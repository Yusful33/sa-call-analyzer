"""
LangGraph RAG with LLM tool calling (bind_tools + agent/tools loop).
Produces traces where the model issues real tool_calls; tools are executed via run_tool_call
with metadata.use_case for online eval filters.
"""

import json as _json
import operator
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END

from ...cost_guard import CostGuard
from ...llm import get_chat_llm
from ...trace_enrichment import (
    invoke_chain_in_context,
    run_guardrail,
    run_in_context,
    run_tool_call,
    tool_definitions_json,
)
from ...use_cases.rag import (
    GUARDRAILS,
    get_vectorstore,
    format_docs,
    search_documents,
    fetch_document_metadata,
)
from ..common_runner_utils import get_query_for_run
from ..registry import RAG


@tool
def search_documents_tool(query: str) -> str:
    """Search the knowledge base for document snippets matching the query."""
    return search_documents(query)


@tool
def fetch_document_metadata_tool(source: str) -> str:
    """Fetch JSON metadata for a document by source id (e.g. refund-policy.md)."""
    return fetch_document_metadata(source)


@tool
def retrieve_from_vector_store(query: str) -> str:
    """Retrieve the most relevant passages from the embedded knowledge base for the question."""
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(query)
    return format_docs(docs)


def _tools_for_run():
    return [search_documents_tool, fetch_document_metadata_tool, retrieve_from_vector_store]


class RAGToolsState(TypedDict):
    query: str
    messages: Annotated[list, operator.add]
    answer: str
    retrieved_docs: list
    guardrail_passed: bool


def run_rag(
    query: str | None = None,
    model: str = "gpt-4o-mini",
    guard: CostGuard | None = None,
    tracer_provider=None,
    prospect_context=None,
    degraded_output=None,
    trace_quality="good",
    **kwargs,
) -> dict:
    """RAG via LangGraph: guardrails -> LLM with bound tools <-> tool execution until final answer."""
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode

    provider = tracer_provider or trace.get_tracer_provider()
    tracer = provider.get_tracer("demo.rag.langgraph.tools")

    from ...use_cases import rag as rag_use_case
    if not query:
        rng = kwargs.get("rng")
        _kw = {k: v for k, v in kwargs.items() if k != "rng"}
        query_spec = get_query_for_run(rag_use_case, prospect_context=prospect_context, rng=rng, **_kw)
        query = query_spec.text
    else:
        query_spec = None

    llm = get_chat_llm(model, temperature=0)
    tools = _tools_for_run()
    llm_with_tools = llm.bind_tools(tools)
    tool_map = {t.name: t for t in tools}

    agent_system = (
        "You are a RAG assistant with retrieval tools. Use tools to gather grounded information. "
        "Prefer retrieve_from_vector_store for factual answers from the knowledge base. "
        "When you have enough information, reply with a single concise answer in plain language only — "
        "no mention of tools or internal steps."
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", agent_system),
        MessagesPlaceholder("messages"),
    ])
    chain = prompt | llm_with_tools

    def guardrails_node(state: RAGToolsState) -> dict:
        for g in GUARDRAILS:
            run_guardrail(
                tracer, g["name"], state["query"], llm, guard,
                system_prompt=g["system_prompt"],
            )
        return {"guardrail_passed": True}

    def agent_node(state: RAGToolsState) -> dict:
        if guard:
            guard.check()
        response = invoke_chain_in_context(chain, {"messages": state["messages"]})
        return {"messages": [response]}

    def tools_node(state: RAGToolsState) -> dict:
        last_msg = state["messages"][-1]
        tool_results = []
        for tc in last_msg.tool_calls:
            if guard:
                guard.check()
            tool_fn = tool_map.get(tc["name"])
            if tool_fn:
                def _invoke(**kwargs):
                    return tool_fn.invoke(tc["args"])

                result = run_tool_call(
                    tracer,
                    tc["name"],
                    _json.dumps(tc["args"]),
                    _invoke,
                    guard=guard,
                    metadata_use_case=RAG,
                )
                tool_results.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
        return {"messages": tool_results}

    def should_continue(state: RAGToolsState) -> str:
        last_msg = state["messages"][-1]
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            return "tools"
        return END

    workflow = StateGraph(RAGToolsState)
    workflow.add_node("guardrails", guardrails_node)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tools_node)
    workflow.set_entry_point("guardrails")
    workflow.add_edge("guardrails", "agent")
    workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    workflow.add_edge("tools", "agent")
    graph = workflow.compile()

    attrs = {
        "openinference.span.kind": "CHAIN",
        "input.value": query,
        "input.mime_type": "text/plain",
        "metadata.framework": "langgraph",
        "metadata.use_case": RAG,
        "metadata.tool_definitions": tool_definitions_json(tools),
    }
    if trace_quality:
        attrs["metadata.trace_quality"] = trace_quality
    if query_spec:
        attrs.update(query_spec.to_span_attributes())

    with tracer.start_as_current_span("rag_pipeline", attributes=attrs) as span:
        result = run_in_context(graph.invoke, {
            "query": query,
            "messages": [HumanMessage(content=query)],
            "answer": "",
            "retrieved_docs": [],
            "guardrail_passed": False,
        })
        last_msg = result["messages"][-1]
        answer = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
        if degraded_output:
            answer = degraded_output
        vectorstore = get_vectorstore()
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        retrieved = run_in_context(retriever.invoke, query)
        ctx = format_docs(retrieved)
        span.set_attribute("rag.retrieval_context", ctx[:8000])
        span.set_attribute("output.value", answer[:5000] if len(answer) > 5000 else answer)
        span.set_attribute("output.mime_type", "text/plain")
        span.set_status(Status(StatusCode.OK))

    retrieved_docs = [
        {"content": d.page_content, "source": d.metadata.get("source", "")}
        for d in retrieved
    ]

    return {
        "query": query,
        "answer": answer,
        "retrieved_docs": retrieved_docs,
    }
