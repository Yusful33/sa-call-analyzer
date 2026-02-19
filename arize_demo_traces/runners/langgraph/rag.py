"""
LangGraph RAG pipeline with guardrails.
Uses a real StateGraph with typed state, nodes, and edges.
Auto-instrumented by LangChainInstrumentor for authentic LangGraph trace patterns.
"""

import random
from typing import TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

from ...cost_guard import CostGuard
from ...llm import get_chat_llm
from ...trace_enrichment import run_guardrail, run_tool_call
from ...use_cases.rag import (
    QUERIES,
    GUARDRAILS,
    SYSTEM_PROMPT,
    get_vectorstore,
    format_docs,
    search_documents,
    fetch_document_metadata,
)


class RAGState(TypedDict):
    query: str
    context: str
    answer: str
    retrieved_docs: list
    guardrail_passed: bool


def run_rag(
    query: str | None = None,
    model: str = "gpt-4o-mini",
    guard: CostGuard | None = None,
    tracer_provider=None,
    prospect_context=None,
) -> dict:
    """Execute a LangGraph RAG pipeline: guardrails -> retrieve -> generate."""
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode

    provider = tracer_provider or trace.get_tracer_provider()
    tracer = provider.get_tracer("demo.rag.langgraph")

    if not query:
        query = random.choice(QUERIES)

    llm = get_chat_llm(model, temperature=0)

    # --- Node functions (closures capturing llm, guard, tracer) ---

    def guardrails_node(state: RAGState) -> dict:
        for g in GUARDRAILS:
            run_guardrail(
                tracer, g["name"], state["query"], llm, guard,
                system_prompt=g["system_prompt"],
            )
        return {"guardrail_passed": True}

    def retrieve_node(state: RAGState) -> dict:
        run_tool_call(tracer, "search_documents", state["query"],
                      search_documents, guard=guard, query=state["query"])
        if guard:
            guard.check()
        vectorstore = get_vectorstore()
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(state["query"])
        if docs:
            run_tool_call(tracer, "fetch_document_metadata", docs[0].metadata.get("source", ""),
                          fetch_document_metadata, guard=guard,
                          source=docs[0].metadata.get("source", "unknown"))
        return {"context": format_docs(docs), "retrieved_docs": docs}

    def generate_node(state: RAGState) -> dict:
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "{question}"),
        ])
        if guard:
            guard.check()
        messages = prompt.format_messages(context=state["context"], question=state["query"])
        response = llm.invoke(messages)
        answer = response.content if hasattr(response, "content") else str(response)
        return {"answer": answer}

    # --- Build graph ---
    workflow = StateGraph(RAGState)
    workflow.add_node("guardrails", guardrails_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    workflow.set_entry_point("guardrails")
    workflow.add_edge("guardrails", "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    graph = workflow.compile()

    # --- Execute with root span ---
    with tracer.start_as_current_span(
        "rag_pipeline",
        attributes={
            "openinference.span.kind": "CHAIN",
            "input.value": query,
            "input.mime_type": "text/plain",
            "metadata.framework": "langgraph",
            "metadata.use_case": "retrieval-augmented-search",
        },
    ) as span:
        result = graph.invoke({
            "query": query,
            "context": "",
            "answer": "",
            "retrieved_docs": [],
            "guardrail_passed": False,
        })
        span.set_attribute("output.value", result["answer"])
        span.set_attribute("output.mime_type", "text/plain")
        span.set_status(Status(StatusCode.OK))

    return {
        "query": query,
        "answer": result["answer"],
        "retrieved_docs": [
            {"content": d.page_content, "source": d.metadata.get("source", "")}
            for d in result.get("retrieved_docs", [])
        ],
    }
