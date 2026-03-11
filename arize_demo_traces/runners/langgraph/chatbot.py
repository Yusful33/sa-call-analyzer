"""
LangGraph chatbot/agent pipeline with tool-calling loop and guardrails.
Uses a real StateGraph with conditional edges for the tool-calling loop.
Auto-instrumented by LangChainInstrumentor for authentic LangGraph trace patterns.
"""

import json as _json
import operator
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END

from ...cost_guard import CostGuard
from ...llm import get_chat_llm
from ...trace_enrichment import invoke_chain_in_context, run_guardrail, run_in_context, run_tool_call, tool_definitions_json
from ...use_cases.chatbot import GUARDRAILS, get_system_prompt, get_tools
from ..common_runner_utils import get_query_for_run


class ChatbotState(TypedDict):
    query: str
    messages: Annotated[list, operator.add]
    answer: str
    tools_used: Annotated[list, operator.add]
    guardrail_passed: bool


def run_chatbot(
    query: str | None = None,
    model: str = "gpt-4o-mini",
    guard: CostGuard | None = None,
    tracer_provider=None,
    prospect_context=None,
    degraded_output=None,
    trace_quality="good",
    **kwargs,
) -> dict:
    """Execute a LangGraph chatbot with tool-calling loop: guardrails -> agent <-> tools."""
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode

    from ...use_cases import chatbot as chatbot_use_case

    provider = tracer_provider or trace.get_tracer_provider()
    tracer = provider.get_tracer("demo.chatbot.langgraph")

    if not query:
        rng = kwargs.get("rng")
        _kw = {k: v for k, v in kwargs.items() if k != "rng"}
        query_spec = get_query_for_run(chatbot_use_case, prospect_context=prospect_context, rng=rng, **_kw)
        query = query_spec.text
    else:
        query_spec = None

    tools = get_tools(prospect_context)
    llm = get_chat_llm(model, temperature=0)
    llm_with_tools = llm.bind_tools(tools)
    tool_map = {t.name: t for t in tools}
    system_prompt = get_system_prompt(prospect_context)

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("messages"),
    ])
    chain = prompt | llm_with_tools

    # --- Node functions ---

    def guardrails_node(state: ChatbotState) -> dict:
        for g in GUARDRAILS:
            run_guardrail(
                tracer, g["name"], state["query"], llm, guard,
                system_prompt=g["system_prompt"],
            )
        return {"guardrail_passed": True}

    def agent_node(state: ChatbotState) -> dict:
        if guard:
            guard.check()
        response = invoke_chain_in_context(chain, {"messages": state["messages"]})
        return {"messages": [response]}

    def tools_node(state: ChatbotState) -> dict:
        last_msg = state["messages"][-1]
        tool_results = []
        tools_used = []
        for tc in last_msg.tool_calls:
            if guard:
                guard.check()
            tool_fn = tool_map.get(tc["name"])
            if tool_fn:
                # Explicit TOOL spans (like RAG's search_documents / fetch_document_metadata) for clear trace tree
                def _invoke(**kwargs):
                    return tool_fn.invoke(tc["args"])
                result = run_tool_call(
                    tracer, tc["name"], str(tc["args"]), _invoke, guard=guard
                )
                tools_used.append({"tool": tc["name"], "args": tc["args"], "result": result})
                tool_results.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
        return {"messages": tool_results, "tools_used": tools_used}

    # --- Routing function ---

    def should_continue(state: ChatbotState) -> str:
        last_msg = state["messages"][-1]
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            return "tools"
        return END

    # --- Build graph ---
    workflow = StateGraph(ChatbotState)
    workflow.add_node("guardrails", guardrails_node)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tools_node)
    workflow.set_entry_point("guardrails")
    workflow.add_edge("guardrails", "agent")
    workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    workflow.add_edge("tools", "agent")
    graph = workflow.compile()

    # --- Execute with root span (same format as rag_pipeline: CHAIN root + LangGraph child) ---
    attrs = {
        "openinference.span.kind": "CHAIN",
        "input.value": query,
        "input.mime_type": "text/plain",
        "metadata.framework": "langgraph",
        "metadata.use_case": "multiturn-chatbot-with-tools",
        "metadata.tool_definitions": tool_definitions_json(tools),
    }
    if trace_quality:
        attrs["metadata.trace_quality"] = trace_quality
    if query_spec:
        attrs.update(query_spec.to_span_attributes())
    with tracer.start_as_current_span("chatbot_pipeline", attributes=attrs) as span:
        result = run_in_context(graph.invoke, {
            "query": query,
            "messages": [HumanMessage(content=query)],
            "answer": "",
            "tools_used": [],
            "guardrail_passed": False,
        })
        last_msg = result["messages"][-1]
        answer = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
        result["answer"] = answer
        # Poisoned traces: override with degraded output so Arize evals can flag quality issues
        if degraded_output:
            answer = degraded_output
            result["answer"] = answer
        span.set_attribute("output.value", answer[:5000] if len(answer) > 5000 else answer)
        span.set_attribute("output.mime_type", "text/plain")
        tools_used = result.get("tools_used", [])
        span.set_attribute("tools.count", len(tools_used))
        if tools_used:
            span.set_attribute("metadata.tools_used", ",".join(t["tool"] for t in tools_used))
            tool_results_text = "\n".join(f"[{t['tool']}]: {t['result']}" for t in tools_used)
            span.set_attribute("context.tool_results", tool_results_text[:2000])
        span.set_status(Status(StatusCode.OK))

    return {
        "query": query,
        "answer": result["answer"],
        "tools_used": tools_used,
    }
