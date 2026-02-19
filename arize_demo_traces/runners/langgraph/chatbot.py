"""
LangGraph chatbot/agent pipeline with tool-calling loop and guardrails.
Uses a real StateGraph with conditional edges for the tool-calling loop.
Auto-instrumented by LangChainInstrumentor for authentic LangGraph trace patterns.
"""

import json as _json
import operator
import random
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END

from ...cost_guard import CostGuard
from ...llm import get_chat_llm
from ...trace_enrichment import run_guardrail
from ...use_cases.chatbot import QUERIES, GUARDRAILS, SYSTEM_PROMPT, TOOLS


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
) -> dict:
    """Execute a LangGraph chatbot with tool-calling loop: guardrails -> agent <-> tools."""
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode

    provider = tracer_provider or trace.get_tracer_provider()
    tracer = provider.get_tracer("demo.chatbot.langgraph")

    if not query:
        query = random.choice(QUERIES)

    llm = get_chat_llm(model, temperature=0)
    llm_with_tools = llm.bind_tools(TOOLS)
    tool_map = {t.name: t for t in TOOLS}

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
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
        response = chain.invoke({"messages": state["messages"]})
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
                result = tool_fn.invoke(tc["args"])
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

    # --- Execute with root span ---
    with tracer.start_as_current_span(
        "customer_support_agent",
        attributes={
            "openinference.span.kind": "AGENT",
            "input.value": query,
            "input.mime_type": "text/plain",
            "metadata.framework": "langgraph",
            "metadata.use_case": "multiturn-chatbot-with-tools",
        },
    ) as span:
        result = graph.invoke({
            "query": query,
            "messages": [HumanMessage(content=query)],
            "answer": "",
            "tools_used": [],
            "guardrail_passed": False,
        })
        last_msg = result["messages"][-1]
        answer = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
        result["answer"] = answer
        span.set_attribute("output.value", answer)
        span.set_attribute("output.mime_type", "text/plain")
        span.set_attribute("tools.count", len(result.get("tools_used", [])))
        span.set_status(Status(StatusCode.OK))

    return {
        "query": query,
        "answer": result["answer"],
        "tools_used": result.get("tools_used", []),
    }
