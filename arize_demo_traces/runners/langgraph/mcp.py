"""
LangGraph MCP (Model Context Protocol) agent pipeline.
Uses a real StateGraph with nodes for tool discovery, planning, execution, and synthesis.
Auto-instrumented by LangChainInstrumentor for authentic LangGraph trace patterns.
"""

import json as _json
import random
from typing import TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

from ...cost_guard import CostGuard
from ...llm import get_chat_llm
from ...trace_enrichment import run_guardrail
from ...use_cases.mcp import (
    QUERIES,
    MCP_SERVERS,
    GUARDRAILS,
    SYSTEM_PROMPT_DISCOVER,
    SYSTEM_PROMPT_PLAN,
    SYSTEM_PROMPT_SYNTHESIZE,
    get_simulated_tool_results,
)


class MCPState(TypedDict):
    query: str
    discovered_tools: str
    execution_plan: str
    tool_results: list
    response: str
    guardrail_passed: bool


def run_mcp(
    query: str | None = None,
    model: str = "gpt-4o-mini",
    guard: CostGuard | None = None,
    tracer_provider=None,
    prospect_context=None,
) -> dict:
    """Execute a LangGraph MCP pipeline: guardrails -> discover -> plan -> execute -> synthesize."""
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode

    provider = tracer_provider or trace.get_tracer_provider()
    tracer = provider.get_tracer("demo.mcp.langgraph")

    if not query:
        query = random.choice(QUERIES)

    llm = get_chat_llm(model, temperature=0)
    servers_info = _json.dumps(
        [{"name": s["name"], "tools": s["tools"]} for s in MCP_SERVERS], indent=2
    )

    # --- Node functions ---

    def guardrails_node(state: MCPState) -> dict:
        for g in GUARDRAILS:
            run_guardrail(
                tracer, g["name"], state["query"], llm, guard,
                system_prompt=g["system_prompt"],
            )
        return {"guardrail_passed": True}

    def discover_tools_node(state: MCPState) -> dict:
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT_DISCOVER),
            ("human", "Available MCP servers:\n{servers}\n\nUser request: {query}"),
        ])
        if guard:
            guard.check()
        messages = prompt.format_messages(servers=servers_info, query=state["query"])
        response = llm.invoke(messages)
        result = response.content if hasattr(response, "content") else str(response)
        return {"discovered_tools": result}

    def plan_node(state: MCPState) -> dict:
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT_PLAN),
            ("human", "User request: {query}\n\nAvailable tools:\n{tools}"),
        ])
        if guard:
            guard.check()
        messages = prompt.format_messages(
            query=state["query"], tools=state["discovered_tools"]
        )
        response = llm.invoke(messages)
        result = response.content if hasattr(response, "content") else str(response)
        return {"execution_plan": result}

    def execute_tools_node(state: MCPState) -> dict:
        tool_results = get_simulated_tool_results(num_tools=3)
        return {"tool_results": tool_results}

    def synthesize_node(state: MCPState) -> dict:
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT_SYNTHESIZE),
            ("human", "User request: {query}\n\nTool results:\n{results}"),
        ])
        if guard:
            guard.check()
        messages = prompt.format_messages(
            query=state["query"],
            results=_json.dumps(state["tool_results"], indent=2, default=str),
        )
        response = llm.invoke(messages)
        result = response.content if hasattr(response, "content") else str(response)
        return {"response": result}

    # --- Build graph ---
    workflow = StateGraph(MCPState)
    workflow.add_node("guardrails", guardrails_node)
    workflow.add_node("discover_tools", discover_tools_node)
    workflow.add_node("plan", plan_node)
    workflow.add_node("execute_tools", execute_tools_node)
    workflow.add_node("synthesize", synthesize_node)
    workflow.set_entry_point("guardrails")
    workflow.add_edge("guardrails", "discover_tools")
    workflow.add_edge("discover_tools", "plan")
    workflow.add_edge("plan", "execute_tools")
    workflow.add_edge("execute_tools", "synthesize")
    workflow.add_edge("synthesize", END)
    graph = workflow.compile()

    # --- Execute with root span ---
    with tracer.start_as_current_span(
        "mcp_agent",
        attributes={
            "openinference.span.kind": "AGENT",
            "input.value": query,
            "input.mime_type": "text/plain",
            "metadata.framework": "langgraph",
            "metadata.use_case": "mcp-tool-use",
        },
    ) as span:
        result = graph.invoke({
            "query": query,
            "discovered_tools": "",
            "execution_plan": "",
            "tool_results": [],
            "response": "",
            "guardrail_passed": False,
        })
        span.set_attribute("output.value", result["response"])
        span.set_attribute("output.mime_type", "text/plain")
        span.set_status(Status(StatusCode.OK))

    return {
        "query": query,
        "discovered_tools": result["discovered_tools"],
        "execution_plan": result["execution_plan"],
        "tool_results": result["tool_results"],
        "response": result["response"],
    }
