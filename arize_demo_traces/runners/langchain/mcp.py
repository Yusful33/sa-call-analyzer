"""
LangChain LCEL MCP (Model Context Protocol) agent pipeline.
Uses prompt | llm chains, auto-instrumented by LangChainInstrumentor.
"""

import json as _json
import random

from langchain_core.prompts import ChatPromptTemplate

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


def run_mcp(
    query: str | None = None,
    model: str = "gpt-4o-mini",
    guard: CostGuard | None = None,
    tracer_provider=None,
    prospect_context=None,
    degraded_output=None,
    trace_quality="good",
    **kwargs,
) -> dict:
    """Execute a LangChain LCEL MCP pipeline: guardrails -> discover -> plan -> execute -> synthesize."""
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode

    provider = tracer_provider or trace.get_tracer_provider()
    tracer = provider.get_tracer("demo.mcp")

    if not query:
        query = random.choice(QUERIES)

    llm = get_chat_llm(model, temperature=0)
    servers_info = _json.dumps(
        [{"name": s["name"], "tools": s["tools"]} for s in MCP_SERVERS], indent=2
    )

    with tracer.start_as_current_span(
        "mcp_agent",
        attributes={
            "openinference.span.kind": "AGENT",
            "input.value": query,
            "input.mime_type": "text/plain",
            "metadata.framework": "langchain",
            "metadata.use_case": "mcp-tool-use",
        },
    ) as agent_span:

        # === GUARDRAILS ===
        for g in GUARDRAILS:
            run_guardrail(
                tracer, g["name"], query, llm, guard,
                system_prompt=g["system_prompt"],
            )

        # === DISCOVER TOOLS ===
        with tracer.start_as_current_span(
            "discover_tools",
            attributes={"openinference.span.kind": "CHAIN", "input.value": query},
        ) as step:
            discover_prompt = ChatPromptTemplate.from_messages([
                ("system", SYSTEM_PROMPT_DISCOVER),
                ("human", "Available MCP servers:\n{servers}\n\nUser request: {query}"),
            ])
            if guard:
                guard.check()
            messages = discover_prompt.format_messages(servers=servers_info, query=query)
            response = llm.invoke(messages)
            discovered = response.content if hasattr(response, "content") else str(response)
            step.set_attribute("output.value", discovered)
            step.set_status(Status(StatusCode.OK))

        # === PLAN EXECUTION ===
        with tracer.start_as_current_span(
            "plan_execution",
            attributes={"openinference.span.kind": "CHAIN", "input.value": query},
        ) as step:
            plan_prompt = ChatPromptTemplate.from_messages([
                ("system", SYSTEM_PROMPT_PLAN),
                ("human", "User request: {query}\n\nAvailable tools:\n{tools}"),
            ])
            if guard:
                guard.check()
            messages = plan_prompt.format_messages(query=query, tools=discovered)
            response = llm.invoke(messages)
            plan = response.content if hasattr(response, "content") else str(response)
            step.set_attribute("output.value", plan)
            step.set_status(Status(StatusCode.OK))

        # === EXECUTE MCP TOOLS ===
        tool_results = get_simulated_tool_results(num_tools=3)
        for tr in tool_results:
            with tracer.start_as_current_span(
                f"mcp.{tr['server']}.{tr['tool']}",
                attributes={
                    "openinference.span.kind": "TOOL",
                    "tool.name": tr["tool"],
                    "input.value": query,
                },
            ) as tool_span:
                tool_span.set_attribute("output.value", _json.dumps(tr["result"], default=str))
                tool_span.set_attribute("output.mime_type", "application/json")
                tool_span.set_attribute("mcp.server", tr["server"])
                tool_span.set_status(Status(StatusCode.OK))

        # === SYNTHESIZE ===
        with tracer.start_as_current_span(
            "synthesize_results",
            attributes={"openinference.span.kind": "CHAIN", "input.value": query},
        ) as step:
            synth_prompt = ChatPromptTemplate.from_messages([
                ("system", SYSTEM_PROMPT_SYNTHESIZE),
                ("human", "User request: {query}\n\nTool results:\n{results}"),
            ])
            if guard:
                guard.check()
            messages = synth_prompt.format_messages(
                query=query,
                results=_json.dumps(tool_results, indent=2, default=str),
            )
            response = llm.invoke(messages)
            synthesized = response.content if hasattr(response, "content") else str(response)
            step.set_attribute("output.value", synthesized)
            step.set_status(Status(StatusCode.OK))

        agent_span.set_attribute("output.value", synthesized)
        agent_span.set_attribute("output.mime_type", "text/plain")
        agent_span.set_status(Status(StatusCode.OK))

    return {
        "query": query,
        "discovered_tools": discovered,
        "execution_plan": plan,
        "tool_results": tool_results,
        "response": synthesized,
    }
