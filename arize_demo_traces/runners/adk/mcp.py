"""
ADK-style MCP (Model Context Protocol) agent with manual OpenTelemetry spans.
Mimics Google Agent Development Kit trace patterns without requiring ADK installation.
Uses real LLM calls inside manually created OTel spans with OpenInference attributes.

Trace structure:
  mcp_agent.run (AGENT)
    -> guardrails (GUARDRAIL)
    -> generate_content (LLM) - discover tools
    -> generate_content (LLM) - plan execution
    -> mcp.server.tool (TOOL) - execute MCP tools (multiple)
    -> generate_content (LLM) - synthesize results
"""

import json as _json
import random

from langchain_core.prompts import ChatPromptTemplate

from ...cost_guard import CostGuard
from ...llm import get_chat_llm
from ...trace_enrichment import invoke_llm_in_context, run_guardrail
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
    """Execute an ADK-style MCP agent: guardrails -> discover -> plan -> execute -> synthesize."""
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode

    provider = tracer_provider or trace.get_tracer_provider()
    tracer = provider.get_tracer("demo.mcp.adk")

    if not query:
        query = random.choice(QUERIES)

    llm = get_chat_llm(model, temperature=0)
    servers_info = _json.dumps(
        [{"name": s["name"], "tools": s["tools"]} for s in MCP_SERVERS], indent=2
    )

    with tracer.start_as_current_span(
        "mcp_agent.run",
        attributes={
            "openinference.span.kind": "AGENT",
            "input.value": query,
            "input.mime_type": "text/plain",
            "metadata.framework": "adk",
            "metadata.use_case": "mcp-tool-use",
        },
    ) as agent_span:

        # ---- Guardrails ----
        for g in GUARDRAILS:
            run_guardrail(
                tracer, g["name"], query, llm, guard,
                system_prompt=g["system_prompt"],
            )

        # ---- generate_content: discover tools ----
        with tracer.start_as_current_span(
            "generate_content",
            attributes={
                "openinference.span.kind": "LLM",
                "input.value": query,
                "input.mime_type": "text/plain",
            },
        ) as discover_span:
            discover_prompt = ChatPromptTemplate.from_messages([
                ("system", SYSTEM_PROMPT_DISCOVER),
                ("human", "Available MCP servers:\n{servers}\n\nUser request: {query}"),
            ])
            if guard:
                guard.check()
            messages = discover_prompt.format_messages(servers=servers_info, query=query)
            response = invoke_llm_in_context(llm, messages)
            discovered = response.content if hasattr(response, "content") else str(response)
            discover_span.set_attribute("output.value", discovered)
            discover_span.set_attribute("output.mime_type", "text/plain")
            discover_span.set_status(Status(StatusCode.OK))

        # ---- generate_content: plan execution ----
        with tracer.start_as_current_span(
            "generate_content",
            attributes={
                "openinference.span.kind": "LLM",
                "input.value": query,
                "input.mime_type": "text/plain",
            },
        ) as plan_span:
            plan_prompt = ChatPromptTemplate.from_messages([
                ("system", SYSTEM_PROMPT_PLAN),
                ("human", "User request: {query}\n\nAvailable tools:\n{tools}"),
            ])
            if guard:
                guard.check()
            messages = plan_prompt.format_messages(query=query, tools=discovered)
            response = invoke_llm_in_context(llm, messages)
            plan = response.content if hasattr(response, "content") else str(response)
            plan_span.set_attribute("output.value", plan)
            plan_span.set_attribute("output.mime_type", "text/plain")
            plan_span.set_status(Status(StatusCode.OK))

        # ---- MCP tool execution (TOOL spans) ----
        tool_results = get_simulated_tool_results(num_tools=3)
        for tr in tool_results:
            with tracer.start_as_current_span(
                f"mcp.{tr['server']}.{tr['tool']}",
                attributes={
                    "openinference.span.kind": "TOOL",
                    "input.value": query,
                    "input.mime_type": "text/plain",
                    "tool.name": tr["tool"],
                },
            ) as tool_span:
                tool_span.set_attribute("output.value", _json.dumps(tr["result"], default=str))
                tool_span.set_attribute("output.mime_type", "application/json")
                tool_span.set_attribute("mcp.server", tr["server"])
                tool_span.set_status(Status(StatusCode.OK))

        # ---- generate_content: synthesize results ----
        with tracer.start_as_current_span(
            "generate_content",
            attributes={
                "openinference.span.kind": "LLM",
                "input.value": query,
                "input.mime_type": "text/plain",
            },
        ) as synth_span:
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
            response = invoke_llm_in_context(llm, messages)
            synthesized = response.content if hasattr(response, "content") else str(response)
            synth_span.set_attribute("output.value", synthesized)
            synth_span.set_attribute("output.mime_type", "text/plain")
            synth_span.set_status(Status(StatusCode.OK))

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
