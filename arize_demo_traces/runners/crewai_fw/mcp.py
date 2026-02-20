"""
CrewAI MCP (Model Context Protocol) pipeline with tool discovery and execution agents.
Uses real CrewAI Crew, Agent, Task objects to generate authentic trace patterns
when instrumented by CrewAIInstrumentor.
"""

import json as _json
import random

from crewai import Agent, Crew, Task, Process

from ...cost_guard import CostGuard
from ...llm import get_chat_llm
from ...trace_enrichment import run_guardrail
from ...use_cases.mcp import (
    QUERIES,
    MCP_SERVERS,
    GUARDRAILS,
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
    """Execute a CrewAI MCP pipeline with discovery, planning, and synthesis agents."""
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode

    provider = tracer_provider or trace.get_tracer_provider()
    tracer = provider.get_tracer("demo.mcp.crewai")
    if not query:
        query = random.choice(QUERIES)

    llm = get_chat_llm(model, temperature=0)
    servers_info = _json.dumps(
        [{"name": s["name"], "tools": s["tools"]} for s in MCP_SERVERS], indent=2
    )

    with tracer.start_as_current_span(
        "mcp_pipeline",
        attributes={
            "openinference.span.kind": "AGENT",
            "input.value": query,
            "input.mime_type": "text/plain",
            "metadata.framework": "crewai",
            "metadata.use_case": "mcp-tool-use",
        },
    ) as pipeline_span:

        # === GUARDRAILS ===
        for g in GUARDRAILS:
            run_guardrail(
                tracer, g["name"], query, llm, guard,
                system_prompt=g["system_prompt"],
            )

        # === CREWAI AGENTS ===
        tool_discovery_agent = Agent(
            role="MCP Tool Discovery Specialist",
            goal="Identify the right MCP servers and tools for the user's request",
            backstory=(
                f"You are an expert at the Model Context Protocol (MCP). You know "
                f"how to discover and select the right tools from available MCP servers "
                f"to fulfill user requests.\n\nAvailable servers:\n{servers_info}"
            ),
            llm=llm,
            verbose=False,
            allow_delegation=False,
        )

        planner_agent = Agent(
            role="Execution Planner",
            goal="Create an optimal execution plan for MCP tool calls",
            backstory=(
                "You are a planning specialist who creates step-by-step execution "
                "plans for MCP tool calls. You determine the right order, parameters, "
                "and dependencies between tool calls."
            ),
            llm=llm,
            verbose=False,
            allow_delegation=False,
        )

        synthesis_agent = Agent(
            role="Results Synthesizer",
            goal="Synthesize MCP tool results into a clear, actionable response",
            backstory=(
                "You are an expert at interpreting tool call results and presenting "
                "findings in a clear, well-structured format that directly answers "
                "the user's question."
            ),
            llm=llm,
            verbose=False,
            allow_delegation=False,
        )

        # === CREWAI TASKS ===
        discover_task = Task(
            description=(
                f"Identify which MCP servers and tools are needed to fulfill this request: "
                f"{query}\n\nAvailable MCP servers:\n{servers_info}"
            ),
            expected_output="A list of relevant MCP servers and tools to use",
            agent=tool_discovery_agent,
        )

        # Simulate tool execution
        tool_results = get_simulated_tool_results(num_tools=3)

        plan_task = Task(
            description=(
                f"Create an execution plan for the MCP tool calls needed to answer: {query}\n\n"
                f"Tool results available:\n{_json.dumps(tool_results, indent=2, default=str)}"
            ),
            expected_output="A step-by-step execution plan",
            agent=planner_agent,
        )

        synthesize_task = Task(
            description=(
                f"Synthesize the following MCP tool results into a clear response "
                f"for the user's request: {query}\n\n"
                f"Tool results:\n{_json.dumps(tool_results, indent=2, default=str)}"
            ),
            expected_output="A clear, well-structured response answering the user's question",
            agent=synthesis_agent,
        )

        # === CREWAI CREW ===
        crew = Crew(
            agents=[tool_discovery_agent, planner_agent, synthesis_agent],
            tasks=[discover_task, plan_task, synthesize_task],
            process=Process.sequential,
            verbose=False,
        )

        if guard:
            guard.check()
        result = crew.kickoff()
        response = str(result)

        # Record tool executions as spans
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

        pipeline_span.set_attribute("output.value", response)
        pipeline_span.set_attribute("output.mime_type", "text/plain")
        pipeline_span.set_status(Status(StatusCode.OK))

    return {
        "query": query,
        "tool_results": tool_results,
        "response": response,
    }
