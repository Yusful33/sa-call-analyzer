"""
CrewAI multi-agent orchestration pipeline with four specialised agents.
Uses real CrewAI Crew, Agent, Task objects to generate authentic trace patterns
when instrumented by CrewAIInstrumentor.
"""

import random

from crewai import Agent, Crew, Task, Process

from ...cost_guard import CostGuard
from ...llm import get_chat_llm
from ...trace_enrichment import run_guardrail, run_tool_call, run_in_context
from ...use_cases.multi_agent import (
    QUERIES,
    AGENTS,
    GUARDRAILS,
    RESEARCH_PROMPT,
    ANALYSIS_PROMPT,
    WRITER_PROMPT,
    REVIEWER_PROMPT,
    search_web,
    analyze_metrics,
)


def run_multi_agent(
    query: str | None = None,
    model: str = "gpt-4o-mini",
    guard: CostGuard | None = None,
    tracer_provider=None,
    prospect_context=None,
    degraded_output=None,
    trace_quality="good",
    **kwargs,
) -> dict:
    """Execute a CrewAI multi-agent orchestration: guardrails -> 4-agent crew."""
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode

    provider = tracer_provider or trace.get_tracer_provider()
    tracer = provider.get_tracer("demo.multi_agent.crewai")
    if not query:
        query = random.choice(QUERIES)

    llm = get_chat_llm(model, temperature=0)

    with tracer.start_as_current_span(
        "multi_agent_orchestration",
        attributes={
            "openinference.span.kind": "AGENT",
            "input.value": query,
            "input.mime_type": "text/plain",
            "metadata.framework": "crewai",
            "metadata.use_case": "multi-agent-orchestration",
        },
    ) as root_span:

        # === GUARDRAILS ===
        for g in GUARDRAILS:
            run_guardrail(
                tracer, g["name"], query, llm, guard,
                system_prompt=g["system_prompt"],
            )

        # === TOOL CALLS ===
        run_tool_call(tracer, "search_web", query, search_web, guard=guard, query=query)
        run_tool_call(tracer, "analyze_metrics", query, analyze_metrics, guard=guard, metric_name="performance")

        # === CREWAI AGENTS ===
        agent_cfgs = AGENTS  # researcher, analyst, writer, reviewer

        researcher = Agent(
            role=agent_cfgs[0]["role"],
            goal=agent_cfgs[0]["goal"],
            backstory=agent_cfgs[0]["backstory"],
            llm=llm,
            verbose=False,
            allow_delegation=False,
        )

        analyst = Agent(
            role=agent_cfgs[1]["role"],
            goal=agent_cfgs[1]["goal"],
            backstory=agent_cfgs[1]["backstory"],
            llm=llm,
            verbose=False,
            allow_delegation=False,
        )

        writer = Agent(
            role=agent_cfgs[2]["role"],
            goal=agent_cfgs[2]["goal"],
            backstory=agent_cfgs[2]["backstory"],
            llm=llm,
            verbose=False,
            allow_delegation=False,
        )

        reviewer = Agent(
            role=agent_cfgs[3]["role"],
            goal=agent_cfgs[3]["goal"],
            backstory=agent_cfgs[3]["backstory"],
            llm=llm,
            verbose=False,
            allow_delegation=False,
        )

        # === CREWAI TASKS ===
        research_task = Task(
            description=(
                f"{RESEARCH_PROMPT}\n\nUser request: {query}"
            ),
            expected_output="Structured research findings with key data points and sources",
            agent=researcher,
        )

        analyze_task = Task(
            description=(
                f"{ANALYSIS_PROMPT}\n\nUser request: {query}"
            ),
            expected_output="Actionable insights with supporting evidence and recommendations",
            agent=analyst,
        )

        write_task = Task(
            description=(
                f"{WRITER_PROMPT}\n\nUser request: {query}"
            ),
            expected_output="A clear, well-structured document addressing the original request",
            agent=writer,
        )

        review_task = Task(
            description=(
                f"{REVIEWER_PROMPT}\n\nUser request: {query}"
            ),
            expected_output="Final reviewed output with quality assessment and any corrections applied",
            agent=reviewer,
        )

        # === CREWAI CREW ===
        crew = Crew(
            agents=[researcher, analyst, writer, reviewer],
            tasks=[research_task, analyze_task, write_task, review_task],
            process=Process.sequential,
            verbose=False,
        )

        if guard:
            guard.check()
        # Run in current trace context so CrewAI-instrumented LLM spans parent to this root
        # (avoids orphan "completion" traces when CrewAI runs in same thread).
        result = run_in_context(crew.kickoff)
        answer = str(result)

        root_span.set_attribute("output.value", answer)
        root_span.set_attribute("output.mime_type", "text/plain")
        root_span.set_status(Status(StatusCode.OK))

    return {
        "query": query,
        "answer": answer,
    }
