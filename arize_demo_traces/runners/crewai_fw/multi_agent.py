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
    AGENTS,
    GUARDRAILS,
    RESEARCH_PROMPT,
    ANALYSIS_PROMPT,
    WRITER_PROMPT,
    REVIEWER_PROMPT,
    search_web,
    analyze_metrics,
)
from ..common_runner_utils import get_query_for_run


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
    from ...use_cases import multi_agent as multi_agent_use_case
    if not query:
        rng = kwargs.get("rng")
        _kw = {k: v for k, v in kwargs.items() if k != "rng"}
        query_spec = get_query_for_run(multi_agent_use_case, prospect_context=prospect_context, rng=rng, **_kw)
        query = query_spec.text
    else:
        query_spec = None

    llm = get_chat_llm(model, temperature=0)

    attrs = {
        "openinference.span.kind": "AGENT",
        "input.value": query,
        "input.mime_type": "text/plain",
        "metadata.framework": "crewai",
        "metadata.use_case": "multi-agent-orchestration",
    }
    if query_spec:
        attrs.update(query_spec.to_span_attributes())
    with tracer.start_as_current_span("multi_agent_orchestration", attributes=attrs) as root_span:

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
        root_span.set_attribute("context.query", query[:1000])
        root_span.set_attribute("context.answer_summary", answer[:2000])
        root_span.set_status(Status(StatusCode.OK))

    return {
        "query": query,
        "answer": answer,
    }
