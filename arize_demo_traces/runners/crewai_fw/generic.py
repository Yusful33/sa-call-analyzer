"""
CrewAI generic LLM pipeline with a knowledge assistant agent.
Uses real CrewAI Crew, Agent, Task objects to generate authentic trace patterns
when instrumented by CrewAIInstrumentor.
"""

import random

from crewai import Agent, Crew, Task, Process

from ...cost_guard import CostGuard
from ...llm import get_chat_llm
from ...trace_enrichment import run_guardrail, run_tool_call
from ...use_cases.generic import QUERIES, GUARDRAILS, SYSTEM_PROMPT, web_search, get_current_context


def run_generic(
    query: str | None = None,
    model: str = "gpt-4o-mini",
    guard: CostGuard | None = None,
    tracer_provider=None,
    prospect_context=None,
    degraded_output=None,
    trace_quality="good",
    **kwargs,
) -> dict:
    """Execute a CrewAI generic pipeline with knowledge assistant agent, guardrails, and evaluation."""
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode

    provider = tracer_provider or trace.get_tracer_provider()
    tracer = provider.get_tracer("demo.generic.crewai")
    if not query:
        query = random.choice(QUERIES)

    llm = get_chat_llm(model, temperature=0)

    with tracer.start_as_current_span(
        "generic_pipeline",
        attributes={
            "openinference.span.kind": "CHAIN",
            "input.value": query,
            "input.mime_type": "text/plain",
            "metadata.framework": "crewai",
            "metadata.use_case": "generic",
        },
    ) as pipeline_span:

        # === GUARDRAILS ===
        for g in GUARDRAILS:
            run_guardrail(
                tracer, g["name"], query, llm, guard,
                system_prompt=g["system_prompt"],
            )

        # === TOOL CALLS ===
        run_tool_call(tracer, "web_search", query, web_search, guard=guard, query=query)
        run_tool_call(tracer, "get_current_context", query, get_current_context, guard=guard)

        # === CREWAI AGENT ===
        assistant_agent = Agent(
            role="Knowledge Assistant",
            goal="Provide clear, accurate, and concise answers to technical questions",
            backstory=(
                "You are a knowledgeable technical assistant with deep expertise in "
                "software engineering, AI/ML systems, cloud architecture, and observability. "
                "You provide well-structured, informative answers."
            ),
            llm=llm,
            verbose=False,
            allow_delegation=False,
        )

        # === CREWAI TASK ===
        answer_task = Task(
            description=(
                f"Answer the following question clearly and concisely.\n\n"
                f"Question: {query}\n\n"
                f"System context: {SYSTEM_PROMPT}"
            ),
            expected_output="A clear, concise, and accurate answer to the question",
            agent=assistant_agent,
        )

        # === CREWAI CREW ===
        crew = Crew(
            agents=[assistant_agent],
            tasks=[answer_task],
            process=Process.sequential,
            verbose=False,
        )

        if guard:
            guard.check()
        result = crew.kickoff()
        answer = str(result)

        pipeline_span.set_attribute("output.value", answer)
        pipeline_span.set_attribute("output.mime_type", "text/plain")
        pipeline_span.set_status(Status(StatusCode.OK))

    return {
        "query": query,
        "answer": answer,
    }
