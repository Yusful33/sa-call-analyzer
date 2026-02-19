"""
CrewAI travel agent pipeline with guardrails and tool calls.
"""

import random

from crewai import Agent, Crew, Task, Process

from ...cost_guard import CostGuard
from ...llm import get_chat_llm
from ...trace_enrichment import run_guardrail, run_tool_call
from ...use_cases.travel_agent import (
    QUERIES,
    GUARDRAILS,
    SYSTEM_PROMPT,
    flight_search,
    hotel_search,
)


def run_travel_agent(
    query: str | None = None,
    model: str = "gpt-4o-mini",
    guard: CostGuard | None = None,
    tracer_provider=None,
    prospect_context=None,
) -> dict:
    """Execute a CrewAI travel agent pipeline."""
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode

    provider = tracer_provider or trace.get_tracer_provider()
    tracer = provider.get_tracer("demo.travel_agent.crewai")
    if not query:
        query = random.choice(QUERIES)

    llm = get_chat_llm(model, temperature=0)

    with tracer.start_as_current_span(
        "travel_agent_pipeline",
        attributes={
            "openinference.span.kind": "CHAIN",
            "input.value": query,
            "input.mime_type": "text/plain",
            "metadata.framework": "crewai",
            "metadata.use_case": "travel-agent",
        },
    ) as pipeline_span:

        for g in GUARDRAILS:
            run_guardrail(
                tracer, g["name"], query, llm, guard,
                system_prompt=g["system_prompt"],
            )

        run_tool_call(
            tracer, "flight_search", query,
            lambda: flight_search("NYC", "Paris", "2025-03-15"),
            guard=guard,
        )
        run_tool_call(
            tracer, "hotel_search", query,
            lambda: hotel_search("Paris", "2025-03-15", "2025-03-17", 2),
            guard=guard,
        )

        travel_agent = Agent(
            role="Travel Agent",
            goal="Find the best flights and hotels for the client and present clear recommendations",
            backstory=(
                "You are an experienced travel agent with expertise in international travel, "
                "airline options, and hotel bookings. You use tool results to give accurate, "
                "actionable recommendations."
            ),
            llm=llm,
            verbose=False,
            allow_delegation=False,
        )

        answer_task = Task(
            description=(
                f"Using the tool results above, answer the client's request clearly.\n\n"
                f"Client request: {query}\n\n"
                f"Guidelines: {SYSTEM_PROMPT}"
            ),
            expected_output="A clear summary of flight and hotel options with prices and practical tips",
            agent=travel_agent,
        )

        crew = Crew(
            agents=[travel_agent],
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
