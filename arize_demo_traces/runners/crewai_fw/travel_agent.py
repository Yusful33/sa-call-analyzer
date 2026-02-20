"""
CrewAI travel agent pipeline with guardrails and tool calls.
"""

import json

from crewai import Agent, Crew, Task, Process

from ...cost_guard import CostGuard
from ...llm import get_chat_llm
from ...trace_enrichment import run_guardrail, run_tool_call
from ...use_cases.travel_agent import (
    GUARDRAILS,
    SYSTEM_PROMPT,
    flight_search,
    hotel_search,
    parse_travel_query,
    build_options_table,
    select_tools_for_query,
    sample_travel_query,
)


def run_travel_agent(
    query: str | None = None,
    model: str = "gpt-4o-mini",
    guard: CostGuard | None = None,
    tracer_provider=None,
    prospect_context=None,
    degraded_output: str | None = None,
    trace_quality: str = "good",
    **kwargs,
) -> dict:
    """Execute a CrewAI travel agent pipeline."""
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode

    provider = tracer_provider or trace.get_tracer_provider()
    tracer = provider.get_tracer("demo.travel_agent.crewai")
    if not query:
        query = sample_travel_query()

    llm = get_chat_llm(model, temperature=0)

    with tracer.start_as_current_span(
        "travel_agent.run",
        attributes={
            "openinference.span.kind": "CHAIN",
            "input.value": query,
            "input.mime_type": "text/plain",
            "metadata.framework": "crewai",
            "metadata.use_case": "travel-agent",
            "metadata.trace_quality": trace_quality,
        },
    ) as pipeline_span:

        for g in GUARDRAILS:
            run_guardrail(
                tracer, g["name"], query, llm, guard,
                system_prompt=g["system_prompt"],
            )

        params = parse_travel_query(query)
        run_flight, run_hotel = select_tools_for_query(query)
        flight_result = run_tool_call(
            tracer, "flight_search", query,
            lambda: flight_search(params["origin"], params["destination"], params["date_out"]),
            guard=guard,
        ) if run_flight else json.dumps({"options": []})
        hotel_result = run_tool_call(
            tracer, "hotel_search", query,
            lambda: hotel_search(
                params["city"], params["check_in"], params["check_out"], params["guests"]
            ),
            guard=guard,
        ) if run_hotel else json.dumps({"options": []})
        options_table = build_options_table(
            flight_result if isinstance(flight_result, str) else str(flight_result),
            hotel_result if isinstance(hotel_result, str) else str(hotel_result),
        )
        assumptions_note = ""
        if params.get("assumptions"):
            assumptions_note = (
                "\n\n[Assumptions used for search: " + ", ".join(params["assumptions"]) + "]"
            )
        if run_flight and not run_hotel:
            assumptions_note += "\n\n[User asked for flights only; hotel search was not run.]"
        elif run_hotel and not run_flight:
            assumptions_note += "\n\n[User asked for accommodation only; flight search was not run.]"

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
                "Using the tool results below, summarize the options and give a clear recommendation. "
                "Do not ask the user for basic info—use the tables.\n\n"
                f"Client request: {query}\n\n"
                f"Available options from search:\n{options_table}{assumptions_note}\n\n"
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

        out = degraded_output if degraded_output else answer
        pipeline_span.set_attribute("output.value", out[:2000] if len(out) > 2000 else out)
        pipeline_span.set_attribute("output.mime_type", "text/plain")
        pipeline_span.set_status(Status(StatusCode.OK))

    return {
        "query": query,
        "answer": degraded_output if degraded_output else answer,
    }
