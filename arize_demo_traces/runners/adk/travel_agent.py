"""
ADK-style travel agent with manual OpenTelemetry spans.
"""

import random

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

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
    """Execute an ADK-style travel agent: guardrails -> tools -> generate."""
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode

    provider = tracer_provider or trace.get_tracer_provider()
    tracer = provider.get_tracer("demo.travel_agent.adk")

    if not query:
        query = random.choice(QUERIES)

    llm = get_chat_llm(model, temperature=0)

    with tracer.start_as_current_span(
        "travel_agent.run",
        attributes={
            "openinference.span.kind": "AGENT",
            "input.value": query,
            "input.mime_type": "text/plain",
            "metadata.framework": "adk",
            "metadata.use_case": "travel-agent",
        },
    ) as agent_span:

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

        with tracer.start_as_current_span(
            "generate_content",
            attributes={
                "openinference.span.kind": "LLM",
                "input.value": query,
                "input.mime_type": "text/plain",
            },
        ) as llm_span:
            prompt = ChatPromptTemplate.from_messages([
                ("system", SYSTEM_PROMPT),
                ("human", "{question}"),
            ])
            chain = prompt | llm | StrOutputParser()
            if guard:
                guard.check()
            answer = chain.invoke({"question": query})
            llm_span.set_attribute("output.value", answer)
            llm_span.set_attribute("output.mime_type", "text/plain")
            llm_span.set_status(Status(StatusCode.OK))

        agent_span.set_attribute("output.value", answer)
        agent_span.set_attribute("output.mime_type", "text/plain")
        agent_span.set_status(Status(StatusCode.OK))

    return {
        "query": query,
        "answer": answer,
    }
