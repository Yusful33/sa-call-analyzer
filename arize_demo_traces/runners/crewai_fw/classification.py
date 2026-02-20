"""
CrewAI classification pipeline with classifier and responder agents.
Uses real CrewAI Crew, Agent, Task objects to generate authentic trace patterns
when instrumented by CrewAIInstrumentor.
"""

import random

from crewai import Agent, Crew, Task, Process

from ...cost_guard import CostGuard
from ...llm import get_chat_llm
from ...trace_enrichment import run_guardrail, run_tool_call
from ...use_cases.classification import (
    QUERIES,
    CATEGORIES,
    GUARDRAILS,
    SYSTEM_PROMPT_CLASSIFY,
    SYSTEM_PROMPT_SENTIMENT,
    SYSTEM_PROMPT_EXTRACT,
    SYSTEM_PROMPT_RESPONSE,
    lookup_routing_rules,
    search_response_templates,
)


def run_classification(
    query: str | None = None,
    model: str = "gpt-4o-mini",
    guard: CostGuard | None = None,
    tracer_provider=None,
    prospect_context=None,
    degraded_output=None,
    trace_quality="good",
    **kwargs,
) -> dict:
    """Execute a CrewAI classification pipeline: guardrails -> classify+extract -> respond."""
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode

    provider = tracer_provider or trace.get_tracer_provider()
    tracer = provider.get_tracer("demo.classification.crewai")
    if not query:
        query = random.choice(QUERIES)

    llm = get_chat_llm(model, temperature=0)

    categories_description = "\n".join(
        f"- {k}: {v['description']}" for k, v in CATEGORIES.items()
    )

    with tracer.start_as_current_span(
        "classification_pipeline",
        attributes={
            "openinference.span.kind": "CHAIN",
            "input.value": query,
            "input.mime_type": "text/plain",
            "metadata.framework": "crewai",
            "metadata.use_case": "classification-routing",
        },
    ) as root_span:

        # === GUARDRAILS ===
        for g in GUARDRAILS:
            run_guardrail(
                tracer, g["name"], query, llm, guard,
                system_prompt=g["system_prompt"],
            )

        # === TOOL CALLS ===
        run_tool_call(tracer, "lookup_routing_rules", query, lookup_routing_rules, guard=guard, category="technical_support")
        run_tool_call(tracer, "search_response_templates", query, search_response_templates, guard=guard, category="technical_support")

        # === CREWAI AGENTS ===
        classifier = Agent(
            role="Ticket Classifier & Analyst",
            goal="Classify customer messages, determine sentiment, and extract key entities",
            backstory=(
                "You are an expert at understanding customer intent. You can quickly "
                "categorize support tickets, gauge customer sentiment, and extract "
                "actionable entities from messages."
            ),
            llm=llm,
            verbose=False,
            allow_delegation=False,
        )

        responder = Agent(
            role="Customer Response Specialist",
            goal="Generate helpful, empathetic customer responses based on classification results",
            backstory=(
                "You are a skilled customer support agent who crafts professional, "
                "empathetic responses tailored to the customer's issue category, "
                "sentiment, and urgency level."
            ),
            llm=llm,
            verbose=False,
            allow_delegation=False,
        )

        # === CREWAI TASKS ===
        classify_and_extract_task = Task(
            description=(
                f"Perform three analyses on the following customer message:\n\n"
                f"Message: {query}\n\n"
                f"1. CLASSIFY the message into one of these categories:\n"
                f"{categories_description}\n\n"
                f"Classification instructions: {SYSTEM_PROMPT_CLASSIFY}\n\n"
                f"2. ANALYZE SENTIMENT:\n{SYSTEM_PROMPT_SENTIMENT}\n\n"
                f"3. EXTRACT ENTITIES:\n{SYSTEM_PROMPT_EXTRACT}\n\n"
                f"Return all three analyses in a structured format."
            ),
            expected_output=(
                "A structured report containing: (1) category classification with confidence, "
                "(2) sentiment analysis with intensity and urgency, "
                "(3) extracted entities including product, issue type, and action requested"
            ),
            agent=classifier,
        )

        generate_response_task = Task(
            description=(
                f"Based on the classification and analysis of the customer message, "
                f"generate a helpful, empathetic response.\n\n"
                f"Original message: {query}\n\n"
                f"Use the classification results from the previous task to craft "
                f"a professional response that acknowledges the customer's concern "
                f"and provides appropriate next steps."
            ),
            expected_output="A professional, empathetic customer response with appropriate next steps",
            agent=responder,
        )

        # === CREWAI CREW ===
        crew = Crew(
            agents=[classifier, responder],
            tasks=[classify_and_extract_task, generate_response_task],
            process=Process.sequential,
            verbose=False,
        )

        if guard:
            guard.check()
        result = crew.kickoff()
        answer = str(result)

        root_span.set_attribute("output.value", answer)
        root_span.set_attribute("output.mime_type", "text/plain")
        root_span.set_status(Status(StatusCode.OK))

    return {
        "query": query,
        "answer": answer,
    }
