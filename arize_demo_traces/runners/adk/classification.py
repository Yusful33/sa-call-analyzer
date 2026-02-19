"""
ADK-style classification agent with manual OpenTelemetry spans.
Mimics Google Agent Development Kit trace patterns without requiring ADK installation.
Uses real LLM calls inside manually created OTel spans with OpenInference attributes.

Trace structure:
  classifier_agent.run (AGENT)
    -> guardrails (GUARDRAIL group)
    -> generate_content [classify] (LLM)
    -> generate_content [sentiment] (LLM)
    -> generate_content [extract] (LLM)
    -> generate_content [respond] (LLM)
"""

import random

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from ...cost_guard import CostGuard
from ...llm import get_chat_llm
from ...trace_enrichment import run_guardrail, run_tool_call
from ...use_cases.classification import (
    QUERIES,
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
) -> dict:
    """Execute an ADK-style classification agent: guardrails -> classify -> sentiment -> extract -> respond."""
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode

    provider = tracer_provider or trace.get_tracer_provider()
    tracer = provider.get_tracer("demo.classification.adk")

    if not query:
        query = random.choice(QUERIES)

    llm = get_chat_llm(model, temperature=0)

    with tracer.start_as_current_span(
        "classifier_agent.run",
        attributes={
            "openinference.span.kind": "AGENT",
            "input.value": query,
            "input.mime_type": "text/plain",
            "metadata.framework": "adk",
            "metadata.use_case": "classification-routing",
        },
    ) as agent_span:

        # ---- Guardrails ----
        for g in GUARDRAILS:
            run_guardrail(
                tracer, g["name"], query, llm, guard,
                system_prompt=g["system_prompt"],
            )

        # ---- generate_content: classify ----
        with tracer.start_as_current_span(
            "generate_content",
            attributes={
                "openinference.span.kind": "LLM",
                "input.value": query,
                "input.mime_type": "text/plain",
            },
        ) as classify_span:
            classify_prompt = ChatPromptTemplate.from_messages([
                ("system", SYSTEM_PROMPT_CLASSIFY),
                ("human", "{input}"),
            ])
            classify_chain = classify_prompt | llm | StrOutputParser()
            if guard:
                guard.check()
            classification = classify_chain.invoke({"input": query})
            classify_span.set_attribute("output.value", classification)
            classify_span.set_attribute("output.mime_type", "text/plain")
            classify_span.set_status(Status(StatusCode.OK))

        # ---- generate_content: sentiment ----
        with tracer.start_as_current_span(
            "generate_content",
            attributes={
                "openinference.span.kind": "LLM",
                "input.value": query,
                "input.mime_type": "text/plain",
            },
        ) as sentiment_span:
            sentiment_prompt = ChatPromptTemplate.from_messages([
                ("system", SYSTEM_PROMPT_SENTIMENT),
                ("human", "{input}"),
            ])
            sentiment_chain = sentiment_prompt | llm | StrOutputParser()
            if guard:
                guard.check()
            sentiment = sentiment_chain.invoke({"input": query})
            sentiment_span.set_attribute("output.value", sentiment)
            sentiment_span.set_attribute("output.mime_type", "text/plain")
            sentiment_span.set_status(Status(StatusCode.OK))

        # ---- generate_content: extract entities ----
        with tracer.start_as_current_span(
            "generate_content",
            attributes={
                "openinference.span.kind": "LLM",
                "input.value": query,
                "input.mime_type": "text/plain",
            },
        ) as extract_span:
            extract_prompt = ChatPromptTemplate.from_messages([
                ("system", SYSTEM_PROMPT_EXTRACT),
                ("human", "{input}"),
            ])
            extract_chain = extract_prompt | llm | StrOutputParser()
            if guard:
                guard.check()
            entities = extract_chain.invoke({"input": query})
            extract_span.set_attribute("output.value", entities)
            extract_span.set_attribute("output.mime_type", "text/plain")
            extract_span.set_status(Status(StatusCode.OK))

        # ---- Tool calls: routing rules and response templates ----
        run_tool_call(tracer, "lookup_routing_rules", classification, lookup_routing_rules, guard=guard, category="technical_support")
        run_tool_call(tracer, "search_response_templates", classification, search_response_templates, guard=guard, category="technical_support")

        # ---- generate_content: respond ----
        with tracer.start_as_current_span(
            "generate_content",
            attributes={
                "openinference.span.kind": "LLM",
                "input.value": query,
                "input.mime_type": "text/plain",
            },
        ) as respond_span:
            respond_prompt = ChatPromptTemplate.from_messages([
                ("system", SYSTEM_PROMPT_RESPONSE.format(
                    classification=classification,
                    sentiment=sentiment,
                    entities=entities,
                )),
                ("human", "{input}"),
            ])
            respond_chain = respond_prompt | llm | StrOutputParser()
            if guard:
                guard.check()
            response = respond_chain.invoke({"input": query})
            respond_span.set_attribute("output.value", response)
            respond_span.set_attribute("output.mime_type", "text/plain")
            respond_span.set_status(Status(StatusCode.OK))

        answer = response

        agent_span.set_attribute("output.value", answer)
        agent_span.set_attribute("output.mime_type", "text/plain")
        agent_span.set_status(Status(StatusCode.OK))

    return {
        "query": query,
        "answer": answer,
        "classification": classification,
        "sentiment": sentiment,
        "entities": entities,
    }
