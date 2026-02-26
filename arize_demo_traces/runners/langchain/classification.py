"""
LangChain LCEL classification pipeline with guardrails.
Uses common_runner_utils for query sampling.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from ...cost_guard import CostGuard
from ...llm import get_chat_llm
from ...trace_enrichment import invoke_chain_in_context, run_guardrail, run_tool_call
from ...use_cases.classification import (
    GUARDRAILS,
    SYSTEM_PROMPT_CLASSIFY,
    SYSTEM_PROMPT_SENTIMENT,
    SYSTEM_PROMPT_EXTRACT,
    SYSTEM_PROMPT_RESPONSE,
    lookup_routing_rules,
    search_response_templates,
)
from ..common_runner_utils import get_query_for_run


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
    """Execute a LangChain LCEL classification pipeline with guardrails."""
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode

    from ...use_cases import classification as classification_use_case

    provider = tracer_provider or trace.get_tracer_provider()
    tracer = provider.get_tracer("demo.classification")

    if not query:
        rng = kwargs.get("rng")
        _kw = {k: v for k, v in kwargs.items() if k != "rng"}
        query_spec = get_query_for_run(classification_use_case, prospect_context=prospect_context, rng=rng, **_kw)
        query = query_spec.text
    else:
        query_spec = None

    llm = get_chat_llm(model, temperature=0)

    attrs = {
        "openinference.span.kind": "CHAIN",
        "input.value": query,
        "input.mime_type": "text/plain",
        "metadata.framework": "langchain",
        "metadata.use_case": "classification-routing",
    }
    if query_spec:
        attrs.update(query_spec.to_span_attributes())
    with tracer.start_as_current_span("classification_pipeline", attributes=attrs) as pipeline_span:

        # === GUARDRAILS ===
        with tracer.start_as_current_span(
            "validate_input",
            attributes={"openinference.span.kind": "GUARDRAIL", "input.value": query},
        ) as guard_span:
            for g in GUARDRAILS:
                run_guardrail(
                    tracer, g["name"], query, llm, guard,
                    system_prompt=g["system_prompt"],
                )
            guard_span.set_attribute("output.value", "All checks passed")
            guard_span.set_attribute("guardrail.passed", True)
            guard_span.set_status(Status(StatusCode.OK))

        # === CLASSIFY TICKET ===
        with tracer.start_as_current_span(
            "classify_ticket",
            attributes={"openinference.span.kind": "CHAIN", "input.value": query},
        ) as step:
            classify_prompt = ChatPromptTemplate.from_messages([
                ("system", SYSTEM_PROMPT_CLASSIFY),
                ("human", "{query}"),
            ])
            classify_chain = classify_prompt | llm | StrOutputParser()
            if guard:
                guard.check()
            category = invoke_chain_in_context(classify_chain, {"query": query})
            step.set_attribute("output.value", category)
            step.set_status(Status(StatusCode.OK))

        # === ANALYZE SENTIMENT ===
        with tracer.start_as_current_span(
            "analyze_sentiment",
            attributes={"openinference.span.kind": "CHAIN", "input.value": query},
        ) as step:
            sentiment_prompt = ChatPromptTemplate.from_messages([
                ("system", SYSTEM_PROMPT_SENTIMENT),
                ("human", "{query}"),
            ])
            sentiment_chain = sentiment_prompt | llm | StrOutputParser()
            if guard:
                guard.check()
            sentiment = invoke_chain_in_context(sentiment_chain, {"query": query})
            step.set_attribute("output.value", sentiment)
            step.set_status(Status(StatusCode.OK))

        # === EXTRACT ENTITIES ===
        with tracer.start_as_current_span(
            "extract_entities",
            attributes={"openinference.span.kind": "CHAIN", "input.value": query},
        ) as step:
            extract_prompt = ChatPromptTemplate.from_messages([
                ("system", SYSTEM_PROMPT_EXTRACT),
                ("human", "{query}"),
            ])
            extract_chain = extract_prompt | llm | StrOutputParser()
            if guard:
                guard.check()
            entities = invoke_chain_in_context(extract_chain, {"query": query})
            step.set_attribute("output.value", entities)
            step.set_status(Status(StatusCode.OK))

        # === TOOL CALLS ===
        run_tool_call(tracer, "lookup_routing_rules", category, lookup_routing_rules, guard=guard, category="technical_support")
        run_tool_call(tracer, "search_response_templates", category, search_response_templates, guard=guard, category="technical_support")

        # === GENERATE RESPONSE ===
        with tracer.start_as_current_span(
            "generate_response",
            attributes={"openinference.span.kind": "CHAIN", "input.value": query},
        ) as step:
            response_prompt = ChatPromptTemplate.from_messages([
                ("system", SYSTEM_PROMPT_RESPONSE),
                ("human", "{query}"),
            ])
            response_chain = response_prompt | llm | StrOutputParser()
            if guard:
                guard.check()
            response = invoke_chain_in_context(response_chain, {
                "query": query,
                "classification": category,
                "sentiment": sentiment,
                "entities": entities,
            })
            step.set_attribute("output.value", response)
            step.set_status(Status(StatusCode.OK))

        pipeline_span.set_attribute("output.value", response)
        pipeline_span.set_attribute("output.mime_type", "text/plain")
        pipeline_span.set_status(Status(StatusCode.OK))

    return {
        "query": query,
        "category": category,
        "sentiment": sentiment,
        "entities": entities,
        "response": response,
    }
