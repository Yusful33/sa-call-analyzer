"""
LangChain LCEL generic LLM pipeline with guardrails.
Uses prompt | llm | parser chains, auto-instrumented by LangChainInstrumentor.
"""

import random

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

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
) -> dict:
    """Execute a LangChain LCEL generic pipeline with guardrail and generation."""
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode

    provider = tracer_provider or trace.get_tracer_provider()
    tracer = provider.get_tracer("demo.generic")

    if not query:
        query = random.choice(QUERIES)

    llm = get_chat_llm(model, temperature=0)

    with tracer.start_as_current_span(
        "llm_pipeline",
        attributes={
            "openinference.span.kind": "CHAIN",
            "input.value": query,
            "input.mime_type": "text/plain",
            "metadata.framework": "langchain",
            "metadata.use_case": "generic",
        },
    ) as span:

        # === GUARDRAIL ===
        for g in GUARDRAILS:
            run_guardrail(
                tracer, g["name"], query, llm, guard,
                system_prompt=g["system_prompt"],
            )

        # === TOOL CALLS ===
        run_tool_call(tracer, "web_search", query, web_search, guard=guard, query=query)
        run_tool_call(tracer, "get_current_context", query, get_current_context, guard=guard)

        # === GENERATE RESPONSE ===
        with tracer.start_as_current_span(
            "generate_response",
            attributes={"openinference.span.kind": "CHAIN", "input.value": query},
        ) as step:
            prompt = ChatPromptTemplate.from_messages([
                ("system", SYSTEM_PROMPT),
                ("human", "{question}"),
            ])
            chain = prompt | llm | StrOutputParser()
            if guard:
                guard.check()
            answer = chain.invoke({"question": query})
            step.set_attribute("output.value", answer)
            step.set_status(Status(StatusCode.OK))

        span.set_attribute("output.value", answer)
        span.set_attribute("output.mime_type", "text/plain")
        span.set_status(Status(StatusCode.OK))

    return {
        "query": query,
        "answer": answer,
    }
