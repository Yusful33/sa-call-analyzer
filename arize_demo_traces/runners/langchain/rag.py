"""
LangChain LCEL RAG pipeline with guardrails.
Uses prompt | llm | parser chains, auto-instrumented by LangChainInstrumentor.
"""

import json as _json
import random
from typing import List

from langchain_core.prompts import ChatPromptTemplate

from ...cost_guard import CostGuard
from ...llm import get_chat_llm
from ...trace_enrichment import run_guardrail
from ...use_cases.rag import (
    QUERIES,
    GUARDRAILS,
    SYSTEM_PROMPT,
    get_vectorstore,
    format_docs,
)


def run_rag(
    query: str | None = None,
    model: str = "gpt-4o-mini",
    guard: CostGuard | None = None,
    tracer_provider=None,
) -> dict:
    """Execute a LangChain LCEL RAG pipeline with guardrails, retrieval, and generation."""
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode

    provider = tracer_provider or trace.get_tracer_provider()
    tracer = provider.get_tracer("demo.rag")

    if not query:
        query = random.choice(QUERIES)

    llm = get_chat_llm(model, temperature=0)

    with tracer.start_as_current_span(
        "user_interaction",
        attributes={
            "openinference.span.kind": "CHAIN",
            "input.value": query,
            "input.mime_type": "text/plain",
        },
    ) as pipeline_span:

        # === GUARDRAILS ===
        with tracer.start_as_current_span(
            "validate_interaction",
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

        # === RETRIEVE DOCUMENTS ===
        with tracer.start_as_current_span(
            "retrieve_documents",
            attributes={"openinference.span.kind": "RETRIEVER", "input.value": query},
        ) as step:
            if guard:
                guard.check()
            vectorstore = get_vectorstore()
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            retrieved = retriever.invoke(query)
            context = format_docs(retrieved)
            step.set_attribute("retrieval.documents", _json.dumps(
                [{"content": d.page_content[:200], "source": d.metadata.get("source", "")} for d in retrieved]
            ))
            step.set_attribute("output.value", context[:2000])
            step.set_status(Status(StatusCode.OK))

        # === GENERATE ANSWER ===
        with tracer.start_as_current_span(
            "generate_answer",
            attributes={"openinference.span.kind": "CHAIN", "input.value": query},
        ) as step:
            prompt = ChatPromptTemplate.from_messages([
                ("system", SYSTEM_PROMPT),
                ("human", "{question}"),
            ])
            if guard:
                guard.check()
            messages = prompt.format_messages(context=context, question=query)
            response = llm.invoke(messages)
            answer = response.content if hasattr(response, "content") else str(response)
            step.set_attribute("output.value", answer)
            step.set_status(Status(StatusCode.OK))

        pipeline_span.set_attribute("output.value", answer)
        pipeline_span.set_attribute("output.mime_type", "text/plain")
        pipeline_span.set_status(Status(StatusCode.OK))

    return {
        "query": query,
        "answer": answer,
        "retrieved_docs": [{"content": d.page_content, "source": d.metadata.get("source", "")} for d in retrieved],
    }
