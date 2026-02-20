"""
ADK-style RAG agent with manual OpenTelemetry spans.
Mimics Google Agent Development Kit trace patterns without requiring ADK installation.
Uses real LLM calls inside manually created OTel spans with OpenInference attributes.

Trace structure:
  rag_agent.run (AGENT)
    -> guardrails (GUARDRAIL group)
    -> generate_content (LLM) - planning step
    -> retrieve_documents (RETRIEVER) - vectorstore retrieval
    -> generate_content (LLM) - generate answer from context
"""

import random

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from ...cost_guard import CostGuard
from ...llm import get_chat_llm
from ...trace_enrichment import run_guardrail, run_tool_call
from ...use_cases.rag import (
    QUERIES,
    GUARDRAILS,
    SYSTEM_PROMPT,
    get_vectorstore,
    format_docs,
    search_documents,
    fetch_document_metadata,
)


def run_rag(
    query: str | None = None,
    model: str = "gpt-4o-mini",
    guard: CostGuard | None = None,
    tracer_provider=None,
    prospect_context=None,
    degraded_output=None,
    trace_quality="good",
    **kwargs,
) -> dict:
    """Execute an ADK-style RAG agent: guardrails -> plan -> retrieve -> generate."""
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode

    provider = tracer_provider or trace.get_tracer_provider()
    tracer = provider.get_tracer("demo.rag.adk")

    if not query:
        query = random.choice(QUERIES)

    llm = get_chat_llm(model, temperature=0)

    with tracer.start_as_current_span(
        "rag_agent.run",
        attributes={
            "openinference.span.kind": "AGENT",
            "input.value": query,
            "input.mime_type": "text/plain",
            "metadata.framework": "adk",
            "metadata.use_case": "retrieval-augmented-search",
        },
    ) as agent_span:

        # ---- Guardrails ----
        for g in GUARDRAILS:
            run_guardrail(
                tracer, g["name"], query, llm, guard,
                system_prompt=g["system_prompt"],
            )

        # ---- generate_content: planning step ----
        with tracer.start_as_current_span(
            "generate_content",
            attributes={
                "openinference.span.kind": "LLM",
                "input.value": query,
                "input.mime_type": "text/plain",
            },
        ) as plan_span:
            plan_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a RAG planning agent. Given the user question, describe what information you need to retrieve to answer it. Be brief."),
                ("human", "{input}"),
            ])
            plan_chain = plan_prompt | llm | StrOutputParser()
            if guard:
                guard.check()
            plan = plan_chain.invoke({"input": query})
            plan_span.set_attribute("output.value", plan)
            plan_span.set_attribute("output.mime_type", "text/plain")
            plan_span.set_status(Status(StatusCode.OK))

        # ---- Tool calls: document search and metadata ----
        run_tool_call(tracer, "search_documents", query, search_documents, guard=guard, query=query)
        run_tool_call(tracer, "fetch_document_metadata", "knowledge-base", fetch_document_metadata, guard=guard, source="knowledge-base")

        # ---- retrieve_documents: vectorstore retrieval ----
        with tracer.start_as_current_span(
            "retrieve_documents",
            attributes={
                "openinference.span.kind": "RETRIEVER",
                "input.value": query,
                "input.mime_type": "text/plain",
            },
        ) as retriever_span:
            vectorstore = get_vectorstore()
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            docs = retriever.invoke(query)
            context = format_docs(docs)
            retriever_span.set_attribute("output.value", context[:1000])
            retriever_span.set_attribute("output.mime_type", "text/plain")
            retriever_span.set_attribute("retrieval.documents", len(docs))
            retriever_span.set_status(Status(StatusCode.OK))

        # ---- generate_content: answer generation ----
        with tracer.start_as_current_span(
            "generate_content",
            attributes={
                "openinference.span.kind": "LLM",
                "input.value": query,
                "input.mime_type": "text/plain",
            },
        ) as answer_span:
            answer_prompt = ChatPromptTemplate.from_messages([
                ("system", SYSTEM_PROMPT),
                ("human", "{question}"),
            ])
            if guard:
                guard.check()
            messages = answer_prompt.format_messages(context=context, question=query)
            response = llm.invoke(messages)
            answer = response.content if hasattr(response, "content") else str(response)
            answer_span.set_attribute("output.value", answer)
            answer_span.set_attribute("output.mime_type", "text/plain")
            answer_span.set_status(Status(StatusCode.OK))

        agent_span.set_attribute("output.value", answer)
        agent_span.set_attribute("output.mime_type", "text/plain")
        agent_span.set_status(Status(StatusCode.OK))

    return {
        "query": query,
        "answer": answer,
        "retrieved_docs": [
            {"content": d.page_content, "source": d.metadata.get("source", "")}
            for d in docs
        ],
    }
