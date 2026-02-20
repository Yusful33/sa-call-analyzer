"""
CrewAI RAG pipeline with retrieval and synthesis agents.
Uses real CrewAI Crew, Agent, Task objects to generate authentic trace patterns
when instrumented by CrewAIInstrumentor.
"""

import json as _json
import random

from crewai import Agent, Crew, Task, Process

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
    """Execute a CrewAI RAG pipeline with retrieval agent, synthesis agent, guardrails, and evaluation."""
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode

    provider = tracer_provider or trace.get_tracer_provider()
    tracer = provider.get_tracer("demo.rag.crewai")
    if not query:
        query = random.choice(QUERIES)

    llm = get_chat_llm(model, temperature=0)

    with tracer.start_as_current_span(
        "rag_pipeline",
        attributes={
            "openinference.span.kind": "CHAIN",
            "input.value": query,
            "input.mime_type": "text/plain",
            "metadata.framework": "crewai",
            "metadata.use_case": "retrieval-augmented-search",
        },
    ) as pipeline_span:

        # === GUARDRAILS ===
        for g in GUARDRAILS:
            run_guardrail(
                tracer, g["name"], query, llm, guard,
                system_prompt=g["system_prompt"],
            )

        # === TOOL CALLS ===
        run_tool_call(tracer, "search_documents", query, search_documents, guard=guard, query=query)
        run_tool_call(tracer, "fetch_document_metadata", "knowledge-base", fetch_document_metadata, guard=guard, source="knowledge-base")

        # === RETRIEVE DOCUMENTS (outside CrewAI since agents can't use vectorstores directly) ===
        with tracer.start_as_current_span(
            "retrieve_documents",
            attributes={"openinference.span.kind": "RETRIEVER", "input.value": query},
        ) as retrieval_span:
            if guard:
                guard.check()
            vectorstore = get_vectorstore()
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            retrieved = retriever.invoke(query)
            context = format_docs(retrieved)
            retrieval_span.set_attribute("retrieval.documents", _json.dumps(
                [{"content": d.page_content[:200], "source": d.metadata.get("source", "")} for d in retrieved]
            ))
            retrieval_span.set_attribute("output.value", context[:2000])
            retrieval_span.set_status(Status(StatusCode.OK))

        # === CREWAI AGENTS ===
        retrieval_agent = Agent(
            role="Document Retrieval Specialist",
            goal="Find and organize the most relevant documents for answering user questions",
            backstory=(
                "You are an expert at analyzing retrieved documents and identifying "
                "the most relevant passages for answering specific questions. You excel "
                "at filtering noise and highlighting key information."
            ),
            llm=llm,
            verbose=False,
            allow_delegation=False,
        )

        synthesis_agent = Agent(
            role="Answer Synthesis Specialist",
            goal="Synthesize accurate, well-grounded answers from provided context documents",
            backstory=(
                "You are a skilled technical writer who excels at synthesizing information "
                "from multiple sources into clear, concise, and accurate answers. You always "
                "ground your responses in the provided context."
            ),
            llm=llm,
            verbose=False,
            allow_delegation=False,
        )

        # === CREWAI TASKS ===
        retrieve_task = Task(
            description=(
                f"Analyze the following retrieved documents and identify the most relevant "
                f"passages for answering the question.\n\n"
                f"Question: {query}\n\n"
                f"Retrieved Documents:\n{context}"
            ),
            expected_output="A summary of the most relevant information from the documents that addresses the question",
            agent=retrieval_agent,
        )

        synthesize_task = Task(
            description=(
                f"Answer the question using this context:\n\n{context}\n\nQuestion: {query}"
            ),
            expected_output="A clear, concise answer grounded in the provided context",
            agent=synthesis_agent,
        )

        # === CREWAI CREW ===
        crew = Crew(
            agents=[retrieval_agent, synthesis_agent],
            tasks=[retrieve_task, synthesize_task],
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
        "retrieved_docs": [{"content": d.page_content, "source": d.metadata.get("source", "")} for d in retrieved],
    }
