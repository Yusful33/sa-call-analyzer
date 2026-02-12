"""
LangChain LCEL multi-agent orchestration pipeline with guardrails.
Uses manual OTel spans with LCEL chains (prompt | llm | StrOutputParser) for each agent.
Auto-instrumented by LangChainInstrumentor for authentic LangChain trace patterns.
"""

import random

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from ...cost_guard import CostGuard
from ...llm import get_chat_llm
from ...trace_enrichment import run_guardrail
from ...use_cases.multi_agent import (
    QUERIES,
    GUARDRAILS,
    SUPERVISOR_PROMPT,
    RESEARCH_PROMPT,
    ANALYSIS_PROMPT,
    WRITER_PROMPT,
    REVIEWER_PROMPT,
)


def run_multi_agent(
    query: str | None = None,
    model: str = "gpt-4o-mini",
    guard: CostGuard | None = None,
    tracer_provider=None,
) -> dict:
    """Execute a LangChain LCEL multi-agent orchestration with guardrails."""
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode

    provider = tracer_provider or trace.get_tracer_provider()
    tracer = provider.get_tracer("demo.multi_agent")

    if not query:
        query = random.choice(QUERIES)

    llm = get_chat_llm(model, temperature=0)

    with tracer.start_as_current_span(
        "multi_agent_orchestration",
        attributes={
            "openinference.span.kind": "AGENT",
            "input.value": query,
            "input.mime_type": "text/plain",
        },
    ) as agent_span:

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

        # === SUPERVISOR PLANNING ===
        with tracer.start_as_current_span(
            "supervisor_planning",
            attributes={"openinference.span.kind": "CHAIN", "input.value": query},
        ) as step:
            supervisor_prompt = ChatPromptTemplate.from_messages([
                ("system", SUPERVISOR_PROMPT),
                ("human", "{query}"),
            ])
            supervisor_chain = supervisor_prompt | llm | StrOutputParser()
            if guard:
                guard.check()
            plan = supervisor_chain.invoke({"query": query})
            step.set_attribute("output.value", plan)
            step.set_status(Status(StatusCode.OK))

        # === RESEARCH AGENT ===
        with tracer.start_as_current_span(
            "research_agent",
            attributes={"openinference.span.kind": "AGENT", "input.value": query},
        ) as step:
            research_prompt = ChatPromptTemplate.from_messages([
                ("system", RESEARCH_PROMPT),
                ("human", "Task: {query}"),
            ])
            research_chain = research_prompt | llm | StrOutputParser()
            if guard:
                guard.check()
            research_output = research_chain.invoke({"query": query})
            step.set_attribute("output.value", research_output)
            step.set_status(Status(StatusCode.OK))

        # === ANALYSIS AGENT ===
        with tracer.start_as_current_span(
            "analysis_agent",
            attributes={"openinference.span.kind": "AGENT", "input.value": query},
        ) as step:
            analysis_prompt = ChatPromptTemplate.from_messages([
                ("system", ANALYSIS_PROMPT),
                ("human", "Original task: {query}\n\nResearch findings:\n{research}"),
            ])
            analysis_chain = analysis_prompt | llm | StrOutputParser()
            if guard:
                guard.check()
            analysis_output = analysis_chain.invoke({"query": query, "research": research_output})
            step.set_attribute("output.value", analysis_output)
            step.set_status(Status(StatusCode.OK))

        # === WRITER AGENT ===
        with tracer.start_as_current_span(
            "writer_agent",
            attributes={"openinference.span.kind": "AGENT", "input.value": query},
        ) as step:
            writer_prompt = ChatPromptTemplate.from_messages([
                ("system", WRITER_PROMPT),
                ("human", "Original task: {query}\n\nResearch:\n{research}\n\nAnalysis:\n{analysis}"),
            ])
            writer_chain = writer_prompt | llm | StrOutputParser()
            if guard:
                guard.check()
            draft = writer_chain.invoke({
                "query": query,
                "research": research_output,
                "analysis": analysis_output,
            })
            step.set_attribute("output.value", draft)
            step.set_status(Status(StatusCode.OK))

        # === REVIEW AGENT ===
        with tracer.start_as_current_span(
            "review_agent",
            attributes={"openinference.span.kind": "AGENT", "input.value": query},
        ) as step:
            reviewer_prompt = ChatPromptTemplate.from_messages([
                ("system", REVIEWER_PROMPT),
                ("human", "Original task: {query}\n\nDraft document:\n{draft}"),
            ])
            reviewer_chain = reviewer_prompt | llm | StrOutputParser()
            if guard:
                guard.check()
            final_output = reviewer_chain.invoke({"query": query, "draft": draft})
            step.set_attribute("output.value", final_output)
            step.set_status(Status(StatusCode.OK))

        agent_span.set_attribute("output.value", final_output)
        agent_span.set_attribute("output.mime_type", "text/plain")
        agent_span.set_status(Status(StatusCode.OK))

    return {
        "query": query,
        "final_output": final_output,
        "research_output": research_output,
        "analysis_output": analysis_output,
        "draft": draft,
    }
