"""
ADK-style multi-agent orchestration with manual OpenTelemetry spans.
Mimics Google Agent Development Kit trace patterns without requiring ADK installation.
Uses real LLM calls inside manually created OTel spans with OpenInference attributes.

Trace structure:
  orchestrator_agent.run (AGENT)
    -> guardrails (GUARDRAIL group)
    -> generate_content [plan] (LLM)
    -> research_agent.run (AGENT)
        -> generate_content [research] (LLM)
    -> analysis_agent.run (AGENT)
        -> generate_content [analyze] (LLM)
    -> writer_agent.run (AGENT)
        -> generate_content [write] (LLM)
    -> review_agent.run (AGENT)
        -> generate_content [review] (LLM)
"""

import random

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from ...cost_guard import CostGuard
from ...llm import get_chat_llm
from ...trace_enrichment import run_guardrail, run_tool_call
from ...use_cases.multi_agent import (
    QUERIES,
    GUARDRAILS,
    SUPERVISOR_PROMPT,
    RESEARCH_PROMPT,
    ANALYSIS_PROMPT,
    WRITER_PROMPT,
    REVIEWER_PROMPT,
    search_web,
    analyze_metrics,
)


def run_multi_agent(
    query: str | None = None,
    model: str = "gpt-4o-mini",
    guard: CostGuard | None = None,
    tracer_provider=None,
    prospect_context=None,
) -> dict:
    """Execute an ADK-style multi-agent orchestration: plan -> research -> analyze -> write -> review."""
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode

    provider = tracer_provider or trace.get_tracer_provider()
    tracer = provider.get_tracer("demo.multi_agent.adk")

    if not query:
        query = random.choice(QUERIES)

    llm = get_chat_llm(model, temperature=0)

    with tracer.start_as_current_span(
        "orchestrator_agent.run",
        attributes={
            "openinference.span.kind": "AGENT",
            "input.value": query,
            "input.mime_type": "text/plain",
            "metadata.framework": "adk",
            "metadata.use_case": "multi-agent-orchestration",
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
                ("system", SUPERVISOR_PROMPT),
                ("human", "{input}"),
            ])
            plan_chain = plan_prompt | llm | StrOutputParser()
            if guard:
                guard.check()
            plan = plan_chain.invoke({"input": query})
            plan_span.set_attribute("output.value", plan)
            plan_span.set_attribute("output.mime_type", "text/plain")
            plan_span.set_status(Status(StatusCode.OK))

        # ---- research_agent.run ----
        with tracer.start_as_current_span(
            "research_agent.run",
            attributes={
                "openinference.span.kind": "AGENT",
                "input.value": query,
                "input.mime_type": "text/plain",
            },
        ) as research_agent_span:
            run_tool_call(tracer, "search_web", query, search_web, guard=guard, query=query)
            with tracer.start_as_current_span(
                "generate_content",
                attributes={
                    "openinference.span.kind": "LLM",
                    "input.value": query,
                    "input.mime_type": "text/plain",
                },
            ) as research_llm_span:
                research_prompt = ChatPromptTemplate.from_messages([
                    ("system", RESEARCH_PROMPT),
                    ("human", "Task plan:\n{plan}\n\nOriginal request: {input}"),
                ])
                research_chain = research_prompt | llm | StrOutputParser()
                if guard:
                    guard.check()
                research_output = research_chain.invoke({"plan": plan, "input": query})
                research_llm_span.set_attribute("output.value", research_output)
                research_llm_span.set_attribute("output.mime_type", "text/plain")
                research_llm_span.set_status(Status(StatusCode.OK))
            research_agent_span.set_attribute("output.value", research_output)
            research_agent_span.set_attribute("output.mime_type", "text/plain")
            research_agent_span.set_status(Status(StatusCode.OK))

        # ---- analysis_agent.run ----
        with tracer.start_as_current_span(
            "analysis_agent.run",
            attributes={
                "openinference.span.kind": "AGENT",
                "input.value": research_output[:1000],
                "input.mime_type": "text/plain",
            },
        ) as analysis_agent_span:
            run_tool_call(tracer, "analyze_metrics", query, analyze_metrics, guard=guard, metric_name="performance")
            with tracer.start_as_current_span(
                "generate_content",
                attributes={
                    "openinference.span.kind": "LLM",
                    "input.value": research_output[:1000],
                    "input.mime_type": "text/plain",
                },
            ) as analysis_llm_span:
                analysis_prompt = ChatPromptTemplate.from_messages([
                    ("system", ANALYSIS_PROMPT),
                    ("human", "Research findings:\n{research}\n\nOriginal request: {input}"),
                ])
                analysis_chain = analysis_prompt | llm | StrOutputParser()
                if guard:
                    guard.check()
                analysis_output = analysis_chain.invoke({"research": research_output, "input": query})
                analysis_llm_span.set_attribute("output.value", analysis_output)
                analysis_llm_span.set_attribute("output.mime_type", "text/plain")
                analysis_llm_span.set_status(Status(StatusCode.OK))
            analysis_agent_span.set_attribute("output.value", analysis_output)
            analysis_agent_span.set_attribute("output.mime_type", "text/plain")
            analysis_agent_span.set_status(Status(StatusCode.OK))

        # ---- writer_agent.run ----
        with tracer.start_as_current_span(
            "writer_agent.run",
            attributes={
                "openinference.span.kind": "AGENT",
                "input.value": analysis_output[:1000],
                "input.mime_type": "text/plain",
            },
        ) as writer_agent_span:
            with tracer.start_as_current_span(
                "generate_content",
                attributes={
                    "openinference.span.kind": "LLM",
                    "input.value": analysis_output[:1000],
                    "input.mime_type": "text/plain",
                },
            ) as writer_llm_span:
                writer_prompt = ChatPromptTemplate.from_messages([
                    ("system", WRITER_PROMPT),
                    ("human", "Analysis:\n{analysis}\n\nResearch:\n{research}\n\nOriginal request: {input}"),
                ])
                writer_chain = writer_prompt | llm | StrOutputParser()
                if guard:
                    guard.check()
                writer_output = writer_chain.invoke({
                    "analysis": analysis_output,
                    "research": research_output,
                    "input": query,
                })
                writer_llm_span.set_attribute("output.value", writer_output)
                writer_llm_span.set_attribute("output.mime_type", "text/plain")
                writer_llm_span.set_status(Status(StatusCode.OK))
            writer_agent_span.set_attribute("output.value", writer_output)
            writer_agent_span.set_attribute("output.mime_type", "text/plain")
            writer_agent_span.set_status(Status(StatusCode.OK))

        # ---- review_agent.run ----
        with tracer.start_as_current_span(
            "review_agent.run",
            attributes={
                "openinference.span.kind": "AGENT",
                "input.value": writer_output[:1000],
                "input.mime_type": "text/plain",
            },
        ) as review_agent_span:
            with tracer.start_as_current_span(
                "generate_content",
                attributes={
                    "openinference.span.kind": "LLM",
                    "input.value": writer_output[:1000],
                    "input.mime_type": "text/plain",
                },
            ) as review_llm_span:
                review_prompt = ChatPromptTemplate.from_messages([
                    ("system", REVIEWER_PROMPT),
                    ("human", "Document to review:\n{document}\n\nOriginal request: {input}"),
                ])
                review_chain = review_prompt | llm | StrOutputParser()
                if guard:
                    guard.check()
                review_output = review_chain.invoke({"document": writer_output, "input": query})
                review_llm_span.set_attribute("output.value", review_output)
                review_llm_span.set_attribute("output.mime_type", "text/plain")
                review_llm_span.set_status(Status(StatusCode.OK))
            review_agent_span.set_attribute("output.value", review_output)
            review_agent_span.set_attribute("output.mime_type", "text/plain")
            review_agent_span.set_status(Status(StatusCode.OK))

        answer = review_output

        agent_span.set_attribute("output.value", answer)
        agent_span.set_attribute("output.mime_type", "text/plain")
        agent_span.set_status(Status(StatusCode.OK))

    return {
        "query": query,
        "answer": answer,
    }
