"""
LangGraph multi-agent orchestration pipeline with guardrails.
Uses a real StateGraph with supervisor planning, specialized agent nodes, and edges.
Auto-instrumented by LangChainInstrumentor for authentic LangGraph trace patterns.
"""

import random
from typing import TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

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


class MultiAgentState(TypedDict):
    query: str
    research_output: str
    analysis_output: str
    draft: str
    final_output: str
    guardrail_passed: bool


def run_multi_agent(
    query: str | None = None,
    model: str = "gpt-4o-mini",
    guard: CostGuard | None = None,
    tracer_provider=None,
    prospect_context=None,
    degraded_output=None,
    trace_quality="good",
    **kwargs,
) -> dict:
    """Execute a LangGraph multi-agent pipeline: guardrails -> supervisor -> research -> analysis -> writing -> review."""
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode

    provider = tracer_provider or trace.get_tracer_provider()
    tracer = provider.get_tracer("demo.multi_agent.langgraph")

    if not query:
        query = random.choice(QUERIES)

    llm = get_chat_llm(model, temperature=0)

    # --- Node functions (closures capturing llm, guard, tracer) ---

    def guardrails_node(state: MultiAgentState) -> dict:
        for g in GUARDRAILS:
            run_guardrail(
                tracer, g["name"], state["query"], llm, guard,
                system_prompt=g["system_prompt"],
            )
        return {"guardrail_passed": True}

    def supervisor_node(state: MultiAgentState) -> dict:
        prompt = ChatPromptTemplate.from_messages([
            ("system", SUPERVISOR_PROMPT),
            ("human", "{query}"),
        ])
        if guard:
            guard.check()
        messages = prompt.format_messages(query=state["query"])
        response = llm.invoke(messages)
        plan = response.content if hasattr(response, "content") else str(response)
        return {}

    def research_node(state: MultiAgentState) -> dict:
        run_tool_call(tracer, "search_web", state["query"],
                      search_web, guard=guard, query=state["query"])
        prompt = ChatPromptTemplate.from_messages([
            ("system", RESEARCH_PROMPT),
            ("human", "Task: {query}"),
        ])
        if guard:
            guard.check()
        messages = prompt.format_messages(query=state["query"])
        response = llm.invoke(messages)
        research = response.content if hasattr(response, "content") else str(response)
        return {"research_output": research}

    def analysis_node(state: MultiAgentState) -> dict:
        run_tool_call(tracer, "analyze_metrics", state["query"],
                      analyze_metrics, guard=guard, metric_name="performance")
        prompt = ChatPromptTemplate.from_messages([
            ("system", ANALYSIS_PROMPT),
            ("human", "Original task: {query}\n\nResearch findings:\n{research}"),
        ])
        if guard:
            guard.check()
        messages = prompt.format_messages(query=state["query"], research=state["research_output"])
        response = llm.invoke(messages)
        analysis = response.content if hasattr(response, "content") else str(response)
        return {"analysis_output": analysis}

    def writing_node(state: MultiAgentState) -> dict:
        prompt = ChatPromptTemplate.from_messages([
            ("system", WRITER_PROMPT),
            ("human", "Original task: {query}\n\nResearch:\n{research}\n\nAnalysis:\n{analysis}"),
        ])
        if guard:
            guard.check()
        messages = prompt.format_messages(
            query=state["query"],
            research=state["research_output"],
            analysis=state["analysis_output"],
        )
        response = llm.invoke(messages)
        draft = response.content if hasattr(response, "content") else str(response)
        return {"draft": draft}

    def review_node(state: MultiAgentState) -> dict:
        prompt = ChatPromptTemplate.from_messages([
            ("system", REVIEWER_PROMPT),
            ("human", "Original task: {query}\n\nDraft document:\n{draft}"),
        ])
        if guard:
            guard.check()
        messages = prompt.format_messages(query=state["query"], draft=state["draft"])
        response = llm.invoke(messages)
        final = response.content if hasattr(response, "content") else str(response)
        return {"final_output": final}

    # --- Build graph ---
    workflow = StateGraph(MultiAgentState)
    workflow.add_node("guardrails", guardrails_node)
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("research", research_node)
    workflow.add_node("analysis", analysis_node)
    workflow.add_node("writing", writing_node)
    workflow.add_node("review", review_node)
    workflow.set_entry_point("guardrails")
    workflow.add_edge("guardrails", "supervisor")
    workflow.add_edge("supervisor", "research")
    workflow.add_edge("research", "analysis")
    workflow.add_edge("analysis", "writing")
    workflow.add_edge("writing", "review")
    workflow.add_edge("review", END)
    graph = workflow.compile()

    # --- Execute with root span ---
    with tracer.start_as_current_span(
        "multi_agent_orchestration",
        attributes={
            "openinference.span.kind": "AGENT",
            "input.value": query,
            "input.mime_type": "text/plain",
            "metadata.framework": "langgraph",
            "metadata.use_case": "multi-agent-orchestration",
        },
    ) as span:
        result = graph.invoke({
            "query": query,
            "research_output": "",
            "analysis_output": "",
            "draft": "",
            "final_output": "",
            "guardrail_passed": False,
        })
        span.set_attribute("output.value", result["final_output"])
        span.set_attribute("output.mime_type", "text/plain")
        span.set_status(Status(StatusCode.OK))

    return {
        "query": query,
        "final_output": result["final_output"],
        "research_output": result["research_output"],
        "analysis_output": result["analysis_output"],
        "draft": result["draft"],
    }
