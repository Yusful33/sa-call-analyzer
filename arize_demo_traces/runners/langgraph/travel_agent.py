"""
LangGraph travel agent pipeline with guardrails and tool calls.
"""

import random
from typing import TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END

from ...cost_guard import CostGuard
from ...llm import get_chat_llm
from ...trace_enrichment import run_guardrail, run_tool_call
from ...use_cases.travel_agent import (
    QUERIES,
    GUARDRAILS,
    SYSTEM_PROMPT,
    flight_search,
    hotel_search,
)


class TravelAgentState(TypedDict):
    query: str
    answer: str
    guardrail_passed: bool


def run_travel_agent(
    query: str | None = None,
    model: str = "gpt-4o-mini",
    guard: CostGuard | None = None,
    tracer_provider=None,
    prospect_context=None,
) -> dict:
    """Execute a LangGraph travel agent pipeline: guardrails -> tools -> generate."""
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode

    provider = tracer_provider or trace.get_tracer_provider()
    tracer = provider.get_tracer("demo.travel_agent.langgraph")

    if not query:
        query = random.choice(QUERIES)

    llm = get_chat_llm(model, temperature=0)

    def guardrails_node(state: TravelAgentState) -> dict:
        for g in GUARDRAILS:
            run_guardrail(
                tracer, g["name"], state["query"], llm, guard,
                system_prompt=g["system_prompt"],
            )
        return {"guardrail_passed": True}

    def generate_node(state: TravelAgentState) -> dict:
        run_tool_call(
            tracer, "flight_search", state["query"],
            lambda: flight_search("NYC", "Paris", "2025-03-15"),
            guard=guard,
        )
        run_tool_call(
            tracer, "hotel_search", state["query"],
            lambda: hotel_search("Paris", "2025-03-15", "2025-03-17", 2),
            guard=guard,
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "{question}"),
        ])
        chain = prompt | llm | StrOutputParser()
        if guard:
            guard.check()
        answer = chain.invoke({"question": state["query"]})
        return {"answer": answer}

    workflow = StateGraph(TravelAgentState)
    workflow.add_node("guardrails", guardrails_node)
    workflow.add_node("generate", generate_node)
    workflow.set_entry_point("guardrails")
    workflow.add_edge("guardrails", "generate")
    workflow.add_edge("generate", END)
    graph = workflow.compile()

    with tracer.start_as_current_span(
        "travel_agent_pipeline",
        attributes={
            "openinference.span.kind": "CHAIN",
            "input.value": query,
            "input.mime_type": "text/plain",
            "metadata.framework": "langgraph",
            "metadata.use_case": "travel-agent",
        },
    ) as span:
        result = graph.invoke({
            "query": query,
            "answer": "",
            "guardrail_passed": False,
        })
        span.set_attribute("output.value", result["answer"])
        span.set_attribute("output.mime_type", "text/plain")
        span.set_status(Status(StatusCode.OK))

    return {
        "query": query,
        "answer": result["answer"],
    }
