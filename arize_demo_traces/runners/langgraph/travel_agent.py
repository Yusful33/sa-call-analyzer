"""
LangGraph travel agent pipeline with guardrails and tool calls.
"""

from typing import TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END

from ...cost_guard import CostGuard
from ...llm import get_chat_llm
from ...trace_enrichment import run_guardrail, run_tool_call
from ...use_cases.travel_agent import (
    GUARDRAILS,
    SYSTEM_PROMPT,
    flight_search,
    hotel_search,
    parse_travel_query,
    build_options_table,
    select_tools_for_query,
    sample_travel_query,
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
    degraded_output: str | None = None,
    trace_quality: str = "good",
    **kwargs,
) -> dict:
    """Execute a LangGraph travel agent pipeline: guardrails -> tools -> generate."""
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode

    provider = tracer_provider or trace.get_tracer_provider()
    tracer = provider.get_tracer("demo.travel_agent.langgraph")

    if not query:
        query = sample_travel_query()

    llm = get_chat_llm(model, temperature=0)

    def guardrails_node(state: TravelAgentState) -> dict:
        for g in GUARDRAILS:
            run_guardrail(
                tracer, g["name"], state["query"], llm, guard,
                system_prompt=g["system_prompt"],
            )
        return {"guardrail_passed": True}

    def generate_node(state: TravelAgentState) -> dict:
        import json
        params = parse_travel_query(state["query"])
        run_flight, run_hotel = select_tools_for_query(state["query"])
        flight_result = run_tool_call(
            tracer, "flight_search", state["query"],
            lambda: flight_search(params["origin"], params["destination"], params["date_out"]),
            guard=guard,
        ) if run_flight else json.dumps({"options": []})
        hotel_result = run_tool_call(
            tracer, "hotel_search", state["query"],
            lambda: hotel_search(
                params["city"], params["check_in"], params["check_out"], params["guests"]
            ),
            guard=guard,
        ) if run_hotel else json.dumps({"options": []})
        options_table = build_options_table(
            flight_result if isinstance(flight_result, str) else str(flight_result),
            hotel_result if isinstance(hotel_result, str) else str(hotel_result),
        )
        assumptions_note = ""
        if params.get("assumptions"):
            assumptions_note = (
                "\n\n[Assumptions used for search: " + ", ".join(params["assumptions"]) + "]"
            )
        if run_flight and not run_hotel:
            assumptions_note += "\n\n[User asked for flights only; hotel search was not run.]"
        elif run_hotel and not run_flight:
            assumptions_note += "\n\n[User asked for accommodation only; flight search was not run.]"
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "User request: {question}\n\nAvailable options from search:\n{table}{assumptions}\n\nSummarize these options and give a clear recommendation."),
        ])
        chain = prompt | llm | StrOutputParser()
        if guard:
            guard.check()
        answer = chain.invoke({
            "question": state["query"],
            "table": options_table,
            "assumptions": assumptions_note,
        })
        return {"answer": answer}

    workflow = StateGraph(TravelAgentState)
    workflow.add_node("guardrails", guardrails_node)
    workflow.add_node("generate", generate_node)
    workflow.set_entry_point("guardrails")
    workflow.add_edge("guardrails", "generate")
    workflow.add_edge("generate", END)
    graph = workflow.compile()

    with tracer.start_as_current_span(
        "travel_agent.run",
        attributes={
            "openinference.span.kind": "CHAIN",
            "input.value": query,
            "input.mime_type": "text/plain",
            "metadata.framework": "langgraph",
            "metadata.use_case": "travel-agent",
            "metadata.trace_quality": trace_quality,
        },
    ) as span:
        result = graph.invoke({
            "query": query,
            "answer": "",
            "guardrail_passed": False,
        })
        out = degraded_output if degraded_output else result["answer"]
        span.set_attribute("output.value", out[:2000] if len(out) > 2000 else out)
        span.set_attribute("output.mime_type", "text/plain")
        span.set_status(Status(StatusCode.OK))

    return {
        "query": query,
        "answer": degraded_output if degraded_output else result["answer"],
    }
