"""
LangGraph generic LLM pipeline with guardrails.
Uses a real StateGraph for a simple linear chain.
Auto-instrumented by LangChainInstrumentor for authentic LangGraph trace patterns.
"""

import random
from typing import TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END

from ...cost_guard import CostGuard
from ...llm import get_chat_llm
from ...trace_enrichment import run_guardrail, run_tool_call
from ...use_cases.generic import QUERIES, GUARDRAILS, SYSTEM_PROMPT, web_search, get_current_context


class GenericState(TypedDict):
    query: str
    answer: str
    guardrail_passed: bool


def run_generic(
    query: str | None = None,
    model: str = "gpt-4o-mini",
    guard: CostGuard | None = None,
    tracer_provider=None,
    prospect_context=None,
    degraded_output=None,
    trace_quality="good",
    **kwargs,
) -> dict:
    """Execute a LangGraph generic pipeline: guardrails -> generate."""
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode

    provider = tracer_provider or trace.get_tracer_provider()
    tracer = provider.get_tracer("demo.generic.langgraph")

    if not query:
        query = random.choice(QUERIES)

    llm = get_chat_llm(model, temperature=0)

    # --- Node functions ---

    def guardrails_node(state: GenericState) -> dict:
        for g in GUARDRAILS:
            run_guardrail(
                tracer, g["name"], state["query"], llm, guard,
                system_prompt=g["system_prompt"],
            )
        return {"guardrail_passed": True}

    def generate_node(state: GenericState) -> dict:
        run_tool_call(tracer, "web_search", state["query"],
                      web_search, guard=guard, query=state["query"])
        run_tool_call(tracer, "get_current_context", state["query"],
                      get_current_context, guard=guard)
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "{question}"),
        ])
        chain = prompt | llm | StrOutputParser()
        if guard:
            guard.check()
        answer = chain.invoke({"question": state["query"]})
        return {"answer": answer}

    # --- Build graph ---
    workflow = StateGraph(GenericState)
    workflow.add_node("guardrails", guardrails_node)
    workflow.add_node("generate", generate_node)
    workflow.set_entry_point("guardrails")
    workflow.add_edge("guardrails", "generate")
    workflow.add_edge("generate", END)
    graph = workflow.compile()

    # --- Execute with root span ---
    with tracer.start_as_current_span(
        "llm_pipeline",
        attributes={
            "openinference.span.kind": "CHAIN",
            "input.value": query,
            "input.mime_type": "text/plain",
            "metadata.framework": "langgraph",
            "metadata.use_case": "generic",
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
