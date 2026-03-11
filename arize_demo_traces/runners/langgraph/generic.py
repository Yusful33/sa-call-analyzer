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
from ...trace_enrichment import invoke_chain_in_context, run_guardrail, run_tool_call, run_in_context
from ...use_cases.generic import GUARDRAILS, SYSTEM_PROMPT, web_search, get_current_context
from ..common_runner_utils import get_query_for_run


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

    from ...use_cases import generic as generic_use_case
    if not query:
        rng = kwargs.get("rng")
        _kw = {k: v for k, v in kwargs.items() if k != "rng"}
        query_spec = get_query_for_run(generic_use_case, prospect_context=prospect_context, rng=rng, **_kw)
        query = query_spec.text
    else:
        query_spec = None

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
        answer = invoke_chain_in_context(chain, {"question": state["query"]})
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
    attrs = {
        "openinference.span.kind": "CHAIN",
        "input.value": query,
        "input.mime_type": "text/plain",
        "metadata.framework": "langgraph",
        "metadata.use_case": "generic",
    }
    if query_spec:
        attrs.update(query_spec.to_span_attributes())
    with tracer.start_as_current_span("llm_pipeline", attributes=attrs) as span:
        result = run_in_context(graph.invoke, {
            "query": query,
            "answer": "",
            "guardrail_passed": False,
        })
        span.set_attribute("output.value", result["answer"])
        span.set_attribute("output.mime_type", "text/plain")
        span.set_attribute("context.query", query[:1000])
        span.set_status(Status(StatusCode.OK))

    return {
        "query": query,
        "answer": result["answer"],
    }
