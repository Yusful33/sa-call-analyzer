"""
LangGraph classification pipeline with guardrails.
Uses a real StateGraph with typed state for ticket classification, sentiment analysis,
entity extraction, and response generation.
Auto-instrumented by LangChainInstrumentor for authentic LangGraph trace patterns.
"""

import random
from typing import TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

from ...cost_guard import CostGuard
from ...llm import get_chat_llm
from ...trace_enrichment import run_guardrail, run_tool_call
from ...use_cases.classification import (
    QUERIES,
    GUARDRAILS,
    SYSTEM_PROMPT_CLASSIFY,
    SYSTEM_PROMPT_SENTIMENT,
    SYSTEM_PROMPT_EXTRACT,
    SYSTEM_PROMPT_RESPONSE,
    lookup_routing_rules,
    search_response_templates,
)


class ClassificationState(TypedDict):
    query: str
    category: str
    confidence: str
    sentiment: str
    entities: str
    response: str
    guardrail_passed: bool


def run_classification(
    query: str | None = None,
    model: str = "gpt-4o-mini",
    guard: CostGuard | None = None,
    tracer_provider=None,
    prospect_context=None,
    degraded_output=None,
    trace_quality="good",
    **kwargs,
) -> dict:
    """Execute a LangGraph classification pipeline: guardrails -> classify -> sentiment -> extract -> respond."""
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode

    provider = tracer_provider or trace.get_tracer_provider()
    tracer = provider.get_tracer("demo.classification.langgraph")

    if not query:
        query = random.choice(QUERIES)

    llm = get_chat_llm(model, temperature=0)

    # --- Node functions (closures capturing llm, guard, tracer) ---

    def guardrails_node(state: ClassificationState) -> dict:
        for g in GUARDRAILS:
            run_guardrail(
                tracer, g["name"], state["query"], llm, guard,
                system_prompt=g["system_prompt"],
            )
        return {"guardrail_passed": True}

    def classify_node(state: ClassificationState) -> dict:
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT_CLASSIFY),
            ("human", "{query}"),
        ])
        if guard:
            guard.check()
        messages = prompt.format_messages(query=state["query"])
        response = llm.invoke(messages)
        result = response.content if hasattr(response, "content") else str(response)
        return {"category": result}

    def sentiment_node(state: ClassificationState) -> dict:
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT_SENTIMENT),
            ("human", "{query}"),
        ])
        if guard:
            guard.check()
        messages = prompt.format_messages(query=state["query"])
        response = llm.invoke(messages)
        result = response.content if hasattr(response, "content") else str(response)
        return {"sentiment": result}

    def extract_entities_node(state: ClassificationState) -> dict:
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT_EXTRACT),
            ("human", "{query}"),
        ])
        if guard:
            guard.check()
        messages = prompt.format_messages(query=state["query"])
        response = llm.invoke(messages)
        result = response.content if hasattr(response, "content") else str(response)
        return {"entities": result}

    def generate_response_node(state: ClassificationState) -> dict:
        run_tool_call(tracer, "lookup_routing_rules", state["category"],
                      lookup_routing_rules, guard=guard, category="technical_support")
        run_tool_call(tracer, "search_response_templates", state["category"],
                      search_response_templates, guard=guard, category="technical_support")
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT_RESPONSE),
            ("human", "{query}"),
        ])
        if guard:
            guard.check()
        messages = prompt.format_messages(
            query=state["query"],
            classification=state["category"],
            sentiment=state["sentiment"],
            entities=state["entities"],
        )
        response = llm.invoke(messages)
        result = response.content if hasattr(response, "content") else str(response)
        return {"response": result}

    # --- Build graph ---
    workflow = StateGraph(ClassificationState)
    workflow.add_node("guardrails", guardrails_node)
    workflow.add_node("classify", classify_node)
    workflow.add_node("sentiment", sentiment_node)
    workflow.add_node("extract_entities", extract_entities_node)
    workflow.add_node("generate_response", generate_response_node)
    workflow.set_entry_point("guardrails")
    workflow.add_edge("guardrails", "classify")
    workflow.add_edge("classify", "sentiment")
    workflow.add_edge("sentiment", "extract_entities")
    workflow.add_edge("extract_entities", "generate_response")
    workflow.add_edge("generate_response", END)
    graph = workflow.compile()

    # --- Execute with root span ---
    with tracer.start_as_current_span(
        "classification_pipeline",
        attributes={
            "openinference.span.kind": "CHAIN",
            "input.value": query,
            "input.mime_type": "text/plain",
            "metadata.framework": "langgraph",
            "metadata.use_case": "classification-routing",
        },
    ) as span:
        result = graph.invoke({
            "query": query,
            "category": "",
            "confidence": "",
            "sentiment": "",
            "entities": "",
            "response": "",
            "guardrail_passed": False,
        })
        span.set_attribute("output.value", result["response"])
        span.set_attribute("output.mime_type", "text/plain")
        span.set_status(Status(StatusCode.OK))

    return {
        "query": query,
        "category": result["category"],
        "sentiment": result["sentiment"],
        "entities": result["entities"],
        "response": result["response"],
    }
