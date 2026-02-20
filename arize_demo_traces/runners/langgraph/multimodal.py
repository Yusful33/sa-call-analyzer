"""
LangGraph multimodal/vision pipeline with guardrails.
Uses a real StateGraph with typed state for image classification, analysis,
entity extraction, and summarization.
Auto-instrumented by LangChainInstrumentor for authentic LangGraph trace patterns.
"""

import random
from typing import TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

from ...cost_guard import CostGuard
from ...llm import get_chat_llm
from ...trace_enrichment import (
    run_guardrail,
    run_local_guardrail,
    run_tool_call,
)
from ...use_cases.multimodal import (
    QUERIES,
    GUARDRAILS,
    LOCAL_GUARDRAILS,
    SYSTEM_PROMPT_VISION,
    SYSTEM_PROMPT_EXTRACT,
    SYSTEM_PROMPT_CLASSIFY_IMAGE,
    SYSTEM_PROMPT_SUMMARIZE_FINDINGS,
    get_random_query,
    analyze_image_content,
    extract_structured_data,
)


class MultimodalState(TypedDict):
    query_text: str
    image_description: str
    classification: str
    analysis: str
    extraction: str
    summary: str
    guardrail_passed: bool


def run_multimodal(
    query: str | None = None,
    model: str = "gpt-4o-mini",
    guard: CostGuard | None = None,
    tracer_provider=None,
    prospect_context=None,
    degraded_output=None,
    trace_quality="good",
    **kwargs,
) -> dict:
    """Execute a LangGraph multimodal pipeline: guardrails -> classify_image -> analyze -> extract -> summarize."""
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode

    provider = tracer_provider or trace.get_tracer_provider()
    tracer = provider.get_tracer("demo.multimodal.langgraph")

    # Handle query: None -> random dict, string -> wrap with default image description
    if query is None:
        q = get_random_query()
        query_text = q["text"]
        image_description = q["image_description"]
    elif isinstance(query, dict):
        query_text = query.get("text", str(query))
        image_description = query.get("image_description", "No image description provided.")
    else:
        query_text = str(query)
        image_description = "A general-purpose image relevant to the user query."

    combined_input = f"{query_text}\n\nImage: {image_description}"
    llm = get_chat_llm(model, temperature=0)

    # --- Node functions (closures capturing llm, guard, tracer) ---

    def guardrails_node(state: MultimodalState) -> dict:
        combined = f"{state['query_text']}\n\nImage: {state['image_description']}"
        for g in GUARDRAILS:
            run_guardrail(
                tracer, g["name"], combined, llm, guard,
                system_prompt=g["system_prompt"],
            )
        for lg in LOCAL_GUARDRAILS:
            run_local_guardrail(
                tracer, lg["name"], combined,
                passed=lg["passed"],
                detail=lg.get("detail", ""),
            )
        return {"guardrail_passed": True}

    def classify_image_node(state: MultimodalState) -> dict:
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT_CLASSIFY_IMAGE),
            ("human", "Image description: {image_description}\n\nQuery: {query}"),
        ])
        if guard:
            guard.check()
        messages = prompt.format_messages(
            image_description=state["image_description"],
            query=state["query_text"],
        )
        response = llm.invoke(messages)
        result = response.content if hasattr(response, "content") else str(response)
        return {"classification": result}

    def analyze_node(state: MultimodalState) -> dict:
        run_tool_call(tracer, "analyze_image_content", state["image_description"][:200],
                      analyze_image_content, guard=guard,
                      image_description=state["image_description"])
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT_VISION),
            ("human", "{query}"),
        ])
        if guard:
            guard.check()
        messages = prompt.format_messages(
            image_description=state["image_description"],
            query=state["query_text"],
        )
        response = llm.invoke(messages)
        result = response.content if hasattr(response, "content") else str(response)
        return {"analysis": result}

    def extract_node(state: MultimodalState) -> dict:
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT_EXTRACT),
            ("human", "Image description: {image_description}\n\nAnalysis: {analysis}"),
        ])
        if guard:
            guard.check()
        messages = prompt.format_messages(
            image_description=state["image_description"],
            analysis=state["analysis"],
        )
        response = llm.invoke(messages)
        result = response.content if hasattr(response, "content") else str(response)
        run_tool_call(tracer, "extract_structured_data", result[:200],
                      extract_structured_data, guard=guard, text=result[:500])
        return {"extraction": result}

    def summarize_node(state: MultimodalState) -> dict:
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT_SUMMARIZE_FINDINGS),
            ("human", "Original query: {query}\n\nExtracted data:\n{extraction}"),
        ])
        if guard:
            guard.check()
        messages = prompt.format_messages(
            query=state["query_text"],
            classification=state["classification"],
            analysis=state["analysis"],
            extraction=state["extraction"],
        )
        response = llm.invoke(messages)
        result = response.content if hasattr(response, "content") else str(response)
        return {"summary": result}

    # --- Build graph ---
    workflow = StateGraph(MultimodalState)
    workflow.add_node("guardrails", guardrails_node)
    workflow.add_node("classify_image", classify_image_node)
    workflow.add_node("analyze", analyze_node)
    workflow.add_node("extract", extract_node)
    workflow.add_node("summarize", summarize_node)
    workflow.set_entry_point("guardrails")
    workflow.add_edge("guardrails", "classify_image")
    workflow.add_edge("classify_image", "analyze")
    workflow.add_edge("analyze", "extract")
    workflow.add_edge("extract", "summarize")
    workflow.add_edge("summarize", END)
    graph = workflow.compile()

    # --- Execute with root span ---
    with tracer.start_as_current_span(
        "multimodal_pipeline",
        attributes={
            "openinference.span.kind": "CHAIN",
            "input.value": combined_input,
            "input.mime_type": "text/plain",
            "metadata.framework": "langgraph",
            "metadata.use_case": "multimodal-ai",
        },
    ) as span:
        result = graph.invoke({
            "query_text": query_text,
            "image_description": image_description,
            "classification": "",
            "analysis": "",
            "extraction": "",
            "summary": "",
            "guardrail_passed": False,
        })
        span.set_attribute("output.value", result["summary"])
        span.set_attribute("output.mime_type", "text/plain")
        span.set_status(Status(StatusCode.OK))

    return {
        "query": query_text,
        "image_description": image_description,
        "classification": result["classification"],
        "analysis": result["analysis"],
        "extraction": result["extraction"],
        "summary": result["summary"],
    }
