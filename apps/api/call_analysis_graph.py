"""LangGraph workflow for call analysis (replaces CrewAI Crew/Task orchestration)."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from call_analysis_llm import chat_invoke_text
from call_analysis_prompts import (
    classification_prompt,
    compile_prompt,
    sales_methodology_prompt,
    technical_prompt,
)

if TYPE_CHECKING:
    from opentelemetry.trace import Tracer


class _AnalysisState(TypedDict, total=False):
    transcript: str
    prior_context: str
    classification_text: str
    tech_output: str
    sales_output: str
    final_report_text: str


def _run_classification(state: _AnalysisState, chat: BaseChatModel) -> dict:
    prompt = classification_prompt(state["transcript"], state.get("prior_context") or "")
    text = chat_invoke_text(chat, prompt)
    return {"classification_text": text}


def _run_technical(state: _AnalysisState, chat: BaseChatModel) -> dict:
    prompt = technical_prompt(state["transcript"], state.get("prior_context") or "")
    text = chat_invoke_text(chat, prompt)
    return {"tech_output": text}


def _run_sales(state: _AnalysisState, chat: BaseChatModel) -> dict:
    prompt = sales_methodology_prompt(state["transcript"], state.get("prior_context") or "")
    text = chat_invoke_text(chat, prompt)
    return {"sales_output": text}


def _run_compile(state: _AnalysisState, chat: BaseChatModel) -> dict:
    prompt = compile_prompt(
        state["transcript"],
        state.get("prior_context") or "",
        state.get("tech_output") or "",
        state.get("sales_output") or "",
    )
    text = chat_invoke_text(chat, prompt)
    return {"final_report_text": text}


def _build_graph(chat: BaseChatModel):
    g = StateGraph(_AnalysisState)

    g.add_node("classify", lambda s: _run_classification(s, chat))
    g.add_node("technical", lambda s: _run_technical(s, chat))
    g.add_node("sales", lambda s: _run_sales(s, chat))
    g.add_node("compile", lambda s: _run_compile(s, chat))

    g.add_edge(START, "classify")
    g.add_edge("classify", "technical")
    g.add_edge("classify", "sales")
    g.add_edge("technical", "compile")
    g.add_edge("sales", "compile")
    g.add_edge("compile", END)

    return g.compile()


def run_call_analysis_pipeline(
    chat: BaseChatModel,
    transcript: str,
    prior_context: str,
    tracer: "Tracer",
) -> tuple[str, str]:
    """
    LangGraph: classify → (technical || sales) → compile.

    Returns (classification_text, final_report_text).
    """
    from opentelemetry.trace import Status, StatusCode

    graph = _build_graph(chat)
    out = graph.invoke(
        {
            "transcript": transcript,
            "prior_context": prior_context or "",
        }
    )

    classification_text = out.get("classification_text") or ""
    tech_output = out.get("tech_output") or ""
    sales_output = out.get("sales_output") or ""
    final_report_text = out.get("final_report_text") or ""

    with tracer.start_as_current_span("call_classification") as class_span:
        class_span.set_attribute("openinference.span.kind", "agent")
        class_span.set_attribute(
            "input.value",
            transcript[:2000] + "..." if len(transcript) > 2000 else transcript,
        )
        class_span.set_attribute("input.mime_type", "text/plain")
        class_span.set_attribute("output.value", classification_text[:3000])
        class_span.set_attribute("output.mime_type", "application/json")
        class_span.add_event("classification_complete", {"result_length": len(classification_text)})
        class_span.set_status(Status(StatusCode.OK))

    with tracer.start_as_current_span("agent.technical_evaluator") as tech_span:
        tech_span.set_attribute("openinference.span.kind", "agent")
        tech_span.set_attribute("agent.name", "technical_evaluator")
        tech_span.set_attribute(
            "input.value",
            transcript[:2000] + "..." if len(transcript) > 2000 else transcript,
        )
        tech_span.set_attribute("input.mime_type", "text/plain")
        tech_span.set_attribute("output.value", tech_output[:5000])
        tech_span.set_attribute("output.mime_type", "text/plain")
        tech_span.set_status(Status(StatusCode.OK))

    with tracer.start_as_current_span("agent.sales_methodology_expert") as sales_span:
        sales_span.set_attribute("openinference.span.kind", "agent")
        sales_span.set_attribute("agent.name", "sales_methodology_expert")
        sales_span.set_attribute(
            "input.value",
            transcript[:2000] + "..." if len(transcript) > 2000 else transcript,
        )
        sales_span.set_attribute("input.mime_type", "text/plain")
        sales_span.set_attribute("output.value", sales_output[:5000])
        sales_span.set_attribute("output.mime_type", "text/plain")
        sales_span.set_status(Status(StatusCode.OK))

    with tracer.start_as_current_span("agent.report_compiler") as compile_span:
        compile_span.set_attribute("openinference.span.kind", "agent")
        compile_span.set_attribute("agent.name", "report_compiler")
        compile_span.set_attribute(
            "input.value",
            str(
                {
                    "technical_output_length": len(tech_output),
                    "sales_output_length": len(sales_output),
                }
            ),
        )
        compile_span.set_attribute("input.mime_type", "application/json")
        compile_span.set_attribute("output.value", final_report_text[:5000])
        compile_span.set_attribute("output.mime_type", "text/plain")
        compile_span.set_status(Status(StatusCode.OK))

    return classification_text, final_report_text
