"""
LangGraph text-to-SQL agent pipeline with guardrails.
Uses a real StateGraph with SQL specialist nodes for generate, validate, execute, summarize.
Auto-instrumented by LangChainInstrumentor for authentic LangGraph trace patterns.
"""

import json as _json
import random
from typing import TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END

from ...cost_guard import CostGuard
from ...llm import get_chat_llm
from ...trace_enrichment import (
    run_guardrail,
    run_local_guardrail,
)
from ...use_cases.text_to_sql import (
    QUERIES,
    SCHEMA,
    GUARDRAILS,
    LOCAL_GUARDRAILS,
    SYSTEM_PROMPT_SQL_GEN,
    SYSTEM_PROMPT_SQL_VALIDATE,
    SYSTEM_PROMPT_ROUTE,
    SYSTEM_PROMPT_SUMMARIZE,
    get_simulated_results,
)


class TextToSQLState(TypedDict):
    query: str
    query_type: str
    generated_sql: str
    validation: str
    results: list
    result_key: str
    summary: str
    guardrail_passed: bool


def run_text_to_sql(
    query: str | None = None,
    model: str = "gpt-4o-mini",
    guard: CostGuard | None = None,
    tracer_provider=None,
) -> dict:
    """Execute a LangGraph text-to-SQL pipeline: guardrails -> route -> generate -> validate -> execute -> summarize."""
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode

    provider = tracer_provider or trace.get_tracer_provider()
    tracer = provider.get_tracer("demo.text-to-sql.langgraph")

    if not query:
        query = random.choice(QUERIES)

    llm = get_chat_llm(model, temperature=0)

    # --- Node functions ---

    def guardrails_node(state: TextToSQLState) -> dict:
        for g in GUARDRAILS:
            run_guardrail(
                tracer, g["name"], state["query"], llm, guard,
                system_prompt=g["system_prompt"],
            )
        for lg in LOCAL_GUARDRAILS:
            run_local_guardrail(
                tracer, lg["name"], state["query"],
                passed=lg["passed"], detail=lg["detail"],
            )
        return {"guardrail_passed": True}

    def route_query_node(state: TextToSQLState) -> dict:
        route_prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT_ROUTE),
            ("human", "{query}"),
        ])
        route_chain = route_prompt | llm | StrOutputParser()
        if guard:
            guard.check()
        query_type = route_chain.invoke({"query": state["query"]})
        return {"query_type": query_type.strip()}

    def generate_sql_node(state: TextToSQLState) -> dict:
        sql_prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT_SQL_GEN),
            ("human", "{question}"),
        ])
        sql_chain = sql_prompt | llm | StrOutputParser()
        if guard:
            guard.check()
        generated_sql = sql_chain.invoke({"question": state["query"]})
        return {"generated_sql": generated_sql}

    def validate_sql_node(state: TextToSQLState) -> dict:
        validate_prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT_SQL_VALIDATE),
            ("human", "SQL: {sql}"),
        ])
        validate_chain = validate_prompt | llm | StrOutputParser()
        if guard:
            guard.check()
        validation = validate_chain.invoke({"sql": state["generated_sql"]})
        return {"validation": validation}

    def execute_query_node(state: TextToSQLState) -> dict:
        result_key, results = get_simulated_results()
        return {"results": results, "result_key": result_key}

    def summarize_node(state: TextToSQLState) -> dict:
        summary_prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT_SUMMARIZE),
            ("human", "Question: {question}\nSQL: {sql}\nResults: {results}"),
        ])
        summary_chain = summary_prompt | llm | StrOutputParser()
        if guard:
            guard.check()
        summary = summary_chain.invoke({
            "question": state["query"],
            "sql": state["generated_sql"],
            "results": str(state["results"]),
        })
        return {"summary": summary}

    # --- Build graph ---
    workflow = StateGraph(TextToSQLState)
    workflow.add_node("guardrails", guardrails_node)
    workflow.add_node("route_query", route_query_node)
    workflow.add_node("generate_sql", generate_sql_node)
    workflow.add_node("validate_sql", validate_sql_node)
    workflow.add_node("execute_query", execute_query_node)
    workflow.add_node("summarize", summarize_node)
    workflow.set_entry_point("guardrails")
    workflow.add_edge("guardrails", "route_query")
    workflow.add_edge("route_query", "generate_sql")
    workflow.add_edge("generate_sql", "validate_sql")
    workflow.add_edge("validate_sql", "execute_query")
    workflow.add_edge("execute_query", "summarize")
    workflow.add_edge("summarize", END)
    graph = workflow.compile()

    # --- Execute with root span ---
    with tracer.start_as_current_span(
        "text_to_sql_agent",
        attributes={
            "openinference.span.kind": "AGENT",
            "input.value": query,
            "input.mime_type": "text/plain",
        },
    ) as span:
        result = graph.invoke({
            "query": query,
            "query_type": "",
            "generated_sql": "",
            "validation": "",
            "results": [],
            "result_key": "",
            "summary": "",
            "guardrail_passed": False,
        })
        span.set_attribute("output.value", result["summary"])
        span.set_attribute("output.mime_type", "text/plain")
        span.set_status(Status(StatusCode.OK))

    return {
        "query": query,
        "generated_sql": result["generated_sql"],
        "validation": result["validation"],
        "results": result["results"],
        "summary": result["summary"],
    }
