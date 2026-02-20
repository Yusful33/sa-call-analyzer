"""
LangGraph text-to-SQL agent pipeline.
Uses a real StateGraph with nodes for route, generate, and validate.
Auto-instrumented by LangChainInstrumentor for authentic LangGraph trace patterns.
"""

import random
from typing import TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END

from ...cost_guard import CostGuard
from ...llm import get_chat_llm
from ...trace_enrichment import run_tool_call
from ...use_cases.text_to_sql import (
    SYSTEM_PROMPT_SQL_GEN,
    SYSTEM_PROMPT_SQL_VALIDATE,
    SYSTEM_PROMPT_ROUTE,
    get_table_schema,
    execute_query,
    get_queries_for_prospect,
    get_context_preamble,
)


class TextToSQLState(TypedDict):
    query: str
    query_type: str
    generated_sql: str
    validation: str


def run_text_to_sql(
    query: str | None = None,
    model: str = "gpt-4o-mini",
    guard: CostGuard | None = None,
    tracer_provider=None,
    prospect_context=None,
    degraded_output=None,
    trace_quality="good",
    **kwargs,
) -> dict:
    """Execute a LangGraph text-to-SQL pipeline: route -> generate -> validate."""
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode

    provider = tracer_provider or trace.get_tracer_provider()
    tracer = provider.get_tracer("demo.text-to-sql.langgraph")

    queries_pool = get_queries_for_prospect(prospect_context)
    if not query:
        query = random.choice(queries_pool)
    sql_gen_system_prompt = get_context_preamble(prospect_context) + SYSTEM_PROMPT_SQL_GEN

    llm = get_chat_llm(model, temperature=0)

    # --- Node functions ---

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
        run_tool_call(tracer, "get_table_schema", state["query"],
                      get_table_schema, guard=guard, table_name="sales")
        sql_prompt = ChatPromptTemplate.from_messages([
            ("system", sql_gen_system_prompt),
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
        run_tool_call(tracer, "execute_query", state["generated_sql"],
                      execute_query, guard=guard, sql=state["generated_sql"])
        return {"validation": validation}

    # --- Build graph ---
    workflow = StateGraph(TextToSQLState)
    workflow.add_node("route_query", route_query_node)
    workflow.add_node("generate_sql", generate_sql_node)
    workflow.add_node("validate_sql", validate_sql_node)
    workflow.set_entry_point("route_query")
    workflow.add_edge("route_query", "generate_sql")
    workflow.add_edge("generate_sql", "validate_sql")
    workflow.add_edge("validate_sql", END)
    graph = workflow.compile()

    # --- Execute with root span ---
    with tracer.start_as_current_span(
        "text_to_sql_agent",
        attributes={
            "openinference.span.kind": "AGENT",
            "input.value": query,
            "input.mime_type": "text/plain",
            "metadata.framework": "langgraph",
            "metadata.use_case": "text-to-sql-bi-agent",
        },
    ) as span:
        result = graph.invoke({
            "query": query,
            "query_type": "",
            "generated_sql": "",
            "validation": "",
        })
        span.set_attribute("output.value", result["generated_sql"])
        span.set_attribute("output.mime_type", "text/plain")
        span.set_status(Status(StatusCode.OK))

    return {
        "query": query,
        "generated_sql": result["generated_sql"],
        "validation": result["validation"],
    }
