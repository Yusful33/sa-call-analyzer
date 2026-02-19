"""
ADK-style text-to-SQL agent with manual OpenTelemetry spans.
Mimics Google Agent Development Kit trace patterns without requiring ADK installation.
Uses real LLM calls inside manually created OTel spans with OpenInference attributes.

Trace structure:
  sql_agent.run (AGENT)
    -> generate_content (LLM) - route query type
    -> generate_content (LLM) - generate SQL
    -> validate_sql (TOOL) - validate the SQL
"""

import random

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

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


def run_text_to_sql(
    query: str | None = None,
    model: str = "gpt-4o-mini",
    guard: CostGuard | None = None,
    tracer_provider=None,
    prospect_context=None,
) -> dict:
    """Execute an ADK-style text-to-SQL agent: route -> generate -> validate."""
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode

    provider = tracer_provider or trace.get_tracer_provider()
    tracer = provider.get_tracer("demo.text-to-sql.adk")

    queries_pool = get_queries_for_prospect(prospect_context)
    if not query:
        query = random.choice(queries_pool)
    sql_gen_system_prompt = get_context_preamble(prospect_context) + SYSTEM_PROMPT_SQL_GEN

    llm = get_chat_llm(model, temperature=0)

    with tracer.start_as_current_span(
        "sql_agent.run",
        attributes={
            "openinference.span.kind": "AGENT",
            "input.value": query,
            "input.mime_type": "text/plain",
            "metadata.framework": "adk",
            "metadata.use_case": "text-to-sql-bi-agent",
        },
    ) as agent_span:

        # ---- generate_content: route query type ----
        with tracer.start_as_current_span(
            "generate_content",
            attributes={
                "openinference.span.kind": "LLM",
                "input.value": query,
                "input.mime_type": "text/plain",
            },
        ) as route_span:
            route_prompt = ChatPromptTemplate.from_messages([
                ("system", SYSTEM_PROMPT_ROUTE),
                ("human", "{query}"),
            ])
            route_chain = route_prompt | llm | StrOutputParser()
            if guard:
                guard.check()
            raw_route = route_chain.invoke({"query": query})
            query_type = (raw_route or "").strip() if isinstance(raw_route, str) else "ANALYTICAL"
            if not query_type:
                query_type = "ANALYTICAL"
            route_span.set_attribute("output.value", query_type)
            route_span.set_attribute("output.mime_type", "text/plain")
            route_span.set_status(Status(StatusCode.OK))

        # ---- get_table_schema: fetch schema for SQL generation ----
        run_tool_call(tracer, "get_table_schema", query, get_table_schema, guard=guard, table_name="sales")

        # ---- generate_content: generate SQL ----
        with tracer.start_as_current_span(
            "generate_content",
            attributes={
                "openinference.span.kind": "LLM",
                "input.value": query,
                "input.mime_type": "text/plain",
            },
        ) as sql_span:
            sql_prompt = ChatPromptTemplate.from_messages([
                ("system", sql_gen_system_prompt),
                ("human", "{question}"),
            ])
            sql_chain = sql_prompt | llm | StrOutputParser()
            if guard:
                guard.check()
            generated_sql = sql_chain.invoke({"question": query})
            sql_span.set_attribute("output.value", generated_sql)
            sql_span.set_attribute("output.mime_type", "text/plain")
            sql_span.set_status(Status(StatusCode.OK))

        # ---- validate_sql: LLM-based SQL validation (TOOL span) ----
        with tracer.start_as_current_span(
            "validate_sql",
            attributes={
                "openinference.span.kind": "TOOL",
                "input.value": generated_sql,
                "input.mime_type": "text/plain",
                "tool.name": "validate_sql",
            },
        ) as validate_span:
            validate_prompt = ChatPromptTemplate.from_messages([
                ("system", SYSTEM_PROMPT_SQL_VALIDATE),
                ("human", "SQL: {sql}"),
            ])
            validate_chain = validate_prompt | llm | StrOutputParser()
            if guard:
                guard.check()
            validation = validate_chain.invoke({"sql": generated_sql})
            validate_span.set_attribute("output.value", validation)
            validate_span.set_attribute("output.mime_type", "text/plain")
            validate_span.set_status(Status(StatusCode.OK))

        # ---- execute_query: run the generated SQL ----
        run_tool_call(tracer, "execute_query", generated_sql, execute_query, guard=guard, sql=generated_sql)

        agent_span.set_attribute("output.value", generated_sql)
        agent_span.set_attribute("output.mime_type", "text/plain")
        agent_span.set_status(Status(StatusCode.OK))

    return {
        "query": query,
        "generated_sql": generated_sql,
        "validation": validation,
    }
