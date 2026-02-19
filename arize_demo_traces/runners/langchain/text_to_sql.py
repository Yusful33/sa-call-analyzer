"""
LangChain LCEL text-to-SQL agent pipeline.
Uses prompt | llm | parser chains, auto-instrumented by LangChainInstrumentor.
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
    """Execute a LangChain LCEL text-to-SQL pipeline: route -> generate -> validate."""
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode

    provider = tracer_provider or trace.get_tracer_provider()
    tracer = provider.get_tracer("demo.text-to-sql")

    queries_pool = get_queries_for_prospect(prospect_context)
    if not query:
        query = random.choice(queries_pool)
    sql_gen_system_prompt = get_context_preamble(prospect_context) + SYSTEM_PROMPT_SQL_GEN

    llm = get_chat_llm(model, temperature=0)

    with tracer.start_as_current_span(
        "text_to_sql_agent",
        attributes={
            "openinference.span.kind": "AGENT",
            "input.value": query,
            "input.mime_type": "text/plain",
            "metadata.framework": "langchain",
            "metadata.use_case": "text-to-sql-bi-agent",
        },
    ) as agent_span:

        # === QUERY ROUTING ===
        with tracer.start_as_current_span(
            "query_routing",
            attributes={"openinference.span.kind": "CHAIN", "input.value": query},
        ) as route_span:
            route_prompt = ChatPromptTemplate.from_messages([
                ("system", SYSTEM_PROMPT_ROUTE),
                ("human", "{query}"),
            ])
            route_chain = route_prompt | llm | StrOutputParser()
            if guard:
                guard.check()
            query_type = route_chain.invoke({"query": query})
            route_span.set_attribute("output.value", query_type.strip())
            route_span.set_status(Status(StatusCode.OK))

        # === SQL GENERATION ===
        with tracer.start_as_current_span(
            "generate_sql",
            attributes={"openinference.span.kind": "CHAIN", "input.value": query},
        ) as step:
            run_tool_call(tracer, "get_table_schema", query, get_table_schema, guard=guard, table_name="sales")
            sql_prompt = ChatPromptTemplate.from_messages([
                ("system", sql_gen_system_prompt),
                ("human", "{question}"),
            ])
            sql_chain = sql_prompt | llm | StrOutputParser()
            if guard:
                guard.check()
            generated_sql = sql_chain.invoke({"question": query})
            step.set_attribute("output.value", generated_sql)
            step.set_status(Status(StatusCode.OK))

        # === SQL VALIDATION ===
        run_tool_call(tracer, "execute_query", generated_sql, execute_query, guard=guard, sql=generated_sql)

        with tracer.start_as_current_span(
            "validate_sql",
            attributes={"openinference.span.kind": "CHAIN", "input.value": generated_sql},
        ) as step:
            validate_prompt = ChatPromptTemplate.from_messages([
                ("system", SYSTEM_PROMPT_SQL_VALIDATE),
                ("human", "SQL: {sql}"),
            ])
            validate_chain = validate_prompt | llm | StrOutputParser()
            if guard:
                guard.check()
            validation = validate_chain.invoke({"sql": generated_sql})
            step.set_attribute("output.value", validation)
            step.set_status(Status(StatusCode.OK))

        agent_span.set_attribute("output.value", generated_sql)
        agent_span.set_attribute("output.mime_type", "text/plain")
        agent_span.set_status(Status(StatusCode.OK))

    return {
        "query": query,
        "generated_sql": generated_sql,
        "validation": validation,
    }
