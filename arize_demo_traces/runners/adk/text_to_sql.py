"""
ADK-style text-to-SQL agent with manual OpenTelemetry spans.
Mimics Google Agent Development Kit trace patterns without requiring ADK installation.
Uses real LLM calls inside manually created OTel spans with OpenInference attributes.

Trace structure:
  sql_agent.run (AGENT)
    -> guardrails (GUARDRAIL group)
    -> generate_content (LLM) - route query type
    -> generate_content (LLM) - generate SQL
    -> validate_sql (TOOL) - validate the SQL
    -> execute_query (TOOL) - simulated execution
    -> generate_content (LLM) - summarize results
"""

import random

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

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


def run_text_to_sql(
    query: str | None = None,
    model: str = "gpt-4o-mini",
    guard: CostGuard | None = None,
    tracer_provider=None,
) -> dict:
    """Execute an ADK-style text-to-SQL agent: guardrails -> route -> generate -> validate -> execute -> summarize."""
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode

    provider = tracer_provider or trace.get_tracer_provider()
    tracer = provider.get_tracer("demo.text-to-sql.adk")

    if not query:
        query = random.choice(QUERIES)

    llm = get_chat_llm(model, temperature=0)

    with tracer.start_as_current_span(
        "sql_agent.run",
        attributes={
            "openinference.span.kind": "AGENT",
            "input.value": query,
            "input.mime_type": "text/plain",
        },
    ) as agent_span:

        # ---- Guardrails ----
        for g in GUARDRAILS:
            run_guardrail(
                tracer, g["name"], query, llm, guard,
                system_prompt=g["system_prompt"],
            )
        for lg in LOCAL_GUARDRAILS:
            run_local_guardrail(
                tracer, lg["name"], query,
                passed=lg["passed"], detail=lg["detail"],
            )

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
            query_type = route_chain.invoke({"query": query})
            route_span.set_attribute("output.value", query_type.strip())
            route_span.set_attribute("output.mime_type", "text/plain")
            route_span.set_status(Status(StatusCode.OK))

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
                ("system", SYSTEM_PROMPT_SQL_GEN),
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

        # ---- execute_query: simulated SQL execution (TOOL span) ----
        with tracer.start_as_current_span(
            "execute_query",
            attributes={
                "openinference.span.kind": "TOOL",
                "input.value": generated_sql,
                "input.mime_type": "text/plain",
                "tool.name": "execute_query",
            },
        ) as exec_span:
            result_key, results = get_simulated_results()
            exec_span.set_attribute("output.value", str(results))
            exec_span.set_attribute("output.mime_type", "text/plain")
            exec_span.set_attribute("tool.result_key", result_key)
            exec_span.set_status(Status(StatusCode.OK))

        # ---- generate_content: summarize results ----
        with tracer.start_as_current_span(
            "generate_content",
            attributes={
                "openinference.span.kind": "LLM",
                "input.value": query,
                "input.mime_type": "text/plain",
            },
        ) as summary_span:
            summary_prompt = ChatPromptTemplate.from_messages([
                ("system", SYSTEM_PROMPT_SUMMARIZE),
                ("human", "Question: {question}\nSQL: {sql}\nResults: {results}"),
            ])
            summary_chain = summary_prompt | llm | StrOutputParser()
            if guard:
                guard.check()
            summary = summary_chain.invoke({
                "question": query,
                "sql": generated_sql,
                "results": str(results),
            })
            summary_span.set_attribute("output.value", summary)
            summary_span.set_attribute("output.mime_type", "text/plain")
            summary_span.set_status(Status(StatusCode.OK))

        agent_span.set_attribute("output.value", summary)
        agent_span.set_attribute("output.mime_type", "text/plain")
        agent_span.set_status(Status(StatusCode.OK))

    return {
        "query": query,
        "generated_sql": generated_sql,
        "validation": validation,
        "results": results,
        "summary": summary,
    }
