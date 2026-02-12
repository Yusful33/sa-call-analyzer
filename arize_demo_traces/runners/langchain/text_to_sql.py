"""
LangChain LCEL text-to-SQL agent pipeline with guardrails.
Uses prompt | llm | parser chains, auto-instrumented by LangChainInstrumentor.
"""

import json as _json
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
    """Execute a LangChain LCEL text-to-SQL pipeline with guardrails."""
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode

    provider = tracer_provider or trace.get_tracer_provider()
    tracer = provider.get_tracer("demo.text-to-sql")

    if not query:
        query = random.choice(QUERIES)

    llm = get_chat_llm(model, temperature=0)

    with tracer.start_as_current_span(
        "text_to_sql_agent",
        attributes={
            "openinference.span.kind": "AGENT",
            "input.value": query,
            "input.mime_type": "text/plain",
        },
    ) as agent_span:

        # === GUARDRAILS ===
        with tracer.start_as_current_span(
            "validate_input",
            attributes={"openinference.span.kind": "GUARDRAIL", "input.value": query},
        ) as guard_span:
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
            guard_span.set_attribute("output.value", "All checks passed")
            guard_span.set_attribute("guardrail.passed", True)
            guard_span.set_status(Status(StatusCode.OK))

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

        # === SQL SPECIALIST AGENT ===
        with tracer.start_as_current_span(
            "sql_specialist",
            attributes={"openinference.span.kind": "AGENT", "input.value": query},
        ) as specialist_span:

            # Generate SQL
            with tracer.start_as_current_span(
                "generate_sql",
                attributes={"openinference.span.kind": "CHAIN", "input.value": query},
            ) as step:
                sql_prompt = ChatPromptTemplate.from_messages([
                    ("system", SYSTEM_PROMPT_SQL_GEN),
                    ("human", "{question}"),
                ])
                sql_chain = sql_prompt | llm | StrOutputParser()
                if guard:
                    guard.check()
                generated_sql = sql_chain.invoke({"question": query})
                step.set_attribute("output.value", generated_sql)
                step.set_status(Status(StatusCode.OK))

            # Validate SQL
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

            # Execute query (simulated)
            with tracer.start_as_current_span(
                "execute_query",
                attributes={
                    "openinference.span.kind": "TOOL",
                    "tool.name": "sql_executor",
                    "input.value": generated_sql,
                },
            ) as step:
                result_key, simulated_results = get_simulated_results()
                step.set_attribute("output.value", _json.dumps(simulated_results))
                step.set_attribute("output.mime_type", "application/json")
                step.set_status(Status(StatusCode.OK))

            # Summarize results
            with tracer.start_as_current_span(
                "summarize_results",
                attributes={"openinference.span.kind": "CHAIN", "input.value": str(simulated_results)[:500]},
            ) as step:
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
                    "results": str(simulated_results),
                })
                step.set_attribute("output.value", summary)
                step.set_status(Status(StatusCode.OK))

            specialist_span.set_attribute("output.value", summary)
            specialist_span.set_status(Status(StatusCode.OK))

        agent_span.set_attribute("output.value", summary)
        agent_span.set_attribute("output.mime_type", "text/plain")
        agent_span.set_status(Status(StatusCode.OK))

    return {
        "query": query,
        "generated_sql": generated_sql,
        "validation": validation,
        "results": simulated_results,
        "summary": summary,
    }
