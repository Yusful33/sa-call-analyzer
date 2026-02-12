"""
CrewAI text-to-SQL pipeline with SQL generator, validator, and data analyst agents.
Uses real CrewAI Crew, Agent, Task objects to generate authentic trace patterns
when instrumented by CrewAIInstrumentor.
"""

import json as _json
import random

from crewai import Agent, Crew, Task, Process

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
    SYSTEM_PROMPT_SUMMARIZE,
    get_simulated_results,
)


def run_text_to_sql(
    query: str | None = None,
    model: str = "gpt-4o-mini",
    guard: CostGuard | None = None,
    tracer_provider=None,
) -> dict:
    """Execute a CrewAI text-to-SQL pipeline with generator, validator, analyst agents, guardrails, and evaluation."""
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode

    provider = tracer_provider or trace.get_tracer_provider()
    tracer = provider.get_tracer("demo.text-to-sql.crewai")
    if not query:
        query = random.choice(QUERIES)

    llm = get_chat_llm(model, temperature=0)

    with tracer.start_as_current_span(
        "text_to_sql_pipeline",
        attributes={
            "openinference.span.kind": "AGENT",
            "input.value": query,
            "input.mime_type": "text/plain",
        },
    ) as pipeline_span:

        # === GUARDRAILS ===
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

        # === CREWAI AGENTS ===
        sql_generator = Agent(
            role="SQL Query Generator",
            goal="Generate accurate SQL queries from natural language questions",
            backstory=(
                f"You are a senior database engineer who specializes in translating "
                f"natural language questions into precise SQL queries. You are deeply "
                f"familiar with the following schema:\n{SCHEMA}"
            ),
            llm=llm,
            verbose=False,
            allow_delegation=False,
        )

        sql_validator = Agent(
            role="SQL Validator",
            goal="Validate SQL queries for correctness, safety, and adherence to the schema",
            backstory=(
                f"You are a database security expert who reviews SQL queries for "
                f"correctness and safety. You check for SQL injection risks, syntax "
                f"errors, and ensure queries only use tables and columns from the schema.\n{SCHEMA}"
            ),
            llm=llm,
            verbose=False,
            allow_delegation=False,
        )

        data_analyst = Agent(
            role="Data Analyst",
            goal="Summarize SQL query results into clear, actionable natural language insights",
            backstory=(
                "You are a skilled data analyst who excels at interpreting query results "
                "and presenting findings in a clear, concise way that business stakeholders "
                "can understand."
            ),
            llm=llm,
            verbose=False,
            allow_delegation=False,
        )

        # === CREWAI TASKS ===
        generate_task = Task(
            description=(
                f"Generate a SQL query to answer the following question. "
                f"Return ONLY the SQL query, no explanation.\n\n"
                f"Question: {query}\n\n"
                f"Database schema:\n{SCHEMA}"
            ),
            expected_output="A valid SQL query that answers the question",
            agent=sql_generator,
        )

        validate_task = Task(
            description=(
                f"Validate the SQL query generated in the previous step. Check that it is "
                f"syntactically correct, safe (no DROP, DELETE, UPDATE), and uses only valid "
                f"tables/columns from the schema.\n\n"
                f"Database schema:\n{SCHEMA}\n\n"
                f"Respond with 'VALID' or 'INVALID: reason'."
            ),
            expected_output="VALID or INVALID with reason",
            agent=sql_validator,
        )

        # Simulate query execution (outside CrewAI since we can't run real SQL)
        result_key, simulated_results = get_simulated_results()

        summarize_task = Task(
            description=(
                f"Summarize the following query results in a clear, concise natural language "
                f"response that answers the original question.\n\n"
                f"Original question: {query}\n\n"
                f"Query results:\n{_json.dumps(simulated_results, indent=2)}"
            ),
            expected_output="A clear natural language summary of the query results",
            agent=data_analyst,
        )

        # === CREWAI CREW ===
        crew = Crew(
            agents=[sql_generator, sql_validator, data_analyst],
            tasks=[generate_task, validate_task, summarize_task],
            process=Process.sequential,
            verbose=False,
        )

        if guard:
            guard.check()
        result = crew.kickoff()
        summary = str(result)

        # Record simulated execution as a span
        with tracer.start_as_current_span(
            "execute_query",
            attributes={
                "openinference.span.kind": "TOOL",
                "tool.name": "sql_executor",
                "input.value": query,
            },
        ) as exec_span:
            exec_span.set_attribute("output.value", _json.dumps(simulated_results))
            exec_span.set_attribute("output.mime_type", "application/json")
            exec_span.set_status(Status(StatusCode.OK))

        pipeline_span.set_attribute("output.value", summary)
        pipeline_span.set_attribute("output.mime_type", "text/plain")
        pipeline_span.set_status(Status(StatusCode.OK))

    return {
        "query": query,
        "results": simulated_results,
        "summary": summary,
    }
