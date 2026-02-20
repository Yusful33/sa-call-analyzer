"""
CrewAI text-to-SQL pipeline with SQL generator and validator agents.
Uses real CrewAI Crew, Agent, Task objects to generate authentic trace patterns
when instrumented by CrewAIInstrumentor.
"""

import random

from crewai import Agent, Crew, Task, Process

from ...cost_guard import CostGuard
from ...llm import get_chat_llm
from ...trace_enrichment import run_tool_call
from ...use_cases.text_to_sql import (
    SCHEMA,
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
    degraded_output=None,
    trace_quality="good",
    **kwargs,
) -> dict:
    """Execute a CrewAI text-to-SQL pipeline with generator and validator agents."""
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode

    provider = tracer_provider or trace.get_tracer_provider()
    tracer = provider.get_tracer("demo.text-to-sql.crewai")
    queries_pool = get_queries_for_prospect(prospect_context)
    if not query:
        query = random.choice(queries_pool)
    context_preamble = get_context_preamble(prospect_context)

    llm = get_chat_llm(model, temperature=0)

    with tracer.start_as_current_span(
        "text_to_sql_pipeline",
        attributes={
            "openinference.span.kind": "AGENT",
            "input.value": query,
            "input.mime_type": "text/plain",
            "metadata.framework": "crewai",
            "metadata.use_case": "text-to-sql-bi-agent",
        },
    ) as pipeline_span:

        # === TOOL CALLS ===
        run_tool_call(tracer, "get_table_schema", query, get_table_schema, guard=guard, table_name="sales")

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

        # === CREWAI TASKS ===
        generate_task = Task(
            description=(
                f"{context_preamble}"
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

        # === CREWAI CREW ===
        crew = Crew(
            agents=[sql_generator, sql_validator],
            tasks=[generate_task, validate_task],
            process=Process.sequential,
            verbose=False,
        )

        if guard:
            guard.check()
        result = crew.kickoff()
        generated_sql = str(result)

        run_tool_call(tracer, "execute_query", "SELECT ...", execute_query, guard=guard, sql="SELECT * FROM sales LIMIT 10")

        pipeline_span.set_attribute("output.value", generated_sql)
        pipeline_span.set_attribute("output.mime_type", "text/plain")
        pipeline_span.set_status(Status(StatusCode.OK))

    return {
        "query": query,
        "generated_sql": generated_sql,
    }
