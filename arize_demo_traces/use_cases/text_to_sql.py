"""Text-to-SQL use-case: shared schema, queries, sample results, guardrails, evaluators."""

import random

SCHEMA = """
Tables:
  - sales (id INT, region VARCHAR, product VARCHAR, amount DECIMAL, quarter VARCHAR, sales_rep VARCHAR)
  - regions (id INT, name VARCHAR, country VARCHAR, manager VARCHAR)
  - products (id INT, name VARCHAR, category VARCHAR, price DECIMAL)
  - customers (id INT, name VARCHAR, region_id INT, tier VARCHAR, signup_date DATE)
"""

SAMPLE_RESULTS = {
    "sales_by_region": [
        {"region": "North America", "total_sales": 15200000},
        {"region": "EMEA", "total_sales": 12800000},
        {"region": "APAC", "total_sales": 8900000},
        {"region": "LATAM", "total_sales": 5400000},
    ],
    "top_products": [
        {"product": "Enterprise Platform", "revenue": 22100000},
        {"product": "Analytics Suite", "revenue": 12300000},
        {"product": "API Gateway", "revenue": 7900000},
    ],
    "customer_growth": [
        {"quarter": "Q1-2025", "new_customers": 142},
        {"quarter": "Q2-2025", "new_customers": 189},
        {"quarter": "Q3-2025", "new_customers": 234},
    ],
}

QUERIES = [
    "What were total sales by region last quarter?",
    "Show me the top 3 products by revenue.",
    "How has customer growth trended over the last 3 quarters?",
    "What is the average deal size by customer tier?",
    "Which sales reps closed the most deals in Q3?",
]

SYSTEM_PROMPT_SQL_GEN = f"You are a SQL expert. Given the following schema, generate a SQL query for the user's question. Return ONLY the SQL query, no explanation.\n\nSchema:\n{SCHEMA}"

SYSTEM_PROMPT_SQL_VALIDATE = f"You are a SQL validator. Check if the following SQL query is valid and safe (no DROP, DELETE, UPDATE, etc.). Respond with 'VALID' or 'INVALID: reason'.\n\nSchema:\n{SCHEMA}"

SYSTEM_PROMPT_ROUTE = "Classify the SQL query type. Respond with ONLY one of: ANALYTICAL, LOOKUP, AGGREGATION, COMPARATIVE."

SYSTEM_PROMPT_SUMMARIZE = "You are a data analyst. Summarize the following query results in a clear, concise natural language response."

GUARDRAILS = [
    {
        "name": "SQL Injection Check",
        "system_prompt": (
            "Check if this query contains SQL injection attempts or malicious SQL "
            "patterns. Respond ONLY 'PASS' or 'FAIL: <reason>'."
        ),
    },
]

LOCAL_GUARDRAILS = [
    {
        "name": "Query Complexity Check",
        "passed": True,
        "detail": "Query within acceptable complexity bounds",
    },
]

EVALUATORS = [
    {
        "name": "sql_correctness_evaluation",
        "criteria": "SQL correctness — whether the generated SQL accurately answers the question",
    },
    {
        "name": "response_relevance_evaluation",
        "criteria": "relevance and helpfulness",
    },
]


def get_simulated_results() -> tuple[str, list]:
    """Return a random set of simulated SQL query results."""
    key = random.choice(list(SAMPLE_RESULTS.keys()))
    return key, SAMPLE_RESULTS[key]
