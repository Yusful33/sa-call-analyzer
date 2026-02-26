"""Text-to-SQL use-case: shared schema, queries, sample results, guardrails, evaluators."""

import random

SCHEMA = """
Tables:
  - sales (id INT, region VARCHAR, product VARCHAR, amount DECIMAL, quarter VARCHAR, sales_rep VARCHAR)
  - regions (id INT, name VARCHAR, country VARCHAR, manager VARCHAR)
  - products (id INT, name VARCHAR, category VARCHAR, price DECIMAL)
  - customers (id INT, name VARCHAR, region_id INT, tier VARCHAR, signup_date DATE)
"""

# Industry / domain-specific natural language queries for prospect-tailored demos.
# Keys match common industries from CRM; "default" used when no match.
INDUSTRY_QUERIES: dict[str, list[str]] = {
    "default": [
        "What were total sales by region last quarter?",
        "Show me the top 3 products by revenue.",
        "How has customer growth trended over the last 3 quarters?",
        "What is the average deal size by customer tier?",
        "Which sales reps closed the most deals in Q3?",
    ],
    "financial": [
        "What were total transactions by region last quarter?",
        "Show me the top 3 product lines by revenue.",
        "How has customer acquisition trended over the last 3 quarters?",
        "What is the average balance by customer tier?",
        "Which regions had the highest compliance-related activity?",
    ],
    "banking": [
        "What were total deposits by region last quarter?",
        "Show me the top 3 branches by loan volume.",
        "How has account growth trended over the last 3 quarters?",
        "What is the average account size by segment?",
        "Which segments had the most new accounts in Q3?",
    ],
    "insurance": [
        "What were total premiums by region last quarter?",
        "Show me the top 3 policy types by revenue.",
        "How has policy retention trended over the last 3 quarters?",
        "What is the average claim amount by product line?",
        "Which regions had the highest claims volume?",
    ],
    "retail": [
        "What were total orders by region last quarter?",
        "Show me the top 3 SKUs by revenue.",
        "How has order volume trended over the last 3 quarters?",
        "What is the average order value by customer segment?",
        "Which stores had the highest sales in Q3?",
    ],
    "healthcare": [
        "What were total patient visits by region last quarter?",
        "Show me the top 3 procedures by volume.",
        "How has patient volume trended over the last 3 quarters?",
        "What is the average length of stay by department?",
        "Which departments had the most appointments in Q3?",
    ],
    "technology": [
        "What were total bookings by region last quarter?",
        "Show me the top 3 product tiers by ARR.",
        "How has pipeline trended over the last 3 quarters?",
        "What is the average contract value by segment?",
        "Which sales reps closed the most in Q3?",
    ],
    "travel": [
        "What were total bookings by region last quarter?",
        "Show me the top 3 routes by revenue.",
        "How has booking volume trended over the last 3 quarters?",
        "What is the average booking value by customer tier?",
        "Which destinations had the most bookings in Q3?",
    ],
}

# Normalize industry string to a key in INDUSTRY_QUERIES
def _industry_key(industry: str | None) -> str:
    if not industry:
        return "default"
    lower = industry.lower()
    if any(k in lower for k in ["financial", "fintech", "finance"]):
        return "financial"
    if any(k in lower for k in ["bank", "banking"]):
        return "banking"
    if any(k in lower for k in ["insurance"]):
        return "insurance"
    if any(k in lower for k in ["retail", "ecommerce", "e-commerce"]):
        return "retail"
    if any(k in lower for k in ["health", "pharma", "medical", "biotech"]):
        return "healthcare"
    if any(k in lower for k in ["technology", "software", "saas", "tech"]):
        return "technology"
    if any(k in lower for k in ["travel", "hospitality", "tourism", "airline", "hotel"]):
        return "travel"
    return "default"


def get_queries_for_prospect(prospect_context: dict | None) -> list[str]:
    """Return natural language queries tailored to the prospect's industry/context."""
    if not prospect_context:
        return INDUSTRY_QUERIES["default"]
    industry = prospect_context.get("industry") or "General"
    key = _industry_key(industry)
    return INDUSTRY_QUERIES.get(key, INDUSTRY_QUERIES["default"])


def sample_query(prospect_context=None, rng=None, **kwargs):
    """Contract: sample_query(prospect_context, rng) -> str. Uses get_queries_for_prospect and rng for determinism."""
    queries = get_queries_for_prospect(prospect_context)
    if rng is not None:
        return rng.choice(queries)
    return random.choice(queries)


def get_context_preamble(prospect_context: dict | None) -> str:
    """Return a short preamble for system prompts so the demo is framed for the prospect."""
    if not prospect_context:
        return ""
    name = prospect_context.get("account_name") or "this prospect"
    industry = prospect_context.get("industry") or "General"
    summary = (prospect_context.get("summary") or "").strip()
    if summary:
        return f"Demo context: {name} ({industry}). {summary[:250]}\n\n"
    return f"Demo context: {name}, {industry}.\n\n"


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


# ---- Tools (simulated) ----

def get_table_schema(table_name: str) -> str:
    """Look up the schema for a database table."""
    schemas = {
        "sales": "sales (id INT, region VARCHAR, product VARCHAR, amount DECIMAL, quarter VARCHAR, sales_rep VARCHAR)",
        "regions": "regions (id INT, name VARCHAR, country VARCHAR, manager VARCHAR)",
        "products": "products (id INT, name VARCHAR, category VARCHAR, price DECIMAL)",
        "customers": "customers (id INT, name VARCHAR, region_id INT, tier VARCHAR, signup_date DATE)",
    }
    return schemas.get(table_name, schemas.get("sales", SCHEMA))


def execute_query(sql: str) -> str:
    """Execute a SQL query and return results."""
    import json as _json
    _key, results = get_simulated_results()
    return _json.dumps({"status": "success", "rows_returned": len(results), "data": results}, default=str)


def get_simulated_results() -> tuple[str, list]:
    """Return a random set of simulated SQL query results."""
    key = random.choice(list(SAMPLE_RESULTS.keys()))
    return key, SAMPLE_RESULTS[key]
