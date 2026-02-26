"""Generic LLM chain use-case: shared prompts, queries, guardrails, evaluators."""

import random

from .common import industry_key

# Default (tech/LLM) and industry-tailored query pools for prospect-aware demos
QUERIES = [
    "What are the key differences between microservices and monolithic architectures?",
    "Explain how rate limiting works in API gateways.",
    "What are best practices for securing LLM applications in production?",
    "How do embedding models work at a high level?",
    "Summarize the benefits of observability in AI systems.",
]

INDUSTRY_QUERIES: dict[str, list[str]] = {
    "default": QUERIES,
    "technology": QUERIES,
    "retail": [
        "What are best practices for inventory forecasting and demand planning?",
        "How can we reduce shrink and loss prevention in stores?",
        "What metrics matter most for grocery and fresh supply chain?",
        "How do loyalty programs affect customer retention and basket size?",
        "What are effective ways to personalize promotions at scale?",
    ],
    "financial": [
        "What are key considerations for fraud detection in transaction systems?",
        "How do banks typically approach credit risk modeling?",
        "What are best practices for regulatory compliance in financial reporting?",
        "How can we improve customer onboarding and KYC workflows?",
        "What metrics matter for deposit and lending growth?",
    ],
    "healthcare": [
        "What are best practices for patient data privacy and HIPAA compliance?",
        "How can we improve clinical documentation and coding accuracy?",
        "What are key considerations for drug discovery and clinical trials?",
        "How do health systems typically approach readmission reduction?",
        "What metrics matter for quality of care and outcomes?",
    ],
    "travel": [
        "What are best practices for dynamic pricing in hospitality?",
        "How can we improve guest experience and personalization?",
        "What are key considerations for airline and hotel distribution?",
        "How do loyalty programs drive repeat bookings?",
        "What metrics matter for occupancy and revenue per available room?",
    ],
}


def get_queries_for_prospect(prospect_context: dict | None) -> list[str]:
    """Return queries tailored to the prospect's industry."""
    if not prospect_context:
        return INDUSTRY_QUERIES["default"]
    key = industry_key(prospect_context.get("industry"))
    return INDUSTRY_QUERIES.get(key, INDUSTRY_QUERIES["default"])


def sample_query(prospect_context=None, rng=None, **kwargs):
    """Contract: sample_query(prospect_context, rng) -> str. Industry-aware sampling."""
    queries = get_queries_for_prospect(prospect_context)
    if rng is not None:
        return rng.choice(queries)
    return random.choice(queries)

SYSTEM_PROMPT = "You are a helpful, knowledgeable assistant. Provide clear and concise answers."

GUARDRAILS = [
    {
        "name": "content_safety_check",
        "system_prompt": (
            "You are a content safety filter. Check if the input contains unsafe, "
            "harmful, or inappropriate content. "
            "Respond ONLY 'PASS' or 'FAIL: <reason>'."
        ),
    },
]

# ---- Tools (simulated) ----

def web_search(query: str) -> str:
    """Search the web for relevant information."""
    import json as _json
    return _json.dumps({
        "results": [
            {"title": f"Understanding {query[:40]}", "url": "https://docs.example.com/article-1", "snippet": f"Comprehensive guide covering {query[:60]}. Key concepts include scalability, reliability, and best practices."},
            {"title": f"Best Practices for {query[:30]}", "url": "https://blog.example.com/best-practices", "snippet": "Industry-standard approaches and recommendations from leading practitioners."},
            {"title": f"{query[:30]} - Technical Deep Dive", "url": "https://engineering.example.com/deep-dive", "snippet": "Detailed technical analysis with benchmarks and real-world case studies."},
        ]
    })


def get_current_context() -> str:
    """Retrieve current context and system information."""
    import json as _json
    return _json.dumps({
        "timestamp": "2025-12-15T10:30:00Z",
        "environment": "production",
        "model_version": "v2.1.0",
        "available_knowledge_bases": ["technical-docs", "api-reference", "best-practices"],
        "session_context": "enterprise-support",
    })


EVALUATORS = [
    {
        "name": "conciseness_evaluation",
        "criteria": "conciseness — whether the response is clear and avoids unnecessary verbosity",
    },
]
