"""Generic LLM chain use-case: shared prompts, queries, guardrails, evaluators."""

QUERIES = [
    "What are the key differences between microservices and monolithic architectures?",
    "Explain how rate limiting works in API gateways.",
    "What are best practices for securing LLM applications in production?",
    "How do embedding models work at a high level?",
    "Summarize the benefits of observability in AI systems.",
]

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
