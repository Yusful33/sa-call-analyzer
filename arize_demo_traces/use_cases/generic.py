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

EVALUATORS = [
    {
        "name": "conciseness_evaluation",
        "criteria": "conciseness — whether the response is clear and avoids unnecessary verbosity",
    },
]
