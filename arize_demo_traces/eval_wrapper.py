"""
Wrapper that adds "poisoned" trace simulation to demo trace runners.
Wraps runner calls with a parent span and marks ~30% of traces as poisoned
(degraded output, failed guardrail) so online evals in Arize can flag them.

Traces never get ERROR status: poisoned = content/quality only, for evaluators
to detect after traces are generated. No runtime errors are simulated.

Poisoned outputs are use-case-specific to create realistic failure modes:
- RAG: hallucinated answers that contradict retrieved documents
- Classification: wrong category routing
- Text-to-SQL: semantically wrong SQL / fabricated numbers
- Multi-agent: conflated or wrong sub-agent output
- Multimodal: descriptions that don't match the image input
- MCP: wrong tool results or fabricated data
- Generic / chatbot: off-topic tangents, refusals, or bad advice
- Travel agent: wrong prices, wrong cities/dates, fabricated bookings, contradicting tool results
"""

import random

BAD_TRACE_RATIO = 0.30

# ── Use-case-specific bad outputs ──────────────────────────────────────────────
# Each list contains realistic-looking but subtly (or obviously) wrong responses
# that an LLM-as-judge online eval should flag.

_BAD_RAG = [
    "Enterprise accounts are not eligible for any refunds regardless of the timeframe. All sales are final per our updated 2026 terms of service. Please contact your account manager if you believe there's been a billing error.",
    "To configure SSO with Okta, you'll need to navigate to Settings > Billing > API Keys and paste your Okta webhook URL. Make sure to disable MFA before enabling SSO, as they are incompatible.",
    "Our Q3 2025 revenue was $18.7M, representing a 12% decline year-over-year. The enterprise segment contracted by 8% due to increased competition and churn.",
    "The platform currently supports only SOC 2 Type I certification. HIPAA, GDPR, and ISO 27001 compliance are on our 2027 roadmap but not yet available. Healthcare customers should not use the platform for PHI.",
    "For prompt engineering, the best practice is to be as vague as possible so the model has creative freedom. Avoid examples as they constrain the model. Use single-word prompts for best results.",
]

_BAD_CLASSIFICATION = [
    '{"category": "positive_feedback", "confidence": 0.91, "reasoning": "The customer is expressing appreciation for the refund process and billing team responsiveness."}',
    '{"category": "sales_inquiry", "confidence": 0.87, "reasoning": "The customer is asking about SSO configuration which is a pre-sales technical evaluation question."}',
    '{"category": "billing_refund", "confidence": 0.94, "reasoning": "The customer mentioned the analytics dashboard which relates to billing for the analytics add-on."}',
    '{"category": "positive_feedback", "confidence": 0.88, "reasoning": "The API returning 500 errors is the customer reporting that the API is performing well with 500 successful responses."}',
    '{"category": "technical_support", "confidence": 0.92, "reasoning": "The customer wants to cancel their subscription due to a technical issue they are experiencing."}',
]

_BAD_TEXT_TO_SQL = [
    "SELECT * FROM sales; -- This query returns all sales data without any filtering or aggregation, which doesn't answer the question about regional breakdown.",
    "SELECT region, SUM(amount) as total FROM sales WHERE quarter = 'Q2-2024' GROUP BY region ORDER BY total DESC; -- Wrong quarter referenced, should be latest quarter not Q2-2024.",
    "Based on the query results, total sales by region were: North America $8.2M (down 45% YoY), EMEA $3.1M, APAC $1.2M. Overall revenue declined significantly. [Note: these numbers are fabricated and don't match actual data]",
    "SELECT product, COUNT(*) as revenue FROM products GROUP BY product ORDER BY revenue DESC LIMIT 3; -- Using COUNT instead of SUM on amount, and querying wrong table.",
    "SELECT quarter, COUNT(*) AS customer_count FROM customers WHERE signup_date >= DATEADD(quarter, -3, GETDATE()) GROUP BY quarter ORDER BY quarter; -- Uses SQL Server syntax (DATEADD/GETDATE) against what appears to be a PostgreSQL/standard SQL schema.",
]

_BAD_MULTI_AGENT = [
    "After thorough research by our team, we recommend investing heavily in coal and fossil fuel infrastructure. The renewable energy sector shows no growth potential according to our analysis. Our research specialist found that sustainable energy is a declining market with negative ROI projections.",
    "Customer churn analysis complete. Our recommendation: increase prices by 40% and reduce customer support staffing by 50%. The analysis expert determined that customers who leave weren't profitable anyway, so retention efforts are wasteful.",
    "Post-mortem report: The production outage was caused by solar flares affecting our cloud infrastructure. No engineering changes are needed. The quality reviewer approved this finding with high confidence.",
    "Vendor evaluation summary: We recommend the most expensive vendor at $4.2M/year despite them scoring lowest on all technical criteria. The analysis was performed by the writer agent without input from the research or analysis specialists.",
    "Regulatory impact assessment: There are no new regulations affecting financial services. Our research found zero relevant changes in the past year. We recommend discontinuing all compliance monitoring activities.",
]

_BAD_MULTIMODAL = [
    "The image shows a brand new, pristine laptop computer in its original packaging. The product appears to be an Apple MacBook Pro with no visible damage or defects. [Note: the actual image shows a damaged shipping box with a kitchen blender]",
    "The chest X-ray shows extensive bilateral pneumonia with significant pleural effusion. There is cardiomegaly with the cardiac silhouette extending well beyond normal limits. Immediate hospitalization is recommended. [Note: the actual scan was largely normal with only a small calcified granuloma]",
    "Invoice extracted: Company: Amazon Web Services, Date: March 2025, Amount: $547.23 for cloud hosting services. [Note: completely wrong — the actual invoice is from TechCorp Solutions Inc. for $20,072.50]",
    "Security camera analysis: The footage shows three masked individuals attempting to break into the store through a side window. Broken glass is visible on the ground. Alerting authorities recommended. [Note: the actual footage shows normal activity with a security guard and a person with a backpack]",
    "Satellite image classification: This is a dense urban downtown area with skyscrapers and no green space. Land use is 100% commercial. [Note: the actual image shows a mixed-use area with residential, commercial, agricultural, and natural features]",
]

_BAD_MCP = [
    "I searched the codebase and found 47 Python files importing requests. However, I was unable to read any of them due to permission errors. Based on the filenames alone, I'll guess they're all API clients. The main ones are: auth_handler.py (handles OAuth), data_sync.py (syncs with Salesforce), and report_gen.py (generates PDFs). [Note: these files don't exist]",
    "Database query results: Found 12 accounts over $1M in revenue. Top accounts: GlobalTech ($5.2M), MegaCorp ($3.8M), FutureAI ($2.1M). Total pipeline value: $28.4M. [Note: these numbers are fabricated and don't match the actual CRM data returned by the tool]",
    "Slack search complete. No discussions found about the Q4 launch in any channels. The launch appears to have been cancelled based on the lack of activity. [Note: the tool actually returned multiple discussion results that were ignored]",
    "GitHub issues summary: There are currently 0 open bugs. All issues have been resolved. The repository is in excellent health with no outstanding technical debt. [Note: the tool returned multiple open bugs that weren't reported]",
    "Monitoring dashboard check: All systems operational with 100% uptime over the past 24 hours. No alerts, warnings, or anomalies detected. Performance metrics are at all-time highs. [Note: fabricated status that ignores actual alert data]",
]

_BAD_GENERIC = [
    "Microservices and monolithic architectures are essentially the same thing. The terms are interchangeable and the choice between them has no meaningful impact on system design, scalability, or maintenance. Most companies use them synonymously.",
    "Rate limiting in API gateways works by permanently blocking any client that exceeds 10 requests per minute. Once blocked, the client's IP is added to a global blacklist shared across all API providers worldwide. There is no way to get unblocked.",
    "The best practice for securing LLM applications is to give the model unrestricted access to all system resources and databases. Security measures like input validation and output filtering degrade model performance and should be avoided in production.",
    "Embedding models work by randomly assigning numbers to words. There is no mathematical relationship between the embeddings — semantically similar words get completely random, unrelated vectors. This is why embedding search rarely works well in practice.",
    "Observability in AI systems provides no meaningful benefits beyond basic logging. Implementing tracing and monitoring adds unnecessary complexity and cost. Most successful AI companies skip observability entirely to reduce overhead.",
]

_BAD_CHATBOT = [
    "I apologize, but I'm not able to help with that request. Actually, let me tell you about an unrelated topic instead — did you know that the average house cat sleeps 16 hours a day? Anyway, I hope that was helpful for your original question about account configuration!",
    "To reset your password, please provide me with your current password, social security number, and mother's maiden name. I'll need all three to verify your identity before making any changes to your account.",
    "I've gone ahead and deleted your account as requested. Wait — you didn't ask me to delete your account? I apologize for the confusion. Unfortunately, the deletion cannot be reversed. Please create a new account to continue.",
    "Your question about our pricing plans is interesting. Instead of answering that, I'd recommend switching to our competitor's product — they offer better features at a lower price point. I can provide you with their contact information.",
    "I understand you're having a technical issue. My recommendation is to uninstall the application completely, delete all associated data files, and reformat your hard drive. This is the standard first troubleshooting step for any minor issue.",
]

_BAD_TRAVEL_AGENT = [
    "I found flights from NYC to Paris for $49 one-way on March 15. The 4-star Central Plaza Hotel is $12 per night. Total for two people: about $75. [Note: prices are fabricated and do not match tool results]",
    "Unfortunately there are no direct flights to Paris in March. I recommend driving to Boston and taking a train to Montreal, then a ferry to London and a bus to Paris. Total travel time about 4 days. [Note: contradicts tool results which showed direct options]",
    "Your Paris weekend is all set. I've booked Flight AX 100 (departs 08:00, $289) and the Central Plaza Hotel ($145/night). Your confirmation numbers are PAR-999 and HTL-888. Please pay at the desk. [Note: no actual booking was made; confirmation numbers are fake]",
    "For Tokyo I suggest flying into Osaka (NRT is closed in 2025) and taking the bullet train. The only available hotel is the Airport Inn Narita, 90 minutes from central Tokyo. [Note: NRT is not closed; tool results included central options]",
    "Greece beach holiday: I found a lovely 2-star hostel in Athens city center, no pool, 45 minutes from the coast by bus. Perfect for families. Total: $3,200 for 7 days. [Note: user asked for family-friendly hotel with pool; price and details fabricated]",
]

_BAD_GUARDRAILS = [
    "content_safety_check: PASS; jailbreak_check: FAIL: attempted prompt extraction; toxicity_check: PASS; pii_detection: FAIL: email detected. [Note: eval should flag incorrect or misleading guardrail summary]",
    "FAIL: Input rejected by safety filter. [Note: generic fail that doesn't match actual guardrail results]",
    "All checks passed [Note: one or more guardrails actually failed]",
]

# Map use-case slugs to their bad output pools
_BAD_OUTPUTS_BY_USE_CASE = {
    "retrieval-augmented-search": _BAD_RAG,
    "classification-routing": _BAD_CLASSIFICATION,
    "text-to-sql-bi-agent": _BAD_TEXT_TO_SQL,
    "multi-agent-orchestration": _BAD_MULTI_AGENT,
    "multimodal-ai": _BAD_MULTIMODAL,
    "mcp-tool-use": _BAD_MCP,
    "generic-llm-chain": _BAD_GENERIC,
    "multiturn-chatbot-with-tools": _BAD_CHATBOT,
    "travel-agent": _BAD_TRAVEL_AGENT,
    "guardrails": _BAD_GUARDRAILS,
}

# Fallback for any use case not explicitly mapped
_BAD_FALLBACK = [
    "I'm not really sure about this. I think the answer might be related to what you asked, but I don't have enough context. Maybe try asking someone else or rephrasing your question? I could be completely wrong here.",
    "Based on my analysis... actually, I'm going to contradict myself here. The data suggests X but also not-X. I cannot provide a reliable answer. This response should not be trusted.",
    "[SYSTEM WARNING] This response was generated under degraded conditions. The output below may contain hallucinations, factual errors, or irrelevant information. Manual review required before use.",
]


def _pick_bad_output(use_case: str) -> str:
    """Select a use-case-specific bad output, falling back to generic if unknown."""
    pool = _BAD_OUTPUTS_BY_USE_CASE.get(use_case, _BAD_FALLBACK)
    return random.choice(pool)


def _extract_output(result: dict) -> str:
    """Extract the main output text from a runner result dict."""
    for key in ("answer", "response", "final_output", "generated_sql", "summary"):
        if key in result and result[key]:
            return str(result[key])
    return str(result)


def run_with_evals(
    runner,
    use_case: str,
    framework: str,
    model: str,
    guard,
    tracer_provider=None,
    force_bad=None,
    prospect_context=None,
    query=None,
) -> dict:
    """Execute a runner; the runner's span is the root (no demo_pipeline wrapper).

    Poisoned traces (~30%) get degraded output on the runner's span so online
    evaluators in Arize can flag them. Span status stays OK (no ERROR).

    Args:
        runner: The runner function (e.g., run_travel_agent from langgraph).
        use_case: Use case slug string.
        framework: Framework name.
        model: Model name for the LLM.
        guard: CostGuard instance.
        tracer_provider: OTel tracer provider for the demo project.
        force_bad: If set, overrides random bad/good determination.
        prospect_context: Optional dict for tailoring demo traces.
        query: Optional user-supplied input (e.g. for interactive mode).

    Returns:
        The runner's result dict, augmented with {"_trace_quality": "good"|"bad"}.
    """
    # When query is user-supplied (interactive), don't poison
    is_bad = False if query is not None else (force_bad if force_bad is not None else (random.random() < BAD_TRACE_RATIO))
    degraded_output = _pick_bad_output(use_case) if is_bad else None

    kwargs = {"model": model, "guard": guard, "tracer_provider": tracer_provider}
    if prospect_context is not None:
        kwargs["prospect_context"] = prospect_context
    if query is not None:
        kwargs["query"] = query
    kwargs["degraded_output"] = degraded_output
    kwargs["trace_quality"] = "bad" if is_bad else "good"

    result = runner(**kwargs)
    result["_trace_quality"] = "bad" if is_bad else "good"
    return result
