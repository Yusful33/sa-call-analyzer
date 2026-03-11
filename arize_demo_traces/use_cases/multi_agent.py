"""Multi-agent orchestration use-case: shared prompts, queries, agent configs, guardrails, evaluators."""

import random

from .common import industry_key

QUERIES = [
    "Research the latest trends in sustainable energy and draft a 500-word executive brief with investment recommendations.",
    "Analyze our Q3 customer churn data and produce an action plan to improve retention by 15%.",
    "Review the competitor landscape for our new product launch and create a positioning document.",
    "Investigate the root cause of the production outage last week and write a post-mortem report.",
    "Evaluate three vendor proposals for our cloud migration and produce a recommendation with cost analysis.",
    "Summarize recent regulatory changes in financial services and assess impact on our compliance posture.",
    "Research best practices for LLM deployment in healthcare and draft internal guidelines.",
    "Analyze customer feedback from the last quarter and identify the top 5 feature requests with business justification.",
]

# Industry-tailored multi-agent tasks for prospect-aware demos
INDUSTRY_QUERIES: dict[str, list[str]] = {
    "default": QUERIES,
    "technology": QUERIES,
    "financial": [
        "Research recent SEC and FINRA regulatory updates and draft a compliance impact brief for our trading platform.",
        "Analyze our Q3 trading volume and margin usage data and produce an action plan to reduce risk exposure.",
        "Review competitor fee structures and create a positioning document for our brokerage offerings.",
        "Investigate the root cause of the settlement delay last week and write a post-mortem report.",
        "Evaluate three vendors for market data feeds and produce a recommendation with cost analysis.",
        "Summarize recent regulatory changes in financial services and assess impact on our compliance posture.",
        "Research best practices for fraud detection in real-time payment systems and draft internal guidelines.",
        "Analyze customer feedback from the last quarter and identify the top 5 feature requests with business justification.",
    ],
    "travel": [
        "Research the latest trends in airline ancillary revenue and draft a 500-word brief with recommendations.",
        "Analyze our Q3 on-time performance and customer satisfaction data and produce an action plan.",
        "Review competitor loyalty programs and create a positioning document for our miles program.",
        "Investigate the root cause of the operational disruption last week and write a post-mortem report.",
        "Evaluate three vendors for crew scheduling and produce a recommendation with cost analysis.",
        "Summarize recent DOT and FAA regulatory updates and assess impact on our operations.",
        "Research best practices for dynamic pricing in airline distribution and draft internal guidelines.",
        "Analyze customer feedback from the last quarter and identify the top 5 complaints with remediation plans.",
    ],
    "retail": [
        "Research the latest trends in omnichannel retail and draft a brief with investment recommendations.",
        "Analyze our Q3 inventory turnover and markdown data and produce an action plan.",
        "Review competitor loyalty programs and create a positioning document.",
        "Investigate the root cause of the fulfillment delay last week and write a post-mortem report.",
        "Evaluate three vendors for demand forecasting and produce a recommendation with cost analysis.",
        "Summarize recent consumer privacy regulations and assess impact on our data practices.",
        "Research best practices for shrink prevention and draft internal guidelines.",
        "Analyze customer feedback from the last quarter and identify the top 5 feature requests.",
    ],
    "healthcare": [
        "Research recent FDA guidance on AI in clinical decision support and draft a compliance brief.",
        "Analyze our Q3 readmission and quality metrics and produce an action plan.",
        "Review competitor EHR integrations and create a positioning document.",
        "Investigate the root cause of the clinical workflow issue last week and write a post-mortem report.",
        "Evaluate three vendors for patient engagement and produce a recommendation with cost analysis.",
        "Summarize recent HIPAA and state privacy updates and assess impact on our posture.",
        "Research best practices for clinical documentation and draft internal guidelines.",
        "Analyze patient feedback from the last quarter and identify the top 5 improvement areas.",
    ],
}


def get_queries_for_prospect(prospect_context: dict | None) -> list[str]:
    """Return multi-agent tasks tailored to the prospect's industry."""
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

# Agent role definitions for the multi-agent orchestration
AGENTS = [
    {
        "role": "Research Specialist",
        "goal": "Gather comprehensive, accurate information from available sources",
        "backstory": "You are an expert researcher who excels at finding relevant data, synthesizing information from multiple sources, and identifying key insights.",
    },
    {
        "role": "Analysis Expert",
        "goal": "Analyze data and research findings to extract actionable insights",
        "backstory": "You are a senior analyst skilled at interpreting data, identifying patterns, and drawing evidence-based conclusions.",
    },
    {
        "role": "Writer",
        "goal": "Produce clear, well-structured written outputs based on research and analysis",
        "backstory": "You are a professional writer who transforms complex analysis into clear, compelling documents tailored to the target audience.",
    },
    {
        "role": "Quality Reviewer",
        "goal": "Review outputs for accuracy, completeness, and quality before delivery",
        "backstory": "You are a meticulous reviewer who ensures deliverables meet high standards for factual accuracy, logical consistency, and clarity.",
    },
]

SUPERVISOR_PROMPT = """You are a supervisor agent coordinating a team of specialists. Break down the user's request into subtasks and delegate to the appropriate agents:
- Research Specialist: for information gathering
- Analysis Expert: for data analysis and insight extraction
- Writer: for drafting documents and reports
- Quality Reviewer: for final review and quality assurance

Plan the workflow and delegate tasks in the optimal order."""

RESEARCH_PROMPT = "You are a research specialist. Given the task, gather comprehensive information and present key findings in a structured format."

ANALYSIS_PROMPT = "You are an analysis expert. Given the research findings, analyze the data to extract actionable insights, identify patterns, and draw evidence-based conclusions."

WRITER_PROMPT = "You are a professional writer. Given the analysis and research, produce a clear, well-structured document that addresses the original request."

REVIEWER_PROMPT = "You are a quality reviewer. Review the following document for factual accuracy, logical consistency, completeness, and clarity. Provide a brief assessment and any corrections needed."

# ---- Tools ----

# Minimal delay so TOOL spans have non-zero duration in traces (avoids 0s warning in Arize).
_TOOL_SPAN_MIN_DURATION_SEC = 0.02


def search_web(query: str) -> str:
    """Search the web for research information."""
    import time as _time
    import json as _json
    _time.sleep(_TOOL_SPAN_MIN_DURATION_SEC)
    return _json.dumps({
        "query": query[:80],
        "results": [
            {"source": "Industry Report 2025", "summary": f"Key findings on {query[:50]}: market growing at 24% CAGR, major players consolidating, regulatory landscape evolving."},
            {"source": "Academic Research", "summary": "Recent studies indicate significant improvements in efficiency and cost reduction through AI-driven automation."},
            {"source": "Market Analysis", "summary": "Current trends show increased enterprise adoption with focus on ROI measurement and governance frameworks."},
        ],
        "total_results": 47,
    })


def analyze_metrics(metric_name: str) -> str:
    """Pull and analyze business metrics."""
    import time as _time
    import json as _json
    _time.sleep(_TOOL_SPAN_MIN_DURATION_SEC)
    return _json.dumps({
        "metric": metric_name,
        "current_value": 87.3,
        "previous_period": 82.1,
        "change_pct": 6.3,
        "trend": "increasing",
        "benchmark": 85.0,
        "status": "above_target",
        "breakdown": {
            "segment_a": 91.2,
            "segment_b": 84.5,
            "segment_c": 86.1,
        },
    })


GUARDRAILS = [
    {
        "name": "Scope Validation",
        "system_prompt": (
            "Check if the user request is within acceptable scope for an enterprise AI assistant. "
            "Reject requests for personal advice, illegal activities, or topics outside business use. "
            "Respond ONLY 'PASS' or 'FAIL: <reason>'."
        ),
    },
]

EVALUATORS = [
    {
        "name": "end_to_end_quality_evaluation",
        "criteria": "overall quality — whether the final output is comprehensive, accurate, and well-structured",
    },
    {
        "name": "agent_delegation_evaluation",
        "criteria": "delegation effectiveness — whether subtasks were appropriately broken down and delegated to the right specialists",
    },
]
