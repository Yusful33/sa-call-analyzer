"""
Centralized runner logic: every runner uses these helpers so query sampling,
parse, and run plan behavior stay consistent across frameworks.
"""

from typing import Any

from ..use_cases.common import get_query_for_run as _get_query_for_run
from ..use_cases.contracts import QuerySpec, RunPlan


def get_query_for_run(
    use_case_module: Any,
    prospect_context: dict | None = None,
    rng: Any = None,
    **kwargs: Any,
) -> QuerySpec:
    """
    Get the next query (and intent) for a demo run.
    Calls use_case_module.sample_query(prospect_context=..., rng=..., **kwargs) if present;
    else random.choice(use_case_module.QUERIES). Always passes prospect_context and rng.
    Returns QuerySpec; runners use .text for the agent and attach .to_span_attributes() to the root span.
    """
    return _get_query_for_run(use_case_module, prospect_context=prospect_context, rng=rng, **kwargs)


def maybe_parse_query(use_case_module: Any, query: str) -> dict[str, Any]:
    """
    Return use_case_module.parse_query(query) if the module defines it; else {}.
    Runners that need params (e.g. travel) use this; others ignore the return value.
    """
    if hasattr(use_case_module, "parse_query"):
        return use_case_module.parse_query(query)
    return {}


def get_run_plan_if_available(
    use_case_module: Any,
    query: str,
    prospect_context: dict | None = None,
) -> RunPlan | None:
    """
    Return use_case_module.get_run_plan(query, prospect_context) if present; else None.
    Runners translate RunPlan to their framework (LangGraph edges, LangChain tool calls, etc.).
    If None, runner falls back to "run all steps/tools".
    """
    if hasattr(use_case_module, "get_run_plan"):
        return use_case_module.get_run_plan(query, prospect_context=prospect_context)
    return None
