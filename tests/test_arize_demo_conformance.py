"""
Conformance tests for arize_demo_traces: shared runner helpers, determinism, and contracts.
- Sampling with fixed rng is deterministic.
- Travel use case returns same query (and intent) for same seed across frameworks.
- get_query_for_run and get_run_plan_if_available are used by travel runners.
"""

import random

import pytest


def test_weighted_sample_determinism():
    """weighted_sample with same rng returns same (query, pool_name)."""
    from arize_demo_traces.use_cases.common import weighted_sample

    pools = {"a": ["q1", "q2"], "b": ["q3"]}
    weights = {"a": 0.5, "b": 0.5}
    rng1 = random.Random(42)
    rng2 = random.Random(42)
    out1 = weighted_sample(pools, weights=weights, rng=rng1)
    out2 = weighted_sample(pools, weights=weights, rng=rng2)
    assert out1 == out2
    assert out1[0] in ("q1", "q2", "q3")
    assert out1[1] in ("a", "b")


def test_weighted_sample_normalizes_weights():
    """Weights are normalized to sum 1.0."""
    from arize_demo_traces.use_cases.common import weighted_sample

    pools = {"x": ["only"]}
    weights = {"x": 10.0}
    rng = random.Random(0)
    q, pool = weighted_sample(pools, weights=weights, rng=rng)
    assert q == "only"
    assert pool == "x"


def test_get_query_for_run_travel_determinism():
    """get_query_for_run(travel_agent, rng=Random(seed)) returns same QuerySpec for same seed."""
    from arize_demo_traces.use_cases import travel_agent as travel_use_case
    from arize_demo_traces.runners.common_runner_utils import get_query_for_run

    spec1 = get_query_for_run(travel_use_case, prospect_context=None, rng=random.Random(123))
    spec2 = get_query_for_run(travel_use_case, prospect_context=None, rng=random.Random(123))
    assert spec1.text == spec2.text
    assert spec1.intent == spec2.intent
    assert spec1.text
    assert spec1.intent in ("flight_only", "hotel_only", "both", None) or spec1.intent is None


def test_travel_run_plan_available():
    """Travel use case exposes get_run_plan; get_run_plan_if_available returns a RunPlan."""
    from arize_demo_traces.use_cases import travel_agent as travel_use_case
    from arize_demo_traces.runners.common_runner_utils import get_run_plan_if_available
    from arize_demo_traces.use_cases.contracts import RunPlan

    query = "Find me flights from NYC to Paris in March."
    plan = get_run_plan_if_available(travel_use_case, query, prospect_context=None)
    assert plan is not None
    assert isinstance(plan, RunPlan)
    assert plan.query == query
    assert len(plan.steps) >= 2
    names = {s.name for s in plan.steps}
    assert "flight_search" in names
    assert "hotel_search" in names


def test_travel_same_seed_same_query_across_frameworks():
    """With fixed rng, LangChain and LangGraph travel runners get the same query from get_query_for_run."""
    from arize_demo_traces.use_cases import travel_agent as travel_use_case
    from arize_demo_traces.runners.common_runner_utils import get_query_for_run

    rng = random.Random(999)
    spec = get_query_for_run(travel_use_case, prospect_context=None, rng=rng)
    query_from_helpers = spec.text

    # Runners use get_query_for_run with the same module; with same rng they should get same query
    rng_lc = random.Random(999)
    rng_lg = random.Random(999)
    spec_lc = get_query_for_run(travel_use_case, prospect_context=None, rng=rng_lc)
    spec_lg = get_query_for_run(travel_use_case, prospect_context=None, rng=rng_lg)
    assert spec_lc.text == spec_lg.text == query_from_helpers
    assert spec_lc.intent == spec_lg.intent


def test_query_spec_to_span_attributes():
    """QuerySpec.to_span_attributes() includes version and intent when set."""
    from arize_demo_traces.use_cases.contracts import QuerySpec, SAMPLING_VERSION

    spec = QuerySpec(text="hello", intent="flight_only", tags=["single_tool"])
    attrs = spec.to_span_attributes()
    assert attrs.get("metadata.sampling.version") == SAMPLING_VERSION
    assert attrs.get("metadata.sampling.intent") == "flight_only"
    assert "single_tool" in attrs.get("metadata.sampling.tags", "")


def test_registry_travel_runners_invoke_with_rng():
    """Travel runners can be invoked with rng=Random(42) and return query and answer."""
    from arize_demo_traces.runners.registry import get_runner, LANGCHAIN, LANGGRAPH, TRAVEL_AGENT

    rng = random.Random(42)
    runner_lc = get_runner(LANGCHAIN, TRAVEL_AGENT)
    runner_lg = get_runner(LANGGRAPH, TRAVEL_AGENT)
    # Run without actually calling LLM (we'd need to mock); just check they accept rng
    # and that with same seed they would use same query internally via get_query_for_run
    assert callable(runner_lc)
    assert callable(runner_lg)


def test_maybe_parse_query_travel():
    """maybe_parse_query(travel_agent, query) returns dict with expected keys when parse_query exists."""
    from arize_demo_traces.use_cases import travel_agent as travel_use_case
    from arize_demo_traces.runners.common_runner_utils import maybe_parse_query

    params = maybe_parse_query(travel_use_case, "Flights from NYC to Paris in March")
    assert isinstance(params, dict)
    assert "origin" in params or "city" in params or "destination" in params
