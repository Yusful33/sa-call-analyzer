"""
Shared query sampling and run helpers for use cases.
Uses contracts.QuerySpec and contracts.RunPlan; runners use common_runner_utils.
"""

import random
from typing import Any

from .contracts import QuerySpec, SAMPLING_VERSION


def industry_key(industry: str | None) -> str:
    """Normalize industry string to a key used by industry-aware query pools (generic, chatbot, rag, text_to_sql)."""
    if not industry:
        return "default"
    lower = (industry or "").lower()
    if any(k in lower for k in ["financial", "fintech", "finance", "bank", "banking"]):
        return "financial"
    if any(k in lower for k in ["insurance"]):
        return "insurance"
    if any(k in lower for k in ["retail", "ecommerce", "e-commerce", "grocery", "store"]):
        return "retail"
    if any(k in lower for k in ["health", "pharma", "medical", "biotech"]):
        return "healthcare"
    if any(k in lower for k in ["technology", "software", "saas", "tech"]):
        return "technology"
    if any(k in lower for k in ["travel", "hospitality", "tourism", "airline", "hotel"]):
        return "travel"
    return "default"


def weighted_sample(
    pools: dict[str, list[str]],
    weights: dict[str, float] | None = None,
    rng: random.Random | None = None,
    skip_empty: bool = True,
    strict_weights: bool = False,
) -> tuple[str, str]:
    """
    Sample one query from pools with optional per-pool weights.
    Returns (chosen_query, pool_name) so callers can attach intent to traces.

    - Weights: If provided, keys must match pool keys. Weights are normalized to sum 1.0.
      Error on negative or mismatched keys when strict_weights=True.
    - Empty pools: Skipped by default (skip_empty=True); no error in production.
    - RNG: If None, uses random module (non-deterministic). If provided, runs are reproducible.
    """
    r = rng if rng is not None else random
    # Filter to non-empty pools
    available = {k: v for k, v in pools.items() if v} if skip_empty else {k: v for k, v in pools.items() if v}
    if not available:
        raise ValueError("weighted_sample: no non-empty pools")

    if weights is not None:
        # Restrict to keys that exist in available
        w = {k: float(weights[k]) for k in available if k in weights}
        if strict_weights and set(weights) != set(available):
            raise ValueError("weighted_sample: weight keys must match pool keys")
        for v in w.values():
            if v < 0:
                raise ValueError("weighted_sample: weights must be non-negative")
        total = sum(w.values())
        if total <= 0:
            raise ValueError("weighted_sample: total weight must be positive")
        # Normalize
        weights_normalized = {k: v / total for k, v in w.items()}
    else:
        weights_normalized = {k: 1.0 / len(available) for k in available}

    roll = r.random()
    cum = 0.0
    for pool_name, queries in available.items():
        p = weights_normalized[pool_name]
        cum += p
        if roll < cum:
            return (r.choice(queries), pool_name)
    # Fallback: last pool
    last_name = list(available)[-1]
    return (r.choice(available[last_name]), last_name)


def get_query_for_run(
    module: Any,
    prospect_context: dict | None = None,
    rng: random.Random | None = None,
    **kwargs: Any,
) -> QuerySpec:
    """
    Get the next query (and intent) for a demo run.
    Calls module.sample_query(prospect_context=..., rng=..., **kwargs) if present;
    else random.choice(module.QUERIES) with rng. Always passes prospect_context and rng.
    Returns QuerySpec; if use case returns str, wraps as QuerySpec(text=s, meta={}).
    """
    if hasattr(module, "sample_query"):
        out = module.sample_query(prospect_context=prospect_context, rng=rng, **kwargs)
        if isinstance(out, str):
            return QuerySpec(text=out, meta={})
        if isinstance(out, QuerySpec):
            return out
        return QuerySpec(text=str(out), meta={})

    queries = getattr(module, "QUERIES", None)
    if not queries:
        raise ValueError(f"Use case module {getattr(module, '__name__', module)} has no sample_query and no QUERIES")
    r = rng if rng is not None else random
    chosen = r.choice(queries)
    return QuerySpec(text=chosen, intent=None, tags=None, meta={})
