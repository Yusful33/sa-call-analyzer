"""
Shared contracts for use-case modules and runners.

Runners must use these types and the documented signatures so behavior stays
consistent across frameworks. Use cases implement the optional functions
they need; runners never interpret opaque return values (e.g. select_tools_for_query)
except via get_run_plan or apply_tool_selection.
"""

from dataclasses import dataclass, field
from typing import Any

# Sampling config version for traceability and eval segmentation
SAMPLING_VERSION = "v2_pool_weighted"


@dataclass
class QuerySpec:
    """Lightweight spec for a sampled query; intent is first-class for traces and evals."""
    text: str
    intent: str | None = None  # e.g. pool name: "flight_only", "both"
    tags: list[str] | None = None  # e.g. ["single_tool", "needs_retrieval"]
    meta: dict[str, Any] = field(default_factory=dict)

    def to_span_attributes(self) -> dict[str, Any]:
        """Attributes to attach to the root span (e.g. metadata.sampling.*)."""
        out = {"metadata.sampling.version": SAMPLING_VERSION}
        if self.intent is not None:
            out["metadata.sampling.intent"] = self.intent
        if self.tags:
            out["metadata.sampling.tags"] = ",".join(self.tags)
        if self.meta:
            for k, v in self.meta.items():
                out[f"metadata.sampling.{k}"] = v
        return out


@dataclass
class StepSpec:
    """Single step in a run plan (e.g. flight_search, hotel_search)."""
    name: str
    enabled: bool
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunPlan:
    """Framework-agnostic plan: which steps to run and with what params."""
    query: str
    steps: list[StepSpec]
    meta: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Contract signatures (docblock only; runners and use cases follow these)
# ---------------------------------------------------------------------------
#
# Query sampling (required for "get query" flow):
#
#   sample_query(prospect_context=None, rng=None, **kwargs) -> str | QuerySpec
#   - Returns the user-facing input. If use case returns QuerySpec, runners use
#     .text for the agent and attach .meta/.intent/.tags to the trace.
#   - Backward compat: returning str is treated as QuerySpec(text=s, meta={}).
#
# Tool/step selection (opaque to runners):
#
#   select_tools_for_query(query: str) -> Any
#   - Return type is use-case-specific and OPAQUE. Runners must not branch on
#     the shape. They only use it via get_run_plan or apply_tool_selection.
#
# Query parsing (when used):
#
#   parse_query(query: str) -> dict[str, Any]
#   - Returns a single dict of params. Runners call only when use case defines it.
#
# Optional:
#
#   get_run_plan(query: str, prospect_context=None) -> RunPlan
#   - Returns which steps are enabled and with what params. Runners translate
#     RunPlan to their framework (LangGraph edges, LangChain tools, etc.).
#
