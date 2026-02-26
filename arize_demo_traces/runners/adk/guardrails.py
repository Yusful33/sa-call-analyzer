"""
ADK-style guardrails-only pipeline.
Runs all guardrail checks (LLM + rule-based) on a single input; no generation step.
"""

from ...cost_guard import CostGuard
from ...llm import get_chat_llm
from ...trace_enrichment import run_guardrail, run_local_guardrail
from ...use_cases.guardrails import GUARDRAILS, LOCAL_GUARDRAILS
from ..common_runner_utils import get_query_for_run


def run_guardrails(
    query: str | None = None,
    model: str = "gpt-4o-mini",
    guard: CostGuard | None = None,
    tracer_provider=None,
    prospect_context=None,
    degraded_output=None,
    trace_quality="good",
    **kwargs,
) -> dict:
    """Execute a guardrails-only pipeline: run all GUARDRAIL spans on the input."""
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode

    from ...use_cases import guardrails as guardrails_use_case

    provider = tracer_provider or trace.get_tracer_provider()
    tracer = provider.get_tracer("demo.guardrails.adk")

    if not query:
        rng = kwargs.get("rng")
        _kw = {k: v for k, v in kwargs.items() if k != "rng"}
        query_spec = get_query_for_run(guardrails_use_case, prospect_context=prospect_context, rng=rng, **_kw)
        query = query_spec.text
    else:
        query_spec = None

    llm = get_chat_llm(model, temperature=0)
    results = []

    attrs = {
        "openinference.span.kind": "CHAIN",
        "input.value": query,
        "input.mime_type": "text/plain",
        "metadata.framework": "adk",
        "metadata.use_case": "guardrails",
    }
    if query_spec:
        attrs.update(query_spec.to_span_attributes())

    with tracer.start_as_current_span("guardrails_pipeline", attributes=attrs) as span:
        for g in GUARDRAILS:
            if guard:
                guard.check()
            passed = run_guardrail(
                tracer, g["name"], query, llm, guard,
                system_prompt=g["system_prompt"],
            )
            results.append(f"{g['name']}: {'PASS' if passed else 'FAIL'}")

        for lg in LOCAL_GUARDRAILS:
            passed = run_local_guardrail(
                tracer, lg["name"], query,
                passed=lg.get("passed", True),
                detail=lg.get("detail", ""),
            )
            results.append(f"{lg['name']}: {'PASS' if passed else 'FAIL'}")

        summary = "All checks passed" if all("PASS" in r for r in results) else "; ".join(results)
        if degraded_output:
            summary = degraded_output
        span.set_attribute("output.value", summary)
        span.set_attribute("output.mime_type", "text/plain")
        span.set_status(Status(StatusCode.OK))

    return {
        "query": query,
        "answer": summary,
        "result": summary,
    }
