"""
Arize-otel tracing setup + OpenInference auto-instrumentors.
Call `setup_tracing()` once at app startup.
Use `get_tracer()` to get a tracer for creating parent spans.

We only attach LangChainInstrumentor (not OpenAIInstrumentor) so each LLM call
produces a single span (e.g. ChatOpenAI). Enabling both would create duplicate
ChatOpenAI + ChatCompletion spans with identical token counts and cost.
"""

import os
from contextlib import contextmanager

from opentelemetry import trace
from arize.otel import register

from openinference.instrumentation.langchain import LangChainInstrumentor


_tracer_provider = None
_tracer = None


def setup_tracing() -> None:
    """Register arize-otel tracer provider and attach OpenInference instrumentors."""
    global _tracer_provider, _tracer

    space_id = os.getenv("ARIZE_SPACE_ID")
    api_key = os.getenv("ARIZE_API_KEY")
    project_name = os.getenv("ARIZE_PROJECT_NAME", "demo-trace-generator")
    endpoint = os.getenv("ARIZE_OTLP_ENDPOINT")

    kwargs = {}
    if space_id:
        kwargs["space_id"] = space_id
    if api_key:
        kwargs["api_key"] = api_key
    if project_name:
        kwargs["project_name"] = project_name
    if endpoint:
        kwargs["endpoint"] = endpoint

    _tracer_provider = register(**kwargs, batch=True)

    # LangChain only; skip OpenAI to avoid duplicate ChatOpenAI + ChatCompletion spans.
    # separate_trace_from_runtime_context=False so instrumented spans parent to our root span
    # (avoids orphaned spans and ensures traces show correctly in Arize).
    LangChainInstrumentor().instrument(
        tracer_provider=_tracer_provider,
        separate_trace_from_runtime_context=False,
    )

    _tracer = trace.get_tracer("arize-demo-trace-service")


def get_tracer() -> trace.Tracer:
    """Return the global tracer (call after setup_tracing)."""
    global _tracer
    if _tracer is None:
        _tracer = trace.get_tracer("arize-demo-trace-service")
    return _tracer


@contextmanager
def traced_pipeline(name: str, attributes: dict | None = None):
    """
    Context manager that creates a parent span.
    All auto-instrumented calls inside this block inherit the trace context.
    """
    tracer = get_tracer()
    with tracer.start_as_current_span(name, attributes=attributes or {}) as span:
        yield span
