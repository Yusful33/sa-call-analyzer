"""
OpenInference observability integration for Arize AX.

This module sets up tracing for CrewAI and LangChain calls and exports telemetry to Arize AX.

IMPORTANT: SpanProcessor Integration
-------------------------------------
We use a SpanProcessor to modify span data BEFORE it reaches Arize by accessing
the internal _attributes dict. This works for:
- Data privacy (PII redaction)
- Cost optimization (truncating large payloads)
- Data quality (extracting JSON, cleaning outputs)

KEY INSIGHT:
- ReadableSpan is immutable from the PUBLIC API (no set_attribute method)
- BUT: span._attributes is a mutable dict we can modify directly
- This is a common pattern in OpenTelemetry extensions
"""
import os
import logging
import contextvars
from typing import Optional
from arize.otel import register
# NOTE: CrewAI Instrumentor disabled - creates separate root traces for each Crew.kickoff()
# from openinference.instrumentation.crewai import CrewAIInstrumentor
from openinference.instrumentation.langchain import LangChainInstrumentor
from openinference.instrumentation.litellm import LiteLLMInstrumentor
from openinference.instrumentation.openai import OpenAIInstrumentor
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from opentelemetry.trace import TracerProvider as TracerProviderBase
from span_processor_fixed import CleaningSpanProcessor


# Per-request provider so concurrent requests don't overwrite each other (LLM spans stay in the right project).
_request_tracer_provider: contextvars.ContextVar[Optional["TracerProviderBase"]] = contextvars.ContextVar(
    "request_tracer_provider", default=None
)


class _ProxyTracer(trace.Tracer):
    """
    Tracer that delegates to the current request's provider on each span creation.
    LangChain/OpenInference instrumentors cache the tracer at instrument() time; by
    returning a _ProxyTracer we ensure every start_span/start_as_current_span
    uses the provider from context (so ChatAnthropic and other LLM spans go to the
    right project and have correct parent/trace_id).
    """

    def __init__(self, proxy_provider: "_ProxyTracerProvider", name: str, version: str = "") -> None:
        self._proxy = proxy_provider
        self._name = name
        self._version = version

    def _get_tracer(self) -> "trace.Tracer":
        provider = _request_tracer_provider.get()
        if provider is None:
            provider = self._proxy._default
        return provider.get_tracer(self._name, self._version)

    def start_span(self, name: str, *args: object, **kwargs: object) -> "trace.Span":
        return self._get_tracer().start_span(name, *args, **kwargs)

    def start_as_current_span(self, name: str, *args: object, **kwargs: object):
        return self._get_tracer().start_as_current_span(name, *args, **kwargs)


class _ProxyTracerProvider(TracerProviderBase):
    """Delegates to the per-request provider (from context) so each request's spans stay in one project."""

    def __init__(self, default_provider: TracerProviderBase):
        self._default = default_provider

    def get_tracer(self, name: str, version: str = "", *args, **kwargs) -> "trace.Tracer":
        """Return a proxy tracer that uses the provider from context at span-creation time (so instrumentors that cache the tracer still get the right project)."""
        return _ProxyTracer(self, name, version)

    def set_current(self, provider: TracerProviderBase) -> None:
        """Legacy: set default when context is not used. Prefer set_request_tracer_provider()."""
        self._default = provider


_proxy_provider: Optional["_ProxyTracerProvider"] = None

# Enable debug logging for OpenTelemetry
logging.basicConfig(level=logging.INFO)
logging.getLogger("opentelemetry").setLevel(logging.DEBUG)
# Also enable debug for arize specifically
logging.getLogger("arize").setLevel(logging.DEBUG)
# Enable debug for HTTP exports
logging.getLogger("opentelemetry.exporter").setLevel(logging.DEBUG)
logging.getLogger("opentelemetry.sdk.trace.export").setLevel(logging.DEBUG)
# Enable gRPC logging to see actual export attempts
logging.getLogger("opentelemetry.exporter.otlp.proto.grpc.trace_exporter").setLevel(logging.DEBUG)
logging.getLogger("grpc").setLevel(logging.DEBUG)

# Four components → four Arize projects (same space)
# 1. Single Call Analysis – analyze, recap, calls-by-account, example
# 2. Prospect Overview – analyze-prospect, analyze-prospect-stream, prospect-overview
# 3. Hypothesis Generator – hypothesis-research
# 4. Custom Demo Builder – classify-demo, generate-demo-stream, generate-demo, export-script, create-online-evals
COMPONENT_SINGLE_CALL = "single_call"
COMPONENT_PROSPECT = "prospect"
COMPONENT_HYPOTHESIS = "hypothesis"
COMPONENT_DEMO = "demo"
PROJECT_SINGLE_CALL = "single-call-analysis"
PROJECT_PROSPECT = "prospect-overview"
PROJECT_HYPOTHESIS = "hypothesis-generator"
PROJECT_DEMO = "custom-demo-builder"
_PROVIDERS_BY_COMPONENT: dict[str, "trace.TracerProvider"] = {}
_DEFAULT_COMPONENT = COMPONENT_SINGLE_CALL


def _add_cleaning_processor(tracer_provider, enable_cleaning: bool) -> None:
    """Add CleaningSpanProcessor to the given provider's chain."""
    if not enable_cleaning:
        return
    cleaning_processor = CleaningSpanProcessor(enabled=True)
    if hasattr(tracer_provider, '_active_span_processor'):
        multi_processor = tracer_provider._active_span_processor
        if hasattr(multi_processor, '_span_processors'):
            existing_processors = multi_processor._span_processors
            multi_processor._span_processors = (cleaning_processor,) + existing_processors


def setup_observability(
    project_name: str = "sa-call-analyzer",
    arize_api_key: Optional[str] = None,
    arize_space_id: Optional[str] = None
):
    """
    Set up OpenInference instrumentation and export traces to Arize AX.

    Args:
        project_name: Name of the project for trace organization
        arize_api_key: Arize API key (defaults to ARIZE_API_KEY env var)
        arize_space_id: Arize space ID (defaults to ARIZE_SPACE_ID env var)
    """

    # Get credentials from environment variables directly
    # Using system environment variables (not .env file) to work with uv/venv
    api_key = arize_api_key or os.getenv("ARIZE_API_KEY")
    space_id = arize_space_id or os.getenv("ARIZE_SPACE_ID")

    # Verify credentials are present
    if api_key and space_id:
        print(f"✅ Arize credentials loaded (API Key: {len(api_key)} chars, Space ID: {len(space_id)} chars)")
        print(f"   Using Space ID: {space_id[:20]}...")

    if not api_key or not space_id:
        print("⚠️  WARNING: Arize credentials not found. Observability disabled.")
        print("   Set ARIZE_API_KEY and ARIZE_SPACE_ID in .env to enable tracing.")
        return None

    try:
        global _PROVIDERS_BY_COMPONENT
        enable_cleaning = os.getenv("ARIZE_ENABLE_OUTPUT_CLEANING", "true").lower() == "true"

        # Register four Arize projects (one per component). Create these in your space if they don't exist.
        projects = [
            (COMPONENT_SINGLE_CALL, PROJECT_SINGLE_CALL),   # Single Call Analysis
            (COMPONENT_PROSPECT, PROJECT_PROSPECT),         # Prospect Overview
            (COMPONENT_HYPOTHESIS, PROJECT_HYPOTHESIS),     # Hypothesis Generator
            (COMPONENT_DEMO, PROJECT_DEMO),                  # Custom Demo Builder
        ]
        for component, proj_name in projects:
            provider = register(
                space_id=space_id,
                api_key=api_key,
                project_name=proj_name,
                set_global_tracer_provider=False,
            )
            _add_cleaning_processor(provider, enable_cleaning)
            _PROVIDERS_BY_COMPONENT[component] = provider
            print(f"   ✅ Registered project: {proj_name} ({component})")

        if enable_cleaning:
            print("   ✅ Cleaning processor added to all project chains")

        default_provider = _PROVIDERS_BY_COMPONENT[COMPONENT_SINGLE_CALL]
        global _proxy_provider
        _proxy_provider = _ProxyTracerProvider(default_provider)
        trace.set_tracer_provider(_proxy_provider)

        # NOTE: CrewAI Instrumentor disabled because it creates separate root traces
        # for each Crew.kickoff() call, breaking the trace hierarchy.
        # We rely on manual spans in crew_analyzer.py instead.
        # 
        # UNCOMMENT BELOW FOR TESTING: Enable to see if CrewAI traces appear in Arize
        # This helps diagnose if CrewAI is bypassing LangChain's callback system
        enable_crewai_instrumentor = os.getenv("ENABLE_CREWAI_INSTRUMENTOR", "false").lower() == "true"
        if enable_crewai_instrumentor:
            from openinference.instrumentation.crewai import CrewAIInstrumentor
            print("   ⚠️  CrewAI Instrumentor enabled for testing (creates separate root traces)")
            CrewAIInstrumentor().instrument(
                tracer_provider=_proxy_provider,
                skip_dep_check=True
            )

        # Instrument OpenAI (for direct OpenAI SDK calls)
        openai_instrumentor = OpenAIInstrumentor()
        openai_instrumentor.instrument(
            tracer_provider=_proxy_provider,
            skip_dep_check=True
        )
        print("   ✅ OpenAI instrumentor initialized")

        # Instrument LangChain (AFTER processor registered).
        # separate_trace_from_runtime_context=False so LLM/chain spans parent to our
        # API spans (one trace per request, no orphan spans).
        langchain_instrumentor = LangChainInstrumentor()
        langchain_instrumentor.instrument(
            tracer_provider=_proxy_provider,
            skip_dep_check=True,
            separate_trace_from_runtime_context=False,
        )

        # Instrument LiteLLM (captures actual LLM calls with token counts)
        litellm_instrumentor = LiteLLMInstrumentor()
        litellm_instrumentor.instrument(
            tracer_provider=_proxy_provider,
            skip_dep_check=True
        )

        # Propagate trace context to new threads (BigQuery, thread pools, etc. create
        # threads that otherwise would have no parent span → orphan spans in Arize).
        try:
            from opentelemetry.instrumentation.threading import ThreadingInstrumentor
            ThreadingInstrumentor().instrument()
            print("   ✅ ThreadingInstrumentor enabled (trace context propagates to new threads)")
        except ImportError:
            print("   ⚠️  ThreadingInstrumentor not available (pip install opentelemetry-instrumentation-threading)")

        print("\n🔍 Instrumentation Verification:")
        if hasattr(langchain_instrumentor, '_tracer') and langchain_instrumentor._tracer:
            print("   ✅ LangChain instrumentor has tracer attached")
        else:
            print("   ❌ LangChain instrumentor tracer is None!")

        print("\n✅ OpenInference tracing enabled (4 projects)")
        print(f"   📊 Projects: {PROJECT_SINGLE_CALL}, {PROJECT_PROSPECT}, {PROJECT_HYPOTHESIS}, {PROJECT_DEMO}")
        print(f"   🔗 View traces at: https://app.arize.com/organizations")
        print(f"   🌐 Space ID: {space_id[:20]}...")

        return default_provider

    except Exception as e:
        print(f"⚠️  WARNING: Failed to initialize observability: {e}")
        print("   Application will continue without tracing.")
        import traceback
        traceback.print_exc()
        return None


def get_tracer(name: str = "sa-call-analyzer"):
    """Get a tracer for manual instrumentation."""
    return trace.get_tracer(name)


# Span name prefix for all API and workflow steps (filter in Arize by "sa_call_analyzer")
SPAN_PREFIX = "sa_call_analyzer"


def get_provider_for_component(component: str) -> "trace.TracerProvider | None":
    """Return the tracer provider for the given component (for middleware)."""
    return _PROVIDERS_BY_COMPONENT.get(component) or _PROVIDERS_BY_COMPONENT.get(_DEFAULT_COMPONENT)


def set_request_tracer_provider(provider: "trace.TracerProvider | None") -> None:
    """Set the active tracer provider for this request (called by middleware). Stored in context so concurrent requests keep separate providers and LLM spans (tokens, cost) stay in the right project."""
    if provider is not None:
        _request_tracer_provider.set(provider)


def get_current_project_name() -> Optional[str]:
    """Return the Arize project name for the current request's provider (for validation logging). Returns None if not in a request context."""
    provider = _request_tracer_provider.get()
    if provider is None:
        return None
    for comp, prov in _PROVIDERS_BY_COMPONENT.items():
        if prov is provider:
            return {
                COMPONENT_SINGLE_CALL: PROJECT_SINGLE_CALL,
                COMPONENT_PROSPECT: PROJECT_PROSPECT,
                COMPONENT_HYPOTHESIS: PROJECT_HYPOTHESIS,
                COMPONENT_DEMO: PROJECT_DEMO,
            }.get(comp)
    return None


def get_component_for_path(path: str) -> str:
    """Map request path to component so middleware can set the right Arize project."""
    if path.startswith("/api/hypothesis-research"):
        return COMPONENT_HYPOTHESIS  # Hypothesis Generator
    if path.startswith("/api/analyze-prospect") or path.startswith("/api/prospect-overview"):
        return COMPONENT_PROSPECT  # Prospect Overview
    # Custom Demo Builder
    if (
        path.startswith("/api/classify-demo")
        or path.startswith("/api/generate-demo")
        or path.startswith("/api/export-script")
        or path.startswith("/api/create-online-evals")
    ):
        return COMPONENT_DEMO
    # Single Call Analysis (analyze, generate-recap-slide, calls-by-account, example)
    return COMPONENT_SINGLE_CALL


def api_span(route_name: str, kind: str = "CHAIN", **attributes):
    """
    Context manager for a top-level API span. Use for every route so the full
    app workflow is visible in Arize with consistent naming.

    Example:
        with api_span("analyze", account_name=request.account_name):
            ...
    """
    from contextlib import contextmanager
    tracer = trace.get_tracer("sa-call-analyzer-api")
    name = f"{SPAN_PREFIX}.{route_name}"
    attrs = {
        "openinference.span.kind": kind.upper(),
        **attributes,
    }
    @contextmanager
    def _ctx():
        with tracer.start_as_current_span(name, attributes=attrs) as span:
            try:
                yield span
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                if hasattr(span, "record_exception"):
                    span.record_exception(e)
                raise
    return _ctx()


def force_flush_spans():
    """
    Force flush all pending spans to Arize.
    Useful for ensuring spans are sent immediately (e.g., in tests or before shutdown).
    """
    try:
        tracer_provider = trace.get_tracer_provider()
        if hasattr(tracer_provider, 'force_flush'):
            success = tracer_provider.force_flush(timeout_millis=30000)  # 30 second timeout
            if success:
                print("✓ Spans flushed to Arize successfully")
            else:
                print("⚠️  Warning: Span flush timed out or failed")
        else:
            print(f"⚠️  Warning: TracerProvider {type(tracer_provider).__name__} does not support force_flush")
    except Exception as e:
        print(f"❌ Error flushing spans: {e}")
        import traceback
        traceback.print_exc()
