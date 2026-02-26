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
from typing import Optional
from arize.otel import register
# NOTE: CrewAI Instrumentor disabled - creates separate root traces for each Crew.kickoff()
# from openinference.instrumentation.crewai import CrewAIInstrumentor
from openinference.instrumentation.langchain import LangChainInstrumentor
from openinference.instrumentation.litellm import LiteLLMInstrumentor
from openinference.instrumentation.openai import OpenAIInstrumentor
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from span_processor_fixed import CleaningSpanProcessor

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
        return

    try:
        # =====================================================================
        # SETUP WITH SPANPROCESSOR (SIMPLER APPROACH)
        # =====================================================================
        # Register with Arize, then add our cleaning SpanProcessor.
        # The processor modifies span._attributes directly in on_end().
        # =====================================================================

        # Check configuration
        enable_cleaning = os.getenv("ARIZE_ENABLE_OUTPUT_CLEANING", "true").lower() == "true"

        # Register with Arize and get tracer provider
        tracer_provider = register(
            space_id=space_id,
            api_key=api_key,
            project_name=project_name,
            set_global_tracer_provider=True,
        )

        # Add cleaning processor to the chain (FIXED VERSION)
        # We insert it at the front of the processor chain so it runs BEFORE export
        if enable_cleaning:
            cleaning_processor = CleaningSpanProcessor(enabled=True)

            # Access the internal processor chain and add our processor
            if hasattr(tracer_provider, '_active_span_processor'):
                multi_processor = tracer_provider._active_span_processor
                if hasattr(multi_processor, '_span_processors'):
                    # Insert at position 0 (runs first, before BatchSpanProcessor exports)
                    existing_processors = multi_processor._span_processors
                    multi_processor._span_processors = (cleaning_processor,) + existing_processors
                    print("   ✅ Cleaning processor added to chain (extracts JSON, truncates large payloads)")
                else:
                    print("   ⚠️  Could not add cleaning processor: no processor chain found")
            else:
                print("   ⚠️  Could not add cleaning processor: no active span processor")
        else:
            print("   ℹ️  Cleaning disabled (set ARIZE_ENABLE_OUTPUT_CLEANING=true to enable)")

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
                tracer_provider=tracer_provider,
                skip_dep_check=True
            )

        # Instrument OpenAI (for direct OpenAI SDK calls)
        openai_instrumentor = OpenAIInstrumentor()
        openai_instrumentor.instrument(
            tracer_provider=tracer_provider,
            skip_dep_check=True
        )
        print("   ✅ OpenAI instrumentor initialized")

        # Instrument LangChain (AFTER processor registered).
        # separate_trace_from_runtime_context=False so LLM/chain spans parent to our
        # API spans (one trace per request, no orphan spans).
        langchain_instrumentor = LangChainInstrumentor()
        langchain_instrumentor.instrument(
            tracer_provider=tracer_provider,
            skip_dep_check=True,
            separate_trace_from_runtime_context=False,
        )

        # Instrument LiteLLM (captures actual LLM calls with token counts)
        litellm_instrumentor = LiteLLMInstrumentor()
        litellm_instrumentor.instrument(
            tracer_provider=tracer_provider,
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

        # DEBUG: Verify instrumentation worked
        print("\n🔍 Instrumentation Verification:")
        if hasattr(langchain_instrumentor, '_tracer') and langchain_instrumentor._tracer:
            print("   ✅ LangChain instrumentor has tracer attached")
        else:
            print("   ❌ LangChain instrumentor tracer is None!")
        
        # Check if BaseCallbackManager is wrapped
        try:
            import langchain_core.callbacks
            is_wrapped = hasattr(langchain_core.callbacks.BaseCallbackManager.__init__, '__wrapped__')
            print(f"   {'✅' if is_wrapped else '❌'} BaseCallbackManager.__init__ wrapped: {is_wrapped}")
        except Exception as e:
            print(f"   ⚠️  Could not verify BaseCallbackManager wrapping: {e}")

        # DEBUG: Check span processor and exporter configuration
        print("\n🔍 Span Export Configuration:")
        if hasattr(tracer_provider, '_active_span_processor'):
            processor = tracer_provider._active_span_processor
            print(f"   Span processor type: {type(processor).__name__}")
            if hasattr(processor, '_span_processors'):
                for i, p in enumerate(processor._span_processors):
                    print(f"   - Processor {i}: {type(p).__name__}")
                    if hasattr(p, 'span_exporter'):
                        print(f"     Exporter: {type(p.span_exporter).__name__}")
                    elif hasattr(p, '_exporter'):
                        print(f"     Exporter (internal): {type(p._exporter).__name__}")
        else:
            print("   ⚠️  No active span processor found")

        print(f"\n✅ OpenInference tracing enabled (LangChain + LiteLLM)")
        print(f"   📊 Sending telemetry to Arize AX (project: {project_name})")
        print(f"   🔗 View traces at: https://app.arize.com/organizations")
        print(f"   🔍 Tracer provider: {type(tracer_provider).__name__}")
        print(f"   🌐 Space ID: {space_id[:20]}...")

        # Verify global tracer provider is set
        global_provider = trace.get_tracer_provider()
        print(f"   ✓ Global tracer provider: {type(global_provider).__name__}")

        return tracer_provider

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
