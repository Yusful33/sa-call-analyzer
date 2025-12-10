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
from opentelemetry import trace
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

    # Get credentials from environment if not provided
    # NOTE: load_dotenv() should be called before this in main.py
    # This ensures .env values are loaded into os.environ
    api_key = arize_api_key or os.getenv("ARIZE_API_KEY")
    space_id = arize_space_id or os.getenv("ARIZE_SPACE_ID")

    # Verify credentials are present
    if api_key and space_id:
        print(f"✅ Arize credentials loaded (API Key: {len(api_key)} chars, Space ID: {len(space_id)} chars)")

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
        # CrewAIInstrumentor().instrument(
        #     tracer_provider=tracer_provider,
        #     skip_dep_check=True
        # )

        # Instrument LangChain (AFTER processor registered)
        LangChainInstrumentor().instrument(
            tracer_provider=tracer_provider,
            skip_dep_check=True
        )

        # Instrument LiteLLM (captures actual LLM calls with token counts)
        LiteLLMInstrumentor().instrument(
            tracer_provider=tracer_provider,
            skip_dep_check=True
        )

        print(f"✅ OpenInference tracing enabled (LangChain + LiteLLM)")
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
