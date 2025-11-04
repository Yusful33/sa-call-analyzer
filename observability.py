"""
OpenInference observability integration for Arize AX.

This module sets up tracing for CrewAI and LangChain calls and exports telemetry to Arize AX.
"""
import os
import logging
from typing import Optional
from arize.otel import register
from openinference.instrumentation.crewai import CrewAIInstrumentor
from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry import trace

# Enable debug logging for OpenTelemetry
logging.basicConfig(level=logging.INFO)
logging.getLogger("opentelemetry").setLevel(logging.DEBUG)


def setup_observability(
    project_name: str = "sa-call-analyzer",
    arize_api_key: Optional[str] = None,
    arize_space_id: Optional[str] = None
) -> None:
    """
    Set up OpenInference instrumentation and export traces to Arize AX.

    Args:
        project_name: Name of the project for trace organization
        arize_api_key: Arize API key (defaults to ARIZE_API_KEY env var)
        arize_space_id: Arize space ID (defaults to ARIZE_SPACE_ID env var)
    """

    # Get credentials from environment if not provided
    api_key = arize_api_key or os.getenv("ARIZE_API_KEY")
    space_id = arize_space_id or os.getenv("ARIZE_SPACE_ID")

    if not api_key or not space_id:
        print("⚠️  WARNING: Arize credentials not found. Observability disabled.")
        print("   Set ARIZE_API_KEY and ARIZE_SPACE_ID in .env to enable tracing.")
        return

    try:
        # Register with Arize AX and set as global tracer provider
        tracer_provider = register(
            space_id=space_id,
            api_key=api_key,
            project_name=project_name,
            set_global_tracer_provider=True,  # Set as global provider for manual spans
        )

        # Instrument CrewAI
        CrewAIInstrumentor().instrument(
            tracer_provider=tracer_provider,
            skip_dep_check=True
        )

        # Instrument LangChain
        LangChainInstrumentor().instrument(
            tracer_provider=tracer_provider,
            skip_dep_check=True
        )

        print(f"✅ OpenInference tracing enabled")
        print(f"   📊 Sending telemetry to Arize AX (project: {project_name})")
        print(f"   🔗 View traces at: https://app.arize.com/organizations")
        print(f"   🔍 Tracer provider: {type(tracer_provider).__name__}")
        print(f"   🌐 Space ID: {space_id[:20]}...")

        # Verify global tracer provider is set
        from opentelemetry import trace as otel_trace
        global_provider = otel_trace.get_tracer_provider()
        print(f"   ✓ Global tracer provider: {type(global_provider).__name__}")

    except Exception as e:
        print(f"⚠️  WARNING: Failed to initialize observability: {e}")
        print("   Application will continue without tracing.")


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
