"""
OpenInference observability integration for Arize Phoenix.

This module sets up tracing for LangChain/CrewAI calls and exports telemetry to Arize.
"""
import os
from typing import Optional
from opentelemetry import trace as trace_api
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from openinference.instrumentation.langchain import LangChainInstrumentor


def setup_observability(
    project_name: str = "sa-call-analyzer",
    arize_api_key: Optional[str] = None,
    arize_space_id: Optional[str] = None,
    phoenix_endpoint: Optional[str] = None
) -> None:
    """
    Set up OpenInference instrumentation and export traces to Arize Phoenix.

    Args:
        project_name: Name of the project for trace organization
        arize_api_key: Arize API key (defaults to ARIZE_API_KEY env var)
        arize_space_id: Arize space ID (defaults to ARIZE_SPACE_ID env var)
        phoenix_endpoint: Phoenix collector endpoint (defaults to PHOENIX_COLLECTOR_ENDPOINT env var)
    """

    # Get credentials from environment if not provided
    api_key = arize_api_key or os.getenv("ARIZE_API_KEY")
    space_id = arize_space_id or os.getenv("ARIZE_SPACE_ID")
    endpoint = phoenix_endpoint or os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "https://app.phoenix.arize.com")

    if not api_key or not space_id:
        print("⚠️  WARNING: Arize credentials not found. Observability disabled.")
        print("   Set ARIZE_API_KEY and ARIZE_SPACE_ID in .env to enable tracing.")
        return

    try:
        # Configure the OTLP exporter to send to Arize Phoenix
        headers = {
            "api_key": api_key,
            "space_id": space_id,
        }

        exporter = OTLPSpanExporter(
            endpoint=f"{endpoint}/v1/traces",
            headers=headers,
        )

        # Set up the tracer provider
        tracer_provider = trace_sdk.TracerProvider()
        tracer_provider.add_span_processor(BatchSpanProcessor(exporter))
        trace_api.set_tracer_provider(tracer_provider)

        # Instrument LangChain (which CrewAI uses under the hood)
        LangChainInstrumentor().instrument(
            tracer_provider=tracer_provider,
            skip_dep_check=True
        )

        print(f"✅ OpenInference tracing enabled")
        print(f"   📊 Sending telemetry to Arize Phoenix")
        print(f"   🔗 View traces at: https://app.phoenix.arize.com")

    except Exception as e:
        print(f"⚠️  WARNING: Failed to initialize observability: {e}")
        print("   Application will continue without tracing.")


def get_tracer(name: str = "sa-call-analyzer"):
    """Get a tracer for manual instrumentation."""
    return trace_api.get_tracer(name)
