"""
OpenInference observability integration for Arize AX.

This module sets up tracing for CrewAI and LangChain calls and exports telemetry to Arize AX.
"""
import os
from typing import Optional
from arize.otel import register
from openinference.instrumentation.crewai import CrewAIInstrumentor
from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry import trace


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
        # Register with Arize AX with enhanced resource attributes
        tracer_provider = register(
            space_id=space_id,
            api_key=api_key,
            project_name=project_name,
            # Add custom resource attributes for better filtering
            resource_attributes={
                "service.name": "sa-call-analyzer",
                "service.version": "0.1.0",
                "deployment.environment": os.getenv("ENVIRONMENT", "production"),
            }
        )

        # Instrument CrewAI with enhanced configuration
        CrewAIInstrumentor().instrument(
            tracer_provider=tracer_provider,
            skip_dep_check=True
        )

        # Instrument LangChain with enhanced configuration
        LangChainInstrumentor().instrument(
            tracer_provider=tracer_provider,
            skip_dep_check=True
        )

        print(f"✅ OpenInference tracing enabled")
        print(f"   📊 Sending telemetry to Arize AX")
        print(f"   🔗 View traces at: https://app.arize.com/organizations")

    except Exception as e:
        print(f"⚠️  WARNING: Failed to initialize observability: {e}")
        print("   Application will continue without tracing.")


def get_tracer(name: str = "sa-call-analyzer"):
    """Get a tracer for manual instrumentation."""
    return trace.get_tracer(name)
