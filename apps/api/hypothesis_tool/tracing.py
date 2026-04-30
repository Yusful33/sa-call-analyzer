"""Arize observability setup for LangGraph tracing.

This module configures OpenTelemetry tracing to send LangGraph
spans to Arize for observability and debugging.
"""

import os
from typing import Any

from .config import get_settings


def setup_arize_tracing() -> Any | None:
    """
    Initialize Arize tracing for LangGraph.
    
    Returns the tracer_provider if successful, None otherwise.
    """
    settings = get_settings()
    
    # Check if Arize credentials are configured
    if not settings.arize_space_id or not settings.arize_api_key:
        print("Arize tracing not configured - ARIZE_SPACE_ID and ARIZE_API_KEY required")
        return None
    
    try:
        from arize.otel import register
        from openinference.instrumentation.langchain import LangChainInstrumentor
        
        # Set environment variables (arize.otel may need them)
        os.environ["ARIZE_SPACE_ID"] = settings.arize_space_id
        os.environ["ARIZE_API_KEY"] = settings.arize_api_key
        
        # Register with Arize
        tracer_provider = register(
            space_id=settings.arize_space_id,
            api_key=settings.arize_api_key,
            project_name=settings.arize_project_name,
        )
        
        # Instrument LangChain (which includes LangGraph). Use runtime context so
        # LLM spans parent to our hypothesis_agent.* node spans (one trace per request).
        LangChainInstrumentor().instrument(
            tracer_provider=tracer_provider,
            separate_trace_from_runtime_context=False,
        )
        
        print(f"Arize tracing enabled - project: {settings.arize_project_name}")
        print(f"View traces at: https://app.arize.com/")
        
        return tracer_provider
        
    except ImportError as e:
        print(f"Arize tracing packages not installed: {e}")
        print("Install with: pip install arize-otel openinference-instrumentation-langchain")
        return None
    except Exception as e:
        print(f"Failed to initialize Arize tracing: {e}")
        return None


def get_trace_url(space_id: str) -> str:
    """Get the Arize dashboard URL for viewing traces."""
    return f"https://app.arize.com/organizations/default/spaces/{space_id}/projects"
