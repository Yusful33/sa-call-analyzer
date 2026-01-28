"""
Enhanced tracing utilities for better observability.

This module provides:
- Decorators for automatic span creation
- Prompt template tracking
- Cost tracking
- Performance metrics
- User/session tracking
- Span linking utilities
"""
import time
import json
from functools import wraps
from typing import Any, Callable, Dict, Optional, List
from contextlib import contextmanager

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode, Link
from openinference.instrumentation.openinference import using_prompt_template, using_attributes

# Cost per 1M tokens (approximate, update as needed)
MODEL_COSTS = {
    "claude-3-5-haiku-20241022": {"input": 0.25, "output": 1.25},  # $0.25/$1.25 per 1M tokens
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
    "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
}


def calculate_cost(model_name: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Calculate cost based on token usage."""
    costs = MODEL_COSTS.get(model_name, {"input": 0.0, "output": 0.0})
    input_cost = (prompt_tokens / 1_000_000) * costs["input"]
    output_cost = (completion_tokens / 1_000_000) * costs["output"]
    return input_cost + output_cost


@contextmanager
def trace_with_prompt_template(
    template: str,
    version: str = "1.0",
    variables: Optional[Dict[str, Any]] = None,
    **kwargs
):
    """
    Context manager that adds prompt template tracking to all spans created within it.
    
    Usage:
        with trace_with_prompt_template(
            template="Analyze this call: {transcript}",
            version="2.0",
            variables={"transcript": transcript}
        ):
            # All LLM calls within this block will have prompt template attributes
            result = llm.invoke(...)
    """
    with using_prompt_template(
        template=template,
        version=version,
        variables=variables or {}
    ):
        yield


@contextmanager
def trace_with_metadata(
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    **kwargs
):
    """
    Context manager that adds user/session tracking and metadata to all spans.
    
    Usage:
        with trace_with_metadata(
            user_id="user123",
            session_id="session456",
            tags=["production", "v2"],
            metadata={"call_id": "call789"}
        ):
            # All spans within this block will have these attributes
            result = analyze_call(...)
    """
    with using_attributes(
        user_id=user_id or "",
        session_id=session_id or "",
        tags=tags or [],
        metadata=metadata or {}
    ):
        yield


def trace_function(
    span_name: Optional[str] = None,
    span_kind: str = "chain",
    capture_args: bool = False,
    capture_result: bool = False,
    add_cost_tracking: bool = False,
    model_name_attr: Optional[str] = None,
):
    """
    Decorator to automatically create spans for function execution.
    
    Args:
        span_name: Name of the span (defaults to function name)
        span_kind: OpenInference span kind (chain, agent, llm, etc.)
        capture_args: Whether to capture function arguments as input.value
        capture_result: Whether to capture return value as output.value
        add_cost_tracking: Whether to track costs (requires model_name_attr)
        model_name_attr: Attribute name in kwargs that contains model name
    
    Usage:
        @trace_function(
            span_name="analyze_transcript",
            span_kind="agent",
            capture_args=True,
            capture_result=True
        )
        def analyze_transcript(transcript: str, model: str):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            tracer = trace.get_tracer(func.__module__ or "sa-call-analyzer")
            name = span_name or f"{func.__name__}"
            
            # Prepare attributes
            attributes = {
                "openinference.span.kind": span_kind,
            }
            
            # Capture input arguments if requested
            if capture_args:
                try:
                    # Get function signature to map args
                    import inspect
                    sig = inspect.signature(func)
                    bound_args = sig.bind(*args, **kwargs)
                    bound_args.apply_defaults()
                    
                    # Create input dict
                    input_dict = {}
                    for param_name, param_value in bound_args.arguments.items():
                        # Truncate long strings
                        if isinstance(param_value, str) and len(param_value) > 1000:
                            input_dict[param_name] = param_value[:1000] + "..."
                        else:
                            input_dict[param_name] = param_value
                    
                    attributes["input.value"] = json.dumps(input_dict, default=str)
                    attributes["input.mime_type"] = "application/json"
                except Exception as e:
                    attributes["input.value"] = f"Error capturing args: {e}"
            
            # Get model name for cost tracking
            model_name = None
            if add_cost_tracking and model_name_attr:
                model_name = kwargs.get(model_name_attr) or kwargs.get("model")
            
            # Create span
            with tracer.start_as_current_span(name, attributes=attributes) as span:
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    
                    # Capture result if requested
                    if capture_result:
                        try:
                            if isinstance(result, str):
                                output_value = result[:5000] + "..." if len(result) > 5000 else result
                            else:
                                output_value = json.dumps(result, default=str)
                                if len(output_value) > 5000:
                                    output_value = output_value[:5000] + "..."
                            span.set_attribute("output.value", output_value)
                            span.set_attribute("output.mime_type", "application/json" if not isinstance(result, str) else "text/plain")
                        except Exception as e:
                            span.set_attribute("output.value", f"Error capturing result: {e}")
                    
                    # Add performance metrics
                    duration = time.time() - start_time
                    span.set_attribute("performance.duration_seconds", duration)
                    span.set_attribute("performance.latency_ms", duration * 1000)
                    
                    # Add cost tracking if enabled and we have token info
                    if add_cost_tracking and model_name:
                        # Try to get token counts from result or span context
                        # This would need to be enhanced based on your token tracking callback
                        span.set_attribute("cost.model_name", model_name)
                        span.add_event("cost_tracking_available", {
                            "model": model_name,
                            "note": "Token counts should be available from TokenTrackingCallback"
                        })
                    
                    span.set_status(Status(StatusCode.OK))
                    return result
                    
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise
                    
        return wrapper
    return decorator


def add_cost_attributes_to_span(
    span: trace.Span,
    model_name: str,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int
):
    """
    Add cost tracking attributes to a span using OpenInference semantic conventions.
    
    Uses llm.cost.* attributes which Arize recognizes for cost calculation.
    
    Usage:
        span = trace.get_current_span()
        add_cost_attributes_to_span(span, "claude-3-5-haiku-20241022", 1000, 500, 1500)
    """
    costs = MODEL_COSTS.get(model_name, {"input": 0.0, "output": 0.0})
    prompt_cost = (prompt_tokens / 1_000_000) * costs["input"]
    completion_cost = (completion_tokens / 1_000_000) * costs["output"]
    total_cost = prompt_cost + completion_cost
    
    # Use OpenInference semantic conventions for cost attributes
    # These are what Arize uses to calculate "Total Cost"
    span.set_attribute("llm.cost.prompt", prompt_cost)
    span.set_attribute("llm.cost.completion", completion_cost)
    span.set_attribute("llm.cost.total", total_cost)
    
    # Also set legacy cost.* attributes for backwards compatibility
    span.set_attribute("cost.model_name", model_name)
    span.set_attribute("cost.prompt_tokens", prompt_tokens)
    span.set_attribute("cost.completion_tokens", completion_tokens)
    span.set_attribute("cost.total_tokens", total_tokens)
    span.set_attribute("cost.total_usd", total_cost)
    span.set_attribute("cost.input_cost_usd", prompt_cost)
    span.set_attribute("cost.output_cost_usd", completion_cost)


def create_span_link(span_context: trace.SpanContext, attributes: Optional[Dict[str, Any]] = None) -> Link:
    """
    Create a span link to connect related operations.
    
    Usage:
        parent_span = trace.get_current_span()
        link = create_span_link(parent_span.get_span_context(), {"link.type": "related_analysis"})
        
        with tracer.start_as_current_span("child_operation", links=[link]):
            ...
    """
    return Link(span_context, attributes or {})


@contextmanager
def trace_agent_execution(
    agent_name: str,
    agent_role: str,
    task_description: str,
    input_data: Optional[Any] = None,
    model_name: Optional[str] = None,
):
    """
    Context manager for tracing individual agent executions with full metadata.
    
    Usage:
        with trace_agent_execution(
            agent_name="technical_evaluator",
            agent_role="Senior Technical Architect",
            task_description="Evaluate technical depth",
            input_data=transcript,
            model_name="claude-3-5-haiku-20241022"
        ) as span:
            result = agent.execute(task)
            span.set_attribute("output.value", json.dumps(result))
    """
    tracer = trace.get_tracer("sa-call-analyzer")
    
    attributes = {
        "openinference.span.kind": "agent",
        "agent.name": agent_name,
        "agent.role": agent_role,
        "agent.task_description": task_description,
    }
    
    if model_name:
        attributes["llm.model_name"] = model_name
        attributes["cost.model_name"] = model_name
    
    if input_data:
        if isinstance(input_data, str):
            attributes["input.value"] = input_data[:2000] + "..." if len(input_data) > 2000 else input_data
            attributes["input.mime_type"] = "text/plain"
        else:
            attributes["input.value"] = json.dumps(input_data, default=str)
            attributes["input.mime_type"] = "application/json"
    
    with tracer.start_as_current_span(f"agent.{agent_name}", attributes=attributes) as span:
        start_time = time.time()
        try:
            yield span
            duration = time.time() - start_time
            span.set_attribute("performance.duration_seconds", duration)
            span.set_status(Status(StatusCode.OK))
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise


def enhance_token_callback_with_cost_tracking(callback_instance):
    """
    Enhance an existing TokenTrackingCallback to also track costs.
    
    Usage:
        token_callback = TokenTrackingCallback()
        enhance_token_callback_with_cost_tracking(token_callback)
    """
    original_on_llm_end = callback_instance.on_llm_end
    
    def enhanced_on_llm_end(self, response, **kwargs):
        # Call original callback
        original_on_llm_end(response, **kwargs)
        
        # Get current span and add cost tracking
        current_span = trace.get_current_span()
        if current_span and hasattr(current_span, 'is_recording') and current_span.is_recording:
            # Try to get token counts from response
            if hasattr(response, 'llm_output') and response.llm_output:
                token_usage = response.llm_output.get('token_usage', {})
                prompt_tokens = token_usage.get('prompt_tokens', 0)
                completion_tokens = token_usage.get('completion_tokens', 0)
                total_tokens = token_usage.get('total_tokens', 0)
                
                # Get model name from span or response
                model_name = None
                if hasattr(response, 'llm_output'):
                    model_name = response.llm_output.get('model_name')
                
                if not model_name:
                    # Try to get from span attributes
                    for attr_key in ['llm.model_name', 'cost.model_name', 'model_name']:
                        if hasattr(current_span, 'attributes'):
                            # This is a simplified check - actual implementation would need proper attribute access
                            pass
                
                if model_name and total_tokens > 0:
                    add_cost_attributes_to_span(
                        current_span,
                        model_name,
                        prompt_tokens,
                        completion_tokens,
                        total_tokens
                    )
    
    callback_instance.on_llm_end = enhanced_on_llm_end.__get__(callback_instance, type(callback_instance))
    return callback_instance

