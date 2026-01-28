# Tracing Improvements Guide

This document describes the enhanced tracing capabilities added to improve observability in Arize.

## What's New

### 1. **Cost Tracking** ✅
- Automatic cost calculation based on token usage and model pricing
- Tracks input/output costs separately
- Total cost per LLM call
- Integrated into `TokenTrackingCallback`

**Example:**
```python
# Automatically added by TokenTrackingCallback
span.set_attribute("cost.model_name", "claude-3-5-haiku-20241022")
span.set_attribute("cost.prompt_tokens", 1000)
span.set_attribute("cost.completion_tokens", 500)
span.set_attribute("cost.total_tokens", 1500)
span.set_attribute("cost.total_usd", 0.000625)  # Calculated automatically
span.set_attribute("cost.input_cost_usd", 0.00025)
span.set_attribute("cost.output_cost_usd", 0.000375)
```

### 2. **Prompt Template Tracking** ✅
Track which prompt templates are used and their versions.

**Usage:**
```python
from tracing_enhancements import trace_with_prompt_template

with trace_with_prompt_template(
    template="Analyze this call: {transcript}",
    version="2.0",
    variables={"transcript": transcript}
):
    # All LLM calls within this block will have prompt template attributes
    result = agent.execute(task)
```

**Attributes Added:**
- `llm.prompt_template.template` - The prompt template
- `llm.prompt_template.version` - Template version
- `llm.prompt_template.variables` - Template variables as JSON

### 3. **User/Session Tracking** ✅
Track user sessions and metadata across spans.

**Usage:**
```python
from tracing_enhancements import trace_with_metadata

with trace_with_metadata(
    user_id="user123",
    session_id="session456",
    tags=["production", "v2"],
    metadata={"call_id": "call789", "customer": "Acme Corp"}
):
    # All spans within this block will have these attributes
    result = analyze_call(...)
```

**Attributes Added:**
- `session.id` - Session identifier
- `user.id` - User identifier
- `metadata.*` - Custom metadata
- `tag.tags` - List of tags

### 4. **Function Decorators** ✅
Automatically create spans for functions with minimal code changes.

**Usage:**
```python
from tracing_enhancements import trace_function

@trace_function(
    span_name="analyze_transcript",
    span_kind="agent",
    capture_args=True,
    capture_result=True,
    add_cost_tracking=True,
    model_name_attr="model"
)
def analyze_transcript(transcript: str, model: str):
    # Function automatically wrapped in a span
    # Args captured as input.value
    # Return value captured as output.value
    # Performance metrics added automatically
    ...
```

### 5. **Agent Execution Tracking** ✅
Detailed spans for individual agent executions.

**Usage:**
```python
from tracing_enhancements import trace_agent_execution

with trace_agent_execution(
    agent_name="technical_evaluator",
    agent_role="Senior Technical Architect",
    task_description="Evaluate technical depth",
    input_data=transcript,
    model_name="claude-3-5-haiku-20241022"
) as span:
    result = agent.execute(task)
    span.set_attribute("output.value", json.dumps(result))
```

### 6. **Performance Metrics** ✅
Automatic latency and duration tracking.

**Attributes Added:**
- `performance.duration_seconds` - Execution time in seconds
- `performance.latency_ms` - Execution time in milliseconds

### 7. **Span Linking** ✅
Connect related operations with span links.

**Usage:**
```python
from tracing_enhancements import create_span_link

parent_span = trace.get_current_span()
link = create_span_link(
    parent_span.get_span_context(),
    {"link.type": "related_analysis"}
)

with tracer.start_as_current_span("child_operation", links=[link]):
    # This span is linked to the parent
    ...
```

## Current Implementation Status

### ✅ Already Integrated
1. **Cost Tracking** - Enhanced `TokenTrackingCallback` in `crew_analyzer.py`
2. **User/Session Tracking** - Added to `main.py` analyze endpoint
3. **Performance Metrics** - Available via decorators

### 🔄 Ready to Use (Optional)
1. **Prompt Template Tracking** - Available but not yet integrated into all prompts
2. **Function Decorators** - Available for new functions
3. **Agent Execution Tracking** - Available for granular agent tracing

## How to Use

### For New Code

1. **Use decorators for functions:**
```python
from tracing_enhancements import trace_function

@trace_function(span_kind="chain", capture_args=True, capture_result=True)
def my_function(arg1, arg2):
    return result
```

2. **Use context managers for blocks:**
```python
from tracing_enhancements import trace_with_prompt_template, trace_with_metadata

with trace_with_metadata(user_id="user123", session_id="session456"):
    with trace_with_prompt_template(template="...", version="1.0"):
        result = llm.invoke(...)
```

3. **Track individual agents:**
```python
from tracing_enhancements import trace_agent_execution

with trace_agent_execution(
    agent_name="my_agent",
    agent_role="Role",
    task_description="Task",
    input_data=data
) as span:
    result = agent.execute(task)
```

### For Existing Code

The enhancements are backward compatible. Existing code continues to work, and you can gradually add:
- Prompt template tracking to key prompts
- User/session tracking to API endpoints
- Function decorators to new functions

## Model Cost Configuration

Costs are configured in `tracing_enhancements.py` in the `MODEL_COSTS` dictionary. Update as needed:

```python
MODEL_COSTS = {
    "claude-3-5-haiku-20241022": {"input": 0.25, "output": 1.25},
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
    # Add more models as needed
}
```

## Viewing in Arize

After deploying, you'll see:

1. **Cost Metrics:**
   - Filter by `cost.total_usd` to see spend per trace
   - Group by `cost.model_name` to see costs by model
   - Track `cost.prompt_tokens` and `cost.completion_tokens`

2. **Prompt Templates:**
   - Filter by `llm.prompt_template.version` to see template usage
   - View `llm.prompt_template.variables` to see what data was used

3. **User Sessions:**
   - Filter by `user.id` or `session.id` to track user journeys
   - View `metadata.*` attributes for custom tracking

4. **Performance:**
   - Filter by `performance.duration_seconds` to find slow operations
   - Compare latency across different agents/tasks

## Next Steps

1. **Add prompt template tracking to all major prompts** - Update classification, evaluation, and compilation prompts
2. **Add user/session tracking to all API endpoints** - Extract from request headers or auth tokens
3. **Use decorators for new utility functions** - Automatically get tracing for free
4. **Add span links** - Connect related operations (e.g., classification → analysis)

## Troubleshooting

### Cost tracking not showing
- Ensure `TokenTrackingCallback` is properly configured
- Check that model name is being captured correctly
- Verify model is in `MODEL_COSTS` dictionary

### Prompt templates not appearing
- Ensure `trace_with_prompt_template` context manager wraps LLM calls
- Check that OpenInference instrumentation is active
- Verify template attributes are set before LLM invocation

### User/session tracking missing
- Ensure `trace_with_metadata` wraps the operation
- Check that user_id/session_id are being passed correctly
- Verify context is propagated to child spans


