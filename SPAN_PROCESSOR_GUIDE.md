# Span Processor Implementation Guide for SAs

> **Purpose**: This guide teaches Arize Solution Architects how to help customers implement custom span processors for LLM observability.

---

## Table of Contents

1. [Overview](#overview)
2. [When to Recommend Span Processors](#when-to-recommend-span-processors)
3. [Architecture & Data Flow](#architecture--data-flow)
4. [Implementation Walkthrough](#implementation-walkthrough)
5. [Common Customer Use Cases](#common-customer-use-cases)
6. [Testing & Validation](#testing--validation)
7. [Troubleshooting](#troubleshooting)
8. [Advanced Patterns](#advanced-patterns)

---

## Overview

### What is a Span Processor?

A **Span Processor** is an OpenTelemetry SDK component that intercepts telemetry data **before** it's exported to Arize. Think of it as middleware for observability data.

```
Application → Spans Created → Span Processor → Span Exporter → Arize Platform
                                    ↑
                              YOUR CUSTOM LOGIC
```

### Key Benefits for Customers

| Benefit | Business Value |
|---------|---------------|
| **Data Privacy** | Automatically redact PII before it leaves their infrastructure |
| **Cost Optimization** | Truncate large payloads to reduce storage/bandwidth costs |
| **Data Quality** | Clean, normalize, and enrich data for better analysis |
| **Compliance** | Meet regulatory requirements (GDPR, HIPAA, SOC 2) |
| **Flexibility** | Customize telemetry without changing application code |

---

## When to Recommend Span Processors

### Discovery Questions

Ask customers:

1. **"Are you concerned about sensitive data in LLM prompts/outputs?"**
   - → PII Redaction Processor

2. **"Are you seeing very large payloads in your traces?"**
   - → Truncation/Summarization Processor

3. **"Do you need to add business context to your telemetry?"**
   - → Metadata Enrichment Processor

4. **"Are your LLM outputs hard to read in Arize (wrapped in markdown, verbose)?"**
   - → JSON Extraction Processor

5. **"Do you have compliance requirements around data handling?"**
   - → Custom Sanitization Processor

### Red Flags (When to Suggest Span Processors)

- Customer mentions "we can't send customer data to observability platforms"
- They're manually sanitizing data in application code
- Trace costs are surprisingly high
- They need to filter or sample traces
- They're struggling with multi-tenant data isolation

---

## Architecture & Data Flow

### Complete Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                     Customer Application                         │
│                                                                   │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐  │
│  │   LangChain  │      │   CrewAI     │      │  Custom Code │  │
│  │   (auto)     │      │   (auto)     │      │  (manual)    │  │
│  └──────┬───────┘      └──────┬───────┘      └──────┬───────┘  │
│         │                     │                      │           │
│         └─────────────────────┴──────────────────────┘           │
│                               │                                   │
│                               ▼                                   │
│                    ┌────────────────────┐                        │
│                    │  TracerProvider    │                        │
│                    │  (OpenTelemetry)   │                        │
│                    └────────┬───────────┘                        │
└─────────────────────────────┼─────────────────────────────────────┘
                              │
                              ▼
            ┌─────────────────────────────────────┐
            │      Span Processor Pipeline        │
            │                                      │
            │  1. PII Redaction Processor          │
            │     • Removes SSN, credit cards     │
            │     • Redacts email addresses        │
            │     • Masks API keys/secrets         │
            │                                      │
            │  2. Output Cleaning Processor        │
            │     • Extracts JSON from markdown   │
            │     • Truncates large payloads      │
            │     • Normalizes formats             │
            │                                      │
            │  3. Metadata Enrichment Processor    │
            │     • Adds environment tags          │
            │     • Adds app version               │
            │     • Adds tenant/customer ID        │
            │                                      │
            └────────────────┬─────────────────────┘
                             │
                             ▼
            ┌─────────────────────────────────────┐
            │         Span Exporter               │
            │    (Batching & Compression)         │
            └────────────────┬─────────────────────┘
                             │
                             ▼
            ┌─────────────────────────────────────┐
            │         Arize Platform              │
            │    (Trace Storage & Analysis)       │
            └─────────────────────────────────────┘
```

### Critical Timing

**MUST register processors AFTER tracer provider creation, BEFORE instrumentation:**

```python
# ✅ CORRECT ORDER
tracer_provider = register(...)           # 1. Create provider
tracer_provider.add_span_processor(...)   # 2. Add processors
LangChainInstrumentor().instrument(...)   # 3. Instrument libraries

# ❌ WRONG ORDER
tracer_provider = register(...)
LangChainInstrumentor().instrument(...)   # 3. Instrumented FIRST
tracer_provider.add_span_processor(...)   # 2. Processor added LATE (won't catch spans!)
```

---

## Implementation Walkthrough

### Step 1: Create the Processor

```python
# span_processor.py
from opentelemetry.sdk.trace import SpanProcessor, ReadableSpan

class CustomProcessor(SpanProcessor):
    """Your custom span processor."""

    def __init__(self, config):
        self.config = config
        print(f"✅ CustomProcessor initialized with config: {config}")

    def on_start(self, span: ReadableSpan, parent_context=None):
        """Called when span is created (rarely used)."""
        pass

    def on_end(self, span: ReadableSpan):
        """Called when span finishes - MAIN LOGIC HERE."""
        try:
            # Get span attributes (read-only)
            attributes = dict(span.attributes or {})

            # Modify attributes via set_attribute
            output = attributes.get("output.value")
            if output:
                # Do your custom processing
                cleaned = self._process_output(output)
                span.set_attribute("output.value", cleaned)

        except Exception as e:
            # CRITICAL: Never crash the app!
            print(f"Processor error: {e}")

    def _process_output(self, output):
        """Your custom logic."""
        # Example: Remove PII
        # Example: Extract JSON
        # Example: Truncate long text
        return output

    def shutdown(self):
        """Cleanup resources."""
        pass

    def force_flush(self, timeout_millis: int = 30000):
        """Force flush buffers (usually no-op)."""
        pass
```

### Step 2: Register the Processor

```python
# observability.py or setup.py
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from span_processor import CustomProcessor

# Create tracer provider
tracer_provider = TracerProvider()

# Register your processor
processor = CustomProcessor(config={"max_length": 5000})
tracer_provider.add_span_processor(processor)

# Set as global provider
trace.set_tracer_provider(tracer_provider)

# Now instrument libraries
from openinference.instrumentation.langchain import LangChainInstrumentor
LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
```

### Step 3: Configure via Environment Variables

```python
# Make it production-ready with configuration
import os

class CustomProcessor(SpanProcessor):
    def __init__(self):
        # Load from environment
        self.enabled = os.getenv("PROCESSOR_ENABLED", "true").lower() == "true"
        self.max_length = int(os.getenv("MAX_OUTPUT_LENGTH", "10000"))
        self.redact_emails = os.getenv("REDACT_EMAILS", "false").lower() == "true"
```

```bash
# .env file
PROCESSOR_ENABLED=true
MAX_OUTPUT_LENGTH=5000
REDACT_EMAILS=true
```

---

## Common Customer Use Cases

### Use Case 1: Financial Services - PII Redaction

**Scenario**: Bank needs to trace LLM interactions but cannot send account numbers, SSNs, or card numbers outside their infrastructure.

**Solution**:

```python
import re

class FinancialPIIProcessor(SpanProcessor):
    def __init__(self):
        self.patterns = {
            'account': r'\b\d{10,12}\b',  # Account numbers
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'card': r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b',
            'routing': r'\b\d{9}\b',
        }

    def on_end(self, span: ReadableSpan):
        try:
            for field in ['input.value', 'output.value']:
                value = span.attributes.get(field)
                if value:
                    for pii_type, pattern in self.patterns.items():
                        value = re.sub(pattern, f'[REDACTED_{pii_type.upper()}]', value)
                    span.set_attribute(field, value)

            span.set_attribute("pii.redacted", True)
        except Exception as e:
            print(f"PII redaction error: {e}")
```

**Business Value**: Enables observability while maintaining compliance.

---

### Use Case 2: SaaS Startup - Cost Optimization

**Scenario**: Startup is sending massive LLM outputs (50KB+) to Arize, causing high costs.

**Solution**:

```python
class TruncationProcessor(SpanProcessor):
    def __init__(self, max_length=10000):
        self.max_length = max_length

    def on_end(self, span: ReadableSpan):
        try:
            output = span.attributes.get("output.value")
            if output and len(output) > self.max_length:
                # Keep beginning and end
                keep_start = int(self.max_length * 0.7)
                keep_end = self.max_length - keep_start - 50

                truncated = (
                    output[:keep_start] +
                    f"\n...[{len(output) - self.max_length} chars truncated]...\n" +
                    output[-keep_end:]
                )

                span.set_attribute("output.value", truncated)
                span.set_attribute("output.original_length", len(output))
                span.set_attribute("output.truncated", True)

        except Exception as e:
            print(f"Truncation error: {e}")
```

**Business Value**: Reduces costs by 60-80% while preserving critical information.

---

### Use Case 3: Enterprise - Multi-Tenant Isolation

**Scenario**: Enterprise SaaS needs to tag traces by customer/tenant for cost attribution and isolation.

**Solution**:

```python
class TenantEnrichmentProcessor(SpanProcessor):
    def __init__(self, tenant_resolver):
        self.tenant_resolver = tenant_resolver

    def on_end(self, span: ReadableSpan):
        try:
            # Extract tenant from context or span attributes
            tenant_id = self._extract_tenant(span)

            if tenant_id:
                # Add tenant metadata
                span.set_attribute("tenant.id", tenant_id)
                span.set_attribute("tenant.tier", self.tenant_resolver.get_tier(tenant_id))
                span.set_attribute("tenant.region", self.tenant_resolver.get_region(tenant_id))

        except Exception as e:
            print(f"Tenant enrichment error: {e}")

    def _extract_tenant(self, span):
        # Look in span attributes, parent context, or thread-local storage
        return span.attributes.get("user.tenant_id")
```

**Business Value**: Enables per-customer cost tracking and chargeback.

---

### Use Case 4: This Project - JSON Extraction

**Scenario**: LLMs wrap JSON responses in markdown code blocks, making them hard to read in Arize.

**Solution**:

```python
import re
import json

class JSONExtractionProcessor(SpanProcessor):
    def on_end(self, span: ReadableSpan):
        try:
            # Only process LLM spans
            if span.attributes.get("openinference.span.kind") != "llm":
                return

            output = span.attributes.get("output.value")
            if output:
                extracted = self._extract_json(output)
                if extracted != output:
                    span.set_attribute("output.value", extracted)
                    span.set_attribute("output.json_extracted", True)

        except Exception as e:
            print(f"JSON extraction error: {e}")

    def _extract_json(self, text):
        """Extract JSON from markdown or verbose text."""
        # Try markdown code blocks
        pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
        matches = re.findall(pattern, text, re.DOTALL)
        if matches and self._is_valid_json(matches[0]):
            return matches[0]

        # Try standalone JSON
        start = text.find('{')
        end = text.rfind('}')
        if start >= 0 and end > start:
            potential = text[start:end+1]
            if self._is_valid_json(potential):
                return potential

        return text  # Return original if no JSON found

    def _is_valid_json(self, text):
        try:
            json.loads(text)
            return True
        except:
            return False
```

**Business Value**: Improves trace readability and reduces cognitive load for engineers.

---

## Testing & Validation

### Manual Testing

```python
# test_processor.py
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from span_processor import CustomProcessor

# Setup
provider = TracerProvider()
processor = CustomProcessor()
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)

# Create test span
tracer = trace.get_tracer(__name__)
with tracer.start_as_current_span("test_span") as span:
    span.set_attribute("output.value", "sensitive data: 123-45-6789")
    # When span ends, processor should modify it

# Check Arize UI to verify:
# 1. Span appears
# 2. Attribute was modified by processor
# 3. Processor metadata attributes are present
```

### Validation Checklist for Customers

- [ ] Processor doesn't crash on malformed data
- [ ] Processor handles missing attributes gracefully
- [ ] Performance impact < 10ms per span
- [ ] Original data is preserved if processing fails
- [ ] Processor metadata attributes are visible in Arize
- [ ] Configuration changes work without code deploy

---

## Troubleshooting

### Issue 1: Processor Not Running

**Symptom**: Spans in Arize show unmodified data

**Root Causes**:
1. Processor registered AFTER instrumentation
2. Processor not added to tracer provider
3. Exception in processor (check logs)

**Fix**:
```python
# Check registration order
tracer_provider = register(...)        # 1. Provider first
tracer_provider.add_span_processor(processor)  # 2. Processor second
LangChainInstrumentor().instrument(...)  # 3. Instrumentation last
```

---

### Issue 2: Application Crashes

**Symptom**: App crashes when processor is enabled

**Root Cause**: Unhandled exception in processor

**Fix**: ALWAYS wrap in try/except
```python
def on_end(self, span: ReadableSpan):
    try:
        # Your logic
        pass
    except Exception as e:
        print(f"⚠️  Processor error: {e}")
        # Don't re-raise!
```

---

### Issue 3: Processor Too Slow

**Symptom**: Application latency increases

**Root Cause**: Expensive operations in processor (API calls, regex, JSON parsing)

**Fix**:
```python
# Keep it fast
def on_end(self, span: ReadableSpan):
    start = time.time()
    try:
        # Your logic
        pass
    finally:
        duration = time.time() - start
        if duration > 0.010:  # 10ms threshold
            print(f"⚠️  Slow processor: {duration*1000:.1f}ms")
```

---

## Advanced Patterns

### Pattern 1: Chaining Multiple Processors

```python
# Order matters! Security first, optimization last
tracer_provider.add_span_processor(PIIRedactionProcessor())
tracer_provider.add_span_processor(TruncationProcessor())
tracer_provider.add_span_processor(MetadataProcessor())
```

### Pattern 2: Conditional Processing

```python
class ConditionalProcessor(SpanProcessor):
    def on_end(self, span: ReadableSpan):
        # Only process production spans
        env = os.getenv("ENVIRONMENT")
        if env != "production":
            return

        # Only process expensive LLM calls
        span_kind = span.attributes.get("openinference.span.kind")
        if span_kind == "llm":
            self._process_llm_span(span)
```

### Pattern 3: Sampling

```python
import random

class SamplingProcessor(SpanProcessor):
    def __init__(self, sample_rate=0.1):  # 10%
        self.sample_rate = sample_rate

    def on_end(self, span: ReadableSpan):
        # Drop 90% of spans
        if random.random() > self.sample_rate:
            # Mark for dropping (requires custom exporter)
            span.set_attribute("sampled", False)
```

---

## Key Takeaways for SAs

### When Presenting to Customers

1. **Start with the problem**: "What data concerns do you have?"
2. **Show the architecture**: Visual diagrams help
3. **Demo live**: Show before/after in Arize UI
4. **Emphasize safety**: "Processor errors won't crash your app"
5. **Provide templates**: Give them working code to start

### Discovery Call Playbook

1. Ask about data sensitivity
2. Ask about trace costs
3. Ask about compliance requirements
4. Ask about multi-tenancy
5. → Recommend span processors as solution

### Success Metrics

- Reduced trace storage costs (measure GB/month)
- Faster time to compliance (days to implement vs months)
- Improved data quality scores (fewer null fields, cleaner JSON)

---

## Additional Resources

- OpenTelemetry Span Processor Docs: https://opentelemetry.io/docs/instrumentation/python/sdk/#span-processor
- Arize OpenInference Semantic Conventions: https://github.com/Arize-ai/openinference
- Example implementations: See `span_processor.py` in this repo

---

## Support & Questions

For SA-specific questions, ping the #solutions-engineering Slack channel with:
- Customer name
- Use case
- What you've tried
- Specific blocker
