# Span Processor: What Gets Cleaned

## Overview

The `WorkingLLMCleanerProcessor` now processes **3 types of spans**:
1. **LLM spans** - Individual AI model calls (extract JSON, extract "raw" key)
2. **Agent spans** - CrewAI agent workflows (summarize input/output)
3. **Chain spans** - Data processing pipelines (summarize input/output)

---

## For `crew_analysis_execution` Span

**Span Kind**: `agent`

### Input Cleaning

**Before** (verbose, full transcript preview):
```json
{
  "sa_name": "Unknown",
  "transcript_stats": {
    "total_length": 45678,
    "line_count": 234,
    "speaker_count": 3
  },
  "transcript_preview": "0:16 | Hakan\nyeah, they're so wealthy...\n[200 more lines]...\n\n... (224 more lines)"
}
```

**After** (clean, truncated preview):
```json
{
  "sa_name": "Unknown",
  "transcript_stats": {
    "total_length": 45678,
    "line_count": 234,
    "speaker_count": 3
  },
  "transcript_preview": "0:16 | Hakan\nyeah, they're so wealthy...\n[first 500 chars only]\n... [truncated]"
}
```

### Output Cleaning

**Before** (full detailed output):
```json
{
  "sa_identified": "Hakan",
  "call_summary": "Analysis of SA performance...",
  "overall_score": 7.5,
  "command_scores": { ... },
  "sa_metrics": { ... },
  "top_insights": [
    { huge object with 10+ fields },
    { huge object with 10+ fields },
    ...
  ],
  "strengths": ["...", "...", "..."],
  "improvement_areas": ["...", "...", "..."],
  "key_moments": [ ... ]
}
```

**After** (summary counts only):
```json
{
  "summary": {
    "call_summary": "Analysis of SA performance...",
    "sa_identified": "Hakan",
    "sa_confidence": "high",
    "insight_count": 5,
    "strength_count": 3,
    "improvement_count": 4
  },
  "top_insights_count": 5,
  "strengths_count": 3,
  "improvement_count": 4,
  "key_moments_count": 6
}
```

---

## For LLM Spans (e.g., `Senior Technical Architect & Evaluator`)

**Span Kind**: `llm`

### Output Cleaning

**Before** (verbose with task description):
```
"Analyze the technical performance of Unknown SA in this call.\n\n
Transcript:\n[FULL 5000 LINE TRANSCRIPT]...\n\n
ALWAYS provide a specific time reference in [MM:SS] format.\n\n",
"expected_output": "Technical evaluation...",
"summary": "Analyze the technical...",
"raw": {
  "technical_depth_score": 8.5,
  "integration_discussion_score": 9,
  ...
},
"pydantic": null,
"tasks_output": [...]
```

**After** (just the evaluation data):
```json
{
  "technical_depth_score": 8.5,
  "integration_discussion_score": 9,
  "demo_quality_score": 9.2,
  "detailed_evaluation": {
    "technical_depth": {
      "strengths": [...]
    }
  }
}
```

---

## Configuration

All cleaning is controlled via environment variables:

```bash
# Enable/disable cleaning
ARIZE_ENABLE_OUTPUT_CLEANING=true  # Set to false to disable

# Truncation limits
ARIZE_MAX_OUTPUT_LENGTH=10000  # Max chars for outputs
ARIZE_MAX_INPUT_LENGTH=5000    # Max chars for inputs

# Metadata
ENVIRONMENT=development
APP_VERSION=1.0.0
```

---

## What Gets Added

All processed spans get these new attributes:

```
deployment.environment = "development"
app.version = "1.0.0"
arize.processor.applied = "WorkingLLMCleanerProcessor"
```

### LLM Spans Also Get:
```
arize.processor.output_cleaned = true
arize.processor.original_length = 25000
arize.processor.truncated = true  # If truncation happened
```

### Agent/Chain Spans Also Get:
```
arize.processor.output_summarized = true  # If summarized
```

---

## How to Test

### 1. Restart Server
```bash
python main.py
```

**Expected output**:
```
✅ Arize credentials loaded
✅ WorkingLLMCleanerProcessor initialized:
   - Max output length: 10000
   - Max input length: 5000
   - JSON extraction: True
   - Raw key extraction: True
   ✅ LLM cleaning processor registered
✅ OpenInference tracing enabled
```

### 2. Run Analysis
- Open http://localhost:8000
- Paste a Gong URL or transcript
- Click "Analyze Call"

### 3. Check Arize UI

**Click on `crew_analysis_execution` span**:

#### Input/Output Tab → Input:
- ✅ Transcript preview truncated to 500 chars
- ✅ Stats preserved
- ✅ Clean JSON formatting

#### Input/Output Tab → Output:
- ✅ Summary counts instead of full data
- ✅ Much shorter and easier to read
- ✅ All key metrics visible at a glance

**Click on `Senior Technical Architect & Evaluator` span**:

#### Input/Output Tab → Output:
- ✅ No verbose task description
- ✅ No full transcript
- ✅ Just the clean evaluation JSON
- ✅ "raw" key extracted (no wrapper)

---

## Benefits

### For You as an SA
- ✅ **Faster debugging** - Less scrolling through verbose data
- ✅ **Cost savings** - Smaller payloads = lower Arize costs
- ✅ **Better demos** - Clean data looks professional
- ✅ **Easier analysis** - Summary data easier to understand

### For Your Customers
- ✅ **Pattern for cleaning LLM data** - Show them this implementation
- ✅ **Production-ready** - Error handling, configurability
- ✅ **Customizable** - Easy to add their own cleaning logic

---

## Troubleshooting

### Issue: Spans still verbose

**Check**:
```bash
# Make sure cleaning is enabled
echo $ARIZE_ENABLE_OUTPUT_CLEANING
# Should output: true (or be unset, defaults to true)

# Check server startup logs
# Should see: ✅ LLM cleaning processor registered
```

### Issue: Processor errors

**Look for**:
```bash
⚠️  Processor error on span ...
```

If you see errors, the processor gracefully skips that span and continues.

---

## Advanced: Customize Cleaning Logic

Want different cleaning for specific spans? Update `_process_agent_or_chain_span`:

```python
def _process_agent_or_chain_span(self, attributes: dict, span_name: str):
    # Custom logic for specific spans
    if span_name == "crew_analysis_execution":
        # Your custom cleaning for crew spans
        pass
    elif span_name == "parse_transcript":
        # Your custom cleaning for parse spans
        pass
```

---

## Summary

✅ **LLM spans**: Extract JSON, extract "raw" key, truncate
✅ **Agent/Chain spans**: Summarize input/output, truncate previews
✅ **All spans**: Add metadata (environment, version, processor info)
✅ **Configurable**: Environment variables control behavior
✅ **Safe**: Never crashes the application

**Test it now and see the difference in Arize!** 🎯
