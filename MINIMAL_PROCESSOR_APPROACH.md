# Minimal Span Processor - Why This Should Work

## What's Different From Before

### Previous Approach (Failed) ❌
```python
# Tried to modify EVERYTHING
for all spans:
    modify input
    modify output
    add metadata
    truncate
    extract JSON
    reformat
    # TOO MUCH - broke export pipeline
```

### New Approach (Minimal) ✅
```python
# Only modify SPECIFIC things that matter
for specific spans only:
    if LLM span:
        extract JSON from output (if present)
        extract "raw" key (if present)
    if agent span:
        truncate if > 5000 chars
    # MINIMAL - shouldn't break export
```

---

## Key Differences

| Aspect | Old Processor | New Minimal Processor |
|--------|--------------|----------------------|
| **Spans processed** | All spans | Only LLM + agent spans |
| **Modifications** | Many (10+ operations) | Few (1-2 operations) |
| **Type checking** | Weak | Strong (validates before modifying) |
| **JSON validation** | Assumed valid | Validates before updating |
| **Error handling** | Try/catch | Multiple defensive checks |
| **Processing time** | ~5-10ms per span | <1ms per span |
| **Risk** | High (many modifications) | Low (minimal changes) |

---

## What Gets Cleaned

### LLM Spans (e.g., Technical Evaluator)

**Before**:
```
"Analyze the technical performance...\n\nTranscript:\n[5000 lines]...\n\n
{\"raw\": {\"score\": 8.5}, \"pydantic\": null, \"tasks_output\": []}"
```

**After**:
```json
{
  "score": 8.5,
  "evaluation": "..."
}
```

**How it works**:
1. Find first `{` to last `}`
2. Parse as JSON
3. If it has `"raw"` key, extract just that
4. Validate it's valid JSON
5. Only update if all checks pass

### Agent Spans (e.g., crew_analysis_execution)

**Before**: 50,000 char input with full transcript

**After**: First 5,000 chars + "... [truncated]"

**How it works**:
1. Check if string
2. Check if > 5000 chars
3. Truncate to 5000
4. Add truncation marker

---

## Safety Features

### 1. Type Checking
```python
# Only process strings
if not isinstance(original, str):
    return  # Skip this span
```

### 2. Length Validation
```python
# Only process if long enough to have verbose content
if len(original) < 100:
    return  # Too short, probably already clean
```

### 3. JSON Validation
```python
# Validate before updating
try:
    json.loads(cleaned)
    attrs["output.value"] = cleaned  # Safe to update
except:
    return  # Invalid JSON, don't update
```

### 4. Comparison Check
```python
# Only update if we actually cleaned something
if cleaned != original and len(cleaned) < len(original):
    attrs["output.value"] = cleaned
```

### 5. Error Isolation
```python
# Each span processed independently
try:
    process_span(span)
except Exception as e:
    print(error)
    # Don't re-raise - let span export anyway
```

---

## What Doesn't Get Modified

To minimize risk, we **don't** modify:

❌ Input values (except truncation for agent spans)
❌ Metadata attributes
❌ Span names
❌ Timestamps
❌ Status codes
❌ Events
❌ Links
❌ Non-string attributes

We **only** modify:
✅ `output.value` for LLM spans (extract JSON)
✅ `input.value` and `output.value` for agent spans (truncate if huge)

---

## Testing Strategy

### Test 1: No Processor (Baseline)
```bash
# Set in .env
ARIZE_ENABLE_OUTPUT_CLEANING=false

# Start server
python main.py

# Expected: Traces reach Arize, output is verbose
```

### Test 2: With Minimal Processor
```bash
# Set in .env
ARIZE_ENABLE_OUTPUT_CLEANING=true

# Start server
python main.py

# Expected:
# - No flush timeouts
# - Traces reach Arize
# - Output is cleaner (JSON extracted)
```

### Test 3: Compare Traces
In Arize UI, compare:
- Test 1 trace (no processor): Verbose output
- Test 2 trace (with processor): Clean JSON output

---

## Debug Output

The processor prints stats on shutdown:

```bash
# When server stops:
MinimalCleaningProcessor stats: processed=45, errors=0
```

**What to look for**:
- `processed > 0` - Processor ran
- `errors = 0` - No errors (good!)
- `errors > 0` - Check logs for details

---

## If It Still Fails

### Symptom: Flush timeouts
**Cause**: Processor is still breaking something

**Debug**:
```python
# In span_processor_minimal.py, add at top of on_end():
def on_end(self, span):
    if not self.enabled:
        return

    print(f"DEBUG: Processing {span.name}")  # Add this
    # rest of code...
```

Then run analysis and check which span causes the timeout.

### Symptom: Output still verbose
**Cause**: Processor not running or extraction failing

**Check**:
1. Server startup shows: `✅ Minimal cleaning processor registered`
2. Server shutdown shows: `processed > 0`
3. Logs don't show errors

**Debug**: Add print statements in `_extract_json()` to see what's happening.

---

## Why This Should Work

1. **Minimal modifications** - Only 1-2 attributes per span
2. **Strong validation** - Every check before modifying
3. **Fast processing** - <1ms per span (won't block)
4. **Safe fallback** - If anything fails, span exports unchanged
5. **Isolated errors** - One bad span doesn't affect others

---

## Expected Results

### LLM Spans
- ✅ Clean JSON output
- ✅ "raw" key extracted
- ✅ No verbose task descriptions

### Agent Spans
- ✅ Truncated to reasonable size
- ✅ Still readable
- ✅ Stats preserved

### All Spans
- ✅ Reach Arize reliably
- ✅ No flush timeouts
- ✅ Fast export

---

## Test Now!

```bash
# 1. Restart server
python main.py

# Expected output:
# ✅ Arize credentials loaded
# ✅ Minimal cleaning processor registered
# ✅ OpenInference tracing enabled

# 2. Run analysis

# 3. Check Arize UI
# - Traces should appear
# - LLM outputs should be cleaner
# - No flush timeout warnings
```

---

## Success Criteria

✅ Server starts without errors
✅ Analysis completes successfully
✅ Traces appear in Arize
✅ No flush timeout warnings
✅ LLM span outputs are clean JSON
✅ Agent span inputs/outputs are truncated
✅ Processor stats show: `processed > 0, errors = 0`

If all 7 pass: **Success!** 🎉

If any fail: Share the logs and we'll debug. 🔍
