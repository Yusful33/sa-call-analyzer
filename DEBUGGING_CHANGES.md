# Debugging Changes for LangChain/LiteLLM Instrumentation

## Changes Made

### 1. Enhanced Debug Logging in `observability.py`
- Added verification that LangChain instrumentor has tracer attached
- Added check to verify BaseCallbackManager is properly wrapped
- Added span processor and exporter configuration logging
- Added optional CrewAI instrumentor (enable with `ENABLE_CREWAI_INSTRUMENTOR=true`)

### 2. Fixed Callback Manager in `crew_analyzer.py`
- Changed from passing `callbacks=[self.token_callback]` to using `CallbackManager([self.token_callback])`
- This ensures LangChainInstrumentor can properly hook into the callback system
- Added debug logging to verify span context and agent LLM configuration

### 3. Added Force Flush in `main.py`
- Added `force_flush_spans()` call after analysis completes
- Ensures spans are sent to Arize immediately instead of waiting for batch export

## How to Test

1. **Run your application** and check the console output for:
   - `✅ LangChain instrumentor has tracer attached`
   - `✅ BaseCallbackManager.__init__ wrapped: True`
   - Span export configuration details
   - Current span context information

2. **Check Arize** for traces:
   - Look for spans with `openinference.span.kind = "llm"`
   - Check if LangChain chain spans appear
   - Verify token counts are being captured

3. **If traces still don't appear**, try enabling CrewAI instrumentor:
   ```bash
   export ENABLE_CREWAI_INSTRUMENTOR=true
   ```
   This will help diagnose if CrewAI is bypassing LangChain's callback system.

## Expected Debug Output

When you run the application, you should see:

```
🔍 Instrumentation Verification:
   ✅ LangChain instrumentor has tracer attached
   ✅ BaseCallbackManager.__init__ wrapped: True

🔍 Span Export Configuration:
   Span processor type: MultiSpanProcessor
   - Processor 0: CleaningSpanProcessor
   - Processor 1: BatchSpanProcessor
     Exporter: OTLPSpanExporter

🔍 Current span context - Trace ID: ..., Span ID: ...
🔍 Agent LLM type: ChatAnthropic
🔍 Agent LLM callbacks: CallbackManager
```

## Next Steps if Issues Persist

1. Check the console for any error messages during instrumentation
2. Verify Arize credentials are correct
3. Check network connectivity to Arize API
4. Review OpenTelemetry debug logs (already enabled)
5. Try enabling CrewAI instrumentor to see if those traces appear


