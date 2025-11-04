# 🔍 Arize AX Granularity Guide

How to get deeper visibility into CrewAI agent execution in Arize.

## What You're Seeing Now ✅

In your screenshot, you can see:
- ✅ Agent execution spans (e.g., "Senior Technical Architect & Evaluator._execute_core")
- ✅ Timing information (16s, 16.85s, 12.64s)
- ✅ Span hierarchy showing agent relationships

## What's Missing (and How to Get It) 🎯

### 1. **LLM Prompts & Responses**

The CrewAI instrumentation automatically captures these, but you need to know where to look:

**How to view**:
1. Click on any agent span (e.g., "Senior Technical Architect & Evaluator._execute_core")
2. Go to the **"Input / Output" tab**
3. You should see:
   - **Input**: The prompt sent to the LLM
   - **Output**: The LLM's response

**Why you might not see them**:
- Spans with `._execute_core` are CrewAI internal - look for child spans with `llm` kind
- CrewAI creates **nested spans** - expand the tree to see LLM calls

### 2. **Agent Reasoning Steps**

CrewAI agents make decisions through multiple steps. To see these:

**In Arize UI**:
1. Click on your agent span
2. Go to **"Events" tab**
3. You'll see timeline events like:
   - `agent_thought` - What the agent is thinking
   - `agent_action` - Actions the agent is taking
   - `tool_invoked` - When tools are called

**We've added these events**:
- `task_1_technical_evaluation`
- `task_2_sales_methodology`
- `task_3_report_compilation`
- `starting_crew_kickoff`
- `crew_kickoff_completed`

### 3. **Token Usage & Cost**

To see LLM token usage and costs:

**In Arize UI**:
1. Look for spans with `openinference.span.kind = llm`
2. These will have attributes like:
   - `llm.token_count.prompt` - Input tokens
   - `llm.token_count.completion` - Output tokens
   - `llm.token_count.total` - Total tokens

**To enable cost tracking**:
- Arize automatically calculates costs if model info is present
- Check the **"Evaluations"** tab on your trace

### 4. **Individual LLM Calls**

To see each LLM call made by CrewAI agents:

**What to look for**:
1. Expand the agent spans in the trace tree
2. Look for child spans like:
   - `ChatAnthropic` or `ChatOpenAI` (the LLM being called)
   - These should have `openinference.span.kind = llm`

**Example hierarchy**:
```
crew_analysis_execution (agent)
└── Senior Technical Architect & Evaluator._execute_core
    └── ChatAnthropic (llm)  ← This is what you want to see!
        └── Attributes: prompt, response, tokens
```

### 5. **Tool Calls Within Agents**

If your agents use tools (they don't currently), you'd see:

```
agent_span (agent)
└── tool_call (tool)
    └── input/output of the tool
```

## 🎛️ Configuration Options

### Option A: Enable More Verbose Logging

Add to your `.env`:
```bash
# Enable detailed OpenTelemetry logging
OTEL_LOG_LEVEL=debug
```

This will print detailed trace information to your console.

### Option B: Increase CrewAI Verbosity

In `crew_analyzer.py`, CrewAI is already set to `verbose=True`:
```python
analysis_crew = Crew(
    agents=[...],
    tasks=[...],
    verbose=True  # ← Already enabled!
)
```

This makes CrewAI agents emit more internal events.

### Option C: Add Custom Tracking

For even more granularity, wrap specific operations:

```python
# Example: Track agent creation
with tracer.start_as_current_span("create_agent_technical_evaluator") as span:
    span.set_attribute("openinference.span.kind", "chain")
    span.set_attribute("agent.role", "technical_evaluator")
    span.set_attribute("agent.goal", "Assess technical depth")

    technical_evaluator = Agent(
        role='Senior Technical Architect & Evaluator',
        goal='...',
        llm=self.llm
    )
```

## 🎯 Recommended Arize Views

### View 1: Agent Graph

1. Go to your trace in Arize
2. Click the **"Agent Graph"** tab
3. This shows:
   - Visual flow of agents
   - Which agents called which tools
   - Agent → LLM relationships

### View 2: Timeline

1. Click the **"Timeline"** tab
2. This shows chronological execution:
   - When each agent started/stopped
   - Parallel vs sequential execution
   - Where time is spent

### View 3: Events (for debugging)

1. Select any span
2. Click **"Events"** tab
3. See all the custom events we added:
   - Task objectives
   - Execution status changes
   - Completion markers

## 🔧 Advanced: Custom Span Processor

If you want even MORE control (e.g., filtering, enriching data), you can add a custom span processor. I've created `span_processor.py` with:

1. **Data truncation** - Limit large payloads to reduce costs
2. **PII filtering** - Redact sensitive data
3. **Performance categorization** - Tag slow operations
4. **Custom enrichment** - Add metadata to all spans

To use it:
```python
from span_processor import SACallAnalyzerSpanProcessor
from opentelemetry.sdk.trace import TracerProvider

# During setup
tracer_provider = register(...)
tracer_provider.add_span_processor(SACallAnalyzerSpanProcessor())
```

## 📊 What Each Span Shows

| Span Name | Kind | What It Contains |
|-----------|------|------------------|
| `analyze_call_request` | chain | API request input, full analysis output |
| `fetch_gong_transcript` | agent | Gong URL, formatted transcript |
| `mcp_tool_call_*` | tool | MCP method, params, response |
| `parse_transcript_lines` | chain | Raw text, parsed structure |
| `sa_call_analysis` | agent | Full transcript, all agent outputs |
| `identify_sa` | agent | Speakers list, identified SA name |
| `crew_analysis_execution` | agent | Transcript preview, crew report |
| `Crew_*._execute_core` | agent | Individual agent execution (CrewAI internal) |
| `ChatAnthropic` | llm | LLM prompts, responses, tokens |

## 🎓 Key Arize Concepts

1. **Spans** = Units of work (API calls, agent tasks, etc.)
2. **Attributes** = Metadata on spans (input, output, custom fields)
3. **Events** = Timeline markers within a span
4. **Span Kind** = Category (agent, tool, chain, llm, retriever)

## 💡 Pro Tips

1. **Use filters**: In Arize, filter by `openinference.span.kind = "llm"` to see just LLM calls
2. **Search attributes**: Use the search box to find specific attributes like `sa_identified`
3. **Compare traces**: Select multiple traces to compare performance
4. **Set up monitors**: Create alerts when traces are slow or error

## ❓ Troubleshooting

**Q: I don't see LLM prompts/responses**
- A: Expand the agent spans - LLM calls are nested children
- Check that `verbose=True` is set on CrewAI agents

**Q: Events tab is empty**
- A: Events are only on spans where we explicitly added them
- Check `crew_analysis_execution` span for our custom events

**Q: Token counts are missing**
- A: LangChain/CrewAI should capture these automatically
- Verify your LLM is properly instrumented (it should be)

**Q: Too much data / high costs**
- A: Use the `SACallAnalyzerSpanProcessor` to truncate large payloads
- Set attribute length limits in `observability.py`

## 🚀 Next Steps

1. ✅ **Your telemetry is already comprehensive!**
2. Run a test analysis and explore the trace in Arize
3. Look for nested LLM spans under agent executions
4. Check the Events tab for our custom markers
5. Use Agent Graph view for visual flow

The granularity is there - you just need to know where to look in the Arize UI! 🎉
