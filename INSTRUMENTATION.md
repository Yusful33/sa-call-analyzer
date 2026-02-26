# SA Call Analyzer – Arize instrumentation

All API and workflow steps are instrumented so traces and spans show up in Arize with consistent naming. Use the span name prefix **`sa_call_analyzer`** to filter in Arize.

## Configuration

- **observability.py**: `LangChainInstrumentor` uses `separate_trace_from_runtime_context=False` so LangChain/LLM spans attach to the current API span (one trace per request, no orphan spans).
- **Hypothesis agent**: Uses the same pattern in `hypothesis_tool/tracing.py` for standalone runs; when invoked from the app, it runs under the app’s tracer.

## Top-level API spans (main.py)

| Span name | Route / action | Description |
|-----------|----------------|-------------|
| `sa_call_analyzer.analyze` | `POST /api/analyze` | Analyze a call (Gong URL or manual transcript). |
| `sa_call_analyzer.generate_recap_slide` | `POST /api/generate-recap-slide` | Generate a recap slide from a transcript. |
| `sa_call_analyzer.get_calls_by_account` | `POST /api/calls-by-account` | List calls for an account (BigQuery + optional fuzzy match). |
| `sa_call_analyzer.analyze_prospect` | `POST /api/analyze-prospect` | Non-streaming prospect analysis. |
| `sa_call_analyzer.analyze_prospect_stream` | `POST /api/analyze-prospect-stream` | Streaming prospect analysis. |
| `sa_call_analyzer.get_prospect_overview` | `POST /api/prospect-overview` | Fetch prospect overview (BigQuery + Gong summary). |
| `sa_call_analyzer.get_example_transcript` | `GET /api/example` | Return example transcript for testing. |
| `sa_call_analyzer.classify_demo` | `POST /api/classify-demo` | Classify prospect use case and framework from CRM/Gong. |
| `sa_call_analyzer.generate_demo_stream` | `POST /api/generate-demo-stream` | Stream demo trace generation (SSE). |
| `sa_call_analyzer.generate_demo` | `POST /api/generate-demo` | Legacy non-streaming demo generation. |
| `sa_call_analyzer.export_script` | `GET /api/export-script` | Generate and download standalone demo script. |
| `sa_call_analyzer.create_online_evals` | `POST /api/create-online-evals` | Create online eval task(s) in Arize. |
| `sa_call_analyzer.hypothesis_research` | `POST /api/hypothesis-research` | Run hypothesis research agent for a company. |

## Hypothesis agent spans (research_agent.py)

When the hypothesis workflow runs (from `POST /api/hypothesis-research` or standalone), each LangGraph node and route is wrapped in a span so the full workflow is visible under one trace.

| Span name pattern | Description |
|-------------------|-------------|
| `hypothesis_agent.<node_name>` | One span per graph node (e.g. `research`, `analyze_signals`, `synthesize_hypotheses`). Attributes include input/output summaries (e.g. `company.name`, `signals_count`, `confidence_score`, `hypotheses_count`). |
| `hypothesis_agent.route.<route_name>` | Conditional edge decisions (e.g. `route.after_research`, `route.after_analysis`). Attribute `hypothesis_agent.route` = chosen branch name. |

## Span kinds

- **CHAIN**: Orchestration (API route, demo stream, hypothesis research).
- **TOOL**: Data/API helpers (e.g. get_calls_by_account).
- LangChain/LiteLLM/OpenAI instrumentors add **LLM**, **CHAIN**, **TOOL** spans under these roots.

## Trace propagation

- **Threads**: Where work is offloaded with `asyncio.to_thread`, the demo stream and multi-agent runners propagate trace context (e.g. `ThreadingInstrumentor`, or explicit `contextvars.copy_context().run(...)`) so child spans are not orphaned.
- **Streaming**: `generate_demo_stream` starts a root span at the beginning of the SSE generator and ends it in a `finally` so the whole stream is one trace.

## Filtering in Arize

- Filter by span name prefix: `sa_call_analyzer` for API-level spans.
- Filter by `hypothesis_agent` for hypothesis workflow steps.
- Use attributes such as `company.name`, `account.name`, `request.input_type`, `openinference.span.kind` for slicing.
