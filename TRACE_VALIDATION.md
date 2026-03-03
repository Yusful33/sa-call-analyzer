# Validating hypothesis-generator traces in Arize

If the **Traces** tab in the **hypothesis-generator** project shows "No Data" or you want to confirm that the correct project and trace tree are used, use the steps below.

## What you should see in hypothesis-generator

- **One root span** per request: `sa_call_analyzer.hypothesis_research` (attributes: `input.value`, `company.name`).
- **Child spans**: LangGraph node spans (`hypothesis_agent.plan_research`, `execute_research`, `check_crm`, `analyze_signals`, `evaluate_confidence`, `load_playbook`, `generate_hypotheses`, `validate_hypotheses`, `finalize_result`), optional BigQuery spans under `check_crm` / `finalize_result`, and **LLM / ChatAnthropic / LiteLLM** spans under the nodes that call the model (e.g. under `generate_hypotheses`, `analyze_signals`), including token/cost when available.

If LLM spans or token/cost are missing, context propagation (e.g. from instrumentors or threads) may need to be checked.

## Validation with debug logging

1. **Enable trace debug**  
   Set in your environment (e.g. in `.env` or when starting the app):
   ```bash
   ARIZE_TRACE_DEBUG=1
   ```

2. **Run a hypothesis request**  
   Call `POST /api/hypothesis-research` (e.g. with a company name).

3. **Check logs**  
   You should see a line like:
   ```text
   [ARIZE_TRACE_DEBUG] hypothesis_research → project='hypothesis-generator' trace_id=abc123...
   ```
   - `project` should be `'hypothesis-generator'`. If it is something else (e.g. `'single-call-analysis'`), the middleware or path routing is wrong.
   - Copy the **trace_id** (full 32-character hex string).

4. **Confirm in Arize**  
   - Open the **hypothesis-generator** project in your Arize space.
   - Go to **Traces** and search or filter by that **trace_id**.
   - You should see the same trace: root `sa_call_analyzer.hypothesis_research` and all child spans (LangGraph, BigQuery, LLM) under it. If the trace appears in another project, the request’s provider was not set to hypothesis-generator.

## If the Traces tab still shows "No Data"

- **Spans** may be present (e.g. 15 spans) while **Traces** shows "No Data" when the root span and children use different trace_ids (e.g. root from default provider, children from hypothesis provider). The fix is to create the root span with the **current** request’s tracer via `api_span(...)` in the hypothesis route (already in place) and to use a **per-request** provider (context var) so concurrent requests don’t overwrite it.
- After the fix, enable `ARIZE_TRACE_DEBUG=1` and confirm in logs that `project='hypothesis-generator'` and then look up the logged `trace_id` in the hypothesis-generator project’s Traces view.
