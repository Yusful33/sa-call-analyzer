# SA Call Analyzer (id-pain)

AI-powered tooling for **Solution Architect sales calls** and related demos: a **CrewAI multi-agent** pipeline scores calls against **Command of the Message**, plus **Gong** and **BigQuery**-backed prospect views, an **Arize**-oriented **custom demo trace generator**, and **hypothesis research** for accounts.

## What’s in the app

| Area | What it does |
|------|----------------|
| **Call analysis** | Paste a transcript or **Gong URL** → structured feedback (timestamps, strengths, improvements). Technical and sales methodology crews run **in parallel** for faster runs (~2–5 minutes typical). |
| **Prospect timeline** | **`/api/analyze-prospect`** (and **SSE** **`/api/analyze-prospect-stream`**) — all Gong calls matching a prospect, analyzed into a cumulative timeline. **`/api/calls-by-account`** lists calls without full analysis. |
| **CRM / analytics** | **`/api/prospect-overview`** — unified view from **BigQuery** (Salesforce, Gong summaries, Pendo, FullStory, etc.) when your warehouse and credentials are configured. |
| **Custom demo builder** | **`/api/classify-demo`**, **`/api/generate-demo`**, **`/api/generate-demo-stream`** — pick use case + framework (e.g. LangGraph, CrewAI), run LLM pipelines, send traces to **Arize** (optional Space ID / API key in the request). **`/api/export-script`** downloads a standalone Python script; **`/api/create-online-evals`** sets up Arize online evals after a run. |
| **Hypothesis research** | **`/api/hypothesis-research`** — LangGraph agent: web search (**Brave**), optional BigQuery, LLM-driven hypotheses (see **`hypothesis_tool/TRACE_FLOW.md`** for trace layout). |
| **Deliverables** | **`/api/generate-recap-slide`** — PowerPoint recap download from structured recap JSON. **`/api/generate-poc-document`** — Word (.docx) PoC SaaS / PoC VPC / PoT: fills in-template placeholders using BigQuery prospect data and an LLM (no separate appendix). |

Open **`http://localhost:8080/docs`** for interactive **OpenAPI** documentation.

---

## Quick start

### 1. Install dependencies

```bash
# Install uv (if you don't have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install project dependencies
uv sync
```

### 2. Configure environment

Copy **`.env.example`** to **`.env`** and set at least **`ANTHROPIC_API_KEY`**. For Gong transcript fetch, set **`GONG_ACCESS_KEY`** and **`GONG_SECRET_KEY`**. For hypothesis search, set **`BRAVE_API_KEY`**. For BigQuery-backed features, configure **Google Application Default Credentials** or a service account (see **Docker** section below for compose mounts).

Get an Anthropic key from [console.anthropic.com](https://console.anthropic.com/).

### 3. Run the app

```bash
uv run python main.py
```

Open **http://localhost:8080** in your browser (or **http://localhost:8080/docs** for the API).

---

## Docker Compose

Runs the **app**, **LiteLLM** (port **4000**), and **Gong MCP** (host port **8081** → container **8080**). Compose uses **`./litellm/config.yaml`** for LiteLLM and wires **`GONG_MCP_URL=http://gong-mcp:8080`**.

```bash
# First time: copy env template (`.env` is not committed)
cp .env.example .env
# Edit .env — at minimum set ANTHROPIC_API_KEY (and OPENAI_API_KEY if your LiteLLM config uses OpenAI)

docker compose up -d

# Logs for the main web app (service name is "app")
docker compose logs -f app
```

**BigQuery from Docker:** the compose file mounts host **`~/.config/gcloud/application_default_credentials.json`** read-only for local ADC; adjust or use a service account file if needed. **`GOOGLE_CLOUD_PROJECT`** defaults in compose for the marketing analytics project—change it if your data lives elsewhere.

---

## Configuration (summary)

| Variable | Purpose |
|----------|---------|
| **`ANTHROPIC_API_KEY`** | Primary LLM provider for analysis and many flows |
| **`OPENAI_API_KEY`** | Used when models or LiteLLM routes go through OpenAI |
| **`MODEL_NAME`** | Default model (retired Anthropic IDs are **aliased** in code to current model IDs) |
| **`USE_LITELLM`**, **`LITELLM_BASE_URL`**, **`LITELLM_API_KEY`** | Route LLMs via LiteLLM instead of direct Anthropic |
| **`GONG_ACCESS_KEY`**, **`GONG_SECRET_KEY`** | Gong API (transcript fetch, prospect/call discovery) |
| **`ARIZE_API_KEY`**, **`ARIZE_SPACE_ID`** | Export traces to Arize AX |
| **`BRAVE_API_KEY`** | Web search for hypothesis research |
| **`LLM_MODEL`** | Hypothesis tool default LLM (see **`hypothesis_tool/config.py`**) |
| **`GCP_CREDENTIALS_BASE64`** | Railway / K8s: base64 service account JSON (decoded at startup) |
| **`GOOGLE_APPLICATION_CREDENTIALS`** | Path to GCP key file when not using ADC |
| **`ARIZE_TRACE_DEBUG`** | Set to **`1`** for extra trace / project logging (e.g. hypothesis debugging) |

For local development **without** Docker, set **`USE_LITELLM=false`** to call Anthropic directly unless you run LiteLLM separately.

---

## How call analysis works

Four CrewAI agents:

1. **Call classifier** — Understands call structure and context  
2. **Technical evaluator** — Technical depth and accuracy  
3. **Sales methodology & discovery** — Discovery and Command of the Message  
4. **Report compiler** — Synthesizes actionable recommendations  

Each insight can include what happened (with timestamp), why it matters, a better approach, and example phrasing.

---

## Cost estimate (call analysis)

| Model | Cost / call (approx.) | Quality |
|-------|------------------------|---------|
| Claude 3.5 Haiku (or mapped successor) | $0.25–0.50 | Strong |
| Claude 3.5 Sonnet (or mapped successor) | $1.50–2.50 | Highest |

Custom demo generation and prospect timelines use additional LLM calls; use **`arize_demo_traces`** cost guards and smaller models for experiments.

---

## Deployment & deep dives

- **[DEPLOYMENT.md](DEPLOYMENT.md)** — AWS **EKS**, ECR images (**id-pain**, **litellm**, **gong-http-server**), secrets, Ingress, Grafana, **`scripts/up.sh`** / **`down.sh`**, pod scale scripts.  
- **[hypothesis_tool/TRACE_FLOW.md](hypothesis_tool/TRACE_FLOW.md)** — Hypothesis / **analyze_signals** LLM inputs and Arize span layout.

---

## API examples

```bash
# Analyze with transcript text
curl -X POST http://localhost:8080/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"transcript": "0:16 | John\nHello everyone..."}'

# Analyze with Gong URL
curl -X POST http://localhost:8080/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"gong_url": "https://app.gong.io/call?id=YOUR_CALL_ID"}'

# Health check
curl http://localhost:8080/health
```

Other notable routes: **`POST /api/prospect-overview`**, **`POST /api/analyze-prospect`**, **`POST /api/hypothesis-research`**, **`POST /api/generate-demo`** / **`/api/generate-demo-stream`**. Full schemas live at **`/docs`**.
