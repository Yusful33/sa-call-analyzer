# SA Call Analyzer (id-pain)

AI-powered tooling for **Solution Architect sales calls** and related demos: a **CrewAI multi-agent** pipeline scores calls against **Command of the Message**, plus **Gong** and **BigQuery**-backed prospect views, **classification + prompts for the Claude arize-synthetic-demo skill**, and **hypothesis research** for accounts.

---

## Local Development (Quick Start)

### 1. Set up environment variables

```bash
# Copy the example env file
cp .env.example .env

# Edit .env and add your API keys (minimum required: ANTHROPIC_API_KEY)
```

**Required:**
| Variable | Description | Get it from |
|----------|-------------|-------------|
| `ANTHROPIC_API_KEY` | Claude API key | [console.anthropic.com](https://console.anthropic.com/) |

**Optional (for specific features):**
| Variable | Required for |
|----------|--------------|
| `GONG_ACCESS_KEY` + `GONG_SECRET_KEY` | Single Call Analysis (Gong URL input) |
| `BRAVE_API_KEY` | Hypothesis Research |
| Google Cloud credentials | Prospect Overview, Demo Builder (BigQuery) |
| `ARIZE_API_KEY` + `ARIZE_SPACE_ID` | Tracing/observability |

### 2. Run with Docker (recommended)

```bash
# Start all services
make dev-docker

# Or manually:
docker compose -f infra/docker-compose.yml up
```

Services will be available at:
- **Web UI**: http://localhost:3000
- **API**: http://localhost:8080
- **API Docs**: http://localhost:8080/docs

### 3. Run without Docker

```bash
# First time setup
make setup

# Start API + Web
make dev
```

Or manually in separate terminals:
```bash
# Terminal 1: API
cd apps/api && uv sync --extra hypothesis && uv run python main.py

# Terminal 2: Web  
cd apps/web && npm install && npm run dev
```

### BigQuery Setup (for Prospect Overview)

```bash
# Authenticate with Google Cloud (one-time)
gcloud auth application-default login
```

For detailed setup, troubleshooting, and Docker configuration, see **[LOCAL_DEV.md](LOCAL_DEV.md)**.

---

## Monorepo layout

| Path | Contents | Vercel project |
|------|-----------|----------------|
| **`apps/api/`** | FastAPI app (`main.py`), Python modules, legacy **`frontend/`**, **`hypothesis_tool/`**, **`pyproject.toml`** + **`requirements.txt`** | `id-pain-api` — **light** bundle (`API_SERVICE_MODE=light`, no CrewAI in `requirements.txt`) |
| **`apps/api-crew/`** | Vercel-only shim: **`vercel.json`**, **`api/index.py`**, crew **`requirements.txt`**; copies **`../api`** into **`_api_src/`** at install | `id-pain-api-crew` — CrewAI-heavy routes (`/api/analyze`, recap, prospect timeline) |
| **`apps/web/`** | Next.js 15 UI | `id-pain-web` |
| **`apps/gong-mcp/`** | Vercel **Node Functions** that call Gong directly (replaces the old MCP HTTP server) | `id-pain-gong-mcp` |
| **`infra/`** | `docker-compose.yml`, **`litellm/`**, **`gong-http-server/`** (legacy), **`k8s/`**, **`scripts/`** | local dev / EKS only |
| **`docs/`** | Extra guides (e.g. **Vercel** deployment) | — |

The **four** Vercel projects all use the **same Git repo**; they differ by **Root Directory**. **LiteLLM** is replaced in production by **Vercel AI Gateway** — there is no separate LiteLLM Vercel project.

**Deploying:** see **[`docs/deploy-vercel.md`](docs/deploy-vercel.md)** for end-to-end steps.

## What’s in the app

| Area | What it does |
|------|----------------|
| **Call analysis** | Paste a transcript or **Gong URL** → structured feedback (timestamps, strengths, improvements). Technical and sales methodology crews run **in parallel** for faster runs (~2–5 minutes typical). |
| **Prospect timeline** | **`/api/analyze-prospect`** (and **SSE** **`/api/analyze-prospect-stream`**) — all Gong calls matching a prospect, analyzed into a cumulative timeline. **`/api/calls-by-account`** lists calls without full analysis. |
| **CRM / analytics** | **`/api/prospect-overview`** — unified view from **BigQuery** (Salesforce, Gong summaries, Pendo, FullStory, etc.) when your warehouse and credentials are configured. |
| **Synthetic demo (skill)** | **`/api/classify-demo`** (+ optional overrides) classifies Gong/CRM signals and returns a ready-to-paste prompt for the **[arize-synthetic-demo](https://github.com/Arize-ai/solutions-resources/blob/main/.claude/skills/arize-synthetic-demo/SKILL.md)** Claude skill (`generator.py` / AX uploads run in Claude — not in this server). **`GET /api/custom-demo/skill`** returns static skill pointers. |
| **Hypothesis research** | **`/api/hypothesis-research`** — LangGraph agent: web search (**Brave**), optional BigQuery, LLM-driven hypotheses (see **`apps/api/hypothesis_tool/TRACE_FLOW.md`** for trace layout). |
| **Deliverables** | **`/api/generate-recap-slide`** — PowerPoint recap download from structured recap JSON. **`/api/generate-poc-document`** — Word (.docx) PoC SaaS / PoC VPC / PoT: fills in-template placeholders using BigQuery prospect data and an LLM (no separate appendix). |

Open **`http://localhost:8080/docs`** for interactive **OpenAPI** documentation.

---

## Quick start

### 1. Install dependencies

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
cd apps/api && uv sync --extra crew --extra litellm --extra hypothesis
```

**`--extra crew`** installs CrewAI + Chroma (needed for **`/api/analyze`**, recap slide, **`/api/analyze-prospect`**). Add **`--extra litellm`** if you use **`USE_LITELLM=true`** (LiteLLM proxy). Add **`--extra hypothesis`** for **`/api/hypothesis-research`**. A truly minimal install (`uv sync` only) matches the **Vercel light** worker — use **`API_SERVICE_MODE=light`** if you skip the crew extra.

### 2. Configure environment

Copy **`.env.example`** from the repo root to **`.env`** at the repo root (`main.py` loads `../.env` relative to **`apps/api/`**). Set at least **`ANTHROPIC_API_KEY`**. For Gong transcript fetch, set **`GONG_ACCESS_KEY`** and **`GONG_SECRET_KEY`**. For hypothesis search, set **`BRAVE_API_KEY`**. For BigQuery-backed features, configure **Google Application Default Credentials** or a service account (see **Docker** section below for compose mounts).

Get an Anthropic key from [console.anthropic.com](https://console.anthropic.com/).

### 3. Run the app

```bash
cd apps/api && uv run python main.py
```

Open **http://localhost:8080** in your browser (or **http://localhost:8080/docs** for the API).

---

## Docker Compose

Runs **`apps/api`** (the Python app), **LiteLLM** (port **4000**), and **Gong MCP** (host **8081** → container **8080**). Config: **`infra/litellm/config.yaml`**. From the **repository root**:

```bash
cp .env.example .env
# Edit .env

docker compose -f infra/docker-compose.yml up -d

docker compose -f infra/docker-compose.yml logs -f app
```

**Next.js (`apps/web/`):** the same compose file can start the UI on **http://localhost:3000**. Legacy FastAPI UI stays on **http://localhost:8080**.

See **[`apps/web/README.md`](apps/web/README.md)** for host-side `npm run dev` without Compose.

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

Prospect timelines and **`/api/classify-demo`** use additional LLM calls; pick smaller models via env where supported and keep experiments bounded.

---

## Deployment & deep dives

- **[DEPLOYMENT.md](DEPLOYMENT.md)** — AWS **EKS**, ECR images (**id-pain**, **litellm**, **gong-http-server**), secrets, Ingress, Grafana, **`infra/scripts/up.sh`** / **`down.sh`**, pod scale scripts.  
- **[docs/deploy-vercel.md](docs/deploy-vercel.md)** — Four Vercel projects from one repo (`apps/web`, `apps/api`, `apps/api-crew`, `apps/gong-mcp`) plus AI Gateway.  
- **[hypothesis_tool/TRACE_FLOW.md](apps/api/hypothesis_tool/TRACE_FLOW.md)** — Hypothesis / **analyze_signals** LLM inputs and Arize span layout.

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

Other notable routes: **`POST /api/prospect-overview`**, **`POST /api/analyze-prospect`**, **`POST /api/hypothesis-research`**, **`POST /api/classify-demo`**. Full schemas live at **`/docs`**.
