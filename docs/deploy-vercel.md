# Deploying everything on Vercel (single repo, four projects)

This monorepo deploys as **four separate Vercel projects** that all point to the same Git repository but use different **Root Directory** settings. The primary Python API (**`id-pain-api`**) ships a **light** dependency set (no CrewAI / Chroma) so it stays under Vercel’s serverless bundle limits; **CrewAI-heavy** routes run on a second Python project (**`id-pain-api-crew`**). The Python API does **not** ship LiteLLM in its default `requirements.txt`; use **provider SDKs** or optional **`OPENAI_BASE_URL`** toward **[Vercel AI Gateway](https://vercel.com/docs/ai-gateway)** for OpenAI-compatible routing. There is no separate LiteLLM service on Vercel.

| # | Vercel project | Root Directory | What it is |
|---|----------------|----------------|------------|
| 1 | `id-pain-web` | `apps/web` | Next.js 15 UI |
| 2 | `id-pain-api` | `apps/api` | FastAPI **light** worker (BigQuery, prospect overview, PoC doc, hypothesis when deps present, etc.) |
| 3 | `id-pain-api-crew` | `apps/api-crew` | FastAPI **Crew** worker — same `main:app` code copied from `../api` at install; `API_SERVICE_MODE=crew` |
| 4 | `id-pain-gong-mcp` | `apps/gong-mcp` | Node Vercel Functions for Gong API |
| — | (no project) | — | LLM routing → **[Vercel AI Gateway](https://vercel.com/docs/ai-gateway)** instead of LiteLLM |

You end up with **four** production URLs (one per project). The web app calls **`id-pain-api`** for most routes and **`id-pain-api-crew`** for analyze / recap / prospect timeline when **`NEXT_PUBLIC_CREW_API_URL`** is set. The API projects call **`id-pain-gong-mcp`** via **`GONG_MCP_URL`**; LLM traffic uses Anthropic / OpenAI SDKs or AI Gateway.

---

## 0. Prerequisites

```bash
npm i -g vercel        # CLI for `vercel deploy`, env, and link
vercel login
```

---

## 1. Project: **id-pain-gong-mcp** (Vercel Functions, Node)

Endpoints provided: `GET /api/health`, `POST /api/calls`, `POST /api/transcript`, `POST /api/call-info` — same shape as the legacy MCP HTTP server.

### Create

1. **Vercel → Add New… → Project** → import this repo.
2. **Root Directory:** `apps/gong-mcp`.
3. **Framework Preset:** **Other** (not Next.js, not Vite).
4. **Build & Output Settings:** leave **Build Command** / **Output Directory** / **Install Command** on **defaults** (toggles off), as in the Vercel UI. The repo’s **`npm run build`** always creates **`public/`** so the default rule (*“`public` if it exists”*) succeeds after the build.
5. **Environment Variables:**
   - `GONG_ACCESS_KEY`
   - `GONG_ACCESS_SECRET` (or `GONG_SECRET_KEY`)
6. **Deploy.** Note the production URL — you'll set it as `GONG_MCP_URL` on both API projects.

### CLI alternative

```bash
cd apps/gong-mcp
vercel link            # creates a new project the first time
vercel env add GONG_ACCESS_KEY production
vercel env add GONG_ACCESS_SECRET production
vercel deploy --prod
```

---

## 2. Project: **id-pain-api** (FastAPI **light** on Vercel Python)

> **Honest constraints:**
> - Vercel Python Functions cap at **300s** by default and a **250 MiB unzipped** bundle per function.
> - **`id-pain-api`** uses **`API_SERVICE_MODE=light`** (set in `apps/api/vercel.json`) so **CrewAI / Chroma are not imported** and **`requirements.txt`** stays smaller.
> - First-request **cold start** depends on imports (lighter without CrewAI).
> - **`USE_LITELLM` must be `false` on Vercel`** (no LiteLLM proxy is running). LLM calls use the Anthropic / OpenAI SDKs (or AI Gateway via `OPENAI_BASE_URL` for OpenAI-compatible routes).

### Serverless bundle size

CrewAI pulls **ChromaDB**, **ONNX Runtime**, and large wheels — that stack is deployed only on **`id-pain-api-crew`**, not on **`id-pain-api`**.

If **`id-pain-api`** still hits the 250 MiB limit (e.g. after adding more deps):

1. Set **`VERCEL_ANALYZE_BUILD_OUTPUT=1`** on the project and redeploy for a size report.
2. Prefer **`apps/api`** on **Railway / Fly.io / Cloud Run** via the root **`Dockerfile`** for an uncapped image.
3. Keep **Vercel** for **`apps/web`**, **`apps/gong-mcp`**, and the split Python workers.

### Create

1. **Vercel → Add New… → Project** → import the same repo.
2. **Root Directory:** `apps/api`.
3. **Framework Preset:** Other (the included `vercel.json` configures `@vercel/python` and routes `/(.*)` to `api/index.py`).
4. **Build & Output:** leave defaults; Vercel reads `requirements.txt`. The **`installCommand`** runs `pip install` then **`scripts/vercel_prune_site_packages.py`**.

### Environment variables

| Name | Why | Notes |
|------|-----|-------|
| `API_SERVICE_MODE` | **`light`** is set in `vercel.json` for this project | Override in dashboard only if you know what you are doing |
| `USE_LITELLM` | **Set to `false`** | Required on Vercel — no proxy is running |
| `ANTHROPIC_API_KEY` | LLM provider | Or use AI Gateway (below) |
| `OPENAI_API_KEY` | LLM provider | Or use AI Gateway (below) |
| `MODEL_NAME` | Default model | e.g. `claude-haiku-4-5` |
| `LLM_MODEL` | Hypothesis tool default | e.g. `claude-sonnet-4-20250514` |
| `BRAVE_API_KEY` | Brave web search (Hypothesis) | optional |
| `GONG_MCP_URL` | URL of **id-pain-gong-mcp** | e.g. `https://id-pain-gong-mcp.vercel.app` |
| `ARIZE_API_KEY`, `ARIZE_SPACE_ID` | Trace export | optional |
| `GCP_CREDENTIALS_BASE64` | base64 of service-account JSON | for BigQuery; written to `/tmp/gcp-credentials.json` at startup |
| `GOOGLE_CLOUD_PROJECT` | BigQuery project | e.g. `mkt-analytics-268801` |

### Use Vercel AI Gateway (OpenAI-compatible traffic)

The app’s **`openai_compat_completion`** helper respects **`OPENAI_BASE_URL`** / **`OPENAI_API_BASE`**. Point them at the Gateway when you want unified routing, fallbacks, or billing:

| Name | Value |
|------|-------|
| `OPENAI_BASE_URL` (or `OPENAI_API_BASE`) | `https://ai-gateway.vercel.sh/v1` |
| `OPENAI_API_KEY` | your **AI Gateway** key |

**Anthropic-native** paths still use **`ANTHROPIC_API_KEY`** unless you route Claude through an OpenAI-compatible surface. See [Vercel AI Gateway docs](https://vercel.com/docs/ai-gateway).

### CLI alternative

```bash
cd apps/api
vercel link
vercel env add USE_LITELLM production       # value: false
vercel env add ANTHROPIC_API_KEY production
vercel env add GONG_MCP_URL production       # production URL of id-pain-gong-mcp
vercel deploy --prod
```

### Notes

- **`/api/hypothesis-research`** needs the **`hypothesis`** optional dependency group. The Vercel **`requirements.txt`** omits that stack to save space; that route returns **503** with a clear message if those packages are missing.
- BigQuery: use `GCP_CREDENTIALS_BASE64`; the app writes it to `/tmp` on startup when `VERCEL` is set.

---

## 3. Project: **id-pain-api-crew** (FastAPI **Crew** worker)

Same **`main:app`** as **`apps/api`**, but:

- **`vercel.json` `installCommand`** runs **`bash vercel_install.sh`** (under the 256-character limit): copy **`../api` → `_api_src/`** (drops **`hypothesis_tool`**, **`tests`**, **`frontend`**, caches), **`pip install -r requirements.txt`**, uninstall **BigQuery** wheels, then **`vercel_prune_site_packages.py`** with **`PRUNE_VERCEL_CREW_WORKER=1`** (sympy/kubernetes/uv CLI, ONNX training trees, every **`tests/`** and **`__pycache__/`** under `site-packages`, etc.).
- Runtime sets **`API_SERVICE_MODE=crew`** via `apps/api-crew/api/index.py` (`os.environ.setdefault`), so **BigQuery is skipped** and only Gong + Crew routes matter for that worker.

**250 MiB cap:** CrewAI + Chroma + ONNX Runtime + gRPC OTLP (via **`arize-otel`**) can still exceed Vercel’s uncompressed function limit on Linux even after pruning. If production deploys keep failing, host **`id-pain-api-crew`** on **Railway / Fly / Cloud Run** using the repo **`Dockerfile`**, or keep only **`id-pain-api`** (light) + **`apps/web`** on Vercel and point **`NEXT_PUBLIC_CREW_API_URL`** at the container URL.

### Create

1. **Vercel → Add New… → Project** → same repo.
2. **Root Directory:** `apps/api-crew`.
3. **Framework Preset:** Other (`vercel.json` routes `/(.*)` → `api/index.py`).
4. **Environment variables:** mirror **`id-pain-api`** for LLM and Gong (**`GONG_MCP_URL`**, **`ANTHROPIC_API_KEY`**, Arize keys, etc.). Omit BigQuery vars if you prefer; crew mode does not initialize BigQuery.

### Regenerate `requirements.txt`

From **`apps/api`**:

```bash
uv export --no-hashes --no-dev --format requirements-txt --no-emit-project --no-annotate --extra crew -o ../api-crew/requirements.txt
```

---

## 4. Project: **id-pain-web** (Next.js)

### Create

1. **Vercel → Add New… → Project** → import the same repo.
2. **Root Directory:** `apps/web`.
3. **Framework Preset:** Next.js (auto-detected).
4. **Environment variables:**
   - **`NEXT_PUBLIC_LEGACY_API_URL`** — production URL of **`id-pain-api`** (no trailing slash).
   - **`NEXT_PUBLIC_CREW_API_URL`** — production URL of **`id-pain-api-crew`** (no trailing slash). When unset, the browser sends Crew routes to the same host as **`NEXT_PUBLIC_LEGACY_API_URL`** (fine for local single-server dev).

### CLI alternative

```bash
cd apps/web
vercel link
vercel env add NEXT_PUBLIC_LEGACY_API_URL production    # https://id-pain-api.vercel.app
vercel env add NEXT_PUBLIC_CREW_API_URL production      # https://id-pain-api-crew.vercel.app
vercel deploy --prod
```

---

## 5. Wiring summary

```
[ Browser ]
     │
     ▼
[ id-pain-web (Vercel) ]   ← apps/web
     │  NEXT_PUBLIC_LEGACY_API_URL  →  thin API (most /api/*)
     │  NEXT_PUBLIC_CREW_API_URL    →  crew API (/api/analyze, recap, analyze-prospect*)
     ▼
[ id-pain-api (Vercel) ]   ← apps/api   (API_SERVICE_MODE=light)
[ id-pain-api-crew ]       ← apps/api-crew (API_SERVICE_MODE=crew, _api_src copy)
     │  GONG_MCP_URL                     OPENAI_API_BASE = ai-gateway.vercel.sh/v1
     ▼                                   │
[ id-pain-gong-mcp ]               [ Vercel AI Gateway ]
     │ Gong API                          │ Anthropic / OpenAI / etc
     ▼                                   ▼
[ Gong (api.gong.io) ]            [ LLM providers ]
```

## 6. CORS

`apps/api/main.py` allows `*` for local dev. Before going public, narrow `allow_origins` to your **`id-pain-web`** and **`id-pain-gong-mcp`** (and both API) Vercel domains.

## 7. Local dev unchanged

Docker Compose still works for local dev:

```bash
docker compose -f infra/docker-compose.yml up
```

It runs LiteLLM locally even though prod uses AI Gateway — that's fine.

For **`apps/api`** without Docker, install Crew locally when you need analyze / recap / prospect timeline:

```bash
cd apps/api && uv sync --extra crew --extra litellm --extra hypothesis
```

`API_SERVICE_MODE` defaults to **`full`** locally (BigQuery + crew routes).
