# Deploying everything on Vercel (single repo, three projects)

This monorepo deploys as **three separate Vercel projects** that all point to the same Git repository but use different **Root Directory** settings. **LiteLLM is replaced by Vercel AI Gateway** — there is no separate LiteLLM project.

| # | Vercel project | Root Directory | What it is |
|---|----------------|----------------|------------|
| 1 | `id-pain-web` | `apps/web` | Next.js 15 UI |
| 2 | `id-pain-api` | `apps/api` | FastAPI Python (Vercel Python runtime, Fluid Compute) |
| 3 | `id-pain-gong-mcp` | `apps/gong-mcp` | Node Vercel Functions for Gong API |
| — | (no project) | — | LLM routing → **[Vercel AI Gateway](https://vercel.com/docs/ai-gateway)** instead of LiteLLM |

You'll end up with three production URLs (one per project). The web project calls the API project; the API project calls the gong-mcp project; the API project sends LLM traffic through AI Gateway.

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
4. **Build & Output Settings** (Project → Settings → General, or during import):
   - **Output Directory:** leave **empty** unless you know you need one. If you see an error like *“No Output Directory named public found”*, either clear **Output Directory** here, or keep the repo’s committed **`public/index.html`** (this repo includes it so a `public` output exists if Vercel still expects it).
5. **Environment Variables:**
   - `GONG_ACCESS_KEY`
   - `GONG_ACCESS_SECRET` (or `GONG_SECRET_KEY`)
6. **Deploy.** Note the production URL — you'll set it as `GONG_MCP_URL` on the API project.

### CLI alternative

```bash
cd apps/gong-mcp
vercel link            # creates a new project the first time
vercel env add GONG_ACCESS_KEY production
vercel env add GONG_ACCESS_SECRET production
vercel deploy --prod
```

---

## 2. Project: **id-pain-api** (FastAPI on Vercel Python)

> **Honest constraints:**
> - Vercel Python Functions cap at **300s** by default and a fixed bundle size. Heavy CrewAI runs near the limit may time out — that's an OK tradeoff for SA call analysis but not for marathon analyses.
> - First-request **cold start ≈ 5–10s** (Arize + LangChain + CrewAI imports). Use a "warm" cron or Vercel Functions warming to mitigate.
> - **`USE_LITELLM` must be `false` on Vercel** (no LiteLLM proxy is running). LLM calls go either to providers directly or through AI Gateway.

### Create

1. **Vercel → Add New… → Project** → import the same repo.
2. **Root Directory:** `apps/api`.
3. **Framework Preset:** Other (the included `vercel.json` configures `@vercel/python` and routes `/(.*)` to `api/index.py`).
4. **Build & Output:** leave defaults; Vercel reads `requirements.txt`.

### Environment variables

| Name | Why | Notes |
|------|-----|-------|
| `USE_LITELLM` | **Set to `false`** | Required on Vercel — no proxy is running |
| `ANTHROPIC_API_KEY` | LLM provider | Or use AI Gateway (below) |
| `OPENAI_API_KEY` | LLM provider | Or use AI Gateway (below) |
| `MODEL_NAME` | Default model | e.g. `claude-haiku-4-5` |
| `LLM_MODEL` | Hypothesis tool default | e.g. `claude-sonnet-4-20250514` |
| `BRAVE_API_KEY` | Brave web search (Hypothesis) | optional |
| `GONG_MCP_URL` | URL of project #1 | e.g. `https://id-pain-gong-mcp.vercel.app` |
| `ARIZE_API_KEY`, `ARIZE_SPACE_ID` | Trace export | optional |
| `GCP_CREDENTIALS_BASE64` | base64 of service-account JSON | for BigQuery; written to `/tmp/gcp-credentials.json` at startup |
| `GOOGLE_CLOUD_PROJECT` | BigQuery project | e.g. `mkt-analytics-268801` |

### Use Vercel AI Gateway (replaces LiteLLM)

Vercel AI Gateway exposes an **OpenAI-compatible** endpoint, so you can point any OpenAI-style client (including `litellm`) at it. Add these instead of provider keys when you want centralized observability/fallbacks/billing:

| Name | Value |
|------|-------|
| `OPENAI_API_BASE` (or `OPENAI_BASE_URL`) | `https://ai-gateway.vercel.sh/v1` |
| `OPENAI_API_KEY` | your **AI Gateway** key |

For Anthropic models routed through the Gateway, use `provider/model` strings (e.g. `anthropic/claude-sonnet-4`) and the OpenAI-compatible URL above. See [Vercel AI Gateway docs](https://vercel.com/docs/ai-gateway) for the latest.

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

- The legacy HTML at `apps/api/frontend/index.html` is still served at `/` from this project; you usually won't hit it on Vercel because the Next app is the user-facing UI.
- BigQuery: Vercel can't mount your gcloud ADC. Use `GCP_CREDENTIALS_BASE64`; the app already writes it to `/tmp` on startup when `VERCEL` is set.
- If your CrewAI bundle is too large, trim `pyproject.toml` (e.g. drop the `chroma` extra) and re-export with `uv export` to regenerate `requirements.txt`.

---

## 3. Project: **id-pain-web** (Next.js)

### Create

1. **Vercel → Add New… → Project** → import the same repo.
2. **Root Directory:** `apps/web`.
3. **Framework Preset:** Next.js (auto-detected).
4. **Environment variables:**
   - `NEXT_PUBLIC_LEGACY_API_URL` = the production URL of **id-pain-api** (no trailing slash).

### CLI alternative

```bash
cd apps/web
vercel link
vercel env add NEXT_PUBLIC_LEGACY_API_URL production    # https://id-pain-api.vercel.app
vercel deploy --prod
```

---

## 4. Wiring summary

```
[ Browser ]
     │
     ▼
[ id-pain-web (Vercel) ]   ← apps/web
     │  fetch NEXT_PUBLIC_LEGACY_API_URL
     ▼
[ id-pain-api (Vercel) ]   ← apps/api
     │  GONG_MCP_URL                     OPENAI_API_BASE = ai-gateway.vercel.sh/v1
     ▼                                   │
[ id-pain-gong-mcp ]               [ Vercel AI Gateway ]
     │ Gong API                          │ Anthropic / OpenAI / etc
     ▼                                   ▼
[ Gong (api.gong.io) ]            [ LLM providers ]
```

## 5. CORS

`apps/api/main.py` allows `*` for local dev. Before going public, narrow `allow_origins` to your `id-pain-web` and `id-pain-gong-mcp` Vercel domains.

## 6. Local dev unchanged

Docker Compose still works for local dev:

```bash
docker compose -f infra/docker-compose.yml up
```

It runs LiteLLM locally even though prod uses AI Gateway — that's fine.
