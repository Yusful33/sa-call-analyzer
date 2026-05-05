# Deploying everything on Vercel (single repo, four projects)

This monorepo deploys as **four or five Vercel projects** (same Git repo, different **Root Directory**). The primary Python API (**`id-pain-api`**) ships a **light** dependency set (no CrewAI / Chroma) so it stays under Vercel’s serverless bundle limits; **CrewAI-heavy** routes run on a second Python project (**`id-pain-api-crew`**) **or**, if that bundle still exceeds the **250 MiB** cap, on a container (Railway / Fly / Cloud Run) with **`id-pain-api-crew-proxy`** on Vercel as a thin forwarder. The Python API does **not** ship LiteLLM in its default `requirements.txt`; use **provider SDKs** or optional **`OPENAI_BASE_URL`** toward **[Vercel AI Gateway](https://vercel.com/docs/ai-gateway)** for OpenAI-compatible routing. There is no separate LiteLLM service on Vercel.

| # | Vercel project | Root Directory | What it is |
|---|----------------|----------------|------------|
| 1 | `id-pain-web` | `apps/web` | Next.js 15 UI |
| 2 | `id-pain-api` | `apps/api` | FastAPI **light** worker (BigQuery, prospect overview, PoC doc, hypothesis research, etc.) |
| 3 | `id-pain-api-crew` | `apps/api-crew` | FastAPI **Crew** worker — same `main:app` code copied from `../api` at install; `API_SERVICE_MODE=crew` |
| 4 | `id-pain-gong-mcp` | `apps/gong-mcp` | Node Vercel Functions for Gong API |
| 5 (optional) | `id-pain-api-crew-proxy` | `apps/api-crew-proxy` | **Tiny** Starlette + httpx reverse proxy → set **`CREW_BACKEND_URL`** to the full Crew worker (container URL); point **`NEXT_PUBLIC_CREW_API_URL`** at this project when the monolithic crew function is too large |
| — | (no project) | — | LLM routing → **[Vercel AI Gateway](https://vercel.com/docs/ai-gateway)** instead of LiteLLM |

You end up with **four or five** production URLs depending on whether you use the crew proxy. The web app calls **`id-pain-api`** for most routes and **`id-pain-api-crew`** (or **`id-pain-api-crew-proxy`**) for analyze / recap / prospect timeline when **`NEXT_PUBLIC_CREW_API_URL`** is set. The API projects call **`id-pain-gong-mcp`** via **`GONG_MCP_URL`**; LLM traffic uses Anthropic / OpenAI SDKs or AI Gateway.

---

## 0. Prerequisites

```bash
npm i -g vercel        # CLI for `vercel deploy`, env, and link
vercel login
```

---

## 1. Project: **id-pain-gong-mcp** (Vercel Functions, Node)

Endpoints: `GET /api/health`, `POST /api/calls`, `POST /api/transcript`, `POST /api/call-info`. Implementations use **`@vercel/node`** `VercelApiHandler` (`(req, res) => …`), which is the reliable pattern for **`api/*.ts`** on the **Other** preset. **`package.json`** sets **`"type": "module"`** so Vercel’s emitted **`export`** syntax in function bundles loads correctly on Node. Relative imports in **`api/`** and **`lib/`** use **`.js` extensions** (e.g. `../lib/gong.js`) so Node’s ESM resolver finds files under `/var/task/...`. **`vercel.json`** rewrites **`/transcript`**, **`/calls`**, **`/call-info`**, and **`/health`** to those `/api/*` routes so the FastAPI **`GongMCPClient`** (which posts to the non-`/api` paths) works without code changes.

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
> - **`id-pain-api`** may use **`API_SERVICE_MODE=light`** (smaller import surface) or **`full`** (call analysis + BigQuery on one worker). Check `apps/api/vercel.json` for the value in your deployment.
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
| `API_SERVICE_MODE` | Set in `apps/api/vercel.json` (often **`full`** or **`light`**) | Use **`crew`** only on the separate crew worker; **`full`** and **`light`** both initialize BigQuery when credentials exist |
| `USE_LITELLM` | **Set to `false`** | Required on Vercel — no proxy is running |
| `ANTHROPIC_API_KEY` | LLM provider | Or use AI Gateway (below) |
| `OPENAI_API_KEY` | LLM provider | Or use AI Gateway (below) |
| `MODEL_NAME` | Default model | e.g. `claude-haiku-4-5` |
| `LLM_MODEL` | Hypothesis tool default | e.g. `claude-sonnet-4-20250514` |
| `BRAVE_API_KEY` | Brave web search (Hypothesis) | optional |
| `GONG_MCP_URL` | URL of **id-pain-gong-mcp** | Prefer full URL with scheme, e.g. `https://id-pain-gong-mcp.vercel.app`. If the scheme is omitted (host only), the API prepends `https://` and strips trailing slashes. |
| `GONG_MCP_VERCEL_BYPASS_SECRET` | **Only if Gong MCP is on Vercel with Deployment Protection** | Same value as **Protection Bypass for Automation** on the **gong-mcp** project ([docs](https://vercel.com/docs/security/deployment-protection/methods-to-bypass-deployment-protection/protection-bypass-automation)). The API sends it as header **`x-vercel-protection-bypass`**. Without it, transcript calls return **401** with an HTML “Authentication Required” page. Alternative: turn off Vercel Authentication / protection on **production** for that project. |
| `ARIZE_API_KEY`, `ARIZE_SPACE_ID` | Trace export | optional |
| `GCP_CREDENTIALS_BASE64` | base64 of service-account JSON | for BigQuery; written to `/tmp/gcp-credentials.json` at startup |
| `GOOGLE_CLOUD_PROJECT` | BigQuery project id for `BigQueryClient` | e.g. `mkt-analytics-268801` (falls back to this if unset); optional alias **`BQ_PROJECT_ID`** |
| `STILLNESS_WEB_URL` | Canonical Next.js origin | e.g. `https://arize-gtm-stillness.vercel.app` — **`GET /`** on the Python project **302-redirects** here so users are not stuck on `id-pain-api.*.vercel.app`. Also set in **`apps/api/vercel.json`** for this repo’s default. |
| `NEXT_PUBLIC_LEGACY_API_URL` | (**`apps/web`**) FastAPI origin for **build-time rewrites** | e.g. **`https://arize-gtm-stillness-api.vercel.app`**. Production UI calls same-origin **`/api/*`**; Next proxies to this URL. Redeploy the **web** app when it changes. |

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
vercel env add GCP_CREDENTIALS_BASE64 production   # base64(minified service account JSON)
vercel env add GOOGLE_CLOUD_PROJECT production     # e.g. mkt-analytics-268801
vercel deploy --prod
```

### Notes

- **`/api/hypothesis-research`** uses **LangGraph** (core), **httpx**, and **`pydantic-settings`** (both in the default **`requirements.txt`** export). Optional **`hypothesis`** / **`full`** extras add **SQLAlchemy**, **BeautifulSoup**, and **aiosqlite** for feedback DB / HTML tooling if you extend the stack; they are not required for the research route. **Bundle tradeoff:** the light worker stays smaller without CrewAI/Chroma, but we keep hypothesis-capable deps in the default export so the route works in production.
- BigQuery: use `GCP_CREDENTIALS_BASE64`; the app writes it to `/tmp` on startup when `VERCEL` is set.
- **PoC / PoT Word (`/api/generate-poc-document`):** needs **BigQuery**, **`ANTHROPIC_API_KEY`** (or OpenAI) for LLM fill-in, and the three template files **`apps/api/templates/poc_pot/{poc_saas,poc_vpc,pot}.docx`** present in the deployment bundle. Check **`GET /health`** → **`poc_pot_workflow`** on the API for `ready` and `word_templates_present`. Commit the `.docx` masters if they are missing from git.

---

## 3. Project: **id-pain-api-crew** (FastAPI **Crew** worker)

Same **`main:app`** as **`apps/api`**, but:

- **`vercel.json` `installCommand`** runs **`bash vercel_install.sh`** (under the 256-character limit): copy **`../api` → `_api_src/`** (drops **`hypothesis_tool`**, **`tests`**, **`frontend`**, caches), **`pip install -r requirements.txt`**, uninstall **BigQuery** wheels, then **`vercel_prune_site_packages.py`** with **`PRUNE_VERCEL_CREW_WORKER=1`** (sympy/kubernetes/uv CLI, ONNX training trees, every **`tests/`** and **`__pycache__/`** under `site-packages`, etc.).
- Runtime sets **`API_SERVICE_MODE=crew`** via `apps/api-crew/api/index.py` (`os.environ.setdefault`), so **BigQuery is skipped** and only Gong + Crew routes matter for that worker.

**250 MiB cap:** CrewAI + Chroma + ONNX Runtime + gRPC OTLP (via **`arize-otel`**) can still exceed Vercel’s uncompressed function limit on Linux even after pruning (the install script runs **`pip uninstall`** for **`uv`**, **`ty`**, **`kubernetes`**, **`uvloop`** and **`vercel_prune_site_packages.py`** removes **`sys.prefix/bin/uv`** and **`ty`**, which Vercel’s size report previously counted as tens of megabytes each). If production deploys keep failing, use **`apps/api-crew-proxy`** on Vercel (**`CREW_BACKEND_URL`** → full worker) or host **`id-pain-api-crew`** entirely on **Railway / Fly / Cloud Run** using the repo **`Dockerfile`**, and set **`NEXT_PUBLIC_CREW_API_URL`** to the proxy or container URL.

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

## 3b. Project: **id-pain-api-crew-proxy** (optional thin edge)

When **`id-pain-api-crew`** cannot fit under **250 MiB**, deploy the real Crew stack from the **`Dockerfile`** (or another image) on a container host, then:

1. **Add** Vercel project **`id-pain-api-crew-proxy`** with **Root Directory** `apps/api-crew-proxy` (preset **Other**; `vercel.json` rewrites `/(.*)` → `api/index.py`).
2. Set **`CREW_BACKEND_URL`** (production) to the container’s **HTTPS origin** with **no** trailing slash, e.g. `https://id-pain-api-crew-xxxx.up.railway.app`.
3. Set **`NEXT_PUBLIC_CREW_API_URL`** on **`id-pain-web`** to the **proxy** production URL (not the container URL), so the browser still talks to Vercel while the proxy streams to the worker.

The proxy forwards method, path, query, headers (minus hop-by-hop), and request body; responses stream back (SSE and large downloads supported).

---

## 4. Project: **id-pain-web** (Next.js)

### Create

1. **Vercel → Add New… → Project** → import the same repo.
2. **Root Directory:** `apps/web`.
3. **Framework Preset:** Next.js (auto-detected).
4. **Environment variables:**
   - **`NEXT_PUBLIC_LEGACY_API_URL`** — production origin of the FastAPI worker (no trailing slash), e.g. **`https://arize-gtm-stillness-api.vercel.app`**. It is read at **build time** by **`apps/web/middleware.ts`** (primary) and **`next.config.ts` fallback rewrites** (backup): same-origin **`/api/:path*`** is proxied to that host so the browser avoids CORS. **`app/api/`** routes (e.g. **`/api/health`**) stay on Next and are not proxied.
   - **`NEXT_PUBLIC_CREW_API_URL`** — optional second origin for analyze / recap / prospect routes (no trailing slash). When unset, those routes use the same rewrite target as the main API (same-origin `/api/...` → FastAPI unless you set a distinct crew URL).

### CLI alternative

```bash
cd apps/web
vercel link
vercel env add NEXT_PUBLIC_LEGACY_API_URL production    # e.g. https://arize-gtm-stillness-api.vercel.app (used at build for rewrites)
vercel env add NEXT_PUBLIC_CREW_API_URL production      # optional second worker; omit if analyze runs on LEGACY host
vercel deploy --prod
```

---

## 5. Wiring summary

```
[ Browser ]
     │
     ▼
[ id-pain-web (Vercel) ]   ← apps/web
     │  Browser → same-origin /api/* ; rewrites → NEXT_PUBLIC_LEGACY_API_URL (FastAPI)
     │  NEXT_PUBLIC_CREW_API_URL    →  optional direct origin for crew-only paths
     ▼
[ id-pain-api (Vercel) ]   ← apps/api   (API_SERVICE_MODE=light)
[ id-pain-api-crew ]       ← apps/api-crew (API_SERVICE_MODE=crew, _api_src copy)
     │  (optional) id-pain-api-crew-proxy → CREW_BACKEND_URL (container crew worker)
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
