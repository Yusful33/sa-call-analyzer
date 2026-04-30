# Deploying this monorepo (Vercel and friends)

The repo is split into deployable **slices** under `apps/` and `infra/`. [Vercel](https://vercel.com) is ideal for the **Next.js** UI; long‑running Python APIs, LLM proxies, and Gong bridges usually belong on a **container** or **VM** host.

## Layout (what goes where)

| Path | Role | Typical host |
|------|------|----------------|
| `apps/web/` | Next.js 15 UI | **Vercel** (recommended) |
| `apps/api/` | FastAPI (CrewAI, BigQuery, legacy HTML, OpenAPI) | **Railway / Fly.io / Render / AWS ECS / EKS** (see `Dockerfile` + `DEPLOYMENT.md`) |
| `infra/litellm/` | LiteLLM proxy config (+ optional custom image) | Same as API or a small always‑on service (not a great fit for serverless) |
| `infra/gong-http-server/` | Gong MCP HTTP bridge | Same as API (must be reachable from `apps/api`) |

---

## 1. Vercel project: **Next.js** (`apps/web`)

1. In Vercel: **Add New → Project** → import this Git repository.
2. Under **Root Directory**, set **`apps/web`** (or “Edit” after import and change monorepo root).
3. **Framework Preset:** Next.js (auto-detected from `package.json`).
4. **Build & Output:** defaults (`npm run build`, output `.next`).
5. **Environment variables** (Production + Preview as needed):

   | Name | Example | Purpose |
   |------|---------|---------|
   | `NEXT_PUBLIC_LEGACY_API_URL` | `https://api.your-domain.com` | Browser calls your FastAPI backend (no trailing slash). |

6. Deploy. Your production URL is the Next app; it proxies API traffic to `NEXT_PUBLIC_LEGACY_API_URL`.

**CORS:** FastAPI already allows `*` in dev; for production, narrow `allow_origins` in `apps/api/main.py` to your Vercel domain.

---

## 2. Should FastAPI go on Vercel?

Vercel’s **Python** support targets **short‑lived** request/response functions. This API runs **long analyses**, **CrewAI**, **streaming**, and **BigQuery** — often **beyond** typical serverless timeouts and bundle limits.

**Practical approach:**

- Host **`apps/api`** on a platform that runs **`docker compose`** or the root **`Dockerfile`** (same image as today), e.g. **Railway**, **Fly.io**, **Render**, **Google Cloud Run** (with a high timeout), or **EKS** (see `DEPLOYMENT.md`).
- Point Vercel’s `NEXT_PUBLIC_LEGACY_API_URL` at that HTTPS URL.

If you still want an experiment on Vercel Python, consult Vercel’s current **Python / FastAPI** docs for route handlers, timeouts, and file-size limits — expect to trim features.

---

## 3. Separate Vercel projects (multiple frontends)

If you later add another Next app under `apps/`, create **another Vercel project** with its own root directory (e.g. `apps/other-ui`). Same repo, multiple projects — each with its own env vars and domain.

---

## 4. LiteLLM and Gong as “separate projects” on Vercel

**Usually no.** LiteLLM and the Gong MCP HTTP server expect a **steady TCP listener** and are built for **containers**, not Edge/serverless routes.

Deploy them beside the API instead:

- **Docker Compose:** from repo root, `docker compose -f infra/docker-compose.yml up` (dev).
- **Production:** duplicate that topology on your host (three services: api, litellm, gong-mcp) or fold LiteLLM + Gong into Kubernetes manifests under `infra/k8s/` (see `DEPLOYMENT.md`).

---

## 5. Checklist after moving folders

- Local API: `cd apps/api && uv sync && uv run python main.py` (`.env` at **repository root** is still loaded).
- Local stack: `docker compose -f infra/docker-compose.yml up` from repo root.
- Docker image for API: `docker build -t id-pain:latest .` from repo root (uses root `Dockerfile` + `apps/api/`).
