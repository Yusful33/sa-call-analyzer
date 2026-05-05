# id-pain-web (Next.js)

Migration shell for Vercel / v0. The FastAPI + legacy UI live in **`../api/`** (same monorepo).

## Local (host Node)

From **repository root**:

```bash
cd apps/web && npm install && npm run dev
```

Open http://localhost:3000. In **development** the UI calls `http://localhost:8080` unless `NEXT_PUBLIC_LEGACY_API_URL` is set. In **production**, the browser calls **same-origin** `/api/...`; `next.config.ts` **fallback rewrites** proxy those to the FastAPI host from `NEXT_PUBLIC_LEGACY_API_URL` (or the default `https://arize-gtm-stillness-api.vercel.app` at build time). Routes defined under `app/api/` in this app (e.g. `/api/health`) are not rewritten.

## Docker Compose

From **repository root**:

```bash
docker compose -f infra/docker-compose.yml up
```

This starts FastAPI (legacy UI on http://localhost:8080), LiteLLM, Gong MCP, and this Next app on http://localhost:3000.

## Vercel

See **[`docs/deploy-vercel.md`](../../docs/deploy-vercel.md)** — set project **Root Directory** to `apps/web`.
