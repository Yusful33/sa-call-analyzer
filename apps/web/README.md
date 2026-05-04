# id-pain-web (Next.js)

Migration shell for Vercel / v0. The FastAPI + legacy UI live in **`../api/`** (same monorepo).

## Local (host Node)

From **repository root**:

```bash
cd apps/web && npm install && npm run dev
```

Open http://localhost:3000. In development the UI calls `http://localhost:8080` unless `NEXT_PUBLIC_LEGACY_API_URL` is set. Production builds default to `https://stillness.vercel.app`; override with `NEXT_PUBLIC_LEGACY_API_URL` if your API lives elsewhere.

## Docker Compose

From **repository root**:

```bash
docker compose -f infra/docker-compose.yml up
```

This starts FastAPI (legacy UI on http://localhost:8080), LiteLLM, Gong MCP, and this Next app on http://localhost:3000.

## Vercel

See **[`docs/deploy-vercel.md`](../../docs/deploy-vercel.md)** — set project **Root Directory** to `apps/web`.
