# id-pain-web (Next.js)

Migration shell for Vercel / v0. The FastAPI + legacy UI live in **`../api/`** (same monorepo).

## Local (host Node)

From **repository root**:

```bash
cd apps/web && npm install && npm run dev
```

Open http://localhost:3000. Set `NEXT_PUBLIC_LEGACY_API_URL` if the API is not on `http://localhost:8080`.

## Docker Compose

From **repository root**:

```bash
docker compose -f infra/docker-compose.yml up
```

This starts FastAPI (legacy UI on http://localhost:8080), LiteLLM, Gong MCP, and this Next app on http://localhost:3000.

## Vercel

See **[`docs/deploy-vercel.md`](../../docs/deploy-vercel.md)** — set project **Root Directory** to `apps/web`.
