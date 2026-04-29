# id-pain-web (Next.js)

Migration shell for Vercel / v0. The FastAPI + legacy UI stay in the parent `id-pain` project.

## Local (host Node)

```bash
cd web && npm install && npm run dev
```

Open http://localhost:3000. Set `NEXT_PUBLIC_LEGACY_API_URL` if the API is not on `http://localhost:8080`.

## Docker Compose

`docker compose up` starts FastAPI (legacy UI on http://localhost:8080), LiteLLM, Gong MCP, and this Next app on http://localhost:3000.
