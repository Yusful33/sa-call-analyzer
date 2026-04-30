# gong-mcp (Vercel Functions)

Direct **Gong API** bridge as Vercel Node Functions — replaces the legacy `infra/gong-http-server` (Python wrapper around an MCP TS subprocess).

## Endpoints

| Path | Body | What it does |
|------|------|---------------|
| `GET /api/health` | — | Liveness check |
| `POST /api/calls` | `{ from_date?, to_date? }` (ISO 8601) | List calls (paginated) with extended metadata |
| `POST /api/transcript` | `{ call_id }` or `{ callIds: [...] }` | Retrieve transcript(s) |
| `POST /api/call-info` | `{ call_id }` | Detailed call metadata (title, scheduled, duration, parties) |

These match the **legacy** routes (`/health`, `/calls`, `/transcript`, `/call-info`) so `apps/api` keeps working with `GONG_MCP_URL=https://gong-mcp.your-domain.com`.

## Environment variables

| Name | Required | Purpose |
|------|----------|---------|
| `GONG_ACCESS_KEY` | yes | Gong API access key |
| `GONG_ACCESS_SECRET` (or `GONG_SECRET_KEY`) | yes | Gong API secret |

## Deploy on Vercel

1. **Vercel → Add New… → Project** → import this repo.
2. **Root Directory:** `apps/gong-mcp`.
3. **Framework preset:** **Other** (Vercel auto-detects the `api/` folder as Functions).
4. Add the env vars above (Production + Preview).
5. Deploy. Your URL becomes the new `GONG_MCP_URL` for the API project.

## Local

```bash
cd apps/gong-mcp
npm install
npx vercel dev
```
