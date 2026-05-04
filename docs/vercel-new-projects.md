# Vercel projects: `arize-gtm-stillness` (web) + `arize-gtm-stillness-api` (API)

This repo was re-linked from **`stillness`** / **`stillness-api`** to new projects. An earlier pass used the personal team **`ycattaneo-5285s-projects`**. **Production for Arize should live under the `arize-ai` team** (see below); the CLI cannot complete **SAML** login for that team in a non-interactive environment, so you must run the commands locally after `vercel login`.

## Production URLs

| App | Vercel project | Default production hostname |
|-----|----------------|------------------------------|
| Next.js (`apps/web`) | **arize-gtm-stillness** | `https://arize-gtm-stillness.vercel.app` |
| FastAPI (`apps/api`) | **arize-gtm-stillness-api** | `https://arize-gtm-stillness-api.vercel.app` |

### If `*.vercel.app` shows `DEPLOYMENT_NOT_FOUND`

After creating or renaming a project, point the hostname at the latest **Ready** production deployment:

```bash
cd apps/web
vercel ls arize-gtm-stillness   # copy the newest Production deployment URL
vercel alias set "https://<deployment-hostname>" arize-gtm-stillness.vercel.app
```

### If you see **401** / Vercel login on the deployment URL

Turn off **Deployment Protection** for **Production** (or allow public access) under **Project → Settings → Deployment Protection**, or authenticate in the browser when testing.

## Environment variables

`vercel env list production` reported **no user-defined** variables on the previous **`stillness`** and **`stillness-api`** projects (only values Vercel injects at build time). If you stored secrets in the **dashboard** under another team, a different environment (Preview), or **Shared Environment Variables**, copy them manually:

1. Old project → **Settings → Environment Variables** → export or copy each name/value.
2. New project → same → **Add** for **Production** (and Preview if needed).

CLI checks:

```bash
cd apps/web && vercel env list production
cd apps/api && vercel env list production
```

To download a local `.env` file from the **currently linked** project (may include short-lived system keys — do not commit):

```bash
cd apps/web && vercel env pull .env.vercel.production.web --environment production --yes
cd apps/api && vercel env pull .env.vercel.production.api --environment production --yes
```

## Old project link backups (local only)

If you need to compare or re-link to the old projects, JSON backups of the previous `.vercel/project.json` files can live under **`.vercel-env-backup/`** on your machine (gitignored). Example previous names: **`stillness`**, **`stillness-api`**.

## Git integration

In Vercel, **disconnect** the Git integration from the old projects or archive them, then **connect** **`arize-gtm-stillness`** and **`arize-gtm-stillness-api`** to the same GitHub repo with root directories **`apps/web`** and **`apps/api`**, so pushes deploy the new projects.

For each project, open **Settings → General → Root Directory** and set **`apps/web`** or **`apps/api`** (not the monorepo root). Set the **Framework Preset** to **Next.js** for the web app and **Other** for the API if the dashboard does not auto-detect correctly.

## Deploy under **Arize AI** (`arize-ai` scope)

Run these in your **own terminal** (browser / device flow so SAML can finish):

```bash
# 1) Authenticate (complete SAML in the browser when prompted)
vercel login

# 2) Create projects on the Arize team (skip if projects already exist)
vercel project add arize-gtm-stillness --scope arize-ai
vercel project add arize-gtm-stillness-api --scope arize-ai

# 3) Link this monorepo
cd /path/to/sa-call-analyzer/apps/web
vercel link --yes --scope arize-ai --project arize-gtm-stillness

cd /path/to/sa-call-analyzer/apps/api
vercel link --yes --scope arize-ai --project arize-gtm-stillness-api

# 4) Production deploy
cd /path/to/sa-call-analyzer/apps/web && vercel deploy --prod --yes
cd /path/to/sa-call-analyzer/apps/api  && vercel deploy --prod --yes
```

Then in the **Arize AI** Vercel dashboard: set **Root Directory** to `apps/web` / `apps/api`, connect **Git**, copy **environment variables**, and adjust **Deployment Protection** if the site should be public.

**CI / automation:** create a token at [vercel.com/account/tokens](https://vercel.com/account/tokens) with access to the **Arize AI** team and run commands with `vercel ... --token "$VERCEL_TOKEN" --scope arize-ai`.

---

## Linking this repo locally (personal team `ycattaneo-5285s-projects`)

Only if you intentionally keep hobby-team deployments:

```bash
cd apps/web && vercel link --yes --scope team_IAApnRALp1SX6qOMpAGrXFz4 --project arize-gtm-stillness
cd apps/api  && vercel link --yes --scope team_IAApnRALp1SX6qOMpAGrXFz4 --project arize-gtm-stillness-api
```

Use `vercel teams ls` if your scope slug differs.
