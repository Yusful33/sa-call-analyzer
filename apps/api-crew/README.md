# id-pain-api-crew (Vercel)

Second Python project for **CrewAI-heavy** routes only (`/api/analyze`, `/api/generate-recap-slide`, `/api/analyze-prospect`, `/api/analyze-prospect-stream`). Shared application code lives in **`../api`**; Vercel’s root is only this folder, so **`vercel.json` `installCommand`** runs **`vercel_install.sh`** (Vercel’s 256-character limit) to copy **`../api` → `_api_src/`** (drops **`hypothesis_tool`**, **`tests`**, **`__pycache__`**), runs **`pip install`**, uninstalls **BigQuery** wheels (unused when `API_SERVICE_MODE=crew`), then runs **`../api/scripts/vercel_prune_site_packages.py`** with **`PRUNE_VERCEL_CREW_WORKER=1`** to trim ONNX tooling trees and other bulk. If deploys still exceed **250 MiB**, enable **`VERCEL_ANALYZE_BUILD_OUTPUT=1`** on this project for a size report, or run the crew image on **Railway / Docker**. Locally you can `cp -R ../api _api_src` once or run the API from **`apps/api`** with `API_SERVICE_MODE=crew` instead.

## Vercel

1. New project → same Git repo → **Root Directory:** `apps/api-crew`.
2. Copy environment variables from **`apps/api`**, especially `ANTHROPIC_API_KEY`, `GONG_MCP_URL`, and Arize keys.
3. Set **`NEXT_PUBLIC_CREW_API_URL`** on **`apps/web`** to this project’s production URL.

## Regenerate `requirements.txt`

From **`apps/api`**:

```bash
uv export --no-hashes --no-dev --format requirements-txt --no-emit-project --no-annotate --extra crew -o ../api-crew/requirements.txt
```
