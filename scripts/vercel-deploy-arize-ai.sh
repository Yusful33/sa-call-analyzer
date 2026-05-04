#!/usr/bin/env bash
# Deploy apps/web and apps/api to Vercel under the Arize AI team.
# Requires: interactive `vercel login` (SAML) or VERCEL_TOKEN with arize-ai access.
set -euo pipefail
SCOPE="${VERCEL_SCOPE:-arize-ai}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

vercel project add arize-gtm-stillness --scope "$SCOPE" 2>/dev/null || true
vercel project add arize-gtm-stillness-api --scope "$SCOPE" 2>/dev/null || true

cd "$ROOT/apps/web"
vercel link --yes --scope "$SCOPE" --project arize-gtm-stillness
vercel deploy --prod --yes

cd "$ROOT/apps/api"
vercel link --yes --scope "$SCOPE" --project arize-gtm-stillness-api
vercel deploy --prod --yes

echo "Done. Open Vercel → team Arize AI → projects arize-gtm-stillness / arize-gtm-stillness-api"
