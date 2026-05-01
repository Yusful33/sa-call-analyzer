#!/usr/bin/env bash
# Vercel installCommand must stay under 256 chars — keep logic here.
set -euo pipefail
export PIP_NO_CACHE_DIR=1
rm -rf _api_src
cp -R ../api _api_src
rm -rf _api_src/hypothesis_tool _api_src/tests _api_src/.pytest_cache
find _api_src -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
python -m pip install -r requirements.txt
python -m pip uninstall -y google-cloud-bigquery google-cloud-core google-crc32c google-resumable-media 2>/dev/null || true
PRUNE_VERCEL_CREW_WORKER=1 python ../api/scripts/vercel_prune_site_packages.py
