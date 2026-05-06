"""
Canonical Arize documentation URLs (docs.arize.com) and runtime availability checks.

PoC/PoT Word templates reference these; `validate_doc_links_sync` is used by /api/doc-links/check.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import httpx

logger = logging.getLogger("sa-call-analyzer")

ARIZE_DOC_REFERENCE_URLS: dict[str, str] = {
    "tracing": "https://docs.arize.com/ax/observe/tracing/",
    "production_monitoring": "https://docs.arize.com/ax/observe/production-monitoring/",
    "dashboards": "https://docs.arize.com/ax/observe/dashboards",
    "datasets": "https://docs.arize.com/ax/develop/datasets",
    "datasets_and_experiments": "https://docs.arize.com/ax/develop/datasets-and-experiments",
    "evaluators": "https://docs.arize.com/ax/evaluate/create-evaluators",
    "session_evaluations": "https://docs.arize.com/ax/evaluate/evaluators/trace-and-session-evals/session-level-evaluations",
    "ml_observability": "https://docs.arize.com/ax/machine-learning/machine-learning/what-is-ml-observability",
    "drift_tracing": "https://docs.arize.com/ax/machine-learning/machine-learning/how-to-ml/drift-tracing",
    "data_quality": "https://docs.arize.com/ax/machine-learning/machine-learning/how-to-ml/data-quality-troubleshooting",
    "custom_metrics": "https://docs.arize.com/ax/machine-learning/machine-learning/how-to-ml/custom-metrics-api",
    "prompt_playground": "https://docs.arize.com/ax/prompts/prompt-playground/",
    "compliance": "https://docs.arize.com/ax/security-and-settings/compliance",
    "sso_rbac": "https://docs.arize.com/ax/security-and-settings/sso-and-rbac",
    "self_hosting": "https://docs.arize.com/ax/selfhosting/architecture",
    "alyx": "https://docs.arize.com/ax/alyx",
    "ai_assistants_setup": "https://docs.arize.com/ax/set-up-with-ai-assistants",
}


def validate_doc_links_sync(
    timeout_seconds: float = 12.0,
    max_workers: int = 8,
) -> dict[str, Any]:
    """
    HEAD each canonical URL with redirect follow. Returns summary for health checks.
    """
    results: dict[str, dict[str, Any]] = {}
    broken: list[str] = []

    def check_one(name: str, url: str) -> tuple[str, bool, str]:
        try:
            with httpx.Client(timeout=timeout_seconds, follow_redirects=True) as client:
                r = client.head(url)
                if r.status_code >= 400:
                    r = client.get(url)
                ok = r.status_code < 400
                fin = str(r.url)
                return name, ok, fin
        except Exception as e:
            logger.debug("doc link check failed for %s: %s", name, e)
            return name, False, str(e)

    items = list(ARIZE_DOC_REFERENCE_URLS.items())
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(check_one, n, u): n for n, u in items}
        for fut in as_completed(futures):
            name, ok, extra = fut.result()
            entry: dict[str, Any] = {"ok": ok, "url": ARIZE_DOC_REFERENCE_URLS[name]}
            if ok and extra.startswith("http"):
                entry["resolved_url"] = extra
            elif not ok:
                entry["error"] = extra
            results[name] = entry
            if not ok:
                broken.append(name)

    return {
        "status": "healthy" if not broken else "degraded",
        "total": len(items),
        "broken_count": len(broken),
        "broken": sorted(broken),
        "links": results,
    }
