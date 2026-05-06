"""Health check and diagnostic routes."""

import os
from pathlib import Path
from fastapi import APIRouter
from fastapi.responses import RedirectResponse, FileResponse, HTMLResponse

router = APIRouter()

BASE_DIR = Path(__file__).resolve().parent.parent


def get_health_status(bq_client, api_service_mode: str) -> dict:
    """Build health status response with dependency checks."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    poc_dirs = [BASE_DIR / "api" / "templates" / "poc_pot", BASE_DIR / "templates" / "poc_pot"]
    poc_templates = {}
    for name in ("poc_saas", "poc_vpc", "pot"):
        poc_templates[name] = any((d / f"{name}.docx").is_file() for d in poc_dirs)
    poc_ready = bool(bq_client) and all(poc_templates.values()) and bool(
        api_key or os.getenv("OPENAI_API_KEY")
    )
    return {
        "status": "healthy",
        "api_key_configured": bool(api_key),
        "service_mode": api_service_mode,
        "canonical_web_url": (os.getenv("STILLNESS_WEB_URL") or os.getenv("PUBLIC_WEB_APP_URL") or "").strip()
        or None,
        "poc_pot_workflow": {
            "ready": poc_ready,
            "bigquery_available": bq_client is not None,
            "word_templates_present": poc_templates,
            "llm_configured": bool(api_key or os.getenv("OPENAI_API_KEY")),
        },
    }


@router.get("/")
async def root():
    """Serve the legacy HTML UI, or redirect to the Next.js app when configured."""
    web_app_url = (os.getenv("STILLNESS_WEB_URL") or os.getenv("PUBLIC_WEB_APP_URL") or "").strip().rstrip("/")
    if web_app_url:
        return RedirectResponse(web_app_url, status_code=302)
    try:
        response = FileResponse(str(BASE_DIR / "frontend" / "index.html"))
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response
    except FileNotFoundError:
        return HTMLResponse("""
        <html>
            <body>
                <h1>Call Analyzer API</h1>
                <p>Frontend not found. API is running at <a href="/docs">/docs</a></p>
            </body>
        </html>
        """)


@router.get("/api/example")
async def get_example_transcript():
    """Get an example transcript for testing."""
    example = """0:16 | Hakan
yeah, they're so wealthy.

0:17 | Juan
Yeah.

2:34 | Anh
Yeah, we don't have any technical questions. So, I think we want to hear more about, you know, if we want to do a POC, how we want to proceed with that.

2:46 | Juan
Okay, perfect. Yeah. So the POC essentially would have our team guiding you through the platform."""
    return {"transcript": example}
