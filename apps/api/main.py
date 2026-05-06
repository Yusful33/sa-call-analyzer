import os
import json
import time
import base64
import asyncio
import logging
import contextvars
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sa-call-analyzer")

# Get the directory where this file is located (apps/api/)
BASE_DIR = Path(__file__).resolve().parent
# Repository root — for .env and docs that stay at monorepo root
REPO_ROOT = BASE_DIR.parent.parent

# ============================================================
# GCP Credentials Setup (for Railway deployment)
# Decode base64 credentials if GCP_CREDENTIALS_BASE64 is set
# ============================================================
def setup_gcp_credentials():
    """Decode GCP credentials from base64 environment variable."""
    gcp_creds_b64 = os.getenv("GCP_CREDENTIALS_BASE64")
    
    logger.info("GCP Credentials Setup: %s", "SET" if gcp_creds_b64 else "NOT SET")
    
    if gcp_creds_b64:
        try:
            # Decode the base64 credentials
            creds_json = base64.b64decode(gcp_creds_b64).decode('utf-8')

            # Vercel serverless disks are typically read-only outside TMPDIR/RAM.
            creds_root = Path(os.getenv("TMPDIR", "/tmp")) if os.getenv("VERCEL") else BASE_DIR
            creds_path = creds_root / "gcp-credentials.json"
            creds_path.parent.mkdir(parents=True, exist_ok=True)

            # Write decoded key for google-auth credential discovery
            with open(creds_path, 'w') as f:
                f.write(creds_json)
            
            # Set the environment variable to point to the file
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(creds_path)
            
            # Verify the JSON is valid
            creds_dict = json.loads(creds_json)
            logger.info("GCP credentials decoded (type: %s)", creds_dict.get('type', 'unknown'))
            
            return True
        except Exception as e:
            logger.error("Failed to decode GCP credentials: %s", e)
            return False
    else:
        # Check if GOOGLE_APPLICATION_CREDENTIALS is already set
        existing_creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if existing_creds and Path(existing_creds).exists():
            logger.info("Using existing GCP credentials")
            return True
        else:
            logger.warning("No GCP credentials available")
            return False

# Run credential setup before anything else
setup_gcp_credentials()

# Use environment variables directly (not .env file)
# This allows uv/venv to manage environment variables
# The .env file is kept for reference/documentation only

# Verify environment variables are set
arize_space_id_env = os.getenv("ARIZE_SPACE_ID")
arize_api_key_env = os.getenv("ARIZE_API_KEY")

logger.info("Environment check: ARIZE_API_KEY=%s, ARIZE_SPACE_ID=%s",
            "SET" if arize_api_key_env else "NOT SET",
            "SET" if arize_space_id_env else "NOT SET")

# Initialize observability before importing analysis modules
from observability import (
    setup_observability,
    get_tracer,
    api_span,
    SPAN_PREFIX,
    get_component_for_path,
    get_provider_for_component,
    set_request_tracer_provider,
    get_current_project_name,
    force_flush_current_request,
    COMPONENT_HYPOTHESIS,
    PROJECT_HYPOTHESIS,
)
tracer_provider = setup_observability(project_name="sa-call-analyzer")

# Now import everything else
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from models import (
    ProspectOverviewRequest,
    ProspectOverview,
    GeneratePocDocumentRequest,
    TransitionToCSRequest,
    TransitionToCSResponse,
    AccountSuggestionsRequest,
    AccountSuggestionsResponse,
)
from poc_document_generator import AppendixGenerationError, build_poc_document
from transition_document_generator import build_transition_document
from gong_mcp_client import GongMCPClient
from arize_doc_links import validate_doc_links_sync
from prospect_intel import build_prospect_intelligence_bundle
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from pydantic import BaseModel
from typing import Optional
from threading import Lock

# ============================================================
# Prospect Overview Cache (TTL-based in-memory cache)
# Reduces redundant BigQuery calls when users click multiple
# capabilities for the same account.
# ============================================================
_PROSPECT_CACHE_TTL = int(os.getenv("PROSPECT_CACHE_TTL_SECONDS", "900"))  # 15 min default
_PROSPECT_CACHE_MAX_SIZE = int(os.getenv("PROSPECT_CACHE_MAX_SIZE", "100"))

class _ProspectCache:
    """Simple TTL cache for ProspectOverview objects."""
    
    def __init__(self, max_size: int = 100, ttl_seconds: int = 900):
        self._cache: dict[str, tuple[float, "ProspectOverview"]] = {}
        self._lock = Lock()
        self._max_size = max_size
        self._ttl = ttl_seconds
    
    def _make_key(self, account_name: str | None, domain: str | None, sfdc_id: str | None) -> str:
        return f"{(account_name or '').lower().strip()}|{(domain or '').lower().strip()}|{(sfdc_id or '').strip()}"
    
    def get(self, account_name: str | None, domain: str | None, sfdc_id: str | None) -> Optional["ProspectOverview"]:
        key = self._make_key(account_name, domain, sfdc_id)
        with self._lock:
            if key in self._cache:
                ts, overview = self._cache[key]
                if time.time() - ts < self._ttl:
                    return overview
                del self._cache[key]
        return None
    
    def set(self, account_name: str | None, domain: str | None, sfdc_id: str | None, overview: "ProspectOverview") -> None:
        key = self._make_key(account_name, domain, sfdc_id)
        with self._lock:
            if len(self._cache) >= self._max_size:
                oldest_key = min(self._cache, key=lambda k: self._cache[k][0])
                del self._cache[oldest_key]
            self._cache[key] = (time.time(), overview)
    
    def invalidate(self, account_name: str | None = None, domain: str | None = None, sfdc_id: str | None = None) -> None:
        if account_name or domain or sfdc_id:
            key = self._make_key(account_name, domain, sfdc_id)
            with self._lock:
                self._cache.pop(key, None)
        else:
            with self._lock:
                self._cache.clear()

_prospect_cache = _ProspectCache(max_size=_PROSPECT_CACHE_MAX_SIZE, ttl_seconds=_PROSPECT_CACHE_TTL)

# Initialize FastAPI app
app = FastAPI(
    title="Call Analyzer - LangGraph Multi-Stage Analysis",
    description="Analyze sales call performance using Command of the Message framework with specialized AI agents"
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    # So browsers can read Content-Disposition on cross-origin blob downloads (e.g. Next :3000 → API :8080)
    expose_headers=["Content-Disposition"],
)


@app.middleware("http")
async def arize_project_middleware(request, call_next):
    """Set the active Arize tracer provider by component so each of the 4 components writes to its own project."""
    if request.url.path.startswith("/api/"):
        component = get_component_for_path(request.url.path)
        provider = get_provider_for_component(component)
        if provider is not None:
            set_request_tracer_provider(provider)
    return await call_next(request)


# Initialize tracer for main API
tracer = trace.get_tracer("sa-call-analyzer-api")

# Verify tracer provider after core imports
current_provider = trace.get_tracer_provider()
logger.debug("Tracer provider: %s", type(current_provider).__name__)
if tracer_provider:
    if current_provider != tracer_provider:
        logger.warning("Tracer provider was overridden: expected %s, got %s",
                       type(tracer_provider).__name__, type(current_provider).__name__)

# Initialize Gong MCP client
try:
    gong_client = GongMCPClient()
    logger.info("Gong MCP client initialized")
except Exception as e:
    gong_client = None
    logger.warning("Gong MCP client not available: %s", e)

_API_SERVICE_MODE = os.environ.get("API_SERVICE_MODE", "full").lower()

# Initialize BigQuery (not needed on crew-only workers)
bq_client = None
if _API_SERVICE_MODE in ("full", "light"):
    try:
        from bigquery_client import BigQueryClient

        bq_client = BigQueryClient()
        logger.info("BigQuery client initialized")
    except Exception as e:
        bq_client = None
        logger.warning("BigQuery client not available: %s", e)
else:
    logger.info("Skipping BigQuery init (API_SERVICE_MODE=crew)")


def _with_optional_gong_mcp_enrichment(overview: ProspectOverview) -> ProspectOverview:
    """When warehouse Gong context is thin, supplement from Gong MCP (GONG_MCP_URL)."""
    from gong_mcp_enrichment import maybe_enrich_overview_with_gong_mcp

    return maybe_enrich_overview_with_gong_mcp(overview, gong_client, bq_client)


if _API_SERVICE_MODE in ("full", "light", "crew"):
    from routes_crew import register_crew_routes

    register_crew_routes(app, gong_client=gong_client, tracer=tracer)
    logger.info("Call analysis routes registered")
else:
    logger.info("API_SERVICE_MODE=%s - call analysis routes not loaded", _API_SERVICE_MODE)

# Register Slack bot routes if configured
if os.getenv("SLACK_BOT_TOKEN"):
    try:
        from slack_bot import SlackBot
        from routes.slack import configure_slack_routes

        _slack_bot = SlackBot(bq_client=bq_client, gong_client=gong_client)
        slack_router = configure_slack_routes(_slack_bot)
        app.include_router(slack_router)
        logger.info("Slack bot routes registered")
    except ImportError as e:
        logger.warning("Slack dependencies not installed, skipping Slack routes: %s", e)
else:
    logger.info("SLACK_BOT_TOKEN not set, Slack routes not loaded")


@app.get("/")
async def root():
    """No bundled UI on the API — redirect to Stillness (Next) when configured."""
    web_app_url = (os.getenv("STILLNESS_WEB_URL") or os.getenv("PUBLIC_WEB_APP_URL") or "").strip().rstrip("/")
    if web_app_url:
        return RedirectResponse(web_app_url, status_code=302)
    return HTMLResponse(
        content="""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/><title>Stillness API</title></head>
<body style="font-family:system-ui,sans-serif;max-width:42rem;margin:2.5rem auto;padding:0 1.25rem;line-height:1.5;color:#1a1d29">
<h1 style="font-weight:600;font-size:1.5rem">Stillness API</h1>
<p>This host runs the <strong>FastAPI</strong> backend only. The unified web UI lives in <code>apps/web</code> (for example <code>http://localhost:3000</code> when developing).</p>
<ul>
<li><a href="/docs">OpenAPI docs</a> (<code>/docs</code>)</li>
<li><a href="/health">Health check</a> (<code>/health</code>)</li>
</ul>
<p style="font-size:0.95rem;color:#5a5f6e">Set <code>STILLNESS_WEB_URL</code> or <code>PUBLIC_WEB_APP_URL</code> to redirect this page to your deployed Next.js app.</p>
</body></html>""",
        status_code=200,
    )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    # Check both locations for templates (api/ folder for Vercel, root for local)
    poc_dirs = [BASE_DIR / "api" / "templates" / "poc_pot", BASE_DIR / "templates" / "poc_pot"]
    poc_templates = {}
    for name in ("poc_saas", "poc_vpc", "pot"):
        poc_templates[name] = any((d / f"{name}.docx").is_file() for d in poc_dirs)
    poc_ready = bool(bq_client) and all(poc_templates.values()) and bool(
        api_key or os.getenv("OPENAI_API_KEY")
    )
    brave_key = (os.getenv("BRAVE_API_KEY") or "").strip()
    return {
        "status": "healthy",
        "api_key_configured": bool(api_key),
        "service_mode": _API_SERVICE_MODE,
        "canonical_web_url": (os.getenv("STILLNESS_WEB_URL") or os.getenv("PUBLIC_WEB_APP_URL") or "").strip()
        or None,
        "poc_pot_workflow": {
            "ready": poc_ready,
            "bigquery_available": bq_client is not None,
            "word_templates_present": poc_templates,
            "llm_configured": bool(api_key or os.getenv("OPENAI_API_KEY")),
        },
        "hypothesis_research": {
            "brave_api_key_configured": bool(brave_key),
            "brave_api_key_length": len(brave_key),
        },
    }


@app.get("/api/debug/brave-search")
async def debug_brave_search(q: str = "Atlassian"):
    """One-shot Brave search probe for diagnosing deployment env issues.
    Returns the raw HTTP status and a snippet of the response body so we can
    see what Brave is actually responding with from the deployed environment.
    Does NOT leak the key — only reports its length."""
    import httpx
    brave_key = (os.getenv("BRAVE_API_KEY") or "").strip()
    if not brave_key:
        return {
            "key_present": False,
            "key_length": 0,
            "error": "BRAVE_API_KEY env var is empty or unset on this deployment",
        }
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                "https://api.search.brave.com/res/v1/web/search",
                headers={
                    "Accept": "application/json",
                    "X-Subscription-Token": brave_key,
                },
                params={"q": q, "count": 3},
            )
        body_text = resp.text[:600]
        result_count = 0
        try:
            data = resp.json()
            result_count = len(((data or {}).get("web") or {}).get("results") or [])
        except Exception:
            pass
        return {
            "key_present": True,
            "key_length": len(brave_key),
            "key_prefix": brave_key[:4],
            "http_status": resp.status_code,
            "result_count": result_count,
            "body_preview": body_text,
        }
    except Exception as e:
        return {
            "key_present": True,
            "key_length": len(brave_key),
            "error": f"{type(e).__name__}: {e}",
        }


@app.get("/api/example")
async def get_example_transcript():
    """Get an example transcript for testing"""
    with api_span("get_example_transcript"):
        example = """0:16 | Hakan
yeah, they're so wealthy.

0:17 | Juan
Yeah.

2:34 | Anh
Yeah, we don't have any technical questions. So, I think we want to hear more about, you know, if we want to do a POC, how we want to proceed with that.

2:46 | Juan
Okay, perfect. Yeah. So the POC essentially would have our team guiding you through the platform."""
        return {"transcript": example}


class CallsByAccountRequest(BaseModel):
    """Request to list calls for an account (lightweight, no analysis)"""
    account_name: str  # Account/company name to search for
    fuzzy_threshold: Optional[float] = 0.85  # Similarity threshold for name matching (0-1)
    from_date: Optional[str] = None  # ISO format start date (e.g., "2024-03-01T00:00:00Z")
    to_date: Optional[str] = None  # ISO format end date


class CallMetadata(BaseModel):
    """Metadata for a single call"""
    call_id: str
    title: Optional[str] = None
    scheduled: Optional[str] = None
    url: Optional[str] = None
    account_name: Optional[str] = None
    participants: Optional[list] = None


class CallsByAccountResponse(BaseModel):
    """Response containing list of calls for an account"""
    account_name_searched: str
    matched_account_names: list[str]
    total_calls: int
    calls: list[CallMetadata]


@app.post("/api/calls-by-account", response_model=CallsByAccountResponse)
async def get_calls_by_account(request: CallsByAccountRequest):
    """
    List all calls for an account/company name (lightweight, no analysis).

    This is a fast endpoint to discover what calls exist for an account
    before running the full analysis. Use /api/analyze-prospect to run
    the full CrewAI analysis on all matching calls.

    Uses fuzzy matching to handle variations like "Acme" vs "Acme Corp".
    """
    with tracer.start_as_current_span(
        f"{SPAN_PREFIX}.get_calls_by_account",
        attributes={
            "account.name": request.account_name,
            "fuzzy.threshold": request.fuzzy_threshold or 0.85,
            "openinference.span.kind": "TOOL",
        }
    ) as span:
        try:
            if not gong_client:
                span.set_status(Status(StatusCode.ERROR, "Gong MCP client not available"))
                raise HTTPException(
                    status_code=503,
                    detail="Gong MCP client not available. Check that Gong MCP server is running."
                )

            # Get matching calls using fuzzy matching
            matching_calls = gong_client.get_calls_by_prospect_name(
                prospect_name=request.account_name,
                from_date=request.from_date,
                to_date=request.to_date,
                fuzzy_threshold=request.fuzzy_threshold or 0.85
            )

            # Extract unique account names that matched
            matched_names = set()
            call_metadata_list = []

            for call in matching_calls:
                call_id = call.get("id")
                if not call_id:
                    continue

                # Get detailed call info for metadata
                try:
                    call_info = gong_client.get_call_info(call_id)

                    # Extract account name
                    account_name = call_info.get("accountName") or ""
                    if not account_name and call_info.get("account"):
                        account_name = call_info.get("account", {}).get("name", "")

                    if account_name:
                        matched_names.add(account_name)

                    # Extract participants
                    participants = []
                    for party in call_info.get("parties", []):
                        if isinstance(party, dict):
                            participants.append({
                                "name": party.get("name"),
                                "title": party.get("title"),
                                "email": party.get("emailAddress"),
                                "company": party.get("companyName")
                            })

                    call_metadata_list.append(CallMetadata(
                        call_id=call_id,
                        title=call_info.get("title") or call.get("title"),
                        scheduled=call_info.get("scheduled") or call.get("scheduled"),
                        url=call_info.get("url") or call.get("url"),
                        account_name=account_name,
                        participants=participants
                    ))
                except Exception as e:
                    # If we can't get call info, still include basic metadata
                    call_metadata_list.append(CallMetadata(
                        call_id=call_id,
                        title=call.get("title"),
                        scheduled=call.get("scheduled"),
                        url=call.get("url")
                    ))

            span.set_attribute("calls.total", len(call_metadata_list))
            span.set_attribute("accounts.matched", ", ".join(sorted(matched_names)))
            span.set_status(Status(StatusCode.OK))

            return CallsByAccountResponse(
                account_name_searched=request.account_name,
                matched_account_names=sorted(list(matched_names)),
                total_calls=len(call_metadata_list),
                calls=call_metadata_list
            )

        except HTTPException:
            raise
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to fetch calls for account: {str(e)}"
            )


@app.post("/api/account-suggestions", response_model=AccountSuggestionsResponse)
async def account_suggestions(request: AccountSuggestionsRequest):
    """
    Resolve a typed account/company name against Salesforce for disambiguation.

    Helps when spacing or punctuation differs from the CRM record
    (e.g. \"Alliance Bernstein\" vs \"AllianceBernstein\").
    """
    if not bq_client:
        raise HTTPException(
            status_code=503,
            detail="BigQuery client not available.",
        )
    data = bq_client.suggest_salesforce_account_names(
        account_name=request.account_name.strip(),
        domain=(request.domain or "").strip() or None,
    )
    return AccountSuggestionsResponse.model_validate(data)


class PrefetchProspectRequest(BaseModel):
    """Request to prefetch/warm the prospect cache."""
    account_name: Optional[str] = None
    domain: Optional[str] = None
    sfdc_account_id: Optional[str] = None


class PrefetchProspectResponse(BaseModel):
    """Response indicating prefetch status."""
    status: str
    cached: bool
    account_name: Optional[str] = None
    message: Optional[str] = None


@app.post("/api/prefetch-prospect", response_model=PrefetchProspectResponse)
async def prefetch_prospect(request: PrefetchProspectRequest):
    """
    Prefetch prospect overview data and warm the cache.
    
    Call this endpoint when the user starts typing an account name to
    pre-load data before they click "Generate PoC" or other capabilities.
    Returns immediately if data is already cached.
    
    This is a fire-and-forget optimization - errors are logged but don't
    fail the request.
    """
    if not bq_client:
        return PrefetchProspectResponse(
            status="skipped",
            cached=False,
            message="BigQuery client not available"
        )
    
    if not (request.account_name or request.domain or request.sfdc_account_id):
        return PrefetchProspectResponse(
            status="skipped",
            cached=False,
            message="No lookup criteria provided"
        )
    
    cached = _prospect_cache.get(
        request.account_name,
        request.domain,
        request.sfdc_account_id
    )
    if cached:
        return PrefetchProspectResponse(
            status="already_cached",
            cached=True,
            account_name=cached.salesforce.name if cached.salesforce else None
        )
    
    try:
        def _do_prefetch():
            overview = bq_client.get_prospect_overview(
                account_name=request.account_name,
                domain=request.domain,
                sfdc_account_id=request.sfdc_account_id,
            )
            overview = _with_optional_gong_mcp_enrichment(overview)
            _prospect_cache.set(
                request.account_name,
                request.domain,
                request.sfdc_account_id,
                overview
            )
            return overview
        
        overview = await asyncio.wait_for(
            asyncio.to_thread(_do_prefetch),
            timeout=60.0
        )
        return PrefetchProspectResponse(
            status="prefetched",
            cached=True,
            account_name=overview.salesforce.name if overview.salesforce else None
        )
    except asyncio.TimeoutError:
        return PrefetchProspectResponse(
            status="timeout",
            cached=False,
            message="Prefetch timed out (data may still be loading)"
        )
    except Exception as e:
        return PrefetchProspectResponse(
            status="error",
            cached=False,
            message=f"Prefetch failed: {str(e)[:200]}"
        )


@app.post("/api/prospect-overview", response_model=ProspectOverview)
async def get_prospect_overview(request: ProspectOverviewRequest):
    """
    Get comprehensive prospect overview from BigQuery.
    
    Aggregates data from multiple sources:
    - Salesforce: Account details, ARR, lifecycle stage, team assignments
    - Salesforce Opportunities: Active deals, stages, amounts
    - Gong: Call analytics, spotlight summaries, engagement metrics
    - Pendo: Product usage, feature adoption, active users
    - FullStory: User session data
    
    Supports multiple lookup methods:
    - account_name: Fuzzy match on account name
    - domain: Match on email/website domain
    - sfdc_account_id: Exact match on Salesforce Account ID
    
    Uses in-memory caching (15 min TTL) to speed up repeated requests.
    Call /api/prefetch-prospect to warm the cache proactively.
    """
    with tracer.start_as_current_span(
        f"{SPAN_PREFIX}.get_prospect_overview",
        attributes={
            "lookup.account_name": request.account_name or "",
            "lookup.domain": request.domain or "",
            "lookup.sfdc_id": request.sfdc_account_id or "",
            "openinference.span.kind": "chain",
        }
    ) as span:
        try:
            if not bq_client:
                span.set_status(Status(StatusCode.ERROR, "BigQuery client not available"))
                raise HTTPException(
                    status_code=503,
                    detail="BigQuery client not available. For local development, run: gcloud auth application-default login"
                )
            
            # Check cache first (fast path)
            cached = _prospect_cache.get(
                request.account_name,
                request.domain,
                request.sfdc_account_id
            )
            if cached:
                span.set_attribute("cache.hit", True)
                span.add_event("cache_hit", {
                    "account_name": request.account_name or "",
                })
                span.set_status(Status(StatusCode.OK))
                return cached
            
            span.set_attribute("cache.hit", False)
            span.add_event("fetching_prospect_overview", {
                "account_name": request.account_name or "",
                "domain": request.domain or "",
                "sfdc_id": request.sfdc_account_id or ""
            })
            
            # Fetch prospect overview from BigQuery (parallelized queries)
            overview = bq_client.get_prospect_overview(
                account_name=request.account_name,
                domain=request.domain,
                sfdc_account_id=request.sfdc_account_id,
                manual_competitors=request.manual_competitors
            )
            overview = _with_optional_gong_mcp_enrichment(overview)
            
            # Populate cache for subsequent requests
            _prospect_cache.set(
                request.account_name,
                request.domain,
                request.sfdc_account_id,
                overview
            )

            # Log results
            span.set_attribute("result.data_sources", ", ".join(overview.data_sources_available))
            if "gong_mcp" in (overview.data_sources_available or []):
                span.set_attribute("result.gong_mcp_enriched", True)
            span.set_attribute("result.has_salesforce", overview.salesforce is not None)
            span.set_attribute("result.opportunity_count", len(overview.all_opportunities))
            span.set_attribute("result.gong_call_count", overview.gong_summary.total_calls if overview.gong_summary else 0)
            
            if overview.errors:
                span.add_event("partial_errors", {"errors": ", ".join(overview.errors)})
            
            span.set_status(Status(StatusCode.OK))
            
            return overview
            
        except HTTPException:
            raise
        except ValueError as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise HTTPException(
                status_code=400,
                detail=str(e)
            )
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to fetch prospect overview: {str(e)}"
            )


@app.get("/api/doc-links/check")
async def doc_links_check():
    """
    Validate canonical docs.arize.com URLs used in PoC/PoT templates (HEAD with redirect follow).
    Intended for ops dashboards; can be slow (~seconds) when run cold.
    """
    return await asyncio.to_thread(validate_doc_links_sync)


@app.post("/api/prospect-intelligence-bundle")
async def prospect_intelligence_bundle(request: ProspectOverviewRequest):
    """
    One round-trip for deal health score, competitive mention rollup, and meeting prep bullets.
    Reuses the same in-memory cache as /api/prospect-overview when possible.
    """
    with tracer.start_as_current_span(
        f"{SPAN_PREFIX}.prospect_intelligence_bundle",
        attributes={
            "lookup.account_name": request.account_name or "",
            "lookup.domain": request.domain or "",
            "lookup.sfdc_id": request.sfdc_account_id or "",
            "openinference.span.kind": "chain",
        },
    ) as span:
        try:
            if not bq_client:
                span.set_status(Status(StatusCode.ERROR, "BigQuery client not available"))
                raise HTTPException(
                    status_code=503,
                    detail="BigQuery client not available. For local development, run: gcloud auth application-default login",
                )

            overview = _prospect_cache.get(
                request.account_name,
                request.domain,
                request.sfdc_account_id,
            )
            if overview is None:

                def _fetch():
                    o = bq_client.get_prospect_overview(
                        account_name=request.account_name,
                        domain=request.domain,
                        sfdc_account_id=request.sfdc_account_id,
                        manual_competitors=request.manual_competitors,
                    )
                    return _with_optional_gong_mcp_enrichment(o)

                overview = await asyncio.wait_for(asyncio.to_thread(_fetch), timeout=60.0)
                _prospect_cache.set(
                    request.account_name,
                    request.domain,
                    request.sfdc_account_id,
                    overview,
                )

            bundle = build_prospect_intelligence_bundle(overview)
            span.set_attribute("intel.competitive_rows", len(bundle.get("competitive_mentions", [])))
            span.set_status(Status(StatusCode.OK))
            return bundle

        except HTTPException:
            raise
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=504,
                detail="Timed out fetching prospect intelligence. Try again.",
            )
        except ValueError as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to build intelligence bundle: {str(e)[:500]}",
            ) from e


# ============================================================
# Custom Demo Builder
# ============================================================

AVAILABLE_USE_CASES = [
    {"value": "text-to-sql-bi-agent", "label": "Text-to-SQL / BI Agent"},
    {"value": "retrieval-augmented-search", "label": "RAG / Retrieval Search"},
    {"value": "multi-agent-orchestration", "label": "Multi-Agent Orchestration"},
    {"value": "classification-routing", "label": "Classification / Routing"},
    {"value": "multimodal-ai", "label": "Multimodal / Vision AI"},
    {"value": "mcp-tool-use", "label": "MCP Tool Use"},
    {"value": "multiturn-chatbot-with-tools", "label": "Chatbot with Tools"},
    {"value": "travel-agent", "label": "Travel Agent"},
    {"value": "guardrails", "label": "Guardrails"},
    {"value": "generic", "label": "Generic LLM Pipeline"},
]

AVAILABLE_FRAMEWORKS = [
    {"value": "langgraph", "label": "LangGraph"},
    {"value": "langchain", "label": "LangChain"},
    {"value": "crewai", "label": "CrewAI"},
    {"value": "adk", "label": "Google ADK"},
]


def _parse_demo_tools_text(text: str | None) -> list[dict[str, str]]:
    """SKILL.md optional ``tools`` — one entry per non-empty line: ``name — description`` (or ``-``, ``:``)."""
    if not (text or "").strip():
        return []
    out: list[dict[str, str]] = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        for sep in (" — ", " – ", " - ", ": ", "\t"):
            if sep in line:
                a, b = line.split(sep, 1)
                out.append({"name": a.strip(), "description": b.strip()})
                break
        else:
            out.append({"name": line, "description": ""})
        if len(out) >= 16:
            break
    return out


def _parse_demo_csv(text: str | None) -> list[str]:
    if not (text or "").strip():
        return []
    return [p.strip() for p in (text or "").split(",") if p.strip()]


def _parse_demo_prompt_versions(text: str | None) -> dict[str, float] | None:
    if not (text or "").strip():
        return None
    try:
        d = json.loads(text)
        if not isinstance(d, dict):
            return None
        return {str(k): float(v) for k, v in d.items()}
    except (json.JSONDecodeError, ValueError, TypeError):
        return None


class ClassifyDemoRequest(BaseModel):
    """Request to classify a prospect for the external **arize-synthetic-demo** skill (SKILL.md inputs + CRM hints)."""

    account_name: str
    # SKILL.md required #2 — primary free-text use case / vertical (not the internal demo-pattern taxonomy).
    industry_or_use_case: str
    # App-only extension: merged into prompts and returned as ``additional_context`` in recommended_inputs.
    additional_context: Optional[str] = None
    # SKILL.md required #6 — explicit output directory for generated files.
    output_dir: Optional[str] = None
    # SKILL.md required #3–#4 — LLM stack and span-tree shape (must match skill enums).
    skill_framework: str
    agent_architecture: str
    num_traces: Optional[int] = None
    with_evals: Optional[bool] = None
    with_dataset_and_experiments: Optional[bool] = None
    scenarios: Optional[list[str]] = None
    # SKILL.md optional — newline-separated "name — description" (or "name: desc") per line; max 16 lines parsed.
    tools_text: Optional[str] = None
    # Comma-separated template names.
    prompt_template_names: Optional[str] = None
    session_size_min: Optional[int] = None
    session_size_max: Optional[int] = None
    # JSON object, e.g. {"v1.0": 0.7, "v2.0": 0.3}
    prompt_versions_json: Optional[str] = None
    # Comma-separated model slugs for the 2-model experiment grid override.
    experiment_grid_models: Optional[str] = None
    # Deprecated: kept for backward compatibility; ignored when industry_or_use_case + skill fields are sent.
    use_case_override: Optional[str] = None
    framework_override: Optional[str] = None


class SyntheticDemoSkillHints(BaseModel):
    """Inputs and copy-paste text for the external **arize-synthetic-demo** Claude skill (see synthetic_demo_skill.py)."""

    skill: dict
    suggested_prompt_for_claude: str
    recommended_inputs: dict
    next_steps: list[str]


class ClassifyDemoResponse(BaseModel):
    """Response from use-case classification + skill-aligned hints (no in-app trace generation)."""
    use_case: str
    framework: str
    reasoning: Optional[str] = None
    industry: Optional[str] = None
    available_use_cases: list[dict]
    available_frameworks: list[dict]
    gong_calls_used: Optional[int] = None
    data_sources_note: Optional[str] = None
    synthetic_demo_skill: SyntheticDemoSkillHints


def _infer_use_case(overview, additional_context: str | None = None) -> tuple[str, str, str | None]:
    """Infer the best demo use case and framework from full prospect data.

    Uses all available signals: user-provided additional_context (when given),
    Gong calls, deal summaries, customer notes, and account metadata. User context
    is prepended so e.g. "HR chatbot for 1:1 prep" drives Chatbot use case.

    Returns:
        Tuple of (use_case, framework, reasoning). framework defaults to "langgraph".
    """
    signals: list[str] = []

    # User-provided scenario context (e.g. "HR chatbot for 1:1 prep") — use first so it drives classification
    if (additional_context or "").strip():
        signals.append("User-provided scenario context: " + additional_context.strip())

    if overview:
        # Gong signals (from BigQuery Gong datasets): spotlight + snippets + metadata
        if overview.gong_summary and overview.gong_summary.recent_calls:
            for call in overview.gong_summary.recent_calls:
                if getattr(call, "call_title", None):
                    signals.append(f"Gong call title: {call.call_title}")
                if call.spotlight_brief:
                    signals.append(f"Gong spotlight brief: {call.spotlight_brief}")
                if call.spotlight_key_points:
                    for kp in call.spotlight_key_points:
                        if isinstance(kp, str):
                            signals.append(f"Gong key point: {kp}")
                        elif isinstance(kp, list):
                            signals.extend([f"Gong key point: {str(item)}" for item in kp if item])
                if getattr(call, "spotlight_next_steps", None):
                    signals.append(f"Gong next steps: {call.spotlight_next_steps}")
                if getattr(call, "spotlight_outcome", None):
                    signals.append(f"Gong outcome: {call.spotlight_outcome}")
                if getattr(call, "spotlight_type", None):
                    signals.append(f"Gong spotlight type: {call.spotlight_type}")
                # Transcript snippet is often the most concrete signal for SQL/ADK mentions
                if getattr(call, "transcript_snippet", None):
                    signals.append(f"Gong transcript snippet: {call.transcript_snippet}")

        # Aggregated key themes
        if overview.gong_summary and overview.gong_summary.key_themes:
            signals.extend([f"Gong theme: {t}" for t in overview.gong_summary.key_themes])

        # Deal summary topics and current state
        if overview.sales_engagement and overview.sales_engagement.deal_summary:
            ds = overview.sales_engagement.deal_summary
            if ds.key_topics_discussed:
                signals.extend([f"Deal topic: {t}" for t in ds.key_topics_discussed])
            if ds.current_state:
                signals.append(f"Deal current state: {ds.current_state}")

        # Customer notes from Salesforce
        if overview.salesforce and overview.salesforce.customer_notes:
            signals.append(overview.salesforce.customer_notes)

        # Account metadata — always useful context for classification
        if overview.salesforce:
            sf = overview.salesforce
            account_info = []
            if sf.name:
                account_info.append(f"Company: {sf.name}")
            if sf.industry:
                account_info.append(f"Industry: {sf.industry}")
            if sf.description:
                account_info.append(f"Description: {sf.description}")
            if sf.is_using_llms:
                account_info.append(f"Using LLMs: {sf.is_using_llms}")
            if account_info:
                signals.append(" | ".join(account_info))

    # Keyword hints: user context (e.g. "chatbot", "1:1") and Gong/CRM keywords drive use case
    hint_use_case, hint_framework = _extract_use_case_framework_hints("\n".join(signals)) if signals else (None, None)
    if hint_use_case and hint_framework:
        return hint_use_case, hint_framework, "Detected explicit keywords for use case and framework in Gong/CRM signals."
    # When user provided scenario context and it matches chatbot/1:1, use it so CRM noise doesn't pick RAG
    if hint_use_case and (additional_context or "").strip():
        return hint_use_case, hint_framework or "langgraph", "User-provided scenario context matched use case."

    # Always try LLM classification when we have any signals
    if signals:
        use_case, framework, reasoning = _classify_use_case_with_llm(
            signals,
            hint_use_case=hint_use_case,
            hint_framework=hint_framework,
        )
        if use_case:
            return use_case, framework or "langgraph", reasoning

    # Fall back to industry-based heuristic only when we have nothing
    industry = overview.salesforce.industry if overview and overview.salesforce else None
    return _industry_heuristic(industry), "langgraph", None


def _extract_use_case_framework_hints(text: str) -> tuple[str | None, str | None]:
    """Lightweight keyword hints from user context and Gong/CRM text."""
    t = (text or "").lower()

    use_case = None
    framework = None

    # --- use case hints (chatbot/1:1 first so user scenario wins over generic CRM keywords) ---
    if any(k in t for k in [
        "chatbot", "1:1", "1-1", "one on one", "one-on-one",
        "hr chatbot", "prep for 1:1", "prep for 1-1", "prep for one on one",
        "prep for one-on-one", "manager meeting", "employee prep",
    ]):
        use_case = "multiturn-chatbot-with-tools"
    elif any(k in t for k in ["text-to-sql", "text to sql", "nl2sql", "natural language to sql", "generate sql", "sql query", "bigquery", "snowflake", "redshift", "postgres", "databricks sql", "warehouse", "semantic layer", "bi agent", "analytics query"]):
        use_case = "text-to-sql-bi-agent"
    elif any(k in t for k in ["rag", "retrieval", "vector db", "embedding", "semantic search", "knowledge base"]):
        use_case = "retrieval-augmented-search"
    elif any(k in t for k in ["mcp", "model context protocol"]):
        use_case = "mcp-tool-use"

    # --- framework hints ---
    if any(k in t for k in ["google adk", " agent development kit", "vertex ai", "agent builder", "google agent builder", "google-genai", "adk"]):
        framework = "adk"
    elif "crewai" in t:
        framework = "crewai"
    elif "langchain" in t and "langgraph" not in t:
        framework = "langchain"

    return use_case, framework


def _classify_use_case_with_llm(
    signals: list[str],
    hint_use_case: str | None = None,
    hint_framework: str | None = None,
) -> tuple[str | None, str | None, str | None]:
    """Use Claude Haiku to classify use case and framework from prospect signals."""
    from openai_compat_completion import completion as llm_completion

    combined_text = "\n".join(signals[:30])  # Use more signals so text-to-SQL/ADK mentions aren't dropped
    hints_text = ""
    if hint_use_case or hint_framework:
        hints_text = (
            "\n\n## Hints (keyword signals detected)\n"
            f"- hinted_use_case: {hint_use_case or 'none'}\n"
            f"- hinted_framework: {hint_framework or 'none'}\n"
            "If these hints are supported by the Conversation Signals, prefer them.\n"
        )

    prompt = f"""You are classifying a prospect's AI/ML use case and orchestration framework to select the most appropriate demo.

Based on the following conversation signals from sales calls, classify:
1. Which demo type best matches the CORE APPLICATION the prospect is building
2. Which orchestration framework the prospect uses

## Available Demo Types (in priority order — prefer the most specific match):
1. "text-to-sql-bi-agent" - Text-to-SQL, natural language to database queries, data querying, BI agents, analytics pipelines. Even if wrapped in a multi-agent chatbot, if text-to-SQL is the primary function, choose this.
2. "retrieval-augmented-search" - RAG pipelines, document search, knowledge bases, semantic search, retrieval-based Q&A.
3. "multi-agent-orchestration" - Multi-agent systems where multiple autonomous agents collaborate: supervisor/worker patterns, research+analysis+writing teams, agent delegation, orchestrator agents. Choose this when MULTIPLE agents cooperate, not just a single chatbot calling tools.
4. "classification-routing" - Classification pipelines, intent detection, ticket routing, document categorization, sentiment analysis, content moderation, auto-labeling workflows.
5. "multimodal-ai" - Vision/image AI, multimodal LLM applications, document extraction (OCR/IDP), image classification, quality inspection, medical imaging analysis.
6. "mcp-tool-use" - MCP (Model Context Protocol) based agents, tool-use pipelines connecting to external servers/services (file systems, databases, APIs, Slack, GitHub), agentic tool discovery and execution.
7. "multiturn-chatbot-with-tools" - Single chatbot or agent with tool calling where no specific pattern (SQL, RAG, multi-agent, classification, vision, MCP) dominates.
8. "travel-agent" - Travel booking agents: flight and hotel search, trip planning, itineraries, destination recommendations, tool-augmented travel assistants.
9. "generic" - When signals are ambiguous or don't clearly match any specific pattern above.

## Available Frameworks:
1. "langgraph" - LangGraph / LangChain graph-based agent orchestration. DEFAULT if no framework is mentioned.
2. "langchain" - Plain LangChain LCEL chains (prompt | llm | parser), no graph.
3. "crewai" - CrewAI multi-agent framework.
4. "adk" - Google ADK (Agent Development Kit), Google Agent Builder, Vertex AI agents, google-genai.

Framework rules (apply in order): If the prospect mentions "ADK", "Google ADK", "Agent Development Kit", "Vertex AI agents", "Google Agent Builder", or "google-genai", use "adk". If they mention "CrewAI", use "crewai". If they mention "LangChain" without LangGraph, use "langchain". Only default to "langgraph" when no framework is mentioned at all.
Use case rules: If there is ANY mention of natural language to database, text-to-SQL, SQL queries, BI agents, analytics queries, or data querying, prefer "text-to-sql-bi-agent" over "multi-agent-orchestration".

## Conversation Signals:
{combined_text}
{hints_text}

## Instructions:
Return ONLY a JSON object with:
- "use_case": one of the nine demo type strings
- "framework": one of the four framework strings
- "reasoning": a brief 1-sentence explanation

Return only valid JSON, no markdown formatting or code blocks."""

    try:
        model = os.environ.get("USE_CASE_CLASSIFICATION_MODEL", "claude-haiku-4-5")

        response = llm_completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0,
        )

        response_text = response.choices[0].message.content.strip()

        # Handle potential markdown code blocks
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
            response_text = response_text.strip()

        result = json.loads(response_text)
        use_case = result.get("use_case", "")
        framework = result.get("framework", "langgraph")
        reasoning = result.get("reasoning", "")

        valid_use_cases = {
            "text-to-sql-bi-agent",
            "retrieval-augmented-search",
            "multi-agent-orchestration",
            "classification-routing",
            "multimodal-ai",
            "mcp-tool-use",
            "multiturn-chatbot-with-tools",
            "travel-agent",
            "generic",
        }
        valid_frameworks = {"langgraph", "langchain", "crewai", "adk"}
        if framework not in valid_frameworks:
            framework = "langgraph"
        if use_case in valid_use_cases:
            return use_case, framework, reasoning
        return None, None, None

    except Exception as e:
        logger.warning("LLM use-case classification failed: %s", e)
        return None, None, None


def _industry_heuristic(industry: str | None) -> str:
    """Fallback: infer demo use case from Salesforce industry field."""
    if not industry:
        return "generic"

    industry_lower = industry.lower()
    if any(k in industry_lower for k in ["financial", "banking", "insurance", "fintech",
                                          "data", "analytics", "intelligence", "research"]):
        return "text-to-sql-bi-agent"
    elif any(k in industry_lower for k in ["healthcare", "pharma", "biotech", "medical"]):
        return "multimodal-ai"
    elif any(k in industry_lower for k in ["manufacturing", "automotive", "industrial"]):
        return "multimodal-ai"
    elif any(k in industry_lower for k in ["retail", "ecommerce", "e-commerce"]):
        return "classification-routing"
    elif any(k in industry_lower for k in ["technology", "software", "saas", "ai"]):
        return "multi-agent-orchestration"
    elif any(k in industry_lower for k in ["consulting", "professional services"]):
        return "multi-agent-orchestration"
    elif any(k in industry_lower for k in ["media", "entertainment", "gaming"]):
        return "retrieval-augmented-search"
    elif any(k in industry_lower for k in ["telecom", "communications"]):
        return "classification-routing"
    elif any(k in industry_lower for k in ["travel", "hospitality", "tourism", "leisure", "airline", "hotel"]):
        return "travel-agent"
    else:
        return "retrieval-augmented-search"


@app.post("/api/classify-demo", response_model=ClassifyDemoResponse)
async def classify_demo(request: ClassifyDemoRequest):
    """
    Classify a prospect's use case and framework from CRM/Gong data.
    Returns the classification along with all available options for user override.
    """
    with api_span("classify_demo", account_name=request.account_name):
        return await _classify_demo_impl(request)


async def _classify_demo_impl(request: ClassifyDemoRequest):
    from synthetic_demo_skill import (
        SKILL_AGENT_ARCHITECTURE_VALUES,
        SKILL_FRAMEWORK_VALUES,
        SKILL_SCENARIO_VALUES,
        build_synthetic_demo_skill_hints,
    )

    if not (request.industry_or_use_case or "").strip():
        raise HTTPException(
            status_code=400,
            detail="industry_or_use_case is required (SKILL.md: free-text industry or use case).",
        )

    sf = request.skill_framework.strip().lower()
    if sf not in SKILL_FRAMEWORK_VALUES:
        raise HTTPException(status_code=400, detail=f"Invalid skill_framework: {request.skill_framework!r}")
    aa = request.agent_architecture.strip()
    if aa not in SKILL_AGENT_ARCHITECTURE_VALUES:
        raise HTTPException(
            status_code=400, detail=f"Invalid agent_architecture: {request.agent_architecture!r}"
        )
    if request.scenarios:
        bad = [s for s in request.scenarios if s not in SKILL_SCENARIO_VALUES]
        if bad:
            raise HTTPException(status_code=400, detail=f"Invalid scenario value(s): {bad}")

    tools_parsed = _parse_demo_tools_text(request.tools_text)
    prompt_names = _parse_demo_csv(request.prompt_template_names)
    exp_models = _parse_demo_csv(request.experiment_grid_models)
    pv = _parse_demo_prompt_versions(request.prompt_versions_json)
    if (request.prompt_versions_json or "").strip() and pv is None:
        raise HTTPException(
            status_code=400,
            detail="prompt_versions_json must be valid JSON object mapping version strings to floats, "
            'e.g. {"v1.0": 0.7, "v2.0": 0.3}',
        )

    smin, smax = request.session_size_min, request.session_size_max
    session_range: tuple[int, int] | None = None
    if smin is not None or smax is not None:
        lo = 3 if smin is None else max(1, min(50, int(smin)))
        hi = 6 if smax is None else max(1, min(50, int(smax)))
        if lo > hi:
            lo, hi = hi, lo
        session_range = (lo, hi)

    overview = None
    industry = None
    try:
        def _bq_lookup():
            from bigquery_client import BigQueryClient
            bq = BigQueryClient()
            return bq.get_prospect_overview(account_name=request.account_name)
        overview = await asyncio.wait_for(asyncio.to_thread(_bq_lookup), timeout=30.0)
        if overview and overview.salesforce:
            industry = overview.salesforce.industry
    except (asyncio.TimeoutError, Exception):
        pass

    gong_calls_used = None
    data_sources_note = None
    if overview:
        gong_calls_used = overview.gong_summary.total_calls if overview.gong_summary else 0
        if gong_calls_used and overview.data_sources_available:
            data_sources_note = f"Classification used Gong ({gong_calls_used} call(s)), Salesforce, and/or deal summary."
        elif overview.data_sources_available:
            data_sources_note = "No Gong calls linked to this account in BigQuery; used Salesforce/CRM only. Classification may be generic."
        else:
            data_sources_note = "No prospect data found for this account name. Using default use case."
    else:
        data_sources_note = "Could not load prospect data (BigQuery timeout or no match). Using default use case."

    additional_context = getattr(request, "additional_context", None) or None
    use_case, framework, reasoning = _infer_use_case(overview, additional_context=additional_context)
    if not use_case:
        use_case = _industry_heuristic(industry) if industry else _industry_heuristic(None)
    if not framework:
        framework = "langgraph"

    ow_uc = getattr(request, "use_case_override", None)
    ow_fw = getattr(request, "framework_override", None)
    if (ow_uc or "").strip():
        use_case = ow_uc.strip()
        reasoning = (reasoning or "") + " (use case overridden for skill mapping.)"
    if (ow_fw or "").strip():
        framework = ow_fw.strip()

    skill_hints = SyntheticDemoSkillHints(
        **build_synthetic_demo_skill_hints(
            company_name=request.account_name,
            industry=industry,
            use_case=use_case,
            framework=framework,
            reasoning=reasoning,
            additional_context=getattr(request, "additional_context", None),
            industry_or_use_case_user=request.industry_or_use_case.strip(),
            output_dir_user=request.output_dir,
            skill_framework=request.skill_framework,
            agent_architecture=request.agent_architecture,
            num_traces=request.num_traces,
            with_evals=request.with_evals,
            with_dataset_and_experiments=request.with_dataset_and_experiments,
            scenarios=request.scenarios,
            tools=tools_parsed or None,
            prompt_template_names=prompt_names or None,
            session_size_range=session_range,
            prompt_versions=pv,
            experiment_grid_models=exp_models or None,
        )
    )

    return ClassifyDemoResponse(
        use_case=use_case,
        framework=framework,
        reasoning=reasoning,
        industry=industry,
        available_use_cases=AVAILABLE_USE_CASES,
        available_frameworks=AVAILABLE_FRAMEWORKS,
        gong_calls_used=gong_calls_used,
        data_sources_note=data_sources_note,
        synthetic_demo_skill=skill_hints,
    )


@app.get("/api/custom-demo/skill")
async def custom_demo_skill_info():
    """Static pointers to the **arize-synthetic-demo** Claude skill (no server-side trace generation)."""
    from synthetic_demo_skill import static_skill_info

    return static_skill_info()


# ============================================================
# Generate Demo (execute arize-synthetic-demo skill)
# ============================================================


class GenerateDemoRequest(BaseModel):
    """Request to generate a synthetic demo by executing the arize-synthetic-demo skill."""

    account_name: str
    industry_or_use_case: str
    skill_framework: str
    agent_architecture: str
    num_traces: Optional[int] = 500
    with_evals: Optional[bool] = True
    with_dataset_and_experiments: Optional[bool] = True
    scenarios: Optional[list[str]] = None
    tools_text: Optional[str] = None
    additional_context: Optional[str] = None


class GenerateDemoResponse(BaseModel):
    """Response with metadata about the generated demo (file returned separately as download)."""

    folder_name: str
    project_name: str
    dataset_name: str
    prompt_name: str
    agent_name: str
    framework: str
    architecture: str
    llm_model: str
    files: list[str]
    domain_content: dict


@app.post("/api/generate-demo")
async def generate_demo(request: GenerateDemoRequest):
    """
    Generate a synthetic demo by executing the arize-synthetic-demo skill logic.
    
    Uses an LLM to generate domain-specific content (queries, tools, prompts),
    fills in the generator skeleton template, and returns a downloadable ZIP file.
    """
    from demo_generator import generate_demo_files, SKILL_PATH
    from synthetic_demo_skill import (
        SKILL_AGENT_ARCHITECTURE_VALUES,
        SKILL_FRAMEWORK_VALUES,
        SKILL_SCENARIO_VALUES,
    )

    with api_span("generate_demo", account_name=request.account_name):
        # Validate inputs
        if not (request.industry_or_use_case or "").strip():
            raise HTTPException(
                status_code=400,
                detail="industry_or_use_case is required.",
            )

        sf = request.skill_framework.strip().lower()
        if sf not in SKILL_FRAMEWORK_VALUES:
            raise HTTPException(status_code=400, detail=f"Invalid skill_framework: {request.skill_framework!r}")
        
        aa = request.agent_architecture.strip()
        if aa not in SKILL_AGENT_ARCHITECTURE_VALUES:
            raise HTTPException(
                status_code=400, detail=f"Invalid agent_architecture: {request.agent_architecture!r}"
            )
        
        if request.scenarios:
            bad = [s for s in request.scenarios if s not in SKILL_SCENARIO_VALUES]
            if bad:
                raise HTTPException(status_code=400, detail=f"Invalid scenario value(s): {bad}")

        # Check that skill templates exist
        if not SKILL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail="arize-synthetic-demo skill not found. Please ensure the skill is installed.",
            )

        # Parse tools from text
        tools_parsed = _parse_demo_tools_text(request.tools_text)

        try:
            zip_bytes, metadata = generate_demo_files(
                company_name=request.account_name.strip(),
                industry_or_use_case=request.industry_or_use_case.strip(),
                framework=sf,
                agent_architecture=aa,
                num_traces=request.num_traces or 500,
                tools=tools_parsed or None,
                additional_context=request.additional_context,
                with_evals=request.with_evals if request.with_evals is not None else True,
                with_dataset_and_experiments=request.with_dataset_and_experiments if request.with_dataset_and_experiments is not None else True,
                scenarios=request.scenarios,
            )
        except FileNotFoundError as e:
            raise HTTPException(
                status_code=503,
                detail=f"Skill template not found: {e}",
            )
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=502,
                detail=f"Failed to parse LLM response: {e}",
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Demo generation failed: {str(e)[:500]}",
            )

        filename = f"{metadata['folder_name']}.zip"

        return Response(
            content=zip_bytes,
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )


# ============================================================
# Demo Insights (auto-populate demo builder from Gong calls)
# ============================================================


class DemoInsightsRequest(BaseModel):
    """Request to fetch demo builder insights from Gong calls."""

    account_name: str


class DemoInsightsResponse(BaseModel):
    """Response with suggested demo builder field values from Gong analysis."""

    account_name: str
    industry_or_use_case: Optional[str] = None
    suggested_framework: Optional[str] = None
    suggested_agent_architecture: Optional[str] = None
    suggested_tools: Optional[str] = None
    additional_context: Optional[str] = None
    gong_calls_analyzed: int = 0
    data_sources_note: Optional[str] = None
    insights_summary: Optional[str] = None


def _extract_demo_insights_with_llm(
    signals: list[str], account_name: str
) -> dict[str, Optional[str]]:
    """Use LLM to extract demo builder suggestions from Gong call signals."""
    from openai_compat_completion import completion as llm_completion

    combined_text = "\n".join(signals[:40])

    prompt = f"""You are analyzing sales call transcripts and summaries for {account_name} to suggest demo configuration for an AI observability platform demo.

Based on the following conversation signals from sales calls, extract:

1. **industry_or_use_case**: A specific description of their AI/ML use case and industry. Be specific and detailed. Examples:
   - "Healthcare claims processing with RAG-based document analysis"
   - "Financial services fraud detection using multi-agent orchestration"
   - "E-commerce product recommendations with real-time embedding search"
   - "Legal document review using retrieval-augmented generation"

2. **suggested_framework**: One of: openai, anthropic, bedrock, vertex, adk, langchain, langgraph, crewai, generic
   - Look for mentions of specific LLM providers or agent frameworks
   - Default to "langgraph" if no framework is mentioned

3. **suggested_agent_architecture**: One of: single_agent, multi_agent_coordinator, retrieval_pipeline, rag_rerank, guarded_rag
   - Infer from their described system architecture
   - Default to "single_agent" if unclear

4. **suggested_tools**: A list of tools/capabilities their system uses, one per line with description. Examples:
   - "search_documents — Retrieves relevant documents from knowledge base"
   - "execute_sql — Runs SQL queries against data warehouse"
   - "send_notification — Sends alerts to stakeholders"
   Leave empty if no specific tools are mentioned.

5. **additional_context**: Key pain points, requirements, or context that would help customize the demo:
   - Compliance requirements (SOC2, HIPAA, etc.)
   - Scale/volume expectations
   - Integration requirements
   - Specific features they're interested in

6. **insights_summary**: A 2-3 sentence summary of what you learned about their AI/ML initiatives and what they're looking for.

## Conversation Signals:
{combined_text}

## Instructions:
Return ONLY a JSON object with these exact fields:
- "industry_or_use_case": string (required, be specific)
- "suggested_framework": string (one of the allowed values)
- "suggested_agent_architecture": string (one of the allowed values)
- "suggested_tools": string (newline-separated tools, or empty string)
- "additional_context": string (key context and requirements)
- "insights_summary": string (brief summary)

Return only valid JSON, no markdown formatting or code blocks."""

    try:
        model = os.environ.get("DEMO_INSIGHTS_MODEL", "claude-haiku-4-5")

        response = llm_completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
            temperature=0,
        )

        response_text = response.choices[0].message.content.strip()

        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
            response_text = response_text.strip()

        result = json.loads(response_text)

        valid_frameworks = {"openai", "anthropic", "bedrock", "vertex", "adk", "langchain", "langgraph", "crewai", "generic"}
        valid_architectures = {"single_agent", "multi_agent_coordinator", "retrieval_pipeline", "rag_rerank", "guarded_rag"}

        fw = result.get("suggested_framework", "").strip().lower()
        if fw not in valid_frameworks:
            fw = "langgraph"

        arch = result.get("suggested_agent_architecture", "").strip()
        if arch not in valid_architectures:
            arch = "single_agent"

        return {
            "industry_or_use_case": result.get("industry_or_use_case", "").strip() or None,
            "suggested_framework": fw,
            "suggested_agent_architecture": arch,
            "suggested_tools": result.get("suggested_tools", "").strip() or None,
            "additional_context": result.get("additional_context", "").strip() or None,
            "insights_summary": result.get("insights_summary", "").strip() or None,
        }

    except Exception as e:
        logger.warning("LLM demo insights extraction failed: %s", e)
        return {
            "industry_or_use_case": None,
            "suggested_framework": "langgraph",
            "suggested_agent_architecture": "single_agent",
            "suggested_tools": None,
            "additional_context": None,
            "insights_summary": None,
        }


@app.post("/api/demo-insights", response_model=DemoInsightsResponse)
async def get_demo_insights(request: DemoInsightsRequest):
    """
    Fetch Gong call insights for a prospect to auto-populate demo builder fields.

    Analyzes recent Gong calls for the account and extracts:
    - Industry/use case description
    - Suggested framework and architecture
    - Tools mentioned
    - Key context and pain points
    """
    with api_span("get_demo_insights", account_name=request.account_name):
        if not request.account_name.strip():
            raise HTTPException(status_code=400, detail="account_name is required.")

        overview = None
        signals: list[str] = []
        gong_calls_analyzed = 0
        data_sources_note = None

        try:
            def _bq_lookup():
                from bigquery_client import BigQueryClient
                bq = BigQueryClient()
                return bq.get_prospect_overview(account_name=request.account_name.strip())

            overview = await asyncio.wait_for(asyncio.to_thread(_bq_lookup), timeout=30.0)
            overview = _with_optional_gong_mcp_enrichment(overview)
        except (asyncio.TimeoutError, Exception) as e:
            logger.warning("Demo insights BQ lookup failed: %s", e)
            overview = None

        if overview:
            if overview.gong_summary and overview.gong_summary.recent_calls:
                gong_calls_analyzed = len(overview.gong_summary.recent_calls)
                for call in overview.gong_summary.recent_calls:
                    if getattr(call, "call_title", None):
                        signals.append(f"Call title: {call.call_title}")
                    if call.spotlight_brief:
                        signals.append(f"Call summary: {call.spotlight_brief}")
                    if call.spotlight_key_points:
                        for kp in call.spotlight_key_points:
                            if isinstance(kp, str):
                                signals.append(f"Key point: {kp}")
                            elif isinstance(kp, list):
                                signals.extend([f"Key point: {str(item)}" for item in kp if item])
                    if getattr(call, "spotlight_next_steps", None):
                        signals.append(f"Next steps: {call.spotlight_next_steps}")
                    if getattr(call, "spotlight_outcome", None):
                        signals.append(f"Outcome: {call.spotlight_outcome}")
                    if getattr(call, "transcript_snippet", None):
                        signals.append(f"Transcript excerpt: {call.transcript_snippet}")

            if overview.gong_summary and overview.gong_summary.key_themes:
                signals.extend([f"Theme: {t}" for t in overview.gong_summary.key_themes])

            if overview.sales_engagement and overview.sales_engagement.deal_summary:
                ds = overview.sales_engagement.deal_summary
                if ds.key_topics_discussed:
                    signals.extend([f"Deal topic: {t}" for t in ds.key_topics_discussed])
                if ds.current_state:
                    signals.append(f"Deal state: {ds.current_state}")

            if overview.salesforce:
                sf = overview.salesforce
                if sf.industry:
                    signals.append(f"Industry: {sf.industry}")
                if sf.description:
                    signals.append(f"Company description: {sf.description}")
                if sf.is_using_llms:
                    signals.append(f"Using LLMs: {sf.is_using_llms}")
                if sf.customer_notes:
                    signals.append(f"Customer notes: {sf.customer_notes}")

            if gong_calls_analyzed > 0:
                data_sources_note = f"Analyzed {gong_calls_analyzed} Gong call(s) and CRM data."
            elif overview.data_sources_available:
                data_sources_note = "No Gong calls found; used CRM data only."
            else:
                data_sources_note = "No data found for this account."
        else:
            data_sources_note = "Could not load prospect data (BigQuery timeout or no match)."

        if not signals:
            return DemoInsightsResponse(
                account_name=request.account_name.strip(),
                gong_calls_analyzed=0,
                data_sources_note=data_sources_note or "No data available for this account.",
                insights_summary="No Gong calls or CRM data found for this account. Please fill in the fields manually.",
            )

        insights = _extract_demo_insights_with_llm(signals, request.account_name.strip())

        return DemoInsightsResponse(
            account_name=request.account_name.strip(),
            industry_or_use_case=insights.get("industry_or_use_case"),
            suggested_framework=insights.get("suggested_framework"),
            suggested_agent_architecture=insights.get("suggested_agent_architecture"),
            suggested_tools=insights.get("suggested_tools"),
            additional_context=insights.get("additional_context"),
            gong_calls_analyzed=gong_calls_analyzed,
            data_sources_note=data_sources_note,
            insights_summary=insights.get("insights_summary"),
        )


# ============================================================
# Hypothesis Research (integrated from ae-hypothesis-tool)
# ============================================================

# Lazy-initialized hypothesis tool components
_hypothesis_agent = None
_hypothesis_init_done = False
# Set when agent is None so 503 reflects import vs setup (not a generic "langgraph missing" false negative).
_hypothesis_init_failure: str | None = None


def _get_hypothesis_agent():
    """Lazy-init the LangGraph research agent (see apps/api/pyproject.toml core + ``hypothesis`` extra)."""
    global _hypothesis_agent, _hypothesis_init_done, _hypothesis_init_failure
    if _hypothesis_init_done:
        return _hypothesis_agent
    _hypothesis_init_done = True
    _hypothesis_init_failure = None

    try:
        from hypothesis_tool.agents.research_agent import ResearchAgent
        from hypothesis_tool.clients.bigquery_client import BigQueryClient as HypBQClient
        from hypothesis_tool.config import get_settings as get_hyp_settings
    except ModuleNotFoundError as e:
        missing = getattr(e, "name", None) or str(e)
        _hypothesis_init_failure = (
            f"Missing Python module {missing!r} required for hypothesis research. "
            "Add it to apps/api pyproject dependencies and regenerate requirements.txt."
        )
        logger.error("Hypothesis research import failed (missing module): %s", e)
        _hypothesis_agent = None
        return None
    except ImportError as e:
        _hypothesis_init_failure = f"Hypothesis research import failed: {e}"
        logger.error("Hypothesis research import failed: %s", e)
        _hypothesis_agent = None
        return None

    try:
        settings = get_hyp_settings()
        bq = None
        try:
            bq = HypBQClient(project_id=settings.bq_project_id)
        except Exception as e:
            logger.warning("Hypothesis BQ client failed: %s", e)
        _hypothesis_agent = ResearchAgent(bq_client=bq)
        logger.info("Hypothesis Research Agent initialized")
    except Exception as e:
        _hypothesis_init_failure = f"Hypothesis agent setup failed after imports: {e}"
        logger.error("Hypothesis agent setup failed: %s", e)
        _hypothesis_agent = None
    return _hypothesis_agent


class HypothesisResearchRequest(BaseModel):
    company_name: str
    company_domain: Optional[str] = None
    known_competitive_situation: Optional[str] = None


@app.post("/api/hypothesis-research")
async def hypothesis_research(request: HypothesisResearchRequest):
    """
    Research a company using AI agent and generate data-driven hypotheses.
    Uses web search (Brave), CRM data (BigQuery), and LLM analysis.
    """
    try:
        if not request.company_name or len(request.company_name.strip()) < 2:
            raise HTTPException(status_code=400, detail="Company name must be at least 2 characters.")

        agent = _get_hypothesis_agent()
        if agent is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    _hypothesis_init_failure
                    or "Hypothesis research failed to initialize on this worker; see server logs."
                ),
            )
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        logger.error("Hypothesis research init error: %s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Research service unavailable: {str(e)[:200]}")

    # Ensure this request's traces go to the Arize "hypothesis-generator" project (explicit connection).
    hyp_provider = get_provider_for_component(COMPONENT_HYPOTHESIS)
    if hyp_provider is not None:
        set_request_tracer_provider(hyp_provider)
        logger.debug("Hypothesis research using project: %s", PROJECT_HYPOTHESIS)
    else:
        logger.warning("Hypothesis tracer provider not available (observability may be disabled)")

    # Use api_span so the root span is created with the current provider's tracer (set above).
    # That way root and all LangGraph/LLM child spans share the same trace_id and the Traces tab shows the tree.
    with api_span(
        "hypothesis_research",
        **{
            "input.value": f"Research {request.company_name}",
            "company.name": request.company_name,
        },
    ) as span:
        project_at_start = get_current_project_name()
        if os.environ.get("ARIZE_TRACE_DEBUG", "").strip().lower() in ("1", "true", "yes"):
            ctx = span.get_span_context() if hasattr(span, "get_span_context") else None
            trace_id = format(ctx.trace_id, "032x") if ctx and hasattr(ctx, "trace_id") else None
            logger.debug("hypothesis_research: project=%s trace_id=%s", project_at_start, trace_id)
        try:
            result, reasoning = await agent.research(
                company_name=request.company_name.strip(),
                company_domain=request.company_domain.strip() if request.company_domain else None,
            )

            # Convert pydantic model to dict for JSON response
            result_dict = result.model_dump() if hasattr(result, 'model_dump') else result

            # Add clean output summary to root span for visibility in Arize
            research = result_dict.get("research", {})
            quality = result_dict.get("research_quality", "unknown")
            hypotheses_count = len(result_dict.get("hypotheses", []))
            signals_count = len(research.get("ai_ml_signals", []))
            industry = research.get("industry", "Unknown")
            confidence = research.get("ai_ml_confidence", "unknown")
            
            # Build clean human-readable output
            hypotheses_list = result_dict.get("hypotheses", [])
            hypothesis_texts = []
            for h in hypotheses_list[:3]:
                text = h.get("hypothesis", h.get("title", "Untitled"))
                if text and len(text) > 100:
                    text = text[:100] + "..."
                hypothesis_texts.append(text)
            
            output_lines = [
                f"Company: {result_dict.get('company_name')}",
                f"Quality: {quality.upper()}",
                f"Industry: {industry}",
                f"AI/ML Confidence: {confidence}",
                f"Signals Found: {signals_count}",
                f"Hypotheses Generated: {hypotheses_count}",
            ]
            if hypothesis_texts:
                output_lines.append("Top Hypotheses:")
                for i, text in enumerate(hypothesis_texts, 1):
                    output_lines.append(f"  {i}. {text}")
            
            warnings = result_dict.get("warnings", [])
            errors = result_dict.get("errors", [])
            if warnings:
                output_lines.append(f"Warnings: {len(warnings)}")
            if errors:
                output_lines.append(f"Errors: {len(errors)}")
            
            span.set_attribute("output.value", "\n".join(output_lines))
            span.set_attribute("research.quality", quality)
            span.set_attribute("research.hypotheses_count", hypotheses_count)
            span.set_attribute("research.signals_count", signals_count)

            # Flush hypothesis provider so traces are exported before response (same pattern as demo)
            hyp_provider = get_provider_for_component(COMPONENT_HYPOTHESIS)
            if hyp_provider is not None and hasattr(hyp_provider, "force_flush"):
                try:
                    hyp_provider.force_flush(timeout_millis=15000)
                except Exception:
                    pass

            return {
                "result": result_dict,
                "agent_reasoning": reasoning,
            }
        except HTTPException:
            raise
        except Exception as e:
            import traceback

            from hypothesis_tool.errors import (
                AmbiguousCompanyError,
                CompanyNotFoundError,
                LLMError,
                SearchAPIError,
            )

            logger.error("Hypothesis research error: %s", traceback.format_exc())
            if isinstance(e, SearchAPIError):
                raise HTTPException(status_code=503, detail="Web search service unavailable.") from e
            if isinstance(e, LLMError):
                raise HTTPException(status_code=503, detail="AI service temporarily unavailable.") from e
            if isinstance(e, (CompanyNotFoundError, AmbiguousCompanyError)):
                raise HTTPException(status_code=400, detail=str(e)) from e

            error_detail = str(e)
            el = error_detail.lower()
            if "timeout" in el:
                raise HTTPException(status_code=504, detail="Request timed out. Try again.") from e
            # Unwrapped SDK errors (not every failure type is converted to LLMError / SearchAPIError).
            if "anthropic" in el or "claude" in el:
                raise HTTPException(status_code=503, detail="AI service temporarily unavailable.") from e
            if "brave" in el:
                raise HTTPException(status_code=503, detail="Web search service unavailable.") from e
            raise HTTPException(status_code=500, detail=f"Research failed: {error_detail[:200]}") from e


# ============================================================
# PoC / PoT document (Word) from BigQuery + LLM appendix
# ============================================================


@app.post("/api/generate-poc-document")
async def generate_poc_document_deliverable(request: GeneratePocDocumentRequest):
    """
    Fetch ProspectOverview from BigQuery, run an LLM to fill in-template placeholders
    in the selected Word master, and return a .docx download.
    """
    with tracer.start_as_current_span(
        f"{SPAN_PREFIX}.generate_poc_document",
        attributes={
            "poc_doc.template": request.document_template,
            "poc_doc.account_name": request.account_name.strip(),
            "openinference.span.kind": "chain",
        },
    ) as span:
        try:
            if not bq_client:
                span.set_status(Status(StatusCode.ERROR, "BigQuery unavailable"))
                raise HTTPException(
                    status_code=503,
                    detail="BigQuery client not available. For local development, run: gcloud auth application-default login",
                )

            domain = (request.domain or "").strip() or None
            span.add_event(
                "fetch_prospect_overview",
                {"account_name": request.account_name.strip(), "domain": domain or ""},
            )
            overview = bq_client.get_prospect_overview(
                account_name=request.account_name.strip(),
                domain=domain,
                sfdc_account_id=None,
                manual_competitors=None,
            )
            overview = _with_optional_gong_mcp_enrichment(overview)

            doc_bytes, filename = build_poc_document(
                overview=overview,
                document_template=request.document_template,
                manual_notes=request.manual_notes,
                llm_model=None,
            )

            span.set_attribute("poc_doc.filename", filename)
            span.set_attribute("poc_doc.size_bytes", len(doc_bytes))
            span.set_status(Status(StatusCode.OK))

            return Response(
                content=doc_bytes,
                media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                headers={"Content-Disposition": f'attachment; filename="{filename}"'},
            )
        except HTTPException:
            raise
        except ValueError as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise HTTPException(status_code=400, detail=str(e))
        except AppendixGenerationError as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise HTTPException(
                status_code=502,
                detail=f"Document generation failed (model output): {str(e)[:400]}",
            )
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate document: {str(e)[:500]}",
            )


# ============================================================
# Transition to CS (Knowledge Transfer markdown)
# ============================================================


@app.post("/api/transition-to-cs")
async def transition_to_cs_deliverable(request: TransitionToCSRequest) -> TransitionToCSResponse:
    """
    Fetch ProspectOverview from BigQuery (with optional Gong MCP enrichment),
    then run the LLM to produce an internal Knowledge Transfer markdown document.
    """
    with tracer.start_as_current_span(
        f"{SPAN_PREFIX}.transition_to_cs",
        attributes={
            "transition.account_name": request.account_name.strip(),
            "openinference.span.kind": "chain",
        },
    ) as span:
        try:
            if not bq_client:
                span.set_status(Status(StatusCode.ERROR, "BigQuery unavailable"))
                raise HTTPException(
                    status_code=503,
                    detail="BigQuery client not available. For local development, run: gcloud auth application-default login",
                )

            span.add_event(
                "fetch_prospect_overview",
                {"account_name": request.account_name.strip()},
            )
            overview = bq_client.get_prospect_overview(
                account_name=request.account_name.strip(),
                domain=None,
                sfdc_account_id=None,
                manual_competitors=None,
            )
            overview = _with_optional_gong_mcp_enrichment(overview)

            payload = build_transition_document(
                overview=overview,
                manual_notes=request.manual_notes,
                llm_model=None,
            )
            span.set_attribute("transition.model", payload.get("model", ""))
            span.set_status(Status(StatusCode.OK))
            return TransitionToCSResponse.model_validate(payload)
        except HTTPException:
            raise
        except ValueError as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise HTTPException(status_code=400, detail=str(e)) from e
        except RuntimeError as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise HTTPException(
                status_code=502,
                detail=f"Transition document generation failed: {str(e)[:400]}",
            ) from e
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate transition document: {str(e)[:500]}",
            ) from e


if __name__ == "__main__":
    import uvicorn

    # Check if API key is configured
    if not os.getenv("ANTHROPIC_API_KEY"):
        logger.warning("ANTHROPIC_API_KEY not found - see .env.example for reference")

    port = int(os.getenv("PORT", 8080))
    logger.info("Starting Call Analyzer on http://localhost:%d (docs at /docs)", port)

    uvicorn.run(app, host="0.0.0.0", port=port)
