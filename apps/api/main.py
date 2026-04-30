import os
import json
import time
import base64
import asyncio
import contextvars
from pathlib import Path
from contextlib import nullcontext

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
    
    print(f"\n🔐 GCP Credentials Setup:")
    print(f"   GCP_CREDENTIALS_BASE64: {'SET' if gcp_creds_b64 else 'NOT SET'} ({len(gcp_creds_b64) if gcp_creds_b64 else 0} chars)")
    
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
            
            print(f"   ✅ Credentials decoded and written to {creds_path}")
            print(f"   GOOGLE_APPLICATION_CREDENTIALS={os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')}")
            
            # Verify the JSON is valid
            creds_dict = json.loads(creds_json)
            print(f"   Credential type: {creds_dict.get('type', 'unknown')}")
            
            return True
        except Exception as e:
            print(f"   ❌ Failed to decode credentials: {e}")
            return False
    else:
        # Check if GOOGLE_APPLICATION_CREDENTIALS is already set
        existing_creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if existing_creds and Path(existing_creds).exists():
            print(f"   Using existing credentials: {existing_creds}")
            return True
        else:
            print(f"   ⚠️  No GCP credentials available")
            return False

# Run credential setup before anything else
setup_gcp_credentials()

# Use environment variables directly (not .env file)
# This allows uv/venv to manage environment variables
# The .env file is kept for reference/documentation only

# DEBUG: Verify environment variables are set and optionally check .env file matches
env_path = REPO_ROOT / ".env"
arize_space_id_env = os.getenv("ARIZE_SPACE_ID")
arize_api_key_env = os.getenv("ARIZE_API_KEY")

print(f"\n🔍 Environment Variable Check:")
print(f"   ARIZE_API_KEY: {'SET' if arize_api_key_env else 'NOT SET'} ({len(arize_api_key_env) if arize_api_key_env else 0} chars)")
print(f"   ARIZE_SPACE_ID: {'SET' if arize_space_id_env else 'NOT SET'} ({len(arize_space_id_env) if arize_space_id_env else 0} chars)")

if arize_space_id_env:
    print(f"   Space ID value (first 20 chars): {arize_space_id_env[:20]}...")

# Optional: Verify .env file matches environment variables (for documentation)
if env_path.exists() and arize_space_id_env:
    try:
        with open(env_path, 'r') as f:
            for line in f:
                if line.startswith('ARIZE_SPACE_ID='):
                    env_file_value = line.split('=', 1)[1].strip()
                    if env_file_value != arize_space_id_env:
                        print(f"   ⚠️  WARNING: .env file Space ID doesn't match environment variable!")
                        print(f"      .env file: {env_file_value[:20]}...")
                        print(f"      env var:   {arize_space_id_env[:20]}...")
                        print(f"      Consider updating .env file to match your environment variables")
                    else:
                        print(f"   ✅ .env file matches environment variable")
                    break
    except Exception as e:
        print(f"   ⚠️  Could not verify .env file: {e}")

# Initialize observability BEFORE importing CrewAI
# This ensures our Arize TracerProvider is set up before CrewAI tries to set up its own
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

# Now import everything else (including CrewAI)
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, Response, StreamingResponse
from models import (
    AnalyzeRequest, AnalysisResult, RecapSlideData, ProspectTimeline,
    ProspectOverviewRequest, ProspectOverview, GeneratePocDocumentRequest,
    AccountSuggestionsRequest, AccountSuggestionsResponse,
)
from poc_document_generator import AppendixGenerationError, build_poc_document
from transcript_parser import TranscriptParser
from crew_analyzer import SACallAnalysisCrew
from gong_mcp_client import GongMCPClient
from prospect_timeline_analyzer import ProspectTimelineAnalyzer
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from pydantic import BaseModel
from typing import Optional

# Initialize FastAPI app
app = FastAPI(
    title="Call Analyzer - CrewAI Multi-Agent System",
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


# #region agent log
def _agent_log(location: str, message: str, data: dict, hypothesis_id: str = "H4"):
    import json
    from datetime import datetime
    try:
        with open("/Users/yusufcattaneo/Projects/.cursor/debug-24e5e3.log", "a") as f:
            f.write(json.dumps({"sessionId": "24e5e3", "hypothesisId": hypothesis_id, "location": location, "message": message, "data": data, "timestamp": datetime.utcnow().isoformat() + "Z"}) + "\n")
    except Exception:
        pass
# #endregion


@app.middleware("http")
async def arize_project_middleware(request, call_next):
    """Set the active Arize tracer provider by component so each of the 4 components writes to its own project."""
    if request.url.path.startswith("/api/"):
        component = get_component_for_path(request.url.path)
        provider = get_provider_for_component(component)
        if provider is not None:
            set_request_tracer_provider(provider)  # update proxy's current provider (instrumentors use the proxy)
        _agent_log("main.py:arize_project_middleware", "path and component", {"path": request.url.path, "component": component}, "H4")
    return await call_next(request)


# Initialize tracer for main API
tracer = trace.get_tracer("sa-call-analyzer-api")

# Verify tracer provider is still ours after CrewAI import
current_provider = trace.get_tracer_provider()
print(f"🔍 Tracer provider after imports: {type(current_provider).__name__}")
if tracer_provider:
    if current_provider == tracer_provider:
        print("   ✅ Arize tracer provider is still active")
    else:
        print(f"   ⚠️  WARNING: Tracer provider was overridden! Expected {type(tracer_provider).__name__}, got {type(current_provider).__name__}")

# Initialize the CrewAI analyzer
analyzer = SACallAnalysisCrew()

# Initialize Gong MCP client
try:
    gong_client = GongMCPClient()
    print("✅ Gong MCP client initialized")
except Exception as e:
    gong_client = None
    print(f"⚠️  Gong MCP client not available: {e}")

# Initialize BigQuery client for Prospect Overview
try:
    from bigquery_client import BigQueryClient
    bq_client = BigQueryClient()
    print("✅ BigQuery client initialized")
except Exception as e:
    bq_client = None
    print(f"⚠️  BigQuery client not available: {e}")


def _with_optional_gong_mcp_enrichment(overview: ProspectOverview) -> ProspectOverview:
    """When warehouse Gong context is thin, supplement from Gong MCP (GONG_MCP_URL)."""
    from gong_mcp_enrichment import maybe_enrich_overview_with_gong_mcp

    return maybe_enrich_overview_with_gong_mcp(overview, gong_client, bq_client)


print("🤖 Using CrewAI Multi-Agent System (4 specialized agents)")
print("   1. 🔍 Call Classifier")
print("   2. 🛠️ Technical Evaluator")
print("   3. 💡 Sales Methodology & Discovery Expert")
print("   4. 📝 Report Compiler")


@app.get("/")
async def root():
    """Serve the frontend HTML"""
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


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    return {
        "status": "healthy",
        "api_key_configured": bool(api_key)
    }


# Map retired Anthropic model IDs to current ones (avoids 404 on Railway/production).
MODEL_ID_ALIASES = {
    "claude-3-5-haiku-20241022": "claude-haiku-4-5",
    "claude-3-5-sonnet-20241022": "claude-sonnet-4-20250514",
}


def _resolve_model_id(model: Optional[str]) -> Optional[str]:
    """Return current model ID; map retired IDs so requests don't 404."""
    if not model:
        return None
    return MODEL_ID_ALIASES.get(model.strip(), model.strip())


@app.post("/api/analyze", response_model=AnalysisResult)
async def analyze_transcript(request: AnalyzeRequest):
    """
    Analyze a call transcript and provide actionable feedback for the sales rep.

    You can provide either:
    - transcript: Manual transcript text (with or without speaker labels)
    - gong_url: Gong call URL (will fetch transcript automatically via MCP)
    """
    # Add user/session tracking if available
    try:
        from tracing_enhancements import trace_with_metadata
        # Extract user/session info from request if available
        user_id = getattr(request, 'user_id', None) or os.getenv('USER_ID')
        session_id = getattr(request, 'session_id', None) or f"session_{int(time.time())}"
        metadata_context = trace_with_metadata(
            user_id=user_id,
            session_id=session_id,
            tags=["api", "call_analysis"],
            metadata={
                "request_id": f"req_{int(time.time())}",
                "model": request.model or "default",
            }
        )
    except ImportError:
        metadata_context = None
        user_id = None
        session_id = None
    
    # Use context manager if available
    context_manager = metadata_context if metadata_context else nullcontext()
    
    with context_manager:
        with tracer.start_as_current_span(
            f"{SPAN_PREFIX}.analyze",
            attributes={
                "request.input_type": "gong_url" if request.gong_url else "manual_transcript",
                "request.model": request.model or "default",
                "input.value": json.dumps({
                    "gong_url": request.gong_url,
                    "transcript": request.transcript[:1000] + "..." if request.transcript and len(request.transcript) > 1000 else request.transcript,
                    "model": request.model
                })[:2000],
                "input.mime_type": "application/json",
                "openinference.span.kind": "CHAIN",
            }
        ) as span:
            try:
                # Fetch transcript from Gong if URL is provided
                if request.gong_url:
                    span.add_event("fetching_from_gong", {
                        "gong.url": request.gong_url
                    })

                    if not gong_client:
                        span.set_status(Status(StatusCode.ERROR, "Gong MCP client not available"))
                        raise HTTPException(
                            status_code=503,
                            detail="Gong MCP client not available. Check that Gong MCP server is running."
                        )

                    try:
                        print(f"📞 Fetching transcript from Gong URL: {request.gong_url}")
                        # Extract call ID and fetch raw transcript data
                        call_id = gong_client.extract_call_id_from_url(request.gong_url)
                        transcript_data = gong_client.get_transcript(call_id)
                        raw_transcript = gong_client.format_transcript_for_analysis(transcript_data)
                        
                        # Get call date from Gong metadata
                        call_date = gong_client.get_call_date(call_id)
                        if call_date:
                            print(f"📅 Call date: {call_date}")
                            span.set_attribute("call.date", call_date)
                        
                        print(f"✅ Fetched {len(raw_transcript)} characters from Gong")
                        span.set_attribute("transcript.source", "gong")
                        span.set_attribute("transcript.raw_length", len(raw_transcript))
                        span.add_event("gong_fetch_success", {
                            "transcript.length": len(raw_transcript),
                            "call.date": call_date or "not_available"
                        })
                    except Exception as e:
                        span.set_status(Status(StatusCode.ERROR, f"Gong fetch failed: {str(e)}"))
                        span.record_exception(e)
                        raise HTTPException(
                            status_code=400,
                            detail=f"Failed to fetch transcript from Gong: {str(e)}"
                        )
                else:
                    raw_transcript = request.transcript
                    transcript_data = None  # No raw data for manual transcripts
                    call_date = ""  # No date for manual transcripts
                    span.set_attribute("transcript.source", "manual")
                    span.set_attribute("transcript.raw_length", len(raw_transcript))
                    span.add_event("using_manual_transcript")

                # Validate transcript is not empty
                if not raw_transcript or not raw_transcript.strip():
                    span.set_status(Status(StatusCode.ERROR, "Empty transcript"))
                    raise HTTPException(status_code=400, detail="Transcript cannot be empty")

                # Parse the transcript
                with tracer.start_as_current_span("parse_transcript") as parse_span:
                    parse_span.set_attribute("openinference.span.kind", "chain")
                    parsed_lines, has_labels = TranscriptParser.parse(raw_transcript)
                    parse_span.set_attribute("transcript.has_labels", has_labels)
                    parse_span.set_attribute("transcript.line_count", len(parsed_lines))
                    span.add_event("transcript_parsed", {
                        "has_labels": has_labels,
                        "line_count": len(parsed_lines)
                    })

                # Format for analysis
                with tracer.start_as_current_span("format_for_analysis") as format_span:
                    format_span.set_attribute("openinference.span.kind", "chain")
                    formatted_transcript = TranscriptParser.format_for_analysis(parsed_lines)
                    format_span.set_attribute("transcript.formatted_length", len(formatted_transcript))

                # Extract speakers if available
                speakers = TranscriptParser.extract_speakers(parsed_lines) if has_labels else []
                span.set_attribute("transcript.speaker_count", len(speakers))
                if speakers:
                    span.set_attribute("transcript.speakers", ", ".join(speakers))

                span.add_event("starting_crew_analysis", {
                    "transcript.length": len(formatted_transcript),
                    "speaker.count": len(speakers)
                })

                # Perform analysis (run in thread with 5min timeout - LLM calls can be slow,
                # but technical + sales crews now run in parallel for ~2x speedup)
                try:
                    model = _resolve_model_id(request.model) or request.model
                    result = await asyncio.wait_for(
                        asyncio.to_thread(
                            analyzer.analyze_call,
                            transcript=formatted_transcript,
                            speakers=speakers,
                            transcript_data=transcript_data,
                            call_date=call_date,
                            model=model
                        ),
                        timeout=300.0
                    )
                except asyncio.TimeoutError:
                    span.set_status(Status(StatusCode.ERROR, "Analysis timed out after 5 minutes"))
                    raise HTTPException(
                        status_code=504,
                        detail="Analysis timed out after 5 minutes. Try a shorter transcript or a faster model (e.g. GPT-4o Mini)."
                    )

                span.set_attribute("analysis.insight_count", len(result.top_insights))
                span.set_attribute("analysis.strength_count", len(result.strengths))
                span.set_attribute("analysis.improvement_count", len(result.improvement_areas))

                # OpenInference output - the complete analysis result
                span.set_attribute("output.value", json.dumps({
                    "call_summary": result.call_summary,
                    "top_insights": [
                        {
                            "category": insight.category,
                            "severity": insight.severity,
                            "timestamp": insight.timestamp,
                            "conversation_snippet": insight.conversation_snippet,
                            "what_happened": insight.what_happened,
                            "why_it_matters": insight.why_it_matters,
                            "better_approach": insight.better_approach
                        }
                        for insight in result.top_insights
                    ],
                    "strengths": result.strengths,
                    "improvement_areas": result.improvement_areas,
                    "key_moments": result.key_moments
                }, indent=2))
                span.set_attribute("output.mime_type", "application/json")

                span.set_status(Status(StatusCode.OK))
                span.add_event("analysis_complete", {
                    "insight_count": len(result.top_insights)
                })

                # Force flush spans to ensure they're sent to Arize immediately
                from observability import force_flush_spans
                force_flush_spans()

                return result

            except HTTPException:
                raise
            except json.JSONDecodeError as e:
                span.set_status(Status(StatusCode.ERROR, f"JSON parse error: {str(e)}"))
                span.record_exception(e)
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to parse AI response: {str(e)}"
                )
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise HTTPException(
                    status_code=500,
                    detail=f"Analysis failed: {str(e)}"
                )


@app.post("/api/generate-recap-slide")
async def generate_recap_slide(recap_data: RecapSlideData):
    """
    Generate a PowerPoint presentation with the recap data.
    
    Returns the PowerPoint file as a download.
    """
    with tracer.start_as_current_span(f"{SPAN_PREFIX}.generate_recap_slide") as span:
        span.set_attribute("openinference.span.kind", "CHAIN")
        
        try:
            from recap_generator import generate_recap_slide as create_slide
            
            # Generate the PowerPoint file
            pptx_bytes = create_slide(recap_data)
            
            # Create filename with date and customer name
            customer_name = recap_data.customer_name or "Call"
            call_date = recap_data.call_date or ""
            
            print(f"📁 Generating filename - Customer: '{customer_name}', Date: '{call_date}'")
            
            safe_name = "".join(c for c in customer_name if c.isalnum() or c in (' ', '-', '_')).strip()
            safe_name = safe_name.replace(' ', '_')
            
            # Include date in filename if available (date first, then name)
            if call_date:
                safe_date = "".join(c for c in call_date if c.isalnum() or c in (' ', '-', '_')).strip()
                safe_date = safe_date.replace(' ', '_')
                filename = f"Recap_{safe_date}_{safe_name}.pptx"
            else:
                filename = f"Recap_{safe_name}.pptx"
            
            print(f"📁 Final filename: {filename}")
            
            span.set_attribute("presentation.filename", filename)
            span.set_attribute("presentation.size_bytes", len(pptx_bytes))
            span.set_status(Status(StatusCode.OK))
            
            return Response(
                content=pptx_bytes,
                media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                headers={
                    "Content-Disposition": f'attachment; filename="{filename}"'
                }
            )
            
        except ImportError as e:
            span.set_status(Status(StatusCode.ERROR, f"Import error: {str(e)}"))
            raise HTTPException(
                status_code=503,
                detail="PowerPoint dependencies not installed. Please install python-pptx."
            )
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate recap slide: {str(e)}"
            )


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


class AnalyzeProspectRequest(BaseModel):
    """Request to analyze all calls for a prospect"""
    prospect_name: str  # Name to search for (e.g., "John Smith" or "Gong")
    fuzzy_threshold: Optional[float] = 0.85  # Similarity threshold for name matching
    from_date: Optional[str] = None  # ISO format start date
    to_date: Optional[str] = None  # ISO format end date
    model: Optional[str] = None  # LLM model to use


@app.post("/api/analyze-prospect", response_model=ProspectTimeline)
async def analyze_prospect(request: AnalyzeProspectRequest):
    """
    Analyze all calls for a prospect and build a cumulative timeline.
    
    Searches Gong for all calls where the prospect name matches any participant,
    analyzes each call with context from prior calls, and returns a timeline view.
    """
    with tracer.start_as_current_span(
        f"{SPAN_PREFIX}.analyze_prospect",
        attributes={
            "prospect.name": request.prospect_name,
            "fuzzy.threshold": request.fuzzy_threshold or 0.85,
            "openinference.span.kind": "chain",
        }
    ) as span:
        try:
            if not gong_client:
                span.set_status(Status(StatusCode.ERROR, "Gong MCP client not available"))
                raise HTTPException(
                    status_code=503,
                    detail="Gong MCP client not available. Check that Gong MCP server is running."
                )
            
            # Initialize timeline analyzer
            timeline_analyzer = ProspectTimelineAnalyzer(
                gong_client=gong_client,
                analyzer=analyzer
            )
            
            span.add_event("analyzing_prospect_timeline", {
                "prospect_name": request.prospect_name,
                "date_range": f"{request.from_date} to {request.to_date}" if request.from_date or request.to_date else "all"
            })
            
            # Analyze prospect timeline
            timeline = timeline_analyzer.analyze_prospect_timeline(
                prospect_name=request.prospect_name,
                from_date=request.from_date,
                to_date=request.to_date,
                fuzzy_threshold=request.fuzzy_threshold or 0.85,
                model=request.model
            )
            
            span.set_attribute("timeline.calls_count", len(timeline.calls))
            span.set_attribute("timeline.matched_names", ", ".join(timeline.matched_participant_names))
            span.set_status(Status(StatusCode.OK))
            
            # Force flush spans
            from observability import force_flush_spans
            force_flush_spans()
            
            return timeline
            
        except ValueError as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise HTTPException(
                status_code=404,
                detail=str(e)
            )
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to analyze prospect timeline: {str(e)}"
            )


@app.post("/api/analyze-prospect-stream")
async def analyze_prospect_stream(request: AnalyzeProspectRequest):
    """
    Analyze all calls for a prospect with real-time progress updates via SSE.

    Returns a Server-Sent Events stream with progress updates as each call is analyzed.
    Progress events include stage, message, and percentage completion.

    Final event contains the complete ProspectTimeline result.
    """
    if not gong_client:
        raise HTTPException(
            status_code=503,
            detail="Gong MCP client not available. Check that Gong MCP server is running."
        )

    # Initialize timeline analyzer
    timeline_analyzer = ProspectTimelineAnalyzer(
        gong_client=gong_client,
        analyzer=analyzer
    )

    def generate_events():
        """Generator that yields SSE-formatted events.

        The OTel span lives INSIDE the generator so it stays active for the
        entire duration of the stream.  Previously the span wrapped the
        generator creation but exited before iteration began, which meant
        every child span created during streaming became a separate root
        trace.
        """
        with tracer.start_as_current_span(
            f"{SPAN_PREFIX}.analyze_prospect_stream",
            attributes={
                "prospect.name": request.prospect_name,
                "fuzzy.threshold": request.fuzzy_threshold or 0.85,
                "openinference.span.kind": "chain",
                "streaming": True,
            }
        ) as span:
            try:
                span.add_event("starting_sse_stream", {
                    "prospect_name": request.prospect_name
                })

                # Use fast mode for quick timeline summaries (1 LLM call per call vs 5)
                for event in timeline_analyzer.analyze_with_progress_fast(
                    prospect_name=request.prospect_name,
                    from_date=request.from_date,
                    to_date=request.to_date,
                    fuzzy_threshold=request.fuzzy_threshold or 0.85,
                    model=request.model
                ):
                    # Handle the final complete event specially - serialize the result
                    if event.get("type") == "complete" and "result" in event:
                        result = event["result"]
                        # Convert ProspectTimeline to dict for JSON serialization
                        event_data = {
                            "type": "complete",
                            "message": event.get("message", "Analysis complete"),
                            "progress": 100,
                            "result": result.model_dump() if hasattr(result, 'model_dump') else result.dict()
                        }
                        yield f"data: {json.dumps(event_data)}\n\n"
                    else:
                        # Regular progress event
                        yield f"data: {json.dumps(event)}\n\n"

                span.set_status(Status(StatusCode.OK))

            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                # Yield error event
                error_event = {
                    "type": "error",
                    "message": f"Analysis failed: {str(e)}"
                }
                yield f"data: {json.dumps(error_event)}\n\n"

        # Flush spans after the span context closes so they are exported
        # before the SSE connection terminates
        from observability import force_flush_spans
        force_flush_spans()

    return StreamingResponse(
        generate_events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
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
            
            span.add_event("fetching_prospect_overview", {
                "account_name": request.account_name or "",
                "domain": request.domain or "",
                "sfdc_id": request.sfdc_account_id or ""
            })
            
            # Fetch prospect overview from BigQuery
            overview = bq_client.get_prospect_overview(
                account_name=request.account_name,
                domain=request.domain,
                sfdc_account_id=request.sfdc_account_id,
                manual_competitors=request.manual_competitors
            )
            overview = _with_optional_gong_mcp_enrichment(overview)

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


class ClassifyDemoRequest(BaseModel):
    """Request to classify a prospect's use case for the external **arize-synthetic-demo** skill."""

    account_name: str
    additional_context: Optional[str] = None  # User scenario (e.g. HR chatbot 1:1 prep); used to pick use case
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
    import litellm

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

        response = litellm.completion(
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
        print(f"⚠️ LLM use-case classification failed: {e}")
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

    from synthetic_demo_skill import build_synthetic_demo_skill_hints

    skill_hints = SyntheticDemoSkillHints(**build_synthetic_demo_skill_hints(
        company_name=request.account_name,
        industry=industry,
        use_case=use_case,
        framework=framework,
        reasoning=reasoning,
        additional_context=getattr(request, "additional_context", None),
    ))

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
# Hypothesis Research (integrated from ae-hypothesis-tool)
# ============================================================

# Lazy-initialized hypothesis tool components
_hypothesis_agent = None


def _get_hypothesis_agent():
    """Lazy-init the LangGraph research agent."""
    global _hypothesis_agent
    if _hypothesis_agent is None:
        try:
            from hypothesis_tool.agents.research_agent import ResearchAgent
            from hypothesis_tool.clients.bigquery_client import BigQueryClient as HypBQClient
            from hypothesis_tool.config import get_settings as get_hyp_settings
            settings = get_hyp_settings()
            bq = None
            try:
                bq = HypBQClient(project_id=settings.bq_project_id)
            except Exception as e:
                print(f"⚠️ Hypothesis BQ client failed: {e}")
            _hypothesis_agent = ResearchAgent(bq_client=bq)
            print("✅ Hypothesis Research Agent initialized")
        except Exception as e:
            print(f"❌ Failed to init hypothesis agent: {e}")
            import traceback
            traceback.print_exc()
            raise
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
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"Hypothesis research init error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Research service unavailable: {str(e)[:200]}")

    # Ensure this request's traces go to the Arize "hypothesis-generator" project (explicit connection).
    hyp_provider = get_provider_for_component(COMPONENT_HYPOTHESIS)
    space_id_used = os.getenv("ARIZE_SPACE_ID", "")
    if hyp_provider is not None:
        set_request_tracer_provider(hyp_provider)
        print(f"[Arize] Hypothesis research → project: {PROJECT_HYPOTHESIS!r}, space_id: {space_id_used!r}")
    else:
        print("[Arize] WARNING: Hypothesis tracer provider not available (observability may be disabled).")

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
        _agent_log("main.py:hypothesis_research", "entry project", {"project": project_at_start}, "H1")
        if os.environ.get("ARIZE_TRACE_DEBUG", "").strip().lower() in ("1", "true", "yes"):
            ctx = span.get_span_context() if hasattr(span, "get_span_context") else None
            trace_id = format(ctx.trace_id, "032x") if ctx and hasattr(ctx, "trace_id") else None
            print(f"[ARIZE_TRACE_DEBUG] hypothesis_research → project={project_at_start!r} trace_id={trace_id}")
        try:
            result, reasoning = await agent.research(
                company_name=request.company_name.strip(),
                company_domain=request.company_domain.strip() if request.company_domain else None,
            )

            # Flush hypothesis provider so traces are exported before response (same pattern as demo)
            hyp_provider = get_provider_for_component(COMPONENT_HYPOTHESIS)
            if hyp_provider is not None and hasattr(hyp_provider, "force_flush"):
                try:
                    hyp_provider.force_flush(timeout_millis=15000)
                    _agent_log("main.py:hypothesis_research", "after flush", {"flushed": True}, "H2")
                except Exception as e:
                    _agent_log("main.py:hypothesis_research", "after flush", {"flushed": False, "error": str(e)}, "H2")

            # Convert pydantic model to dict for JSON response
            result_dict = result.model_dump() if hasattr(result, 'model_dump') else result
            return {
                "result": result_dict,
                "agent_reasoning": reasoning,
            }
        except HTTPException:
            raise
        except Exception as e:
            import traceback
            print(f"Hypothesis research error: {traceback.format_exc()}")
            error_detail = str(e)
            if "anthropic" in error_detail.lower() or "claude" in error_detail.lower():
                raise HTTPException(status_code=503, detail="AI service temporarily unavailable.")
            elif "brave" in error_detail.lower() or "search" in error_detail.lower():
                raise HTTPException(status_code=503, detail="Web search service unavailable.")
            elif "timeout" in error_detail.lower():
                raise HTTPException(status_code=504, detail="Request timed out. Try again.")
            else:
                raise HTTPException(status_code=500, detail=f"Research failed: {error_detail[:200]}")


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


_static_dir = BASE_DIR / "frontend" / "static"
if _static_dir.is_dir():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


if __name__ == "__main__":
    import uvicorn

    # Check if API key is configured
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("⚠️  WARNING: ANTHROPIC_API_KEY not found in environment variables")
        print("   Please create a .env file with your API key")
        print("   See .env.example for reference")

    port = int(os.getenv("PORT", 8080))
    print("🚀 Starting Call Analyzer...")
    print(f"📝 Open http://localhost:{port} in your browser")
    print(f"📚 API docs available at http://localhost:{port}/docs")

    uvicorn.run(app, host="0.0.0.0", port=port)
