import os
import json
import time
import base64
import asyncio
import contextvars
from pathlib import Path
from contextlib import nullcontext

# Get the directory where this file is located
BASE_DIR = Path(__file__).parent

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
            
            # Write to file
            creds_path = BASE_DIR / "gcp-credentials.json"
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
env_path = BASE_DIR / '.env'
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
from observability import setup_observability
tracer_provider = setup_observability(project_name="sa-call-analyzer")

# Now import everything else (including CrewAI)
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, Response, StreamingResponse
from models import (
    AnalyzeRequest, AnalysisResult, RecapSlideData, ProspectTimeline,
    ProspectOverviewRequest, ProspectOverview
)
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
)

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

print("🤖 Using CrewAI Multi-Agent System (4 specialized agents)")
print("   1. 🔍 Call Classifier")
print("   2. 🛠️ Technical Evaluator")
print("   3. 💡 Sales Methodology & Discovery Expert")
print("   4. 📝 Report Compiler")


@app.get("/")
async def root():
    """Serve the frontend HTML"""
    try:
        response = FileResponse("frontend/index.html")
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
            "analyze_call_request",
            attributes={
                "request.input_type": "gong_url" if request.gong_url else "manual_transcript",
                "request.model": request.model or "default",
                # OpenInference input - the API request
                "input.value": json.dumps({
                    "gong_url": request.gong_url,
                    "transcript": request.transcript[:1000] + "..." if request.transcript and len(request.transcript) > 1000 else request.transcript,
                    "model": request.model
                }),
                "input.mime_type": "application/json",
                # OpenInference span kind - this is a chain orchestrating the workflow
                "openinference.span.kind": "chain",
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
                    result = await asyncio.wait_for(
                        asyncio.to_thread(
                            analyzer.analyze_call,
                            transcript=formatted_transcript,
                            speakers=speakers,
                            transcript_data=transcript_data,
                            call_date=call_date,
                            model=request.model
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
    with tracer.start_as_current_span("generate_recap_slide") as span:
        span.set_attribute("openinference.span.kind", "chain")
        
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
        "get_calls_by_account",
        attributes={
            "account.name": request.account_name,
            "fuzzy.threshold": request.fuzzy_threshold or 0.85,
            "openinference.span.kind": "tool",
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
        "analyze_prospect_request",
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
            "analyze_prospect_stream_request",
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
        "get_prospect_overview",
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
            
            # Log results
            span.set_attribute("result.data_sources", ", ".join(overview.data_sources_available))
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
    {"value": "generic", "label": "Generic LLM Pipeline"},
]

AVAILABLE_FRAMEWORKS = [
    {"value": "langgraph", "label": "LangGraph"},
    {"value": "langchain", "label": "LangChain"},
    {"value": "crewai", "label": "CrewAI"},
    {"value": "adk", "label": "Google ADK"},
]


class ClassifyDemoRequest(BaseModel):
    """Request to classify a prospect's use case before demo generation."""
    account_name: str


class ClassifyDemoResponse(BaseModel):
    """Response from use-case classification."""
    use_case: str
    framework: str
    reasoning: Optional[str] = None
    industry: Optional[str] = None
    available_use_cases: list[dict]
    available_frameworks: list[dict]
    gong_calls_used: Optional[int] = None
    data_sources_note: Optional[str] = None


class GenerateDemoRequest(BaseModel):
    """Request to generate demo traces for a prospect."""
    account_name: str
    project_name: Optional[str] = None
    model: Optional[str] = None
    use_case: Optional[str] = None
    framework: Optional[str] = None
    arize_api_key: Optional[str] = None
    arize_space_id: Optional[str] = None


class GenerateDemoResponse(BaseModel):
    """Response from demo generation."""
    prospect_name: str
    industry: Optional[str] = None
    use_case: str
    framework: Optional[str] = None
    use_case_reasoning: Optional[str] = None
    model_used: str
    llm_calls_made: int
    project_name: str
    arize_url: Optional[str] = None
    result: Optional[dict] = None
    error_message: Optional[str] = None
    eval_created: Optional[bool] = None
    eval_message: Optional[str] = None


def _infer_use_case(overview) -> tuple[str, str, str | None]:
    """Infer the best demo use case and framework from full prospect data.

    Uses all available signals: Gong calls, deal summaries, customer notes,
    and account metadata (name, industry, description). Always attempts LLM
    classification when any signals exist — falls back to industry heuristic
    only when we have nothing to work with.

    Returns:
        Tuple of (use_case, framework, reasoning). framework defaults to "langgraph".
    """
    signals: list[str] = []

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

    # Keyword hints (general, not account-specific): make sure SQL/ADK mentions win.
    hint_use_case, hint_framework = _extract_use_case_framework_hints("\n".join(signals)) if signals else (None, None)
    if hint_use_case and hint_framework:
        return hint_use_case, hint_framework, "Detected explicit keywords for use case and framework in Gong/CRM signals."

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
    """Lightweight keyword hints from Gong/CRM text (not account-specific)."""
    t = (text or "").lower()

    use_case = None
    framework = None

    # --- use case hints ---
    if any(k in t for k in ["text-to-sql", "text to sql", "nl2sql", "natural language to sql", "generate sql", "sql query", "bigquery", "snowflake", "redshift", "postgres", "databricks sql", "warehouse", "semantic layer", "bi agent", "analytics query"]):
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
        model = os.environ.get("USE_CASE_CLASSIFICATION_MODEL", "claude-3-5-haiku-20241022")

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


def _build_prospect_demo_context(overview, account_name: str) -> dict | None:
    """Build a short prospect context dict for tailoring demo traces (e.g. text-to-SQL).

    Uses organization and call data: industry, key themes, deal state, account info.
    Returns None if overview is missing or empty.
    """
    if not overview or not account_name:
        return None
    parts = []
    industry = None
    if getattr(overview, "salesforce", None) and overview.salesforce:
        sf = overview.salesforce
        if sf.industry:
            industry = (sf.industry or "").strip()
        if sf.name:
            parts.append(f"Company: {sf.name}")
        if sf.description:
            parts.append(sf.description.strip()[:300])
    if getattr(overview, "gong_summary", None) and overview.gong_summary and overview.gong_summary.key_themes:
        parts.append("Key themes from calls: " + ", ".join(overview.gong_summary.key_themes[:5]))
    if getattr(overview, "sales_engagement", None) and overview.sales_engagement and overview.sales_engagement.deal_summary:
        ds = overview.sales_engagement.deal_summary
        if ds.current_state:
            parts.append(f"Deal context: {ds.current_state[:200]}")
        if ds.key_topics_discussed:
            parts.append("Topics discussed: " + ", ".join(ds.key_topics_discussed[:5]))
    summary = " ".join(parts).strip() if parts else None
    return {
        "account_name": account_name,
        "industry": industry or "General",
        "summary": summary or f"Prospect: {account_name}.",
    }


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


def _sse_event(event: str, data: dict | str) -> str:
    """Format data as Server-Sent Event."""
    import json
    payload = json.dumps(data) if isinstance(data, dict) else data
    return f"event: {event}\ndata: {payload}\n\n"


@app.post("/api/classify-demo", response_model=ClassifyDemoResponse)
async def classify_demo(request: ClassifyDemoRequest):
    """
    Classify a prospect's use case and framework from CRM/Gong data.
    Returns the classification along with all available options for user override.
    """
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

    use_case, framework, reasoning = (
        _infer_use_case(overview) if overview
        else (_industry_heuristic(None), "langgraph", None)
    )

    # Fallback if classification returned None (e.g. LLM parsing failed)
    if not use_case:
        use_case = _industry_heuristic(industry)
    if not framework:
        framework = "langgraph"

    return ClassifyDemoResponse(
        use_case=use_case,
        framework=framework,
        reasoning=reasoning,
        industry=industry,
        available_use_cases=AVAILABLE_USE_CASES,
        available_frameworks=AVAILABLE_FRAMEWORKS,
        gong_calls_used=gong_calls_used,
        data_sources_note=data_sources_note,
    )


@app.post("/api/generate-demo-stream")
async def generate_demo_stream(request: GenerateDemoRequest):
    """
    Stream demo generation with progress updates via Server-Sent Events.
    """
    import json
    tracer = trace.get_tracer(__name__)
    project_name = request.project_name or request.account_name.lower().replace(" ", "-") + "-demo"

    async def event_stream():
        try:
            overview = None
            industry = None
            # If use_case and framework are already provided (user confirmed via classify step),
            # skip BigQuery lookup and classification
            if request.use_case and request.framework:
                use_case = request.use_case
                framework = request.framework
                use_case_reasoning = "User confirmed"
            else:
                yield _sse_event("progress", {"message": "Fetching prospect profile from BigQuery..."})
                overview = None  # ensure defined before try (classification path)
                industry = None
                try:
                    def _bq_lookup():
                        from bigquery_client import BigQueryClient
                        bq = BigQueryClient()
                        return bq.get_prospect_overview(account_name=request.account_name)
                    overview = await asyncio.wait_for(asyncio.to_thread(_bq_lookup), timeout=30.0)
                    if overview and overview.salesforce:
                        industry = overview.salesforce.industry
                except asyncio.TimeoutError:
                    yield _sse_event("progress", {"message": "BigQuery timeout (30s). Using default use case..."})
                except Exception:
                    pass

                yield _sse_event("progress", {"message": "Analyzing prospect data to select best demo type..."})
                use_case, framework, use_case_reasoning = _infer_use_case(overview) if overview else (_industry_heuristic(None), "langgraph", None)

            # If we don't have overview yet (e.g. user confirmed use case), fetch for prospect context to tailor traces
            if overview is None and request.account_name:
                try:
                    def _bq_context():
                        from bigquery_client import BigQueryClient
                        bq = BigQueryClient()
                        return bq.get_prospect_overview(account_name=request.account_name)
                    overview = await asyncio.wait_for(asyncio.to_thread(_bq_context), timeout=15.0)
                except (asyncio.TimeoutError, Exception):
                    pass
            prospect_context = _build_prospect_demo_context(overview, request.account_name) if overview else None

            model = request.model or "gpt-5.2"

            yield _sse_event("progress", {"message": f"Use case: {use_case} | Framework: {framework}. Running LLM pipeline (model: {model})..."})

            num_traces = 10
            from arize_demo_traces.cost_guard import CostGuard
            from opentelemetry import context as otel_context
            # CrewAI uses more LLM calls per trace (multiple agents)
            calls_per_trace = 12 if framework == "crewai" or use_case == "multi-agent-orchestration" else 8
            guard = CostGuard(max_calls=num_traces * calls_per_trace)

            original_provider = trace.get_tracer_provider()
            demo_provider = None
            try:
                from arize.otel import register as arize_register
                from openinference.instrumentation.langchain import LangChainInstrumentor
                from openinference.instrumentation.openai import OpenAIInstrumentor
                from openinference.instrumentation.litellm import LiteLLMInstrumentor
                crewai_instrumentor = None
                if framework == "crewai":
                    try:
                        from openinference.instrumentation.crewai import CrewAIInstrumentor
                        crewai_instrumentor = CrewAIInstrumentor()
                    except ImportError:
                        pass
                api_key = request.arize_api_key or os.getenv("ARIZE_API_KEY")
                space_id = request.arize_space_id or os.getenv("ARIZE_SPACE_ID")
                if api_key and space_id:
                    demo_provider = arize_register(
                        space_id=space_id,
                        api_key=api_key,
                        project_name=project_name,
                        set_global_tracer_provider=False,
                    )
                    # Re-instrument libraries with demo provider so LLM calls
                    # are traced to the demo project instead of sa-call-analyzer
                    LangChainInstrumentor().uninstrument()
                    OpenAIInstrumentor().uninstrument()
                    LiteLLMInstrumentor().uninstrument()
                    LangChainInstrumentor().instrument(tracer_provider=demo_provider, skip_dep_check=True)
                    OpenAIInstrumentor().instrument(tracer_provider=demo_provider, skip_dep_check=True)
                    LiteLLMInstrumentor().instrument(tracer_provider=demo_provider, skip_dep_check=True)
                    if crewai_instrumentor:
                        try:
                            crewai_instrumentor.uninstrument()
                        except Exception:
                            pass
                        crewai_instrumentor.instrument(tracer_provider=demo_provider, skip_dep_check=True)

                # Generate multiple traces, each as a separate root trace
                # Pass demo_provider directly to runners (trace.set_tracer_provider
                # is a one-time setter and cannot override the app's global provider)
                all_results = []
                trace_error = None  # set when first trace fails so we can surface it in the response
                # Deterministic poisoned trace distribution (~30%)
                import random as _random
                bad_indices = set(_random.sample(range(num_traces), k=max(1, int(num_traces * 0.3))))
                for i in range(num_traces):
                    quality_label = "poisoned" if i in bad_indices else "good"
                    yield _sse_event("progress", {"message": f"Generating trace {i + 1}/{num_traces} ({quality_label})..."})

                    def _run_pipeline(trace_idx=i):
                        # Run pipeline in an isolated context so trace hierarchy is correct.
                        # asyncio.to_thread does NOT propagate OTel context to worker threads;
                        # we must run with explicit context to avoid orphaned/incomplete traces.
                        def _body():
                            ctx = trace.set_span_in_context(trace.INVALID_SPAN)
                            token = otel_context.attach(ctx)
                            try:
                                from arize_demo_traces.runners.registry import get_runner
                                from arize_demo_traces.eval_wrapper import run_with_evals
                                runner = get_runner(framework, use_case)
                                return run_with_evals(
                                    runner=runner,
                                    use_case=use_case,
                                    framework=framework,
                                    model=model,
                                    guard=guard,
                                    tracer_provider=demo_provider,
                                    force_bad=(trace_idx in bad_indices),
                                    prospect_context=prospect_context,
                                )
                            finally:
                                otel_context.detach(token)

                        # Run in fresh context so worker thread has proper OTel context from the start
                        run_ctx = contextvars.copy_context()
                        return run_ctx.run(_body)

                    try:
                        result = await asyncio.wait_for(asyncio.to_thread(_run_pipeline), timeout=120.0)
                        all_results.append(result)
                        # Flush after each trace so spans export before next run (avoids partial traces)
                        if demo_provider is not None:
                            try:
                                demo_provider.force_flush(timeout_millis=5000)
                            except Exception:
                                pass
                    except asyncio.TimeoutError:
                        trace_error = f"Trace {i + 1} timed out (120s)"
                        yield _sse_event("progress", {"message": f"Stopped at trace {i + 1}: timeout (120s)"})
                        break
                    except RuntimeError as e:
                        trace_error = f"Trace {i + 1}: {e or 'runtime error'}"
                        yield _sse_event("progress", {"message": f"Stopped at trace {i + 1}: {e or 'runtime error'}"})
                        break
                    except Exception as e:
                        import traceback
                        print(f"❌ Demo trace {i + 1} error: {traceback.format_exc()}")
                        trace_error = f"Trace {i + 1} failed: {type(e).__name__}: {e}"
                        yield _sse_event("progress", {"message": trace_error})
                        break

                result = {"traces_generated": len(all_results), "results": all_results}
                if trace_error is not None:
                    result["error"] = trace_error
            except Exception as e:
                import traceback
                print(f"❌ Demo pipeline error: {traceback.format_exc()}")
                yield _sse_event("error", {"detail": str(e)})
                return
            finally:
                if demo_provider is not None:
                    try:
                        demo_provider.force_flush(timeout_millis=30000)
                    except Exception:
                        pass
                    # Restore instrumentors to original provider
                    try:
                        LangChainInstrumentor().uninstrument()
                        OpenAIInstrumentor().uninstrument()
                        LiteLLMInstrumentor().uninstrument()
                        LangChainInstrumentor().instrument(tracer_provider=original_provider, skip_dep_check=True)
                        OpenAIInstrumentor().instrument(tracer_provider=original_provider, skip_dep_check=True)
                        LiteLLMInstrumentor().instrument(tracer_provider=original_provider, skip_dep_check=True)
                        if crewai_instrumentor:
                            crewai_instrumentor.uninstrument()
                    except Exception:
                        pass
                    demo_provider.shutdown()

            yield _sse_event("progress", {"message": "Pipeline complete. Sending traces to Arize..."})

            # Create online eval task and run backfill on newly sent traces
            eval_created = False
            eval_message = None
            space_id = request.arize_space_id or os.getenv("ARIZE_SPACE_ID")
            api_key_arize = request.arize_api_key or os.getenv("ARIZE_API_KEY")
            if space_id and api_key_arize and result.get("traces_generated", 0) > 0:
                yield _sse_event("progress", {"message": "Creating online evaluation task (runs on new spans continuously)..."})
                try:
                    from arize_demo_traces.online_evals import create_and_run_online_eval
                    eval_result = await asyncio.to_thread(
                        create_and_run_online_eval,
                        project_name=project_name,
                        use_case=use_case,
                        space_id=space_id,
                        api_key=api_key_arize,
                        delay_seconds_before_backfill=30,
                        minutes_back=60,
                        max_spans=500,
                    )
                    if eval_result.get("success"):
                        eval_created = True
                        eval_label = eval_result.get("eval_name", eval_result.get("task_name"))
                        msg = f"Online eval '{eval_label}' created. Backfill run on recent spans (run_id={eval_result.get('run_id') or 'n/a'}). New traces will be evaluated automatically."
                        if eval_result.get("backfill_task_error"):
                            err = eval_result["backfill_task_error"]
                            err_msg = err.get("message", err) if isinstance(err, dict) else str(err)
                            msg += f" Backfill note: {err_msg}"
                            eval_message = f"Task created; backfill had an issue: {err_msg}. You can re-run the eval from Arize Online Evals."
                        elif eval_result.get("note"):
                            msg += f" {eval_result['note']}"
                            eval_message = eval_result.get("note")
                        else:
                            eval_message = "Eval task created and backfill started. Evaluations may take 1–2 min to appear in Arize."
                        yield _sse_event("progress", {"message": msg})
                    elif eval_result.get("error"):
                        eval_message = eval_result["error"]
                        yield _sse_event("progress", {"message": f"Online eval setup: {eval_result['error']}"})
                except Exception as e:
                    eval_message = f"Online eval setup failed: {e}"
                    yield _sse_event("progress", {"message": f"Online eval setup skipped: {e}"})
            elif result.get("traces_generated", 0) > 0 and (not space_id or not api_key_arize):
                eval_message = "No Arize Space ID or API key provided; online eval was not created. Set them in Custom Demo or env to enable evals."

            arize_url = None
            if space_id:
                arize_url = f"https://app.arize.com/?space_id={space_id}&project_id={project_name}"

            response = GenerateDemoResponse(
                prospect_name=request.account_name,
                industry=industry,
                use_case=use_case,
                framework=framework,
                use_case_reasoning=use_case_reasoning,
                model_used=model,
                llm_calls_made=guard.calls_made,
                project_name=project_name,
                arize_url=arize_url,
                result=result,
                error_message=result.get("error") if result else None,
                eval_created=eval_created,
                eval_message=eval_message,
            )
            yield _sse_event("done", response.model_dump())
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield _sse_event("error", {"detail": str(e)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/generate-demo", response_model=GenerateDemoResponse)
async def generate_demo_legacy(request: GenerateDemoRequest):
    """
    Generate tailored demo traces for a prospect.
    1. Fetch prospect profile from BigQuery
    2. Infer use case from industry/tech stack
    3. Run a real LLM pipeline
    4. Traces flow to Arize automatically via arize-otel (under demo project)
    """
    project_name = request.project_name or request.account_name.lower().replace(" ", "-") + "-demo"

    original_provider = trace.get_tracer_provider()
    demo_provider = None
    try:
        from arize.otel import register as arize_register
        from openinference.instrumentation.langchain import LangChainInstrumentor
        from openinference.instrumentation.openai import OpenAIInstrumentor
        from openinference.instrumentation.litellm import LiteLLMInstrumentor
        crewai_instrumentor = None
        api_key = request.arize_api_key or os.getenv("ARIZE_API_KEY")
        space_id = request.arize_space_id or os.getenv("ARIZE_SPACE_ID")
        if api_key and space_id:
            demo_provider = arize_register(
                space_id=space_id,
                api_key=api_key,
                project_name=project_name,
                set_global_tracer_provider=False,
            )
            # Re-instrument libraries with demo provider
            LangChainInstrumentor().uninstrument()
            OpenAIInstrumentor().uninstrument()
            LiteLLMInstrumentor().uninstrument()
            LangChainInstrumentor().instrument(tracer_provider=demo_provider, skip_dep_check=True)
            OpenAIInstrumentor().instrument(tracer_provider=demo_provider, skip_dep_check=True)
            LiteLLMInstrumentor().instrument(tracer_provider=demo_provider, skip_dep_check=True)

        overview = None
        industry = None
        # If use_case and framework are already provided (user confirmed), skip classification
        if request.use_case and request.framework:
            use_case = request.use_case
            framework = request.framework
            use_case_reasoning = "User confirmed"
        else:
            # Step 1: Fetch prospect profile from BigQuery (classification path)
            overview = None
            industry = None
            try:
                from bigquery_client import BigQueryClient
                bq = BigQueryClient()
                overview = bq.get_prospect_overview(account_name=request.account_name)
                if overview and overview.salesforce:
                    industry = overview.salesforce.industry
            except Exception as e:
                pass

            # Step 2: Infer use case from full prospect data (Gong calls, industry, etc.)
            use_case, framework, use_case_reasoning = _infer_use_case(overview) if overview else (_industry_heuristic(None), "langgraph", None)

        # If we don't have overview yet (e.g. user confirmed use case), fetch for prospect context
        if overview is None and request.account_name:
            try:
                from bigquery_client import BigQueryClient
                bq = BigQueryClient()
                overview = bq.get_prospect_overview(account_name=request.account_name)
            except Exception:
                pass
        prospect_context = _build_prospect_demo_context(overview, request.account_name) if overview else None

        # Step 3: Run the real pipeline multiple times
        num_traces = 10
        model = request.model or "gpt-5.2"

        from arize_demo_traces.cost_guard import CostGuard
        calls_per_trace = 12 if framework == "crewai" or use_case == "multi-agent-orchestration" else 8
        guard = CostGuard(max_calls=num_traces * calls_per_trace)

        # Instrument CrewAI if needed (after framework is known)
        if framework == "crewai" and demo_provider:
            try:
                from openinference.instrumentation.crewai import CrewAIInstrumentor
                crewai_instrumentor = CrewAIInstrumentor()
                try:
                    crewai_instrumentor.uninstrument()
                except Exception:
                    pass
                crewai_instrumentor.instrument(tracer_provider=demo_provider, skip_dep_check=True)
            except ImportError:
                pass

        all_results = []
        import random as _random
        bad_indices = set(_random.sample(range(num_traces), k=max(1, int(num_traces * 0.3))))
        for i in range(num_traces):
            def _run_pipeline(_trace_idx=i):
                from opentelemetry import context as otel_context
                def _body():
                    ctx = trace.set_span_in_context(trace.INVALID_SPAN)
                    token = otel_context.attach(ctx)
                    try:
                        from arize_demo_traces.runners.registry import get_runner
                        from arize_demo_traces.eval_wrapper import run_with_evals
                        runner = get_runner(framework, use_case)
                        return run_with_evals(
                            runner=runner,
                            use_case=use_case,
                            framework=framework,
                            model=model,
                            guard=guard,
                            tracer_provider=demo_provider,
                            force_bad=(_trace_idx in bad_indices),
                            prospect_context=prospect_context,
                        )
                    finally:
                        otel_context.detach(token)
                return contextvars.copy_context().run(_body)

            try:
                result = await asyncio.to_thread(_run_pipeline)
                all_results.append(result)
                if demo_provider is not None:
                    try:
                        demo_provider.force_flush(timeout_millis=5000)
                    except Exception:
                        pass
            except (asyncio.TimeoutError, RuntimeError):
                break
            except Exception as e:
                import traceback
                print(f"❌ Demo pipeline error: {traceback.format_exc()}")
                raise HTTPException(status_code=500, detail=f"Demo pipeline failed: {e}")

        result = {"traces_generated": len(all_results), "results": all_results}

        eval_created = False
        eval_message = None
        space_id_val = request.arize_space_id or os.getenv("ARIZE_SPACE_ID")
        api_key_val = request.arize_api_key or os.getenv("ARIZE_API_KEY")
        if space_id_val and api_key_val and result.get("traces_generated", 0) > 0:
            try:
                from arize_demo_traces.online_evals import create_and_run_online_eval
                eval_result = await asyncio.to_thread(
                    create_and_run_online_eval,
                    project_name=project_name,
                    use_case=use_case,
                    space_id=space_id_val,
                    api_key=api_key_val,
                    delay_seconds_before_backfill=30,
                    minutes_back=60,
                    max_spans=500,
                )
                if eval_result.get("success"):
                    eval_created = True
                    if eval_result.get("backfill_task_error"):
                        err = eval_result["backfill_task_error"]
                        err_msg = err.get("message", err) if isinstance(err, dict) else str(err)
                        eval_message = f"Task created; backfill had an issue: {err_msg}. Re-run the eval from Arize Online Evals."
                    else:
                        eval_message = "Eval task created and backfill started. Evaluations may take 1–2 min to appear in Arize."
                elif eval_result.get("error"):
                    eval_message = eval_result["error"]
            except Exception as e:
                eval_message = f"Online eval setup failed: {e}"
        elif result.get("traces_generated", 0) > 0 and (not space_id_val or not api_key_val):
            eval_message = "No Arize Space ID or API key provided; online eval was not created."

        arize_url = None
        if space_id_val:
            arize_url = f"https://app.arize.com/?space_id={space_id_val}&project_id={project_name}"

        return GenerateDemoResponse(
            prospect_name=request.account_name,
            industry=industry,
            use_case=use_case,
            use_case_reasoning=use_case_reasoning,
            model_used=model,
            llm_calls_made=guard.calls_made,
            project_name=project_name,
            arize_url=arize_url,
            framework=framework,
            result=result,
            eval_created=eval_created,
            eval_message=eval_message,
        )
    finally:
        if demo_provider is not None:
            try:
                demo_provider.force_flush(timeout_millis=30000)
            except Exception:
                pass
            # Restore instrumentors to original provider
            try:
                LangChainInstrumentor().uninstrument()
                OpenAIInstrumentor().uninstrument()
                LiteLLMInstrumentor().uninstrument()
                LangChainInstrumentor().instrument(tracer_provider=original_provider, skip_dep_check=True)
                OpenAIInstrumentor().instrument(tracer_provider=original_provider, skip_dep_check=True)
                LiteLLMInstrumentor().instrument(tracer_provider=original_provider, skip_dep_check=True)
                if crewai_instrumentor:
                    crewai_instrumentor.uninstrument()
            except Exception:
                pass
            demo_provider.shutdown()


# ============================================================
# Export Customer Demo Script
# ============================================================

@app.get("/api/export-script")
async def export_script(
    use_case: str,
    framework: str,
    model: str = "gpt-4o-mini",
    project_name: str = "customer-demo",
):
    """
    Generate and download a standalone Python demo script.

    The script includes all necessary code inlined so the customer can
    run it in a fresh environment with just `pip install` + env vars.
    """
    from scripts.generate_customer_script import generate_script

    try:
        script_content = generate_script(
            use_case=use_case,
            framework=framework,
            model=model,
            project_name=project_name,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate script: {e}")

    safe_name = "".join(c for c in project_name if c.isalnum() or c in ('-', '_')).strip()
    filename = f"{safe_name}_demo.py" if safe_name else "demo.py"

    return Response(
        content=script_content,
        media_type="text/x-python",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


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
    agent = _get_hypothesis_agent()

    if not request.company_name or len(request.company_name.strip()) < 2:
        raise HTTPException(status_code=400, detail="Company name must be at least 2 characters.")

    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span(
        f"hypothesis_research:{request.company_name}",
        attributes={
            "openinference.span.kind": "CHAIN",
            "input.value": f"Research {request.company_name}",
            "company.name": request.company_name,
        },
    ):
        try:
            result, reasoning = await agent.research(
                company_name=request.company_name.strip(),
                company_domain=request.company_domain.strip() if request.company_domain else None,
            )

            # Convert pydantic model to dict for JSON response
            result_dict = result.model_dump() if hasattr(result, 'model_dump') else result
            return {
                "result": result_dict,
                "agent_reasoning": reasoning,
            }
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
                raise HTTPException(status_code=500, detail=f"Research failed: {error_detail}")


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
