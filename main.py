import os
import json
import time
import base64
import asyncio
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
        return FileResponse("frontend/index.html")
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

                # Perform analysis
                result = analyzer.analyze_call(
                    transcript=formatted_transcript,
                    speakers=speakers,
                    transcript_data=transcript_data,  # Pass raw data for hybrid sampling if available
                    call_date=call_date,  # Pass call date for recap generation (from Gong metadata)
                    model=request.model  # Pass selected model from UI
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

class GenerateDemoRequest(BaseModel):
    """Request to generate demo traces for a prospect."""
    account_name: str
    project_name: Optional[str] = None
    model: Optional[str] = None


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
        # Spotlight briefs and key points from recent Gong calls
        if overview.gong_summary and overview.gong_summary.recent_calls:
            for call in overview.gong_summary.recent_calls:
                if call.spotlight_brief:
                    signals.append(call.spotlight_brief)
                if call.spotlight_key_points:
                    for kp in call.spotlight_key_points:
                        if isinstance(kp, str):
                            signals.append(kp)
                        elif isinstance(kp, list):
                            signals.extend([str(item) for item in kp if item])

        # Aggregated key themes
        if overview.gong_summary and overview.gong_summary.key_themes:
            signals.extend(overview.gong_summary.key_themes)

        # Deal summary topics and current state
        if overview.sales_engagement and overview.sales_engagement.deal_summary:
            ds = overview.sales_engagement.deal_summary
            if ds.key_topics_discussed:
                signals.extend(ds.key_topics_discussed)
            if ds.current_state:
                signals.append(ds.current_state)

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

    # Always try LLM classification when we have any signals
    if signals:
        use_case, framework, reasoning = _classify_use_case_with_llm(signals)
        if use_case:
            return use_case, framework or "langgraph", reasoning

    # Fall back to industry-based heuristic only when we have nothing
    industry = overview.salesforce.industry if overview and overview.salesforce else None
    return _industry_heuristic(industry), "langgraph", None


def _classify_use_case_with_llm(signals: list[str]) -> tuple[str | None, str | None, str | None]:
    """Use Claude Haiku to classify use case and framework from prospect signals."""
    import litellm

    combined_text = "\n".join(signals[:20])  # Cap to control token usage

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
6. "multiturn-chatbot-with-tools" - Single chatbot or agent with tool calling where no specific pattern (SQL, RAG, multi-agent, classification, vision) dominates.
7. "generic" - When signals are ambiguous or don't clearly match any specific pattern above.

## Available Frameworks:
1. "langgraph" - LangGraph / LangChain graph-based agent orchestration. DEFAULT if no framework is mentioned.
2. "langchain" - Plain LangChain LCEL chains (prompt | llm | parser), no graph.
3. "crewai" - CrewAI multi-agent framework.
4. "adk" - Google ADK (Agent Development Kit), Google Agent Builder, Vertex AI agents, google-genai.

Look for explicit framework mentions in the signals. If the prospect mentions "ADK", "Google ADK", "Agent Development Kit", or "google-genai", use "adk". If they mention "CrewAI", use "crewai". If they mention "LangChain" without LangGraph, use "langchain". Default to "langgraph" if unclear.

## Conversation Signals:
{combined_text}

## Instructions:
Return ONLY a JSON object with:
- "use_case": one of the seven demo type strings
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
            "multiturn-chatbot-with-tools",
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
    else:
        return "retrieval-augmented-search"


def _sse_event(event: str, data: dict | str) -> str:
    """Format data as Server-Sent Event."""
    import json
    payload = json.dumps(data) if isinstance(data, dict) else data
    return f"event: {event}\ndata: {payload}\n\n"


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
            yield _sse_event("progress", {"message": "Fetching prospect profile from BigQuery..."})

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
            except asyncio.TimeoutError:
                yield _sse_event("progress", {"message": "BigQuery timeout (30s). Using default use case..."})
            except Exception:
                pass

            yield _sse_event("progress", {"message": "Analyzing prospect data to select best demo type..."})
            use_case, framework, use_case_reasoning = _infer_use_case(overview) if overview else (_industry_heuristic(None), "langgraph", None)
            model = request.model or "claude-opus-4-6"

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
                api_key = os.getenv("ARIZE_API_KEY")
                space_id = os.getenv("ARIZE_SPACE_ID")
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
                for i in range(num_traces):
                    yield _sse_event("progress", {"message": f"Generating trace {i + 1}/{num_traces}..."})

                    def _run_pipeline(trace_idx=i):
                        # Clear inherited parent span context so each run creates
                        # a true root trace (not orphaned under sa-call-analyzer)
                        ctx = trace.set_span_in_context(trace.INVALID_SPAN)
                        token = otel_context.attach(ctx)
                        try:
                            from arize_demo_traces.runners.registry import get_runner
                            runner = get_runner(framework, use_case)
                            return runner(model=model, guard=guard, tracer_provider=demo_provider)
                        finally:
                            otel_context.detach(token)

                    try:
                        result = await asyncio.wait_for(asyncio.to_thread(_run_pipeline), timeout=60.0)
                        all_results.append(result)
                    except (asyncio.TimeoutError, RuntimeError) as e:
                        # Stop generating if we hit cost guard or timeout
                        yield _sse_event("progress", {"message": f"Stopped at trace {i + 1}: {e}"})
                        break

                result = {"traces_generated": len(all_results), "results": all_results}
            except Exception as e:
                yield _sse_event("error", {"detail": str(e)})
                return
            finally:
                if demo_provider is not None:
                    try:
                        demo_provider.force_flush(timeout_millis=10000)
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

            space_id = os.getenv("ARIZE_SPACE_ID")
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
        api_key = os.getenv("ARIZE_API_KEY")
        space_id = os.getenv("ARIZE_SPACE_ID")
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

        # Step 1: Fetch prospect profile from BigQuery
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

        # Step 3: Run the real pipeline multiple times
        num_traces = 10
        model = request.model or "claude-opus-4-6"

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
        for i in range(num_traces):
            def _run_pipeline():
                from opentelemetry import context as otel_context
                ctx = trace.set_span_in_context(trace.INVALID_SPAN)
                token = otel_context.attach(ctx)
                try:
                    from arize_demo_traces.runners.registry import get_runner
                    runner = get_runner(framework, use_case)
                    return runner(model=model, guard=guard, tracer_provider=demo_provider)
                finally:
                    otel_context.detach(token)

            try:
                result = await asyncio.to_thread(_run_pipeline)
                all_results.append(result)
            except (asyncio.TimeoutError, RuntimeError):
                break
            except Exception as e:
                import traceback
                print(f"❌ Demo pipeline error: {traceback.format_exc()}")
                raise HTTPException(status_code=500, detail=f"Demo pipeline failed: {e}")

        result = {"traces_generated": len(all_results), "results": all_results}

        arize_url = None
        space_id_val = os.getenv("ARIZE_SPACE_ID")
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
        )
    finally:
        if demo_provider is not None:
            try:
                demo_provider.force_flush(timeout_millis=10000)
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
