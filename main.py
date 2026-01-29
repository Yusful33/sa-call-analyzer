import os
import json
import time
import base64
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
    with tracer.start_as_current_span(
        "analyze_prospect_stream_request",
        attributes={
            "prospect.name": request.prospect_name,
            "fuzzy.threshold": request.fuzzy_threshold or 0.85,
            "openinference.span.kind": "chain",
            "streaming": True,
        }
    ) as span:
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

        def generate_events():
            """Generator that yields SSE-formatted events."""
            try:
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

            except Exception as e:
                # Yield error event
                error_event = {
                    "type": "error",
                    "message": f"Analysis failed: {str(e)}"
                }
                yield f"data: {json.dumps(error_event)}\n\n"

        span.add_event("starting_sse_stream", {
            "prospect_name": request.prospect_name
        })

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
