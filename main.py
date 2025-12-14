import os
import json
from dotenv import load_dotenv

# Load environment variables FIRST
# override=True ensures .env file takes precedence over system env vars
load_dotenv(override=True)

# Initialize observability BEFORE importing CrewAI
# This ensures our Arize TracerProvider is set up before CrewAI tries to set up its own
from observability import setup_observability
tracer_provider = setup_observability(project_name="sa-call-analyzer")

# Now import everything else (including CrewAI)
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, Response
from models import AnalyzeRequest, AnalysisResult, RecapSlideData
from transcript_parser import TranscriptParser
from crew_analyzer import SACallAnalysisCrew
from gong_mcp_client import GongMCPClient
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

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
