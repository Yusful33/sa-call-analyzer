import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from dotenv import load_dotenv
from models import AnalyzeRequest, AnalysisResult
from transcript_parser import TranscriptParser
from crew_analyzer import SACallAnalysisCrew

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="SA Call Analyzer - CrewAI Multi-Agent System",
    description="Analyze Solution Architect performance using Command of the Message framework with 4 specialized AI agents"
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the CrewAI analyzer
analyzer = SACallAnalysisCrew()
print("🤖 Using CrewAI Multi-Agent System (4 specialized agents)")
print("   1. 🔍 SA Identifier")
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
                <h1>SA Call Analyzer API</h1>
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
    Analyze a call transcript and provide actionable feedback for the SA.

    The transcript can be in any format - with or without speaker labels.
    If speaker labels exist, we'll use them. If not, we'll try to infer.
    """
    if not request.transcript.strip():
        raise HTTPException(status_code=400, detail="Transcript cannot be empty")

    try:
        # Parse the transcript
        parsed_lines, has_labels = TranscriptParser.parse(request.transcript)

        # Format for analysis
        formatted_transcript = TranscriptParser.format_for_analysis(parsed_lines)

        # Extract speakers if available
        speakers = TranscriptParser.extract_speakers(parsed_lines) if has_labels else []

        # Perform analysis
        result = analyzer.analyze_call(
            transcript=formatted_transcript,
            speakers=speakers,
            manual_sa=request.sa_name
        )

        return result

    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to parse AI response: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
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

    print("🚀 Starting SA Call Analyzer...")
    print("📝 Open http://localhost:8000 in your browser")
    print("📚 API docs available at http://localhost:8000/docs")

    uvicorn.run(app, host="0.0.0.0", port=8000)
