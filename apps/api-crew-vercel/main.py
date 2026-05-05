"""CrewAI routes for Vercel deployment.

This is a separate Vercel project to stay within the 500MB Lambda limit.
The main API (apps/api) handles light routes, this handles CrewAI-heavy routes.
"""
import os
import sys
from pathlib import Path

# Add parent api directory to path for shared modules
API_DIR = Path(__file__).resolve().parent.parent / "api"
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from opentelemetry import trace

from observability import setup_observability
from gong_mcp_client import GongMCPClient
from crew_handlers import (
    analyze_prospect,
    analyze_prospect_stream,
    analyze_transcript,
    configure_crew_runtime,
    CrewRuntime,
    generate_recap_slide,
)
from models import AnalysisResult, ProspectTimeline

# Initialize observability
setup_observability(project_name="sa-call-analyzer-crew")
tracer = trace.get_tracer("sa-call-analyzer-crew")

# Initialize Gong client
gong_client = GongMCPClient()

# Create FastAPI app
app = FastAPI(
    title="Call Analyzer - CrewAI Routes",
    description="CrewAI-powered analysis routes (separate Vercel project for size limits)"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],
)

# Initialize CrewAI analyzer
from crew_analyzer import SACallAnalysisCrew

analyzer = SACallAnalysisCrew()
configure_crew_runtime(CrewRuntime(analyzer=analyzer, gong_client=gong_client, tracer=tracer))

# Register routes
app.post("/api/analyze", response_model=AnalysisResult)(analyze_transcript)
app.post("/api/generate-recap-slide")(generate_recap_slide)
app.post("/api/analyze-prospect", response_model=ProspectTimeline)(analyze_prospect)
app.post("/api/analyze-prospect-stream")(analyze_prospect_stream)


@app.get("/health")
async def health():
    """Health check for crew function."""
    return {"status": "healthy", "service": "id-pain-api-crew", "mode": "crewai"}


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "id-pain-api-crew",
        "routes": [
            "/api/analyze",
            "/api/generate-recap-slide", 
            "/api/analyze-prospect",
            "/api/analyze-prospect-stream",
            "/health"
        ]
    }
