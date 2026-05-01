"""Register LangGraph-backed call-analysis routes on the shared FastAPI `app`."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import FastAPI

from crew_handlers import (
    analyze_prospect,
    analyze_prospect_stream,
    analyze_transcript,
    configure_crew_runtime,
    CrewRuntime,
    generate_recap_slide,
)
from models import AnalysisResult, ProspectTimeline, RecapSlideData

if TYPE_CHECKING:
    from opentelemetry.trace import Tracer


def register_crew_routes(app: FastAPI, *, gong_client, tracer: "Tracer") -> None:
    from crew_analyzer import SACallAnalysisCrew

    analyzer = SACallAnalysisCrew()
    configure_crew_runtime(CrewRuntime(analyzer=analyzer, gong_client=gong_client, tracer=tracer))

    app.post("/api/analyze", response_model=AnalysisResult)(analyze_transcript)
    app.post("/api/generate-recap-slide")(generate_recap_slide)
    app.post("/api/analyze-prospect", response_model=ProspectTimeline)(analyze_prospect)
    app.post("/api/analyze-prospect-stream")(analyze_prospect_stream)
