"""Call-analysis HTTP handlers (mounted from `routes_crew`; LangGraph pipeline)."""

from __future__ import annotations

import asyncio
import json
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from fastapi import HTTPException
from fastapi.responses import Response, StreamingResponse
from model_routing import resolve_model_id
from models import (
    AnalyzeRequest,
    AnalysisResult,
    AnalyzeProspectRequest,
    ProspectTimeline,
    RecapSlideData,
)
from observability import SPAN_PREFIX
from opentelemetry.trace import Status, StatusCode
from transcript_parser import TranscriptParser
from prospect_timeline_analyzer import ProspectTimelineAnalyzer

if TYPE_CHECKING:
    from opentelemetry.trace import Tracer


@dataclass
class CrewRuntime:
    analyzer: Any
    gong_client: Any
    tracer: "Tracer"


_rt: Optional[CrewRuntime] = None


def configure_crew_runtime(rt: CrewRuntime) -> None:
    global _rt
    _rt = rt


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
        with _rt.tracer.start_as_current_span(
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

                    if not _rt.gong_client:
                        span.set_status(Status(StatusCode.ERROR, "Gong MCP client not available"))
                        raise HTTPException(
                            status_code=503,
                            detail="Gong MCP client not available. Check that Gong MCP server is running."
                        )

                    try:
                        print(f"📞 Fetching transcript from Gong URL: {request.gong_url}")
                        # Extract call ID and fetch raw transcript data
                        call_id = _rt.gong_client.extract_call_id_from_url(request.gong_url)
                        transcript_data = _rt.gong_client.get_transcript(call_id)
                        raw_transcript = _rt.gong_client.format_transcript_for_analysis(transcript_data)
                        
                        # Get call date from Gong metadata
                        call_date = _rt.gong_client.get_call_date(call_id)
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
                with _rt.tracer.start_as_current_span("parse_transcript") as parse_span:
                    parse_span.set_attribute("openinference.span.kind", "chain")
                    parsed_lines, has_labels = TranscriptParser.parse(raw_transcript)
                    parse_span.set_attribute("transcript.has_labels", has_labels)
                    parse_span.set_attribute("transcript.line_count", len(parsed_lines))
                    span.add_event("transcript_parsed", {
                        "has_labels": has_labels,
                        "line_count": len(parsed_lines)
                    })

                # Format for analysis
                with _rt.tracer.start_as_current_span("format_for_analysis") as format_span:
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
                    model = resolve_model_id(request.model) or request.model
                    result = await asyncio.wait_for(
                        asyncio.to_thread(
                            _rt.analyzer.analyze_call,
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


async def generate_recap_slide(recap_data: RecapSlideData):
    """
    Generate a PowerPoint presentation with the recap data.

    Returns the PowerPoint file as a download.
    """
    with _rt.tracer.start_as_current_span(f"{SPAN_PREFIX}.generate_recap_slide") as span:
        span.set_attribute("openinference.span.kind", "CHAIN")

        try:
            from recap_generator import generate_recap_slide as create_slide

            pptx_bytes = create_slide(recap_data)

            customer_name = recap_data.customer_name or "Call"
            call_date = recap_data.call_date or ""

            print(f"📁 Generating filename - Customer: '{customer_name}', Date: '{call_date}'")

            safe_name = "".join(c for c in customer_name if c.isalnum() or c in (" ", "-", "_")).strip()
            safe_name = safe_name.replace(" ", "_")

            if call_date:
                safe_date = "".join(c for c in call_date if c.isalnum() or c in (" ", "-", "_")).strip()
                safe_date = safe_date.replace(" ", "_")
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
                },
            )

        except ImportError as e:
            span.set_status(Status(StatusCode.ERROR, f"Import error: {str(e)}"))
            raise HTTPException(
                status_code=503,
                detail="PowerPoint dependencies not installed. Please install python-pptx.",
            ) from e
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate recap slide: {str(e)}",
            ) from e


async def analyze_prospect(request: AnalyzeProspectRequest):
    """
    Analyze all calls for a prospect and build a cumulative timeline.
    
    Searches Gong for all calls where the prospect name matches any participant,
    analyzes each call with context from prior calls, and returns a timeline view.
    """
    with _rt.tracer.start_as_current_span(
        f"{SPAN_PREFIX}.analyze_prospect",
        attributes={
            "prospect.name": request.prospect_name,
            "fuzzy.threshold": request.fuzzy_threshold or 0.85,
            "openinference.span.kind": "chain",
        }
    ) as span:
        try:
            if not _rt.gong_client:
                span.set_status(Status(StatusCode.ERROR, "Gong MCP client not available"))
                raise HTTPException(
                    status_code=503,
                    detail="Gong MCP client not available. Check that Gong MCP server is running."
                )
            
            # Initialize timeline analyzer
            timeline_analyzer = ProspectTimelineAnalyzer(
                gong_client=_rt.gong_client,
                analyzer=_rt.analyzer,
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


async def analyze_prospect_stream(request: AnalyzeProspectRequest):
    """
    Analyze all calls for a prospect with real-time progress updates via SSE.

    Returns a Server-Sent Events stream with progress updates as each call is analyzed.
    Progress events include stage, message, and percentage completion.

    Final event contains the complete ProspectTimeline result.
    """
    if not _rt.gong_client:
        raise HTTPException(
            status_code=503,
            detail="Gong MCP client not available. Check that Gong MCP server is running."
        )

    # Initialize timeline analyzer
    timeline_analyzer = ProspectTimelineAnalyzer(
        gong_client=_rt.gong_client,
        analyzer=_rt.analyzer,
    )

    def generate_events():
        """Generator that yields SSE-formatted events.

        The OTel span lives INSIDE the generator so it stays active for the
        entire duration of the stream.  Previously the span wrapped the
        generator creation but exited before iteration began, which meant
        every child span created during streaming became a separate root
        trace.
        """
        with _rt.tracer.start_as_current_span(
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

