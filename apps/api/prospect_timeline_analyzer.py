"""
Prospect Timeline Analyzer

Analyzes multiple calls from the same prospect and builds a cumulative timeline view.
"""
import json
from typing import List, Dict, Optional, Any, Generator, Union
from datetime import datetime
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from models import (
    ProspectTimeline, CallTimelineEntry, AnalysisResult,
    AccountAnalysisSummary, AggregatedActionableInsight,
    CommandOfMessageScore, SAPerformanceMetrics, RecapSlideData,
    QuickCallSummary
)
from gong_mcp_client import GongMCPClient
from crew_analyzer import SACallAnalysisCrew


# Type alias for progress events
ProgressEvent = Dict[str, Any]


class ProspectTimelineAnalyzer:
    """Analyzes multiple calls from a prospect and builds a cumulative timeline."""
    
    def __init__(self, gong_client: GongMCPClient, analyzer: SACallAnalysisCrew):
        """
        Initialize the prospect timeline analyzer.
        
        Args:
            gong_client: GongMCPClient instance for fetching calls
            analyzer: SACallAnalysisCrew instance for analyzing calls
        """
        self.gong_client = gong_client
        self.analyzer = analyzer
        self.tracer = trace.get_tracer("prospect-timeline-analyzer")
    
    def analyze_prospect_timeline(
        self,
        prospect_name: str,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        fuzzy_threshold: float = 0.85,
        model: Optional[str] = None
    ) -> ProspectTimeline:
        """
        Analyze all calls for a prospect and build a cumulative timeline.
        
        Args:
            prospect_name: Name of the prospect to search for
            from_date: Optional start date for filtering calls
            to_date: Optional end date for filtering calls
            fuzzy_threshold: Similarity threshold for name matching (default 0.85)
            model: Optional LLM model to use for analysis
            
        Returns:
            ProspectTimeline with all calls and cumulative insights
        """
        with self.tracer.start_as_current_span(
            "analyze_prospect_timeline",
            attributes={
                "prospect.name": prospect_name,
                "openinference.span.kind": "chain",
            }
        ) as span:
            try:
                # Fetch all matching calls
                matching_calls = self.gong_client.get_calls_by_prospect_name(
                    prospect_name=prospect_name,
                    from_date=from_date,
                    to_date=to_date,
                    fuzzy_threshold=fuzzy_threshold
                )
                
                span.set_attribute("calls.found", len(matching_calls))
                
                if not matching_calls:
                    span.set_status(Status(StatusCode.ERROR, "No matching calls found"))
                    raise ValueError(f"No calls found for prospect: {prospect_name}")
                
                # Sort calls chronologically by date
                sorted_calls = self._sort_calls_by_date(matching_calls)

                # Extract matched participant names
                matched_names = self._extract_matched_names(sorted_calls)

                # Pre-fetch all transcripts in parallel for better performance
                call_ids = [call.get("id") for call in sorted_calls if call.get("id")]
                prefetched_transcripts = self.gong_client.get_transcripts_parallel(call_ids)

                # Analyze each call with cumulative context
                timeline_entries = []
                prior_insights = []  # Accumulate insights from prior calls

                for idx, call in enumerate(sorted_calls):
                    call_id = call.get("id")
                    call_date = call.get("scheduled") or call.get("started") or call.get("date", "")

                    span.add_event("analyzing_call", {
                        "call_id": call_id,
                        "call_index": idx + 1,
                        "total_calls": len(sorted_calls)
                    })

                    # Use pre-fetched transcript
                    try:
                        transcript_data = prefetched_transcripts.get(call_id)
                        if not transcript_data:
                            # Fallback to fetching if pre-fetch failed
                            transcript_data = self.gong_client.get_transcript(call_id)
                        formatted_transcript = self.gong_client.format_transcript_for_analysis(transcript_data)

                        # Format call date from call object (no API call needed)
                        formatted_date = self._format_date_string(call_date)

                        # Analyze call with prior context
                        analysis_result = self.analyzer.analyze_call(
                            transcript=formatted_transcript,
                            speakers=[],  # Will be extracted from transcript
                            transcript_data=transcript_data,
                            call_date=formatted_date,
                            model=model,
                            prior_call_insights=prior_insights if prior_insights else None
                        )
                        
                        # Extract key insights from this call
                        key_insights = self._extract_key_insights(analysis_result)
                        
                        # Determine progression indicators
                        progression_indicators = self._determine_progression(
                            key_insights,
                            prior_insights
                        )
                        
                        # Create timeline entry
                        call_type = None
                        if analysis_result.call_classification:
                            call_type = analysis_result.call_classification.call_type.value
                        
                        entry = CallTimelineEntry(
                            call_id=call_id,
                            call_date=formatted_date,
                            call_title=call.get("title"),
                            call_type=call_type,
                            call_url=call.get("url"),
                            analysis=analysis_result,
                            key_insights=key_insights,
                            progression_indicators=progression_indicators
                        )
                        
                        timeline_entries.append(entry)
                        
                        # Accumulate insights for next call
                        prior_insights.append({
                            "call_date": formatted_date,
                            "call_type": call_type,
                            "insights": key_insights,
                            "summary": analysis_result.call_summary,
                            "top_insights": [
                                {
                                    "category": insight.category,
                                    "what_happened": insight.what_happened,
                                    "why_it_matters": insight.why_it_matters
                                }
                                for insight in analysis_result.top_insights[:5]  # Top 5 insights
                            ]
                        })
                        
                    except Exception as e:
                        span.add_event("call_analysis_failed", {
                            "call_id": call_id,
                            "error": str(e)
                        })
                        # Still create entry without analysis
                        entry = CallTimelineEntry(
                            call_id=call_id,
                            call_date=call_date,
                            call_title=call.get("title"),
                            call_url=call.get("url"),
                            key_insights=[f"Analysis failed: {str(e)}"]
                        )
                        timeline_entries.append(entry)
                        continue
                
                # Build cumulative insights
                cumulative_insights = self._build_cumulative_insights(timeline_entries)

                # Generate progression summary
                progression_summary = self._generate_progression_summary(timeline_entries)

                # Extract key themes
                key_themes = self._extract_key_themes(timeline_entries)

                # Generate next steps
                next_steps = self._generate_next_steps(timeline_entries)

                # Assess overall account health
                account_health = self._assess_account_health(timeline_entries)

                # Build aggregated account analysis (similar to single-call analysis but aggregated)
                account_analysis = self._build_account_analysis(timeline_entries)

                timeline = ProspectTimeline(
                    prospect_name=prospect_name,
                    matched_participant_names=matched_names,
                    calls=timeline_entries,
                    cumulative_insights=cumulative_insights,
                    progression_summary=progression_summary,
                    overall_account_health=account_health,
                    key_themes=key_themes,
                    next_steps=next_steps,
                    account_analysis=account_analysis
                )
                
                span.set_attribute("timeline.calls_count", len(timeline_entries))
                span.set_status(Status(StatusCode.OK))

                return timeline

            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

    def analyze_with_progress(
        self,
        prospect_name: str,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        fuzzy_threshold: float = 0.85,
        model: Optional[str] = None
    ) -> Generator[ProgressEvent, None, ProspectTimeline]:
        """
        Analyze all calls for a prospect with progress events.

        This is a generator that yields progress events as it processes,
        enabling real-time progress updates via SSE.

        Yields:
            ProgressEvent dicts with type, stage, message, progress (0-100)

        Returns:
            ProspectTimeline with all calls and cumulative insights (as final yield)
        """
        with self.tracer.start_as_current_span(
            "analyze_prospect_timeline_with_progress",
            attributes={
                "prospect.name": prospect_name,
                "openinference.span.kind": "chain",
            }
        ) as span:
            try:
                # Stage 1: Searching for calls (0-5%)
                yield {
                    "type": "progress",
                    "stage": "searching",
                    "message": f"Searching for calls matching '{prospect_name}'...",
                    "progress": 0
                }

                # Use generator for granular progress during search
                matching_calls = []
                for event in self.gong_client.get_calls_by_prospect_name_with_progress(
                    prospect_name=prospect_name,
                    from_date=from_date,
                    to_date=to_date,
                    fuzzy_threshold=fuzzy_threshold
                ):
                    stage = event.get("stage")

                    if stage == "querying_api":
                        yield {
                            "type": "progress",
                            "stage": "searching",
                            "message": event["message"],
                            "progress": 1
                        }
                    elif stage == "filtering":
                        total_calls = event.get("total_calls_to_check", 0)
                        yield {
                            "type": "progress",
                            "stage": "searching",
                            "message": event["message"],
                            "progress": 2,
                            "details": f"{total_calls} calls to check"
                        }
                    elif stage == "matching":
                        # Progress from 2% to 4% during matching
                        checked = event.get("checked", 0)
                        total = event.get("total", 1)
                        match_progress = 2 + (checked / total) * 2 if total > 0 else 3
                        yield {
                            "type": "progress",
                            "stage": "searching",
                            "message": event["message"],
                            "progress": round(match_progress, 1)
                        }
                    elif stage == "complete":
                        matching_calls = event.get("result", [])
                    elif stage == "error":
                        yield {
                            "type": "error",
                            "message": event["message"]
                        }
                        return

                span.set_attribute("calls.found", len(matching_calls))

                if not matching_calls:
                    span.set_status(Status(StatusCode.ERROR, "No matching calls found"))
                    yield {
                        "type": "error",
                        "message": f"No calls found for prospect: {prospect_name}"
                    }
                    return

                # Stage 2: Found calls
                yield {
                    "type": "progress",
                    "stage": "found",
                    "message": f"Found {len(matching_calls)} calls, sorting chronologically...",
                    "progress": 5,
                    "total_calls": len(matching_calls)
                }

                # Sort calls chronologically by date
                sorted_calls = self._sort_calls_by_date(matching_calls)

                # Extract matched participant names
                yield {
                    "type": "progress",
                    "stage": "extracting_names",
                    "message": "Extracting participant names...",
                    "progress": 6
                }
                matched_names = self._extract_matched_names(sorted_calls)

                # Pre-fetch all transcripts in parallel for better performance
                yield {
                    "type": "progress",
                    "stage": "prefetching_transcripts",
                    "message": f"Pre-fetching transcripts for {len(sorted_calls)} calls...",
                    "progress": 8
                }
                call_ids = [call.get("id") for call in sorted_calls if call.get("id")]
                prefetched_transcripts = self.gong_client.get_transcripts_parallel(call_ids)

                # Stage 3: Analyze each call with progress updates
                timeline_entries = []
                prior_insights = []
                total_calls = len(sorted_calls)

                for idx, call in enumerate(sorted_calls):
                    call_id = call.get("id")
                    call_title = call.get("title", "Untitled")
                    call_date = call.get("scheduled") or call.get("started") or call.get("date", "")

                    # Calculate progress: calls take up 10% to 90% of total progress
                    base_progress = 10
                    call_progress_range = 80  # 10% to 90%
                    current_progress = base_progress + (idx / total_calls) * call_progress_range

                    # Yield progress for starting this call
                    yield {
                        "type": "progress",
                        "stage": "analyzing",
                        "message": f"Analyzing call {idx + 1}/{total_calls}: {call_title}",
                        "progress": round(current_progress, 1),
                        "current_call": idx + 1,
                        "total_calls": total_calls,
                        "call_title": call_title,
                        "call_date": call_date
                    }

                    span.add_event("analyzing_call", {
                        "call_id": call_id,
                        "call_index": idx + 1,
                        "total_calls": total_calls
                    })

                    # Analyze transcript (already pre-fetched)
                    try:
                        # Use pre-fetched transcript (no API call needed)
                        yield {
                            "type": "progress",
                            "stage": "processing_transcript",
                            "message": f"Processing transcript for call {idx + 1}...",
                            "progress": round(current_progress + 2, 1),
                            "current_call": idx + 1
                        }

                        transcript_data = prefetched_transcripts.get(call_id)
                        if not transcript_data:
                            # Fallback to fetching if pre-fetch failed
                            transcript_data = self.gong_client.get_transcript(call_id)
                        formatted_transcript = self.gong_client.format_transcript_for_analysis(transcript_data)

                        # Format call date from call object (no API call needed)
                        formatted_date = self._format_date_string(call_date)

                        # Sub-stage: Running AI analysis
                        yield {
                            "type": "progress",
                            "stage": "running_analysis",
                            "message": f"Running AI analysis on call {idx + 1}...",
                            "progress": round(current_progress + 5, 1),
                            "current_call": idx + 1
                        }

                        # Analyze call with prior context
                        analysis_result = self.analyzer.analyze_call(
                            transcript=formatted_transcript,
                            speakers=[],
                            transcript_data=transcript_data,
                            call_date=formatted_date,
                            model=model,
                            prior_call_insights=prior_insights if prior_insights else None
                        )

                        # Extract key insights
                        key_insights = self._extract_key_insights(analysis_result)
                        progression_indicators = self._determine_progression(key_insights, prior_insights)

                        # Create timeline entry
                        call_type = None
                        if analysis_result.call_classification:
                            call_type = analysis_result.call_classification.call_type.value

                        entry = CallTimelineEntry(
                            call_id=call_id,
                            call_date=formatted_date,
                            call_title=call.get("title"),
                            call_type=call_type,
                            call_url=call.get("url"),
                            analysis=analysis_result,
                            key_insights=key_insights,
                            progression_indicators=progression_indicators
                        )

                        timeline_entries.append(entry)

                        # Accumulate insights for next call
                        prior_insights.append({
                            "call_date": formatted_date,
                            "call_type": call_type,
                            "insights": key_insights,
                            "summary": analysis_result.call_summary,
                            "top_insights": [
                                {
                                    "category": insight.category,
                                    "what_happened": insight.what_happened,
                                    "why_it_matters": insight.why_it_matters
                                }
                                for insight in analysis_result.top_insights[:5]
                            ]
                        })

                        # Yield call complete
                        call_end_progress = base_progress + ((idx + 1) / total_calls) * call_progress_range
                        yield {
                            "type": "progress",
                            "stage": "call_complete",
                            "message": f"Call {idx + 1}/{total_calls} complete",
                            "progress": round(call_end_progress, 1),
                            "current_call": idx + 1,
                            "total_calls": total_calls
                        }

                    except Exception as e:
                        span.add_event("call_analysis_failed", {
                            "call_id": call_id,
                            "error": str(e)
                        })

                        # Yield error for this call but continue
                        yield {
                            "type": "call_error",
                            "message": f"Analysis failed for call {idx + 1}: {str(e)}",
                            "current_call": idx + 1,
                            "call_title": call_title
                        }

                        # Still create entry without analysis
                        entry = CallTimelineEntry(
                            call_id=call_id,
                            call_date=call_date,
                            call_title=call.get("title"),
                            call_url=call.get("url"),
                            key_insights=[f"Analysis failed: {str(e)}"]
                        )
                        timeline_entries.append(entry)
                        continue

                # Stage 4: Build cumulative insights (90-100%)
                yield {
                    "type": "progress",
                    "stage": "finalizing",
                    "message": "Building cumulative insights...",
                    "progress": 91
                }
                cumulative_insights = self._build_cumulative_insights(timeline_entries)

                yield {
                    "type": "progress",
                    "stage": "finalizing",
                    "message": "Generating progression summary...",
                    "progress": 93
                }
                progression_summary = self._generate_progression_summary(timeline_entries)

                yield {
                    "type": "progress",
                    "stage": "finalizing",
                    "message": "Extracting key themes...",
                    "progress": 95
                }
                key_themes = self._extract_key_themes(timeline_entries)
                next_steps = self._generate_next_steps(timeline_entries)

                yield {
                    "type": "progress",
                    "stage": "finalizing",
                    "message": "Assessing account health...",
                    "progress": 97
                }
                account_health = self._assess_account_health(timeline_entries)

                yield {
                    "type": "progress",
                    "stage": "finalizing",
                    "message": "Building aggregated account analysis...",
                    "progress": 99
                }
                account_analysis = self._build_account_analysis(timeline_entries)

                # Build final timeline
                timeline = ProspectTimeline(
                    prospect_name=prospect_name,
                    matched_participant_names=matched_names,
                    calls=timeline_entries,
                    cumulative_insights=cumulative_insights,
                    progression_summary=progression_summary,
                    overall_account_health=account_health,
                    key_themes=key_themes,
                    next_steps=next_steps,
                    account_analysis=account_analysis
                )

                span.set_attribute("timeline.calls_count", len(timeline_entries))
                span.set_status(Status(StatusCode.OK))

                # Yield final complete event with result
                yield {
                    "type": "complete",
                    "message": "Analysis complete",
                    "progress": 100,
                    "result": timeline
                }

            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                yield {
                    "type": "error",
                    "message": f"Analysis failed: {str(e)}"
                }

    def analyze_with_progress_fast(
        self,
        prospect_name: str,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        fuzzy_threshold: float = 0.85,
        model: Optional[str] = None
    ) -> Generator[ProgressEvent, None, ProspectTimeline]:
        """
        Fast version of analyze_with_progress that produces quick summaries.

        Uses 1 LLM call per call instead of 5 (4 agents + recap).
        Ideal for quickly getting someone up to speed on an account.

        Yields:
            ProgressEvent dicts with type, stage, message, progress (0-100)

        Returns:
            ProspectTimeline with quick summaries instead of deep analysis
        """
        with self.tracer.start_as_current_span(
            "analyze_prospect_timeline_fast",
            attributes={
                "prospect.name": prospect_name,
                "mode": "fast",
                "openinference.span.kind": "chain",
            }
        ) as span:
            try:
                # Stage 1: Searching for calls (0-5%) - same as regular mode
                yield {
                    "type": "progress",
                    "stage": "searching",
                    "message": f"Searching for calls matching '{prospect_name}'...",
                    "progress": 0
                }

                # Use generator for granular progress during search
                matching_calls = []
                for event in self.gong_client.get_calls_by_prospect_name_with_progress(
                    prospect_name=prospect_name,
                    from_date=from_date,
                    to_date=to_date,
                    fuzzy_threshold=fuzzy_threshold
                ):
                    stage = event.get("stage")

                    if stage == "querying_api":
                        yield {
                            "type": "progress",
                            "stage": "searching",
                            "message": event["message"],
                            "progress": 1
                        }
                    elif stage == "filtering":
                        total_calls = event.get("total_calls_to_check", 0)
                        yield {
                            "type": "progress",
                            "stage": "searching",
                            "message": event["message"],
                            "progress": 2,
                            "details": f"{total_calls} calls to check"
                        }
                    elif stage == "matching":
                        checked = event.get("checked", 0)
                        total = event.get("total", 1)
                        match_progress = 2 + (checked / total) * 2 if total > 0 else 3
                        yield {
                            "type": "progress",
                            "stage": "searching",
                            "message": event["message"],
                            "progress": round(match_progress, 1)
                        }
                    elif stage == "complete":
                        matching_calls = event.get("result", [])
                    elif stage == "error":
                        yield {
                            "type": "error",
                            "message": event["message"]
                        }
                        return

                span.set_attribute("calls.found", len(matching_calls))

                if not matching_calls:
                    span.set_status(Status(StatusCode.ERROR, "No matching calls found"))
                    yield {
                        "type": "error",
                        "message": f"No calls found for prospect: {prospect_name}"
                    }
                    return

                # Stage 2: Found calls
                yield {
                    "type": "progress",
                    "stage": "found",
                    "message": f"Found {len(matching_calls)} calls, sorting chronologically...",
                    "progress": 5,
                    "total_calls": len(matching_calls)
                }

                # Sort calls chronologically
                sorted_calls = self._sort_calls_by_date(matching_calls)

                # Extract participant names from call data (faster - no extra API calls)
                matched_names = set()
                for call in sorted_calls:
                    parties = call.get("parties", [])
                    for party in parties:
                        if isinstance(party, dict):
                            name = party.get("name")
                            if name:
                                matched_names.add(name)

                # Pre-fetch all transcripts in parallel for better performance
                yield {
                    "type": "progress",
                    "stage": "prefetching_transcripts",
                    "message": f"Pre-fetching transcripts for {len(sorted_calls)} calls...",
                    "progress": 4
                }
                call_ids = [call.get("id") for call in sorted_calls if call.get("id")]
                prefetched_transcripts = self.gong_client.get_transcripts_parallel(call_ids)

                # Stage 3: Quick summarize each call (5% to 95%)
                timeline_entries = []
                total_calls = len(sorted_calls)

                for idx, call in enumerate(sorted_calls):
                    call_id = call.get("id")
                    call_title = call.get("title", "Untitled")
                    call_date = call.get("scheduled") or call.get("started") or call.get("date", "")

                    # Calculate progress: calls take up 5% to 95% (90% range)
                    base_progress = 5
                    call_progress_range = 90
                    current_progress = base_progress + (idx / total_calls) * call_progress_range

                    # Yield progress for starting this call
                    yield {
                        "type": "progress",
                        "stage": "summarizing",
                        "message": f"Summarizing call {idx + 1}/{total_calls}: {call_title}",
                        "progress": round(current_progress, 1),
                        "current_call": idx + 1,
                        "total_calls": total_calls,
                        "call_title": call_title,
                        "call_date": call_date
                    }

                    try:
                        # Use pre-fetched transcript (no API call needed)
                        transcript_data = prefetched_transcripts.get(call_id)
                        if not transcript_data:
                            # Fallback to fetching if pre-fetch failed
                            transcript_data = self.gong_client.get_transcript(call_id)
                        formatted_transcript = self.gong_client.format_transcript_for_analysis(transcript_data)

                        # Format call date from call object (no API call needed)
                        formatted_date = self._format_date_string(call_date)

                        # Quick summarize (single LLM call)
                        summary_dict = self.analyzer.quick_summarize_call(
                            transcript=formatted_transcript,
                            call_date=formatted_date,
                            model=model
                        )

                        # Create QuickCallSummary from dict
                        quick_summary = QuickCallSummary(
                            call_type=summary_dict.get("call_type", "other"),
                            one_liner=summary_dict.get("one_liner", ""),
                            key_points=summary_dict.get("key_points", []),
                            participants_mentioned=summary_dict.get("participants_mentioned", []),
                            sentiment=summary_dict.get("sentiment", "neutral")
                        )

                        # Create timeline entry with quick summary
                        entry = CallTimelineEntry(
                            call_id=call_id,
                            call_date=formatted_date,
                            call_title=call.get("title"),
                            call_type=quick_summary.call_type,
                            call_url=call.get("url"),
                            quick_summary=quick_summary,
                            key_insights=quick_summary.key_points[:3]  # Use key points as insights
                        )

                        timeline_entries.append(entry)

                        # Update matched names from summary
                        for name in quick_summary.participants_mentioned:
                            matched_names.add(name)

                    except Exception as e:
                        span.add_event("call_summarization_failed", {
                            "call_id": call_id,
                            "error": str(e)
                        })

                        # Still create entry without summary
                        entry = CallTimelineEntry(
                            call_id=call_id,
                            call_date=call_date,
                            call_title=call.get("title"),
                            call_url=call.get("url"),
                            key_insights=[f"Summary failed: {str(e)}"]
                        )
                        timeline_entries.append(entry)
                        continue

                # Stage 4: Build simple timeline (95-100%)
                yield {
                    "type": "progress",
                    "stage": "finalizing",
                    "message": "Building timeline...",
                    "progress": 95
                }

                # Simple progression summary
                call_types = [e.call_type for e in timeline_entries if e.call_type]
                progression_summary = f"Analyzed {len(timeline_entries)} call(s)"
                if call_types:
                    unique_types = list(set(call_types))
                    progression_summary += f": {', '.join(unique_types)}"

                # Simple key themes from call types
                key_themes = list(set(call_types)) if call_types else ["General"]

                # Build timeline
                timeline = ProspectTimeline(
                    prospect_name=prospect_name,
                    matched_participant_names=sorted(list(matched_names)),
                    calls=timeline_entries,
                    cumulative_insights={
                        "total_calls": len(timeline_entries),
                        "call_types": list(set(call_types)),
                        "mode": "fast"
                    },
                    progression_summary=progression_summary,
                    overall_account_health=None,  # Not calculated in fast mode
                    key_themes=key_themes,
                    next_steps=[]  # Not calculated in fast mode
                )

                span.set_attribute("timeline.calls_count", len(timeline_entries))
                span.set_attribute("timeline.mode", "fast")
                span.set_status(Status(StatusCode.OK))

                # Yield final complete event
                yield {
                    "type": "complete",
                    "message": "Timeline ready",
                    "progress": 100,
                    "result": timeline
                }

            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                yield {
                    "type": "error",
                    "message": f"Analysis failed: {str(e)}"
                }

    def _sort_calls_by_date(self, calls: List[Dict]) -> List[Dict]:
        """Sort calls chronologically by date."""
        def get_date(call):
            date_str = call.get("scheduled") or call.get("started") or call.get("date", "")
            if not date_str:
                return datetime.min
            try:
                # Try parsing ISO format
                if isinstance(date_str, str):
                    return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                elif isinstance(date_str, (int, float)):
                    return datetime.fromtimestamp(date_str / 1000 if date_str > 1e12 else date_str)
            except:
                return datetime.min
            return datetime.min

        return sorted(calls, key=get_date)

    def _format_date_string(self, date_value: Any) -> str:
        """
        Format a date value into a human-readable string.

        Args:
            date_value: ISO format string, Unix timestamp, or already formatted string

        Returns:
            Formatted date string (e.g., "December 10, 2025") or the original value
        """
        if not date_value:
            return ""

        try:
            if isinstance(date_value, str):
                # Check if already formatted (e.g., "December 10, 2025")
                if any(month in date_value for month in ["January", "February", "March", "April",
                       "May", "June", "July", "August", "September", "October", "November", "December"]):
                    return date_value

                # Try parsing ISO format
                try:
                    dt = datetime.fromisoformat(date_value.replace('Z', '+00:00'))
                    return dt.strftime("%B %d, %Y")
                except ValueError:
                    # Try other formats
                    for fmt in ["%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"]:
                        try:
                            dt = datetime.strptime(date_value[:len(fmt.replace('%', ''))+4], fmt)
                            return dt.strftime("%B %d, %Y")
                        except ValueError:
                            continue
                    return date_value  # Return as-is if can't parse

            elif isinstance(date_value, (int, float)):
                # Unix timestamp (milliseconds or seconds)
                dt = datetime.fromtimestamp(date_value / 1000 if date_value > 1e12 else date_value)
                return dt.strftime("%B %d, %Y")

        except Exception:
            pass

        return str(date_value) if date_value else ""
    
    def _extract_matched_names(self, calls: List[Dict]) -> List[str]:
        """
        Extract unique participant names from calls.

        Uses party data already present in call objects from list_calls(),
        avoiding redundant API calls to get_call_info().
        """
        matched_names = set()
        for call in calls:
            # Use party data already in call object from list_calls()
            parties = call.get("parties", [])
            for party in parties:
                if isinstance(party, dict):
                    name = party.get("name")
                    if name and name.strip():
                        matched_names.add(name.strip())
        return sorted(list(matched_names))
    
    def _extract_key_insights(self, analysis: AnalysisResult) -> List[str]:
        """Extract key insights from analysis result."""
        insights = []
        
        # Add top insights
        for insight in analysis.top_insights[:5]:
            insights.append(f"{insight.category}: {insight.what_happened}")
        
        # Add key moments
        for moment in analysis.key_moments[:3]:
            insights.append(f"Key moment: {moment.get('description', '')}")
        
        return insights
    
    def _determine_progression(
        self,
        current_insights: List[str],
        prior_insights: List[Dict]
    ) -> Dict[str, str]:
        """Determine what's new vs carried forward from prior calls."""
        if not prior_insights:
            return {
                "new": "Initial call - establishing baseline",
                "carried_forward": "None"
            }
        
        # Simple heuristic: mark as new if not in prior insights
        prior_text = " ".join([
            " ".join(p.get("insights", [])) + " " + p.get("summary", "")
            for p in prior_insights
        ]).lower()
        
        new_items = []
        carried_forward = []
        
        for insight in current_insights:
            insight_lower = insight.lower()
            # Check if similar insight exists in prior calls
            if any(keyword in prior_text for keyword in insight_lower.split()[:3]):
                carried_forward.append(insight)
            else:
                new_items.append(insight)
        
        return {
            "new": "; ".join(new_items[:3]) if new_items else "Continuation of prior discussions",
            "carried_forward": "; ".join(carried_forward[:3]) if carried_forward else "None"
        }
    
    def _build_cumulative_insights(self, entries: List[CallTimelineEntry]) -> Dict[str, any]:
        """Build cumulative insights across all calls."""
        all_insights = []
        all_strengths = []
        all_improvements = []
        
        for entry in entries:
            if entry.analysis:
                all_insights.extend(entry.key_insights)
                all_strengths.extend(entry.analysis.strengths)
                all_improvements.extend(entry.analysis.improvement_areas)
        
        return {
            "total_insights": len(all_insights),
            "total_strengths": len(all_strengths),
            "total_improvements": len(all_improvements),
            "insights": all_insights[:20],  # Top 20 insights
            "strengths": list(set(all_strengths))[:10],  # Unique strengths
            "improvements": list(set(all_improvements))[:10]  # Unique improvements
        }
    
    def _generate_progression_summary(self, entries: List[CallTimelineEntry]) -> str:
        """Generate a summary of progression across calls."""
        if not entries:
            return "No calls to summarize."
        
        call_types = [e.call_type for e in entries if e.call_type]
        progression = []
        
        if "discovery" in call_types:
            progression.append("Discovery phase completed")
        if "poc_scoping" in call_types:
            progression.append("PoC scoping completed")
        if any("poc" in str(ct).lower() for ct in call_types if ct):
            progression.append("PoC phase in progress")
        
        if progression:
            return " → ".join(progression)
        else:
            return f"Analyzed {len(entries)} call(s) with progression through the sales cycle."
    
    def _extract_key_themes(self, entries: List[CallTimelineEntry]) -> List[str]:
        """Extract key themes across all calls."""
        themes = set()
        
        for entry in entries:
            if entry.analysis:
                # Extract themes from insights
                for insight in entry.analysis.top_insights:
                    themes.add(insight.category)
                
                # Extract from call summary
                summary_lower = entry.analysis.call_summary.lower()
                if "pain" in summary_lower or "challenge" in summary_lower:
                    themes.add("Pain Points")
                if "stakeholder" in summary_lower or "decision" in summary_lower:
                    themes.add("Stakeholder Mapping")
                if "requirement" in summary_lower or "capability" in summary_lower:
                    themes.add("Requirements")
                if "poc" in summary_lower or "proof" in summary_lower:
                    themes.add("PoC Planning")
        
        return sorted(list(themes))[:10]
    
    def _generate_next_steps(self, entries: List[CallTimelineEntry]) -> List[str]:
        """Generate recommended next steps based on timeline."""
        if not entries:
            return ["Schedule initial discovery call"]
        
        last_entry = entries[-1]
        if not last_entry.analysis:
            return ["Review call analysis"]
        
        next_steps = []
        
        # Check call type to suggest next steps
        if last_entry.call_type == "discovery":
            next_steps.append("Schedule Pre-PoC scoping call")
            if last_entry.analysis.call_classification:
                if last_entry.analysis.call_classification.discovery_completion_score < 70:
                    next_steps.append("Complete remaining discovery criteria")
        elif last_entry.call_type == "poc_scoping":
            next_steps.append("Proceed with PoC implementation")
            next_steps.append("Schedule PoC sync calls")
        elif "poc" in str(last_entry.call_type).lower():
            next_steps.append("Continue PoC execution")
            next_steps.append("Schedule regular check-ins")
        
        # Add recommendations from analysis
        if last_entry.analysis.call_classification:
            next_steps.extend(last_entry.analysis.call_classification.recommendations[:3])
        
        return next_steps[:5]  # Top 5 next steps
    
    def _assess_account_health(self, entries: List[CallTimelineEntry]) -> str:
        """Assess overall account health based on timeline."""
        if not entries:
            return "No calls analyzed"
        
        # Calculate average scores
        scores = []
        for entry in entries:
            if entry.analysis and entry.analysis.overall_score:
                scores.append(entry.analysis.overall_score)
        
        if not scores:
            return "Insufficient data for assessment"
        
        avg_score = sum(scores) / len(scores)
        
        # Assess based on progression
        call_types = [e.call_type for e in entries if e.call_type]
        has_discovery = "discovery" in call_types
        has_poc_scoping = "poc_scoping" in call_types
        
        if avg_score >= 8.0:
            health = "Excellent"
        elif avg_score >= 7.0:
            health = "Good"
        elif avg_score >= 6.0:
            health = "Fair"
        else:
            health = "Needs Improvement"
        
        progression_status = ""
        if has_discovery and has_poc_scoping:
            progression_status = " - Strong progression through sales cycle"
        elif has_discovery:
            progression_status = " - Discovery complete, ready for PoC scoping"
        else:
            progression_status = " - Early stage"
        
        return f"{health} (avg score: {avg_score:.1f}/10){progression_status}"

    def _build_account_analysis(self, entries: List[CallTimelineEntry]) -> AccountAnalysisSummary:
        """
        Build aggregated analysis summary across all calls.

        This creates an analysis similar to single-call AnalysisResult but
        aggregated across all calls for the account.
        """
        if not entries:
            return AccountAnalysisSummary()

        # Collect data from all calls with analysis
        analyzed_entries = [e for e in entries if e.analysis]
        if not analyzed_entries:
            return AccountAnalysisSummary(
                total_calls_analyzed=len(entries),
                account_summary="No calls were successfully analyzed."
            )

        # Calculate date range
        dates = [e.call_date for e in entries if e.call_date]
        date_range = ""
        if dates:
            # Try to format dates nicely
            try:
                date_range = f"{dates[0]} - {dates[-1]}" if len(dates) > 1 else dates[0]
            except:
                date_range = f"{len(dates)} calls"

        # Aggregate scores
        overall_scores = []
        problem_scores = []
        differentiation_scores = []
        proof_scores = []
        required_cap_scores = []
        tech_depth_scores = []
        discovery_quality_scores = []
        active_listening_scores = []
        value_articulation_scores = []

        for entry in analyzed_entries:
            analysis = entry.analysis
            if analysis.overall_score:
                overall_scores.append(analysis.overall_score)
            if analysis.command_scores:
                if analysis.command_scores.problem_identification:
                    problem_scores.append(analysis.command_scores.problem_identification)
                if analysis.command_scores.differentiation:
                    differentiation_scores.append(analysis.command_scores.differentiation)
                if analysis.command_scores.proof_evidence:
                    proof_scores.append(analysis.command_scores.proof_evidence)
                if analysis.command_scores.required_capabilities:
                    required_cap_scores.append(analysis.command_scores.required_capabilities)
            if analysis.sa_metrics:
                if analysis.sa_metrics.technical_depth:
                    tech_depth_scores.append(analysis.sa_metrics.technical_depth)
                if analysis.sa_metrics.discovery_quality:
                    discovery_quality_scores.append(analysis.sa_metrics.discovery_quality)
                if analysis.sa_metrics.active_listening:
                    active_listening_scores.append(analysis.sa_metrics.active_listening)
                if analysis.sa_metrics.value_articulation:
                    value_articulation_scores.append(analysis.sa_metrics.value_articulation)

        def avg(lst): return sum(lst) / len(lst) if lst else None

        # Build aggregated command scores
        command_scores = CommandOfMessageScore(
            problem_identification=round(avg(problem_scores), 1) if problem_scores else None,
            differentiation=round(avg(differentiation_scores), 1) if differentiation_scores else None,
            proof_evidence=round(avg(proof_scores), 1) if proof_scores else None,
            required_capabilities=round(avg(required_cap_scores), 1) if required_cap_scores else None
        )

        # Build aggregated SA metrics
        sa_metrics = SAPerformanceMetrics(
            technical_depth=round(avg(tech_depth_scores), 1) if tech_depth_scores else None,
            discovery_quality=round(avg(discovery_quality_scores), 1) if discovery_quality_scores else None,
            active_listening=round(avg(active_listening_scores), 1) if active_listening_scores else None,
            value_articulation=round(avg(value_articulation_scores), 1) if value_articulation_scores else None
        )

        # Aggregate top insights from all calls (prioritize critical, then important)
        all_insights: List[AggregatedActionableInsight] = []
        for entry in analyzed_entries:
            if entry.analysis and entry.analysis.top_insights:
                for insight in entry.analysis.top_insights:
                    all_insights.append(AggregatedActionableInsight(
                        category=insight.category,
                        severity=insight.severity,
                        call_date=entry.call_date,
                        call_title=entry.call_title,
                        timestamp=insight.timestamp,
                        conversation_snippet=insight.conversation_snippet,
                        what_happened=insight.what_happened,
                        why_it_matters=insight.why_it_matters,
                        better_approach=insight.better_approach,
                        example_phrasing=insight.example_phrasing
                    ))

        # Sort by severity (critical first) and take top 10
        severity_order = {"critical": 0, "important": 1, "minor": 2}
        all_insights.sort(key=lambda x: severity_order.get(x.severity, 3))
        top_insights = all_insights[:10]

        # Aggregate unique strengths and improvements
        all_strengths = set()
        all_improvements = set()
        for entry in analyzed_entries:
            if entry.analysis:
                all_strengths.update(entry.analysis.strengths)
                all_improvements.update(entry.analysis.improvement_areas)

        # Aggregate key moments with call context
        all_key_moments = []
        for entry in analyzed_entries:
            if entry.analysis and entry.analysis.key_moments:
                for moment in entry.analysis.key_moments:
                    moment_with_context = dict(moment)
                    moment_with_context["call_date"] = entry.call_date
                    moment_with_context["call_title"] = entry.call_title
                    all_key_moments.append(moment_with_context)

        # Aggregate discovery criteria completion (take max across calls)
        discovery_completion = {}
        for entry in analyzed_entries:
            if entry.analysis and entry.analysis.call_classification:
                cls = entry.analysis.call_classification
                if cls.discovery_criteria:
                    dc = cls.discovery_criteria
                    discovery_completion["pain_current_state"] = max(
                        discovery_completion.get("pain_current_state", 0),
                        dc.pain_current_state.completion_score
                    )
                    discovery_completion["stakeholder_map"] = max(
                        discovery_completion.get("stakeholder_map", 0),
                        dc.stakeholder_map.completion_score
                    )
                    discovery_completion["required_capabilities"] = max(
                        discovery_completion.get("required_capabilities", 0),
                        dc.required_capabilities.completion_score
                    )
                    discovery_completion["competitive_landscape"] = max(
                        discovery_completion.get("competitive_landscape", 0),
                        dc.competitive_landscape.completion_score
                    )

        # Aggregate PoC scoping criteria completion (take max across calls)
        poc_scoping_completion = {}
        for entry in analyzed_entries:
            if entry.analysis and entry.analysis.call_classification:
                cls = entry.analysis.call_classification
                if cls.poc_scoping_criteria:
                    pc = cls.poc_scoping_criteria
                    poc_scoping_completion["use_case_scoped"] = max(
                        poc_scoping_completion.get("use_case_scoped", 0),
                        pc.use_case_scoped.completion_score
                    )
                    poc_scoping_completion["implementation_requirements"] = max(
                        poc_scoping_completion.get("implementation_requirements", 0),
                        pc.implementation_requirements.completion_score
                    )
                    poc_scoping_completion["metrics_success_criteria"] = max(
                        poc_scoping_completion.get("metrics_success_criteria", 0),
                        pc.metrics_success_criteria.completion_score
                    )
                    poc_scoping_completion["timeline_milestones"] = max(
                        poc_scoping_completion.get("timeline_milestones", 0),
                        pc.timeline_milestones.completion_score
                    )
                    poc_scoping_completion["resources_committed"] = max(
                        poc_scoping_completion.get("resources_committed", 0),
                        pc.resources_committed.completion_score
                    )

        # Aggregate all missed opportunities
        all_missed_opportunities = []
        for entry in analyzed_entries:
            if entry.analysis and entry.analysis.call_classification:
                cls = entry.analysis.call_classification
                # From discovery criteria
                if cls.discovery_criteria:
                    for section in [cls.discovery_criteria.pain_current_state,
                                    cls.discovery_criteria.stakeholder_map,
                                    cls.discovery_criteria.required_capabilities,
                                    cls.discovery_criteria.competitive_landscape]:
                        for mo in section.missed_opportunities:
                            all_missed_opportunities.append({
                                "call_date": entry.call_date,
                                "call_title": entry.call_title,
                                "criteria_name": mo.criteria_name,
                                "context": mo.context,
                                "suggested_question": mo.suggested_question,
                                "why_important": mo.why_important
                            })
                # From PoC scoping criteria
                if cls.poc_scoping_criteria:
                    for section in [cls.poc_scoping_criteria.use_case_scoped,
                                    cls.poc_scoping_criteria.implementation_requirements,
                                    cls.poc_scoping_criteria.metrics_success_criteria,
                                    cls.poc_scoping_criteria.timeline_milestones,
                                    cls.poc_scoping_criteria.resources_committed]:
                        for mo in section.missed_opportunities:
                            all_missed_opportunities.append({
                                "call_date": entry.call_date,
                                "call_title": entry.call_title,
                                "criteria_name": mo.criteria_name,
                                "context": mo.context,
                                "suggested_question": mo.suggested_question,
                                "why_important": mo.why_important
                            })

        # Build aggregated recap data from last call (most relevant for next call)
        recap_data = None
        if analyzed_entries and analyzed_entries[-1].analysis:
            last_analysis = analyzed_entries[-1].analysis
            if last_analysis.recap_data:
                # Combine recap data from all calls
                all_initiatives = []
                all_challenges = []
                all_requirements = []
                all_follow_ups = []
                for entry in analyzed_entries:
                    if entry.analysis and entry.analysis.recap_data:
                        rd = entry.analysis.recap_data
                        all_initiatives.extend(rd.key_initiatives)
                        all_challenges.extend(rd.challenges)
                        all_requirements.extend(rd.solution_requirements)
                        all_follow_ups.extend(rd.follow_up_questions)

                recap_data = RecapSlideData(
                    customer_name=last_analysis.recap_data.customer_name,
                    call_date=f"{len(entries)} calls ({date_range})",
                    key_initiatives=list(set(all_initiatives))[:5],
                    challenges=list(set(all_challenges))[:5],
                    solution_requirements=list(set(all_requirements))[:5],
                    follow_up_questions=list(set(all_follow_ups))[:5]
                )

        # Build account summary
        call_types = [e.call_type for e in analyzed_entries if e.call_type]
        call_summaries = [e.analysis.call_summary for e in analyzed_entries if e.analysis]

        avg_score = avg(overall_scores)
        score_desc = ""
        if avg_score:
            if avg_score >= 8:
                score_desc = "excellent"
            elif avg_score >= 7:
                score_desc = "good"
            elif avg_score >= 6:
                score_desc = "fair"
            else:
                score_desc = "needs improvement"

        account_summary = f"""Analyzed {len(analyzed_entries)} call(s) for this account spanning {date_range}.

Call types: {', '.join(set(call_types)) if call_types else 'Not classified'}. Overall engagement quality is {score_desc} with an average score of {avg_score:.1f}/10.

Key strengths across calls: {'; '.join(list(all_strengths)[:3]) if all_strengths else 'None identified'}.

Areas for improvement: {'; '.join(list(all_improvements)[:3]) if all_improvements else 'None identified'}."""

        return AccountAnalysisSummary(
            account_summary=account_summary,
            total_calls_analyzed=len(analyzed_entries),
            date_range=date_range,
            average_overall_score=round(avg_score, 1) if avg_score else None,
            command_scores=command_scores,
            sa_metrics=sa_metrics,
            top_insights=top_insights,
            strengths=list(all_strengths)[:10],
            improvement_areas=list(all_improvements)[:10],
            key_moments=all_key_moments[:15],
            discovery_completion=discovery_completion,
            poc_scoping_completion=poc_scoping_completion,
            all_missed_opportunities=all_missed_opportunities[:20],
            recap_data=recap_data
        )

