import os
import json
import re
import contextvars
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional
from models import (
    AnalysisResult,
    ActionableInsight,
    CommandOfMessageScore,
    SAPerformanceMetrics,
    CallClassification,
    CallType,
    DiscoveryCriteria,
    PocScopingCriteria,
    PainCurrentState,
    StakeholderMap,
    RequiredCapabilities,
    CompetitiveLandscape,
    UseCaseScoped,
    ImplementationRequirements,
    MetricsSuccessCriteria,
    TimelineMilestones,
    ResourcesCommitted,
    CriteriaEvidence,
    MissedOpportunity,
    MissingElements,
    RecapSlideData
)
from dotenv import load_dotenv
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from observability import force_flush_spans

load_dotenv()


class SACallAnalysisCrew:
    """
    LangGraph-orchestrated Call Analysis System.

    Uses specialized LLM stages to comprehensively analyze sales call performance:
    1. Call Classifier - Classifies the call type (Discovery vs PoC Scoping)
    2. Technical Evaluator - Assesses technical discussion quality
    3. Sales Methodology & Discovery Expert - Evaluates discovery and Command of Message framework
    4. Report Compiler - Synthesizes feedback into actionable insights
    """

    def __init__(self):
        # Determine which LLM to use based on environment
        self.use_litellm = os.getenv("USE_LITELLM", "false").lower() == "true"
        self.default_model = os.getenv("MODEL_NAME", "claude-sonnet-4-20250514")

        # Initialize with default model (LangChain chat model)
        self._chat = None
        self._configure_chat(self.default_model)

    def _configure_chat(self, model_name: str) -> None:
        """Configure LangChain chat model for analysis and recap."""
        from call_analysis_llm import build_chat_model

        self._chat = build_chat_model(model_name)
        self.model_name = model_name

    def _extract_snippet_from_transcript(self, transcript: str, timestamp: str, max_chars: int = 500) -> Optional[str]:
        """
        Extract actual conversation lines from transcript based on timestamp.
        
        Args:
            transcript: The full transcript text
            timestamp: Timestamp like "[12:34]", "[0:16]", "[~10:00]"
            max_chars: Maximum characters to extract (default 500)
            
        Returns:
            Actual conversation snippet from transcript, or None if timestamp not found
        """
        if not timestamp or not transcript:
            return None
            
        # Clean timestamp - remove brackets, tildes, etc. to get core time
        clean_ts = timestamp.strip().strip('[]').strip('~').strip()
        
        # Split transcript into lines
        lines = transcript.split('\n')
        
        # Build list of timestamp variations to search for
        search_patterns = [clean_ts, f"[{clean_ts}]"]
        
        # Handle different timestamp formats: "02:16", "2:16", "00:02:16"
        if ':' in clean_ts:
            parts = clean_ts.split(':')
            if len(parts) == 2:
                mm, ss = parts
                # Try with and without leading zeros
                search_patterns.extend([
                    f"{int(mm)}:{ss}",  # "2:16"
                    f"{int(mm):02d}:{ss}",  # "02:16"
                    f"[{int(mm)}:{ss}]",
                    f"[{int(mm):02d}:{ss}]",
                ])
            elif len(parts) == 3:
                hh, mm, ss = parts
                search_patterns.extend([
                    f"{int(mm)}:{ss}",  # Just MM:SS
                    f"{int(hh)}:{int(mm)}:{ss}",
                ])
        
        # Try to find the timestamp in the transcript
        target_line_idx = None
        
        for i, line in enumerate(lines):
            for pattern in search_patterns:
                if pattern in line:
                    target_line_idx = i
                    break
            if target_line_idx is not None:
                break
        
        if target_line_idx is None:
            return None
        
        # Extract text starting from the timestamp, limited by character count
        snippet_parts = []
        char_count = 0
        
        for i in range(target_line_idx, min(target_line_idx + 6, len(lines))):
            line = lines[i].strip()
            if not line:
                continue
                
            # If adding this line would exceed limit, truncate it
            if char_count + len(line) > max_chars:
                remaining = max_chars - char_count
                if remaining > 50:  # Only add if we can fit meaningful content
                    # Try to break at a sentence or word boundary
                    truncated = line[:remaining]
                    last_period = truncated.rfind('.')
                    last_space = truncated.rfind(' ')
                    if last_period > remaining * 0.5:
                        truncated = truncated[:last_period + 1]
                    elif last_space > remaining * 0.5:
                        truncated = truncated[:last_space] + "..."
                    else:
                        truncated = truncated + "..."
                    snippet_parts.append(truncated)
                break
            
            snippet_parts.append(line)
            char_count += len(line) + 1  # +1 for newline
        
        if not snippet_parts:
            return None
            
        return '\n'.join(snippet_parts)

    def _populate_snippets_in_insights(self, insights: list, transcript: str) -> list:
        """
        Post-process insights to populate conversation_snippet from actual transcript.
        
        Args:
            insights: List of insight dictionaries with timestamps
            transcript: The full transcript text
            
        Returns:
            Updated insights with actual transcript snippets
        """
        for insight in insights:
            timestamp = insight.get('timestamp')
            if timestamp:
                actual_snippet = self._extract_snippet_from_transcript(transcript, timestamp)
                if actual_snippet:
                    insight['conversation_snippet'] = actual_snippet
                else:
                    # If we can't find the timestamp, set to None instead of hallucinated text
                    insight['conversation_snippet'] = None
        return insights

    def quick_summarize_call(
        self,
        transcript: str,
        call_date: str = "",
        model: str = None
    ) -> dict:
        """
        Generate a quick summary of a call with a single LLM call.

        Much faster than full analyze_call() which uses the multi-stage LangGraph pipeline.
        Ideal for getting someone up to speed quickly on an account.

        Args:
            transcript: The call transcript text
            call_date: Optional date of the call
            model: Optional model to use (defaults to configured model)

        Returns:
            Dict with call_type, one_liner, key_points, participants, sentiment
        """
        tracer = trace.get_tracer("quick-summarize")

        with tracer.start_as_current_span("quick_summarize_call") as span:
            try:
                # Configure model if specified
                if model and model != self.model_name:
                    self._configure_chat(model)

                # Limit transcript size for speed (first ~15000 chars is usually enough)
                truncated_transcript = transcript[:15000]
                if len(transcript) > 15000:
                    truncated_transcript += "\n\n[... transcript truncated for speed ...]"

                prompt = f"""Summarize this sales call transcript in a brief, scannable format.

CALL DATE: {call_date or "Unknown"}

TRANSCRIPT:
{truncated_transcript}

Analyze this transcript and respond with ONLY a JSON object (no other text) in this exact format:
{{
    "call_type": "discovery|poc_scoping|check_in|demo|follow_up|other",
    "one_liner": "Single sentence summary of what this call was about",
    "key_points": [
        "Key point 1 - most important thing discussed or decided",
        "Key point 2 - another important topic or outcome",
        "Key point 3 - relevant detail, next step, or action item",
        "Key point 4 - additional context if needed",
        "Key point 5 - optional additional point"
    ],
    "participants_mentioned": ["Name 1", "Name 2"],
    "sentiment": "positive|neutral|negative|mixed"
}}

Guidelines:
- call_type: "discovery" for initial exploration calls, "poc_scoping" for proof-of-concept planning, "check_in" for status updates, "demo" for product demonstrations, "follow_up" for continuing discussions
- key_points: 3-5 bullet points capturing what happened, decisions made, and next steps
- participants_mentioned: Names of key people mentioned (customer contacts, stakeholders)
- sentiment: Overall tone of the conversation

Return ONLY the JSON object, no additional text."""

                span.set_attribute("input.transcript_length", len(transcript))
                span.set_attribute("input.truncated", len(transcript) > 15000)

                from call_analysis_llm import chat_invoke_text

                result_text = chat_invoke_text(self._chat, prompt)
                if not isinstance(result_text, str):
                    result_text = str(result_text)

                span.set_attribute("output.raw_length", len(result_text))

                # Parse JSON from response
                # Try to find JSON in the response
                json_start = result_text.find('{')
                json_end = result_text.rfind('}') + 1

                if json_start >= 0 and json_end > json_start:
                    json_str = result_text[json_start:json_end]
                    summary = json.loads(json_str)

                    # Validate required fields
                    summary.setdefault("call_type", "other")
                    summary.setdefault("one_liner", "Call summary unavailable")
                    summary.setdefault("key_points", [])
                    summary.setdefault("participants_mentioned", [])
                    summary.setdefault("sentiment", "neutral")

                    span.set_attribute("output.call_type", summary["call_type"])
                    span.set_attribute("output.key_points_count", len(summary["key_points"]))
                    span.set_status(Status(StatusCode.OK))

                    return summary
                else:
                    # Fallback if JSON parsing fails
                    span.set_status(Status(StatusCode.ERROR, "Failed to parse JSON"))
                    return {
                        "call_type": "other",
                        "one_liner": "Summary generation failed - could not parse response",
                        "key_points": [result_text[:200] if result_text else "No response"],
                        "participants_mentioned": [],
                        "sentiment": "neutral"
                    }

            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                return {
                    "call_type": "other",
                    "one_liner": f"Summary generation failed: {str(e)}",
                    "key_points": [],
                    "participants_mentioned": [],
                    "sentiment": "neutral"
                }

    def _format_prior_insights(self, prior_call_insights: Optional[List[Dict]]) -> str:
        """
        Format prior call insights into a context string for prompts.
        
        Args:
            prior_call_insights: List of insights from prior calls
            
        Returns:
            Formatted context string
        """
        if not prior_call_insights:
            return ""
        
        context_parts = ["\n=== CONTEXT FROM PRIOR CALLS ===\n"]
        
        for idx, prior in enumerate(prior_call_insights, 1):
            call_date = prior.get("call_date", "Unknown date")
            call_type = prior.get("call_type", "Unknown type")
            summary = prior.get("summary", "")
            insights = prior.get("insights", [])
            
            context_parts.append(f"\nPrior Call #{idx} ({call_date}, {call_type}):")
            if summary:
                context_parts.append(f"Summary: {summary}")
            if insights:
                context_parts.append("Key Insights:")
                for insight in insights[:5]:  # Top 5 insights per call
                    context_parts.append(f"  - {insight}")
        
        context_parts.append("\n=== END PRIOR CALL CONTEXT ===\n")
        context_parts.append("\nWhen analyzing the current call, reference information from prior calls when relevant.")
        context_parts.append("For example: 'As discussed in the discovery call...' or 'Building on the previous PoC scoping call...'")
        
        return "\n".join(context_parts)

    def analyze_call(
        self,
        transcript: str,
        speakers: List[str],
        transcript_data: Optional[dict] = None,
        call_date: str = "",
        model: Optional[str] = None,
        prior_call_insights: Optional[List[Dict]] = None
    ) -> AnalysisResult:
        """
        Analyze a call transcript using the crew of specialized agents.

        Args:
            transcript: The call transcript (formatted)
            speakers: List of speaker names (if available)
            transcript_data: Raw transcript data from Gong (optional, for hybrid sampling)
            call_date: The date of the call (optional, from Gong metadata)
            model: LLM model to use (optional, defaults to environment config)
            prior_call_insights: Optional list of insights from prior calls for cumulative context

        Returns:
            AnalysisResult with structured analysis
        """
        
        # Configure model if specified (allows per-request model selection)
        if model and model != self.model_name:
            print(f"🔄 Switching model from {self.model_name} to {model}")
            self._configure_chat(model)

        # Format prior call insights for context
        prior_context = self._format_prior_insights(prior_call_insights)

        # Get tracer (must be obtained after setup_observability is called)
        tracer = trace.get_tracer("sa-call-analyzer")
        print(f"🔍 Tracer obtained: {type(tracer).__name__}")

        # Create a custom span for the entire analysis workflow
        with tracer.start_as_current_span(
            "sa_call_analysis",
            attributes={
                # OpenInference semantic conventions for input - full transcript
                "input.value": transcript,
                "input.mime_type": "text/plain",
                # OpenInference span kind - this is an agent workflow
                "openinference.span.kind": "agent",
            }
        ) as analysis_span:
            print(f"📊 Created span: {analysis_span.get_span_context().span_id if analysis_span else 'None'}")
            
            # DEBUG: Check current span context
            current_span = trace.get_current_span()
            if current_span:
                span_ctx = current_span.get_span_context()
                print(f"🔍 Current span context - Trace ID: {format(span_ctx.trace_id, '032x')}, Span ID: {format(span_ctx.span_id, '016x')}")
            else:
                print("🔍 No current span context found")
            
            result = None  # Initialize result
            try:
                from call_analysis_graph import run_call_analysis_pipeline

                classification_text, final_report_text = run_call_analysis_pipeline(
                    self._chat, transcript, prior_context, tracer
                )
                call_classification = self._parse_classification(classification_text)
                analysis_data_from_crew = None

                # Parse JSON from the final LLM report
                with tracer.start_as_current_span("parse_crew_report") as parse_span:
                    parse_span.set_attribute("openinference.span.kind", "chain")
                    parse_span.set_attribute("report.raw_length", len(final_report_text))
                    # OpenInference input - raw report text
                    parse_span.set_attribute("input.value", final_report_text)
                    parse_span.set_attribute("input.mime_type", "text/plain")

                    try:
                        analysis_data = None

                        # Pre-parsed JSON path (unused with LangGraph; kept for compatibility)
                        if analysis_data_from_crew:
                            analysis_data = analysis_data_from_crew
                            parse_span.add_event("using_pre_parsed_json")
                            parse_span.set_attribute("parsing.method", "pre_parsed_json")

                        # If not, try to extract JSON from the report
                        if analysis_data is None:
                            # Try markdown code blocks first
                            code_block_pattern = r'```(?:json)?\s*([\s\S]*?)```'
                            code_blocks = re.findall(code_block_pattern, final_report_text)

                            for block in code_blocks:
                                block = block.strip()
                                if block.startswith('{'):
                                    try:
                                        analysis_data = json.loads(block)
                                        parse_span.add_event("json_parsed_from_code_block")
                                        parse_span.set_attribute("parsing.method", "code_block")
                                        break
                                    except json.JSONDecodeError:
                                        continue

                        # If still no data, try finding raw JSON
                        if analysis_data is None:
                            json_start = final_report_text.find('{')
                            json_end = final_report_text.rfind('}') + 1

                            if json_start >= 0 and json_end > json_start:
                                json_str = final_report_text[json_start:json_end]
                                # Clean up common issues
                                json_str = re.sub(r',\s*}', '}', json_str)  # Remove trailing commas
                                json_str = re.sub(r',\s*]', ']', json_str)  # Remove trailing commas in arrays
                                analysis_data = json.loads(json_str)
                                parse_span.add_event("json_parsed_from_raw")
                                parse_span.set_attribute("parsing.method", "raw_extraction")
                            else:
                                raise ValueError("No JSON found in report")

                        parse_span.set_attribute("parsing.success", True)

                    except (json.JSONDecodeError, ValueError) as e:
                        print(f"⚠️  WARNING: Could not parse JSON from crew report: {e}")
                        print(f"📄 Raw report length: {len(final_report_text)} chars")
                        print(f"📄 Raw report (first 1000 chars):\n{final_report_text[:1000]}")
                        print(f"📄 Raw report (last 500 chars):\n{final_report_text[-500:]}")

                        # Try one more time with aggressive cleaning
                        try:
                            # Remove any control characters and clean the text
                            cleaned_text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', final_report_text)
                            # Remove ANSI escape codes
                            cleaned_text = re.sub(r'\x1b\[[0-9;]*m', '', cleaned_text)
                            # Remove any box-drawing characters
                            cleaned_text = re.sub(r'[│╭╮╯╰─]', '', cleaned_text)

                            json_start = cleaned_text.find('{')
                            json_end = cleaned_text.rfind('}') + 1

                            if json_start >= 0 and json_end > json_start:
                                json_str = cleaned_text[json_start:json_end]
                                json_str = re.sub(r',\s*}', '}', json_str)
                                json_str = re.sub(r',\s*]', ']', json_str)
                                analysis_data = json.loads(json_str)
                                parse_span.add_event("json_parsed_after_cleaning")
                                parse_span.set_attribute("parsing.success", True)
                                parse_span.set_attribute("parsing.method", "cleaned")
                                print(f"✅ JSON parsed successfully after cleaning!")
                            else:
                                raise ValueError("No JSON found after cleaning")
                        except (json.JSONDecodeError, ValueError) as e2:
                            print(f"⚠️  Cleaning also failed: {e2}")
                            parse_span.add_event("json_parse_failed", {
                                "error": str(e),
                                "cleaning_error": str(e2),
                                "report_preview": final_report_text[:500]
                            })
                            parse_span.set_attribute("parsing.success", False)
                            parse_span.set_attribute("parsing.fallback", True)
                            parse_span.set_attribute("error.message", str(e))

                            # Smart fallback: Extract meaningful content from raw text
                            analysis_data = self._extract_insights_from_raw_text(final_report_text)

                    # Helper function to safely convert to float
                    def safe_float(value, default=7.0):
                        """Convert value to float, return default if None or invalid."""
                        if value is None:
                            return default
                        try:
                            return float(value)
                        except (ValueError, TypeError):
                            return default

                    # Convert to Pydantic models
                    parse_span.add_event("constructing_pydantic_models")

                    # Parse insights and sort chronologically
                    # Note: conversation_snippet is now a synthesized summary generated by the LLM
                    raw_insights = analysis_data.get("top_insights", [])

                    insights = []
                    for insight in raw_insights:
                        insights.append(ActionableInsight(**insight))

                    # Sort insights by timestamp (chronological order)
                    def extract_timestamp_value(timestamp_str):
                        """Extract numeric value from timestamp for sorting."""
                        if not timestamp_str:
                            return 999999  # Put items without timestamps at the end

                        # Remove brackets and handle different formats: [5:23], [0:16], [~10:00]
                        clean_ts = timestamp_str.strip('[]~')

                        try:
                            # Split by : to get minutes and seconds
                            parts = clean_ts.split(':')
                            if len(parts) == 2:
                                minutes, seconds = parts
                                return int(minutes) * 60 + int(seconds)
                            elif len(parts) == 3:  # HH:MM:SS format
                                hours, minutes, seconds = parts
                                return int(hours) * 3600 + int(minutes) * 60 + int(seconds)
                        except (ValueError, AttributeError):
                            return 999999  # If parsing fails, put at end

                        return 999999

                    insights.sort(key=lambda x: extract_timestamp_value(x.timestamp))

                    result = AnalysisResult(
                        call_summary=analysis_data.get("call_summary", ""),
                        overall_score=safe_float(analysis_data.get("overall_score")),
                        command_scores=CommandOfMessageScore(
                            **{k: safe_float(v) for k, v in analysis_data.get("command_scores", {}).items()}
                        ),
                        sa_metrics=SAPerformanceMetrics(
                            **{k: safe_float(v) for k, v in analysis_data.get("sa_metrics", {}).items()}
                        ),
                        top_insights=insights,  # Now sorted chronologically
                        strengths=analysis_data.get("strengths", []),
                        improvement_areas=analysis_data.get("improvement_areas", []),
                        key_moments=analysis_data.get("key_moments", []),
                        call_classification=call_classification  # Add classification result
                    )

                    # Generate recap slide data
                    try:
                        recap_data = self.generate_recap_data(transcript, call_classification, analysis_data, call_date)
                        result.recap_data = recap_data
                        print(f"✅ Recap data generated - Customer: '{recap_data.customer_name}', Date: '{recap_data.call_date}'")
                    except Exception as e:
                        print(f"⚠️ Could not generate recap data: {e}")

                    # Add metadata about the parsed results
                    parse_span.set_attribute("analysis.insight_count", len(result.top_insights))
                    parse_span.set_attribute("analysis.strength_count", len(result.strengths))
                    parse_span.set_attribute("analysis.improvement_count", len(result.improvement_areas))
                    parse_span.set_attribute("analysis.key_moment_count", len(result.key_moments))
                    # OpenInference output - parsed and structured analysis result
                    parse_span.set_attribute("output.value", json.dumps({
                        "call_summary": result.call_summary,
                        "insight_count": len(result.top_insights),
                        "strength_count": len(result.strengths),
                        "improvement_count": len(result.improvement_areas)
                    }))
                    parse_span.set_attribute("output.mime_type", "application/json")
                    parse_span.set_status(Status(StatusCode.OK))

                # Create separate child spans for each insight
                for idx, insight in enumerate(result.top_insights):
                    with tracer.start_as_current_span(
                        f"insight_{idx+1}_{insight.category.lower().replace(' ', '_')}",
                        attributes={
                            "openinference.span.kind": "chain",
                            "insight.index": idx + 1,
                            "insight.category": insight.category,
                            "insight.severity": insight.severity,
                            "insight.timestamp": insight.timestamp or "unknown",
                            "output.value": json.dumps({
                                "category": insight.category,
                                "severity": insight.severity,
                                "timestamp": insight.timestamp,
                                "conversation_snippet": insight.conversation_snippet,
                                "what_happened": insight.what_happened,
                                "why_it_matters": insight.why_it_matters,
                                "better_approach": insight.better_approach,
                                "example_phrasing": insight.example_phrasing,
                            }, indent=2),
                            "output.mime_type": "application/json",
                        }
                    ) as insight_span:
                        insight_span.add_event("insight_recorded", {
                            "category": insight.category,
                            "timestamp": insight.timestamp or "unknown"
                        })

                # Add result metadata to analysis_span (parent span)
                analysis_span.set_attributes({
                    "output.value": json.dumps({
                        "call_summary": result.call_summary,
                        "overall_score": result.overall_score,
                        "command_scores": result.command_scores.model_dump(),
                        "sa_metrics": result.sa_metrics.model_dump(),
                        "insight_count": len(result.top_insights),
                        "strengths": result.strengths,
                        "improvement_areas": result.improvement_areas,
                        "key_moments": result.key_moments,
                    }, indent=2),
                    "output.mime_type": "application/json",
                })

            except Exception as e:
                analysis_span.set_status(Status(StatusCode.ERROR, str(e)))
                analysis_span.record_exception(e)
                raise

        # Force flush spans to ensure they're sent to Arize immediately
        force_flush_spans()

        return result

    def _parse_evidence(self, evidence_list: List[dict]) -> List[CriteriaEvidence]:
        """Parse evidence items from JSON data"""
        evidence = []
        for item in evidence_list:
            try:
                evidence.append(CriteriaEvidence(
                    criteria_name=item.get("criteria_name", ""),
                    captured=item.get("captured", False),
                    timestamp=item.get("timestamp"),
                    conversation_snippet=item.get("conversation_snippet"),
                    speaker=item.get("speaker")
                ))
            except Exception:
                pass  # Skip malformed evidence items
        return evidence

    def _parse_missed_opportunities(self, opportunities_list: List[dict]) -> List[MissedOpportunity]:
        """Parse missed opportunity items from JSON data"""
        opportunities = []
        for item in opportunities_list:
            try:
                opportunities.append(MissedOpportunity(
                    criteria_name=item.get("criteria_name", ""),
                    timestamp=item.get("timestamp"),
                    context=item.get("context", ""),
                    suggested_question=item.get("suggested_question", ""),
                    why_important=item.get("why_important", ""),
                    collected_later=item.get("collected_later", False)
                ))
            except Exception:
                pass  # Skip malformed items
        return opportunities
    
    def _validate_missed_opportunities(
        self, 
        missed_opportunities: List[MissedOpportunity], 
        evidence: List[CriteriaEvidence]
    ) -> List[MissedOpportunity]:
        """
        Filter out missed opportunities where the same information was collected later in the call.
        
        This provides a safety net to catch cases where the LLM didn't perfectly follow
        the instruction to check for later evidence.
        """
        if not missed_opportunities or not evidence:
            return missed_opportunities
        
        # Create a map of criteria_name -> list of evidence timestamps
        evidence_by_criteria = {}
        for ev in evidence:
            if ev.captured and ev.timestamp:
                criteria = ev.criteria_name
                if criteria not in evidence_by_criteria:
                    evidence_by_criteria[criteria] = []
                evidence_by_criteria[criteria].append(ev.timestamp)
        
        def parse_timestamp(ts: str) -> Optional[int]:
            """Convert timestamp string like '[05:23]' or '[1:15:30]' to seconds"""
            if not ts:
                return None
            try:
                # Remove brackets and whitespace
                clean_ts = ts.strip().strip('[]').strip('~').strip()
                parts = clean_ts.split(':')
                if len(parts) == 2:  # MM:SS
                    return int(parts[0]) * 60 + int(parts[1])
                elif len(parts) == 3:  # HH:MM:SS
                    return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
            except (ValueError, IndexError):
                return None
        
        validated_opportunities = []
        for mo in missed_opportunities:
            # Skip if explicitly marked as collected_later
            if mo.collected_later:
                continue
            
            # Check if same criteria appears in evidence
            if mo.criteria_name in evidence_by_criteria:
                mo_timestamp_sec = parse_timestamp(mo.timestamp) if mo.timestamp else None
                
                # Check if any evidence timestamp is later than the missed opportunity timestamp
                # Only filter out if we can definitively compare timestamps
                evidence_collected_later = False
                for ev_timestamp in evidence_by_criteria[mo.criteria_name]:
                    ev_timestamp_sec = parse_timestamp(ev_timestamp)
                    # Only filter if we can parse both timestamps and evidence is clearly later
                    if mo_timestamp_sec is not None and ev_timestamp_sec is not None:
                        if ev_timestamp_sec > mo_timestamp_sec:
                            evidence_collected_later = True
                            break
                
                if evidence_collected_later:
                    # Information was collected later, so this is not a true missed opportunity
                    continue
            
            validated_opportunities.append(mo)
        
        return validated_opportunities

    def _parse_classification(self, classification_text: str) -> Optional[CallClassification]:
        """Parse classification result from the Call Classifier agent"""
        import sys
        print(f"📊 Parsing classification result ({len(classification_text)} chars)...", flush=True)
        print(f"📊 Classification text preview: {classification_text[:500]}...", flush=True)

        try:
            # Extract JSON from the result
            json_start = classification_text.find('{')
            json_end = classification_text.rfind('}') + 1
            print(f"📊 JSON bounds: start={json_start}, end={json_end}", flush=True)

            if json_start >= 0 and json_end > json_start:
                json_str = classification_text[json_start:json_end]
                print(f"📊 Extracted JSON length: {len(json_str)}", flush=True)
                data = json.loads(json_str)
                print(f"📊 Parsed JSON keys: {list(data.keys())}", flush=True)
            else:
                print("⚠️  No JSON found in classification result", flush=True)
                return None

            # Parse call type
            call_type_str = data.get("call_type", "unclear").lower()
            call_type_map = {
                "discovery": CallType.DISCOVERY,
                "poc_scoping": CallType.POC_SCOPING,
                "mixed": CallType.MIXED,
                "unclear": CallType.UNCLEAR
            }
            call_type = call_type_map.get(call_type_str, CallType.UNCLEAR)

            # Parse discovery criteria
            discovery_data = data.get("discovery_criteria", {})
            discovery_criteria = None
            discovery_score = 0.0

            if discovery_data:
                try:
                    pain_data = discovery_data.get("pain_current_state", {})
                    pain = PainCurrentState(
                        primary_use_case=pain_data.get("primary_use_case"),
                        prompt_model_iteration_understood=pain_data.get("prompt_model_iteration_understood", False),
                        debugging_process_documented=pain_data.get("debugging_process_documented", False),
                        situation_understood=pain_data.get("situation_understood", False),
                        resolution_attempts_documented=pain_data.get("resolution_attempts_documented", False),
                        outcomes_documented=pain_data.get("outcomes_documented", False),
                        frequency_quantified=pain_data.get("frequency_quantified", False),
                        duration_quantified=pain_data.get("duration_quantified", False),
                        impact_quantified=pain_data.get("impact_quantified", False),
                        people_impact_understood=pain_data.get("people_impact_understood", False),
                        process_impact_understood=pain_data.get("process_impact_understood", False),
                        technology_impact_understood=pain_data.get("technology_impact_understood", False),
                        mttd_mttr_quantified=pain_data.get("mttd_mttr_quantified", False),
                        experiment_time_quantified=pain_data.get("experiment_time_quantified", False),
                        notes=pain_data.get("notes"),
                        evidence=self._parse_evidence(pain_data.get("evidence", [])),
                        missed_opportunities=self._validate_missed_opportunities(
                            self._parse_missed_opportunities(pain_data.get("missed_opportunities", [])),
                            self._parse_evidence(pain_data.get("evidence", []))
                        )
                    )

                    stakeholder_data = discovery_data.get("stakeholder_map", {})
                    stakeholder = StakeholderMap(
                        technical_champion_identified=stakeholder_data.get("technical_champion_identified", False),
                        technical_champion_engaged=stakeholder_data.get("technical_champion_engaged", False),
                        economic_buyer_identified=stakeholder_data.get("economic_buyer_identified", False),
                        decision_maker_confirmed=stakeholder_data.get("decision_maker_confirmed", False),
                        notes=stakeholder_data.get("notes"),
                        evidence=self._parse_evidence(stakeholder_data.get("evidence", [])),
                        missed_opportunities=self._validate_missed_opportunities(
                            self._parse_missed_opportunities(stakeholder_data.get("missed_opportunities", [])),
                            self._parse_evidence(stakeholder_data.get("evidence", []))
                        )
                    )

                    rc_data = discovery_data.get("required_capabilities", {})
                    required_caps = RequiredCapabilities(
                        top_rcs_ranked=rc_data.get("top_rcs_ranked", False),
                        llm_agent_tracing_important=rc_data.get("llm_agent_tracing_important"),
                        llm_evaluations_important=rc_data.get("llm_evaluations_important"),
                        production_monitoring_important=rc_data.get("production_monitoring_important"),
                        prompt_management_important=rc_data.get("prompt_management_important"),
                        prompt_experimentation_important=rc_data.get("prompt_experimentation_important"),
                        monitoring_important=rc_data.get("monitoring_important"),
                        compliance_important=rc_data.get("compliance_important"),
                        must_have_vs_nice_to_have_distinguished=rc_data.get("must_have_vs_nice_to_have_distinguished", False),
                        deal_breakers_identified=rc_data.get("deal_breakers_identified", False),
                        notes=rc_data.get("notes"),
                        evidence=self._parse_evidence(rc_data.get("evidence", [])),
                        missed_opportunities=self._validate_missed_opportunities(
                            self._parse_missed_opportunities(rc_data.get("missed_opportunities", [])),
                            self._parse_evidence(rc_data.get("evidence", []))
                        )
                    )

                    comp_data = discovery_data.get("competitive_landscape", {})
                    competitive = CompetitiveLandscape(
                        current_tools_evaluated=comp_data.get("current_tools_evaluated", False),
                        tools_mentioned=comp_data.get("tools_mentioned", []),
                        why_looking_vs_staying=comp_data.get("why_looking_vs_staying", False),
                        key_differentiators_identified=comp_data.get("key_differentiators_identified", False),
                        notes=comp_data.get("notes"),
                        evidence=self._parse_evidence(comp_data.get("evidence", [])),
                        missed_opportunities=self._validate_missed_opportunities(
                            self._parse_missed_opportunities(comp_data.get("missed_opportunities", [])),
                            self._parse_evidence(comp_data.get("evidence", []))
                        )
                    )

                    discovery_criteria = DiscoveryCriteria(
                        pain_current_state=pain,
                        stakeholder_map=stakeholder,
                        required_capabilities=required_caps,
                        competitive_landscape=competitive
                    )
                    discovery_score = discovery_criteria.overall_completion_score
                except Exception as e:
                    print(f"⚠️  Error parsing discovery criteria: {e}")

            # Parse PoC scoping criteria
            poc_data = data.get("poc_scoping_criteria", {})
            poc_criteria = None
            poc_score = 0.0

            if poc_data:
                try:
                    use_case_data = poc_data.get("use_case_scoped", {})
                    use_case = UseCaseScoped(
                        llm_applications_selected=use_case_data.get("llm_applications_selected", False),
                        applications_list=use_case_data.get("applications_list", []),
                        environment_decided=use_case_data.get("environment_decided", False),
                        environment_type=use_case_data.get("environment_type"),
                        trace_volume_estimated=use_case_data.get("trace_volume_estimated", False),
                        estimated_volume=use_case_data.get("estimated_volume"),
                        llm_provider_identified=use_case_data.get("llm_provider_identified", False),
                        llm_provider=use_case_data.get("llm_provider"),
                        integration_complexity_assessed=use_case_data.get("integration_complexity_assessed", False),
                        notes=use_case_data.get("notes"),
                        evidence=self._parse_evidence(use_case_data.get("evidence", [])),
                        missed_opportunities=self._validate_missed_opportunities(
                            self._parse_missed_opportunities(use_case_data.get("missed_opportunities", [])),
                            self._parse_evidence(use_case_data.get("evidence", []))
                        )
                    )

                    impl_data = poc_data.get("implementation_requirements", {})
                    impl_req = ImplementationRequirements(
                        data_residency_confirmed=impl_data.get("data_residency_confirmed", False),
                        deployment_model=impl_data.get("deployment_model"),
                        blockers_identified=impl_data.get("blockers_identified", False),
                        blockers_list=impl_data.get("blockers_list", []),
                        notes=impl_data.get("notes"),
                        evidence=self._parse_evidence(impl_data.get("evidence", [])),
                        missed_opportunities=self._validate_missed_opportunities(
                            self._parse_missed_opportunities(impl_data.get("missed_opportunities", [])),
                            self._parse_evidence(impl_data.get("evidence", []))
                        )
                    )

                    metrics_data = poc_data.get("metrics_success_criteria", {})
                    metrics = MetricsSuccessCriteria(
                        specific_metrics_defined=metrics_data.get("specific_metrics_defined", False),
                        example_metrics=metrics_data.get("example_metrics", []),
                        baseline_captured=metrics_data.get("baseline_captured", False),
                        success_measurement_agreed=metrics_data.get("success_measurement_agreed", False),
                        competitive_favorable_criteria=metrics_data.get("competitive_favorable_criteria", False),
                        notes=metrics_data.get("notes"),
                        evidence=self._parse_evidence(metrics_data.get("evidence", [])),
                        missed_opportunities=self._validate_missed_opportunities(
                            self._parse_missed_opportunities(metrics_data.get("missed_opportunities", [])),
                            self._parse_evidence(metrics_data.get("evidence", []))
                        )
                    )

                    timeline_data = poc_data.get("timeline_milestones", {})
                    timeline = TimelineMilestones(
                        poc_duration_defined=timeline_data.get("poc_duration_defined", False),
                        duration_weeks=timeline_data.get("duration_weeks"),
                        key_milestones_with_dates=timeline_data.get("key_milestones_with_dates", False),
                        milestones=timeline_data.get("milestones", []),
                        decision_date_committed=timeline_data.get("decision_date_committed", False),
                        decision_date=timeline_data.get("decision_date"),
                        next_steps_discussed=timeline_data.get("next_steps_discussed", False),
                        notes=timeline_data.get("notes"),
                        evidence=self._parse_evidence(timeline_data.get("evidence", [])),
                        missed_opportunities=self._validate_missed_opportunities(
                            self._parse_missed_opportunities(timeline_data.get("missed_opportunities", [])),
                            self._parse_evidence(timeline_data.get("evidence", []))
                        )
                    )

                    resources_data = poc_data.get("resources_committed", {})
                    resources = ResourcesCommitted(
                        engineering_resources_allocated=resources_data.get("engineering_resources_allocated", False),
                        resource_names=resources_data.get("resource_names", []),
                        checkin_cadence_established=resources_data.get("checkin_cadence_established", False),
                        cadence=resources_data.get("cadence"),
                        communication_channel_created=resources_data.get("communication_channel_created", False),
                        notes=resources_data.get("notes"),
                        evidence=self._parse_evidence(resources_data.get("evidence", [])),
                        missed_opportunities=self._validate_missed_opportunities(
                            self._parse_missed_opportunities(resources_data.get("missed_opportunities", [])),
                            self._parse_evidence(resources_data.get("evidence", []))
                        )
                    )

                    poc_criteria = PocScopingCriteria(
                        use_case_scoped=use_case,
                        implementation_requirements=impl_req,
                        metrics_success_criteria=metrics,
                        timeline_milestones=timeline,
                        resources_committed=resources
                    )
                    poc_score = poc_criteria.overall_completion_score
                except Exception as e:
                    print(f"⚠️  Error parsing PoC criteria: {e}")

            # Create the classification result
            classification = CallClassification(
                call_type=call_type,
                confidence=data.get("confidence", "medium"),
                reasoning=data.get("reasoning", ""),
                discovery_criteria=discovery_criteria,
                discovery_completion_score=discovery_score,
                poc_scoping_criteria=poc_criteria,
                poc_scoping_completion_score=poc_score,
                missing_elements=self._parse_missing_elements(data.get("missing_elements", {})),
                recommendations=data.get("recommendations", [])
            )

            print(f"✅ Classification: {call_type.value} (confidence: {classification.confidence})", flush=True)
            print(f"✅ Discovery score: {classification.discovery_completion_score}%, PoC score: {classification.poc_scoping_completion_score}%", flush=True)
            print(f"   Discovery score: {discovery_score:.1f}%")
            print(f"   PoC Scoping score: {poc_score:.1f}%")

            return classification

        except json.JSONDecodeError as e:
            print(f"⚠️  Failed to parse classification JSON: {e}")
            return None
        except Exception as e:
            print(f"⚠️  Error parsing classification: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _parse_missing_elements(self, data) -> MissingElements:
        """Parse missing elements from classification data, handling both old and new formats."""
        # Handle new nested format: {"discovery": [...], "poc_scoping": [...]}
        if isinstance(data, dict):
            return MissingElements(
                discovery=data.get("discovery", []),
                poc_scoping=data.get("poc_scoping", [])
            )
        # Handle old flat list format for backwards compatibility
        elif isinstance(data, list):
            # Put all items in both categories as a fallback
            return MissingElements(
                discovery=data,
                poc_scoping=data
            )
        # Handle empty/None
        return MissingElements()

    def generate_recap_data(self, transcript: str, call_classification: Optional[CallClassification], analysis_data: dict, call_date: str = "") -> RecapSlideData:
        """
        Generate recap slide data using an LLM to synthesize Key Initiatives, Challenges,
        Solution Requirements, and Follow-up Questions.
        
        Args:
            transcript: The call transcript
            call_classification: The call classification result (if available)
            analysis_data: The parsed analysis data from the crew
            call_date: The date of the call (from Gong metadata or empty)
            
        Returns:
            RecapSlideData with the recap sections and follow-up questions
        """
        tracer = trace.get_tracer("recap-generator")
        
        with tracer.start_as_current_span("generate_recap_data") as span:
            span.set_attribute("openinference.span.kind", "chain")
            
            # Build context from classification data
            classification_context = ""
            missed_opportunities = []
            missing_elements_list = []
            recommendations_list = []
            
            if call_classification:
                # Extract Missing Elements (from classification)
                if call_classification.missing_elements:
                    if call_classification.missing_elements.discovery:
                        missing_elements_list.extend(call_classification.missing_elements.discovery)
                    if call_classification.missing_elements.poc_scoping:
                        missing_elements_list.extend(call_classification.missing_elements.poc_scoping)
                
                # Extract Recommendations for Next Call
                if call_classification.recommendations:
                    recommendations_list = call_classification.recommendations[:5]
                
                if call_classification.discovery_criteria:
                    dc = call_classification.discovery_criteria
                    classification_context += f"""
Discovery Information:
- Pain/Current State Notes: {dc.pain_current_state.notes if dc.pain_current_state else 'N/A'}
- Required Capabilities Notes: {dc.required_capabilities.notes if dc.required_capabilities else 'N/A'}
- Competitive Landscape: {dc.competitive_landscape.notes if dc.competitive_landscape else 'N/A'}
"""
                    # Collect missed opportunities for question generation
                    if dc.pain_current_state and dc.pain_current_state.missed_opportunities:
                        missed_opportunities.extend(dc.pain_current_state.missed_opportunities)
                    if dc.stakeholder_map and dc.stakeholder_map.missed_opportunities:
                        missed_opportunities.extend(dc.stakeholder_map.missed_opportunities)
                    if dc.required_capabilities and dc.required_capabilities.missed_opportunities:
                        missed_opportunities.extend(dc.required_capabilities.missed_opportunities)
                        
                if call_classification.poc_scoping_criteria:
                    pc = call_classification.poc_scoping_criteria
                    classification_context += f"""
PoC Scoping Information:
- Use Case Notes: {pc.use_case_scoped.notes if pc.use_case_scoped else 'N/A'}
- Success Metrics: {pc.metrics_success_criteria.notes if pc.metrics_success_criteria else 'N/A'}
"""
                    # Collect missed opportunities from PoC scoping
                    if pc.use_case_scoped and pc.use_case_scoped.missed_opportunities:
                        missed_opportunities.extend(pc.use_case_scoped.missed_opportunities)
                    if pc.metrics_success_criteria and pc.metrics_success_criteria.missed_opportunities:
                        missed_opportunities.extend(pc.metrics_success_criteria.missed_opportunities)
            
            # Extract suggested questions from missed opportunities
            suggested_questions = []
            for mo in missed_opportunities[:8]:
                if hasattr(mo, 'suggested_question') and mo.suggested_question:
                    suggested_questions.append(mo.suggested_question)
            
            from call_analysis_llm import chat_invoke_text
            from call_analysis_prompts import recap_slide_prompt

            recap_prompt = recap_slide_prompt(
                transcript,
                classification_context,
                analysis_data.get("call_summary", "N/A"),
                json.dumps(missing_elements_list[:8], indent=2) if missing_elements_list else "",
                json.dumps(recommendations_list, indent=2) if recommendations_list else "",
                json.dumps(suggested_questions, indent=2) if suggested_questions else "",
            )

            try:
                result_text = chat_invoke_text(self._chat, recap_prompt)
                
                # Parse the JSON result
                json_start = result_text.find('{')
                json_end = result_text.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    recap_json = json.loads(result_text[json_start:json_end])
                    
                    # Use LLM-extracted date, or fall back to Gong metadata date
                    extracted_date = recap_json.get("call_date", "")
                    final_date = extracted_date if extracted_date else call_date
                    
                    recap_data = RecapSlideData(
                        customer_name=recap_json.get("customer_name", ""),
                        call_date=final_date,
                        key_initiatives=recap_json.get("key_initiatives", []),
                        challenges=recap_json.get("challenges", []),
                        solution_requirements=recap_json.get("solution_requirements", []),
                        follow_up_questions=recap_json.get("follow_up_questions", [])
                    )
                    
                    print(f"✅ Recap data extracted - Customer: '{recap_data.customer_name}', Date: '{recap_data.call_date}' (from {'LLM' if extracted_date else 'Gong metadata'})")
                    
                    span.set_attribute("recap.customer_name", recap_data.customer_name)
                    span.set_attribute("recap.call_date", recap_data.call_date)
                    span.set_attribute("recap.sections_generated", 4)
                    span.set_attribute("recap.questions_count", len(recap_data.follow_up_questions))
                    span.set_status(Status(StatusCode.OK))
                    return recap_data
                    
            except Exception as e:
                print(f"⚠️ Failed to generate recap data: {e}")
                span.set_status(Status(StatusCode.ERROR, str(e)))
            
            # Return empty recap data on failure
            return RecapSlideData()

    def _extract_insights_from_raw_text(self, raw_text: str) -> dict:
        """
        Extract meaningful insights from raw LLM output when JSON parsing fails.
        Uses regex patterns and heuristics to find strengths, improvements, and key moments.
        """
        insights = []
        strengths = []
        improvements = []
        key_moments = []
        call_summary = ""
        
        # Try to extract call summary
        summary_patterns = [
            r'(?:call[_\s]*summary|summary)[:\s]*["\']?([^"\'\n]+)["\']?',
            r'(?:overview|assessment)[:\s]*["\']?([^"\'\n]+)["\']?',
        ]
        for pattern in summary_patterns:
            match = re.search(pattern, raw_text, re.IGNORECASE)
            if match:
                call_summary = match.group(1).strip()
                break
        
        if not call_summary:
            # Look for first substantive sentence
            sentences = re.split(r'[.!?]\s+', raw_text[:500])
            for sentence in sentences:
                if len(sentence) > 30 and not sentence.startswith('{'):
                    call_summary = sentence.strip() + "."
                    break
        
        # Extract strengths - look for positive indicators
        strength_patterns = [
            r'(?:strength|positive|well|good|excellent|effective)[s]?[:\s]*[-•*]?\s*([^\n•*]+)',
            r'✓\s*([^\n]+)',
            r'(?:did well|succeeded|strong)[:\s]*([^\n]+)',
        ]
        for pattern in strength_patterns:
            matches = re.findall(pattern, raw_text, re.IGNORECASE)
            for match in matches:
                clean_match = match.strip().strip('",').strip()
                if clean_match and len(clean_match) > 10 and clean_match not in strengths:
                    strengths.append(clean_match[:200])
        
        # Extract improvements - look for negative indicators or suggestions
        improvement_patterns = [
            r'(?:improvement|could\s*improve|area[s]?\s*for\s*improvement|missed|opportunity|should)[:\s]*[-•*]?\s*([^\n•*]+)',
            r'(?:→|➤|>)\s*([^\n]+)',
            r'(?:consider|recommend|suggest)[:\s]*([^\n]+)',
            r'(?:instead of|rather than|better approach)[:\s]*([^\n]+)',
        ]
        for pattern in improvement_patterns:
            matches = re.findall(pattern, raw_text, re.IGNORECASE)
            for match in matches:
                clean_match = match.strip().strip('",').strip()
                if clean_match and len(clean_match) > 10 and clean_match not in improvements:
                    improvements.append(clean_match[:200])
        
        # Extract timestamps and key moments
        timestamp_pattern = r'\[(\d{1,2}:\d{2}(?::\d{2})?)\][:\s]*([^\n\[]+)'
        timestamp_matches = re.findall(timestamp_pattern, raw_text)
        for ts, description in timestamp_matches[:10]:  # Limit to 10 moments
            key_moments.append({
                "timestamp": f"[{ts}]",
                "description": description.strip()[:150]
            })
        
        # Extract structured insights - look for what_happened/better_approach patterns
        insight_patterns = [
            r'(?:what\s*happened|issue|problem)[:\s]*([^\n]+)(?:.*?(?:why\s*it\s*matters|impact)[:\s]*([^\n]+))?(?:.*?(?:better\s*approach|recommendation)[:\s]*([^\n]+))?',
        ]
        for pattern in insight_patterns:
            matches = re.findall(pattern, raw_text, re.IGNORECASE | re.DOTALL)
            for match in matches[:5]:  # Limit to 5 insights
                what_happened = match[0].strip() if len(match) > 0 and match[0] else ""
                why_matters = match[1].strip() if len(match) > 1 and match[1] else ""
                better_approach = match[2].strip() if len(match) > 2 and match[2] else ""
                
                if what_happened and len(what_happened) > 15:
                    insights.append({
                        "category": "Call Feedback",
                        "severity": "important",
                        "timestamp": None,
                        "conversation_snippet": None,
                        "what_happened": what_happened[:300],
                        "why_it_matters": why_matters[:300] if why_matters else "This affects call effectiveness and customer engagement.",
                        "better_approach": better_approach[:300] if better_approach else "Review specific recommendations in the analysis.",
                        "example_phrasing": None
                    })
        
        # If we didn't find structured insights, create them from improvements
        if not insights and improvements:
            for i, imp in enumerate(improvements[:5]):
                insights.append({
                    "category": "Improvement Opportunity",
                    "severity": "important",
                    "timestamp": None,
                    "conversation_snippet": None,
                    "what_happened": imp,
                    "why_it_matters": "Addressing this will improve call effectiveness.",
                    "better_approach": "See the detailed analysis for specific recommendations.",
                    "example_phrasing": None
                })
        
        # Ensure we have at least some content
        if not strengths:
            # Try to find any positive statements
            positive_words = re.findall(r'(?:effectively|successfully|clearly|well)[^.]*\.', raw_text, re.IGNORECASE)
            strengths = [pw.strip() for pw in positive_words[:3] if len(pw) > 20]
        
        if not improvements:
            # Try to find any suggestion statements  
            suggestion_words = re.findall(r'(?:could have|should have|next time|in future)[^.]*\.', raw_text, re.IGNORECASE)
            improvements = [sw.strip() for sw in suggestion_words[:3] if len(sw) > 20]
        
        # Final fallback - provide the raw text as context
        if not strengths:
            strengths = ["Analysis completed - see detailed output for specific strengths identified during the call."]
        if not improvements:
            improvements = ["Analysis completed - see detailed output for specific improvement opportunities."]
        if not insights:
            # Create a meaningful fallback insight with context
            insights = [{
                "category": "Analysis Summary",
                "severity": "important",
                "timestamp": None,
                "conversation_snippet": None,
                "what_happened": "Comprehensive analysis performed for the sales rep's call performance.",
                "why_it_matters": "Multi-agent review provides thorough evaluation of technical depth, discovery quality, and sales methodology.",
                "better_approach": "Review the strengths and improvement areas sections below for actionable feedback.",
                "example_phrasing": None
            }]
        
        if not call_summary:
            call_summary = "Call analysis complete. Review insights below for specific feedback."
        
        return {
            "call_summary": call_summary,
            "overall_score": 7.0,
            "command_scores": {
                "problem_identification": 7.0,
                "differentiation": 7.0,
                "proof_evidence": 7.0,
                "required_capabilities": 7.0
            },
            "sa_metrics": {
                "technical_depth": 7.0,
                "discovery_quality": 7.0,
                "active_listening": 7.0,
                "value_articulation": 7.0
            },
            "top_insights": insights,
            "strengths": strengths[:5],  # Limit to 5
            "improvement_areas": improvements[:5],  # Limit to 5
            "key_moments": key_moments[:10]  # Limit to 10
        }
