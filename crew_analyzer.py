import os
import json
import re
from typing import List, Optional
from crewai import Agent, Task, Crew, Process
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
    MissingElements
)
from dotenv import load_dotenv
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from observability import force_flush_spans
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from typing import Any, Dict, List
from uuid import UUID

load_dotenv()


class TokenTrackingCallback(BaseCallbackHandler):
    """
    LangChain callback that captures token usage and adds it to the current OpenTelemetry span.
    This enables cost tracking in Arize by providing token counts.
    """
    
    def __init__(self):
        self.tracer = trace.get_tracer("token-tracking")
    
    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        """Capture token usage when LLM call completes."""
        current_span = trace.get_current_span()
        
        if response.llm_output:
            token_usage = response.llm_output.get("token_usage", {})
            model_name = response.llm_output.get("model_name", "unknown")
            
            # Also check for usage in different formats
            if not token_usage and "usage" in response.llm_output:
                token_usage = response.llm_output["usage"]
            
            prompt_tokens = token_usage.get("prompt_tokens", 0)
            completion_tokens = token_usage.get("completion_tokens", 0)
            total_tokens = token_usage.get("total_tokens", prompt_tokens + completion_tokens)
            
            if current_span and current_span.is_recording():
                # OpenInference semantic conventions for token counts
                current_span.set_attribute("llm.token_count.prompt", prompt_tokens)
                current_span.set_attribute("llm.token_count.completion", completion_tokens)
                current_span.set_attribute("llm.token_count.total", total_tokens)
                current_span.set_attribute("llm.model_name", model_name)
                
                # Also add as an event for visibility
                current_span.add_event("llm_token_usage", {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "model": model_name
                })
                
            print(f"📊 Token usage: {prompt_tokens} prompt + {completion_tokens} completion = {total_tokens} total ({model_name})")


class SACallAnalysisCrew:
    """
    CrewAI-powered SA Call Analysis System.

    Uses 4 specialized agents to comprehensively analyze Solution Architect performance:
    1. SA Identifier - Identifies the Solution Architect
    2. Technical Evaluator - Assesses technical skills
    3. Sales Methodology & Discovery Expert - Evaluates discovery and Command of Message framework
    4. Report Compiler - Synthesizes feedback
    """

    def __init__(self):
        # Determine which LLM to use based on environment
        use_litellm = os.getenv("USE_LITELLM", "false").lower() == "true"
        model_name = os.getenv("MODEL_NAME", "claude-3-5-sonnet-20241022")
        
        # Create token tracking callback for cost visibility
        self.token_callback = TokenTrackingCallback()

        if use_litellm:
            # Use LiteLLM directly for LLM calls (enables LiteLLMInstrumentor to capture spans)
            from langchain_litellm import ChatLiteLLM
            print(f"🔧 LiteLLM Config: model={model_name}")
            self.llm = ChatLiteLLM(
                model=model_name,
                temperature=0.7,
                max_tokens=4096,
                callbacks=[self.token_callback]
            )
        else:
            # Use Anthropic directly
            from langchain_anthropic import ChatAnthropic
            self.llm = ChatAnthropic(
                model=model_name,
                anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
                temperature=0.7,
                callbacks=[self.token_callback]
            )
        
        # Store model name for cost calculations
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

    def create_agents(self):
        """Create all specialized agents"""

        sa_identifier = Agent(
            role='Solution Architect Identifier',
            goal='Accurately identify who the Solution Architect is in the call transcript',
            backstory="""You are an expert at analyzing conversation patterns and roles.
            You can identify Solution Architects by looking for:
            - Technical architecture discussions
            - Data integration and infrastructure talks
            - Technical scoping and requirements gathering
            - Feature and implementation details
            - Technical problem-solving""",
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )

        technical_evaluator = Agent(
            role='Senior Technical Architect & Evaluator',
            goal='Assess the Solution Architect\'s technical depth, accuracy, and demonstration quality',
            backstory="""You are a seasoned technical architect with 15+ years of experience.
            You evaluate:
            - Technical accuracy and depth of explanations
            - Architecture and design discussions
            - Demo quality and effectiveness
            - Ability to explain complex technical concepts simply
            - Integration and implementation feasibility
            You provide specific, actionable feedback on technical performance.""",
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )

        sales_methodology_expert = Agent(
            role='Sales Methodology & Discovery Expert',
            goal='Evaluate discovery quality and score Command of the Message framework performance',
            backstory="""You are a world-class sales coach certified in Command of the Message framework.

            IMPORTANT: Recognize that small talk and rapport building at the beginning of calls is NORMAL and VALUABLE.
            Do not penalize SAs for spending 1-3 minutes on greetings, weather, weekend plans, or casual conversation.
            This is an essential part of building trust and relationships with customers.

            You evaluate TWO key areas:

            1. DISCOVERY & ENGAGEMENT:
               - Quality and depth of discovery questions (after initial rapport building)
               - Active listening skills and follow-up questions
               - Engagement with customer responses
               - Uncovering pain points and needs
               - Building rapport and trust (including appropriate small talk)

            2. COMMAND OF THE MESSAGE FRAMEWORK (score each 1-10):
               - Problem Identification: Did they uncover real business problems?
               - Differentiation: Did they articulate unique value vs competitors?
               - Proof/Evidence: Did they provide case studies, metrics, demos?
               - Required Capabilities: Did they tie technical features to business outcomes?

            You provide specific examples with timestamps and actionable recommendations.""",
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )

        report_compiler = Agent(
            role='Executive Performance Coach & Report Writer',
            goal='Synthesize all feedback into actionable, specific recommendations with timestamps',
            backstory="""You are an executive coach who creates clear, actionable performance reports.
            You compile insights from all analysts and create:
            - Overall performance scores
            - Top 3-5 high-impact improvements with timestamps
            - Specific alternative phrases and approaches
            - Strengths to reinforce
            - Key moments from the call
            Your reports are concise, specific, and immediately actionable.""",
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )

        call_classifier = Agent(
            role='Sales Call Type Classifier & Criteria Assessor',
            goal='Classify calls as Discovery or PoC Scoping and assess completion of required criteria',
            backstory="""You are an expert sales operations analyst who specializes in classifying 
            sales calls and assessing deal progression. You understand the critical difference between:
            
            DISCOVERY CALLS - Focus on:
            1. Pain & Current State: Understanding how they debug LLM/agent issues, quantifying MTTD/MTTR
            2. Stakeholder Map: Identifying technical champion, economic buyer, decision maker
            3. Required Capabilities: Prioritizing RCs (tracing, evals, monitoring, datasets, compliance)
            4. Competitive Landscape: Understanding current tools (LangSmith, W&B, Braintrust, Datadog)
            
            POC SCOPING CALLS - Focus on:
            1. Use Case Scoped: Specific LLM apps selected, environment, trace volume, integration complexity
            2. Implementation Requirements: Data residency, deployment model, blockers identified
            3. Metrics & Success Criteria: Specific metrics defined, baselines captured
            4. Timeline & Milestones: PoC duration, key dates, decision date committed
            5. Resources Committed: Engineering resources allocated, check-in cadence, Slack channel
            
            You provide structured assessments with specific evidence from the transcript.""",
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )

        return {
            'sa_identifier': sa_identifier,
            'technical_evaluator': technical_evaluator,
            'sales_methodology_expert': sales_methodology_expert,
            'report_compiler': report_compiler,
            'call_classifier': call_classifier
        }

    def analyze_call(
        self,
        transcript: str,
        speakers: List[str],
        manual_sa: Optional[str] = None,
        transcript_data: Optional[dict] = None
    ) -> AnalysisResult:
        """
        Analyze a call transcript using the crew of specialized agents.

        Args:
            transcript: The call transcript (formatted)
            speakers: List of speaker names (if available)
            manual_sa: Manually specified SA name (optional)
            transcript_data: Raw transcript data from Gong (optional, for hybrid sampling)

        Returns:
            AnalysisResult with structured analysis
        """

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
            result = None  # Initialize result
            try:
                agents = self.create_agents()

                # Task 1: Identify SA
                with tracer.start_as_current_span("identify_sa") as sa_span:
                    sa_span.set_attribute("openinference.span.kind", "agent")
                    sa_span.set_attribute("speakers.available", ", ".join(speakers) if speakers else "none")
                    sa_span.set_attribute("speakers.count", len(speakers))
                    # OpenInference input
                    sa_span.set_attribute("input.value", json.dumps({
                        "speakers": speakers,
                        "transcript_preview": transcript[:500] + "..." if len(transcript) > 500 else transcript,
                        "manual_sa": manual_sa
                    }))
                    sa_span.set_attribute("input.mime_type", "application/json")

                    if manual_sa:
                        sa_name = manual_sa
                        sa_identification_result = f"SA manually specified: {manual_sa}"
                        sa_span.set_attribute("identification.method", "manual")
                        sa_span.add_event("manual_sa_provided", {"sa_name": manual_sa})
                    else:
                        sa_span.set_attribute("identification.method", "crew_analysis")

                        # Use hybrid sampling if raw transcript data is available
                        if transcript_data:
                            sa_span.set_attribute("identification.sampling_method", "hybrid")
                            from gong_mcp_client import GongMCPClient
                            gong_client = GongMCPClient()

                            # Extract call_id from transcript_data if available
                            call_id = None
                            if "callTranscripts" in transcript_data:
                                call_id = transcript_data["callTranscripts"][0].get("callId")

                            transcript_sample = gong_client.create_hybrid_sample_for_sa_identification(
                                transcript_data,
                                call_id=call_id,
                                max_speakers=3
                            )
                            sa_span.add_event("hybrid_sampling_used", {
                                "sample_length": len(transcript_sample)
                            })
                            print(f"📊 Using hybrid sampling: {len(transcript_sample)} chars")
                        else:
                            # Fallback to old character-based sampling
                            sa_span.set_attribute("identification.sampling_method", "character_based")
                            transcript_len = len(transcript)
                            if transcript_len <= 5000:
                                transcript_sample = transcript
                            else:
                                # Sample from beginning, middle, and end
                                beginning = transcript[:2000]
                                middle_start = transcript_len // 2 - 1000
                                middle = transcript[middle_start:middle_start + 2000]
                                end = transcript[-1000:]
                                transcript_sample = f"{beginning}\n\n... [middle section] ...\n\n{middle}\n\n... [later section] ...\n\n{end}"
                            print(f"📊 Using character-based sampling: {len(transcript_sample)} chars")

                        identify_sa_task = Task(
                            description=f"""Analyze this call transcript sample and identify who the Solution Architect (SA) is.

                            Available speakers: {', '.join(speakers) if speakers else 'Unknown (no speaker labels)'}

                            {transcript_sample}

                            The Solution Architect is typically the person who:
                            - Discusses technical architecture and implementation
                            - Explains product features and capabilities
                            - Answers technical questions
                            - Conducts demos or technical walkthroughs
                            - Discusses integration and data architecture
                            - Has higher word count and longer, more technical turns

                            NOTE: Initial small talk and pleasantries are normal - look beyond that for technical content.
                            If statistics are provided, heavily weight the speaker with higher word count and SA likelihood score.

                            Provide:
                            1. The SA's name (if names are provided, use the actual name; otherwise use the speaker ID)
                            2. Confidence level (high/medium/low)
                            3. Brief reasoning based on the statistics and content

                            Format: "SA Name: <name or speaker_id>, Confidence: <level>, Reasoning: <brief explanation>"
                            """,
                            agent=agents['sa_identifier'],
                            expected_output="SA identification with speaker ID, confidence, and reasoning"
                        )

                        # Execute SA identification
                        sa_span.add_event("starting_sa_crew")
                        sa_crew = Crew(
                            agents=[agents['sa_identifier']],
                            tasks=[identify_sa_task],
                            process=Process.sequential,
                            verbose=True
                        )
                        sa_identification_result = sa_crew.kickoff()
                        sa_name = self._extract_sa_name(str(sa_identification_result))
                        print(f"🔍 SA Identification Result: {sa_identification_result}")
                        print(f"✅ Extracted SA Name: {sa_name}")
                        sa_span.add_event("sa_crew_completed", {
                            "raw_result": str(sa_identification_result)[:200],
                            "extracted_name": sa_name
                        })

                    sa_span.set_attribute("sa.identified_name", sa_name)
                    # OpenInference output
                    sa_span.set_attribute("output.value", json.dumps({
                        "sa_name": sa_name,
                        "method": "manual" if manual_sa else "crew_analysis"
                    }))
                    sa_span.set_attribute("output.mime_type", "application/json")
                    sa_span.set_status(Status(StatusCode.OK))

                # Add SA identification to span
                analysis_span.add_event("sa_identified", {"sa_name": sa_name})

                # ============================================================
                # STEP 1: Run Call Classification (separate crew for clean output)
                # ============================================================
                with tracer.start_as_current_span("call_classification") as class_span:
                    class_span.set_attribute("openinference.span.kind", "agent")
                    class_span.set_attribute("input.value", transcript[:2000] + "..." if len(transcript) > 2000 else transcript)
                    class_span.set_attribute("input.mime_type", "text/plain")

                    classification_task = Task(
                        description=f"""Analyze this call transcript and classify it as either a DISCOVERY call or a POC SCOPING call.

                    Transcript:
                    {transcript}

                    DISCOVERY CALL CRITERIA - Look for evidence of:

                    1. PAIN & CURRENT STATE VALIDATED:
                       - What is the primary Use Case? (Development or Production focus?)
                       - DEVELOPMENT: How do they iterate on prompts/models today?
                       - PRODUCTION: How do they debug LLM/agent issues today?
                       - For either: What is the situation? What have they done to resolve? What outcomes?
                       - Frequency quantified (how often?)
                       - Duration quantified (how long?)
                       - Impact quantified (how much?) - People, Process, Technology
                       - METRICS: MTTD/MTTR for LLM failures quantified?
                       - METRICS: Average time to experiment with new prompts/models?

                    2. STAKEHOLDER MAP COMPLETE:
                       - Technical Champion identified and engaged?
                       - Economic Buyer identified?
                       - Decision Maker confirmed?

                    3. REQUIRED CAPABILITIES (RCs) PRIORITIZED:
                       - Top 2-3 RCs ranked by prospect?
                       - Core capabilities discussed:
                         * LLM/Agent Tracing
                         * LLM Evaluations
                         * Production Monitoring
                         * Prompt Management
                         * Prompt Experimentation
                         * Monitoring
                         * Compliance (SOC2, SSO, GDPR, etc.)
                       - "Must have" vs "nice to have" distinguished?
                       - Deal-breakers identified?

                    4. COMPETITIVE LANDSCAPE UNDERSTOOD:
                       - Current/prior tools evaluated (LangSmith, W&B, Braintrust, Datadog LLM)?
                       - Why they're looking vs. staying with current solution?
                       - Key differentiators that matter to this prospect?

                    POC SCOPING CALL CRITERIA - Look for evidence of:

                    1. USE CASE SCOPED:
                       - Specific LLM application(s) selected for PoC (not to exceed one use case)?
                       - Production vs. staging environment decision made?
                       - Expected trace volume estimated?
                       - LLM Provider identified (for gateway implementation)?
                       - Integration complexity assessed (# services, frameworks)?

                    2. IMPLEMENTATION REQUIREMENTS VALIDATED:
                       - Data residency / deployment model confirmed (Cloud, VPC, On-prem)?
                       - Any blockers identified (firewall, procurement, security review)?

                    3. METRICS & SUCCESS CRITERIA DEFINED:
                       - Prospect-specific metrics defined (e.g., "reduce debugging from 4hr to 30min")?
                       - Baseline measurements captured for before/after comparison?
                       - Agreement on how success will be measured?
                       - Success criteria favorable for Arize AX vs competitors (Galileo, Braintrust, LangSmith)?

                    4. TIMELINE & MILESTONES AGREED:
                       - PoC duration defined (typically 2-4 weeks)?
                       - Key milestones with dates (kickoff, integration complete, eval review, decision)?
                       - Decision date committed?
                       - Next steps after successful PoC discussed?

                    5. RESOURCES COMMITTED:
                       - Prospect engineering resources allocated (names, % time)?
                       - Weekly check-in cadence established?
                       - Slack/communication channel created?

                    IMPORTANT: For each criteria section, you MUST provide:
                    1. "evidence" - Array of evidence items showing WHERE and WHAT was captured
                    2. "missed_opportunities" - Array of moments where we COULD have gathered more info

                    Return your analysis as JSON with this EXACT structure:
                    {{
                        "call_type": "discovery" | "poc_scoping" | "mixed" | "unclear",
                        "confidence": "high" | "medium" | "low",
                        "reasoning": "Brief explanation of classification",
                        "discovery_criteria": {{
                            "pain_current_state": {{
                                "primary_use_case": "development" | "production" | "both" | null,
                                "prompt_model_iteration_understood": true/false,
                                "debugging_process_documented": true/false,
                                "situation_understood": true/false,
                                "resolution_attempts_documented": true/false,
                                "outcomes_documented": true/false,
                                "frequency_quantified": true/false,
                                "duration_quantified": true/false,
                                "impact_quantified": true/false,
                                "people_impact_understood": true/false,
                                "process_impact_understood": true/false,
                                "technology_impact_understood": true/false,
                                "mttd_mttr_quantified": true/false,
                                "experiment_time_quantified": true/false,
                                "notes": "summary of pain/current state understanding",
                                "evidence": [
                                    {{
                                        "criteria_name": "situation_understood",
                                        "captured": true,
                                        "timestamp": "[MM:SS]",
                                        "conversation_snippet": "Customer: 'We're struggling with...'",
                                        "speaker": "Customer Name"
                                    }}
                                ],
                                "missed_opportunities": [
                                    {{
                                        "criteria_name": "mttd_mttr_quantified",
                                        "timestamp": "[MM:SS]",
                                        "context": "Customer mentioned debugging issues but SA moved on",
                                        "suggested_question": "How long does it typically take your team to detect when an LLM response is incorrect?",
                                        "why_important": "Quantifying MTTD establishes baseline for ROI calculation"
                                    }}
                                ]
                            }},
                            "stakeholder_map": {{
                                "technical_champion_identified": true/false,
                                "technical_champion_engaged": true/false,
                                "economic_buyer_identified": true/false,
                                "decision_maker_confirmed": true/false,
                                "notes": "stakeholder summary",
                                "evidence": [],
                                "missed_opportunities": []
                            }},
                            "required_capabilities": {{
                                "top_rcs_ranked": true/false,
                                "llm_agent_tracing_important": true/false/null,
                                "llm_evaluations_important": true/false/null,
                                "production_monitoring_important": true/false/null,
                                "prompt_management_important": true/false/null,
                                "prompt_experimentation_important": true/false/null,
                                "monitoring_important": true/false/null,
                                "compliance_important": true/false/null,
                                "must_have_vs_nice_to_have_distinguished": true/false,
                                "deal_breakers_identified": true/false,
                                "notes": "RC summary",
                                "evidence": [],
                                "missed_opportunities": []
                            }},
                            "competitive_landscape": {{
                                "current_tools_evaluated": true/false,
                                "tools_mentioned": ["list", "of", "tools"],
                                "why_looking_vs_staying": true/false,
                                "key_differentiators_identified": true/false,
                                "notes": "competitive summary",
                                "evidence": [],
                                "missed_opportunities": []
                            }}
                        }},
                        "poc_scoping_criteria": {{
                            "use_case_scoped": {{
                                "llm_applications_selected": true/false,
                                "applications_list": ["list", "of", "apps"],
                                "environment_decided": true/false,
                                "environment_type": "production" | "staging" | "both" | null,
                                "trace_volume_estimated": true/false,
                                "estimated_volume": "e.g., 10K traces/day" | null,
                                "llm_provider_identified": true/false,
                                "llm_provider": "e.g., OpenAI, Anthropic, Azure OpenAI" | null,
                                "integration_complexity_assessed": true/false,
                                "notes": "use case summary",
                                "evidence": [],
                                "missed_opportunities": []
                            }},
                            "implementation_requirements": {{
                                "data_residency_confirmed": true/false,
                                "deployment_model": "cloud" | "vpc" | "on-prem" | null,
                                "blockers_identified": true/false,
                                "blockers_list": ["list", "of", "blockers"],
                                "notes": "implementation summary",
                                "evidence": [],
                                "missed_opportunities": []
                            }},
                            "metrics_success_criteria": {{
                                "specific_metrics_defined": true/false,
                                "example_metrics": ["list", "of", "metrics"],
                                "baseline_captured": true/false,
                                "success_measurement_agreed": true/false,
                                "competitive_favorable_criteria": true/false,
                                "notes": "metrics summary",
                                "evidence": [],
                                "missed_opportunities": []
                            }},
                            "timeline_milestones": {{
                                "poc_duration_defined": true/false,
                                "duration_weeks": 2-4 | null,
                                "key_milestones_with_dates": true/false,
                                "milestones": ["list", "of", "milestones"],
                                "decision_date_committed": true/false,
                                "decision_date": "date string" | null,
                                "next_steps_discussed": true/false,
                                "notes": "timeline summary",
                                "evidence": [],
                                "missed_opportunities": []
                            }},
                            "resources_committed": {{
                                "engineering_resources_allocated": true/false,
                                "resource_names": ["names"],
                                "checkin_cadence_established": true/false,
                                "cadence": "weekly" | "bi-weekly" | null,
                                "communication_channel_created": true/false,
                                "notes": "resources summary",
                                "evidence": [],
                                "missed_opportunities": []
                            }}
                        }},
                        "missing_elements": {{
                            "discovery": ["REQUIRED: 3-5 bullet points summarizing the KEY discovery gaps, e.g. 'Did not quantify business impact', 'Missing stakeholder map'"],
                            "poc_scoping": ["REQUIRED: 3-5 bullet points summarizing the KEY PoC scoping gaps, e.g. 'No success metrics defined', 'Timeline not discussed'"]
                        }},
                        "recommendations": ["Specific actions with example questions for next call"]
                        
                        IMPORTANT: The "missing_elements" field is REQUIRED and must contain a summary of the most important gaps.
                        - For discovery calls: Focus on pain/impact, stakeholders, competitive landscape gaps
                        - For poc_scoping calls: Focus on use case, metrics, timeline, resources gaps
                        - Always include at least 3 items in the relevant category based on call_type
                    }}
                    """,
                        agent=agents['call_classifier'],
                        expected_output="JSON classification with detailed criteria assessment"
                    )

                    # Run classification crew
                    class_span.add_event("starting_classification_crew")
                    classification_crew = Crew(
                        agents=[agents['call_classifier']],
                        tasks=[classification_task],
                        process=Process.sequential,
                        verbose=True
                    )
                    classification_result = classification_crew.kickoff()
                    classification_text = str(classification_result)
                    class_span.set_attribute("output.value", classification_text[:3000])
                    class_span.set_attribute("output.mime_type", "application/json")
                    class_span.add_event("classification_complete", {
                        "result_length": len(classification_text)
                    })

                    # Parse classification result
                    call_classification = self._parse_classification(classification_text)
                    if call_classification:
                        class_span.set_attribute("classification.call_type", call_classification.call_type.value)
                        class_span.set_attribute("classification.confidence", call_classification.confidence)
                        class_span.set_attribute("classification.discovery_score", call_classification.discovery_completion_score)
                        class_span.set_attribute("classification.poc_score", call_classification.poc_scoping_completion_score)
                    class_span.set_status(Status(StatusCode.OK))

                # ============================================================
                # STEP 2: Run Main Analysis (technical, sales methodology, report)
                # ============================================================

                # Task 3-5: Parallel analysis by different experts
                technical_task = Task(
                    description=f"""Analyze the technical performance of {sa_name} in this call.

                    Transcript:
                    {transcript}

                    Evaluate:
                    - Technical depth and accuracy
                    - Architecture/integration discussions
                    - Demo quality

                    CRITICAL REQUIREMENT: EXACT TIMESTAMPS
                    For EVERY finding, extract the EXACT timestamp from the transcript where this occurred.
                    Format: "[MM:SS]" or "[HH:MM:SS]" (e.g., "[05:23]", "[24:49]")

                    Return your analysis as JSON with an array of findings:
                    {{
                        "findings": [
                            {{
                                "timestamp": "[24:49]",
                                "issue": "Brief description of the technical issue or observation",
                                "recommendation": "Specific actionable recommendation"
                            }}
                        ]
                    }}

                    NOTE: Do NOT include conversation_snippet - we will extract actual quotes programmatically.
                    """,
                    agent=agents['technical_evaluator'],
                    expected_output="Technical evaluation with timestamps and specific feedback"
                )

                sales_methodology_task = Task(
                    description=f"""Analyze {sa_name}'s sales methodology and discovery performance in this call.

                    Transcript:
                    {transcript}

                    IMPORTANT CONTEXT: Small talk and rapport building at the start of calls (1-3 minutes) is EXPECTED and VALUABLE.
                    Do NOT penalize the SA for casual conversation, greetings, or relationship building at the beginning.
                    Focus your evaluation on the discovery and sales methodology AFTER the initial rapport phase.

                    Evaluate TWO areas:

                    1. DISCOVERY & ENGAGEMENT:
                       - Discovery question quality (after initial small talk)
                       - Active listening
                       - Engagement with customer
                       - Rapport building (including appropriate small talk)
                       Identify missed opportunities and suggest better questions.

                    2. COMMAND OF THE MESSAGE FRAMEWORK:
                       - Problem Identification: Uncovering business problems
                       - Differentiation: Unique value vs competitors
                       - Proof/Evidence: Case studies, metrics, demos
                       - Required Capabilities: Features tied to business outcomes

                    CRITICAL REQUIREMENT: EXACT TIMESTAMPS
                    For EVERY finding, extract the EXACT timestamp from the transcript where this occurred.
                    Format: "[MM:SS]" or "[HH:MM:SS]" (e.g., "[05:23]", "[24:49]")

                    Return your analysis as JSON:
                    {{
                        "findings": [
                            {{
                                "timestamp": "[12:34]",
                                "category": "Discovery Depth",
                                "observation": "What the SA did well or needs improvement",
                                "recommendation": "Specific actionable recommendation"
                            }}
                        ]
                    }}

                    NOTE: Do NOT include conversation_snippet - we will extract actual quotes programmatically.
                    """,
                    agent=agents['sales_methodology_expert'],
                    expected_output="Sales methodology evaluation with timestamps and Command of Message feedback"
                )

                # Task 4: Compile report (depends on all previous tasks)
                compile_task = Task(
                    description=f"""Compile a comprehensive, actionable performance report for {sa_name}.

                    Based on all the analysis from technical and sales methodology experts,
                    create a final report with:

                    1. Top 3-5 actionable insights with:
                       - Category (which skill area)
                       - Severity (critical/important/minor)
                       - Timestamp (REQUIRED - MUST be included for EVERY insight)
                       - Conversation snippet (1-2 sentence SYNTHESIS of what was discussed at this moment - NOT a verbatim quote)
                       - What happened
                       - Why it matters
                       - Better approach
                       - Example phrasing

                    CRITICAL: Sort all insights in CHRONOLOGICAL ORDER by timestamp (earliest first).
                    This allows readers to follow the call from beginning to end.
                    2. List of strengths (2-3 items)
                    3. List of improvement areas (2-3 items)
                    4. Key moments with timestamps

                    CRITICAL TIMESTAMP REQUIREMENT:
                    EVERY insight in "top_insights" MUST have an EXACT timestamp in [MM:SS] or [HH:MM:SS] format.

                    Extract exact timestamps from the expert analyses (they have already identified specific moments).
                    Examples of CORRECT timestamps: "[05:23]", "[0:16]", "[15:30]", "[~10:00]"
                    Examples of INCORRECT timestamps: "Early in call", "Mid-call", "During demo", null, empty string

                    CONVERSATION SNIPPET REQUIREMENT:
                    For each insight, include a "conversation_snippet" that is a BRIEF 1-2 sentence synthesis
                    of what was discussed at that moment. Do NOT try to quote verbatim - just summarize the exchange.
                    Example: "Customer expressed concerns about integration complexity; SA pivoted to demo without addressing."

                    Make it specific and actionable. Focus on high-impact improvements.
                    Return as valid JSON that matches this structure:
                    {{
                        "call_summary": "2-3 sentence summary",
                        "top_insights": [
                            {{
                                "category": "Discovery Depth",
                                "severity": "critical",
                                "timestamp": "[05:23]",
                                "conversation_snippet": "Brief 1-2 sentence synthesis of the exchange at this moment",
                                "what_happened": "Brief description",
                                "why_it_matters": "Business impact",
                                "better_approach": "What to do differently",
                                "example_phrasing": "Exact words to use"
                            }}
                        ],
                        "strengths": [...],
                        "improvement_areas": [...],
                        "key_moments": [
                            {{
                                "timestamp": "[12:45]",
                                "description": "Key moment description"
                            }}
                        ]
                    }}

                    Sort insights chronologically (earliest timestamp first).
                    """,
                    agent=agents['report_compiler'],
                    expected_output="Complete JSON analysis report",
                    context=[technical_task, sales_methodology_task]
                )

                # Create and run the analysis crew
                with tracer.start_as_current_span("crew_analysis_execution") as crew_span:
                    crew_span.set_attribute("openinference.span.kind", "agent")
                    crew_span.set_attribute("crew.agent_count", 3)
                    crew_span.set_attribute("crew.task_count", 3)
                    crew_span.set_attribute("crew.sa_name", sa_name)

                    # OpenInference input - structured and readable format
                    # Extract first few exchanges for preview
                    all_lines = transcript.split('\n')
                    transcript_lines = all_lines[:10]  # First 10 lines
                    transcript_preview = '\n'.join(transcript_lines)
                    remaining_lines = len(all_lines) - 10
                    if remaining_lines > 0:
                        transcript_preview += f"\n\n... ({remaining_lines} more lines)"

                    crew_span.set_attribute("input.value", json.dumps({
                        "sa_name": sa_name,
                        "transcript_stats": {
                            "total_length": len(transcript),
                            "line_count": len(all_lines),
                            "speaker_count": len(speakers)
                        },
                        "transcript_preview": transcript_preview
                    }, indent=2))
                    crew_span.set_attribute("input.mime_type", "application/json")

                    # Add detailed task information as events
                    crew_span.add_event("task_1_technical_evaluation", {
                        "agent": "Senior Technical Architect & Evaluator",
                        "objective": "Assess technical depth and accuracy"
                    })
                    crew_span.add_event("task_2_sales_methodology", {
                        "agent": "Sales Methodology & Discovery Expert",
                        "objective": "Evaluate discovery and Command of Message"
                    })
                    crew_span.add_event("task_3_report_compilation", {
                        "agent": "Executive Performance Coach & Report Writer",
                        "objective": "Synthesize feedback into actionable report"
                    })

                    crew_span.add_event("creating_analysis_crew")

                    analysis_crew = Crew(
                        agents=[
                            agents['technical_evaluator'],
                            agents['sales_methodology_expert'],
                            agents['report_compiler']
                        ],
                        tasks=[
                            technical_task,
                            sales_methodology_task,
                            compile_task
                        ],
                        process=Process.sequential,
                        verbose=True
                    )

                    # Execute the crew with detailed progress tracking
                    crew_span.add_event("starting_crew_kickoff")
                    crew_span.set_attribute("execution.status", "in_progress")

                    final_report = analysis_crew.kickoff()
                    
                    # Use .raw to get clean output without terminal formatting
                    # str(final_report) includes box-drawing characters that break JSON parsing
                    final_report_text = final_report.raw if hasattr(final_report, 'raw') else str(final_report)
                    
                    # Check if CrewAI already parsed the JSON for us
                    if hasattr(final_report, 'json_dict') and final_report.json_dict:
                        print(f"✅ Using pre-parsed json_dict from CrewOutput")
                        analysis_data_from_crew = final_report.json_dict
                    else:
                        analysis_data_from_crew = None

                    crew_span.add_event("crew_kickoff_completed", {
                        "report.length": len(final_report_text),
                        "tasks_completed": 3
                    })
                    crew_span.set_attribute("execution.status", "completed")

                    # OpenInference output - formatted structured report
                    # Create a summary-first format for better readability
                    try:
                        # Try to parse the report as JSON for structured output
                        json_start = final_report_text.find('{')
                        json_end = final_report_text.rfind('}') + 1
                        if json_start >= 0 and json_end > json_start:
                            parsed_report = json.loads(final_report_text[json_start:json_end])

                            # Create a readable summary format
                            formatted_output = {
                                "summary": {
                                    "call_summary": parsed_report.get("call_summary", "N/A"),
                                    "sa_identified": sa_name,
                                    "insight_count": len(parsed_report.get("top_insights", [])),
                                    "strength_count": len(parsed_report.get("strengths", [])),
                                    "improvement_count": len(parsed_report.get("improvement_areas", []))
                                },
                                "top_insights": [
                                    {
                                        "category": insight.get("category"),
                                        "severity": insight.get("severity"),
                                        "timestamp": insight.get("timestamp"),
                                        "what_happened": insight.get("what_happened", "")[:200] + "..." if len(insight.get("what_happened", "")) > 200 else insight.get("what_happened", ""),
                                        "why_it_matters": insight.get("why_it_matters", "")[:200] + "..." if len(insight.get("why_it_matters", "")) > 200 else insight.get("why_it_matters", ""),
                                        "better_approach": insight.get("better_approach", "")[:200] + "..." if len(insight.get("better_approach", "")) > 200 else insight.get("better_approach", ""),
                                    }
                                    for insight in parsed_report.get("top_insights", [])
                                ],
                                "strengths": parsed_report.get("strengths", []),
                                "improvement_areas": parsed_report.get("improvement_areas", []),
                                "key_moments_count": len(parsed_report.get("key_moments", []))
                            }

                            crew_span.set_attribute("output.value", json.dumps(formatted_output, indent=2))
                        else:
                            # Fallback to raw text if JSON parsing fails
                            crew_span.set_attribute("output.value", final_report_text[:5000] + "..." if len(final_report_text) > 5000 else final_report_text)
                    except (json.JSONDecodeError, Exception):
                        # If parsing fails, use truncated raw text
                        crew_span.set_attribute("output.value", final_report_text[:5000] + "..." if len(final_report_text) > 5000 else final_report_text)

                    crew_span.set_attribute("output.mime_type", "application/json")

                    # Parse JSON from the final report (inside crew_analysis_execution)
                    with tracer.start_as_current_span("parse_crew_report") as parse_span:
                        parse_span.set_attribute("openinference.span.kind", "chain")
                        parse_span.set_attribute("report.raw_length", len(final_report_text))
                        # OpenInference input - raw crew report text
                        parse_span.set_attribute("input.value", final_report_text)
                        parse_span.set_attribute("input.mime_type", "text/plain")

                        try:
                            analysis_data = None
                            
                            # First, check if CrewAI already parsed the JSON for us
                            if analysis_data_from_crew:
                                analysis_data = analysis_data_from_crew
                                parse_span.add_event("using_crew_json_dict")
                                parse_span.set_attribute("parsing.method", "crew_json_dict")
                            
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
                                analysis_data = self._extract_insights_from_raw_text(final_report_text, sa_name)

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
                            sa_identified=sa_name,
                            sa_confidence="high",  # Crew consensus
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

                    # Add metadata about the parsed results
                        parse_span.set_attribute("analysis.insight_count", len(result.top_insights))
                        parse_span.set_attribute("analysis.strength_count", len(result.strengths))
                        parse_span.set_attribute("analysis.improvement_count", len(result.improvement_areas))
                        parse_span.set_attribute("analysis.key_moment_count", len(result.key_moments))
                    # OpenInference output - parsed and structured analysis result
                        parse_span.set_attribute("output.value", json.dumps({
                            "sa_identified": result.sa_identified,
                            "sa_confidence": result.sa_confidence,
                            "call_summary": result.call_summary,
                            "insight_count": len(result.top_insights),
                            "strength_count": len(result.strengths),
                            "improvement_count": len(result.improvement_areas)
                        }))
                        parse_span.set_attribute("output.mime_type", "application/json")
                        parse_span.set_status(Status(StatusCode.OK))

                    # Create separate child spans for each insight (inside crew_span)
                for idx, insight in enumerate(result.top_insights):
                    with tracer.start_as_current_span(
                        f"insight_{idx+1}_{insight.category.lower().replace(' ', '_')}",
                        attributes={
                            "openinference.span.kind": "chain",
                            "insight.index": idx + 1,
                            "insight.category": insight.category,
                            "insight.severity": insight.severity,
                            "insight.timestamp": insight.timestamp or "unknown",
                            # Full insight as output
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

                    # Mark crew_analysis_execution as complete
                    crew_span.set_status(Status(StatusCode.OK))

                # Add result metadata to analysis_span (parent span)
                analysis_span.set_attributes({
                    "output.value": json.dumps({
                        "sa_identified": sa_name,
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

    def _extract_sa_name(self, identification_result: str) -> str:
        """Extract SA name from identification result"""
        print(f"🔍 Extracting SA name from: {identification_result[:200]}...")

        # Try multiple patterns
        patterns = [
            "SA Name:",
            "Solution Architect:",
            "SA is",
            "identified as",
            "The SA is",
            "The Solution Architect is"
        ]

        for pattern in patterns:
            if pattern in identification_result:
                # Extract text after pattern
                parts = identification_result.split(pattern)[1].split(",")[0].split("\n")[0]
                extracted = parts.strip().strip('"').strip("'")
                if extracted and extracted != "Unknown" and len(extracted) > 0:
                    print(f"✅ Extracted SA name using pattern '{pattern}': {extracted}")
                    return extracted

        # Fallback - log warning
        print(f"⚠️  Could not extract SA name from identification result. Using 'Unknown SA'")
        return "Unknown SA"

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
                    why_important=item.get("why_important", "")
                ))
            except Exception:
                pass  # Skip malformed items
        return opportunities

    def _parse_classification(self, classification_text: str) -> Optional[CallClassification]:
        """Parse classification result from the Call Classifier agent"""
        print(f"📊 Parsing classification result ({len(classification_text)} chars)...")

        try:
            # Extract JSON from the result
            json_start = classification_text.find('{')
            json_end = classification_text.rfind('}') + 1

            if json_start >= 0 and json_end > json_start:
                json_str = classification_text[json_start:json_end]
                data = json.loads(json_str)
            else:
                print("⚠️  No JSON found in classification result")
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
                        missed_opportunities=self._parse_missed_opportunities(pain_data.get("missed_opportunities", []))
                    )

                    stakeholder_data = discovery_data.get("stakeholder_map", {})
                    stakeholder = StakeholderMap(
                        technical_champion_identified=stakeholder_data.get("technical_champion_identified", False),
                        technical_champion_engaged=stakeholder_data.get("technical_champion_engaged", False),
                        economic_buyer_identified=stakeholder_data.get("economic_buyer_identified", False),
                        decision_maker_confirmed=stakeholder_data.get("decision_maker_confirmed", False),
                        notes=stakeholder_data.get("notes"),
                        evidence=self._parse_evidence(stakeholder_data.get("evidence", [])),
                        missed_opportunities=self._parse_missed_opportunities(stakeholder_data.get("missed_opportunities", []))
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
                        missed_opportunities=self._parse_missed_opportunities(rc_data.get("missed_opportunities", []))
                    )

                    comp_data = discovery_data.get("competitive_landscape", {})
                    competitive = CompetitiveLandscape(
                        current_tools_evaluated=comp_data.get("current_tools_evaluated", False),
                        tools_mentioned=comp_data.get("tools_mentioned", []),
                        why_looking_vs_staying=comp_data.get("why_looking_vs_staying", False),
                        key_differentiators_identified=comp_data.get("key_differentiators_identified", False),
                        notes=comp_data.get("notes"),
                        evidence=self._parse_evidence(comp_data.get("evidence", [])),
                        missed_opportunities=self._parse_missed_opportunities(comp_data.get("missed_opportunities", []))
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
                        missed_opportunities=self._parse_missed_opportunities(use_case_data.get("missed_opportunities", []))
                    )

                    impl_data = poc_data.get("implementation_requirements", {})
                    impl_req = ImplementationRequirements(
                        data_residency_confirmed=impl_data.get("data_residency_confirmed", False),
                        deployment_model=impl_data.get("deployment_model"),
                        blockers_identified=impl_data.get("blockers_identified", False),
                        blockers_list=impl_data.get("blockers_list", []),
                        notes=impl_data.get("notes"),
                        evidence=self._parse_evidence(impl_data.get("evidence", [])),
                        missed_opportunities=self._parse_missed_opportunities(impl_data.get("missed_opportunities", []))
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
                        missed_opportunities=self._parse_missed_opportunities(metrics_data.get("missed_opportunities", []))
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
                        missed_opportunities=self._parse_missed_opportunities(timeline_data.get("missed_opportunities", []))
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
                        missed_opportunities=self._parse_missed_opportunities(resources_data.get("missed_opportunities", []))
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

            print(f"✅ Classification: {call_type.value} (confidence: {classification.confidence})")
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

    def _extract_insights_from_raw_text(self, raw_text: str, sa_name: str) -> dict:
        """
        Extract meaningful insights from raw crew output when JSON parsing fails.
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
                "what_happened": f"Comprehensive analysis performed for {sa_name}'s call performance.",
                "why_it_matters": "Multi-agent review provides thorough evaluation of technical depth, discovery quality, and sales methodology.",
                "better_approach": "Review the strengths and improvement areas sections below for actionable feedback.",
                "example_phrasing": None
            }]
        
        if not call_summary:
            call_summary = f"Call analysis for {sa_name}. Review insights below for specific feedback."
        
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
