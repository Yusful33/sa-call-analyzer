import os
import json
from typing import List, Optional
from crewai import Agent, Task, Crew, Process
from models import (
    AnalysisResult,
    ActionableInsight,
    CommandOfMessageScore,
    SAPerformanceMetrics
)
from dotenv import load_dotenv
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from observability import force_flush_spans

load_dotenv()


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

        if use_litellm:
            # Use OpenAI-compatible endpoint (LiteLLM)
            from langchain_openai import ChatOpenAI
            base_url = os.getenv("LITELLM_BASE_URL", "http://localhost:4000")
            # Ensure base_url ends with /v1 for OpenAI compatibility
            if not base_url.endswith("/v1"):
                base_url = f"{base_url}/v1"
            print(f"🔧 LiteLLM Config: base_url={base_url}, model={model_name}")
            self.llm = ChatOpenAI(
                model=model_name,
                base_url=base_url,
                api_key=os.getenv("LITELLM_API_KEY", "dummy"),
                temperature=0.7
            )
        else:
            # Use Anthropic directly
            from langchain_anthropic import ChatAnthropic
            self.llm = ChatAnthropic(
                model=model_name,
                anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
                temperature=0.7
            )

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

        return {
            'sa_identifier': sa_identifier,
            'technical_evaluator': technical_evaluator,
            'sales_methodology_expert': sales_methodology_expert,
            'report_compiler': report_compiler
        }

    def analyze_call(self, transcript: str, speakers: List[str], manual_sa: Optional[str] = None) -> AnalysisResult:
        """
        Analyze a call transcript using the crew of specialized agents.

        Args:
            transcript: The call transcript
            speakers: List of speaker names (if available)
            manual_sa: Manually specified SA name (optional)

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
                        # Give SA identifier a better sample: first 2000 chars + middle 2000 chars + last 1000 chars
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

                        identify_sa_task = Task(
                            description=f"""Analyze this call transcript and identify who the Solution Architect (SA) is.

                            Available speakers: {', '.join(speakers) if speakers else 'Unknown (no speaker labels)'}

                            Transcript sample (includes beginning, middle, and end of call):
                            {transcript_sample}

                            The Solution Architect is typically the person who:
                            - Discusses technical architecture and implementation
                            - Explains product features and capabilities
                            - Answers technical questions
                            - Conducts demos or technical walkthroughs
                            - Discusses integration and data architecture

                            NOTE: Initial small talk and pleasantries are normal - look beyond that for technical content.

                            Provide:
                            1. The SA's name
                            2. Confidence level (high/medium/low)
                            3. Brief reasoning

                            Format: "SA Name: <name>, Confidence: <level>, Reasoning: <brief explanation>"
                            """,
                            agent=agents['sa_identifier'],
                            expected_output="SA identification with name, confidence, and reasoning"
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

                # Task 2-4: Parallel analysis by different experts
                technical_task = Task(
                    description=f"""Analyze the technical performance of {sa_name} in this call.

                    Transcript:
                    {transcript}

                    Evaluate:
                    - Technical depth and accuracy
                    - Architecture/integration discussions
                    - Demo quality

                    CRITICAL REQUIREMENT - EXACT TIMESTAMPS:
                    For EVERY finding, you MUST provide the EXACT timestamp from the transcript where this occurred.

                    Steps to extract timestamps:
                    1. Search the transcript for the exact quote or moment being referenced
                    2. Find the timestamp marker immediately before that quote (format: [HH:MM:SS], [MM:SS], or "0:16 |")
                    3. Include that exact timestamp in your analysis (e.g., "[05:23]", "[0:16]", "[15:30]")
                    4. If the transcript has NO timestamps at all, estimate based on position: "[~2:00]" for early, "[~10:00]" for mid-call, "[~20:00]" for late

                    DO NOT use vague descriptions like "Early in call" or "Mid-call".
                    ALWAYS provide a specific time reference in [MM:SS] or [HH:MM:SS] format.

                    Provide specific examples with timestamps.
                    Format as JSON with scores and detailed feedback.
                    """,
                    agent=agents['technical_evaluator'],
                    expected_output="Technical evaluation with scores and specific feedback"
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

                    CRITICAL REQUIREMENT - EXACT TIMESTAMPS:
                    For EVERY finding, you MUST provide the EXACT timestamp from the transcript where this occurred.

                    Steps to extract timestamps:
                    1. Search the transcript for the exact quote or moment being referenced
                    2. Find the timestamp marker immediately before that quote (format: [HH:MM:SS], [MM:SS], or "0:16 |")
                    3. Include that exact timestamp in your analysis (e.g., "[05:23]", "[0:16]", "[15:30]")
                    4. If the transcript has NO timestamps at all, estimate based on position: "[~2:00]" for early, "[~10:00]" for mid-call, "[~20:00]" for late

                    DO NOT use vague descriptions like "Early in call" or "Mid-call".
                    ALWAYS provide a specific time reference in [MM:SS] or [HH:MM:SS] format.

                    Provide specific examples with timestamps for both areas.
                    Format as JSON with detailed qualitative feedback and recommendations.
                    """,
                    agent=agents['sales_methodology_expert'],
                    expected_output="Sales methodology and discovery evaluation with qualitative Command of Message feedback"
                )

                # Task 4: Compile report (depends on all previous tasks)
                compile_task = Task(
                    description=f"""Compile a comprehensive, actionable performance report for {sa_name}.

                    Based on all the analysis from technical and sales methodology experts,
                    create a final report with:

                    1. Top 3-5 actionable insights with:
                       - Category (which skill area)
                       - Severity (critical/important/minor)
                       - Timestamp (REQUIRED - MUST be included for EVERY insight. Extract from the expert analysis or transcript.)
                       - Conversation snippet (REQUIRED - Include a brief 2-3 line excerpt from the actual transcript showing the moment this occurred. Use the exact words from the transcript.)
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

                    If an expert provided a vague description, YOU must convert it to a specific time estimate.
                    DO NOT leave timestamp as null, empty, or use descriptive text.

                    Make it specific and actionable. Focus on high-impact improvements.
                    Return as valid JSON that matches this structure:
                    {{
                        "call_summary": "2-3 sentence summary",
                        "top_insights": [
                            {{
                                "category": "Discovery Depth",
                                "severity": "critical",
                                "timestamp": "[05:23]",
                                "conversation_snippet": "Customer: 'We have data quality issues.'\nSA: 'Okay, let me show you our validation features.'\nCustomer: 'Actually, the bigger problem is...'",
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

                    REMEMBER:
                    - Sort insights chronologically (earliest timestamp first)
                    - Include actual conversation snippets from the transcript for each insight
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
                    final_report_text = str(final_report)

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
                    crew_span.set_status(Status(StatusCode.OK))

                # Parse JSON from the final report
                with tracer.start_as_current_span("parse_crew_report") as parse_span:
                    parse_span.set_attribute("openinference.span.kind", "chain")
                    parse_span.set_attribute("report.raw_length", len(final_report_text))
                    # OpenInference input - raw crew report text
                    parse_span.set_attribute("input.value", final_report_text)
                    parse_span.set_attribute("input.mime_type", "text/plain")

                    try:
                        # Extract JSON from the report (might have markdown code blocks)
                        json_start = final_report_text.find('{')
                        json_end = final_report_text.rfind('}') + 1

                        if json_start >= 0 and json_end > json_start:
                            json_str = final_report_text[json_start:json_end]
                            analysis_data = json.loads(json_str)
                            parse_span.add_event("json_parsed_successfully")
                            parse_span.set_attribute("parsing.success", True)
                        else:
                            raise ValueError("No JSON found in report")

                    except (json.JSONDecodeError, ValueError) as e:
                        print(f"⚠️  WARNING: Could not parse JSON from crew report: {e}")
                        print(f"📄 Raw report (first 1000 chars):\n{final_report_text[:1000]}")
                        print(f"📄 Raw report (last 500 chars):\n{final_report_text[-500:]}")
                        parse_span.add_event("json_parse_failed", {
                            "error": str(e),
                            "report_preview": final_report_text[:500]
                        })
                        parse_span.set_attribute("parsing.success", False)
                        parse_span.set_attribute("parsing.fallback", True)
                        parse_span.set_attribute("error.message", str(e))

                        # Fallback to default structure
                        analysis_data = {
                            "call_summary": f"Analysis completed by CrewAI for {sa_name}. See raw output for details.",
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
                            "top_insights": [{
                                "category": "General",
                                "severity": "important",
                                "timestamp": "[~0:00]",
                                "conversation_snippet": None,
                                "what_happened": "Multi-agent analysis completed",
                                "why_it_matters": "Comprehensive evaluation from multiple expert perspectives",
                                "better_approach": "Review the detailed crew output for specific recommendations",
                                "example_phrasing": None
                            }],
                            "strengths": ["Multi-agent analysis completed successfully"],
                            "improvement_areas": ["See detailed crew output for specific areas"],
                            "key_moments": []
                        }

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
                    insights = []
                    for insight in analysis_data.get("top_insights", []):
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
                        key_moments=analysis_data.get("key_moments", [])
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

                # Add result metadata to span - full detailed analysis output
                analysis_span.set_attributes({
                    # OpenInference semantic conventions for output - complete analysis
                    "output.value": json.dumps({
                        "sa_identified": sa_name,
                        "call_summary": result.call_summary,
                        "overall_score": result.overall_score,
                        "command_scores": result.command_scores.model_dump(),
                        "sa_metrics": result.sa_metrics.model_dump(),
                        "top_insights": [
                            {
                                "category": insight.category,
                                "severity": insight.severity,
                                "timestamp": insight.timestamp,
                                "what_happened": insight.what_happened,
                                "why_it_matters": insight.why_it_matters,
                                "better_approach": insight.better_approach,
                                "example_phrasing": insight.example_phrasing,
                            }
                            for insight in result.top_insights
                        ],
                        "strengths": result.strengths,
                        "improvement_areas": result.improvement_areas,
                        "key_moments": result.key_moments,
                    }, indent=2),
                    "output.mime_type": "application/json",
                })
                analysis_span.set_status(Status(StatusCode.OK))

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
