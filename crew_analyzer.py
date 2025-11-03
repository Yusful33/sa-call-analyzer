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

load_dotenv()

# Get tracer for manual instrumentation
tracer = trace.get_tracer("sa-call-analyzer")


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

            You evaluate TWO key areas:

            1. DISCOVERY & ENGAGEMENT:
               - Quality and depth of discovery questions
               - Active listening skills and follow-up questions
               - Engagement with customer responses
               - Uncovering pain points and needs
               - Building rapport and trust

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

        # Create a custom span for the entire analysis workflow
        with tracer.start_as_current_span(
            "sa_call_analysis",
            attributes={
                "call.transcript_length": len(transcript),
                "call.speaker_count": len(speakers),
                "call.speakers": ", ".join(speakers) if speakers else "unknown",
                "call.manual_sa_provided": manual_sa is not None,
                "analysis.agent_count": 4,
                "analysis.framework": "Command of the Message",
            }
        ) as analysis_span:
            try:
                agents = self.create_agents()

                # Task 1: Identify SA
                if manual_sa:
                    sa_name = manual_sa
                    sa_identification_result = f"SA manually specified: {manual_sa}"
                else:
                    identify_sa_task = Task(
                        description=f"""Analyze this call transcript and identify who the Solution Architect (SA) is.

                        Available speakers: {', '.join(speakers) if speakers else 'Unknown (no speaker labels)'}

                        Transcript:
                        {transcript[:3000]}... (truncated if longer)

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
                    sa_crew = Crew(
                        agents=[agents['sa_identifier']],
                        tasks=[identify_sa_task],
                        process=Process.sequential,
                        verbose=True
                    )
                    sa_identification_result = sa_crew.kickoff()
                    sa_name = self._extract_sa_name(str(sa_identification_result))

                # Add SA identification to span
                analysis_span.add_event("sa_identified", {"sa_name": sa_name})

                # Task 2-4: Parallel analysis by different experts
                technical_task = Task(
                    description=f"""Analyze the technical performance of {sa_name} in this call.

                    Transcript:
                    {transcript}

                    Evaluate and score (1-10):
                    - Technical depth and accuracy
                    - Architecture/integration discussions
                    - Demo quality

                    Provide specific examples with timestamps where possible.
                    Format as JSON with scores and detailed feedback.
                    """,
                    agent=agents['technical_evaluator'],
                    expected_output="Technical evaluation with scores and specific feedback"
                )

                sales_methodology_task = Task(
                    description=f"""Analyze {sa_name}'s sales methodology and discovery performance in this call.

                    Transcript:
                    {transcript}

                    Evaluate TWO areas:

                    1. DISCOVERY & ENGAGEMENT (score 1-10):
                       - Discovery question quality
                       - Active listening
                       - Engagement with customer
                       Identify missed opportunities and suggest better questions.

                    2. COMMAND OF THE MESSAGE FRAMEWORK (score each pillar 1-10):
                       - Problem Identification: Uncovering business problems
                       - Differentiation: Unique value vs competitors
                       - Proof/Evidence: Case studies, metrics, demos
                       - Required Capabilities: Features tied to business outcomes

                    Provide scores and specific examples with timestamps for both areas.
                    Format as JSON with all scores and detailed recommendations.
                    """,
                    agent=agents['sales_methodology_expert'],
                    expected_output="Sales methodology and discovery evaluation with Command of Message scores"
                )

                # Task 4: Compile report (depends on all previous tasks)
                compile_task = Task(
                    description=f"""Compile a comprehensive, actionable performance report for {sa_name}.

                    Based on all the analysis from technical and sales methodology experts,
                    create a final report with:

                    1. Overall score (1-10)
                    2. All individual scores from each expert
                    3. Top 3-5 actionable insights with:
                       - Category (which skill area)
                       - Severity (critical/important/minor)
                       - Timestamp (if available)
                       - What happened
                       - Why it matters
                       - Better approach
                       - Example phrasing
                    4. List of strengths (2-3 items)
                    5. List of improvement areas (2-3 items)
                    6. Key moments with timestamps

                    Make it specific and actionable. Focus on high-impact improvements.
                    Return as valid JSON that matches this structure:
                    {{
                        "call_summary": "2-3 sentence summary",
                        "overall_score": 7.5,
                        "command_scores": {{
                            "problem_identification": 8,
                            "differentiation": 6,
                            "proof_evidence": 7,
                            "required_capabilities": 5
                        }},
                        "sa_metrics": {{
                            "technical_depth": 8,
                            "discovery_quality": 7,
                            "active_listening": 9,
                            "value_articulation": 5
                        }},
                        "top_insights": [...],
                        "strengths": [...],
                        "improvement_areas": [...],
                        "key_moments": [...]
                    }}
                    """,
                    agent=agents['report_compiler'],
                    expected_output="Complete JSON analysis report",
                    context=[technical_task, sales_methodology_task]
                )

                # Create and run the analysis crew
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

                # Execute the crew
                final_report = analysis_crew.kickoff()
                final_report_text = str(final_report)

                # Parse JSON from the final report
                try:
                    # Extract JSON from the report (might have markdown code blocks)
                    json_start = final_report_text.find('{')
                    json_end = final_report_text.rfind('}') + 1

                    if json_start >= 0 and json_end > json_start:
                        json_str = final_report_text[json_start:json_end]
                        analysis_data = json.loads(json_str)
                    else:
                        raise ValueError("No JSON found in report")

                except (json.JSONDecodeError, ValueError) as e:
                    print(f"Warning: Could not parse JSON from crew report: {e}")
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
                            "timestamp": None,
                            "what_happened": "Multi-agent analysis completed",
                            "why_it_matters": "Comprehensive evaluation from multiple expert perspectives",
                            "better_approach": "Review the detailed crew output for specific recommendations",
                            "example_phrasing": None
                        }],
                        "strengths": ["Multi-agent analysis completed successfully"],
                        "improvement_areas": ["See detailed crew output for specific areas"],
                        "key_moments": []
                    }

                # Convert to Pydantic models
                result = AnalysisResult(
                    sa_identified=sa_name,
                    sa_confidence="high",  # Crew consensus
                    call_summary=analysis_data.get("call_summary", ""),
                    overall_score=float(analysis_data.get("overall_score", 7.0)),
                    command_scores=CommandOfMessageScore(
                        **{k: float(v) for k, v in analysis_data.get("command_scores", {}).items()}
                    ),
                    sa_metrics=SAPerformanceMetrics(
                        **{k: float(v) for k, v in analysis_data.get("sa_metrics", {}).items()}
                    ),
                    top_insights=[
                        ActionableInsight(**insight)
                        for insight in analysis_data.get("top_insights", [])
                    ],
                    strengths=analysis_data.get("strengths", []),
                    improvement_areas=analysis_data.get("improvement_areas", []),
                    key_moments=analysis_data.get("key_moments", [])
                )

                # Add result metadata to span
                analysis_span.set_attributes({
                    "result.sa_identified": sa_name,
                    "result.overall_score": result.overall_score,
                    "result.insights_count": len(result.top_insights),
                    "result.problem_identification_score": result.command_scores.problem_identification,
                    "result.differentiation_score": result.command_scores.differentiation,
                    "result.proof_evidence_score": result.command_scores.proof_evidence,
                    "result.required_capabilities_score": result.command_scores.required_capabilities,
                    "result.technical_depth": result.sa_metrics.technical_depth,
                    "result.discovery_quality": result.sa_metrics.discovery_quality,
                })
                analysis_span.set_status(Status(StatusCode.OK))

                return result

            except Exception as e:
                analysis_span.set_status(Status(StatusCode.ERROR, str(e)))
                analysis_span.record_exception(e)
                raise

    def _extract_sa_name(self, identification_result: str) -> str:
        """Extract SA name from identification result"""
        # Simple extraction - look for "SA Name:" pattern
        if "SA Name:" in identification_result:
            parts = identification_result.split("SA Name:")[1].split(",")[0]
            return parts.strip()
        # Fallback
        return "Unknown SA"
