from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Literal
from enum import Enum


class CallType(str, Enum):
    """Type of sales call"""
    DISCOVERY = "discovery"
    POC_SCOPING = "poc_scoping"
    MIXED = "mixed"  # Contains elements of both
    UNCLEAR = "unclear"  # Cannot determine


class TranscriptLine(BaseModel):
    """A single line from the transcript"""
    timestamp: Optional[str] = None
    speaker: Optional[str] = None
    text: str


class AnalyzeRequest(BaseModel):
    """Request to analyze a transcript"""
    transcript: Optional[str] = None  # Manual transcript text
    gong_url: Optional[str] = None  # Gong call URL (alternative to transcript)
    sa_name: Optional[str] = None  # Manual override for SA identification

    def model_post_init(self, __context):
        """Validate that either transcript or gong_url is provided."""
        if not self.transcript and not self.gong_url:
            raise ValueError("Either 'transcript' or 'gong_url' must be provided")
        if self.transcript and self.gong_url:
            raise ValueError("Provide either 'transcript' or 'gong_url', not both")


# ============================================================================
# EVIDENCE & OPPORTUNITY TRACKING
# ============================================================================

class CriteriaEvidence(BaseModel):
    """Evidence captured for a specific criteria item"""
    criteria_name: str  # e.g., "debugging_process_documented"
    captured: bool = False  # Whether this criteria was captured
    timestamp: Optional[str] = None  # When in the call this was discussed
    conversation_snippet: Optional[str] = None  # Brief excerpt showing evidence
    speaker: Optional[str] = None  # Who provided this information


class MissedOpportunity(BaseModel):
    """A missed opportunity to gather information"""
    criteria_name: str  # Which criteria this relates to
    timestamp: Optional[str] = None  # When the opportunity occurred
    context: str  # What was happening when the opportunity arose
    suggested_question: str  # What question could have been asked
    why_important: str  # Why this information matters


# ============================================================================
# DISCOVERY CALL CRITERIA
# ============================================================================

class PainCurrentState(BaseModel):
    """1. Pain & Current State Validated"""
    # Primary Use Case Focus
    primary_use_case: Optional[str] = None  # "development", "production", or "both"
    
    # Development Focus
    prompt_model_iteration_understood: bool = False  # How they iterate on prompts/models today
    
    # Production Focus
    debugging_process_documented: bool = False  # How they debug LLM/agent issues today
    
    # Common criteria for both
    situation_understood: bool = False  # What is the situation?
    resolution_attempts_documented: bool = False  # What have you done to resolve?
    outcomes_documented: bool = False  # What outcome did those actions have?
    frequency_quantified: bool = False  # How often
    duration_quantified: bool = False  # How long
    impact_quantified: bool = False  # How much (impact)
    people_impact_understood: bool = False  # People affected
    process_impact_understood: bool = False  # Process affected
    technology_impact_understood: bool = False  # Technology affected
    
    # Metrics
    mttd_mttr_quantified: bool = False  # Mean Time to Detection/Remediation for LLM failures
    experiment_time_quantified: bool = False  # Average time to experiment with new prompts/models
    
    notes: Optional[str] = None
    evidence: List[CriteriaEvidence] = Field(default_factory=list)
    missed_opportunities: List[MissedOpportunity] = Field(default_factory=list)

    @property
    def completion_score(self) -> float:
        """Calculate completion percentage (0-100)"""
        fields = [
            self.prompt_model_iteration_understood, self.debugging_process_documented,
            self.situation_understood, self.resolution_attempts_documented,
            self.outcomes_documented, self.frequency_quantified, self.duration_quantified,
            self.impact_quantified, self.people_impact_understood,
            self.process_impact_understood, self.technology_impact_understood,
            self.mttd_mttr_quantified, self.experiment_time_quantified
        ]
        return (sum(fields) / len(fields)) * 100


class StakeholderMap(BaseModel):
    """2. Stakeholder Map Complete"""
    technical_champion_identified: bool = False
    technical_champion_engaged: bool = False
    economic_buyer_identified: bool = False
    decision_maker_confirmed: bool = False
    notes: Optional[str] = None
    evidence: List[CriteriaEvidence] = Field(default_factory=list)
    missed_opportunities: List[MissedOpportunity] = Field(default_factory=list)

    @property
    def completion_score(self) -> float:
        fields = [
            self.technical_champion_identified, self.technical_champion_engaged,
            self.economic_buyer_identified, self.decision_maker_confirmed
        ]
        return (sum(fields) / len(fields)) * 100


class RequiredCapabilities(BaseModel):
    """3. Required Capabilities (RCs) Prioritized"""
    top_rcs_ranked: bool = False  # Top 2-3 RCs ranked by prospect
    
    # Core Capabilities - which are important to the prospect?
    llm_agent_tracing_important: Optional[bool] = None
    llm_evaluations_important: Optional[bool] = None
    production_monitoring_important: Optional[bool] = None
    prompt_management_important: Optional[bool] = None  # NEW
    prompt_experimentation_important: Optional[bool] = None  # NEW
    monitoring_important: Optional[bool] = None
    compliance_important: Optional[bool] = None  # SOC2, SSO, GDPR, etc.
    
    must_have_vs_nice_to_have_distinguished: bool = False
    deal_breakers_identified: bool = False
    notes: Optional[str] = None
    evidence: List[CriteriaEvidence] = Field(default_factory=list)
    missed_opportunities: List[MissedOpportunity] = Field(default_factory=list)

    @property
    def completion_score(self) -> float:
        fields = [
            self.top_rcs_ranked, self.must_have_vs_nice_to_have_distinguished,
            self.deal_breakers_identified
        ]
        return (sum(fields) / len(fields)) * 100


class CompetitiveLandscape(BaseModel):
    """4. Competitive Landscape Understood"""
    current_tools_evaluated: bool = False  # LangSmith, W&B, Braintrust, Datadog LLM
    tools_mentioned: List[str] = Field(default_factory=list)
    why_looking_vs_staying: bool = False  # Why they're looking vs. staying
    key_differentiators_identified: bool = False  # Differentiators that matter to prospect
    notes: Optional[str] = None
    evidence: List[CriteriaEvidence] = Field(default_factory=list)
    missed_opportunities: List[MissedOpportunity] = Field(default_factory=list)

    @property
    def completion_score(self) -> float:
        fields = [
            self.current_tools_evaluated, self.why_looking_vs_staying,
            self.key_differentiators_identified
        ]
        return (sum(fields) / len(fields)) * 100


class DiscoveryCriteria(BaseModel):
    """Complete Discovery Call Criteria"""
    pain_current_state: PainCurrentState = Field(default_factory=PainCurrentState)
    stakeholder_map: StakeholderMap = Field(default_factory=StakeholderMap)
    required_capabilities: RequiredCapabilities = Field(default_factory=RequiredCapabilities)
    competitive_landscape: CompetitiveLandscape = Field(default_factory=CompetitiveLandscape)

    @property
    def overall_completion_score(self) -> float:
        return (
            self.pain_current_state.completion_score +
            self.stakeholder_map.completion_score +
            self.required_capabilities.completion_score +
            self.competitive_landscape.completion_score
        ) / 4


# ============================================================================
# POC SCOPING CALL CRITERIA
# ============================================================================

class UseCaseScoped(BaseModel):
    """1. Use Case Scoped"""
    llm_applications_selected: bool = False  # Specific LLM application(s) for PoC (not to exceed one)
    applications_list: List[str] = Field(default_factory=list)
    environment_decided: bool = False  # Production vs. staging
    environment_type: Optional[str] = None  # "production", "staging", "both"
    trace_volume_estimated: bool = False  # Expected trace volume
    estimated_volume: Optional[str] = None  # e.g., "10K traces/day"
    llm_provider_identified: bool = False  # LLM Provider for gateway implementation
    llm_provider: Optional[str] = None  # e.g., "OpenAI", "Anthropic", "Azure OpenAI"
    integration_complexity_assessed: bool = False  # # of services, frameworks
    notes: Optional[str] = None
    evidence: List[CriteriaEvidence] = Field(default_factory=list)
    missed_opportunities: List[MissedOpportunity] = Field(default_factory=list)

    @property
    def completion_score(self) -> float:
        fields = [
            self.llm_applications_selected, self.environment_decided,
            self.trace_volume_estimated, self.llm_provider_identified,
            self.integration_complexity_assessed
        ]
        return (sum(fields) / len(fields)) * 100


class ImplementationRequirements(BaseModel):
    """2. Implementation Requirements Validated"""
    data_residency_confirmed: bool = False  # Cloud, VPC, On-prem
    deployment_model: Optional[str] = None  # "cloud", "vpc", "on-prem"
    blockers_identified: bool = False  # Firewall, procurement, security review
    blockers_list: List[str] = Field(default_factory=list)
    notes: Optional[str] = None
    evidence: List[CriteriaEvidence] = Field(default_factory=list)
    missed_opportunities: List[MissedOpportunity] = Field(default_factory=list)

    @property
    def completion_score(self) -> float:
        fields = [self.data_residency_confirmed, self.blockers_identified]
        return (sum(fields) / len(fields)) * 100


class MetricsSuccessCriteria(BaseModel):
    """3. Metrics & Success Criteria Defined"""
    specific_metrics_defined: bool = False  # Prospect-specific "X" values
    example_metrics: List[str] = Field(default_factory=list)  # e.g., "reduce debugging from 4hr to 30min"
    baseline_captured: bool = False  # Before/after comparison baseline
    success_measurement_agreed: bool = False  # Agreement on how to measure success
    competitive_favorable_criteria: bool = False  # Success criteria favorable vs competitors (Galileo, Braintrust, LangSmith)
    notes: Optional[str] = None
    evidence: List[CriteriaEvidence] = Field(default_factory=list)
    missed_opportunities: List[MissedOpportunity] = Field(default_factory=list)

    @property
    def completion_score(self) -> float:
        fields = [
            self.specific_metrics_defined, self.baseline_captured,
            self.success_measurement_agreed, self.competitive_favorable_criteria
        ]
        return (sum(fields) / len(fields)) * 100


class TimelineMilestones(BaseModel):
    """4. Timeline & Milestones Agreed"""
    poc_duration_defined: bool = False  # Typically 2-4 weeks
    duration_weeks: Optional[int] = None
    key_milestones_with_dates: bool = False  # Kickoff, integration, eval review, decision
    milestones: List[str] = Field(default_factory=list)
    decision_date_committed: bool = False
    decision_date: Optional[str] = None
    next_steps_discussed: bool = False  # What happens after successful PoC
    notes: Optional[str] = None
    evidence: List[CriteriaEvidence] = Field(default_factory=list)
    missed_opportunities: List[MissedOpportunity] = Field(default_factory=list)

    @property
    def completion_score(self) -> float:
        fields = [
            self.poc_duration_defined, self.key_milestones_with_dates,
            self.decision_date_committed, self.next_steps_discussed
        ]
        return (sum(fields) / len(fields)) * 100


class ResourcesCommitted(BaseModel):
    """5. Resources Committed"""
    engineering_resources_allocated: bool = False  # Names, % time
    resource_names: List[str] = Field(default_factory=list)
    checkin_cadence_established: bool = False  # Weekly check-ins
    cadence: Optional[str] = None  # e.g., "weekly", "bi-weekly"
    communication_channel_created: bool = False  # Slack channel created
    notes: Optional[str] = None
    evidence: List[CriteriaEvidence] = Field(default_factory=list)
    missed_opportunities: List[MissedOpportunity] = Field(default_factory=list)

    @property
    def completion_score(self) -> float:
        fields = [
            self.engineering_resources_allocated, self.checkin_cadence_established,
            self.communication_channel_created
        ]
        return (sum(fields) / len(fields)) * 100


class PocScopingCriteria(BaseModel):
    """Complete PoC Scoping Call Criteria"""
    use_case_scoped: UseCaseScoped = Field(default_factory=UseCaseScoped)
    implementation_requirements: ImplementationRequirements = Field(default_factory=ImplementationRequirements)
    metrics_success_criteria: MetricsSuccessCriteria = Field(default_factory=MetricsSuccessCriteria)
    timeline_milestones: TimelineMilestones = Field(default_factory=TimelineMilestones)
    resources_committed: ResourcesCommitted = Field(default_factory=ResourcesCommitted)

    @property
    def overall_completion_score(self) -> float:
        return (
            self.use_case_scoped.completion_score +
            self.implementation_requirements.completion_score +
            self.metrics_success_criteria.completion_score +
            self.timeline_milestones.completion_score +
            self.resources_committed.completion_score
        ) / 5


# ============================================================================
# CALL CLASSIFICATION RESULT
# ============================================================================

class MissingElements(BaseModel):
    """Missing elements segmented by call type"""
    discovery: List[str] = Field(default_factory=list)  # Missing discovery criteria
    poc_scoping: List[str] = Field(default_factory=list)  # Missing PoC scoping criteria


class CallClassification(BaseModel):
    """Complete call classification with criteria assessment"""
    call_type: CallType
    confidence: str  # "high", "medium", "low"
    reasoning: str  # Why this classification was made
    
    # Discovery criteria (populated if call has discovery elements)
    discovery_criteria: Optional[DiscoveryCriteria] = None
    discovery_completion_score: float = 0.0
    
    # PoC Scoping criteria (populated if call has PoC scoping elements)
    poc_scoping_criteria: Optional[PocScopingCriteria] = None
    poc_scoping_completion_score: float = 0.0
    
    # Key gaps segmented by call type
    missing_elements: MissingElements = Field(default_factory=MissingElements)
    # Recommendations for next steps
    recommendations: List[str] = Field(default_factory=list)


class ActionableInsight(BaseModel):
    """A single actionable insight with specific recommendations"""
    category: str  # e.g., "Problem Identification", "Differentiation", etc.
    severity: str  # "critical", "important", "minor"
    timestamp: Optional[str] = None
    conversation_snippet: Optional[str] = None  # Brief excerpt from the actual conversation
    what_happened: str  # What the SA did/said
    why_it_matters: str  # Business impact
    better_approach: str  # Specific alternative
    example_phrasing: Optional[str] = None  # Exact words they could use


class CommandOfMessageScore(BaseModel):
    """Scoring for Command of the Message framework (optional fields)"""
    problem_identification: Optional[float] = 7.0  # 1-10 (can be decimal like 6.5)
    differentiation: Optional[float] = 7.0  # 1-10 (can be decimal like 6.5)
    proof_evidence: Optional[float] = 7.0  # 1-10 (can be decimal like 6.5)
    required_capabilities: Optional[float] = 7.0  # 1-10 (can be decimal like 6.5)


class SAPerformanceMetrics(BaseModel):
    """Additional SA-specific metrics (optional fields)"""
    technical_depth: Optional[float] = 7.0  # 1-10 (can be decimal like 6.5)
    discovery_quality: Optional[float] = 7.0  # 1-10 (can be decimal like 6.5)
    active_listening: Optional[float] = 7.0  # 1-10 (can be decimal like 6.5)
    value_articulation: Optional[float] = 7.0  # 1-10 (can be decimal like 6.5)


class RecapSlideData(BaseModel):
    """Data for generating a recap slide for the next call"""
    customer_name: str = ""
    call_date: str = ""
    current_state: List[str] = Field(default_factory=list)  # Current customer situation/pain points
    future_state: List[str] = Field(default_factory=list)  # Desired outcome after implementation
    negative_consequences: List[str] = Field(default_factory=list)  # What happens if they don't act
    positive_business_outcomes: List[str] = Field(default_factory=list)  # Benefits of moving forward
    required_capabilities: List[str] = Field(default_factory=list)  # What they need from a solution


class AnalysisResult(BaseModel):
    """Complete analysis results"""
    sa_identified: str  # Who we identified as the SA
    sa_confidence: str  # "high", "medium", "low"
    call_summary: str
    overall_score: Optional[float] = 7.0  # 1-10 (optional, defaults to 7.0)
    command_scores: CommandOfMessageScore
    sa_metrics: SAPerformanceMetrics
    top_insights: List[ActionableInsight]
    strengths: List[str]
    improvement_areas: List[str]
    key_moments: List[Dict[str, str]]  # timestamp, description
    
    # Call classification (new)
    call_classification: Optional[CallClassification] = None
    
    # Recap slide data for next call
    recap_data: Optional[RecapSlideData] = None
