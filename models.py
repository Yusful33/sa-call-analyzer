from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Literal, Any
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
    model: Optional[str] = None  # LLM model to use (e.g., claude-haiku-4-5, gpt-4o-mini)

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
    collected_later: Optional[bool] = False  # True if this info was collected later in the call (should be excluded from missed_opportunities)


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
    key_initiatives: List[str] = Field(default_factory=list)  # Customer's key goals/initiatives
    challenges: List[str] = Field(default_factory=list)  # Pain points and challenges discussed
    solution_requirements: List[str] = Field(default_factory=list)  # What they need from a solution
    follow_up_questions: List[str] = Field(default_factory=list)  # Probing questions for next call


class AnalysisResult(BaseModel):
    """Complete analysis results"""
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


# ============================================================================
# PROSPECT TIMELINE MODELS
# ============================================================================

class AggregatedActionableInsight(BaseModel):
    """An actionable insight aggregated across multiple calls"""
    category: str  # e.g., "Problem Identification", "Differentiation", etc.
    severity: str  # "critical", "important", "minor"
    call_date: Optional[str] = None  # Which call this came from
    call_title: Optional[str] = None  # Title of the call
    timestamp: Optional[str] = None
    conversation_snippet: Optional[str] = None
    what_happened: str
    why_it_matters: str
    better_approach: str
    example_phrasing: Optional[str] = None


class AccountAnalysisSummary(BaseModel):
    """Aggregated analysis summary across all calls for an account - similar to AnalysisResult but for multiple calls"""

    # Summary
    account_summary: str = ""  # 2-3 paragraph overview of engagement across all calls
    total_calls_analyzed: int = 0
    date_range: str = ""  # e.g., "December 15, 2025 - December 23, 2025"

    # Aggregated Scores (averages across all calls)
    average_overall_score: Optional[float] = None  # 1-10 average
    command_scores: CommandOfMessageScore = Field(default_factory=CommandOfMessageScore)  # Averaged
    sa_metrics: SAPerformanceMetrics = Field(default_factory=SAPerformanceMetrics)  # Averaged

    # Aggregated Insights (top insights across all calls, with call context)
    top_insights: List[AggregatedActionableInsight] = Field(default_factory=list)  # Top 10 across all calls

    # Aggregated Strengths & Improvements (unique across all calls)
    strengths: List[str] = Field(default_factory=list)
    improvement_areas: List[str] = Field(default_factory=list)

    # Key Moments across all calls
    key_moments: List[Dict[str, str]] = Field(default_factory=list)  # With call_date added

    # Discovery criteria completion (aggregated - highest values across calls)
    discovery_completion: Dict[str, float] = Field(default_factory=dict)  # Section -> score

    # PoC Scoping criteria completion (aggregated - highest values across calls)
    poc_scoping_completion: Dict[str, float] = Field(default_factory=dict)  # Section -> score

    # All missed opportunities across calls (for coaching)
    all_missed_opportunities: List[Dict[str, Any]] = Field(default_factory=list)

    # Aggregated recap data for next call
    recap_data: Optional[RecapSlideData] = None


class QuickCallSummary(BaseModel):
    """Fast call summary from single LLM call (vs full multi-agent analysis)"""
    call_type: str = "other"  # discovery, poc_scoping, check_in, demo, follow_up, other
    one_liner: str = ""  # Single sentence summary
    key_points: List[str] = Field(default_factory=list)  # 3-5 bullet points
    participants_mentioned: List[str] = Field(default_factory=list)  # Key people mentioned
    sentiment: str = "neutral"  # positive, neutral, negative, mixed


class CallTimelineEntry(BaseModel):
    """A single call entry in the prospect timeline"""
    call_id: str
    call_date: Optional[str] = None  # Formatted date string
    call_title: Optional[str] = None
    call_type: Optional[str] = None  # discovery, poc_scoping, poc_sync, etc.
    call_url: Optional[str] = None
    analysis: Optional[AnalysisResult] = None  # Full analysis result (slow mode)
    quick_summary: Optional[QuickCallSummary] = None  # Fast summary (fast mode)
    key_insights: List[str] = Field(default_factory=list)  # Extracted key insights
    progression_indicators: Dict[str, str] = Field(default_factory=dict)  # What's new vs carried forward


class ProspectTimeline(BaseModel):
    """Complete timeline of calls for a prospect"""
    prospect_name: str  # Name as searched
    matched_participant_names: List[str] = Field(default_factory=list)  # Actual names found in Gong
    calls: List[CallTimelineEntry] = Field(default_factory=list)  # Calls in chronological order
    cumulative_insights: Dict[str, Any] = Field(default_factory=dict)  # Cumulative insights across all calls
    progression_summary: str = ""  # Summary of progression across calls
    overall_account_health: Optional[str] = None  # Overall assessment
    key_themes: List[str] = Field(default_factory=list)  # Key themes across all calls
    next_steps: List[str] = Field(default_factory=list)  # Recommended next steps

    # NEW: Aggregated analysis summary across all calls (similar to single-call AnalysisResult)
    account_analysis: Optional[AccountAnalysisSummary] = None


# ============================================================================
# PROSPECT OVERVIEW MODELS (BigQuery Integration)
# ============================================================================

class SalesforceAccountData(BaseModel):
    """Salesforce account data from BigQuery"""
    id: str
    name: str
    website: Optional[str] = None
    industry: Optional[str] = None
    annual_revenue: Optional[float] = None
    number_of_employees: Optional[int] = None
    total_active_arr: Optional[float] = None
    lifecycle_stage: Optional[str] = None
    product_tier: Optional[str] = None
    status: Optional[str] = None
    customer_success_tier: Optional[str] = None
    # Team assignments
    assigned_sa: Optional[str] = None
    assigned_cse: Optional[str] = None
    assigned_csm: Optional[str] = None
    assigned_ai_se: Optional[str] = None
    owner_name: Optional[str] = None  # Account owner
    # Pendo linkage
    pendo_account_id: Optional[str] = None
    pendo_time_on_site: Optional[float] = None
    # Product usage
    num_models: Optional[float] = None
    is_using_llms: Optional[str] = None
    deployment_types: Optional[str] = None
    # Dates
    created_date: Optional[str] = None
    last_activity_date: Optional[str] = None
    # Notes
    next_steps: Optional[str] = None
    customer_notes: Optional[str] = None
    description: Optional[str] = None


class OpportunityData(BaseModel):
    """Salesforce opportunity data from BigQuery"""
    id: str
    name: str
    stage_name: str
    amount: Optional[float] = None
    close_date: Optional[str] = None
    probability: Optional[float] = None
    is_closed: bool = False
    is_won: bool = False
    forecast_category: Optional[str] = None
    created_date: Optional[str] = None
    last_modified_date: Optional[str] = None
    age_in_days: Optional[int] = None
    next_step: Optional[str] = None
    description: Optional[str] = None
    lead_source: Optional[str] = None
    type: Optional[str] = None
    owner_name: Optional[str] = None


class SalesforceTaskData(BaseModel):
    """Salesforce task/activity data from BigQuery"""
    id: str
    subject: Optional[str] = None
    type: Optional[str] = None  # Call, Email, Meeting, etc.
    status: str
    activity_date: Optional[str] = None
    description: Optional[str] = None
    owner_name: Optional[str] = None
    is_closed: bool = False
    created_date: Optional[str] = None


class GongParticipant(BaseModel):
    """Gong call participant"""
    name: Optional[str] = None
    email: Optional[str] = None
    affiliation: Optional[str] = None  # "internal" or "external"
    speaker_id: Optional[str] = None


class GongCallData(BaseModel):
    """Individual Gong call data from BigQuery"""
    conversation_key: Optional[str] = None
    call_url: Optional[str] = None
    call_title: Optional[str] = None
    call_date: Optional[str] = None
    duration_minutes: Optional[float] = None
    # Call Spotlight AI summary
    spotlight_brief: Optional[str] = None
    spotlight_key_points: Optional[List[str]] = None
    spotlight_next_steps: Optional[str] = None
    spotlight_outcome: Optional[str] = None
    spotlight_type: Optional[str] = None
    # Interaction metrics
    talk_ratio: Optional[float] = None
    interactivity: Optional[float] = None
    patience: Optional[float] = None
    question_rate: Optional[float] = None
    longest_monologue: Optional[float] = None
    longest_customer_story: Optional[float] = None
    # Participants
    participants: List[GongParticipant] = Field(default_factory=list)
    # Transcript snippets
    transcript_snippet: Optional[str] = None  # First few exchanges


class GongSummaryData(BaseModel):
    """Aggregated Gong analytics from BigQuery"""
    total_calls: int = 0
    total_duration_minutes: float = 0
    avg_talk_ratio: Optional[float] = None
    avg_interactivity: Optional[float] = None
    avg_patience: Optional[float] = None
    avg_question_rate: Optional[float] = None
    # Engagement timeline
    first_call_date: Optional[str] = None
    last_call_date: Optional[str] = None
    days_since_last_call: Optional[int] = None
    # Key themes from spotlight briefs
    key_themes: List[str] = Field(default_factory=list)
    # Recent calls with full details
    recent_calls: List[GongCallData] = Field(default_factory=list)


class PendoVisitorActivity(BaseModel):
    """Individual Pendo visitor activity"""
    visitor_id: str
    email: Optional[str] = None
    display_name: Optional[str] = None
    total_events: int = 0
    total_minutes: int = 0
    last_visit: Optional[str] = None
    first_visit: Optional[str] = None
    visit_count: int = 0


class PendoFeatureUsage(BaseModel):
    """Feature usage data from Pendo"""
    feature_id: str
    feature_name: Optional[str] = None
    event_count: int = 0
    unique_users: int = 0
    last_used: Optional[str] = None


class PendoPageUsage(BaseModel):
    """Page usage data from Pendo"""
    page_id: str
    page_name: Optional[str] = None
    view_count: int = 0
    unique_viewers: int = 0
    total_minutes: int = 0


class PendoUsageData(BaseModel):
    """Enhanced Pendo product usage data from BigQuery"""
    # Overall metrics
    total_events: int = 0
    total_minutes: int = 0
    unique_visitors: int = 0
    
    # Timeline
    first_activity: Optional[str] = None
    last_activity: Optional[str] = None
    days_since_last_activity: Optional[int] = None
    
    # Usage frequency
    active_days_last_30: int = 0
    active_days_last_7: int = 0
    avg_daily_minutes: Optional[float] = None
    avg_session_duration_minutes: Optional[float] = None
    
    # Most recent visitors (who last used it)
    recent_visitors: List[PendoVisitorActivity] = Field(default_factory=list)
    
    # Feature-level data
    top_features: List[PendoFeatureUsage] = Field(default_factory=list)
    
    # Page-level data
    top_pages: List[PendoPageUsage] = Field(default_factory=list)
    
    # Weekly trend (events per week for last 12 weeks)
    weekly_trend: List[Dict[str, Any]] = Field(default_factory=list)


class FullStoryUserData(BaseModel):
    """FullStory user data from BigQuery"""
    id: str
    display_name: Optional[str] = None
    email: Optional[str] = None
    app_url: Optional[str] = None  # Link to FullStory user


class DealSummary(BaseModel):
    """AI-generated summary of deal status from Gong calls"""
    current_state: str = ""  # High-level summary of where the deal stands
    key_topics_discussed: List[str] = Field(default_factory=list)
    blockers_identified: List[str] = Field(default_factory=list)
    next_steps_from_calls: List[str] = Field(default_factory=list)
    champion_sentiment: Optional[str] = None  # positive, neutral, concerned
    risk_factors: List[str] = Field(default_factory=list)


class UserIssueEvent(BaseModel):
    """Individual issue event from FullStory (errors, dead clicks, frustrated moments)"""
    issue_type: str  # e.g., "dead_click", "error", "frustrated", "rage_click"
    error_kind: Optional[str] = None  # e.g., "rateLimit", "TypeError"
    page_url: Optional[str] = None
    page_context: Optional[str] = None  # Extracted context from URL (e.g., "template_evaluation", "models")
    user_email: Optional[str] = None
    user_name: Optional[str] = None
    session_id: Optional[str] = None
    recording_url: Optional[str] = None  # Direct link to FullStory recording
    timestamp: Optional[str] = None
    count: int = 1


class UserErrorEvent(BaseModel):
    """Deprecated - use UserIssueEvent instead"""
    error_type: str
    error_kind: Optional[str] = None
    page_url: Optional[str] = None
    user_email: Optional[str] = None
    user_name: Optional[str] = None
    timestamp: Optional[str] = None
    count: int = 1


class AdoptionMilestone(BaseModel):
    """Tracks whether a user/account has completed a key adoption milestone"""
    name: str  # e.g., "created_project", "sent_traces"
    completed: bool = False
    count: int = 0  # How many times (e.g., 3 projects created)
    first_date: Optional[str] = None  # When first completed
    last_date: Optional[str] = None  # Most recent activity


class UserBehaviorAnalysis(BaseModel):
    """Analysis of user behavior from Pendo/FullStory data"""
    summary: str = ""  # What users are doing in the application
    hypothesis: str = ""  # What they're likely trying to accomplish
    key_workflows_used: List[str] = Field(default_factory=list)
    
    # Adoption milestones - key product activities
    adoption_milestones: List[AdoptionMilestone] = Field(default_factory=list)
    
    critical_issues: List[Dict[str, str]] = Field(default_factory=list)  # Issues/errors encountered
    user_issues: List[UserIssueEvent] = Field(default_factory=list)  # Detailed issue events from FullStory
    issues_summary: Optional[str] = None  # High-level summary of issues with context
    engagement_level: str = "unknown"  # high, medium, low, unknown
    # Prescriptive recommendations with competitive insights, next steps, and internal contacts
    recommendations: List[Any] = Field(default_factory=list)


class SalesEngagementSummary(BaseModel):
    """Summary of sales engagement journey"""
    # Journey timeline
    first_touch_date: Optional[str] = None
    days_in_sales_cycle: Optional[int] = None
    current_stage: Optional[str] = None
    
    # Activity summary
    total_calls: int = 0
    total_emails: int = 0
    total_meetings: int = 0
    total_tasks: int = 0
    
    # Recent activity
    last_sales_activity_date: Optional[str] = None
    days_since_last_activity: Optional[int] = None
    
    # Deal analysis from Gong
    deal_summary: Optional[DealSummary] = None


class ProductUsageSummary(BaseModel):
    """Summary of product usage patterns"""
    # Adoption status
    adoption_status: str = "not_started"  # not_started, exploring, active, power_user, churning
    
    # Usage metrics
    total_users: int = 0
    active_users_last_7_days: int = 0
    active_users_last_30_days: int = 0
    
    # Time on platform
    total_time_minutes: int = 0
    avg_session_minutes: Optional[float] = None
    
    # Last activity
    last_platform_activity: Optional[str] = None
    last_active_user: Optional[str] = None
    days_since_last_activity: Optional[int] = None
    
    # Usage trend
    trend: str = "stable"  # growing, stable, declining


class ProspectOverviewRequest(BaseModel):
    """Request to get prospect overview from BigQuery"""
    # Support multiple lookup methods
    account_name: Optional[str] = None  # Fuzzy match on account name
    domain: Optional[str] = None  # Match on website domain
    sfdc_account_id: Optional[str] = None  # Exact match on Salesforce ID
    
    # Manual competitor input - known competitors for this deal
    manual_competitors: Optional[List[str]] = None  # e.g., ["BrainTrust", "LangSmith"]

    def model_post_init(self, __context):
        """Validate that at least one lookup method is provided."""
        if not self.account_name and not self.domain and not self.sfdc_account_id:
            raise ValueError("At least one of 'account_name', 'domain', or 'sfdc_account_id' must be provided")


class ProspectOverview(BaseModel):
    """Complete prospect overview aggregating data from BigQuery"""
    # Lookup info
    lookup_method: str  # "name", "domain", or "sfdc_id"
    lookup_value: str
    
    # ============================================================================
    # SALES ENGAGEMENT CONTEXT
    # ============================================================================
    
    # Salesforce Account
    salesforce: Optional[SalesforceAccountData] = None
    
    # Latest open opportunity (primary focus)
    latest_opportunity: Optional[OpportunityData] = None
    
    # All opportunities (for context)
    all_opportunities: List[OpportunityData] = Field(default_factory=list)
    
    # Gong Call Analytics with transcripts and themes
    gong_summary: Optional[GongSummaryData] = None
    
    # Sales engagement summary with deal analysis
    sales_engagement: Optional[SalesEngagementSummary] = None
    
    # ============================================================================
    # PRODUCT USAGE PATTERNS
    # ============================================================================
    
    # Pendo Product Usage - detailed
    pendo_usage: Optional[PendoUsageData] = None
    
    # User behavior analysis (from Pendo + FullStory)
    user_behavior: Optional[UserBehaviorAnalysis] = None
    
    # Product usage summary
    product_usage: Optional[ProductUsageSummary] = None
    
    # ============================================================================
    # METADATA
    # ============================================================================
    
    # Which data sources returned data
    data_sources_available: List[str] = Field(default_factory=list)
    
    # Any errors during data fetch
    errors: List[str] = Field(default_factory=list)
