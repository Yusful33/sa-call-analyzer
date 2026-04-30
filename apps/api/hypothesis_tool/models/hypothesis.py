"""Pydantic models for hypothesis generation and research results."""

from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field


class SignalConfidence(str, Enum):
    """Confidence level for detected signals."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class HypothesisConfidence(str, Enum):
    """Confidence level for a hypothesis."""

    HIGH = "high"  # Industry has n>=10 won deals, specific use case match
    MEDIUM = "medium"  # Industry has 5-9 deals, OR adjacent industry pattern
    LOW = "low"  # Industry has <5 deals, using general ML observability pitch
    EXPLORATORY = "exploratory"  # New vertical, no data


class ValueCategory(str, Enum):
    """Value category for hypothesis - maps to business outcomes."""

    REDUCE_RISK = "reduce_risk"  # Reduce operational/business risk
    INCREASE_EFFICIENCY = "increase_efficiency"  # Go faster, reduce troubleshooting time
    INCREASE_REVENUE = "increase_revenue"  # Better model performance = more revenue
    REDUCE_COST = "reduce_cost"  # Avoid build, reduce token costs, retain talent


class CompetitiveSituation(str, Enum):
    """Detected competitive situation."""

    GREENFIELD = "greenfield"  # No ML observability in place
    SWITCHING = "switching"  # Using a competitor
    BUILD_VS_BUY = "build_vs_buy"  # Using homegrown solution
    UNKNOWN = "unknown"


class AIMLSignal(BaseModel):
    """An AI/ML signal detected from research."""

    signal_type: str = Field(..., description="Type of signal (blog, job, product, etc)")
    evidence: str = Field(..., description="The actual evidence found")
    confidence: SignalConfidence = Field(..., description="Confidence in this signal")
    source_url: str | None = Field(None, description="URL where this was found")


class CompetitorEvidence(BaseModel):
    """Evidence for competitor tool detection."""

    tool: str = Field(..., description="The competitor tool detected")
    keyword_matched: str = Field(..., description="The keyword that triggered detection")
    source_title: str = Field(..., description="Title of the source where found")
    source_description: str = Field(..., description="Description/snippet from the source")
    source_url: str | None = Field(None, description="URL where this was found")


class CompanyResearch(BaseModel):
    """Research results for a company."""

    company_name: str
    domain: str | None = None
    industry: str | None = None
    employee_count: int | None = None

    # AI/ML signals
    ai_ml_signals: list[AIMLSignal] = Field(default_factory=list)
    ai_ml_confidence: SignalConfidence = SignalConfidence.LOW

    # Competitive situation
    competitive_situation: CompetitiveSituation = CompetitiveSituation.UNKNOWN
    detected_tools: list[str] = Field(
        default_factory=list, description="ML tools detected (LangSmith, W&B, etc)"
    )
    competitor_evidence: list[CompetitorEvidence] = Field(
        default_factory=list, description="Evidence for why competitors were detected"
    )

    # CRM data
    exists_in_crm: bool = False
    prior_opportunities: int = 0
    last_contact_date: datetime | None = None

    # Summary
    company_summary: str | None = Field(
        None, description="Brief summary of the company"
    )

    # Prospect's known GenAI product(s) – e.g. Gusto → Gus. Aligns hypotheses with their actual offering.
    genai_products: list[dict] = Field(
        default_factory=list,
        description="Known GenAI product(s) for this prospect: product_name, product_description, use_cases, arize_angles",
    )

    research_completed_at: datetime = Field(default_factory=datetime.utcnow)


class DiscoveryQuestion(BaseModel):
    """A discovery question tailored to the prospect."""

    question: str = Field(..., description="The discovery question")
    rationale: str = Field(
        ..., description="Why this question is relevant for this prospect"
    )
    tied_to_signal: str | None = Field(
        None, description="What research finding triggered this"
    )
    follow_up_if_yes: str | None = None
    follow_up_if_no: str | None = None


class SimilarCustomer(BaseModel):
    """A similar customer for proof points."""

    customer_name: str = Field(..., description="Customer name (may be anonymized)")
    industry: str
    deal_size: float | None = None
    use_case: str | None = None
    outcome: str | None = Field(
        None, description="Brief description of outcome/value delivered"
    )


class Hypothesis(BaseModel):
    """A hypothesis about where Arize could add value.
    
    Structured around the Command of the Message framework:
    - Current State: What they're doing now and challenges
    - Future State: What they could achieve
    - Required Capabilities: What they need (Arize value)
    - Negative Consequences: Risks of inaction
    """

    hypothesis: str = Field(..., description="The hypothesis statement")
    confidence: HypothesisConfidence = Field(..., description="Confidence level")
    confidence_reasoning: str = Field(
        ..., description="Why we have this confidence level"
    )

    # Command of the Message framework
    current_state: str | None = Field(
        None, description="What they're doing today and the challenges they face"
    )
    future_state: str | None = Field(
        None, description="What they could achieve with the right solution"
    )
    required_capabilities: list[str] = Field(
        default_factory=list, description="Capabilities they need (maps to Arize features)"
    )
    negative_consequences: str | None = Field(
        None, description="What happens if they don't address this"
    )
    value_category: ValueCategory | None = Field(
        None, description="Business value category (reduce_risk, increase_efficiency, increase_revenue, reduce_cost)"
    )

    # Supporting evidence
    supporting_signals: list["SupportingSignal"] = Field(
        default_factory=list, description="Research signals that support this hypothesis"
    )
    playbook_matches: list[str] = Field(
        default_factory=list,
        description="Patterns from won deals that match",
    )

    # Actionable outputs
    discovery_questions: list[DiscoveryQuestion] = Field(
        default_factory=list, description="Questions to validate this hypothesis"
    )
    similar_customers: list[SimilarCustomer] = Field(
        default_factory=list, description="Similar customers for proof points"
    )


class SupportingSignal(BaseModel):
    """A signal that supports a hypothesis, with source link."""
    
    description: str = Field(..., description="Description of the signal")
    source_url: str | None = Field(None, description="URL to verify the signal")
    confidence: str = Field("medium", description="Signal confidence level")


class CompetitiveBattlecard(BaseModel):
    """Competitive context and battlecard."""

    situation: CompetitiveSituation
    detected_competitor: str | None = None

    positioning: str = Field(..., description="How to position against this situation")
    key_questions: list[str] = Field(
        default_factory=list, description="Key questions to ask"
    )
    advantages: list[str] = Field(
        default_factory=list, description="Arize advantages in this situation"
    )
    watch_outs: list[str] = Field(
        default_factory=list, description="Things to be careful about"
    )


class ResearchQuality(str, Enum):
    """Quality indicator for research results."""
    
    HIGH = "high"  # Found rich signals, confident hypotheses
    MEDIUM = "medium"  # Some signals, reasonable hypotheses
    LOW = "low"  # Limited signals, generic hypotheses
    INSUFFICIENT = "insufficient"  # Not enough data to be useful


class HypothesisResult(BaseModel):
    """Complete hypothesis result returned to AE."""

    # Input
    company_name: str
    requested_at: datetime = Field(default_factory=datetime.utcnow)

    # Research
    research: CompanyResearch

    # Hypotheses (ranked by confidence)
    hypotheses: list[Hypothesis] = Field(
        default_factory=list, description="Ranked hypotheses"
    )

    # Competitive context
    competitive_context: CompetitiveBattlecard | None = None

    # Metadata
    playbook_used: str | None = Field(
        None, description="Which industry playbook was used"
    )
    playbook_deal_count: int | None = Field(
        None, description="How many deals the playbook is based on"
    )

    # Confidence note
    confidence_note: str | None = Field(
        None,
        description="Note about data availability (e.g., 'Limited data in this industry')",
    )
    
    # Quality and status indicators
    research_quality: ResearchQuality = Field(
        default=ResearchQuality.MEDIUM,
        description="Overall quality of the research results"
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Non-critical warnings about the research"
    )
    errors: list[str] = Field(
        default_factory=list,
        description="Critical errors that occurred during research"
    )

    # Processing time
    processing_time_seconds: float | None = None
