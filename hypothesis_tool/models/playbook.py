"""Pydantic models for industry playbooks."""

from datetime import datetime
from pydantic import BaseModel, Field


class PainPoint(BaseModel):
    """A pain point identified from won deals."""

    pain: str = Field(..., description="Description of the pain point")
    frequency: int = Field(..., description="Number of deals where this was mentioned")
    example_quote: str | None = Field(
        None, description="Example quote from a call demonstrating this pain"
    )


class ValueProp(BaseModel):
    """A value proposition that resonated with customers."""

    value: str = Field(..., description="The value proposition")
    frequency: int = Field(..., description="Number of deals where this resonated")
    use_case: str | None = Field(
        None, description="Specific use case this applies to"
    )


class Objection(BaseModel):
    """A common objection and how to handle it."""

    objection: str = Field(..., description="The objection text")
    stage_typically_appears: str = Field(
        "early", description="When this typically comes up (early/mid/late)"
    )
    effective_response: str = Field(
        ..., description="Effective response that led to won deals"
    )
    frequency: int = Field(..., description="How often this objection appears")


class DiscoveryQuestionTemplate(BaseModel):
    """A discovery question template tied to a signal."""

    question: str = Field(..., description="The discovery question")
    tied_to_signal: str = Field(
        ..., description="What research finding triggers this question"
    )
    validates_hypothesis: str = Field(
        ..., description="Which hypothesis this question tests"
    )
    follow_up_if_yes: str | None = Field(
        None, description="Follow-up if they confirm"
    )
    follow_up_if_no: str | None = Field(None, description="Follow-up if they deny")


class IndustryPlaybook(BaseModel):
    """A complete playbook for an industry based on won deal analysis."""

    industry: str = Field(..., description="Industry name")
    deal_count: int = Field(..., description="Number of won deals analyzed")
    generated_at: datetime = Field(
        default_factory=datetime.utcnow, description="When this playbook was generated"
    )

    top_pain_points: list[PainPoint] = Field(
        default_factory=list, description="Top pain points from won deals"
    )
    winning_value_props: list[ValueProp] = Field(
        default_factory=list, description="Value props that resonated"
    )
    common_objections: list[Objection] = Field(
        default_factory=list, description="Common objections and responses"
    )
    discovery_questions: list[DiscoveryQuestionTemplate] = Field(
        default_factory=list, description="Recommended discovery questions"
    )

    # Metadata
    sample_customers: list[str] = Field(
        default_factory=list, description="Example customer names (anonymized)"
    )
    avg_deal_size: float | None = Field(
        None, description="Average deal size in this industry"
    )
    avg_sales_cycle_days: int | None = Field(
        None, description="Average sales cycle length"
    )
