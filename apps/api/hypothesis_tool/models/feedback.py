"""Pydantic models for feedback tracking."""

from datetime import datetime
from pydantic import BaseModel, Field
from typing import Optional


class FeedbackCreate(BaseModel):
    """Model for creating new feedback."""

    # What was researched
    company_name: str
    hypothesis_id: str | None = Field(
        None, description="ID of the specific hypothesis being rated"
    )

    # Immediate feedback
    was_helpful: bool | None = Field(None, description="Thumbs up/down")
    helpfulness_comment: str | None = Field(
        None, description="Optional comment on helpfulness"
    )

    # Post-call feedback
    hypothesis_came_up: bool | None = Field(
        None, description="Did this hypothesis come up on the call?"
    )
    hypothesis_resonated: bool | None = Field(
        None, description="Did the hypothesis resonate with the prospect?"
    )

    # Outcome tracking
    led_to_meeting: bool | None = Field(
        None, description="Did this research help book a meeting?"
    )
    led_to_opportunity: bool | None = Field(
        None, description="Did this lead to an opportunity?"
    )

    # Additional context
    ae_notes: str | None = Field(
        None, description="Any additional notes from the AE"
    )


class Feedback(FeedbackCreate):
    """Full feedback model with metadata."""

    id: int
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime | None = None

    # Link to the research session
    research_session_id: str | None = None

    class Config:
        from_attributes = True
