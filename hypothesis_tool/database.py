"""Database setup and feedback storage using SQLAlchemy."""

from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from .config import get_settings

Base = declarative_base()


class FeedbackRecord(Base):
    """SQLAlchemy model for feedback storage."""

    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, onupdate=datetime.utcnow)

    # What was researched
    company_name = Column(String(255), nullable=False)
    hypothesis_id = Column(String(255), nullable=True)
    research_session_id = Column(String(255), nullable=True)

    # Immediate feedback
    was_helpful = Column(Boolean, nullable=True)
    helpfulness_comment = Column(Text, nullable=True)

    # Post-call feedback
    hypothesis_came_up = Column(Boolean, nullable=True)
    hypothesis_resonated = Column(Boolean, nullable=True)

    # Outcome tracking
    led_to_meeting = Column(Boolean, nullable=True)
    led_to_opportunity = Column(Boolean, nullable=True)

    # Additional context
    ae_notes = Column(Text, nullable=True)


# Database setup
def get_engine():
    """Get SQLAlchemy engine."""
    settings = get_settings()
    return create_engine(settings.database_url, echo=settings.debug)


def init_db():
    """Initialize the database and create tables."""
    engine = get_engine()
    Base.metadata.create_all(engine)
    return engine


def get_session():
    """Get a database session."""
    engine = get_engine()
    Session = sessionmaker(bind=engine)
    return Session()


class FeedbackStore:
    """Store for feedback data."""

    def __init__(self):
        self.engine = init_db()
        self.Session = sessionmaker(bind=self.engine)

    def create_feedback(
        self,
        company_name: str,
        was_helpful: bool | None = None,
        hypothesis_id: str | None = None,
        helpfulness_comment: str | None = None,
        ae_notes: str | None = None,
    ) -> FeedbackRecord:
        """Create a new feedback record."""
        session = self.Session()
        try:
            record = FeedbackRecord(
                company_name=company_name,
                was_helpful=was_helpful,
                hypothesis_id=hypothesis_id,
                helpfulness_comment=helpfulness_comment,
                ae_notes=ae_notes,
            )
            session.add(record)
            session.commit()
            session.refresh(record)
            return record
        finally:
            session.close()

    def update_feedback(
        self,
        feedback_id: int,
        hypothesis_came_up: bool | None = None,
        hypothesis_resonated: bool | None = None,
        led_to_meeting: bool | None = None,
        led_to_opportunity: bool | None = None,
        ae_notes: str | None = None,
    ) -> FeedbackRecord | None:
        """Update an existing feedback record with post-call information."""
        session = self.Session()
        try:
            record = session.query(FeedbackRecord).filter_by(id=feedback_id).first()
            if not record:
                return None

            if hypothesis_came_up is not None:
                record.hypothesis_came_up = hypothesis_came_up
            if hypothesis_resonated is not None:
                record.hypothesis_resonated = hypothesis_resonated
            if led_to_meeting is not None:
                record.led_to_meeting = led_to_meeting
            if led_to_opportunity is not None:
                record.led_to_opportunity = led_to_opportunity
            if ae_notes is not None:
                record.ae_notes = ae_notes

            session.commit()
            session.refresh(record)
            return record
        finally:
            session.close()

    def get_feedback_stats(self) -> dict:
        """Get aggregate feedback statistics."""
        session = self.Session()
        try:
            total = session.query(FeedbackRecord).count()
            helpful = (
                session.query(FeedbackRecord)
                .filter(FeedbackRecord.was_helpful == True)
                .count()
            )
            not_helpful = (
                session.query(FeedbackRecord)
                .filter(FeedbackRecord.was_helpful == False)
                .count()
            )
            led_to_meetings = (
                session.query(FeedbackRecord)
                .filter(FeedbackRecord.led_to_meeting == True)
                .count()
            )

            return {
                "total_feedback": total,
                "helpful_count": helpful,
                "not_helpful_count": not_helpful,
                "helpfulness_rate": helpful / total if total > 0 else 0,
                "meetings_booked": led_to_meetings,
            }
        finally:
            session.close()

    def get_recent_feedback(self, limit: int = 20) -> list[FeedbackRecord]:
        """Get recent feedback records."""
        session = self.Session()
        try:
            return (
                session.query(FeedbackRecord)
                .order_by(FeedbackRecord.created_at.desc())
                .limit(limit)
                .all()
            )
        finally:
            session.close()
