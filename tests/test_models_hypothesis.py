"""Tests for hypothesis pipeline Pydantic model validation."""

import pytest
from datetime import datetime

from hypothesis_tool.models.hypothesis import (
    AIMLSignal,
    SignalConfidence,
    HypothesisConfidence,
    ValueCategory,
    CompetitiveSituation,
    ResearchQuality,
    CompetitorEvidence,
    CompanyResearch,
    Hypothesis,
    SupportingSignal,
    DiscoveryQuestion,
    SimilarCustomer,
    CompetitiveBattlecard,
    HypothesisResult,
)
from hypothesis_tool.clients.brave_client import SearchResult


class TestSearchResult:
    def test_creation(self):
        sr = SearchResult(title="Test", url="https://x.com", description="Desc")
        assert sr.title == "Test"
        assert sr.url == "https://x.com"
        assert sr.description == "Desc"

    def test_optional_page_age(self):
        sr = SearchResult(title="T", url="https://x.com", description="D")
        assert sr.page_age is None

    def test_model_dump(self):
        sr = SearchResult(title="T", url="https://x.com", description="D", page_age="2d")
        d = sr.model_dump()
        assert d["page_age"] == "2d"


class TestAIMLSignal:
    def test_creation(self):
        sig = AIMLSignal(
            signal_type="engineering",
            evidence="Found: llm in 'Blog post'",
            confidence=SignalConfidence.HIGH,
            source_url="https://example.com",
        )
        assert sig.signal_type == "engineering"
        assert sig.confidence == SignalConfidence.HIGH

    def test_optional_source_url(self):
        sig = AIMLSignal(
            signal_type="marketing",
            evidence="Found: ai-powered",
            confidence=SignalConfidence.MEDIUM,
        )
        assert sig.source_url is None


class TestEnumValues:
    def test_signal_confidence_values(self):
        assert SignalConfidence.HIGH == "high"
        assert SignalConfidence.MEDIUM == "medium"
        assert SignalConfidence.LOW == "low"

    def test_hypothesis_confidence_values(self):
        assert HypothesisConfidence.HIGH == "high"
        assert HypothesisConfidence.MEDIUM == "medium"
        assert HypothesisConfidence.LOW == "low"
        assert HypothesisConfidence.EXPLORATORY == "exploratory"

    def test_value_category_values(self):
        assert ValueCategory.REDUCE_RISK == "reduce_risk"
        assert ValueCategory.INCREASE_EFFICIENCY == "increase_efficiency"
        assert ValueCategory.INCREASE_REVENUE == "increase_revenue"
        assert ValueCategory.REDUCE_COST == "reduce_cost"

    def test_competitive_situation_values(self):
        assert CompetitiveSituation.GREENFIELD == "greenfield"
        assert CompetitiveSituation.SWITCHING == "switching"
        assert CompetitiveSituation.BUILD_VS_BUY == "build_vs_buy"
        assert CompetitiveSituation.UNKNOWN == "unknown"

    def test_research_quality_values(self):
        assert ResearchQuality.HIGH == "high"
        assert ResearchQuality.MEDIUM == "medium"
        assert ResearchQuality.LOW == "low"
        assert ResearchQuality.INSUFFICIENT == "insufficient"


class TestCompetitorEvidence:
    def test_creation(self):
        ce = CompetitorEvidence(
            tool="langsmith",
            keyword_matched="langsmith",
            source_title="Blog",
            source_description="Using langsmith for tracing",
            source_url="https://blog.com/langsmith",
        )
        assert ce.tool == "langsmith"
        assert ce.source_url == "https://blog.com/langsmith"


class TestCompanyResearch:
    def test_minimal_creation(self):
        cr = CompanyResearch(company_name="Acme Corp")
        assert cr.company_name == "Acme Corp"
        assert cr.ai_ml_signals == []
        assert cr.ai_ml_confidence == SignalConfidence.LOW
        assert cr.competitive_situation == CompetitiveSituation.UNKNOWN
        assert cr.exists_in_crm is False

    def test_full_creation(self):
        cr = CompanyResearch(
            company_name="Acme Corp",
            domain="acme.com",
            industry="Technology",
            employee_count=500,
            ai_ml_signals=[
                AIMLSignal(
                    signal_type="engineering",
                    evidence="test",
                    confidence=SignalConfidence.HIGH,
                ),
            ],
            ai_ml_confidence=SignalConfidence.HIGH,
            competitive_situation=CompetitiveSituation.GREENFIELD,
            detected_tools=["langsmith"],
            exists_in_crm=True,
            company_summary="A tech company using AI.",
        )
        assert cr.employee_count == 500
        assert len(cr.ai_ml_signals) == 1
        assert cr.detected_tools == ["langsmith"]


class TestHypothesis:
    def test_creation_with_command_of_message(self):
        h = Hypothesis(
            hypothesis="Company needs LLM observability",
            confidence=HypothesisConfidence.HIGH,
            confidence_reasoning="Strong signals",
            current_state="No visibility into LLM pipeline",
            future_state="Full tracing and evaluation",
            required_capabilities=["LLM tracing", "Evals"],
            negative_consequences="Blind to production issues",
            value_category=ValueCategory.REDUCE_RISK,
        )
        assert h.hypothesis == "Company needs LLM observability"
        assert len(h.required_capabilities) == 2

    def test_optional_fields_default(self):
        h = Hypothesis(
            hypothesis="Test",
            confidence=HypothesisConfidence.LOW,
            confidence_reasoning="Limited data",
        )
        assert h.current_state is None
        assert h.future_state is None
        assert h.required_capabilities == []
        assert h.supporting_signals == []
        assert h.discovery_questions == []
        assert h.similar_customers == []


class TestSupportingSignal:
    def test_default_confidence(self):
        ss = SupportingSignal(description="Found blog post")
        assert ss.confidence == "medium"
        assert ss.source_url is None


class TestDiscoveryQuestion:
    def test_optional_follow_ups(self):
        dq = DiscoveryQuestion(
            question="How do you debug LLM failures?",
            rationale="Validates pain",
        )
        assert dq.follow_up_if_yes is None
        assert dq.follow_up_if_no is None


class TestHypothesisResultRoundTrip:
    def test_serialization_roundtrip(self, sample_hypothesis_result):
        """Serialize to dict, then validate back to model."""
        d = sample_hypothesis_result.model_dump()
        restored = HypothesisResult.model_validate(d)

        assert restored.company_name == sample_hypothesis_result.company_name
        assert restored.research_quality == sample_hypothesis_result.research_quality
        assert len(restored.hypotheses) == len(sample_hypothesis_result.hypotheses)
        assert restored.hypotheses[0].hypothesis == sample_hypothesis_result.hypotheses[0].hypothesis
        assert restored.competitive_context is not None
        assert restored.processing_time_seconds == 12.5

    def test_json_roundtrip(self, sample_hypothesis_result):
        """Serialize to JSON string and back."""
        json_str = sample_hypothesis_result.model_dump_json()
        restored = HypothesisResult.model_validate_json(json_str)
        assert restored.company_name == "Acme Corp"
        assert len(restored.research.ai_ml_signals) == 1
