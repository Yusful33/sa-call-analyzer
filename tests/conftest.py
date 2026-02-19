"""Shared test fixtures and configuration."""

import os
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime


# Set test environment variables BEFORE any application imports
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-123")
os.environ.setdefault("BRAVE_API_KEY", "test-brave-key")
os.environ.setdefault("ARIZE_API_KEY", "")
os.environ.setdefault("ARIZE_SPACE_ID", "")


@pytest.fixture(autouse=True, scope="session")
def clear_settings_cache():
    """Clear the lru_cache on get_settings to prevent stale config."""
    from hypothesis_tool.config import get_settings
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def signal_extractor():
    """Create a SignalExtractor with mocked Anthropic client."""
    mock_settings = MagicMock()
    mock_settings.anthropic_api_key = "test-key-123"
    mock_settings.brave_api_key = "test-brave-key"
    mock_settings.llm_model = "claude-sonnet-4-20250514"

    with patch(
        "hypothesis_tool.analyzers.signal_extractor.get_settings",
        return_value=mock_settings,
    ):
        with patch("hypothesis_tool.analyzers.signal_extractor.anthropic.Anthropic"):
            from hypothesis_tool.analyzers.signal_extractor import SignalExtractor
            return SignalExtractor()


@pytest.fixture
def make_search_result():
    """Factory fixture to create SearchResult objects."""
    from hypothesis_tool.clients.brave_client import SearchResult

    def _make(title="Test Result", url="https://example.com", description="Test description", page_age=None):
        return SearchResult(title=title, url=url, description=description, page_age=page_age)

    return _make


@pytest.fixture
def sample_search_results(make_search_result):
    """A varied set of SearchResult objects for testing."""
    return [
        make_search_result(
            title="Company deploys LLM-powered chatbot",
            url="https://company.com/blog/llm-chatbot",
            description="We built a generative ai chatbot using LangChain and RAG",
        ),
        make_search_result(
            title="Company uses AI-powered analytics",
            url="https://company.com/features",
            description="Our machine learning platform provides predictive insights",
        ),
        make_search_result(
            title="Company evaluates LangSmith for tracing",
            url="https://devblog.com/langsmith-review",
            description="Using langsmith to trace our LLM pipelines and evaluate prompts",
        ),
        make_search_result(
            title="Braintrust Capital quarterly report",
            url="https://finance.com/braintrust-capital",
            description="Braintrust Capital announced strong portfolio returns and new fund investments",
        ),
        make_search_result(
            title="Braintrust AI eval framework",
            url="https://tech.com/braintrust-eval",
            description="Using braintrust for LLM evaluation and prompt testing",
        ),
        make_search_result(
            title="Company office furniture catalog",
            url="https://company.com/furniture",
            description="Browse our latest collection of ergonomic office chairs and standing desks",
        ),
    ]


@pytest.fixture
def sample_hypothesis_result():
    """A fully populated HypothesisResult for serialization tests."""
    from hypothesis_tool.models.hypothesis import (
        HypothesisResult,
        CompanyResearch,
        Hypothesis,
        AIMLSignal,
        SignalConfidence,
        HypothesisConfidence,
        CompetitiveSituation,
        ValueCategory,
        ResearchQuality,
        SupportingSignal,
        DiscoveryQuestion,
        SimilarCustomer,
        CompetitiveBattlecard,
    )

    research = CompanyResearch(
        company_name="Acme Corp",
        domain="acme.com",
        industry="Technology",
        employee_count=500,
        ai_ml_signals=[
            AIMLSignal(
                signal_type="engineering",
                evidence="Found: llm, langchain in 'Acme deploys LLM pipeline'",
                confidence=SignalConfidence.HIGH,
                source_url="https://acme.com/blog",
            ),
        ],
        ai_ml_confidence=SignalConfidence.HIGH,
        competitive_situation=CompetitiveSituation.GREENFIELD,
    )

    hypothesis = Hypothesis(
        hypothesis="Acme needs LLM observability for their chatbot pipeline",
        confidence=HypothesisConfidence.HIGH,
        confidence_reasoning="Strong AI signals and greenfield opportunity",
        current_state="Using print statements to debug LLM issues",
        future_state="Full visibility into LLM pipeline performance",
        required_capabilities=["LLM tracing", "Prompt evaluation"],
        negative_consequences="Increased debugging time and user-facing errors",
        value_category=ValueCategory.INCREASE_EFFICIENCY,
        supporting_signals=[
            SupportingSignal(
                description="Blog post about LLM deployment challenges",
                source_url="https://acme.com/blog/challenges",
                confidence="high",
            ),
        ],
        discovery_questions=[
            DiscoveryQuestion(
                question="How do you currently debug LLM failures?",
                rationale="Validates pain around observability",
            ),
        ],
        similar_customers=[
            SimilarCustomer(
                customer_name="TechCo",
                industry="Technology",
                use_case="LLM pipeline monitoring",
                outcome="Reduced debugging time by 80%",
            ),
        ],
    )

    return HypothesisResult(
        company_name="Acme Corp",
        research=research,
        hypotheses=[hypothesis],
        competitive_context=CompetitiveBattlecard(
            situation=CompetitiveSituation.GREENFIELD,
            positioning="First-mover advantage in LLM observability",
            key_questions=["What tools do you use today?"],
            advantages=["Full-stack LLM observability"],
            watch_outs=["May evaluate open-source alternatives"],
        ),
        research_quality=ResearchQuality.HIGH,
        processing_time_seconds=12.5,
    )
