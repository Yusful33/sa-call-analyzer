"""Tests for SignalExtractor pure business logic."""

import pytest
from hypothesis_tool.clients.brave_client import SearchResult
from hypothesis_tool.models.hypothesis import (
    AIMLSignal,
    SignalConfidence,
    CompetitiveSituation,
)


# =============================================================================
# _check_keywords
# =============================================================================

class TestCheckKeywords:
    def test_finds_exact_keyword_match(self, signal_extractor):
        found = signal_extractor._check_keywords("We use LLM models", ["llm"])
        assert "llm" in found

    def test_finds_multiple_keywords(self, signal_extractor):
        found = signal_extractor._check_keywords(
            "Our LLM uses RAG for retrieval", ["llm", "rag", "missing"]
        )
        assert "llm" in found
        assert "rag" in found
        assert "missing" not in found

    def test_returns_empty_for_no_match(self, signal_extractor):
        found = signal_extractor._check_keywords("We sell furniture", ["llm", "rag"])
        assert found == []

    def test_case_insensitive(self, signal_extractor):
        found = signal_extractor._check_keywords("Using LangChain framework", ["langchain"])
        assert "langchain" in found

    def test_empty_text(self, signal_extractor):
        found = signal_extractor._check_keywords("", ["llm"])
        assert found == []

    def test_empty_keywords(self, signal_extractor):
        found = signal_extractor._check_keywords("We use LLM", [])
        assert found == []


# =============================================================================
# _has_financial_context
# =============================================================================

class TestHasFinancialContext:
    def test_detects_financial_keywords(self, signal_extractor):
        assert signal_extractor._has_financial_context(
            "Braintrust Capital portfolio returns and fund investments"
        )

    def test_no_financial_context(self, signal_extractor):
        assert not signal_extractor._has_financial_context(
            "AI engineering team deploys LLM pipeline"
        )

    def test_single_financial_keyword_sufficient(self, signal_extractor):
        assert signal_extractor._has_financial_context("New investment round announced")

    def test_case_insensitive(self, signal_extractor):
        assert signal_extractor._has_financial_context("CAPITAL FUND returns")


# =============================================================================
# _has_ai_context
# =============================================================================

class TestHasAIContext:
    def test_detects_ai_keywords(self, signal_extractor):
        assert signal_extractor._has_ai_context("LLM model training pipeline")

    def test_no_ai_context(self, signal_extractor):
        assert not signal_extractor._has_ai_context("Financial quarterly report summary")

    def test_single_ai_keyword_sufficient(self, signal_extractor):
        assert signal_extractor._has_ai_context("Using latest ai technology")


# =============================================================================
# _is_ambiguous_keyword
# =============================================================================

class TestIsAmbiguousKeyword:
    def test_braintrust_is_ambiguous(self, signal_extractor):
        assert signal_extractor._is_ambiguous_keyword("braintrust")

    def test_wandb_is_ambiguous(self, signal_extractor):
        assert signal_extractor._is_ambiguous_keyword("wandb")

    def test_weights_and_biases_is_ambiguous(self, signal_extractor):
        assert signal_extractor._is_ambiguous_keyword("weights and biases")

    def test_langsmith_not_ambiguous(self, signal_extractor):
        assert not signal_extractor._is_ambiguous_keyword("langsmith")

    def test_langfuse_not_ambiguous(self, signal_extractor):
        assert not signal_extractor._is_ambiguous_keyword("langfuse")

    def test_case_insensitive(self, signal_extractor):
        assert signal_extractor._is_ambiguous_keyword("BRAINTRUST")


# =============================================================================
# extract_signals_from_search_results
# =============================================================================

class TestExtractSignals:
    def test_high_confidence_signal(self, signal_extractor, make_search_result):
        results = [make_search_result(
            title="Company deploys LLM pipeline",
            description="Using generative ai for customer support",
        )]
        signals = signal_extractor.extract_signals_from_search_results(results)
        assert len(signals) == 1
        assert signals[0].confidence == SignalConfidence.HIGH

    def test_medium_confidence_signal(self, signal_extractor, make_search_result):
        results = [make_search_result(
            title="AI-powered features launch",
            description="New machine learning capabilities for data analysis",
        )]
        signals = signal_extractor.extract_signals_from_search_results(results)
        assert len(signals) == 1
        assert signals[0].confidence == SignalConfidence.MEDIUM

    def test_no_signals_from_irrelevant(self, signal_extractor, make_search_result):
        results = [make_search_result(
            title="Office furniture catalog",
            description="Ergonomic chairs and standing desks on sale",
        )]
        signals = signal_extractor.extract_signals_from_search_results(results)
        assert signals == []

    def test_high_takes_priority_over_medium(self, signal_extractor, make_search_result):
        """When both high and medium keywords present, only one HIGH signal is produced."""
        results = [make_search_result(
            title="LLM-powered AI analytics",
            description="Machine learning platform with generative ai capabilities",
        )]
        signals = signal_extractor.extract_signals_from_search_results(results)
        assert len(signals) == 1
        assert signals[0].confidence == SignalConfidence.HIGH

    def test_multiple_results_multiple_signals(self, signal_extractor, make_search_result):
        results = [
            make_search_result(title="LLM chatbot", description="Using RAG"),
            make_search_result(title="ML platform", description="Computer vision system"),
            make_search_result(title="Office supplies", description="Pens and paper"),
        ]
        signals = signal_extractor.extract_signals_from_search_results(results)
        assert len(signals) == 2  # LLM (high) + computer vision (medium), not furniture

    def test_empty_results(self, signal_extractor):
        signals = signal_extractor.extract_signals_from_search_results([])
        assert signals == []

    def test_signal_type_engineering_for_high(self, signal_extractor, make_search_result):
        results = [make_search_result(title="Using LangChain for RAG", description="Vector database embeddings")]
        signals = signal_extractor.extract_signals_from_search_results(results)
        assert signals[0].signal_type == "engineering"

    def test_signal_type_marketing_for_medium(self, signal_extractor, make_search_result):
        results = [make_search_result(
            title="AI-powered analytics dashboard",
            description="Predictive insights for business teams",
        )]
        signals = signal_extractor.extract_signals_from_search_results(results)
        assert signals[0].signal_type == "marketing"

    def test_source_url_captured(self, signal_extractor, make_search_result):
        url = "https://company.com/blog/llm"
        results = [make_search_result(title="LLM deployment", description="RAG pipeline", url=url)]
        signals = signal_extractor.extract_signals_from_search_results(results)
        assert signals[0].source_url == url


# =============================================================================
# detect_competitors
# =============================================================================

class TestDetectCompetitors:
    def test_detects_langsmith(self, signal_extractor, make_search_result):
        results = [make_search_result(
            title="LangSmith tracing setup",
            description="Using langsmith for LLM observability and tracing",
        )]
        situation, tools, evidence = signal_extractor.detect_competitors(results)
        assert situation == CompetitiveSituation.SWITCHING
        assert "langsmith" in tools
        assert len(evidence) > 0

    def test_detects_specific_braintrust_keyword(self, signal_extractor, make_search_result):
        """braintrust.dev is a high-specificity keyword, no AI context needed."""
        results = [make_search_result(
            title="Checking out braintrust.dev",
            description="New platform for evaluations",
        )]
        situation, tools, evidence = signal_extractor.detect_competitors(results)
        assert situation == CompetitiveSituation.SWITCHING
        assert "braintrust" in tools

    def test_filters_braintrust_with_financial_context(self, signal_extractor, make_search_result):
        results = [make_search_result(
            title="Braintrust Capital quarterly report",
            description="Braintrust announced strong portfolio returns and new fund investments",
        )]
        situation, tools, evidence = signal_extractor.detect_competitors(results)
        assert situation != CompetitiveSituation.SWITCHING
        assert "braintrust" not in tools

    def test_accepts_braintrust_with_ai_context(self, signal_extractor, make_search_result):
        results = [make_search_result(
            title="Braintrust AI eval framework",
            description="Using braintrust for LLM evaluation and prompt testing",
        )]
        situation, tools, evidence = signal_extractor.detect_competitors(results)
        assert situation == CompetitiveSituation.SWITCHING
        assert "braintrust" in tools

    def test_filters_wandb_without_ai_context(self, signal_extractor, make_search_result):
        results = [make_search_result(
            title="wandb outdoor gear reviews",
            description="Best boots and outdoor equipment from wandb store",
        )]
        situation, tools, evidence = signal_extractor.detect_competitors(results)
        assert "weights_and_biases" not in tools

    def test_accepts_wandb_with_ai_context(self, signal_extractor, make_search_result):
        results = [make_search_result(
            title="wandb experiment tracking",
            description="Using wandb for ML model training and experiment tracking",
        )]
        situation, tools, evidence = signal_extractor.detect_competitors(results)
        assert situation == CompetitiveSituation.SWITCHING
        assert "weights_and_biases" in tools

    def test_multiple_competitors_detected(self, signal_extractor, make_search_result):
        results = [
            make_search_result(
                title="LangSmith vs Langfuse comparison",
                description="Comparing langsmith and langfuse for LLM tracing",
            ),
        ]
        situation, tools, evidence = signal_extractor.detect_competitors(results)
        assert situation == CompetitiveSituation.SWITCHING
        assert "langsmith" in tools
        assert "langfuse" in tools

    def test_homegrown_detection(self, signal_extractor, make_search_result):
        results = [make_search_result(
            title="Company monitoring stack",
            description="Using grafana dashboards for internal ML monitoring",
        )]
        situation, tools, evidence = signal_extractor.detect_competitors(results)
        assert situation == CompetitiveSituation.BUILD_VS_BUY

    def test_no_competitors_returns_unknown(self, signal_extractor, make_search_result):
        results = [make_search_result(
            title="Company website",
            description="We provide cloud consulting services",
        )]
        situation, tools, evidence = signal_extractor.detect_competitors(results)
        assert situation == CompetitiveSituation.UNKNOWN
        assert tools == []
        assert evidence == []

    def test_evidence_fields_populated(self, signal_extractor, make_search_result):
        results = [make_search_result(
            title="Langfuse tracing",
            url="https://blog.com/langfuse",
            description="Using langfuse for LLM observability",
        )]
        _, _, evidence = signal_extractor.detect_competitors(results)
        assert len(evidence) > 0
        e = evidence[0]
        assert e.tool == "langfuse"
        assert e.keyword_matched is not None
        assert e.source_title == "Langfuse tracing"
        assert e.source_url == "https://blog.com/langfuse"

    def test_description_truncated_at_200(self, signal_extractor, make_search_result):
        long_desc = "langfuse tracing for LLM " + "x" * 300
        results = [make_search_result(title="Test", description=long_desc)]
        _, _, evidence = signal_extractor.detect_competitors(results)
        assert len(evidence) > 0
        assert evidence[0].source_description.endswith("...")
        assert len(evidence[0].source_description) <= 204  # 200 + "..."


# =============================================================================
# calculate_overall_confidence
# =============================================================================

class TestCalculateOverallConfidence:
    def _make_signal(self, confidence):
        return AIMLSignal(
            signal_type="test",
            evidence="test",
            confidence=confidence,
        )

    def test_empty_signals_returns_low(self, signal_extractor):
        assert signal_extractor.calculate_overall_confidence([]) == SignalConfidence.LOW

    def test_single_high_returns_medium(self, signal_extractor):
        signals = [self._make_signal(SignalConfidence.HIGH)]
        assert signal_extractor.calculate_overall_confidence(signals) == SignalConfidence.MEDIUM

    def test_two_high_returns_high(self, signal_extractor):
        signals = [
            self._make_signal(SignalConfidence.HIGH),
            self._make_signal(SignalConfidence.HIGH),
        ]
        assert signal_extractor.calculate_overall_confidence(signals) == SignalConfidence.HIGH

    def test_three_medium_returns_medium(self, signal_extractor):
        signals = [self._make_signal(SignalConfidence.MEDIUM)] * 3
        assert signal_extractor.calculate_overall_confidence(signals) == SignalConfidence.MEDIUM

    def test_two_medium_returns_low(self, signal_extractor):
        signals = [self._make_signal(SignalConfidence.MEDIUM)] * 2
        assert signal_extractor.calculate_overall_confidence(signals) == SignalConfidence.LOW

    def test_one_high_one_medium_returns_medium(self, signal_extractor):
        signals = [
            self._make_signal(SignalConfidence.HIGH),
            self._make_signal(SignalConfidence.MEDIUM),
        ]
        assert signal_extractor.calculate_overall_confidence(signals) == SignalConfidence.MEDIUM

    def test_mixed_signals_high_dominates(self, signal_extractor):
        signals = (
            [self._make_signal(SignalConfidence.HIGH)] * 2
            + [self._make_signal(SignalConfidence.MEDIUM)] * 5
        )
        assert signal_extractor.calculate_overall_confidence(signals) == SignalConfidence.HIGH
