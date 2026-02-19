"""Performance regression tests.

These tests establish timing baselines for CPU-bound operations.
Bounds are intentionally generous (10x expected) to avoid CI flakiness
while still catching algorithmic regressions.
"""

import time
import pytest

from hypothesis_tool.clients.brave_client import SearchResult
from hypothesis_tool.models.hypothesis import (
    AIMLSignal,
    SignalConfidence,
    HypothesisConfidence,
    CompetitiveSituation,
    ValueCategory,
    ResearchQuality,
    CompanyResearch,
    Hypothesis,
    HypothesisResult,
    CompetitiveBattlecard,
)


def _make_search_results(n: int) -> list[SearchResult]:
    """Generate n varied SearchResult objects."""
    results = []
    templates = [
        ("Company deploys LLM pipeline", "Using langchain and RAG for AI chatbot"),
        ("AI-powered analytics", "Machine learning predictive platform"),
        ("LangSmith tracing setup", "langsmith for LLM observability"),
        ("Office furniture sale", "Ergonomic chairs and standing desks"),
        ("Company engineering blog", "Building scalable microservices"),
    ]
    for i in range(n):
        title, desc = templates[i % len(templates)]
        results.append(SearchResult(
            title=f"{title} #{i}",
            url=f"https://example.com/{i}",
            description=f"{desc} - variant {i}",
        ))
    return results


def _make_hypothesis_result() -> HypothesisResult:
    """Generate a fully populated HypothesisResult."""
    return HypothesisResult(
        company_name="Test Corp",
        research=CompanyResearch(
            company_name="Test Corp",
            domain="test.com",
            industry="Technology",
            employee_count=1000,
            ai_ml_signals=[
                AIMLSignal(
                    signal_type="engineering",
                    evidence=f"Signal evidence #{i}",
                    confidence=SignalConfidence.HIGH if i % 2 == 0 else SignalConfidence.MEDIUM,
                    source_url=f"https://test.com/signal/{i}",
                )
                for i in range(5)
            ],
            ai_ml_confidence=SignalConfidence.HIGH,
            competitive_situation=CompetitiveSituation.GREENFIELD,
        ),
        hypotheses=[
            Hypothesis(
                hypothesis=f"Hypothesis #{i}",
                confidence=HypothesisConfidence.HIGH,
                confidence_reasoning="Strong signals",
                current_state="Current state description",
                future_state="Future state description",
                required_capabilities=["Cap A", "Cap B", "Cap C"],
                negative_consequences="Risk of inaction",
                value_category=ValueCategory.REDUCE_RISK,
            )
            for i in range(3)
        ],
        competitive_context=CompetitiveBattlecard(
            situation=CompetitiveSituation.GREENFIELD,
            positioning="First mover advantage",
            key_questions=["Q1", "Q2"],
            advantages=["Adv1", "Adv2"],
            watch_outs=["Watch1"],
        ),
        research_quality=ResearchQuality.HIGH,
        processing_time_seconds=10.0,
    )


@pytest.mark.slow
class TestSignalExtractionPerformance:
    def test_extract_signals_100_results(self, signal_extractor):
        """Signal extraction from 100 results must complete under 50ms."""
        results = _make_search_results(100)
        start = time.perf_counter()
        signal_extractor.extract_signals_from_search_results(results)
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 50, f"Signal extraction took {elapsed_ms:.1f}ms (limit: 50ms)"

    def test_competitor_detection_100_results(self, signal_extractor):
        """Competitor detection from 100 results must complete under 50ms."""
        results = _make_search_results(100)
        start = time.perf_counter()
        signal_extractor.detect_competitors(results)
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 50, f"Competitor detection took {elapsed_ms:.1f}ms (limit: 50ms)"


@pytest.mark.slow
class TestConfidenceCalculationPerformance:
    def test_confidence_from_1000_signals(self, signal_extractor):
        """Confidence calculation from 1000 signals must complete under 10ms."""
        signals = [
            AIMLSignal(
                signal_type="test",
                evidence=f"Evidence #{i}",
                confidence=SignalConfidence.HIGH if i % 3 == 0 else SignalConfidence.MEDIUM,
            )
            for i in range(1000)
        ]
        start = time.perf_counter()
        signal_extractor.calculate_overall_confidence(signals)
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 10, f"Confidence calc took {elapsed_ms:.1f}ms (limit: 10ms)"


@pytest.mark.slow
class TestModelSerializationPerformance:
    def test_hypothesis_result_roundtrip_50(self):
        """Serialize/deserialize 50 HypothesisResult objects under 100ms."""
        results = [_make_hypothesis_result() for _ in range(50)]

        start = time.perf_counter()
        for r in results:
            d = r.model_dump()
            HypothesisResult.model_validate(d)
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 100, f"Serialization roundtrip took {elapsed_ms:.1f}ms (limit: 100ms)"

    def test_search_result_creation_1000(self):
        """Create 1000 SearchResult objects under 20ms."""
        start = time.perf_counter()
        _make_search_results(1000)
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 20, f"SearchResult creation took {elapsed_ms:.1f}ms (limit: 20ms)"


@pytest.mark.slow
class TestKeywordCheckPerformance:
    def test_keyword_check_500_texts(self, signal_extractor):
        """Check keywords against 500 text strings under 20ms."""
        from hypothesis_tool.analyzers.signal_extractor import HIGH_CONFIDENCE_KEYWORDS
        texts = [f"Company #{i} uses LLM and RAG for AI chatbot deployment" for i in range(500)]

        start = time.perf_counter()
        for text in texts:
            signal_extractor._check_keywords(text, HIGH_CONFIDENCE_KEYWORDS)
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 20, f"Keyword check took {elapsed_ms:.1f}ms (limit: 20ms)"
