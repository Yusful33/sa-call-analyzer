"""Tests for call analysis Pydantic model validation."""

import pytest
from pydantic import ValidationError

from models import (
    AnalyzeRequest,
    CallType,
    PainCurrentState,
    StakeholderMap,
    RequiredCapabilities,
    CompetitiveLandscape,
    DiscoveryCriteria,
    PocScopingCriteria,
    AnalysisResult,
    CommandOfMessageScore,
    SAPerformanceMetrics,
    RecapSlideData,
    ProspectOverviewRequest,
)


class TestAnalyzeRequest:
    def test_transcript_only(self):
        req = AnalyzeRequest(transcript="Hello, welcome to the call")
        assert req.transcript is not None
        assert req.gong_url is None

    def test_gong_url_only(self):
        req = AnalyzeRequest(gong_url="https://app.gong.io/call?id=123")
        assert req.gong_url is not None
        assert req.transcript is None

    def test_both_raises(self):
        with pytest.raises(ValueError, match="not both"):
            AnalyzeRequest(transcript="text", gong_url="https://gong.io/call")

    def test_neither_raises(self):
        with pytest.raises(ValueError, match="must be provided"):
            AnalyzeRequest()


class TestCallType:
    def test_enum_values(self):
        assert CallType.DISCOVERY == "discovery"
        assert CallType.POC_SCOPING == "poc_scoping"
        assert CallType.MIXED == "mixed"
        assert CallType.UNCLEAR == "unclear"


class TestPainCurrentState:
    def test_all_false_score_zero(self):
        pcs = PainCurrentState()
        assert pcs.completion_score == 0.0

    def test_all_true_score_hundred(self):
        pcs = PainCurrentState(
            prompt_model_iteration_understood=True,
            debugging_process_documented=True,
            situation_understood=True,
            resolution_attempts_documented=True,
            outcomes_documented=True,
            frequency_quantified=True,
            duration_quantified=True,
            impact_quantified=True,
            people_impact_understood=True,
            process_impact_understood=True,
            technology_impact_understood=True,
            mttd_mttr_quantified=True,
            experiment_time_quantified=True,
        )
        assert pcs.completion_score == 100.0

    def test_partial_score(self):
        pcs = PainCurrentState(situation_understood=True, impact_quantified=True)
        score = pcs.completion_score
        assert 0 < score < 100


class TestDiscoveryCriteria:
    def test_overall_completion_is_average(self):
        dc = DiscoveryCriteria()
        assert dc.overall_completion_score == 0.0

    def test_partial_completion(self):
        dc = DiscoveryCriteria(
            pain_current_state=PainCurrentState(situation_understood=True),
            stakeholder_map=StakeholderMap(technical_champion_identified=True),
        )
        assert dc.overall_completion_score > 0.0


class TestPocScopingCriteria:
    def test_overall_completion_is_average_of_five(self):
        psc = PocScopingCriteria()
        assert psc.overall_completion_score == 0.0


class TestAnalysisResult:
    def test_creation(self):
        result = AnalysisResult(
            call_summary="Good discovery call",
            overall_score=7.5,
            command_scores=CommandOfMessageScore(
                problem_identification=8.0,
                differentiation=7.0,
                proof_evidence=6.5,
                required_capabilities=7.5,
            ),
            sa_metrics=SAPerformanceMetrics(),
            top_insights=[],
            strengths=["Good questioning"],
            improvement_areas=["Ask more follow-ups"],
            key_moments=[{"timestamp": "5:30", "description": "Key pain uncovered"}],
        )
        assert result.overall_score == 7.5
        assert result.command_scores.problem_identification == 8.0

    def test_default_scores(self):
        result = AnalysisResult(
            call_summary="Test",
            command_scores=CommandOfMessageScore(),
            sa_metrics=SAPerformanceMetrics(),
            top_insights=[],
            strengths=[],
            improvement_areas=[],
            key_moments=[],
        )
        assert result.overall_score == 7.0
        assert result.command_scores.problem_identification == 7.0
        assert result.sa_metrics.technical_depth == 7.0


class TestRecapSlideData:
    def test_creation(self):
        rsd = RecapSlideData(
            customer_name="Acme",
            call_date="2025-01-15",
            key_initiatives=["Deploy LLM pipeline"],
            challenges=["No observability"],
        )
        assert rsd.customer_name == "Acme"
        assert len(rsd.key_initiatives) == 1

    def test_defaults(self):
        rsd = RecapSlideData()
        assert rsd.customer_name == ""
        assert rsd.key_initiatives == []


class TestProspectOverviewRequest:
    def test_account_name_only(self):
        req = ProspectOverviewRequest(account_name="Acme Corp")
        assert req.account_name == "Acme Corp"

    def test_domain_only(self):
        req = ProspectOverviewRequest(domain="acme.com")
        assert req.domain == "acme.com"

    def test_sfdc_id_only(self):
        req = ProspectOverviewRequest(sfdc_account_id="001ABC")
        assert req.sfdc_account_id == "001ABC"

    def test_none_raises(self):
        with pytest.raises(ValueError, match="At least one"):
            ProspectOverviewRequest()
