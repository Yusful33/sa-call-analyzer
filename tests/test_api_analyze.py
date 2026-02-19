"""Tests for the /api/analyze endpoint."""

import os
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from models import AnalysisResult, CommandOfMessageScore, SAPerformanceMetrics


def _make_mock_analysis_result():
    """Create a mock AnalysisResult dict."""
    return AnalysisResult(
        call_summary="Good discovery call covering LLM observability needs",
        overall_score=7.5,
        command_scores=CommandOfMessageScore(
            problem_identification=8.0,
            differentiation=7.0,
            proof_evidence=6.5,
            required_capabilities=7.5,
        ),
        sa_metrics=SAPerformanceMetrics(
            technical_depth=7.5,
            discovery_quality=8.0,
            active_listening=7.0,
            value_articulation=6.5,
        ),
        top_insights=[],
        strengths=["Good questioning technique"],
        improvement_areas=["Follow up on pain quantification"],
        key_moments=[{"timestamp": "5:30", "description": "Uncovered key pain"}],
    )


@pytest.fixture
def test_client_analyze():
    """Create a TestClient with a mocked CrewAI analyzer."""
    mock_analyzer = MagicMock()
    mock_result = _make_mock_analysis_result()
    mock_analyzer.analyze_call = MagicMock(return_value=mock_result)

    with patch.dict(os.environ, {
        "ANTHROPIC_API_KEY": "test-key",
        "ARIZE_API_KEY": "",
        "ARIZE_SPACE_ID": "",
    }):
        with patch("observability.setup_observability", return_value=None):
            with patch("crew_analyzer.SACallAnalysisCrew", return_value=mock_analyzer):
                with patch("gong_mcp_client.GongMCPClient", side_effect=Exception("mock")):
                    mock_bq = MagicMock()
                    with patch.dict("sys.modules", {"bigquery_client": mock_bq}):
                        mock_bq.BigQueryClient = MagicMock(side_effect=Exception("mock"))
                        import importlib
                        import main as main_module
                        importlib.reload(main_module)
                        # Replace the analyzer with our mock
                        main_module.analyzer = mock_analyzer
                        from fastapi.testclient import TestClient
                        yield TestClient(main_module.app)


class TestAnalyzeEndpoint:
    def test_valid_transcript_returns_200(self, test_client_analyze):
        response = test_client_analyze.post(
            "/api/analyze",
            json={"transcript": "Speaker 1: Hello, welcome to the call. Let's discuss your LLM pipeline."},
        )
        assert response.status_code == 200

    def test_empty_transcript_returns_400(self, test_client_analyze):
        response = test_client_analyze.post(
            "/api/analyze",
            json={"transcript": "   "},
        )
        assert response.status_code == 400

    def test_response_has_analysis_fields(self, test_client_analyze):
        response = test_client_analyze.post(
            "/api/analyze",
            json={"transcript": "Speaker 1: Let's discuss your AI needs."},
        )
        data = response.json()
        assert "call_summary" in data
        assert "overall_score" in data
        assert "command_scores" in data
        assert "sa_metrics" in data

    def test_gong_url_without_client_returns_503(self, test_client_analyze):
        """When gong_client is None, gong_url requests should fail."""
        response = test_client_analyze.post(
            "/api/analyze",
            json={"gong_url": "https://app.gong.io/call?id=12345"},
        )
        assert response.status_code == 503

    def test_missing_both_fields_returns_422(self, test_client_analyze):
        """Omitting both transcript and gong_url should fail Pydantic validation."""
        response = test_client_analyze.post(
            "/api/analyze",
            json={},
        )
        assert response.status_code == 422
