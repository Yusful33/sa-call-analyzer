"""Tests for the /api/hypothesis-research endpoint."""

import os
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime


@pytest.fixture
def test_client_with_agent(sample_hypothesis_result):
    """Create a TestClient with a mocked hypothesis research agent."""
    mock_agent = MagicMock()
    mock_agent.research = AsyncMock(return_value=(
        sample_hypothesis_result,
        ["Step 1: Research company", "Step 2: Generate hypotheses"],
    ))

    with patch.dict(os.environ, {
        "ANTHROPIC_API_KEY": "test-key",
        "ARIZE_API_KEY": "",
        "ARIZE_SPACE_ID": "",
    }):
        with patch("observability.setup_observability", return_value=None):
            with patch("crew_analyzer.SACallAnalysisCrew", return_value=MagicMock()):
                with patch("gong_mcp_client.GongMCPClient", side_effect=Exception("mock")):
                    mock_bq = MagicMock()
                    with patch.dict("sys.modules", {"bigquery_client": mock_bq}):
                        mock_bq.BigQueryClient = MagicMock(side_effect=Exception("mock"))
                        import importlib
                        import main as main_module
                        importlib.reload(main_module)
                        # Inject our mock agent
                        main_module._hypothesis_agent = mock_agent
                        from fastapi.testclient import TestClient
                        yield TestClient(main_module.app), mock_agent


class TestHypothesisResearchEndpoint:
    def test_valid_request_returns_200(self, test_client_with_agent):
        client, _ = test_client_with_agent
        response = client.post(
            "/api/hypothesis-research",
            json={"company_name": "Acme Corp"},
        )
        assert response.status_code == 200

    def test_response_has_result_and_reasoning(self, test_client_with_agent):
        client, _ = test_client_with_agent
        response = client.post(
            "/api/hypothesis-research",
            json={"company_name": "Acme Corp"},
        )
        data = response.json()
        assert "result" in data
        assert "agent_reasoning" in data
        assert isinstance(data["agent_reasoning"], list)

    def test_short_company_name_returns_400(self, test_client_with_agent):
        client, _ = test_client_with_agent
        response = client.post(
            "/api/hypothesis-research",
            json={"company_name": "A"},
        )
        assert response.status_code == 400

    def test_empty_company_name_returns_422(self, test_client_with_agent):
        """Empty string should still be sent but fail validation."""
        client, _ = test_client_with_agent
        response = client.post(
            "/api/hypothesis-research",
            json={"company_name": ""},
        )
        # Empty string is < 2 chars
        assert response.status_code == 400

    def test_optional_company_domain(self, test_client_with_agent):
        client, _ = test_client_with_agent
        response = client.post(
            "/api/hypothesis-research",
            json={"company_name": "Acme Corp", "company_domain": "acme.com"},
        )
        assert response.status_code == 200

    def test_llm_failure_returns_503(self, test_client_with_agent):
        client, mock_agent = test_client_with_agent
        mock_agent.research = AsyncMock(side_effect=Exception("anthropic API rate limit"))
        response = client.post(
            "/api/hypothesis-research",
            json={"company_name": "Acme Corp"},
        )
        assert response.status_code == 503

    def test_search_failure_returns_503(self, test_client_with_agent):
        client, mock_agent = test_client_with_agent
        mock_agent.research = AsyncMock(side_effect=Exception("brave search failed"))
        response = client.post(
            "/api/hypothesis-research",
            json={"company_name": "Acme Corp"},
        )
        assert response.status_code == 503

    def test_timeout_returns_504(self, test_client_with_agent):
        client, mock_agent = test_client_with_agent
        mock_agent.research = AsyncMock(side_effect=Exception("Request timeout exceeded"))
        response = client.post(
            "/api/hypothesis-research",
            json={"company_name": "Acme Corp"},
        )
        assert response.status_code == 504

    def test_generic_error_returns_500(self, test_client_with_agent):
        client, mock_agent = test_client_with_agent
        mock_agent.research = AsyncMock(side_effect=Exception("Something unexpected"))
        response = client.post(
            "/api/hypothesis-research",
            json={"company_name": "Acme Corp"},
        )
        assert response.status_code == 500
