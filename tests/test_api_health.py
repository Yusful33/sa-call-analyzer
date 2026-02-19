"""Tests for the /health endpoint."""

import os
import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture
def test_client():
    """Create a FastAPI TestClient with all external services mocked."""
    # Patch module-level side effects before importing main
    with patch.dict(os.environ, {
        "ANTHROPIC_API_KEY": "test-key",
        "ARIZE_API_KEY": "",
        "ARIZE_SPACE_ID": "",
    }):
        with patch("observability.setup_observability", return_value=None):
            with patch("crew_analyzer.SACallAnalysisCrew", return_value=MagicMock()):
                with patch("gong_mcp_client.GongMCPClient", side_effect=Exception("mock")):
                    # Mock BigQuery import inside main.py
                    mock_bq = MagicMock()
                    with patch.dict("sys.modules", {"bigquery_client": mock_bq}):
                        mock_bq.BigQueryClient = MagicMock(side_effect=Exception("mock"))
                        import importlib
                        import main as main_module
                        importlib.reload(main_module)
                        from fastapi.testclient import TestClient
                        yield TestClient(main_module.app)


class TestHealthEndpoint:
    def test_returns_200(self, test_client):
        response = test_client.get("/health")
        assert response.status_code == 200

    def test_response_structure(self, test_client):
        response = test_client.get("/health")
        data = response.json()
        assert "status" in data
        assert "api_key_configured" in data

    def test_status_is_healthy(self, test_client):
        response = test_client.get("/health")
        assert response.json()["status"] == "healthy"

    def test_api_key_configured_true(self, test_client):
        """When ANTHROPIC_API_KEY is set, api_key_configured should be True."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            response = test_client.get("/health")
            assert response.json()["api_key_configured"] is True

    def test_api_key_configured_false(self, test_client):
        """When ANTHROPIC_API_KEY is empty, api_key_configured should be False."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""}):
            response = test_client.get("/health")
            assert response.json()["api_key_configured"] is False
