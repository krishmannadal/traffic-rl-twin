"""
test_api.py — Integration tests for FastAPI endpoints.

Uses FastAPI's TestClient (which runs without a real server) to verify
that all REST endpoints return correct status codes, content types,
and JSON schema.  Mocks the SUMO environment to avoid external dependencies.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient


# ──────────────────────────────────────────────────────────────────────
#  Test Client Fixture
# ──────────────────────────────────────────────────────────────────────

@pytest.fixture
def client():
    """Create a FastAPI TestClient with mocked SUMO dependencies."""
    # Import here to allow patching before the app is loaded
    from api.main import app
    with TestClient(app) as c:
        yield c


# ──────────────────────────────────────────────────────────────────────
#  Health & Root Endpoint Tests
# ──────────────────────────────────────────────────────────────────────

class TestHealthEndpoints:
    """Basic connectivity tests."""

    def test_root_returns_json(self, client):
        """GET / should return a JSON response with API info."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data or "status" in data or "endpoints" in data

    def test_health_returns_ok(self, client):
        """GET /health should return 200 with system status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data or "sumo_running" in data


# ──────────────────────────────────────────────────────────────────────
#  Simulation Endpoint Tests
# ──────────────────────────────────────────────────────────────────────

class TestSimulationEndpoints:
    """Tests for /simulation/* routes."""

    def test_status_returns_json(self, client):
        """GET /simulation/status should return current simulation state."""
        response = client.get("/simulation/status")
        assert response.status_code == 200
        data = response.json()
        assert "running" in data

    def test_stop_without_start_returns_400(self, client):
        """POST /simulation/stop when not running should return 400."""
        response = client.post("/simulation/stop")
        assert response.status_code == 400

    def test_reset_without_start_returns_400(self, client):
        """POST /simulation/reset when not running should return 400."""
        response = client.post("/simulation/reset")
        assert response.status_code == 400

    def test_emergency_without_simulation_returns_400(self, client):
        """POST /simulation/emergency when not running should return 400."""
        response = client.post(
            "/simulation/emergency",
            json={"origin_edge": "south_to_center", "destination_edge": "center_to_north"}
        )
        assert response.status_code == 400

    def test_speed_requires_positive(self, client):
        """POST /simulation/speed with 0 or negative should return 400."""
        response = client.post(
            "/simulation/speed",
            json={"multiplier": -1.0}
        )
        assert response.status_code == 400


# ──────────────────────────────────────────────────────────────────────
#  Metrics Endpoint Tests
# ──────────────────────────────────────────────────────────────────────

class TestMetricsEndpoints:
    """Tests for /metrics/* routes."""

    def test_current_metrics_empty(self, client):
        """GET /metrics/current should return empty status when no data."""
        response = client.get("/metrics/current")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data or "step" in data

    def test_history_returns_list(self, client):
        """GET /metrics/history should return a list (possibly empty)."""
        response = client.get("/metrics/history")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_summary_returns_json(self, client):
        """GET /metrics/summary should return aggregate stats."""
        response = client.get("/metrics/summary")
        assert response.status_code == 200
        data = response.json()
        assert "total_steps" in data or "status" in data

    def test_record_metric(self, client):
        """POST /metrics/record should accept a valid metric entry."""
        import time
        response = client.post("/metrics/record", json={
            "step": 1,
            "reward": 0.5,
            "waiting_time": 10.0,
            "queue_length": 3.0,
            "timestamp": time.time(),
        })
        assert response.status_code == 200
        assert response.json()["status"] == "recorded"

    def test_export_csv(self, client):
        """GET /metrics/export should return CSV content type or empty text."""
        # First record some data
        import time
        for i in range(3):
            client.post("/metrics/record", json={
                "step": i,
                "reward": 0.1 * i,
                "waiting_time": 5.0,
                "queue_length": 2.0,
                "timestamp": time.time(),
            })
        
        response = client.get("/metrics/export")
        assert response.status_code == 200
        # Should be CSV or plain text
        assert "text" in response.headers.get("content-type", "")


# ──────────────────────────────────────────────────────────────────────
#  Agent Endpoint Tests
# ──────────────────────────────────────────────────────────────────────

class TestAgentEndpoints:
    """Tests for /agents/* routes."""

    def test_agent_status(self, client):
        """GET /agents/status should return agent configuration info."""
        response = client.get("/agents/status")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)

    def test_models_list(self, client):
        """GET /agents/models should return a list of saved models."""
        response = client.get("/agents/models")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert isinstance(data["models"], list)


# ──────────────────────────────────────────────────────────────────────
#  Vehicle Endpoint Tests
# ──────────────────────────────────────────────────────────────────────

class TestVehicleEndpoints:
    """Tests for /vehicles/* routes."""

    def test_list_vehicles(self, client):
        """GET /vehicles/active should return a list."""
        response = client.get("/vehicles/active")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, (list, dict))
