"""Tests for health endpoints."""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")


class TestHealthEndpoints:
    """Tests for /api/v1/health/* endpoints."""

    def test_system_health(self, client):
        """GET /api/v1/health returns system health summary."""
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["system_status"] in ("HEALTHY", "DEGRADED", "CRITICAL", "HALTED")
        assert "agents" in data
        assert "pipeline" in data
        assert data["agents"]["total"] == 8  # test fixture has 8 agents
        assert data["agent_details"] is None  # not requested

    def test_system_health_with_agents(self, client):
        """GET /api/v1/health?include_agents=true includes per-agent details."""
        resp = client.get("/api/v1/health?include_agents=true")
        assert resp.status_code == 200
        data = resp.json()
        assert data["agent_details"] is not None
        assert "PERCEPT-PRICE" in data["agent_details"]
        assert "COGNIT-TECHNICAL" in data["agent_details"]

    def test_readiness_check(self, client):
        """GET /api/v1/health/ready returns ready when agents exist."""
        resp = client.get("/api/v1/health/ready")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ready"

    def test_liveness_check(self, client):
        """GET /api/v1/health/live always returns alive."""
        resp = client.get("/api/v1/health/live")
        assert resp.status_code == 200
        assert resp.json()["status"] == "alive"


class TestRootEndpoint:
    """Tests for root endpoint."""

    def test_root(self, client):
        """GET / returns API info."""
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "Providence API"
        assert "docs" in data
