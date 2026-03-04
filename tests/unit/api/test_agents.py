"""Tests for agent registry endpoints."""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")


class TestAgentEndpoints:
    """Tests for /api/v1/agents/* endpoints."""

    def test_list_all_agents(self, client):
        """GET /api/v1/agents returns all registered agents."""
        resp = client.get("/api/v1/agents")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 8  # test fixture
        agent_ids = {a["agent_id"] for a in data}
        assert "PERCEPT-PRICE" in agent_ids
        assert "COGNIT-TECHNICAL" in agent_ids

    def test_list_agents_by_subsystem(self, client):
        """GET /api/v1/agents?subsystem=cognition filters correctly."""
        resp = client.get("/api/v1/agents?subsystem=cognition")
        assert resp.status_code == 200
        data = resp.json()
        assert all(a["subsystem"] == "cognition" for a in data)
        agent_ids = {a["agent_id"] for a in data}
        assert "COGNIT-TECHNICAL" in agent_ids
        assert "COGNIT-FUNDAMENTAL" in agent_ids

    def test_list_agents_by_classification(self, client):
        """GET /api/v1/agents?classification=FROZEN filters correctly."""
        resp = client.get("/api/v1/agents?classification=FROZEN")
        assert resp.status_code == 200
        data = resp.json()
        assert all(a["classification"] == "FROZEN" for a in data)
        # COGNIT-FUNDAMENTAL is ADAPTIVE, should NOT appear
        agent_ids = {a["agent_id"] for a in data}
        assert "COGNIT-FUNDAMENTAL" not in agent_ids
        assert "COGNIT-TECHNICAL" in agent_ids

    def test_get_agent(self, client):
        """GET /api/v1/agents/{agent_id} returns agent info."""
        resp = client.get("/api/v1/agents/PERCEPT-PRICE")
        assert resp.status_code == 200
        data = resp.json()
        assert data["agent_id"] == "PERCEPT-PRICE"
        assert data["subsystem"] == "perception"
        assert data["classification"] == "PERCEPTION"

    def test_get_agent_not_found(self, client):
        """GET /api/v1/agents/{bad_id} returns 404."""
        resp = client.get("/api/v1/agents/DOES-NOT-EXIST")
        assert resp.status_code == 404

    def test_get_agent_health(self, client):
        """GET /api/v1/agents/{agent_id}/health returns health status."""
        resp = client.get("/api/v1/agents/COGNIT-TECHNICAL/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["agent_id"] == "COGNIT-TECHNICAL"
        assert "status" in data

    def test_agent_classification(self, client):
        """Verify classification logic for each agent type."""
        resp = client.get("/api/v1/agents/COGNIT-FUNDAMENTAL")
        data = resp.json()
        assert data["classification"] == "ADAPTIVE"
        assert data["subsystem"] == "cognition"

        resp = client.get("/api/v1/agents/DECIDE-OPTIM")
        data = resp.json()
        assert data["classification"] == "FROZEN"
        assert data["subsystem"] == "decision"
