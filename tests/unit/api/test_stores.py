"""Tests for store query endpoints."""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")


class TestFragmentEndpoints:
    """Tests for /api/v1/stores/fragments/* endpoints."""

    def test_fragment_stats(self, client):
        """GET /api/v1/stores/fragments/stats returns statistics."""
        resp = client.get("/api/v1/stores/fragments/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_count"] == 3  # AAPL, MSFT, GOOG
        assert "by_type" in data
        assert "by_validation_status" in data

    def test_list_fragments(self, client):
        """GET /api/v1/stores/fragments returns fragment summaries."""
        resp = client.get("/api/v1/stores/fragments")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 3
        # Should NOT include payload
        for frag in data:
            assert "fragment_id" in frag
            assert "entity" in frag
            assert "payload" not in frag

    def test_list_fragments_filter_entity(self, client):
        """GET /api/v1/stores/fragments?entity=AAPL filters by entity."""
        resp = client.get("/api/v1/stores/fragments?entity=AAPL")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["entity"] == "AAPL"

    def test_list_fragments_limit(self, client):
        """GET /api/v1/stores/fragments?limit=1 respects limit."""
        resp = client.get("/api/v1/stores/fragments?limit=1")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1

    def test_get_fragment_detail(self, client, test_state):
        """GET /api/v1/stores/fragments/{id} returns full detail with payload."""
        frags = test_state.fragment_store.query(entities={"AAPL"}, limit=1)
        assert len(frags) == 1
        fid = frags[0].fragment_id

        resp = client.get(f"/api/v1/stores/fragments/{fid}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["entity"] == "AAPL"
        assert "payload" in data
        assert data["payload"]["test"] is True

    def test_get_fragment_not_found(self, client):
        """GET /api/v1/stores/fragments/{bad_id} returns 404."""
        resp = client.get("/api/v1/stores/fragments/00000000-0000-0000-0000-000000000000")
        assert resp.status_code == 404

    def test_list_fragments_invalid_data_type(self, client):
        """GET /api/v1/stores/fragments?data_type=BOGUS returns 400."""
        resp = client.get("/api/v1/stores/fragments?data_type=BOGUS")
        assert resp.status_code == 400


class TestBeliefEndpoints:
    """Tests for /api/v1/stores/beliefs/* endpoints."""

    def test_belief_stats(self, client):
        """GET /api/v1/stores/beliefs/stats returns statistics."""
        resp = client.get("/api/v1/stores/beliefs/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_count"] == 0  # no beliefs in fixture
        assert data["agents"] == []
        assert data["tickers"] == []

    def test_list_beliefs_empty(self, client):
        """GET /api/v1/stores/beliefs returns empty list when no beliefs."""
        resp = client.get("/api/v1/stores/beliefs")
        assert resp.status_code == 200
        data = resp.json()
        assert data == []
