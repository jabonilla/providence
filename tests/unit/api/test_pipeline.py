"""Tests for pipeline execution and run history endpoints."""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")


class TestRunHistory:
    """Tests for /api/v1/pipeline/runs/* endpoints."""

    def test_list_runs(self, client):
        """GET /api/v1/pipeline/runs returns run history."""
        resp = client.get("/api/v1/pipeline/runs")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 3  # 2 MAIN + 1 EXIT in fixture

    def test_list_runs_filter_loop_type(self, client):
        """GET /api/v1/pipeline/runs?loop_type=MAIN filters correctly."""
        resp = client.get("/api/v1/pipeline/runs?loop_type=MAIN")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
        assert all(r["loop_type"] == "MAIN" for r in data)

    def test_list_runs_filter_status(self, client):
        """GET /api/v1/pipeline/runs?status=SUCCEEDED filters by status."""
        resp = client.get("/api/v1/pipeline/runs?status=SUCCEEDED")
        assert resp.status_code == 200
        data = resp.json()
        assert all(r["status"] == "SUCCEEDED" for r in data)

    def test_list_runs_invalid_status(self, client):
        """GET /api/v1/pipeline/runs?status=BOGUS returns 400."""
        resp = client.get("/api/v1/pipeline/runs?status=BOGUS")
        assert resp.status_code == 400

    def test_list_runs_limit(self, client):
        """GET /api/v1/pipeline/runs?limit=1 respects limit."""
        resp = client.get("/api/v1/pipeline/runs?limit=1")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1

    def test_get_latest_run(self, client):
        """GET /api/v1/pipeline/runs/latest returns most recent run."""
        resp = client.get("/api/v1/pipeline/runs/latest")
        assert resp.status_code == 200
        data = resp.json()
        assert "run_id" in data
        assert "stage_results" in data

    def test_get_run_by_id(self, client, test_state):
        """GET /api/v1/pipeline/runs/{run_id} returns specific run."""
        # Get a known run_id from the store
        latest = test_state.run_store.get_latest()
        assert latest is not None

        resp = client.get(f"/api/v1/pipeline/runs/{latest.run_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["run_id"] == str(latest.run_id)

    def test_get_run_not_found(self, client):
        """GET /api/v1/pipeline/runs/{bad_id} returns 404."""
        resp = client.get("/api/v1/pipeline/runs/00000000-0000-0000-0000-000000000000")
        assert resp.status_code == 404


class TestPipelineStats:
    """Tests for /api/v1/pipeline/stats endpoint."""

    def test_pipeline_stats(self, client):
        """GET /api/v1/pipeline/stats returns execution statistics."""
        resp = client.get("/api/v1/pipeline/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_count"] == 3
        assert "by_loop_type" in data
        assert data["by_loop_type"]["MAIN"] == 2
        assert data["by_loop_type"]["EXIT"] == 1
        assert "success_rate" in data
        assert "success_rate_by_loop" in data


class TestTriggerRun:
    """Tests for pipeline trigger endpoints (no runner = 503)."""

    def test_trigger_run_no_runner(self, client):
        """POST /api/v1/pipeline/run returns 503 when runner not set."""
        resp = client.post("/api/v1/pipeline/run")
        assert resp.status_code == 503

    def test_trigger_learning_no_runner(self, client):
        """POST /api/v1/pipeline/run/learning returns 503 when runner not set."""
        resp = client.post("/api/v1/pipeline/run/learning")
        assert resp.status_code == 503
