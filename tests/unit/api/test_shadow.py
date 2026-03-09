"""Tests for shadow mode API endpoints."""

from datetime import datetime, timezone
from uuid import uuid4

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from providence.api.app import create_app
from providence.api.deps import AppState
from providence.schemas.enums import Action, Direction, SystemMode
from providence.schemas.shadow import ShadowRunSummary, ShadowSignal
from providence.services.shadow_execution import ShadowSignalStore


@pytest.fixture
def shadow_store():
    """Create a shadow store with some test signals."""
    store = ShadowSignalStore()
    run_id = uuid4()

    s1 = ShadowSignal(
        run_id=run_id, ticker="AAPL", action=Action.OPEN_LONG,
        direction=Direction.LONG, target_weight=0.05,
        confidence=0.75, approved=True, adjusted_weight=0.05,
        price_at_signal=180.0,
        simulated_entry_price=180.0, simulated_fill_qty=27,
        simulated_notional=5000.0,
    )
    s2 = ShadowSignal(
        run_id=run_id, ticker="MSFT", action=Action.OPEN_LONG,
        direction=Direction.LONG, target_weight=0.04,
        confidence=0.60, approved=True, adjusted_weight=0.04,
        price_at_signal=400.0,
    )
    s3 = ShadowSignal(
        run_id=run_id, ticker="TSLA", action=Action.OPEN_SHORT,
        direction=Direction.SHORT, target_weight=0.03,
        confidence=0.30, approved=False,
        rejection_reasons=["Confidence below minimum"],
    )
    store.append(s1)
    store.append(s2)
    store.append(s3)

    summary = ShadowRunSummary(
        run_id=run_id,
        system_mode=SystemMode.SHADOW,
        total_signals=3,
        approved_signals=2,
        rejected_signals=1,
        long_signals=2,
        short_signals=1,
    )
    store.append_summary(summary)

    return store, run_id, [s1, s2, s3]


@pytest.fixture
def client(shadow_store):
    """Create test client with shadow store."""
    store, _, _ = shadow_store
    state = AppState(shadow_signal_store=store)
    app = create_app(state, enable_auth=False)
    return TestClient(app)


@pytest.fixture
def client_no_shadow():
    """Create test client WITHOUT shadow store."""
    state = AppState(shadow_signal_store=None)
    app = create_app(state, enable_auth=False)
    return TestClient(app)


class TestShadowStats:
    """Tests for GET /api/v1/shadow/stats."""

    def test_get_stats(self, client):
        resp = client.get("/api/v1/shadow/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_signals"] == 3
        assert data["total_runs"] == 1
        assert data["approved_signals"] == 2
        assert data["rejected_signals"] == 1
        assert data["unique_tickers"] == 3

    def test_stats_no_shadow_store(self, client_no_shadow):
        resp = client_no_shadow.get("/api/v1/shadow/stats")
        assert resp.status_code == 404


class TestShadowSignals:
    """Tests for GET /api/v1/shadow/signals."""

    def test_list_all_signals(self, client):
        resp = client.get("/api/v1/shadow/signals")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 3

    def test_filter_by_ticker(self, client):
        resp = client.get("/api/v1/shadow/signals?ticker=AAPL")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["ticker"] == "AAPL"

    def test_filter_approved_only(self, client):
        resp = client.get("/api/v1/shadow/signals?approved_only=true")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
        assert all(s["approved"] for s in data)

    def test_filter_by_run_id(self, client, shadow_store):
        _, run_id, _ = shadow_store
        resp = client.get(f"/api/v1/shadow/signals?run_id={run_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 3

    def test_limit(self, client):
        resp = client.get("/api/v1/shadow/signals?limit=1")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1

    def test_signal_fields(self, client):
        resp = client.get("/api/v1/shadow/signals?ticker=AAPL")
        data = resp.json()[0]
        assert data["action"] == "OPEN_LONG"
        assert data["direction"] == "LONG"
        assert data["confidence"] == 0.75
        assert data["price_at_signal"] == 180.0
        assert data["simulated_fill_qty"] == 27


class TestShadowSignalDetail:
    """Tests for GET /api/v1/shadow/signals/{signal_id}."""

    def test_get_signal_by_id(self, client, shadow_store):
        _, _, signals = shadow_store
        signal_id = signals[0].signal_id
        resp = client.get(f"/api/v1/shadow/signals/{signal_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["ticker"] == "AAPL"

    def test_signal_not_found(self, client):
        resp = client.get(f"/api/v1/shadow/signals/{uuid4()}")
        assert resp.status_code == 404


class TestShadowSummaries:
    """Tests for GET /api/v1/shadow/summaries."""

    def test_list_summaries(self, client):
        resp = client.get("/api/v1/shadow/summaries")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["total_signals"] == 3
        assert data[0]["system_mode"] == "SHADOW"


class TestShadowReport:
    """Tests for GET /api/v1/shadow/report."""

    def test_get_report(self, client):
        resp = client.get("/api/v1/shadow/report")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_signals"] == 3
        assert data["total_approved"] == 2
        assert "phase_b_criteria" in data


class TestShadowBackfill:
    """Tests for POST /api/v1/shadow/backfill."""

    def test_backfill_no_price_client(self, client):
        """Backfill fails gracefully when no price client configured."""
        resp = client.post("/api/v1/shadow/backfill")
        assert resp.status_code == 503
