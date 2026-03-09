"""Tests for ShadowExecutionService and ShadowSignalStore."""

import json
import tempfile
from pathlib import Path
from uuid import uuid4

import pytest

from providence.schemas.enums import Action, Direction, SystemMode
from providence.schemas.shadow import ShadowPerformanceReport, ShadowRunSummary, ShadowSignal
from providence.services.shadow_execution import ShadowExecutionService, ShadowSignalStore


# ---------------------------------------------------------------------------
# ShadowSignalStore tests
# ---------------------------------------------------------------------------

class TestShadowSignalStore:
    """Tests for ShadowSignalStore (append-only, indexed, persistent)."""

    def test_append_and_retrieve(self):
        store = ShadowSignalStore()
        run_id = uuid4()
        signal = ShadowSignal(
            run_id=run_id,
            ticker="AAPL",
            action=Action.OPEN_LONG,
            direction=Direction.LONG,
            target_weight=0.05,
            confidence=0.75,
            approved=True,
            adjusted_weight=0.05,
        )
        assert store.append(signal) is True
        assert store.count == 1

    def test_deduplication(self):
        store = ShadowSignalStore()
        run_id = uuid4()
        signal = ShadowSignal(
            run_id=run_id,
            ticker="AAPL",
            action=Action.OPEN_LONG,
            direction=Direction.LONG,
            target_weight=0.05,
            confidence=0.75,
            approved=True,
        )
        assert store.append(signal) is True
        assert store.append(signal) is False  # Duplicate
        assert store.count == 1

    def test_index_by_run(self):
        store = ShadowSignalStore()
        run1 = uuid4()
        run2 = uuid4()

        s1 = ShadowSignal(run_id=run1, ticker="AAPL", action=Action.OPEN_LONG,
                           direction=Direction.LONG, target_weight=0.05,
                           confidence=0.7, approved=True)
        s2 = ShadowSignal(run_id=run1, ticker="MSFT", action=Action.OPEN_LONG,
                           direction=Direction.LONG, target_weight=0.04,
                           confidence=0.6, approved=True)
        s3 = ShadowSignal(run_id=run2, ticker="AAPL", action=Action.CLOSE,
                           direction=Direction.NEUTRAL, target_weight=0.0,
                           confidence=0.5, approved=False)

        store.append(s1)
        store.append(s2)
        store.append(s3)

        assert len(store.get_by_run(run1)) == 2
        assert len(store.get_by_run(run2)) == 1

    def test_index_by_ticker(self):
        store = ShadowSignalStore()
        run_id = uuid4()

        s1 = ShadowSignal(run_id=run_id, ticker="AAPL", action=Action.OPEN_LONG,
                           direction=Direction.LONG, target_weight=0.05,
                           confidence=0.7, approved=True)
        s2 = ShadowSignal(run_id=run_id, ticker="AAPL", action=Action.ADJUST,
                           direction=Direction.LONG, target_weight=0.03,
                           confidence=0.6, approved=True)

        store.append(s1)
        store.append(s2)

        assert len(store.get_by_ticker("AAPL")) == 2
        assert len(store.get_by_ticker("MSFT")) == 0

    def test_get_all_newest_first(self):
        store = ShadowSignalStore()
        run_id = uuid4()

        s1 = ShadowSignal(run_id=run_id, ticker="AAPL", action=Action.OPEN_LONG,
                           direction=Direction.LONG, target_weight=0.05,
                           confidence=0.7, approved=True)
        s2 = ShadowSignal(run_id=run_id, ticker="MSFT", action=Action.OPEN_SHORT,
                           direction=Direction.SHORT, target_weight=0.04,
                           confidence=0.6, approved=True)

        store.append(s1)
        store.append(s2)

        all_signals = store.get_all()
        assert len(all_signals) == 2
        assert all_signals[0].ticker == "MSFT"  # Newest first

    def test_persistence_roundtrip(self, tmp_path: Path):
        """Verify signals persist to JSONL and reload correctly."""
        signal_path = tmp_path / "shadow_signals.jsonl"

        # Write signals
        store1 = ShadowSignalStore(persist_path=signal_path)
        run_id = uuid4()
        s1 = ShadowSignal(run_id=run_id, ticker="AAPL", action=Action.OPEN_LONG,
                           direction=Direction.LONG, target_weight=0.05,
                           confidence=0.75, approved=True, adjusted_weight=0.05)
        s2 = ShadowSignal(run_id=run_id, ticker="GOOGL", action=Action.OPEN_SHORT,
                           direction=Direction.SHORT, target_weight=0.04,
                           confidence=0.65, approved=True, adjusted_weight=0.04)
        store1.append(s1)
        store1.append(s2)
        assert store1.count == 2

        # Reload from disk
        store2 = ShadowSignalStore(persist_path=signal_path)
        assert store2.count == 2
        assert len(store2.get_by_ticker("AAPL")) == 1
        assert len(store2.get_by_ticker("GOOGL")) == 1

    def test_stats(self):
        store = ShadowSignalStore()
        run_id = uuid4()

        s1 = ShadowSignal(run_id=run_id, ticker="AAPL", action=Action.OPEN_LONG,
                           direction=Direction.LONG, target_weight=0.05,
                           confidence=0.7, approved=True)
        s2 = ShadowSignal(run_id=run_id, ticker="MSFT", action=Action.OPEN_LONG,
                           direction=Direction.LONG, target_weight=0.04,
                           confidence=0.3, approved=False,
                           rejection_reasons=["Confidence below minimum"])

        store.append(s1)
        store.append(s2)

        stats = store.stats()
        assert stats["total_signals"] == 2
        assert stats["approved_signals"] == 1
        assert stats["rejected_signals"] == 1
        assert stats["unique_tickers"] == 2
        assert stats["total_runs"] == 1

    def test_summaries(self):
        store = ShadowSignalStore()
        summary = ShadowRunSummary(
            run_id=uuid4(),
            system_mode=SystemMode.SHADOW,
            total_signals=5,
            approved_signals=3,
            rejected_signals=2,
        )
        store.append_summary(summary)
        summaries = store.get_summaries()
        assert len(summaries) == 1
        assert summaries[0].total_signals == 5


# ---------------------------------------------------------------------------
# ShadowExecutionService tests
# ---------------------------------------------------------------------------

class TestShadowExecutionService:
    """Tests for ShadowExecutionService.record_signals()."""

    def _make_validated_proposal(
        self,
        positions: list[dict] | None = None,
    ) -> dict:
        """Create a mock ValidatedProposal dict."""
        if positions is None:
            positions = [
                {
                    "ticker": "AAPL",
                    "action": "OPEN_LONG",
                    "direction": "LONG",
                    "target_weight": 0.06,
                    "confidence": 0.78,
                    "approved": True,
                    "rejection_reasons": [],
                    "adjusted_weight": 0.06,
                },
                {
                    "ticker": "TSLA",
                    "action": "OPEN_SHORT",
                    "direction": "SHORT",
                    "target_weight": 0.04,
                    "confidence": 0.55,
                    "approved": True,
                    "rejection_reasons": [],
                    "adjusted_weight": 0.04,
                },
                {
                    "ticker": "XOM",
                    "action": "OPEN_LONG",
                    "direction": "LONG",
                    "target_weight": 0.10,
                    "confidence": 0.15,
                    "approved": False,
                    "rejection_reasons": ["Confidence 0.15 below minimum 0.20"],
                    "adjusted_weight": 0.0,
                },
            ]
        return {
            "agent_id": "EXEC-VALIDATE",
            "approved_count": sum(1 for p in positions if p["approved"]),
            "rejected_count": sum(1 for p in positions if not p["approved"]),
            "risk_mode_applied": "NORMAL",
            "results": positions,
        }

    def test_record_signals_basic(self):
        store = ShadowSignalStore()
        svc = ShadowExecutionService(store)
        run_id = uuid4()

        proposal = self._make_validated_proposal()
        summary = svc.record_signals(run_id, proposal)

        assert summary.total_signals == 3
        assert summary.approved_signals == 2
        assert summary.rejected_signals == 1
        assert summary.long_signals == 2
        assert summary.short_signals == 1
        assert summary.system_mode == SystemMode.SHADOW
        assert store.count == 3

    def test_record_signals_with_price(self):
        store = ShadowSignalStore()
        svc = ShadowExecutionService(store)
        run_id = uuid4()

        proposal = self._make_validated_proposal()
        metadata = {
            "current_prices": {"AAPL": 185.50, "TSLA": 245.00},
            "shadow_equity": 100_000.0,
        }

        summary = svc.record_signals(run_id, proposal, metadata=metadata)
        signals = store.get_by_ticker("AAPL")
        assert len(signals) == 1
        assert signals[0].price_at_signal == 185.50
        assert signals[0].simulated_entry_price == 185.50
        assert signals[0].simulated_notional == 6000.0  # 100K * 0.06
        assert signals[0].simulated_fill_qty == 32  # int(6000 / 185.50)

    def test_record_signals_empty_proposal(self):
        store = ShadowSignalStore()
        svc = ShadowExecutionService(store)
        run_id = uuid4()

        summary = svc.record_signals(run_id, {"results": []})
        assert summary.total_signals == 0
        assert store.count == 0

    def test_record_signals_regime_state(self):
        store = ShadowSignalStore()
        svc = ShadowExecutionService(store)
        run_id = uuid4()

        proposal = self._make_validated_proposal()
        regime = {
            "system_risk_mode": "CAUTIOUS",
            "statistical_regime": "HIGH_VOL_MEAN_REVERTING",
        }

        summary = svc.record_signals(run_id, proposal, regime_state=regime)
        assert summary.risk_mode == "CAUTIOUS"
        assert summary.regime_state == "HIGH_VOL_MEAN_REVERTING"

    def test_record_signals_invalid_action_graceful(self):
        """Signals with invalid action strings should fall back to CLOSE."""
        store = ShadowSignalStore()
        svc = ShadowExecutionService(store)
        run_id = uuid4()

        proposal = self._make_validated_proposal(positions=[{
            "ticker": "AAPL",
            "action": "INVALID_ACTION",
            "direction": "LONG",
            "target_weight": 0.05,
            "confidence": 0.5,
            "approved": True,
            "rejection_reasons": [],
            "adjusted_weight": 0.05,
        }])

        summary = svc.record_signals(run_id, proposal)
        assert summary.total_signals == 1
        signals = store.get_all()
        assert signals[0].action == Action.CLOSE  # Fallback

    def test_store_property(self):
        store = ShadowSignalStore()
        svc = ShadowExecutionService(store)
        assert svc.store is store


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------

class TestShadowSchemas:
    """Test shadow mode Pydantic schemas."""

    def test_shadow_signal_frozen(self):
        signal = ShadowSignal(
            run_id=uuid4(),
            ticker="AAPL",
            action=Action.OPEN_LONG,
            direction=Direction.LONG,
            target_weight=0.05,
            confidence=0.75,
            approved=True,
        )
        with pytest.raises(Exception):
            signal.ticker = "MSFT"  # Frozen

    def test_shadow_signal_serialization(self):
        signal = ShadowSignal(
            run_id=uuid4(),
            ticker="AAPL",
            action=Action.OPEN_LONG,
            direction=Direction.LONG,
            target_weight=0.05,
            confidence=0.75,
            approved=True,
        )
        data = json.loads(signal.model_dump_json())
        assert data["ticker"] == "AAPL"
        assert data["action"] == "OPEN_LONG"
        assert data["approved"] is True

        # Roundtrip
        restored = ShadowSignal.model_validate(data)
        assert restored.signal_id == signal.signal_id

    def test_shadow_performance_report_defaults(self):
        report = ShadowPerformanceReport()
        assert report.total_runs == 0
        assert report.ready_for_paper_trading is False

    def test_shadow_run_summary(self):
        summary = ShadowRunSummary(
            run_id=uuid4(),
            system_mode=SystemMode.SHADOW,
            total_signals=10,
            approved_signals=7,
            rejected_signals=3,
            long_signals=5,
            short_signals=2,
        )
        assert summary.total_signals == 10
        assert summary.system_mode == SystemMode.SHADOW
