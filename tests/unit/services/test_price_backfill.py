"""Tests for PriceBackfillService — realized return computation."""

import asyncio
from datetime import datetime, timedelta, timezone, date
from uuid import uuid4

import pytest

from providence.schemas.enums import Action, Direction
from providence.schemas.shadow import ShadowSignal
from providence.services.price_backfill import (
    PriceBackfillService,
    advance_trading_days,
    _is_trading_day,
)
from providence.services.shadow_execution import ShadowSignalStore


# ---------------------------------------------------------------------------
# Trading day utilities
# ---------------------------------------------------------------------------

class TestTradingDayUtils:
    """Tests for trading day calculation functions."""

    def test_weekday_is_trading_day(self):
        # Monday 2026-03-09
        assert _is_trading_day(date(2026, 3, 9)) is True

    def test_saturday_is_not_trading_day(self):
        assert _is_trading_day(date(2026, 3, 7)) is False

    def test_sunday_is_not_trading_day(self):
        assert _is_trading_day(date(2026, 3, 8)) is False

    def test_holiday_is_not_trading_day(self):
        # Christmas 2026
        assert _is_trading_day(date(2026, 12, 25)) is False

    def test_advance_1_trading_day(self):
        # Monday → Tuesday
        result = advance_trading_days(date(2026, 3, 9), 1)
        assert result == date(2026, 3, 10)

    def test_advance_over_weekend(self):
        # Friday + 1 trading day = Monday
        result = advance_trading_days(date(2026, 3, 6), 1)
        assert result == date(2026, 3, 9)

    def test_advance_5_trading_days(self):
        # Monday + 5 trading days = following Monday
        result = advance_trading_days(date(2026, 3, 9), 5)
        assert result == date(2026, 3, 16)

    def test_advance_20_trading_days(self):
        # Roughly 4 calendar weeks
        result = advance_trading_days(date(2026, 3, 9), 20)
        # 20 trading days from March 9 = ~April 6 (Monday)
        assert result.weekday() < 5  # Must be a weekday
        assert (result - date(2026, 3, 9)).days >= 27  # At least 27 calendar days


# ---------------------------------------------------------------------------
# Mock price client
# ---------------------------------------------------------------------------

class MockPriceClient:
    """Minimal mock for PolygonClient.get_daily_bars()."""

    def __init__(self, prices: dict[tuple[str, str], float]):
        """prices: mapping of (ticker, date_str) → close price."""
        self._prices = prices
        self.call_count = 0

    async def get_daily_bars(self, ticker: str, date_str: str) -> dict:
        self.call_count += 1
        price = self._prices.get((ticker, date_str))
        if price is not None:
            return {"results": [{"c": price}]}
        return {"results": []}


# ---------------------------------------------------------------------------
# ShadowSignalStore.update_signal tests
# ---------------------------------------------------------------------------

class TestSignalStoreUpdate:
    """Tests for ShadowSignalStore.update_signal()."""

    def test_update_replaces_signal(self):
        store = ShadowSignalStore()
        run_id = uuid4()
        signal = ShadowSignal(
            run_id=run_id, ticker="AAPL", action=Action.OPEN_LONG,
            direction=Direction.LONG, target_weight=0.05,
            confidence=0.7, approved=True, price_at_signal=180.0,
        )
        store.append(signal)

        # Create updated version with realized return
        updated = ShadowSignal(
            signal_id=signal.signal_id,
            run_id=run_id, ticker="AAPL", action=Action.OPEN_LONG,
            direction=Direction.LONG, target_weight=0.05,
            confidence=0.7, approved=True, price_at_signal=180.0,
            price_1d_later=182.0,
            realized_return_1d=(182.0 - 180.0) / 180.0,
        )

        assert store.update_signal(signal.signal_id, updated) is True
        # Verify the store has the updated version
        retrieved = store.get_by_ticker("AAPL")
        assert len(retrieved) == 1
        assert retrieved[0].price_1d_later == 182.0
        assert retrieved[0].realized_return_1d is not None

    def test_update_nonexistent_returns_false(self):
        store = ShadowSignalStore()
        fake_id = uuid4()
        signal = ShadowSignal(
            signal_id=fake_id,
            run_id=uuid4(), ticker="AAPL", action=Action.OPEN_LONG,
            direction=Direction.LONG, target_weight=0.05,
            confidence=0.7, approved=True,
        )
        assert store.update_signal(fake_id, signal) is False

    def test_update_mismatched_id_returns_false(self):
        store = ShadowSignalStore()
        run_id = uuid4()
        signal = ShadowSignal(
            run_id=run_id, ticker="AAPL", action=Action.OPEN_LONG,
            direction=Direction.LONG, target_weight=0.05,
            confidence=0.7, approved=True,
        )
        store.append(signal)

        # Try to update with wrong signal_id
        wrong_signal = ShadowSignal(
            signal_id=uuid4(),  # Different ID
            run_id=run_id, ticker="AAPL", action=Action.OPEN_LONG,
            direction=Direction.LONG, target_weight=0.05,
            confidence=0.7, approved=True,
        )
        assert store.update_signal(signal.signal_id, wrong_signal) is False

    def test_update_persists(self, tmp_path):
        """Verify update_signal re-persists to JSONL."""
        path = tmp_path / "signals.jsonl"
        store = ShadowSignalStore(persist_path=path)
        run_id = uuid4()
        signal = ShadowSignal(
            run_id=run_id, ticker="AAPL", action=Action.OPEN_LONG,
            direction=Direction.LONG, target_weight=0.05,
            confidence=0.7, approved=True, price_at_signal=180.0,
        )
        store.append(signal)

        updated = ShadowSignal(
            signal_id=signal.signal_id,
            run_id=run_id, ticker="AAPL", action=Action.OPEN_LONG,
            direction=Direction.LONG, target_weight=0.05,
            confidence=0.7, approved=True, price_at_signal=180.0,
            realized_return_1d=0.011,
        )
        store.update_signal(signal.signal_id, updated)

        # Reload from disk
        store2 = ShadowSignalStore(persist_path=path)
        assert store2.count == 1
        reloaded = store2.get_by_ticker("AAPL")[0]
        assert reloaded.realized_return_1d == 0.011


# ---------------------------------------------------------------------------
# PriceBackfillService tests
# ---------------------------------------------------------------------------

class TestPriceBackfillService:
    """Tests for PriceBackfillService.run()."""

    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def _make_signal(
        self,
        ticker: str = "AAPL",
        price: float = 180.0,
        approved: bool = True,
        age_days: int = 30,
        **kwargs,
    ) -> ShadowSignal:
        """Create a signal that is old enough for backfill."""
        return ShadowSignal(
            run_id=uuid4(),
            ticker=ticker,
            action=Action.OPEN_LONG,
            direction=Direction.LONG,
            target_weight=0.05,
            confidence=0.7,
            approved=approved,
            adjusted_weight=0.05,
            price_at_signal=price,
            timestamp=datetime.now(timezone.utc) - timedelta(days=age_days),
            **kwargs,
        )

    def test_backfill_basic(self):
        """Signals get realized returns after backfill."""
        store = ShadowSignalStore()
        signal = self._make_signal(ticker="AAPL", price=180.0)
        store.append(signal)

        signal_date = signal.timestamp.date()
        t1 = advance_trading_days(signal_date, 1)
        t5 = advance_trading_days(signal_date, 5)
        t20 = advance_trading_days(signal_date, 20)

        prices = {
            ("AAPL", t1.isoformat()): 182.0,
            ("AAPL", t5.isoformat()): 185.0,
            ("AAPL", t20.isoformat()): 190.0,
        }
        client = MockPriceClient(prices)
        svc = PriceBackfillService(store, client, inter_request_delay=0)

        result = self._run(svc.run())
        assert result["updated"] == 1
        assert result["errors"] == 0

        # Verify the signal was updated
        updated_signal = store.get_by_ticker("AAPL")[0]
        assert updated_signal.realized_return_1d is not None
        assert abs(updated_signal.realized_return_1d - (182.0 - 180.0) / 180.0) < 1e-6

    def test_backfill_skips_unapproved(self):
        """Unapproved signals are not backfilled."""
        store = ShadowSignalStore()
        signal = self._make_signal(approved=False)
        store.append(signal)

        client = MockPriceClient({})
        svc = PriceBackfillService(store, client, inter_request_delay=0)

        result = self._run(svc.run())
        assert result["processed"] == 0

    def test_backfill_skips_no_entry_price(self):
        """Signals without entry price are skipped."""
        store = ShadowSignalStore()
        signal = self._make_signal(price=None)
        store.append(signal)

        client = MockPriceClient({})
        svc = PriceBackfillService(store, client, inter_request_delay=0)

        result = self._run(svc.run())
        assert result["processed"] == 0

    def test_backfill_skips_already_computed(self):
        """Signals with all returns already computed are skipped."""
        store = ShadowSignalStore()
        signal = self._make_signal(
            realized_return_1d=0.01,
            realized_return_5d=0.02,
            realized_return_20d=0.05,
        )
        store.append(signal)

        client = MockPriceClient({})
        svc = PriceBackfillService(store, client, inter_request_delay=0)

        result = self._run(svc.run())
        assert result["processed"] == 0

    def test_backfill_skips_too_recent(self):
        """Signals less than 2 days old are skipped."""
        store = ShadowSignalStore()
        signal = self._make_signal(age_days=0)  # Just created
        store.append(signal)

        client = MockPriceClient({})
        svc = PriceBackfillService(store, client, inter_request_delay=0)

        result = self._run(svc.run())
        assert result["processed"] == 0

    def test_backfill_partial_prices(self):
        """If only some horizons have prices, only those are filled."""
        store = ShadowSignalStore()
        signal = self._make_signal(ticker="AAPL", price=180.0, age_days=5)
        store.append(signal)

        signal_date = signal.timestamp.date()
        t1 = advance_trading_days(signal_date, 1)
        # Only provide 1d price, not 5d or 20d
        prices = {("AAPL", t1.isoformat()): 182.0}
        client = MockPriceClient(prices)
        svc = PriceBackfillService(store, client, inter_request_delay=0)

        result = self._run(svc.run())
        assert result["updated"] == 1

        updated = store.get_by_ticker("AAPL")[0]
        assert updated.realized_return_1d is not None
        # 5d and 20d may or may not be filled depending on date availability

    def test_backfill_multiple_tickers(self):
        """Backfill works across multiple tickers."""
        store = ShadowSignalStore()
        s1 = self._make_signal(ticker="AAPL", price=180.0)
        s2 = self._make_signal(ticker="MSFT", price=400.0)
        store.append(s1)
        store.append(s2)

        sd1 = s1.timestamp.date()
        sd2 = s2.timestamp.date()

        prices = {
            ("AAPL", advance_trading_days(sd1, 1).isoformat()): 182.0,
            ("AAPL", advance_trading_days(sd1, 5).isoformat()): 185.0,
            ("AAPL", advance_trading_days(sd1, 20).isoformat()): 190.0,
            ("MSFT", advance_trading_days(sd2, 1).isoformat()): 405.0,
            ("MSFT", advance_trading_days(sd2, 5).isoformat()): 410.0,
            ("MSFT", advance_trading_days(sd2, 20).isoformat()): 420.0,
        }
        client = MockPriceClient(prices)
        svc = PriceBackfillService(store, client, inter_request_delay=0)

        result = self._run(svc.run())
        assert result["updated"] == 2

    def test_backfill_max_signals_limit(self):
        """max_signals parameter limits processing."""
        store = ShadowSignalStore()
        for i in range(10):
            store.append(self._make_signal(ticker=f"T{i}", price=100.0))

        client = MockPriceClient({})
        svc = PriceBackfillService(store, client, inter_request_delay=0)

        result = self._run(svc.run(max_signals=3))
        assert result["processed"] == 3

    def test_backfill_empty_store(self):
        """Empty store produces zero results."""
        store = ShadowSignalStore()
        client = MockPriceClient({})
        svc = PriceBackfillService(store, client, inter_request_delay=0)

        result = self._run(svc.run())
        assert result["processed"] == 0
        assert result["updated"] == 0
