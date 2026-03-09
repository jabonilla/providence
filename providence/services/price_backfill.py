"""Price Backfill Service — computes realized returns for shadow signals.

After shadow signals are recorded, this service fetches closing prices
at T+1, T+5, and T+20 trading days to compute directional accuracy
and hypothetical returns for the shadow report.

The backfill runs offline (batch), fetching from Polygon.io:
  1. Scans ShadowSignalStore for approved signals missing realized returns
  2. Groups by ticker to minimize API calls
  3. Fetches daily closes for target dates
  4. Computes returns and updates signals in the store

Trading day calculation uses a simple heuristic (skip weekends,
skip known US market holidays). Not perfect but sufficient for
5-day and 20-day horizons where off-by-one doesn't matter much.

Usage:
    backfill = PriceBackfillService(signal_store, polygon_client)
    result = await backfill.run()
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional

import structlog

from providence.schemas.enums import Direction
from providence.schemas.shadow import ShadowSignal
from providence.services.shadow_execution import ShadowSignalStore

logger = structlog.get_logger()

# US market holidays (approximate, covers major ones)
US_HOLIDAYS = {
    # 2025
    date(2025, 1, 1), date(2025, 1, 20), date(2025, 2, 17),
    date(2025, 4, 18), date(2025, 5, 26), date(2025, 6, 19),
    date(2025, 7, 4), date(2025, 9, 1), date(2025, 11, 27),
    date(2025, 12, 25),
    # 2026
    date(2026, 1, 1), date(2026, 1, 19), date(2026, 2, 16),
    date(2026, 4, 3), date(2026, 5, 25), date(2026, 6, 19),
    date(2026, 7, 3), date(2026, 9, 7), date(2026, 11, 26),
    date(2026, 12, 25),
    # 2027
    date(2027, 1, 1), date(2027, 1, 18), date(2027, 2, 15),
    date(2027, 4, 2), date(2027, 5, 31), date(2027, 6, 18),
    date(2027, 7, 5), date(2027, 9, 6), date(2027, 11, 25),
    date(2027, 12, 24),
}


def _is_trading_day(d: date) -> bool:
    """Check if a date is a US equity trading day."""
    if d.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    if d in US_HOLIDAYS:
        return False
    return True


def advance_trading_days(start: date, n: int) -> date:
    """Advance n trading days from start date.

    Args:
        start: Starting date.
        n: Number of trading days to advance.

    Returns:
        The date that is n trading days after start.
    """
    current = start
    advanced = 0
    while advanced < n:
        current += timedelta(days=1)
        if _is_trading_day(current):
            advanced += 1
    return current


class PriceBackfillService:
    """Fetches historical closes and computes realized returns for shadow signals.

    Works with any price source that implements get_daily_bars(ticker, date_str).
    Designed for batch operation — call run() periodically after shadow signals
    accumulate (e.g., daily after market close).
    """

    # Horizons to backfill: (field_suffix, trading_days)
    HORIZONS = [
        ("1d", 1),
        ("5d", 5),
        ("20d", 20),
    ]

    def __init__(
        self,
        signal_store: ShadowSignalStore,
        price_client: Any,  # PolygonClient or compatible
        inter_request_delay: float = 0.5,
    ) -> None:
        """Initialize backfill service.

        Args:
            signal_store: The shadow signal store to update.
            price_client: Async price client with get_daily_bars(ticker, date_str).
            inter_request_delay: Seconds between API requests (rate limiting).
        """
        self._store = signal_store
        self._price_client = price_client
        self._delay = inter_request_delay

    async def run(self, max_signals: int = 500) -> dict[str, Any]:
        """Run the backfill process.

        Scans for signals missing realized returns, fetches prices,
        computes returns, and updates the store.

        Args:
            max_signals: Maximum number of signals to process in one batch.

        Returns:
            Summary dict with counts and errors.
        """
        log = logger.bind(service="price_backfill")
        log.info("Starting price backfill")

        # Find signals needing backfill
        candidates = self._find_backfill_candidates(max_signals)
        if not candidates:
            log.info("No signals need backfill")
            return {"processed": 0, "updated": 0, "errors": 0, "skipped": 0}

        log.info("Backfill candidates found", count=len(candidates))

        # Group required price fetches by (ticker, date) to minimize API calls
        price_requests = self._build_price_requests(candidates)
        log.info("Price requests needed", unique_requests=len(price_requests))

        # Fetch prices
        price_cache = await self._fetch_prices(price_requests)
        log.info("Prices fetched", cached=len(price_cache))

        # Update signals with realized returns
        updated = 0
        errors = 0
        skipped = 0

        for signal in candidates:
            try:
                new_signal = self._compute_returns(signal, price_cache)
                if new_signal is not None:
                    self._store.update_signal(signal.signal_id, new_signal)
                    updated += 1
                else:
                    skipped += 1
            except Exception as exc:
                log.debug(
                    "Failed to update signal",
                    signal_id=str(signal.signal_id),
                    error=str(exc),
                )
                errors += 1

        result = {
            "processed": len(candidates),
            "updated": updated,
            "errors": errors,
            "skipped": skipped,
            "prices_fetched": len(price_cache),
        }
        log.info("Price backfill complete", **result)
        return result

    def _find_backfill_candidates(self, max_count: int) -> list[ShadowSignal]:
        """Find approved signals that are missing realized return data.

        Only looks at signals where enough time has passed for at
        least the 1-day horizon. Signals without price_at_signal
        are skipped (no entry price to compute return from).
        """
        now = datetime.now(timezone.utc)
        min_age = timedelta(days=2)  # Need at least 1 trading day to have passed
        candidates = []

        for signal in self._store.get_all():
            if len(candidates) >= max_count:
                break

            # Skip unapproved or signals without entry price
            if not signal.approved:
                continue
            if signal.price_at_signal is None or signal.price_at_signal <= 0:
                continue

            # Skip if all returns already computed
            if (
                signal.realized_return_1d is not None
                and signal.realized_return_5d is not None
                and signal.realized_return_20d is not None
            ):
                continue

            # Skip if too recent for any backfill
            age = now - signal.timestamp
            if age < min_age:
                continue

            candidates.append(signal)

        return candidates

    def _build_price_requests(
        self, signals: list[ShadowSignal]
    ) -> set[tuple[str, str]]:
        """Build unique (ticker, date_str) pairs needed for backfill.

        For each signal, computes the target dates for each horizon
        and adds them to the request set.
        """
        requests: set[tuple[str, str]] = set()
        now = date.today()

        for signal in signals:
            signal_date = signal.timestamp.date()
            for suffix, trading_days in self.HORIZONS:
                # Check if this horizon needs backfill
                existing = getattr(signal, f"realized_return_{suffix}", None)
                if existing is not None:
                    continue

                target_date = advance_trading_days(signal_date, trading_days)
                # Only request if target date is in the past
                if target_date <= now:
                    requests.add((signal.ticker, target_date.isoformat()))

        return requests

    async def _fetch_prices(
        self, requests: set[tuple[str, str]]
    ) -> dict[tuple[str, str], float]:
        """Fetch closing prices for all (ticker, date) pairs.

        Returns a cache mapping (ticker, date_str) → closing_price.
        Failed fetches are silently skipped (the signal won't be updated).
        """
        cache: dict[tuple[str, str], float] = {}

        for ticker, date_str in sorted(requests):
            try:
                response = await self._price_client.get_daily_bars(ticker, date_str)
                results = response.get("results", [])
                if results and len(results) > 0:
                    close = results[0].get("c")  # Polygon uses 'c' for close
                    if close is not None:
                        cache[(ticker, date_str)] = float(close)
            except Exception as exc:
                logger.debug(
                    "Price fetch failed",
                    ticker=ticker,
                    date=date_str,
                    error=str(exc),
                )

            # Rate limiting
            if self._delay > 0:
                await asyncio.sleep(self._delay)

        return cache

    def _compute_returns(
        self,
        signal: ShadowSignal,
        price_cache: dict[tuple[str, str], float],
    ) -> ShadowSignal | None:
        """Compute realized returns for a signal using cached prices.

        Creates a new ShadowSignal with price and return fields filled in.
        Returns None if no new data is available (nothing to update).
        """
        signal_date = signal.timestamp.date()
        entry_price = signal.price_at_signal

        if entry_price is None or entry_price <= 0:
            return None

        updates: dict[str, Any] = {}
        any_update = False

        for suffix, trading_days in self.HORIZONS:
            price_field = f"price_{suffix}_later"
            return_field = f"realized_return_{suffix}"

            # Skip already-computed horizons
            if getattr(signal, return_field, None) is not None:
                updates[price_field] = getattr(signal, price_field)
                updates[return_field] = getattr(signal, return_field)
                continue

            target_date = advance_trading_days(signal_date, trading_days)
            cache_key = (signal.ticker, target_date.isoformat())
            close_price = price_cache.get(cache_key)

            if close_price is not None:
                raw_return = (close_price - entry_price) / entry_price
                updates[price_field] = close_price
                updates[return_field] = raw_return
                any_update = True
            else:
                # Keep existing values (None)
                updates[price_field] = getattr(signal, price_field)
                updates[return_field] = getattr(signal, return_field)

        if not any_update:
            return None

        # Create new frozen signal with updated fields
        return ShadowSignal(
            signal_id=signal.signal_id,
            run_id=signal.run_id,
            timestamp=signal.timestamp,
            ticker=signal.ticker,
            action=signal.action,
            direction=signal.direction,
            target_weight=signal.target_weight,
            confidence=signal.confidence,
            approved=signal.approved,
            rejection_reasons=signal.rejection_reasons,
            adjusted_weight=signal.adjusted_weight,
            risk_mode_applied=signal.risk_mode_applied,
            simulated_entry_price=signal.simulated_entry_price,
            simulated_fill_qty=signal.simulated_fill_qty,
            simulated_notional=signal.simulated_notional,
            price_at_signal=signal.price_at_signal,
            price_1d_later=updates.get("price_1d_later"),
            price_5d_later=updates.get("price_5d_later"),
            price_20d_later=updates.get("price_20d_later"),
            realized_return_1d=updates.get("realized_return_1d"),
            realized_return_5d=updates.get("realized_return_5d"),
            realized_return_20d=updates.get("realized_return_20d"),
        )
