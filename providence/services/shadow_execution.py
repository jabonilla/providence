"""Shadow Execution Service — records pipeline signals without broker interaction.

In shadow mode, the full pipeline runs (Cognition → Regime → Decision →
Execution validation), but instead of submitting orders to a broker,
signals are recorded for offline analysis.

This replaces ExecutionService when SystemMode is SHADOW. The pipeline
output is captured as ShadowSignals stored in a ShadowSignalStore
(append-only JSONL, same pattern as FragmentStore/BeliefStore).

Usage:
    shadow_svc = ShadowExecutionService(signal_store)
    summary = shadow_svc.record_signals(run_id, validated_proposal, regime_state, metadata)
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from uuid import UUID, uuid4

import structlog

from providence.schemas.enums import Action, Direction, SystemMode
from providence.schemas.shadow import ShadowRunSummary, ShadowSignal

logger = structlog.get_logger()


class ShadowSignalStore:
    """Append-only store for shadow mode signals.

    Thread-safe. Indexed by run_id and ticker.
    Optionally persists to JSONL file.

    Same pattern as FragmentStore / BeliefStore / RunStore.
    """

    def __init__(self, persist_path: Path | None = None) -> None:
        self._signals: list[ShadowSignal] = []
        self._summaries: list[ShadowRunSummary] = []
        self._by_run: dict[UUID, list[ShadowSignal]] = {}
        self._by_ticker: dict[str, list[ShadowSignal]] = {}
        self._signal_ids: set[UUID] = set()
        self._lock = threading.RLock()

        self._persist_path = persist_path
        # Summaries stored in a sibling file: shadow_signals.jsonl → shadow_summaries.jsonl
        self._summaries_path: Path | None = None
        if persist_path:
            self._summaries_path = persist_path.parent / persist_path.name.replace(
                "signals", "summaries"
            )
        if persist_path and persist_path.exists():
            self._load_from_disk()
        if self._summaries_path and self._summaries_path.exists():
            self._load_summaries_from_disk()

    def _load_from_disk(self) -> None:
        """Load signals from JSONL file."""
        count = 0
        try:
            with open(self._persist_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        signal = ShadowSignal.model_validate(data)
                        self._index_signal(signal)
                        count += 1
                    except Exception as exc:
                        logger.debug("Skipping corrupt shadow signal line", error=str(exc))
            logger.info("Loaded shadow signals from disk", count=count)
        except Exception as exc:
            logger.warning("Failed to load shadow signals", error=str(exc))

    def _index_signal(self, signal: ShadowSignal) -> None:
        """Add signal to in-memory indices (no lock, caller must hold lock)."""
        if signal.signal_id in self._signal_ids:
            return
        self._signal_ids.add(signal.signal_id)
        self._signals.append(signal)
        self._by_run.setdefault(signal.run_id, []).append(signal)
        self._by_ticker.setdefault(signal.ticker, []).append(signal)

    def _persist_signal(self, signal: ShadowSignal) -> None:
        """Append a single signal to JSONL file."""
        if self._persist_path is None:
            return
        try:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._persist_path, "a") as f:
                f.write(signal.model_dump_json() + "\n")
        except Exception as exc:
            logger.warning("Failed to persist shadow signal", error=str(exc))

    def append(self, signal: ShadowSignal) -> bool:
        """Append a shadow signal. Returns True if new (not duplicate)."""
        with self._lock:
            if signal.signal_id in self._signal_ids:
                return False
            self._index_signal(signal)
            self._persist_signal(signal)
            return True

    def _load_summaries_from_disk(self) -> None:
        """Load run summaries from JSONL file."""
        count = 0
        try:
            with open(self._summaries_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        summary = ShadowRunSummary.model_validate(data)
                        self._summaries.append(summary)
                        count += 1
                    except Exception as exc:
                        logger.debug("Skipping corrupt summary line", error=str(exc))
            logger.info("Loaded shadow summaries from disk", count=count)
        except Exception as exc:
            logger.warning("Failed to load shadow summaries", error=str(exc))

    def _persist_summary(self, summary: ShadowRunSummary) -> None:
        """Append a single summary to JSONL file."""
        if self._summaries_path is None:
            return
        try:
            self._summaries_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._summaries_path, "a") as f:
                f.write(summary.model_dump_json() + "\n")
        except Exception as exc:
            logger.warning("Failed to persist shadow summary", error=str(exc))

    def append_summary(self, summary: ShadowRunSummary) -> None:
        """Append a run summary."""
        with self._lock:
            self._summaries.append(summary)
            self._persist_summary(summary)

    def get_by_run(self, run_id: UUID) -> list[ShadowSignal]:
        """Get all signals for a specific pipeline run."""
        with self._lock:
            return list(self._by_run.get(run_id, []))

    def get_by_ticker(self, ticker: str) -> list[ShadowSignal]:
        """Get all signals for a specific ticker."""
        with self._lock:
            return list(self._by_ticker.get(ticker, []))

    def get_all(self) -> list[ShadowSignal]:
        """Get all signals, newest first."""
        with self._lock:
            return list(reversed(self._signals))

    def get_summaries(self) -> list[ShadowRunSummary]:
        """Get all run summaries, newest first."""
        with self._lock:
            return list(reversed(self._summaries))

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._signals)

    @property
    def run_count(self) -> int:
        with self._lock:
            return len(self._by_run)

    def update_signal(self, signal_id: UUID, new_signal: ShadowSignal) -> bool:
        """Replace a signal by ID with an updated version.

        Used by PriceBackfillService to fill in realized return fields.
        The signal_id of the new signal MUST match the target signal_id.

        Args:
            signal_id: The signal to replace.
            new_signal: Updated signal (same signal_id, new price/return data).

        Returns:
            True if updated, False if signal_id not found.
        """
        if new_signal.signal_id != signal_id:
            return False

        with self._lock:
            if signal_id not in self._signal_ids:
                return False

            # Replace in main list
            for i, s in enumerate(self._signals):
                if s.signal_id == signal_id:
                    self._signals[i] = new_signal
                    break

            # Replace in run index
            run_signals = self._by_run.get(new_signal.run_id, [])
            for i, s in enumerate(run_signals):
                if s.signal_id == signal_id:
                    run_signals[i] = new_signal
                    break

            # Replace in ticker index
            ticker_signals = self._by_ticker.get(new_signal.ticker, [])
            for i, s in enumerate(ticker_signals):
                if s.signal_id == signal_id:
                    ticker_signals[i] = new_signal
                    break

            # Re-persist entire store (since we modified in place)
            self._re_persist_all()
            return True

    def _re_persist_all(self) -> None:
        """Rewrite the entire JSONL file from in-memory state.

        Called after update_signal to ensure persistence reflects updates.
        """
        if self._persist_path is None:
            return
        try:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._persist_path, "w") as f:
                for signal in self._signals:
                    f.write(signal.model_dump_json() + "\n")
        except Exception as exc:
            logger.warning("Failed to re-persist shadow signals", error=str(exc))

    def stats(self) -> dict[str, Any]:
        """Return store statistics."""
        with self._lock:
            tickers = set(self._by_ticker.keys())
            approved = sum(1 for s in self._signals if s.approved)
            return {
                "total_signals": len(self._signals),
                "total_runs": len(self._by_run),
                "total_summaries": len(self._summaries),
                "unique_tickers": len(tickers),
                "approved_signals": approved,
                "rejected_signals": len(self._signals) - approved,
            }


class ShadowExecutionService:
    """Records pipeline signals without broker interaction.

    Replaces ExecutionService when SystemMode is SHADOW.
    Extracts signals from EXEC-VALIDATE output and records them
    as ShadowSignals for offline analysis.
    """

    def __init__(self, signal_store: ShadowSignalStore) -> None:
        self._store = signal_store

    def record_signals(
        self,
        run_id: UUID,
        validated_proposal: dict,
        regime_state: dict | None = None,
        metadata: dict | None = None,
    ) -> ShadowRunSummary:
        """Record signals from a validated proposal.

        Extracts each position from the EXEC-VALIDATE output,
        creates a ShadowSignal for it, and stores it.

        Args:
            run_id: Pipeline run ID.
            validated_proposal: Serialized ValidatedProposal dict from EXEC-VALIDATE.
            regime_state: Current regime state dict (optional).
            metadata: Additional pipeline metadata (optional).

        Returns:
            ShadowRunSummary for this run.
        """
        log = logger.bind(run_id=str(run_id), mode="SHADOW")
        log.info("Recording shadow signals")

        regime_state = regime_state or {}
        metadata = metadata or {}
        risk_mode = regime_state.get("system_risk_mode", "NORMAL")
        regime_label = regime_state.get("statistical_regime", "LOW_VOL_TRENDING")

        results = validated_proposal.get("results", [])
        signals: list[ShadowSignal] = []
        approved_count = 0
        rejected_count = 0
        long_count = 0
        short_count = 0

        for result in results:
            if not isinstance(result, dict):
                continue

            ticker = result.get("ticker", "UNKNOWN")
            action_str = result.get("action", "CLOSE")
            direction_str = result.get("direction", "NEUTRAL")
            approved = result.get("approved", False)

            try:
                action = Action(action_str)
            except ValueError:
                action = Action.CLOSE
            try:
                direction = Direction(direction_str)
            except ValueError:
                direction = Direction.NEUTRAL

            # Get current price from metadata if available
            price_at_signal = self._get_current_price(ticker, metadata)

            # Simulate fill for approved signals
            simulated_entry = None
            simulated_qty = None
            simulated_notional = None
            adjusted_weight = float(result.get("adjusted_weight", 0.0))

            if approved and price_at_signal and price_at_signal > 0:
                # Assume $100K portfolio for shadow simulation
                shadow_equity = float(metadata.get("shadow_equity", 100_000.0))
                simulated_notional = shadow_equity * adjusted_weight
                simulated_entry = price_at_signal
                simulated_qty = int(simulated_notional / price_at_signal) if price_at_signal > 0 else 0

            signal = ShadowSignal(
                run_id=run_id,
                ticker=ticker,
                action=action,
                direction=direction,
                target_weight=float(result.get("target_weight", 0.0)),
                confidence=float(result.get("confidence", 0.0)),
                approved=approved,
                rejection_reasons=result.get("rejection_reasons", []),
                adjusted_weight=adjusted_weight,
                risk_mode_applied=risk_mode,
                simulated_entry_price=simulated_entry,
                simulated_fill_qty=simulated_qty,
                simulated_notional=simulated_notional,
                price_at_signal=price_at_signal,
            )

            self._store.append(signal)
            signals.append(signal)

            if approved:
                approved_count += 1
            else:
                rejected_count += 1

            if direction == Direction.LONG:
                long_count += 1
            elif direction == Direction.SHORT:
                short_count += 1

        summary = ShadowRunSummary(
            run_id=run_id,
            system_mode=SystemMode.SHADOW,
            signals=signals,
            total_signals=len(signals),
            approved_signals=approved_count,
            rejected_signals=rejected_count,
            long_signals=long_count,
            short_signals=short_count,
            risk_mode=risk_mode,
            regime_state=regime_label,
        )

        self._store.append_summary(summary)

        log.info(
            "Shadow signals recorded",
            total=len(signals),
            approved=approved_count,
            rejected=rejected_count,
            longs=long_count,
            shorts=short_count,
        )

        return summary

    def _get_current_price(self, ticker: str, metadata: dict) -> Optional[float]:
        """Extract current price for a ticker from pipeline metadata.

        Looks for price data in the metadata that perception agents
        would have populated.
        """
        # Try direct price lookup in metadata
        prices = metadata.get("current_prices", {})
        if ticker in prices:
            try:
                return float(prices[ticker])
            except (ValueError, TypeError):
                pass

        # Try extracting from fragment data
        fragments = metadata.get("fragments", {})
        if isinstance(fragments, dict):
            for key, frag in fragments.items():
                if isinstance(frag, dict) and frag.get("entity") == ticker:
                    payload = frag.get("payload", {})
                    close = payload.get("close")
                    if close is not None:
                        try:
                            return float(close)
                        except (ValueError, TypeError):
                            pass

        return None

    @property
    def store(self) -> ShadowSignalStore:
        """Access the underlying signal store."""
        return self._store
