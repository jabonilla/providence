"""Shadow mode API endpoints — signals, summaries, report, and backfill.

Exposes shadow signal store data and performance metrics for monitoring
signal quality during Launch Plan Phase B (shadow mode evaluation).
"""

from __future__ import annotations

from typing import Optional
from uuid import UUID

import structlog
from fastapi import APIRouter, HTTPException, Query

from providence.api.deps import get_state
from providence.api.schemas import (
    BackfillTriggerResponse,
    ShadowReportResponse,
    ShadowRunSummaryResponse,
    ShadowSignalResponse,
    ShadowStoreStatsResponse,
)

logger = structlog.get_logger()

router = APIRouter(prefix="/shadow", tags=["shadow"])


def _get_shadow_store():
    """Get shadow signal store from app state, 404 if not active."""
    state = get_state()
    store = state.shadow_signal_store
    if store is None:
        raise HTTPException(
            status_code=404,
            detail="Shadow mode not active — system is not running in SHADOW mode",
        )
    return store


# ── Stats ──────────────────────────────────────────────────────────

@router.get("/stats", response_model=ShadowStoreStatsResponse)
async def get_shadow_stats() -> ShadowStoreStatsResponse:
    """Get shadow signal store statistics."""
    store = _get_shadow_store()
    stats = store.stats()
    return ShadowStoreStatsResponse(**stats)


# ── Signals ────────────────────────────────────────────────────────

@router.get("/signals", response_model=list[ShadowSignalResponse])
async def list_signals(
    ticker: Optional[str] = Query(None, description="Filter by ticker"),
    run_id: Optional[UUID] = Query(None, description="Filter by run ID"),
    approved_only: bool = Query(False, description="Only show approved signals"),
    limit: int = Query(100, ge=1, le=1000, description="Max results"),
) -> list[ShadowSignalResponse]:
    """List shadow signals with optional filtering."""
    store = _get_shadow_store()

    if ticker:
        signals = store.get_by_ticker(ticker.upper())
    elif run_id:
        signals = store.get_by_run(run_id)
    else:
        signals = store.get_all()

    if approved_only:
        signals = [s for s in signals if s.approved]

    signals = signals[:limit]

    return [
        ShadowSignalResponse(
            signal_id=s.signal_id,
            run_id=s.run_id,
            timestamp=s.timestamp,
            ticker=s.ticker,
            action=s.action.value,
            direction=s.direction.value,
            target_weight=s.target_weight,
            confidence=s.confidence,
            approved=s.approved,
            rejection_reasons=s.rejection_reasons,
            adjusted_weight=s.adjusted_weight,
            risk_mode_applied=s.risk_mode_applied,
            simulated_entry_price=s.simulated_entry_price,
            simulated_fill_qty=s.simulated_fill_qty,
            simulated_notional=s.simulated_notional,
            price_at_signal=s.price_at_signal,
            price_1d_later=s.price_1d_later,
            price_5d_later=s.price_5d_later,
            price_20d_later=s.price_20d_later,
            realized_return_1d=s.realized_return_1d,
            realized_return_5d=s.realized_return_5d,
            realized_return_20d=s.realized_return_20d,
        )
        for s in signals
    ]


@router.get("/signals/{signal_id}", response_model=ShadowSignalResponse)
async def get_signal(signal_id: UUID) -> ShadowSignalResponse:
    """Get a specific shadow signal by ID."""
    store = _get_shadow_store()

    # Search through all signals
    for s in store.get_all():
        if s.signal_id == signal_id:
            return ShadowSignalResponse(
                signal_id=s.signal_id,
                run_id=s.run_id,
                timestamp=s.timestamp,
                ticker=s.ticker,
                action=s.action.value,
                direction=s.direction.value,
                target_weight=s.target_weight,
                confidence=s.confidence,
                approved=s.approved,
                rejection_reasons=s.rejection_reasons,
                adjusted_weight=s.adjusted_weight,
                risk_mode_applied=s.risk_mode_applied,
                simulated_entry_price=s.simulated_entry_price,
                simulated_fill_qty=s.simulated_fill_qty,
                simulated_notional=s.simulated_notional,
                price_at_signal=s.price_at_signal,
                price_1d_later=s.price_1d_later,
                price_5d_later=s.price_5d_later,
                price_20d_later=s.price_20d_later,
                realized_return_1d=s.realized_return_1d,
                realized_return_5d=s.realized_return_5d,
                realized_return_20d=s.realized_return_20d,
            )

    raise HTTPException(status_code=404, detail="Signal not found")


# ── Run Summaries ──────────────────────────────────────────────────

@router.get("/summaries", response_model=list[ShadowRunSummaryResponse])
async def list_summaries(
    limit: int = Query(50, ge=1, le=200, description="Max results"),
) -> list[ShadowRunSummaryResponse]:
    """List shadow run summaries (newest first)."""
    store = _get_shadow_store()
    summaries = store.get_summaries()[:limit]

    return [
        ShadowRunSummaryResponse(
            run_id=s.run_id,
            timestamp=s.timestamp,
            system_mode=s.system_mode.value,
            total_signals=s.total_signals,
            approved_signals=s.approved_signals,
            rejected_signals=s.rejected_signals,
            long_signals=s.long_signals,
            short_signals=s.short_signals,
            risk_mode=s.risk_mode,
            regime_state=s.regime_state,
        )
        for s in summaries
    ]


# ── Performance Report ─────────────────────────────────────────────

@router.get("/report", response_model=ShadowReportResponse)
async def get_shadow_report() -> ShadowReportResponse:
    """Generate and return the shadow mode performance report.

    Computes directional accuracy, hypothetical returns,
    and Phase B readiness criteria from stored signals.
    """
    store = _get_shadow_store()
    signals = store.get_all()
    stats = store.stats()

    if not signals:
        return ShadowReportResponse(status="NO_DATA", **{
            k: v for k, v in stats.items()
            if k in ("total_runs", "total_signals")
        })

    from datetime import datetime, timezone

    approved = [s for s in signals if s.approved]
    longs = [s for s in signals if s.direction.value == "LONG"]
    shorts = [s for s in signals if s.direction.value == "SHORT"]
    confidences = [s.confidence for s in signals]

    # Directional accuracy at each horizon
    def _accuracy(horizon: str) -> float | None:
        attr = f"realized_return_{horizon}"
        measured = [(s, getattr(s, attr, None)) for s in approved]
        measured = [(s, r) for s, r in measured if r is not None]
        if not measured:
            return None
        correct = sum(
            1 for s, r in measured
            if (r > 0 and s.direction.value == "LONG")
            or (r > 0 and s.direction.value == "SHORT")  # short return already signed
            or (r == 0)
        )
        return round(correct / len(measured), 4)

    def _hypo_return(horizon: str) -> float | None:
        attr = f"realized_return_{horizon}"
        returns = [getattr(s, attr) for s in approved if getattr(s, attr, None) is not None]
        if not returns:
            return None
        return round(sum(returns) / len(returns), 6)

    acc_1d = _accuracy("1d")
    acc_5d = _accuracy("5d")
    acc_20d = _accuracy("20d")

    # Phase B criteria
    phase_b = {
        "accuracy_above_55pct": (acc_5d or 0) > 0.55,
        "min_20_signals": len(approved) >= 20,
        "stability": stats["total_runs"] >= 5,
    }

    # Per-ticker breakdown
    ticker_breakdown: dict = {}
    for ticker in set(s.ticker for s in signals):
        t_sigs = [s for s in signals if s.ticker == ticker]
        t_approved = [s for s in t_sigs if s.approved]
        t_ret_5d = [s.realized_return_5d for s in t_approved if s.realized_return_5d is not None]
        ticker_breakdown[ticker] = {
            "total": len(t_sigs),
            "approved": len(t_approved),
            "avg_confidence": round(sum(s.confidence for s in t_sigs) / len(t_sigs), 3) if t_sigs else 0,
            "accuracy_5d": round(
                sum(1 for r in t_ret_5d if r > 0) / len(t_ret_5d), 3
            ) if t_ret_5d else None,
            "avg_return_5d": round(sum(t_ret_5d) / len(t_ret_5d), 5) if t_ret_5d else None,
        }

    return ShadowReportResponse(
        status="OK",
        generated_at=datetime.now(timezone.utc).isoformat(),
        period_start=signals[-1].timestamp.isoformat() if signals else None,
        period_end=signals[0].timestamp.isoformat() if signals else None,
        total_runs=stats["total_runs"],
        total_signals=stats["total_signals"],
        total_approved=stats["approved_signals"],
        total_rejected=stats["rejected_signals"],
        total_longs=len(longs),
        total_shorts=len(shorts),
        unique_tickers=stats["unique_tickers"],
        avg_confidence=round(sum(confidences) / len(confidences), 3) if confidences else 0,
        avg_signals_per_run=round(len(signals) / max(stats["total_runs"], 1), 1),
        long_short_ratio=round(len(longs) / max(len(shorts), 1), 2),
        accuracy_1d=acc_1d,
        accuracy_5d=acc_5d,
        accuracy_20d=acc_20d,
        hypothetical_return_1d=_hypo_return("1d"),
        hypothetical_return_5d=_hypo_return("5d"),
        hypothetical_return_20d=_hypo_return("20d"),
        phase_b_criteria=phase_b,
        ticker_breakdown=ticker_breakdown,
    )


# ── Backfill Trigger ───────────────────────────────────────────────

@router.post("/backfill", response_model=BackfillTriggerResponse)
async def trigger_backfill(
    max_signals: int = Query(500, ge=1, le=5000, description="Max signals to process"),
) -> BackfillTriggerResponse:
    """Trigger a price backfill for shadow signals missing realized returns.

    Requires POLYGON_API_KEY to be configured. Fetches historical
    closing prices and computes directional accuracy metrics.
    """
    store = _get_shadow_store()
    state = get_state()

    # Try to get a price client from the app state
    price_client = state.extra.get("polygon_client")
    if price_client is None:
        raise HTTPException(
            status_code=503,
            detail="Price client not available — ensure POLYGON_API_KEY is configured",
        )

    from providence.services.price_backfill import PriceBackfillService

    backfill_svc = PriceBackfillService(
        signal_store=store,
        price_client=price_client,
    )

    try:
        result = await backfill_svc.run(max_signals=max_signals)
    except Exception as exc:
        logger.error("Backfill failed", error=str(exc))
        raise HTTPException(status_code=500, detail="Backfill operation failed")

    return BackfillTriggerResponse(**result)
