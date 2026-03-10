"""Portfolio management API endpoints — positions, orders, and snapshots.

Exposes portfolio tracker and order manager data for monitoring positions
and order lifecycle during trading operations.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Optional
from uuid import UUID

import structlog
from fastapi import APIRouter, HTTPException, Query

from providence.api.deps import get_state
from providence.api.schemas import (
    OrderResponse,
    OrderStatsResponse,
    PositionResponse,
    PortfolioSnapshotResponse,
)
from providence.portfolio.tracker import PositionSide

logger = structlog.get_logger()

router = APIRouter(prefix="/portfolio", tags=["portfolio"])


def _get_portfolio_tracker():
    """Get portfolio tracker from app state, 404 if not available."""
    state = get_state()
    tracker = state.portfolio_tracker
    if tracker is None:
        raise HTTPException(
            status_code=404,
            detail="Portfolio tracking not enabled",
        )
    return tracker


def _get_order_manager():
    """Get order manager from app state, 404 if not available."""
    state = get_state()
    manager = state.order_manager
    if manager is None:
        raise HTTPException(
            status_code=404,
            detail="Order management not enabled",
        )
    return manager


# ── Snapshot ────────────────────────────────────────────────────────

@router.get("/snapshot", response_model=PortfolioSnapshotResponse)
async def get_snapshot() -> PortfolioSnapshotResponse:
    """Get current portfolio snapshot."""
    tracker = _get_portfolio_tracker()
    snap = tracker.snapshot()

    # Build position responses
    positions_resp = {}
    for ticker, pos in snap.positions.items():
        positions_resp[ticker] = PositionResponse(
            ticker=pos.ticker,
            side=pos.side.value,
            quantity=str(pos.quantity),
            avg_entry_price=str(pos.avg_entry_price),
            current_price=str(pos.current_price),
            market_value=str(pos.market_value),
            unrealized_pnl=str(pos.unrealized_pnl),
            unrealized_pnl_pct=pos.unrealized_pnl_pct,
            realized_pnl=str(pos.realized_pnl),
            weight=pos.weight,
            sector=pos.sector,
            days_held=pos.days_held,
        )

    return PortfolioSnapshotResponse(
        snapshot_id=snap.snapshot_id,
        timestamp=snap.timestamp,
        equity=str(snap.equity),
        cash=str(snap.cash),
        buying_power=str(snap.buying_power),
        positions=positions_resp,
        gross_exposure=snap.gross_exposure,
        net_exposure=snap.net_exposure,
        long_exposure=snap.long_exposure,
        short_exposure=snap.short_exposure,
        sector_exposure=snap.sector_exposure,
        position_count=snap.position_count,
        total_unrealized_pnl=str(snap.total_unrealized_pnl),
        total_realized_pnl=str(snap.total_realized_pnl),
        drawdown_pct=snap.drawdown_pct,
    )


# ── Positions ───────────────────────────────────────────────────────

@router.get("/positions", response_model=list[PositionResponse])
async def list_positions() -> list[PositionResponse]:
    """List all open positions."""
    tracker = _get_portfolio_tracker()
    positions = tracker.positions

    return [
        PositionResponse(
            ticker=pos.ticker,
            side=pos.side.value,
            quantity=str(pos.quantity),
            avg_entry_price=str(pos.avg_entry_price),
            current_price=str(pos.current_price),
            market_value=str(pos.market_value),
            unrealized_pnl=str(pos.unrealized_pnl),
            unrealized_pnl_pct=pos.unrealized_pnl_pct,
            realized_pnl=str(pos.realized_pnl),
            weight=pos.weight,
            sector=pos.sector,
            days_held=pos.days_held,
        )
        for pos in positions.values()
        if pos.side != PositionSide.FLAT
    ]


@router.get("/positions/{ticker}", response_model=PositionResponse)
async def get_position(ticker: str) -> PositionResponse:
    """Get a specific position by ticker."""
    tracker = _get_portfolio_tracker()
    pos = tracker.get_position(ticker.upper())

    if pos is None or pos.side == PositionSide.FLAT:
        raise HTTPException(status_code=404, detail=f"Position {ticker} not found")

    return PositionResponse(
        ticker=pos.ticker,
        side=pos.side.value,
        quantity=str(pos.quantity),
        avg_entry_price=str(pos.avg_entry_price),
        current_price=str(pos.current_price),
        market_value=str(pos.market_value),
        unrealized_pnl=str(pos.unrealized_pnl),
        unrealized_pnl_pct=pos.unrealized_pnl_pct,
        realized_pnl=str(pos.realized_pnl),
        weight=pos.weight,
        sector=pos.sector,
        days_held=pos.days_held,
    )


# ── Orders ──────────────────────────────────────────────────────────

@router.get("/orders", response_model=list[OrderResponse])
async def list_orders(
    status: Optional[str] = Query(None, description="Filter by status"),
    ticker: Optional[str] = Query(None, description="Filter by ticker"),
    limit: int = Query(100, ge=1, le=1000, description="Max results"),
) -> list[OrderResponse]:
    """List orders with optional filtering."""
    manager = _get_order_manager()

    # Get all orders
    all_orders = list(manager._orders.values())

    # Filter by status if provided
    if status:
        status_upper = status.upper()
        all_orders = [o for o in all_orders if o.status.value == status_upper]

    # Filter by ticker if provided
    if ticker:
        ticker_upper = ticker.upper()
        all_orders = [o for o in all_orders if o.ticker == ticker_upper]

    # Sort by created_at descending (newest first)
    all_orders.sort(key=lambda o: o.created_at, reverse=True)

    # Apply limit
    all_orders = all_orders[:limit]

    return [
        OrderResponse(
            order_id=o.order_id,
            broker_order_id=o.broker_order_id,
            client_order_id=o.client_order_id,
            ticker=o.ticker,
            side=o.side.value,
            order_type=o.order_type,
            time_in_force=o.time_in_force,
            qty=str(o.qty) if o.qty is not None else None,
            notional=str(o.notional) if o.notional is not None else None,
            limit_price=str(o.limit_price) if o.limit_price is not None else None,
            status=o.status.value,
            filled_qty=str(o.filled_qty),
            filled_avg_price=str(o.filled_avg_price),
            execution_strategy=o.execution_strategy,
            target_weight=o.target_weight,
            confidence=o.confidence,
            created_at=o.created_at,
            submitted_at=o.submitted_at,
            filled_at=o.filled_at,
            retry_count=o.retry_count,
            last_error=o.last_error,
        )
        for o in all_orders
    ]


@router.get("/orders/{order_id}", response_model=OrderResponse)
async def get_order(order_id: UUID) -> OrderResponse:
    """Get a specific order by ID."""
    manager = _get_order_manager()
    order = manager.get_order(order_id)

    if order is None:
        raise HTTPException(status_code=404, detail=f"Order {order_id} not found")

    return OrderResponse(
        order_id=order.order_id,
        broker_order_id=order.broker_order_id,
        client_order_id=order.client_order_id,
        ticker=order.ticker,
        side=order.side.value,
        order_type=order.order_type,
        time_in_force=order.time_in_force,
        qty=str(order.qty) if order.qty is not None else None,
        notional=str(order.notional) if order.notional is not None else None,
        limit_price=str(order.limit_price) if order.limit_price is not None else None,
        status=order.status.value,
        filled_qty=str(order.filled_qty),
        filled_avg_price=str(order.filled_avg_price),
        execution_strategy=order.execution_strategy,
        target_weight=order.target_weight,
        confidence=order.confidence,
        created_at=order.created_at,
        submitted_at=order.submitted_at,
        filled_at=order.filled_at,
        retry_count=order.retry_count,
        last_error=order.last_error,
    )


@router.get("/orders/stats", response_model=OrderStatsResponse)
async def get_order_stats() -> OrderStatsResponse:
    """Get order statistics (counts by status)."""
    manager = _get_order_manager()
    stats = manager.stats

    return OrderStatsResponse(
        total=stats.get("total", 0),
        active=stats.get("active", 0),
        by_status={k: v for k, v in stats.items() if k not in ("total", "active")},
    )


# ── History ─────────────────────────────────────────────────────────

@router.get("/history", response_model=list[PortfolioSnapshotResponse])
async def get_snapshot_history(
    limit: int = Query(50, ge=1, le=500, description="Max results"),
) -> list[PortfolioSnapshotResponse]:
    """Get recent portfolio snapshots (newest first)."""
    tracker = _get_portfolio_tracker()

    # Access internal snapshots (newest first)
    snapshots = tracker._snapshots[::-1][:limit]

    result = []
    for snap in snapshots:
        # Build position responses
        positions_resp = {}
        for ticker, pos in snap.positions.items():
            positions_resp[ticker] = PositionResponse(
                ticker=pos.ticker,
                side=pos.side.value,
                quantity=str(pos.quantity),
                avg_entry_price=str(pos.avg_entry_price),
                current_price=str(pos.current_price),
                market_value=str(pos.market_value),
                unrealized_pnl=str(pos.unrealized_pnl),
                unrealized_pnl_pct=pos.unrealized_pnl_pct,
                realized_pnl=str(pos.realized_pnl),
                weight=pos.weight,
                sector=pos.sector,
                days_held=pos.days_held,
            )

        result.append(
            PortfolioSnapshotResponse(
                snapshot_id=snap.snapshot_id,
                timestamp=snap.timestamp,
                equity=str(snap.equity),
                cash=str(snap.cash),
                buying_power=str(snap.buying_power),
                positions=positions_resp,
                gross_exposure=snap.gross_exposure,
                net_exposure=snap.net_exposure,
                long_exposure=snap.long_exposure,
                short_exposure=snap.short_exposure,
                sector_exposure=snap.sector_exposure,
                position_count=snap.position_count,
                total_unrealized_pnl=str(snap.total_unrealized_pnl),
                total_realized_pnl=str(snap.total_realized_pnl),
                drawdown_pct=snap.drawdown_pct,
            )
        )

    return result
