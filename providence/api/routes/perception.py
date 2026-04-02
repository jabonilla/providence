"""Perception sweep endpoints — fetch real market data.

Triggers perception agents to pull live data from Polygon, EDGAR,
FRED, etc. and populate the FragmentStore so cognition agents
have real market context to analyze.
"""

from __future__ import annotations

from typing import Optional

import structlog
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from providence.api.deps import get_state
from providence.services.perception_scheduler import PerceptionScheduler

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/perception", tags=["perception"])


class SweepRequest(BaseModel):
    """Request to trigger a perception sweep."""
    tickers: Optional[list[str]] = None  # None = full watchlist
    priority: Optional[int] = None  # 1=high priority only, None=all


class SweepResponse(BaseModel):
    """Result of a perception sweep."""
    status: str
    sweep_start: Optional[str] = None
    sweep_end: Optional[str] = None
    duration_seconds: float = 0.0
    tickers_processed: int = 0
    fragments_created: int = 0
    agents_run: int = 0
    errors: int = 0
    per_ticker_results: dict = {}


def _build_scheduler() -> PerceptionScheduler:
    """Build a PerceptionScheduler from current app state."""
    state = get_state()

    # Extract perception agents from the registry
    perception_agents = {
        agent_id: agent
        for agent_id, agent in state.agent_registry.items()
        if agent_id.startswith("PERCEPT-")
    }

    if not perception_agents:
        raise HTTPException(
            status_code=503,
            detail="No perception agents loaded. Check that API keys are configured and server was started without --skip-perception.",
        )

    if state.watchlist is None:
        raise HTTPException(
            status_code=503,
            detail="Watchlist not configured.",
        )

    return PerceptionScheduler(
        perception_agents=perception_agents,
        fragment_store=state.fragment_store,
        watchlist=state.watchlist,
        inter_ticker_delay=1.5,  # slightly faster for on-demand
        inter_agent_delay=0.5,
        max_concurrent_agents=2,  # conservative to avoid rate limits
    )


@router.post("/sweep", response_model=SweepResponse)
async def trigger_sweep(request: SweepRequest | None = None) -> SweepResponse:
    """Trigger a perception sweep to fetch live market data.

    Runs perception agents (PERCEPT-PRICE, PERCEPT-FILING, PERCEPT-NEWS,
    PERCEPT-OPTIONS, PERCEPT-CDS, PERCEPT-MACRO) for the specified tickers.

    If no tickers specified, sweeps the full watchlist.
    If priority is specified, only sweeps tickers at that priority level or higher.
    """
    scheduler = _build_scheduler()
    req = request or SweepRequest()

    try:
        if req.tickers:
            # Run specific tickers one at a time
            all_results = {}
            total_fragments = 0
            total_agents = 0
            total_errors = 0

            for ticker in req.tickers:
                result = await scheduler.run_single(ticker)
                all_results[ticker] = result
                total_fragments += result.get("fragments", 0)
                total_agents += result.get("agents_run", 0)
                total_errors += result.get("errors", 0)

            return SweepResponse(
                status="completed",
                tickers_processed=len(req.tickers),
                fragments_created=total_fragments,
                agents_run=total_agents,
                errors=total_errors,
                per_ticker_results=all_results,
            )

        elif req.priority:
            result = await scheduler.run_priority_sweep(max_priority=req.priority)
        else:
            result = await scheduler.run_full_sweep()

        return SweepResponse(
            status="completed",
            sweep_start=result.get("sweep_start"),
            sweep_end=result.get("sweep_end"),
            duration_seconds=result.get("duration_seconds", 0),
            tickers_processed=result.get("tickers_processed", 0),
            fragments_created=result.get("fragments_created", 0),
            agents_run=result.get("agents_run", 0),
            errors=result.get("errors", 0),
            per_ticker_results=result.get("per_ticker_results", {}),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Perception sweep failed", error=str(exc))
        raise HTTPException(status_code=500, detail="Perception sweep failed")


@router.get("/status")
async def perception_status() -> dict:
    """Get status of perception agents and last sweep info."""
    state = get_state()

    perception_agents = {
        agent_id: "loaded"
        for agent_id, agent in state.agent_registry.items()
        if agent_id.startswith("PERCEPT-")
    }

    # Fragment store stats
    fragment_count = state.fragment_store.count() if state.fragment_store else 0

    return {
        "perception_agents": perception_agents,
        "agent_count": len(perception_agents),
        "fragment_count": fragment_count,
        "watchlist_configured": state.watchlist is not None,
        "watchlist_tickers": state.watchlist.tickers if state.watchlist else [],
    }
