"""Configuration and watchlist endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from providence.api.deps import get_state
from providence.api.schemas import WatchlistEntryResponse, WatchlistResponse

router = APIRouter(prefix="/config", tags=["config"])


@router.get("/watchlist", response_model=WatchlistResponse)
async def get_watchlist() -> WatchlistResponse:
    """Get the current watchlist configuration."""
    state = get_state()
    if state.watchlist is None:
        raise HTTPException(status_code=404, detail="Watchlist not configured")

    entries = [
        WatchlistEntryResponse(
            ticker=e.ticker,
            sector=e.sector,
            enabled=e.enabled,
            priority=e.priority,
            tags=list(e.tags),
        )
        for e in state.watchlist.entries
    ]

    return WatchlistResponse(
        name=state.watchlist.name,
        max_positions=state.watchlist.max_positions,
        entries=entries,
        active_tickers=state.watchlist.tickers,
    )
