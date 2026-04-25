"""Configuration endpoints — watchlist, agent weights, preferences, tiers."""

from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, HTTPException

from providence.api.deps import get_state
from providence.api.schemas import (
    AccountInfoResponse,
    AgentPreferencesResponse,
    AgentPreferencesUpdateRequest,
    AgentWeightResponse,
    AgentWeightUpdateRequest,
    TierInfoResponse,
    TierLimitsResponse,
    WatchlistEntryResponse,
    WatchlistResponse,
)
from providence.config.account_tiers import TIER_LIMITS, AccountTier, get_tier_limits
from providence.config.agent_preferences import AgentPreferences
from providence.config.agent_weights import DEFAULT_WEIGHTS
from providence.config.watchlist import Watchlist, WatchlistEntry

router = APIRouter(prefix="/config", tags=["config"])


# ── Watchlist ──────────────────────────────────────────────────────

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


@router.post("/watchlist/entries", response_model=WatchlistEntryResponse)
async def add_watchlist_entry(
    ticker: str,
    sector: str = "",
    priority: int = 1,
    tags: list[str] | None = None,
) -> WatchlistEntryResponse:
    """Add a ticker to the watchlist."""
    state = get_state()
    if state.watchlist is None:
        state.watchlist = Watchlist.default()

    # Check for duplicate
    existing_tickers = [e.ticker for e in state.watchlist.entries]
    if ticker.upper() in existing_tickers:
        raise HTTPException(status_code=409, detail="Ticker already in watchlist")

    new_entry = WatchlistEntry(
        ticker=ticker.upper(),
        sector=sector,
        enabled=True,
        priority=priority,
        tags=tuple(tags) if tags else (),
    )

    # Rebuild watchlist with new entry (frozen dataclass)
    state.watchlist = Watchlist(
        entries=state.watchlist.entries + (new_entry,),
        name=state.watchlist.name,
        max_positions=state.watchlist.max_positions,
    )

    return WatchlistEntryResponse(
        ticker=new_entry.ticker,
        sector=new_entry.sector,
        enabled=new_entry.enabled,
        priority=new_entry.priority,
        tags=list(new_entry.tags),
    )


@router.put("/watchlist/entries/{ticker}")
async def update_watchlist_entry(
    ticker: str,
    sector: str | None = None,
    priority: int | None = None,
    enabled: bool | None = None,
    tags: list[str] | None = None,
) -> WatchlistEntryResponse:
    """Update a watchlist entry."""
    state = get_state()
    if state.watchlist is None:
        raise HTTPException(status_code=404, detail="Watchlist not configured")

    ticker_upper = ticker.upper()
    found = False
    new_entries = []
    updated_entry = None

    for e in state.watchlist.entries:
        if e.ticker == ticker_upper:
            found = True
            updated_entry = WatchlistEntry(
                ticker=e.ticker,
                sector=sector if sector is not None else e.sector,
                enabled=enabled if enabled is not None else e.enabled,
                priority=priority if priority is not None else e.priority,
                tags=tuple(tags) if tags is not None else e.tags,
            )
            new_entries.append(updated_entry)
        else:
            new_entries.append(e)

    if not found:
        raise HTTPException(status_code=404, detail="Ticker not in watchlist")

    state.watchlist = Watchlist(
        entries=tuple(new_entries),
        name=state.watchlist.name,
        max_positions=state.watchlist.max_positions,
    )

    return WatchlistEntryResponse(
        ticker=updated_entry.ticker,
        sector=updated_entry.sector,
        enabled=updated_entry.enabled,
        priority=updated_entry.priority,
        tags=list(updated_entry.tags),
    )


@router.delete("/watchlist/entries/{ticker}")
async def remove_watchlist_entry(ticker: str):
    """Remove a ticker from the watchlist."""
    state = get_state()
    if state.watchlist is None:
        raise HTTPException(status_code=404, detail="Watchlist not configured")

    ticker_upper = ticker.upper()
    original_count = len(state.watchlist.entries)
    new_entries = tuple(e for e in state.watchlist.entries if e.ticker != ticker_upper)

    if len(new_entries) == original_count:
        raise HTTPException(status_code=404, detail="Ticker not in watchlist")

    state.watchlist = Watchlist(
        entries=new_entries,
        name=state.watchlist.name,
        max_positions=state.watchlist.max_positions,
    )

    return {"status": "ok", "ticker": ticker_upper, "remaining": len(new_entries)}


# ── Agent Weights ──────────────────────────────────────────────────

@router.get("/agent-weights", response_model=AgentWeightResponse)
async def get_agent_weights() -> AgentWeightResponse:
    """Get current agent synthesis weights."""
    state = get_state()
    if state.agent_weight_store is None:
        return AgentWeightResponse(weights=DEFAULT_WEIGHTS, is_default=True)

    config = state.agent_weight_store.get("default")
    return AgentWeightResponse(
        weights=config.normalized(),
        is_default=config.weights == DEFAULT_WEIGHTS,
    )


@router.put("/agent-weights", response_model=AgentWeightResponse)
async def update_agent_weights(req: AgentWeightUpdateRequest) -> AgentWeightResponse:
    """Update agent synthesis weights."""
    state = get_state()
    if state.agent_weight_store is None:
        raise HTTPException(status_code=503, detail="Weight store not initialized")

    config = state.agent_weight_store.set("default", req.weights)
    return AgentWeightResponse(
        weights=config.normalized(),
        is_default=config.weights == DEFAULT_WEIGHTS,
    )


@router.post("/agent-weights/reset", response_model=AgentWeightResponse)
async def reset_agent_weights() -> AgentWeightResponse:
    """Reset agent weights to defaults."""
    state = get_state()
    if state.agent_weight_store is None:
        return AgentWeightResponse(weights=DEFAULT_WEIGHTS, is_default=True)

    config = state.agent_weight_store.reset("default")
    return AgentWeightResponse(
        weights=config.normalized(),
        is_default=True,
    )


# ── Agent Preferences ─────────────────────────────────────────────

@router.get("/agent-preferences")
async def get_all_agent_preferences() -> dict[str, AgentPreferencesResponse]:
    """Get all agent preferences for the current user."""
    state = get_state()
    if state.agent_preferences_store is None:
        return {}

    prefs = state.agent_preferences_store.get_all("default")
    return {
        agent_id: AgentPreferencesResponse(
            agent_id=p.agent_id,
            time_horizon_days=p.time_horizon_days,
            risk_threshold=p.risk_threshold,
            regime_sensitivity=p.regime_sensitivity,
            sector_filters=p.sector_filters,
            enabled=p.enabled,
        )
        for agent_id, p in prefs.items()
    }


@router.get(
    "/agent-preferences/{agent_id}",
    response_model=AgentPreferencesResponse,
)
async def get_agent_preferences(agent_id: str) -> AgentPreferencesResponse:
    """Get preferences for a specific agent."""
    state = get_state()
    if state.agent_preferences_store is None:
        # Return defaults
        return AgentPreferencesResponse(agent_id=agent_id)

    prefs = state.agent_preferences_store.get("default", agent_id)
    if prefs is None:
        return AgentPreferencesResponse(agent_id=agent_id)

    return AgentPreferencesResponse(
        agent_id=prefs.agent_id,
        time_horizon_days=prefs.time_horizon_days,
        risk_threshold=prefs.risk_threshold,
        regime_sensitivity=prefs.regime_sensitivity,
        sector_filters=prefs.sector_filters,
        enabled=prefs.enabled,
    )


@router.put(
    "/agent-preferences/{agent_id}",
    response_model=AgentPreferencesResponse,
)
async def update_agent_preferences(
    agent_id: str,
    req: AgentPreferencesUpdateRequest,
) -> AgentPreferencesResponse:
    """Update preferences for a specific agent (partial update)."""
    state = get_state()
    if state.agent_preferences_store is None:
        raise HTTPException(status_code=503, detail="Preferences store not initialized")

    # Get existing or defaults
    existing = state.agent_preferences_store.get("default", agent_id)
    if existing is None:
        existing = AgentPreferences(agent_id=agent_id)

    # Merge partial update
    update_data: dict = {}
    if req.time_horizon_days is not None:
        update_data["time_horizon_days"] = req.time_horizon_days
    if req.risk_threshold is not None:
        update_data["risk_threshold"] = req.risk_threshold
    if req.regime_sensitivity is not None:
        update_data["regime_sensitivity"] = req.regime_sensitivity
    if req.sector_filters is not None:
        update_data["sector_filters"] = req.sector_filters
    if req.enabled is not None:
        update_data["enabled"] = req.enabled

    updated = existing.model_copy(update=update_data)
    stored = state.agent_preferences_store.set("default", agent_id, updated)

    return AgentPreferencesResponse(
        agent_id=stored.agent_id,
        time_horizon_days=stored.time_horizon_days,
        risk_threshold=stored.risk_threshold,
        regime_sensitivity=stored.regime_sensitivity,
        sector_filters=stored.sector_filters,
        enabled=stored.enabled,
    )


# ── Account Tiers ─────────────────────────────────────────────────

@router.get("/tiers", response_model=list[TierInfoResponse])
async def list_tiers() -> list[TierInfoResponse]:
    """List all account tiers with their limits."""
    result = []
    for tier, limits in TIER_LIMITS.items():
        result.append(
            TierInfoResponse(
                tier=tier.value,
                limits=TierLimitsResponse(**asdict(limits)),
            )
        )
    return result


@router.get("/tiers/{tier_name}", response_model=TierInfoResponse)
async def get_tier(tier_name: str) -> TierInfoResponse:
    """Get limits for a specific tier."""
    try:
        tier = AccountTier(tier_name.upper())
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid tier. Valid tiers: {[t.value for t in AccountTier]}",
        )
    limits = get_tier_limits(tier)
    return TierInfoResponse(
        tier=tier.value,
        limits=TierLimitsResponse(**asdict(limits)),
    )


@router.get("/account", response_model=AccountInfoResponse)
async def get_account_info() -> AccountInfoResponse:
    """Get current user account info and tier."""
    # For now, default to EXPLORER tier for all users
    # Multi-tenancy (F9-F10) will add proper user/tier resolution
    tier = AccountTier.EXPLORER
    state = get_state()
    user_tier = state.extra.get("account_tier", tier.value)
    try:
        resolved_tier = AccountTier(user_tier)
    except ValueError:
        resolved_tier = tier
    limits = get_tier_limits(resolved_tier)
    return AccountInfoResponse(
        user_id="default",
        tier=resolved_tier.value,
        limits=TierLimitsResponse(**asdict(limits)),
    )
