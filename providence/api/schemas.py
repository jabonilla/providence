"""API response/request schemas — Pydantic models for REST endpoints.

Separate from internal domain models to decouple API shape from internals.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, Field


# ── Health ──────────────────────────────────────────────────────────

class AgentHealthResponse(BaseModel):
    """Single agent health status."""

    agent_id: str
    status: str
    last_run: Optional[datetime] = None
    last_success: Optional[datetime] = None
    error_count_24h: int = 0
    avg_latency_ms: float = 0.0
    message: Optional[str] = None


class SystemHealthResponse(BaseModel):
    """Aggregated system health."""

    timestamp: datetime
    system_status: str
    agents: dict[str, int] = Field(
        description="Counts by status: total, healthy, degraded, unhealthy, offline",
    )
    pipeline: dict[str, Any] = Field(
        description="Run count and success rate",
    )
    agent_details: Optional[dict[str, AgentHealthResponse]] = None


# ── Pipeline ────────────────────────────────────────────────────────

class StageResultResponse(BaseModel):
    """Single stage execution result."""

    stage_name: str
    agent_id: str
    status: str
    started_at: datetime
    finished_at: datetime
    duration_ms: float
    error: Optional[str] = None


class PipelineRunResponse(BaseModel):
    """Pipeline execution summary."""

    run_id: UUID
    loop_type: str
    status: str
    started_at: datetime
    finished_at: Optional[datetime] = None
    stage_results: list[StageResultResponse]
    succeeded_count: int
    failed_count: int
    skipped_count: int
    total_duration_ms: float
    content_hash: str


class RunTriggerRequest(BaseModel):
    """Request to trigger a pipeline run."""

    run_exit: bool = Field(default=True, description="Include exit loop")
    run_governance: bool = Field(default=True, description="Include governance loop")


class RunTriggerResponse(BaseModel):
    """Response from a triggered pipeline run."""

    runs: dict[str, PipelineRunResponse]
    summary: str


# ── Agents ──────────────────────────────────────────────────────────

class AgentInfoResponse(BaseModel):
    """Agent identity and classification."""

    agent_id: str
    agent_type: str
    version: str
    subsystem: str
    classification: str  # FROZEN, ADAPTIVE, PERCEPTION


# ── Fragments ───────────────────────────────────────────────────────

class FragmentSummaryResponse(BaseModel):
    """Market state fragment summary (excludes payload for brevity)."""

    fragment_id: UUID
    agent_id: str
    timestamp: datetime
    entity: str
    data_type: str
    validation_status: str
    schema_version: str


class FragmentDetailResponse(FragmentSummaryResponse):
    """Full fragment including payload."""

    payload: dict[str, Any]


class FragmentStoreStatsResponse(BaseModel):
    """Fragment store statistics."""

    total_count: int
    by_type: dict[str, int]
    by_validation_status: dict[str, int]


# ── Beliefs ─────────────────────────────────────────────────────────

class BeliefSummaryResponse(BaseModel):
    """Belief object summary."""

    belief_id: UUID
    agent_id: str
    timestamp: datetime
    belief_count: int
    tickers: list[str]


class BeliefDetailResponse(BaseModel):
    """Full belief object."""

    belief_id: UUID
    agent_id: str
    timestamp: datetime
    context_window_hash: str
    beliefs: list[dict[str, Any]]


class BeliefStoreStatsResponse(BaseModel):
    """Belief store statistics."""

    total_count: int
    agents: list[str]
    tickers: list[str]


# ── Runs ────────────────────────────────────────────────────────────

class RunStoreStatsResponse(BaseModel):
    """Run store statistics."""

    total_count: int
    by_loop_type: dict[str, int]
    success_rate: float
    success_rate_by_loop: dict[str, float]


# ── Watchlist ───────────────────────────────────────────────────────

class WatchlistEntryResponse(BaseModel):
    """Single watchlist entry."""

    ticker: str
    sector: str
    enabled: bool
    priority: int
    tags: list[str]


class WatchlistResponse(BaseModel):
    """Full watchlist."""

    name: str
    max_positions: int
    entries: list[WatchlistEntryResponse]
    active_tickers: list[str]


# ── Portfolio ───────────────────────────────────────────────────────

class PositionResponse(BaseModel):
    """Single portfolio position."""

    ticker: str
    side: str
    quantity: str  # Decimal as string
    avg_entry_price: str
    current_price: str
    market_value: str
    unrealized_pnl: str
    unrealized_pnl_pct: float
    realized_pnl: str
    weight: float
    sector: str
    days_held: int


class PortfolioSnapshotResponse(BaseModel):
    """Portfolio snapshot."""

    snapshot_id: UUID
    timestamp: datetime
    equity: str
    cash: str
    buying_power: str
    positions: dict[str, PositionResponse]
    gross_exposure: float
    net_exposure: float
    long_exposure: float
    short_exposure: float
    sector_exposure: dict[str, float]
    position_count: int
    total_unrealized_pnl: str
    total_realized_pnl: str
    drawdown_pct: float


# ── Generic ─────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str
    detail: Optional[str] = None


class StatusResponse(BaseModel):
    """Simple status message."""

    status: str
    message: str
