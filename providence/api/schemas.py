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


class OrderResponse(BaseModel):
    """Single portfolio order."""

    order_id: UUID
    broker_order_id: Optional[str] = None
    client_order_id: str
    ticker: str
    side: str
    order_type: str
    time_in_force: str
    qty: Optional[str] = None
    notional: Optional[str] = None
    limit_price: Optional[str] = None
    status: str
    filled_qty: str = "0"
    filled_avg_price: str = "0"
    execution_strategy: str = "MARKET"
    target_weight: float = 0.0
    confidence: float = 0.0
    created_at: datetime
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    retry_count: int = 0
    last_error: Optional[str] = None


class OrderStatsResponse(BaseModel):
    """Order statistics."""

    total: int
    active: int
    by_status: dict[str, int]


# ── Shadow Mode ────────────────────────────────────────────────────

class ShadowSignalResponse(BaseModel):
    """Shadow mode trading signal."""

    signal_id: UUID
    run_id: UUID
    timestamp: datetime
    ticker: str
    action: str
    direction: str
    target_weight: float
    confidence: float
    approved: bool
    rejection_reasons: list[str] = Field(default_factory=list)
    adjusted_weight: float = 0.0
    risk_mode_applied: str = "NORMAL"
    simulated_entry_price: Optional[float] = None
    simulated_fill_qty: Optional[int] = None
    simulated_notional: Optional[float] = None
    price_at_signal: Optional[float] = None
    price_1d_later: Optional[float] = None
    price_5d_later: Optional[float] = None
    price_20d_later: Optional[float] = None
    realized_return_1d: Optional[float] = None
    realized_return_5d: Optional[float] = None
    realized_return_20d: Optional[float] = None


class ShadowRunSummaryResponse(BaseModel):
    """Summary of a single shadow pipeline run."""

    run_id: UUID
    timestamp: datetime
    system_mode: str
    total_signals: int
    approved_signals: int
    rejected_signals: int
    long_signals: int = 0
    short_signals: int = 0
    risk_mode: str = "NORMAL"
    regime_state: str = "LOW_VOL_TRENDING"


class ShadowStoreStatsResponse(BaseModel):
    """Shadow signal store statistics."""

    total_signals: int
    total_runs: int
    total_summaries: int
    unique_tickers: int
    approved_signals: int
    rejected_signals: int


class ShadowReportResponse(BaseModel):
    """Shadow mode performance report."""

    status: str
    generated_at: Optional[str] = None
    period_start: Optional[str] = None
    period_end: Optional[str] = None
    total_runs: int = 0
    total_signals: int = 0
    total_approved: int = 0
    total_rejected: int = 0
    total_longs: int = 0
    total_shorts: int = 0
    unique_tickers: int = 0
    avg_confidence: float = 0.0
    avg_signals_per_run: float = 0.0
    long_short_ratio: Optional[float] = None
    accuracy_1d: Optional[float] = None
    accuracy_5d: Optional[float] = None
    accuracy_20d: Optional[float] = None
    hypothetical_return_1d: Optional[float] = None
    hypothetical_return_5d: Optional[float] = None
    hypothetical_return_20d: Optional[float] = None
    phase_b_criteria: dict[str, bool] = Field(default_factory=dict)
    ticker_breakdown: dict[str, dict[str, Any]] = Field(default_factory=dict)


class BackfillTriggerResponse(BaseModel):
    """Response from a price backfill run."""

    processed: int
    updated: int
    errors: int
    skipped: int
    prices_fetched: int = 0


# ── Regime ─────────────────────────────────────────────────────────

class SectorRegimeOverlayResponse(BaseModel):
    """Sector-level regime overlay."""

    sector: str
    regime: str = Field(description="StatisticalRegime enum value")
    regime_confidence: float = Field(ge=0.0, le=1.0)
    regime_probabilities: dict[str, float] = Field(default_factory=dict)
    relative_stress: float = Field(
        default=0.0,
        description="Relative stress vs market: -1.0 (calmer) to +1.0 (more stressed)",
    )
    key_signals: list[str] = Field(default_factory=list)
    ticker_count: int = Field(default=0, ge=0)


class RegimeStateResponse(BaseModel):
    """Full regime state including global regime, narrative, and sector overlays."""

    statistical_regime: str = Field(description="Global regime classification")
    regime_confidence: float = Field(ge=0.0, le=1.0)
    regime_probabilities: dict[str, float] = Field(default_factory=dict)
    system_risk_mode: str = Field(description="NORMAL/CAUTIOUS/DEFENSIVE/HALTED")
    sector_overlays: list[SectorRegimeOverlayResponse] = Field(default_factory=list)
    narrative_label: Optional[str] = None
    narrative_confidence: Optional[float] = None
    narrative_key_signals: list[str] = Field(default_factory=list)
    narrative_affected_sectors: list[str] = Field(default_factory=list)
    narrative_summary: Optional[str] = None
    run_id: str = Field(description="Pipeline run that produced this regime state")
    timestamp: str = Field(description="When this regime was computed")


# ── Chat ──────────────────────────────────────────────────────────

class ChatMessageRequest(BaseModel):
    """Request to send a chat message."""

    message: str = Field(min_length=1, max_length=2000, description="User message text")
    conversation_id: Optional[str] = Field(
        default=None,
        description="Existing conversation ID. If null, a new conversation is created.",
    )


class ChatCitation(BaseModel):
    """A structured citation referencing a Providence resource."""

    type: str = Field(description="Resource type: position, belief, regime, agent, pipeline")
    id: str = Field(description="Resource identifier")
    label: str = Field(description="Human-readable label")
    url: str = Field(description="API URL to the cited resource")


class ChatMessageInResponse(BaseModel):
    """Single message returned in a chat response."""

    id: str = Field(description="Unique message ID")
    role: str = Field(description="Message role: user or assistant")
    content: str = Field(description="Message text content")
    citations: list[ChatCitation] = Field(default_factory=list)
    timestamp: datetime


class ChatSendResponse(BaseModel):
    """Response from sending a chat message."""

    message: ChatMessageInResponse
    conversation_id: str = Field(description="Conversation ID (new or existing)")


class ChatMessageResponse(BaseModel):
    """Response from the chat engine (legacy shape, kept for compatibility)."""

    response: str = Field(description="Natural language response text")
    citations: list[ChatCitation] = Field(default_factory=list)
    conversation_id: str = Field(description="Conversation ID (new or existing)")
    timestamp: datetime


class ConversationMessage(BaseModel):
    """A single message in a conversation."""

    role: str = Field(description="Message role: user or assistant")
    content: str
    citations: list[ChatCitation] = Field(default_factory=list)
    timestamp: datetime


class ConversationSummary(BaseModel):
    """Summary of a conversation for listing."""

    id: str
    title: str
    message_count: int
    last_message_at: Optional[datetime] = None
    created_at: datetime


class ConversationDetail(BaseModel):
    """Full conversation with message history."""

    id: str
    title: str
    messages: list[ConversationMessage]
    created_at: datetime


# ── Document Upload ──────────────────────────────────────────────────

class DocumentUploadResponse(BaseModel):
    """Response from document upload."""

    status: str
    filename: str
    fragment_id: str
    content_type: str
    text_length: int
    message: str


# ── Agent Weights ──────────────────────────────────────────────────

class AgentWeightResponse(BaseModel):
    """Current agent synthesis weights."""

    weights: dict[str, float]
    is_default: bool = True


class AgentWeightUpdateRequest(BaseModel):
    """Request to update agent weights."""

    weights: dict[str, float]


# ── Agent Preferences ─────────────────────────────────────────────

class AgentPreferencesResponse(BaseModel):
    """Agent configuration preferences."""

    agent_id: str
    time_horizon_days: int = 30
    risk_threshold: float = 0.5
    regime_sensitivity: float = 1.0
    sector_filters: list[str] = Field(default_factory=list)
    enabled: bool = True


class AgentPreferencesUpdateRequest(BaseModel):
    """Partial update for agent preferences."""

    time_horizon_days: Optional[int] = None
    risk_threshold: Optional[float] = None
    regime_sensitivity: Optional[float] = None
    sector_filters: Optional[list[str]] = None
    enabled: Optional[bool] = None


# ── Account Tiers ─────────────────────────────────────────────────

class TierLimitsResponse(BaseModel):
    """Resource limits for a tier."""

    max_agents: int
    max_positions: int
    custom_weights: bool
    custom_config: bool
    api_access: bool
    max_watchlist: int
    shadow_mode: bool
    paper_trading: bool
    live_trading: bool
    doc_uploads_per_day: int


class TierInfoResponse(BaseModel):
    """Tier with its limits."""

    tier: str
    limits: TierLimitsResponse


class AccountInfoResponse(BaseModel):
    """Current user account info."""

    user_id: str
    tier: str
    limits: TierLimitsResponse


# ── Generic ─────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str
    detail: Optional[str] = None


class StatusResponse(BaseModel):
    """Simple status message."""

    status: str
    message: str
