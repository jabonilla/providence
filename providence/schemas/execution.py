"""Execution subsystem schemas — validated orders, routing, and trailing stops.

Output types for EXEC-VALIDATE, EXEC-ROUTER, EXEC-GUARDIAN, and EXEC-CAPTURE.

Spec Reference: Technical Spec v2.3, Sections 2.6-2.9
"""

import hashlib
import json
from datetime import datetime
from typing import Any, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from providence.schemas.enums import Action, Direction, SystemRiskMode, ExecutionStatus


# ---------------------------------------------------------------------------
# EXEC-VALIDATE output
# ---------------------------------------------------------------------------

class ValidationResult(BaseModel):
    """Result of pre-trade validation for a single proposed position.

    Produced by EXEC-VALIDATE. Consumed by EXEC-ROUTER.
    """

    model_config = ConfigDict(frozen=True)

    ticker: str = Field(..., description="Ticker being validated")
    action: Action = Field(..., description="Proposed action")
    direction: Direction = Field(..., description="Proposed direction")
    target_weight: float = Field(..., ge=0.0, le=0.20)
    confidence: float = Field(..., ge=0.0, le=1.0)
    source_intent_id: UUID = Field(..., description="Original intent ID")
    approved: bool = Field(..., description="Whether this position passed validation")
    rejection_reasons: list[str] = Field(
        default_factory=list,
        description="Reasons for rejection (empty if approved)",
    )
    adjusted_weight: float = Field(
        default=0.0, ge=0.0, le=0.20,
        description="Weight after constraint adjustments (may differ from target)",
    )


class ValidatedProposal(BaseModel):
    """Complete output of EXEC-VALIDATE for one validation cycle.

    Contains per-position validation results and aggregate stats.
    Content-hashed.
    """

    model_config = ConfigDict(frozen=True)

    validation_id: UUID = Field(default_factory=uuid4)
    agent_id: str = Field(...)
    timestamp: datetime = Field(...)
    context_window_hash: str = Field(...)
    results: list[ValidationResult] = Field(default_factory=list)
    approved_count: int = Field(default=0, ge=0)
    rejected_count: int = Field(default=0, ge=0)
    risk_mode_applied: str = Field(default="NORMAL")
    content_hash: str = Field(default="")

    @field_validator("timestamp")
    @classmethod
    def must_have_timezone(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("Timestamp must include timezone information")
        return v

    @model_validator(mode="after")
    def _compute_content_hash(self) -> "ValidatedProposal":
        if not self.content_hash:
            data = {
                "agent_id": self.agent_id,
                "results": [
                    {"ticker": r.ticker, "approved": r.approved, "weight": r.adjusted_weight}
                    for r in self.results
                ],
            }
            serialized = json.dumps(data, sort_keys=True, default=str).encode("utf-8")
            object.__setattr__(self, "content_hash", hashlib.sha256(serialized).hexdigest())
        return self


# ---------------------------------------------------------------------------
# EXEC-ROUTER output
# ---------------------------------------------------------------------------

class RoutedOrder(BaseModel):
    """A single order with routing metadata.

    Produced by EXEC-ROUTER. Consumed by EXEC-GUARDIAN.
    """

    model_config = ConfigDict(frozen=True)

    order_id: UUID = Field(default_factory=uuid4)
    ticker: str = Field(...)
    action: Action = Field(...)
    direction: Direction = Field(...)
    target_weight: float = Field(..., ge=0.0, le=0.20)
    confidence: float = Field(..., ge=0.0, le=1.0)
    source_intent_id: UUID = Field(...)
    execution_strategy: str = Field(
        default="TWAP",
        description="Execution algo: TWAP, VWAP, LIMIT, MARKET",
    )
    urgency: str = Field(
        default="NORMAL",
        description="Execution urgency: LOW, NORMAL, HIGH, IMMEDIATE",
    )
    time_horizon_days: int = Field(default=60, gt=0)
    max_slippage_bps: int = Field(
        default=50, ge=0,
        description="Maximum acceptable slippage in basis points",
    )


class RoutingPlan(BaseModel):
    """Complete output of EXEC-ROUTER for one routing cycle.

    Content-hashed.
    """

    model_config = ConfigDict(frozen=True)

    routing_id: UUID = Field(default_factory=uuid4)
    agent_id: str = Field(...)
    timestamp: datetime = Field(...)
    context_window_hash: str = Field(...)
    orders: list[RoutedOrder] = Field(default_factory=list)
    total_orders: int = Field(default=0, ge=0)
    risk_mode_applied: str = Field(default="NORMAL")
    content_hash: str = Field(default="")

    @field_validator("timestamp")
    @classmethod
    def must_have_timezone(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("Timestamp must include timezone information")
        return v

    @model_validator(mode="after")
    def _compute_content_hash(self) -> "RoutingPlan":
        if not self.content_hash:
            data = {
                "agent_id": self.agent_id,
                "orders": [
                    {"ticker": o.ticker, "action": o.action.value, "strategy": o.execution_strategy}
                    for o in self.orders
                ],
            }
            serialized = json.dumps(data, sort_keys=True, default=str).encode("utf-8")
            object.__setattr__(self, "content_hash", hashlib.sha256(serialized).hexdigest())
        return self


# ---------------------------------------------------------------------------
# EXEC-GUARDIAN output
# ---------------------------------------------------------------------------

class GuardianCheck(BaseModel):
    """Result of kill-switch / circuit-breaker check for a single order."""

    model_config = ConfigDict(frozen=True)

    order_id: UUID = Field(...)
    ticker: str = Field(...)
    approved: bool = Field(...)
    halt_reason: str = Field(default="", description="Reason for halt (empty if approved)")


class GuardianVerdict(BaseModel):
    """Complete output of EXEC-GUARDIAN for one cycle.

    Content-hashed.
    """

    model_config = ConfigDict(frozen=True)

    verdict_id: UUID = Field(default_factory=uuid4)
    agent_id: str = Field(...)
    timestamp: datetime = Field(...)
    context_window_hash: str = Field(...)
    checks: list[GuardianCheck] = Field(default_factory=list)
    system_halt: bool = Field(default=False, description="Whether system-wide halt is triggered")
    halt_reason: str = Field(default="")
    approved_count: int = Field(default=0, ge=0)
    halted_count: int = Field(default=0, ge=0)
    risk_mode_applied: str = Field(default="NORMAL")
    content_hash: str = Field(default="")

    @field_validator("timestamp")
    @classmethod
    def must_have_timezone(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("Timestamp must include timezone information")
        return v

    @model_validator(mode="after")
    def _compute_content_hash(self) -> "GuardianVerdict":
        if not self.content_hash:
            data = {
                "agent_id": self.agent_id,
                "system_halt": self.system_halt,
                "checks": [
                    {"ticker": c.ticker, "approved": c.approved}
                    for c in self.checks
                ],
            }
            serialized = json.dumps(data, sort_keys=True, default=str).encode("utf-8")
            object.__setattr__(self, "content_hash", hashlib.sha256(serialized).hexdigest())
        return self


# ---------------------------------------------------------------------------
# EXEC-CAPTURE output
# ---------------------------------------------------------------------------

class TrailingStopState(BaseModel):
    """Per-position trailing stop state.

    Spec: Activation at 2.0x expected_return, trail at 30%/20%,
    hard giveback at 50%, minimum hold 5 days.
    """

    model_config = ConfigDict(frozen=True)

    ticker: str = Field(...)
    is_active: bool = Field(default=False, description="Whether trailing stop is activated")
    peak_unrealized_pnl: float = Field(default=0.0, description="Peak unrealized PnL (high water mark)")
    current_unrealized_pnl: float = Field(default=0.0, description="Current unrealized PnL")
    trail_pct: float = Field(default=0.30, description="Trail percentage (0.20 or 0.30)")
    trigger_level: float = Field(default=0.0, description="PnL level that triggers exit")
    days_held: int = Field(default=0, ge=0, description="Days position has been held")
    trim_stage: int = Field(default=0, ge=0, le=3, description="Current trim stage (max 3)")


class CaptureDecision(BaseModel):
    """Per-position exit/hold decision from EXEC-CAPTURE."""

    model_config = ConfigDict(frozen=True)

    ticker: str = Field(...)
    action: str = Field(
        ..., description="HOLD, TRIM, or CLOSE",
    )
    trim_pct: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Percentage of REMAINING position to trim",
    )
    exit_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Confidence in exit decision",
    )
    reason: str = Field(default="", description="Reason for the decision")
    trailing_stop: TrailingStopState = Field(
        ..., description="Current trailing stop state",
    )


class CaptureOutput(BaseModel):
    """Complete output of EXEC-CAPTURE for one cycle.

    EXEC-CAPTURE has supremacy: its decisions override all upstream agents.
    Content-hashed.
    """

    model_config = ConfigDict(frozen=True)

    capture_id: UUID = Field(default_factory=uuid4)
    agent_id: str = Field(...)
    timestamp: datetime = Field(...)
    context_window_hash: str = Field(...)
    decisions: list[CaptureDecision] = Field(default_factory=list)
    positions_held: int = Field(default=0, ge=0)
    positions_trimmed: int = Field(default=0, ge=0)
    positions_closed: int = Field(default=0, ge=0)
    content_hash: str = Field(default="")

    @field_validator("timestamp")
    @classmethod
    def must_have_timezone(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("Timestamp must include timezone information")
        return v

    @model_validator(mode="after")
    def _compute_content_hash(self) -> "CaptureOutput":
        if not self.content_hash:
            data = {
                "agent_id": self.agent_id,
                "decisions": [
                    {"ticker": d.ticker, "action": d.action, "trim_pct": d.trim_pct}
                    for d in self.decisions
                ],
            }
            serialized = json.dumps(data, sort_keys=True, default=str).encode("utf-8")
            object.__setattr__(self, "content_hash", hashlib.sha256(serialized).hexdigest())
        return self


# ---------------------------------------------------------------------------
# EXECUTION REPORT output (Spec Section 2.6)
# ---------------------------------------------------------------------------

class ExecutionReport(BaseModel):
    """Complete execution report for a filled/partial/rejected position.

    Links back to the PositionProposal and tracks actual fill price, slippage,
    market impact, and any constraint violations during execution.
    Content-hashed.
    """

    model_config = ConfigDict(frozen=True)

    execution_id: UUID = Field(default_factory=uuid4)
    proposal_id: UUID = Field(..., description="Links to PositionProposal")
    ticker: str = Field(...)
    status: ExecutionStatus = Field(...)
    requested_weight: float = Field(..., ge=0.0, le=0.20)
    achieved_weight: float = Field(..., ge=0.0, le=0.20)
    avg_fill_price: float = Field(..., gt=0.0)
    benchmark_price: float = Field(..., gt=0.0, description="Arrival price for slippage calc")
    slippage_bps: float = Field(..., ge=0.0, description="Basis points")
    market_impact_bps: float = Field(..., ge=0.0, description="Estimated permanent impact")
    execution_algo: str = Field(
        ..., description="TWAP_15, VWAP_30, LIMIT, or MARKET"
    )
    execution_duration_ms: int = Field(..., ge=0)
    venue: str = Field(..., description="Exchange or venue name")
    fees_bps: float = Field(..., ge=0.0)
    constraint_violations: list[str] = Field(
        default_factory=list,
        description="Should be empty if execution succeeded",
    )
    timestamp_start: datetime = Field(...)
    timestamp_end: datetime = Field(...)
    content_hash: str = Field(default="")

    @field_validator("timestamp_start", "timestamp_end")
    @classmethod
    def must_have_timezone(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("Timestamp must include timezone information")
        return v

    @model_validator(mode="after")
    def _compute_content_hash(self) -> "ExecutionReport":
        if not self.content_hash:
            data = {
                "proposal_id": str(self.proposal_id),
                "ticker": self.ticker,
                "status": self.status.value,
                "achieved_weight": self.achieved_weight,
                "avg_fill_price": self.avg_fill_price,
                "slippage_bps": self.slippage_bps,
            }
            serialized = json.dumps(data, sort_keys=True, default=str).encode("utf-8")
            object.__setattr__(self, "content_hash", hashlib.sha256(serialized).hexdigest())
        return self
