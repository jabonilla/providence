"""Exit system schemas — structured outputs for position lifecycle management.

Output types for COGNIT-EXIT, INVALID-MON, THESIS-RENEW, SHADOW-EXIT, and RENEW-MON.

Spec Reference: Technical Spec v2.3, Sections 2.8, Phase 3

Critical Rules:
  - COGNIT-EXIT defers CLOSE if renewal pending AND asymmetry > 0.5
  - EXEC-CAPTURE has SUPREMACY — overrides COGNIT-EXIT decisions
  - Max 3 trim stages, then mandatory CLOSE
  - trim_pct applies to REMAINING position, not original
  - All invalidation conditions machine-evaluable
"""

import hashlib
import json
from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# COGNIT-EXIT output
# ---------------------------------------------------------------------------

class ExitAssessment(BaseModel):
    """Per-position exit assessment from COGNIT-EXIT.

    Spec Reference: Section 2.8 (ExitAssessment)
    - exit_confidence [0-1]: confidence that position should be exited
    - regret_estimate_bps: estimated regret of exiting now (basis points)
    - regret_direction: MISSED_UPSIDE or SUFFERED_GIVEBACK
    """

    model_config = ConfigDict(frozen=True)

    ticker: str = Field(..., description="Ticker being assessed")
    exit_action: str = Field(
        ..., description="Recommended action: HOLD, REDUCE, or EXIT",
    )
    exit_confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="Confidence that position should be exited",
    )
    regret_estimate_bps: float = Field(
        default=0.0, ge=0.0,
        description="Estimated regret of exiting now in basis points",
    )
    regret_direction: str = Field(
        default="MISSED_UPSIDE",
        description="Direction of regret: MISSED_UPSIDE or SUFFERED_GIVEBACK",
    )
    thesis_health_score: float = Field(
        default=1.0, ge=0.0, le=1.0,
        description="Overall thesis health (1.0 = fully healthy)",
    )
    conditions_triggered: int = Field(
        default=0, ge=0,
        description="Number of invalidation conditions triggered",
    )
    conditions_total: int = Field(
        default=0, ge=0,
        description="Total number of invalidation conditions",
    )
    renewal_pending: bool = Field(
        default=False,
        description="Whether a thesis renewal is pending for this position",
    )
    renewal_asymmetry: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Renewal asymmetry score (>0.5 defers CLOSE)",
    )
    rationale: str = Field(
        default="", description="LLM-generated rationale for exit assessment",
    )


class ExitOutput(BaseModel):
    """Complete output of COGNIT-EXIT for one cycle.

    COGNIT-EXIT is ADAPTIVE (uses LLM). Its decisions are subject to
    EXEC-CAPTURE supremacy — trailing stop overrides everything.
    Content-hashed.
    """

    model_config = ConfigDict(frozen=True)

    exit_id: UUID = Field(default_factory=uuid4)
    agent_id: str = Field(...)
    timestamp: datetime = Field(...)
    context_window_hash: str = Field(...)
    assessments: list[ExitAssessment] = Field(default_factory=list)
    positions_hold: int = Field(default=0, ge=0)
    positions_reduce: int = Field(default=0, ge=0)
    positions_exit: int = Field(default=0, ge=0)
    content_hash: str = Field(default="")

    @field_validator("timestamp")
    @classmethod
    def must_have_timezone(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("Timestamp must include timezone information")
        return v

    @model_validator(mode="after")
    def _compute_content_hash(self) -> "ExitOutput":
        if not self.content_hash:
            data = {
                "agent_id": self.agent_id,
                "assessments": [
                    {
                        "ticker": a.ticker,
                        "exit_action": a.exit_action,
                        "exit_confidence": a.exit_confidence,
                    }
                    for a in self.assessments
                ],
            }
            serialized = json.dumps(data, sort_keys=True, default=str).encode("utf-8")
            object.__setattr__(self, "content_hash", hashlib.sha256(serialized).hexdigest())
        return self


# ---------------------------------------------------------------------------
# INVALID-MON output
# ---------------------------------------------------------------------------

class MonitoredCondition(BaseModel):
    """Result of monitoring a single invalidation condition."""

    model_config = ConfigDict(frozen=True)

    condition_id: UUID = Field(...)
    source_thesis_id: str = Field(..., description="Thesis that owns this condition")
    source_agent_id: str = Field(..., description="Agent that produced the thesis")
    ticker: str = Field(...)
    metric: str = Field(..., description="Metric being monitored")
    operator: str = Field(..., description="Comparison operator")
    threshold: float = Field(...)
    current_value: Optional[float] = Field(default=None)
    is_breached: bool = Field(default=False, description="Whether condition is breached")
    breach_magnitude: float = Field(
        default=0.0, ge=0.0,
        description="|current - threshold| / |threshold| (0 = at threshold)",
    )
    breach_velocity: float = Field(
        default=0.0,
        description="Rate of approach per day (trailing 5-day avg)",
    )
    confidence_impact: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Suggested confidence reduction (0.0-1.0)",
    )


class InvalidationMonitorOutput(BaseModel):
    """Complete output of INVALID-MON for one monitoring cycle.

    FROZEN agent. Evaluates all active invalidation conditions
    across active positions against current market data.
    Content-hashed.
    """

    model_config = ConfigDict(frozen=True)

    monitor_id: UUID = Field(default_factory=uuid4)
    agent_id: str = Field(...)
    timestamp: datetime = Field(...)
    context_window_hash: str = Field(...)
    conditions: list[MonitoredCondition] = Field(default_factory=list)
    total_conditions: int = Field(default=0, ge=0)
    conditions_breached: int = Field(default=0, ge=0)
    conditions_approaching: int = Field(
        default=0, ge=0,
        description="Conditions within 10% of threshold",
    )
    content_hash: str = Field(default="")

    @field_validator("timestamp")
    @classmethod
    def must_have_timezone(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("Timestamp must include timezone information")
        return v

    @model_validator(mode="after")
    def _compute_content_hash(self) -> "InvalidationMonitorOutput":
        if not self.content_hash:
            data = {
                "agent_id": self.agent_id,
                "conditions": [
                    {
                        "ticker": c.ticker,
                        "metric": c.metric,
                        "is_breached": c.is_breached,
                        "breach_magnitude": c.breach_magnitude,
                    }
                    for c in self.conditions
                ],
            }
            serialized = json.dumps(data, sort_keys=True, default=str).encode("utf-8")
            object.__setattr__(self, "content_hash", hashlib.sha256(serialized).hexdigest())
        return self


# ---------------------------------------------------------------------------
# THESIS-RENEW output
# ---------------------------------------------------------------------------

class RenewalCandidate(BaseModel):
    """A thesis eligible for renewal with updated parameters."""

    model_config = ConfigDict(frozen=True)

    thesis_id: str = Field(..., description="Thesis being renewed")
    ticker: str = Field(...)
    original_confidence: float = Field(..., ge=0.0, le=1.0)
    renewed_confidence: float = Field(..., ge=0.0, le=1.0)
    confidence_delta: float = Field(
        default=0.0,
        description="renewed - original confidence",
    )
    original_horizon_days: int = Field(..., gt=0)
    renewed_horizon_days: int = Field(..., gt=0)
    renewal_reason: str = Field(default="", description="Why the thesis is being renewed")
    asymmetry_score: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Upside asymmetry score (>0.5 favors renewal)",
    )
    conditions_healthy: int = Field(default=0, ge=0)
    conditions_total: int = Field(default=0, ge=0)
    is_renewed: bool = Field(default=False, description="Whether renewal is approved")


class ThesisRenewalOutput(BaseModel):
    """Complete output of THESIS-RENEW for one cycle.

    FROZEN agent. Evaluates thesis renewal candidates based on
    time decay, condition health, regime alignment, and asymmetry.
    Content-hashed.
    """

    model_config = ConfigDict(frozen=True)

    renewal_id: UUID = Field(default_factory=uuid4)
    agent_id: str = Field(...)
    timestamp: datetime = Field(...)
    context_window_hash: str = Field(...)
    candidates: list[RenewalCandidate] = Field(default_factory=list)
    total_evaluated: int = Field(default=0, ge=0)
    total_renewed: int = Field(default=0, ge=0)
    total_expired: int = Field(default=0, ge=0)
    content_hash: str = Field(default="")

    @field_validator("timestamp")
    @classmethod
    def must_have_timezone(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("Timestamp must include timezone information")
        return v

    @model_validator(mode="after")
    def _compute_content_hash(self) -> "ThesisRenewalOutput":
        if not self.content_hash:
            data = {
                "agent_id": self.agent_id,
                "candidates": [
                    {
                        "thesis_id": c.thesis_id,
                        "ticker": c.ticker,
                        "is_renewed": c.is_renewed,
                        "renewed_confidence": c.renewed_confidence,
                    }
                    for c in self.candidates
                ],
            }
            serialized = json.dumps(data, sort_keys=True, default=str).encode("utf-8")
            object.__setattr__(self, "content_hash", hashlib.sha256(serialized).hexdigest())
        return self


# ---------------------------------------------------------------------------
# SHADOW-EXIT output
# ---------------------------------------------------------------------------

class ShadowExitSignal(BaseModel):
    """Shadow exit tracking for a single position.

    Compares COGNIT-EXIT recommendations against EXEC-CAPTURE decisions
    to detect agreement/divergence and estimate exit probability.
    """

    model_config = ConfigDict(frozen=True)

    ticker: str = Field(...)
    cognit_exit_action: str = Field(
        ..., description="COGNIT-EXIT recommendation: HOLD, REDUCE, EXIT",
    )
    capture_action: str = Field(
        ..., description="EXEC-CAPTURE decision: HOLD, TRIM, CLOSE",
    )
    signals_agree: bool = Field(
        default=False,
        description="Whether both systems agree on exit direction",
    )
    exit_probability: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Combined exit probability",
    )
    divergence_reason: str = Field(
        default="",
        description="Why signals diverge (empty if they agree)",
    )
    days_in_shadow: int = Field(
        default=0, ge=0,
        description="Days this position has been in shadow-exit state",
    )


class ShadowExitOutput(BaseModel):
    """Complete output of SHADOW-EXIT for one cycle.

    FROZEN agent. Tracks positions where exit signals are developing
    but not yet triggered. Detects COGNIT-EXIT vs EXEC-CAPTURE divergence.
    Content-hashed.
    """

    model_config = ConfigDict(frozen=True)

    shadow_id: UUID = Field(default_factory=uuid4)
    agent_id: str = Field(...)
    timestamp: datetime = Field(...)
    context_window_hash: str = Field(...)
    signals: list[ShadowExitSignal] = Field(default_factory=list)
    total_positions: int = Field(default=0, ge=0)
    positions_agreeing: int = Field(default=0, ge=0)
    positions_diverging: int = Field(default=0, ge=0)
    content_hash: str = Field(default="")

    @field_validator("timestamp")
    @classmethod
    def must_have_timezone(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("Timestamp must include timezone information")
        return v

    @model_validator(mode="after")
    def _compute_content_hash(self) -> "ShadowExitOutput":
        if not self.content_hash:
            data = {
                "agent_id": self.agent_id,
                "signals": [
                    {
                        "ticker": s.ticker,
                        "signals_agree": s.signals_agree,
                        "exit_probability": s.exit_probability,
                    }
                    for s in self.signals
                ],
            }
            serialized = json.dumps(data, sort_keys=True, default=str).encode("utf-8")
            object.__setattr__(self, "content_hash", hashlib.sha256(serialized).hexdigest())
        return self


# ---------------------------------------------------------------------------
# RENEW-MON output
# ---------------------------------------------------------------------------

class BeliefHealthReport(BaseModel):
    """Health report for a single active belief/thesis."""

    model_config = ConfigDict(frozen=True)

    thesis_id: str = Field(...)
    ticker: str = Field(...)
    agent_id: str = Field(..., description="Agent that produced the thesis")
    health_score: float = Field(
        ..., ge=0.0, le=1.0,
        description="Overall belief health (1.0 = fully healthy)",
    )
    conditions_healthy: int = Field(default=0, ge=0)
    conditions_breached: int = Field(default=0, ge=0)
    conditions_approaching: int = Field(default=0, ge=0)
    days_remaining: int = Field(
        default=0, ge=0,
        description="Days remaining on thesis time horizon",
    )
    confidence_decay: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Time-based confidence decay factor",
    )
    is_renewal_candidate: bool = Field(
        default=False,
        description="Whether this thesis should be considered for renewal",
    )
    renewal_urgency: str = Field(
        default="NONE",
        description="Renewal urgency: NONE, LOW, MEDIUM, HIGH",
    )


class RenewalMonitorOutput(BaseModel):
    """Complete output of RENEW-MON for one cycle.

    FROZEN agent. Aggregates condition health, time decay, and renewal
    status across all active beliefs. Identifies renewal candidates.
    Content-hashed.
    """

    model_config = ConfigDict(frozen=True)

    monitor_id: UUID = Field(default_factory=uuid4)
    agent_id: str = Field(...)
    timestamp: datetime = Field(...)
    context_window_hash: str = Field(...)
    reports: list[BeliefHealthReport] = Field(default_factory=list)
    total_beliefs: int = Field(default=0, ge=0)
    healthy_beliefs: int = Field(default=0, ge=0)
    degraded_beliefs: int = Field(default=0, ge=0)
    critical_beliefs: int = Field(default=0, ge=0)
    renewal_candidates: int = Field(default=0, ge=0)
    content_hash: str = Field(default="")

    @field_validator("timestamp")
    @classmethod
    def must_have_timezone(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("Timestamp must include timezone information")
        return v

    @model_validator(mode="after")
    def _compute_content_hash(self) -> "RenewalMonitorOutput":
        if not self.content_hash:
            data = {
                "agent_id": self.agent_id,
                "reports": [
                    {
                        "thesis_id": r.thesis_id,
                        "ticker": r.ticker,
                        "health_score": r.health_score,
                        "is_renewal_candidate": r.is_renewal_candidate,
                    }
                    for r in self.reports
                ],
            }
            serialized = json.dumps(data, sort_keys=True, default=str).encode("utf-8")
            object.__setattr__(self, "content_hash", hashlib.sha256(serialized).hexdigest())
        return self
