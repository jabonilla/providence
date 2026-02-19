"""Learning subsystem schemas — offline attribution, calibration, and retraining.

Output types for LEARN-ATTRIB, LEARN-CALIB, LEARN-RETRAIN, and LEARN-BACKTEST.

Spec Reference: Technical Spec v2.3, Phase 4

Critical Rules:
  - No live learning. All retraining is offline.
  - Shadow mode before live deployment.
  - All data immutable and content-hashed.
"""

import hashlib
import json
from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# LEARN-ATTRIB output
# ---------------------------------------------------------------------------

class AgentAttribution(BaseModel):
    """Performance attribution for a single agent over an evaluation window."""

    model_config = ConfigDict(frozen=True)

    agent_id: str = Field(...)
    total_beliefs_produced: int = Field(default=0, ge=0)
    beliefs_acted_on: int = Field(default=0, ge=0)
    hit_rate: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Fraction of beliefs where direction was correct",
    )
    avg_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Average raw_confidence across beliefs",
    )
    avg_return_bps: float = Field(
        default=0.0,
        description="Average return in basis points for acted-on beliefs",
    )
    sharpe_contribution: float = Field(
        default=0.0,
        description="Agent's contribution to portfolio Sharpe ratio",
    )
    information_ratio: float = Field(
        default=0.0,
        description="Agent's information ratio (excess return / tracking error)",
    )
    max_drawdown_contribution_bps: float = Field(
        default=0.0, ge=0.0,
        description="Agent's contribution to max portfolio drawdown (bps)",
    )
    value_added_bps: float = Field(
        default=0.0,
        description="Net value added vs equal-weight benchmark (bps)",
    )


class TickerAttribution(BaseModel):
    """Performance attribution for a single ticker over an evaluation window."""

    model_config = ConfigDict(frozen=True)

    ticker: str = Field(...)
    total_pnl_bps: float = Field(default=0.0, description="Total PnL in basis points")
    holding_days: int = Field(default=0, ge=0)
    avg_weight: float = Field(default=0.0, ge=0.0, le=1.0)
    contributing_agents: list[str] = Field(
        default_factory=list,
        description="Agents whose beliefs contributed to this position",
    )
    exit_reason: str = Field(default="", description="Why the position was exited")
    regime_during_hold: str = Field(default="", description="Dominant regime during hold")


class AttributionOutput(BaseModel):
    """Complete output of LEARN-ATTRIB for one evaluation window.

    FROZEN agent. Offline-only performance attribution across agents,
    tickers, and regime contexts.
    Content-hashed.
    """

    model_config = ConfigDict(frozen=True)

    attribution_id: UUID = Field(default_factory=uuid4)
    agent_id: str = Field(...)
    timestamp: datetime = Field(...)
    context_window_hash: str = Field(...)
    evaluation_start: datetime = Field(...)
    evaluation_end: datetime = Field(...)
    agent_attributions: list[AgentAttribution] = Field(default_factory=list)
    ticker_attributions: list[TickerAttribution] = Field(default_factory=list)
    portfolio_sharpe: float = Field(default=0.0)
    portfolio_return_bps: float = Field(default=0.0)
    portfolio_max_drawdown_bps: float = Field(default=0.0, ge=0.0)
    total_trades: int = Field(default=0, ge=0)
    content_hash: str = Field(default="")

    @field_validator("timestamp", "evaluation_start", "evaluation_end")
    @classmethod
    def must_have_timezone(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("Timestamp must include timezone information")
        return v

    @model_validator(mode="after")
    def _compute_content_hash(self) -> "AttributionOutput":
        if not self.content_hash:
            data = {
                "agent_id": self.agent_id,
                "agent_attributions": [
                    {"agent_id": a.agent_id, "hit_rate": a.hit_rate, "value_added_bps": a.value_added_bps}
                    for a in self.agent_attributions
                ],
                "portfolio_return_bps": self.portfolio_return_bps,
            }
            serialized = json.dumps(data, sort_keys=True, default=str).encode("utf-8")
            object.__setattr__(self, "content_hash", hashlib.sha256(serialized).hexdigest())
        return self


# ---------------------------------------------------------------------------
# LEARN-CALIB output
# ---------------------------------------------------------------------------

class CalibrationBucket(BaseModel):
    """Calibration analysis for a confidence bucket.

    Compares stated confidence to realized accuracy to detect
    overconfidence or underconfidence.
    """

    model_config = ConfigDict(frozen=True)

    bucket_lower: float = Field(..., ge=0.0, le=1.0)
    bucket_upper: float = Field(..., ge=0.0, le=1.0)
    sample_count: int = Field(default=0, ge=0)
    avg_stated_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    realized_accuracy: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Fraction of beliefs that were directionally correct",
    )
    calibration_error: float = Field(
        default=0.0,
        description="avg_stated_confidence - realized_accuracy (positive = overconfident)",
    )
    brier_score: float = Field(
        default=0.0, ge=0.0,
        description="Mean squared difference between confidence and outcome",
    )


class AgentCalibration(BaseModel):
    """Calibration profile for a single agent."""

    model_config = ConfigDict(frozen=True)

    agent_id: str = Field(...)
    total_beliefs_evaluated: int = Field(default=0, ge=0)
    overall_brier_score: float = Field(
        default=0.0, ge=0.0,
        description="Overall Brier score (lower = better calibrated)",
    )
    overall_calibration_error: float = Field(
        default=0.0,
        description="Mean calibration error across buckets",
    )
    is_overconfident: bool = Field(
        default=False,
        description="Whether agent is systematically overconfident",
    )
    recommended_adjustment: float = Field(
        default=0.0,
        description="Suggested confidence multiplier (e.g. 0.85 = reduce by 15%)",
    )
    buckets: list[CalibrationBucket] = Field(default_factory=list)


class CalibrationOutput(BaseModel):
    """Complete output of LEARN-CALIB for one calibration run.

    FROZEN agent. Offline-only confidence calibration analysis.
    Content-hashed.
    """

    model_config = ConfigDict(frozen=True)

    calibration_id: UUID = Field(default_factory=uuid4)
    agent_id: str = Field(...)
    timestamp: datetime = Field(...)
    context_window_hash: str = Field(...)
    evaluation_start: datetime = Field(...)
    evaluation_end: datetime = Field(...)
    agent_calibrations: list[AgentCalibration] = Field(default_factory=list)
    system_brier_score: float = Field(
        default=0.0, ge=0.0,
        description="System-wide Brier score",
    )
    agents_overconfident: int = Field(default=0, ge=0)
    agents_underconfident: int = Field(default=0, ge=0)
    agents_well_calibrated: int = Field(default=0, ge=0)
    content_hash: str = Field(default="")

    @field_validator("timestamp", "evaluation_start", "evaluation_end")
    @classmethod
    def must_have_timezone(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("Timestamp must include timezone information")
        return v

    @model_validator(mode="after")
    def _compute_content_hash(self) -> "CalibrationOutput":
        if not self.content_hash:
            data = {
                "agent_id": self.agent_id,
                "agent_calibrations": [
                    {
                        "agent_id": a.agent_id,
                        "overall_brier_score": a.overall_brier_score,
                        "recommended_adjustment": a.recommended_adjustment,
                    }
                    for a in self.agent_calibrations
                ],
                "system_brier_score": self.system_brier_score,
            }
            serialized = json.dumps(data, sort_keys=True, default=str).encode("utf-8")
            object.__setattr__(self, "content_hash", hashlib.sha256(serialized).hexdigest())
        return self


# ---------------------------------------------------------------------------
# LEARN-RETRAIN output
# ---------------------------------------------------------------------------

class RetrainRecommendation(BaseModel):
    """Retraining recommendation for a single agent."""

    model_config = ConfigDict(frozen=True)

    agent_id: str = Field(...)
    needs_retrain: bool = Field(default=False)
    reason: str = Field(default="", description="Why retraining is needed")
    priority: str = Field(
        default="LOW",
        description="Retrain priority: LOW, MEDIUM, HIGH, CRITICAL",
    )
    performance_degradation_pct: float = Field(
        default=0.0, ge=0.0,
        description="Performance degradation vs baseline (%)",
    )
    suggested_changes: list[str] = Field(
        default_factory=list,
        description="Suggested prompt or parameter changes",
    )
    shadow_mode_required: bool = Field(
        default=True,
        description="Whether shadow mode is required before deployment",
    )


class RetrainOutput(BaseModel):
    """Complete output of LEARN-RETRAIN for one evaluation.

    FROZEN agent. Offline-only retraining recommendations.
    Content-hashed.
    """

    model_config = ConfigDict(frozen=True)

    retrain_id: UUID = Field(default_factory=uuid4)
    agent_id: str = Field(...)
    timestamp: datetime = Field(...)
    context_window_hash: str = Field(...)
    recommendations: list[RetrainRecommendation] = Field(default_factory=list)
    total_agents_evaluated: int = Field(default=0, ge=0)
    agents_needing_retrain: int = Field(default=0, ge=0)
    content_hash: str = Field(default="")

    @field_validator("timestamp")
    @classmethod
    def must_have_timezone(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("Timestamp must include timezone information")
        return v

    @model_validator(mode="after")
    def _compute_content_hash(self) -> "RetrainOutput":
        if not self.content_hash:
            data = {
                "agent_id": self.agent_id,
                "recommendations": [
                    {"agent_id": r.agent_id, "needs_retrain": r.needs_retrain, "priority": r.priority}
                    for r in self.recommendations
                ],
            }
            serialized = json.dumps(data, sort_keys=True, default=str).encode("utf-8")
            object.__setattr__(self, "content_hash", hashlib.sha256(serialized).hexdigest())
        return self


# ---------------------------------------------------------------------------
# LEARN-BACKTEST output
# ---------------------------------------------------------------------------

class BacktestPeriod(BaseModel):
    """Results for a single backtest sub-period (e.g., monthly)."""

    model_config = ConfigDict(frozen=True)

    period_start: datetime = Field(...)
    period_end: datetime = Field(...)
    return_bps: float = Field(default=0.0)
    sharpe_ratio: float = Field(default=0.0)
    max_drawdown_bps: float = Field(default=0.0, ge=0.0)
    trade_count: int = Field(default=0, ge=0)
    win_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    avg_holding_days: float = Field(default=0.0, ge=0.0)
    dominant_regime: str = Field(default="")

    @field_validator("period_start", "period_end")
    @classmethod
    def must_have_timezone(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("Timestamp must include timezone information")
        return v


class BacktestOutput(BaseModel):
    """Complete output of LEARN-BACKTEST for one backtest run.

    FROZEN agent. Offline-only backtesting over historical data.
    Content-hashed.
    """

    model_config = ConfigDict(frozen=True)

    backtest_id: UUID = Field(default_factory=uuid4)
    agent_id: str = Field(...)
    timestamp: datetime = Field(...)
    context_window_hash: str = Field(...)
    backtest_start: datetime = Field(...)
    backtest_end: datetime = Field(...)
    periods: list[BacktestPeriod] = Field(default_factory=list)
    total_return_bps: float = Field(default=0.0)
    annualized_return_bps: float = Field(default=0.0)
    annualized_sharpe: float = Field(default=0.0)
    max_drawdown_bps: float = Field(default=0.0, ge=0.0)
    total_trades: int = Field(default=0, ge=0)
    overall_win_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    profit_factor: float = Field(
        default=0.0, ge=0.0,
        description="Gross profits / gross losses",
    )
    content_hash: str = Field(default="")

    @field_validator("timestamp", "backtest_start", "backtest_end")
    @classmethod
    def must_have_timezone(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("Timestamp must include timezone information")
        return v

    @model_validator(mode="after")
    def _compute_content_hash(self) -> "BacktestOutput":
        if not self.content_hash:
            data = {
                "agent_id": self.agent_id,
                "total_return_bps": self.total_return_bps,
                "annualized_sharpe": self.annualized_sharpe,
                "total_trades": self.total_trades,
            }
            serialized = json.dumps(data, sort_keys=True, default=str).encode("utf-8")
            object.__setattr__(self, "content_hash", hashlib.sha256(serialized).hexdigest())
        return self
