"""Governance subsystem schemas — capital tiers, maturity gates, oversight.

Output types for GOVERN-CAPITAL, GOVERN-MATURITY, GOVERN-OVERSIGHT, and GOVERN-POLICY.

Spec Reference: Technical Spec v2.3, Phase 4 (Governance)

Critical Rules:
  - Human-in-the-loop for all tier/maturity transitions.
  - All governance decisions are immutable and content-hashed.
  - Shadow mode before live deployment for all adaptive agents.
  - Capital tier determines system autonomy level.
"""

import hashlib
import json
from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from providence.schemas.enums import (
    CapitalTier,
    IncidentSeverity,
    MaturityStage,
    SystemRiskMode,
)


# ---------------------------------------------------------------------------
# GOVERN-CAPITAL output
# ---------------------------------------------------------------------------

class TierConstraints(BaseModel):
    """Execution constraints derived from the current capital tier."""

    model_config = ConfigDict(frozen=True)

    max_position_weight: float = Field(..., ge=0.0, le=1.0)
    max_gross_exposure: float = Field(..., ge=0.0)
    max_single_sector_pct: float = Field(..., ge=0.0, le=1.0)
    min_confidence_threshold: float = Field(..., ge=0.0, le=1.0)
    live_execution_enabled: bool = Field(...)
    max_positions: int = Field(..., ge=0)


class CapitalTierOutput(BaseModel):
    """Complete output of GOVERN-CAPITAL for one evaluation.

    FROZEN agent. Classifies AUM into capital tiers and derives constraints.
    Content-hashed.
    """

    model_config = ConfigDict(frozen=True)

    tier_id: UUID = Field(default_factory=uuid4)
    agent_id: str = Field(...)
    timestamp: datetime = Field(...)
    context_window_hash: str = Field(...)
    current_aum: float = Field(..., ge=0.0, description="Current AUM in USD")
    current_tier: CapitalTier = Field(...)
    previous_tier: Optional[CapitalTier] = Field(default=None)
    tier_changed: bool = Field(default=False)
    constraints: TierConstraints = Field(...)
    headroom_to_next_tier_pct: float = Field(
        default=0.0, ge=0.0,
        description="How close to next tier threshold (0-100%)",
    )
    content_hash: str = Field(default="")

    @field_validator("timestamp")
    @classmethod
    def must_have_timezone(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("Timestamp must include timezone information")
        return v

    @model_validator(mode="after")
    def _compute_content_hash(self) -> "CapitalTierOutput":
        if not self.content_hash:
            data = {
                "agent_id": self.agent_id,
                "current_aum": self.current_aum,
                "current_tier": self.current_tier.value,
                "tier_changed": self.tier_changed,
            }
            serialized = json.dumps(data, sort_keys=True, default=str).encode("utf-8")
            object.__setattr__(self, "content_hash", hashlib.sha256(serialized).hexdigest())
        return self


# ---------------------------------------------------------------------------
# GOVERN-MATURITY output
# ---------------------------------------------------------------------------

class AgentMaturityRecord(BaseModel):
    """Maturity classification for a single agent."""

    model_config = ConfigDict(frozen=True)

    agent_id: str = Field(...)
    current_stage: MaturityStage = Field(...)
    previous_stage: Optional[MaturityStage] = Field(default=None)
    stage_changed: bool = Field(default=False)
    days_in_current_stage: int = Field(default=0, ge=0)
    promotion_eligible: bool = Field(
        default=False,
        description="Whether agent meets criteria for next maturity stage",
    )
    promotion_blockers: list[str] = Field(
        default_factory=list,
        description="Reasons preventing promotion to next stage",
    )
    confidence_weight: float = Field(
        default=1.0, ge=0.0, le=1.0,
        description="Confidence weighting: SHADOW=0.0, LIMITED=0.5, FULL=1.0",
    )


class MaturityGateOutput(BaseModel):
    """Complete output of GOVERN-MATURITY for one evaluation.

    FROZEN agent. Evaluates agent readiness for deployment stage transitions.
    Content-hashed.
    """

    model_config = ConfigDict(frozen=True)

    maturity_id: UUID = Field(default_factory=uuid4)
    agent_id: str = Field(...)
    timestamp: datetime = Field(...)
    context_window_hash: str = Field(...)
    agent_records: list[AgentMaturityRecord] = Field(default_factory=list)
    agents_in_shadow: int = Field(default=0, ge=0)
    agents_in_limited: int = Field(default=0, ge=0)
    agents_in_full: int = Field(default=0, ge=0)
    promotions_recommended: int = Field(default=0, ge=0)
    content_hash: str = Field(default="")

    @field_validator("timestamp")
    @classmethod
    def must_have_timezone(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("Timestamp must include timezone information")
        return v

    @model_validator(mode="after")
    def _compute_content_hash(self) -> "MaturityGateOutput":
        if not self.content_hash:
            data = {
                "agent_id": self.agent_id,
                "agent_records": [
                    {"agent_id": r.agent_id, "current_stage": r.current_stage.value}
                    for r in self.agent_records
                ],
                "promotions_recommended": self.promotions_recommended,
            }
            serialized = json.dumps(data, sort_keys=True, default=str).encode("utf-8")
            object.__setattr__(self, "content_hash", hashlib.sha256(serialized).hexdigest())
        return self


# ---------------------------------------------------------------------------
# GOVERN-OVERSIGHT output
# ---------------------------------------------------------------------------

class GovernanceIncident(BaseModel):
    """A governance incident flagged for human review."""

    model_config = ConfigDict(frozen=True)

    incident_id: UUID = Field(default_factory=uuid4)
    severity: IncidentSeverity = Field(...)
    source_agent_id: str = Field(...)
    title: str = Field(...)
    description: str = Field(default="")
    requires_human_action: bool = Field(default=False)


class SystemHealthSummary(BaseModel):
    """Aggregate health summary across all agents."""

    model_config = ConfigDict(frozen=True)

    total_agents: int = Field(default=0, ge=0)
    healthy_count: int = Field(default=0, ge=0)
    degraded_count: int = Field(default=0, ge=0)
    unhealthy_count: int = Field(default=0, ge=0)
    offline_count: int = Field(default=0, ge=0)
    total_errors_24h: int = Field(default=0, ge=0)


class OversightOutput(BaseModel):
    """Complete output of GOVERN-OVERSIGHT for one evaluation.

    FROZEN agent. Aggregates system health, flags incidents, provides
    dashboard data for human operators.
    Content-hashed.
    """

    model_config = ConfigDict(frozen=True)

    oversight_id: UUID = Field(default_factory=uuid4)
    agent_id: str = Field(...)
    timestamp: datetime = Field(...)
    context_window_hash: str = Field(...)
    health_summary: SystemHealthSummary = Field(...)
    current_risk_mode: SystemRiskMode = Field(...)
    current_tier: CapitalTier = Field(...)
    incidents: list[GovernanceIncident] = Field(default_factory=list)
    active_positions_count: int = Field(default=0, ge=0)
    gross_exposure_pct: float = Field(default=0.0, ge=0.0)
    net_exposure_pct: float = Field(default=0.0)
    retraining_queue_size: int = Field(default=0, ge=0)
    shadow_exit_divergences: int = Field(default=0, ge=0)
    content_hash: str = Field(default="")

    @field_validator("timestamp")
    @classmethod
    def must_have_timezone(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("Timestamp must include timezone information")
        return v

    @model_validator(mode="after")
    def _compute_content_hash(self) -> "OversightOutput":
        if not self.content_hash:
            data = {
                "agent_id": self.agent_id,
                "health_summary": {
                    "total_agents": self.health_summary.total_agents,
                    "healthy_count": self.health_summary.healthy_count,
                },
                "incidents_count": len(self.incidents),
                "current_risk_mode": self.current_risk_mode.value,
            }
            serialized = json.dumps(data, sort_keys=True, default=str).encode("utf-8")
            object.__setattr__(self, "content_hash", hashlib.sha256(serialized).hexdigest())
        return self


# ---------------------------------------------------------------------------
# GOVERN-POLICY output
# ---------------------------------------------------------------------------

class PolicyViolation(BaseModel):
    """A detected policy violation."""

    model_config = ConfigDict(frozen=True)

    violation_id: UUID = Field(default_factory=uuid4)
    policy_name: str = Field(...)
    severity: IncidentSeverity = Field(...)
    description: str = Field(default="")
    violating_agent_id: str = Field(default="")
    auto_enforced: bool = Field(
        default=False,
        description="Whether the system auto-corrected the violation",
    )


class PolicyOutput(BaseModel):
    """Complete output of GOVERN-POLICY for one evaluation.

    FROZEN agent. Enforces capital constraints, position limits, and
    risk mode compliance. Flags violations for human review.
    Content-hashed.
    """

    model_config = ConfigDict(frozen=True)

    policy_id: UUID = Field(default_factory=uuid4)
    agent_id: str = Field(...)
    timestamp: datetime = Field(...)
    context_window_hash: str = Field(...)
    violations: list[PolicyViolation] = Field(default_factory=list)
    total_policies_checked: int = Field(default=0, ge=0)
    total_violations: int = Field(default=0, ge=0)
    auto_enforced_count: int = Field(default=0, ge=0)
    requires_human_review: bool = Field(default=False)
    content_hash: str = Field(default="")

    @field_validator("timestamp")
    @classmethod
    def must_have_timezone(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("Timestamp must include timezone information")
        return v

    @model_validator(mode="after")
    def _compute_content_hash(self) -> "PolicyOutput":
        if not self.content_hash:
            data = {
                "agent_id": self.agent_id,
                "total_violations": self.total_violations,
                "auto_enforced_count": self.auto_enforced_count,
                "requires_human_review": self.requires_human_review,
            }
            serialized = json.dumps(data, sort_keys=True, default=str).encode("utf-8")
            object.__setattr__(self, "content_hash", hashlib.sha256(serialized).hexdigest())
        return self
