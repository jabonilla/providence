"""Decision subsystem schemas — SynthesizedPositionIntent.

The SynthesizedPositionIntent is the output of DECIDE-SYNTH.
It synthesizes beliefs from independent research agents into
unified position intents with conflict resolution metadata.

Spec Reference: Technical Spec v2.3, Section 2.4
"""

import hashlib
import json
from datetime import datetime
from typing import Any, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from providence.schemas.enums import Action, Direction, Magnitude


class ContributingThesis(BaseModel):
    """Reference to a thesis that contributes to a position intent.

    Tracks which agent produced it, its direction, confidence, and
    how much weight it received in the synthesis.
    """

    model_config = ConfigDict(frozen=True)

    thesis_id: str = Field(..., description="Original thesis ID from the research agent")
    agent_id: str = Field(..., description="Research agent that produced this thesis")
    ticker: str = Field(..., description="Ticker targeted by this thesis")
    direction: Direction = Field(..., description="LONG, SHORT, or NEUTRAL")
    raw_confidence: float = Field(..., ge=0.0, le=1.0, description="Agent's confidence")
    magnitude: Magnitude = Field(..., description="SMALL, MODERATE, or LARGE")
    synthesis_weight: float = Field(
        ..., ge=0.0, le=1.0,
        description="Weight assigned to this thesis during synthesis",
    )


class ConflictResolution(BaseModel):
    """Metadata about how conflicts between agents were resolved.

    When agents disagree (e.g., FUNDAMENTAL says LONG, MACRO says SHORT),
    the synthesizer records the resolution logic.
    """

    model_config = ConfigDict(frozen=True)

    has_conflict: bool = Field(default=False, description="Whether conflicting theses exist")
    conflict_type: str = Field(
        default="",
        description="Type of conflict: DIRECTIONAL, MAGNITUDE, TIMING, NONE",
    )
    resolution_method: str = Field(
        default="",
        description="How the conflict was resolved: CONFIDENCE_WEIGHTED, REGIME_ADJUSTED, MAJORITY, DEFERRED",
    )
    resolution_rationale: str = Field(
        default="",
        description="LLM-generated explanation of the resolution (1-2 sentences)",
    )
    net_conviction_delta: float = Field(
        default=0.0,
        description="How much the conflict reduced overall conviction (-1.0 to 0.0)",
    )


class ActiveInvalidation(BaseModel):
    """An invalidation condition carried forward from contributing theses.

    The most critical invalidation conditions from all contributing
    theses are aggregated into the position intent.
    """

    model_config = ConfigDict(frozen=True)

    condition_id: UUID = Field(default_factory=uuid4, description="Unique condition ID")
    source_thesis_id: str = Field(..., description="Which thesis this condition came from")
    source_agent_id: str = Field(..., description="Which agent produced the source thesis")
    metric: str = Field(..., description="Metric to monitor")
    operator: str = Field(..., description="Comparison operator (GT, LT, etc.)")
    threshold: float = Field(..., description="Threshold value")
    description: str = Field(default="", description="Human-readable description")


class SynthesizedPositionIntent(BaseModel):
    """A unified position intent synthesized from multiple research agent beliefs.

    Produced by DECIDE-SYNTH. Consumed by DECIDE-OPTIM for portfolio construction.
    Immutable (frozen=True). Content-hashed.
    """

    model_config = ConfigDict(frozen=True)

    intent_id: UUID = Field(default_factory=uuid4, description="Unique intent ID")
    ticker: str = Field(..., description="Target ticker symbol")
    net_direction: Direction = Field(
        ..., description="Synthesized net direction: LONG, SHORT, or NEUTRAL"
    )
    synthesized_confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="Synthesized confidence after conflict resolution and regime adjustment",
    )
    contributing_theses: list[ContributingThesis] = Field(
        default_factory=list,
        description="Theses that support this position intent",
    )
    conflicting_theses: list[ContributingThesis] = Field(
        default_factory=list,
        description="Theses that oppose this position intent",
    )
    conflict_resolution: ConflictResolution = Field(
        default_factory=ConflictResolution,
        description="How conflicts between agents were resolved",
    )
    time_horizon_days: int = Field(
        ..., gt=0,
        description="Synthesized time horizon (weighted average of contributing theses)",
    )
    regime_adjustment: float = Field(
        default=0.0,
        description="Confidence adjustment from regime overlay (-0.3 to +0.1)",
    )
    active_invalidations: list[ActiveInvalidation] = Field(
        default_factory=list,
        description="Aggregated invalidation conditions from contributing theses",
    )
    synthesis_rationale: str = Field(
        default="",
        description="LLM-generated rationale for the synthesized position (1-3 sentences)",
    )


class SynthesisOutput(BaseModel):
    """Complete output of DECIDE-SYNTH for one processing cycle.

    Contains one or more SynthesizedPositionIntents along with
    context metadata. Content-hashed.
    """

    model_config = ConfigDict(frozen=True)

    synthesis_id: UUID = Field(default_factory=uuid4, description="Unique synthesis ID")
    agent_id: str = Field(..., description="ID of the agent that produced this")
    timestamp: datetime = Field(..., description="When this synthesis was produced")
    context_window_hash: str = Field(
        ..., description="Hash of input context for reproducibility"
    )
    position_intents: list[SynthesizedPositionIntent] = Field(
        default_factory=list,
        description="Synthesized position intents",
    )
    regime_context: str = Field(
        default="",
        description="Current regime label used for adjustments",
    )
    total_beliefs_consumed: int = Field(
        default=0, ge=0,
        description="Total number of individual beliefs consumed from all agents",
    )
    content_hash: str = Field(
        default="",
        description="SHA-256 of synthesis data for content-addressing",
    )

    @field_validator("timestamp")
    @classmethod
    def must_have_timezone(cls, v: datetime) -> datetime:
        """Timestamp must include timezone information."""
        if v.tzinfo is None:
            raise ValueError("Timestamp must include timezone information")
        return v

    @model_validator(mode="after")
    def _compute_content_hash(self) -> "SynthesisOutput":
        """Compute content hash from position intents."""
        if not self.content_hash:
            intents_data = [
                {
                    "ticker": intent.ticker,
                    "net_direction": intent.net_direction.value,
                    "synthesized_confidence": intent.synthesized_confidence,
                    "time_horizon_days": intent.time_horizon_days,
                    "regime_adjustment": intent.regime_adjustment,
                }
                for intent in self.position_intents
            ]
            data = {
                "agent_id": self.agent_id,
                "position_intents": intents_data,
                "regime_context": self.regime_context,
            }
            serialized = json.dumps(data, sort_keys=True, default=str).encode("utf-8")
            computed = hashlib.sha256(serialized).hexdigest()
            object.__setattr__(self, "content_hash", computed)
        return self


# ---------------------------------------------------------------------------
# PositionProposal — output of DECIDE-OPTIM
# ---------------------------------------------------------------------------


class PortfolioMetadata(BaseModel):
    """Portfolio-level metrics computed by DECIDE-OPTIM.

    Describes the aggregate risk/exposure characteristics of the
    proposed portfolio after Black-Litterman optimization.
    """

    model_config = ConfigDict(frozen=True)

    gross_exposure: float = Field(
        ..., ge=0.0, le=2.0,
        description="Sum of absolute position weights (0.0 to 2.0)",
    )
    net_exposure: float = Field(
        ..., ge=-1.0, le=1.0,
        description="Net long minus short exposure (-1.0 to 1.0)",
    )
    sector_concentrations: dict[str, float] = Field(
        default_factory=dict,
        description="Map of GICS sector -> absolute weight concentration",
    )
    estimated_sharpe: float = Field(
        default=0.0,
        description="Estimated ex-ante Sharpe ratio from optimization",
    )
    position_count: int = Field(
        default=0, ge=0,
        description="Number of positions in the proposed portfolio",
    )
    risk_mode_applied: str = Field(
        default="NORMAL",
        description="SystemRiskMode that governed exposure limits",
    )


class ProposedPosition(BaseModel):
    """A single proposed position from portfolio optimization.

    Produced by DECIDE-OPTIM. Consumed by EXEC-VALIDATE.
    Immutable (frozen=True).
    """

    model_config = ConfigDict(frozen=True)

    ticker: str = Field(..., description="Target ticker symbol")
    action: Action = Field(..., description="OPEN_LONG, OPEN_SHORT, CLOSE, or ADJUST")
    target_weight: float = Field(
        ..., ge=0.0, le=0.20,
        description="Target portfolio weight (0.0 to 0.20, max 20% per position)",
    )
    direction: Direction = Field(..., description="LONG, SHORT, or NEUTRAL")
    confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="Confidence carried from synthesized intent",
    )
    source_intent_id: UUID = Field(
        ..., description="Reference to the SynthesizedPositionIntent that produced this",
    )
    time_horizon_days: int = Field(
        ..., gt=0,
        description="Time horizon from the source intent",
    )
    regime_adjustment: float = Field(
        default=0.0,
        description="Regime confidence adjustment applied (-0.3 to +0.1)",
    )
    sector: str = Field(
        default="Unknown",
        description="GICS sector of this ticker",
    )


class PositionProposal(BaseModel):
    """Complete output of DECIDE-OPTIM for one optimization cycle.

    Contains proposed positions and portfolio-level metadata.
    Content-hashed for audit trail.
    """

    model_config = ConfigDict(frozen=True)

    proposal_id: UUID = Field(default_factory=uuid4, description="Unique proposal ID")
    agent_id: str = Field(..., description="ID of the agent that produced this")
    timestamp: datetime = Field(..., description="When this proposal was produced")
    context_window_hash: str = Field(
        ..., description="Hash of input context for reproducibility",
    )
    proposals: list[ProposedPosition] = Field(
        default_factory=list,
        description="Proposed positions after optimization",
    )
    portfolio_metadata: PortfolioMetadata = Field(
        ..., description="Portfolio-level risk and exposure metrics",
    )
    total_intents_consumed: int = Field(
        default=0, ge=0,
        description="Number of position intents consumed from DECIDE-SYNTH",
    )
    content_hash: str = Field(
        default="",
        description="SHA-256 of proposal data for content-addressing",
    )

    @field_validator("timestamp")
    @classmethod
    def must_have_timezone(cls, v: datetime) -> datetime:
        """Timestamp must include timezone information."""
        if v.tzinfo is None:
            raise ValueError("Timestamp must include timezone information")
        return v

    @model_validator(mode="after")
    def _compute_content_hash(self) -> "PositionProposal":
        """Compute content hash from proposals."""
        if not self.content_hash:
            proposals_data = [
                {
                    "ticker": p.ticker,
                    "action": p.action.value,
                    "target_weight": p.target_weight,
                    "direction": p.direction.value,
                    "confidence": p.confidence,
                }
                for p in self.proposals
            ]
            data = {
                "agent_id": self.agent_id,
                "proposals": proposals_data,
                "gross_exposure": self.portfolio_metadata.gross_exposure,
                "net_exposure": self.portfolio_metadata.net_exposure,
            }
            serialized = json.dumps(data, sort_keys=True, default=str).encode("utf-8")
            computed = hashlib.sha256(serialized).hexdigest()
            object.__setattr__(self, "content_hash", computed)
        return self
