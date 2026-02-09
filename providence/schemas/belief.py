"""BeliefObject schema and nested types — the core output of Research Agents.

Spec Reference: Technical Spec v2.3, Section 2.2

Produced by: Cognition System (Research Agents)
Consumed by: Decision System, Learning System

Each Research Agent independently produces BeliefObjects containing one or more
Belief entries. Beliefs include machine-evaluable invalidation conditions and
evidence references linking back to MarketStateFragments.
"""

import hashlib
import json
from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from providence.schemas.enums import (
    CatalystType,
    ComparisonOperator,
    ConditionStatus,
    Direction,
    Magnitude,
    MarketCapBucket,
)


class EvidenceRef(BaseModel):
    """Reference to a specific observation in a MarketStateFragment.

    Links a belief back to the data that supports it, enabling
    full provenance tracking through the system.
    """

    model_config = ConfigDict(frozen=True)

    source_fragment_id: UUID = Field(..., description="Links to MarketStateFragment.fragment_id")
    field_path: str = Field(..., description="JSON path within the fragment payload")
    observation: str = Field(..., description="What the agent observed at this path")
    weight: float = Field(..., ge=0.0, le=1.0, description="Importance to thesis (0.0-1.0)")


class InvalidationCondition(BaseModel):
    """A machine-evaluable condition that would invalidate a thesis.

    Every condition must specify a concrete metric, operator, and threshold
    so it can be monitored automatically by INVALID-MON. No vague prose.

    Breach semantics (v2.1+):
    - breach_magnitude: |current - threshold| / threshold
      * < 0.05 → marginal, reduce confidence 10-20%
      * 0.05-0.20 → moderate, reduce confidence 30-50%
      * > 0.20 → strong, reduce confidence 60-80%
    - breach_velocity: rate of approach to threshold (per day, trailing 5-day avg)
    """

    model_config = ConfigDict(frozen=True)

    condition_id: UUID = Field(default_factory=uuid4, description="Unique condition identifier")
    description: str = Field(..., description="Human-readable description of the condition")
    data_source_agent: str = Field(..., description="Which data agent monitors this condition")
    metric: str = Field(..., description="Specific field/metric to watch")
    operator: ComparisonOperator = Field(..., description="Comparison operator for evaluation")
    threshold: float = Field(..., description="Threshold value for the condition")
    status: ConditionStatus = Field(
        default=ConditionStatus.ACTIVE,
        description="Current status of this condition",
    )
    current_value: Optional[float] = Field(default=None, description="Latest observed value")
    breach_magnitude: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="|current - threshold| / threshold (0.0 = at threshold)",
    )
    breach_velocity: Optional[float] = Field(
        default=None,
        description="Rate of approach to threshold (per day, trailing 5-day avg)",
    )

    @field_validator("metric")
    @classmethod
    def metric_must_not_be_empty(cls, v: str) -> str:
        """Metric must be a non-empty string specifying a concrete field."""
        if not v.strip():
            raise ValueError("Metric must be a non-empty string specifying a concrete field to watch")
        return v


class BeliefMetadata(BaseModel):
    """Metadata attached to each belief for classification and analysis."""

    model_config = ConfigDict(frozen=True)

    sector: str = Field(..., description="Sector classification")
    market_cap_bucket: MarketCapBucket = Field(..., description="Market cap classification")
    catalyst_type: Optional[CatalystType] = Field(
        default=None,
        description="Type of catalyst driving this thesis",
    )


class Belief(BaseModel):
    """A single investment thesis produced by a Research Agent.

    Contains the thesis direction, confidence, evidence chain,
    and machine-evaluable invalidation conditions.
    """

    model_config = ConfigDict(frozen=True)

    thesis_id: str = Field(..., description="Unique, human-readable thesis identifier")
    ticker: str = Field(..., description="Target ticker symbol")
    thesis_summary: str = Field(..., description="One-line thesis description")
    direction: Direction = Field(..., description="Investment direction: LONG, SHORT, or NEUTRAL")
    magnitude: Magnitude = Field(..., description="Expected magnitude: SMALL, MODERATE, or LARGE")
    raw_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Agent self-assessment of confidence (0.0-1.0)",
    )
    time_horizon_days: int = Field(..., gt=0, description="Expected time horizon in days")
    evidence: list[EvidenceRef] = Field(
        default_factory=list,
        description="Evidence references linking to MarketStateFragments",
    )
    invalidation_conditions: list[InvalidationCondition] = Field(
        default_factory=list,
        description="Machine-evaluable conditions that would invalidate this thesis",
    )
    correlated_beliefs: list[str] = Field(
        default_factory=list,
        description="Thesis IDs of correlated beliefs (cross-agent linkage)",
    )
    metadata: BeliefMetadata = Field(..., description="Classification metadata")


class BeliefObject(BaseModel):
    """The complete output of a Research Agent for one processing cycle.

    Contains one or more Belief entries along with the context window hash
    for exact input reproducibility. Content hash is computed from all beliefs.
    """

    model_config = ConfigDict(frozen=True)

    belief_id: UUID = Field(default_factory=uuid4, description="Unique identifier for this belief set")
    agent_id: str = Field(..., description="ID of the Research Agent that produced this")
    timestamp: datetime = Field(..., description="When this belief set was produced")
    context_window_hash: str = Field(
        ...,
        description="SHA-256 hash of the input context for reproducibility",
    )
    beliefs: list[Belief] = Field(
        default_factory=list,
        description="Array of individual beliefs/theses",
    )
    content_hash: str = Field(default="", description="SHA-256 hash of all beliefs, computed automatically")

    @field_validator("timestamp")
    @classmethod
    def must_have_timezone(cls, v: datetime) -> datetime:
        """Timestamp must include timezone information."""
        if v.tzinfo is None:
            raise ValueError("Timestamp must include timezone information (tzinfo cannot be None)")
        return v

    @model_validator(mode="after")
    def compute_content_hash(self) -> "BeliefObject":
        """Compute SHA-256 content hash from all beliefs.

        Serializes all beliefs deterministically (sorted keys) and
        computes a single hash for the entire belief set.
        """
        beliefs_data = [belief.model_dump(mode="json") for belief in self.beliefs]
        beliefs_bytes = json.dumps(beliefs_data, sort_keys=True, default=str).encode("utf-8")
        content_hash = hashlib.sha256(beliefs_bytes).hexdigest()
        object.__setattr__(self, "content_hash", content_hash)
        return self
