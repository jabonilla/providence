"""Regime subsystem schemas — RegimeStateObject, SectorRegimeOverlay, NarrativeRegimeOverlay.

The RegimeStateObject is the output of REGIME-STAT, REGIME-SECTOR, and REGIME-NARR.
It captures the classified market regime, confidence, state probabilities,
risk mode, features used, optional sector-level overlays, and narrative overlay.

Spec Reference: Technical Spec v2.3, Section 2.3
"""

import hashlib
import json
from datetime import datetime
from typing import Any, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from providence.schemas.enums import StatisticalRegime, SystemRiskMode


class SectorRegimeOverlay(BaseModel):
    """Sector-level regime overlay produced by REGIME-SECTOR.

    Modifies how the global regime applies to a specific sector.
    For example, during a global HIGH_VOL regime, Technology might
    be in CRISIS_DISLOCATION while Utilities remains LOW_VOL_TRENDING.

    Immutable (frozen=True).
    """

    model_config = ConfigDict(frozen=True)

    sector: str = Field(..., description="GICS sector name (e.g., 'Technology', 'Healthcare')")
    regime: StatisticalRegime = Field(..., description="Sector-specific regime classification")
    regime_confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence in the sector regime"
    )
    regime_probabilities: dict[str, float] = Field(
        ..., description="Per-state probabilities for this sector"
    )
    relative_stress: float = Field(
        default=0.0,
        description="Relative stress vs market: -1.0 (calmer) to +1.0 (more stressed)",
    )
    key_signals: list[str] = Field(
        default_factory=list,
        description="Top driving factors for this sector's regime (e.g., 'high realized vol')",
    )
    ticker_count: int = Field(
        default=0, ge=0, description="Number of tickers used to compute this overlay"
    )


class NarrativeRegimeOverlay(BaseModel):
    """Narrative regime overlay produced by REGIME-NARR.

    LLM-generated qualitative assessment of the dominant market narrative.
    Captures the "story" the market is telling beyond statistical signals.
    Examples: "AI-driven tech euphoria", "Fed pivot anticipation",
    "Banking contagion fear", "Risk-off flight to quality".

    Immutable (frozen=True).
    """

    model_config = ConfigDict(frozen=True)

    label: str = Field(
        ..., min_length=1, max_length=200,
        description="LLM-generated narrative regime label",
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="Confidence in the narrative assessment",
    )
    key_signals: list[str] = Field(
        default_factory=list,
        description="Key narrative drivers identified by the LLM",
    )
    affected_sectors: list[str] = Field(
        default_factory=list,
        description="Sectors most affected by this narrative",
    )
    regime_alignment: str = Field(
        default="",
        description="How this narrative aligns with statistical regime (CONFIRMS/DIVERGES/NEUTRAL)",
    )
    summary: str = Field(
        default="",
        description="LLM-generated summary of current market narrative (1-3 sentences)",
    )


class RegimeStateObject(BaseModel):
    """Output of regime agents: classified market regime with probabilities.

    Produced by REGIME-STAT (global), enriched by REGIME-SECTOR
    (sector overlays) and REGIME-NARR (narrative overlay).
    Immutable (frozen=True). Content-hashed.
    """

    model_config = ConfigDict(frozen=True)

    regime_id: UUID = Field(default_factory=uuid4, description="Unique ID for this regime snapshot")
    agent_id: str = Field(..., description="ID of the agent that produced this")
    timestamp: datetime = Field(..., description="When this regime was classified")
    context_window_hash: str = Field(..., description="Hash of input context for reproducibility")

    statistical_regime: StatisticalRegime = Field(
        ..., description="The classified regime state"
    )
    regime_confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence in the classified regime"
    )
    regime_probabilities: dict[str, float] = Field(
        ..., description="Probability for each regime state from HMM forward algorithm"
    )
    system_risk_mode: SystemRiskMode = Field(
        ..., description="Derived risk mode governing position sizing and exposure limits"
    )
    features_used: dict[str, float] = Field(
        default_factory=dict,
        description="Feature values used for classification (realized_vol, yield_spread, etc.)",
    )
    sector_overlays: dict[str, SectorRegimeOverlay] = Field(
        default_factory=dict,
        description="Sector-level regime overlays (populated by REGIME-SECTOR)",
    )
    narrative_overlay: NarrativeRegimeOverlay | None = Field(
        default=None,
        description="Narrative regime overlay (populated by REGIME-NARR)",
    )
    content_hash: str = Field(
        default="",
        description="SHA-256 of regime data for content-addressing",
    )

    @model_validator(mode="after")
    def _compute_content_hash(self) -> "RegimeStateObject":
        """Compute content hash from regime data if not already set."""
        if not self.content_hash:
            # Serialize sector overlays for hashing
            overlay_data = {}
            for sector, overlay in sorted(self.sector_overlays.items()):
                overlay_data[sector] = {
                    "regime": overlay.regime.value,
                    "regime_confidence": overlay.regime_confidence,
                    "relative_stress": overlay.relative_stress,
                }

            # Serialize narrative overlay for hashing
            narrative_data = None
            if self.narrative_overlay is not None:
                narrative_data = {
                    "label": self.narrative_overlay.label,
                    "confidence": self.narrative_overlay.confidence,
                    "regime_alignment": self.narrative_overlay.regime_alignment,
                }

            data = {
                "agent_id": self.agent_id,
                "statistical_regime": self.statistical_regime.value,
                "regime_confidence": self.regime_confidence,
                "regime_probabilities": self.regime_probabilities,
                "system_risk_mode": self.system_risk_mode.value,
                "features_used": self.features_used,
                "sector_overlays": overlay_data,
                "narrative_overlay": narrative_data,
            }
            serialized = json.dumps(data, sort_keys=True, default=str).encode("utf-8")
            computed = hashlib.sha256(serialized).hexdigest()
            object.__setattr__(self, "content_hash", computed)
        return self

    @field_validator("regime_probabilities")
    @classmethod
    def _validate_probabilities(cls, v: dict[str, float]) -> dict[str, float]:
        """Probabilities should be non-negative."""
        for key, prob in v.items():
            if prob < 0.0:
                raise ValueError(f"Probability for {key} must be non-negative, got {prob}")
        return v
