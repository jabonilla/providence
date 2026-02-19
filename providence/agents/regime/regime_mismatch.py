"""REGIME-MISMATCH: Regime Divergence Detection Agent.

Compares upstream regime classifications (statistical, narrative, sector)
and flags divergences that warrant risk mode escalation. Mismatch between
statistical and narrative regimes is itself a risk signal — it often
precedes regime transitions.

Spec Reference: Technical Spec v2.3, Section 4.3

Classification: FROZEN — zero LLM calls. Pure computation.

Input: AgentContext with metadata carrying upstream RegimeStateObjects:
  - metadata["stat_regime"]: Serialized RegimeStateObject from REGIME-STAT
  - metadata["narr_regime"]: Serialized RegimeStateObject from REGIME-NARR
  - metadata["sector_regime"]: Serialized RegimeStateObject from REGIME-SECTOR

Output: RegimeStateObject with:
  - Final regime = stat regime (statistical takes precedence)
  - Risk mode = potentially ESCALATED based on mismatch severity
  - features_used includes mismatch metrics
  - All overlays merged from upstream
"""

from datetime import datetime, timezone

import structlog

from providence.agents.base import AgentContext, AgentStatus, BaseAgent, HealthStatus
from providence.exceptions import AgentProcessingError
from providence.schemas.enums import StatisticalRegime, SystemRiskMode
from providence.schemas.regime import (
    NarrativeRegimeOverlay,
    RegimeStateObject,
    SectorRegimeOverlay,
)

logger = structlog.get_logger()

# Regime severity ordering (higher = more stressed)
REGIME_SEVERITY = {
    StatisticalRegime.LOW_VOL_TRENDING: 0,
    StatisticalRegime.TRANSITION_UNCERTAIN: 1,
    StatisticalRegime.HIGH_VOL_MEAN_REVERTING: 2,
    StatisticalRegime.CRISIS_DISLOCATION: 3,
}

# Risk mode severity ordering
RISK_MODE_SEVERITY = {
    SystemRiskMode.NORMAL: 0,
    SystemRiskMode.CAUTIOUS: 1,
    SystemRiskMode.DEFENSIVE: 2,
    SystemRiskMode.HALTED: 3,
}

# Ordered risk modes for escalation
RISK_MODES_ORDERED = [
    SystemRiskMode.NORMAL,
    SystemRiskMode.CAUTIOUS,
    SystemRiskMode.DEFENSIVE,
    SystemRiskMode.HALTED,
]


def compute_regime_distance(
    regime_a: StatisticalRegime,
    regime_b: StatisticalRegime,
) -> int:
    """Compute distance between two regimes on the severity scale.

    Args:
        regime_a: First regime.
        regime_b: Second regime.

    Returns:
        Absolute distance (0 = same, 3 = maximum divergence).
    """
    return abs(REGIME_SEVERITY[regime_a] - REGIME_SEVERITY[regime_b])


def compute_mismatch_score(
    stat_regime: StatisticalRegime,
    stat_confidence: float,
    narrative_alignment: str,
    narrative_confidence: float,
    sector_divergence_count: int,
    total_sectors: int,
) -> float:
    """Compute overall mismatch score between regime signals.

    Higher score = more divergence = more risk.

    Components:
    1. Stat-narrative alignment mismatch (0.40 weight)
    2. Confidence-weighted divergence (0.30 weight)
    3. Sector divergence ratio (0.30 weight)

    Args:
        stat_regime: Statistical regime from HMM.
        stat_confidence: Confidence in statistical regime.
        narrative_alignment: "CONFIRMS", "DIVERGES", or "NEUTRAL".
        narrative_confidence: Confidence in narrative assessment.
        sector_divergence_count: Number of sectors diverging from global.
        total_sectors: Total number of sectors with overlays.

    Returns:
        Mismatch score between 0.0 and 1.0.
    """
    scores: list[float] = []
    weights: list[float] = []

    # Component 1: Stat-narrative alignment
    if narrative_alignment == "DIVERGES":
        # Divergence is the primary mismatch signal
        alignment_score = 0.8 + 0.2 * narrative_confidence
    elif narrative_alignment == "NEUTRAL":
        # Neutral is a mild signal — ambiguity
        alignment_score = 0.3
    else:
        # CONFIRMS = no mismatch from this component
        alignment_score = 0.0
    scores.append(alignment_score)
    weights.append(0.40)

    # Component 2: Confidence-weighted divergence
    # High confidence on BOTH sides + disagreement = strong mismatch
    if narrative_alignment == "DIVERGES":
        combined_conf = stat_confidence * narrative_confidence
        conf_score = combined_conf  # Both high → near 1.0; either low → < 0.5
    elif narrative_alignment == "NEUTRAL":
        conf_score = 0.2 * stat_confidence
    else:
        conf_score = 0.0
    scores.append(conf_score)
    weights.append(0.30)

    # Component 3: Sector divergence
    if total_sectors > 0:
        sector_score = sector_divergence_count / total_sectors
    else:
        sector_score = 0.0
    scores.append(sector_score)
    weights.append(0.30)

    total_weight = sum(weights)
    composite = sum(s * w for s, w in zip(scores, weights)) / total_weight
    return max(0.0, min(1.0, composite))


def count_sector_divergences(
    global_regime: StatisticalRegime,
    sector_overlays: dict[str, SectorRegimeOverlay],
) -> int:
    """Count sectors whose regime diverges from the global regime.

    A sector diverges if its regime is >=2 steps away on the severity scale.

    Args:
        global_regime: Global statistical regime.
        sector_overlays: Per-sector regime overlays.

    Returns:
        Number of diverging sectors.
    """
    count = 0
    for overlay in sector_overlays.values():
        distance = compute_regime_distance(global_regime, overlay.regime)
        if distance >= 2:
            count += 1
    return count


def escalate_risk_mode(
    base_mode: SystemRiskMode,
    mismatch_score: float,
) -> SystemRiskMode:
    """Escalate risk mode based on mismatch severity.

    Escalation rules:
    - mismatch_score < 0.30: no escalation
    - mismatch_score 0.30-0.60: escalate by 1 level
    - mismatch_score > 0.60: escalate by 2 levels
    - Never exceed HALTED

    Args:
        base_mode: Base risk mode from statistical regime.
        mismatch_score: Overall mismatch score [0, 1].

    Returns:
        Potentially escalated risk mode.
    """
    if mismatch_score < 0.30:
        return base_mode

    current_idx = RISK_MODE_SEVERITY[base_mode]

    if mismatch_score > 0.60:
        escalation = 2
    else:
        escalation = 1

    new_idx = min(current_idx + escalation, 3)
    return RISK_MODES_ORDERED[new_idx]


def identify_mismatch_signals(
    stat_regime: StatisticalRegime,
    narrative_alignment: str,
    narrative_label: str,
    sector_divergence_count: int,
    total_sectors: int,
    mismatch_score: float,
) -> list[str]:
    """Generate human-readable mismatch signal descriptions.

    Args:
        stat_regime: Statistical regime.
        narrative_alignment: Narrative alignment with statistical regime.
        narrative_label: Narrative regime label.
        sector_divergence_count: Number of diverging sectors.
        total_sectors: Total sectors.
        mismatch_score: Overall mismatch score.

    Returns:
        List of signal descriptions.
    """
    signals: list[str] = []

    if narrative_alignment == "DIVERGES":
        signals.append(
            f"stat-narrative divergence: HMM says {stat_regime.value} but narrative is '{narrative_label}'"
        )
    elif narrative_alignment == "NEUTRAL":
        signals.append("narrative ambiguous — unclear alignment with statistical regime")

    if total_sectors > 0 and sector_divergence_count > 0:
        pct = round(100 * sector_divergence_count / total_sectors)
        signals.append(
            f"{sector_divergence_count}/{total_sectors} sectors ({pct}%) diverge from global regime"
        )

    if mismatch_score > 0.60:
        signals.append("severe mismatch — risk mode escalated by 2 levels")
    elif mismatch_score > 0.30:
        signals.append("moderate mismatch — risk mode escalated by 1 level")

    if not signals:
        signals.append("no significant regime mismatch detected")

    return signals


class RegimeMismatch(BaseAgent[RegimeStateObject]):
    """Regime divergence detection agent.

    Compares statistical (HMM), narrative (LLM), and sector-level
    regime classifications to detect divergences. Mismatch between
    signals warrants risk mode escalation.

    Receives upstream regime data via AgentContext metadata:
      - metadata["stat_regime"]: dict from REGIME-STAT output
      - metadata["narr_regime"]: dict from REGIME-NARR output (optional)
      - metadata["sector_regime"]: dict from REGIME-SECTOR output (optional)

    FROZEN: No LLM calls. All analysis is deterministic computation.
    """

    # REGIME-MISMATCH doesn't directly consume raw market data.
    # It reads upstream regime outputs from context metadata.
    CONSUMED_DATA_TYPES: set = set()

    def __init__(self) -> None:
        super().__init__(
            agent_id="REGIME-MISMATCH",
            agent_type="regime",
            version="1.0.0",
        )
        self._last_run: datetime | None = None
        self._last_success: datetime | None = None
        self._error_count_24h: int = 0

    async def process(self, context: AgentContext) -> RegimeStateObject:
        """Execute the regime mismatch detection pipeline.

        Steps:
          1. RECEIVE CONTEXT     → Extract upstream regime data from metadata
          2. PARSE UPSTREAM      → Reconstruct regime classifications
          3. COMPUTE MISMATCH    → Compare statistical vs narrative vs sector
          4. ESCALATE RISK MODE  → Adjust risk mode based on mismatch severity
          5. EMIT                → Return merged RegimeStateObject

        Args:
            context: AgentContext with upstream regime data in metadata.

        Returns:
            RegimeStateObject with final risk mode and mismatch signals.

        Raises:
            AgentProcessingError: If processing fails.
        """
        self._last_run = datetime.now(timezone.utc)

        try:
            log = logger.bind(
                agent_id=self.agent_id,
                context_hash=context.context_window_hash,
            )
            log.info("Starting regime mismatch detection")

            # Step 1-2: EXTRACT UPSTREAM REGIME DATA
            stat_data = context.metadata.get("stat_regime", {})
            narr_data = context.metadata.get("narr_regime", {})
            sector_data = context.metadata.get("sector_regime", {})

            # Parse statistical regime (required)
            stat_regime_str = stat_data.get("statistical_regime", "LOW_VOL_TRENDING")
            stat_regime = StatisticalRegime(stat_regime_str)
            stat_confidence = float(stat_data.get("regime_confidence", 0.5))
            stat_risk_mode_str = stat_data.get("system_risk_mode", "NORMAL")
            stat_risk_mode = SystemRiskMode(stat_risk_mode_str)
            stat_probs = stat_data.get("regime_probabilities", {
                r.value: 0.25 for r in StatisticalRegime
            })
            stat_features = stat_data.get("features_used", {})

            # Parse narrative overlay (optional)
            narrative_overlay = None
            narrative_alignment = "NEUTRAL"
            narrative_confidence = 0.0
            narrative_label = ""

            narr_overlay_data = narr_data.get("narrative_overlay")
            if narr_overlay_data and isinstance(narr_overlay_data, dict):
                label = narr_overlay_data.get("label", "")
                if label:
                    narrative_overlay = NarrativeRegimeOverlay(
                        label=label,
                        confidence=float(narr_overlay_data.get("confidence", 0.0)),
                        key_signals=narr_overlay_data.get("key_signals", []),
                        affected_sectors=narr_overlay_data.get("affected_sectors", []),
                        regime_alignment=narr_overlay_data.get("regime_alignment", "NEUTRAL"),
                        summary=narr_overlay_data.get("summary", ""),
                    )
                    narrative_alignment = narrative_overlay.regime_alignment or "NEUTRAL"
                    narrative_confidence = narrative_overlay.confidence
                    narrative_label = narrative_overlay.label

            # Parse sector overlays (optional)
            sector_overlays: dict[str, SectorRegimeOverlay] = {}
            sector_overlays_data = sector_data.get("sector_overlays", {})
            if isinstance(sector_overlays_data, dict):
                for sector_name, overlay_dict in sector_overlays_data.items():
                    if isinstance(overlay_dict, dict):
                        try:
                            sector_overlays[sector_name] = SectorRegimeOverlay(
                                sector=sector_name,
                                regime=StatisticalRegime(
                                    overlay_dict.get("regime", "LOW_VOL_TRENDING")
                                ),
                                regime_confidence=float(
                                    overlay_dict.get("regime_confidence", 0.5)
                                ),
                                regime_probabilities=overlay_dict.get(
                                    "regime_probabilities",
                                    {r.value: 0.25 for r in StatisticalRegime},
                                ),
                                relative_stress=float(
                                    overlay_dict.get("relative_stress", 0.0)
                                ),
                                key_signals=overlay_dict.get("key_signals", []),
                                ticker_count=int(overlay_dict.get("ticker_count", 0)),
                            )
                        except (ValueError, TypeError):
                            continue

            # Step 3: COMPUTE MISMATCH
            sector_divergences = count_sector_divergences(stat_regime, sector_overlays)

            mismatch_score = compute_mismatch_score(
                stat_regime=stat_regime,
                stat_confidence=stat_confidence,
                narrative_alignment=narrative_alignment,
                narrative_confidence=narrative_confidence,
                sector_divergence_count=sector_divergences,
                total_sectors=len(sector_overlays),
            )

            log.info(
                "Mismatch computed",
                mismatch_score=round(mismatch_score, 4),
                narrative_alignment=narrative_alignment,
                sector_divergences=sector_divergences,
                total_sectors=len(sector_overlays),
            )

            # Step 4: ESCALATE RISK MODE
            final_risk_mode = escalate_risk_mode(stat_risk_mode, mismatch_score)

            mismatch_signals = identify_mismatch_signals(
                stat_regime=stat_regime,
                narrative_alignment=narrative_alignment,
                narrative_label=narrative_label,
                sector_divergence_count=sector_divergences,
                total_sectors=len(sector_overlays),
                mismatch_score=mismatch_score,
            )

            log.info(
                "Risk mode resolved",
                base_mode=stat_risk_mode.value,
                final_mode=final_risk_mode.value,
                escalated=final_risk_mode != stat_risk_mode,
            )

            # Step 5: EMIT — build final RegimeStateObject
            features_dict = dict(stat_features)
            features_dict["mismatch_score"] = round(mismatch_score, 6)
            features_dict["sector_divergence_count"] = float(sector_divergences)
            features_dict["total_sectors"] = float(len(sector_overlays))

            regime_state = RegimeStateObject(
                agent_id=self.agent_id,
                timestamp=datetime.now(timezone.utc),
                context_window_hash=context.context_window_hash,
                statistical_regime=stat_regime,
                regime_confidence=round(stat_confidence, 6),
                regime_probabilities=stat_probs,
                system_risk_mode=final_risk_mode,
                features_used=features_dict,
                sector_overlays=sector_overlays,
                narrative_overlay=narrative_overlay,
            )

            self._last_success = datetime.now(timezone.utc)
            log.info(
                "Regime mismatch detection complete",
                mismatch_signals=mismatch_signals,
            )
            return regime_state

        except AgentProcessingError:
            raise
        except Exception as e:
            self._error_count_24h += 1
            raise AgentProcessingError(
                message=f"REGIME-MISMATCH processing failed: {e}",
                agent_id=self.agent_id,
            ) from e

    def get_health(self) -> HealthStatus:
        """Report health status."""
        if self._error_count_24h > 10:
            status = AgentStatus.UNHEALTHY
        elif self._error_count_24h > 3:
            status = AgentStatus.DEGRADED
        else:
            status = AgentStatus.HEALTHY

        return HealthStatus(
            agent_id=self.agent_id,
            status=status,
            last_run=self._last_run,
            last_success=self._last_success,
            error_count_24h=self._error_count_24h,
        )
