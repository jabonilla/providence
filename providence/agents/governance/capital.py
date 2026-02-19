"""GOVERN-CAPITAL: Capital Tier Classification Agent.

Classifies AUM into capital tiers and derives execution constraints.
Capital tiers control system autonomy: SEED (shadow only) → GROWTH
(limited execution) → SCALE (full execution) → INSTITUTIONAL (enhanced monitoring).

Spec Reference: Technical Spec v2.3, Phase 4 (Governance)

Classification: FROZEN — zero LLM calls. Pure computation.

Input: AgentContext with metadata:
  - metadata["current_aum"]: float (current AUM in USD)
  - metadata["previous_tier"]: str (previous capital tier, optional)

Output: CapitalTierOutput with tier classification and constraints.
"""

from datetime import datetime, timezone
from typing import Any, Optional

import structlog

from providence.agents.base import AgentContext, AgentStatus, BaseAgent, HealthStatus
from providence.exceptions import AgentProcessingError
from providence.schemas.enums import CapitalTier
from providence.schemas.governance import CapitalTierOutput, TierConstraints

logger = structlog.get_logger()

# Tier thresholds in USD
TIER_THRESHOLDS: dict[CapitalTier, float] = {
    CapitalTier.SEED: 0.0,
    CapitalTier.GROWTH: 10_000_000.0,       # $10M
    CapitalTier.SCALE: 100_000_000.0,        # $100M
    CapitalTier.INSTITUTIONAL: 500_000_000.0, # $500M
}

# Execution constraints per tier
TIER_CONSTRAINTS: dict[CapitalTier, dict[str, Any]] = {
    CapitalTier.SEED: {
        "max_position_weight": 0.0,
        "max_gross_exposure": 0.0,
        "max_single_sector_pct": 0.0,
        "min_confidence_threshold": 1.0,  # Unreachable → no live execution
        "live_execution_enabled": False,
        "max_positions": 0,
    },
    CapitalTier.GROWTH: {
        "max_position_weight": 0.05,
        "max_gross_exposure": 0.80,
        "max_single_sector_pct": 0.25,
        "min_confidence_threshold": 0.70,
        "live_execution_enabled": True,
        "max_positions": 15,
    },
    CapitalTier.SCALE: {
        "max_position_weight": 0.10,
        "max_gross_exposure": 1.20,
        "max_single_sector_pct": 0.30,
        "min_confidence_threshold": 0.60,
        "live_execution_enabled": True,
        "max_positions": 30,
    },
    CapitalTier.INSTITUTIONAL: {
        "max_position_weight": 0.08,
        "max_gross_exposure": 1.60,
        "max_single_sector_pct": 0.25,
        "min_confidence_threshold": 0.55,
        "live_execution_enabled": True,
        "max_positions": 50,
    },
}


def classify_tier(aum: float) -> CapitalTier:
    """Classify AUM into a capital tier.

    Args:
        aum: Current AUM in USD.

    Returns:
        CapitalTier classification.
    """
    if aum >= TIER_THRESHOLDS[CapitalTier.INSTITUTIONAL]:
        return CapitalTier.INSTITUTIONAL
    if aum >= TIER_THRESHOLDS[CapitalTier.SCALE]:
        return CapitalTier.SCALE
    if aum >= TIER_THRESHOLDS[CapitalTier.GROWTH]:
        return CapitalTier.GROWTH
    return CapitalTier.SEED


def get_tier_constraints(tier: CapitalTier) -> TierConstraints:
    """Get execution constraints for a capital tier.

    Args:
        tier: Capital tier.

    Returns:
        TierConstraints for the given tier.
    """
    return TierConstraints(**TIER_CONSTRAINTS[tier])


def compute_headroom(aum: float, current_tier: CapitalTier) -> float:
    """Compute how close AUM is to the next tier threshold.

    Args:
        aum: Current AUM.
        current_tier: Current tier classification.

    Returns:
        Percentage (0-100) of progress toward next tier.
    """
    tier_order = [CapitalTier.SEED, CapitalTier.GROWTH, CapitalTier.SCALE, CapitalTier.INSTITUTIONAL]
    idx = tier_order.index(current_tier)

    if idx >= len(tier_order) - 1:
        return 100.0  # Already at top tier

    next_threshold = TIER_THRESHOLDS[tier_order[idx + 1]]
    current_threshold = TIER_THRESHOLDS[current_tier]
    range_size = next_threshold - current_threshold

    if range_size <= 0:
        return 100.0

    progress = (aum - current_threshold) / range_size * 100.0
    return round(min(max(progress, 0.0), 100.0), 2)


class GovernCapital(BaseAgent[CapitalTierOutput]):
    """Capital tier classification agent.

    Classifies AUM into capital tiers and derives execution constraints
    that control system autonomy level.

    FROZEN: Zero LLM calls. Pure computation.
    """

    CONSUMED_DATA_TYPES: set = set()

    def __init__(self) -> None:
        super().__init__(
            agent_id="GOVERN-CAPITAL",
            agent_type="governance",
            version="1.0.0",
        )
        self._last_run: Optional[datetime] = None
        self._last_success: Optional[datetime] = None
        self._error_count_24h: int = 0

    async def process(self, context: AgentContext) -> CapitalTierOutput:
        """Classify AUM and derive capital tier constraints.

        Steps:
          1. EXTRACT current AUM and previous tier from metadata
          2. CLASSIFY AUM into capital tier
          3. DERIVE execution constraints
          4. COMPUTE headroom to next tier
          5. EMIT CapitalTierOutput

        Args:
            context: AgentContext with AUM in metadata.

        Returns:
            CapitalTierOutput with tier classification and constraints.
        """
        self._last_run = datetime.now(timezone.utc)

        try:
            log = logger.bind(agent_id=self.agent_id, context_hash=context.context_window_hash)
            log.info("Starting capital tier classification")

            current_aum = float(context.metadata.get("current_aum", 0.0))
            prev_tier_str = context.metadata.get("previous_tier", "")

            # Classify
            tier = classify_tier(current_aum)
            constraints = get_tier_constraints(tier)
            headroom = compute_headroom(current_aum, tier)

            # Previous tier
            previous_tier: Optional[CapitalTier] = None
            if prev_tier_str:
                try:
                    previous_tier = CapitalTier(prev_tier_str)
                except ValueError:
                    previous_tier = None

            tier_changed = previous_tier is not None and previous_tier != tier

            output = CapitalTierOutput(
                agent_id=self.agent_id,
                timestamp=datetime.now(timezone.utc),
                context_window_hash=context.context_window_hash,
                current_aum=current_aum,
                current_tier=tier,
                previous_tier=previous_tier,
                tier_changed=tier_changed,
                constraints=constraints,
                headroom_to_next_tier_pct=headroom,
            )

            self._last_success = datetime.now(timezone.utc)
            log.info(
                "Capital tier classified",
                aum=current_aum,
                tier=tier.value,
                changed=tier_changed,
                headroom=headroom,
            )
            return output

        except AgentProcessingError:
            raise
        except Exception as e:
            self._error_count_24h += 1
            raise AgentProcessingError(
                message=f"GOVERN-CAPITAL processing failed: {e}",
                agent_id=self.agent_id,
            ) from e

    def get_health(self) -> HealthStatus:
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
