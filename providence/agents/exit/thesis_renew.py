"""THESIS-RENEW: Thesis Renewal Agent.

Evaluates active theses approaching time horizon expiry for renewal
eligibility. Considers condition health, regime alignment, confidence
decay, and upside asymmetry to determine whether a thesis should be
renewed with updated parameters or allowed to expire.

Spec Reference: Technical Spec v2.3, Phase 3

Classification: FROZEN — zero LLM calls. Pure computation.

Critical Rules:
  - COGNIT-EXIT defers CLOSE if renewal_pending AND asymmetry > 0.5
  - Renewal extends time horizon and adjusts confidence based on health
  - Only theses within renewal_window_days of expiry are evaluated
  - Minimum health_score of 0.40 required for renewal

Input: AgentContext with metadata:
  - metadata["active_beliefs"]: list of belief dicts
  - metadata["invalidation_state"]: dict of thesis_id -> condition status
  - metadata["regime_state"]: serialized RegimeStateObject dict

Output: ThesisRenewalOutput with per-thesis renewal decisions.
"""

from datetime import datetime, timezone
from typing import Any, Optional

import structlog

from providence.agents.base import AgentContext, AgentStatus, BaseAgent, HealthStatus
from providence.exceptions import AgentProcessingError
from providence.schemas.exit import RenewalCandidate, ThesisRenewalOutput

logger = structlog.get_logger()

# Renewal parameters
RENEWAL_WINDOW_DAYS = 14  # Evaluate theses within 14 days of expiry
MIN_HEALTH_FOR_RENEWAL = 0.40  # Minimum health score to qualify
MIN_CONFIDENCE_FOR_RENEWAL = 0.25  # Minimum confidence to qualify
MAX_HORIZON_EXTENSION_DAYS = 90  # Maximum horizon extension
DEFAULT_HORIZON_EXTENSION_DAYS = 30  # Default extension

# Confidence decay: linear decay over time horizon
# At 50% through horizon → 10% decay, at 75% → 25% decay, at 100% → 40% decay
DECAY_AT_50_PCT = 0.10
DECAY_AT_75_PCT = 0.25
DECAY_AT_100_PCT = 0.40

# Regime alignment bonuses/penalties
REGIME_ALIGNMENT_BONUS: dict[str, float] = {
    "NORMAL": 0.0,
    "CAUTIOUS": -0.05,
    "DEFENSIVE": -0.10,
    "HALTED": -1.0,  # No renewals in HALTED
}

# Asymmetry thresholds
ASYMMETRY_STRONG = 0.7  # Strong upside asymmetry
ASYMMETRY_MODERATE = 0.5  # Moderate asymmetry (deferral threshold)


def compute_confidence_decay(
    days_elapsed: int,
    time_horizon_days: int,
) -> float:
    """Compute time-based confidence decay factor.

    Args:
        days_elapsed: Days since thesis was created.
        time_horizon_days: Original time horizon in days.

    Returns:
        Decay factor [0.0, 1.0] where 0 = no decay, 1 = full decay.
    """
    if time_horizon_days <= 0:
        return DECAY_AT_100_PCT

    pct_elapsed = min(1.0, days_elapsed / time_horizon_days)

    if pct_elapsed <= 0.5:
        # Linear interpolation from 0 to DECAY_AT_50_PCT
        return DECAY_AT_50_PCT * (pct_elapsed / 0.5)
    elif pct_elapsed <= 0.75:
        # Linear interpolation from DECAY_AT_50_PCT to DECAY_AT_75_PCT
        t = (pct_elapsed - 0.5) / 0.25
        return DECAY_AT_50_PCT + t * (DECAY_AT_75_PCT - DECAY_AT_50_PCT)
    else:
        # Linear interpolation from DECAY_AT_75_PCT to DECAY_AT_100_PCT
        t = (pct_elapsed - 0.75) / 0.25
        return DECAY_AT_75_PCT + t * (DECAY_AT_100_PCT - DECAY_AT_75_PCT)


def compute_asymmetry_score(
    health_score: float,
    confidence: float,
    regime_adjustment: float,
) -> float:
    """Compute upside asymmetry score for renewal decision.

    Higher scores indicate more favorable risk/reward for renewal.

    Args:
        health_score: Thesis health [0, 1].
        confidence: Current confidence [0, 1].
        regime_adjustment: Regime alignment bonus/penalty.

    Returns:
        Asymmetry score [0, 1].
    """
    # Base asymmetry from health and confidence
    base = (health_score * 0.6) + (confidence * 0.4)

    # Apply regime adjustment
    adjusted = base + regime_adjustment

    return round(max(0.0, min(1.0, adjusted)), 4)


def evaluate_renewal(
    belief: dict[str, Any],
    invalidation_state: dict[str, Any],
    risk_mode: str,
) -> dict[str, Any]:
    """Evaluate a single thesis for renewal eligibility.

    Args:
        belief: Active belief dict.
        invalidation_state: Dict with condition health for this thesis.
        risk_mode: Current system risk mode.

    Returns:
        Dict with renewal evaluation results.
    """
    thesis_id = belief.get("thesis_id", "")
    ticker = belief.get("ticker", "")
    original_confidence = float(belief.get("raw_confidence", 0.5))
    time_horizon_days = int(belief.get("time_horizon_days", 60))
    days_elapsed = int(belief.get("days_elapsed", 0))

    # Compute days remaining
    days_remaining = max(0, time_horizon_days - days_elapsed)

    # Check if within renewal window
    in_window = days_remaining <= RENEWAL_WINDOW_DAYS

    # Get condition health
    thesis_state = invalidation_state.get(thesis_id, {})
    if isinstance(thesis_state, dict):
        conditions_healthy = int(thesis_state.get("conditions_healthy", 0))
        conditions_total = int(thesis_state.get("conditions_total", 0))
    else:
        conditions_healthy = 0
        conditions_total = 0

    # Compute health score
    if conditions_total > 0:
        health_score = conditions_healthy / conditions_total
    else:
        health_score = 1.0

    # Compute confidence decay
    decay = compute_confidence_decay(days_elapsed, time_horizon_days)

    # Regime adjustment
    regime_adj = REGIME_ALIGNMENT_BONUS.get(risk_mode, 0.0)

    # Compute asymmetry
    asymmetry = compute_asymmetry_score(
        health_score, original_confidence, regime_adj,
    )

    # Compute renewed confidence
    renewed_confidence = original_confidence * (1.0 - decay) + regime_adj
    renewed_confidence = max(0.0, min(1.0, renewed_confidence))

    # Compute renewed horizon
    if health_score >= 0.70:
        extension = DEFAULT_HORIZON_EXTENSION_DAYS
    elif health_score >= MIN_HEALTH_FOR_RENEWAL:
        extension = DEFAULT_HORIZON_EXTENSION_DAYS // 2
    else:
        extension = 0
    renewed_horizon = min(
        time_horizon_days + extension,
        time_horizon_days + MAX_HORIZON_EXTENSION_DAYS,
    )

    # Determine if renewal is approved
    is_renewed = (
        in_window
        and health_score >= MIN_HEALTH_FOR_RENEWAL
        and renewed_confidence >= MIN_CONFIDENCE_FOR_RENEWAL
        and risk_mode != "HALTED"
    )

    # Build renewal reason
    if not in_window:
        reason = f"Not in renewal window ({days_remaining} days remaining > {RENEWAL_WINDOW_DAYS})"
    elif risk_mode == "HALTED":
        reason = "System HALTED — no renewals"
    elif health_score < MIN_HEALTH_FOR_RENEWAL:
        reason = f"Health score {health_score:.2f} below minimum {MIN_HEALTH_FOR_RENEWAL}"
    elif renewed_confidence < MIN_CONFIDENCE_FOR_RENEWAL:
        reason = f"Renewed confidence {renewed_confidence:.2f} below minimum {MIN_CONFIDENCE_FOR_RENEWAL}"
    elif is_renewed:
        reason = (
            f"Renewed: health={health_score:.2f}, "
            f"asymmetry={asymmetry:.2f}, "
            f"confidence {original_confidence:.2f}→{renewed_confidence:.2f}"
        )
    else:
        reason = "Renewal conditions not met"

    return {
        "thesis_id": thesis_id,
        "ticker": ticker,
        "original_confidence": round(original_confidence, 4),
        "renewed_confidence": round(renewed_confidence, 4),
        "confidence_delta": round(renewed_confidence - original_confidence, 4),
        "original_horizon_days": time_horizon_days,
        "renewed_horizon_days": renewed_horizon,
        "renewal_reason": reason,
        "asymmetry_score": asymmetry,
        "conditions_healthy": conditions_healthy,
        "conditions_total": conditions_total,
        "is_renewed": is_renewed,
    }


class ThesisRenew(BaseAgent[ThesisRenewalOutput]):
    """Thesis renewal agent.

    Evaluates active theses approaching time horizon expiry for
    renewal eligibility based on condition health, regime alignment,
    confidence decay, and upside asymmetry.

    Critical: COGNIT-EXIT defers CLOSE if renewal_pending AND asymmetry > 0.5.

    FROZEN: Zero LLM calls.
    """

    CONSUMED_DATA_TYPES: set = set()

    def __init__(self) -> None:
        super().__init__(
            agent_id="THESIS-RENEW",
            agent_type="exit",
            version="1.0.0",
        )
        self._last_run: Optional[datetime] = None
        self._last_success: Optional[datetime] = None
        self._error_count_24h: int = 0

    async def process(self, context: AgentContext) -> ThesisRenewalOutput:
        """Evaluate theses for renewal.

        Steps:
          1. EXTRACT active beliefs, invalidation state, regime
          2. FILTER to theses within renewal window
          3. EVALUATE each thesis for renewal eligibility
          4. EMIT ThesisRenewalOutput

        Args:
            context: AgentContext with belief and invalidation data.

        Returns:
            ThesisRenewalOutput with per-thesis renewal decisions.
        """
        self._last_run = datetime.now(timezone.utc)

        try:
            log = logger.bind(
                agent_id=self.agent_id,
                context_hash=context.context_window_hash,
            )
            log.info("Starting thesis renewal evaluation")

            active_beliefs = context.metadata.get("active_beliefs", [])
            invalidation_state = context.metadata.get("invalidation_state", {})
            regime_state = context.metadata.get("regime_state", {})

            if not isinstance(active_beliefs, list):
                active_beliefs = []
            if not isinstance(invalidation_state, dict):
                invalidation_state = {}
            if not isinstance(regime_state, dict):
                regime_state = {}

            risk_mode = regime_state.get("system_risk_mode", "NORMAL")

            candidates: list[RenewalCandidate] = []
            total_renewed = 0
            total_expired = 0

            for belief in active_beliefs:
                if not isinstance(belief, dict):
                    continue

                result = evaluate_renewal(belief, invalidation_state, risk_mode)
                candidates.append(RenewalCandidate(**result))

                if result["is_renewed"]:
                    total_renewed += 1
                else:
                    total_expired += 1

            output = ThesisRenewalOutput(
                agent_id=self.agent_id,
                timestamp=datetime.now(timezone.utc),
                context_window_hash=context.context_window_hash,
                candidates=candidates,
                total_evaluated=len(candidates),
                total_renewed=total_renewed,
                total_expired=total_expired,
            )

            self._last_success = datetime.now(timezone.utc)
            log.info(
                "Thesis renewal evaluation complete",
                evaluated=len(candidates),
                renewed=total_renewed,
                expired=total_expired,
            )
            return output

        except AgentProcessingError:
            raise
        except Exception as e:
            self._error_count_24h += 1
            raise AgentProcessingError(
                message=f"THESIS-RENEW processing failed: {e}",
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
