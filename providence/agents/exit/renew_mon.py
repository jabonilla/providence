"""RENEW-MON: Renewal Monitoring Agent.

Aggregates condition health, time decay, and renewal status across all
active beliefs. Identifies renewal candidates and computes per-belief
health scores. Feeds into THESIS-RENEW and COGNIT-EXIT.

Spec Reference: Technical Spec v2.3, Phase 3

Classification: FROZEN — zero LLM calls. Pure computation.

Health score semantics:
  - [0.80, 1.0] → HEALTHY: all conditions intact, time horizon safe
  - [0.50, 0.80) → DEGRADED: some conditions breached or approaching
  - [0.0, 0.50) → CRITICAL: multiple breaches or near expiry
  - is_renewal_candidate when days_remaining <= renewal_window AND health >= min

Input: AgentContext with metadata:
  - metadata["active_beliefs"]: list of belief dicts
  - metadata["invalidation_results"]: list of MonitoredCondition dicts from INVALID-MON
  - metadata["regime_state"]: serialized RegimeStateObject dict

Output: RenewalMonitorOutput with per-belief health reports.
"""

from datetime import datetime, timezone
from typing import Any, Optional

import structlog

from providence.agents.base import AgentContext, AgentStatus, BaseAgent, HealthStatus
from providence.exceptions import AgentProcessingError
from providence.schemas.exit import BeliefHealthReport, RenewalMonitorOutput

logger = structlog.get_logger()

# Health thresholds
HEALTHY_THRESHOLD = 0.80
DEGRADED_THRESHOLD = 0.50

# Renewal eligibility
RENEWAL_WINDOW_DAYS = 14
MIN_HEALTH_FOR_RENEWAL = 0.40

# Urgency levels based on days remaining and health
URGENCY_IMMEDIATE_DAYS = 3
URGENCY_HIGH_DAYS = 7

# Approaching weight in health calculation
APPROACHING_PENALTY = 0.10  # Per approaching condition


def compute_belief_health(
    thesis_id: str,
    ticker: str,
    conditions_by_thesis: dict[str, list[dict[str, Any]]],
) -> tuple[float, int, int, int]:
    """Compute health score for a single belief from its conditions.

    Args:
        thesis_id: Thesis identifier.
        ticker: Ticker for this thesis.
        conditions_by_thesis: Dict of thesis_id -> list of condition results.

    Returns:
        Tuple of (health_score, healthy, breached, approaching).
    """
    conditions = conditions_by_thesis.get(thesis_id, [])

    if not conditions:
        # No conditions monitored → assume healthy
        return 1.0, 0, 0, 0

    healthy = 0
    breached = 0
    approaching = 0

    for cond in conditions:
        if not isinstance(cond, dict):
            continue

        is_breached = cond.get("is_breached", False)
        breach_magnitude = float(cond.get("breach_magnitude", 0.0))

        if is_breached:
            breached += 1
        else:
            # Check if approaching (breach_magnitude < 0.10 means close)
            threshold = cond.get("threshold", 0)
            current = cond.get("current_value")
            if current is not None and abs(threshold) > 1e-9:
                distance_pct = abs(current - threshold) / abs(threshold)
                if distance_pct < 0.10:
                    approaching += 1
                else:
                    healthy += 1
            else:
                healthy += 1

    total = healthy + breached + approaching
    if total == 0:
        return 1.0, 0, 0, 0

    # Health = (healthy / total) - penalty for approaching conditions
    base_health = healthy / total
    approach_penalty = approaching * APPROACHING_PENALTY
    health = max(0.0, base_health - approach_penalty)

    return round(health, 4), healthy, breached, approaching


def compute_confidence_decay(
    days_elapsed: int,
    time_horizon_days: int,
) -> float:
    """Compute time-based confidence decay.

    Same formula as THESIS-RENEW for consistency.

    Args:
        days_elapsed: Days since thesis created.
        time_horizon_days: Original time horizon.

    Returns:
        Decay factor [0, 1].
    """
    if time_horizon_days <= 0:
        return 0.40

    pct_elapsed = min(1.0, days_elapsed / time_horizon_days)

    if pct_elapsed <= 0.5:
        return 0.10 * (pct_elapsed / 0.5)
    elif pct_elapsed <= 0.75:
        t = (pct_elapsed - 0.5) / 0.25
        return 0.10 + t * (0.25 - 0.10)
    else:
        t = (pct_elapsed - 0.75) / 0.25
        return 0.25 + t * (0.40 - 0.25)


def determine_renewal_urgency(
    days_remaining: int,
    health_score: float,
    is_candidate: bool,
) -> str:
    """Determine renewal urgency level.

    Args:
        days_remaining: Days remaining on thesis horizon.
        health_score: Current health score.
        is_candidate: Whether this is a renewal candidate.

    Returns:
        Urgency: NONE, LOW, MEDIUM, or HIGH.
    """
    if not is_candidate:
        return "NONE"

    if days_remaining <= URGENCY_IMMEDIATE_DAYS:
        return "HIGH"
    elif days_remaining <= URGENCY_HIGH_DAYS:
        if health_score < DEGRADED_THRESHOLD:
            return "HIGH"
        return "MEDIUM"
    elif health_score < DEGRADED_THRESHOLD:
        return "MEDIUM"
    else:
        return "LOW"


class RenewMon(BaseAgent[RenewalMonitorOutput]):
    """Renewal monitoring agent.

    Aggregates condition health, time decay, and renewal status across
    all active beliefs. Identifies renewal candidates and computes
    per-belief health scores.

    FROZEN: Zero LLM calls.
    """

    CONSUMED_DATA_TYPES: set = set()

    def __init__(self) -> None:
        super().__init__(
            agent_id="RENEW-MON",
            agent_type="exit",
            version="1.0.0",
        )
        self._last_run: Optional[datetime] = None
        self._last_success: Optional[datetime] = None
        self._error_count_24h: int = 0

    async def process(self, context: AgentContext) -> RenewalMonitorOutput:
        """Monitor belief health and identify renewal candidates.

        Steps:
          1. EXTRACT active beliefs, invalidation results, regime
          2. INDEX invalidation results by thesis_id
          3. COMPUTE health score per belief
          4. DETERMINE renewal candidacy and urgency
          5. EMIT RenewalMonitorOutput

        Args:
            context: AgentContext with beliefs and invalidation results.

        Returns:
            RenewalMonitorOutput with per-belief health reports.
        """
        self._last_run = datetime.now(timezone.utc)

        try:
            log = logger.bind(
                agent_id=self.agent_id,
                context_hash=context.context_window_hash,
            )
            log.info("Starting renewal monitoring")

            active_beliefs = context.metadata.get("active_beliefs", [])
            invalidation_results = context.metadata.get("invalidation_results", [])
            regime_state = context.metadata.get("regime_state", {})

            if not isinstance(active_beliefs, list):
                active_beliefs = []
            if not isinstance(invalidation_results, list):
                invalidation_results = []
            if not isinstance(regime_state, dict):
                regime_state = {}

            risk_mode = regime_state.get("system_risk_mode", "NORMAL")

            # Index invalidation results by thesis_id
            conditions_by_thesis: dict[str, list[dict[str, Any]]] = {}
            for result in invalidation_results:
                if not isinstance(result, dict):
                    continue
                thesis_id = result.get("source_thesis_id", "")
                if thesis_id:
                    if thesis_id not in conditions_by_thesis:
                        conditions_by_thesis[thesis_id] = []
                    conditions_by_thesis[thesis_id].append(result)

            reports: list[BeliefHealthReport] = []
            healthy_count = 0
            degraded_count = 0
            critical_count = 0
            renewal_count = 0

            for belief in active_beliefs:
                if not isinstance(belief, dict):
                    continue

                thesis_id = belief.get("thesis_id", "")
                ticker = belief.get("ticker", "")
                agent_id = belief.get("agent_id", "")
                time_horizon = int(belief.get("time_horizon_days", 60))
                days_elapsed = int(belief.get("days_elapsed", 0))
                days_remaining = max(0, time_horizon - days_elapsed)

                # Compute health
                health, cond_healthy, cond_breached, cond_approaching = (
                    compute_belief_health(thesis_id, ticker, conditions_by_thesis)
                )

                # Compute decay
                decay = compute_confidence_decay(days_elapsed, time_horizon)

                # Determine renewal candidacy
                is_candidate = (
                    days_remaining <= RENEWAL_WINDOW_DAYS
                    and health >= MIN_HEALTH_FOR_RENEWAL
                    and risk_mode != "HALTED"
                )

                # Determine urgency
                urgency = determine_renewal_urgency(
                    days_remaining, health, is_candidate,
                )

                report = BeliefHealthReport(
                    thesis_id=thesis_id,
                    ticker=ticker,
                    agent_id=agent_id,
                    health_score=health,
                    conditions_healthy=cond_healthy,
                    conditions_breached=cond_breached,
                    conditions_approaching=cond_approaching,
                    days_remaining=days_remaining,
                    confidence_decay=round(decay, 4),
                    is_renewal_candidate=is_candidate,
                    renewal_urgency=urgency,
                )
                reports.append(report)

                # Classify
                if health >= HEALTHY_THRESHOLD:
                    healthy_count += 1
                elif health >= DEGRADED_THRESHOLD:
                    degraded_count += 1
                else:
                    critical_count += 1

                if is_candidate:
                    renewal_count += 1

            output = RenewalMonitorOutput(
                agent_id=self.agent_id,
                timestamp=datetime.now(timezone.utc),
                context_window_hash=context.context_window_hash,
                reports=reports,
                total_beliefs=len(reports),
                healthy_beliefs=healthy_count,
                degraded_beliefs=degraded_count,
                critical_beliefs=critical_count,
                renewal_candidates=renewal_count,
            )

            self._last_success = datetime.now(timezone.utc)
            log.info(
                "Renewal monitoring complete",
                total=len(reports),
                healthy=healthy_count,
                degraded=degraded_count,
                critical=critical_count,
                candidates=renewal_count,
            )
            return output

        except AgentProcessingError:
            raise
        except Exception as e:
            self._error_count_24h += 1
            raise AgentProcessingError(
                message=f"RENEW-MON processing failed: {e}",
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
