"""INVALID-MON: Invalidation Condition Monitoring Agent.

Monitors all active invalidation conditions across active positions.
Evaluates current market data against thresholds and computes breach
magnitude, velocity, and suggested confidence impact.

Spec Reference: Technical Spec v2.3, InvalidationCondition schema

Classification: FROZEN — zero LLM calls. Pure computation.

Breach semantics:
  - breach_magnitude: |current - threshold| / |threshold|
    * < 0.05 → marginal, confidence impact 0.10-0.20
    * 0.05-0.20 → moderate, confidence impact 0.30-0.50
    * > 0.20 → strong, confidence impact 0.60-0.80
  - breach_velocity: rate of approach per day (trailing 5-day avg)

Input: AgentContext with metadata:
  - metadata["active_beliefs"]: list of belief dicts with invalidation_conditions
  - metadata["current_values"]: dict of metric -> current_value for evaluation

Output: InvalidationMonitorOutput with per-condition monitoring results.
"""

from datetime import datetime, timezone
from typing import Any, Optional
from uuid import UUID

import structlog

from providence.agents.base import AgentContext, AgentStatus, BaseAgent, HealthStatus
from providence.exceptions import AgentProcessingError
from providence.schemas.exit import InvalidationMonitorOutput, MonitoredCondition

logger = structlog.get_logger()

# Operator evaluation functions
OPERATORS: dict[str, Any] = {
    "GT": lambda current, threshold: current > threshold,
    "LT": lambda current, threshold: current < threshold,
    "EQ": lambda current, threshold: abs(current - threshold) < 1e-9,
    "CROSSES_ABOVE": lambda current, threshold: current > threshold,
    "CROSSES_BELOW": lambda current, threshold: current < threshold,
}

# Breach magnitude -> confidence impact mapping
BREACH_IMPACT_MARGINAL = 0.15  # magnitude < 0.05
BREACH_IMPACT_MODERATE = 0.40  # 0.05 <= magnitude < 0.20
BREACH_IMPACT_STRONG = 0.70  # magnitude >= 0.20

# Approaching threshold (within 10%)
APPROACH_THRESHOLD_PCT = 0.10


def evaluate_condition(
    condition: dict[str, Any],
    current_values: dict[str, float],
    historical_values: dict[str, list[float]] | None = None,
) -> dict[str, Any]:
    """Evaluate a single invalidation condition against current data.

    Args:
        condition: Invalidation condition dict from a belief.
        current_values: Dict of metric -> current value.
        historical_values: Optional dict of metric -> recent values (for velocity).

    Returns:
        Dict with evaluation results.
    """
    metric = condition.get("metric", "")
    operator = condition.get("operator", "GT")
    threshold = float(condition.get("threshold", 0.0))
    condition_id = condition.get("condition_id", "")

    current_value = current_values.get(metric)

    result = {
        "condition_id": condition_id,
        "metric": metric,
        "operator": operator,
        "threshold": threshold,
        "current_value": current_value,
        "is_breached": False,
        "breach_magnitude": 0.0,
        "breach_velocity": 0.0,
        "confidence_impact": 0.0,
    }

    if current_value is None:
        return result

    # Check if breached
    eval_fn = OPERATORS.get(operator)
    if eval_fn is None:
        return result

    is_breached = eval_fn(current_value, threshold)
    result["is_breached"] = is_breached

    # Compute breach magnitude: |current - threshold| / |threshold|
    if abs(threshold) > 1e-9:
        magnitude = abs(current_value - threshold) / abs(threshold)
    else:
        magnitude = abs(current_value - threshold)
    result["breach_magnitude"] = round(magnitude, 6)

    # Compute confidence impact based on breach magnitude
    if is_breached:
        if magnitude < 0.05:
            result["confidence_impact"] = BREACH_IMPACT_MARGINAL
        elif magnitude < 0.20:
            result["confidence_impact"] = BREACH_IMPACT_MODERATE
        else:
            result["confidence_impact"] = BREACH_IMPACT_STRONG

    # Compute breach velocity from historical values
    if historical_values and metric in historical_values:
        hist = historical_values[metric]
        if len(hist) >= 2:
            # trailing 5-day average velocity (or fewer days if less data)
            window = hist[-min(5, len(hist)):]
            if len(window) >= 2:
                daily_changes = [
                    window[i] - window[i - 1]
                    for i in range(1, len(window))
                ]
                avg_velocity = sum(daily_changes) / len(daily_changes)
                result["breach_velocity"] = round(avg_velocity, 6)

    return result


def is_approaching(
    current_value: float | None,
    threshold: float,
    operator: str,
) -> bool:
    """Check if a metric is within 10% of its invalidation threshold.

    Args:
        current_value: Current metric value.
        threshold: Invalidation threshold.
        operator: Comparison operator.

    Returns:
        True if approaching (within 10% of threshold).
    """
    if current_value is None:
        return False

    if abs(threshold) < 1e-9:
        return abs(current_value) < APPROACH_THRESHOLD_PCT

    distance_pct = abs(current_value - threshold) / abs(threshold)

    # For GT/CROSSES_ABOVE: approaching if current is below threshold and close
    if operator in ("GT", "CROSSES_ABOVE"):
        return current_value <= threshold and distance_pct < APPROACH_THRESHOLD_PCT

    # For LT/CROSSES_BELOW: approaching if current is above threshold and close
    if operator in ("LT", "CROSSES_BELOW"):
        return current_value >= threshold and distance_pct < APPROACH_THRESHOLD_PCT

    return distance_pct < APPROACH_THRESHOLD_PCT


class InvalidMon(BaseAgent[InvalidationMonitorOutput]):
    """Invalidation condition monitoring agent.

    Monitors all active invalidation conditions across active positions.
    Evaluates current market data against thresholds and computes breach
    magnitude, velocity, and suggested confidence impact.

    FROZEN: Zero LLM calls.
    """

    CONSUMED_DATA_TYPES: set = set()

    def __init__(self) -> None:
        super().__init__(
            agent_id="INVALID-MON",
            agent_type="exit",
            version="1.0.0",
        )
        self._last_run: Optional[datetime] = None
        self._last_success: Optional[datetime] = None
        self._error_count_24h: int = 0

    async def process(self, context: AgentContext) -> InvalidationMonitorOutput:
        """Monitor all active invalidation conditions.

        Steps:
          1. EXTRACT active beliefs and current values from metadata
          2. ITERATE over all invalidation conditions
          3. EVALUATE each condition against current data
          4. COMPUTE breach magnitude, velocity, confidence impact
          5. EMIT InvalidationMonitorOutput

        Args:
            context: AgentContext with active_beliefs and current_values.

        Returns:
            InvalidationMonitorOutput with per-condition results.
        """
        self._last_run = datetime.now(timezone.utc)

        try:
            log = logger.bind(
                agent_id=self.agent_id,
                context_hash=context.context_window_hash,
            )
            log.info("Starting invalidation monitoring")

            active_beliefs = context.metadata.get("active_beliefs", [])
            current_values = context.metadata.get("current_values", {})
            historical_values = context.metadata.get("historical_values")

            if not isinstance(active_beliefs, list):
                active_beliefs = []
            if not isinstance(current_values, dict):
                current_values = {}

            conditions: list[MonitoredCondition] = []
            total_breached = 0
            total_approaching = 0

            for belief in active_beliefs:
                if not isinstance(belief, dict):
                    continue

                thesis_id = belief.get("thesis_id", "")
                agent_id = belief.get("agent_id", "")
                ticker = belief.get("ticker", "")
                inv_conditions = belief.get("invalidation_conditions", [])

                if not isinstance(inv_conditions, list):
                    continue

                for cond in inv_conditions:
                    if not isinstance(cond, dict):
                        continue

                    # Skip already triggered/expired conditions
                    status = cond.get("status", "ACTIVE")
                    if status != "ACTIVE":
                        continue

                    result = evaluate_condition(
                        cond, current_values, historical_values,
                    )

                    # Parse condition_id
                    cond_id_raw = result["condition_id"]
                    try:
                        cond_id = UUID(str(cond_id_raw))
                    except (ValueError, TypeError):
                        from uuid import uuid4
                        cond_id = uuid4()

                    monitored = MonitoredCondition(
                        condition_id=cond_id,
                        source_thesis_id=thesis_id,
                        source_agent_id=agent_id,
                        ticker=ticker,
                        metric=result["metric"],
                        operator=result["operator"],
                        threshold=result["threshold"],
                        current_value=result["current_value"],
                        is_breached=result["is_breached"],
                        breach_magnitude=round(result["breach_magnitude"], 6),
                        breach_velocity=round(result["breach_velocity"], 6),
                        confidence_impact=round(result["confidence_impact"], 4),
                    )
                    conditions.append(monitored)

                    if result["is_breached"]:
                        total_breached += 1
                    elif is_approaching(
                        result["current_value"],
                        result["threshold"],
                        result["operator"],
                    ):
                        total_approaching += 1

            output = InvalidationMonitorOutput(
                agent_id=self.agent_id,
                timestamp=datetime.now(timezone.utc),
                context_window_hash=context.context_window_hash,
                conditions=conditions,
                total_conditions=len(conditions),
                conditions_breached=total_breached,
                conditions_approaching=total_approaching,
            )

            self._last_success = datetime.now(timezone.utc)
            log.info(
                "Invalidation monitoring complete",
                total=len(conditions),
                breached=total_breached,
                approaching=total_approaching,
            )
            return output

        except AgentProcessingError:
            raise
        except Exception as e:
            self._error_count_24h += 1
            raise AgentProcessingError(
                message=f"INVALID-MON processing failed: {e}",
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
