"""LEARN-CALIB: Confidence Calibration Agent.

Analyzes confidence calibration of each Research Agent by comparing
stated confidence levels to realized directional accuracy. Detects
systematic overconfidence or underconfidence and suggests adjustments.

Spec Reference: Technical Spec v2.3, Phase 4 (Learning)

Classification: FROZEN — zero LLM calls. Pure computation.

Critical Rules:
  - Offline only. Never runs during live trading.
  - Brier score: mean( (confidence - outcome)^2 ), lower = better.
  - Calibration error: avg_stated_confidence - realized_accuracy.
    Positive = overconfident, negative = underconfident.
  - Well-calibrated: |calibration_error| < 0.05 per bucket.

Input: AgentContext with metadata:
  - metadata["belief_outcomes"]: list of dicts with agent_id, raw_confidence,
    direction, was_correct (bool), realized_return_bps
  - metadata["evaluation_start"]: ISO timestamp string
  - metadata["evaluation_end"]: ISO timestamp string

Output: CalibrationOutput with per-agent calibration profiles.
"""

import math
from datetime import datetime, timezone
from typing import Any, Optional

import structlog

from providence.agents.base import AgentContext, AgentStatus, BaseAgent, HealthStatus
from providence.exceptions import AgentProcessingError
from providence.schemas.learning import (
    AgentCalibration,
    CalibrationBucket,
    CalibrationOutput,
)

logger = structlog.get_logger()

# Calibration bucket boundaries
DEFAULT_BUCKETS = [
    (0.0, 0.2),
    (0.2, 0.4),
    (0.4, 0.6),
    (0.6, 0.8),
    (0.8, 1.0),
]

# Calibration thresholds
WELL_CALIBRATED_THRESHOLD = 0.05  # |calibration_error| < 5%
OVERCONFIDENT_THRESHOLD = 0.05  # overall calibration_error > 5%

# Degradation threshold for recommended adjustment
ADJUSTMENT_SCALE = 0.10  # Reduce/increase confidence by 10% per 0.10 calibration error


def compute_brier_score(confidence: float, outcome: bool) -> float:
    """Compute Brier score for a single prediction.

    Brier = (confidence - outcome)^2 where outcome is 1.0 (correct) or 0.0.

    Args:
        confidence: Stated confidence [0, 1].
        outcome: Whether the prediction was correct.

    Returns:
        Brier score [0, 1].
    """
    outcome_val = 1.0 if outcome else 0.0
    return (confidence - outcome_val) ** 2


def calibrate_bucket(
    beliefs: list[dict[str, Any]],
    lower: float,
    upper: float,
) -> dict[str, Any]:
    """Compute calibration for a single confidence bucket.

    Args:
        beliefs: Beliefs with confidence in [lower, upper).
        lower: Lower bound of bucket.
        upper: Upper bound of bucket.

    Returns:
        Dict with bucket calibration fields.
    """
    bucket_beliefs = [
        b for b in beliefs
        if isinstance(b, dict)
        and lower <= float(b.get("raw_confidence", 0.0)) < upper
    ]
    # Include upper bound for the last bucket
    if upper >= 1.0:
        bucket_beliefs = [
            b for b in beliefs
            if isinstance(b, dict)
            and lower <= float(b.get("raw_confidence", 0.0)) <= upper
        ]

    count = len(bucket_beliefs)
    if count == 0:
        return {
            "bucket_lower": lower,
            "bucket_upper": upper,
            "sample_count": 0,
            "avg_stated_confidence": 0.0,
            "realized_accuracy": 0.0,
            "calibration_error": 0.0,
            "brier_score": 0.0,
        }

    confidences = [float(b.get("raw_confidence", 0.0)) for b in bucket_beliefs]
    avg_conf = sum(confidences) / count

    correct = sum(1 for b in bucket_beliefs if b.get("was_correct", False))
    accuracy = correct / count

    brier_scores = [
        compute_brier_score(float(b.get("raw_confidence", 0.0)), b.get("was_correct", False))
        for b in bucket_beliefs
    ]
    avg_brier = sum(brier_scores) / count

    return {
        "bucket_lower": lower,
        "bucket_upper": upper,
        "sample_count": count,
        "avg_stated_confidence": round(avg_conf, 4),
        "realized_accuracy": round(accuracy, 4),
        "calibration_error": round(avg_conf - accuracy, 4),
        "brier_score": round(avg_brier, 4),
    }


def calibrate_agent(
    agent_id: str,
    beliefs: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute full calibration profile for a single agent.

    Args:
        agent_id: Agent identifier.
        beliefs: Beliefs with outcomes for this agent.

    Returns:
        Dict with agent calibration fields.
    """
    total = len(beliefs)

    # Compute buckets
    buckets = []
    for lower, upper in DEFAULT_BUCKETS:
        bucket = calibrate_bucket(beliefs, lower, upper)
        buckets.append(bucket)

    # Overall Brier score
    brier_scores = [
        compute_brier_score(float(b.get("raw_confidence", 0.0)), b.get("was_correct", False))
        for b in beliefs
        if isinstance(b, dict)
    ]
    overall_brier = round(sum(brier_scores) / len(brier_scores), 4) if brier_scores else 0.0

    # Overall calibration error (weighted by sample count)
    total_weighted_error = 0.0
    total_samples = 0
    for bucket in buckets:
        count = bucket["sample_count"]
        if count > 0:
            total_weighted_error += bucket["calibration_error"] * count
            total_samples += count

    overall_cal_error = round(total_weighted_error / total_samples, 4) if total_samples > 0 else 0.0

    is_overconfident = overall_cal_error > OVERCONFIDENT_THRESHOLD

    # Compute recommended adjustment
    # If overconfident by 0.10, recommend multiplying confidence by 0.90
    if abs(overall_cal_error) > WELL_CALIBRATED_THRESHOLD:
        adjustment = round(1.0 - overall_cal_error, 4)
        adjustment = max(0.50, min(1.50, adjustment))  # Clamp to reasonable range
    else:
        adjustment = 1.0  # Well-calibrated, no adjustment needed

    return {
        "agent_id": agent_id,
        "total_beliefs_evaluated": total,
        "overall_brier_score": overall_brier,
        "overall_calibration_error": overall_cal_error,
        "is_overconfident": is_overconfident,
        "recommended_adjustment": adjustment,
        "buckets": buckets,
    }


class LearnCalib(BaseAgent[CalibrationOutput]):
    """Offline confidence calibration agent.

    Analyzes each Research Agent's confidence calibration by comparing
    stated confidence to realized accuracy. Detects overconfidence and
    suggests adjustments.

    FROZEN: Zero LLM calls. Offline only.
    """

    CONSUMED_DATA_TYPES: set = set()

    def __init__(self) -> None:
        super().__init__(
            agent_id="LEARN-CALIB",
            agent_type="learning",
            version="1.0.0",
        )
        self._last_run: Optional[datetime] = None
        self._last_success: Optional[datetime] = None
        self._error_count_24h: int = 0

    async def process(self, context: AgentContext) -> CalibrationOutput:
        """Run offline confidence calibration analysis.

        Steps:
          1. EXTRACT belief outcomes from metadata
          2. GROUP by agent_id
          3. COMPUTE per-agent calibration (buckets, Brier, error)
          4. CLASSIFY agents as over/under/well-calibrated
          5. EMIT CalibrationOutput

        Args:
            context: AgentContext with belief outcomes in metadata.

        Returns:
            CalibrationOutput with per-agent calibration profiles.
        """
        self._last_run = datetime.now(timezone.utc)

        try:
            log = logger.bind(agent_id=self.agent_id, context_hash=context.context_window_hash)
            log.info("Starting offline calibration")

            belief_outcomes = context.metadata.get("belief_outcomes", [])
            eval_start_str = context.metadata.get("evaluation_start", "")
            eval_end_str = context.metadata.get("evaluation_end", "")

            if not isinstance(belief_outcomes, list):
                belief_outcomes = []

            try:
                eval_start = datetime.fromisoformat(eval_start_str) if eval_start_str else context.timestamp
            except (ValueError, TypeError):
                eval_start = context.timestamp
            try:
                eval_end = datetime.fromisoformat(eval_end_str) if eval_end_str else datetime.now(timezone.utc)
            except (ValueError, TypeError):
                eval_end = datetime.now(timezone.utc)

            if eval_start.tzinfo is None:
                eval_start = eval_start.replace(tzinfo=timezone.utc)
            if eval_end.tzinfo is None:
                eval_end = eval_end.replace(tzinfo=timezone.utc)

            # Group by agent
            beliefs_by_agent: dict[str, list[dict]] = {}
            for b in belief_outcomes:
                if not isinstance(b, dict):
                    continue
                aid = b.get("agent_id", "")
                if aid:
                    beliefs_by_agent.setdefault(aid, []).append(b)

            # Per-agent calibration
            agent_cals: list[AgentCalibration] = []
            overconfident = 0
            underconfident = 0
            well_calibrated = 0
            all_briers: list[float] = []

            for aid, beliefs in sorted(beliefs_by_agent.items()):
                cal_dict = calibrate_agent(aid, beliefs)
                buckets = [CalibrationBucket(**b) for b in cal_dict.pop("buckets")]
                cal_dict["buckets"] = buckets
                cal = AgentCalibration(**cal_dict)
                agent_cals.append(cal)

                if cal.is_overconfident:
                    overconfident += 1
                elif cal.overall_calibration_error < -OVERCONFIDENT_THRESHOLD:
                    underconfident += 1
                else:
                    well_calibrated += 1

                all_briers.append(cal.overall_brier_score)

            system_brier = round(sum(all_briers) / len(all_briers), 4) if all_briers else 0.0

            output = CalibrationOutput(
                agent_id=self.agent_id,
                timestamp=datetime.now(timezone.utc),
                context_window_hash=context.context_window_hash,
                evaluation_start=eval_start,
                evaluation_end=eval_end,
                agent_calibrations=agent_cals,
                system_brier_score=system_brier,
                agents_overconfident=overconfident,
                agents_underconfident=underconfident,
                agents_well_calibrated=well_calibrated,
            )

            self._last_success = datetime.now(timezone.utc)
            log.info(
                "Calibration complete",
                agents=len(agent_cals),
                overconfident=overconfident,
                underconfident=underconfident,
                well_calibrated=well_calibrated,
            )
            return output

        except AgentProcessingError:
            raise
        except Exception as e:
            self._error_count_24h += 1
            raise AgentProcessingError(
                message=f"LEARN-CALIB processing failed: {e}",
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
