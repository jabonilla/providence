"""LEARN-RETRAIN: Offline Retraining Recommendation Agent.

Analyzes attribution and calibration results to determine which agents
need retraining, prompt adjustments, or parameter tuning. Enforces
shadow mode before any retrained agent goes live.

Spec Reference: Technical Spec v2.3, Phase 4 (Learning)

Classification: FROZEN — zero LLM calls. Pure computation.

Critical Rules:
  - No live learning. All retraining offline.
  - Shadow mode before live deployment (always True).
  - Priority: CRITICAL if hit_rate < 0.40 AND Brier > 0.30
  - Priority: HIGH if performance degradation > 20%
  - Priority: MEDIUM if calibration error > 0.10
  - Priority: LOW if any degradation detected

Input: AgentContext with metadata:
  - metadata["attribution_results"]: list of AgentAttribution dicts
  - metadata["calibration_results"]: list of AgentCalibration dicts
  - metadata["baseline_metrics"]: dict of agent_id -> baseline performance

Output: RetrainOutput with per-agent retraining recommendations.
"""

from datetime import datetime, timezone
from typing import Any, Optional

import structlog

from providence.agents.base import AgentContext, AgentStatus, BaseAgent, HealthStatus
from providence.exceptions import AgentProcessingError
from providence.schemas.learning import RetrainOutput, RetrainRecommendation

logger = structlog.get_logger()

# Retrain thresholds
CRITICAL_HIT_RATE = 0.40
CRITICAL_BRIER = 0.30
HIGH_DEGRADATION_PCT = 20.0
MEDIUM_CAL_ERROR = 0.10
LOW_DEGRADATION_PCT = 5.0

# Adaptive agents that can be retrained (FROZEN agents excluded)
ADAPTIVE_AGENTS = {
    "COGNIT-FUNDAMENTAL",
    "COGNIT-NARRATIVE",
    "COGNIT-MACRO",
    "COGNIT-EVENT",
    "COGNIT-CROSSSEC",
    "COGNIT-EXIT",
    "REGIME-NARR",
    "DECIDE-SYNTH",
}


def compute_degradation(
    agent_id: str,
    current_metrics: dict[str, Any],
    baseline_metrics: dict[str, Any],
) -> float:
    """Compute performance degradation vs baseline.

    Args:
        agent_id: Agent identifier.
        current_metrics: Current attribution metrics.
        baseline_metrics: Baseline metrics for comparison.

    Returns:
        Degradation percentage (positive = worse than baseline).
    """
    baseline = baseline_metrics.get(agent_id, {})
    if not isinstance(baseline, dict):
        return 0.0

    baseline_hit = float(baseline.get("hit_rate", 0.5))
    current_hit = float(current_metrics.get("hit_rate", 0.5))

    if baseline_hit <= 0:
        return 0.0

    degradation = (baseline_hit - current_hit) / baseline_hit * 100.0
    return round(max(0.0, degradation), 2)


def determine_priority(
    hit_rate: float,
    brier_score: float,
    degradation_pct: float,
    cal_error: float,
) -> str:
    """Determine retrain priority level.

    Args:
        hit_rate: Agent's directional hit rate.
        brier_score: Agent's Brier score.
        degradation_pct: Performance degradation vs baseline.
        cal_error: Absolute calibration error.

    Returns:
        Priority: CRITICAL, HIGH, MEDIUM, or LOW.
    """
    if hit_rate < CRITICAL_HIT_RATE and brier_score > CRITICAL_BRIER:
        return "CRITICAL"
    if degradation_pct > HIGH_DEGRADATION_PCT:
        return "HIGH"
    if abs(cal_error) > MEDIUM_CAL_ERROR:
        return "MEDIUM"
    return "LOW"


def suggest_changes(
    hit_rate: float,
    brier_score: float,
    cal_error: float,
    is_overconfident: bool,
) -> list[str]:
    """Generate suggested prompt/parameter changes.

    Args:
        hit_rate: Agent hit rate.
        brier_score: Agent Brier score.
        cal_error: Calibration error.
        is_overconfident: Whether agent is overconfident.

    Returns:
        List of suggested changes.
    """
    suggestions: list[str] = []

    if hit_rate < 0.45:
        suggestions.append("Review prompt for directional bias — hit rate below 45%")

    if is_overconfident:
        suggestions.append(f"Add confidence dampening — agent is overconfident by {cal_error:.2f}")

    if brier_score > 0.25:
        suggestions.append("Increase invalidation condition specificity — Brier score > 0.25")

    if not is_overconfident and cal_error < -0.05:
        suggestions.append("Consider increasing confidence range — agent is underconfident")

    if not suggestions:
        suggestions.append("Minor parameter tuning recommended — overall performance acceptable")

    return suggestions


def evaluate_agent(
    agent_id: str,
    attribution: dict[str, Any],
    calibration: dict[str, Any],
    baseline_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate a single agent for retraining need.

    Args:
        agent_id: Agent identifier.
        attribution: Attribution results for this agent.
        calibration: Calibration results for this agent.
        baseline_metrics: Baseline metrics for comparison.

    Returns:
        Dict with retraining recommendation fields.
    """
    hit_rate = float(attribution.get("hit_rate", 0.5))
    brier_score = float(calibration.get("overall_brier_score", 0.0))
    cal_error = float(calibration.get("overall_calibration_error", 0.0))
    is_overconfident = calibration.get("is_overconfident", False)

    degradation = compute_degradation(agent_id, attribution, baseline_metrics)

    # Determine if retrain is needed
    needs_retrain = (
        hit_rate < CRITICAL_HIT_RATE
        or degradation > LOW_DEGRADATION_PCT
        or abs(cal_error) > MEDIUM_CAL_ERROR
        or brier_score > CRITICAL_BRIER
    )

    priority = determine_priority(hit_rate, brier_score, degradation, cal_error)

    # Only suggest changes for ADAPTIVE agents
    if agent_id in ADAPTIVE_AGENTS:
        changes = suggest_changes(hit_rate, brier_score, cal_error, is_overconfident)
    else:
        changes = ["FROZEN agent — parameter tuning only (no prompt changes)"]

    reason = ""
    if needs_retrain:
        parts = []
        if hit_rate < CRITICAL_HIT_RATE:
            parts.append(f"hit_rate={hit_rate:.2f} < {CRITICAL_HIT_RATE}")
        if degradation > LOW_DEGRADATION_PCT:
            parts.append(f"degradation={degradation:.1f}%")
        if abs(cal_error) > MEDIUM_CAL_ERROR:
            parts.append(f"cal_error={cal_error:.2f}")
        if brier_score > CRITICAL_BRIER:
            parts.append(f"brier={brier_score:.2f}")
        reason = "Retrain needed: " + ", ".join(parts)
    else:
        reason = "Performance within acceptable bounds"

    return {
        "agent_id": agent_id,
        "needs_retrain": needs_retrain,
        "reason": reason,
        "priority": priority,
        "performance_degradation_pct": degradation,
        "suggested_changes": changes,
        "shadow_mode_required": True,  # Always True per spec
    }


class LearnRetrain(BaseAgent[RetrainOutput]):
    """Offline retraining recommendation agent.

    Analyzes attribution and calibration to determine which agents
    need retraining. Enforces shadow mode before deployment.

    FROZEN: Zero LLM calls. Offline only.
    """

    CONSUMED_DATA_TYPES: set = set()

    def __init__(self) -> None:
        super().__init__(
            agent_id="LEARN-RETRAIN",
            agent_type="learning",
            version="1.0.0",
        )
        self._last_run: Optional[datetime] = None
        self._last_success: Optional[datetime] = None
        self._error_count_24h: int = 0

    async def process(self, context: AgentContext) -> RetrainOutput:
        """Generate retraining recommendations.

        Steps:
          1. EXTRACT attribution, calibration, baseline from metadata
          2. MATCH attribution + calibration per agent
          3. EVALUATE each agent for retraining need
          4. EMIT RetrainOutput

        Args:
            context: AgentContext with attribution/calibration results.

        Returns:
            RetrainOutput with per-agent retraining recommendations.
        """
        self._last_run = datetime.now(timezone.utc)

        try:
            log = logger.bind(agent_id=self.agent_id, context_hash=context.context_window_hash)
            log.info("Starting retraining evaluation")

            attribution_results = context.metadata.get("attribution_results", [])
            calibration_results = context.metadata.get("calibration_results", [])
            baseline_metrics = context.metadata.get("baseline_metrics", {})

            if not isinstance(attribution_results, list):
                attribution_results = []
            if not isinstance(calibration_results, list):
                calibration_results = []
            if not isinstance(baseline_metrics, dict):
                baseline_metrics = {}

            # Index by agent_id
            attr_by_agent = {}
            for a in attribution_results:
                if isinstance(a, dict) and "agent_id" in a:
                    attr_by_agent[a["agent_id"]] = a

            cal_by_agent = {}
            for c in calibration_results:
                if isinstance(c, dict) and "agent_id" in c:
                    cal_by_agent[c["agent_id"]] = c

            # Union of all agents
            all_agents = set(attr_by_agent.keys()) | set(cal_by_agent.keys())

            recommendations: list[RetrainRecommendation] = []
            needs_retrain_count = 0

            for aid in sorted(all_agents):
                attr = attr_by_agent.get(aid, {})
                cal = cal_by_agent.get(aid, {})

                rec_dict = evaluate_agent(aid, attr, cal, baseline_metrics)
                recommendations.append(RetrainRecommendation(**rec_dict))

                if rec_dict["needs_retrain"]:
                    needs_retrain_count += 1

            output = RetrainOutput(
                agent_id=self.agent_id,
                timestamp=datetime.now(timezone.utc),
                context_window_hash=context.context_window_hash,
                recommendations=recommendations,
                total_agents_evaluated=len(recommendations),
                agents_needing_retrain=needs_retrain_count,
            )

            self._last_success = datetime.now(timezone.utc)
            log.info(
                "Retraining evaluation complete",
                evaluated=len(recommendations),
                needs_retrain=needs_retrain_count,
            )
            return output

        except AgentProcessingError:
            raise
        except Exception as e:
            self._error_count_24h += 1
            raise AgentProcessingError(
                message=f"LEARN-RETRAIN processing failed: {e}",
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
