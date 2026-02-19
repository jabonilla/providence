"""GOVERN-MATURITY: Agent Maturity Gate Agent.

Evaluates agent readiness for deployment stage transitions:
SHADOW → LIMITED → FULL. Uses attribution, calibration, and backtest
results to determine promotion eligibility.

Spec Reference: Technical Spec v2.3, Phase 4 (Governance)

Classification: FROZEN — zero LLM calls. Pure computation.

Promotion criteria:
  - SHADOW → LIMITED: min 30 days in shadow, hit_rate ≥ 0.50, Brier ≤ 0.30
  - LIMITED → FULL: min 60 days in limited, hit_rate ≥ 0.55, Brier ≤ 0.25,
    no CRITICAL retrain recommendations

Confidence weighting:
  - SHADOW: 0.0 (outputs not used in live trading)
  - LIMITED: 0.5 (outputs used with 50% weighting)
  - FULL: 1.0 (outputs used at full weight)

Input: AgentContext with metadata:
  - metadata["agent_maturity_state"]: list of dicts with agent_id, current_stage, days_in_stage
  - metadata["attribution_results"]: list of agent attribution dicts
  - metadata["calibration_results"]: list of agent calibration dicts
  - metadata["retrain_recommendations"]: list of retrain recommendation dicts

Output: MaturityGateOutput with per-agent maturity records.
"""

from datetime import datetime, timezone
from typing import Any, Optional

import structlog

from providence.agents.base import AgentContext, AgentStatus, BaseAgent, HealthStatus
from providence.exceptions import AgentProcessingError
from providence.schemas.enums import MaturityStage
from providence.schemas.governance import AgentMaturityRecord, MaturityGateOutput

logger = structlog.get_logger()

# Promotion thresholds
SHADOW_MIN_DAYS = 30
LIMITED_MIN_DAYS = 60
SHADOW_MIN_HIT_RATE = 0.50
LIMITED_MIN_HIT_RATE = 0.55
SHADOW_MAX_BRIER = 0.30
LIMITED_MAX_BRIER = 0.25

# Confidence weights per stage
STAGE_WEIGHTS: dict[MaturityStage, float] = {
    MaturityStage.SHADOW: 0.0,
    MaturityStage.LIMITED: 0.5,
    MaturityStage.FULL: 1.0,
}


def evaluate_promotion(
    agent_id: str,
    current_stage: MaturityStage,
    days_in_stage: int,
    attribution: dict[str, Any],
    calibration: dict[str, Any],
    retrain_rec: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate whether an agent is eligible for promotion.

    Args:
        agent_id: Agent identifier.
        current_stage: Current maturity stage.
        days_in_stage: Days in current stage.
        attribution: Attribution results for this agent.
        calibration: Calibration results for this agent.
        retrain_rec: Retrain recommendation for this agent.

    Returns:
        Dict with promotion eligibility and blockers.
    """
    blockers: list[str] = []
    hit_rate = float(attribution.get("hit_rate", 0.0))
    brier = float(calibration.get("overall_brier_score", 1.0))
    retrain_priority = retrain_rec.get("priority", "LOW")

    if current_stage == MaturityStage.SHADOW:
        if days_in_stage < SHADOW_MIN_DAYS:
            blockers.append(f"Needs {SHADOW_MIN_DAYS - days_in_stage} more days in SHADOW")
        if hit_rate < SHADOW_MIN_HIT_RATE:
            blockers.append(f"Hit rate {hit_rate:.2f} below {SHADOW_MIN_HIT_RATE} threshold")
        if brier > SHADOW_MAX_BRIER:
            blockers.append(f"Brier score {brier:.2f} above {SHADOW_MAX_BRIER} threshold")

    elif current_stage == MaturityStage.LIMITED:
        if days_in_stage < LIMITED_MIN_DAYS:
            blockers.append(f"Needs {LIMITED_MIN_DAYS - days_in_stage} more days in LIMITED")
        if hit_rate < LIMITED_MIN_HIT_RATE:
            blockers.append(f"Hit rate {hit_rate:.2f} below {LIMITED_MIN_HIT_RATE} threshold")
        if brier > LIMITED_MAX_BRIER:
            blockers.append(f"Brier score {brier:.2f} above {LIMITED_MAX_BRIER} threshold")
        if retrain_priority == "CRITICAL":
            blockers.append("CRITICAL retrain recommendation blocks FULL promotion")

    # FULL stage agents cannot be promoted further
    elif current_stage == MaturityStage.FULL:
        pass  # No promotion possible

    eligible = len(blockers) == 0 and current_stage != MaturityStage.FULL

    return {
        "agent_id": agent_id,
        "current_stage": current_stage,
        "previous_stage": None,
        "stage_changed": False,
        "days_in_current_stage": days_in_stage,
        "promotion_eligible": eligible,
        "promotion_blockers": blockers,
        "confidence_weight": STAGE_WEIGHTS[current_stage],
    }


class GovernMaturity(BaseAgent[MaturityGateOutput]):
    """Agent maturity gate evaluation agent.

    Evaluates each agent's readiness for deployment stage transitions.
    Promotion requires meeting hit rate, Brier score, and time-in-stage
    thresholds. Human approval is required for actual transitions.

    FROZEN: Zero LLM calls. Pure computation.
    """

    CONSUMED_DATA_TYPES: set = set()

    def __init__(self) -> None:
        super().__init__(
            agent_id="GOVERN-MATURITY",
            agent_type="governance",
            version="1.0.0",
        )
        self._last_run: Optional[datetime] = None
        self._last_success: Optional[datetime] = None
        self._error_count_24h: int = 0

    async def process(self, context: AgentContext) -> MaturityGateOutput:
        """Evaluate all agents for maturity stage transitions.

        Steps:
          1. EXTRACT maturity state, attribution, calibration from metadata
          2. EVALUATE each agent for promotion eligibility
          3. COUNT agents per stage
          4. EMIT MaturityGateOutput

        Args:
            context: AgentContext with agent maturity state.

        Returns:
            MaturityGateOutput with per-agent maturity records.
        """
        self._last_run = datetime.now(timezone.utc)

        try:
            log = logger.bind(agent_id=self.agent_id, context_hash=context.context_window_hash)
            log.info("Starting maturity gate evaluation")

            maturity_state = context.metadata.get("agent_maturity_state", [])
            attribution_results = context.metadata.get("attribution_results", [])
            calibration_results = context.metadata.get("calibration_results", [])
            retrain_recs = context.metadata.get("retrain_recommendations", [])

            if not isinstance(maturity_state, list):
                maturity_state = []
            if not isinstance(attribution_results, list):
                attribution_results = []
            if not isinstance(calibration_results, list):
                calibration_results = []
            if not isinstance(retrain_recs, list):
                retrain_recs = []

            # Index by agent_id
            attr_by_agent = {a["agent_id"]: a for a in attribution_results if isinstance(a, dict) and "agent_id" in a}
            cal_by_agent = {c["agent_id"]: c for c in calibration_results if isinstance(c, dict) and "agent_id" in c}
            retrain_by_agent = {r["agent_id"]: r for r in retrain_recs if isinstance(r, dict) and "agent_id" in r}

            records: list[AgentMaturityRecord] = []
            promotions = 0
            shadow_count = 0
            limited_count = 0
            full_count = 0

            for state in maturity_state:
                if not isinstance(state, dict) or "agent_id" not in state:
                    continue

                aid = state["agent_id"]
                stage_str = state.get("current_stage", "SHADOW")
                try:
                    stage = MaturityStage(stage_str)
                except ValueError:
                    stage = MaturityStage.SHADOW

                days = int(state.get("days_in_stage", 0))
                attr = attr_by_agent.get(aid, {})
                cal = cal_by_agent.get(aid, {})
                retrain = retrain_by_agent.get(aid, {})

                rec_dict = evaluate_promotion(aid, stage, days, attr, cal, retrain)
                records.append(AgentMaturityRecord(**rec_dict))

                if rec_dict["promotion_eligible"]:
                    promotions += 1

                if stage == MaturityStage.SHADOW:
                    shadow_count += 1
                elif stage == MaturityStage.LIMITED:
                    limited_count += 1
                elif stage == MaturityStage.FULL:
                    full_count += 1

            output = MaturityGateOutput(
                agent_id=self.agent_id,
                timestamp=datetime.now(timezone.utc),
                context_window_hash=context.context_window_hash,
                agent_records=records,
                agents_in_shadow=shadow_count,
                agents_in_limited=limited_count,
                agents_in_full=full_count,
                promotions_recommended=promotions,
            )

            self._last_success = datetime.now(timezone.utc)
            log.info(
                "Maturity evaluation complete",
                agents=len(records),
                promotions=promotions,
                shadow=shadow_count,
                limited=limited_count,
                full=full_count,
            )
            return output

        except AgentProcessingError:
            raise
        except Exception as e:
            self._error_count_24h += 1
            raise AgentProcessingError(
                message=f"GOVERN-MATURITY processing failed: {e}",
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
