"""GOVERN-OVERSIGHT: System Oversight Dashboard Agent.

Aggregates health from all agents, flags incidents, and produces
dashboard data for human operators. Detects agent failures, shadow
exit divergences, retraining backlogs, and exposure breaches.

Spec Reference: Technical Spec v2.3, Phase 4 (Governance)

Classification: FROZEN — zero LLM calls. Pure computation.

Input: AgentContext with metadata:
  - metadata["agent_health_reports"]: list of HealthStatus-like dicts
  - metadata["current_risk_mode"]: str (NORMAL/CAUTIOUS/DEFENSIVE/HALTED)
  - metadata["current_tier"]: str (SEED/GROWTH/SCALE/INSTITUTIONAL)
  - metadata["active_positions_count"]: int
  - metadata["gross_exposure_pct"]: float
  - metadata["net_exposure_pct"]: float
  - metadata["retraining_queue_size"]: int
  - metadata["shadow_exit_divergences"]: int

Output: OversightOutput with health summary, incidents, and dashboard data.
"""

from datetime import datetime, timezone
from typing import Any, Optional

import structlog

from providence.agents.base import AgentContext, AgentStatus, BaseAgent, HealthStatus
from providence.exceptions import AgentProcessingError
from providence.schemas.enums import CapitalTier, IncidentSeverity, SystemRiskMode
from providence.schemas.governance import (
    GovernanceIncident,
    OversightOutput,
    SystemHealthSummary,
)

logger = structlog.get_logger()

# Thresholds for incident generation
UNHEALTHY_THRESHOLD = 1  # ≥ 1 unhealthy agent → CRITICAL incident
DEGRADED_THRESHOLD = 3   # ≥ 3 degraded agents → WARNING incident
ERROR_THRESHOLD_24H = 50  # Total errors > 50 → WARNING
RETRAIN_QUEUE_WARNING = 3  # ≥ 3 agents needing retrain → WARNING
SHADOW_DIVERGENCE_WARNING = 5  # ≥ 5 divergences → WARNING
GROSS_EXPOSURE_WARNING = 1.50  # > 150% gross exposure → WARNING


def aggregate_health(reports: list[dict[str, Any]]) -> SystemHealthSummary:
    """Aggregate health reports from all agents.

    Args:
        reports: List of HealthStatus-like dicts.

    Returns:
        SystemHealthSummary with counts per status.
    """
    healthy = 0
    degraded = 0
    unhealthy = 0
    offline = 0
    total_errors = 0

    for report in reports:
        if not isinstance(report, dict):
            continue

        status = report.get("status", "HEALTHY")
        if status == "HEALTHY":
            healthy += 1
        elif status == "DEGRADED":
            degraded += 1
        elif status == "UNHEALTHY":
            unhealthy += 1
        elif status == "OFFLINE":
            offline += 1

        total_errors += int(report.get("error_count_24h", 0))

    return SystemHealthSummary(
        total_agents=healthy + degraded + unhealthy + offline,
        healthy_count=healthy,
        degraded_count=degraded,
        unhealthy_count=unhealthy,
        offline_count=offline,
        total_errors_24h=total_errors,
    )


def detect_incidents(
    health: SystemHealthSummary,
    risk_mode: str,
    retraining_queue: int,
    shadow_divergences: int,
    gross_exposure: float,
) -> list[GovernanceIncident]:
    """Detect governance incidents from system state.

    Args:
        health: Aggregated health summary.
        risk_mode: Current system risk mode.
        retraining_queue: Number of agents needing retrain.
        shadow_divergences: COGNIT-EXIT vs EXEC-CAPTURE divergences.
        gross_exposure: Current gross exposure percentage.

    Returns:
        List of GovernanceIncident objects.
    """
    incidents: list[GovernanceIncident] = []

    # Unhealthy agents
    if health.unhealthy_count >= UNHEALTHY_THRESHOLD:
        incidents.append(GovernanceIncident(
            severity=IncidentSeverity.CRITICAL,
            source_agent_id="GOVERN-OVERSIGHT",
            title="Unhealthy agents detected",
            description=f"{health.unhealthy_count} agent(s) are UNHEALTHY",
            requires_human_action=True,
        ))

    # Offline agents
    if health.offline_count > 0:
        incidents.append(GovernanceIncident(
            severity=IncidentSeverity.CRITICAL,
            source_agent_id="GOVERN-OVERSIGHT",
            title="Offline agents detected",
            description=f"{health.offline_count} agent(s) are OFFLINE",
            requires_human_action=True,
        ))

    # Many degraded agents
    if health.degraded_count >= DEGRADED_THRESHOLD:
        incidents.append(GovernanceIncident(
            severity=IncidentSeverity.WARNING,
            source_agent_id="GOVERN-OVERSIGHT",
            title="Multiple degraded agents",
            description=f"{health.degraded_count} agents are DEGRADED",
            requires_human_action=False,
        ))

    # High error count
    if health.total_errors_24h > ERROR_THRESHOLD_24H:
        incidents.append(GovernanceIncident(
            severity=IncidentSeverity.WARNING,
            source_agent_id="GOVERN-OVERSIGHT",
            title="High error count",
            description=f"{health.total_errors_24h} total errors in last 24h",
            requires_human_action=False,
        ))

    # HALTED risk mode
    if risk_mode == "HALTED":
        incidents.append(GovernanceIncident(
            severity=IncidentSeverity.CRITICAL,
            source_agent_id="GOVERN-OVERSIGHT",
            title="System HALTED",
            description="Risk mode is HALTED — all execution suspended",
            requires_human_action=True,
        ))

    # Retrain backlog
    if retraining_queue >= RETRAIN_QUEUE_WARNING:
        incidents.append(GovernanceIncident(
            severity=IncidentSeverity.WARNING,
            source_agent_id="GOVERN-OVERSIGHT",
            title="Retraining backlog",
            description=f"{retraining_queue} agents need retraining",
            requires_human_action=False,
        ))

    # Shadow exit divergences
    if shadow_divergences >= SHADOW_DIVERGENCE_WARNING:
        incidents.append(GovernanceIncident(
            severity=IncidentSeverity.WARNING,
            source_agent_id="GOVERN-OVERSIGHT",
            title="Shadow exit divergences",
            description=f"{shadow_divergences} COGNIT-EXIT vs EXEC-CAPTURE divergences",
            requires_human_action=False,
        ))

    # Gross exposure breach
    if gross_exposure > GROSS_EXPOSURE_WARNING:
        incidents.append(GovernanceIncident(
            severity=IncidentSeverity.CRITICAL,
            source_agent_id="GOVERN-OVERSIGHT",
            title="Gross exposure breach",
            description=f"Gross exposure {gross_exposure:.1%} exceeds {GROSS_EXPOSURE_WARNING:.0%} threshold",
            requires_human_action=True,
        ))

    return incidents


class GovernOversight(BaseAgent[OversightOutput]):
    """System oversight dashboard agent.

    Aggregates health, detects incidents, and produces dashboard data
    for human operators.

    FROZEN: Zero LLM calls. Pure computation.
    """

    CONSUMED_DATA_TYPES: set = set()

    def __init__(self) -> None:
        super().__init__(
            agent_id="GOVERN-OVERSIGHT",
            agent_type="governance",
            version="1.0.0",
        )
        self._last_run: Optional[datetime] = None
        self._last_success: Optional[datetime] = None
        self._error_count_24h: int = 0

    async def process(self, context: AgentContext) -> OversightOutput:
        """Aggregate system health and detect governance incidents.

        Steps:
          1. EXTRACT health reports and system state from metadata
          2. AGGREGATE health across all agents
          3. DETECT incidents from thresholds
          4. EMIT OversightOutput

        Args:
            context: AgentContext with health reports in metadata.

        Returns:
            OversightOutput with health summary and incidents.
        """
        self._last_run = datetime.now(timezone.utc)

        try:
            log = logger.bind(agent_id=self.agent_id, context_hash=context.context_window_hash)
            log.info("Starting oversight aggregation")

            health_reports = context.metadata.get("agent_health_reports", [])
            risk_mode_str = context.metadata.get("current_risk_mode", "NORMAL")
            tier_str = context.metadata.get("current_tier", "SEED")
            positions = int(context.metadata.get("active_positions_count", 0))
            gross_exp = float(context.metadata.get("gross_exposure_pct", 0.0))
            net_exp = float(context.metadata.get("net_exposure_pct", 0.0))
            retrain_queue = int(context.metadata.get("retraining_queue_size", 0))
            shadow_div = int(context.metadata.get("shadow_exit_divergences", 0))

            if not isinstance(health_reports, list):
                health_reports = []

            # Parse enums
            try:
                risk_mode = SystemRiskMode(risk_mode_str)
            except ValueError:
                risk_mode = SystemRiskMode.NORMAL

            try:
                tier = CapitalTier(tier_str)
            except ValueError:
                tier = CapitalTier.SEED

            # Aggregate and detect
            health = aggregate_health(health_reports)
            incidents = detect_incidents(
                health, risk_mode_str, retrain_queue, shadow_div, gross_exp,
            )

            output = OversightOutput(
                agent_id=self.agent_id,
                timestamp=datetime.now(timezone.utc),
                context_window_hash=context.context_window_hash,
                health_summary=health,
                current_risk_mode=risk_mode,
                current_tier=tier,
                incidents=incidents,
                active_positions_count=positions,
                gross_exposure_pct=gross_exp,
                net_exposure_pct=net_exp,
                retraining_queue_size=retrain_queue,
                shadow_exit_divergences=shadow_div,
            )

            self._last_success = datetime.now(timezone.utc)
            log.info(
                "Oversight complete",
                total_agents=health.total_agents,
                incidents=len(incidents),
                risk_mode=risk_mode.value,
            )
            return output

        except AgentProcessingError:
            raise
        except Exception as e:
            self._error_count_24h += 1
            raise AgentProcessingError(
                message=f"GOVERN-OVERSIGHT processing failed: {e}",
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
