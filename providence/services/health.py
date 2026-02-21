"""System health aggregation service.

Collects health from all agents, computes system-level metrics,
and produces a structured health report for monitoring and governance.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import structlog

from providence.agents.base import AgentStatus, BaseAgent, HealthStatus
from providence.storage.run_store import RunStore

logger = structlog.get_logger()


@dataclass(frozen=True)
class SystemHealth:
    """Aggregated system health report."""

    timestamp: datetime
    total_agents: int
    healthy_count: int
    degraded_count: int
    unhealthy_count: int
    offline_count: int
    agent_health: dict[str, HealthStatus]
    run_success_rate: float
    run_count: int
    system_status: str  # HEALTHY, DEGRADED, CRITICAL, HALTED

    @property
    def healthy_pct(self) -> float:
        if self.total_agents == 0:
            return 0.0
        return self.healthy_count / self.total_agents

    def summary(self) -> dict[str, Any]:
        """Produce a summary dict for logging or API response."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "system_status": self.system_status,
            "agents": {
                "total": self.total_agents,
                "healthy": self.healthy_count,
                "degraded": self.degraded_count,
                "unhealthy": self.unhealthy_count,
                "offline": self.offline_count,
            },
            "pipeline": {
                "run_count": self.run_count,
                "success_rate": round(self.run_success_rate, 4),
            },
        }


class HealthService:
    """Aggregates health from all agents and pipeline runs.

    Used by the CLI 'health' command and GOVERN-OVERSIGHT agent.
    """

    def __init__(
        self,
        agent_registry: dict[str, BaseAgent],
        run_store: RunStore | None = None,
    ) -> None:
        self._agents = agent_registry
        self._run_store = run_store

    def check(self) -> SystemHealth:
        """Collect health from all agents and compute system status."""
        agent_health: dict[str, HealthStatus] = {}
        counts = {
            AgentStatus.HEALTHY: 0,
            AgentStatus.DEGRADED: 0,
            AgentStatus.UNHEALTHY: 0,
            AgentStatus.OFFLINE: 0,
        }

        for agent_id, agent in sorted(self._agents.items()):
            try:
                health = agent.get_health()
                agent_health[agent_id] = health
                status = AgentStatus(health.status) if isinstance(health.status, str) else health.status
                counts[status] = counts.get(status, 0) + 1
            except Exception as exc:
                logger.warning("Agent health check failed", agent_id=agent_id, error=str(exc))
                fallback = HealthStatus(
                    agent_id=agent_id,
                    status=AgentStatus.OFFLINE,
                    message=f"Health check failed: {exc}",
                )
                agent_health[agent_id] = fallback
                counts[AgentStatus.OFFLINE] += 1

        # Pipeline metrics from RunStore
        run_success_rate = 0.0
        run_count = 0
        if self._run_store is not None:
            run_count = self._run_store.count()
            run_success_rate = self._run_store.success_rate()

        # Determine system status
        total = len(self._agents)
        system_status = self._compute_system_status(
            total=total,
            healthy=counts[AgentStatus.HEALTHY],
            unhealthy=counts[AgentStatus.UNHEALTHY],
            offline=counts[AgentStatus.OFFLINE],
            run_success_rate=run_success_rate,
        )

        report = SystemHealth(
            timestamp=datetime.now(timezone.utc),
            total_agents=total,
            healthy_count=counts[AgentStatus.HEALTHY],
            degraded_count=counts[AgentStatus.DEGRADED],
            unhealthy_count=counts[AgentStatus.UNHEALTHY],
            offline_count=counts[AgentStatus.OFFLINE],
            agent_health=agent_health,
            run_success_rate=run_success_rate,
            run_count=run_count,
            system_status=system_status,
        )

        logger.info("System health check", **report.summary())
        return report

    @staticmethod
    def _compute_system_status(
        total: int,
        healthy: int,
        unhealthy: int,
        offline: int,
        run_success_rate: float,
    ) -> str:
        """Classify overall system health.

        - HALTED: >50% agents offline or all pipeline runs failing
        - CRITICAL: >25% unhealthy/offline or success rate < 50%
        - DEGRADED: any agent unhealthy/offline or success rate < 90%
        - HEALTHY: all agents healthy and success rate >= 90%
        """
        if total == 0:
            return "HALTED"

        offline_pct = offline / total
        problem_pct = (unhealthy + offline) / total

        if offline_pct > 0.5 or (run_success_rate < 0.01 and total > 0):
            return "HALTED"
        if problem_pct > 0.25 or run_success_rate < 0.5:
            return "CRITICAL"
        if unhealthy > 0 or offline > 0 or run_success_rate < 0.9:
            return "DEGRADED"
        return "HEALTHY"
