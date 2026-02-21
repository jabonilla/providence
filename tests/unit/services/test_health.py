"""Tests for HealthService — system health aggregation."""

from datetime import datetime, timezone

import pytest

from providence.agents.base import AgentContext, AgentStatus, BaseAgent, HealthStatus
from providence.orchestration.models import PipelineRun, RunStatus, StageResult, StageStatus
from providence.services.health import HealthService, SystemHealth
from providence.storage.run_store import RunStore


NOW = datetime(2026, 2, 20, 12, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Mock Agents
# ---------------------------------------------------------------------------


class HealthyAgent(BaseAgent[dict]):
    def __init__(self, agent_id: str) -> None:
        super().__init__(agent_id=agent_id, agent_type="test", version="1.0.0")

    async def process(self, context: AgentContext) -> dict:
        return {}

    def get_health(self) -> HealthStatus:
        return HealthStatus(agent_id=self.agent_id, status=AgentStatus.HEALTHY, last_run=NOW, error_count_24h=0)


class UnhealthyAgent(BaseAgent[dict]):
    def __init__(self, agent_id: str) -> None:
        super().__init__(agent_id=agent_id, agent_type="test", version="1.0.0")

    async def process(self, context: AgentContext) -> dict:
        return {}

    def get_health(self) -> HealthStatus:
        return HealthStatus(agent_id=self.agent_id, status=AgentStatus.UNHEALTHY, error_count_24h=10)


class OfflineAgent(BaseAgent[dict]):
    def __init__(self, agent_id: str) -> None:
        super().__init__(agent_id=agent_id, agent_type="test", version="1.0.0")

    async def process(self, context: AgentContext) -> dict:
        return {}

    def get_health(self) -> HealthStatus:
        return HealthStatus(agent_id=self.agent_id, status=AgentStatus.OFFLINE, message="No connection")


class BrokenAgent(BaseAgent[dict]):
    """Agent whose get_health() raises an exception."""

    def __init__(self, agent_id: str) -> None:
        super().__init__(agent_id=agent_id, agent_type="test", version="1.0.0")

    async def process(self, context: AgentContext) -> dict:
        return {}

    def get_health(self) -> HealthStatus:
        raise RuntimeError("Health check crashed")


def _make_run(status: RunStatus = RunStatus.SUCCEEDED) -> PipelineRun:
    from datetime import timedelta
    from uuid import uuid4

    return PipelineRun(
        run_id=uuid4(),
        loop_type="MAIN",
        status=status,
        started_at=NOW,
        finished_at=NOW + timedelta(seconds=5),
        stage_results=[
            StageResult(
                stage_name="TEST",
                agent_id="TEST",
                status=StageStatus.SUCCEEDED,
                started_at=NOW,
                finished_at=NOW + timedelta(seconds=3),
                duration_ms=3000.0,
                output={},
                error=None,
            )
        ],
        metadata={},
    )


# ===========================================================================
# All Healthy
# ===========================================================================


class TestAllHealthy:
    def test_all_healthy_system_status(self):
        registry = {f"AGENT-{i}": HealthyAgent(f"AGENT-{i}") for i in range(5)}
        svc = HealthService(registry)
        report = svc.check()

        assert report.system_status == "HEALTHY"
        assert report.healthy_count == 5
        assert report.total_agents == 5
        assert report.unhealthy_count == 0
        assert report.offline_count == 0

    def test_healthy_pct(self):
        registry = {f"AGENT-{i}": HealthyAgent(f"AGENT-{i}") for i in range(4)}
        svc = HealthService(registry)
        report = svc.check()
        assert report.healthy_pct == 1.0


# ===========================================================================
# Degraded
# ===========================================================================


class TestDegraded:
    def test_one_unhealthy_degrades_system(self):
        registry = {
            "HEALTHY-1": HealthyAgent("HEALTHY-1"),
            "HEALTHY-2": HealthyAgent("HEALTHY-2"),
            "SICK-1": UnhealthyAgent("SICK-1"),
        }
        svc = HealthService(registry)
        report = svc.check()

        assert report.system_status == "DEGRADED"
        assert report.unhealthy_count == 1

    def test_one_offline_degrades_system(self):
        registry = {
            "HEALTHY-1": HealthyAgent("HEALTHY-1"),
            "HEALTHY-2": HealthyAgent("HEALTHY-2"),
            "HEALTHY-3": HealthyAgent("HEALTHY-3"),
            "OFFLINE-1": OfflineAgent("OFFLINE-1"),
        }
        svc = HealthService(registry)
        report = svc.check()

        assert report.system_status == "DEGRADED"
        assert report.offline_count == 1


# ===========================================================================
# Critical
# ===========================================================================


class TestCritical:
    def test_many_unhealthy_is_critical(self):
        """If >25% of agents are unhealthy, status is CRITICAL."""
        registry = {
            "HEALTHY-1": HealthyAgent("HEALTHY-1"),
            "HEALTHY-2": HealthyAgent("HEALTHY-2"),
            "SICK-1": UnhealthyAgent("SICK-1"),
            "SICK-2": UnhealthyAgent("SICK-2"),
        }
        svc = HealthService(registry)
        report = svc.check()

        # 2/4 = 50% unhealthy → CRITICAL
        assert report.system_status == "CRITICAL"


# ===========================================================================
# Halted
# ===========================================================================


class TestHalted:
    def test_majority_offline_is_halted(self):
        """If >50% of agents are offline, status is HALTED."""
        registry = {
            "HEALTHY-1": HealthyAgent("HEALTHY-1"),
            "OFFLINE-1": OfflineAgent("OFFLINE-1"),
            "OFFLINE-2": OfflineAgent("OFFLINE-2"),
            "OFFLINE-3": OfflineAgent("OFFLINE-3"),
        }
        svc = HealthService(registry)
        report = svc.check()

        assert report.system_status == "HALTED"

    def test_empty_registry_is_halted(self):
        svc = HealthService({})
        report = svc.check()
        assert report.system_status == "HALTED"
        assert report.total_agents == 0


# ===========================================================================
# Broken Agent Handling
# ===========================================================================


class TestBrokenAgent:
    def test_broken_health_check_counts_as_offline(self):
        registry = {
            "HEALTHY-1": HealthyAgent("HEALTHY-1"),
            "BROKEN-1": BrokenAgent("BROKEN-1"),
        }
        svc = HealthService(registry)
        report = svc.check()

        assert report.offline_count == 1
        assert "BROKEN-1" in report.agent_health
        assert report.agent_health["BROKEN-1"].status == AgentStatus.OFFLINE


# ===========================================================================
# RunStore Integration
# ===========================================================================


class TestRunStoreIntegration:
    def test_success_rate_from_run_store(self):
        run_store = RunStore()
        run_store.append(_make_run(RunStatus.SUCCEEDED))
        run_store.append(_make_run(RunStatus.SUCCEEDED))
        run_store.append(_make_run(RunStatus.FAILED))

        registry = {"AGENT-1": HealthyAgent("AGENT-1")}
        svc = HealthService(registry, run_store=run_store)
        report = svc.check()

        assert report.run_count == 3
        assert abs(report.run_success_rate - 2 / 3) < 0.01

    def test_low_success_rate_degrades_status(self):
        """<90% success rate should degrade even if all agents healthy."""
        run_store = RunStore()
        for _ in range(5):
            run_store.append(_make_run(RunStatus.SUCCEEDED))
        for _ in range(2):
            run_store.append(_make_run(RunStatus.FAILED))

        registry = {"AGENT-1": HealthyAgent("AGENT-1")}
        svc = HealthService(registry, run_store=run_store)
        report = svc.check()

        # 5/7 ≈ 71% success rate < 90% → DEGRADED
        assert report.system_status == "DEGRADED"

    def test_no_run_store(self):
        registry = {"AGENT-1": HealthyAgent("AGENT-1")}
        svc = HealthService(registry, run_store=None)
        report = svc.check()

        assert report.run_count == 0
        assert report.run_success_rate == 0.0


# ===========================================================================
# Summary Output
# ===========================================================================


class TestSummary:
    def test_summary_structure(self):
        registry = {"AGENT-1": HealthyAgent("AGENT-1")}
        svc = HealthService(registry)
        report = svc.check()
        summary = report.summary()

        assert "timestamp" in summary
        assert "system_status" in summary
        assert "agents" in summary
        assert "pipeline" in summary
        assert summary["agents"]["total"] == 1
        assert summary["agents"]["healthy"] == 1
