"""Tests for Orchestrator — DAG coordinator.

Tests cover:
  - Main loop: parallel cognition, sequential execution, metadata flow
  - Exit/Learning/Governance loops: sequential execution
  - Partial failure handling
  - Skipped stages on upstream failure
  - PipelineRun tracking and content hash
"""

import asyncio
from datetime import datetime, timezone
from typing import Any, Optional
from unittest.mock import MagicMock

import pytest

from providence.agents.base import AgentContext, AgentStatus, BaseAgent, HealthStatus
from providence.config.agent_config import AgentConfigRegistry
from providence.exceptions import AgentProcessingError
from providence.orchestration.models import PipelineRun, RunStatus, StageStatus
from providence.orchestration.orchestrator import (
    COGNITION_AGENTS,
    EXECUTION_AGENTS,
    EXIT_AGENTS,
    GOVERNANCE_AGENTS,
    LEARNING_AGENTS,
    REGIME_PARALLEL_AGENTS,
    Orchestrator,
)
from providence.services.context_svc import ContextService

NOW = datetime.now(timezone.utc)


class SimpleAgent(BaseAgent):
    """Simple mock agent that returns a dict output."""

    CONSUMED_DATA_TYPES: set = set()

    def __init__(
        self,
        agent_id: str,
        output: dict | None = None,
        error: Exception | None = None,
    ):
        super().__init__(agent_id=agent_id, agent_type="test", version="1.0.0")
        self._output = output or {"agent_id": agent_id, "status": "ok"}
        self._error = error
        self.call_count = 0

    async def process(self, context: AgentContext) -> dict:
        self.call_count += 1
        if self._error:
            raise self._error
        return self._output

    def get_health(self) -> HealthStatus:
        return HealthStatus(agent_id=self.agent_id, status=AgentStatus.HEALTHY)


def _build_agent_registry(
    agent_ids: list[str],
    failing: dict[str, Exception] | None = None,
) -> dict[str, BaseAgent]:
    """Build a mock agent registry with optional failures."""
    failing = failing or {}
    registry = {}
    for aid in agent_ids:
        if aid in failing:
            registry[aid] = SimpleAgent(aid, error=failing[aid])
        else:
            registry[aid] = SimpleAgent(aid)
    return registry


def _mock_context_service() -> ContextService:
    """Create a mock ContextService that falls through to fallback."""
    mock = MagicMock(spec=ContextService)
    mock.assemble_context.side_effect = Exception("Use fallback")
    return mock


def _mock_config_registry() -> AgentConfigRegistry:
    return MagicMock(spec=AgentConfigRegistry)


def _all_main_agents() -> list[str]:
    return (
        COGNITION_AGENTS
        + REGIME_PARALLEL_AGENTS
        + ["REGIME-MISMATCH", "DECIDE-SYNTH", "DECIDE-OPTIM"]
        + EXECUTION_AGENTS
    )


# ===========================================================================
# Main Loop Tests
# ===========================================================================

class TestMainLoop:
    @pytest.mark.asyncio
    async def test_all_stages_succeed(self):
        agents = _build_agent_registry(_all_main_agents())
        orch = Orchestrator(agents, _mock_context_service(), _mock_config_registry())

        run = await orch.run_main_loop(fragments=[])
        assert run.status == RunStatus.SUCCEEDED
        assert run.loop_type == "MAIN"
        assert len(run.stage_results) == len(_all_main_agents())
        assert all(s.status == StageStatus.SUCCEEDED for s in run.stage_results)

    @pytest.mark.asyncio
    async def test_cognition_runs_parallel(self):
        """All 6 cognition agents should be called."""
        agents = _build_agent_registry(_all_main_agents())
        orch = Orchestrator(agents, _mock_context_service(), _mock_config_registry())

        run = await orch.run_main_loop(fragments=[])

        cog_results = [s for s in run.stage_results if s.agent_id in COGNITION_AGENTS]
        assert len(cog_results) == 6
        assert all(s.status == StageStatus.SUCCEEDED for s in cog_results)

    @pytest.mark.asyncio
    async def test_metadata_accumulates(self):
        agents = _build_agent_registry(_all_main_agents())
        orch = Orchestrator(agents, _mock_context_service(), _mock_config_registry())

        run = await orch.run_main_loop(fragments=[])
        # Metadata should contain belief_objects and regime_outputs
        assert "belief_objects" in run.metadata
        assert "regime_outputs" in run.metadata

    @pytest.mark.asyncio
    async def test_cognition_failure_partial(self):
        """One cognition agent failing → PARTIAL_FAILURE (not FAILED)."""
        failing = {"COGNIT-MACRO": AgentProcessingError("LLM error", agent_id="COGNIT-MACRO")}
        agents = _build_agent_registry(_all_main_agents(), failing=failing)
        orch = Orchestrator(agents, _mock_context_service(), _mock_config_registry())

        run = await orch.run_main_loop(fragments=[])
        assert run.status == RunStatus.PARTIAL_FAILURE
        assert run.failed_count == 1

    @pytest.mark.asyncio
    async def test_execution_failure_is_critical(self):
        """Execution stage failure → FAILED status."""
        failing = {"EXEC-VALIDATE": AgentProcessingError("Validation error")}
        agents = _build_agent_registry(_all_main_agents(), failing=failing)
        orch = Orchestrator(agents, _mock_context_service(), _mock_config_registry())

        run = await orch.run_main_loop(fragments=[])
        assert run.status == RunStatus.FAILED

    @pytest.mark.asyncio
    async def test_execution_skips_on_upstream_failure(self):
        """EXEC-ROUTER should be skipped if EXEC-VALIDATE failed."""
        failing = {"EXEC-VALIDATE": AgentProcessingError("fail")}
        agents = _build_agent_registry(_all_main_agents(), failing=failing)
        orch = Orchestrator(agents, _mock_context_service(), _mock_config_registry())

        run = await orch.run_main_loop(fragments=[])
        router_result = next(
            (s for s in run.stage_results if s.agent_id == "EXEC-ROUTER"), None,
        )
        assert router_result is not None
        assert router_result.status == StageStatus.SKIPPED

    @pytest.mark.asyncio
    async def test_content_hash(self):
        agents = _build_agent_registry(_all_main_agents())
        orch = Orchestrator(agents, _mock_context_service(), _mock_config_registry())
        run = await orch.run_main_loop(fragments=[])
        assert len(run.content_hash) == 64

    @pytest.mark.asyncio
    async def test_initial_metadata_passed(self):
        agents = _build_agent_registry(_all_main_agents())
        orch = Orchestrator(agents, _mock_context_service(), _mock_config_registry())
        run = await orch.run_main_loop(fragments=[], metadata={"custom_key": "custom_value"})
        assert run.metadata.get("custom_key") == "custom_value"

    @pytest.mark.asyncio
    async def test_missing_agent_handled(self):
        """Missing agents should not crash the pipeline."""
        # Only register cognition agents — everything else missing
        agents = _build_agent_registry(COGNITION_AGENTS)
        orch = Orchestrator(agents, _mock_context_service(), _mock_config_registry())

        run = await orch.run_main_loop(fragments=[])
        # Should still complete (missing agents → skipped, not crashed)
        assert run.status in (RunStatus.SUCCEEDED, RunStatus.PARTIAL_FAILURE, RunStatus.FAILED)
        cog_results = [s for s in run.stage_results if s.agent_id in COGNITION_AGENTS]
        assert all(s.status == StageStatus.SUCCEEDED for s in cog_results)
        # Non-cognition agents should be skipped (not in registry)
        skipped = [s for s in run.stage_results if s.status == StageStatus.SKIPPED]
        assert len(skipped) > 0


# ===========================================================================
# Exit Loop Tests
# ===========================================================================

class TestExitLoop:
    @pytest.mark.asyncio
    async def test_all_succeed(self):
        agents = _build_agent_registry(EXIT_AGENTS)
        orch = Orchestrator(agents, _mock_context_service(), _mock_config_registry())
        run = await orch.run_exit_loop()
        assert run.status == RunStatus.SUCCEEDED
        assert run.loop_type == "EXIT"
        assert len(run.stage_results) == 5

    @pytest.mark.asyncio
    async def test_partial_failure(self):
        failing = {"INVALID-MON": AgentProcessingError("error")}
        agents = _build_agent_registry(EXIT_AGENTS, failing=failing)
        orch = Orchestrator(agents, _mock_context_service(), _mock_config_registry())
        run = await orch.run_exit_loop()
        assert run.status == RunStatus.PARTIAL_FAILURE

    @pytest.mark.asyncio
    async def test_sequential_skip_on_failure(self):
        failing = {"COGNIT-EXIT": AgentProcessingError("error")}
        agents = _build_agent_registry(EXIT_AGENTS, failing=failing)
        orch = Orchestrator(agents, _mock_context_service(), _mock_config_registry())
        run = await orch.run_exit_loop()
        # INVALID-MON should be skipped since COGNIT-EXIT failed
        invalid_mon = next(s for s in run.stage_results if s.agent_id == "INVALID-MON")
        assert invalid_mon.status == StageStatus.SKIPPED


# ===========================================================================
# Learning Loop Tests
# ===========================================================================

class TestLearningLoop:
    @pytest.mark.asyncio
    async def test_all_succeed(self):
        agents = _build_agent_registry(LEARNING_AGENTS)
        orch = Orchestrator(agents, _mock_context_service(), _mock_config_registry())
        run = await orch.run_learning_loop()
        assert run.status == RunStatus.SUCCEEDED
        assert run.loop_type == "LEARNING"
        assert len(run.stage_results) == 4

    @pytest.mark.asyncio
    async def test_metadata_passed(self):
        agents = _build_agent_registry(LEARNING_AGENTS)
        orch = Orchestrator(agents, _mock_context_service(), _mock_config_registry())
        run = await orch.run_learning_loop(metadata={"trade_history": []})
        assert "trade_history" in run.metadata


# ===========================================================================
# Governance Loop Tests
# ===========================================================================

class TestGovernanceLoop:
    @pytest.mark.asyncio
    async def test_all_succeed(self):
        agents = _build_agent_registry(GOVERNANCE_AGENTS)
        orch = Orchestrator(agents, _mock_context_service(), _mock_config_registry())
        run = await orch.run_governance_loop()
        assert run.status == RunStatus.SUCCEEDED
        assert run.loop_type == "GOVERNANCE"
        assert len(run.stage_results) == 4


# ===========================================================================
# PipelineRun Model Tests
# ===========================================================================

class TestPipelineRunModel:
    @pytest.mark.asyncio
    async def test_succeeded_count(self):
        agents = _build_agent_registry(LEARNING_AGENTS)
        orch = Orchestrator(agents, _mock_context_service(), _mock_config_registry())
        run = await orch.run_learning_loop()
        assert run.succeeded_count == 4
        assert run.failed_count == 0
        assert run.skipped_count == 0

    @pytest.mark.asyncio
    async def test_total_duration(self):
        agents = _build_agent_registry(LEARNING_AGENTS)
        orch = Orchestrator(agents, _mock_context_service(), _mock_config_registry())
        run = await orch.run_learning_loop()
        assert run.total_duration_ms > 0

    @pytest.mark.asyncio
    async def test_timestamps(self):
        agents = _build_agent_registry(LEARNING_AGENTS)
        orch = Orchestrator(agents, _mock_context_service(), _mock_config_registry())
        run = await orch.run_learning_loop()
        assert run.started_at is not None
        assert run.finished_at is not None
        assert run.finished_at >= run.started_at


# ===========================================================================
# ProvidenceRunner Tests
# ===========================================================================

class TestProvidenceRunner:
    @pytest.mark.asyncio
    async def test_run_once(self):
        from providence.orchestration.runner import ProvidenceRunner

        agents = _build_agent_registry(
            _all_main_agents() + EXIT_AGENTS + GOVERNANCE_AGENTS,
        )
        orch = Orchestrator(agents, _mock_context_service(), _mock_config_registry())
        runner = ProvidenceRunner(orch)

        runs = await runner.run_once(fragments=[])
        assert "MAIN" in runs
        assert "EXIT" in runs
        assert "GOVERNANCE" in runs
        assert all(r.status == RunStatus.SUCCEEDED for r in runs.values())

    @pytest.mark.asyncio
    async def test_run_once_without_exit(self):
        from providence.orchestration.runner import ProvidenceRunner

        agents = _build_agent_registry(_all_main_agents())
        orch = Orchestrator(agents, _mock_context_service(), _mock_config_registry())
        runner = ProvidenceRunner(orch)

        runs = await runner.run_once(fragments=[], run_exit=False, run_governance=False)
        assert "MAIN" in runs
        assert "EXIT" not in runs
        assert "GOVERNANCE" not in runs

    @pytest.mark.asyncio
    async def test_learning_batch(self):
        from providence.orchestration.runner import ProvidenceRunner

        agents = _build_agent_registry(LEARNING_AGENTS)
        orch = Orchestrator(agents, _mock_context_service(), _mock_config_registry())
        runner = ProvidenceRunner(orch)

        run = await runner.run_learning_batch(metadata={"trade_history": []})
        assert run.status == RunStatus.SUCCEEDED

    def test_shutdown(self):
        from providence.orchestration.runner import ProvidenceRunner

        orch = Orchestrator({}, _mock_context_service(), _mock_config_registry())
        runner = ProvidenceRunner(orch)
        assert not runner.is_running
        runner.request_shutdown()
        assert runner._shutdown_event.is_set()
