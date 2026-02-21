"""End-to-end integration test for the full Providence pipeline.

Validates:
  1. Full pipeline cycle: Main loop → Exit loop → Governance loop
  2. Storage integration: FragmentStore → Runner → BeliefStore + RunStore
  3. Learning batch runs correctly
  4. Continuous mode starts/stops cleanly
  5. All 35 agents execute without errors (using mock implementations)

Uses mock agents throughout (no real API calls or LLM inference).
"""

import asyncio
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from providence.agents.base import AgentContext, BaseAgent, HealthStatus
from providence.config.agent_config import AgentConfigRegistry
from providence.orchestration.models import RunStatus, StageStatus
from providence.orchestration.orchestrator import (
    COGNITION_AGENTS,
    EXECUTION_AGENTS,
    EXIT_AGENTS,
    GOVERNANCE_AGENTS,
    LEARNING_AGENTS,
    REGIME_PARALLEL_AGENTS,
    Orchestrator,
)
from providence.orchestration.runner import ProvidenceRunner
from providence.schemas.enums import DataType, ValidationStatus
from providence.schemas.market_state import MarketStateFragment
from providence.services.context_svc import ContextService
from providence.storage.belief_store import BeliefStore
from providence.storage.fragment_store import FragmentStore
from providence.storage.run_store import RunStore


NOW = datetime(2026, 2, 20, 12, 0, 0, tzinfo=timezone.utc)

ALL_MAIN_AGENTS = (
    COGNITION_AGENTS
    + REGIME_PARALLEL_AGENTS
    + ["REGIME-MISMATCH", "DECIDE-SYNTH", "DECIDE-OPTIM"]
    + EXECUTION_AGENTS
)


# ---------------------------------------------------------------------------
# Mock Agents
# ---------------------------------------------------------------------------


class MockAgent(BaseAgent[dict]):
    """Simple mock that returns a dict output."""

    def __init__(self, agent_id: str) -> None:
        super().__init__(agent_id=agent_id, agent_type="test", version="1.0.0")

    async def process(self, context: AgentContext) -> dict:
        return {"agent_id": self.agent_id, "status": "ok"}

    def get_health(self) -> HealthStatus:
        return HealthStatus(
            agent_id=self.agent_id,
            status="HEALTHY",
            last_run=NOW,
            error_count_24h=0,
        )


class MockCognitionAgent(BaseAgent[dict]):
    """Mock cognition agent that returns a BeliefObject-like dict."""

    def __init__(self, agent_id: str) -> None:
        super().__init__(agent_id=agent_id, agent_type="cognition", version="1.0.0")

    async def process(self, context: AgentContext) -> dict:
        return {
            "belief_id": str(uuid4()),
            "agent_id": self.agent_id,
            "timestamp": NOW.isoformat(),
            "context_window_hash": "test-hash",
            "beliefs": [],
            "content_hash": "test-content-hash",
        }

    def get_health(self) -> HealthStatus:
        return HealthStatus(
            agent_id=self.agent_id,
            status="HEALTHY",
            last_run=NOW,
            error_count_24h=0,
        )


PERCEPTION_AGENTS = [
    "PERCEPT-PRICE", "PERCEPT-FILING", "PERCEPT-NEWS",
    "PERCEPT-OPTIONS", "PERCEPT-CDS", "PERCEPT-MACRO",
]


def _build_registry() -> dict[str, BaseAgent]:
    """Build a full 35-agent registry with mocks."""
    registry: dict[str, BaseAgent] = {}
    # 6 Perception agents (not orchestrated, but part of the registry)
    for aid in PERCEPTION_AGENTS:
        registry[aid] = MockAgent(aid)
    # 16 Main loop agents (6 cognition + 3 regime parallel + 1 mismatch + 2 decision + 4 execution)
    for aid in ALL_MAIN_AGENTS:
        if aid.startswith("COGNIT-"):
            registry[aid] = MockCognitionAgent(aid)
        else:
            registry[aid] = MockAgent(aid)
    # 5 Exit + 4 Learning + 4 Governance
    for aid in EXIT_AGENTS + LEARNING_AGENTS + GOVERNANCE_AGENTS:
        registry[aid] = MockAgent(aid)
    return registry


def _make_fragment(entity: str = "AAPL", data_type: DataType = DataType.PRICE_OHLCV) -> MarketStateFragment:
    return MarketStateFragment(
        fragment_id=uuid4(),
        agent_id="PERCEPT-PRICE",
        timestamp=NOW,
        source_timestamp=NOW - timedelta(hours=1),
        entity=entity,
        data_type=data_type,
        schema_version="1.0.0",
        source_hash=f"test-{uuid4().hex[:8]}",
        validation_status=ValidationStatus.VALID,
        payload={"close": 150.0, "volume": 50000},
    )


def _build_system(
    fragment_store: FragmentStore | None = None,
    belief_store: BeliefStore | None = None,
    run_store: RunStore | None = None,
) -> ProvidenceRunner:
    """Build a complete system with all stores."""
    registry = _build_registry()
    config_registry = AgentConfigRegistry(configs={})
    context_service = ContextService(config_registry)
    orchestrator = Orchestrator(
        agent_registry=registry,
        context_service=context_service,
        config_registry=config_registry,
    )
    return ProvidenceRunner(
        orchestrator,
        fragment_store=fragment_store,
        belief_store=belief_store,
        run_store=run_store,
    )


# ===========================================================================
# Test 1: Full Pipeline Cycle
# ===========================================================================


class TestFullPipelineCycle:
    """End-to-end test: fragments → Main → Exit → Governance → storage."""

    @pytest.mark.asyncio
    async def test_complete_cycle_with_all_stores(self):
        """Run full cycle with all stores and verify everything persists."""
        fragment_store = FragmentStore()
        belief_store = BeliefStore()
        run_store = RunStore()

        # Pre-load fragments
        fragment_store.append(_make_fragment("AAPL"))
        fragment_store.append(_make_fragment("NVDA"))
        fragment_store.append(_make_fragment("JPM"))

        runner = _build_system(
            fragment_store=fragment_store,
            belief_store=belief_store,
            run_store=run_store,
        )

        # Run full cycle — fragments pulled from store
        runs = await runner.run_once(
            fragments=None,
            run_exit=True,
            run_governance=True,
        )

        # All 3 loops should complete
        assert "MAIN" in runs
        assert "EXIT" in runs
        assert "GOVERNANCE" in runs

        # All should succeed
        assert runs["MAIN"].status == RunStatus.SUCCEEDED
        assert runs["EXIT"].status == RunStatus.SUCCEEDED
        assert runs["GOVERNANCE"].status == RunStatus.SUCCEEDED

        # RunStore should have 3 entries
        assert run_store.count() == 3
        assert run_store.count(loop_type="MAIN") == 1
        assert run_store.count(loop_type="EXIT") == 1
        assert run_store.count(loop_type="GOVERNANCE") == 1

        # All runs should have stage results
        for loop_type, run in runs.items():
            assert len(run.stage_results) > 0
            assert run.started_at is not None
            assert run.finished_at is not None
            assert run.total_duration_ms >= 0

    @pytest.mark.asyncio
    async def test_main_loop_stage_count(self):
        """Main loop should produce results for all expected agents."""
        run_store = RunStore()
        runner = _build_system(run_store=run_store)

        runs = await runner.run_once(fragments=[], run_exit=False, run_governance=False)

        main_run = runs["MAIN"]
        # 6 cognition + 3 regime parallel + 1 REGIME-MISMATCH + 2 decision + 4 execution = 16
        assert len(main_run.stage_results) == 16
        assert all(sr.status == StageStatus.SUCCEEDED for sr in main_run.stage_results)

    @pytest.mark.asyncio
    async def test_exit_loop_stage_count(self):
        """Exit loop should produce results for all 5 exit agents."""
        run_store = RunStore()
        runner = _build_system(run_store=run_store)

        runs = await runner.run_once(fragments=[], run_exit=True, run_governance=False)

        exit_run = runs["EXIT"]
        assert len(exit_run.stage_results) == 5
        assert all(sr.status == StageStatus.SUCCEEDED for sr in exit_run.stage_results)

    @pytest.mark.asyncio
    async def test_governance_loop_stage_count(self):
        """Governance loop should produce results for all 4 governance agents."""
        run_store = RunStore()
        runner = _build_system(run_store=run_store)

        runs = await runner.run_once(fragments=[], run_exit=False, run_governance=True)

        gov_run = runs["GOVERNANCE"]
        assert len(gov_run.stage_results) == 4
        assert all(sr.status == StageStatus.SUCCEEDED for sr in gov_run.stage_results)

    @pytest.mark.asyncio
    async def test_metadata_flows_through_stages(self):
        """Main loop metadata should accumulate outputs from all stages."""
        runner = _build_system()
        runs = await runner.run_once(fragments=[])

        meta = runs["MAIN"].metadata
        # Cognition outputs should be in metadata
        assert "belief_objects" in meta
        assert len(meta["belief_objects"]) == 6  # 6 cognition agents

        # Regime outputs should be in metadata
        assert "regime_outputs" in meta


# ===========================================================================
# Test 2: Storage Roundtrip (disk persistence)
# ===========================================================================


class TestStorageRoundtrip:
    """Test that pipeline data persists to disk and reloads correctly."""

    @pytest.mark.asyncio
    async def test_full_persistence_roundtrip(self, tmp_path):
        """Run pipeline, persist to disk, reload, and verify."""
        frag_path = tmp_path / "fragments.jsonl"
        belief_path = tmp_path / "beliefs.jsonl"
        run_path = tmp_path / "runs.jsonl"

        # Session 1: create stores, load fragments, run pipeline
        frag_store1 = FragmentStore(persist_path=frag_path)
        belief_store1 = BeliefStore(persist_path=belief_path)
        run_store1 = RunStore(persist_path=run_path)

        frag_store1.append(_make_fragment("AAPL"))
        frag_store1.append(_make_fragment("NVDA"))

        runner1 = _build_system(
            fragment_store=frag_store1,
            belief_store=belief_store1,
            run_store=run_store1,
        )
        await runner1.run_once(fragments=None, run_exit=True, run_governance=True)

        # Verify session 1 state
        assert frag_store1.count() == 2
        assert run_store1.count() == 3

        # Session 2: reload from disk
        frag_store2 = FragmentStore(persist_path=frag_path)
        belief_store2 = BeliefStore(persist_path=belief_path)
        run_store2 = RunStore(persist_path=run_path)

        # Fragments should survive
        assert frag_store2.count() == 2

        # Runs should survive
        assert run_store2.count() == 3
        latest = run_store2.get_latest()
        assert latest is not None

        # Run pipeline again from reloaded stores
        runner2 = _build_system(
            fragment_store=frag_store2,
            belief_store=belief_store2,
            run_store=run_store2,
        )
        await runner2.run_once(fragments=None, run_exit=False, run_governance=False)

        # Should now have 4 total runs (3 from session 1 + 1 MAIN from session 2)
        assert run_store2.count() == 4


# ===========================================================================
# Test 3: Learning Batch
# ===========================================================================


class TestLearningBatch:
    @pytest.mark.asyncio
    async def test_learning_batch_runs_all_agents(self):
        """Learning batch should run all 4 learning agents."""
        run_store = RunStore()
        runner = _build_system(run_store=run_store)

        run = await runner.run_learning_batch()

        assert run.loop_type == "LEARNING"
        assert run.status == RunStatus.SUCCEEDED
        assert len(run.stage_results) == 4

        # Verify in RunStore
        assert run_store.count() == 1
        assert run_store.count(loop_type="LEARNING") == 1

    @pytest.mark.asyncio
    async def test_learning_batch_with_metadata(self):
        """Learning batch should pass metadata to agents."""
        runner = _build_system()

        metadata = {
            "trade_history": [{"ticker": "AAPL", "pnl_bps": 150}],
            "evaluation_period": "2026-01",
        }
        run = await runner.run_learning_batch(metadata=metadata)

        assert run.status == RunStatus.SUCCEEDED
        # Metadata should flow through
        assert "trade_history" in run.metadata


# ===========================================================================
# Test 4: Continuous Mode
# ===========================================================================


class TestContinuousMode:
    @pytest.mark.asyncio
    async def test_continuous_mode_starts_and_stops(self):
        """Continuous mode should run at least one cycle and stop gracefully."""
        run_store = RunStore()
        runner = _build_system(run_store=run_store)

        # Schedule shutdown after a short delay
        async def schedule_shutdown():
            await asyncio.sleep(0.5)
            runner.request_shutdown()

        # Run continuous with very short interval
        shutdown_task = asyncio.create_task(schedule_shutdown())
        await runner.run_continuous(
            interval_seconds=0.1,
            run_exit=False,
            run_governance=False,
        )
        await shutdown_task

        # Should have completed at least 1 cycle
        assert run_store.count() >= 1
        assert not runner.is_running

    @pytest.mark.asyncio
    async def test_continuous_with_fragment_provider(self):
        """Continuous mode should accept a fragment provider callable."""
        run_store = RunStore()
        runner = _build_system(run_store=run_store)

        call_count = 0

        async def fragment_provider():
            nonlocal call_count
            call_count += 1
            return [_make_fragment("AAPL")]

        async def schedule_shutdown():
            await asyncio.sleep(0.3)
            runner.request_shutdown()

        shutdown_task = asyncio.create_task(schedule_shutdown())
        await runner.run_continuous(
            fragment_provider=fragment_provider,
            interval_seconds=0.1,
            run_exit=False,
            run_governance=False,
        )
        await shutdown_task

        assert call_count >= 1
        assert run_store.count() >= 1


# ===========================================================================
# Test 5: Multi-Cycle Analytics
# ===========================================================================


class TestMultiCycleAnalytics:
    @pytest.mark.asyncio
    async def test_success_rate_across_cycles(self):
        """RunStore success_rate should track across multiple cycles."""
        run_store = RunStore()
        runner = _build_system(run_store=run_store)

        # Run 3 cycles (all should succeed with mock agents)
        for _ in range(3):
            await runner.run_once(fragments=[], run_exit=False, run_governance=False)

        assert run_store.count() == 3
        assert run_store.success_rate() == 1.0
        assert run_store.success_rate(loop_type="MAIN") == 1.0

    @pytest.mark.asyncio
    async def test_fragment_store_queries_after_pipeline(self):
        """FragmentStore should support queries after pipeline runs."""
        frag_store = FragmentStore()
        frag_store.append(_make_fragment("AAPL", DataType.PRICE_OHLCV))
        frag_store.append(_make_fragment("AAPL", DataType.FILING_10Q))
        frag_store.append(_make_fragment("NVDA", DataType.PRICE_OHLCV))

        runner = _build_system(fragment_store=frag_store)
        await runner.run_once(fragments=None)

        # Store should still be queryable
        assert frag_store.count() == 3
        assert len(frag_store.query(entities={"AAPL"})) == 2
        assert len(frag_store.query(data_types={DataType.PRICE_OHLCV})) == 2
        assert frag_store.all_entities() == {"AAPL", "NVDA"}


# ===========================================================================
# Test 6: Agent Count Verification
# ===========================================================================


class TestAgentCountVerification:
    def test_all_35_agents_in_registry(self):
        """Registry should contain exactly 35 agents."""
        registry = _build_registry()
        assert len(registry) == 35

    def test_all_agent_groups_present(self):
        """All agent group lists should sum to 35."""
        total = (
            len(COGNITION_AGENTS)  # 6
            + len(REGIME_PARALLEL_AGENTS)  # 3
            + 1  # REGIME-MISMATCH
            + 2  # DECIDE-SYNTH, DECIDE-OPTIM
            + len(EXECUTION_AGENTS)  # 4
            + len(EXIT_AGENTS)  # 5
            + len(LEARNING_AGENTS)  # 4
            + len(GOVERNANCE_AGENTS)  # 4
        )
        # 6 + 3 + 1 + 2 + 4 + 5 + 4 + 4 = 29... missing 6 perception agents
        # Perception agents are not in orchestrator groups (they run externally)
        # Main loop agents: 16. Plus Exit (5) + Learning (4) + Governance (4) = 29
        # The remaining 6 are Perception agents (not orchestrated in loops)
        assert total == 29  # Orchestrated agents

    def test_registry_agent_ids_match_groups(self):
        """Every agent in group constants should be in the registry."""
        registry = _build_registry()
        all_group_agents = (
            ALL_MAIN_AGENTS
            + EXIT_AGENTS
            + LEARNING_AGENTS
            + GOVERNANCE_AGENTS
        )
        for aid in all_group_agents:
            assert aid in registry, f"Agent {aid} missing from registry"
