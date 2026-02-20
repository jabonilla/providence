"""Orchestrator — DAG coordinator for the Providence pipeline.

Wires all 31 agents together across 4 loops:
  - Main loop: Cognition → Regime → Decision → Execution
  - Exit loop: COGNIT-EXIT → INVALID-MON → THESIS-RENEW → SHADOW-EXIT → RENEW-MON
  - Learning loop: LEARN-ATTRIB → LEARN-CALIB → LEARN-RETRAIN → LEARN-BACKTEST
  - Governance loop: GOVERN-CAPITAL → GOVERN-MATURITY → GOVERN-OVERSIGHT → GOVERN-POLICY

Cognition agents run in parallel (agent independence).
Regime agents (STAT, SECTOR, NARR) run in parallel, then MISMATCH sequential.
Execution pipeline is strictly sequential.
"""

import asyncio
from datetime import datetime, timezone
from typing import Any, Optional

import structlog

from providence.agents.base import AgentContext, BaseAgent
from providence.config.agent_config import AgentConfigRegistry
from providence.orchestration.models import PipelineRun, RunStatus, StageStatus
from providence.orchestration.stage import PipelineStage
from providence.schemas.market_state import MarketStateFragment
from providence.services.context_svc import ContextService

logger = structlog.get_logger()

# Agent groups
COGNITION_AGENTS = [
    "COGNIT-FUNDAMENTAL",
    "COGNIT-TECHNICAL",
    "COGNIT-NARRATIVE",
    "COGNIT-MACRO",
    "COGNIT-EVENT",
    "COGNIT-CROSSSEC",
]

REGIME_PARALLEL_AGENTS = [
    "REGIME-STAT",
    "REGIME-SECTOR",
    "REGIME-NARR",
]

EXECUTION_AGENTS = [
    "EXEC-VALIDATE",
    "EXEC-ROUTER",
    "EXEC-GUARDIAN",
    "EXEC-CAPTURE",
]

EXIT_AGENTS = [
    "COGNIT-EXIT",
    "INVALID-MON",
    "THESIS-RENEW",
    "SHADOW-EXIT",
    "RENEW-MON",
]

LEARNING_AGENTS = [
    "LEARN-ATTRIB",
    "LEARN-CALIB",
    "LEARN-RETRAIN",
    "LEARN-BACKTEST",
]

GOVERNANCE_AGENTS = [
    "GOVERN-CAPITAL",
    "GOVERN-MATURITY",
    "GOVERN-OVERSIGHT",
    "GOVERN-POLICY",
]


class Orchestrator:
    """DAG coordinator for the Providence pipeline.

    Manages agent execution order, parallel stages, metadata flow,
    and error isolation across the 4 pipeline loops.
    """

    def __init__(
        self,
        agent_registry: dict[str, BaseAgent],
        context_service: ContextService,
        config_registry: AgentConfigRegistry,
        default_timeout: float = 120.0,
    ) -> None:
        self._agents = agent_registry
        self._context_svc = context_service
        self._config_registry = config_registry
        self._default_timeout = default_timeout

    def _get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """Get agent from registry, or None if not found."""
        return self._agents.get(agent_id)

    def _make_context(
        self,
        agent_id: str,
        trigger: str,
        fragments: list[MarketStateFragment],
        metadata: dict[str, Any],
    ) -> AgentContext:
        """Assemble AgentContext for an agent using ContextService.

        Falls back to direct construction if ContextService is unavailable
        or agent config is not found.
        """
        try:
            ctx = self._context_svc.assemble_context(
                agent_id=agent_id,
                trigger=trigger,
                available_fragments=fragments,
                metadata=metadata,
            )
            return ctx
        except Exception:
            # Fallback: construct directly
            from providence.utils.hashing import compute_context_window_hash

            frag_hashes = sorted(f.version for f in fragments if hasattr(f, "version"))
            ctx_hash = compute_context_window_hash(frag_hashes) if frag_hashes else "empty"

            return AgentContext(
                agent_id=agent_id,
                trigger=trigger,
                fragments=fragments,
                context_window_hash=ctx_hash,
                timestamp=datetime.now(timezone.utc),
                metadata=metadata,
            )

    def _make_metadata_context(
        self,
        agent_id: str,
        trigger: str,
        metadata: dict[str, Any],
    ) -> AgentContext:
        """Create AgentContext with metadata only (no fragments).

        Used for downstream agents that read from metadata, not raw fragments.
        """
        return AgentContext(
            agent_id=agent_id,
            trigger=trigger,
            fragments=[],
            context_window_hash="metadata-only",
            timestamp=datetime.now(timezone.utc),
            metadata=metadata,
        )

    async def _run_parallel_stages(
        self,
        agent_ids: list[str],
        trigger: str,
        fragments: list[MarketStateFragment],
        metadata: dict[str, Any],
    ) -> list[tuple[str, Any]]:
        """Run multiple agents in parallel using asyncio.gather.

        Args:
            agent_ids: List of agent IDs to run in parallel.
            trigger: Trigger type.
            fragments: Available fragments for context assembly.
            metadata: Shared metadata dict.

        Returns:
            List of (agent_id, StageResult) tuples.
        """
        tasks = []
        valid_ids = []

        for aid in agent_ids:
            agent = self._get_agent(aid)
            if agent is None:
                continue
            valid_ids.append(aid)
            ctx = self._make_context(aid, trigger, fragments, metadata)
            stage = PipelineStage(aid, agent, self._default_timeout)
            tasks.append(stage.execute(ctx))

        if not tasks:
            return []

        results = await asyncio.gather(*tasks, return_exceptions=True)
        output = []
        for aid, result in zip(valid_ids, results):
            if isinstance(result, Exception):
                from providence.orchestration.stage import PipelineStage as PS
                sr = PS.make_skipped(aid, aid, f"gather exception: {result}")
                output.append((aid, sr))
            else:
                output.append((aid, result))

        return output

    async def _run_sequential_stage(
        self,
        agent_id: str,
        trigger: str,
        metadata: dict[str, Any],
        fragments: Optional[list[MarketStateFragment]] = None,
        required_upstream: Optional[list[str]] = None,
        completed_stages: Optional[dict[str, Any]] = None,
    ) -> Any:
        """Run a single agent sequentially with dependency checking.

        Args:
            agent_id: Agent to execute.
            trigger: Trigger type.
            metadata: Current metadata.
            fragments: Optional fragments (None = metadata-only context).
            required_upstream: Stage names that must have succeeded.
            completed_stages: Map of completed stage names → StageResults.

        Returns:
            StageResult for this execution.
        """
        agent = self._get_agent(agent_id)
        if agent is None:
            return PipelineStage.make_skipped(agent_id, agent_id, "Agent not in registry")

        # Check upstream dependencies
        if required_upstream and completed_stages:
            for dep in required_upstream:
                dep_result = completed_stages.get(dep)
                if dep_result and hasattr(dep_result, "status") and dep_result.status == StageStatus.FAILED:
                    return PipelineStage.make_skipped(
                        agent_id, agent_id, f"Skipped: upstream {dep} failed",
                    )

        if fragments is not None:
            ctx = self._make_context(agent_id, trigger, fragments, metadata)
        else:
            ctx = self._make_metadata_context(agent_id, trigger, metadata)

        stage = PipelineStage(agent_id, agent, self._default_timeout)
        return await stage.execute(ctx)

    def _inject_output(
        self,
        metadata: dict[str, Any],
        key: str,
        result: Any,
    ) -> None:
        """Inject a stage's output into metadata for downstream consumption.

        Args:
            metadata: Metadata dict to update.
            key: Key to store the output under.
            result: StageResult with output.
        """
        if hasattr(result, "status") and result.status == StageStatus.SUCCEEDED and result.output:
            metadata[key] = result.output

    # ------------------------------------------------------------------
    # Main Loop
    # ------------------------------------------------------------------

    async def run_main_loop(
        self,
        fragments: list[MarketStateFragment],
        metadata: Optional[dict[str, Any]] = None,
        trigger: str = "schedule",
    ) -> PipelineRun:
        """Execute the main trading loop.

        Cognition (parallel) → Regime (parallel + MISMATCH) →
        Decision (sequential) → Execution (strictly sequential).

        Args:
            fragments: MarketStateFragments from Perception.
            metadata: Optional initial metadata.
            trigger: Trigger type.

        Returns:
            PipelineRun with all stage results.
        """
        log = logger.bind(loop="MAIN")
        log.info("Main loop starting")
        started_at = datetime.now(timezone.utc)
        meta = dict(metadata or {})
        all_results = []
        completed = {}

        # Stage 1: Cognition (parallel)
        log.info("Running cognition stage (parallel)")
        cog_results = await self._run_parallel_stages(
            COGNITION_AGENTS, trigger, fragments, meta,
        )
        belief_objects = []
        for aid, sr in cog_results:
            all_results.append(sr)
            completed[aid] = sr
            if sr.status == StageStatus.SUCCEEDED and sr.output:
                belief_objects.append(sr.output)
        meta["belief_objects"] = belief_objects

        # Stage 2: Regime (parallel: STAT, SECTOR, NARR)
        log.info("Running regime stage (parallel)")
        regime_results = await self._run_parallel_stages(
            REGIME_PARALLEL_AGENTS, trigger, fragments, meta,
        )
        regime_outputs = {}
        for aid, sr in regime_results:
            all_results.append(sr)
            completed[aid] = sr
            if sr.status == StageStatus.SUCCEEDED and sr.output:
                regime_outputs[aid] = sr.output
        meta["regime_outputs"] = regime_outputs

        # Stage 3: REGIME-MISMATCH (sequential, depends on regime)
        log.info("Running regime mismatch stage")
        mismatch_result = await self._run_sequential_stage(
            "REGIME-MISMATCH", trigger, meta,
            required_upstream=REGIME_PARALLEL_AGENTS,
            completed_stages=completed,
        )
        all_results.append(mismatch_result)
        completed["REGIME-MISMATCH"] = mismatch_result
        self._inject_output(meta, "regime_mismatch", mismatch_result)

        # Stage 4: DECIDE-SYNTH (sequential)
        log.info("Running decision synthesis stage")
        synth_result = await self._run_sequential_stage(
            "DECIDE-SYNTH", trigger, meta, fragments=fragments,
        )
        all_results.append(synth_result)
        completed["DECIDE-SYNTH"] = synth_result
        self._inject_output(meta, "synthesis_output", synth_result)

        # Stage 5: DECIDE-OPTIM (sequential)
        log.info("Running decision optimization stage")
        optim_result = await self._run_sequential_stage(
            "DECIDE-OPTIM", trigger, meta,
            required_upstream=["DECIDE-SYNTH"],
            completed_stages=completed,
        )
        all_results.append(optim_result)
        completed["DECIDE-OPTIM"] = optim_result
        self._inject_output(meta, "optimization_output", optim_result)

        # Stage 6: Execution (strictly sequential)
        log.info("Running execution pipeline (sequential)")
        prev_exec = None
        for exec_aid in EXECUTION_AGENTS:
            deps = [prev_exec] if prev_exec else []
            exec_result = await self._run_sequential_stage(
                exec_aid, trigger, meta,
                required_upstream=deps,
                completed_stages=completed,
            )
            all_results.append(exec_result)
            completed[exec_aid] = exec_result
            self._inject_output(meta, f"exec_{exec_aid.lower()}", exec_result)
            prev_exec = exec_aid

        # Determine run status
        failed = sum(1 for r in all_results if r.status == StageStatus.FAILED)
        status = RunStatus.SUCCEEDED
        if failed > 0:
            # Critical failure if any execution stage failed
            exec_failed = any(
                r.status == StageStatus.FAILED
                for r in all_results
                if r.agent_id in EXECUTION_AGENTS
            )
            status = RunStatus.FAILED if exec_failed else RunStatus.PARTIAL_FAILURE

        run = PipelineRun(
            loop_type="MAIN",
            status=status,
            started_at=started_at,
            finished_at=datetime.now(timezone.utc),
            stage_results=all_results,
            metadata=meta,
        )

        log.info(
            "Main loop complete",
            status=status.value,
            stages=len(all_results),
            failed=failed,
            duration_ms=run.total_duration_ms,
        )
        return run

    # ------------------------------------------------------------------
    # Exit Loop
    # ------------------------------------------------------------------

    async def run_exit_loop(
        self,
        metadata: Optional[dict[str, Any]] = None,
        trigger: str = "schedule",
    ) -> PipelineRun:
        """Execute the exit monitoring loop (post-trade).

        Sequential: COGNIT-EXIT → INVALID-MON → THESIS-RENEW →
        SHADOW-EXIT → RENEW-MON.
        """
        return await self._run_sequential_loop(
            "EXIT", EXIT_AGENTS, metadata, trigger,
        )

    # ------------------------------------------------------------------
    # Learning Loop
    # ------------------------------------------------------------------

    async def run_learning_loop(
        self,
        metadata: Optional[dict[str, Any]] = None,
        trigger: str = "manual",
    ) -> PipelineRun:
        """Execute the offline learning loop.

        Sequential: LEARN-ATTRIB → LEARN-CALIB → LEARN-RETRAIN → LEARN-BACKTEST.
        """
        return await self._run_sequential_loop(
            "LEARNING", LEARNING_AGENTS, metadata, trigger,
        )

    # ------------------------------------------------------------------
    # Governance Loop
    # ------------------------------------------------------------------

    async def run_governance_loop(
        self,
        metadata: Optional[dict[str, Any]] = None,
        trigger: str = "schedule",
    ) -> PipelineRun:
        """Execute the governance loop.

        Sequential: GOVERN-CAPITAL → GOVERN-MATURITY →
        GOVERN-OVERSIGHT → GOVERN-POLICY.
        """
        return await self._run_sequential_loop(
            "GOVERNANCE", GOVERNANCE_AGENTS, metadata, trigger,
        )

    # ------------------------------------------------------------------
    # Internal: Sequential Loop Runner
    # ------------------------------------------------------------------

    async def _run_sequential_loop(
        self,
        loop_type: str,
        agent_ids: list[str],
        metadata: Optional[dict[str, Any]],
        trigger: str,
    ) -> PipelineRun:
        """Run a list of agents sequentially, each feeding the next.

        Args:
            loop_type: Loop identifier (EXIT, LEARNING, GOVERNANCE).
            agent_ids: Ordered list of agents to run.
            metadata: Initial metadata.
            trigger: Trigger type.

        Returns:
            PipelineRun with all stage results.
        """
        log = logger.bind(loop=loop_type)
        log.info(f"{loop_type} loop starting")
        started_at = datetime.now(timezone.utc)
        meta = dict(metadata or {})
        all_results = []
        completed = {}
        prev_agent = None

        for aid in agent_ids:
            deps = [prev_agent] if prev_agent else []
            result = await self._run_sequential_stage(
                aid, trigger, meta,
                required_upstream=deps,
                completed_stages=completed,
            )
            all_results.append(result)
            completed[aid] = result
            self._inject_output(meta, f"{loop_type.lower()}_{aid.lower()}", result)
            prev_agent = aid

        failed = sum(1 for r in all_results if r.status == StageStatus.FAILED)
        if failed == len(all_results):
            status = RunStatus.FAILED
        elif failed > 0:
            status = RunStatus.PARTIAL_FAILURE
        else:
            status = RunStatus.SUCCEEDED

        run = PipelineRun(
            loop_type=loop_type,
            status=status,
            started_at=started_at,
            finished_at=datetime.now(timezone.utc),
            stage_results=all_results,
            metadata=meta,
        )

        log.info(
            f"{loop_type} loop complete",
            status=status.value,
            stages=len(all_results),
            failed=failed,
        )
        return run
