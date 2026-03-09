"""ProvidenceRunner — scheduler and entry point for pipeline execution.

Wraps the Orchestrator with scheduling, graceful shutdown, storage
integration, and convenience methods for running pipeline loops.

Storage integration:
  - FragmentStore: pulls available fragments before each cycle
  - BeliefStore: captures cognition outputs after main loop
  - RunStore: persists every PipelineRun for audit and analytics
  - ShadowSignalStore: captures signals in shadow mode (no broker)

Shadow mode integration:
  When system_mode is SHADOW, the pipeline runs fully through
  EXEC-VALIDATE but does not submit orders to any broker. Instead,
  ShadowExecutionService records signals for offline analysis.
"""

import asyncio
from datetime import datetime, timezone
from typing import Any, Optional

import structlog

from providence.orchestration.models import PipelineRun, StageStatus
from providence.orchestration.orchestrator import Orchestrator
from providence.schemas.enums import SystemMode
from providence.schemas.market_state import MarketStateFragment
from providence.services.shadow_execution import ShadowExecutionService, ShadowSignalStore
from providence.storage.belief_store import BeliefStore
from providence.storage.fragment_store import FragmentStore
from providence.storage.run_store import RunStore

logger = structlog.get_logger()


class ProvidenceRunner:
    """Entry point for running Providence pipeline loops.

    Wraps the Orchestrator with:
      - Single run (manual trigger)
      - Continuous scheduling (asyncio.sleep loop)
      - Offline batch runs (learning)
      - Graceful shutdown
      - Storage integration (fragments, beliefs, run history)
      - Shadow mode (signal recording without broker interaction)
    """

    def __init__(
        self,
        orchestrator: Orchestrator,
        fragment_store: FragmentStore | None = None,
        belief_store: BeliefStore | None = None,
        run_store: RunStore | None = None,
        shadow_signal_store: ShadowSignalStore | None = None,
        system_mode: SystemMode = SystemMode.SHADOW,
    ) -> None:
        self._orchestrator = orchestrator
        self._fragment_store = fragment_store
        self._belief_store = belief_store
        self._run_store = run_store
        self._system_mode = system_mode
        self._shutdown_event = asyncio.Event()
        self._running = False

        # Shadow mode services
        self._shadow_signal_store = shadow_signal_store
        self._shadow_svc: ShadowExecutionService | None = None
        if shadow_signal_store is not None:
            self._shadow_svc = ShadowExecutionService(shadow_signal_store)

    @property
    def system_mode(self) -> SystemMode:
        return self._system_mode

    @property
    def shadow_signal_store(self) -> ShadowSignalStore | None:
        return self._shadow_signal_store

    @property
    def is_running(self) -> bool:
        return self._running

    def _persist_run(self, run: PipelineRun) -> None:
        """Persist a PipelineRun to the RunStore if available."""
        if self._run_store is not None:
            self._run_store.append(run)

    def _extract_and_store_beliefs(self, run: PipelineRun) -> int:
        """Extract BeliefObjects from cognition stage results and store them.

        Cognition agents produce BeliefObjects as their output. The
        orchestrator serializes these to dicts in stage_result.output.
        We attempt to reconstruct and store them.

        Returns the number of beliefs stored.
        """
        if self._belief_store is None:
            return 0

        stored = 0
        from providence.schemas.belief import BeliefObject

        for sr in run.stage_results:
            if not sr.agent_id.startswith("COGNIT-"):
                continue
            if sr.status != StageStatus.SUCCEEDED or sr.output is None:
                continue
            try:
                if isinstance(sr.output, dict) and "belief_id" in sr.output:
                    belief = BeliefObject.model_validate(sr.output)
                    if self._belief_store.append(belief):
                        stored += 1
            except Exception as exc:
                logger.debug(
                    "Could not extract belief from stage output",
                    agent_id=sr.agent_id,
                    error=str(exc),
                )
        return stored

    def _get_fragments_from_store(self) -> list[MarketStateFragment]:
        """Pull all fragments from the FragmentStore (including quarantined).

        Quarantined fragments may still contain useful data (e.g. SEC filings
        with metadata but missing XBRL). Downstream agents handle data quality.
        """
        if self._fragment_store is None:
            return []
        return self._fragment_store.query(exclude_quarantined=False)

    async def run_once(
        self,
        fragments: list[MarketStateFragment] | None = None,
        metadata: Optional[dict[str, Any]] = None,
        run_exit: bool = True,
        run_governance: bool = True,
    ) -> dict[str, PipelineRun]:
        """Execute a single full pipeline cycle.

        Runs main loop, then optionally exit and governance loops.
        If fragments is None, pulls from FragmentStore automatically.

        In SHADOW mode, after the main loop completes, signals are
        extracted from EXEC-VALIDATE output and recorded via
        ShadowExecutionService instead of submitting to a broker.

        Args:
            fragments: MarketStateFragments from Perception. If None,
                reads from FragmentStore.
            metadata: Optional initial metadata.
            run_exit: Whether to run the exit loop after main.
            run_governance: Whether to run the governance loop.

        Returns:
            Dict of loop_type → PipelineRun.
        """
        log = logger.bind(trigger="manual", system_mode=self._system_mode.value)
        log.info("Starting single pipeline run")

        # Pull fragments from store if not provided
        if fragments is None:
            fragments = self._get_fragments_from_store()
            log.info("Fragments loaded from store", count=len(fragments))

        # Inject system mode into metadata for downstream agents
        if metadata is None:
            metadata = {}
        metadata["system_mode"] = self._system_mode.value
        if self._system_mode == SystemMode.SHADOW:
            metadata["capital_tier"] = "SEED"  # Shadow mode = SEED tier

        runs: dict[str, PipelineRun] = {}

        # Main loop
        main_run = await self._orchestrator.run_main_loop(
            fragments=fragments,
            metadata=metadata,
            trigger="manual",
        )
        runs["MAIN"] = main_run
        self._persist_run(main_run)
        beliefs_stored = self._extract_and_store_beliefs(main_run)
        if beliefs_stored:
            log.info("Beliefs stored", count=beliefs_stored)

        # Shadow mode: record signals instead of real execution
        if self._system_mode == SystemMode.SHADOW and self._shadow_svc is not None:
            self._record_shadow_signals(main_run)

        # Exit loop (uses main loop metadata)
        if run_exit:
            exit_run = await self._orchestrator.run_exit_loop(
                metadata=main_run.metadata,
                trigger="manual",
            )
            runs["EXIT"] = exit_run
            self._persist_run(exit_run)

        # Governance loop (uses main loop metadata)
        if run_governance:
            gov_run = await self._orchestrator.run_governance_loop(
                metadata=main_run.metadata,
                trigger="manual",
            )
            runs["GOVERNANCE"] = gov_run
            self._persist_run(gov_run)

        log.info(
            "Single pipeline run complete",
            loops=list(runs.keys()),
            statuses={k: v.status.value for k, v in runs.items()},
            system_mode=self._system_mode.value,
        )
        return runs

    def _record_shadow_signals(self, main_run: PipelineRun) -> None:
        """Extract EXEC-VALIDATE output from main run and record shadow signals."""
        if self._shadow_svc is None:
            return

        validated_proposal = None
        regime_state = None

        for sr in main_run.stage_results:
            if sr.agent_id == "EXEC-VALIDATE" and sr.status == StageStatus.SUCCEEDED:
                validated_proposal = sr.output
            if sr.agent_id == "REGIME-MISMATCH" and sr.status == StageStatus.SUCCEEDED:
                regime_state = sr.output

        if validated_proposal and isinstance(validated_proposal, dict):
            summary = self._shadow_svc.record_signals(
                run_id=main_run.run_id,
                validated_proposal=validated_proposal,
                regime_state=regime_state if isinstance(regime_state, dict) else {},
                metadata=main_run.metadata,
            )
            logger.info(
                "Shadow signals captured",
                run_id=str(main_run.run_id),
                signals=summary.total_signals,
                approved=summary.approved_signals,
            )
        else:
            logger.debug(
                "No EXEC-VALIDATE output found for shadow recording",
                run_id=str(main_run.run_id),
            )

    async def run_continuous(
        self,
        fragment_provider: Any = None,
        interval_seconds: float = 300.0,
        run_exit: bool = True,
        run_governance: bool = True,
    ) -> None:
        """Run the pipeline continuously on a schedule.

        Runs until shutdown is requested via request_shutdown().

        Args:
            fragment_provider: Callable or async callable that returns
                list[MarketStateFragment] for each cycle. If None,
                reads from FragmentStore automatically.
            interval_seconds: Seconds between pipeline runs.
            run_exit: Whether to run exit loop each cycle.
            run_governance: Whether to run governance loop each cycle.
        """
        log = logger.bind(interval=interval_seconds)
        log.info("Starting continuous pipeline")
        self._running = True
        cycle = 0

        try:
            while not self._shutdown_event.is_set():
                cycle += 1
                log.info("Pipeline cycle starting", cycle=cycle)

                try:
                    # Get fragments from provider or store
                    if fragment_provider is not None:
                        if asyncio.iscoroutinefunction(fragment_provider):
                            fragments = await fragment_provider()
                        else:
                            fragments = fragment_provider()
                    else:
                        fragments = None  # run_once will pull from store

                    # Run cycle
                    runs = await self.run_once(
                        fragments=fragments,
                        run_exit=run_exit,
                        run_governance=run_governance,
                    )

                    log.info(
                        "Pipeline cycle complete",
                        cycle=cycle,
                        main_status=runs["MAIN"].status.value,
                    )

                except Exception as e:
                    log.error("Pipeline cycle failed", cycle=cycle, error=str(e))

                # Wait for next cycle or shutdown
                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=interval_seconds,
                    )
                    break  # Shutdown requested
                except asyncio.TimeoutError:
                    pass  # Timeout = next cycle

        finally:
            self._running = False
            log.info("Continuous pipeline stopped", total_cycles=cycle)

    async def run_learning_batch(
        self,
        metadata: dict[str, Any] | None = None,
    ) -> PipelineRun:
        """Execute an offline learning batch.

        Args:
            metadata: Metadata with trade history, belief outcomes, etc.

        Returns:
            PipelineRun for the learning loop.
        """
        log = logger.bind(trigger="batch")
        log.info("Starting learning batch")

        run = await self._orchestrator.run_learning_loop(
            metadata=metadata or {},
            trigger="manual",
        )

        self._persist_run(run)

        log.info(
            "Learning batch complete",
            status=run.status.value,
            stages=len(run.stage_results),
        )
        return run

    def request_shutdown(self) -> None:
        """Request graceful shutdown of continuous pipeline."""
        logger.info("Shutdown requested")
        self._shutdown_event.set()
