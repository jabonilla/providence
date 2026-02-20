"""ProvidenceRunner — scheduler and entry point for pipeline execution.

Wraps the Orchestrator with scheduling, graceful shutdown, and
convenience methods for running single or continuous pipeline loops.
"""

import asyncio
from datetime import datetime, timezone
from typing import Any, Optional

import structlog

from providence.orchestration.models import PipelineRun
from providence.orchestration.orchestrator import Orchestrator
from providence.schemas.market_state import MarketStateFragment

logger = structlog.get_logger()


class ProvidenceRunner:
    """Entry point for running Providence pipeline loops.

    Wraps the Orchestrator with:
      - Single run (manual trigger)
      - Continuous scheduling (asyncio.sleep loop)
      - Offline batch runs (learning)
      - Graceful shutdown
    """

    def __init__(self, orchestrator: Orchestrator) -> None:
        self._orchestrator = orchestrator
        self._shutdown_event = asyncio.Event()
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    async def run_once(
        self,
        fragments: list[MarketStateFragment],
        metadata: Optional[dict[str, Any]] = None,
        run_exit: bool = True,
        run_governance: bool = True,
    ) -> dict[str, PipelineRun]:
        """Execute a single full pipeline cycle.

        Runs main loop, then optionally exit and governance loops.

        Args:
            fragments: MarketStateFragments from Perception.
            metadata: Optional initial metadata.
            run_exit: Whether to run the exit loop after main.
            run_governance: Whether to run the governance loop.

        Returns:
            Dict of loop_type → PipelineRun.
        """
        log = logger.bind(trigger="manual")
        log.info("Starting single pipeline run")

        runs: dict[str, PipelineRun] = {}

        # Main loop
        main_run = await self._orchestrator.run_main_loop(
            fragments=fragments,
            metadata=metadata,
            trigger="manual",
        )
        runs["MAIN"] = main_run

        # Exit loop (uses main loop metadata)
        if run_exit:
            exit_run = await self._orchestrator.run_exit_loop(
                metadata=main_run.metadata,
                trigger="manual",
            )
            runs["EXIT"] = exit_run

        # Governance loop (uses main loop metadata)
        if run_governance:
            gov_run = await self._orchestrator.run_governance_loop(
                metadata=main_run.metadata,
                trigger="manual",
            )
            runs["GOVERNANCE"] = gov_run

        log.info(
            "Single pipeline run complete",
            loops=list(runs.keys()),
            statuses={k: v.status.value for k, v in runs.items()},
        )
        return runs

    async def run_continuous(
        self,
        fragment_provider: Any,
        interval_seconds: float = 300.0,
        run_exit: bool = True,
        run_governance: bool = True,
    ) -> None:
        """Run the pipeline continuously on a schedule.

        Runs until shutdown is requested via request_shutdown().

        Args:
            fragment_provider: Callable or async callable that returns
                list[MarketStateFragment] for each cycle.
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
                    # Get fragments
                    if asyncio.iscoroutinefunction(fragment_provider):
                        fragments = await fragment_provider()
                    else:
                        fragments = fragment_provider()

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
        metadata: dict[str, Any],
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
            metadata=metadata,
            trigger="manual",
        )

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
