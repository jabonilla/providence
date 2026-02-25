"""Perception scheduler — cycles through watchlist tickers.

Runs perception agents for each ticker in the watchlist, managing
rate limits and scheduling. Produces MarketStateFragments that feed
into the main analysis pipeline.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Optional

import structlog

from providence.agents.base import AgentContext, BaseAgent
from providence.config.watchlist import Watchlist, WatchlistEntry
from providence.storage.fragment_store import FragmentStore

logger = structlog.get_logger()


class PerceptionScheduler:
    """Schedules perception agent runs across the watchlist.

    Manages concurrent and sequential runs through the watchlist,
    with configurable rate limiting between tickers and agents.

    Modes:
    - run_full_sweep(): Process all tickers sequentially
    - run_priority_sweep(): Process only priority-N tickers
    - run_single(ticker): Process one ticker

    Each run calls all enabled perception agents for the ticker(s)
    and stores fragments in FragmentStore.
    """

    def __init__(
        self,
        perception_agents: dict[str, BaseAgent],  # agent_id -> agent
        fragment_store: FragmentStore,
        watchlist: Watchlist,
        *,
        inter_ticker_delay: float = 2.0,  # seconds between tickers (rate limiting)
        inter_agent_delay: float = 1.0,    # seconds between agents per ticker
        max_concurrent_agents: int = 3,    # max concurrent agent runs per ticker
    ):
        """Initialize the PerceptionScheduler.

        Args:
            perception_agents: Dict mapping agent_id to BaseAgent instances
            fragment_store: FragmentStore for persisting output fragments
            watchlist: Watchlist with tickers to process
            inter_ticker_delay: Seconds to wait between processing tickers
            inter_agent_delay: Seconds to wait between running agents on same ticker
            max_concurrent_agents: Maximum concurrent agent tasks per ticker
        """
        self._agents = perception_agents
        self._fragment_store = fragment_store
        self._watchlist = watchlist
        self._inter_ticker_delay = inter_ticker_delay
        self._inter_agent_delay = inter_agent_delay
        self._max_concurrent_agents = max_concurrent_agents

        # Tracking for last sweep
        self._last_sweep_time: Optional[datetime] = None
        self._last_sweep_stats: dict[str, Any] = {}

    async def run_full_sweep(self) -> dict[str, Any]:
        """Run all perception agents for all enabled tickers.

        Processes tickers sequentially, with rate limiting between them.
        For each ticker, runs all enabled perception agents.

        Returns:
            Sweep summary with structure:
            {
                "sweep_start": ISO timestamp,
                "sweep_end": ISO timestamp,
                "duration_seconds": float,
                "tickers_processed": int,
                "fragments_created": int,
                "agents_run": int,
                "errors": int,
                "per_ticker_results": {
                    "AAPL": {
                        "agents_run": 2,
                        "fragments": 2,
                        "errors": 0,
                    },
                    ...
                }
            }
        """
        sweep_start = datetime.now(timezone.utc)
        logger.info(
            "Starting full watchlist sweep",
            watchlist_name=self._watchlist.name,
            ticker_count=len(self._watchlist.tickers),
            agent_count=len(self._agents),
        )

        per_ticker_results = {}
        total_fragments = 0
        total_agents_run = 0
        total_errors = 0

        for ticker in self._watchlist.tickers:
            ticker_result = await self._process_ticker(ticker)

            per_ticker_results[ticker] = ticker_result
            total_fragments += ticker_result.get("fragments", 0)
            total_agents_run += ticker_result.get("agents_run", 0)
            total_errors += ticker_result.get("errors", 0)

            # Rate limiting between tickers
            if ticker != self._watchlist.tickers[-1]:  # Don't wait after last ticker
                await asyncio.sleep(self._inter_ticker_delay)

        sweep_end = datetime.now(timezone.utc)
        duration = (sweep_end - sweep_start).total_seconds()

        result = {
            "sweep_start": sweep_start.isoformat(),
            "sweep_end": sweep_end.isoformat(),
            "duration_seconds": duration,
            "tickers_processed": len(self._watchlist.tickers),
            "fragments_created": total_fragments,
            "agents_run": total_agents_run,
            "errors": total_errors,
            "per_ticker_results": per_ticker_results,
        }

        self._last_sweep_time = sweep_end
        self._last_sweep_stats = result

        logger.info(
            "Full sweep complete",
            duration_seconds=duration,
            fragments_created=total_fragments,
            agents_run=total_agents_run,
            errors=total_errors,
        )

        return result

    async def run_priority_sweep(self, max_priority: int = 1) -> dict[str, Any]:
        """Run perception for high-priority tickers only.

        Args:
            max_priority: Include tickers with priority <= this value.
                         1 = highest priority, 2 = medium, 3 = low

        Returns:
            Sweep summary (same structure as run_full_sweep)
        """
        priority_entries = self._watchlist.by_priority(max_priority)
        priority_tickers = [e.ticker for e in priority_entries]

        logger.info(
            "Starting priority sweep",
            max_priority=max_priority,
            ticker_count=len(priority_tickers),
        )

        sweep_start = datetime.now(timezone.utc)
        per_ticker_results = {}
        total_fragments = 0
        total_agents_run = 0
        total_errors = 0

        for ticker in priority_tickers:
            entry = next((e for e in priority_entries if e.ticker == ticker), None)
            ticker_result = await self._process_ticker(ticker, entry)

            per_ticker_results[ticker] = ticker_result
            total_fragments += ticker_result.get("fragments", 0)
            total_agents_run += ticker_result.get("agents_run", 0)
            total_errors += ticker_result.get("errors", 0)

            # Rate limiting between tickers
            if ticker != priority_tickers[-1]:
                await asyncio.sleep(self._inter_ticker_delay)

        sweep_end = datetime.now(timezone.utc)
        duration = (sweep_end - sweep_start).total_seconds()

        result = {
            "sweep_start": sweep_start.isoformat(),
            "sweep_end": sweep_end.isoformat(),
            "duration_seconds": duration,
            "tickers_processed": len(priority_tickers),
            "fragments_created": total_fragments,
            "agents_run": total_agents_run,
            "errors": total_errors,
            "per_ticker_results": per_ticker_results,
        }

        self._last_sweep_time = sweep_end
        self._last_sweep_stats = result

        logger.info(
            "Priority sweep complete",
            max_priority=max_priority,
            duration_seconds=duration,
            fragments_created=total_fragments,
        )

        return result

    async def run_single(self, ticker: str) -> dict[str, Any]:
        """Run all perception agents for a single ticker.

        Args:
            ticker: Stock ticker to process

        Returns:
            Result dict with structure:
            {
                "ticker": str,
                "agents_run": int,
                "fragments": int,
                "errors": int,
                "duration_seconds": float,
                "agent_results": {
                    "agent_id": {
                        "success": bool,
                        "fragments": int,
                        "error": str or None,
                    },
                    ...
                }
            }
        """
        logger.info("Running single ticker sweep", ticker=ticker)

        run_start = datetime.now(timezone.utc)

        # Find entry in watchlist
        entry = next(
            (e for e in self._watchlist.entries if e.ticker == ticker),
            None,
        )

        result = await self._process_ticker(ticker, entry)
        duration = (datetime.now(timezone.utc) - run_start).total_seconds()

        logger.info(
            "Single ticker sweep complete",
            ticker=ticker,
            duration_seconds=duration,
            fragments=result.get("fragments", 0),
        )

        return result

    async def _process_ticker(
        self,
        ticker: str,
        entry: Optional[WatchlistEntry] = None,
    ) -> dict[str, Any]:
        """Run all perception agents for one ticker.

        Creates AgentContext with ticker in metadata, runs each agent,
        stores resulting fragments.

        Args:
            ticker: Stock ticker
            entry: Optional WatchlistEntry with metadata

        Returns:
            Result dict with structure:
            {
                "ticker": str,
                "agents_run": int,
                "fragments": int,
                "errors": int,
                "duration_seconds": float,
                "agent_results": {
                    "agent_id": {
                        "success": bool,
                        "fragments": int,
                        "error": str or None,
                    },
                    ...
                }
            }
        """
        ticker_start = datetime.now(timezone.utc)
        agent_results = {}
        total_fragments = 0
        errors = 0

        # Build metadata from entry
        metadata = {
            "ticker": ticker,
            "trigger": "schedule",
        }
        if entry:
            metadata["sector"] = entry.sector
            metadata["priority"] = entry.priority
            metadata["tags"] = list(entry.tags)

        # Prepare agent tasks
        tasks = []
        agent_ids = list(self._agents.keys())

        logger.debug(
            "Processing ticker with agents",
            ticker=ticker,
            agent_count=len(agent_ids),
        )

        for agent_id in agent_ids:
            agent = self._agents[agent_id]
            task = self._run_agent_for_ticker(
                agent_id,
                agent,
                ticker,
                metadata,
                agent_results,
            )
            tasks.append(task)

            # Rate limiting between agent starts
            if agent_id != agent_ids[-1]:  # Don't wait after last agent
                await asyncio.sleep(self._inter_agent_delay)

        # Run agents concurrently with semaphore
        semaphore = asyncio.Semaphore(self._max_concurrent_agents)

        async def sem_task(task):
            async with semaphore:
                return await task

        try:
            results = await asyncio.gather(*[sem_task(t) for t in tasks], return_exceptions=True)

            for result in results:
                if isinstance(result, Exception):
                    logger.error("Agent task raised exception", error=str(result))
                    errors += 1
                elif result:
                    total_fragments += result.get("fragments", 0)
                    if result.get("error"):
                        errors += 1

        except Exception as e:
            logger.error(
                "Error running concurrent agent tasks",
                error=str(e),
            )
            errors += 1

        ticker_duration = (datetime.now(timezone.utc) - ticker_start).total_seconds()

        return {
            "ticker": ticker,
            "agents_run": len(agent_ids),
            "fragments": total_fragments,
            "errors": errors,
            "duration_seconds": ticker_duration,
            "agent_results": agent_results,
        }

    async def _run_agent_for_ticker(
        self,
        agent_id: str,
        agent: BaseAgent,
        ticker: str,
        metadata: dict[str, Any],
        results_dict: dict[str, Any],
    ) -> dict[str, Any]:
        """Run a single perception agent for one ticker.

        Args:
            agent_id: ID of the agent to run
            agent: BaseAgent instance
            ticker: Stock ticker
            metadata: Metadata dict with ticker info
            results_dict: Dict to populate with results

        Returns:
            Result dict to accumulate stats
        """
        try:
            # Create AgentContext for this agent
            # Note: In a real system, this would be assembled by CONTEXT-SVC
            # For now, we create a minimal context
            context = AgentContext(
                agent_id=agent_id,
                trigger="schedule",
                fragments=[],  # Empty for now; would be filled by CONTEXT-SVC
                context_window_hash="",  # Would be computed by CONTEXT-SVC
                timestamp=datetime.now(timezone.utc),
                metadata=metadata,
            )

            logger.debug(
                "Running perception agent",
                agent_id=agent_id,
                ticker=ticker,
            )

            # Run agent
            agent_start = datetime.now(timezone.utc)
            try:
                output = await agent.process(context)
            except AttributeError:
                # Agent might not be async
                output = agent.process(context)

            agent_duration = (datetime.now(timezone.utc) - agent_start).total_seconds()

            # Store output
            fragment_count = 0
            if output:
                # Output should be a list of MarketStateFragments
                if isinstance(output, list):
                    fragment_count = self._fragment_store.append_many(output)
                else:
                    # Single fragment
                    fragment_count = 1 if self._fragment_store.append(output) else 0

            result = {
                "success": True,
                "fragments": fragment_count,
                "error": None,
                "duration_seconds": agent_duration,
            }

            results_dict[agent_id] = result

            logger.debug(
                "Agent run complete",
                agent_id=agent_id,
                ticker=ticker,
                fragments=fragment_count,
                duration_seconds=agent_duration,
            )

            return result

        except Exception as e:
            logger.error(
                "Error running perception agent",
                agent_id=agent_id,
                ticker=ticker,
                error=str(e),
            )

            result = {
                "success": False,
                "fragments": 0,
                "error": str(e),
            }

            results_dict[agent_id] = result
            return result

    @property
    def last_sweep_time(self) -> Optional[datetime]:
        """Timestamp of the last completed sweep."""
        return self._last_sweep_time

    @property
    def sweep_stats(self) -> dict[str, Any]:
        """Statistics from the last completed sweep.

        Returns dict with keys:
        - sweep_start: ISO timestamp
        - sweep_end: ISO timestamp
        - duration_seconds: float
        - tickers_processed: int
        - fragments_created: int
        - agents_run: int
        - errors: int
        - per_ticker_results: dict[str, dict]
        """
        return self._last_sweep_stats or {
            "sweep_start": None,
            "sweep_end": None,
            "duration_seconds": 0.0,
            "tickers_processed": 0,
            "fragments_created": 0,
            "agents_run": 0,
            "errors": 0,
            "per_ticker_results": {},
        }

    def get_watchlist_summary(self) -> dict[str, Any]:
        """Get summary of the current watchlist configuration.

        Returns:
            Dict with watchlist metadata
        """
        by_sector = self._watchlist.by_sector()
        enabled_count = len(self._watchlist.enabled_entries)

        return {
            "name": self._watchlist.name,
            "total_tickers": len(self._watchlist.entries),
            "enabled_tickers": enabled_count,
            "max_positions": self._watchlist.max_positions,
            "sector_distribution": {
                sector: len(entries)
                for sector, entries in by_sector.items()
            },
            "priority_distribution": {
                priority: len(self._watchlist.by_priority(priority))
                for priority in [1, 2, 3]
            },
        }
