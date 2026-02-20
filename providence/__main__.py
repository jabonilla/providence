"""Providence CLI — entry point for running the hedge fund pipeline.

Usage:
    python -m providence run-once          Run a single main loop cycle
    python -m providence run-continuous    Run continuously with scheduling
    python -m providence run-learning      Run an offline learning batch
    python -m providence health            Print agent health status
    python -m providence list-agents       List all registered agents
"""

from __future__ import annotations

import argparse
import asyncio
import signal
import sys
from pathlib import Path

import structlog

from providence.config.agent_config import AgentConfigRegistry
from providence.factory import ALL_AGENT_IDS, build_agent_registry
from providence.orchestration.orchestrator import Orchestrator
from providence.orchestration.runner import ProvidenceRunner
from providence.services.context_svc import ContextService

logger = structlog.get_logger()

DEFAULT_CONFIG_PATH = Path("config/agents.yaml")
DEFAULT_INTERVAL = 300  # 5 minutes


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="providence",
        description="Providence AI-Native Hedge Fund — Pipeline Runner",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to agents.yaml config file",
    )
    parser.add_argument(
        "--skip-perception",
        action="store_true",
        help="Skip perception agents (run from cached fragments)",
    )
    parser.add_argument(
        "--skip-adaptive",
        action="store_true",
        help="Skip adaptive (LLM) agents (frozen pipeline only)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Default stage timeout in seconds (default: 120)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # run-once
    run_once = subparsers.add_parser(
        "run-once", help="Run a single main loop cycle"
    )
    run_once.add_argument(
        "--with-exit",
        action="store_true",
        help="Also run exit loop after main loop",
    )
    run_once.add_argument(
        "--with-governance",
        action="store_true",
        help="Also run governance loop after main loop",
    )

    # run-continuous
    continuous = subparsers.add_parser(
        "run-continuous", help="Run continuously with scheduling"
    )
    continuous.add_argument(
        "--interval",
        type=int,
        default=DEFAULT_INTERVAL,
        help=f"Seconds between cycles (default: {DEFAULT_INTERVAL})",
    )

    # run-learning
    subparsers.add_parser(
        "run-learning", help="Run an offline learning batch"
    )

    # health
    subparsers.add_parser("health", help="Print agent health status")

    # list-agents
    subparsers.add_parser("list-agents", help="List all registered agents")

    return parser.parse_args()


def _configure_logging(level: str) -> None:
    """Configure structlog with the given level."""
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(structlog, level, structlog.INFO)
        ),
    )


def _build_system(
    args: argparse.Namespace,
) -> tuple[ProvidenceRunner, dict]:
    """Build the full Providence system from CLI args.

    Returns (runner, agent_registry) tuple.
    """
    # Load agent configs
    config_registry = AgentConfigRegistry.from_yaml(args.config)

    # Build agent registry
    # NOTE: In production, pass real Polygon/EDGAR/FRED clients here.
    # For now, skip_perception=True by default since we don't have
    # live API credentials wired up yet.
    registry = build_agent_registry(
        skip_perception=args.skip_perception,
        skip_adaptive=args.skip_adaptive,
    )

    # Build context service and orchestrator
    context_service = ContextService(config_registry)
    orchestrator = Orchestrator(
        agent_registry=registry,
        context_service=context_service,
        config_registry=config_registry,
        default_timeout=args.timeout,
    )

    runner = ProvidenceRunner(orchestrator)
    return runner, registry


async def _cmd_run_once(args: argparse.Namespace) -> int:
    """Execute a single pipeline cycle."""
    runner, _ = _build_system(args)
    logger.info(
        "Running single cycle",
        with_exit=args.with_exit,
        with_governance=args.with_governance,
    )

    run = await runner.run_once(
        fragments=[],
        metadata={},
        run_exit=args.with_exit,
        run_governance=args.with_governance,
    )

    logger.info(
        "Cycle complete",
        status=run.status.value,
        stages=len(run.stage_results),
        succeeded=run.succeeded_count,
        failed=run.failed_count,
        skipped=run.skipped_count,
        duration_ms=run.total_duration_ms,
    )
    return 0 if run.status.value in ("SUCCEEDED", "PARTIAL_FAILURE") else 1


async def _cmd_run_continuous(args: argparse.Namespace) -> int:
    """Run continuously with graceful shutdown."""
    runner, _ = _build_system(args)

    # Set up graceful shutdown on SIGINT/SIGTERM
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, runner.request_shutdown)

    logger.info("Starting continuous mode", interval_seconds=args.interval)

    async def _fragment_provider():
        """Placeholder — returns empty fragments for now."""
        return []

    await runner.run_continuous(
        fragment_provider=_fragment_provider,
        interval_seconds=args.interval,
    )
    logger.info("Shutdown complete")
    return 0


async def _cmd_run_learning(args: argparse.Namespace) -> int:
    """Run an offline learning batch."""
    runner, _ = _build_system(args)
    logger.info("Running learning batch")

    run = await runner.run_learning_batch(metadata={})

    logger.info(
        "Learning batch complete",
        status=run.status.value,
        succeeded=run.succeeded_count,
        failed=run.failed_count,
        duration_ms=run.total_duration_ms,
    )
    return 0 if run.status.value in ("SUCCEEDED", "PARTIAL_FAILURE") else 1


def _cmd_health(args: argparse.Namespace) -> int:
    """Print health status of all agents."""
    _, registry = _build_system(args)

    print(f"\n{'Agent ID':<25} {'Type':<12} {'Status':<12} {'Version'}")
    print("-" * 65)

    for agent_id in sorted(registry):
        agent = registry[agent_id]
        health = agent.get_health()
        print(
            f"{agent_id:<25} {agent.agent_type:<12} "
            f"{health.status:<12} {agent.version}"
        )

    print(f"\nTotal: {len(registry)} agents registered")
    print(f"Expected: {len(ALL_AGENT_IDS)} agents")
    missing = set(ALL_AGENT_IDS) - set(registry)
    if missing:
        print(f"Missing: {', '.join(sorted(missing))}")
    return 0


def _cmd_list_agents(_args: argparse.Namespace) -> int:
    """List all known agent IDs."""
    print(f"\nProvidence — {len(ALL_AGENT_IDS)} agents\n")

    subsystems = {
        "Perception": [a for a in ALL_AGENT_IDS if a.startswith("PERCEPT-")],
        "Cognition": [a for a in ALL_AGENT_IDS if a.startswith("COGNIT-")],
        "Regime": [a for a in ALL_AGENT_IDS if a.startswith("REGIME-")],
        "Decision": [a for a in ALL_AGENT_IDS if a.startswith("DECIDE-")],
        "Execution": [a for a in ALL_AGENT_IDS if a.startswith("EXEC-")],
        "Exit": [
            a
            for a in ALL_AGENT_IDS
            if a.startswith(("INVALID-", "THESIS-", "SHADOW-", "RENEW-"))
        ],
        "Learning": [a for a in ALL_AGENT_IDS if a.startswith("LEARN-")],
        "Governance": [a for a in ALL_AGENT_IDS if a.startswith("GOVERN-")],
    }

    for name, agents in subsystems.items():
        print(f"  {name} ({len(agents)}):")
        for agent_id in agents:
            print(f"    {agent_id}")
        print()

    return 0


def main() -> int:
    args = _parse_args()
    _configure_logging(args.log_level)

    if args.command == "list-agents":
        return _cmd_list_agents(args)
    elif args.command == "health":
        return _cmd_health(args)
    elif args.command == "run-once":
        return asyncio.run(_cmd_run_once(args))
    elif args.command == "run-continuous":
        return asyncio.run(_cmd_run_continuous(args))
    elif args.command == "run-learning":
        return asyncio.run(_cmd_run_learning(args))
    else:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
