"""Standalone API server entry point.

Usage:
    python -m providence.api.server [--host HOST] [--port PORT] [--data-dir DIR]
    python -m providence.api.server --skip-perception --skip-adaptive

Creates all stores, builds agent registry, wires up the runner,
and starts the FastAPI server via uvicorn.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

import structlog
import uvicorn

from providence.api.app import create_app
from providence.api.deps import AppState

logger = structlog.get_logger()


def build_state(
    *,
    data_dir: Path | None = None,
    skip_perception: bool = False,
    skip_adaptive: bool = False,
) -> AppState:
    """Build the full AppState with stores, agents, and services."""
    from providence.config.agent_config import AgentConfigRegistry
    from providence.config.watchlist import Watchlist
    from providence.factory import build_agent_registry
    from providence.orchestration.orchestrator import Orchestrator
    from providence.orchestration.runner import ProvidenceRunner
    from providence.portfolio.order_manager import OrderManager
    from providence.portfolio.tracker import PortfolioTracker
    from providence.services.context_svc import ContextService
    from providence.schemas.enums import SystemMode
    from providence.services.health import HealthService
    from providence.services.shadow_execution import ShadowSignalStore
    from providence.storage.belief_store import BeliefStore
    from providence.storage.fragment_store import FragmentStore
    from providence.storage.run_store import RunStore

    # Stores
    frag_path = data_dir / "fragments.jsonl" if data_dir else None
    belief_path = data_dir / "beliefs.jsonl" if data_dir else None
    run_path = data_dir / "runs.jsonl" if data_dir else None
    portfolio_path = data_dir / "portfolio.jsonl" if data_dir else None
    orders_path = data_dir / "orders.jsonl" if data_dir else None

    if data_dir:
        data_dir.mkdir(parents=True, exist_ok=True)

    fragment_store = FragmentStore(persist_path=frag_path)
    belief_store = BeliefStore(persist_path=belief_path)
    run_store = RunStore(persist_path=run_path)

    # Agent registry
    registry = build_agent_registry(
        skip_perception=skip_perception,
        skip_adaptive=skip_adaptive,
    )

    # Config
    config_path = Path(__file__).parent.parent / "config" / "agents.yaml"
    config_registry = AgentConfigRegistry.from_yaml(config_path)

    watchlist_path = Path(__file__).parent.parent / "config" / "watchlist.yaml"
    watchlist = None
    if watchlist_path.exists():
        watchlist = Watchlist.from_yaml(watchlist_path)

    # Services
    context_service = ContextService(config_registry=config_registry)
    orchestrator = Orchestrator(
        agent_registry=registry,
        context_service=context_service,
        config_registry=config_registry,
    )
    # Shadow signal store (always create for API visibility)
    shadow_path = data_dir / "shadow_signals.jsonl" if data_dir else None
    shadow_signal_store = ShadowSignalStore(persist_path=shadow_path)

    # Portfolio tracking
    portfolio_tracker = PortfolioTracker(persist_path=portfolio_path)
    order_manager = OrderManager(persist_path=orders_path)

    runner = ProvidenceRunner(
        orchestrator=orchestrator,
        fragment_store=fragment_store,
        belief_store=belief_store,
        run_store=run_store,
        shadow_signal_store=shadow_signal_store,
        system_mode=SystemMode.SHADOW,
    )
    health_service = HealthService(
        agent_registry=registry,
        run_store=run_store,
    )

    return AppState(
        fragment_store=fragment_store,
        belief_store=belief_store,
        run_store=run_store,
        shadow_signal_store=shadow_signal_store,
        agent_registry=registry,
        health_service=health_service,
        runner=runner,
        config_registry=config_registry,
        watchlist=watchlist,
        portfolio_tracker=portfolio_tracker,
        order_manager=order_manager,
    )


def main() -> None:
    """Parse args and launch the API server."""
    parser = argparse.ArgumentParser(description="Providence REST API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    parser.add_argument("--data-dir", type=Path, default=None, help="Persistent storage directory")
    parser.add_argument("--skip-perception", action="store_true", help="Skip perception agents")
    parser.add_argument("--skip-adaptive", action="store_true", help="Skip adaptive (LLM) agents")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (dev only)")
    parser.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error"])
    args = parser.parse_args()

    # Build state
    logger.info(
        "Building application state",
        data_dir=str(args.data_dir) if args.data_dir else None,
        skip_perception=args.skip_perception,
        skip_adaptive=args.skip_adaptive,
    )
    state = build_state(
        data_dir=args.data_dir,
        skip_perception=args.skip_perception,
        skip_adaptive=args.skip_adaptive,
    )

    # Create app
    app = create_app(state=state)

    # Launch
    logger.info("Starting Providence API", host=args.host, port=args.port)
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
