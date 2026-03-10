"""API dependency injection — shared state and service accessors.

All stateful objects (stores, registries, runner) are held in AppState
and injected into route handlers via FastAPI Depends().
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from providence.agents.base import BaseAgent
from providence.config.agent_config import AgentConfigRegistry
from providence.config.watchlist import Watchlist
from providence.orchestration.runner import ProvidenceRunner
from providence.portfolio.order_manager import OrderManager
from providence.portfolio.tracker import PortfolioTracker
from providence.services.health import HealthService
from providence.services.shadow_execution import ShadowSignalStore
from providence.storage.belief_store import BeliefStore
from providence.storage.fragment_store import FragmentStore
from providence.storage.run_store import RunStore


@dataclass
class AppState:
    """Mutable container for all shared application state.

    Created once at startup, injected into route handlers.
    """

    # Core stores
    fragment_store: FragmentStore = field(default_factory=FragmentStore)
    belief_store: BeliefStore = field(default_factory=BeliefStore)
    run_store: RunStore = field(default_factory=RunStore)

    # Agent registry
    agent_registry: dict[str, BaseAgent] = field(default_factory=dict)

    # Shadow mode
    shadow_signal_store: Optional[ShadowSignalStore] = None

    # Portfolio management
    portfolio_tracker: Optional[PortfolioTracker] = None
    order_manager: Optional[OrderManager] = None

    # Services
    health_service: Optional[HealthService] = None
    runner: Optional[ProvidenceRunner] = None

    # Config
    config_registry: Optional[AgentConfigRegistry] = None
    watchlist: Optional[Watchlist] = None

    # Extra metadata
    extra: dict[str, Any] = field(default_factory=dict)


# Singleton — set at startup, read by route handlers
_state: Optional[AppState] = None


def set_state(state: AppState) -> None:
    """Set the global app state (called once at startup)."""
    global _state
    _state = state


def get_state() -> AppState:
    """Get the global app state. Raises if not initialized."""
    if _state is None:
        raise RuntimeError("AppState not initialized — call set_state() at startup")
    return _state
