"""Agent factory — instantiates all 35 Providence agents.

Maps agent IDs to their concrete classes and handles dependency
injection (API clients for perception, LLM clients for adaptive agents).
Frozen agents are instantiated with zero external dependencies.

API clients auto-discover credentials from environment variables:
  POLYGON_API_KEY, EDGAR_USER_AGENT, FRED_API_KEY, ANTHROPIC_API_KEY
"""

from __future__ import annotations

from typing import Any

import structlog

from providence.agents.base import BaseAgent

# --- Perception ---
from providence.agents.perception.price import PerceptPrice
from providence.agents.perception.filing import PerceptFiling
from providence.agents.perception.news import PerceptNews
from providence.agents.perception.options import PerceptOptions
from providence.agents.perception.cds import PerceptCds
from providence.agents.perception.macro import PerceptMacro

# --- Cognition ---
from providence.agents.cognition import (
    CognitCrossSec,
    CognitEvent,
    CognitFundamental,
    CognitMacro,
    CognitNarrative,
    CognitTechnical,
)

# --- Regime ---
from providence.agents.regime import (
    RegimeMismatch,
    RegimeNarr,
    RegimeSector,
    RegimeStat,
)

# --- Decision ---
from providence.agents.decision import DecideOptim, DecideSynth

# --- Execution ---
from providence.agents.execution import (
    ExecCapture,
    ExecGuardian,
    ExecRouter,
    ExecValidate,
)

# --- Exit ---
from providence.agents.exit import (
    CognitExit,
    InvalidMon,
    RenewMon,
    ShadowExit,
    ThesisRenew,
)

# --- Learning ---
from providence.agents.learning import (
    LearnAttrib,
    LearnBacktest,
    LearnCalib,
    LearnRetrain,
)

# --- Governance ---
from providence.agents.governance import (
    GovernCapital,
    GovernMaturity,
    GovernOversight,
    GovernPolicy,
)

# --- Infrastructure ---
from providence.infra.polygon_client import PolygonClient
from providence.infra.edgar_client import EdgarClient
from providence.infra.fred_client import FredClient
from providence.infra.llm_client import LLMClient

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Agent ID → class mapping
# ---------------------------------------------------------------------------

# Frozen agents that take ZERO constructor arguments
_FROZEN_NO_ARGS: dict[str, type[BaseAgent]] = {
    "COGNIT-TECHNICAL": CognitTechnical,
    "REGIME-STAT": RegimeStat,
    "REGIME-SECTOR": RegimeSector,
    "REGIME-MISMATCH": RegimeMismatch,
    "DECIDE-OPTIM": DecideOptim,
    "EXEC-VALIDATE": ExecValidate,
    "EXEC-ROUTER": ExecRouter,
    "EXEC-GUARDIAN": ExecGuardian,
    "EXEC-CAPTURE": ExecCapture,
    "INVALID-MON": InvalidMon,
    "THESIS-RENEW": ThesisRenew,
    "SHADOW-EXIT": ShadowExit,
    "RENEW-MON": RenewMon,
    "LEARN-ATTRIB": LearnAttrib,
    "LEARN-CALIB": LearnCalib,
    "LEARN-RETRAIN": LearnRetrain,
    "LEARN-BACKTEST": LearnBacktest,
    "GOVERN-CAPITAL": GovernCapital,
    "GOVERN-MATURITY": GovernMaturity,
    "GOVERN-OVERSIGHT": GovernOversight,
    "GOVERN-POLICY": GovernPolicy,
}

# Adaptive agents that need an LLM client
_ADAPTIVE_LLM: dict[str, type[BaseAgent]] = {
    "COGNIT-FUNDAMENTAL": CognitFundamental,
    "COGNIT-MACRO": CognitMacro,
    "COGNIT-EVENT": CognitEvent,
    "COGNIT-NARRATIVE": CognitNarrative,
    "COGNIT-CROSSSEC": CognitCrossSec,
    "COGNIT-EXIT": CognitExit,
    "REGIME-NARR": RegimeNarr,
    "DECIDE-SYNTH": DecideSynth,
}

# Perception agents that need specific API clients
_PERCEPTION_POLYGON: dict[str, type[BaseAgent]] = {
    "PERCEPT-PRICE": PerceptPrice,
    "PERCEPT-NEWS": PerceptNews,
    "PERCEPT-OPTIONS": PerceptOptions,
}

_PERCEPTION_EDGAR: dict[str, type[BaseAgent]] = {
    "PERCEPT-FILING": PerceptFiling,
}

_PERCEPTION_FRED: dict[str, type[BaseAgent]] = {
    "PERCEPT-CDS": PerceptCds,
    "PERCEPT-MACRO": PerceptMacro,
}

ALL_AGENT_IDS = sorted(
    list(_FROZEN_NO_ARGS)
    + list(_ADAPTIVE_LLM)
    + list(_PERCEPTION_POLYGON)
    + list(_PERCEPTION_EDGAR)
    + list(_PERCEPTION_FRED)
)


def build_agent_registry(
    *,
    polygon_client: PolygonClient | None = None,
    edgar_client: EdgarClient | None = None,
    fred_client: FredClient | None = None,
    llm_client: LLMClient | None = None,
    skip_perception: bool = False,
    skip_adaptive: bool = False,
    agent_filter: set[str] | None = None,
) -> dict[str, BaseAgent]:
    """Build a complete agent registry mapping agent_id → agent instance.

    Parameters
    ----------
    polygon_client:
        Shared Polygon.io client for perception agents. Required unless
        skip_perception=True.
    edgar_client:
        Shared SEC EDGAR client. Required unless skip_perception=True.
    fred_client:
        Shared FRED client. Required unless skip_perception=True.
    llm_client:
        Shared LLM client for adaptive agents. If None, each adaptive
        agent will create its own AnthropicClient (default behaviour).
    skip_perception:
        If True, skip perception agents (useful for backtesting or
        when running from cached fragments).
    skip_adaptive:
        If True, skip adaptive (LLM) agents. Useful for testing the
        frozen pipeline in isolation.
    agent_filter:
        If provided, only instantiate agents whose IDs are in this set.
        Overrides skip_perception and skip_adaptive for included agents.

    Returns
    -------
    dict[str, BaseAgent]
        Registry suitable for passing to Orchestrator.__init__.
    """
    registry: dict[str, BaseAgent] = {}

    def _should_include(agent_id: str) -> bool:
        if agent_filter is not None:
            return agent_id in agent_filter
        return True

    # 1. Frozen agents (zero args)
    for agent_id, agent_cls in _FROZEN_NO_ARGS.items():
        if _should_include(agent_id):
            try:
                registry[agent_id] = agent_cls()
                logger.debug("Agent instantiated", agent_id=agent_id, type="frozen")
            except Exception as exc:
                logger.error(
                    "Failed to instantiate frozen agent",
                    agent_id=agent_id,
                    error=str(exc),
                )

    # 2. Adaptive agents (LLM client)
    if not skip_adaptive or agent_filter:
        for agent_id, agent_cls in _ADAPTIVE_LLM.items():
            if not _should_include(agent_id):
                continue
            if skip_adaptive and (agent_filter is None or agent_id not in agent_filter):
                continue
            try:
                kwargs: dict[str, Any] = {}
                if llm_client is not None:
                    kwargs["llm_client"] = llm_client
                registry[agent_id] = agent_cls(**kwargs)
                logger.debug(
                    "Agent instantiated",
                    agent_id=agent_id,
                    type="adaptive",
                    shared_llm=llm_client is not None,
                )
            except Exception as exc:
                logger.error(
                    "Failed to instantiate adaptive agent",
                    agent_id=agent_id,
                    error=str(exc),
                )

    # 3. Perception agents (external API clients)
    if not skip_perception or agent_filter:
        # Polygon-based
        for agent_id, agent_cls in _PERCEPTION_POLYGON.items():
            if not _should_include(agent_id):
                continue
            if skip_perception and (agent_filter is None or agent_id not in agent_filter):
                continue
            if polygon_client is None:
                logger.warning(
                    "Skipping agent — no PolygonClient provided",
                    agent_id=agent_id,
                )
                continue
            try:
                registry[agent_id] = agent_cls(polygon_client)
                logger.debug(
                    "Agent instantiated", agent_id=agent_id, type="perception"
                )
            except Exception as exc:
                logger.error(
                    "Failed to instantiate perception agent",
                    agent_id=agent_id,
                    error=str(exc),
                )

        # EDGAR-based
        for agent_id, agent_cls in _PERCEPTION_EDGAR.items():
            if not _should_include(agent_id):
                continue
            if skip_perception and (agent_filter is None or agent_id not in agent_filter):
                continue
            if edgar_client is None:
                logger.warning(
                    "Skipping agent — no EdgarClient provided",
                    agent_id=agent_id,
                )
                continue
            try:
                registry[agent_id] = agent_cls(edgar_client)
                logger.debug(
                    "Agent instantiated", agent_id=agent_id, type="perception"
                )
            except Exception as exc:
                logger.error(
                    "Failed to instantiate perception agent",
                    agent_id=agent_id,
                    error=str(exc),
                )

        # FRED-based
        for agent_id, agent_cls in _PERCEPTION_FRED.items():
            if not _should_include(agent_id):
                continue
            if skip_perception and (agent_filter is None or agent_id not in agent_filter):
                continue
            if fred_client is None:
                logger.warning(
                    "Skipping agent — no FredClient provided",
                    agent_id=agent_id,
                )
                continue
            try:
                registry[agent_id] = agent_cls(fred_client)
                logger.debug(
                    "Agent instantiated", agent_id=agent_id, type="perception"
                )
            except Exception as exc:
                logger.error(
                    "Failed to instantiate perception agent",
                    agent_id=agent_id,
                    error=str(exc),
                )

    logger.info(
        "Agent registry built",
        total=len(registry),
        frozen=sum(1 for a in registry if a in _FROZEN_NO_ARGS),
        adaptive=sum(1 for a in registry if a in _ADAPTIVE_LLM),
        perception=sum(
            1
            for a in registry
            if a in _PERCEPTION_POLYGON
            or a in _PERCEPTION_EDGAR
            or a in _PERCEPTION_FRED
        ),
    )
    return registry


def build_agent_registry_from_env(
    *,
    skip_perception: bool = False,
    skip_adaptive: bool = False,
    agent_filter: set[str] | None = None,
) -> dict[str, BaseAgent]:
    """Build agent registry with API clients auto-created from environment.

    Clients are only instantiated when their env vars are set.
    Missing keys cause the corresponding agents to be skipped
    (with a warning), not a hard failure.

    Environment variables:
        POLYGON_API_KEY: Polygon.io API key
        EDGAR_USER_AGENT: SEC EDGAR user agent string
        FRED_API_KEY: FRED API key
        ANTHROPIC_API_KEY: Anthropic (Claude) API key

    Returns:
        Agent registry ready for Orchestrator.
    """
    import os

    polygon_client = None
    edgar_client = None
    fred_client = None
    llm_client = None

    if not skip_perception:
        if os.environ.get("POLYGON_API_KEY"):
            polygon_client = PolygonClient()
            logger.info("PolygonClient initialized from environment")
        if os.environ.get("EDGAR_USER_AGENT"):
            edgar_client = EdgarClient()
            logger.info("EdgarClient initialized from environment")
        if os.environ.get("FRED_API_KEY"):
            fred_client = FredClient()
            logger.info("FredClient initialized from environment")

    if not skip_adaptive:
        if os.environ.get("ANTHROPIC_API_KEY"):
            from providence.infra.llm_client import AnthropicClient
            llm_client = AnthropicClient()
            logger.info("AnthropicClient initialized from environment")

    return build_agent_registry(
        polygon_client=polygon_client,
        edgar_client=edgar_client,
        fred_client=fred_client,
        llm_client=llm_client,
        skip_perception=skip_perception,
        skip_adaptive=skip_adaptive,
        agent_filter=agent_filter,
    )
