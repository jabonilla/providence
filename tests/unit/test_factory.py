"""Tests for the agent factory — bootstrap and registry building."""

import pytest

from providence.agents.base import BaseAgent
from providence.factory import (
    ALL_AGENT_IDS,
    _ADAPTIVE_LLM,
    _FROZEN_NO_ARGS,
    _PERCEPTION_ALPHAVANTAGE,
    _PERCEPTION_EDGAR,
    _PERCEPTION_FAMAFRENCH,
    _PERCEPTION_FRED,
    _PERCEPTION_PLAID,
    _PERCEPTION_POLYGON,
    _PERCEPTION_YFINANCE,
    build_agent_registry,
)


# All perception groups, keyed by the client they need.
_PERCEPTION_GROUPS = (
    _PERCEPTION_POLYGON,
    _PERCEPTION_EDGAR,
    _PERCEPTION_FRED,
    _PERCEPTION_YFINANCE,
    _PERCEPTION_ALPHAVANTAGE,
    _PERCEPTION_FAMAFRENCH,
    _PERCEPTION_PLAID,
)


def _perception_ids() -> set[str]:
    """Every perception agent ID across all source-specific groups."""
    return {aid for group in _PERCEPTION_GROUPS for aid in group}


# ===========================================================================
# Agent ID Completeness
# ===========================================================================


class TestAgentIDCompleteness:
    def test_total_agent_count(self):
        """Every agent group is accounted for in ALL_AGENT_IDS."""
        total = (
            len(_FROZEN_NO_ARGS)
            + len(_ADAPTIVE_LLM)
            + len(_perception_ids())
        )
        assert total == len(ALL_AGENT_IDS)

    def test_all_agent_ids_sorted(self):
        assert ALL_AGENT_IDS == sorted(ALL_AGENT_IDS)
        assert len(ALL_AGENT_IDS) == len(set(ALL_AGENT_IDS))
        assert set(ALL_AGENT_IDS) == (
            set(_FROZEN_NO_ARGS) | set(_ADAPTIVE_LLM) | _perception_ids()
        )

    def test_no_duplicate_ids(self):
        all_ids = (
            list(_FROZEN_NO_ARGS)
            + list(_ADAPTIVE_LLM)
            + [aid for group in _PERCEPTION_GROUPS for aid in group]
        )
        assert len(all_ids) == len(set(all_ids))

    def test_perception_agents_present(self):
        expected = {
            "PERCEPT-PRICE",
            "PERCEPT-FILING",
            "PERCEPT-NEWS",
            "PERCEPT-OPTIONS",
            "PERCEPT-CDS",
            "PERCEPT-MACRO",
            "PERCEPT-YFINANCE",
            "PERCEPT-ALPHAVANTAGE",
            "PERCEPT-FACTORS",
            "PERCEPT-FUNDFLOW",
        }
        assert _perception_ids() == expected

    def test_adaptive_agents_present(self):
        expected = {
            "COGNIT-FUNDAMENTAL",
            "COGNIT-MACRO",
            "COGNIT-EVENT",
            "COGNIT-NARRATIVE",
            "COGNIT-CROSSSEC",
            "COGNIT-EXIT",
            "REGIME-NARR",
            "DECIDE-SYNTH",
        }
        assert set(_ADAPTIVE_LLM) == expected

    def test_frozen_no_args_count(self):
        """Frozen no-arg agents are everything that is neither adaptive
        nor perception."""
        expected = len(ALL_AGENT_IDS) - len(_ADAPTIVE_LLM) - len(_perception_ids())
        assert len(_FROZEN_NO_ARGS) == expected


# ===========================================================================
# Frozen Agent Instantiation
# ===========================================================================


class TestFrozenInstantiation:
    def test_all_frozen_agents_instantiate(self):
        """Every frozen agent should instantiate with no args."""
        registry = build_agent_registry(
            skip_perception=True,
            skip_adaptive=True,
        )
        for agent_id in _FROZEN_NO_ARGS:
            assert agent_id in registry, f"{agent_id} not in registry"
            assert isinstance(registry[agent_id], BaseAgent)

    def test_frozen_count(self):
        registry = build_agent_registry(
            skip_perception=True,
            skip_adaptive=True,
        )
        assert len(registry) == len(_FROZEN_NO_ARGS)

    def test_each_frozen_agent_has_correct_id(self):
        registry = build_agent_registry(
            skip_perception=True,
            skip_adaptive=True,
        )
        for agent_id, agent in registry.items():
            assert agent.agent_id == agent_id

    def test_each_frozen_agent_has_health(self):
        registry = build_agent_registry(
            skip_perception=True,
            skip_adaptive=True,
        )
        for agent_id, agent in registry.items():
            health = agent.get_health()
            assert health is not None
            assert hasattr(health, "status")


# ===========================================================================
# Adaptive Agent Instantiation
# ===========================================================================


class TestAdaptiveInstantiation:
    def test_adaptive_agents_instantiate_without_llm(self):
        """Adaptive agents should default to creating their own LLM client."""
        registry = build_agent_registry(
            skip_perception=True,
            skip_adaptive=False,
        )
        for agent_id in _ADAPTIVE_LLM:
            assert agent_id in registry, f"{agent_id} not in registry"

    def test_skip_adaptive(self):
        registry = build_agent_registry(
            skip_perception=True,
            skip_adaptive=True,
        )
        for agent_id in _ADAPTIVE_LLM:
            assert agent_id not in registry


# ===========================================================================
# Perception Agent Instantiation
# ===========================================================================


class TestPerceptionInstantiation:
    def test_skip_perception(self):
        registry = build_agent_registry(
            skip_perception=True,
            skip_adaptive=True,
        )
        all_perception = (
            set(_PERCEPTION_POLYGON)
            | set(_PERCEPTION_EDGAR)
            | set(_PERCEPTION_FRED)
        )
        for agent_id in all_perception:
            assert agent_id not in registry

    def test_perception_skipped_without_clients(self):
        """Perception agents need clients — should be skipped if not provided."""
        registry = build_agent_registry(
            skip_perception=False,
            skip_adaptive=True,
            # No clients provided
        )
        all_perception = (
            set(_PERCEPTION_POLYGON)
            | set(_PERCEPTION_EDGAR)
            | set(_PERCEPTION_FRED)
        )
        for agent_id in all_perception:
            assert agent_id not in registry


# ===========================================================================
# Agent Filter
# ===========================================================================


class TestAgentFilter:
    def test_filter_single_agent(self):
        registry = build_agent_registry(
            skip_perception=True,
            skip_adaptive=True,
            agent_filter={"EXEC-VALIDATE"},
        )
        assert "EXEC-VALIDATE" in registry
        assert len(registry) == 1

    def test_filter_multiple_agents(self):
        target = {"EXEC-VALIDATE", "EXEC-ROUTER", "GOVERN-CAPITAL"}
        registry = build_agent_registry(
            skip_perception=True,
            skip_adaptive=True,
            agent_filter=target,
        )
        assert set(registry.keys()) == target

    def test_filter_includes_adaptive_despite_skip(self):
        """agent_filter should override skip_adaptive for included agents."""
        registry = build_agent_registry(
            skip_perception=True,
            skip_adaptive=True,
            agent_filter={"COGNIT-FUNDAMENTAL", "EXEC-VALIDATE"},
        )
        # COGNIT-FUNDAMENTAL is adaptive but should be included via filter
        assert "COGNIT-FUNDAMENTAL" in registry
        assert "EXEC-VALIDATE" in registry

    def test_filter_nonexistent_agent(self):
        registry = build_agent_registry(
            skip_perception=True,
            skip_adaptive=True,
            agent_filter={"FAKE-AGENT"},
        )
        assert "FAKE-AGENT" not in registry
        assert len(registry) == 0


# ===========================================================================
# Full Bootstrap (frozen + adaptive, no perception)
# ===========================================================================


class TestFullBootstrap:
    def test_frozen_plus_adaptive_count(self):
        registry = build_agent_registry(
            skip_perception=True,
            skip_adaptive=False,
        )
        expected_count = len(_FROZEN_NO_ARGS) + len(_ADAPTIVE_LLM)
        assert len(registry) == expected_count

    def test_all_agents_are_base_agent(self):
        registry = build_agent_registry(
            skip_perception=True,
            skip_adaptive=False,
        )
        for agent_id, agent in registry.items():
            assert isinstance(agent, BaseAgent), f"{agent_id} is not BaseAgent"
