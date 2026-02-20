"""Tests for the agent factory — bootstrap and registry building."""

import pytest

from providence.agents.base import BaseAgent
from providence.factory import (
    ALL_AGENT_IDS,
    _ADAPTIVE_LLM,
    _FROZEN_NO_ARGS,
    _PERCEPTION_EDGAR,
    _PERCEPTION_FRED,
    _PERCEPTION_POLYGON,
    build_agent_registry,
)


# ===========================================================================
# Agent ID Completeness
# ===========================================================================


class TestAgentIDCompleteness:
    def test_total_agent_count(self):
        """All 31 agents should be accounted for."""
        total = (
            len(_FROZEN_NO_ARGS)
            + len(_ADAPTIVE_LLM)
            + len(_PERCEPTION_POLYGON)
            + len(_PERCEPTION_EDGAR)
            + len(_PERCEPTION_FRED)
        )
        assert total == 31

    def test_all_agent_ids_sorted(self):
        assert ALL_AGENT_IDS == sorted(ALL_AGENT_IDS)
        assert len(ALL_AGENT_IDS) == 31

    def test_no_duplicate_ids(self):
        all_ids = (
            list(_FROZEN_NO_ARGS)
            + list(_ADAPTIVE_LLM)
            + list(_PERCEPTION_POLYGON)
            + list(_PERCEPTION_EDGAR)
            + list(_PERCEPTION_FRED)
        )
        assert len(all_ids) == len(set(all_ids))

    def test_perception_agents_present(self):
        perception = set(_PERCEPTION_POLYGON) | set(_PERCEPTION_EDGAR) | set(_PERCEPTION_FRED)
        expected = {
            "PERCEPT-PRICE",
            "PERCEPT-FILING",
            "PERCEPT-NEWS",
            "PERCEPT-OPTIONS",
            "PERCEPT-CDS",
            "PERCEPT-MACRO",
        }
        assert perception == expected

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
        # 31 total - 8 adaptive - 6 perception = 17 frozen no-args
        assert len(_FROZEN_NO_ARGS) == 17


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
        assert len(registry) == expected_count  # 17 + 8 = 25

    def test_all_agents_are_base_agent(self):
        registry = build_agent_registry(
            skip_perception=True,
            skip_adaptive=False,
        )
        for agent_id, agent in registry.items():
            assert isinstance(agent, BaseAgent), f"{agent_id} is not BaseAgent"
