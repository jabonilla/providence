"""Tests for EXEC-VALIDATE shadow mode and capital tier enforcement."""

import asyncio
from uuid import uuid4

import pytest

from providence.agents.base import AgentContext
from providence.agents.execution.validate import ExecValidate
from providence.schemas.enums import SystemMode


def _make_context(
    positions: list[dict],
    risk_mode: str = "NORMAL",
    capital_tier: str = "SEED",
    system_mode: str = "SHADOW",
) -> AgentContext:
    """Build an AgentContext with proposal and regime in metadata."""
    return AgentContext(
        agent_id="EXEC-VALIDATE",
        context_window_hash="test-hash",
        fragments=[],
        metadata={
            "proposal": {
                "proposals": positions,
            },
            "regime_state": {
                "system_risk_mode": risk_mode,
            },
            "capital_tier": capital_tier,
            "system_mode": system_mode,
        },
    )


class TestExecValidateShadowMode:
    """Tests for capital tier enforcement in EXEC-VALIDATE."""

    @pytest.fixture
    def agent(self):
        return ExecValidate()

    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_seed_tier_shadow_mode_approves(self, agent):
        """In SHADOW mode with SEED tier, validation still runs normally
        (shadow signals are recorded, not blocked)."""
        positions = [{
            "ticker": "AAPL",
            "action": "OPEN_LONG",
            "direction": "LONG",
            "target_weight": 0.05,
            "confidence": 0.70,
            "sector": "Technology",
        }]
        ctx = _make_context(positions, capital_tier="SEED", system_mode="SHADOW")
        result = self._run(agent.process(ctx))
        assert result.approved_count == 1

    def test_seed_tier_live_mode_blocks(self, agent):
        """In LIVE mode with SEED tier, all orders are blocked (HALTED)."""
        positions = [{
            "ticker": "AAPL",
            "action": "OPEN_LONG",
            "direction": "LONG",
            "target_weight": 0.05,
            "confidence": 0.70,
            "sector": "Technology",
        }]
        ctx = _make_context(positions, capital_tier="SEED", system_mode="LIVE")
        result = self._run(agent.process(ctx))
        # HALTED mode: min_confidence = 1.0, max_weight = 0.0 → rejected
        assert result.approved_count == 0
        assert result.rejected_count == 1
        assert result.risk_mode_applied == "HALTED"

    def test_growth_tier_live_mode_allows(self, agent):
        """In LIVE mode with GROWTH tier, normal validation applies."""
        positions = [{
            "ticker": "AAPL",
            "action": "OPEN_LONG",
            "direction": "LONG",
            "target_weight": 0.05,
            "confidence": 0.70,
            "sector": "Technology",
        }]
        ctx = _make_context(positions, capital_tier="GROWTH", system_mode="LIVE")
        result = self._run(agent.process(ctx))
        assert result.approved_count == 1

    def test_paper_mode_seed_tier_allows(self, agent):
        """In PAPER mode with SEED tier, validation runs normally
        (paper trading is allowed for SEED)."""
        positions = [{
            "ticker": "AAPL",
            "action": "OPEN_LONG",
            "direction": "LONG",
            "target_weight": 0.05,
            "confidence": 0.70,
            "sector": "Technology",
        }]
        ctx = _make_context(positions, capital_tier="SEED", system_mode="PAPER")
        result = self._run(agent.process(ctx))
        assert result.approved_count == 1

    def test_invalid_system_mode_defaults_shadow(self, agent):
        """Invalid system mode string falls back to SHADOW."""
        positions = [{
            "ticker": "AAPL",
            "action": "OPEN_LONG",
            "direction": "LONG",
            "target_weight": 0.05,
            "confidence": 0.70,
            "sector": "Technology",
        }]
        ctx = _make_context(positions, capital_tier="SEED", system_mode="INVALID")
        result = self._run(agent.process(ctx))
        # SHADOW mode with SEED = normal validation (not halted)
        assert result.approved_count == 1
