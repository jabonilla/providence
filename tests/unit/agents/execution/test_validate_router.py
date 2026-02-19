"""Tests for EXEC-VALIDATE and EXEC-ROUTER agents.

Tests pre-trade validation (constraints, position limits, sector caps)
and order routing (strategy selection, urgency, slippage).

Both agents are FROZEN: zero LLM calls, pure computation.
"""

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from providence.agents.base import AgentContext, AgentStatus
from providence.agents.execution.validate import (
    ExecValidate,
    VALIDATION_LIMITS,
    validate_position,
)
from providence.agents.execution.router import (
    ExecRouter,
    STRATEGY_RULES,
    compute_max_slippage,
    determine_strategy,
    determine_urgency,
)
from providence.exceptions import AgentProcessingError
from providence.schemas.enums import Action, Direction
from providence.schemas.execution import (
    RoutedOrder,
    RoutingPlan,
    ValidatedProposal,
    ValidationResult,
)

NOW = datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_position(
    ticker: str = "AAPL",
    action: str = "OPEN_LONG",
    direction: str = "LONG",
    weight: float = 0.06,
    confidence: float = 0.60,
    sector: str = "Information Technology",
    time_horizon: int = 60,
) -> dict:
    return {
        "ticker": ticker,
        "action": action,
        "direction": direction,
        "target_weight": weight,
        "confidence": confidence,
        "source_intent_id": str(uuid4()),
        "time_horizon_days": time_horizon,
        "sector": sector,
    }


def _make_proposal(positions: list[dict] | None = None) -> dict:
    return {"proposals": positions or []}


def _make_regime(risk_mode: str = "NORMAL") -> dict:
    return {"system_risk_mode": risk_mode, "statistical_regime": "LOW_VOL_TRENDING"}


def _make_context(
    proposal: dict | None = None,
    regime: dict | None = None,
    validated: dict | None = None,
) -> AgentContext:
    metadata = {}
    if proposal is not None:
        metadata["proposal"] = proposal
    if regime is not None:
        metadata["regime_state"] = regime
    if validated is not None:
        metadata["validated_proposal"] = validated
    return AgentContext(
        agent_id="TEST",
        trigger="schedule",
        fragments=[],
        context_window_hash="test-hash-exec-001",
        timestamp=NOW,
        metadata=metadata,
    )


# ===========================================================================
# Tests: validate_position
# ===========================================================================
class TestValidatePosition:
    """Tests for the validate_position pure function."""

    def test_valid_position(self):
        pos = _make_position()
        limits = VALIDATION_LIMITS["NORMAL"]
        approved, reasons, weight = validate_position(pos, limits, {}, 0.0)
        assert approved is True
        assert weight > 0

    def test_low_confidence_rejected(self):
        pos = _make_position(confidence=0.10)
        limits = VALIDATION_LIMITS["NORMAL"]
        approved, reasons, _ = validate_position(pos, limits, {}, 0.0)
        assert approved is False
        assert any("Confidence" in r for r in reasons)

    def test_weight_clamped_to_limit(self):
        pos = _make_position(weight=0.15)
        limits = VALIDATION_LIMITS["NORMAL"]
        approved, reasons, weight = validate_position(pos, limits, {}, 0.0)
        assert approved is True
        assert weight <= limits["max_position_weight"]

    def test_gross_exposure_limit(self):
        pos = _make_position(weight=0.06)
        limits = VALIDATION_LIMITS["NORMAL"]
        # Already at gross exposure limit
        approved, reasons, _ = validate_position(pos, limits, {}, 1.60)
        assert approved is False
        assert any("Gross exposure" in r for r in reasons)

    def test_sector_concentration_limit(self):
        pos = _make_position(sector="Technology")
        limits = VALIDATION_LIMITS["NORMAL"]
        sector_totals = {"Technology": 0.35}
        approved, reasons, _ = validate_position(pos, limits, sector_totals, 0.0)
        assert approved is False
        assert any("Sector" in r or "sector" in r for r in reasons)

    def test_missing_ticker_rejected(self):
        pos = _make_position()
        pos["ticker"] = ""
        limits = VALIDATION_LIMITS["NORMAL"]
        approved, reasons, _ = validate_position(pos, limits, {}, 0.0)
        assert approved is False

    def test_invalid_action_rejected(self):
        pos = _make_position()
        pos["action"] = "BUY"
        limits = VALIDATION_LIMITS["NORMAL"]
        approved, reasons, _ = validate_position(pos, limits, {}, 0.0)
        assert approved is False
        assert any("Invalid action" in r for r in reasons)


# ===========================================================================
# Tests: ExecValidate agent
# ===========================================================================
class TestExecValidate:
    """Integration tests for ExecValidate."""

    @pytest.mark.asyncio
    async def test_process_single_approved(self):
        agent = ExecValidate()
        ctx = _make_context(
            proposal=_make_proposal([_make_position()]),
            regime=_make_regime("NORMAL"),
        )
        result = await agent.process(ctx)
        assert isinstance(result, ValidatedProposal)
        assert result.approved_count == 1
        assert result.rejected_count == 0

    @pytest.mark.asyncio
    async def test_process_multiple(self):
        agent = ExecValidate()
        ctx = _make_context(
            proposal=_make_proposal([
                _make_position("AAPL"),
                _make_position("JPM", sector="Financials"),
            ]),
            regime=_make_regime("NORMAL"),
        )
        result = await agent.process(ctx)
        assert result.approved_count == 2

    @pytest.mark.asyncio
    async def test_halted_rejects_all(self):
        agent = ExecValidate()
        ctx = _make_context(
            proposal=_make_proposal([_make_position()]),
            regime=_make_regime("HALTED"),
        )
        result = await agent.process(ctx)
        assert result.approved_count == 0
        assert result.rejected_count == 1

    @pytest.mark.asyncio
    async def test_defensive_tighter_limits(self):
        agent = ExecValidate()
        pos = _make_position(weight=0.07, confidence=0.28)
        # NORMAL would approve (min_confidence=0.20)
        ctx_normal = _make_context(
            proposal=_make_proposal([pos]),
            regime=_make_regime("NORMAL"),
        )
        result_normal = await agent.process(ctx_normal)
        # DEFENSIVE might reject (min_confidence=0.30)
        ctx_def = _make_context(
            proposal=_make_proposal([pos]),
            regime=_make_regime("DEFENSIVE"),
        )
        result_def = await agent.process(ctx_def)
        assert result_normal.approved_count >= result_def.approved_count

    @pytest.mark.asyncio
    async def test_empty_proposal(self):
        agent = ExecValidate()
        ctx = _make_context(
            proposal=_make_proposal([]),
            regime=_make_regime("NORMAL"),
        )
        result = await agent.process(ctx)
        assert result.approved_count == 0
        assert result.rejected_count == 0

    @pytest.mark.asyncio
    async def test_content_hash(self):
        agent = ExecValidate()
        ctx = _make_context(
            proposal=_make_proposal([_make_position()]),
            regime=_make_regime("NORMAL"),
        )
        result = await agent.process(ctx)
        assert len(result.content_hash) == 64


# ===========================================================================
# Tests: determine_strategy / urgency / slippage
# ===========================================================================
class TestRoutingFunctions:
    """Tests for routing pure functions."""

    def test_close_action_market(self):
        assert determine_strategy("CLOSE", 0.70, 60, STRATEGY_RULES["NORMAL"]) == "MARKET"

    def test_short_horizon_market(self):
        assert determine_strategy("OPEN_LONG", 0.50, 3, STRATEGY_RULES["NORMAL"]) == "MARKET"

    def test_high_confidence_limit(self):
        strat = determine_strategy("OPEN_LONG", 0.80, 60, STRATEGY_RULES["NORMAL"])
        assert strat == "LIMIT"

    def test_default_twap(self):
        strat = determine_strategy("OPEN_LONG", 0.50, 60, STRATEGY_RULES["NORMAL"])
        assert strat == "TWAP"

    def test_close_urgency_immediate(self):
        assert determine_urgency("CLOSE", 0.50, 60) == "IMMEDIATE"

    def test_short_horizon_high(self):
        assert determine_urgency("OPEN_LONG", 0.50, 2) == "HIGH"

    def test_long_horizon_low(self):
        assert determine_urgency("OPEN_LONG", 0.40, 90) == "LOW"

    def test_default_urgency_normal(self):
        assert determine_urgency("OPEN_LONG", 0.50, 30) == "NORMAL"

    def test_close_slippage_doubled(self):
        slippage = compute_max_slippage("CLOSE", 0.50, 50)
        assert slippage == 100  # 2x base

    def test_high_confidence_tighter_slippage(self):
        slippage = compute_max_slippage("OPEN_LONG", 0.80, 50)
        assert slippage < 50


# ===========================================================================
# Tests: ExecRouter agent
# ===========================================================================
class TestExecRouter:
    """Integration tests for ExecRouter."""

    @pytest.mark.asyncio
    async def test_process_approved_orders(self):
        agent = ExecRouter()
        validated = {
            "results": [
                {
                    "ticker": "AAPL",
                    "action": "OPEN_LONG",
                    "direction": "LONG",
                    "confidence": 0.65,
                    "adjusted_weight": 0.06,
                    "approved": True,
                    "source_intent_id": str(uuid4()),
                    "time_horizon_days": 60,
                },
            ],
        }
        ctx = _make_context(validated=validated, regime=_make_regime("NORMAL"))
        result = await agent.process(ctx)
        assert isinstance(result, RoutingPlan)
        assert result.total_orders == 1
        assert result.orders[0].ticker == "AAPL"

    @pytest.mark.asyncio
    async def test_rejected_not_routed(self):
        agent = ExecRouter()
        validated = {
            "results": [
                {"ticker": "AAPL", "approved": False, "action": "OPEN_LONG"},
            ],
        }
        ctx = _make_context(validated=validated, regime=_make_regime("NORMAL"))
        result = await agent.process(ctx)
        assert result.total_orders == 0

    @pytest.mark.asyncio
    async def test_defensive_vwap_strategy(self):
        agent = ExecRouter()
        validated = {
            "results": [
                {
                    "ticker": "AAPL",
                    "action": "OPEN_LONG",
                    "direction": "LONG",
                    "confidence": 0.50,
                    "adjusted_weight": 0.04,
                    "approved": True,
                    "source_intent_id": str(uuid4()),
                    "time_horizon_days": 60,
                },
            ],
        }
        ctx = _make_context(validated=validated, regime=_make_regime("DEFENSIVE"))
        result = await agent.process(ctx)
        assert result.orders[0].execution_strategy == "VWAP"

    @pytest.mark.asyncio
    async def test_content_hash(self):
        agent = ExecRouter()
        validated = {
            "results": [
                {
                    "ticker": "AAPL",
                    "action": "OPEN_LONG",
                    "direction": "LONG",
                    "confidence": 0.65,
                    "adjusted_weight": 0.06,
                    "approved": True,
                    "source_intent_id": str(uuid4()),
                },
            ],
        }
        ctx = _make_context(validated=validated, regime=_make_regime("NORMAL"))
        result = await agent.process(ctx)
        assert len(result.content_hash) == 64


# ===========================================================================
# Tests: Health and Properties for both agents
# ===========================================================================
class TestExecValidateHealth:
    def test_healthy(self):
        agent = ExecValidate()
        assert agent.get_health().status == AgentStatus.HEALTHY

    def test_degraded(self):
        agent = ExecValidate()
        agent._error_count_24h = 5
        assert agent.get_health().status == AgentStatus.DEGRADED

    def test_agent_id(self):
        assert ExecValidate().agent_id == "EXEC-VALIDATE"

    def test_agent_type(self):
        assert ExecValidate().agent_type == "execution"

    def test_consumed_empty(self):
        assert ExecValidate.CONSUMED_DATA_TYPES == set()


class TestExecRouterHealth:
    def test_healthy(self):
        agent = ExecRouter()
        assert agent.get_health().status == AgentStatus.HEALTHY

    def test_agent_id(self):
        assert ExecRouter().agent_id == "EXEC-ROUTER"

    def test_consumed_empty(self):
        assert ExecRouter.CONSUMED_DATA_TYPES == set()
