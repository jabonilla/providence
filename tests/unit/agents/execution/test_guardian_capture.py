"""Tests for EXEC-GUARDIAN and EXEC-CAPTURE agents.

Tests kill-switch/circuit-breaker checks and trailing stop management
including trim stages, hard giveback, and EXEC-CAPTURE supremacy.

Both agents are FROZEN: zero LLM calls, pure computation.
"""

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from providence.agents.base import AgentContext, AgentStatus
from providence.agents.execution.guardian import (
    ExecGuardian,
    CIRCUIT_BREAKERS,
    check_order,
    check_system_halt,
)
from providence.agents.execution.capture import (
    ExecCapture,
    MAX_TRIM_STAGES,
    TRAIL_PARAMS,
    compute_trailing_stop,
    evaluate_position,
)
from providence.exceptions import AgentProcessingError
from providence.schemas.execution import (
    CaptureDecision,
    CaptureOutput,
    GuardianCheck,
    GuardianVerdict,
    TrailingStopState,
)

NOW = datetime.now(timezone.utc)


def _make_regime(risk_mode: str = "NORMAL") -> dict:
    return {"system_risk_mode": risk_mode}


def _make_order(
    ticker: str = "AAPL",
    weight: float = 0.06,
) -> dict:
    return {
        "order_id": str(uuid4()),
        "ticker": ticker,
        "action": "OPEN_LONG",
        "direction": "LONG",
        "target_weight": weight,
        "confidence": 0.65,
        "source_intent_id": str(uuid4()),
    }


def _make_portfolio(
    drawdown_pct: float = 0.0,
    daily_trades: int = 0,
    turnover_pct: float = 0.0,
) -> dict:
    return {
        "daily_drawdown_pct": drawdown_pct,
        "daily_trade_count": daily_trades,
        "daily_turnover_pct": turnover_pct,
    }


def _make_position(
    ticker: str = "AAPL",
    unrealized_pnl: float = 0.05,
    peak_pnl: float = 0.10,
    expected_return: float = 0.04,
    days_held: int = 10,
    trim_stage: int = 0,
) -> dict:
    return {
        "ticker": ticker,
        "unrealized_pnl": unrealized_pnl,
        "peak_unrealized_pnl": peak_pnl,
        "expected_return": expected_return,
        "days_held": days_held,
        "trim_stage": trim_stage,
    }


def _make_context(**metadata_kwargs) -> AgentContext:
    return AgentContext(
        agent_id="TEST",
        trigger="schedule",
        fragments=[],
        context_window_hash="test-hash-exec-002",
        timestamp=NOW,
        metadata=metadata_kwargs,
    )


# ===========================================================================
# Tests: check_system_halt
# ===========================================================================
class TestCheckSystemHalt:
    """Tests for the check_system_halt function."""

    def test_halted_mode(self):
        halt, reason = check_system_halt("HALTED", {}, CIRCUIT_BREAKERS["HALTED"])
        assert halt is True
        assert "HALTED" in reason

    def test_drawdown_triggers_halt(self):
        halt, reason = check_system_halt(
            "NORMAL",
            {"daily_drawdown_pct": 6.0},
            CIRCUIT_BREAKERS["NORMAL"],
        )
        assert halt is True
        assert "drawdown" in reason.lower()

    def test_trade_count_triggers_halt(self):
        halt, reason = check_system_halt(
            "NORMAL",
            {"daily_trade_count": 55},
            CIRCUIT_BREAKERS["NORMAL"],
        )
        assert halt is True
        assert "trade count" in reason.lower()

    def test_normal_no_halt(self):
        halt, _ = check_system_halt("NORMAL", {}, CIRCUIT_BREAKERS["NORMAL"])
        assert halt is False


# ===========================================================================
# Tests: check_order
# ===========================================================================
class TestCheckOrder:
    """Tests for the check_order function."""

    def test_within_turnover(self):
        approved, _ = check_order(
            _make_order(weight=0.05), {}, CIRCUIT_BREAKERS["NORMAL"], 0.0,
        )
        assert approved is True

    def test_turnover_exceeded(self):
        approved, reason = check_order(
            _make_order(weight=0.10), {}, CIRCUIT_BREAKERS["NORMAL"], 39.0,
        )
        assert approved is False
        assert "turnover" in reason.lower()


# ===========================================================================
# Tests: ExecGuardian agent
# ===========================================================================
class TestExecGuardian:

    @pytest.mark.asyncio
    async def test_process_approved(self):
        agent = ExecGuardian()
        ctx = _make_context(
            routing_plan={"orders": [_make_order()]},
            regime_state=_make_regime("NORMAL"),
            portfolio_state=_make_portfolio(),
        )
        result = await agent.process(ctx)
        assert isinstance(result, GuardianVerdict)
        assert result.system_halt is False
        assert result.approved_count == 1

    @pytest.mark.asyncio
    async def test_halted_blocks_all(self):
        agent = ExecGuardian()
        ctx = _make_context(
            routing_plan={"orders": [_make_order(), _make_order("MSFT")]},
            regime_state=_make_regime("HALTED"),
            portfolio_state=_make_portfolio(),
        )
        result = await agent.process(ctx)
        assert result.system_halt is True
        assert result.halted_count == 2
        assert result.approved_count == 0

    @pytest.mark.asyncio
    async def test_drawdown_halt(self):
        agent = ExecGuardian()
        ctx = _make_context(
            routing_plan={"orders": [_make_order()]},
            regime_state=_make_regime("NORMAL"),
            portfolio_state=_make_portfolio(drawdown_pct=6.0),
        )
        result = await agent.process(ctx)
        assert result.system_halt is True

    @pytest.mark.asyncio
    async def test_empty_orders(self):
        agent = ExecGuardian()
        ctx = _make_context(
            routing_plan={"orders": []},
            regime_state=_make_regime("NORMAL"),
        )
        result = await agent.process(ctx)
        assert result.approved_count == 0

    @pytest.mark.asyncio
    async def test_content_hash(self):
        agent = ExecGuardian()
        ctx = _make_context(
            routing_plan={"orders": [_make_order()]},
            regime_state=_make_regime("NORMAL"),
            portfolio_state=_make_portfolio(),
        )
        result = await agent.process(ctx)
        assert len(result.content_hash) == 64


# ===========================================================================
# Tests: compute_trailing_stop
# ===========================================================================
class TestComputeTrailingStop:
    """Tests for trailing stop state computation."""

    def test_not_activated_below_threshold(self):
        pos = _make_position(peak_pnl=0.06, expected_return=0.04)
        # Needs 2.0x = 0.08 to activate, peak is 0.06
        state = compute_trailing_stop(pos, TRAIL_PARAMS["NORMAL"])
        assert state.is_active is False

    def test_activated_above_threshold(self):
        pos = _make_position(peak_pnl=0.10, expected_return=0.04)
        # Needs 2.0x = 0.08, peak is 0.10 → active
        state = compute_trailing_stop(pos, TRAIL_PARAMS["NORMAL"])
        assert state.is_active is True

    def test_trigger_level_computed(self):
        pos = _make_position(peak_pnl=0.10, expected_return=0.04)
        state = compute_trailing_stop(pos, TRAIL_PARAMS["NORMAL"])
        # Trail 30% → trigger at 0.10 * 0.70 = 0.07
        assert state.trigger_level == pytest.approx(0.07, abs=0.001)

    def test_cautious_tighter_trail(self):
        pos = _make_position(peak_pnl=0.10, expected_return=0.04)
        state = compute_trailing_stop(pos, TRAIL_PARAMS["CAUTIOUS"])
        # Trail 20% → trigger at 0.10 * 0.80 = 0.08
        assert state.trigger_level == pytest.approx(0.08, abs=0.001)

    def test_peak_updated(self):
        pos = _make_position(unrealized_pnl=0.15, peak_pnl=0.10)
        state = compute_trailing_stop(pos, TRAIL_PARAMS["NORMAL"])
        assert state.peak_unrealized_pnl == 0.15  # Updated to current


# ===========================================================================
# Tests: evaluate_position
# ===========================================================================
class TestEvaluatePosition:
    """Tests for position evaluation logic."""

    def test_max_trim_stages_close(self):
        pos = _make_position(trim_stage=3)
        state = compute_trailing_stop(pos, TRAIL_PARAMS["NORMAL"])
        action, _, conf, reason = evaluate_position(pos, state, TRAIL_PARAMS["NORMAL"])
        assert action == "CLOSE"
        assert "Max trim stages" in reason

    def test_hard_giveback_close(self):
        # Peak was 0.10, now at 0.04 → lost 60% > 50% threshold
        pos = _make_position(unrealized_pnl=0.04, peak_pnl=0.10, days_held=10)
        state = compute_trailing_stop(pos, TRAIL_PARAMS["NORMAL"])
        action, _, _, reason = evaluate_position(pos, state, TRAIL_PARAMS["NORMAL"])
        assert action == "CLOSE"
        assert "giveback" in reason.lower()

    def test_min_hold_prevents_exit(self):
        pos = _make_position(days_held=3, peak_pnl=0.10, expected_return=0.04)
        state = compute_trailing_stop(pos, TRAIL_PARAMS["NORMAL"])
        action, _, _, reason = evaluate_position(pos, state, TRAIL_PARAMS["NORMAL"])
        assert action == "HOLD"
        assert "hold period" in reason.lower()

    def test_trailing_stop_breached_trim(self):
        # Peak 0.10, current 0.06, trail 30% → trigger at 0.07
        pos = _make_position(
            unrealized_pnl=0.06, peak_pnl=0.10,
            expected_return=0.04, days_held=10, trim_stage=0,
        )
        state = compute_trailing_stop(pos, TRAIL_PARAMS["NORMAL"])
        action, trim_pct, _, reason = evaluate_position(pos, state, TRAIL_PARAMS["NORMAL"])
        assert action == "TRIM"
        assert trim_pct == 0.30  # Stage 0 default

    def test_trailing_stop_not_breached_hold(self):
        # Peak 0.10, current 0.09, trail 30% → trigger at 0.07
        pos = _make_position(
            unrealized_pnl=0.09, peak_pnl=0.10,
            expected_return=0.04, days_held=10,
        )
        state = compute_trailing_stop(pos, TRAIL_PARAMS["NORMAL"])
        action, _, _, _ = evaluate_position(pos, state, TRAIL_PARAMS["NORMAL"])
        assert action == "HOLD"

    def test_stage_2_full_close(self):
        # Stage 2 trim_pct = 1.0 → CLOSE
        pos = _make_position(
            unrealized_pnl=0.06, peak_pnl=0.10,
            expected_return=0.04, days_held=10, trim_stage=2,
        )
        state = compute_trailing_stop(pos, TRAIL_PARAMS["NORMAL"])
        action, _, _, _ = evaluate_position(pos, state, TRAIL_PARAMS["NORMAL"])
        assert action == "CLOSE"

    def test_default_hold(self):
        # Not activated, within min hold, no giveback
        pos = _make_position(
            unrealized_pnl=0.03, peak_pnl=0.03,
            expected_return=0.04, days_held=10,
        )
        state = compute_trailing_stop(pos, TRAIL_PARAMS["NORMAL"])
        action, _, _, _ = evaluate_position(pos, state, TRAIL_PARAMS["NORMAL"])
        assert action == "HOLD"


# ===========================================================================
# Tests: ExecCapture agent
# ===========================================================================
class TestExecCapture:

    @pytest.mark.asyncio
    async def test_process_hold(self):
        agent = ExecCapture()
        pos = _make_position(
            unrealized_pnl=0.03, peak_pnl=0.03,
            expected_return=0.04, days_held=10,
        )
        ctx = _make_context(
            active_positions=[pos],
            regime_state=_make_regime("NORMAL"),
        )
        result = await agent.process(ctx)
        assert isinstance(result, CaptureOutput)
        assert result.positions_held == 1
        assert result.positions_closed == 0

    @pytest.mark.asyncio
    async def test_process_trim(self):
        agent = ExecCapture()
        pos = _make_position(
            unrealized_pnl=0.06, peak_pnl=0.10,
            expected_return=0.04, days_held=10, trim_stage=0,
        )
        ctx = _make_context(
            active_positions=[pos],
            regime_state=_make_regime("NORMAL"),
        )
        result = await agent.process(ctx)
        assert result.positions_trimmed == 1

    @pytest.mark.asyncio
    async def test_process_close_hard_giveback(self):
        agent = ExecCapture()
        pos = _make_position(
            unrealized_pnl=0.04, peak_pnl=0.10,
            expected_return=0.04, days_held=10,
        )
        ctx = _make_context(
            active_positions=[pos],
            regime_state=_make_regime("NORMAL"),
        )
        result = await agent.process(ctx)
        assert result.positions_closed == 1

    @pytest.mark.asyncio
    async def test_halted_closes_all(self):
        agent = ExecCapture()
        positions = [
            _make_position("AAPL"),
            _make_position("MSFT"),
        ]
        ctx = _make_context(
            active_positions=positions,
            regime_state=_make_regime("HALTED"),
        )
        result = await agent.process(ctx)
        assert result.positions_closed == 2
        assert result.positions_held == 0

    @pytest.mark.asyncio
    async def test_max_trim_stages_mandatory_close(self):
        agent = ExecCapture()
        pos = _make_position(trim_stage=3)
        ctx = _make_context(
            active_positions=[pos],
            regime_state=_make_regime("NORMAL"),
        )
        result = await agent.process(ctx)
        assert result.positions_closed == 1
        assert result.decisions[0].action == "CLOSE"

    @pytest.mark.asyncio
    async def test_empty_positions(self):
        agent = ExecCapture()
        ctx = _make_context(
            active_positions=[],
            regime_state=_make_regime("NORMAL"),
        )
        result = await agent.process(ctx)
        assert len(result.decisions) == 0

    @pytest.mark.asyncio
    async def test_content_hash(self):
        agent = ExecCapture()
        pos = _make_position()
        ctx = _make_context(
            active_positions=[pos],
            regime_state=_make_regime("NORMAL"),
        )
        result = await agent.process(ctx)
        assert len(result.content_hash) == 64

    @pytest.mark.asyncio
    async def test_trim_pct_applies_to_remaining(self):
        """trim_pct applies to REMAINING position, not original."""
        agent = ExecCapture()
        # Stage 0 breached → trim 30% of REMAINING
        pos = _make_position(
            unrealized_pnl=0.06, peak_pnl=0.10,
            expected_return=0.04, days_held=10, trim_stage=0,
        )
        ctx = _make_context(
            active_positions=[pos],
            regime_state=_make_regime("NORMAL"),
        )
        result = await agent.process(ctx)
        decision = result.decisions[0]
        assert decision.action == "TRIM"
        assert decision.trim_pct == 0.30  # 30% of remaining


# ===========================================================================
# Tests: Health and Properties
# ===========================================================================
class TestExecGuardianHealth:
    def test_healthy(self):
        assert ExecGuardian().get_health().status == AgentStatus.HEALTHY

    def test_agent_id(self):
        assert ExecGuardian().agent_id == "EXEC-GUARDIAN"

    def test_consumed_empty(self):
        assert ExecGuardian.CONSUMED_DATA_TYPES == set()


class TestExecCaptureHealth:
    def test_healthy(self):
        assert ExecCapture().get_health().status == AgentStatus.HEALTHY

    def test_agent_id(self):
        assert ExecCapture().agent_id == "EXEC-CAPTURE"

    def test_consumed_empty(self):
        assert ExecCapture.CONSUMED_DATA_TYPES == set()

    def test_supremacy_documented(self):
        """EXEC-CAPTURE docstring should mention supremacy."""
        assert "supremacy" in ExecCapture.__doc__.lower()
