"""Tests for LEARN-RETRAIN and LEARN-BACKTEST agents.

Tests cover:
  - LEARN-RETRAIN: Priority determination, degradation, suggestion generation
  - LEARN-BACKTEST: Period metrics, profit factor, sub-period partitioning
"""

import math
from datetime import datetime, timedelta, timezone

import pytest

from providence.agents.base import AgentContext, AgentStatus
from providence.agents.learning.retrain import (
    ADAPTIVE_AGENTS,
    CRITICAL_BRIER,
    CRITICAL_HIT_RATE,
    HIGH_DEGRADATION_PCT,
    LOW_DEGRADATION_PCT,
    MEDIUM_CAL_ERROR,
    LearnRetrain,
    compute_degradation,
    determine_priority,
    evaluate_agent,
    suggest_changes,
)
from providence.agents.learning.backtest import (
    DEFAULT_PERIOD_DAYS,
    TRADING_DAYS_PER_YEAR,
    LearnBacktest,
    compute_period_metrics,
    compute_profit_factor,
)
from providence.schemas.learning import (
    BacktestOutput,
    BacktestPeriod,
    RetrainOutput,
    RetrainRecommendation,
)

NOW = datetime.now(timezone.utc)
BT_START = datetime(2024, 1, 1, tzinfo=timezone.utc)
BT_END = datetime(2024, 6, 30, tzinfo=timezone.utc)


def _make_context(**metadata_kwargs) -> AgentContext:
    return AgentContext(
        agent_id="TEST",
        trigger="manual",
        fragments=[],
        context_window_hash="test-hash-learn-002",
        timestamp=NOW,
        metadata=metadata_kwargs,
    )


def _make_attribution(
    agent_id: str = "COGNIT-FUNDAMENTAL",
    hit_rate: float = 0.65,
    avg_return_bps: float = 25.0,
    sharpe_contribution: float = 0.15,
) -> dict:
    return {
        "agent_id": agent_id,
        "hit_rate": hit_rate,
        "avg_return_bps": avg_return_bps,
        "sharpe_contribution": sharpe_contribution,
    }


def _make_calibration(
    agent_id: str = "COGNIT-FUNDAMENTAL",
    overall_brier_score: float = 0.15,
    overall_calibration_error: float = 0.05,
    is_overconfident: bool = False,
) -> dict:
    return {
        "agent_id": agent_id,
        "overall_brier_score": overall_brier_score,
        "overall_calibration_error": overall_calibration_error,
        "is_overconfident": is_overconfident,
    }


def _make_trade(
    realized_pnl_bps: float = 50.0,
    holding_days: int = 10,
    exit_timestamp: str | None = None,
) -> dict:
    if exit_timestamp is None:
        exit_timestamp = datetime(2024, 3, 15, tzinfo=timezone.utc).isoformat()
    return {
        "realized_pnl_bps": realized_pnl_bps,
        "holding_days": holding_days,
        "exit_timestamp": exit_timestamp,
    }


def _make_regime(
    timestamp: str | None = None,
    statistical_regime: str = "LOW_VOL_TRENDING",
) -> dict:
    if timestamp is None:
        timestamp = datetime(2024, 3, 15, tzinfo=timezone.utc).isoformat()
    return {
        "timestamp": timestamp,
        "statistical_regime": statistical_regime,
    }


# ===========================================================================
# compute_degradation Tests
# ===========================================================================

class TestComputeDegradation:
    def test_no_degradation(self):
        result = compute_degradation(
            "AGENT-A",
            {"hit_rate": 0.70},
            {"AGENT-A": {"hit_rate": 0.60}},
        )
        assert result == 0.0  # Current > baseline → no degradation

    def test_positive_degradation(self):
        result = compute_degradation(
            "AGENT-A",
            {"hit_rate": 0.40},
            {"AGENT-A": {"hit_rate": 0.80}},
        )
        assert result == 50.0  # (0.80-0.40)/0.80 * 100

    def test_missing_baseline(self):
        result = compute_degradation("AGENT-X", {"hit_rate": 0.50}, {})
        assert result == 0.0

    def test_zero_baseline(self):
        result = compute_degradation(
            "AGENT-A",
            {"hit_rate": 0.50},
            {"AGENT-A": {"hit_rate": 0.0}},
        )
        assert result == 0.0

    def test_non_dict_baseline(self):
        result = compute_degradation(
            "AGENT-A",
            {"hit_rate": 0.50},
            {"AGENT-A": "invalid"},
        )
        assert result == 0.0


# ===========================================================================
# determine_priority Tests
# ===========================================================================

class TestDeterminePriority:
    def test_critical(self):
        # Low hit rate AND high Brier → CRITICAL
        assert determine_priority(0.30, 0.40, 10.0, 0.05) == "CRITICAL"

    def test_high(self):
        # High degradation → HIGH
        assert determine_priority(0.60, 0.10, 25.0, 0.05) == "HIGH"

    def test_medium(self):
        # High calibration error → MEDIUM
        assert determine_priority(0.60, 0.10, 5.0, 0.15) == "MEDIUM"

    def test_low(self):
        # Everything within bounds → LOW
        assert determine_priority(0.60, 0.10, 3.0, 0.05) == "LOW"

    def test_critical_takes_precedence(self):
        # CRITICAL check comes first even with high degradation
        assert determine_priority(0.30, 0.40, 50.0, 0.20) == "CRITICAL"

    def test_high_over_medium(self):
        assert determine_priority(0.50, 0.10, 25.0, 0.15) == "HIGH"


# ===========================================================================
# suggest_changes Tests
# ===========================================================================

class TestSuggestChanges:
    def test_low_hit_rate(self):
        suggestions = suggest_changes(0.30, 0.10, 0.05, False)
        assert any("hit rate" in s.lower() for s in suggestions)

    def test_overconfident(self):
        suggestions = suggest_changes(0.60, 0.10, 0.15, True)
        assert any("confidence dampening" in s.lower() for s in suggestions)

    def test_high_brier(self):
        suggestions = suggest_changes(0.60, 0.30, 0.05, False)
        assert any("brier" in s.lower() for s in suggestions)

    def test_underconfident(self):
        suggestions = suggest_changes(0.60, 0.10, -0.10, False)
        assert any("underconfident" in s.lower() for s in suggestions)

    def test_no_issues(self):
        suggestions = suggest_changes(0.60, 0.10, 0.02, False)
        assert any("acceptable" in s.lower() for s in suggestions)


# ===========================================================================
# evaluate_agent Tests
# ===========================================================================

class TestEvaluateAgent:
    def test_needs_retrain_critical(self):
        result = evaluate_agent(
            "COGNIT-FUNDAMENTAL",
            {"hit_rate": 0.30},
            {"overall_brier_score": 0.40, "overall_calibration_error": 0.05, "is_overconfident": False},
            {},
        )
        assert result["needs_retrain"] is True
        assert result["priority"] == "CRITICAL"
        assert result["shadow_mode_required"] is True

    def test_no_retrain_needed(self):
        result = evaluate_agent(
            "COGNIT-FUNDAMENTAL",
            {"hit_rate": 0.70},
            {"overall_brier_score": 0.10, "overall_calibration_error": 0.03, "is_overconfident": False},
            {"COGNIT-FUNDAMENTAL": {"hit_rate": 0.65}},
        )
        assert result["needs_retrain"] is False
        assert result["priority"] == "LOW"

    def test_adaptive_agent_gets_prompt_suggestions(self):
        result = evaluate_agent(
            "COGNIT-EXIT",  # ADAPTIVE agent
            {"hit_rate": 0.30},
            {"overall_brier_score": 0.40, "overall_calibration_error": 0.20, "is_overconfident": True},
            {},
        )
        assert not any("FROZEN" in s for s in result["suggested_changes"])

    def test_frozen_agent_no_prompt_changes(self):
        result = evaluate_agent(
            "SOME-FROZEN-AGENT",
            {"hit_rate": 0.30},
            {"overall_brier_score": 0.40, "overall_calibration_error": 0.20, "is_overconfident": True},
            {},
        )
        assert any("FROZEN" in s for s in result["suggested_changes"])

    def test_degradation_triggers_retrain(self):
        result = evaluate_agent(
            "COGNIT-FUNDAMENTAL",
            {"hit_rate": 0.50},
            {"overall_brier_score": 0.15, "overall_calibration_error": 0.03, "is_overconfident": False},
            {"COGNIT-FUNDAMENTAL": {"hit_rate": 0.80}},  # 37.5% degradation
        )
        assert result["needs_retrain"] is True
        assert result["performance_degradation_pct"] > 20.0


# ===========================================================================
# LearnRetrain Integration Tests
# ===========================================================================

class TestLearnRetrain:
    @pytest.mark.asyncio
    async def test_process_basic(self):
        agent = LearnRetrain()
        ctx = _make_context(
            attribution_results=[
                _make_attribution("COGNIT-FUNDAMENTAL", hit_rate=0.65),
                _make_attribution("COGNIT-EXIT", hit_rate=0.30),
            ],
            calibration_results=[
                _make_calibration("COGNIT-FUNDAMENTAL", overall_brier_score=0.10),
                _make_calibration("COGNIT-EXIT", overall_brier_score=0.40),
            ],
            baseline_metrics={},
        )
        result = await agent.process(ctx)
        assert isinstance(result, RetrainOutput)
        assert result.total_agents_evaluated == 2
        assert result.agents_needing_retrain >= 1  # COGNIT-EXIT should need retrain

    @pytest.mark.asyncio
    async def test_process_empty(self):
        agent = LearnRetrain()
        ctx = _make_context(attribution_results=[], calibration_results=[])
        result = await agent.process(ctx)
        assert result.total_agents_evaluated == 0
        assert result.agents_needing_retrain == 0

    @pytest.mark.asyncio
    async def test_content_hash(self):
        agent = LearnRetrain()
        ctx = _make_context(
            attribution_results=[_make_attribution()],
            calibration_results=[_make_calibration()],
        )
        result = await agent.process(ctx)
        assert len(result.content_hash) == 64

    @pytest.mark.asyncio
    async def test_shadow_mode_always_true(self):
        agent = LearnRetrain()
        ctx = _make_context(
            attribution_results=[
                _make_attribution("A", hit_rate=0.30),
            ],
            calibration_results=[
                _make_calibration("A", overall_brier_score=0.40),
            ],
        )
        result = await agent.process(ctx)
        for rec in result.recommendations:
            assert rec.shadow_mode_required is True

    @pytest.mark.asyncio
    async def test_union_of_agents(self):
        """Agents in attribution but not calibration (and vice versa) still evaluated."""
        agent = LearnRetrain()
        ctx = _make_context(
            attribution_results=[_make_attribution("AGENT-A")],
            calibration_results=[_make_calibration("AGENT-B")],
        )
        result = await agent.process(ctx)
        assert result.total_agents_evaluated == 2
        agent_ids = {r.agent_id for r in result.recommendations}
        assert "AGENT-A" in agent_ids
        assert "AGENT-B" in agent_ids

    @pytest.mark.asyncio
    async def test_sorted_output(self):
        agent = LearnRetrain()
        ctx = _make_context(
            attribution_results=[
                _make_attribution("Z-AGENT"),
                _make_attribution("A-AGENT"),
            ],
            calibration_results=[],
        )
        result = await agent.process(ctx)
        agent_ids = [r.agent_id for r in result.recommendations]
        assert agent_ids == sorted(agent_ids)


class TestLearnRetrainHealth:
    def test_agent_id(self):
        assert LearnRetrain().agent_id == "LEARN-RETRAIN"

    def test_healthy(self):
        assert LearnRetrain().get_health().status == AgentStatus.HEALTHY

    def test_degraded(self):
        agent = LearnRetrain()
        agent._error_count_24h = 5
        assert agent.get_health().status == AgentStatus.DEGRADED

    def test_unhealthy(self):
        agent = LearnRetrain()
        agent._error_count_24h = 15
        assert agent.get_health().status == AgentStatus.UNHEALTHY


class TestRetrainSchemas:
    def test_recommendation_frozen(self):
        rec = RetrainRecommendation(agent_id="TEST", needs_retrain=True, priority="HIGH")
        with pytest.raises(Exception):
            rec.needs_retrain = False

    def test_shadow_mode_default_true(self):
        rec = RetrainRecommendation(agent_id="TEST")
        assert rec.shadow_mode_required is True

    def test_retrain_output_hash(self):
        ro = RetrainOutput(
            agent_id="TEST",
            timestamp=NOW,
            context_window_hash="test",
        )
        assert ro.content_hash != ""

    def test_retrain_output_timezone_required(self):
        with pytest.raises(Exception):
            RetrainOutput(
                agent_id="TEST",
                timestamp=datetime(2026, 1, 1),  # No tz
                context_window_hash="test",
            )


# ===========================================================================
# compute_period_metrics Tests
# ===========================================================================

class TestComputePeriodMetrics:
    def test_basic_period(self):
        trades = [
            _make_trade(50.0, 10),
            _make_trade(-20.0, 5),
            _make_trade(30.0, 8),
        ]
        result = compute_period_metrics(
            trades, BT_START, BT_END, [],
        )
        assert result["return_bps"] == 60.0  # 50 + (-20) + 30
        assert result["trade_count"] == 3
        assert result["win_rate"] == pytest.approx(2 / 3, rel=1e-2)

    def test_empty_trades(self):
        result = compute_period_metrics([], BT_START, BT_END, [])
        assert result["return_bps"] == 0.0
        assert result["trade_count"] == 0
        assert result["win_rate"] == 0.0

    def test_max_drawdown(self):
        # Returns: +100, -200, +50 → peak 100, drawdown to -100 → dd = 200
        trades = [
            _make_trade(100.0),
            _make_trade(-200.0),
            _make_trade(50.0),
        ]
        result = compute_period_metrics(trades, BT_START, BT_END, [])
        assert result["max_drawdown_bps"] == 200.0

    def test_sharpe_ratio(self):
        # Multiple trades with variance → non-zero Sharpe
        trades = [
            _make_trade(50.0),
            _make_trade(60.0),
            _make_trade(40.0),
        ]
        result = compute_period_metrics(trades, BT_START, BT_END, [])
        assert result["sharpe_ratio"] != 0.0

    def test_single_trade_no_sharpe(self):
        trades = [_make_trade(50.0)]
        result = compute_period_metrics(trades, BT_START, BT_END, [])
        assert result["sharpe_ratio"] == 0.0  # Need ≥ 2 trades

    def test_dominant_regime(self):
        regimes = [
            _make_regime(datetime(2024, 2, 1, tzinfo=timezone.utc).isoformat(), "LOW_VOL_TRENDING"),
            _make_regime(datetime(2024, 3, 1, tzinfo=timezone.utc).isoformat(), "LOW_VOL_TRENDING"),
            _make_regime(datetime(2024, 4, 1, tzinfo=timezone.utc).isoformat(), "HIGH_VOL_MEAN_REV"),
        ]
        result = compute_period_metrics(
            [_make_trade()], BT_START, BT_END, regimes,
        )
        assert result["dominant_regime"] == "LOW_VOL_TRENDING"

    def test_avg_holding_days(self):
        trades = [_make_trade(holding_days=10), _make_trade(holding_days=20)]
        result = compute_period_metrics(trades, BT_START, BT_END, [])
        assert result["avg_holding_days"] == 15.0


# ===========================================================================
# compute_profit_factor Tests
# ===========================================================================

class TestComputeProfitFactor:
    def test_profitable(self):
        # Gross profits = 150, gross losses = 50 → PF = 3.0
        returns = [100.0, 50.0, -30.0, -20.0]
        assert compute_profit_factor(returns) == 3.0

    def test_no_losses(self):
        returns = [100.0, 50.0]
        pf = compute_profit_factor(returns)
        assert pf == 150.0  # Just gross_profits rounded

    def test_no_profits(self):
        returns = [-50.0, -30.0]
        assert compute_profit_factor(returns) == 0.0

    def test_empty(self):
        assert compute_profit_factor([]) == 0.0

    def test_breakeven(self):
        returns = [50.0, -50.0]
        assert compute_profit_factor(returns) == 1.0

    def test_all_zero(self):
        returns = [0.0, 0.0]
        assert compute_profit_factor(returns) == 0.0


# ===========================================================================
# LearnBacktest Integration Tests
# ===========================================================================

class TestLearnBacktest:
    @pytest.mark.asyncio
    async def test_process_basic(self):
        trades = [
            _make_trade(50.0, 10, datetime(2024, 1, 15, tzinfo=timezone.utc).isoformat()),
            _make_trade(-20.0, 5, datetime(2024, 2, 15, tzinfo=timezone.utc).isoformat()),
            _make_trade(30.0, 8, datetime(2024, 3, 15, tzinfo=timezone.utc).isoformat()),
        ]
        agent = LearnBacktest()
        ctx = _make_context(
            trade_history=trades,
            regime_history=[],
            backtest_start=BT_START.isoformat(),
            backtest_end=BT_END.isoformat(),
            period_days=30,
        )
        result = await agent.process(ctx)
        assert isinstance(result, BacktestOutput)
        assert result.total_return_bps == 60.0
        assert result.total_trades == 3
        assert len(result.periods) > 0

    @pytest.mark.asyncio
    async def test_process_empty(self):
        agent = LearnBacktest()
        ctx = _make_context(
            trade_history=[],
            backtest_start=BT_START.isoformat(),
            backtest_end=BT_END.isoformat(),
        )
        result = await agent.process(ctx)
        assert result.total_trades == 0
        assert result.total_return_bps == 0.0
        assert len(result.periods) == 0

    @pytest.mark.asyncio
    async def test_content_hash(self):
        agent = LearnBacktest()
        ctx = _make_context(
            trade_history=[_make_trade()],
            backtest_start=BT_START.isoformat(),
            backtest_end=BT_END.isoformat(),
        )
        result = await agent.process(ctx)
        assert len(result.content_hash) == 64

    @pytest.mark.asyncio
    async def test_annualized_return(self):
        agent = LearnBacktest()
        ctx = _make_context(
            trade_history=[_make_trade(100.0)],
            backtest_start=BT_START.isoformat(),
            backtest_end=BT_END.isoformat(),
        )
        result = await agent.process(ctx)
        total_days = (BT_END - BT_START).days
        expected_annualized = 100.0 * (TRADING_DAYS_PER_YEAR / total_days)
        assert result.annualized_return_bps == pytest.approx(expected_annualized, rel=1e-2)

    @pytest.mark.asyncio
    async def test_win_rate(self):
        trades = [
            _make_trade(50.0, exit_timestamp=datetime(2024, 2, 1, tzinfo=timezone.utc).isoformat()),
            _make_trade(-20.0, exit_timestamp=datetime(2024, 2, 15, tzinfo=timezone.utc).isoformat()),
            _make_trade(30.0, exit_timestamp=datetime(2024, 3, 1, tzinfo=timezone.utc).isoformat()),
        ]
        agent = LearnBacktest()
        ctx = _make_context(
            trade_history=trades,
            backtest_start=BT_START.isoformat(),
            backtest_end=BT_END.isoformat(),
        )
        result = await agent.process(ctx)
        assert result.overall_win_rate == pytest.approx(2 / 3, rel=1e-2)

    @pytest.mark.asyncio
    async def test_profit_factor(self):
        trades = [
            _make_trade(100.0, exit_timestamp=datetime(2024, 2, 1, tzinfo=timezone.utc).isoformat()),
            _make_trade(-50.0, exit_timestamp=datetime(2024, 3, 1, tzinfo=timezone.utc).isoformat()),
        ]
        agent = LearnBacktest()
        ctx = _make_context(
            trade_history=trades,
            backtest_start=BT_START.isoformat(),
            backtest_end=BT_END.isoformat(),
        )
        result = await agent.process(ctx)
        assert result.profit_factor == 2.0

    @pytest.mark.asyncio
    async def test_sub_period_partitioning(self):
        """Trades spread across multiple 30-day periods."""
        trades = [
            _make_trade(50.0, exit_timestamp=datetime(2024, 1, 15, tzinfo=timezone.utc).isoformat()),
            _make_trade(30.0, exit_timestamp=datetime(2024, 3, 15, tzinfo=timezone.utc).isoformat()),
            _make_trade(-10.0, exit_timestamp=datetime(2024, 5, 15, tzinfo=timezone.utc).isoformat()),
        ]
        agent = LearnBacktest()
        ctx = _make_context(
            trade_history=trades,
            backtest_start=BT_START.isoformat(),
            backtest_end=BT_END.isoformat(),
            period_days=30,
        )
        result = await agent.process(ctx)
        # 181 days / 30 = ~6 periods, but only 3 have trades
        assert len(result.periods) == 3

    @pytest.mark.asyncio
    async def test_max_drawdown(self):
        trades = [
            _make_trade(100.0, exit_timestamp=datetime(2024, 2, 1, tzinfo=timezone.utc).isoformat()),
            _make_trade(-200.0, exit_timestamp=datetime(2024, 3, 1, tzinfo=timezone.utc).isoformat()),
            _make_trade(50.0, exit_timestamp=datetime(2024, 4, 1, tzinfo=timezone.utc).isoformat()),
        ]
        agent = LearnBacktest()
        ctx = _make_context(
            trade_history=trades,
            backtest_start=BT_START.isoformat(),
            backtest_end=BT_END.isoformat(),
        )
        result = await agent.process(ctx)
        assert result.max_drawdown_bps == 200.0


class TestLearnBacktestHealth:
    def test_agent_id(self):
        assert LearnBacktest().agent_id == "LEARN-BACKTEST"

    def test_healthy(self):
        assert LearnBacktest().get_health().status == AgentStatus.HEALTHY

    def test_degraded(self):
        agent = LearnBacktest()
        agent._error_count_24h = 5
        assert agent.get_health().status == AgentStatus.DEGRADED

    def test_unhealthy(self):
        agent = LearnBacktest()
        agent._error_count_24h = 15
        assert agent.get_health().status == AgentStatus.UNHEALTHY


class TestBacktestSchemas:
    def test_period_frozen(self):
        bp = BacktestPeriod(
            period_start=NOW,
            period_end=NOW,
            return_bps=50.0,
        )
        with pytest.raises(Exception):
            bp.return_bps = 100.0

    def test_period_timezone_required(self):
        with pytest.raises(Exception):
            BacktestPeriod(
                period_start=datetime(2026, 1, 1),  # No tz
                period_end=NOW,
            )

    def test_backtest_output_hash(self):
        bo = BacktestOutput(
            agent_id="TEST",
            timestamp=NOW,
            context_window_hash="test",
            backtest_start=NOW,
            backtest_end=NOW,
        )
        assert bo.content_hash != ""

    def test_backtest_output_timezone_required(self):
        with pytest.raises(Exception):
            BacktestOutput(
                agent_id="TEST",
                timestamp=datetime(2026, 1, 1),  # No tz
                context_window_hash="test",
                backtest_start=NOW,
                backtest_end=NOW,
            )

    def test_profit_factor_non_negative(self):
        with pytest.raises(Exception):
            BacktestOutput(
                agent_id="TEST",
                timestamp=NOW,
                context_window_hash="test",
                backtest_start=NOW,
                backtest_end=NOW,
                profit_factor=-1.0,
            )
