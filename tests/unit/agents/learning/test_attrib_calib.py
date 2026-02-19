"""Tests for LEARN-ATTRIB and LEARN-CALIB agents.

Tests cover:
  - LEARN-ATTRIB: Hit rate, information ratio, Sharpe contribution, attribution
  - LEARN-CALIB: Brier score, bucket calibration, overconfidence detection
"""

import math
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from providence.agents.base import AgentContext, AgentStatus
from providence.agents.learning.attrib import (
    LearnAttrib,
    attribute_agent,
    attribute_ticker,
    compute_hit_rate,
    compute_information_ratio,
    compute_sharpe_contribution,
)
from providence.agents.learning.calib import (
    LearnCalib,
    calibrate_agent,
    calibrate_bucket,
    compute_brier_score,
)
from providence.schemas.learning import (
    AgentAttribution,
    AgentCalibration,
    AttributionOutput,
    CalibrationBucket,
    CalibrationOutput,
    TickerAttribution,
)

NOW = datetime.now(timezone.utc)
EVAL_START = datetime(2025, 1, 1, tzinfo=timezone.utc)
EVAL_END = datetime(2025, 6, 30, tzinfo=timezone.utc)


def _make_context(**metadata_kwargs) -> AgentContext:
    return AgentContext(
        agent_id="TEST",
        trigger="manual",
        fragments=[],
        context_window_hash="test-hash-learn-001",
        timestamp=NOW,
        metadata=metadata_kwargs,
    )


def _make_belief(
    agent_id: str = "COGNIT-FUNDAMENTAL",
    direction: str = "LONG",
    raw_confidence: float = 0.70,
    realized_return_bps: float = 50.0,
    was_acted_on: bool = True,
    was_correct: bool = True,
) -> dict:
    return {
        "agent_id": agent_id,
        "direction": direction,
        "raw_confidence": raw_confidence,
        "realized_return_bps": realized_return_bps,
        "was_acted_on": was_acted_on,
        "was_correct": was_correct,
    }


def _make_position(
    ticker: str = "AAPL",
    realized_pnl_bps: float = 50.0,
    holding_days: int = 30,
    avg_weight: float = 0.05,
    contributing_agents: list | None = None,
) -> dict:
    return {
        "ticker": ticker,
        "realized_pnl_bps": realized_pnl_bps,
        "holding_days": holding_days,
        "avg_weight": avg_weight,
        "contributing_agents": contributing_agents or ["COGNIT-FUNDAMENTAL"],
        "exit_reason": "trailing_stop",
        "regime_during_hold": "LOW_VOL_TRENDING",
    }


# ===========================================================================
# compute_hit_rate Tests
# ===========================================================================

class TestComputeHitRate:
    def test_all_correct_long(self):
        beliefs = [
            _make_belief(direction="LONG", realized_return_bps=50.0),
            _make_belief(direction="LONG", realized_return_bps=30.0),
        ]
        assert compute_hit_rate(beliefs) == 1.0

    def test_all_wrong(self):
        beliefs = [
            _make_belief(direction="LONG", realized_return_bps=-50.0),
            _make_belief(direction="SHORT", realized_return_bps=30.0),
        ]
        assert compute_hit_rate(beliefs) == 0.0

    def test_mixed(self):
        beliefs = [
            _make_belief(direction="LONG", realized_return_bps=50.0),
            _make_belief(direction="LONG", realized_return_bps=-20.0),
        ]
        assert compute_hit_rate(beliefs) == 0.5

    def test_short_correct(self):
        beliefs = [_make_belief(direction="SHORT", realized_return_bps=-50.0)]
        assert compute_hit_rate(beliefs) == 1.0

    def test_neutral_excluded(self):
        beliefs = [
            _make_belief(direction="NEUTRAL", realized_return_bps=50.0),
        ]
        assert compute_hit_rate(beliefs) == 0.0

    def test_empty(self):
        assert compute_hit_rate([]) == 0.0


# ===========================================================================
# compute_information_ratio Tests
# ===========================================================================

class TestInformationRatio:
    def test_positive_excess(self):
        returns = [10.0, 20.0, 15.0, 25.0]
        benchmark = [5.0, 5.0, 5.0, 5.0]
        ir = compute_information_ratio(returns, benchmark)
        assert ir > 0

    def test_zero_excess(self):
        returns = [10.0, 10.0, 10.0]
        benchmark = [10.0, 10.0, 10.0]
        # Excess = [0,0,0], no variance → 0.0
        ir = compute_information_ratio(returns, benchmark)
        assert ir == 0.0

    def test_insufficient_data(self):
        assert compute_information_ratio([10.0], [5.0]) == 0.0
        assert compute_information_ratio([], []) == 0.0


# ===========================================================================
# attribute_agent Tests
# ===========================================================================

class TestAttributeAgent:
    def test_basic_attribution(self):
        beliefs = [
            _make_belief(realized_return_bps=50.0, was_acted_on=True),
            _make_belief(realized_return_bps=-20.0, was_acted_on=True),
            _make_belief(realized_return_bps=30.0, was_acted_on=False),
        ]
        result = attribute_agent("COGNIT-FUNDAMENTAL", beliefs, [10.0, 10.0], [50.0, -20.0])
        assert result["total_beliefs_produced"] == 3
        assert result["beliefs_acted_on"] == 2
        assert result["hit_rate"] == 0.5  # 1 of 2 acted-on correct (LONG + positive)
        assert result["avg_return_bps"] == 15.0  # (50 + -20) / 2

    def test_empty_beliefs(self):
        result = attribute_agent("AGENT", [], [0.0], [0.0])
        assert result["total_beliefs_produced"] == 0
        assert result["hit_rate"] == 0.0


# ===========================================================================
# attribute_ticker Tests
# ===========================================================================

class TestAttributeTicker:
    def test_basic_ticker(self):
        positions = [
            _make_position(realized_pnl_bps=50.0, holding_days=30),
            _make_position(realized_pnl_bps=-20.0, holding_days=15),
        ]
        result = attribute_ticker("AAPL", positions)
        assert result["total_pnl_bps"] == 30.0
        assert result["holding_days"] == 45
        assert "COGNIT-FUNDAMENTAL" in result["contributing_agents"]

    def test_empty_positions(self):
        result = attribute_ticker("AAPL", [])
        assert result["total_pnl_bps"] == 0.0


# ===========================================================================
# LearnAttrib Integration Tests
# ===========================================================================

class TestLearnAttrib:
    @pytest.mark.asyncio
    async def test_process_basic(self):
        agent = LearnAttrib()
        ctx = _make_context(
            closed_positions=[
                _make_position("AAPL", 50.0),
                _make_position("MSFT", -20.0),
            ],
            belief_history=[
                _make_belief("COGNIT-FUNDAMENTAL", realized_return_bps=50.0),
                _make_belief("COGNIT-TECHNICAL", realized_return_bps=-20.0),
            ],
            evaluation_start=EVAL_START.isoformat(),
            evaluation_end=EVAL_END.isoformat(),
        )
        result = await agent.process(ctx)
        assert isinstance(result, AttributionOutput)
        assert len(result.agent_attributions) == 2
        assert len(result.ticker_attributions) == 2
        assert result.total_trades == 2
        assert result.portfolio_return_bps == 30.0

    @pytest.mark.asyncio
    async def test_process_empty(self):
        agent = LearnAttrib()
        ctx = _make_context(closed_positions=[], belief_history=[])
        result = await agent.process(ctx)
        assert result.total_trades == 0
        assert result.portfolio_return_bps == 0.0

    @pytest.mark.asyncio
    async def test_content_hash(self):
        agent = LearnAttrib()
        ctx = _make_context(
            closed_positions=[_make_position()],
            belief_history=[_make_belief()],
        )
        result = await agent.process(ctx)
        assert len(result.content_hash) == 64

    @pytest.mark.asyncio
    async def test_portfolio_sharpe(self):
        # Multiple positions with variance → Sharpe should be non-zero
        agent = LearnAttrib()
        ctx = _make_context(
            closed_positions=[
                _make_position("AAPL", 100.0),
                _make_position("MSFT", 50.0),
                _make_position("GOOG", -30.0),
            ],
            belief_history=[],
        )
        result = await agent.process(ctx)
        assert result.portfolio_sharpe != 0.0


class TestLearnAttribHealth:
    def test_agent_id(self):
        assert LearnAttrib().agent_id == "LEARN-ATTRIB"

    def test_healthy(self):
        assert LearnAttrib().get_health().status == AgentStatus.HEALTHY


# ===========================================================================
# compute_brier_score Tests
# ===========================================================================

class TestBrierScore:
    def test_perfect_prediction(self):
        assert compute_brier_score(1.0, True) == 0.0

    def test_worst_prediction(self):
        assert compute_brier_score(0.0, True) == 1.0

    def test_50_50(self):
        assert compute_brier_score(0.5, True) == 0.25
        assert compute_brier_score(0.5, False) == 0.25

    def test_overconfident_wrong(self):
        assert compute_brier_score(0.9, False) == pytest.approx(0.81, rel=1e-2)


# ===========================================================================
# calibrate_bucket Tests
# ===========================================================================

class TestCalibrateBucket:
    def test_perfect_calibration(self):
        # All beliefs at 0.7 confidence, 70% correct
        beliefs = [
            _make_belief(raw_confidence=0.70, was_correct=True),
            _make_belief(raw_confidence=0.70, was_correct=True),
            _make_belief(raw_confidence=0.70, was_correct=True),
            _make_belief(raw_confidence=0.70, was_correct=True),
            _make_belief(raw_confidence=0.70, was_correct=True),
            _make_belief(raw_confidence=0.70, was_correct=True),
            _make_belief(raw_confidence=0.70, was_correct=True),
            _make_belief(raw_confidence=0.70, was_correct=False),
            _make_belief(raw_confidence=0.70, was_correct=False),
            _make_belief(raw_confidence=0.70, was_correct=False),
        ]
        result = calibrate_bucket(beliefs, 0.6, 0.8)
        assert result["sample_count"] == 10
        assert abs(result["calibration_error"]) == 0.0  # 0.7 stated, 0.7 realized

    def test_overconfident_bucket(self):
        # All at 0.80 confidence but only 50% correct
        beliefs = [
            _make_belief(raw_confidence=0.80, was_correct=True),
            _make_belief(raw_confidence=0.80, was_correct=False),
        ]
        result = calibrate_bucket(beliefs, 0.6, 1.0)
        assert result["calibration_error"] == 0.30  # 0.80 - 0.50

    def test_empty_bucket(self):
        result = calibrate_bucket([], 0.0, 0.2)
        assert result["sample_count"] == 0


# ===========================================================================
# calibrate_agent Tests
# ===========================================================================

class TestCalibrateAgent:
    def test_overconfident_agent(self):
        beliefs = [
            _make_belief(raw_confidence=0.85, was_correct=False),
            _make_belief(raw_confidence=0.90, was_correct=False),
            _make_belief(raw_confidence=0.80, was_correct=True),
            _make_belief(raw_confidence=0.75, was_correct=False),
        ]
        result = calibrate_agent("AGENT", beliefs)
        assert result["is_overconfident"] is True
        assert result["recommended_adjustment"] < 1.0  # Reduce confidence

    def test_well_calibrated_agent(self):
        # 70% confidence, 70% accuracy → well calibrated
        beliefs = []
        for i in range(10):
            beliefs.append(_make_belief(raw_confidence=0.70, was_correct=(i < 7)))
        result = calibrate_agent("AGENT", beliefs)
        assert abs(result["overall_calibration_error"]) < 0.10


# ===========================================================================
# LearnCalib Integration Tests
# ===========================================================================

class TestLearnCalib:
    @pytest.mark.asyncio
    async def test_process_basic(self):
        agent = LearnCalib()
        beliefs = [
            _make_belief("AGENT-A", raw_confidence=0.80, was_correct=True),
            _make_belief("AGENT-A", raw_confidence=0.80, was_correct=False),
            _make_belief("AGENT-B", raw_confidence=0.50, was_correct=True),
            _make_belief("AGENT-B", raw_confidence=0.50, was_correct=False),
        ]
        ctx = _make_context(
            belief_outcomes=beliefs,
            evaluation_start=EVAL_START.isoformat(),
            evaluation_end=EVAL_END.isoformat(),
        )
        result = await agent.process(ctx)
        assert isinstance(result, CalibrationOutput)
        assert len(result.agent_calibrations) == 2

    @pytest.mark.asyncio
    async def test_process_empty(self):
        agent = LearnCalib()
        ctx = _make_context(belief_outcomes=[])
        result = await agent.process(ctx)
        assert len(result.agent_calibrations) == 0

    @pytest.mark.asyncio
    async def test_overconfident_detected(self):
        agent = LearnCalib()
        # Agent with 0.90 confidence but only 40% accuracy
        beliefs = [
            _make_belief("AGENT-A", raw_confidence=0.90, was_correct=True),
            _make_belief("AGENT-A", raw_confidence=0.90, was_correct=True),
            _make_belief("AGENT-A", raw_confidence=0.90, was_correct=False),
            _make_belief("AGENT-A", raw_confidence=0.90, was_correct=False),
            _make_belief("AGENT-A", raw_confidence=0.90, was_correct=False),
        ]
        ctx = _make_context(belief_outcomes=beliefs)
        result = await agent.process(ctx)
        assert result.agents_overconfident >= 1

    @pytest.mark.asyncio
    async def test_content_hash(self):
        agent = LearnCalib()
        ctx = _make_context(belief_outcomes=[_make_belief()])
        result = await agent.process(ctx)
        assert len(result.content_hash) == 64

    @pytest.mark.asyncio
    async def test_system_brier_score(self):
        agent = LearnCalib()
        beliefs = [
            _make_belief("A", raw_confidence=0.90, was_correct=True),
            _make_belief("B", raw_confidence=0.10, was_correct=False),
        ]
        ctx = _make_context(belief_outcomes=beliefs)
        result = await agent.process(ctx)
        assert result.system_brier_score >= 0.0


class TestLearnCalibHealth:
    def test_agent_id(self):
        assert LearnCalib().agent_id == "LEARN-CALIB"

    def test_healthy(self):
        assert LearnCalib().get_health().status == AgentStatus.HEALTHY

    def test_degraded(self):
        agent = LearnCalib()
        agent._error_count_24h = 5
        assert agent.get_health().status == AgentStatus.DEGRADED


# ===========================================================================
# Schema Tests
# ===========================================================================

class TestLearningSchemas:
    def test_agent_attribution_frozen(self):
        aa = AgentAttribution(agent_id="TEST", hit_rate=0.65)
        with pytest.raises(Exception):
            aa.hit_rate = 0.80

    def test_attribution_output_timezone(self):
        with pytest.raises(Exception):
            AttributionOutput(
                agent_id="TEST",
                timestamp=datetime(2026, 1, 1),  # No tz
                context_window_hash="test",
                evaluation_start=NOW,
                evaluation_end=NOW,
            )

    def test_calibration_bucket_frozen(self):
        cb = CalibrationBucket(
            bucket_lower=0.0, bucket_upper=0.2,
            sample_count=10, avg_stated_confidence=0.1,
        )
        with pytest.raises(Exception):
            cb.sample_count = 20

    def test_calibration_output_hash(self):
        co = CalibrationOutput(
            agent_id="TEST",
            timestamp=NOW,
            context_window_hash="test",
            evaluation_start=NOW,
            evaluation_end=NOW,
        )
        assert co.content_hash != ""
