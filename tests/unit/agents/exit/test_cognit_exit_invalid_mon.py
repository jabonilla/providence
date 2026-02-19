"""Tests for COGNIT-EXIT and INVALID-MON agents.

Tests cover:
  - COGNIT-EXIT: LLM response parsing, renewal deferral, thesis health, process flow
  - INVALID-MON: Condition evaluation, breach detection, approaching detection, process flow
"""

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

from providence.agents.base import AgentContext, AgentStatus
from providence.agents.exit.cognit_exit import (
    CognitExit,
    apply_renewal_deferral,
    compute_thesis_health,
    parse_exit_response,
)
from providence.agents.exit.invalid_mon import (
    BREACH_IMPACT_MARGINAL,
    BREACH_IMPACT_MODERATE,
    BREACH_IMPACT_STRONG,
    InvalidMon,
    evaluate_condition,
    is_approaching,
)
from providence.exceptions import AgentProcessingError
from providence.schemas.exit import (
    ExitAssessment,
    ExitOutput,
    InvalidationMonitorOutput,
    MonitoredCondition,
)

NOW = datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_context(**metadata_kwargs) -> AgentContext:
    return AgentContext(
        agent_id="TEST",
        trigger="schedule",
        fragments=[],
        context_window_hash="test-hash-exit-001",
        timestamp=NOW,
        metadata=metadata_kwargs,
    )


def _make_belief(
    thesis_id: str = "FUND-AAPL-001",
    ticker: str = "AAPL",
    agent_id: str = "COGNIT-FUNDAMENTAL",
    raw_confidence: float = 0.70,
    time_horizon_days: int = 60,
    days_elapsed: int = 30,
    conditions: list | None = None,
) -> dict:
    if conditions is None:
        conditions = [
            {
                "condition_id": str(uuid4()),
                "metric": "gross_margin_pct",
                "operator": "LT",
                "threshold": 44.0,
                "status": "ACTIVE",
                "current_value": 46.5,
            },
            {
                "condition_id": str(uuid4()),
                "metric": "close_price",
                "operator": "LT",
                "threshold": 165.0,
                "status": "ACTIVE",
                "current_value": 195.0,
            },
        ]
    return {
        "thesis_id": thesis_id,
        "ticker": ticker,
        "agent_id": agent_id,
        "raw_confidence": raw_confidence,
        "time_horizon_days": time_horizon_days,
        "days_elapsed": days_elapsed,
        "invalidation_conditions": conditions,
    }


def _make_position(
    ticker: str = "AAPL",
    unrealized_pnl: float = 0.05,
    days_held: int = 10,
) -> dict:
    return {
        "ticker": ticker,
        "unrealized_pnl": unrealized_pnl,
        "days_held": days_held,
    }


# ===========================================================================
# parse_exit_response Tests
# ===========================================================================

class TestParseExitResponse:
    def test_valid_response(self):
        raw = json.dumps({
            "assessments": [{
                "ticker": "AAPL",
                "exit_action": "HOLD",
                "exit_confidence": 0.20,
                "regret_estimate_bps": 15.0,
                "regret_direction": "MISSED_UPSIDE",
                "thesis_health_score": 0.85,
                "rationale": "Thesis healthy.",
            }],
        })
        result = parse_exit_response(raw)
        assert result is not None
        assert len(result) == 1
        assert result[0]["ticker"] == "AAPL"
        assert result[0]["exit_action"] == "HOLD"
        assert result[0]["exit_confidence"] == 0.20

    def test_multi_ticker_response(self):
        raw = json.dumps({
            "assessments": [
                {"ticker": "AAPL", "exit_action": "HOLD", "exit_confidence": 0.10},
                {"ticker": "MSFT", "exit_action": "EXIT", "exit_confidence": 0.85},
            ],
        })
        result = parse_exit_response(raw)
        assert result is not None
        assert len(result) == 2
        assert result[1]["exit_action"] == "EXIT"

    def test_malformed_no_assessments(self):
        raw = json.dumps({"wrong_key": "data"})
        result = parse_exit_response(raw)
        assert result is None

    def test_empty_assessments(self):
        raw = json.dumps({"assessments": []})
        result = parse_exit_response(raw)
        assert result is None

    def test_invalid_action_defaults_to_hold(self):
        raw = json.dumps({
            "assessments": [{
                "ticker": "AAPL",
                "exit_action": "BUY_MORE",
                "exit_confidence": 0.50,
            }],
        })
        result = parse_exit_response(raw)
        assert result is not None
        assert result[0]["exit_action"] == "HOLD"

    def test_confidence_clamped_to_max(self):
        raw = json.dumps({
            "assessments": [{
                "ticker": "AAPL",
                "exit_action": "EXIT",
                "exit_confidence": 1.5,
            }],
        })
        result = parse_exit_response(raw)
        assert result is not None
        assert result[0]["exit_confidence"] <= 0.95

    def test_negative_confidence_clamped(self):
        raw = json.dumps({
            "assessments": [{
                "ticker": "AAPL",
                "exit_action": "HOLD",
                "exit_confidence": -0.5,
            }],
        })
        result = parse_exit_response(raw)
        assert result is not None
        assert result[0]["exit_confidence"] >= 0.0

    def test_negative_regret_clamped(self):
        raw = json.dumps({
            "assessments": [{
                "ticker": "AAPL",
                "exit_action": "HOLD",
                "exit_confidence": 0.2,
                "regret_estimate_bps": -50.0,
            }],
        })
        result = parse_exit_response(raw)
        assert result[0]["regret_estimate_bps"] >= 0.0

    def test_invalid_regret_direction_defaults(self):
        raw = json.dumps({
            "assessments": [{
                "ticker": "AAPL",
                "exit_action": "HOLD",
                "exit_confidence": 0.2,
                "regret_direction": "WRONG",
            }],
        })
        result = parse_exit_response(raw)
        assert result[0]["regret_direction"] == "MISSED_UPSIDE"

    def test_markdown_fences_stripped(self):
        inner = json.dumps({
            "assessments": [{
                "ticker": "AAPL",
                "exit_action": "REDUCE",
                "exit_confidence": 0.60,
            }],
        })
        raw = f"```json\n{inner}\n```"
        result = parse_exit_response(raw)
        assert result is not None
        assert result[0]["exit_action"] == "REDUCE"

    def test_json_embedded_in_text(self):
        inner = json.dumps({
            "assessments": [{
                "ticker": "AAPL",
                "exit_action": "EXIT",
                "exit_confidence": 0.80,
            }],
        })
        raw = f"Here is my analysis: {inner} That's all."
        result = parse_exit_response(raw)
        assert result is not None
        assert result[0]["exit_action"] == "EXIT"

    def test_non_json_string(self):
        result = parse_exit_response("This is not JSON at all.")
        assert result is None

    def test_missing_ticker_skipped(self):
        raw = json.dumps({
            "assessments": [
                {"exit_action": "HOLD", "exit_confidence": 0.2},
                {"ticker": "MSFT", "exit_action": "EXIT", "exit_confidence": 0.8},
            ],
        })
        result = parse_exit_response(raw)
        assert result is not None
        assert len(result) == 1
        assert result[0]["ticker"] == "MSFT"

    def test_ticker_uppercased(self):
        raw = json.dumps({
            "assessments": [{
                "ticker": "aapl",
                "exit_action": "HOLD",
                "exit_confidence": 0.2,
            }],
        })
        result = parse_exit_response(raw)
        assert result[0]["ticker"] == "AAPL"


# ===========================================================================
# apply_renewal_deferral Tests
# ===========================================================================

class TestApplyRenewalDeferral:
    def test_no_renewal_no_change(self):
        assessment = {
            "ticker": "AAPL",
            "exit_action": "EXIT",
            "exit_confidence": 0.85,
            "rationale": "Thesis broken.",
        }
        result = apply_renewal_deferral(assessment, {}, [])
        assert result["exit_action"] == "EXIT"
        assert result["renewal_pending"] is False

    def test_renewal_pending_high_asymmetry_defers_exit(self):
        assessment = {
            "ticker": "AAPL",
            "exit_action": "EXIT",
            "exit_confidence": 0.80,
            "rationale": "Original rationale.",
        }
        beliefs = [{"ticker": "AAPL", "thesis_id": "FUND-AAPL-001"}]
        renewal_state = {
            "FUND-AAPL-001": {"is_renewed": True, "asymmetry_score": 0.70},
        }
        result = apply_renewal_deferral(assessment, renewal_state, beliefs)
        assert result["exit_action"] == "REDUCE"
        assert result["renewal_pending"] is True
        assert result["renewal_asymmetry"] == 0.70
        assert "[DEFERRED]" in result["rationale"]

    def test_renewal_pending_low_asymmetry_no_deferral(self):
        assessment = {
            "ticker": "AAPL",
            "exit_action": "EXIT",
            "exit_confidence": 0.80,
            "rationale": "Exit.",
        }
        beliefs = [{"ticker": "AAPL", "thesis_id": "FUND-AAPL-001"}]
        renewal_state = {
            "FUND-AAPL-001": {"is_renewed": True, "asymmetry_score": 0.30},
        }
        result = apply_renewal_deferral(assessment, renewal_state, beliefs)
        assert result["exit_action"] == "EXIT"  # No deferral

    def test_hold_not_affected_by_renewal(self):
        assessment = {
            "ticker": "AAPL",
            "exit_action": "HOLD",
            "exit_confidence": 0.10,
            "rationale": "Hold.",
        }
        beliefs = [{"ticker": "AAPL", "thesis_id": "FUND-AAPL-001"}]
        renewal_state = {
            "FUND-AAPL-001": {"is_renewed": True, "asymmetry_score": 0.90},
        }
        result = apply_renewal_deferral(assessment, renewal_state, beliefs)
        assert result["exit_action"] == "HOLD"

    def test_multiple_theses_max_asymmetry(self):
        assessment = {
            "ticker": "AAPL",
            "exit_action": "EXIT",
            "exit_confidence": 0.75,
            "rationale": "Exit.",
        }
        beliefs = [
            {"ticker": "AAPL", "thesis_id": "T1"},
            {"ticker": "AAPL", "thesis_id": "T2"},
        ]
        renewal_state = {
            "T1": {"is_renewed": True, "asymmetry_score": 0.40},
            "T2": {"is_renewed": True, "asymmetry_score": 0.65},
        }
        result = apply_renewal_deferral(assessment, renewal_state, beliefs)
        assert result["renewal_asymmetry"] == 0.65
        assert result["exit_action"] == "REDUCE"  # Deferred (0.65 > 0.5)


# ===========================================================================
# compute_thesis_health Tests
# ===========================================================================

class TestComputeThesisHealth:
    def test_all_active(self):
        beliefs = [_make_belief(conditions=[
            {"status": "ACTIVE", "ticker": "AAPL"},
            {"status": "ACTIVE", "ticker": "AAPL"},
        ])]
        health, triggered, total = compute_thesis_health("AAPL", beliefs)
        assert health == 1.0
        assert triggered == 0
        assert total == 2

    def test_one_triggered(self):
        beliefs = [_make_belief(conditions=[
            {"status": "ACTIVE", "ticker": "AAPL"},
            {"status": "TRIGGERED", "ticker": "AAPL"},
        ])]
        health, triggered, total = compute_thesis_health("AAPL", beliefs)
        assert health == 0.5
        assert triggered == 1

    def test_all_triggered(self):
        beliefs = [_make_belief(conditions=[
            {"status": "TRIGGERED", "ticker": "AAPL"},
            {"status": "TRIGGERED", "ticker": "AAPL"},
        ])]
        health, triggered, total = compute_thesis_health("AAPL", beliefs)
        assert health == 0.0
        assert triggered == 2

    def test_no_conditions(self):
        beliefs = [_make_belief(conditions=[])]
        health, triggered, total = compute_thesis_health("AAPL", beliefs)
        assert health == 1.0
        assert total == 0

    def test_wrong_ticker_ignored(self):
        beliefs = [_make_belief(ticker="MSFT")]
        health, triggered, total = compute_thesis_health("AAPL", beliefs)
        assert health == 1.0
        assert total == 0


# ===========================================================================
# CognitExit Integration Tests
# ===========================================================================

class TestCognitExit:
    @pytest.mark.asyncio
    async def test_process_valid(self):
        mock_llm = AsyncMock()
        mock_llm.complete = AsyncMock(return_value=json.dumps({
            "assessments": [{
                "ticker": "AAPL",
                "exit_action": "HOLD",
                "exit_confidence": 0.20,
                "thesis_health_score": 0.85,
                "rationale": "Healthy thesis.",
            }],
        }))
        agent = CognitExit(llm_client=mock_llm)
        ctx = _make_context(
            active_positions=[_make_position()],
            active_beliefs=[_make_belief()],
            regime_state={"system_risk_mode": "NORMAL"},
        )
        result = await agent.process(ctx)
        assert isinstance(result, ExitOutput)
        assert len(result.assessments) == 1
        assert result.assessments[0].exit_action == "HOLD"
        assert result.positions_hold == 1

    @pytest.mark.asyncio
    async def test_process_multi_position(self):
        mock_llm = AsyncMock()
        mock_llm.complete = AsyncMock(return_value=json.dumps({
            "assessments": [
                {"ticker": "AAPL", "exit_action": "HOLD", "exit_confidence": 0.1},
                {"ticker": "MSFT", "exit_action": "REDUCE", "exit_confidence": 0.6},
                {"ticker": "TSLA", "exit_action": "EXIT", "exit_confidence": 0.9},
            ],
        }))
        agent = CognitExit(llm_client=mock_llm)
        ctx = _make_context(
            active_positions=[
                _make_position("AAPL"),
                _make_position("MSFT"),
                _make_position("TSLA"),
            ],
            active_beliefs=[],
        )
        result = await agent.process(ctx)
        assert result.positions_hold == 1
        assert result.positions_reduce == 1
        assert result.positions_exit == 1

    @pytest.mark.asyncio
    async def test_process_with_renewal_deferral(self):
        mock_llm = AsyncMock()
        mock_llm.complete = AsyncMock(return_value=json.dumps({
            "assessments": [{
                "ticker": "AAPL",
                "exit_action": "EXIT",
                "exit_confidence": 0.80,
                "rationale": "Thesis deteriorating.",
            }],
        }))
        agent = CognitExit(llm_client=mock_llm)
        ctx = _make_context(
            active_positions=[_make_position()],
            active_beliefs=[_make_belief(thesis_id="T1")],
            renewal_state={"T1": {"is_renewed": True, "asymmetry_score": 0.70}},
        )
        result = await agent.process(ctx)
        # EXIT deferred to REDUCE
        assert result.assessments[0].exit_action == "REDUCE"
        assert result.positions_reduce == 1
        assert result.positions_exit == 0

    @pytest.mark.asyncio
    async def test_empty_positions_returns_empty(self):
        mock_llm = AsyncMock()
        agent = CognitExit(llm_client=mock_llm)
        ctx = _make_context(active_positions=[])
        result = await agent.process(ctx)
        assert len(result.assessments) == 0
        mock_llm.complete.assert_not_called()

    @pytest.mark.asyncio
    async def test_malformed_llm_response_raises(self):
        mock_llm = AsyncMock()
        mock_llm.complete = AsyncMock(return_value=json.dumps({"bad": "data"}))
        agent = CognitExit(llm_client=mock_llm)
        ctx = _make_context(
            active_positions=[_make_position()],
        )
        with pytest.raises(AgentProcessingError):
            await agent.process(ctx)

    @pytest.mark.asyncio
    async def test_llm_exception_raises(self):
        mock_llm = AsyncMock()
        mock_llm.complete = AsyncMock(side_effect=RuntimeError("LLM down"))
        agent = CognitExit(llm_client=mock_llm)
        ctx = _make_context(active_positions=[_make_position()])
        with pytest.raises(AgentProcessingError, match="COGNIT-EXIT processing failed"):
            await agent.process(ctx)

    @pytest.mark.asyncio
    async def test_content_hash_computed(self):
        mock_llm = AsyncMock()
        mock_llm.complete = AsyncMock(return_value=json.dumps({
            "assessments": [{
                "ticker": "AAPL",
                "exit_action": "HOLD",
                "exit_confidence": 0.2,
            }],
        }))
        agent = CognitExit(llm_client=mock_llm)
        ctx = _make_context(active_positions=[_make_position()])
        result = await agent.process(ctx)
        assert result.content_hash != ""
        assert len(result.content_hash) == 64  # SHA-256 hex


class TestCognitExitHealth:
    def test_healthy(self):
        agent = CognitExit(llm_client=AsyncMock())
        health = agent.get_health()
        assert health.status == AgentStatus.HEALTHY

    def test_degraded(self):
        agent = CognitExit(llm_client=AsyncMock())
        agent._error_count_24h = 5
        health = agent.get_health()
        assert health.status == AgentStatus.DEGRADED

    def test_unhealthy(self):
        agent = CognitExit(llm_client=AsyncMock())
        agent._error_count_24h = 15
        health = agent.get_health()
        assert health.status == AgentStatus.UNHEALTHY

    def test_agent_id(self):
        agent = CognitExit(llm_client=AsyncMock())
        assert agent.agent_id == "COGNIT-EXIT"


# ===========================================================================
# evaluate_condition Tests
# ===========================================================================

class TestEvaluateCondition:
    def test_gt_breached(self):
        cond = {"metric": "rsi_14", "operator": "GT", "threshold": 70.0, "condition_id": str(uuid4())}
        result = evaluate_condition(cond, {"rsi_14": 75.0})
        assert result["is_breached"] is True
        assert result["breach_magnitude"] > 0

    def test_lt_breached(self):
        cond = {"metric": "close_price", "operator": "LT", "threshold": 165.0, "condition_id": str(uuid4())}
        result = evaluate_condition(cond, {"close_price": 150.0})
        assert result["is_breached"] is True

    def test_gt_not_breached(self):
        cond = {"metric": "rsi_14", "operator": "GT", "threshold": 70.0, "condition_id": str(uuid4())}
        result = evaluate_condition(cond, {"rsi_14": 65.0})
        assert result["is_breached"] is False

    def test_missing_current_value(self):
        cond = {"metric": "rsi_14", "operator": "GT", "threshold": 70.0, "condition_id": str(uuid4())}
        result = evaluate_condition(cond, {})
        assert result["is_breached"] is False
        assert result["current_value"] is None

    def test_marginal_breach_impact(self):
        cond = {"metric": "rsi_14", "operator": "GT", "threshold": 70.0, "condition_id": str(uuid4())}
        # 71.0 → magnitude = 1/70 ≈ 0.014 (marginal)
        result = evaluate_condition(cond, {"rsi_14": 71.0})
        assert result["is_breached"] is True
        assert result["confidence_impact"] == BREACH_IMPACT_MARGINAL

    def test_moderate_breach_impact(self):
        cond = {"metric": "rsi_14", "operator": "GT", "threshold": 70.0, "condition_id": str(uuid4())}
        # 80.0 → magnitude = 10/70 ≈ 0.143 (moderate)
        result = evaluate_condition(cond, {"rsi_14": 80.0})
        assert result["is_breached"] is True
        assert result["confidence_impact"] == BREACH_IMPACT_MODERATE

    def test_strong_breach_impact(self):
        cond = {"metric": "rsi_14", "operator": "GT", "threshold": 70.0, "condition_id": str(uuid4())}
        # 90.0 → magnitude = 20/70 ≈ 0.286 (strong)
        result = evaluate_condition(cond, {"rsi_14": 90.0})
        assert result["is_breached"] is True
        assert result["confidence_impact"] == BREACH_IMPACT_STRONG

    def test_velocity_computation(self):
        cond = {"metric": "rsi_14", "operator": "GT", "threshold": 70.0, "condition_id": str(uuid4())}
        historical = {"rsi_14": [60.0, 62.0, 64.0, 66.0, 68.0]}
        result = evaluate_condition(cond, {"rsi_14": 68.0}, historical)
        assert result["breach_velocity"] == 2.0  # +2 per day average

    def test_unknown_operator(self):
        cond = {"metric": "rsi_14", "operator": "UNKNOWN", "threshold": 70.0, "condition_id": str(uuid4())}
        result = evaluate_condition(cond, {"rsi_14": 75.0})
        assert result["is_breached"] is False

    def test_crosses_above_breached(self):
        cond = {"metric": "price", "operator": "CROSSES_ABOVE", "threshold": 200.0, "condition_id": str(uuid4())}
        result = evaluate_condition(cond, {"price": 210.0})
        assert result["is_breached"] is True

    def test_eq_breached(self):
        cond = {"metric": "count", "operator": "EQ", "threshold": 5.0, "condition_id": str(uuid4())}
        result = evaluate_condition(cond, {"count": 5.0})
        assert result["is_breached"] is True


# ===========================================================================
# is_approaching Tests
# ===========================================================================

class TestIsApproaching:
    def test_approaching_lt(self):
        # threshold=165, current=170, distance=5/165≈3% (within 10%)
        assert is_approaching(170.0, 165.0, "LT") is True

    def test_not_approaching_lt(self):
        # threshold=165, current=195, distance=30/165≈18% (outside 10%)
        assert is_approaching(195.0, 165.0, "LT") is False

    def test_approaching_gt(self):
        # threshold=70, current=66, distance=4/70≈5.7% (within 10%)
        assert is_approaching(66.0, 70.0, "GT") is True

    def test_none_value(self):
        assert is_approaching(None, 70.0, "GT") is False

    def test_already_breached_not_approaching(self):
        # Already past threshold
        assert is_approaching(75.0, 70.0, "GT") is False


# ===========================================================================
# InvalidMon Integration Tests
# ===========================================================================

class TestInvalidMon:
    @pytest.mark.asyncio
    async def test_process_no_breaches(self):
        agent = InvalidMon()
        ctx = _make_context(
            active_beliefs=[_make_belief()],
            current_values={"gross_margin_pct": 46.5, "close_price": 195.0},
        )
        result = await agent.process(ctx)
        assert isinstance(result, InvalidationMonitorOutput)
        assert result.conditions_breached == 0
        assert result.total_conditions == 2

    @pytest.mark.asyncio
    async def test_process_with_breach(self):
        agent = InvalidMon()
        ctx = _make_context(
            active_beliefs=[_make_belief(conditions=[
                {
                    "condition_id": str(uuid4()),
                    "metric": "gross_margin_pct",
                    "operator": "LT",
                    "threshold": 44.0,
                    "status": "ACTIVE",
                },
            ])],
            current_values={"gross_margin_pct": 42.0},
        )
        result = await agent.process(ctx)
        assert result.conditions_breached == 1
        assert result.conditions[0].is_breached is True
        assert result.conditions[0].confidence_impact > 0

    @pytest.mark.asyncio
    async def test_triggered_conditions_skipped(self):
        agent = InvalidMon()
        ctx = _make_context(
            active_beliefs=[_make_belief(conditions=[
                {
                    "condition_id": str(uuid4()),
                    "metric": "rsi_14",
                    "operator": "GT",
                    "threshold": 70.0,
                    "status": "TRIGGERED",
                },
            ])],
            current_values={"rsi_14": 75.0},
        )
        result = await agent.process(ctx)
        assert result.total_conditions == 0  # TRIGGERED skipped

    @pytest.mark.asyncio
    async def test_empty_beliefs(self):
        agent = InvalidMon()
        ctx = _make_context(active_beliefs=[], current_values={})
        result = await agent.process(ctx)
        assert result.total_conditions == 0

    @pytest.mark.asyncio
    async def test_content_hash(self):
        agent = InvalidMon()
        ctx = _make_context(
            active_beliefs=[_make_belief()],
            current_values={"gross_margin_pct": 46.5, "close_price": 195.0},
        )
        result = await agent.process(ctx)
        assert result.content_hash != ""
        assert len(result.content_hash) == 64

    @pytest.mark.asyncio
    async def test_approaching_counted(self):
        agent = InvalidMon()
        # gross_margin_pct at 44.2, threshold 44.0 → distance = 0.2/44 ≈ 0.45% (approaching)
        ctx = _make_context(
            active_beliefs=[_make_belief(conditions=[
                {
                    "condition_id": str(uuid4()),
                    "metric": "gross_margin_pct",
                    "operator": "LT",
                    "threshold": 44.0,
                    "status": "ACTIVE",
                },
            ])],
            current_values={"gross_margin_pct": 44.2},
        )
        result = await agent.process(ctx)
        # 44.2 is above 44.0 (not breached for LT), but within 10%
        assert result.conditions_breached == 0
        assert result.conditions_approaching == 1


class TestInvalidMonHealth:
    def test_healthy(self):
        agent = InvalidMon()
        health = agent.get_health()
        assert health.status == AgentStatus.HEALTHY

    def test_agent_id(self):
        agent = InvalidMon()
        assert agent.agent_id == "INVALID-MON"

    def test_degraded(self):
        agent = InvalidMon()
        agent._error_count_24h = 5
        assert agent.get_health().status == AgentStatus.DEGRADED


# ===========================================================================
# Schema Tests
# ===========================================================================

class TestExitSchemas:
    def test_exit_assessment_frozen(self):
        ea = ExitAssessment(
            ticker="AAPL",
            exit_action="HOLD",
            exit_confidence=0.20,
        )
        with pytest.raises(Exception):
            ea.ticker = "MSFT"

    def test_exit_output_content_hash(self):
        eo = ExitOutput(
            agent_id="COGNIT-EXIT",
            timestamp=NOW,
            context_window_hash="test",
            assessments=[
                ExitAssessment(
                    ticker="AAPL",
                    exit_action="HOLD",
                    exit_confidence=0.2,
                ),
            ],
            positions_hold=1,
        )
        assert eo.content_hash != ""

    def test_exit_output_timezone_required(self):
        with pytest.raises(Exception):
            ExitOutput(
                agent_id="TEST",
                timestamp=datetime(2026, 1, 1),  # No timezone
                context_window_hash="test",
            )

    def test_monitored_condition_frozen(self):
        mc = MonitoredCondition(
            condition_id=uuid4(),
            source_thesis_id="T1",
            source_agent_id="COGNIT-FUNDAMENTAL",
            ticker="AAPL",
            metric="rsi_14",
            operator="GT",
            threshold=70.0,
        )
        with pytest.raises(Exception):
            mc.is_breached = True

    def test_invalidation_output_content_hash(self):
        imo = InvalidationMonitorOutput(
            agent_id="INVALID-MON",
            timestamp=NOW,
            context_window_hash="test",
        )
        assert imo.content_hash != ""
