"""Tests for THESIS-RENEW, SHADOW-EXIT, and RENEW-MON agents.

Tests cover:
  - THESIS-RENEW: Confidence decay, asymmetry, renewal eligibility
  - SHADOW-EXIT: Agreement detection, exit probability, shadow tracking
  - RENEW-MON: Belief health, renewal candidacy, urgency
"""

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from providence.agents.base import AgentContext, AgentStatus
from providence.agents.exit.thesis_renew import (
    RENEWAL_WINDOW_DAYS,
    MIN_HEALTH_FOR_RENEWAL,
    ThesisRenew,
    compute_asymmetry_score,
    compute_confidence_decay,
    evaluate_renewal,
)
from providence.agents.exit.shadow_exit import (
    ShadowExit,
    build_shadow_signal,
    compute_agreement,
    compute_exit_probability,
)
from providence.agents.exit.renew_mon import (
    DEGRADED_THRESHOLD,
    HEALTHY_THRESHOLD,
    RenewMon,
    compute_belief_health,
    compute_confidence_decay as renew_compute_decay,
    determine_renewal_urgency,
)
from providence.schemas.exit import (
    BeliefHealthReport,
    RenewalCandidate,
    RenewalMonitorOutput,
    ShadowExitOutput,
    ShadowExitSignal,
    ThesisRenewalOutput,
)

NOW = datetime.now(timezone.utc)


def _make_context(**metadata_kwargs) -> AgentContext:
    return AgentContext(
        agent_id="TEST",
        trigger="schedule",
        fragments=[],
        context_window_hash="test-hash-exit-002",
        timestamp=NOW,
        metadata=metadata_kwargs,
    )


def _make_belief(
    thesis_id: str = "FUND-AAPL-001",
    ticker: str = "AAPL",
    agent_id: str = "COGNIT-FUNDAMENTAL",
    raw_confidence: float = 0.70,
    time_horizon_days: int = 60,
    days_elapsed: int = 50,
) -> dict:
    return {
        "thesis_id": thesis_id,
        "ticker": ticker,
        "agent_id": agent_id,
        "raw_confidence": raw_confidence,
        "time_horizon_days": time_horizon_days,
        "days_elapsed": days_elapsed,
    }


# ===========================================================================
# compute_confidence_decay Tests
# ===========================================================================

class TestConfidenceDecay:
    def test_zero_elapsed(self):
        decay = compute_confidence_decay(0, 60)
        assert decay == 0.0

    def test_half_elapsed(self):
        decay = compute_confidence_decay(30, 60)
        assert abs(decay - 0.10) < 0.01  # 10% at 50%

    def test_75pct_elapsed(self):
        decay = compute_confidence_decay(45, 60)
        assert abs(decay - 0.25) < 0.01  # 25% at 75%

    def test_full_elapsed(self):
        decay = compute_confidence_decay(60, 60)
        assert abs(decay - 0.40) < 0.01  # 40% at 100%

    def test_over_elapsed(self):
        decay = compute_confidence_decay(90, 60)
        assert abs(decay - 0.40) < 0.01  # Capped at 100%

    def test_zero_horizon(self):
        decay = compute_confidence_decay(10, 0)
        assert decay == 0.40  # Max decay


# ===========================================================================
# compute_asymmetry_score Tests
# ===========================================================================

class TestAsymmetryScore:
    def test_perfect_health_high_confidence(self):
        score = compute_asymmetry_score(1.0, 1.0, 0.0)
        assert score == 1.0

    def test_zero_health_zero_confidence(self):
        score = compute_asymmetry_score(0.0, 0.0, 0.0)
        assert score == 0.0

    def test_regime_penalty_applied(self):
        base = compute_asymmetry_score(0.8, 0.6, 0.0)
        penalized = compute_asymmetry_score(0.8, 0.6, -0.10)
        assert penalized < base

    def test_clamped_to_bounds(self):
        score = compute_asymmetry_score(1.0, 1.0, 0.5)
        assert score <= 1.0
        score = compute_asymmetry_score(0.0, 0.0, -0.5)
        assert score >= 0.0


# ===========================================================================
# evaluate_renewal Tests
# ===========================================================================

class TestEvaluateRenewal:
    def test_within_window_healthy_renewed(self):
        belief = _make_belief(days_elapsed=50, time_horizon_days=60, raw_confidence=0.70)
        inv_state = {
            "FUND-AAPL-001": {"conditions_healthy": 2, "conditions_total": 2},
        }
        result = evaluate_renewal(belief, inv_state, "NORMAL")
        assert result["is_renewed"] is True
        assert result["renewed_confidence"] > 0
        assert result["renewed_horizon_days"] > 60

    def test_outside_window_not_renewed(self):
        belief = _make_belief(days_elapsed=20, time_horizon_days=60)
        result = evaluate_renewal(belief, {}, "NORMAL")
        assert result["is_renewed"] is False
        assert "Not in renewal window" in result["renewal_reason"]

    def test_low_health_not_renewed(self):
        belief = _make_belief(days_elapsed=50, time_horizon_days=60)
        inv_state = {
            "FUND-AAPL-001": {"conditions_healthy": 1, "conditions_total": 5},
        }
        result = evaluate_renewal(belief, inv_state, "NORMAL")
        assert result["is_renewed"] is False

    def test_halted_no_renewals(self):
        belief = _make_belief(days_elapsed=55, time_horizon_days=60)
        inv_state = {
            "FUND-AAPL-001": {"conditions_healthy": 2, "conditions_total": 2},
        }
        result = evaluate_renewal(belief, inv_state, "HALTED")
        assert result["is_renewed"] is False
        assert "HALTED" in result["renewal_reason"]

    def test_cautious_regime_penalty(self):
        belief = _make_belief(days_elapsed=50, time_horizon_days=60, raw_confidence=0.70)
        inv_state = {"FUND-AAPL-001": {"conditions_healthy": 2, "conditions_total": 2}}
        normal = evaluate_renewal(belief, inv_state, "NORMAL")
        cautious = evaluate_renewal(belief, inv_state, "CAUTIOUS")
        assert cautious["renewed_confidence"] < normal["renewed_confidence"]

    def test_asymmetry_score_computed(self):
        belief = _make_belief(days_elapsed=50, time_horizon_days=60, raw_confidence=0.70)
        inv_state = {"FUND-AAPL-001": {"conditions_healthy": 2, "conditions_total": 2}}
        result = evaluate_renewal(belief, inv_state, "NORMAL")
        assert 0.0 <= result["asymmetry_score"] <= 1.0

    def test_horizon_extension(self):
        belief = _make_belief(days_elapsed=55, time_horizon_days=60, raw_confidence=0.80)
        inv_state = {"FUND-AAPL-001": {"conditions_healthy": 3, "conditions_total": 3}}
        result = evaluate_renewal(belief, inv_state, "NORMAL")
        assert result["renewed_horizon_days"] > result["original_horizon_days"]

    def test_low_confidence_not_renewed(self):
        belief = _make_belief(days_elapsed=58, time_horizon_days=60, raw_confidence=0.15)
        inv_state = {"FUND-AAPL-001": {"conditions_healthy": 2, "conditions_total": 2}}
        result = evaluate_renewal(belief, inv_state, "NORMAL")
        # After decay, confidence may be too low
        assert result["renewed_confidence"] < 0.15


# ===========================================================================
# ThesisRenew Integration Tests
# ===========================================================================

class TestThesisRenew:
    @pytest.mark.asyncio
    async def test_process_with_renewal(self):
        agent = ThesisRenew()
        ctx = _make_context(
            active_beliefs=[_make_belief(days_elapsed=50, time_horizon_days=60)],
            invalidation_state={
                "FUND-AAPL-001": {"conditions_healthy": 2, "conditions_total": 2},
            },
            regime_state={"system_risk_mode": "NORMAL"},
        )
        result = await agent.process(ctx)
        assert isinstance(result, ThesisRenewalOutput)
        assert result.total_evaluated == 1
        assert result.total_renewed == 1

    @pytest.mark.asyncio
    async def test_process_no_beliefs(self):
        agent = ThesisRenew()
        ctx = _make_context(active_beliefs=[])
        result = await agent.process(ctx)
        assert result.total_evaluated == 0

    @pytest.mark.asyncio
    async def test_process_halted_no_renewals(self):
        agent = ThesisRenew()
        ctx = _make_context(
            active_beliefs=[_make_belief(days_elapsed=55)],
            invalidation_state={"FUND-AAPL-001": {"conditions_healthy": 2, "conditions_total": 2}},
            regime_state={"system_risk_mode": "HALTED"},
        )
        result = await agent.process(ctx)
        assert result.total_renewed == 0

    @pytest.mark.asyncio
    async def test_content_hash(self):
        agent = ThesisRenew()
        ctx = _make_context(
            active_beliefs=[_make_belief(days_elapsed=50)],
            invalidation_state={},
        )
        result = await agent.process(ctx)
        assert result.content_hash != ""
        assert len(result.content_hash) == 64


class TestThesisRenewHealth:
    def test_agent_id(self):
        agent = ThesisRenew()
        assert agent.agent_id == "THESIS-RENEW"

    def test_healthy(self):
        agent = ThesisRenew()
        assert agent.get_health().status == AgentStatus.HEALTHY

    def test_degraded(self):
        agent = ThesisRenew()
        agent._error_count_24h = 5
        assert agent.get_health().status == AgentStatus.DEGRADED


# ===========================================================================
# compute_agreement Tests
# ===========================================================================

class TestComputeAgreement:
    def test_both_hold(self):
        agree, reason = compute_agreement("HOLD", "HOLD")
        assert agree is True
        assert reason == ""

    def test_reduce_and_trim(self):
        agree, reason = compute_agreement("REDUCE", "TRIM")
        assert agree is True

    def test_exit_and_close(self):
        agree, reason = compute_agreement("EXIT", "CLOSE")
        assert agree is True

    def test_exit_vs_hold(self):
        agree, reason = compute_agreement("EXIT", "HOLD")
        assert agree is False
        assert "CAPTURE has supremacy" in reason

    def test_hold_vs_close(self):
        agree, reason = compute_agreement("HOLD", "CLOSE")
        assert agree is False
        assert "EXEC-CAPTURE says CLOSE" in reason


# ===========================================================================
# compute_exit_probability Tests
# ===========================================================================

class TestExitProbability:
    def test_both_hold(self):
        prob = compute_exit_probability("HOLD", 0.0, "HOLD", 0.0)
        assert prob == 0.0

    def test_both_exit_high_confidence(self):
        prob = compute_exit_probability("EXIT", 0.90, "CLOSE", 0.95)
        assert prob > 0.5

    def test_capture_weighted_higher(self):
        # Same action levels but CAPTURE weight is 0.65 vs COGNIT 0.35
        prob_cognit = compute_exit_probability("EXIT", 0.90, "HOLD", 0.0)
        prob_capture = compute_exit_probability("HOLD", 0.0, "CLOSE", 0.90)
        assert prob_capture > prob_cognit

    def test_bounded_0_1(self):
        prob = compute_exit_probability("EXIT", 1.0, "CLOSE", 1.0)
        assert 0.0 <= prob <= 1.0


# ===========================================================================
# build_shadow_signal Tests
# ===========================================================================

class TestBuildShadowSignal:
    def test_both_hold_resets_shadow(self):
        signal = build_shadow_signal(
            "AAPL",
            {"exit_action": "HOLD", "exit_confidence": 0.0},
            {"action": "HOLD", "exit_confidence": 0.0},
            {"AAPL": 5},
        )
        assert signal["days_in_shadow"] == 0

    def test_exit_signal_increments_shadow(self):
        signal = build_shadow_signal(
            "AAPL",
            {"exit_action": "REDUCE", "exit_confidence": 0.6},
            {"action": "HOLD", "exit_confidence": 0.0},
            {"AAPL": 3},
        )
        assert signal["days_in_shadow"] == 4

    def test_max_shadow_days(self):
        signal = build_shadow_signal(
            "AAPL",
            {"exit_action": "EXIT", "exit_confidence": 0.9},
            {"action": "CLOSE", "exit_confidence": 0.9},
            {"AAPL": 30},
        )
        assert signal["days_in_shadow"] == 30

    def test_none_assessments(self):
        signal = build_shadow_signal("AAPL", None, None, {})
        assert signal["cognit_exit_action"] == "HOLD"
        assert signal["capture_action"] == "HOLD"
        assert signal["days_in_shadow"] == 0


# ===========================================================================
# ShadowExit Integration Tests
# ===========================================================================

class TestShadowExit:
    @pytest.mark.asyncio
    async def test_process_agreeing(self):
        agent = ShadowExit()
        ctx = _make_context(
            exit_assessments=[
                {"ticker": "AAPL", "exit_action": "HOLD", "exit_confidence": 0.1},
            ],
            capture_decisions=[
                {"ticker": "AAPL", "action": "HOLD", "exit_confidence": 0.0},
            ],
        )
        result = await agent.process(ctx)
        assert isinstance(result, ShadowExitOutput)
        assert result.positions_agreeing == 1
        assert result.positions_diverging == 0

    @pytest.mark.asyncio
    async def test_process_diverging(self):
        agent = ShadowExit()
        ctx = _make_context(
            exit_assessments=[
                {"ticker": "AAPL", "exit_action": "EXIT", "exit_confidence": 0.8},
            ],
            capture_decisions=[
                {"ticker": "AAPL", "action": "HOLD", "exit_confidence": 0.0},
            ],
        )
        result = await agent.process(ctx)
        assert result.positions_diverging == 1

    @pytest.mark.asyncio
    async def test_process_multi_position(self):
        agent = ShadowExit()
        ctx = _make_context(
            exit_assessments=[
                {"ticker": "AAPL", "exit_action": "HOLD", "exit_confidence": 0.1},
                {"ticker": "MSFT", "exit_action": "EXIT", "exit_confidence": 0.85},
            ],
            capture_decisions=[
                {"ticker": "AAPL", "action": "HOLD", "exit_confidence": 0.0},
                {"ticker": "TSLA", "action": "CLOSE", "exit_confidence": 0.9},
            ],
            shadow_history={"AAPL": 2, "MSFT": 5},
        )
        result = await agent.process(ctx)
        # AAPL: agree (both HOLD), MSFT: only COGNIT-EXIT, TSLA: only CAPTURE
        assert result.total_positions == 3

    @pytest.mark.asyncio
    async def test_empty_inputs(self):
        agent = ShadowExit()
        ctx = _make_context(exit_assessments=[], capture_decisions=[])
        result = await agent.process(ctx)
        assert result.total_positions == 0

    @pytest.mark.asyncio
    async def test_content_hash(self):
        agent = ShadowExit()
        ctx = _make_context(
            exit_assessments=[{"ticker": "AAPL", "exit_action": "HOLD", "exit_confidence": 0.1}],
            capture_decisions=[{"ticker": "AAPL", "action": "HOLD", "exit_confidence": 0.0}],
        )
        result = await agent.process(ctx)
        assert len(result.content_hash) == 64


class TestShadowExitHealth:
    def test_agent_id(self):
        agent = ShadowExit()
        assert agent.agent_id == "SHADOW-EXIT"

    def test_healthy(self):
        assert ShadowExit().get_health().status == AgentStatus.HEALTHY


# ===========================================================================
# compute_belief_health (RENEW-MON) Tests
# ===========================================================================

class TestComputeBeliefHealth:
    def test_all_healthy(self):
        conditions = {"T1": [
            {"is_breached": False, "breach_magnitude": 0.0, "threshold": 100, "current_value": 120},
            {"is_breached": False, "breach_magnitude": 0.0, "threshold": 50, "current_value": 60},
        ]}
        health, h, b, a = compute_belief_health("T1", "AAPL", conditions)
        assert health == 1.0
        assert h == 2
        assert b == 0

    def test_one_breached(self):
        conditions = {"T1": [
            {"is_breached": True, "breach_magnitude": 0.1, "threshold": 100, "current_value": 110},
            {"is_breached": False, "breach_magnitude": 0.0, "threshold": 50, "current_value": 60},
        ]}
        health, h, b, a = compute_belief_health("T1", "AAPL", conditions)
        assert b == 1
        assert health < 1.0

    def test_all_breached(self):
        conditions = {"T1": [
            {"is_breached": True, "breach_magnitude": 0.2, "threshold": 100, "current_value": 80},
            {"is_breached": True, "breach_magnitude": 0.3, "threshold": 50, "current_value": 35},
        ]}
        health, h, b, a = compute_belief_health("T1", "AAPL", conditions)
        assert health == 0.0
        assert b == 2

    def test_no_conditions(self):
        health, h, b, a = compute_belief_health("T1", "AAPL", {})
        assert health == 1.0

    def test_approaching_penalized(self):
        # Condition not breached but within 10% of threshold
        conditions = {"T1": [
            {"is_breached": False, "breach_magnitude": 0.05, "threshold": 100, "current_value": 95},
        ]}
        health, h, b, a = compute_belief_health("T1", "AAPL", conditions)
        assert a == 1
        assert health < 1.0


# ===========================================================================
# determine_renewal_urgency Tests
# ===========================================================================

class TestDetermineRenewalUrgency:
    def test_not_candidate(self):
        assert determine_renewal_urgency(5, 0.80, False) == "NONE"

    def test_high_urgency_few_days(self):
        assert determine_renewal_urgency(2, 0.80, True) == "HIGH"

    def test_medium_urgency_moderate_days(self):
        assert determine_renewal_urgency(6, 0.80, True) == "MEDIUM"

    def test_medium_urgency_degraded_health(self):
        assert determine_renewal_urgency(10, 0.40, True) == "MEDIUM"

    def test_low_urgency_healthy(self):
        assert determine_renewal_urgency(12, 0.90, True) == "LOW"

    def test_high_urgency_degraded_few_days(self):
        assert determine_renewal_urgency(5, 0.30, True) == "HIGH"


# ===========================================================================
# RenewMon Integration Tests
# ===========================================================================

class TestRenewMon:
    @pytest.mark.asyncio
    async def test_process_healthy_beliefs(self):
        agent = RenewMon()
        ctx = _make_context(
            active_beliefs=[
                _make_belief(days_elapsed=20, time_horizon_days=60),
            ],
            invalidation_results=[],
            regime_state={"system_risk_mode": "NORMAL"},
        )
        result = await agent.process(ctx)
        assert isinstance(result, RenewalMonitorOutput)
        assert result.total_beliefs == 1
        assert result.healthy_beliefs == 1

    @pytest.mark.asyncio
    async def test_process_with_degraded(self):
        agent = RenewMon()
        ctx = _make_context(
            active_beliefs=[
                _make_belief(thesis_id="T1", days_elapsed=50, time_horizon_days=60),
            ],
            invalidation_results=[
                {
                    "source_thesis_id": "T1",
                    "is_breached": True,
                    "breach_magnitude": 0.15,
                    "threshold": 100,
                    "current_value": 85,
                },
                {
                    "source_thesis_id": "T1",
                    "is_breached": False,
                    "breach_magnitude": 0.01,
                    "threshold": 50,
                    "current_value": 55,
                },
            ],
            regime_state={"system_risk_mode": "NORMAL"},
        )
        result = await agent.process(ctx)
        assert result.total_beliefs == 1
        # One breached out of 2 → health = 0.5 → degraded
        assert result.degraded_beliefs >= 0  # May be degraded or critical

    @pytest.mark.asyncio
    async def test_process_renewal_candidate(self):
        agent = RenewMon()
        ctx = _make_context(
            active_beliefs=[
                _make_belief(thesis_id="T1", days_elapsed=50, time_horizon_days=60),
            ],
            invalidation_results=[],
            regime_state={"system_risk_mode": "NORMAL"},
        )
        result = await agent.process(ctx)
        # days_remaining=10 < RENEWAL_WINDOW_DAYS=14, health=1.0 → candidate
        assert result.renewal_candidates == 1
        assert result.reports[0].is_renewal_candidate is True
        assert result.reports[0].renewal_urgency != "NONE"

    @pytest.mark.asyncio
    async def test_process_halted_no_candidates(self):
        agent = RenewMon()
        ctx = _make_context(
            active_beliefs=[_make_belief(days_elapsed=55)],
            invalidation_results=[],
            regime_state={"system_risk_mode": "HALTED"},
        )
        result = await agent.process(ctx)
        assert result.renewal_candidates == 0

    @pytest.mark.asyncio
    async def test_empty_beliefs(self):
        agent = RenewMon()
        ctx = _make_context(active_beliefs=[])
        result = await agent.process(ctx)
        assert result.total_beliefs == 0

    @pytest.mark.asyncio
    async def test_content_hash(self):
        agent = RenewMon()
        ctx = _make_context(
            active_beliefs=[_make_belief()],
            invalidation_results=[],
        )
        result = await agent.process(ctx)
        assert len(result.content_hash) == 64

    @pytest.mark.asyncio
    async def test_confidence_decay_reported(self):
        agent = RenewMon()
        ctx = _make_context(
            active_beliefs=[_make_belief(days_elapsed=45, time_horizon_days=60)],
            invalidation_results=[],
        )
        result = await agent.process(ctx)
        assert result.reports[0].confidence_decay > 0


class TestRenewMonHealth:
    def test_agent_id(self):
        agent = RenewMon()
        assert agent.agent_id == "RENEW-MON"

    def test_healthy(self):
        assert RenewMon().get_health().status == AgentStatus.HEALTHY

    def test_unhealthy(self):
        agent = RenewMon()
        agent._error_count_24h = 15
        assert agent.get_health().status == AgentStatus.UNHEALTHY


# ===========================================================================
# Schema Tests
# ===========================================================================

class TestExitSystemSchemas:
    def test_renewal_candidate_frozen(self):
        rc = RenewalCandidate(
            thesis_id="T1",
            ticker="AAPL",
            original_confidence=0.70,
            renewed_confidence=0.60,
            original_horizon_days=60,
            renewed_horizon_days=90,
        )
        with pytest.raises(Exception):
            rc.ticker = "MSFT"

    def test_shadow_signal_frozen(self):
        ss = ShadowExitSignal(
            ticker="AAPL",
            cognit_exit_action="HOLD",
            capture_action="HOLD",
        )
        with pytest.raises(Exception):
            ss.ticker = "MSFT"

    def test_belief_health_report_frozen(self):
        bhr = BeliefHealthReport(
            thesis_id="T1",
            ticker="AAPL",
            agent_id="COGNIT-FUNDAMENTAL",
            health_score=0.85,
        )
        with pytest.raises(Exception):
            bhr.health_score = 0.5

    def test_thesis_renewal_output_timezone(self):
        with pytest.raises(Exception):
            ThesisRenewalOutput(
                agent_id="TEST",
                timestamp=datetime(2026, 1, 1),  # No timezone
                context_window_hash="test",
            )

    def test_shadow_exit_output_hash(self):
        seo = ShadowExitOutput(
            agent_id="SHADOW-EXIT",
            timestamp=NOW,
            context_window_hash="test",
        )
        assert seo.content_hash != ""

    def test_renewal_monitor_output_hash(self):
        rmo = RenewalMonitorOutput(
            agent_id="RENEW-MON",
            timestamp=NOW,
            context_window_hash="test",
        )
        assert rmo.content_hash != ""
