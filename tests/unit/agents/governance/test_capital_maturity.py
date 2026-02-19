"""Tests for GOVERN-CAPITAL and GOVERN-MATURITY agents.

Tests cover:
  - GOVERN-CAPITAL: Tier classification, constraint derivation, headroom computation
  - GOVERN-MATURITY: Promotion eligibility, blocker detection, confidence weights
"""

from datetime import datetime, timezone

import pytest

from providence.agents.base import AgentContext, AgentStatus
from providence.agents.governance.capital import (
    TIER_CONSTRAINTS,
    TIER_THRESHOLDS,
    GovernCapital,
    classify_tier,
    compute_headroom,
    get_tier_constraints,
)
from providence.agents.governance.maturity import (
    LIMITED_MAX_BRIER,
    LIMITED_MIN_DAYS,
    LIMITED_MIN_HIT_RATE,
    SHADOW_MAX_BRIER,
    SHADOW_MIN_DAYS,
    SHADOW_MIN_HIT_RATE,
    STAGE_WEIGHTS,
    GovernMaturity,
    evaluate_promotion,
)
from providence.schemas.enums import CapitalTier, MaturityStage
from providence.schemas.governance import (
    AgentMaturityRecord,
    CapitalTierOutput,
    MaturityGateOutput,
    TierConstraints,
)

NOW = datetime.now(timezone.utc)


def _make_context(**metadata_kwargs) -> AgentContext:
    return AgentContext(
        agent_id="TEST",
        trigger="manual",
        fragments=[],
        context_window_hash="test-hash-gov-001",
        timestamp=NOW,
        metadata=metadata_kwargs,
    )


# ===========================================================================
# classify_tier Tests
# ===========================================================================

class TestClassifyTier:
    def test_seed(self):
        assert classify_tier(0.0) == CapitalTier.SEED
        assert classify_tier(5_000_000.0) == CapitalTier.SEED
        assert classify_tier(9_999_999.0) == CapitalTier.SEED

    def test_growth(self):
        assert classify_tier(10_000_000.0) == CapitalTier.GROWTH
        assert classify_tier(50_000_000.0) == CapitalTier.GROWTH
        assert classify_tier(99_999_999.0) == CapitalTier.GROWTH

    def test_scale(self):
        assert classify_tier(100_000_000.0) == CapitalTier.SCALE
        assert classify_tier(300_000_000.0) == CapitalTier.SCALE
        assert classify_tier(499_999_999.0) == CapitalTier.SCALE

    def test_institutional(self):
        assert classify_tier(500_000_000.0) == CapitalTier.INSTITUTIONAL
        assert classify_tier(1_000_000_000.0) == CapitalTier.INSTITUTIONAL


# ===========================================================================
# get_tier_constraints Tests
# ===========================================================================

class TestGetTierConstraints:
    def test_seed_no_execution(self):
        c = get_tier_constraints(CapitalTier.SEED)
        assert c.live_execution_enabled is False
        assert c.max_positions == 0

    def test_growth_limited(self):
        c = get_tier_constraints(CapitalTier.GROWTH)
        assert c.live_execution_enabled is True
        assert c.max_position_weight == 0.05
        assert c.max_positions == 15

    def test_scale_standard(self):
        c = get_tier_constraints(CapitalTier.SCALE)
        assert c.max_position_weight == 0.10
        assert c.max_positions == 30

    def test_institutional_enhanced(self):
        c = get_tier_constraints(CapitalTier.INSTITUTIONAL)
        assert c.max_positions == 50
        assert c.live_execution_enabled is True

    def test_all_tiers_have_constraints(self):
        for tier in CapitalTier:
            c = get_tier_constraints(tier)
            assert isinstance(c, TierConstraints)


# ===========================================================================
# compute_headroom Tests
# ===========================================================================

class TestComputeHeadroom:
    def test_seed_halfway(self):
        headroom = compute_headroom(5_000_000.0, CapitalTier.SEED)
        assert headroom == 50.0  # 5M / 10M = 50%

    def test_seed_start(self):
        assert compute_headroom(0.0, CapitalTier.SEED) == 0.0

    def test_institutional_top(self):
        assert compute_headroom(1_000_000_000.0, CapitalTier.INSTITUTIONAL) == 100.0

    def test_growth_near_scale(self):
        headroom = compute_headroom(90_000_000.0, CapitalTier.GROWTH)
        # (90M - 10M) / (100M - 10M) * 100 = 88.89%
        assert headroom == pytest.approx(88.89, rel=1e-2)


# ===========================================================================
# GovernCapital Integration Tests
# ===========================================================================

class TestGovernCapital:
    @pytest.mark.asyncio
    async def test_process_seed(self):
        agent = GovernCapital()
        ctx = _make_context(current_aum=5_000_000.0)
        result = await agent.process(ctx)
        assert isinstance(result, CapitalTierOutput)
        assert result.current_tier == CapitalTier.SEED
        assert result.constraints.live_execution_enabled is False
        assert result.tier_changed is False

    @pytest.mark.asyncio
    async def test_process_growth(self):
        agent = GovernCapital()
        ctx = _make_context(current_aum=50_000_000.0)
        result = await agent.process(ctx)
        assert result.current_tier == CapitalTier.GROWTH
        assert result.constraints.live_execution_enabled is True

    @pytest.mark.asyncio
    async def test_tier_change_detected(self):
        agent = GovernCapital()
        ctx = _make_context(current_aum=50_000_000.0, previous_tier="SEED")
        result = await agent.process(ctx)
        assert result.tier_changed is True
        assert result.previous_tier == CapitalTier.SEED
        assert result.current_tier == CapitalTier.GROWTH

    @pytest.mark.asyncio
    async def test_no_tier_change(self):
        agent = GovernCapital()
        ctx = _make_context(current_aum=50_000_000.0, previous_tier="GROWTH")
        result = await agent.process(ctx)
        assert result.tier_changed is False

    @pytest.mark.asyncio
    async def test_content_hash(self):
        agent = GovernCapital()
        ctx = _make_context(current_aum=100_000_000.0)
        result = await agent.process(ctx)
        assert len(result.content_hash) == 64

    @pytest.mark.asyncio
    async def test_zero_aum(self):
        agent = GovernCapital()
        ctx = _make_context(current_aum=0.0)
        result = await agent.process(ctx)
        assert result.current_tier == CapitalTier.SEED


class TestGovernCapitalHealth:
    def test_agent_id(self):
        assert GovernCapital().agent_id == "GOVERN-CAPITAL"

    def test_healthy(self):
        assert GovernCapital().get_health().status == AgentStatus.HEALTHY

    def test_degraded(self):
        agent = GovernCapital()
        agent._error_count_24h = 5
        assert agent.get_health().status == AgentStatus.DEGRADED


# ===========================================================================
# evaluate_promotion Tests
# ===========================================================================

class TestEvaluatePromotion:
    def test_shadow_eligible(self):
        result = evaluate_promotion(
            "AGENT-A", MaturityStage.SHADOW, 45,
            {"hit_rate": 0.60}, {"overall_brier_score": 0.20}, {},
        )
        assert result["promotion_eligible"] is True
        assert result["confidence_weight"] == 0.0

    def test_shadow_too_early(self):
        result = evaluate_promotion(
            "AGENT-A", MaturityStage.SHADOW, 15,
            {"hit_rate": 0.60}, {"overall_brier_score": 0.20}, {},
        )
        assert result["promotion_eligible"] is False
        assert any("more days" in b for b in result["promotion_blockers"])

    def test_shadow_low_hit_rate(self):
        result = evaluate_promotion(
            "AGENT-A", MaturityStage.SHADOW, 45,
            {"hit_rate": 0.35}, {"overall_brier_score": 0.20}, {},
        )
        assert result["promotion_eligible"] is False
        assert any("hit rate" in b.lower() for b in result["promotion_blockers"])

    def test_shadow_high_brier(self):
        result = evaluate_promotion(
            "AGENT-A", MaturityStage.SHADOW, 45,
            {"hit_rate": 0.60}, {"overall_brier_score": 0.40}, {},
        )
        assert result["promotion_eligible"] is False
        assert any("brier" in b.lower() for b in result["promotion_blockers"])

    def test_limited_eligible(self):
        result = evaluate_promotion(
            "AGENT-A", MaturityStage.LIMITED, 90,
            {"hit_rate": 0.65}, {"overall_brier_score": 0.15}, {},
        )
        assert result["promotion_eligible"] is True
        assert result["confidence_weight"] == 0.5

    def test_limited_critical_retrain_blocks(self):
        result = evaluate_promotion(
            "AGENT-A", MaturityStage.LIMITED, 90,
            {"hit_rate": 0.65}, {"overall_brier_score": 0.15},
            {"priority": "CRITICAL"},
        )
        assert result["promotion_eligible"] is False
        assert any("CRITICAL" in b for b in result["promotion_blockers"])

    def test_full_no_promotion(self):
        result = evaluate_promotion(
            "AGENT-A", MaturityStage.FULL, 180,
            {"hit_rate": 0.70}, {"overall_brier_score": 0.10}, {},
        )
        assert result["promotion_eligible"] is False  # Already at top
        assert result["confidence_weight"] == 1.0

    def test_multiple_blockers(self):
        result = evaluate_promotion(
            "AGENT-A", MaturityStage.SHADOW, 10,
            {"hit_rate": 0.30}, {"overall_brier_score": 0.40}, {},
        )
        assert result["promotion_eligible"] is False
        assert len(result["promotion_blockers"]) == 3


# ===========================================================================
# GovernMaturity Integration Tests
# ===========================================================================

class TestGovernMaturity:
    @pytest.mark.asyncio
    async def test_process_basic(self):
        agent = GovernMaturity()
        ctx = _make_context(
            agent_maturity_state=[
                {"agent_id": "COGNIT-FUNDAMENTAL", "current_stage": "SHADOW", "days_in_stage": 45},
                {"agent_id": "COGNIT-EXIT", "current_stage": "FULL", "days_in_stage": 200},
            ],
            attribution_results=[
                {"agent_id": "COGNIT-FUNDAMENTAL", "hit_rate": 0.60},
            ],
            calibration_results=[
                {"agent_id": "COGNIT-FUNDAMENTAL", "overall_brier_score": 0.20},
            ],
        )
        result = await agent.process(ctx)
        assert isinstance(result, MaturityGateOutput)
        assert len(result.agent_records) == 2
        assert result.agents_in_shadow == 1
        assert result.agents_in_full == 1

    @pytest.mark.asyncio
    async def test_process_empty(self):
        agent = GovernMaturity()
        ctx = _make_context(agent_maturity_state=[])
        result = await agent.process(ctx)
        assert len(result.agent_records) == 0
        assert result.promotions_recommended == 0

    @pytest.mark.asyncio
    async def test_promotion_counted(self):
        agent = GovernMaturity()
        ctx = _make_context(
            agent_maturity_state=[
                {"agent_id": "A", "current_stage": "SHADOW", "days_in_stage": 45},
            ],
            attribution_results=[{"agent_id": "A", "hit_rate": 0.65}],
            calibration_results=[{"agent_id": "A", "overall_brier_score": 0.15}],
        )
        result = await agent.process(ctx)
        assert result.promotions_recommended == 1

    @pytest.mark.asyncio
    async def test_content_hash(self):
        agent = GovernMaturity()
        ctx = _make_context(agent_maturity_state=[
            {"agent_id": "A", "current_stage": "SHADOW", "days_in_stage": 10},
        ])
        result = await agent.process(ctx)
        assert len(result.content_hash) == 64

    @pytest.mark.asyncio
    async def test_invalid_stage_defaults_shadow(self):
        agent = GovernMaturity()
        ctx = _make_context(agent_maturity_state=[
            {"agent_id": "A", "current_stage": "INVALID", "days_in_stage": 10},
        ])
        result = await agent.process(ctx)
        assert result.agent_records[0].current_stage == MaturityStage.SHADOW


class TestGovernMaturityHealth:
    def test_agent_id(self):
        assert GovernMaturity().agent_id == "GOVERN-MATURITY"

    def test_healthy(self):
        assert GovernMaturity().get_health().status == AgentStatus.HEALTHY

    def test_unhealthy(self):
        agent = GovernMaturity()
        agent._error_count_24h = 15
        assert agent.get_health().status == AgentStatus.UNHEALTHY


# ===========================================================================
# Schema Tests
# ===========================================================================

class TestGovernanceSchemas:
    def test_tier_constraints_frozen(self):
        tc = TierConstraints(
            max_position_weight=0.10, max_gross_exposure=1.20,
            max_single_sector_pct=0.30, min_confidence_threshold=0.60,
            live_execution_enabled=True, max_positions=30,
        )
        with pytest.raises(Exception):
            tc.max_positions = 50

    def test_capital_output_timezone(self):
        with pytest.raises(Exception):
            CapitalTierOutput(
                agent_id="TEST",
                timestamp=datetime(2026, 1, 1),  # No tz
                context_window_hash="test",
                current_aum=100.0,
                current_tier=CapitalTier.SEED,
                constraints=get_tier_constraints(CapitalTier.SEED),
            )

    def test_maturity_record_frozen(self):
        mr = AgentMaturityRecord(
            agent_id="TEST", current_stage=MaturityStage.SHADOW,
        )
        with pytest.raises(Exception):
            mr.current_stage = MaturityStage.FULL

    def test_maturity_output_hash(self):
        mo = MaturityGateOutput(
            agent_id="TEST", timestamp=NOW, context_window_hash="test",
        )
        assert mo.content_hash != ""

    def test_capital_tier_enum_values(self):
        assert CapitalTier.SEED.value == "SEED"
        assert CapitalTier.GROWTH.value == "GROWTH"
        assert CapitalTier.SCALE.value == "SCALE"
        assert CapitalTier.INSTITUTIONAL.value == "INSTITUTIONAL"

    def test_maturity_stage_enum_values(self):
        assert MaturityStage.SHADOW.value == "SHADOW"
        assert MaturityStage.LIMITED.value == "LIMITED"
        assert MaturityStage.FULL.value == "FULL"
