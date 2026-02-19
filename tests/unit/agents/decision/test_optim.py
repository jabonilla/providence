"""Tests for DECIDE-OPTIM Portfolio Optimization Agent.

Tests the full optimization pipeline: Black-Litterman core functions,
exposure/sector/position constraints, intent filtering, portfolio
metadata computation, error handling, and health reporting.

DECIDE-OPTIM is FROZEN: zero LLM calls, pure computation.
"""

import math
from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest

from providence.agents.base import AgentContext, AgentStatus
from providence.agents.decision.optim import (
    DecideOptim,
    RISK_MODE_LIMITS,
    apply_exposure_limits,
    apply_position_limits,
    black_litterman_weights,
    compute_equilibrium_weights,
    compute_sector_concentrations,
    compute_view_confidence_matrix,
    enforce_sector_limits,
    estimate_sharpe,
    intent_to_action,
)
from providence.exceptions import AgentProcessingError
from providence.schemas.decision import (
    PortfolioMetadata,
    PositionProposal,
    ProposedPosition,
)
from providence.schemas.enums import Action, Direction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
NOW = datetime.now(timezone.utc)


def _make_intent(
    ticker: str = "AAPL",
    direction: str = "LONG",
    confidence: float = 0.60,
    time_horizon: int = 60,
    regime_adj: float = -0.05,
    intent_id: str | None = None,
) -> dict:
    """Create a serialized SynthesizedPositionIntent dict."""
    return {
        "intent_id": intent_id or str(uuid4()),
        "ticker": ticker,
        "net_direction": direction,
        "synthesized_confidence": confidence,
        "time_horizon_days": time_horizon,
        "regime_adjustment": regime_adj,
    }


def _make_regime(
    risk_mode: str = "NORMAL",
    statistical_regime: str = "LOW_VOL_TRENDING",
) -> dict:
    return {
        "statistical_regime": statistical_regime,
        "regime_confidence": 0.75,
        "system_risk_mode": risk_mode,
        "features_used": {"realized_vol": 0.12},
    }


def _make_context(
    intents: list[dict] | None = None,
    regime: dict | None = None,
) -> AgentContext:
    metadata = {}
    if intents is not None:
        metadata["position_intents"] = intents
    if regime is not None:
        metadata["regime_state"] = regime
    return AgentContext(
        agent_id="DECIDE-OPTIM",
        trigger="schedule",
        fragments=[],
        context_window_hash="test-hash-optim-001",
        timestamp=NOW,
        metadata=metadata,
    )


# ===========================================================================
# Tests: equilibrium weights
# ===========================================================================
class TestEquilibriumWeights:
    """Tests for compute_equilibrium_weights."""

    def test_single_asset(self):
        w = compute_equilibrium_weights(1)
        assert len(w) == 1
        assert w[0] == pytest.approx(1.0)

    def test_three_assets(self):
        w = compute_equilibrium_weights(3)
        assert len(w) == 3
        for wi in w:
            assert wi == pytest.approx(1.0 / 3.0, abs=1e-4)

    def test_zero_assets(self):
        assert compute_equilibrium_weights(0) == []

    def test_negative_assets(self):
        assert compute_equilibrium_weights(-1) == []

    def test_weights_sum_to_one(self):
        w = compute_equilibrium_weights(10)
        assert sum(w) == pytest.approx(1.0, abs=1e-4)


# ===========================================================================
# Tests: view confidence matrix
# ===========================================================================
class TestViewConfidenceMatrix:
    """Tests for compute_view_confidence_matrix."""

    def test_high_confidence_low_omega(self):
        omega = compute_view_confidence_matrix([0.90], tau=0.05)
        assert len(omega) == 1
        # High confidence → low uncertainty
        assert omega[0] < 0.10

    def test_low_confidence_high_omega(self):
        omega = compute_view_confidence_matrix([0.10], tau=0.05)
        assert len(omega) == 1
        # Low confidence → high uncertainty
        assert omega[0] > 0.30

    def test_confidence_clamped_to_floor(self):
        omega = compute_view_confidence_matrix([0.01], tau=0.05)
        # Should use 0.05 (floor) not 0.01
        expected = 0.05 / 0.05
        assert omega[0] == pytest.approx(expected, abs=1e-4)

    def test_multiple_views(self):
        omega = compute_view_confidence_matrix([0.5, 0.8], tau=0.05)
        assert len(omega) == 2
        assert omega[0] > omega[1]  # Lower confidence → higher uncertainty


# ===========================================================================
# Tests: Black-Litterman weights
# ===========================================================================
class TestBlackLittermanWeights:
    """Tests for black_litterman_weights."""

    def test_single_long_view(self):
        prior = [1.0]
        directions = [1.0]
        confidences = [0.70]
        weights = black_litterman_weights(prior, directions, confidences)
        assert len(weights) == 1
        assert weights[0] > 0  # Should be positive (LONG)

    def test_single_short_view(self):
        prior = [1.0]
        directions = [-1.0]
        confidences = [0.70]
        weights = black_litterman_weights(prior, directions, confidences)
        assert len(weights) == 1
        assert weights[0] < 0  # Should be negative (SHORT)

    def test_mixed_views(self):
        prior = compute_equilibrium_weights(2)
        directions = [1.0, -1.0]
        confidences = [0.70, 0.60]
        weights = black_litterman_weights(prior, directions, confidences)
        assert len(weights) == 2
        assert weights[0] > 0  # LONG
        assert weights[1] < 0  # SHORT

    def test_empty_assets(self):
        assert black_litterman_weights([], [], []) == []

    def test_higher_confidence_gets_more_weight(self):
        prior = compute_equilibrium_weights(2)
        # Both LONG, but first has higher confidence
        directions = [1.0, 1.0]
        confidences = [0.80, 0.40]
        weights = black_litterman_weights(prior, directions, confidences)
        assert abs(weights[0]) > abs(weights[1])

    def test_weights_normalized(self):
        prior = compute_equilibrium_weights(3)
        directions = [1.0, -1.0, 1.0]
        confidences = [0.60, 0.50, 0.70]
        weights = black_litterman_weights(prior, directions, confidences)
        abs_sum = sum(abs(w) for w in weights)
        assert abs_sum == pytest.approx(1.0, abs=0.01)


# ===========================================================================
# Tests: position limits
# ===========================================================================
class TestPositionLimits:
    """Tests for apply_position_limits."""

    def test_weights_within_limits(self):
        weights = [0.05, -0.03, 0.08]
        clamped = apply_position_limits(weights, max_weight=0.10)
        assert clamped == [0.05, -0.03, 0.08]

    def test_long_position_clamped(self):
        weights = [0.15, 0.05]
        clamped = apply_position_limits(weights, max_weight=0.10)
        assert clamped[0] == 0.10
        assert clamped[1] == 0.05

    def test_short_position_clamped(self):
        weights = [-0.15, 0.05]
        clamped = apply_position_limits(weights, max_weight=0.10)
        assert clamped[0] == -0.10

    def test_empty_list(self):
        assert apply_position_limits([], max_weight=0.10) == []


# ===========================================================================
# Tests: exposure limits
# ===========================================================================
class TestExposureLimits:
    """Tests for apply_exposure_limits."""

    def test_within_limits(self):
        weights = [0.30, -0.20, 0.25]
        result = apply_exposure_limits(weights, max_gross=1.6, max_net=0.8)
        # Gross = 0.75, net = 0.35 → no scaling needed
        assert len(result) == 3

    def test_gross_exposure_scaled(self):
        weights = [0.60, -0.50, 0.50]  # Gross = 1.60
        result = apply_exposure_limits(weights, max_gross=0.80, max_net=0.80)
        gross = sum(abs(w) for w in result)
        assert gross <= 0.80 + 0.001

    def test_net_exposure_scaled(self):
        weights = [0.50, 0.50, 0.10]  # Net = 1.10
        result = apply_exposure_limits(weights, max_gross=2.0, max_net=0.80)
        net = sum(result)
        assert abs(net) <= 0.80 + 0.001

    def test_empty_list(self):
        assert apply_exposure_limits([], max_gross=1.0, max_net=0.5) == []


# ===========================================================================
# Tests: sector concentrations
# ===========================================================================
class TestSectorConcentrations:
    """Tests for compute_sector_concentrations."""

    def test_known_tickers(self):
        conc = compute_sector_concentrations(
            ["AAPL", "MSFT", "JPM"],
            [0.10, 0.08, 0.05],
        )
        assert "Information Technology" in conc or "Technology" in conc or len(conc) > 0

    def test_unknown_ticker(self):
        conc = compute_sector_concentrations(["UNKNOWN_TICKER"], [0.10])
        assert "Unknown" in conc

    def test_empty_list(self):
        conc = compute_sector_concentrations([], [])
        assert conc == {}


# ===========================================================================
# Tests: sector limits
# ===========================================================================
class TestSectorLimits:
    """Tests for enforce_sector_limits."""

    def test_within_limits(self):
        tickers = ["AAPL", "JPM"]
        weights = [0.10, 0.10]
        result = enforce_sector_limits(tickers, weights, max_sector_concentration=0.35)
        # Different sectors, both under limit → no change
        for w_in, w_out in zip(weights, result):
            assert abs(w_in - w_out) < 0.001

    def test_sector_scaled_down(self):
        # Two tech stocks exceeding sector limit
        tickers = ["AAPL", "MSFT"]
        weights = [0.20, 0.20]  # Both tech → 0.40 total
        result = enforce_sector_limits(tickers, weights, max_sector_concentration=0.30)
        sector_total = abs(result[0]) + abs(result[1])
        assert sector_total <= 0.30 + 0.001


# ===========================================================================
# Tests: Sharpe estimation
# ===========================================================================
class TestEstimateSharpe:
    """Tests for estimate_sharpe."""

    def test_aligned_positions(self):
        sharpe = estimate_sharpe(
            weights=[0.10, 0.08],
            confidences=[0.70, 0.65],
            directions=[1.0, 1.0],
        )
        assert sharpe > 0.5  # Good alignment

    def test_misaligned_positions(self):
        sharpe = estimate_sharpe(
            weights=[0.10, -0.08],
            confidences=[0.70, 0.65],
            directions=[1.0, 1.0],  # Both views LONG but second weight SHORT
        )
        # Partial misalignment → lower Sharpe
        aligned = estimate_sharpe([0.10, 0.08], [0.70, 0.65], [1.0, 1.0])
        assert sharpe < aligned

    def test_empty_portfolio(self):
        assert estimate_sharpe([], [], []) == 0.0

    def test_sharpe_capped_at_two(self):
        sharpe = estimate_sharpe([1.0], [0.99], [1.0])
        assert sharpe <= 2.0


# ===========================================================================
# Tests: intent_to_action
# ===========================================================================
class TestIntentToAction:
    """Tests for intent_to_action."""

    def test_long_positive(self):
        assert intent_to_action("LONG", 0.10) == Action.OPEN_LONG

    def test_short_negative(self):
        assert intent_to_action("SHORT", -0.10) == Action.OPEN_SHORT

    def test_neutral_close(self):
        assert intent_to_action("NEUTRAL", 0.05) == Action.CLOSE

    def test_negligible_weight_close(self):
        assert intent_to_action("LONG", 0.0005) == Action.CLOSE

    def test_direction_weight_mismatch_adjust(self):
        # SHORT direction but positive weight → ADJUST
        assert intent_to_action("SHORT", 0.10) == Action.ADJUST


# ===========================================================================
# Tests: DecideOptim agent
# ===========================================================================
class TestDecideOptim:
    """Integration tests for the DecideOptim agent pipeline."""

    @pytest.mark.asyncio
    async def test_process_single_intent(self):
        agent = DecideOptim()
        ctx = _make_context(
            intents=[_make_intent("AAPL", "LONG", 0.65)],
            regime=_make_regime("NORMAL"),
        )
        result = await agent.process(ctx)
        assert isinstance(result, PositionProposal)
        assert result.agent_id == "DECIDE-OPTIM"
        assert len(result.proposals) == 1
        assert result.proposals[0].ticker == "AAPL"
        assert result.proposals[0].direction == Direction.LONG

    @pytest.mark.asyncio
    async def test_process_multi_intent(self):
        agent = DecideOptim()
        ctx = _make_context(
            intents=[
                _make_intent("AAPL", "LONG", 0.65),
                _make_intent("SPY", "SHORT", 0.55),
                _make_intent("MSFT", "LONG", 0.60),
            ],
            regime=_make_regime("NORMAL"),
        )
        result = await agent.process(ctx)
        assert len(result.proposals) >= 2
        tickers = [p.ticker for p in result.proposals]
        assert "AAPL" in tickers

    @pytest.mark.asyncio
    async def test_halted_returns_empty(self):
        agent = DecideOptim()
        ctx = _make_context(
            intents=[_make_intent("AAPL", "LONG", 0.65)],
            regime=_make_regime("HALTED"),
        )
        result = await agent.process(ctx)
        assert len(result.proposals) == 0
        assert result.portfolio_metadata.gross_exposure == 0.0
        assert result.portfolio_metadata.risk_mode_applied == "HALTED"

    @pytest.mark.asyncio
    async def test_neutral_intents_filtered(self):
        agent = DecideOptim()
        ctx = _make_context(
            intents=[_make_intent("AAPL", "NEUTRAL", 0.65)],
            regime=_make_regime("NORMAL"),
        )
        result = await agent.process(ctx)
        assert len(result.proposals) == 0

    @pytest.mark.asyncio
    async def test_low_confidence_filtered(self):
        agent = DecideOptim()
        ctx = _make_context(
            intents=[_make_intent("AAPL", "LONG", 0.10)],  # Below floor
            regime=_make_regime("NORMAL"),
        )
        result = await agent.process(ctx)
        assert len(result.proposals) == 0

    @pytest.mark.asyncio
    async def test_defensive_tighter_limits(self):
        agent = DecideOptim()
        intents = [
            _make_intent("AAPL", "LONG", 0.65),
            _make_intent("MSFT", "LONG", 0.60),
            _make_intent("GOOG", "LONG", 0.55),
            _make_intent("NVDA", "LONG", 0.50),
        ]
        # NORMAL
        ctx_normal = _make_context(intents=intents, regime=_make_regime("NORMAL"))
        result_normal = await agent.process(ctx_normal)

        # DEFENSIVE — tighter limits, higher confidence floor
        ctx_def = _make_context(intents=intents, regime=_make_regime("DEFENSIVE"))
        result_def = await agent.process(ctx_def)

        # Defensive should have lower gross exposure
        assert (
            result_def.portfolio_metadata.gross_exposure
            <= result_normal.portfolio_metadata.gross_exposure + 0.001
        )

    @pytest.mark.asyncio
    async def test_gross_exposure_within_limits(self):
        agent = DecideOptim()
        intents = [_make_intent(f"T{i}", "LONG", 0.70) for i in range(10)]
        # Use known tickers for sector mapping
        for i, t in enumerate(["AAPL", "MSFT", "GOOG", "AMZN", "JPM", "GS", "JNJ", "PFE", "XOM", "CVX"]):
            intents[i]["ticker"] = t
        ctx = _make_context(intents=intents, regime=_make_regime("NORMAL"))
        result = await agent.process(ctx)
        limits = RISK_MODE_LIMITS["NORMAL"]
        assert result.portfolio_metadata.gross_exposure <= limits["max_gross_exposure"] + 0.01

    @pytest.mark.asyncio
    async def test_position_weight_within_limits(self):
        agent = DecideOptim()
        ctx = _make_context(
            intents=[_make_intent("AAPL", "LONG", 0.85)],
            regime=_make_regime("NORMAL"),
        )
        result = await agent.process(ctx)
        for p in result.proposals:
            assert p.target_weight <= RISK_MODE_LIMITS["NORMAL"]["max_position_weight"] + 0.001

    @pytest.mark.asyncio
    async def test_content_hash_computed(self):
        agent = DecideOptim()
        ctx = _make_context(
            intents=[_make_intent("AAPL", "LONG", 0.65)],
            regime=_make_regime("NORMAL"),
        )
        result = await agent.process(ctx)
        assert result.content_hash != ""
        assert len(result.content_hash) == 64

    @pytest.mark.asyncio
    async def test_context_window_hash_passthrough(self):
        agent = DecideOptim()
        ctx = _make_context(
            intents=[_make_intent("AAPL", "LONG", 0.65)],
            regime=_make_regime("NORMAL"),
        )
        result = await agent.process(ctx)
        assert result.context_window_hash == "test-hash-optim-001"

    @pytest.mark.asyncio
    async def test_portfolio_metadata_populated(self):
        agent = DecideOptim()
        ctx = _make_context(
            intents=[
                _make_intent("AAPL", "LONG", 0.65),
                _make_intent("JPM", "SHORT", 0.55),
            ],
            regime=_make_regime("NORMAL"),
        )
        result = await agent.process(ctx)
        meta = result.portfolio_metadata
        assert meta.gross_exposure > 0
        assert meta.position_count == len(result.proposals)
        assert meta.risk_mode_applied == "NORMAL"
        assert meta.estimated_sharpe >= 0.0

    @pytest.mark.asyncio
    async def test_empty_intents_returns_empty(self):
        agent = DecideOptim()
        ctx = _make_context(intents=[], regime=_make_regime("NORMAL"))
        result = await agent.process(ctx)
        assert len(result.proposals) == 0

    @pytest.mark.asyncio
    async def test_no_metadata_defaults(self):
        agent = DecideOptim()
        ctx = _make_context()
        result = await agent.process(ctx)
        assert len(result.proposals) == 0
        assert result.portfolio_metadata.risk_mode_applied == "NORMAL"

    @pytest.mark.asyncio
    async def test_short_position_action(self):
        agent = DecideOptim()
        ctx = _make_context(
            intents=[_make_intent("SPY", "SHORT", 0.65)],
            regime=_make_regime("NORMAL"),
        )
        result = await agent.process(ctx)
        if result.proposals:
            spy = [p for p in result.proposals if p.ticker == "SPY"]
            if spy:
                assert spy[0].action == Action.OPEN_SHORT
                assert spy[0].direction == Direction.SHORT


# ===========================================================================
# Tests: health reporting
# ===========================================================================
class TestDecideOptimHealth:
    """Tests for DecideOptim health reporting."""

    def test_healthy_initial(self):
        agent = DecideOptim()
        health = agent.get_health()
        assert health.status == AgentStatus.HEALTHY
        assert health.error_count_24h == 0

    def test_degraded_after_errors(self):
        agent = DecideOptim()
        agent._error_count_24h = 5
        health = agent.get_health()
        assert health.status == AgentStatus.DEGRADED

    def test_unhealthy_after_many_errors(self):
        agent = DecideOptim()
        agent._error_count_24h = 15
        health = agent.get_health()
        assert health.status == AgentStatus.UNHEALTHY

    def test_health_agent_id(self):
        agent = DecideOptim()
        health = agent.get_health()
        assert health.agent_id == "DECIDE-OPTIM"


# ===========================================================================
# Tests: agent properties
# ===========================================================================
class TestDecideOptimProperties:
    """Tests for DecideOptim static properties."""

    def test_agent_id(self):
        agent = DecideOptim()
        assert agent.agent_id == "DECIDE-OPTIM"

    def test_agent_type(self):
        agent = DecideOptim()
        assert agent.agent_type == "decision"

    def test_version(self):
        agent = DecideOptim()
        assert agent.version == "1.0.0"

    def test_consumed_data_types_empty(self):
        """DECIDE-OPTIM reads from metadata, not raw fragments."""
        assert DecideOptim.CONSUMED_DATA_TYPES == set()


# ===========================================================================
# Tests: schema validation
# ===========================================================================
class TestOptimSchemas:
    """Tests for PositionProposal/ProposedPosition schemas."""

    def test_proposed_position_frozen(self):
        pp = ProposedPosition(
            ticker="AAPL",
            action=Action.OPEN_LONG,
            target_weight=0.08,
            direction=Direction.LONG,
            confidence=0.65,
            source_intent_id=uuid4(),
            time_horizon_days=60,
        )
        with pytest.raises(Exception):
            pp.ticker = "MSFT"

    def test_target_weight_max(self):
        with pytest.raises(Exception):
            ProposedPosition(
                ticker="AAPL",
                action=Action.OPEN_LONG,
                target_weight=0.25,  # Over 0.20 limit
                direction=Direction.LONG,
                confidence=0.65,
                source_intent_id=uuid4(),
                time_horizon_days=60,
            )

    def test_portfolio_metadata_bounds(self):
        meta = PortfolioMetadata(
            gross_exposure=1.20,
            net_exposure=0.50,
        )
        assert meta.gross_exposure == 1.20
        assert meta.net_exposure == 0.50

    def test_position_proposal_content_hash(self):
        pp = ProposedPosition(
            ticker="AAPL",
            action=Action.OPEN_LONG,
            target_weight=0.08,
            direction=Direction.LONG,
            confidence=0.65,
            source_intent_id=uuid4(),
            time_horizon_days=60,
        )
        proposal = PositionProposal(
            agent_id="DECIDE-OPTIM",
            timestamp=NOW,
            context_window_hash="test-hash",
            proposals=[pp],
            portfolio_metadata=PortfolioMetadata(
                gross_exposure=0.08,
                net_exposure=0.08,
            ),
        )
        assert proposal.content_hash != ""
        assert len(proposal.content_hash) == 64

    def test_position_proposal_requires_tz(self):
        with pytest.raises(Exception):
            PositionProposal(
                agent_id="DECIDE-OPTIM",
                timestamp=datetime(2026, 1, 1),  # No timezone
                context_window_hash="test-hash",
                proposals=[],
                portfolio_metadata=PortfolioMetadata(
                    gross_exposure=0.0,
                    net_exposure=0.0,
                ),
            )

    def test_action_enum_values(self):
        assert Action.OPEN_LONG.value == "OPEN_LONG"
        assert Action.OPEN_SHORT.value == "OPEN_SHORT"
        assert Action.CLOSE.value == "CLOSE"
        assert Action.ADJUST.value == "ADJUST"

    def test_risk_mode_limits_defined(self):
        for mode in ["NORMAL", "CAUTIOUS", "DEFENSIVE", "HALTED"]:
            assert mode in RISK_MODE_LIMITS
            limits = RISK_MODE_LIMITS[mode]
            assert "max_gross_exposure" in limits
            assert "max_net_exposure" in limits
            assert "max_position_weight" in limits
            assert "max_sector_concentration" in limits
            assert "confidence_floor" in limits
