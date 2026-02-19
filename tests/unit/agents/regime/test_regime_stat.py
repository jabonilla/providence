"""Tests for REGIME-STAT Statistical Regime Classification Agent.

Tests the full regime classification pipeline with real HMM computation.
Validates feature extraction, HMM forward algorithm, regime classification,
risk mode derivation, and error handling.

REGIME-STAT is FROZEN: zero LLM calls. Pure computation.
"""

import math
from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

import pytest

from providence.agents.base import AgentContext, AgentStatus
from providence.agents.regime.hmm_model import (
    DEFAULT_HMM_PARAMS,
    REGIME_STATES,
    classify_regime,
    derive_risk_mode,
    features_to_composite_score,
    forward_algorithm,
    gaussian_emission_prob,
)
from providence.agents.regime.regime_features import (
    RegimeFeatures,
    compute_drawdown,
    compute_realized_vol,
    compute_vol_of_vol,
    extract_regime_features,
)
from providence.agents.regime.regime_stat import RegimeStat
from providence.exceptions import AgentProcessingError
from providence.schemas.enums import (
    DataType,
    StatisticalRegime,
    SystemRiskMode,
    ValidationStatus,
)
from providence.schemas.market_state import MarketStateFragment
from providence.schemas.regime import RegimeStateObject


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
NOW = datetime.now(timezone.utc)


def _make_price_fragment(
    close: float = 195.0,
    hours_ago: float = 1,
    entity: str = "SPY",
    fragment_id: UUID | None = None,
) -> MarketStateFragment:
    """Create a PRICE_OHLCV test fragment."""
    ts = NOW - timedelta(hours=hours_ago)
    return MarketStateFragment(
        fragment_id=fragment_id or uuid4(),
        agent_id="PERCEPT-PRICE",
        timestamp=ts,
        source_timestamp=ts,
        entity=entity,
        data_type=DataType.PRICE_OHLCV,
        schema_version="1.0.0",
        source_hash=f"hash-price-{entity}-{hours_ago}",
        validation_status=ValidationStatus.VALID,
        payload={"open": close * 0.99, "high": close * 1.01, "low": close * 0.98, "close": close, "volume": 50000000},
    )


def _make_yield_curve_fragment(
    spread_2s10s: float = 50.0,
    hours_ago: float = 1,
) -> MarketStateFragment:
    """Create a MACRO_YIELD_CURVE test fragment."""
    ts = NOW - timedelta(hours=hours_ago)
    return MarketStateFragment(
        fragment_id=uuid4(),
        agent_id="PERCEPT-MACRO",
        timestamp=ts,
        source_timestamp=ts,
        entity=None,
        data_type=DataType.MACRO_YIELD_CURVE,
        schema_version="1.0.0",
        source_hash=f"hash-yield-{hours_ago}",
        validation_status=ValidationStatus.VALID,
        payload={"spreads": {"2s10s": spread_2s10s}, "curve_date": "2026-02-12"},
    )


def _make_cds_fragment(
    spread_bps: float = 100.0,
    hours_ago: float = 1,
) -> MarketStateFragment:
    """Create a MACRO_CDS test fragment."""
    ts = NOW - timedelta(hours=hours_ago)
    return MarketStateFragment(
        fragment_id=uuid4(),
        agent_id="PERCEPT-CDS",
        timestamp=ts,
        source_timestamp=ts,
        entity=None,
        data_type=DataType.MACRO_CDS,
        schema_version="1.0.0",
        source_hash=f"hash-cds-{hours_ago}",
        validation_status=ValidationStatus.VALID,
        payload={"spread_bps": spread_bps, "tenor": "5Y", "reference_entity": "CDX.NA.IG"},
    )


def _make_options_fragment(
    implied_vol: float = 0.18,
    hours_ago: float = 1,
) -> MarketStateFragment:
    """Create an OPTIONS_CHAIN test fragment."""
    ts = NOW - timedelta(hours=hours_ago)
    return MarketStateFragment(
        fragment_id=uuid4(),
        agent_id="PERCEPT-OPTIONS",
        timestamp=ts,
        source_timestamp=ts,
        entity="SPY",
        data_type=DataType.OPTIONS_CHAIN,
        schema_version="1.0.0",
        source_hash=f"hash-options-{hours_ago}",
        validation_status=ValidationStatus.VALID,
        payload={"implied_volatility": implied_vol, "strike_price": 500, "contract_type": "CALL"},
    )


def _make_macro_economic_fragment(
    value: float = 2.5,
    indicator: str = "GDP",
    hours_ago: float = 1,
) -> MarketStateFragment:
    """Create a MACRO_ECONOMIC test fragment."""
    ts = NOW - timedelta(hours=hours_ago)
    return MarketStateFragment(
        fragment_id=uuid4(),
        agent_id="PERCEPT-MACRO",
        timestamp=ts,
        source_timestamp=ts,
        entity=None,
        data_type=DataType.MACRO_ECONOMIC,
        schema_version="1.0.0",
        source_hash=f"hash-macro-{hours_ago}",
        validation_status=ValidationStatus.VALID,
        payload={"indicator": indicator, "value": value, "period": "2026-Q1"},
    )


def _make_calm_price_series(n: int = 65, base: float = 500.0) -> list[MarketStateFragment]:
    """Create a calm, steadily rising price series (low vol)."""
    frags = []
    price = base
    for i in range(n):
        price *= 1.0005  # ~0.05% daily drift, low vol
        frags.append(_make_price_fragment(close=price, hours_ago=n - i, entity="SPY"))
    return frags


def _make_volatile_price_series(n: int = 65, base: float = 500.0) -> list[MarketStateFragment]:
    """Create a volatile, oscillating price series (high vol)."""
    frags = []
    price = base
    for i in range(n):
        swing = 1.03 if i % 2 == 0 else 0.97  # ~3% daily swings
        price *= swing
        frags.append(_make_price_fragment(close=price, hours_ago=n - i, entity="SPY"))
    return frags


def _make_context(
    fragments: list[MarketStateFragment] | None = None,
) -> AgentContext:
    """Create a test AgentContext for REGIME-STAT."""
    return AgentContext(
        agent_id="REGIME-STAT",
        trigger="schedule",
        fragments=fragments if fragments is not None else _make_calm_price_series(),
        context_window_hash="regime_test_hash",
        timestamp=NOW,
        metadata={},
    )


# ---------------------------------------------------------------------------
# Feature extraction tests
# ---------------------------------------------------------------------------
class TestRegimeFeatureExtraction:
    """Tests for regime feature extraction from fragments."""

    def test_realized_vol_from_calm_prices(self):
        """Calm prices produce low realized volatility."""
        prices = [100.0 * (1.0005 ** i) for i in range(25)]
        vol = compute_realized_vol(prices, window=20)
        assert vol is not None
        assert vol < 0.15  # Low vol

    def test_realized_vol_from_volatile_prices(self):
        """Volatile prices produce high realized volatility."""
        prices = []
        p = 100.0
        for i in range(25):
            p *= 1.03 if i % 2 == 0 else 0.97
            prices.append(p)
        vol = compute_realized_vol(prices, window=20)
        assert vol is not None
        assert vol > 0.30  # High vol

    def test_realized_vol_insufficient_data(self):
        """Returns None when insufficient price points."""
        vol = compute_realized_vol([100.0, 101.0], window=20)
        assert vol is None

    def test_drawdown_from_declining_prices(self):
        """Declining prices produce negative drawdown."""
        prices = [100.0, 95.0, 90.0, 85.0, 80.0]
        dd = compute_drawdown(prices)
        assert dd is not None
        assert dd == pytest.approx(-0.20, abs=0.01)

    def test_drawdown_from_rising_prices(self):
        """Rising prices produce zero drawdown."""
        prices = [100.0, 105.0, 110.0, 115.0, 120.0]
        dd = compute_drawdown(prices)
        assert dd is not None
        assert dd == 0.0

    def test_extract_features_with_all_data_types(self):
        """Feature extraction works with all fragment types."""
        frags = (
            _make_calm_price_series(n=25) +
            [_make_yield_curve_fragment(spread_2s10s=100.0)] +
            [_make_cds_fragment(spread_bps=80.0)] +
            [_make_options_fragment(implied_vol=0.15)] +
            [_make_macro_economic_fragment(value=2.8)]
        )
        features = extract_regime_features(frags)
        assert features.realized_vol_20d is not None
        assert features.yield_spread_2s10s == 100.0
        assert features.cds_ig_spread == 80.0
        assert features.vix_proxy == 0.15
        assert features.macro_momentum == 2.8

    def test_extract_features_price_only(self):
        """Feature extraction works with just price data."""
        frags = _make_calm_price_series(n=25)
        features = extract_regime_features(frags)
        assert features.realized_vol_20d is not None
        assert features.yield_spread_2s10s is None
        assert features.cds_ig_spread is None


# ---------------------------------------------------------------------------
# HMM model tests
# ---------------------------------------------------------------------------
class TestHMMModel:
    """Tests for HMM classification logic."""

    def test_gaussian_emission_prob_at_mean(self):
        """Emission probability is highest at the mean."""
        p_at_mean = gaussian_emission_prob(0.5, mean=0.5, std=0.1)
        p_off_mean = gaussian_emission_prob(0.7, mean=0.5, std=0.1)
        assert p_at_mean > p_off_mean

    def test_forward_algorithm_produces_valid_probs(self):
        """Forward algorithm produces probabilities summing to ~1.0."""
        posteriors = forward_algorithm(0.2, DEFAULT_HMM_PARAMS)
        assert len(posteriors) == 4
        assert sum(posteriors) == pytest.approx(1.0, abs=1e-6)

    def test_low_stress_classifies_low_vol(self):
        """Low composite score classifies as LOW_VOL_TRENDING."""
        features = RegimeFeatures(
            realized_vol_20d=0.10,
            yield_spread_2s10s=150.0,
            cds_ig_spread=60.0,
            vix_proxy=0.12,
        )
        regime, conf, probs = classify_regime(features)
        assert regime == StatisticalRegime.LOW_VOL_TRENDING

    def test_high_stress_classifies_crisis(self):
        """High composite score classifies as CRISIS_DISLOCATION."""
        features = RegimeFeatures(
            realized_vol_20d=0.65,
            vol_of_vol=0.30,
            yield_spread_2s10s=-80.0,
            cds_ig_spread=400.0,
            vix_proxy=0.55,
            price_drawdown_pct=-0.25,
        )
        regime, conf, probs = classify_regime(features)
        assert regime == StatisticalRegime.CRISIS_DISLOCATION

    def test_composite_score_no_features(self):
        """Empty features produce neutral composite score."""
        features = RegimeFeatures()
        score = features_to_composite_score(features)
        assert score == pytest.approx(0.45, abs=0.01)

    def test_derive_risk_mode_normal(self):
        """LOW_VOL_TRENDING → NORMAL."""
        assert derive_risk_mode(StatisticalRegime.LOW_VOL_TRENDING, 0.7) == SystemRiskMode.NORMAL

    def test_derive_risk_mode_halted(self):
        """CRISIS_DISLOCATION with high confidence → HALTED."""
        assert derive_risk_mode(StatisticalRegime.CRISIS_DISLOCATION, 0.95) == SystemRiskMode.HALTED

    def test_derive_risk_mode_defensive(self):
        """CRISIS_DISLOCATION with moderate confidence → DEFENSIVE."""
        assert derive_risk_mode(StatisticalRegime.CRISIS_DISLOCATION, 0.7) == SystemRiskMode.DEFENSIVE

    def test_derive_risk_mode_cautious(self):
        """HIGH_VOL_MEAN_REVERTING → CAUTIOUS."""
        assert derive_risk_mode(StatisticalRegime.HIGH_VOL_MEAN_REVERTING, 0.6) == SystemRiskMode.CAUTIOUS


# ---------------------------------------------------------------------------
# RegimeStat agent tests
# ---------------------------------------------------------------------------
class TestRegimeStat:
    """Tests for the RegimeStat agent."""

    @pytest.mark.asyncio
    async def test_process_returns_regime_state_object(self):
        """Agent process() returns a valid RegimeStateObject."""
        agent = RegimeStat()
        context = _make_context()

        result = await agent.process(context)

        assert isinstance(result, RegimeStateObject)
        assert result.agent_id == "REGIME-STAT"
        assert result.statistical_regime in StatisticalRegime
        assert result.system_risk_mode in SystemRiskMode

    @pytest.mark.asyncio
    async def test_process_calm_market(self):
        """Calm price series classifies as LOW_VOL_TRENDING."""
        agent = RegimeStat()
        frags = _make_calm_price_series(n=65)
        context = _make_context(fragments=frags)

        result = await agent.process(context)

        assert result.statistical_regime == StatisticalRegime.LOW_VOL_TRENDING
        assert result.system_risk_mode == SystemRiskMode.NORMAL

    @pytest.mark.asyncio
    async def test_process_volatile_market(self):
        """Volatile prices + stressed macro classifies as HIGH_VOL or CRISIS."""
        agent = RegimeStat()
        # Volatile prices alone may not overcome strong LOW_VOL prior,
        # so add stressed macro data (inverted yield curve, wide CDS, high IV)
        frags = (
            _make_volatile_price_series(n=65)
            + [_make_yield_curve_fragment(spread_2s10s=-50.0)]
            + [_make_cds_fragment(spread_bps=350.0)]
            + [_make_options_fragment(implied_vol=0.45)]
        )
        context = _make_context(fragments=frags)

        result = await agent.process(context)

        # With stressed multi-asset data, should NOT be LOW_VOL
        assert result.statistical_regime != StatisticalRegime.LOW_VOL_TRENDING

    @pytest.mark.asyncio
    async def test_process_with_macro_data(self):
        """Agent works with mixed price and macro fragments."""
        agent = RegimeStat()
        frags = (
            _make_calm_price_series(n=30) +
            [_make_yield_curve_fragment(spread_2s10s=120.0)] +
            [_make_cds_fragment(spread_bps=75.0)]
        )
        context = _make_context(fragments=frags)

        result = await agent.process(context)

        assert isinstance(result, RegimeStateObject)
        assert "yield_spread_2s10s" in result.features_used
        assert "cds_ig_spread" in result.features_used

    @pytest.mark.asyncio
    async def test_process_empty_fragments(self):
        """Agent handles empty fragment list (falls back to priors)."""
        agent = RegimeStat()
        context = _make_context(fragments=[])

        result = await agent.process(context)

        # Should still produce a result using HMM priors
        assert isinstance(result, RegimeStateObject)

    @pytest.mark.asyncio
    async def test_context_window_hash_preserved(self):
        """RegimeStateObject carries the context_window_hash."""
        agent = RegimeStat()
        context = _make_context()

        result = await agent.process(context)

        assert result.context_window_hash == "regime_test_hash"

    @pytest.mark.asyncio
    async def test_content_hash_computed(self):
        """RegimeStateObject has a non-empty content hash."""
        agent = RegimeStat()
        context = _make_context()

        result = await agent.process(context)

        assert result.content_hash
        assert len(result.content_hash) == 64

    @pytest.mark.asyncio
    async def test_probabilities_sum_to_one(self):
        """Regime probabilities sum to approximately 1.0."""
        agent = RegimeStat()
        context = _make_context()

        result = await agent.process(context)

        total = sum(result.regime_probabilities.values())
        assert total == pytest.approx(1.0, abs=1e-4)


# ---------------------------------------------------------------------------
# Health status tests
# ---------------------------------------------------------------------------
class TestRegimeStatHealth:
    """Tests for agent health reporting."""

    def test_initial_health_is_healthy(self):
        """Agent starts in HEALTHY state."""
        agent = RegimeStat()
        health = agent.get_health()
        assert health.agent_id == "REGIME-STAT"
        assert health.status == AgentStatus.HEALTHY
        assert health.error_count_24h == 0

    @pytest.mark.asyncio
    async def test_health_after_success(self):
        """Health reports last_success after successful run."""
        agent = RegimeStat()
        context = _make_context()

        await agent.process(context)

        health = agent.get_health()
        assert health.last_run is not None
        assert health.last_success is not None
        assert health.status == AgentStatus.HEALTHY

    def test_health_degraded_after_errors(self):
        """Health degrades after repeated errors."""
        agent = RegimeStat()
        agent._error_count_24h = 4
        health = agent.get_health()
        assert health.status == AgentStatus.DEGRADED

    def test_health_unhealthy_after_many_errors(self):
        """Health becomes UNHEALTHY after 11+ errors."""
        agent = RegimeStat()
        agent._error_count_24h = 11
        health = agent.get_health()
        assert health.status == AgentStatus.UNHEALTHY


# ---------------------------------------------------------------------------
# Agent properties
# ---------------------------------------------------------------------------
class TestRegimeStatProperties:
    """Test agent identity properties."""

    def test_agent_id(self):
        agent = RegimeStat()
        assert agent.agent_id == "REGIME-STAT"

    def test_agent_type(self):
        agent = RegimeStat()
        assert agent.agent_type == "regime"

    def test_version(self):
        agent = RegimeStat()
        assert agent.version == "1.0.0"

    def test_consumed_data_types(self):
        """Agent consumes the correct data types."""
        expected = {
            DataType.PRICE_OHLCV,
            DataType.MACRO_YIELD_CURVE,
            DataType.MACRO_CDS,
            DataType.MACRO_ECONOMIC,
            DataType.OPTIONS_CHAIN,
        }
        assert RegimeStat.CONSUMED_DATA_TYPES == expected
