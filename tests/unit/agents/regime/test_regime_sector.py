"""Tests for REGIME-SECTOR Sector-Level Regime Overlay Agent.

Tests the full sector regime pipeline: ticker→sector mapping,
per-sector feature extraction, per-sector HMM classification,
relative stress computation, and key signal identification.

REGIME-SECTOR is FROZEN: zero LLM calls. Pure computation.
"""

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from providence.agents.base import AgentContext, AgentStatus
from providence.agents.regime.hmm_model import features_to_composite_score
from providence.agents.regime.regime_features import RegimeFeatures
from providence.agents.regime.regime_sector import RegimeSector
from providence.agents.regime.sector_features import (
    GICS_SECTORS,
    TICKER_SECTOR_MAP,
    SectorFragmentGroup,
    compute_relative_stress,
    extract_sector_features,
    get_sector,
    group_fragments_by_sector,
    identify_key_signals,
)
from providence.exceptions import AgentProcessingError
from providence.schemas.enums import (
    DataType,
    StatisticalRegime,
    SystemRiskMode,
    ValidationStatus,
)
from providence.schemas.market_state import MarketStateFragment
from providence.schemas.regime import RegimeStateObject, SectorRegimeOverlay


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
NOW = datetime.now(timezone.utc)


def _make_price_fragment(
    close: float = 195.0,
    hours_ago: float = 1,
    entity: str = "AAPL",
) -> MarketStateFragment:
    """Create a PRICE_OHLCV test fragment."""
    ts = NOW - timedelta(hours=hours_ago)
    return MarketStateFragment(
        fragment_id=uuid4(),
        agent_id="PERCEPT-PRICE",
        timestamp=ts,
        source_timestamp=ts,
        entity=entity,
        data_type=DataType.PRICE_OHLCV,
        schema_version="1.0.0",
        source_hash=f"hash-{entity}-{hours_ago}",
        validation_status=ValidationStatus.VALID,
        payload={
            "open": close * 0.99,
            "high": close * 1.01,
            "low": close * 0.98,
            "close": close,
            "volume": 50000000,
        },
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
        payload={"spreads": {"2s10s": spread_2s10s}, "curve_date": "2026-02-16"},
    )


def _make_sector_price_series(
    entity: str,
    n: int = 25,
    base: float = 200.0,
    daily_drift: float = 1.0005,
) -> list[MarketStateFragment]:
    """Create a price series for a single ticker."""
    frags = []
    price = base
    for i in range(n):
        price *= daily_drift
        frags.append(_make_price_fragment(close=price, hours_ago=n - i, entity=entity))
    return frags


def _make_volatile_sector_series(
    entity: str,
    n: int = 25,
    base: float = 200.0,
) -> list[MarketStateFragment]:
    """Create a volatile price series for a single ticker."""
    frags = []
    price = base
    for i in range(n):
        swing = 1.03 if i % 2 == 0 else 0.97
        price *= swing
        frags.append(_make_price_fragment(close=price, hours_ago=n - i, entity=entity))
    return frags


def _make_multi_sector_fragments() -> list[MarketStateFragment]:
    """Create fragments spanning multiple sectors (calm)."""
    frags = []
    # Technology: AAPL, MSFT
    frags.extend(_make_sector_price_series("AAPL", n=25, base=190.0))
    frags.extend(_make_sector_price_series("MSFT", n=25, base=400.0))
    # Healthcare: JNJ
    frags.extend(_make_sector_price_series("JNJ", n=25, base=160.0))
    # Financials: JPM
    frags.extend(_make_sector_price_series("JPM", n=25, base=200.0))
    return frags


def _make_context(
    fragments: list[MarketStateFragment] | None = None,
) -> AgentContext:
    """Create a test AgentContext for REGIME-SECTOR."""
    return AgentContext(
        agent_id="REGIME-SECTOR",
        trigger="schedule",
        fragments=fragments if fragments is not None else _make_multi_sector_fragments(),
        context_window_hash="sector_test_hash",
        timestamp=NOW,
        metadata={},
    )


# ---------------------------------------------------------------------------
# Ticker-to-sector mapping tests
# ---------------------------------------------------------------------------
class TestTickerSectorMapping:
    """Tests for GICS sector mapping."""

    def test_known_tech_ticker(self):
        """AAPL maps to Technology."""
        assert get_sector("AAPL") == "Technology"

    def test_known_healthcare_ticker(self):
        """JNJ maps to Healthcare."""
        assert get_sector("JNJ") == "Healthcare"

    def test_known_financial_ticker(self):
        """JPM maps to Financials."""
        assert get_sector("JPM") == "Financials"

    def test_index_returns_none(self):
        """SPY (index) returns None — not a sector."""
        assert get_sector("SPY") is None

    def test_unknown_ticker_returns_none(self):
        """Unknown ticker returns None."""
        assert get_sector("ZZZZ") is None

    def test_gics_sectors_list(self):
        """GICS_SECTORS contains expected sectors."""
        assert "Technology" in GICS_SECTORS
        assert "Healthcare" in GICS_SECTORS
        assert "Financials" in GICS_SECTORS
        assert "Energy" in GICS_SECTORS
        assert len(GICS_SECTORS) == 11  # 11 GICS sectors


# ---------------------------------------------------------------------------
# Fragment grouping tests
# ---------------------------------------------------------------------------
class TestFragmentGrouping:
    """Tests for grouping fragments by sector."""

    def test_groups_by_sector(self):
        """Fragments are correctly grouped by sector."""
        frags = _make_multi_sector_fragments()
        groups = group_fragments_by_sector(frags)

        assert "Technology" in groups
        assert "Healthcare" in groups
        assert "Financials" in groups

    def test_tech_has_two_tickers(self):
        """Technology group has AAPL + MSFT = 2 tickers."""
        frags = _make_multi_sector_fragments()
        groups = group_fragments_by_sector(frags)
        assert groups["Technology"].ticker_count == 2

    def test_healthcare_has_one_ticker(self):
        """Healthcare group has JNJ = 1 ticker."""
        frags = _make_multi_sector_fragments()
        groups = group_fragments_by_sector(frags)
        assert groups["Healthcare"].ticker_count == 1

    def test_ignores_index_fragments(self):
        """SPY fragments (no sector) are excluded from grouping."""
        frags = _make_sector_price_series("SPY", n=10)
        groups = group_fragments_by_sector(frags)
        assert len(groups) == 0

    def test_ignores_non_price_fragments(self):
        """Non-PRICE_OHLCV fragments are excluded from sector grouping."""
        frags = [_make_yield_curve_fragment()]
        groups = group_fragments_by_sector(frags)
        assert len(groups) == 0

    def test_prices_sorted_chronologically(self):
        """Close prices within a group are sorted by timestamp."""
        frags = _make_sector_price_series("AAPL", n=10)
        groups = group_fragments_by_sector(frags)
        prices = groups["Technology"].price_closes
        # With positive drift, each price should be >= previous
        for i in range(1, len(prices)):
            assert prices[i] >= prices[i - 1]


# ---------------------------------------------------------------------------
# Sector feature extraction tests
# ---------------------------------------------------------------------------
class TestSectorFeatureExtraction:
    """Tests for per-sector feature extraction."""

    def test_calm_sector_low_vol(self):
        """Calm sector prices produce low realized vol."""
        frags = _make_sector_price_series("AAPL", n=25)
        groups = group_fragments_by_sector(frags)
        features = extract_sector_features(groups["Technology"])
        assert features.realized_vol_20d is not None
        assert features.realized_vol_20d < 0.15

    def test_volatile_sector_high_vol(self):
        """Volatile sector prices produce high realized vol."""
        frags = _make_volatile_sector_series("AAPL", n=25)
        groups = group_fragments_by_sector(frags)
        features = extract_sector_features(groups["Technology"])
        assert features.realized_vol_20d is not None
        assert features.realized_vol_20d > 0.30

    def test_insufficient_data_returns_none(self):
        """Too few prices yield None for vol features."""
        group = SectorFragmentGroup(
            sector="Technology", price_closes=[100.0, 101.0], ticker_count=1
        )
        features = extract_sector_features(group)
        assert features.realized_vol_20d is None

    def test_drawdown_computed(self):
        """Drawdown is computed from sector price data."""
        # Declining prices
        frags = _make_sector_price_series("AAPL", n=25, daily_drift=0.99)
        groups = group_fragments_by_sector(frags)
        features = extract_sector_features(groups["Technology"])
        assert features.price_drawdown_pct is not None
        assert features.price_drawdown_pct < 0  # Negative drawdown

    def test_momentum_computed(self):
        """20-day momentum is computed when sufficient data."""
        frags = _make_sector_price_series("AAPL", n=25, daily_drift=1.005)
        groups = group_fragments_by_sector(frags)
        features = extract_sector_features(groups["Technology"])
        assert features.price_momentum_20d is not None
        assert features.price_momentum_20d > 0  # Positive momentum


# ---------------------------------------------------------------------------
# Relative stress and key signal tests
# ---------------------------------------------------------------------------
class TestRelativeStressAndSignals:
    """Tests for relative stress computation and key signal identification."""

    def test_equal_stress_yields_zero(self):
        """Equal sector and market stress → relative stress ≈ 0."""
        rel = compute_relative_stress(0.5, 0.5)
        assert rel == pytest.approx(0.0, abs=0.01)

    def test_sector_more_stressed(self):
        """Sector more stressed than market → positive relative stress."""
        rel = compute_relative_stress(0.7, 0.3)
        assert rel > 0.0

    def test_sector_calmer(self):
        """Sector calmer than market → negative relative stress."""
        rel = compute_relative_stress(0.2, 0.6)
        assert rel < 0.0

    def test_relative_stress_clamped(self):
        """Relative stress is clamped to [-1, +1]."""
        rel_high = compute_relative_stress(1.0, 0.0)
        rel_low = compute_relative_stress(0.0, 1.0)
        assert rel_high <= 1.0
        assert rel_low >= -1.0

    def test_key_signals_high_vol(self):
        """High realized vol produces 'high realized vol' signal."""
        features = RegimeFeatures(realized_vol_20d=0.55)
        signals = identify_key_signals(features, 0.7, 0.3)
        assert "high realized vol" in signals

    def test_key_signals_low_vol(self):
        """Low realized vol produces 'low realized vol' signal."""
        features = RegimeFeatures(realized_vol_20d=0.08)
        signals = identify_key_signals(features, 0.2, 0.3)
        assert "low realized vol" in signals

    def test_key_signals_drawdown(self):
        """Significant drawdown produces appropriate signal."""
        features = RegimeFeatures(price_drawdown_pct=-0.20)
        signals = identify_key_signals(features, 0.6, 0.3)
        assert "significant drawdown" in signals

    def test_key_signals_elevated_vs_market(self):
        """Sector much more stressed than market → 'elevated vs market'."""
        features = RegimeFeatures()
        signals = identify_key_signals(features, 0.7, 0.3)
        assert "elevated vs market" in signals


# ---------------------------------------------------------------------------
# RegimeSector agent tests
# ---------------------------------------------------------------------------
class TestRegimeSector:
    """Tests for the RegimeSector agent."""

    @pytest.mark.asyncio
    async def test_process_returns_regime_state_object(self):
        """Agent process() returns a valid RegimeStateObject."""
        agent = RegimeSector()
        context = _make_context()

        result = await agent.process(context)

        assert isinstance(result, RegimeStateObject)
        assert result.agent_id == "REGIME-SECTOR"

    @pytest.mark.asyncio
    async def test_process_populates_sector_overlays(self):
        """Output has sector overlays for each detected sector."""
        agent = RegimeSector()
        context = _make_context()

        result = await agent.process(context)

        # Should have Technology, Healthcare, Financials
        assert len(result.sector_overlays) >= 3
        assert "Technology" in result.sector_overlays
        assert "Healthcare" in result.sector_overlays
        assert "Financials" in result.sector_overlays

    @pytest.mark.asyncio
    async def test_overlay_is_sector_regime_overlay(self):
        """Each overlay is a valid SectorRegimeOverlay."""
        agent = RegimeSector()
        context = _make_context()

        result = await agent.process(context)

        for sector, overlay in result.sector_overlays.items():
            assert isinstance(overlay, SectorRegimeOverlay)
            assert overlay.sector == sector
            assert overlay.regime in StatisticalRegime
            assert 0.0 <= overlay.regime_confidence <= 1.0

    @pytest.mark.asyncio
    async def test_overlay_probabilities_sum_to_one(self):
        """Each overlay's regime probabilities sum to ~1.0."""
        agent = RegimeSector()
        context = _make_context()

        result = await agent.process(context)

        for overlay in result.sector_overlays.values():
            total = sum(overlay.regime_probabilities.values())
            assert total == pytest.approx(1.0, abs=1e-4)

    @pytest.mark.asyncio
    async def test_overlay_ticker_count(self):
        """Ticker count is correct for each sector."""
        agent = RegimeSector()
        context = _make_context()

        result = await agent.process(context)

        assert result.sector_overlays["Technology"].ticker_count == 2  # AAPL, MSFT
        assert result.sector_overlays["Healthcare"].ticker_count == 1  # JNJ
        assert result.sector_overlays["Financials"].ticker_count == 1  # JPM

    @pytest.mark.asyncio
    async def test_calm_sectors_classify_low_vol(self):
        """Calm sector data classifies as LOW_VOL_TRENDING."""
        agent = RegimeSector()
        frags = _make_sector_price_series("AAPL", n=65, daily_drift=1.0005)
        context = _make_context(fragments=frags)

        result = await agent.process(context)

        tech_overlay = result.sector_overlays.get("Technology")
        assert tech_overlay is not None
        assert tech_overlay.regime == StatisticalRegime.LOW_VOL_TRENDING

    @pytest.mark.asyncio
    async def test_empty_fragments_no_overlays(self):
        """Empty fragments produce no sector overlays."""
        agent = RegimeSector()
        context = _make_context(fragments=[])

        result = await agent.process(context)

        assert isinstance(result, RegimeStateObject)
        assert len(result.sector_overlays) == 0

    @pytest.mark.asyncio
    async def test_index_only_fragments_no_overlays(self):
        """SPY-only fragments produce no sector overlays (SPY has no sector)."""
        agent = RegimeSector()
        frags = _make_sector_price_series("SPY", n=25)
        context = _make_context(fragments=frags)

        result = await agent.process(context)

        assert len(result.sector_overlays) == 0

    @pytest.mark.asyncio
    async def test_global_regime_populated(self):
        """Output includes valid global regime classification."""
        agent = RegimeSector()
        context = _make_context()

        result = await agent.process(context)

        assert result.statistical_regime in StatisticalRegime
        assert result.system_risk_mode in SystemRiskMode
        assert result.regime_confidence > 0.0

    @pytest.mark.asyncio
    async def test_content_hash_includes_overlays(self):
        """Content hash changes when sector overlays differ."""
        agent = RegimeSector()

        # Run with multi-sector data
        context1 = _make_context()
        result1 = await agent.process(context1)

        # Run with single-sector data
        frags2 = _make_sector_price_series("AAPL", n=25)
        context2 = _make_context(fragments=frags2)
        result2 = await agent.process(context2)

        # Different overlays → different content hash
        assert result1.content_hash != result2.content_hash

    @pytest.mark.asyncio
    async def test_context_window_hash_preserved(self):
        """RegimeStateObject carries the context_window_hash."""
        agent = RegimeSector()
        context = _make_context()

        result = await agent.process(context)

        assert result.context_window_hash == "sector_test_hash"

    @pytest.mark.asyncio
    async def test_relative_stress_in_overlays(self):
        """Each overlay has a relative_stress value."""
        agent = RegimeSector()
        context = _make_context()

        result = await agent.process(context)

        for overlay in result.sector_overlays.values():
            assert -1.0 <= overlay.relative_stress <= 1.0


# ---------------------------------------------------------------------------
# Health status tests
# ---------------------------------------------------------------------------
class TestRegimeSectorHealth:
    """Tests for agent health reporting."""

    def test_initial_health_is_healthy(self):
        """Agent starts in HEALTHY state."""
        agent = RegimeSector()
        health = agent.get_health()
        assert health.agent_id == "REGIME-SECTOR"
        assert health.status == AgentStatus.HEALTHY
        assert health.error_count_24h == 0

    @pytest.mark.asyncio
    async def test_health_after_success(self):
        """Health reports last_success after successful run."""
        agent = RegimeSector()
        context = _make_context()

        await agent.process(context)

        health = agent.get_health()
        assert health.last_run is not None
        assert health.last_success is not None
        assert health.status == AgentStatus.HEALTHY

    def test_health_degraded_after_errors(self):
        """Health degrades after repeated errors."""
        agent = RegimeSector()
        agent._error_count_24h = 4
        health = agent.get_health()
        assert health.status == AgentStatus.DEGRADED

    def test_health_unhealthy_after_many_errors(self):
        """Health becomes UNHEALTHY after 11+ errors."""
        agent = RegimeSector()
        agent._error_count_24h = 11
        health = agent.get_health()
        assert health.status == AgentStatus.UNHEALTHY


# ---------------------------------------------------------------------------
# Agent properties
# ---------------------------------------------------------------------------
class TestRegimeSectorProperties:
    """Test agent identity properties."""

    def test_agent_id(self):
        agent = RegimeSector()
        assert agent.agent_id == "REGIME-SECTOR"

    def test_agent_type(self):
        agent = RegimeSector()
        assert agent.agent_type == "regime"

    def test_version(self):
        agent = RegimeSector()
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
        assert RegimeSector.CONSUMED_DATA_TYPES == expected
