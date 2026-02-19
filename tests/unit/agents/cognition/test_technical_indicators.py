"""Tests for technical indicator computations.

Tests the pure-function indicator library used by COGNIT-TECHNICAL.
All functions are FROZEN — pure computation, no LLM calls, no I/O.
"""

import math

import pytest

from providence.agents.cognition.technical_indicators import (
    TechnicalSignals,
    compute_all_signals,
    compute_bollinger_bands,
    compute_ema,
    compute_macd,
    compute_momentum,
    compute_rsi,
    compute_sma,
)


# ---------------------------------------------------------------------------
# Helpers — generate synthetic price series
# ---------------------------------------------------------------------------
def _uptrend(n: int = 250, start: float = 100.0, step: float = 0.5) -> list[float]:
    """Create a steadily rising price series."""
    return [start + i * step for i in range(n)]


def _downtrend(n: int = 250, start: float = 200.0, step: float = 0.5) -> list[float]:
    """Create a steadily falling price series."""
    return [start - i * step for i in range(n)]


def _flat(n: int = 250, price: float = 100.0) -> list[float]:
    """Create a flat price series."""
    return [price] * n


def _oscillating(n: int = 250, base: float = 100.0, amplitude: float = 10.0) -> list[float]:
    """Create a price series oscillating around a base."""
    return [base + amplitude * math.sin(i * 0.2) for i in range(n)]


# ---------------------------------------------------------------------------
# compute_sma
# ---------------------------------------------------------------------------
class TestComputeSMA:
    """Tests for Simple Moving Average computation."""

    def test_exact_period_data(self):
        """SMA with exactly enough data points."""
        prices = [10.0, 20.0, 30.0]
        assert compute_sma(prices, 3) == pytest.approx(20.0)

    def test_more_data_than_period(self):
        """SMA uses only the last `period` points."""
        prices = [1.0, 2.0, 3.0, 4.0, 5.0]
        # SMA(3) of last 3: (3+4+5)/3 = 4.0
        assert compute_sma(prices, 3) == pytest.approx(4.0)

    def test_insufficient_data_returns_none(self):
        """SMA returns None when there are fewer points than the period."""
        assert compute_sma([1.0, 2.0], 3) is None

    def test_single_period(self):
        """SMA with period=1 returns the last price."""
        prices = [5.0, 10.0, 15.0]
        assert compute_sma(prices, 1) == pytest.approx(15.0)

    def test_empty_list(self):
        """SMA of empty list returns None."""
        assert compute_sma([], 5) is None

    def test_known_values(self):
        """SMA with a known hand-calculated result."""
        prices = [44.0, 44.34, 44.09, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08]
        expected = sum(prices[-5:]) / 5  # SMA(5) of last 5
        assert compute_sma(prices, 5) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# compute_ema
# ---------------------------------------------------------------------------
class TestComputeEMA:
    """Tests for Exponential Moving Average computation."""

    def test_insufficient_data_returns_none(self):
        """EMA returns None when there are fewer points than the period."""
        assert compute_ema([1.0, 2.0], 3) is None

    def test_exact_period_data_equals_sma(self):
        """With exactly `period` data points, EMA equals SMA (seed only)."""
        prices = [10.0, 20.0, 30.0]
        expected_sma = 20.0
        assert compute_ema(prices, 3) == pytest.approx(expected_sma)

    def test_ema_responds_to_recent_prices(self):
        """EMA is closer to recent prices than SMA would be."""
        prices = [10.0] * 10 + [20.0]  # flat then spike
        ema = compute_ema(prices, 10)
        sma = compute_sma(prices, 10)
        # EMA should be between SMA and the latest price because of the spike
        assert ema is not None
        assert sma is not None
        assert ema > sma  # EMA weights the spike more heavily

    def test_ema_multiplier(self):
        """Manually verify EMA step computation."""
        prices = [10.0, 10.0, 10.0, 15.0]  # period=3, then one new price
        # Seed: SMA(first 3) = 10.0
        # Multiplier: 2/(3+1) = 0.5
        # EMA = (15.0 - 10.0) * 0.5 + 10.0 = 12.5
        assert compute_ema(prices, 3) == pytest.approx(12.5)


# ---------------------------------------------------------------------------
# compute_rsi
# ---------------------------------------------------------------------------
class TestComputeRSI:
    """Tests for Relative Strength Index computation."""

    def test_insufficient_data_returns_none(self):
        """RSI returns None with fewer than period+1 data points."""
        assert compute_rsi([100.0, 101.0], period=14) is None

    def test_all_gains_returns_100(self):
        """Pure uptrend (all gains, no losses) yields RSI = 100."""
        prices = [float(i) for i in range(1, 20)]  # 1,2,...,19
        rsi = compute_rsi(prices, period=14)
        assert rsi == pytest.approx(100.0)

    def test_all_losses_returns_0(self):
        """Pure downtrend (all losses, no gains) yields RSI near 0."""
        prices = [float(20 - i) for i in range(20)]  # 20,19,...,1
        rsi = compute_rsi(prices, period=14)
        # With Wilder smoothing, pure losses push RSI close to 0
        assert rsi is not None
        assert rsi < 5.0

    def test_rsi_range(self):
        """RSI is always between 0 and 100."""
        for prices in [_uptrend(50), _downtrend(50), _oscillating(50), _flat(50)]:
            rsi = compute_rsi(prices, period=14)
            if rsi is not None:
                assert 0.0 <= rsi <= 100.0

    def test_rsi_flat_prices_is_neutral(self):
        """Flat prices produce no gains or losses — RSI is undefined or 100 (all-zero loss)."""
        prices = _flat(20)
        rsi = compute_rsi(prices, period=14)
        # All changes are 0. avg_gain=0, avg_loss=0 → RS undefined → our code returns 100
        assert rsi == pytest.approx(100.0)

    def test_rsi_overbought(self):
        """Strong uptrend gives RSI > 70."""
        prices = _uptrend(50, start=100.0, step=2.0)
        rsi = compute_rsi(prices, period=14)
        assert rsi is not None
        assert rsi > 70

    def test_rsi_oversold(self):
        """Strong downtrend gives RSI < 30."""
        prices = _downtrend(50, start=200.0, step=2.0)
        rsi = compute_rsi(prices, period=14)
        assert rsi is not None
        assert rsi < 30


# ---------------------------------------------------------------------------
# compute_macd
# ---------------------------------------------------------------------------
class TestComputeMACD:
    """Tests for MACD computation."""

    def test_insufficient_data_returns_nones(self):
        """MACD returns (None, None, None) with too little data."""
        result = compute_macd([1.0, 2.0, 3.0])
        assert result == (None, None, None)

    def test_sufficient_data_returns_values(self):
        """MACD with enough data returns non-None values."""
        prices = _uptrend(60)
        macd_line, signal, histogram = compute_macd(prices)
        assert macd_line is not None
        assert signal is not None
        assert histogram is not None

    def test_histogram_is_macd_minus_signal(self):
        """Histogram = MACD line - signal line."""
        prices = _oscillating(60)
        macd_line, signal, histogram = compute_macd(prices)
        if macd_line is not None and signal is not None and histogram is not None:
            assert histogram == pytest.approx(macd_line - signal, abs=1e-10)

    def test_uptrend_macd_positive(self):
        """In an uptrend, MACD line should be positive (fast EMA > slow EMA)."""
        prices = _uptrend(80, start=50, step=1.0)
        macd_line, _, _ = compute_macd(prices)
        assert macd_line is not None
        assert macd_line > 0

    def test_downtrend_macd_negative(self):
        """In a downtrend, MACD line should be negative."""
        prices = _downtrend(80, start=200, step=1.0)
        macd_line, _, _ = compute_macd(prices)
        assert macd_line is not None
        assert macd_line < 0

    def test_below_threshold_returns_all_none(self):
        """Below slow_period + signal_period (35), all values are None."""
        prices = _uptrend(30)
        macd_line, signal, histogram = compute_macd(prices)
        assert macd_line is None
        assert signal is None
        assert histogram is None

    def test_at_threshold_returns_full_result(self):
        """At exactly 35 points, full MACD (line + signal + histogram) is returned."""
        prices = _uptrend(35)
        macd_line, signal, histogram = compute_macd(prices)
        assert macd_line is not None
        assert signal is not None
        assert histogram is not None


# ---------------------------------------------------------------------------
# compute_bollinger_bands
# ---------------------------------------------------------------------------
class TestComputeBollingerBands:
    """Tests for Bollinger Bands computation."""

    def test_insufficient_data_returns_nones(self):
        """Returns (None, None, None) with too few data points."""
        assert compute_bollinger_bands([1.0, 2.0], period=20) == (None, None, None)

    def test_middle_band_is_sma(self):
        """Middle band equals SMA(period)."""
        prices = _oscillating(30)
        upper, middle, lower = compute_bollinger_bands(prices, period=20)
        sma = compute_sma(prices, 20)
        assert middle is not None
        assert sma is not None
        assert middle == pytest.approx(sma)

    def test_upper_above_middle_above_lower(self):
        """Upper > Middle > Lower always holds."""
        prices = _oscillating(30)
        upper, middle, lower = compute_bollinger_bands(prices, period=20)
        assert upper is not None and middle is not None and lower is not None
        assert upper > middle > lower

    def test_flat_prices_bands_collapse(self):
        """Flat prices have zero standard deviation — upper == middle == lower."""
        prices = _flat(25, price=50.0)
        upper, middle, lower = compute_bollinger_bands(prices, period=20)
        assert upper == pytest.approx(50.0)
        assert middle == pytest.approx(50.0)
        assert lower == pytest.approx(50.0)

    def test_wider_std_multiplier_gives_wider_bands(self):
        """Higher num_std gives wider bands."""
        prices = _oscillating(30)
        upper_2, _, lower_2 = compute_bollinger_bands(prices, period=20, num_std=2.0)
        upper_3, _, lower_3 = compute_bollinger_bands(prices, period=20, num_std=3.0)
        assert upper_3 > upper_2
        assert lower_3 < lower_2


# ---------------------------------------------------------------------------
# compute_momentum
# ---------------------------------------------------------------------------
class TestComputeMomentum:
    """Tests for momentum (percentage change) computation."""

    def test_positive_momentum(self):
        """Price increase gives positive momentum."""
        prices = [100.0, 105.0, 110.0, 115.0, 120.0, 125.0]
        momentum = compute_momentum(prices, period=5)
        assert momentum is not None
        assert momentum > 0

    def test_negative_momentum(self):
        """Price decrease gives negative momentum."""
        prices = [125.0, 120.0, 115.0, 110.0, 105.0, 100.0]
        momentum = compute_momentum(prices, period=5)
        assert momentum is not None
        assert momentum < 0

    def test_zero_momentum(self):
        """Flat prices give zero momentum."""
        prices = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
        momentum = compute_momentum(prices, period=5)
        assert momentum == pytest.approx(0.0)

    def test_exact_percentage(self):
        """Verify exact percentage calculation."""
        prices = [100.0, 110.0, 120.0, 130.0, 140.0, 150.0]
        momentum = compute_momentum(prices, period=5)
        # (150 - 100) / 100 * 100 = 50%
        assert momentum == pytest.approx(50.0)

    def test_insufficient_data_returns_none(self):
        """Returns None when not enough data for the period."""
        prices = [100.0, 110.0]
        assert compute_momentum(prices, period=5) is None

    def test_zero_base_price_returns_none(self):
        """Returns None when the base price is zero (division by zero guard)."""
        prices = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]
        assert compute_momentum(prices, period=5) is None


# ---------------------------------------------------------------------------
# compute_all_signals (integration)
# ---------------------------------------------------------------------------
class TestComputeAllSignals:
    """Integration tests for the aggregated signal computation."""

    def test_uptrend_produces_bullish_signals(self):
        """Strong uptrend generates net positive (bullish) signals."""
        prices = _uptrend(250, start=50.0, step=0.5)
        signals = compute_all_signals(prices)

        assert signals.bullish_signals > 0
        assert signals.net_signal > 0
        # In a strong uptrend we expect golden cross, positive momentum, price > SMA200
        assert signals.golden_cross is True
        assert signals.death_cross is False

    def test_downtrend_produces_bearish_signals(self):
        """Strong downtrend generates net negative (bearish) signals."""
        prices = _downtrend(250, start=200.0, step=0.5)
        signals = compute_all_signals(prices)

        assert signals.bearish_signals > 0
        assert signals.net_signal < 0
        assert signals.death_cross is True
        assert signals.golden_cross is False

    def test_flat_prices_mixed_signals(self):
        """Flat prices produce no directional signals."""
        prices = _flat(250, price=100.0)
        signals = compute_all_signals(prices)

        # SMAs converge on the flat price
        assert signals.sma_20 == pytest.approx(100.0)
        assert signals.sma_50 == pytest.approx(100.0)
        assert signals.sma_200 == pytest.approx(100.0)
        # No golden or death cross (SMA50 == SMA200 exactly doesn't trigger >/<)
        assert signals.golden_cross is False
        assert signals.death_cross is False

    def test_all_indicators_computed_with_enough_data(self):
        """With 250 data points, all indicator fields are non-None."""
        prices = _oscillating(250)
        signals = compute_all_signals(prices)

        assert signals.sma_20 is not None
        assert signals.sma_50 is not None
        assert signals.sma_200 is not None
        assert signals.rsi_14 is not None
        assert signals.macd_line is not None
        assert signals.macd_signal is not None
        assert signals.macd_histogram is not None
        assert signals.bb_upper is not None
        assert signals.bb_middle is not None
        assert signals.bb_lower is not None
        assert signals.momentum_5d is not None
        assert signals.momentum_20d is not None

    def test_minimal_data_partial_signals(self):
        """With only 25 data points, some indicators are None."""
        prices = _uptrend(25)
        signals = compute_all_signals(prices)

        assert signals.sma_20 is not None
        assert signals.sma_50 is None  # Need 50 points
        assert signals.sma_200 is None  # Need 200 points
        assert signals.rsi_14 is not None  # Need 15 points

    def test_volume_data_used(self):
        """Volume data is incorporated when provided."""
        prices = _uptrend(25)
        volumes = [1_000_000.0 + i * 10_000 for i in range(25)]

        signals = compute_all_signals(prices, volumes=volumes)

        assert signals.volume_sma_20 is not None
        assert signals.volume_ratio is not None
        assert signals.volume_ratio > 0

    def test_no_volume_data(self):
        """Volume fields are None when no volume data provided."""
        prices = _uptrend(25)
        signals = compute_all_signals(prices, volumes=None)

        assert signals.volume_sma_20 is None
        assert signals.volume_ratio is None

    def test_price_vs_sma_percentages(self):
        """Price vs SMA fields are correct percentages."""
        # Use a flat series at 100 with the last price at 110
        prices = _flat(20, price=100.0)
        prices[-1] = 110.0
        signals = compute_all_signals(prices)

        if signals.price_vs_sma20 is not None:
            # Current = 110, SMA20 ≈ 100.5 (19 * 100 + 110) / 20 = 100.5
            expected_sma = (19 * 100.0 + 110.0) / 20
            expected_pct = (110.0 - expected_sma) / expected_sma * 100
            assert signals.price_vs_sma20 == pytest.approx(expected_pct, rel=0.01)

    def test_rsi_extremes_flagged(self):
        """Overbought/oversold flags align with RSI thresholds."""
        # Strong uptrend => overbought
        up_prices = _uptrend(50, start=100.0, step=2.0)
        up_signals = compute_all_signals(up_prices)
        if up_signals.rsi_14 is not None and up_signals.rsi_14 > 70:
            assert up_signals.rsi_overbought is True

        # Strong downtrend => oversold
        down_prices = _downtrend(50, start=200.0, step=2.0)
        down_signals = compute_all_signals(down_prices)
        if down_signals.rsi_14 is not None and down_signals.rsi_14 < 30:
            assert down_signals.rsi_oversold is True

    def test_net_signal_is_bullish_minus_bearish(self):
        """net_signal always equals bullish_signals - bearish_signals."""
        for gen in [_uptrend, _downtrend, _oscillating, _flat]:
            prices = gen(250)
            signals = compute_all_signals(prices)
            assert signals.net_signal == signals.bullish_signals - signals.bearish_signals

    def test_bollinger_band_flags(self):
        """Price above upper / below lower BB flags are set correctly."""
        # Create a series where the last price is extremely high
        prices = _flat(25, price=100.0)
        prices[-1] = 200.0  # Way above upper band
        signals = compute_all_signals(prices)
        if signals.bb_upper is not None:
            assert signals.price_above_upper_bb is True

    def test_bb_width_calculated(self):
        """Bollinger Band width is (upper - lower) / middle."""
        prices = _oscillating(25)
        signals = compute_all_signals(prices)
        if signals.bb_upper is not None and signals.bb_lower is not None and signals.bb_middle:
            expected_width = (signals.bb_upper - signals.bb_lower) / signals.bb_middle
            assert signals.bb_width == pytest.approx(expected_width)


# ---------------------------------------------------------------------------
# TechnicalSignals dataclass
# ---------------------------------------------------------------------------
class TestTechnicalSignals:
    """Tests for the TechnicalSignals frozen dataclass."""

    def test_frozen_immutability(self):
        """TechnicalSignals is frozen (immutable)."""
        signals = compute_all_signals(_uptrend(250))
        with pytest.raises(AttributeError):
            signals.sma_20 = 999.0  # type: ignore[misc]

    def test_all_fields_present(self):
        """TechnicalSignals has all documented fields."""
        signals = compute_all_signals(_uptrend(250))
        expected_fields = {
            "sma_20", "sma_50", "sma_200",
            "price_vs_sma20", "price_vs_sma50", "price_vs_sma200",
            "golden_cross", "death_cross",
            "rsi_14", "rsi_overbought", "rsi_oversold",
            "macd_line", "macd_signal", "macd_histogram",
            "macd_bullish_crossover", "macd_bearish_crossover",
            "bb_upper", "bb_middle", "bb_lower", "bb_width",
            "price_above_upper_bb", "price_below_lower_bb",
            "volume_sma_20", "volume_ratio",
            "momentum_5d", "momentum_20d",
            "bullish_signals", "bearish_signals", "net_signal",
        }
        actual_fields = {f.name for f in signals.__dataclass_fields__.values()}
        assert expected_fields == actual_fields
