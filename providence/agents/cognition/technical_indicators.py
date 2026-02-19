"""Technical indicator computations for COGNIT-TECHNICAL.

Pure functions — no LLM calls, no external dependencies.
Classification: FROZEN component.

All indicators operate on lists of float values (typically close prices).
Lists are expected in chronological order (oldest first).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class TechnicalSignals:
    """Aggregated technical signals from all indicator computations."""
    # Moving averages
    sma_20: float | None
    sma_50: float | None
    sma_200: float | None
    price_vs_sma20: float | None  # % above/below SMA20
    price_vs_sma50: float | None
    price_vs_sma200: float | None
    golden_cross: bool  # SMA50 > SMA200 (bullish)
    death_cross: bool   # SMA50 < SMA200 (bearish)

    # RSI
    rsi_14: float | None  # 0-100
    rsi_overbought: bool  # RSI > 70
    rsi_oversold: bool    # RSI < 30

    # MACD
    macd_line: float | None
    macd_signal: float | None
    macd_histogram: float | None
    macd_bullish_crossover: bool  # MACD crosses above signal
    macd_bearish_crossover: bool  # MACD crosses below signal

    # Bollinger Bands
    bb_upper: float | None
    bb_middle: float | None  # = SMA20
    bb_lower: float | None
    bb_width: float | None   # (upper - lower) / middle
    price_above_upper_bb: bool
    price_below_lower_bb: bool

    # Volume
    volume_sma_20: float | None
    volume_ratio: float | None  # current volume / SMA20 volume

    # Momentum
    momentum_5d: float | None   # % change over 5 days
    momentum_20d: float | None  # % change over 20 days

    # Overall signal
    bullish_signals: int   # count of bullish signals
    bearish_signals: int   # count of bearish signals
    net_signal: int        # bullish - bearish


def compute_sma(prices: list[float], period: int) -> float | None:
    """Compute Simple Moving Average over the given period.

    Args:
        prices: Chronological price list (oldest first).
        period: Number of periods for the average.

    Returns:
        SMA value, or None if insufficient data.
    """
    if len(prices) < period:
        return None
    return sum(prices[-period:]) / period


def compute_ema(prices: list[float], period: int) -> float | None:
    """Compute Exponential Moving Average over the given period.

    Uses the standard smoothing factor: 2 / (period + 1).

    Returns:
        EMA value, or None if insufficient data.
    """
    if len(prices) < period:
        return None
    multiplier = 2.0 / (period + 1)
    ema = sum(prices[:period]) / period  # Seed with SMA
    for price in prices[period:]:
        ema = (price - ema) * multiplier + ema
    return ema


def compute_rsi(prices: list[float], period: int = 14) -> float | None:
    """Compute Relative Strength Index.

    Uses Wilder's smoothing method (exponential).

    Args:
        prices: Chronological prices (oldest first).
        period: RSI lookback period (default 14).

    Returns:
        RSI value (0-100), or None if insufficient data.
    """
    if len(prices) < period + 1:
        return None

    # Compute price changes
    changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]

    # Separate gains and losses
    gains = [max(c, 0.0) for c in changes]
    losses = [abs(min(c, 0.0)) for c in changes]

    # Initial average gain/loss (first `period` changes)
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    # Wilder's smoothing for remaining changes
    for i in range(period, len(changes)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def compute_macd(
    prices: list[float],
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> tuple[float | None, float | None, float | None]:
    """Compute MACD (Moving Average Convergence Divergence).

    Returns:
        Tuple of (macd_line, signal_line, histogram), or (None, None, None) if insufficient data.
    """
    if len(prices) < slow_period + signal_period:
        return None, None, None

    fast_ema = compute_ema(prices, fast_period)
    slow_ema = compute_ema(prices, slow_period)

    if fast_ema is None or slow_ema is None:
        return None, None, None

    macd_line = fast_ema - slow_ema

    # Compute MACD history for signal line
    # We need enough data points to compute MACD over time for the signal EMA
    macd_history = []
    for i in range(slow_period, len(prices) + 1):
        sub_prices = prices[:i]
        fast = compute_ema(sub_prices, fast_period)
        slow = compute_ema(sub_prices, slow_period)
        if fast is not None and slow is not None:
            macd_history.append(fast - slow)

    if len(macd_history) < signal_period:
        return macd_line, None, None

    signal_line = compute_ema(macd_history, signal_period)

    if signal_line is None:
        return macd_line, None, None

    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_bollinger_bands(
    prices: list[float], period: int = 20, num_std: float = 2.0
) -> tuple[float | None, float | None, float | None]:
    """Compute Bollinger Bands.

    Returns:
        Tuple of (upper_band, middle_band, lower_band), or (None, None, None) if insufficient data.
    """
    if len(prices) < period:
        return None, None, None

    window = prices[-period:]
    middle = sum(window) / period

    # Standard deviation
    variance = sum((p - middle) ** 2 for p in window) / period
    std_dev = variance ** 0.5

    upper = middle + num_std * std_dev
    lower = middle - num_std * std_dev

    return upper, middle, lower


def compute_momentum(prices: list[float], period: int) -> float | None:
    """Compute price momentum as percentage change over the period.

    Returns:
        Percentage change (e.g., 5.0 = 5%), or None if insufficient data.
    """
    if len(prices) < period + 1 or prices[-(period + 1)] == 0:
        return None
    return ((prices[-1] - prices[-(period + 1)]) / prices[-(period + 1)]) * 100.0


def compute_all_signals(
    close_prices: list[float],
    volumes: list[float] | None = None,
) -> TechnicalSignals:
    """Compute all technical signals from price and volume data.

    Args:
        close_prices: Chronological close prices (oldest first). Need at least 200 for full signals.
        volumes: Chronological volumes (oldest first). Optional.

    Returns:
        TechnicalSignals with all indicators and signal counts.
    """
    current_price = close_prices[-1] if close_prices else 0.0

    # Moving averages
    sma_20 = compute_sma(close_prices, 20)
    sma_50 = compute_sma(close_prices, 50)
    sma_200 = compute_sma(close_prices, 200)

    price_vs_sma20 = ((current_price - sma_20) / sma_20 * 100) if sma_20 else None
    price_vs_sma50 = ((current_price - sma_50) / sma_50 * 100) if sma_50 else None
    price_vs_sma200 = ((current_price - sma_200) / sma_200 * 100) if sma_200 else None

    golden_cross = (sma_50 is not None and sma_200 is not None and sma_50 > sma_200)
    death_cross = (sma_50 is not None and sma_200 is not None and sma_50 < sma_200)

    # RSI
    rsi_14 = compute_rsi(close_prices, 14)
    rsi_overbought = (rsi_14 is not None and rsi_14 > 70)
    rsi_oversold = (rsi_14 is not None and rsi_14 < 30)

    # MACD
    macd_line, macd_signal, macd_histogram = compute_macd(close_prices)
    macd_bullish = (macd_histogram is not None and macd_histogram > 0 and
                    macd_line is not None and macd_signal is not None and
                    macd_line > macd_signal)
    macd_bearish = (macd_histogram is not None and macd_histogram < 0 and
                    macd_line is not None and macd_signal is not None and
                    macd_line < macd_signal)

    # Bollinger Bands
    bb_upper, bb_middle, bb_lower = compute_bollinger_bands(close_prices)
    bb_width = ((bb_upper - bb_lower) / bb_middle) if (bb_upper is not None and bb_middle and bb_lower is not None) else None
    price_above_upper = (bb_upper is not None and current_price > bb_upper)
    price_below_lower = (bb_lower is not None and current_price < bb_lower)

    # Volume
    vol_sma_20 = compute_sma(volumes, 20) if volumes else None
    vol_ratio = (volumes[-1] / vol_sma_20) if (volumes and vol_sma_20 and vol_sma_20 > 0) else None

    # Momentum
    momentum_5d = compute_momentum(close_prices, 5)
    momentum_20d = compute_momentum(close_prices, 20)

    # Count signals
    bullish = 0
    bearish = 0

    if golden_cross: bullish += 1
    if death_cross: bearish += 1
    if rsi_oversold: bullish += 1
    if rsi_overbought: bearish += 1
    if macd_bullish: bullish += 1
    if macd_bearish: bearish += 1
    if price_below_lower: bullish += 1  # mean reversion signal
    if price_above_upper: bearish += 1  # mean reversion signal
    if price_vs_sma200 is not None and price_vs_sma200 > 0: bullish += 1
    if price_vs_sma200 is not None and price_vs_sma200 < 0: bearish += 1
    if momentum_20d is not None and momentum_20d > 5: bullish += 1
    if momentum_20d is not None and momentum_20d < -5: bearish += 1

    return TechnicalSignals(
        sma_20=sma_20,
        sma_50=sma_50,
        sma_200=sma_200,
        price_vs_sma20=price_vs_sma20,
        price_vs_sma50=price_vs_sma50,
        price_vs_sma200=price_vs_sma200,
        golden_cross=golden_cross,
        death_cross=death_cross,
        rsi_14=rsi_14,
        rsi_overbought=rsi_overbought,
        rsi_oversold=rsi_oversold,
        macd_line=macd_line,
        macd_signal=macd_signal,
        macd_histogram=macd_histogram,
        macd_bullish_crossover=macd_bullish,
        macd_bearish_crossover=macd_bearish,
        bb_upper=bb_upper,
        bb_middle=bb_middle,
        bb_lower=bb_lower,
        bb_width=bb_width,
        price_above_upper_bb=price_above_upper,
        price_below_lower_bb=price_below_lower,
        volume_sma_20=vol_sma_20,
        volume_ratio=vol_ratio,
        momentum_5d=momentum_5d,
        momentum_20d=momentum_20d,
        bullish_signals=bullish,
        bearish_signals=bearish,
        net_signal=bullish - bearish,
    )
