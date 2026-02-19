"""Regime feature extraction for REGIME-STAT.

Pure functions — no LLM calls, no external dependencies.
Classification: FROZEN component.

Extracts volatility, yield curve, credit spread, and macro features
from MarketStateFragments for HMM regime classification.
"""

import math
from dataclasses import dataclass

from providence.schemas.enums import DataType
from providence.schemas.market_state import MarketStateFragment


@dataclass(frozen=True)
class RegimeFeatures:
    """Aggregated features for HMM regime classification.

    All fields are optional — the HMM uses whatever features
    are available and falls back to priors for missing data.
    """
    # Volatility features (from PRICE_OHLCV)
    realized_vol_20d: float | None = None   # Annualized 20-day realized vol
    realized_vol_60d: float | None = None   # Annualized 60-day realized vol
    vol_of_vol: float | None = None         # Volatility of volatility (instability)
    price_drawdown_pct: float | None = None # Max drawdown from recent peak
    price_momentum_20d: float | None = None # 20-day price momentum (% change)

    # Yield curve features (from MACRO_YIELD_CURVE)
    yield_spread_2s10s: float | None = None  # 2-10Y spread in bps

    # Credit features (from MACRO_CDS)
    cds_ig_spread: float | None = None  # Investment-grade CDS spread in bps

    # Implied volatility (from OPTIONS_CHAIN)
    vix_proxy: float | None = None  # Average implied vol as VIX proxy

    # Macro momentum (from MACRO_ECONOMIC)
    macro_momentum: float | None = None  # Composite macro momentum score


def compute_realized_vol(prices: list[float], window: int = 20) -> float | None:
    """Compute annualized realized volatility from close prices.

    Uses log returns and annualizes by sqrt(252).

    Args:
        prices: Close prices in chronological order (oldest first).
        window: Lookback window for volatility calculation.

    Returns:
        Annualized realized volatility, or None if insufficient data.
    """
    if len(prices) < window + 1:
        return None

    # Compute log returns for the window
    recent = prices[-(window + 1):]
    log_returns = []
    for i in range(1, len(recent)):
        if recent[i - 1] > 0 and recent[i] > 0:
            log_returns.append(math.log(recent[i] / recent[i - 1]))

    if len(log_returns) < 2:
        return None

    # Standard deviation of log returns
    mean_ret = sum(log_returns) / len(log_returns)
    variance = sum((r - mean_ret) ** 2 for r in log_returns) / (len(log_returns) - 1)
    daily_vol = math.sqrt(variance)

    # Annualize
    return daily_vol * math.sqrt(252)


def compute_vol_of_vol(
    prices: list[float], window: int = 60, sub_window: int = 5
) -> float | None:
    """Compute volatility of volatility (vol-of-vol).

    Measures instability: rolling sub_window vol computed over window,
    then takes std dev of those rolling vols.

    Args:
        prices: Close prices in chronological order.
        window: Total lookback window.
        sub_window: Rolling sub-window for vol estimates.

    Returns:
        Vol-of-vol, or None if insufficient data.
    """
    if len(prices) < window + 1:
        return None

    recent = prices[-(window + 1):]

    # Compute rolling realized vols
    rolling_vols = []
    for i in range(sub_window + 1, len(recent) + 1):
        sub_prices = recent[i - sub_window - 1: i]
        vol = compute_realized_vol(sub_prices, window=sub_window)
        if vol is not None:
            rolling_vols.append(vol)

    if len(rolling_vols) < 2:
        return None

    # Standard deviation of rolling vols
    mean_vol = sum(rolling_vols) / len(rolling_vols)
    variance = sum((v - mean_vol) ** 2 for v in rolling_vols) / (len(rolling_vols) - 1)
    return math.sqrt(variance)


def compute_drawdown(prices: list[float]) -> float | None:
    """Compute max drawdown from peak over the price series.

    Args:
        prices: Close prices in chronological order.

    Returns:
        Max drawdown as a negative percentage (e.g., -0.15 for -15%), or None.
    """
    if len(prices) < 2:
        return None

    peak = prices[0]
    max_dd = 0.0

    for price in prices[1:]:
        if price > peak:
            peak = price
        dd = (price - peak) / peak if peak > 0 else 0.0
        if dd < max_dd:
            max_dd = dd

    return max_dd


def extract_regime_features(
    fragments: list[MarketStateFragment],
) -> RegimeFeatures:
    """Extract regime features from a collection of MarketStateFragments.

    Groups fragments by data type and extracts relevant features.

    Args:
        fragments: MarketStateFragments from CONTEXT-SVC.

    Returns:
        RegimeFeatures with whatever data is available.
    """
    # Collect price data (sorted by timestamp)
    price_frags = sorted(
        [f for f in fragments if f.data_type == DataType.PRICE_OHLCV],
        key=lambda f: f.timestamp,
    )
    close_prices = []
    for frag in price_frags:
        if "close" in frag.payload:
            close_prices.append(float(frag.payload["close"]))

    # Compute volatility features from prices
    realized_vol_20d = compute_realized_vol(close_prices, window=20)
    realized_vol_60d = compute_realized_vol(close_prices, window=60)
    vol_of_vol = compute_vol_of_vol(close_prices, window=60, sub_window=5)
    price_drawdown_pct = compute_drawdown(close_prices)

    # 20-day momentum
    price_momentum_20d = None
    if len(close_prices) >= 21 and close_prices[-21] > 0:
        price_momentum_20d = (close_prices[-1] / close_prices[-21]) - 1.0

    # Extract yield curve spread (latest)
    yield_spread_2s10s = None
    yield_frags = [f for f in fragments if f.data_type == DataType.MACRO_YIELD_CURVE]
    if yield_frags:
        latest = max(yield_frags, key=lambda f: f.timestamp)
        spreads = latest.payload.get("spreads", {})
        yield_spread_2s10s = spreads.get("2s10s")
        if yield_spread_2s10s is None:
            yield_spread_2s10s = latest.payload.get("spread_2s10s")

    # Extract CDS spread (latest IG)
    cds_ig_spread = None
    cds_frags = [f for f in fragments if f.data_type == DataType.MACRO_CDS]
    if cds_frags:
        latest = max(cds_frags, key=lambda f: f.timestamp)
        cds_ig_spread = latest.payload.get("spread_bps")

    # Extract VIX proxy from options (average implied vol)
    vix_proxy = None
    options_frags = [f for f in fragments if f.data_type == DataType.OPTIONS_CHAIN]
    if options_frags:
        ivs = []
        for frag in options_frags:
            iv = frag.payload.get("implied_volatility")
            if iv is not None:
                ivs.append(float(iv))
        if ivs:
            vix_proxy = sum(ivs) / len(ivs)

    # Extract macro momentum (latest economic indicator value)
    macro_momentum = None
    macro_frags = [f for f in fragments if f.data_type == DataType.MACRO_ECONOMIC]
    if macro_frags:
        latest = max(macro_frags, key=lambda f: f.timestamp)
        macro_momentum = latest.payload.get("value")

    return RegimeFeatures(
        realized_vol_20d=realized_vol_20d,
        realized_vol_60d=realized_vol_60d,
        vol_of_vol=vol_of_vol,
        price_drawdown_pct=price_drawdown_pct,
        price_momentum_20d=price_momentum_20d,
        yield_spread_2s10s=yield_spread_2s10s,
        cds_ig_spread=cds_ig_spread,
        vix_proxy=vix_proxy,
        macro_momentum=macro_momentum,
    )
