"""Sector-level feature extraction for REGIME-SECTOR.

Pure functions — no LLM calls, no external dependencies.
Classification: FROZEN component.

Groups MarketStateFragments by GICS sector via ticker→sector mapping,
then extracts per-sector RegimeFeatures for independent HMM classification.
"""

from dataclasses import dataclass

from providence.agents.regime.regime_features import (
    RegimeFeatures,
    compute_drawdown,
    compute_realized_vol,
    compute_vol_of_vol,
)
from providence.schemas.enums import DataType
from providence.schemas.market_state import MarketStateFragment


# ---------------------------------------------------------------------------
# Ticker-to-GICS sector mapping (MVP hardcoded, Phase 4 → dynamic lookup)
# ---------------------------------------------------------------------------
TICKER_SECTOR_MAP: dict[str, str] = {
    # Technology
    "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology",
    "GOOG": "Technology", "META": "Technology", "NVDA": "Technology",
    "AVGO": "Technology", "ADBE": "Technology", "CRM": "Technology",
    "AMD": "Technology", "INTC": "Technology", "CSCO": "Technology",
    "ORCL": "Technology", "IBM": "Technology", "QCOM": "Technology",
    # Healthcare
    "JNJ": "Healthcare", "UNH": "Healthcare", "PFE": "Healthcare",
    "ABBV": "Healthcare", "MRK": "Healthcare", "LLY": "Healthcare",
    "TMO": "Healthcare", "ABT": "Healthcare", "AMGN": "Healthcare",
    "BMY": "Healthcare", "GILD": "Healthcare", "MDT": "Healthcare",
    # Financials
    "JPM": "Financials", "BAC": "Financials", "WFC": "Financials",
    "GS": "Financials", "MS": "Financials", "C": "Financials",
    "BLK": "Financials", "AXP": "Financials", "SCHW": "Financials",
    "BRK.B": "Financials", "V": "Financials", "MA": "Financials",
    # Consumer Discretionary
    "AMZN": "Consumer Discretionary", "TSLA": "Consumer Discretionary",
    "HD": "Consumer Discretionary", "NKE": "Consumer Discretionary",
    "MCD": "Consumer Discretionary", "SBUX": "Consumer Discretionary",
    "LOW": "Consumer Discretionary", "TJX": "Consumer Discretionary",
    # Consumer Staples
    "PG": "Consumer Staples", "KO": "Consumer Staples",
    "PEP": "Consumer Staples", "WMT": "Consumer Staples",
    "COST": "Consumer Staples", "CL": "Consumer Staples",
    "MDLZ": "Consumer Staples", "PM": "Consumer Staples",
    # Energy
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy",
    "SLB": "Energy", "EOG": "Energy", "MPC": "Energy",
    "PSX": "Energy", "VLO": "Energy", "OXY": "Energy",
    # Industrials
    "CAT": "Industrials", "BA": "Industrials", "HON": "Industrials",
    "UNP": "Industrials", "GE": "Industrials", "RTX": "Industrials",
    "MMM": "Industrials", "LMT": "Industrials", "DE": "Industrials",
    # Utilities
    "NEE": "Utilities", "DUK": "Utilities", "SO": "Utilities",
    "D": "Utilities", "AEP": "Utilities", "SRE": "Utilities",
    # Real Estate
    "AMT": "Real Estate", "PLD": "Real Estate", "CCI": "Real Estate",
    "EQIX": "Real Estate", "SPG": "Real Estate", "O": "Real Estate",
    # Materials
    "LIN": "Materials", "APD": "Materials", "SHW": "Materials",
    "ECL": "Materials", "FCX": "Materials", "NEM": "Materials",
    # Communication Services
    "DIS": "Communication Services", "NFLX": "Communication Services",
    "CMCSA": "Communication Services", "T": "Communication Services",
    "VZ": "Communication Services", "TMUS": "Communication Services",
    # Index / Broad market (no sector)
    "SPY": None, "QQQ": None, "IWM": None, "DIA": None,
    "VIX": None, "^GSPC": None, "^DJI": None, "^IXIC": None,
}

# All valid GICS sectors
GICS_SECTORS = sorted({
    s for s in TICKER_SECTOR_MAP.values() if s is not None
})


def get_sector(ticker: str) -> str | None:
    """Map a ticker to its GICS sector.

    Args:
        ticker: Ticker symbol (e.g., "AAPL").

    Returns:
        GICS sector name, or None if ticker is an index / unknown.
    """
    return TICKER_SECTOR_MAP.get(ticker)


@dataclass(frozen=True)
class SectorFragmentGroup:
    """Fragments for a single sector, ready for feature extraction."""
    sector: str
    price_closes: list[float]
    ticker_count: int


def group_fragments_by_sector(
    fragments: list[MarketStateFragment],
) -> dict[str, SectorFragmentGroup]:
    """Group PRICE_OHLCV fragments by sector and extract close prices.

    Aggregates close prices across all tickers in each sector,
    sorted chronologically. This gives a sector-level price composite.

    Args:
        fragments: All MarketStateFragments from CONTEXT-SVC.

    Returns:
        Dict of sector name → SectorFragmentGroup.
    """
    # Collect price fragments per sector
    sector_prices: dict[str, list[tuple[float, float]]] = {}  # sector → [(timestamp_epoch, close)]
    sector_tickers: dict[str, set[str]] = {}

    for frag in fragments:
        if frag.data_type != DataType.PRICE_OHLCV:
            continue
        if frag.entity is None:
            continue

        sector = get_sector(frag.entity)
        if sector is None:
            continue

        close = frag.payload.get("close")
        if close is None:
            continue

        if sector not in sector_prices:
            sector_prices[sector] = []
            sector_tickers[sector] = set()

        sector_prices[sector].append((frag.timestamp.timestamp(), float(close)))
        sector_tickers[sector].add(frag.entity)

    # Build SectorFragmentGroups
    result: dict[str, SectorFragmentGroup] = {}
    for sector, ts_prices in sector_prices.items():
        # Sort by timestamp
        ts_prices.sort(key=lambda x: x[0])
        closes = [p for _, p in ts_prices]

        result[sector] = SectorFragmentGroup(
            sector=sector,
            price_closes=closes,
            ticker_count=len(sector_tickers[sector]),
        )

    return result


def extract_sector_features(group: SectorFragmentGroup) -> RegimeFeatures:
    """Extract regime features for a single sector from its price data.

    Uses the same volatility computations as the global regime features,
    but applied to sector-level aggregated prices.

    Args:
        group: SectorFragmentGroup with chronological close prices.

    Returns:
        RegimeFeatures with price-based features populated.
    """
    prices = group.price_closes

    realized_vol_20d = compute_realized_vol(prices, window=20)
    realized_vol_60d = compute_realized_vol(prices, window=60)
    vol_of_vol = compute_vol_of_vol(prices, window=60, sub_window=5)
    price_drawdown_pct = compute_drawdown(prices)

    # 20-day momentum
    price_momentum_20d = None
    if len(prices) >= 21 and prices[-21] > 0:
        price_momentum_20d = (prices[-1] / prices[-21]) - 1.0

    return RegimeFeatures(
        realized_vol_20d=realized_vol_20d,
        realized_vol_60d=realized_vol_60d,
        vol_of_vol=vol_of_vol,
        price_drawdown_pct=price_drawdown_pct,
        price_momentum_20d=price_momentum_20d,
    )


def compute_relative_stress(
    sector_composite: float,
    market_composite: float,
) -> float:
    """Compute relative stress of a sector vs the overall market.

    Args:
        sector_composite: Sector's composite stress score [0, 1].
        market_composite: Market-wide composite stress score [0, 1].

    Returns:
        Relative stress in range [-1.0, +1.0].
        Positive = sector more stressed than market.
        Negative = sector calmer than market.
    """
    # Simple difference, clamped to [-1, +1]
    diff = sector_composite - market_composite
    return max(-1.0, min(1.0, diff * 2.0))  # Scale by 2 for more differentiation


def identify_key_signals(
    features: RegimeFeatures,
    sector_composite: float,
    market_composite: float,
) -> list[str]:
    """Identify top driving signals for a sector's regime classification.

    Args:
        features: Sector's extracted features.
        sector_composite: Sector's composite stress score.
        market_composite: Market-wide composite stress score.

    Returns:
        List of human-readable signal descriptions.
    """
    signals: list[str] = []

    if features.realized_vol_20d is not None:
        if features.realized_vol_20d > 0.40:
            signals.append("high realized vol")
        elif features.realized_vol_20d < 0.12:
            signals.append("low realized vol")

    if features.vol_of_vol is not None and features.vol_of_vol > 0.15:
        signals.append("unstable volatility")

    if features.price_drawdown_pct is not None:
        if features.price_drawdown_pct < -0.15:
            signals.append("significant drawdown")
        elif features.price_drawdown_pct < -0.05:
            signals.append("moderate drawdown")

    if features.price_momentum_20d is not None:
        if features.price_momentum_20d > 0.05:
            signals.append("positive momentum")
        elif features.price_momentum_20d < -0.05:
            signals.append("negative momentum")

    if sector_composite > market_composite + 0.15:
        signals.append("elevated vs market")
    elif sector_composite < market_composite - 0.15:
        signals.append("calmer than market")

    return signals
