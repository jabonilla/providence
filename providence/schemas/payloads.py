"""Typed payload schemas for MarketStateFragment data.

Each DataType in the Data Type Registry has a corresponding payload
schema defined here. Payloads are stored as the `payload` field of
MarketStateFragment.

Spec Reference: Technical Spec v2.3, Sections 2.1 and 4.1
"""

from datetime import date
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class PricePayload(BaseModel):
    """OHLCV price data payload for DataType.PRICE_OHLCV.

    Produced by: PERCEPT-PRICE
    Source: Polygon.io REST API
    """

    model_config = ConfigDict(frozen=True)

    open: float = Field(..., description="Opening price")
    high: float = Field(..., description="Highest price")
    low: float = Field(..., description="Lowest price")
    close: float = Field(..., description="Closing price")
    volume: int = Field(..., ge=0, description="Trading volume")
    vwap: Optional[float] = Field(default=None, description="Volume-weighted average price")
    num_trades: Optional[int] = Field(default=None, ge=0, description="Number of trades")
    timeframe: str = Field(..., description="Timeframe: '1D', '1H', '1min', etc.")


class FilingType(str, Enum):
    """SEC filing type classification."""
    FORM_10K = "10-K"
    FORM_10Q = "10-Q"
    FORM_8K = "8-K"


class FilingPayload(BaseModel):
    """SEC filing payload for FILING_10K, FILING_10Q, and FILING_8K data types.

    Produced by: PERCEPT-FILING
    Source: SEC EDGAR full-text search and XBRL API

    For 10-K/10-Q: includes key financial metrics extracted from XBRL.
    For 8-K: includes event type, description, and material impact flag.
    """

    model_config = ConfigDict(frozen=True)

    filing_type: FilingType = Field(..., description="Type of SEC filing")
    company_name: str = Field(..., description="Company legal name")
    cik: str = Field(..., description="SEC Central Index Key")
    ticker: str = Field(..., description="Stock ticker symbol")
    filed_date: date = Field(..., description="Date the filing was submitted to SEC")
    period_of_report: date = Field(..., description="Period covered by the filing")

    # Financial metrics (10-K / 10-Q only)
    revenue: Optional[float] = Field(default=None, description="Total revenue")
    net_income: Optional[float] = Field(default=None, description="Net income")
    eps: Optional[float] = Field(default=None, description="Earnings per share")
    total_assets: Optional[float] = Field(default=None, description="Total assets")
    total_liabilities: Optional[float] = Field(default=None, description="Total liabilities")
    operating_cash_flow: Optional[float] = Field(default=None, description="Operating cash flow")
    key_ratios: dict[str, float] = Field(
        default_factory=dict,
        description="Key financial ratios (e.g., debt_to_equity, current_ratio)",
    )

    # Event fields (8-K only)
    event_type: Optional[str] = Field(default=None, description="Type of material event")
    event_description: Optional[str] = Field(default=None, description="Description of the event")
    material_impact: Optional[bool] = Field(default=None, description="Whether the event is material")

    # Text excerpt for agent context
    raw_text_excerpt: str = Field(
        default="",
        description="First N chars of relevant section for agent context",
    )


class NewsPayload(BaseModel):
    """News and sentiment payload for DataType.SENTIMENT_NEWS.

    Produced by: PERCEPT-SENTIMENT
    Source: News API, Finnhub, or third-party news aggregators
    """

    model_config = ConfigDict(frozen=True)

    source: str = Field(..., description="News source name")
    headline: str = Field(..., description="Article headline")
    summary: str = Field(..., description="Article summary/excerpt")
    sentiment_score: float = Field(
        ..., ge=-1.0, le=1.0, description="Sentiment score: -1.0=bearish, 0.0=neutral, 1.0=bullish"
    )
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Relevance score 0.0-1.0")
    tickers_mentioned: list[str] = Field(..., description="Stock tickers referenced in article")
    published_utc: str = Field(..., description="ISO timestamp of publication")
    article_url: Optional[str] = Field(default=None, description="URL to full article")
    keywords: list[str] = Field(default_factory=list, description="Extracted keywords from article")


class OptionsPayload(BaseModel):
    """Options chain payload for DataType.OPTIONS_CHAIN.

    Produced by: PERCEPT-OPTIONS
    Source: Polygon.io Options API or other derivatives data provider
    """

    model_config = ConfigDict(frozen=True)

    contract_type: str = Field(..., description="Option type: 'call' or 'put'")
    strike_price: float = Field(..., description="Strike price of the option")
    expiration_date: str = Field(..., description="Expiration date in YYYY-MM-DD format")
    underlying_price: float = Field(..., description="Current price of underlying asset")
    last_price: float = Field(..., description="Last trade price of the option")
    bid: Optional[float] = Field(default=None, description="Current bid price")
    ask: Optional[float] = Field(default=None, description="Current ask price")
    volume: int = Field(..., ge=0, description="Trading volume for the option")
    open_interest: int = Field(..., ge=0, description="Open interest for the contract")
    implied_volatility: Optional[float] = Field(default=None, description="Implied volatility as decimal")
    delta: Optional[float] = Field(default=None, description="Greek: delta")
    gamma: Optional[float] = Field(default=None, description="Greek: gamma")
    theta: Optional[float] = Field(default=None, description="Greek: theta")
    vega: Optional[float] = Field(default=None, description="Greek: vega")
    snapshot_time: str = Field(..., description="ISO timestamp when snapshot was taken")


class CdsPayload(BaseModel):
    """Credit Default Swap payload for DataType.MACRO_CDS.

    Produced by: PERCEPT-CDS
    Source: Bloomberg, Markit, or other CDS data provider
    """

    model_config = ConfigDict(frozen=True)

    reference_entity: str = Field(..., description="Company or sovereign name")
    tenor: str = Field(..., description="CDS tenor (e.g., '5Y', '1Y', '10Y')")
    spread_bps: float = Field(..., description="CDS spread in basis points")
    previous_spread_bps: Optional[float] = Field(default=None, description="Prior day's CDS spread in bps")
    spread_change_bps: Optional[float] = Field(default=None, description="Daily change in spread (bps)")
    recovery_rate: float = Field(default=0.4, description="Assumed recovery rate for calculation")
    currency: str = Field(default="USD", description="Currency of the CDS")
    observation_date: str = Field(..., description="Observation date in YYYY-MM-DD format")


class MacroYieldPayload(BaseModel):
    """Yield curve payload for DataType.MACRO_YIELD_CURVE.

    Produced by: PERCEPT-YIELD-CURVE
    Source: FRED (Federal Reserve Economic Data) or Treasury.gov
    """

    model_config = ConfigDict(frozen=True)

    curve_date: str = Field(..., description="Yield curve date in YYYY-MM-DD format")
    tenors: dict[str, float] = Field(
        ..., description="Mapping of tenor labels to yields (e.g., {'1M': 5.32, '10Y': 3.95})"
    )
    spread_2s10s: Optional[float] = Field(default=None, description="2Y-10Y spread in basis points")
    spread_3m10y: Optional[float] = Field(default=None, description="3M-10Y spread in basis points")
    curve_source: str = Field(..., description="Source of yield curve data (e.g., 'FRED', 'Treasury.gov')")
    previous_tenors: Optional[dict[str, float]] = Field(
        default=None, description="Prior day's yield curve for comparison"
    )


class MacroEconomicPayload(BaseModel):
    """Macroeconomic indicator payload for DataType.MACRO_ECONOMIC.

    Produced by: PERCEPT-MACRO
    Source: FRED, BLS (Bureau of Labor Statistics), or other official government sources
    """

    model_config = ConfigDict(frozen=True)

    indicator: str = Field(
        ..., description="Economic indicator name (e.g., 'GDP', 'CPI', 'UNEMPLOYMENT_RATE')"
    )
    value: float = Field(..., description="The indicator value")
    previous_value: Optional[float] = Field(default=None, description="Prior period's value")
    period: str = Field(..., description="Period of the indicator (e.g., '2026-Q1', '2026-01')")
    frequency: str = Field(..., description="Frequency (e.g., 'MONTHLY', 'QUARTERLY', 'ANNUAL')")
    unit: str = Field(..., description="Unit of measurement (e.g., 'PERCENT', 'BILLIONS_USD', 'THOUSANDS')")
    source_series_id: str = Field(..., description="Source series ID (e.g., FRED series like 'CPIAUCSL')")
    observation_date: str = Field(..., description="Observation date in YYYY-MM-DD format")
    revision_number: int = Field(default=0, description="Revision number for data releases")
