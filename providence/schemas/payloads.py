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
