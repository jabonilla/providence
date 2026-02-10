"""Typed payload schemas for MarketStateFragment data.

Each DataType in the Data Type Registry has a corresponding payload
schema defined here. Payloads are stored as the `payload` field of
MarketStateFragment.

Spec Reference: Technical Spec v2.3, Sections 2.1 and 4.1
"""

from typing import Optional

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
