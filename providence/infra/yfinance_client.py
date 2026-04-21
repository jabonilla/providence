"""Yahoo Finance client for market data and fundamentals ingestion.

Provides async access to stock fundamentals, price history, and institutional
holdings via the yfinance library. Uses asyncio.to_thread since yfinance is sync.

Spec Reference: Technical Spec v2.3, Section 4.1 (Perception Agents)

Usage:
    client = YFinanceClient()
    fundamentals = await client.get_fundamentals("AAPL")
"""

import asyncio
import time
from typing import Any

import structlog

from providence.exceptions import DataIngestionError, ExternalAPIError

logger = structlog.get_logger()


class YFinanceClient:
    """Async wrapper around the yfinance library.

    Handles sync-to-async conversion, rate limiting, retries, and error mapping.
    No API key needed — yfinance is free.
    """

    MIN_REQUEST_INTERVAL = 0.5  # seconds between requests to avoid Yahoo throttling
    MAX_RETRIES = 3
    RETRY_BACKOFF_BASE = 1.0  # seconds

    def __init__(self, timeout: float = 30.0) -> None:
        """Initialize the YFinance client.

        Args:
            timeout: Not used directly by yfinance, but stored for consistency.
        """
        self._timeout = timeout
        self._last_request_time: float = 0.0
        self._yf = None

    def _ensure_yfinance(self) -> Any:
        """Lazy-import yfinance and cache the module reference."""
        if self._yf is None:
            try:
                import yfinance as yf
                self._yf = yf
            except ImportError as e:
                raise ExternalAPIError(
                    message="yfinance library required. Install with: pip install yfinance",
                    service="yfinance",
                ) from e
        return self._yf

    async def _rate_limit(self) -> None:
        """Enforce rate limiting to avoid Yahoo Finance throttling."""
        now = time.monotonic()
        elapsed = now - self._last_request_time
        if elapsed < self.MIN_REQUEST_INTERVAL:
            await asyncio.sleep(self.MIN_REQUEST_INTERVAL - elapsed)
        self._last_request_time = time.monotonic()

    async def close(self) -> None:
        """No persistent connection to close, but matches client interface."""
        pass

    def _sync_get_fundamentals(self, ticker: str) -> dict[str, Any]:
        """Synchronous fundamentals fetch — runs inside asyncio.to_thread."""
        yf = self._ensure_yfinance()
        stock = yf.Ticker(ticker)
        info = stock.info or {}
        if not info or info.get("regularMarketPrice") is None:
            raise DataIngestionError(
                message=f"No fundamental data returned for {ticker}"
            )
        return info

    async def get_fundamentals(self, ticker: str) -> dict[str, Any]:
        """Fetch fundamental data for a ticker.

        Returns a dict with keys like: marketCap, trailingPE, forwardPE,
        profitMargins, returnOnEquity, debtToEquity, beta, sector, industry, etc.

        Args:
            ticker: Stock ticker symbol (e.g., "AAPL").

        Returns:
            Dict of fundamental metrics from Yahoo Finance.

        Raises:
            ExternalAPIError: On network or API failures.
            DataIngestionError: If no data is returned.
        """
        last_error: Exception | None = None

        for attempt in range(self.MAX_RETRIES):
            try:
                await self._rate_limit()
                logger.info("Fetching fundamentals", ticker=ticker, attempt=attempt + 1)
                data = await asyncio.to_thread(self._sync_get_fundamentals, ticker)
                return data
            except DataIngestionError:
                raise
            except Exception as e:
                last_error = ExternalAPIError(
                    message=f"yfinance error for {ticker} on attempt {attempt + 1}: {e}",
                    service="yfinance",
                )
                if attempt < self.MAX_RETRIES - 1:
                    await asyncio.sleep(self.RETRY_BACKOFF_BASE * (2 ** attempt))
                    continue

        raise last_error or ExternalAPIError(
            message=f"yfinance request failed for {ticker} after all retries",
            service="yfinance",
        )

    def _sync_get_price_history(
        self, ticker: str, period: str, interval: str
    ) -> list[dict[str, Any]]:
        """Synchronous price history fetch."""
        yf = self._ensure_yfinance()
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)
        if df is None or df.empty:
            raise DataIngestionError(
                message=f"No price history returned for {ticker} (period={period})"
            )
        result = []
        for date_idx, row in df.iterrows():
            if hasattr(date_idx, "strftime"):
                date_str = date_idx.strftime("%Y-%m-%d")
            else:
                date_str = str(date_idx).split()[0]
            result.append({
                "date": date_str,
                "open": float(row.get("Open", 0.0)),
                "high": float(row.get("High", 0.0)),
                "low": float(row.get("Low", 0.0)),
                "close": float(row.get("Close", 0.0)),
                "volume": int(row.get("Volume", 0)),
                "dividends": float(row.get("Dividends", 0.0)),
                "stock_splits": float(row.get("Stock Splits", 0.0)),
            })
        return result

    async def get_price_history(
        self, ticker: str, period: str = "3mo", interval: str = "1d"
    ) -> list[dict[str, Any]]:
        """Fetch price history for a ticker.

        Args:
            ticker: Stock ticker symbol.
            period: Lookback period (e.g., "1mo", "3mo", "1y", "5y", "max").
            interval: Bar interval (e.g., "1d", "1wk", "1mo").

        Returns:
            List of bar dicts with date, open, high, low, close, volume.
        """
        last_error: Exception | None = None
        for attempt in range(self.MAX_RETRIES):
            try:
                await self._rate_limit()
                logger.info("Fetching price history", ticker=ticker, period=period)
                data = await asyncio.to_thread(
                    self._sync_get_price_history, ticker, period, interval
                )
                return data
            except DataIngestionError:
                raise
            except Exception as e:
                last_error = ExternalAPIError(
                    message=f"yfinance history error for {ticker}: {e}",
                    service="yfinance",
                )
                if attempt < self.MAX_RETRIES - 1:
                    await asyncio.sleep(self.RETRY_BACKOFF_BASE * (2 ** attempt))
                    continue

        raise last_error or ExternalAPIError(
            message=f"yfinance history failed for {ticker} after all retries",
            service="yfinance",
        )

    def _sync_get_institutional_holders(self, ticker: str) -> list[dict[str, Any]]:
        """Synchronous institutional holders fetch."""
        yf = self._ensure_yfinance()
        stock = yf.Ticker(ticker)
        df = stock.institutional_holders
        if df is None or df.empty:
            return []
        result = []
        for _, row in df.iterrows():
            result.append({
                "holder": str(row.get("Holder", "")),
                "shares": int(row.get("Shares", 0)),
                "date_reported": str(row.get("Date Reported", "")),
                "pct_out": float(row.get("% Out", 0.0)) if row.get("% Out") else None,
                "value": float(row.get("Value", 0.0)) if row.get("Value") else None,
            })
        return result

    async def get_institutional_holders(self, ticker: str) -> list[dict[str, Any]]:
        """Fetch institutional holders for a ticker.

        Args:
            ticker: Stock ticker symbol.

        Returns:
            List of holder dicts with holder name, shares, pct_out, value.
        """
        await self._rate_limit()
        try:
            return await asyncio.to_thread(self._sync_get_institutional_holders, ticker)
        except Exception as e:
            logger.warning("Failed to fetch institutional holders", ticker=ticker, error=str(e))
            return []

    def _sync_get_info(self, ticker: str) -> dict[str, Any]:
        """Synchronous full info fetch."""
        yf = self._ensure_yfinance()
        stock = yf.Ticker(ticker)
        return stock.info or {}

    async def get_info(self, ticker: str) -> dict[str, Any]:
        """Fetch full info dict for a ticker.

        Returns the complete Yahoo Finance info dictionary.
        """
        await self._rate_limit()
        try:
            return await asyncio.to_thread(self._sync_get_info, ticker)
        except Exception as e:
            raise ExternalAPIError(
                message=f"yfinance info error for {ticker}: {e}",
                service="yfinance",
            ) from e
