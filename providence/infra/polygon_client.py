"""Polygon.io REST API client for market data ingestion.

Provides async access to OHLCV price data with retry logic,
rate limiting, and error handling.

Spec Reference: Technical Spec v2.3, Section 4.1 (PERCEPT-PRICE)

Usage:
    client = PolygonClient(api_key="your_key")
    bars = await client.get_daily_bars("AAPL", "2026-02-09")
"""

import asyncio
import os
from typing import Any

import httpx

from providence.exceptions import DataIngestionError, ExternalAPIError


class PolygonClient:
    """Async HTTP client for the Polygon.io REST API.

    Handles authentication, rate limiting, retries, and error mapping.
    """

    BASE_URL = "https://api.polygon.io"
    DEFAULT_TIMEOUT = 30.0
    MAX_RETRIES = 3
    RETRY_BACKOFF_BASE = 1.0  # seconds

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        """Initialize the Polygon client.

        Args:
            api_key: Polygon.io API key. Falls back to POLYGON_API_KEY env var.
            base_url: Override base URL (useful for testing).
            timeout: Request timeout in seconds.
        """
        self._api_key = api_key or os.environ.get("POLYGON_API_KEY", "")
        if not self._api_key:
            raise ValueError(
                "Polygon API key required. Pass api_key or set POLYGON_API_KEY env var."
            )
        self._base_url = base_url or self.BASE_URL
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=self._timeout,
                params={"apiKey": self._api_key},
            )
        return self._client

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def _request(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Make a GET request with retry logic.

        Args:
            path: API endpoint path (e.g., /v2/aggs/ticker/AAPL/range/1/day/...).
            params: Additional query parameters.

        Returns:
            Parsed JSON response as a dictionary.

        Raises:
            ExternalAPIError: On HTTP errors or unexpected responses.
            DataIngestionError: On response parsing failures.
        """
        client = await self._get_client()
        last_error: Exception | None = None

        for attempt in range(self.MAX_RETRIES):
            try:
                response = await client.get(path, params=params or {})

                if response.status_code == 429:
                    # Rate limited — back off and retry
                    wait = self.RETRY_BACKOFF_BASE * (2 ** attempt)
                    await asyncio.sleep(wait)
                    continue

                if response.status_code != 200:
                    raise ExternalAPIError(
                        message=f"Polygon API returned {response.status_code}: {response.text[:200]}",
                        service="polygon",
                        status_code=response.status_code,
                    )

                data = response.json()
                if not isinstance(data, dict):
                    raise DataIngestionError(
                        message=f"Expected dict response, got {type(data).__name__}"
                    )
                return data

            except httpx.TimeoutException as e:
                last_error = ExternalAPIError(
                    message=f"Polygon API timeout on attempt {attempt + 1}: {e}",
                    service="polygon",
                )
                if attempt < self.MAX_RETRIES - 1:
                    await asyncio.sleep(self.RETRY_BACKOFF_BASE * (2 ** attempt))
                    continue

            except httpx.HTTPError as e:
                last_error = ExternalAPIError(
                    message=f"Polygon API HTTP error: {e}",
                    service="polygon",
                )
                if attempt < self.MAX_RETRIES - 1:
                    await asyncio.sleep(self.RETRY_BACKOFF_BASE * (2 ** attempt))
                    continue

            except (ExternalAPIError, DataIngestionError):
                raise

        raise last_error or ExternalAPIError(
            message="Polygon API request failed after all retries",
            service="polygon",
        )

    async def get_daily_bars(self, ticker: str, date: str) -> dict[str, Any]:
        """Fetch daily OHLCV bars for a ticker on a given date.

        Args:
            ticker: Stock ticker symbol (e.g., "AAPL").
            date: Date string in YYYY-MM-DD format.

        Returns:
            Raw API response dict with 'results' containing bar data.
        """
        path = f"/v2/aggs/ticker/{ticker}/range/1/day/{date}/{date}"
        return await self._request(path)

    async def get_intraday_bars(
        self,
        ticker: str,
        date: str,
        timeframe: str = "1",
        timespan: str = "hour",
    ) -> list[dict[str, Any]]:
        """Fetch intraday OHLCV bars for a ticker on a given date.

        Args:
            ticker: Stock ticker symbol (e.g., "AAPL").
            date: Date string in YYYY-MM-DD format.
            timeframe: Multiplier (e.g., "1", "5", "15").
            timespan: Time span unit ("minute", "hour").

        Returns:
            List of bar dicts from the 'results' field.
        """
        path = f"/v2/aggs/ticker/{ticker}/range/{timeframe}/{timespan}/{date}/{date}"
        data = await self._request(path)
        results = data.get("results", [])
        if not isinstance(results, list):
            raise DataIngestionError(
                message=f"Expected list of results, got {type(results).__name__}"
            )
        return results
