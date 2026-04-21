"""Alpha Vantage REST API client for fundamental data ingestion.

Provides async access to earnings, income statement, and company overview data
with retry logic, rate limiting, and error handling.

Spec Reference: Technical Spec v2.3, Section 4.2 (Perception Agents)

Usage:
    client = AlphaVantageClient(api_key="your_key")
    earnings = await client.get_earnings("AAPL")
"""

import asyncio
import json
import os
import time
from typing import Any

import httpx
import structlog

from providence.exceptions import DataIngestionError, ExternalAPIError

logger = structlog.get_logger()


class AlphaVantageClient:
    """Async HTTP client for the Alpha Vantage REST API.

    Handles authentication, rate limiting, retries, and error mapping.
    """

    BASE_URL = "https://www.alphavantage.co"
    DEFAULT_TIMEOUT = 30.0
    MAX_RETRIES = 3
    RETRY_BACKOFF_BASE = 1.0  # seconds
    MIN_REQUEST_INTERVAL = 15.0  # Free tier = ~25 req/day, so 1 req per 15 seconds (conservative)

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        """Initialize the Alpha Vantage client.

        Args:
            api_key: Alpha Vantage API key. Falls back to ALPHAVANTAGE_API_KEY env var.
            base_url: Override base URL (useful for testing).
            timeout: Request timeout in seconds.
        """
        self._api_key = api_key or os.environ.get("ALPHAVANTAGE_API_KEY", "")
        if not self._api_key:
            raise ValueError(
                "Alpha Vantage API key required. Pass api_key or set ALPHAVANTAGE_API_KEY env var."
            )
        self._base_url = base_url or self.BASE_URL
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None
        self._last_request_time: float = 0.0

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=self._timeout,
            )
        return self._client

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def _rate_limit(self) -> None:
        """Enforce Alpha Vantage rate limit (free tier: ~25 req/day)."""
        now = time.monotonic()
        elapsed = now - self._last_request_time
        if elapsed < self.MIN_REQUEST_INTERVAL:
            await asyncio.sleep(self.MIN_REQUEST_INTERVAL - elapsed)
        self._last_request_time = time.monotonic()

    async def _request(self, function: str, symbol: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Make a GET request with rate limiting and retry logic.

        Args:
            function: Alpha Vantage function name (e.g., EARNINGS, INCOME_STATEMENT, OVERVIEW).
            symbol: Stock ticker symbol.
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
                await self._rate_limit()

                query_params = {
                    "function": function,
                    "symbol": symbol,
                    "apikey": self._api_key,
                }
                if params:
                    query_params.update(params)

                response = await client.get("/query", params=query_params)

                if response.status_code == 429:
                    # Rate limited — back off and retry
                    wait = self.RETRY_BACKOFF_BASE * (2 ** attempt)
                    logger.warning(
                        "Alpha Vantage rate limited — backing off",
                        wait_seconds=wait,
                        attempt=attempt + 1,
                    )
                    await asyncio.sleep(wait)
                    continue

                if response.status_code != 200:
                    raise ExternalAPIError(
                        message=f"Alpha Vantage API returned {response.status_code}: {response.text[:200]}",
                        service="alphavantage",
                        status_code=response.status_code,
                    )

                data = response.json()
                if not isinstance(data, dict):
                    raise DataIngestionError(
                        message=f"Expected dict response, got {type(data).__name__}"
                    )

                # Check for API error messages
                if "Error Message" in data:
                    raise ExternalAPIError(
                        message=f"Alpha Vantage API error: {data['Error Message']}",
                        service="alphavantage",
                    )

                if "Note" in data:
                    raise ExternalAPIError(
                        message=f"Alpha Vantage API note: {data['Note']}",
                        service="alphavantage",
                    )

                return data

            except httpx.TimeoutException as e:
                last_error = ExternalAPIError(
                    message=f"Alpha Vantage API timeout on attempt {attempt + 1}: {e}",
                    service="alphavantage",
                )
                if attempt < self.MAX_RETRIES - 1:
                    await asyncio.sleep(self.RETRY_BACKOFF_BASE * (2 ** attempt))
                    continue

            except httpx.HTTPError as e:
                last_error = ExternalAPIError(
                    message=f"Alpha Vantage API HTTP error: {e}",
                    service="alphavantage",
                )
                if attempt < self.MAX_RETRIES - 1:
                    await asyncio.sleep(self.RETRY_BACKOFF_BASE * (2 ** attempt))
                    continue

            except (ExternalAPIError, DataIngestionError):
                raise

        raise last_error or ExternalAPIError(
            message="Alpha Vantage API request failed after all retries",
            service="alphavantage",
        )

    async def get_earnings(self, ticker: str) -> dict[str, Any]:
        """Fetch quarterly earnings data for a ticker.

        Args:
            ticker: Stock ticker symbol (e.g., "AAPL").

        Returns:
            Raw API response dict with 'quarterlyEarnings' array.

        Raises:
            ExternalAPIError: On API errors or HTTP failures.
            DataIngestionError: On response parsing failures.
        """
        try:
            logger.info(
                "Fetching earnings data",
                ticker=ticker,
                function="EARNINGS",
            )
            data = await self._request("EARNINGS", ticker)
            return data
        except (ExternalAPIError, DataIngestionError) as e:
            logger.error(
                "Failed to fetch earnings data",
                ticker=ticker,
                error=str(e),
            )
            raise

    async def get_income_statement(self, ticker: str) -> dict[str, Any]:
        """Fetch annual income statement data for a ticker.

        Args:
            ticker: Stock ticker symbol (e.g., "AAPL").

        Returns:
            Raw API response dict with 'annualReports' and 'quarterlyReports' arrays.

        Raises:
            ExternalAPIError: On API errors or HTTP failures.
            DataIngestionError: On response parsing failures.
        """
        try:
            logger.info(
                "Fetching income statement data",
                ticker=ticker,
                function="INCOME_STATEMENT",
            )
            data = await self._request("INCOME_STATEMENT", ticker)
            return data
        except (ExternalAPIError, DataIngestionError) as e:
            logger.error(
                "Failed to fetch income statement data",
                ticker=ticker,
                error=str(e),
            )
            raise

    async def get_overview(self, ticker: str) -> dict[str, Any]:
        """Fetch company overview data for a ticker.

        Args:
            ticker: Stock ticker symbol (e.g., "AAPL").

        Returns:
            Raw API response dict with company fundamentals (PE ratio, dividend, etc.).

        Raises:
            ExternalAPIError: On API errors or HTTP failures.
            DataIngestionError: On response parsing failures.
        """
        try:
            logger.info(
                "Fetching company overview data",
                ticker=ticker,
                function="OVERVIEW",
            )
            data = await self._request("OVERVIEW", ticker)
            return data
        except (ExternalAPIError, DataIngestionError) as e:
            logger.error(
                "Failed to fetch company overview data",
                ticker=ticker,
                error=str(e),
            )
            raise
