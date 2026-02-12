"""Federal Reserve Economic Data (FRED) API client for macroeconomic data ingestion.

Provides async access to FRED time series data with rate limiting (max 120 req/min),
retry logic, and error handling. Supports fetching economic indicators like Treasury
yields and macroeconomic series.

Spec Reference: Technical Spec v2.3, Section 4.1 (PERCEPT-MACRO)

Usage:
    client = FredClient(api_key="your_key")
    yields = await client.get_treasury_yields("2026-02-09")
    obs = await client.get_series_observations("DGS10", "2026-01-01", "2026-02-09")
"""

import asyncio
import os
import time
from typing import Any

import httpx

from providence.exceptions import DataIngestionError, ExternalAPIError


class FredClient:
    """Async HTTP client for the Federal Reserve Economic Data (FRED) API.

    Handles authentication, rate limiting, retries, and error mapping.
    """

    BASE_URL = "https://api.stlouisfed.org"
    DEFAULT_TIMEOUT = 30.0
    MAX_RETRIES = 3
    RETRY_BACKOFF_BASE = 1.0  # seconds
    MIN_REQUEST_INTERVAL = 0.5  # 120 req/min = 1 req per 0.5 seconds

    def __init__(
        self,
        api_key: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        """Initialize the FRED client.

        Args:
            api_key: FRED API key (free from https://research.stlouisfed.org/docs/api/).
                     Falls back to FRED_API_KEY env var.
            timeout: Request timeout in seconds.
        """
        self._api_key = api_key or os.environ.get("FRED_API_KEY", "")
        if not self._api_key:
            raise ValueError(
                "FRED API key required. Pass api_key or set FRED_API_KEY env var. "
                "Get a free key at https://research.stlouisfed.org/docs/api/"
            )
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None
        self._last_request_time: float = 0.0

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.BASE_URL,
                timeout=self._timeout,
                headers={
                    "Accept": "application/json",
                },
            )
        return self._client

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def _rate_limit(self) -> None:
        """Enforce FRED rate limit of 120 requests per minute."""
        now = time.monotonic()
        elapsed = now - self._last_request_time
        if elapsed < self.MIN_REQUEST_INTERVAL:
            await asyncio.sleep(self.MIN_REQUEST_INTERVAL - elapsed)
        self._last_request_time = time.monotonic()

    async def _request(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Make a GET request with rate limiting and retry logic.

        Args:
            path: API endpoint path (e.g., /fred/series/observations).
            params: Additional query parameters (API key will be added automatically).

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
                # Add API key to params for this request
                request_params = params or {}
                request_params["api_key"] = self._api_key

                response = await client.get(path, params=request_params)

                if response.status_code == 429:
                    # Rate limited — back off and retry
                    wait = self.RETRY_BACKOFF_BASE * (2 ** attempt)
                    await asyncio.sleep(wait)
                    continue

                if response.status_code != 200:
                    raise ExternalAPIError(
                        message=f"FRED API returned {response.status_code}: {response.text[:200]}",
                        service="fred",
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
                    message=f"FRED API timeout on attempt {attempt + 1}: {e}",
                    service="fred",
                )
                if attempt < self.MAX_RETRIES - 1:
                    await asyncio.sleep(self.RETRY_BACKOFF_BASE * (2 ** attempt))
                    continue

            except httpx.HTTPError as e:
                last_error = ExternalAPIError(
                    message=f"FRED API HTTP error: {e}",
                    service="fred",
                )
                if attempt < self.MAX_RETRIES - 1:
                    await asyncio.sleep(self.RETRY_BACKOFF_BASE * (2 ** attempt))
                    continue

            except (ExternalAPIError, DataIngestionError):
                raise

        raise last_error or ExternalAPIError(
            message="FRED API request failed after all retries",
            service="fred",
        )

    async def get_series_observations(
        self, series_id: str, start_date: str, end_date: str, limit: int = 10
    ) -> list[dict[str, Any]]:
        """Fetch observations for a FRED economic series.

        Args:
            series_id: FRED series ID (e.g., "DGS10" for 10-year Treasury yield).
            start_date: Start date in YYYY-MM-DD format.
            end_date: End date in YYYY-MM-DD format.
            limit: Maximum number of observations to return.

        Returns:
            List of observation dicts with keys: date, value, realtime_start, realtime_end.
        """
        path = "/fred/series/observations"
        params = {
            "series_id": series_id,
            "observation_start": start_date,
            "observation_end": end_date,
            "sort_order": "desc",
            "limit": str(limit),
            "file_type": "json",
        }
        data = await self._request(path, params)
        observations = data.get("observations", [])
        if not isinstance(observations, list):
            raise DataIngestionError(
                message=f"Expected list of observations, got {type(observations).__name__}"
            )
        return observations

    async def get_latest_observation(self, series_id: str) -> dict[str, Any]:
        """Fetch the latest observation for a FRED economic series.

        Args:
            series_id: FRED series ID (e.g., "DGS10" for 10-year Treasury yield).

        Returns:
            Single observation dict with keys: date, value, realtime_start, realtime_end.

        Raises:
            DataIngestionError: If no observations are found.
        """
        observations = await self.get_series_observations(series_id, "1980-01-01", "2099-12-31", limit=1)
        if not observations:
            raise DataIngestionError(
                message=f"No observations found for series {series_id}"
            )
        return observations[0]

    async def get_treasury_yields(self, date: str) -> dict[str, float]:
        """Fetch the Treasury yield curve for a given date.

        Retrieves observations from FRED series covering the full yield curve:
        1M, 3M, 6M, 1Y, 2Y, 3Y, 5Y, 7Y, 10Y, 20Y, and 30Y tenors.

        Args:
            date: Date in YYYY-MM-DD format (typically the observation date you want).

        Returns:
            Dict mapping tenor labels (e.g., "10Y") to yields as floats.
            Missing values (marked as "." in FRED) are skipped.

        Example:
            {"1M": 5.32, "3M": 5.45, "10Y": 3.95, ...}
        """
        # Mapping of FRED series IDs to tenor labels
        series_map = {
            "DGS1MO": "1M",
            "DGS3MO": "3M",
            "DGS6MO": "6M",
            "DGS1": "1Y",
            "DGS2": "2Y",
            "DGS3": "3Y",
            "DGS5": "5Y",
            "DGS7": "7Y",
            "DGS10": "10Y",
            "DGS20": "20Y",
            "DGS30": "30Y",
        }

        yields = {}

        # Fetch observations for each series
        for series_id, tenor in series_map.items():
            try:
                observations = await self.get_series_observations(
                    series_id, date, date, limit=1
                )
                if observations:
                    obs = observations[0]
                    value = obs.get("value")
                    # Skip missing values (FRED uses "." for missing data)
                    if value and value != ".":
                        try:
                            yields[tenor] = float(value)
                        except (ValueError, TypeError):
                            # Skip if value can't be converted to float
                            pass
            except ExternalAPIError:
                # Skip series if we get an API error (some tenors may not be available)
                pass

        return yields
