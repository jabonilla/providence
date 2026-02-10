"""SEC EDGAR API client for filing data ingestion.

Provides async access to SEC EDGAR full-text search and XBRL API
with rate limiting (max 10 req/sec per SEC policy) and required
User-Agent header.

Spec Reference: Technical Spec v2.3, Section 4.1 (PERCEPT-FILING)
"""

import asyncio
import os
import time
from typing import Any

import httpx

from providence.exceptions import DataIngestionError, ExternalAPIError


class EdgarClient:
    """Async HTTP client for SEC EDGAR APIs.

    Respects SEC rate limiting (max 10 requests per second) and
    includes the required User-Agent header per SEC policy.
    """

    EFTS_BASE_URL = "https://efts.sec.gov/LATEST"
    XBRL_BASE_URL = "https://data.sec.gov"
    DEFAULT_TIMEOUT = 30.0
    MAX_RETRIES = 3
    RETRY_BACKOFF_BASE = 1.0
    MIN_REQUEST_INTERVAL = 0.1  # 10 req/sec max

    def __init__(
        self,
        user_agent: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        """Initialize the EDGAR client.

        Args:
            user_agent: Required User-Agent string per SEC policy.
                        Format: "Company Name email@example.com"
                        Falls back to EDGAR_USER_AGENT env var.
            timeout: Request timeout in seconds.
        """
        self._user_agent = user_agent or os.environ.get("EDGAR_USER_AGENT", "")
        if not self._user_agent:
            raise ValueError(
                "User-Agent required per SEC policy. Pass user_agent or set "
                "EDGAR_USER_AGENT env var. Format: 'Company Name email@example.com'"
            )
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None
        self._last_request_time: float = 0.0

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=self._timeout,
                headers={
                    "User-Agent": self._user_agent,
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
        """Enforce SEC rate limit of 10 requests per second."""
        now = time.monotonic()
        elapsed = now - self._last_request_time
        if elapsed < self.MIN_REQUEST_INTERVAL:
            await asyncio.sleep(self.MIN_REQUEST_INTERVAL - elapsed)
        self._last_request_time = time.monotonic()

    async def _request(self, url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Make a GET request with rate limiting and retry logic.

        Args:
            url: Full URL to request.
            params: Query parameters.

        Returns:
            Parsed JSON response.
        """
        client = await self._get_client()
        last_error: Exception | None = None

        for attempt in range(self.MAX_RETRIES):
            try:
                await self._rate_limit()
                response = await client.get(url, params=params or {})

                if response.status_code == 429:
                    wait = self.RETRY_BACKOFF_BASE * (2 ** attempt)
                    await asyncio.sleep(wait)
                    continue

                if response.status_code != 200:
                    raise ExternalAPIError(
                        message=f"EDGAR API returned {response.status_code}: {response.text[:200]}",
                        service="edgar",
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
                    message=f"EDGAR API timeout on attempt {attempt + 1}: {e}",
                    service="edgar",
                )
                if attempt < self.MAX_RETRIES - 1:
                    await asyncio.sleep(self.RETRY_BACKOFF_BASE * (2 ** attempt))

            except httpx.HTTPError as e:
                last_error = ExternalAPIError(
                    message=f"EDGAR API HTTP error: {e}",
                    service="edgar",
                )
                if attempt < self.MAX_RETRIES - 1:
                    await asyncio.sleep(self.RETRY_BACKOFF_BASE * (2 ** attempt))

            except (ExternalAPIError, DataIngestionError):
                raise

        raise last_error or ExternalAPIError(
            message="EDGAR API request failed after all retries",
            service="edgar",
        )

    async def get_recent_filings(
        self, ticker: str, filing_type: str, count: int = 5
    ) -> list[dict[str, Any]]:
        """Fetch recent filings for a company.

        Args:
            ticker: Stock ticker symbol.
            filing_type: Filing type (e.g., "10-K", "10-Q", "8-K").
            count: Number of filings to retrieve.

        Returns:
            List of filing metadata dicts.
        """
        url = f"{self.EFTS_BASE_URL}/search-index"
        params = {
            "q": f'"{ticker}"',
            "dateRange": "custom",
            "forms": filing_type,
            "from": "0",
            "size": str(count),
        }
        data = await self._request(url, params)
        hits = data.get("hits", {}).get("hits", [])
        return [hit.get("_source", {}) for hit in hits]

    async def get_filing_detail(self, accession_number: str) -> dict[str, Any]:
        """Fetch detailed filing data using the XBRL API.

        Args:
            accession_number: SEC accession number (e.g., "0000320193-24-000123").

        Returns:
            Filing detail dict with financial data.
        """
        # Format accession number for URL (remove dashes)
        acc_clean = accession_number.replace("-", "")
        url = f"{self.XBRL_BASE_URL}/api/xbrl/companyfacts/{acc_clean}.json"
        return await self._request(url)

    async def get_company_facts(self, cik: str) -> dict[str, Any]:
        """Fetch XBRL company facts for a CIK.

        Args:
            cik: SEC Central Index Key (zero-padded to 10 digits).

        Returns:
            Company facts dict with all XBRL data.
        """
        cik_padded = cik.zfill(10)
        url = f"{self.XBRL_BASE_URL}/api/xbrl/companyfacts/CIK{cik_padded}.json"
        return await self._request(url)
