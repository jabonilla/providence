"""Plaid API client for fund flow and investment transaction data.

Provides async access to investment transactions, holdings, and account data
with retry logic, rate limiting, and error handling.

Spec Reference: Technical Spec v2.3, Section 4.1 (Perception Agents)

Usage:
    client = PlaidClient(client_id="id", secret="secret")
    txns = await client.get_investment_transactions(access_token, "2026-01-01", "2026-03-01")
"""

import asyncio
import os
import time
from typing import Any

import httpx
import structlog

from providence.exceptions import DataIngestionError, ExternalAPIError

logger = structlog.get_logger()


_ENVIRONMENT_URLS = {
    "sandbox": "https://sandbox.plaid.com",
    "development": "https://development.plaid.com",
    "production": "https://production.plaid.com",
}


class PlaidClient:
    """Async HTTP client for the Plaid API.

    Handles authentication, rate limiting, retries, and error mapping.
    """

    DEFAULT_TIMEOUT = 30.0
    MAX_RETRIES = 3
    RETRY_BACKOFF_BASE = 1.0  # seconds
    MIN_REQUEST_INTERVAL = 0.2  # 10 req/s is generous

    def __init__(
        self,
        client_id: str | None = None,
        secret: str | None = None,
        environment: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        """Initialize the Plaid client.

        Args:
            client_id: Plaid client ID. Falls back to PLAID_CLIENT_ID env var.
            secret: Plaid secret. Falls back to PLAID_SECRET env var.
            environment: Plaid environment (sandbox/development/production).
                Falls back to PLAID_ENV env var, defaults to "sandbox".
            timeout: Request timeout in seconds.
        """
        self._client_id = client_id or os.environ.get("PLAID_CLIENT_ID", "")
        self._secret = secret or os.environ.get("PLAID_SECRET", "")
        if not self._client_id or not self._secret:
            raise ValueError(
                "Plaid credentials required. Pass client_id/secret or set "
                "PLAID_CLIENT_ID and PLAID_SECRET env vars."
            )
        env = environment or os.environ.get("PLAID_ENV", "sandbox")
        self._base_url = _ENVIRONMENT_URLS.get(env, _ENVIRONMENT_URLS["sandbox"])
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None
        self._last_request_time: float = 0.0

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=self._timeout,
                headers={"Content-Type": "application/json"},
            )
        return self._client

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def _rate_limit(self) -> None:
        """Enforce Plaid rate limit."""
        now = time.monotonic()
        elapsed = now - self._last_request_time
        if elapsed < self.MIN_REQUEST_INTERVAL:
            await asyncio.sleep(self.MIN_REQUEST_INTERVAL - elapsed)
        self._last_request_time = time.monotonic()

    async def _request(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        """Make a POST request with rate limiting and retry logic.

        All Plaid API calls are POST with JSON body containing client_id and secret.

        Args:
            path: API endpoint path (e.g., /investments/transactions/get).
            body: Request body (client_id and secret are injected automatically).

        Returns:
            Parsed JSON response as a dictionary.

        Raises:
            ExternalAPIError: On HTTP errors or Plaid API errors.
            DataIngestionError: On response parsing failures.
        """
        client = await self._get_client()
        last_error: Exception | None = None

        full_body = {
            "client_id": self._client_id,
            "secret": self._secret,
            **body,
        }

        for attempt in range(self.MAX_RETRIES):
            try:
                await self._rate_limit()
                response = await client.post(path, json=full_body)

                if response.status_code == 429:
                    wait = self.RETRY_BACKOFF_BASE * (2 ** attempt)
                    logger.warning(
                        "Plaid rate limited — backing off",
                        wait_seconds=wait,
                        attempt=attempt + 1,
                    )
                    await asyncio.sleep(wait)
                    continue

                data = response.json()
                if not isinstance(data, dict):
                    raise DataIngestionError(
                        message=f"Expected dict response, got {type(data).__name__}"
                    )

                # Check for Plaid error response
                if "error_type" in data and data.get("error_type"):
                    raise ExternalAPIError(
                        message=(
                            f"Plaid API error: [{data.get('error_type')}] "
                            f"{data.get('error_code', 'UNKNOWN')}: "
                            f"{data.get('error_message', 'No message')}"
                        ),
                        service="plaid",
                        status_code=response.status_code,
                    )

                if response.status_code != 200:
                    raise ExternalAPIError(
                        message=f"Plaid API returned {response.status_code}: {response.text[:200]}",
                        service="plaid",
                        status_code=response.status_code,
                    )

                return data

            except httpx.TimeoutException as e:
                last_error = ExternalAPIError(
                    message=f"Plaid API timeout on attempt {attempt + 1}: {e}",
                    service="plaid",
                )
                if attempt < self.MAX_RETRIES - 1:
                    await asyncio.sleep(self.RETRY_BACKOFF_BASE * (2 ** attempt))
                    continue

            except httpx.HTTPError as e:
                last_error = ExternalAPIError(
                    message=f"Plaid API HTTP error: {e}",
                    service="plaid",
                )
                if attempt < self.MAX_RETRIES - 1:
                    await asyncio.sleep(self.RETRY_BACKOFF_BASE * (2 ** attempt))
                    continue

            except (ExternalAPIError, DataIngestionError):
                raise

        raise last_error or ExternalAPIError(
            message="Plaid API request failed after all retries",
            service="plaid",
        )

    async def get_investment_transactions(
        self,
        access_token: str,
        start_date: str,
        end_date: str,
        count: int = 500,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Fetch investment transactions for a linked account.

        Args:
            access_token: Plaid access token for the linked institution.
            start_date: Start date in YYYY-MM-DD format.
            end_date: End date in YYYY-MM-DD format.
            count: Maximum number of transactions to return (max 500).
            offset: Pagination offset.

        Returns:
            Dict with 'investment_transactions', 'accounts', 'securities', 'total_investment_transactions'.
        """
        logger.info(
            "Fetching investment transactions",
            start_date=start_date,
            end_date=end_date,
        )
        return await self._request(
            "/investments/transactions/get",
            {
                "access_token": access_token,
                "start_date": start_date,
                "end_date": end_date,
                "options": {"count": count, "offset": offset},
            },
        )

    async def get_accounts(self, access_token: str) -> dict[str, Any]:
        """Fetch linked accounts.

        Args:
            access_token: Plaid access token.

        Returns:
            Dict with 'accounts' and 'item' information.
        """
        logger.info("Fetching accounts")
        return await self._request(
            "/accounts/get",
            {"access_token": access_token},
        )

    async def get_holdings(self, access_token: str) -> dict[str, Any]:
        """Fetch investment holdings for a linked account.

        Args:
            access_token: Plaid access token.

        Returns:
            Dict with 'holdings', 'accounts', 'securities'.
        """
        logger.info("Fetching investment holdings")
        return await self._request(
            "/investments/holdings/get",
            {"access_token": access_token},
        )
