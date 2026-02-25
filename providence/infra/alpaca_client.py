"""Alpaca Markets REST API client for order placement and portfolio management.

Provides async access to order execution, position management, and account
data with retry logic, rate limiting, and error handling. Supports both
paper and live trading via environment configuration.

Spec Reference: Technical Spec v2.3, Section 5.2 (EXEC-SVC)

Usage:
    client = AlpacaClient()  # Paper trading by default
    account = await client.get_account()
    order = await client.submit_order("AAPL", qty=10, side="buy")
"""

import asyncio
import os
import time
from typing import Any

import httpx
import structlog

from providence.exceptions import DataIngestionError, ExternalAPIError

logger = structlog.get_logger()


class AlpacaClient:
    """Async HTTP client for the Alpaca Markets REST API.

    Handles authentication, rate limiting, retries, and error mapping.
    Supports both paper and live trading via environment configuration.
    Default: paper trading (ALPACA_PAPER=true).
    """

    PAPER_URL = "https://paper-api.alpaca.markets"
    LIVE_URL = "https://api.alpaca.markets"
    DATA_URL = "https://data.alpaca.markets"
    DEFAULT_TIMEOUT = 30.0
    MAX_RETRIES = 3
    RETRY_BACKOFF_BASE = 1.0  # seconds

    def __init__(
        self,
        api_key: str | None = None,
        secret_key: str | None = None,
        paper: bool | None = None,
        base_url: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        """Initialize the Alpaca client.

        Args:
            api_key: Alpaca API key. Falls back to ALPACA_API_KEY env var.
            secret_key: Alpaca secret key. Falls back to ALPACA_SECRET_KEY env var.
            paper: Use paper trading (True) or live (False). Falls back to ALPACA_PAPER env var, defaults to True.
            base_url: Override base URL (useful for testing).
            timeout: Request timeout in seconds.
        """
        self._api_key = api_key or os.environ.get("ALPACA_API_KEY", "")
        self._secret_key = secret_key or os.environ.get("ALPACA_SECRET_KEY", "")

        if not self._api_key or not self._secret_key:
            raise ValueError(
                "Alpaca API credentials required. Pass api_key/secret_key or set ALPACA_API_KEY/ALPACA_SECRET_KEY env vars."
            )

        # Determine if paper or live trading
        if paper is None:
            paper = os.environ.get("ALPACA_PAPER", "true").lower() == "true"
        self._paper = paper

        # Set base URL
        if base_url:
            self._base_url = base_url
        else:
            self._base_url = self.PAPER_URL if paper else self.LIVE_URL

        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None
        self._last_request_time: float = 0.0

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None or self._client.is_closed:
            headers = {
                "APCA-API-KEY-ID": self._api_key,
                "APCA-API-SECRET-KEY": self._secret_key,
            }
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=self._timeout,
                headers=headers,
            )
        return self._client

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def _request(
        self,
        method: str,
        path: str,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Make an HTTP request with retry logic and error handling.

        Args:
            method: HTTP method (GET, POST, DELETE, etc.).
            path: API endpoint path (e.g., /v2/orders).
            json: JSON body for POST/PUT requests.
            params: Query parameters.

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
                logger.debug(
                    "alpaca_request",
                    method=method,
                    path=path,
                    attempt=attempt + 1,
                    paper=self._paper,
                )

                response = await client.request(
                    method,
                    path,
                    json=json,
                    params=params or {},
                )

                # Handle rate limiting
                if response.status_code == 429:
                    wait = self.RETRY_BACKOFF_BASE * (2 ** attempt)
                    logger.debug(
                        "alpaca_rate_limited",
                        path=path,
                        wait_seconds=wait,
                        attempt=attempt + 1,
                    )
                    await asyncio.sleep(wait)
                    continue

                # Handle validation errors (expected in business logic)
                if response.status_code == 422:
                    error_text = response.text[:500]
                    logger.debug(
                        "alpaca_validation_error",
                        path=path,
                        status=422,
                        error=error_text,
                    )
                    raise ExternalAPIError(
                        message=f"Alpaca validation error: {error_text}",
                        service="alpaca",
                        status_code=422,
                    )

                # Handle insufficient funds
                if response.status_code == 403:
                    error_text = response.text[:500]
                    logger.debug(
                        "alpaca_insufficient_funds",
                        path=path,
                        status=403,
                        error=error_text,
                    )
                    raise ExternalAPIError(
                        message=f"Alpaca insufficient funds: {error_text}",
                        service="alpaca",
                        status_code=403,
                    )

                # Handle unexpected status codes
                if response.status_code not in (200, 201, 204):
                    raise ExternalAPIError(
                        message=f"Alpaca API returned {response.status_code}: {response.text[:200]}",
                        service="alpaca",
                        status_code=response.status_code,
                    )

                # Handle empty 204 responses
                if response.status_code == 204:
                    logger.debug("alpaca_success", path=path, status=204)
                    return {}

                data = response.json()
                if not isinstance(data, dict):
                    raise DataIngestionError(
                        message=f"Expected dict response, got {type(data).__name__}"
                    )

                logger.debug("alpaca_success", path=path, status=response.status_code)
                return data

            except httpx.TimeoutException as e:
                last_error = ExternalAPIError(
                    message=f"Alpaca API timeout on attempt {attempt + 1}: {e}",
                    service="alpaca",
                )
                logger.debug(
                    "alpaca_timeout",
                    path=path,
                    attempt=attempt + 1,
                    error=str(e),
                )
                if attempt < self.MAX_RETRIES - 1:
                    await asyncio.sleep(self.RETRY_BACKOFF_BASE * (2 ** attempt))
                    continue

            except httpx.HTTPError as e:
                last_error = ExternalAPIError(
                    message=f"Alpaca API HTTP error: {e}",
                    service="alpaca",
                )
                logger.debug(
                    "alpaca_http_error",
                    path=path,
                    attempt=attempt + 1,
                    error=str(e),
                )
                if attempt < self.MAX_RETRIES - 1:
                    await asyncio.sleep(self.RETRY_BACKOFF_BASE * (2 ** attempt))
                    continue

            except (ExternalAPIError, DataIngestionError):
                raise

        raise last_error or ExternalAPIError(
            message="Alpaca API request failed after all retries",
            service="alpaca",
        )

    # Account
    async def get_account(self) -> dict[str, Any]:
        """Fetch account details.

        Returns:
            Account dict with portfolio_value, cash, buying_power, equity, etc.
        """
        return await self._request("GET", "/v2/account")

    # Orders
    async def submit_order(
        self,
        ticker: str,
        qty: int | None = None,
        notional: float | None = None,
        side: str = "buy",
        type: str = "market",
        time_in_force: str = "day",
        limit_price: float | None = None,
        stop_price: float | None = None,
        client_order_id: str | None = None,
    ) -> dict[str, Any]:
        """Submit an order.

        Args:
            ticker: Stock ticker symbol (e.g., "AAPL").
            qty: Quantity of shares. Either qty or notional must be provided.
            notional: Dollar amount to trade. Either qty or notional must be provided.
            side: "buy" or "sell".
            type: "market", "limit", "stop", or "stop_limit".
            time_in_force: "day", "gtc", "opg", "cls", "ioc", or "fok".
            limit_price: Required for limit and stop_limit orders.
            stop_price: Required for stop and stop_limit orders.
            client_order_id: Optional idempotency key.

        Returns:
            Order dict with order_id, status, etc.
        """
        body: dict[str, Any] = {
            "symbol": ticker,
            "side": side,
            "type": type,
            "time_in_force": time_in_force,
        }

        if qty is not None:
            body["qty"] = qty
        if notional is not None:
            body["notional"] = notional
        if limit_price is not None:
            body["limit_price"] = limit_price
        if stop_price is not None:
            body["stop_price"] = stop_price
        if client_order_id is not None:
            body["client_order_id"] = client_order_id

        return await self._request("POST", "/v2/orders", json=body)

    async def get_order(self, order_id: str) -> dict[str, Any]:
        """Fetch a specific order by order_id.

        Args:
            order_id: The order ID.

        Returns:
            Order dict.
        """
        return await self._request("GET", f"/v2/orders/{order_id}")

    async def get_order_by_client_id(self, client_order_id: str) -> dict[str, Any]:
        """Fetch a specific order by client_order_id.

        Args:
            client_order_id: The client-provided order ID.

        Returns:
            Order dict.
        """
        return await self._request(
            "GET",
            "/v2/orders:by_client_order_id",
            params={"client_order_id": client_order_id},
        )

    async def list_orders(
        self, status: str = "open", limit: int = 100
    ) -> list[dict[str, Any]]:
        """List orders with optional filtering.

        Args:
            status: Filter by status (e.g., "open", "closed", "all").
            limit: Maximum number of orders to return.

        Returns:
            List of order dicts.
        """
        params = {"status": status, "limit": limit}
        response = await self._request("GET", "/v2/orders", params=params)

        # Response may be a list or dict; ensure we return a list
        if isinstance(response, list):
            return response
        if isinstance(response, dict):
            return response.get("orders", [])
        return []

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order.

        Args:
            order_id: The order ID to cancel.

        Returns:
            True if successful.
        """
        await self._request("DELETE", f"/v2/orders/{order_id}")
        return True

    async def cancel_all_orders(self) -> bool:
        """Cancel all open orders.

        Returns:
            True if successful.
        """
        await self._request("DELETE", "/v2/orders")
        return True

    # Positions
    async def list_positions(self) -> list[dict[str, Any]]:
        """List all open positions.

        Returns:
            List of position dicts.
        """
        response = await self._request("GET", "/v2/positions")

        # Response may be a list or dict
        if isinstance(response, list):
            return response
        if isinstance(response, dict):
            return response.get("positions", [])
        return []

    async def get_position(self, ticker: str) -> dict[str, Any] | None:
        """Fetch a specific position by ticker.

        Args:
            ticker: Stock ticker symbol.

        Returns:
            Position dict or None if not found.
        """
        try:
            return await self._request("GET", f"/v2/positions/{ticker}")
        except ExternalAPIError as e:
            if e.status_code == 404:
                return None
            raise

    async def close_position(
        self, ticker: str, qty: int | None = None, percentage: float | None = None
    ) -> dict[str, Any]:
        """Close a position (fully or partially).

        Args:
            ticker: Stock ticker symbol.
            qty: Quantity to close. If not provided, closes entire position.
            percentage: Percentage of position to close (0-100). Mutually exclusive with qty.

        Returns:
            Order dict for the closing order.
        """
        params: dict[str, Any] = {}
        if qty is not None:
            params["qty"] = qty
        if percentage is not None:
            params["percentage"] = percentage

        return await self._request(
            "DELETE", f"/v2/positions/{ticker}", params=params if params else None
        )

    async def close_all_positions(self) -> list[dict[str, Any]]:
        """Close all open positions.

        Returns:
            List of order dicts for the closing orders.
        """
        response = await self._request("DELETE", "/v2/positions")

        # Response may be a list or dict
        if isinstance(response, list):
            return response
        if isinstance(response, dict):
            return response.get("orders", [])
        return []

    # Portfolio History
    async def get_portfolio_history(
        self, period: str = "1M", timeframe: str = "1D"
    ) -> dict[str, Any]:
        """Fetch portfolio history (equity curve).

        Args:
            period: Time period (e.g., "1M", "3M", "1A").
            timeframe: Aggregation timeframe (e.g., "1M", "1D", "1H").

        Returns:
            Portfolio history dict with timestamps, equity values, etc.
        """
        params = {"period": period, "timeframe": timeframe}
        return await self._request("GET", "/v2/account/portfolio/history", params=params)

    # Clock & Calendar
    async def get_clock(self) -> dict[str, Any]:
        """Get market clock status.

        Returns:
            Clock dict with is_open, next_open, next_close, etc.
        """
        return await self._request("GET", "/v2/clock")
