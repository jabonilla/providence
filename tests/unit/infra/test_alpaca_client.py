"""Tests for AlpacaClient at providence/infra/alpaca_client.py.

All tests mock httpx — NO real HTTP calls.
"""
import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

import httpx
import pytest

from providence.infra.alpaca_client import AlpacaClient
from providence.exceptions import ExternalAPIError, DataIngestionError


class TestAlpacaClientInit:
    """Test AlpacaClient initialization."""

    def test_init_default_paper_mode(self):
        """Paper mode should be default."""
        with patch.dict(os.environ, {"ALPACA_API_KEY": "test_key", "ALPACA_SECRET_KEY": "test_secret"}):
            client = AlpacaClient()
            assert client._paper is True
            assert client._base_url == AlpacaClient.PAPER_URL

    def test_init_live_mode(self):
        """Can explicitly set live mode."""
        with patch.dict(os.environ, {"ALPACA_API_KEY": "test_key", "ALPACA_SECRET_KEY": "test_secret"}):
            client = AlpacaClient(paper=False)
            assert client._paper is False
            assert client._base_url == AlpacaClient.LIVE_URL

    def test_init_env_var_paper_mode(self):
        """ALPACA_PAPER env var sets mode."""
        with patch.dict(
            os.environ,
            {
                "ALPACA_API_KEY": "test_key",
                "ALPACA_SECRET_KEY": "test_secret",
                "ALPACA_PAPER": "false",
            },
        ):
            client = AlpacaClient()
            assert client._paper is False

    def test_init_missing_api_key_raises(self):
        """Missing API key raises ValueError."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Alpaca API credentials required"):
                AlpacaClient()

    def test_init_missing_secret_key_raises(self):
        """Missing secret key raises ValueError."""
        with patch.dict(os.environ, {"ALPACA_API_KEY": "test_key"}, clear=True):
            with pytest.raises(ValueError, match="Alpaca API credentials required"):
                AlpacaClient()

    def test_init_custom_base_url(self):
        """Can override base URL."""
        with patch.dict(os.environ, {"ALPACA_API_KEY": "test_key", "ALPACA_SECRET_KEY": "test_secret"}):
            client = AlpacaClient(base_url="https://custom.example.com")
            assert client._base_url == "https://custom.example.com"

    def test_init_custom_timeout(self):
        """Can set custom timeout."""
        with patch.dict(os.environ, {"ALPACA_API_KEY": "test_key", "ALPACA_SECRET_KEY": "test_secret"}):
            client = AlpacaClient(timeout=60.0)
            assert client._timeout == 60.0


class TestAlpacaClientOrders:
    """Test order-related methods."""

    @pytest.fixture
    def client(self):
        """Create a client for testing."""
        with patch.dict(os.environ, {"ALPACA_API_KEY": "test_key", "ALPACA_SECRET_KEY": "test_secret"}):
            return AlpacaClient()

    @pytest.mark.asyncio
    async def test_submit_order_success(self, client):
        """submit_order succeeds with broker response."""
        with patch.object(client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {
                "id": "order123",
                "symbol": "AAPL",
                "qty": 10,
                "side": "buy",
                "status": "new",
            }

            result = await client.submit_order("AAPL", qty=10, side="buy")

            assert result["id"] == "order123"
            assert result["symbol"] == "AAPL"
            mock_request.assert_called_once_with("POST", "/v2/orders", json={
                "symbol": "AAPL",
                "qty": 10,
                "side": "buy",
                "type": "market",
                "time_in_force": "day",
            })

    @pytest.mark.asyncio
    async def test_submit_order_with_limit_price(self, client):
        """submit_order includes limit_price when provided."""
        with patch.object(client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {"id": "order123"}

            await client.submit_order(
                "AAPL",
                qty=10,
                side="buy",
                type="limit",
                limit_price=150.00
            )

            call_args = mock_request.call_args
            assert call_args[1]["json"]["limit_price"] == 150.00

    @pytest.mark.asyncio
    async def test_get_order(self, client):
        """get_order fetches a specific order."""
        with patch.object(client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {"id": "order123", "status": "filled"}

            result = await client.get_order("order123")

            assert result["id"] == "order123"
            mock_request.assert_called_once_with("GET", "/v2/orders/order123")

    @pytest.mark.asyncio
    async def test_list_orders(self, client):
        """list_orders returns list of orders."""
        with patch.object(client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = [
                {"id": "order1", "status": "filled"},
                {"id": "order2", "status": "new"},
            ]

            result = await client.list_orders(status="open")

            assert len(result) == 2
            assert result[0]["id"] == "order1"

    @pytest.mark.asyncio
    async def test_list_orders_dict_response(self, client):
        """list_orders handles dict response with orders key."""
        with patch.object(client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {
                "orders": [
                    {"id": "order1"},
                    {"id": "order2"},
                ]
            }

            result = await client.list_orders()

            assert len(result) == 2

    @pytest.mark.asyncio
    async def test_cancel_order(self, client):
        """cancel_order returns True on success."""
        with patch.object(client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {}

            result = await client.cancel_order("order123")

            assert result is True
            mock_request.assert_called_once_with("DELETE", "/v2/orders/order123")


class TestAlpacaClientPositions:
    """Test position-related methods."""

    @pytest.fixture
    def client(self):
        with patch.dict(os.environ, {"ALPACA_API_KEY": "test_key", "ALPACA_SECRET_KEY": "test_secret"}):
            return AlpacaClient()

    @pytest.mark.asyncio
    async def test_list_positions(self, client):
        """list_positions returns list of positions."""
        with patch.object(client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = [
                {"symbol": "AAPL", "qty": 100},
                {"symbol": "MSFT", "qty": 50},
            ]

            result = await client.list_positions()

            assert len(result) == 2
            assert result[0]["symbol"] == "AAPL"

    @pytest.mark.asyncio
    async def test_get_position(self, client):
        """get_position returns position by ticker."""
        with patch.object(client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {"symbol": "AAPL", "qty": 100}

            result = await client.get_position("AAPL")

            assert result["symbol"] == "AAPL"

    @pytest.mark.asyncio
    async def test_get_position_not_found_returns_none(self, client):
        """get_position returns None on 404."""
        with patch.object(client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.side_effect = ExternalAPIError(
                message="Not found",
                service="alpaca",
                status_code=404
            )

            result = await client.get_position("NONEXISTENT")

            assert result is None

    @pytest.mark.asyncio
    async def test_get_position_other_error_raises(self, client):
        """get_position raises on non-404 errors."""
        with patch.object(client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.side_effect = ExternalAPIError(
                message="Server error",
                service="alpaca",
                status_code=500
            )

            with pytest.raises(ExternalAPIError):
                await client.get_position("AAPL")

    @pytest.mark.asyncio
    async def test_close_position(self, client):
        """close_position closes a position."""
        with patch.object(client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {"id": "order456"}

            result = await client.close_position("AAPL")

            assert result["id"] == "order456"
            mock_request.assert_called_once_with("DELETE", "/v2/positions/AAPL", params=None)

    @pytest.mark.asyncio
    async def test_close_position_partial(self, client):
        """close_position with qty parameter."""
        with patch.object(client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {"id": "order456"}

            await client.close_position("AAPL", qty=50)

            call_args = mock_request.call_args
            assert call_args[1]["params"]["qty"] == 50


class TestAlpacaClientAccount:
    """Test account-related methods."""

    @pytest.fixture
    def client(self):
        with patch.dict(os.environ, {"ALPACA_API_KEY": "test_key", "ALPACA_SECRET_KEY": "test_secret"}):
            return AlpacaClient()

    @pytest.mark.asyncio
    async def test_get_account(self, client):
        """get_account returns account details."""
        with patch.object(client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {
                "equity": "100000.00",
                "cash": "50000.00",
                "buying_power": "100000.00",
                "portfolio_value": "105000.00",
            }

            result = await client.get_account()

            assert result["equity"] == "100000.00"
            assert result["buying_power"] == "100000.00"
            mock_request.assert_called_once_with("GET", "/v2/account")


class TestAlpacaClientRetry:
    """Test retry logic."""

    @pytest.fixture
    def client(self):
        with patch.dict(os.environ, {"ALPACA_API_KEY": "test_key", "ALPACA_SECRET_KEY": "test_secret"}):
            return AlpacaClient()

    @pytest.mark.asyncio
    async def test_retry_on_429(self, client):
        """_request retries on 429 rate limit."""
        with patch.object(client, "_get_client", new_callable=AsyncMock) as mock_get_client:
            mock_client = AsyncMock(spec=httpx.AsyncClient)
            mock_get_client.return_value = mock_client

            # First call returns 429, second returns 200
            response_429 = MagicMock()
            response_429.status_code = 429

            response_200 = MagicMock()
            response_200.status_code = 200
            response_200.json.return_value = {"result": "success"}

            mock_client.request.side_effect = [response_429, response_200]

            result = await client._request("GET", "/v2/test")

            assert result["result"] == "success"
            assert mock_client.request.call_count == 2

    @pytest.mark.asyncio
    async def test_retry_on_timeout(self, client):
        """_request retries on timeout."""
        with patch.object(client, "_get_client", new_callable=AsyncMock) as mock_get_client:
            mock_client = AsyncMock(spec=httpx.AsyncClient)
            mock_get_client.return_value = mock_client

            response_200 = MagicMock()
            response_200.status_code = 200
            response_200.json.return_value = {"result": "success"}

            mock_client.request.side_effect = [
                httpx.TimeoutException("timeout"),
                response_200,
            ]

            result = await client._request("GET", "/v2/test")

            assert result["result"] == "success"

    @pytest.mark.asyncio
    async def test_no_retry_on_422(self, client):
        """_request does NOT retry on 422 validation error."""
        with patch.object(client, "_get_client", new_callable=AsyncMock) as mock_get_client:
            mock_client = AsyncMock(spec=httpx.AsyncClient)
            mock_get_client.return_value = mock_client

            response_422 = MagicMock()
            response_422.status_code = 422
            response_422.text = "Invalid order parameters"

            mock_client.request.return_value = response_422

            with pytest.raises(ExternalAPIError):
                await client._request("POST", "/v2/orders", json={})

            # Should only be called once (no retry)
            assert mock_client.request.call_count == 1

    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self, client):
        """_request raises after MAX_RETRIES attempts."""
        with patch.object(client, "_get_client", new_callable=AsyncMock) as mock_get_client:
            mock_client = AsyncMock(spec=httpx.AsyncClient)
            mock_get_client.return_value = mock_client

            # Always timeout
            mock_client.request.side_effect = httpx.TimeoutException("timeout")

            with pytest.raises(ExternalAPIError):
                await client._request("GET", "/v2/test")

            # Should retry MAX_RETRIES times
            assert mock_client.request.call_count == AlpacaClient.MAX_RETRIES


class TestAlpacaClientClock:
    """Test clock methods."""

    @pytest.fixture
    def client(self):
        with patch.dict(os.environ, {"ALPACA_API_KEY": "test_key", "ALPACA_SECRET_KEY": "test_secret"}):
            return AlpacaClient()

    @pytest.mark.asyncio
    async def test_get_clock(self, client):
        """get_clock returns market status."""
        with patch.object(client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {
                "is_open": True,
                "next_open": "2025-01-20T09:30:00Z",
                "next_close": "2025-01-20T16:00:00Z",
            }

            result = await client.get_clock()

            assert result["is_open"] is True
            assert "next_open" in result
            mock_request.assert_called_once_with("GET", "/v2/clock")
