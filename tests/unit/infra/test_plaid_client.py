"""Tests for PlaidClient at providence/infra/plaid_client.py.

All tests mock httpx.AsyncClient — NO real API calls.
"""

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from providence.exceptions import DataIngestionError, ExternalAPIError
from providence.infra.plaid_client import PlaidClient


class TestPlaidClientInit:
    """Test PlaidClient initialization."""

    def test_init_with_args(self):
        """Client should initialize with explicit credentials."""
        client = PlaidClient(client_id="test_id", secret="test_secret")
        assert client._client_id == "test_id"
        assert client._secret == "test_secret"
        assert client._base_url == "https://sandbox.plaid.com"
        assert client._timeout == PlaidClient.DEFAULT_TIMEOUT

    def test_init_from_env_vars(self):
        """Client should read credentials from environment."""
        with patch.dict(os.environ, {
            "PLAID_CLIENT_ID": "env_id",
            "PLAID_SECRET": "env_secret",
            "PLAID_ENV": "development",
        }):
            client = PlaidClient()
            assert client._client_id == "env_id"
            assert client._secret == "env_secret"
            assert client._base_url == "https://development.plaid.com"

    def test_init_missing_client_id_raises_error(self):
        """Should raise ValueError when client_id is missing."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError) as exc_info:
                PlaidClient(secret="test_secret")
            assert "Plaid credentials required" in str(exc_info.value)

    def test_init_missing_secret_raises_error(self):
        """Should raise ValueError when secret is missing."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError) as exc_info:
                PlaidClient(client_id="test_id")
            assert "Plaid credentials required" in str(exc_info.value)

    def test_init_environment_url_mapping(self):
        """Should map environment parameter to correct URL."""
        test_cases = [
            ("sandbox", "https://sandbox.plaid.com"),
            ("development", "https://development.plaid.com"),
            ("production", "https://production.plaid.com"),
        ]

        for env_name, expected_url in test_cases:
            client = PlaidClient(
                client_id="test_id",
                secret="test_secret",
                environment=env_name,
            )
            assert client._base_url == expected_url

    def test_init_invalid_environment_defaults_to_sandbox(self):
        """Should default to sandbox for invalid environment."""
        client = PlaidClient(
            client_id="test_id",
            secret="test_secret",
            environment="invalid",
        )
        assert client._base_url == "https://sandbox.plaid.com"

    def test_init_custom_timeout(self):
        """Client should accept custom timeout."""
        client = PlaidClient(
            client_id="test_id",
            secret="test_secret",
            timeout=60.0,
        )
        assert client._timeout == 60.0


class TestPlaidClientGetClient:
    """Test HTTP client management."""

    @pytest.mark.asyncio
    async def test_get_client_creates_client(self):
        """Should create httpx.AsyncClient on first call."""
        client = PlaidClient(client_id="test_id", secret="test_secret")
        http_client = await client._get_client()

        assert isinstance(http_client, httpx.AsyncClient)
        assert http_client.base_url == "https://sandbox.plaid.com"
        assert http_client.timeout == PlaidClient.DEFAULT_TIMEOUT

        await client.close()

    @pytest.mark.asyncio
    async def test_get_client_reuses_client(self):
        """Should reuse the same client on subsequent calls."""
        client = PlaidClient(client_id="test_id", secret="test_secret")

        http_client1 = await client._get_client()
        http_client2 = await client._get_client()

        assert http_client1 is http_client2

        await client.close()

    @pytest.mark.asyncio
    async def test_close_closes_client(self):
        """Should close the HTTP client."""
        client = PlaidClient(client_id="test_id", secret="test_secret")
        http_client = await client._get_client()

        assert not http_client.is_closed

        await client.close()

        assert http_client.is_closed
        assert client._client is None

    @pytest.mark.asyncio
    async def test_close_on_already_closed_client(self):
        """Should handle closing an already-closed client."""
        client = PlaidClient(client_id="test_id", secret="test_secret")
        await client.close()
        await client.close()  # Should not raise


class TestPlaidClientRateLimit:
    """Test rate limiting behavior."""

    @pytest.mark.asyncio
    async def test_rate_limit_enforced(self):
        """Rate limit should enforce minimum interval between requests."""
        client = PlaidClient(client_id="test_id", secret="test_secret")
        client._last_request_time = asyncio.get_event_loop().time()

        import time

        before = time.time()
        await client._rate_limit()
        elapsed = time.time() - before

        assert elapsed >= client.MIN_REQUEST_INTERVAL * 0.9  # Allow 10% variance

    @pytest.mark.asyncio
    async def test_rate_limit_updates_time(self):
        """Rate limit should update the last request time."""
        client = PlaidClient(client_id="test_id", secret="test_secret")
        client._last_request_time = 0.0

        await client._rate_limit()
        assert client._last_request_time > 0.0


class TestPlaidClientRequest:
    """Test the underlying _request method."""

    @pytest.mark.asyncio
    async def test_request_success(self):
        """Should return parsed JSON response."""
        client = PlaidClient(client_id="test_id", secret="test_secret")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"accounts": [], "item": {}}

        with patch.object(client, "_get_client") as mock_get:
            mock_http_client = AsyncMock()
            mock_http_client.post.return_value = mock_response
            mock_get.return_value = mock_http_client

            result = await client._request("/accounts/get", {"access_token": "token"})

        assert result == {"accounts": [], "item": {}}
        assert mock_response.json.called

    @pytest.mark.asyncio
    async def test_request_includes_credentials(self):
        """Should inject client_id and secret into request body."""
        client = PlaidClient(client_id="test_id", secret="test_secret")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {}

        with patch.object(client, "_get_client") as mock_get:
            mock_http_client = AsyncMock()
            mock_http_client.post.return_value = mock_response
            mock_get.return_value = mock_http_client

            await client._request("/accounts/get", {"access_token": "token"})

            # Verify credentials were included in the request
            call_args = mock_http_client.post.call_args
            body = call_args.kwargs["json"]
            assert body["client_id"] == "test_id"
            assert body["secret"] == "test_secret"
            assert body["access_token"] == "token"

        await client.close()

    @pytest.mark.asyncio
    async def test_request_plaid_error_response(self):
        """Should raise ExternalAPIError on Plaid error_type."""
        client = PlaidClient(client_id="test_id", secret="test_secret")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "error_type": "INVALID_REQUEST",
            "error_code": "INVALID_BODY",
            "error_message": "Invalid request body",
        }

        with patch.object(client, "_get_client") as mock_get:
            mock_http_client = AsyncMock()
            mock_http_client.post.return_value = mock_response
            mock_get.return_value = mock_http_client

            with pytest.raises(ExternalAPIError) as exc_info:
                await client._request("/accounts/get", {"access_token": "token"})

            assert "INVALID_REQUEST" in str(exc_info.value)
            assert "INVALID_BODY" in str(exc_info.value)

        await client.close()

    @pytest.mark.asyncio
    async def test_request_http_error_response(self):
        """Should raise ExternalAPIError on non-200 status."""
        client = PlaidClient(client_id="test_id", secret="test_secret")

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal server error"

        with patch.object(client, "_get_client") as mock_get:
            mock_http_client = AsyncMock()
            mock_http_client.post.return_value = mock_response
            mock_get.return_value = mock_http_client

            with pytest.raises(ExternalAPIError) as exc_info:
                await client._request("/accounts/get", {"access_token": "token"})

            assert "500" in str(exc_info.value)

        await client.close()

    @pytest.mark.asyncio
    async def test_request_non_dict_response(self):
        """Should raise DataIngestionError on non-dict response."""
        client = PlaidClient(client_id="test_id", secret="test_secret")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = ["not", "a", "dict"]

        with patch.object(client, "_get_client") as mock_get:
            mock_http_client = AsyncMock()
            mock_http_client.post.return_value = mock_response
            mock_get.return_value = mock_http_client

            with pytest.raises(DataIngestionError) as exc_info:
                await client._request("/accounts/get", {"access_token": "token"})

            assert "Expected dict response" in str(exc_info.value)

        await client.close()

    @pytest.mark.asyncio
    async def test_request_rate_limit_retry(self):
        """Should retry on 429 rate limit response."""
        client = PlaidClient(client_id="test_id", secret="test_secret")

        # First response is 429, second is success
        mock_response_429 = MagicMock()
        mock_response_429.status_code = 429

        mock_response_200 = MagicMock()
        mock_response_200.status_code = 200
        mock_response_200.json.return_value = {"accounts": []}

        with patch.object(client, "_get_client") as mock_get:
            mock_http_client = AsyncMock()
            mock_http_client.post.side_effect = [mock_response_429, mock_response_200]
            mock_get.return_value = mock_http_client

            with patch("asyncio.sleep", new_callable=AsyncMock):
                result = await client._request("/accounts/get", {"access_token": "token"})

        assert result == {"accounts": []}

        await client.close()

    @pytest.mark.asyncio
    async def test_request_timeout_retry(self):
        """Should retry on timeout exception."""
        client = PlaidClient(client_id="test_id", secret="test_secret")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"accounts": []}

        with patch.object(client, "_get_client") as mock_get:
            mock_http_client = AsyncMock()
            # First call raises timeout, second succeeds
            mock_http_client.post.side_effect = [
                httpx.TimeoutException("Timeout"),
                mock_response,
            ]
            mock_get.return_value = mock_http_client

            with patch("asyncio.sleep", new_callable=AsyncMock):
                result = await client._request("/accounts/get", {"access_token": "token"})

        assert result == {"accounts": []}

        await client.close()

    @pytest.mark.asyncio
    async def test_request_max_retries_exceeded(self):
        """Should raise ExternalAPIError after max retries."""
        client = PlaidClient(client_id="test_id", secret="test_secret")

        with patch.object(client, "_get_client") as mock_get:
            mock_http_client = AsyncMock()
            mock_http_client.post.side_effect = httpx.TimeoutException("Always timeout")
            mock_get.return_value = mock_http_client

            with patch("asyncio.sleep", new_callable=AsyncMock):
                with pytest.raises(ExternalAPIError) as exc_info:
                    await client._request("/accounts/get", {"access_token": "token"})

                assert "timeout" in str(exc_info.value).lower() or "failed" in str(exc_info.value).lower()

        await client.close()


class TestGetInvestmentTransactions:
    """Test get_investment_transactions method."""

    @pytest.mark.asyncio
    async def test_get_investment_transactions_success(self):
        """Should return investment transactions dict."""
        client = PlaidClient(client_id="test_id", secret="test_secret")

        expected_response = {
            "investment_transactions": [
                {
                    "investment_transaction_id": "txn1",
                    "account_id": "acct1",
                    "security_id": "sec1",
                    "date": "2026-02-01",
                    "amount": 100.0,
                }
            ],
            "accounts": [{"account_id": "acct1", "name": "Brokerage"}],
            "securities": [{"security_id": "sec1", "ticker_symbol": "AAPL"}],
            "total_investment_transactions": 1,
        }

        with patch.object(client, "_request") as mock_request:
            mock_request.return_value = expected_response

            result = await client.get_investment_transactions(
                "test_token", "2026-02-01", "2026-02-28"
            )

        assert result == expected_response
        mock_request.assert_called_once()
        call_args = mock_request.call_args
        assert call_args[0][0] == "/investments/transactions/get"
        assert call_args[0][1]["access_token"] == "test_token"
        assert call_args[0][1]["start_date"] == "2026-02-01"
        assert call_args[0][1]["end_date"] == "2026-02-28"

    @pytest.mark.asyncio
    async def test_get_investment_transactions_with_pagination(self):
        """Should pass pagination parameters."""
        client = PlaidClient(client_id="test_id", secret="test_secret")

        with patch.object(client, "_request") as mock_request:
            mock_request.return_value = {"investment_transactions": []}

            await client.get_investment_transactions(
                "test_token", "2026-02-01", "2026-02-28", count=100, offset=50
            )

            call_args = mock_request.call_args
            options = call_args[0][1]["options"]
            assert options["count"] == 100
            assert options["offset"] == 50


class TestGetAccounts:
    """Test get_accounts method."""

    @pytest.mark.asyncio
    async def test_get_accounts_success(self):
        """Should return accounts dict."""
        client = PlaidClient(client_id="test_id", secret="test_secret")

        expected_response = {
            "accounts": [{"account_id": "acct1", "name": "Checking"}],
            "item": {"item_id": "item1"},
        }

        with patch.object(client, "_request") as mock_request:
            mock_request.return_value = expected_response

            result = await client.get_accounts("test_token")

        assert result == expected_response
        mock_request.assert_called_once()
        call_args = mock_request.call_args
        assert call_args[0][0] == "/accounts/get"
        assert call_args[0][1]["access_token"] == "test_token"


class TestGetHoldings:
    """Test get_holdings method."""

    @pytest.mark.asyncio
    async def test_get_holdings_success(self):
        """Should return holdings dict."""
        client = PlaidClient(client_id="test_id", secret="test_secret")

        expected_response = {
            "holdings": [{"security_id": "sec1", "quantity": 10.0}],
            "accounts": [{"account_id": "acct1"}],
            "securities": [{"security_id": "sec1", "ticker_symbol": "AAPL"}],
        }

        with patch.object(client, "_request") as mock_request:
            mock_request.return_value = expected_response

            result = await client.get_holdings("test_token")

        assert result == expected_response
        mock_request.assert_called_once()
        call_args = mock_request.call_args
        assert call_args[0][0] == "/investments/holdings/get"
        assert call_args[0][1]["access_token"] == "test_token"
