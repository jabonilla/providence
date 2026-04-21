"""Tests for AlphaVantageClient at providence/infra/alphavantage_client.py.

All tests mock httpx — NO real HTTP calls.
"""
import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

import httpx
import pytest

from providence.infra.alphavantage_client import AlphaVantageClient
from providence.exceptions import ExternalAPIError, DataIngestionError


class TestAlphaVantageClientInit:
    """Test AlphaVantageClient initialization."""

    def test_init_with_api_key_parameter(self):
        """Can initialize with explicit api_key parameter."""
        client = AlphaVantageClient(api_key="test_key_123")
        assert client._api_key == "test_key_123"
        assert client._base_url == AlphaVantageClient.BASE_URL
        assert client._timeout == AlphaVantageClient.DEFAULT_TIMEOUT

    def test_init_from_env_var(self):
        """Can initialize from ALPHAVANTAGE_API_KEY env var."""
        with patch.dict(os.environ, {"ALPHAVANTAGE_API_KEY": "env_key_456"}):
            client = AlphaVantageClient()
            assert client._api_key == "env_key_456"

    def test_init_prefers_parameter_over_env(self):
        """Explicit api_key parameter takes precedence over env var."""
        with patch.dict(os.environ, {"ALPHAVANTAGE_API_KEY": "env_key"}):
            client = AlphaVantageClient(api_key="param_key")
            assert client._api_key == "param_key"

    def test_init_raises_value_error_no_key(self):
        """Raises ValueError when no api_key provided and env var not set."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Alpha Vantage API key required"):
                AlphaVantageClient()

    def test_init_custom_base_url(self):
        """Can override base_url."""
        client = AlphaVantageClient(
            api_key="test_key",
            base_url="https://custom.example.com"
        )
        assert client._base_url == "https://custom.example.com"

    def test_init_custom_timeout(self):
        """Can set custom timeout."""
        client = AlphaVantageClient(api_key="test_key", timeout=60.0)
        assert client._timeout == 60.0


class TestAlphaVantageClientGetEarnings:
    """Test get_earnings method."""

    @pytest.fixture
    def client(self):
        """Create a client for testing."""
        return AlphaVantageClient(api_key="test_key")

    @pytest.mark.asyncio
    async def test_get_earnings_success(self, client):
        """get_earnings returns dict with quarterlyEarnings."""
        earnings_response = {
            "quarterlyEarnings": [
                {
                    "fiscalDateEnding": "2025-03-31",
                    "reportedEPS": "1.50",
                    "estimatedEPS": "1.45",
                    "surprise": "0.05",
                    "surprisePercentage": "3.45",
                    "reportedDate": "2025-04-15"
                },
                {
                    "fiscalDateEnding": "2024-12-31",
                    "reportedEPS": "1.30",
                    "estimatedEPS": "1.28",
                    "surprise": "0.02",
                    "surprisePercentage": "1.56",
                    "reportedDate": "2025-01-15"
                },
            ]
        }
        with patch.object(client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = earnings_response

            result = await client.get_earnings("AAPL")

            assert result == earnings_response
            assert "quarterlyEarnings" in result
            assert len(result["quarterlyEarnings"]) == 2
            mock_request.assert_called_once_with("EARNINGS", "AAPL")

    @pytest.mark.asyncio
    async def test_get_earnings_calls_request_with_function(self, client):
        """get_earnings passes EARNINGS function to _request."""
        with patch.object(client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {"quarterlyEarnings": []}

            await client.get_earnings("MSFT")

            mock_request.assert_called_once_with("EARNINGS", "MSFT")


class TestAlphaVantageClientGetIncomeStatement:
    """Test get_income_statement method."""

    @pytest.fixture
    def client(self):
        return AlphaVantageClient(api_key="test_key")

    @pytest.mark.asyncio
    async def test_get_income_statement_success(self, client):
        """get_income_statement returns dict with annualReports and quarterlyReports."""
        income_response = {
            "annualReports": [
                {
                    "fiscalDateEnding": "2024-12-31",
                    "totalRevenue": "391035000000",
                    "grossProfit": "114337000000",
                    "netIncome": "93736000000",
                }
            ],
            "quarterlyReports": [
                {
                    "fiscalDateEnding": "2025-03-31",
                    "totalRevenue": "95000000000",
                    "grossProfit": "30000000000",
                    "netIncome": "22000000000",
                }
            ]
        }
        with patch.object(client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = income_response

            result = await client.get_income_statement("AAPL")

            assert result == income_response
            assert "annualReports" in result
            assert "quarterlyReports" in result
            mock_request.assert_called_once_with("INCOME_STATEMENT", "AAPL")

    @pytest.mark.asyncio
    async def test_get_income_statement_calls_request_with_function(self, client):
        """get_income_statement passes INCOME_STATEMENT function to _request."""
        with patch.object(client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {"annualReports": []}

            await client.get_income_statement("GOOGL")

            mock_request.assert_called_once_with("INCOME_STATEMENT", "GOOGL")


class TestAlphaVantageClientGetOverview:
    """Test get_overview method."""

    @pytest.fixture
    def client(self):
        return AlphaVantageClient(api_key="test_key")

    @pytest.mark.asyncio
    async def test_get_overview_success(self, client):
        """get_overview returns dict with company fundamentals."""
        overview_response = {
            "Symbol": "AAPL",
            "Name": "Apple Inc",
            "PERatio": "28.5",
            "DividendYield": "0.005",
            "EPS": "6.05",
            "MarketCapitalization": "3000000000000",
            "52WeekHigh": "250.00",
            "52WeekLow": "150.00",
        }
        with patch.object(client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = overview_response

            result = await client.get_overview("AAPL")

            assert result == overview_response
            assert result["Symbol"] == "AAPL"
            mock_request.assert_called_once_with("OVERVIEW", "AAPL")

    @pytest.mark.asyncio
    async def test_get_overview_calls_request_with_function(self, client):
        """get_overview passes OVERVIEW function to _request."""
        with patch.object(client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {}

            await client.get_overview("AMZN")

            mock_request.assert_called_once_with("OVERVIEW", "AMZN")


class TestAlphaVantageClientRateLimiting:
    """Test rate limiting behavior."""

    @pytest.fixture
    def client(self):
        return AlphaVantageClient(api_key="test_key")

    @pytest.mark.asyncio
    async def test_rate_limit_enforces_minimum_interval(self, client):
        """_rate_limit enforces minimum 15 second interval between requests."""
        import time

        # First request should go through immediately
        start = time.monotonic()
        await client._rate_limit()
        first_time = time.monotonic() - start

        # Should be nearly instant
        assert first_time < 0.1

        # Simulate a second request immediately after
        start = time.monotonic()
        # Manually set last_request_time to current time - 5 seconds
        # This means we need to wait 10 more seconds
        client._last_request_time = time.monotonic() - 5.0
        await client._rate_limit()
        elapsed = time.monotonic() - start

        # Should sleep ~10 seconds (MIN_REQUEST_INTERVAL - 5)
        assert elapsed >= 9.0  # Allow some tolerance

    @pytest.mark.asyncio
    async def test_rate_limit_skips_sleep_if_enough_time_passed(self, client):
        """_rate_limit skips sleep if enough time has already passed."""
        import time

        # Set last request time to 20 seconds ago (more than MIN_REQUEST_INTERVAL)
        client._last_request_time = time.monotonic() - 20.0

        start = time.monotonic()
        await client._rate_limit()
        elapsed = time.monotonic() - start

        # Should not sleep (already 20 seconds have passed)
        assert elapsed < 1.0


class TestAlphaVantageClientRetry:
    """Test retry logic."""

    @pytest.fixture
    def client(self):
        return AlphaVantageClient(api_key="test_key")

    @pytest.mark.asyncio
    async def test_retry_on_429_rate_limit(self, client):
        """_request retries on 429 Too Many Requests."""
        with patch.object(client, "_get_client", new_callable=AsyncMock) as mock_get_client:
            mock_http_client = AsyncMock(spec=httpx.AsyncClient)
            mock_get_client.return_value = mock_http_client

            # First call returns 429, second returns 200
            response_429 = MagicMock()
            response_429.status_code = 429

            response_200 = MagicMock()
            response_200.status_code = 200
            response_200.json.return_value = {"quarterlyEarnings": []}

            mock_http_client.get.side_effect = [response_429, response_200]

            with patch.object(client, "_rate_limit", new_callable=AsyncMock):
                result = await client._request("EARNINGS", "AAPL")

            assert result["quarterlyEarnings"] == []
            assert mock_http_client.get.call_count == 2

    @pytest.mark.asyncio
    async def test_retry_on_timeout(self, client):
        """_request retries on timeout exception."""
        with patch.object(client, "_get_client", new_callable=AsyncMock) as mock_get_client:
            mock_http_client = AsyncMock(spec=httpx.AsyncClient)
            mock_get_client.return_value = mock_http_client

            response_200 = MagicMock()
            response_200.status_code = 200
            response_200.json.return_value = {"annualReports": []}

            # First call times out, second succeeds
            mock_http_client.get.side_effect = [
                httpx.TimeoutException("timeout"),
                response_200,
            ]

            with patch.object(client, "_rate_limit", new_callable=AsyncMock):
                result = await client._request("INCOME_STATEMENT", "MSFT")

            assert result["annualReports"] == []
            assert mock_http_client.get.call_count == 2

    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self, client):
        """_request raises after MAX_RETRIES attempts."""
        with patch.object(client, "_get_client", new_callable=AsyncMock) as mock_get_client:
            mock_http_client = AsyncMock(spec=httpx.AsyncClient)
            mock_get_client.return_value = mock_http_client

            # Always timeout
            mock_http_client.get.side_effect = httpx.TimeoutException("timeout")

            with patch.object(client, "_rate_limit", new_callable=AsyncMock):
                with pytest.raises(ExternalAPIError):
                    await client._request("EARNINGS", "AAPL")

            # Should retry MAX_RETRIES times
            assert mock_http_client.get.call_count == AlphaVantageClient.MAX_RETRIES

    @pytest.mark.asyncio
    async def test_no_retry_on_non_retryable_error(self, client):
        """_request does NOT retry on 400 Bad Request."""
        with patch.object(client, "_get_client", new_callable=AsyncMock) as mock_get_client:
            mock_http_client = AsyncMock(spec=httpx.AsyncClient)
            mock_get_client.return_value = mock_http_client

            response_400 = MagicMock()
            response_400.status_code = 400
            response_400.text = "Bad request"

            mock_http_client.get.return_value = response_400

            with patch.object(client, "_rate_limit", new_callable=AsyncMock):
                with pytest.raises(ExternalAPIError):
                    await client._request("EARNINGS", "AAPL")

            # Should only be called once (no retry)
            assert mock_http_client.get.call_count == 1


class TestAlphaVantageClientErrorHandling:
    """Test error response handling."""

    @pytest.fixture
    def client(self):
        return AlphaVantageClient(api_key="test_key")

    @pytest.mark.asyncio
    async def test_error_message_key_in_response(self, client):
        """_request raises when response contains 'Error Message' key."""
        with patch.object(client, "_get_client", new_callable=AsyncMock) as mock_get_client:
            mock_http_client = AsyncMock(spec=httpx.AsyncClient)
            mock_get_client.return_value = mock_http_client

            response = MagicMock()
            response.status_code = 200
            response.json.return_value = {
                "Error Message": "Invalid symbol or API key"
            }

            mock_http_client.get.return_value = response

            with patch.object(client, "_rate_limit", new_callable=AsyncMock):
                with pytest.raises(ExternalAPIError, match="API error"):
                    await client._request("EARNINGS", "INVALID")

    @pytest.mark.asyncio
    async def test_note_key_in_response(self, client):
        """_request raises when response contains 'Note' key (rate limit note)."""
        with patch.object(client, "_get_client", new_callable=AsyncMock) as mock_get_client:
            mock_http_client = AsyncMock(spec=httpx.AsyncClient)
            mock_get_client.return_value = mock_http_client

            response = MagicMock()
            response.status_code = 200
            response.json.return_value = {
                "Note": "Thank you for using Alpha Vantage! Our standard API call frequency is 5 calls per minute."
            }

            mock_http_client.get.return_value = response

            with patch.object(client, "_rate_limit", new_callable=AsyncMock):
                with pytest.raises(ExternalAPIError, match="API note"):
                    await client._request("EARNINGS", "AAPL")

    @pytest.mark.asyncio
    async def test_non_200_status_code_raises(self, client):
        """_request raises ExternalAPIError on non-200 status codes."""
        with patch.object(client, "_get_client", new_callable=AsyncMock) as mock_get_client:
            mock_http_client = AsyncMock(spec=httpx.AsyncClient)
            mock_get_client.return_value = mock_http_client

            response = MagicMock()
            response.status_code = 500
            response.text = "Internal Server Error"

            mock_http_client.get.return_value = response

            with patch.object(client, "_rate_limit", new_callable=AsyncMock):
                with pytest.raises(ExternalAPIError, match="500"):
                    await client._request("EARNINGS", "AAPL")

    @pytest.mark.asyncio
    async def test_non_dict_response_raises(self, client):
        """_request raises DataIngestionError if response is not a dict."""
        with patch.object(client, "_get_client", new_callable=AsyncMock) as mock_get_client:
            mock_http_client = AsyncMock(spec=httpx.AsyncClient)
            mock_get_client.return_value = mock_http_client

            response = MagicMock()
            response.status_code = 200
            response.json.return_value = ["not", "a", "dict"]  # List instead of dict

            mock_http_client.get.return_value = response

            with patch.object(client, "_rate_limit", new_callable=AsyncMock):
                with pytest.raises(DataIngestionError, match="Expected dict"):
                    await client._request("EARNINGS", "AAPL")


class TestAlphaVantageClientClose:
    """Test client lifecycle methods."""

    @pytest.mark.asyncio
    async def test_close_closes_http_client(self):
        """close() closes the underlying HTTP client."""
        client = AlphaVantageClient(api_key="test_key")

        # Access _get_client to create the client
        with patch.object(client, "_get_client", new_callable=AsyncMock):
            client._client = AsyncMock(spec=httpx.AsyncClient)
            client._client.is_closed = False

            await client.close()

            client._client.aclose.assert_called_once()
            assert client._client is None

    @pytest.mark.asyncio
    async def test_close_idempotent(self):
        """close() is idempotent and handles already-closed clients."""
        client = AlphaVantageClient(api_key="test_key")
        client._client = None

        # Should not raise
        await client.close()
        assert client._client is None
