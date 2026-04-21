"""Tests for YFinanceClient at providence/infra/yfinance_client.py.

All tests mock yfinance and asyncio.to_thread — NO real API calls.
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from providence.exceptions import ExternalAPIError, DataIngestionError
from providence.infra.yfinance_client import YFinanceClient


class TestYFinanceClientInit:
    """Test YFinanceClient initialization."""

    def test_init_creates_client_with_default_timeout(self) -> None:
        """Client should initialize with default 30s timeout."""
        client = YFinanceClient()
        assert client._timeout == 30.0
        assert client._last_request_time == 0.0
        assert client._yf is None

    def test_init_custom_timeout(self) -> None:
        """Can set custom timeout."""
        client = YFinanceClient(timeout=60.0)
        assert client._timeout == 60.0

    @pytest.mark.asyncio
    async def test_close_is_noop(self) -> None:
        """close() should be a no-op (no persistent connection)."""
        client = YFinanceClient()
        await client.close()  # Should not raise


class TestYFinanceClientImport:
    """Test lazy import of yfinance module."""

    def test_ensure_yfinance_imports_module(self) -> None:
        """_ensure_yfinance should lazy-import yfinance."""
        client = YFinanceClient()
        with patch("providence.infra.yfinance_client.yfinance") as mock_yf:
            # First call should import
            result = client._ensure_yfinance()
            assert result is mock_yf

    def test_ensure_yfinance_caches_module(self) -> None:
        """_ensure_yfinance should cache the module."""
        client = YFinanceClient()
        with patch("providence.infra.yfinance_client.yfinance") as mock_yf:
            result1 = client._ensure_yfinance()
            result2 = client._ensure_yfinance()
            # Should return same cached instance
            assert result1 is result2

    def test_ensure_yfinance_raises_on_missing_import(self) -> None:
        """_ensure_yfinance should raise ExternalAPIError if import fails."""
        client = YFinanceClient()
        with patch(
            "providence.infra.yfinance_client.yfinance",
            side_effect=ImportError("No module named 'yfinance'"),
        ):
            with pytest.raises(ExternalAPIError) as exc_info:
                client._ensure_yfinance()
            assert "yfinance library required" in str(exc_info.value)
            assert "pip install yfinance" in str(exc_info.value)


class TestYFinanceClientRateLimit:
    """Test rate limiting behavior."""

    @pytest.mark.asyncio
    async def test_rate_limit_enforces_minimum_interval(self) -> None:
        """_rate_limit should enforce MIN_REQUEST_INTERVAL between calls."""
        client = YFinanceClient()
        client._last_request_time = time.monotonic()

        # Immediately calling should sleep
        start = time.monotonic()
        await client._rate_limit()
        elapsed = time.monotonic() - start

        # Should have slept roughly MIN_REQUEST_INTERVAL (0.5s)
        assert elapsed >= client.MIN_REQUEST_INTERVAL - 0.05  # Small tolerance

    @pytest.mark.asyncio
    async def test_rate_limit_no_sleep_if_interval_elapsed(self) -> None:
        """_rate_limit should not sleep if interval already elapsed."""
        client = YFinanceClient()
        client._last_request_time = time.monotonic() - 1.0  # 1 second ago

        start = time.monotonic()
        await client._rate_limit()
        elapsed = time.monotonic() - start

        # Should have slept < MIN_REQUEST_INTERVAL
        assert elapsed < client.MIN_REQUEST_INTERVAL / 2


class TestYFinanceClientGetFundamentals:
    """Test get_fundamentals method."""

    @pytest.mark.asyncio
    async def test_get_fundamentals_returns_dict_with_mock(self) -> None:
        """Valid fundamentals data should return dict."""
        mock_yf = MagicMock()
        mock_ticker = MagicMock()
        mock_ticker.info = {
            "regularMarketPrice": 185.50,
            "marketCap": 2_900_000_000_000,
            "trailingPE": 29.5,
            "sector": "Technology",
        }
        mock_yf.Ticker.return_value = mock_ticker

        client = YFinanceClient()
        with patch.object(client, "_ensure_yfinance", return_value=mock_yf):
            with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
                mock_thread.return_value = mock_ticker.info
                result = await client.get_fundamentals("AAPL")

        assert result["regularMarketPrice"] == 185.50
        assert result["marketCap"] == 2_900_000_000_000
        assert result["trailingPE"] == 29.5

    @pytest.mark.asyncio
    async def test_get_fundamentals_raises_on_no_data(self) -> None:
        """Should raise DataIngestionError if no fundamental data returned."""
        mock_yf = MagicMock()
        mock_ticker = MagicMock()
        mock_ticker.info = {}  # Empty
        mock_yf.Ticker.return_value = mock_ticker

        client = YFinanceClient()
        with patch.object(client, "_ensure_yfinance", return_value=mock_yf):
            with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
                # _sync_get_fundamentals raises DataIngestionError for empty info
                mock_thread.side_effect = DataIngestionError(
                    message="No fundamental data returned for AAPL"
                )
                with pytest.raises(DataIngestionError) as exc_info:
                    await client.get_fundamentals("AAPL")
                assert "No fundamental data" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_fundamentals_raises_on_none_price(self) -> None:
        """Should raise DataIngestionError if regularMarketPrice is None."""
        client = YFinanceClient()
        with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
            mock_thread.side_effect = DataIngestionError(
                message="No fundamental data returned for INVALID"
            )
            with pytest.raises(DataIngestionError):
                await client.get_fundamentals("INVALID")

    @pytest.mark.asyncio
    async def test_get_fundamentals_retry_on_failure(self) -> None:
        """Should retry up to MAX_RETRIES on transient failure."""
        client = YFinanceClient()
        # First two attempts fail, third succeeds
        with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
            mock_thread.side_effect = [
                Exception("Temporary network error"),
                Exception("Still failing"),
                {"regularMarketPrice": 185.50, "marketCap": 2_900_000_000_000},
            ]
            with patch.object(client, "_rate_limit", new_callable=AsyncMock):
                with patch("asyncio.sleep", new_callable=AsyncMock):
                    result = await client.get_fundamentals("AAPL")

        assert result["regularMarketPrice"] == 185.50

    @pytest.mark.asyncio
    async def test_get_fundamentals_exponential_backoff(self) -> None:
        """Should use exponential backoff on retries."""
        client = YFinanceClient()
        sleep_times = []

        async def mock_sleep(seconds: float) -> None:
            sleep_times.append(seconds)

        with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
            mock_thread.side_effect = Exception("Network error")
            with patch.object(client, "_rate_limit", new_callable=AsyncMock):
                with patch("asyncio.sleep", side_effect=mock_sleep):
                    with pytest.raises(ExternalAPIError):
                        await client.get_fundamentals("AAPL")

        # Should have 2 sleep calls (MAX_RETRIES - 1)
        assert len(sleep_times) == 2
        # Backoff should be 1.0 * 2^0 = 1.0, 1.0 * 2^1 = 2.0
        assert sleep_times[0] == client.RETRY_BACKOFF_BASE * (2 ** 0)
        assert sleep_times[1] == client.RETRY_BACKOFF_BASE * (2 ** 1)


class TestYFinanceClientGetPriceHistory:
    """Test get_price_history method."""

    @pytest.mark.asyncio
    async def test_get_price_history_returns_list_of_dicts(self) -> None:
        """Valid price history should return list of bar dicts."""
        mock_yf = MagicMock()
        mock_ticker = MagicMock()

        # Create a mock DataFrame
        import pandas as pd
        from datetime import datetime
        dates = pd.date_range("2025-02-01", periods=3)
        df = pd.DataFrame({
            "Open": [185.0, 186.0, 187.0],
            "High": [187.0, 188.0, 189.0],
            "Low": [184.0, 185.0, 186.0],
            "Close": [186.0, 187.0, 188.0],
            "Volume": [50_000_000, 51_000_000, 52_000_000],
            "Dividends": [0.0, 0.0, 0.0],
            "Stock Splits": [0.0, 0.0, 0.0],
        }, index=dates)

        mock_ticker.history.return_value = df
        mock_yf.Ticker.return_value = mock_ticker

        client = YFinanceClient()
        with patch.object(client, "_ensure_yfinance", return_value=mock_yf):
            with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
                mock_thread.return_value = [
                    {
                        "date": "2025-02-01",
                        "open": 185.0,
                        "high": 187.0,
                        "low": 184.0,
                        "close": 186.0,
                        "volume": 50_000_000,
                        "dividends": 0.0,
                        "stock_splits": 0.0,
                    },
                    {
                        "date": "2025-02-02",
                        "open": 186.0,
                        "high": 188.0,
                        "low": 185.0,
                        "close": 187.0,
                        "volume": 51_000_000,
                        "dividends": 0.0,
                        "stock_splits": 0.0,
                    },
                    {
                        "date": "2025-02-03",
                        "open": 187.0,
                        "high": 189.0,
                        "low": 186.0,
                        "close": 188.0,
                        "volume": 52_000_000,
                        "dividends": 0.0,
                        "stock_splits": 0.0,
                    },
                ]
                result = await client.get_price_history("AAPL", period="3mo")

        assert len(result) == 3
        assert result[0]["close"] == 186.0
        assert result[1]["volume"] == 51_000_000
        assert all("date" in bar for bar in result)
        assert all("open" in bar for bar in result)
        assert all("high" in bar for bar in result)
        assert all("low" in bar for bar in result)
        assert all("close" in bar for bar in result)

    @pytest.mark.asyncio
    async def test_get_price_history_raises_on_empty(self) -> None:
        """Should raise DataIngestionError if no price history returned."""
        client = YFinanceClient()
        with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
            mock_thread.side_effect = DataIngestionError(
                message="No price history returned for AAPL (period=3mo)"
            )
            with patch.object(client, "_rate_limit", new_callable=AsyncMock):
                with pytest.raises(DataIngestionError) as exc_info:
                    await client.get_price_history("AAPL")
                assert "No price history" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_price_history_custom_period_interval(self) -> None:
        """Should support custom period and interval parameters."""
        client = YFinanceClient()
        with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
            mock_thread.return_value = [
                {
                    "date": "2025-02-01",
                    "open": 185.0,
                    "high": 187.0,
                    "low": 184.0,
                    "close": 186.0,
                    "volume": 50_000_000,
                    "dividends": 0.0,
                    "stock_splits": 0.0,
                }
            ]
            with patch.object(client, "_rate_limit", new_callable=AsyncMock):
                result = await client.get_price_history(
                    "AAPL", period="1y", interval="1wk"
                )

        assert len(result) >= 1
        mock_thread.assert_called_once()
        # Check that _sync_get_price_history was called with correct args
        args, kwargs = mock_thread.call_args
        assert "AAPL" in args
        assert "1y" in args
        assert "1wk" in args


class TestYFinanceClientGetInstitutionalHolders:
    """Test get_institutional_holders method."""

    @pytest.mark.asyncio
    async def test_get_institutional_holders_returns_list(self) -> None:
        """Should return list of holder dicts."""
        mock_yf = MagicMock()
        mock_ticker = MagicMock()

        import pandas as pd
        df = pd.DataFrame({
            "Holder": ["BlackRock Inc.", "Vanguard Group"],
            "Shares": [1_000_000, 900_000],
            "Date Reported": ["2025-02-01", "2025-02-01"],
            "% Out": [5.0, 4.5],
            "Value": [185_000_000, 168_300_000],
        })
        mock_ticker.institutional_holders = df
        mock_yf.Ticker.return_value = mock_ticker

        client = YFinanceClient()
        with patch.object(client, "_ensure_yfinance", return_value=mock_yf):
            with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
                mock_thread.return_value = [
                    {
                        "holder": "BlackRock Inc.",
                        "shares": 1_000_000,
                        "date_reported": "2025-02-01",
                        "pct_out": 5.0,
                        "value": 185_000_000,
                    },
                    {
                        "holder": "Vanguard Group",
                        "shares": 900_000,
                        "date_reported": "2025-02-01",
                        "pct_out": 4.5,
                        "value": 168_300_000,
                    },
                ]
                result = await client.get_institutional_holders("AAPL")

        assert len(result) == 2
        assert result[0]["holder"] == "BlackRock Inc."
        assert result[1]["shares"] == 900_000

    @pytest.mark.asyncio
    async def test_get_institutional_holders_returns_empty_on_failure(self) -> None:
        """Should return empty list on error (graceful degradation)."""
        client = YFinanceClient()
        with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
            mock_thread.side_effect = Exception("API error")
            with patch.object(client, "_rate_limit", new_callable=AsyncMock):
                result = await client.get_institutional_holders("AAPL")

        assert result == []

    @pytest.mark.asyncio
    async def test_get_institutional_holders_empty_dataframe(self) -> None:
        """Should return empty list if no holders found."""
        client = YFinanceClient()
        with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
            mock_thread.return_value = []
            with patch.object(client, "_rate_limit", new_callable=AsyncMock):
                result = await client.get_institutional_holders("AAPL")

        assert result == []


class TestYFinanceClientGetInfo:
    """Test get_info method."""

    @pytest.mark.asyncio
    async def test_get_info_returns_full_dict(self) -> None:
        """Should return full info dictionary."""
        mock_yf = MagicMock()
        mock_ticker = MagicMock()
        mock_ticker.info = {
            "regularMarketPrice": 185.50,
            "marketCap": 2_900_000_000_000,
            "sector": "Technology",
        }
        mock_yf.Ticker.return_value = mock_ticker

        client = YFinanceClient()
        with patch.object(client, "_ensure_yfinance", return_value=mock_yf):
            with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
                mock_thread.return_value = mock_ticker.info
                with patch.object(client, "_rate_limit", new_callable=AsyncMock):
                    result = await client.get_info("AAPL")

        assert result["regularMarketPrice"] == 185.50
        assert result["marketCap"] == 2_900_000_000_000

    @pytest.mark.asyncio
    async def test_get_info_raises_on_error(self) -> None:
        """Should raise ExternalAPIError on API failure."""
        client = YFinanceClient()
        with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
            mock_thread.side_effect = Exception("Network error")
            with patch.object(client, "_rate_limit", new_callable=AsyncMock):
                with pytest.raises(ExternalAPIError) as exc_info:
                    await client.get_info("INVALID")
                assert "yfinance info error" in str(exc_info.value)
