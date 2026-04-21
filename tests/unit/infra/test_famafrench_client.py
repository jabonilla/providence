"""Tests for FamaFrenchClient at providence/infra/famafrench_client.py.

All tests mock pandas_datareader — NO real API calls.
"""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from providence.exceptions import DataIngestionError, ExternalAPIError
from providence.infra.famafrench_client import FamaFrenchClient


class TestFamaFrenchClientInit:
    """Test FamaFrenchClient initialization."""

    def test_init_no_args(self):
        """Client should initialize with no arguments."""
        client = FamaFrenchClient()
        assert client._last_request_time == 0.0
        assert client.MIN_REQUEST_INTERVAL == 1.0

    def test_multiple_instances_independent(self):
        """Multiple instances should have independent request timing."""
        client1 = FamaFrenchClient()
        client2 = FamaFrenchClient()
        assert client1._last_request_time == client2._last_request_time
        assert client1 is not client2


class TestFamaFrenchClientRateLimit:
    """Test rate limiting behavior."""

    @pytest.mark.asyncio
    async def test_rate_limit_enforced(self):
        """Rate limit should enforce minimum interval between requests."""
        client = FamaFrenchClient()
        client._last_request_time = asyncio.get_event_loop().time()

        # First call should be immediate
        await client._rate_limit()

        # Second call without delay should sleep
        import time
        before = time.time()
        await client._rate_limit()
        elapsed = time.time() - before

        # Should have slept roughly the MIN_REQUEST_INTERVAL
        assert elapsed >= client.MIN_REQUEST_INTERVAL * 0.9  # Allow 10% variance

    @pytest.mark.asyncio
    async def test_rate_limit_updates_time(self):
        """Rate limit should update the last request time."""
        client = FamaFrenchClient()
        client._last_request_time = 0.0

        await client._rate_limit()
        assert client._last_request_time > 0.0


class TestGetFiveFactorsDaily:
    """Test get_five_factors_daily method."""

    @pytest.mark.asyncio
    async def test_get_five_factors_daily_success(self):
        """Should return list of dicts with factor fields."""
        client = FamaFrenchClient()

        mock_df = {
            "Mkt-RF": [1.23, 0.45],
            "SMB": [0.12, -0.05],
            "HML": [0.34, 0.56],
            "RMW": [0.10, 0.20],
            "CMA": [0.08, -0.12],
            "RF": [0.02, 0.02],
        }

        import pandas as pd

        df = pd.DataFrame(mock_df)
        df.index = pd.date_range("2026-02-01", periods=2)

        with patch("asyncio.to_thread") as mock_thread:
            mock_thread.return_value = df

            result = await client.get_five_factors_daily("2026-02-01", "2026-02-03")

        assert len(result) == 2
        assert result[0]["date"] == "2026-02-01"
        assert result[0]["mkt_rf"] == 1.23
        assert result[0]["smb"] == 0.12
        assert result[0]["hml"] == 0.34
        assert result[0]["rmw"] == 0.10
        assert result[0]["cma"] == 0.08
        assert result[0]["rf"] == 0.02

    @pytest.mark.asyncio
    async def test_get_five_factors_tuple_response(self):
        """Should unpack tuple response (df, description)."""
        client = FamaFrenchClient()

        import pandas as pd

        df = pd.DataFrame({
            "Mkt-RF": [1.0],
            "SMB": [0.1],
            "HML": [0.2],
            "RMW": [0.05],
            "CMA": [0.03],
            "RF": [0.01],
        })
        df.index = pd.date_range("2026-02-01", periods=1)

        # DataReader returns tuple (df, description)
        tuple_response = (df, "Some description")

        with patch("asyncio.to_thread") as mock_thread:
            mock_thread.return_value = tuple_response

            result = await client.get_five_factors_daily("2026-02-01", "2026-02-03")

        assert len(result) == 1
        assert result[0]["date"] == "2026-02-01"

    @pytest.mark.asyncio
    async def test_get_five_factors_empty_raises_error(self):
        """Should raise DataIngestionError when data is empty."""
        client = FamaFrenchClient()

        import pandas as pd

        empty_df = pd.DataFrame()

        with patch("asyncio.to_thread") as mock_thread:
            mock_thread.return_value = empty_df

            with pytest.raises(DataIngestionError) as exc_info:
                await client.get_five_factors_daily("2026-02-01", "2026-02-03")

            assert "No 5-factor data returned" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_five_factors_none_raises_error(self):
        """Should raise DataIngestionError when data is None."""
        client = FamaFrenchClient()

        with patch("asyncio.to_thread") as mock_thread:
            mock_thread.return_value = None

            with pytest.raises(DataIngestionError):
                await client.get_five_factors_daily("2026-02-01", "2026-02-03")

    @pytest.mark.asyncio
    async def test_get_five_factors_import_error(self):
        """Should raise ExternalAPIError when pandas_datareader not available."""
        client = FamaFrenchClient()

        with patch.dict("sys.modules", {"pandas_datareader.data": None}):
            with patch("providence.infra.famafrench_client.web", side_effect=ImportError("No module")):
                with pytest.raises(ExternalAPIError) as exc_info:
                    await client.get_five_factors_daily("2026-02-01", "2026-02-03")

                assert "pandas_datareader" in str(exc_info.value)


class TestGetMomentumDaily:
    """Test get_momentum_daily method."""

    @pytest.mark.asyncio
    async def test_get_momentum_daily_success(self):
        """Should return list of dicts with mom field."""
        client = FamaFrenchClient()

        import pandas as pd

        # Note: FF dataset has trailing spaces in column name
        df = pd.DataFrame({"Mom   ": [1.5, 2.1]})
        df.index = pd.date_range("2026-02-01", periods=2)

        with patch("asyncio.to_thread") as mock_thread:
            mock_thread.return_value = df

            result = await client.get_momentum_daily("2026-02-01", "2026-02-03")

        assert len(result) == 2
        assert result[0]["date"] == "2026-02-01"
        assert result[0]["mom"] == 1.5
        assert result[1]["date"] == "2026-02-02"
        assert result[1]["mom"] == 2.1

    @pytest.mark.asyncio
    async def test_get_momentum_tuple_response(self):
        """Should unpack tuple response (df, description)."""
        client = FamaFrenchClient()

        import pandas as pd

        df = pd.DataFrame({"Mom   ": [1.0]})
        df.index = pd.date_range("2026-02-01", periods=1)

        tuple_response = (df, "Momentum description")

        with patch("asyncio.to_thread") as mock_thread:
            mock_thread.return_value = tuple_response

            result = await client.get_momentum_daily("2026-02-01", "2026-02-03")

        assert len(result) == 1
        assert result[0]["mom"] == 1.0

    @pytest.mark.asyncio
    async def test_get_momentum_empty_raises_error(self):
        """Should raise DataIngestionError when data is empty."""
        client = FamaFrenchClient()

        import pandas as pd

        empty_df = pd.DataFrame()

        with patch("asyncio.to_thread") as mock_thread:
            mock_thread.return_value = empty_df

            with pytest.raises(DataIngestionError) as exc_info:
                await client.get_momentum_daily("2026-02-01", "2026-02-03")

            assert "No momentum data returned" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_momentum_none_raises_error(self):
        """Should raise DataIngestionError when data is None."""
        client = FamaFrenchClient()

        with patch("asyncio.to_thread") as mock_thread:
            mock_thread.return_value = None

            with pytest.raises(DataIngestionError):
                await client.get_momentum_daily("2026-02-01", "2026-02-03")

    @pytest.mark.asyncio
    async def test_get_momentum_api_error(self):
        """Should raise ExternalAPIError on API failures."""
        client = FamaFrenchClient()

        with patch("asyncio.to_thread") as mock_thread:
            mock_thread.side_effect = RuntimeError("Data source error")

            with pytest.raises(ExternalAPIError) as exc_info:
                await client.get_momentum_daily("2026-02-01", "2026-02-03")

            assert "Failed to fetch momentum data" in str(exc_info.value)
            assert "fama_french" in str(exc_info.value)


class TestRateLimitingBehavior:
    """Integration-level tests for rate limiting across calls."""

    @pytest.mark.asyncio
    async def test_rate_limit_across_calls(self):
        """Rate limiting should be shared across different methods."""
        client = FamaFrenchClient()

        import pandas as pd
        import time

        df = pd.DataFrame({
            "Mkt-RF": [1.0],
            "SMB": [0.1],
            "HML": [0.2],
            "RMW": [0.05],
            "CMA": [0.03],
            "RF": [0.01],
            "Mom   ": [1.5],
        })
        df.index = pd.date_range("2026-02-01", periods=1)

        with patch("asyncio.to_thread") as mock_thread:
            mock_thread.return_value = df

            # First call
            before = time.time()
            await client.get_five_factors_daily("2026-02-01", "2026-02-03")
            first_elapsed = time.time() - before

            # Second call should respect rate limit
            before = time.time()
            await client.get_momentum_daily("2026-02-01", "2026-02-03")
            second_elapsed = time.time() - before

            # Second call should sleep due to rate limit
            total_elapsed = first_elapsed + second_elapsed
            assert total_elapsed >= client.MIN_REQUEST_INTERVAL * 0.9
