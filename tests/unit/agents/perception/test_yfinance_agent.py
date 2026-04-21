"""Tests for PERCEPT-YFINANCE agent.

Uses mocked YFinanceClient to test the full Perception loop:
FETCH → VALIDATE → NORMALIZE → VERSION → STORE/ALERT

All tests run without real API calls.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch
from uuid import UUID

import pytest

from providence.agents.base import AgentContext, AgentStatus
from providence.agents.perception.yfinance_agent import PerceptYFinance
from providence.exceptions import AgentProcessingError, DataIngestionError
from providence.infra.yfinance_client import YFinanceClient
from providence.schemas.enums import DataType, ValidationStatus


# ===================================================================
# Helpers
# ===================================================================
def _make_context(
    tickers: list[str],
    date: str = "2026-02-09",
) -> AgentContext:
    """Create an AgentContext for PERCEPT-YFINANCE testing."""
    return AgentContext(
        agent_id="PERCEPT-YFINANCE",
        trigger="schedule",
        context_window_hash="test_hash",
        timestamp=datetime.now(timezone.utc),
        metadata={"tickers": tickers, "date": date},
    )


def _make_agent(mock_client: AsyncMock) -> PerceptYFinance:
    """Create a PerceptYFinance agent with a mocked YFinanceClient."""
    return PerceptYFinance(yfinance_client=mock_client)


# ===================================================================
# Valid Data Tests
# ===================================================================
class TestPerceptYFinanceValidData:
    """Test PERCEPT-YFINANCE with valid fundamentals data."""

    @pytest.mark.asyncio
    async def test_single_ticker_produces_valid_fragment(self) -> None:
        """Valid AAPL data should produce a VALID MarketStateFragment."""
        mock_client = AsyncMock(spec=YFinanceClient)
        mock_client.get_fundamentals.return_value = {
            "regularMarketPrice": 185.50,
            "marketCap": 2_900_000_000_000,
            "trailingPE": 29.5,
            "sector": "Technology",
            "industry": "Software",
        }
        agent = _make_agent(mock_client)

        context = _make_context(["AAPL"])
        fragments = await agent.process(context)

        assert len(fragments) == 1
        frag = fragments[0]
        assert frag.entity == "AAPL"
        assert frag.data_type == DataType.YFINANCE_FUNDAMENTALS
        assert frag.validation_status == ValidationStatus.VALID
        assert frag.agent_id == "PERCEPT-YFINANCE"
        assert isinstance(frag.fragment_id, UUID)
        assert frag.schema_version == "1.0.0"

    @pytest.mark.asyncio
    async def test_payload_fields_normalized_correctly(self) -> None:
        """Payload should contain normalized YFinanceFundamentalsPayload fields."""
        mock_client = AsyncMock(spec=YFinanceClient)
        mock_client.get_fundamentals.return_value = {
            "regularMarketPrice": 185.50,
            "marketCap": 2_900_000_000_000,
            "enterpriseValue": 2_850_000_000_000,
            "trailingPE": 29.5,
            "forwardPE": 28.0,
            "pegRatio": 2.1,
            "priceToBook": 42.5,
            "priceToSalesTrailing12Months": 8.2,
            "profitMargins": 0.25,
            "operatingMargins": 0.30,
            "returnOnEquity": 0.85,
            "returnOnAssets": 0.15,
            "totalRevenue": 391_000_000_000,
            "revenueGrowth": 0.10,
            "earningsGrowth": 0.12,
            "debtToEquity": 1.5,
            "currentRatio": 2.1,
            "freeCashflow": 110_000_000_000,
            "dividendYield": 0.005,
            "beta": 1.2,
            "fiftyTwoWeekHigh": 199.0,
            "fiftyTwoWeekLow": 164.0,
            "averageVolume": 52_000_000,
            "sharesOutstanding": 15_600_000_000,
            "heldPercentInstitutions": 0.62,
            "shortRatio": 0.02,
            "sector": "Technology",
            "industry": "Software",
        }
        agent = _make_agent(mock_client)

        context = _make_context(["AAPL"])
        fragments = await agent.process(context)

        payload = fragments[0].payload
        assert payload["ticker"] == "AAPL"
        assert payload["market_cap"] == 2_900_000_000_000
        assert payload["enterprise_value"] == 2_850_000_000_000
        assert payload["trailing_pe"] == 29.5
        assert payload["forward_pe"] == 28.0
        assert payload["peg_ratio"] == 2.1
        assert payload["price_to_book"] == 42.5
        assert payload["profit_margin"] == 0.25
        assert payload["operating_margin"] == 0.30
        assert payload["roe"] == 0.85
        assert payload["roa"] == 0.15
        assert payload["revenue"] == 391_000_000_000
        assert payload["sector"] == "Technology"
        assert payload["industry"] == "Software"
        assert payload["observation_date"] == "2026-02-09"

    @pytest.mark.asyncio
    async def test_multiple_tickers_produce_one_fragment_each(self) -> None:
        """Multiple tickers should produce one fragment each."""
        mock_client = AsyncMock(spec=YFinanceClient)
        aapl_data = {
            "regularMarketPrice": 185.50,
            "marketCap": 2_900_000_000_000,
            "sector": "Technology",
        }
        msft_data = {
            "regularMarketPrice": 380.0,
            "marketCap": 2_800_000_000_000,
            "sector": "Technology",
        }
        mock_client.get_fundamentals.side_effect = [aapl_data, msft_data]
        agent = _make_agent(mock_client)

        context = _make_context(["AAPL", "MSFT"])
        fragments = await agent.process(context)

        assert len(fragments) == 2
        entities = {f.entity for f in fragments}
        assert entities == {"AAPL", "MSFT"}
        assert all(f.validation_status == ValidationStatus.VALID for f in fragments)
        assert all(f.data_type == DataType.YFINANCE_FUNDAMENTALS for f in fragments)

    @pytest.mark.asyncio
    async def test_observation_date_from_context(self) -> None:
        """Observation date should come from context metadata."""
        mock_client = AsyncMock(spec=YFinanceClient)
        mock_client.get_fundamentals.return_value = {
            "regularMarketPrice": 185.50,
            "marketCap": 2_900_000_000_000,
        }
        agent = _make_agent(mock_client)

        context = _make_context(["AAPL"], date="2026-01-15")
        fragments = await agent.process(context)

        payload = fragments[0].payload
        assert payload["observation_date"] == "2026-01-15"

    @pytest.mark.asyncio
    async def test_observation_date_defaults_to_today(self) -> None:
        """If no date in metadata, should default to current date."""
        mock_client = AsyncMock(spec=YFinanceClient)
        mock_client.get_fundamentals.return_value = {
            "regularMarketPrice": 185.50,
            "marketCap": 2_900_000_000_000,
        }
        agent = _make_agent(mock_client)

        # Create context without date
        context = AgentContext(
            agent_id="PERCEPT-YFINANCE",
            trigger="schedule",
            context_window_hash="test_hash",
            timestamp=datetime.now(timezone.utc),
            metadata={"tickers": ["AAPL"]},  # No date key
        )
        fragments = await agent.process(context)

        payload = fragments[0].payload
        # Should be YYYY-MM-DD format
        assert len(payload["observation_date"]) == 10
        assert payload["observation_date"].count("-") == 2


# ===================================================================
# Validation Status Tests
# ===================================================================
class TestPerceptYFinanceValidation:
    """Test validation of Yahoo Finance data."""

    @pytest.mark.asyncio
    async def test_missing_core_fields_produces_quarantined_fragment(self) -> None:
        """Missing core fields should produce QUARANTINED fragment."""
        mock_client = AsyncMock(spec=YFinanceClient)
        # Missing both regularMarketPrice and marketCap
        mock_client.get_fundamentals.return_value = {
            "trailingPE": 29.5,
            "sector": "Technology",
        }
        agent = _make_agent(mock_client)

        context = _make_context(["AAPL"])
        fragments = await agent.process(context)

        assert len(fragments) == 1
        frag = fragments[0]
        assert frag.validation_status == ValidationStatus.QUARANTINED
        assert frag.entity == "AAPL"

    @pytest.mark.asyncio
    async def test_partial_core_fields_produces_partial_status(self) -> None:
        """Missing one core field should produce PARTIAL validation status."""
        mock_client = AsyncMock(spec=YFinanceClient)
        # Has regularMarketPrice but missing marketCap
        mock_client.get_fundamentals.return_value = {
            "regularMarketPrice": 185.50,
            "trailingPE": 29.5,
            "sector": "Technology",
        }
        agent = _make_agent(mock_client)

        context = _make_context(["AAPL"])
        fragments = await agent.process(context)

        assert len(fragments) == 1
        frag = fragments[0]
        assert frag.validation_status == ValidationStatus.PARTIAL

    @pytest.mark.asyncio
    async def test_all_core_fields_produces_valid_status(self) -> None:
        """Having all core fields should produce VALID status."""
        mock_client = AsyncMock(spec=YFinanceClient)
        mock_client.get_fundamentals.return_value = {
            "regularMarketPrice": 185.50,
            "marketCap": 2_900_000_000_000,
            "trailingPE": 29.5,  # Optional but good to have
        }
        agent = _make_agent(mock_client)

        context = _make_context(["AAPL"])
        fragments = await agent.process(context)

        assert fragments[0].validation_status == ValidationStatus.VALID

    @pytest.mark.asyncio
    async def test_empty_data_produces_quarantined(self) -> None:
        """Empty data dict should produce QUARANTINED status."""
        mock_client = AsyncMock(spec=YFinanceClient)
        mock_client.get_fundamentals.return_value = {}
        agent = _make_agent(mock_client)

        context = _make_context(["AAPL"])
        fragments = await agent.process(context)

        assert fragments[0].validation_status == ValidationStatus.QUARANTINED


# ===================================================================
# Error Handling Tests
# ===================================================================
class TestPerceptYFinanceErrorHandling:
    """Test error handling and graceful degradation."""

    @pytest.mark.asyncio
    async def test_api_failure_produces_quarantined_fragment(self) -> None:
        """API failure should produce QUARANTINED fragment, not crash."""
        mock_client = AsyncMock(spec=YFinanceClient)
        mock_client.get_fundamentals.side_effect = DataIngestionError(
            message="No fundamental data returned for AAPL"
        )
        agent = _make_agent(mock_client)

        context = _make_context(["AAPL"])
        fragments = await agent.process(context)

        # Should return a fragment, not raise
        assert len(fragments) == 1
        frag = fragments[0]
        assert frag.validation_status == ValidationStatus.QUARANTINED
        assert frag.entity == "AAPL"
        assert frag.payload["error"] is not None

    @pytest.mark.asyncio
    async def test_partial_failure_mixed_results(self) -> None:
        """If some tickers succeed and some fail, return mixed fragments."""
        mock_client = AsyncMock(spec=YFinanceClient)
        aapl_data = {
            "regularMarketPrice": 185.50,
            "marketCap": 2_900_000_000_000,
        }
        mock_client.get_fundamentals.side_effect = [
            aapl_data,  # AAPL succeeds
            DataIngestionError(message="MSFT failed"),  # MSFT fails
        ]
        agent = _make_agent(mock_client)

        context = _make_context(["AAPL", "MSFT"])
        fragments = await agent.process(context)

        assert len(fragments) == 2
        aapl_frag = next(f for f in fragments if f.entity == "AAPL")
        msft_frag = next(f for f in fragments if f.entity == "MSFT")
        assert aapl_frag.validation_status == ValidationStatus.VALID
        assert msft_frag.validation_status == ValidationStatus.QUARANTINED

    @pytest.mark.asyncio
    async def test_no_tickers_raises_agent_processing_error(self) -> None:
        """No tickers in metadata should raise AgentProcessingError."""
        mock_client = AsyncMock(spec=YFinanceClient)
        agent = _make_agent(mock_client)

        context = AgentContext(
            agent_id="PERCEPT-YFINANCE",
            trigger="schedule",
            context_window_hash="test_hash",
            timestamp=datetime.now(timezone.utc),
            metadata={"tickers": []},  # Empty
        )

        with pytest.raises(AgentProcessingError) as exc_info:
            await agent.process(context)
        assert "No tickers" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_missing_tickers_key_raises_agent_processing_error(self) -> None:
        """Missing tickers key should raise AgentProcessingError."""
        mock_client = AsyncMock(spec=YFinanceClient)
        agent = _make_agent(mock_client)

        context = AgentContext(
            agent_id="PERCEPT-YFINANCE",
            trigger="schedule",
            context_window_hash="test_hash",
            timestamp=datetime.now(timezone.utc),
            metadata={"date": "2026-02-09"},  # No tickers key
        )

        with pytest.raises(AgentProcessingError) as exc_info:
            await agent.process(context)
        assert "No tickers" in str(exc_info.value)


# ===================================================================
# Source Hash Tests
# ===================================================================
class TestPerceptYFinanceSourceHash:
    """Test content hashing for provenance."""

    @pytest.mark.asyncio
    async def test_same_data_same_source_hash(self) -> None:
        """Same data should produce the same source hash (deterministic)."""
        mock_client = AsyncMock(spec=YFinanceClient)
        data = {
            "regularMarketPrice": 185.50,
            "marketCap": 2_900_000_000_000,
            "trailingPE": 29.5,
        }
        mock_client.get_fundamentals.return_value = data
        agent = _make_agent(mock_client)

        context = _make_context(["AAPL"])
        fragments1 = await agent.process(context)

        mock_client.get_fundamentals.return_value = data
        fragments2 = await agent.process(context)

        assert fragments1[0].source_hash == fragments2[0].source_hash

    @pytest.mark.asyncio
    async def test_different_data_different_source_hash(self) -> None:
        """Different data should produce different source hash."""
        mock_client = AsyncMock(spec=YFinanceClient)
        agent = _make_agent(mock_client)

        data1 = {
            "regularMarketPrice": 185.50,
            "marketCap": 2_900_000_000_000,
        }
        mock_client.get_fundamentals.return_value = data1
        context = _make_context(["AAPL"])
        fragments1 = await agent.process(context)

        data2 = {
            "regularMarketPrice": 190.00,
            "marketCap": 2_950_000_000_000,
        }
        mock_client.get_fundamentals.return_value = data2
        fragments2 = await agent.process(context)

        assert fragments1[0].source_hash != fragments2[0].source_hash


# ===================================================================
# Health Status Tests
# ===================================================================
class TestPerceptYFinanceHealth:
    """Test health status reporting."""

    def test_health_healthy_on_init(self) -> None:
        """Agent should start in HEALTHY status."""
        mock_client = AsyncMock(spec=YFinanceClient)
        agent = _make_agent(mock_client)

        health = agent.get_health()
        assert health.status == AgentStatus.HEALTHY
        assert health.agent_id == "PERCEPT-YFINANCE"
        assert health.error_count_24h == 0
        assert health.last_run is None
        assert health.last_success is None

    @pytest.mark.asyncio
    async def test_health_updates_on_successful_run(self) -> None:
        """Health status should update after successful run."""
        mock_client = AsyncMock(spec=YFinanceClient)
        mock_client.get_fundamentals.return_value = {
            "regularMarketPrice": 185.50,
            "marketCap": 2_900_000_000_000,
        }
        agent = _make_agent(mock_client)

        context = _make_context(["AAPL"])
        await agent.process(context)

        health = agent.get_health()
        assert health.status == AgentStatus.HEALTHY
        assert health.last_run is not None
        assert health.last_success is not None

    @pytest.mark.asyncio
    async def test_health_degraded_after_3_errors(self) -> None:
        """Health status should be DEGRADED after 3+ errors."""
        mock_client = AsyncMock(spec=YFinanceClient)
        mock_client.get_fundamentals.side_effect = DataIngestionError(
            message="API error"
        )
        agent = _make_agent(mock_client)

        # Trigger 3 errors
        for _ in range(3):
            context = _make_context(["AAPL"])
            await agent.process(context)

        health = agent.get_health()
        assert health.status == AgentStatus.DEGRADED
        assert health.error_count_24h == 3

    @pytest.mark.asyncio
    async def test_health_unhealthy_after_10_errors(self) -> None:
        """Health status should be UNHEALTHY after 10+ errors."""
        mock_client = AsyncMock(spec=YFinanceClient)
        mock_client.get_fundamentals.side_effect = DataIngestionError(
            message="API error"
        )
        agent = _make_agent(mock_client)

        # Trigger 11 errors (each ticker is one error)
        for _ in range(11):
            context = _make_context(["AAPL"])
            await agent.process(context)

        health = agent.get_health()
        assert health.status == AgentStatus.UNHEALTHY
        assert health.error_count_24h >= 10

    @pytest.mark.asyncio
    async def test_health_success_after_errors_recovers(self) -> None:
        """Health status should recover after errors if new success occurs."""
        mock_client = AsyncMock(spec=YFinanceClient)
        agent = _make_agent(mock_client)

        # Trigger 4 errors
        mock_client.get_fundamentals.side_effect = DataIngestionError("error")
        for _ in range(4):
            context = _make_context(["AAPL"])
            await agent.process(context)

        health1 = agent.get_health()
        assert health1.status == AgentStatus.DEGRADED

        # Now succeed
        mock_client.get_fundamentals.side_effect = None
        mock_client.get_fundamentals.return_value = {
            "regularMarketPrice": 185.50,
            "marketCap": 2_900_000_000_000,
        }
        context = _make_context(["AAPL"])
        await agent.process(context)

        health2 = agent.get_health()
        assert health2.last_success is not None
        # Still shows all 4 errors, but last_success is now recent
        assert health2.error_count_24h == 4


# ===================================================================
# Fragment Metadata Tests
# ===================================================================
class TestPerceptYFinanceFragmentMetadata:
    """Test MarketStateFragment metadata fields."""

    @pytest.mark.asyncio
    async def test_fragment_has_valid_metadata(self) -> None:
        """Fragment should have all required metadata fields."""
        mock_client = AsyncMock(spec=YFinanceClient)
        mock_client.get_fundamentals.return_value = {
            "regularMarketPrice": 185.50,
            "marketCap": 2_900_000_000_000,
        }
        agent = _make_agent(mock_client)

        context = _make_context(["AAPL"], date="2026-02-09")
        fragments = await agent.process(context)

        frag = fragments[0]
        assert frag.fragment_id is not None
        assert isinstance(frag.fragment_id, UUID)
        assert frag.agent_id == "PERCEPT-YFINANCE"
        assert frag.timestamp is not None
        assert frag.source_timestamp is not None
        assert frag.entity == "AAPL"
        assert frag.data_type == DataType.YFINANCE_FUNDAMENTALS
        assert frag.schema_version == "1.0.0"
        assert isinstance(frag.source_hash, str)
        assert len(frag.source_hash) == 64  # SHA-256 hex = 64 chars
        assert frag.validation_status in [ValidationStatus.VALID, ValidationStatus.PARTIAL, ValidationStatus.QUARANTINED]
        assert isinstance(frag.payload, dict)

    @pytest.mark.asyncio
    async def test_fragment_timestamp_is_utc(self) -> None:
        """Fragment timestamps should be in UTC timezone."""
        mock_client = AsyncMock(spec=YFinanceClient)
        mock_client.get_fundamentals.return_value = {
            "regularMarketPrice": 185.50,
            "marketCap": 2_900_000_000_000,
        }
        agent = _make_agent(mock_client)

        context = _make_context(["AAPL"])
        fragments = await agent.process(context)

        frag = fragments[0]
        assert frag.timestamp.tzinfo is not None
        assert frag.source_timestamp.tzinfo is not None

    @pytest.mark.asyncio
    async def test_quarantined_fragment_has_error_payload(self) -> None:
        """Quarantined fragments should have error message in payload."""
        mock_client = AsyncMock(spec=YFinanceClient)
        mock_client.get_fundamentals.return_value = {}
        agent = _make_agent(mock_client)

        context = _make_context(["AAPL"])
        fragments = await agent.process(context)

        frag = fragments[0]
        assert frag.validation_status == ValidationStatus.QUARANTINED
        assert "error" in frag.payload
        assert frag.payload["ticker"] == "AAPL"
