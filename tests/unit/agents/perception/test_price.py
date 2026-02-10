"""Tests for PERCEPT-PRICE agent.

Uses mocked Polygon client to test the full Perception loop:
FETCH → VALIDATE → NORMALIZE → VERSION → STORE/ALERT

All tests run without real API calls.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch
from uuid import UUID

import pytest

from providence.agents.base import AgentContext, AgentStatus
from providence.agents.perception.price import PerceptPrice
from providence.exceptions import AgentProcessingError, DataIngestionError
from providence.infra.polygon_client import PolygonClient
from providence.schemas.enums import DataType, ValidationStatus
from tests.fixtures.polygon_responses import (
    daily_bars_aapl,
    daily_bars_empty,
    daily_bars_missing_fields,
    daily_bars_no_ohlcv,
    daily_bars_nvda,
    intraday_bars_aapl,
)


# ===================================================================
# Helpers
# ===================================================================
def _make_context(
    tickers: list[str],
    date: str = "2026-02-09",
    timeframe: str = "1D",
) -> AgentContext:
    """Create an AgentContext for PERCEPT-PRICE testing."""
    return AgentContext(
        agent_id="PERCEPT-PRICE",
        trigger="schedule",
        context_window_hash="test_hash",
        timestamp=datetime.now(timezone.utc),
        metadata={"tickers": tickers, "date": date, "timeframe": timeframe},
    )


def _make_agent(mock_client: AsyncMock) -> PerceptPrice:
    """Create a PerceptPrice agent with a mocked Polygon client."""
    # We pass the mock directly — it quacks like a PolygonClient
    return PerceptPrice(polygon_client=mock_client)


# ===================================================================
# Valid Data Tests
# ===================================================================
class TestPerceptPriceValidData:
    """Test PERCEPT-PRICE with valid Polygon responses."""

    @pytest.mark.asyncio
    async def test_single_ticker_produces_valid_fragment(self) -> None:
        """Valid AAPL data should produce a VALID MarketStateFragment."""
        mock_client = AsyncMock(spec=PolygonClient)
        mock_client.get_daily_bars.return_value = daily_bars_aapl()
        agent = _make_agent(mock_client)

        context = _make_context(["AAPL"])
        fragments = await agent.process(context)

        assert len(fragments) == 1
        frag = fragments[0]
        assert frag.entity == "AAPL"
        assert frag.data_type == DataType.PRICE_OHLCV
        assert frag.validation_status == ValidationStatus.VALID
        assert frag.agent_id == "PERCEPT-PRICE"
        assert isinstance(frag.fragment_id, UUID)

    @pytest.mark.asyncio
    async def test_payload_fields_correct(self) -> None:
        """Payload should contain normalized OHLCV fields."""
        mock_client = AsyncMock(spec=PolygonClient)
        mock_client.get_daily_bars.return_value = daily_bars_aapl()
        agent = _make_agent(mock_client)

        context = _make_context(["AAPL"])
        fragments = await agent.process(context)

        payload = fragments[0].payload
        assert payload["open"] == 185.50
        assert payload["high"] == 187.20
        assert payload["low"] == 184.80
        assert payload["close"] == 186.90
        assert payload["volume"] == 52_345_678
        assert payload["vwap"] == 186.15
        assert payload["num_trades"] == 623_456
        assert payload["timeframe"] == "1D"

    @pytest.mark.asyncio
    async def test_multiple_tickers(self) -> None:
        """Multiple tickers should produce one fragment each."""
        mock_client = AsyncMock(spec=PolygonClient)
        mock_client.get_daily_bars.side_effect = [
            daily_bars_aapl(),
            daily_bars_nvda(),
        ]
        agent = _make_agent(mock_client)

        context = _make_context(["AAPL", "NVDA"])
        fragments = await agent.process(context)

        assert len(fragments) == 2
        entities = {f.entity for f in fragments}
        assert entities == {"AAPL", "NVDA"}
        assert all(f.validation_status == ValidationStatus.VALID for f in fragments)

    @pytest.mark.asyncio
    async def test_source_timestamp_extracted(self) -> None:
        """Source timestamp should be extracted from Polygon 't' field."""
        mock_client = AsyncMock(spec=PolygonClient)
        mock_client.get_daily_bars.return_value = daily_bars_aapl()
        agent = _make_agent(mock_client)

        context = _make_context(["AAPL"])
        fragments = await agent.process(context)

        frag = fragments[0]
        assert frag.source_timestamp.tzinfo is not None
        # Polygon timestamp 1739059200000 = 2025-02-09 00:00 UTC
        assert frag.source_timestamp.year >= 2025


# ===================================================================
# Content Hash Determinism Tests
# ===================================================================
class TestPerceptPriceContentHash:
    """Test content hash determinism."""

    @pytest.mark.asyncio
    async def test_same_data_same_hash(self) -> None:
        """Same Polygon data should produce the same content hash."""
        mock_client = AsyncMock(spec=PolygonClient)
        mock_client.get_daily_bars.return_value = daily_bars_aapl()
        agent = _make_agent(mock_client)

        context = _make_context(["AAPL"])
        fragments1 = await agent.process(context)

        mock_client.get_daily_bars.return_value = daily_bars_aapl()
        fragments2 = await agent.process(context)

        assert fragments1[0].version == fragments2[0].version

    @pytest.mark.asyncio
    async def test_different_data_different_hash(self) -> None:
        """Different Polygon data should produce different hashes."""
        mock_client = AsyncMock(spec=PolygonClient)

        mock_client.get_daily_bars.return_value = daily_bars_aapl()
        agent = _make_agent(mock_client)
        context = _make_context(["AAPL"])
        fragments_aapl = await agent.process(context)

        mock_client.get_daily_bars.return_value = daily_bars_nvda()
        context = _make_context(["NVDA"])
        fragments_nvda = await agent.process(context)

        assert fragments_aapl[0].version != fragments_nvda[0].version

    @pytest.mark.asyncio
    async def test_source_hash_computed(self) -> None:
        """Source hash should be a valid SHA-256 of raw response."""
        mock_client = AsyncMock(spec=PolygonClient)
        mock_client.get_daily_bars.return_value = daily_bars_aapl()
        agent = _make_agent(mock_client)

        context = _make_context(["AAPL"])
        fragments = await agent.process(context)

        assert len(fragments[0].source_hash) == 64


# ===================================================================
# Validation & Quarantine Tests
# ===================================================================
class TestPerceptPriceValidation:
    """Test validation and quarantine behavior."""

    @pytest.mark.asyncio
    async def test_empty_results_quarantined(self) -> None:
        """Empty results (e.g., market holiday) should be QUARANTINED."""
        mock_client = AsyncMock(spec=PolygonClient)
        mock_client.get_daily_bars.return_value = daily_bars_empty()
        agent = _make_agent(mock_client)

        context = _make_context(["AAPL"])
        fragments = await agent.process(context)

        assert len(fragments) == 1
        assert fragments[0].validation_status == ValidationStatus.QUARANTINED

    @pytest.mark.asyncio
    async def test_missing_fields_partial(self) -> None:
        """Missing OHLCV fields should result in PARTIAL status."""
        mock_client = AsyncMock(spec=PolygonClient)
        mock_client.get_daily_bars.return_value = daily_bars_missing_fields()
        agent = _make_agent(mock_client)

        context = _make_context(["AAPL"])
        fragments = await agent.process(context)

        assert len(fragments) == 1
        assert fragments[0].validation_status == ValidationStatus.PARTIAL

    @pytest.mark.asyncio
    async def test_no_ohlcv_fields_quarantined(self) -> None:
        """Results with no OHLCV fields at all should be QUARANTINED."""
        mock_client = AsyncMock(spec=PolygonClient)
        mock_client.get_daily_bars.return_value = daily_bars_no_ohlcv()
        agent = _make_agent(mock_client)

        context = _make_context(["AAPL"])
        fragments = await agent.process(context)

        assert len(fragments) == 1
        assert fragments[0].validation_status == ValidationStatus.QUARANTINED


# ===================================================================
# Error Handling Tests
# ===================================================================
class TestPerceptPriceErrorHandling:
    """Test error handling and graceful degradation."""

    @pytest.mark.asyncio
    async def test_no_tickers_raises(self) -> None:
        """Missing tickers in metadata should raise AgentProcessingError."""
        mock_client = AsyncMock(spec=PolygonClient)
        agent = _make_agent(mock_client)

        context = _make_context([])
        with pytest.raises(AgentProcessingError, match="No tickers"):
            await agent.process(context)

    @pytest.mark.asyncio
    async def test_no_date_raises(self) -> None:
        """Missing date in metadata should raise AgentProcessingError."""
        mock_client = AsyncMock(spec=PolygonClient)
        agent = _make_agent(mock_client)

        context = AgentContext(
            agent_id="PERCEPT-PRICE",
            trigger="schedule",
            context_window_hash="test",
            timestamp=datetime.now(timezone.utc),
            metadata={"tickers": ["AAPL"]},
        )
        with pytest.raises(AgentProcessingError, match="No date"):
            await agent.process(context)

    @pytest.mark.asyncio
    async def test_api_error_produces_quarantined_fragment(self) -> None:
        """API failure should produce a quarantined fragment, not crash."""
        mock_client = AsyncMock(spec=PolygonClient)
        mock_client.get_daily_bars.side_effect = Exception("Connection refused")
        agent = _make_agent(mock_client)

        context = _make_context(["AAPL"])
        fragments = await agent.process(context)

        assert len(fragments) == 1
        assert fragments[0].validation_status == ValidationStatus.QUARANTINED
        assert "error" in fragments[0].payload

    @pytest.mark.asyncio
    async def test_partial_failure_produces_mixed_results(self) -> None:
        """One ticker failing shouldn't prevent others from succeeding."""
        mock_client = AsyncMock(spec=PolygonClient)
        mock_client.get_daily_bars.side_effect = [
            daily_bars_aapl(),                      # AAPL succeeds
            Exception("Timeout"),                    # NVDA fails
        ]
        agent = _make_agent(mock_client)

        context = _make_context(["AAPL", "NVDA"])
        fragments = await agent.process(context)

        assert len(fragments) == 2
        assert fragments[0].validation_status == ValidationStatus.VALID
        assert fragments[1].validation_status == ValidationStatus.QUARANTINED


# ===================================================================
# Health Status Tests
# ===================================================================
class TestPerceptPriceHealth:
    """Test health reporting."""

    @pytest.mark.asyncio
    async def test_healthy_after_success(self) -> None:
        """Agent should report HEALTHY after successful processing."""
        mock_client = AsyncMock(spec=PolygonClient)
        mock_client.get_daily_bars.return_value = daily_bars_aapl()
        agent = _make_agent(mock_client)

        context = _make_context(["AAPL"])
        await agent.process(context)

        health = agent.get_health()
        assert health.status == AgentStatus.HEALTHY
        assert health.last_run is not None
        assert health.last_success is not None

    @pytest.mark.asyncio
    async def test_degraded_after_errors(self) -> None:
        """Agent should report DEGRADED after multiple errors."""
        mock_client = AsyncMock(spec=PolygonClient)
        mock_client.get_daily_bars.side_effect = Exception("fail")
        agent = _make_agent(mock_client)

        # Process 4 failing tickers to cross the degraded threshold
        context = _make_context(["A", "B", "C", "D"])
        await agent.process(context)

        health = agent.get_health()
        assert health.status == AgentStatus.DEGRADED
        assert health.error_count_24h == 4

    def test_agent_properties(self) -> None:
        """Agent should have correct identity properties."""
        mock_client = AsyncMock(spec=PolygonClient)
        agent = _make_agent(mock_client)

        assert agent.agent_id == "PERCEPT-PRICE"
        assert agent.agent_type == "perception"
        assert agent.version == "1.0.0"
