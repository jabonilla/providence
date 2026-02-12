"""Tests for PERCEPT-OPTIONS agent.

Uses mocked Polygon client to test the full Perception loop:
FETCH → VALIDATE → NORMALIZE → VERSION → STORE/ALERT

All tests run without real API calls.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock
from uuid import UUID

import pytest

from providence.agents.base import AgentContext, AgentStatus
from providence.agents.perception.options import PerceptOptions
from providence.exceptions import AgentProcessingError
from providence.infra.polygon_client import PolygonClient
from providence.schemas.enums import DataType, ValidationStatus
from tests.fixtures.options_responses import (
    options_chain_aapl,
    options_chain_empty,
    options_chain_no_greeks,
)


# ===================================================================
# Helpers
# ===================================================================
def _make_context(
    tickers: list[str],
    expiration_date: str | None = None,
    contract_type: str | None = None,
    limit: int = 50,
) -> AgentContext:
    """Create an AgentContext for PERCEPT-OPTIONS testing."""
    metadata = {"tickers": tickers, "limit": limit}
    if expiration_date:
        metadata["expiration_date"] = expiration_date
    if contract_type:
        metadata["contract_type"] = contract_type

    return AgentContext(
        agent_id="PERCEPT-OPTIONS",
        trigger="schedule",
        context_window_hash="test_hash",
        timestamp=datetime.now(timezone.utc),
        metadata=metadata,
    )


def _make_agent(mock_client: AsyncMock) -> PerceptOptions:
    """Create a PerceptOptions agent with a mocked Polygon client."""
    # We pass the mock directly — it quacks like a PolygonClient
    return PerceptOptions(polygon_client=mock_client)


# ===================================================================
# Valid Data Tests
# ===================================================================
class TestPerceptOptionsValidData:
    """Test PERCEPT-OPTIONS with valid Polygon responses."""

    @pytest.mark.asyncio
    async def test_valid_options_produces_fragment(self) -> None:
        """Valid AAPL options data should produce a VALID MarketStateFragment."""
        mock_client = AsyncMock(spec=PolygonClient)
        mock_client.get_options_chain.return_value = options_chain_aapl()
        agent = _make_agent(mock_client)

        context = _make_context(["AAPL"])
        fragments = await agent.process(context)

        assert len(fragments) == 1
        frag = fragments[0]
        assert frag.entity == "AAPL"
        assert frag.data_type == DataType.OPTIONS_CHAIN
        assert frag.validation_status == ValidationStatus.VALID
        assert frag.agent_id == "PERCEPT-OPTIONS"
        assert isinstance(frag.fragment_id, UUID)

    @pytest.mark.asyncio
    async def test_payload_has_contracts(self) -> None:
        """Payload should contain contracts, contract_count, and put_call_ratio."""
        mock_client = AsyncMock(spec=PolygonClient)
        mock_client.get_options_chain.return_value = options_chain_aapl()
        agent = _make_agent(mock_client)

        context = _make_context(["AAPL"])
        fragments = await agent.process(context)

        payload = fragments[0].payload
        assert "contracts" in payload
        assert "contract_count" in payload
        assert "put_call_ratio" in payload
        assert isinstance(payload["contracts"], list)
        assert isinstance(payload["contract_count"], int)
        assert isinstance(payload["put_call_ratio"], float)

    @pytest.mark.asyncio
    async def test_put_call_ratio_computed(self) -> None:
        """With 2 calls and 2 puts, put_call_ratio should be 1.0."""
        mock_client = AsyncMock(spec=PolygonClient)
        mock_client.get_options_chain.return_value = options_chain_aapl()
        agent = _make_agent(mock_client)

        context = _make_context(["AAPL"])
        fragments = await agent.process(context)

        payload = fragments[0].payload
        # options_chain_aapl has 2 calls and 2 puts
        assert payload["put_call_ratio"] == 1.0
        assert payload["contract_count"] == 4

    @pytest.mark.asyncio
    async def test_multiple_tickers(self) -> None:
        """Multiple tickers should produce one fragment each."""
        mock_client = AsyncMock(spec=PolygonClient)
        mock_client.get_options_chain.side_effect = [
            options_chain_aapl(),
            options_chain_aapl(),
        ]
        agent = _make_agent(mock_client)

        context = _make_context(["AAPL", "MSFT"])
        fragments = await agent.process(context)

        assert len(fragments) == 2
        entities = {f.entity for f in fragments}
        assert entities == {"AAPL", "MSFT"}
        assert all(f.validation_status == ValidationStatus.VALID for f in fragments)

    @pytest.mark.asyncio
    async def test_source_timestamp_extracted(self) -> None:
        """Source timestamp should be set on fragment."""
        mock_client = AsyncMock(spec=PolygonClient)
        mock_client.get_options_chain.return_value = options_chain_aapl()
        agent = _make_agent(mock_client)

        context = _make_context(["AAPL"])
        fragments = await agent.process(context)

        frag = fragments[0]
        assert frag.source_timestamp.tzinfo is not None
        assert frag.source_timestamp is not None


# ===================================================================
# Content Hash Determinism Tests
# ===================================================================
class TestPerceptOptionsContentHash:
    """Test content hash determinism."""

    @pytest.mark.asyncio
    async def test_same_data_same_hash(self) -> None:
        """Same Polygon data should produce the same content hash when snapshot_time is fixed."""
        from unittest.mock import patch
        fixed_time = datetime(2026, 2, 9, 12, 0, 0, tzinfo=timezone.utc)

        mock_client = AsyncMock(spec=PolygonClient)
        mock_client.get_options_chain.return_value = options_chain_aapl()
        agent = _make_agent(mock_client)

        context = _make_context(["AAPL"])
        with patch("providence.agents.perception.options.datetime") as mock_dt:
            mock_dt.now.return_value = fixed_time
            mock_dt.strptime = datetime.strptime
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            fragments1 = await agent.process(context)

            mock_client.get_options_chain.return_value = options_chain_aapl()
            fragments2 = await agent.process(context)

        assert fragments1[0].version == fragments2[0].version

    @pytest.mark.asyncio
    async def test_different_data_different_hash(self) -> None:
        """Different Polygon data should produce different hashes."""
        mock_client = AsyncMock(spec=PolygonClient)

        mock_client.get_options_chain.return_value = options_chain_aapl()
        agent = _make_agent(mock_client)
        context = _make_context(["AAPL"])
        fragments_aapl = await agent.process(context)

        mock_client.get_options_chain.return_value = options_chain_empty()
        context = _make_context(["AAPL"])
        fragments_empty = await agent.process(context)

        assert fragments_aapl[0].version != fragments_empty[0].version

    @pytest.mark.asyncio
    async def test_source_hash_computed(self) -> None:
        """Source hash should be a valid SHA-256 of raw response."""
        mock_client = AsyncMock(spec=PolygonClient)
        mock_client.get_options_chain.return_value = options_chain_aapl()
        agent = _make_agent(mock_client)

        context = _make_context(["AAPL"])
        fragments = await agent.process(context)

        assert len(fragments[0].source_hash) == 64


# ===================================================================
# Validation & Quarantine Tests
# ===================================================================
class TestPerceptOptionsValidation:
    """Test validation and quarantine behavior."""

    @pytest.mark.asyncio
    async def test_empty_chain_quarantined(self) -> None:
        """Empty options chain should be QUARANTINED."""
        mock_client = AsyncMock(spec=PolygonClient)
        mock_client.get_options_chain.return_value = options_chain_empty()
        agent = _make_agent(mock_client)

        context = _make_context(["AAPL"])
        fragments = await agent.process(context)

        assert len(fragments) == 1
        assert fragments[0].validation_status == ValidationStatus.QUARANTINED

    @pytest.mark.asyncio
    async def test_no_greeks_partial(self) -> None:
        """Options without Greeks data should result in PARTIAL status."""
        mock_client = AsyncMock(spec=PolygonClient)
        mock_client.get_options_chain.return_value = options_chain_no_greeks()
        agent = _make_agent(mock_client)

        context = _make_context(["AAPL"])
        fragments = await agent.process(context)

        assert len(fragments) == 1
        assert fragments[0].validation_status == ValidationStatus.PARTIAL


# ===================================================================
# Error Handling Tests
# ===================================================================
class TestPerceptOptionsErrors:
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
    async def test_api_error_produces_quarantined(self) -> None:
        """API failure should produce a quarantined fragment, not crash."""
        mock_client = AsyncMock(spec=PolygonClient)
        mock_client.get_options_chain.side_effect = Exception("Connection refused")
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
        mock_client.get_options_chain.side_effect = [
            options_chain_aapl(),
            Exception("Timeout"),
        ]
        agent = _make_agent(mock_client)

        context = _make_context(["AAPL", "MSFT"])
        fragments = await agent.process(context)

        assert len(fragments) == 2
        assert fragments[0].validation_status == ValidationStatus.VALID
        assert fragments[1].validation_status == ValidationStatus.QUARANTINED


# ===================================================================
# Health Status Tests
# ===================================================================
class TestPerceptOptionsHealth:
    """Test health reporting."""

    @pytest.mark.asyncio
    async def test_healthy_after_success(self) -> None:
        """Agent should report HEALTHY after successful processing."""
        mock_client = AsyncMock(spec=PolygonClient)
        mock_client.get_options_chain.return_value = options_chain_aapl()
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
        mock_client.get_options_chain.side_effect = Exception("fail")
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

        assert agent.agent_id == "PERCEPT-OPTIONS"
        assert agent.agent_type == "perception"
        assert agent.version == "1.0.0"
