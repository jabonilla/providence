"""Tests for PERCEPT-CDS agent.

Uses mocked FRED client to test the full Perception loop:
FETCH → VALIDATE → NORMALIZE → VERSION → STORE/ALERT

All tests run without real API calls.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock
from uuid import UUID

import pytest

from providence.agents.base import AgentContext, AgentStatus
from providence.agents.perception.cds import PerceptCds
from providence.exceptions import AgentProcessingError
from providence.schemas.enums import DataType, ValidationStatus
from tests.fixtures.fred_responses import (
    cds_observations,
    cds_observations_single,
    cds_observations_empty,
    cds_observations_missing_value,
)


# ===================================================================
# Helpers
# ===================================================================
def _make_context(
    date: str = "2026-02-09",
    entities: list[dict] | None = None,
) -> AgentContext:
    """Create an AgentContext for PERCEPT-CDS testing."""
    if entities is None:
        entities = [{"name": "IG_CDX", "series_id": "BAMLC0A0CM", "tenor": "5Y"}]

    return AgentContext(
        agent_id="PERCEPT-CDS",
        trigger="schedule",
        context_window_hash="test_hash",
        timestamp=datetime.now(timezone.utc),
        metadata={"date": date, "entities": entities},
    )


def _make_agent(mock_client: AsyncMock) -> PerceptCds:
    """Create a PerceptCds agent with a mocked FRED client."""
    return PerceptCds(fred_client=mock_client)


# ===================================================================
# Valid Data Tests
# ===================================================================
class TestPerceptCdsValidData:
    """Test PERCEPT-CDS with valid CDS observations."""

    @pytest.mark.asyncio
    async def test_valid_cds_produces_fragment(self) -> None:
        """Valid CDS data should produce a VALID MarketStateFragment."""
        mock_client = AsyncMock()
        mock_client.get_series_observations.return_value = cds_observations()
        agent = _make_agent(mock_client)

        context = _make_context()
        fragments = await agent.process(context)

        assert len(fragments) == 1
        frag = fragments[0]
        assert frag.entity == "IG_CDX"
        assert frag.data_type == DataType.MACRO_CDS
        assert frag.validation_status == ValidationStatus.VALID
        assert frag.agent_id == "PERCEPT-CDS"
        assert isinstance(frag.fragment_id, UUID)

    @pytest.mark.asyncio
    async def test_payload_has_spread_data(self) -> None:
        """Payload should contain spread_bps, reference_entity, and tenor."""
        mock_client = AsyncMock()
        mock_client.get_series_observations.return_value = cds_observations()
        agent = _make_agent(mock_client)

        context = _make_context()
        fragments = await agent.process(context)

        payload = fragments[0].payload
        assert "spread_bps" in payload
        assert "reference_entity" in payload
        assert "tenor" in payload
        assert payload["reference_entity"] == "IG_CDX"
        assert payload["tenor"] == "5Y"
        assert payload["spread_bps"] == 125.5

    @pytest.mark.asyncio
    async def test_spread_change_computed(self) -> None:
        """With 2 observations, spread_change_bps should be current - previous."""
        mock_client = AsyncMock()
        mock_client.get_series_observations.return_value = cds_observations()
        agent = _make_agent(mock_client)

        context = _make_context()
        fragments = await agent.process(context)

        payload = fragments[0].payload
        assert "spread_change_bps" in payload
        # spread_change_bps = 125.5 - 122.3 = 3.2
        assert payload["spread_change_bps"] == pytest.approx(3.2, abs=0.01)
        assert payload["previous_spread_bps"] == 122.3

    @pytest.mark.asyncio
    async def test_single_observation_no_change(self) -> None:
        """Single observation should not have spread_change_bps."""
        mock_client = AsyncMock()
        mock_client.get_series_observations.return_value = cds_observations_single()
        agent = _make_agent(mock_client)

        context = _make_context()
        fragments = await agent.process(context)

        payload = fragments[0].payload
        assert payload["spread_change_bps"] is None
        assert payload["previous_spread_bps"] is None
        assert payload["spread_bps"] == 125.5


# ===================================================================
# Validation Tests
# ===================================================================
class TestPerceptCdsValidation:
    """Test validation and quarantine behavior."""

    @pytest.mark.asyncio
    async def test_empty_observations_quarantined(self) -> None:
        """Empty observations should be QUARANTINED."""
        mock_client = AsyncMock()
        mock_client.get_series_observations.return_value = cds_observations_empty()
        agent = _make_agent(mock_client)

        context = _make_context()
        fragments = await agent.process(context)

        assert len(fragments) == 1
        assert fragments[0].validation_status == ValidationStatus.QUARANTINED

    @pytest.mark.asyncio
    async def test_missing_value_quarantined(self) -> None:
        """Observations with missing value (.) should be QUARANTINED."""
        mock_client = AsyncMock()
        mock_client.get_series_observations.return_value = cds_observations_missing_value()
        agent = _make_agent(mock_client)

        context = _make_context()
        fragments = await agent.process(context)

        assert len(fragments) == 1
        assert fragments[0].validation_status == ValidationStatus.QUARANTINED


# ===================================================================
# Error Handling Tests
# ===================================================================
class TestPerceptCdsErrors:
    """Test error handling and graceful degradation."""

    @pytest.mark.asyncio
    async def test_no_date_raises(self) -> None:
        """Missing date in metadata should raise AgentProcessingError."""
        mock_client = AsyncMock()
        agent = _make_agent(mock_client)

        context = AgentContext(
            agent_id="PERCEPT-CDS",
            trigger="schedule",
            context_window_hash="test",
            timestamp=datetime.now(timezone.utc),
            metadata={"entities": [{"name": "IG_CDX", "series_id": "BAMLC0A0CM"}]},
        )
        with pytest.raises(AgentProcessingError, match="No date"):
            await agent.process(context)

    @pytest.mark.asyncio
    async def test_no_entities_raises(self) -> None:
        """Missing entities in metadata should raise AgentProcessingError."""
        mock_client = AsyncMock()
        agent = _make_agent(mock_client)

        context = AgentContext(
            agent_id="PERCEPT-CDS",
            trigger="schedule",
            context_window_hash="test",
            timestamp=datetime.now(timezone.utc),
            metadata={"date": "2026-02-09"},
        )
        with pytest.raises(AgentProcessingError, match="No entities"):
            await agent.process(context)

    @pytest.mark.asyncio
    async def test_api_error_produces_quarantined(self) -> None:
        """API failure should produce a quarantined fragment, not crash."""
        mock_client = AsyncMock()
        mock_client.get_series_observations.side_effect = Exception("API timeout")
        agent = _make_agent(mock_client)

        context = _make_context()
        fragments = await agent.process(context)

        assert len(fragments) == 1
        assert fragments[0].validation_status == ValidationStatus.QUARANTINED
        assert "error" in fragments[0].payload

    @pytest.mark.asyncio
    async def test_partial_failure_multiple_entities(self) -> None:
        """One entity failing shouldn't prevent others from succeeding."""
        mock_client = AsyncMock()
        mock_client.get_series_observations.side_effect = [
            cds_observations(),  # IG_CDX succeeds
            Exception("Series not found"),  # HY_CDX fails
        ]
        agent = _make_agent(mock_client)

        entities = [
            {"name": "IG_CDX", "series_id": "BAMLC0A0CM", "tenor": "5Y"},
            {"name": "HY_CDX", "series_id": "BAMLH0A0HYM2", "tenor": "5Y"},
        ]
        context = _make_context(entities=entities)
        fragments = await agent.process(context)

        assert len(fragments) == 2
        assert fragments[0].validation_status == ValidationStatus.VALID
        assert fragments[1].validation_status == ValidationStatus.QUARANTINED


# ===================================================================
# Health Status Tests
# ===================================================================
class TestPerceptCdsHealth:
    """Test health reporting."""

    @pytest.mark.asyncio
    async def test_healthy_after_success(self) -> None:
        """Agent should report HEALTHY after successful processing."""
        mock_client = AsyncMock()
        mock_client.get_series_observations.return_value = cds_observations()
        agent = _make_agent(mock_client)

        context = _make_context()
        await agent.process(context)

        health = agent.get_health()
        assert health.status == AgentStatus.HEALTHY
        assert health.last_run is not None
        assert health.last_success is not None

    def test_agent_properties(self) -> None:
        """Agent should have correct identity properties."""
        mock_client = AsyncMock()
        agent = _make_agent(mock_client)

        assert agent.agent_id == "PERCEPT-CDS"
        assert agent.agent_type == "perception"
        assert agent.version == "1.0.0"
