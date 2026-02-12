"""Tests for PERCEPT-MACRO agent.

Uses mocked FRED client to test the full Perception loop:
FETCH → VALIDATE → NORMALIZE → VERSION → STORE/ALERT

All tests run without real API calls.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock
from uuid import UUID

import pytest

from providence.agents.base import AgentContext, AgentStatus
from providence.agents.perception.macro import PerceptMacro
from providence.exceptions import AgentProcessingError
from providence.schemas.enums import DataType, ValidationStatus
from tests.fixtures.fred_responses import (
    treasury_yields_response,
    treasury_yields_partial,
    treasury_yields_empty,
    gdp_observation,
    cpi_observation,
    missing_observation,
)


# ===================================================================
# Helpers
# ===================================================================
def _make_context(
    date: str = "2026-02-09",
    yield_curve: bool = True,
    indicators: list[dict] | None = None,
) -> AgentContext:
    """Create an AgentContext for PERCEPT-MACRO testing."""
    metadata = {"date": date, "yield_curve": yield_curve}
    if indicators is not None:
        metadata["indicators"] = indicators
    else:
        metadata["indicators"] = []

    return AgentContext(
        agent_id="PERCEPT-MACRO",
        trigger="schedule",
        context_window_hash="test_hash",
        timestamp=datetime.now(timezone.utc),
        metadata=metadata,
    )


def _make_agent(mock_client: AsyncMock) -> PerceptMacro:
    """Create a PerceptMacro agent with a mocked FRED client."""
    return PerceptMacro(fred_client=mock_client)


# ===================================================================
# Yield Curve Tests
# ===================================================================
class TestPerceptMacroYieldCurve:
    """Test PERCEPT-MACRO with valid yield curve data."""

    @pytest.mark.asyncio
    async def test_valid_yield_curve_produces_fragment(self) -> None:
        """Valid yield curve data should produce a VALID MarketStateFragment."""
        mock_client = AsyncMock()
        mock_client.get_treasury_yields.return_value = treasury_yields_response()
        agent = _make_agent(mock_client)

        context = _make_context(yield_curve=True)
        fragments = await agent.process(context)

        assert len(fragments) == 1
        frag = fragments[0]
        assert frag.entity is None
        assert frag.data_type == DataType.MACRO_YIELD_CURVE
        assert frag.validation_status == ValidationStatus.VALID
        assert frag.agent_id == "PERCEPT-MACRO"
        assert isinstance(frag.fragment_id, UUID)

    @pytest.mark.asyncio
    async def test_yield_curve_payload_has_tenors(self) -> None:
        """Payload should contain tenors, curve_date, and curve_source."""
        mock_client = AsyncMock()
        mock_client.get_treasury_yields.return_value = treasury_yields_response()
        agent = _make_agent(mock_client)

        context = _make_context(yield_curve=True)
        fragments = await agent.process(context)

        payload = fragments[0].payload
        assert "tenors" in payload
        assert "curve_date" in payload
        assert "curve_source" in payload
        assert payload["curve_source"] == "FRED"
        assert payload["curve_date"] == "2026-02-09"

    @pytest.mark.asyncio
    async def test_yield_curve_spreads_computed(self) -> None:
        """Payload should include computed spreads (2s10s and 3m10y)."""
        mock_client = AsyncMock()
        mock_client.get_treasury_yields.return_value = treasury_yields_response()
        agent = _make_agent(mock_client)

        context = _make_context(yield_curve=True)
        fragments = await agent.process(context)

        payload = fragments[0].payload
        assert "spread_2s10s" in payload
        assert "spread_3m10y" in payload

        # 2s10s = (10Y - 2Y) * 100 = (3.95 - 4.45) * 100 = -50
        assert payload["spread_2s10s"] == pytest.approx(-50.0, abs=0.1)

        # 3m10y = (10Y - 3M) * 100 = (3.95 - 5.28) * 100 = -133
        assert payload["spread_3m10y"] == pytest.approx(-133.0, abs=0.1)

    @pytest.mark.asyncio
    async def test_partial_yield_curve(self) -> None:
        """Partial yield curve (fewer than 6 tenors) should be PARTIAL."""
        mock_client = AsyncMock()
        mock_client.get_treasury_yields.return_value = treasury_yields_partial()
        agent = _make_agent(mock_client)

        context = _make_context(yield_curve=True)
        fragments = await agent.process(context)

        assert len(fragments) == 1
        assert fragments[0].validation_status == ValidationStatus.PARTIAL

    @pytest.mark.asyncio
    async def test_empty_yield_curve_quarantined(self) -> None:
        """Empty yield curve response should be QUARANTINED."""
        mock_client = AsyncMock()
        mock_client.get_treasury_yields.return_value = treasury_yields_empty()
        agent = _make_agent(mock_client)

        context = _make_context(yield_curve=True)
        fragments = await agent.process(context)

        assert len(fragments) == 1
        assert fragments[0].validation_status == ValidationStatus.QUARANTINED


# ===================================================================
# Economic Indicators Tests
# ===================================================================
class TestPerceptMacroIndicators:
    """Test PERCEPT-MACRO with economic indicator data."""

    @pytest.mark.asyncio
    async def test_indicator_produces_fragment(self) -> None:
        """Valid indicator data should produce a VALID MarketStateFragment."""
        mock_client = AsyncMock()
        mock_client.get_latest_observation.return_value = gdp_observation()[0]
        agent = _make_agent(mock_client)

        indicators = [{"series_id": "A191RA1Q225SBEA", "name": "GDP"}]
        context = _make_context(yield_curve=False, indicators=indicators)
        fragments = await agent.process(context)

        assert len(fragments) == 1
        frag = fragments[0]
        assert frag.data_type == DataType.MACRO_ECONOMIC
        assert frag.validation_status == ValidationStatus.VALID
        assert frag.agent_id == "PERCEPT-MACRO"

    @pytest.mark.asyncio
    async def test_indicator_payload_fields(self) -> None:
        """Payload should contain indicator, value, source_series_id, observation_date."""
        mock_client = AsyncMock()
        mock_client.get_latest_observation.return_value = gdp_observation()[0]
        agent = _make_agent(mock_client)

        indicators = [{"series_id": "A191RA1Q225SBEA", "name": "GDP"}]
        context = _make_context(yield_curve=False, indicators=indicators)
        fragments = await agent.process(context)

        payload = fragments[0].payload
        assert "indicator" in payload
        assert payload["indicator"] == "GDP"
        assert "value" in payload
        assert "source_series_id" in payload
        assert payload["source_series_id"] == "A191RA1Q225SBEA"
        assert "observation_date" in payload
        assert payload["observation_date"] == "2026-01-01"

    @pytest.mark.asyncio
    async def test_missing_indicator_quarantined(self) -> None:
        """Indicator with missing value (.) should be QUARANTINED."""
        mock_client = AsyncMock()
        mock_client.get_latest_observation.return_value = missing_observation()[0]
        agent = _make_agent(mock_client)

        indicators = [{"series_id": "TESTVAL", "name": "TEST"}]
        context = _make_context(yield_curve=False, indicators=indicators)
        fragments = await agent.process(context)

        assert len(fragments) == 1
        assert fragments[0].validation_status == ValidationStatus.QUARANTINED

    @pytest.mark.asyncio
    async def test_multiple_indicators(self) -> None:
        """Multiple indicators should produce one fragment each (plus yield curve)."""
        mock_client = AsyncMock()
        mock_client.get_treasury_yields.return_value = treasury_yields_response()
        mock_client.get_latest_observation.side_effect = [
            gdp_observation()[0],
            cpi_observation()[0],
        ]
        agent = _make_agent(mock_client)

        indicators = [
            {"series_id": "A191RA1Q225SBEA", "name": "GDP"},
            {"series_id": "CPIAUCSL", "name": "CPI"},
        ]
        context = _make_context(yield_curve=True, indicators=indicators)
        fragments = await agent.process(context)

        assert len(fragments) == 3  # 1 yield curve + 2 indicators
        assert fragments[0].data_type == DataType.MACRO_YIELD_CURVE
        assert fragments[1].data_type == DataType.MACRO_ECONOMIC
        assert fragments[2].data_type == DataType.MACRO_ECONOMIC


# ===================================================================
# Error Handling Tests
# ===================================================================
class TestPerceptMacroErrors:
    """Test error handling and graceful degradation."""

    @pytest.mark.asyncio
    async def test_no_date_raises(self) -> None:
        """Missing date in metadata should raise AgentProcessingError."""
        mock_client = AsyncMock()
        agent = _make_agent(mock_client)

        context = AgentContext(
            agent_id="PERCEPT-MACRO",
            trigger="schedule",
            context_window_hash="test",
            timestamp=datetime.now(timezone.utc),
            metadata={"yield_curve": True},
        )
        with pytest.raises(AgentProcessingError, match="No date"):
            await agent.process(context)

    @pytest.mark.asyncio
    async def test_api_error_produces_quarantined(self) -> None:
        """API failure should produce a quarantined fragment, not crash."""
        mock_client = AsyncMock()
        mock_client.get_treasury_yields.side_effect = Exception("API connection failed")
        agent = _make_agent(mock_client)

        context = _make_context(yield_curve=True)
        fragments = await agent.process(context)

        assert len(fragments) == 1
        assert fragments[0].validation_status == ValidationStatus.QUARANTINED
        assert "error" in fragments[0].payload

    @pytest.mark.asyncio
    async def test_partial_failure_with_indicators(self) -> None:
        """One indicator failing shouldn't prevent others from succeeding."""
        mock_client = AsyncMock()
        mock_client.get_treasury_yields.return_value = treasury_yields_response()
        mock_client.get_latest_observation.side_effect = [
            gdp_observation()[0],  # First indicator succeeds
            Exception("Timeout"),  # Second indicator fails
        ]
        agent = _make_agent(mock_client)

        indicators = [
            {"series_id": "A191RA1Q225SBEA", "name": "GDP"},
            {"series_id": "CPIAUCSL", "name": "CPI"},
        ]
        context = _make_context(yield_curve=True, indicators=indicators)
        fragments = await agent.process(context)

        assert len(fragments) == 3
        assert fragments[0].validation_status == ValidationStatus.VALID  # Yield curve
        assert fragments[1].validation_status == ValidationStatus.VALID  # GDP
        assert fragments[2].validation_status == ValidationStatus.QUARANTINED  # CPI


# ===================================================================
# Health Status Tests
# ===================================================================
class TestPerceptMacroHealth:
    """Test health reporting."""

    @pytest.mark.asyncio
    async def test_healthy_after_success(self) -> None:
        """Agent should report HEALTHY after successful processing."""
        mock_client = AsyncMock()
        mock_client.get_treasury_yields.return_value = treasury_yields_response()
        agent = _make_agent(mock_client)

        context = _make_context(yield_curve=True)
        await agent.process(context)

        health = agent.get_health()
        assert health.status == AgentStatus.HEALTHY
        assert health.last_run is not None
        assert health.last_success is not None

    def test_agent_properties(self) -> None:
        """Agent should have correct identity properties."""
        mock_client = AsyncMock()
        agent = _make_agent(mock_client)

        assert agent.agent_id == "PERCEPT-MACRO"
        assert agent.agent_type == "perception"
        assert agent.version == "1.0.0"
