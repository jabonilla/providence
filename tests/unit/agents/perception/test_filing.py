"""Tests for PERCEPT-FILING agent.

Uses mocked EDGAR client to test the full Perception loop for
10-K, 10-Q, and 8-K filings.

All tests run without real API calls.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from providence.agents.base import AgentContext, AgentStatus
from providence.agents.perception.filing import PerceptFiling
from providence.exceptions import AgentProcessingError
from providence.infra.edgar_client import EdgarClient
from providence.schemas.enums import DataType, ValidationStatus
from tests.fixtures.edgar_responses import (
    filing_10k_aapl,
    filing_10q_aapl,
    filing_8k_aapl,
    filing_8k_no_event_type,
    filing_empty,
    filing_missing_xbrl,
)


# ===================================================================
# Helpers
# ===================================================================
def _make_context(
    tickers: list[str],
    filing_types: list[str] | None = None,
    cik_map: dict[str, str] | None = None,
    company_names: dict[str, str] | None = None,
) -> AgentContext:
    """Create an AgentContext for PERCEPT-FILING testing."""
    return AgentContext(
        agent_id="PERCEPT-FILING",
        trigger="schedule",
        context_window_hash="test_hash",
        timestamp=datetime.now(timezone.utc),
        metadata={
            "tickers": tickers,
            "filing_types": filing_types or ["10-Q"],
            "count": 1,
            "cik_map": cik_map or {"AAPL": "0000320193"},
            "company_names": company_names or {"AAPL": "Apple Inc."},
        },
    )


def _make_agent(mock_client: AsyncMock) -> PerceptFiling:
    """Create a PerceptFiling agent with a mocked EDGAR client."""
    return PerceptFiling(edgar_client=mock_client)


# ===================================================================
# 10-Q Tests
# ===================================================================
class TestPerceptFiling10Q:
    """Test PERCEPT-FILING with 10-Q filings."""

    @pytest.mark.asyncio
    async def test_valid_10q_produces_fragment(self) -> None:
        """Valid 10-Q should produce a VALID MarketStateFragment."""
        mock_client = AsyncMock(spec=EdgarClient)
        mock_client.get_recent_filings.return_value = [filing_10q_aapl()]
        agent = _make_agent(mock_client)

        context = _make_context(["AAPL"], ["10-Q"])
        fragments = await agent.process(context)

        assert len(fragments) == 1
        frag = fragments[0]
        assert frag.entity == "AAPL"
        assert frag.data_type == DataType.FILING_10Q
        assert frag.validation_status == ValidationStatus.VALID
        assert frag.agent_id == "PERCEPT-FILING"

    @pytest.mark.asyncio
    async def test_10q_payload_has_financial_metrics(self) -> None:
        """10-Q payload should contain extracted financial metrics."""
        mock_client = AsyncMock(spec=EdgarClient)
        mock_client.get_recent_filings.return_value = [filing_10q_aapl()]
        agent = _make_agent(mock_client)

        context = _make_context(["AAPL"], ["10-Q"])
        fragments = await agent.process(context)

        payload = fragments[0].payload
        assert payload["filing_type"] == "10-Q"
        assert payload["ticker"] == "AAPL"
        assert payload["revenue"] == 124_300_000_000
        assert payload["net_income"] == 33_900_000_000
        assert payload["eps"] == 2.18
        assert payload["total_assets"] == 352_600_000_000

    @pytest.mark.asyncio
    async def test_10q_computes_ratios(self) -> None:
        """10-Q should compute key financial ratios when data is available."""
        mock_client = AsyncMock(spec=EdgarClient)
        mock_client.get_recent_filings.return_value = [filing_10q_aapl()]
        agent = _make_agent(mock_client)

        context = _make_context(["AAPL"], ["10-Q"])
        fragments = await agent.process(context)

        ratios = fragments[0].payload.get("key_ratios", {})
        assert "current_ratio" in ratios
        assert ratios["current_ratio"] > 0


# ===================================================================
# 10-K Tests
# ===================================================================
class TestPerceptFiling10K:
    """Test PERCEPT-FILING with 10-K filings."""

    @pytest.mark.asyncio
    async def test_valid_10k_produces_fragment(self) -> None:
        """Valid 10-K should produce a FILING_10K fragment."""
        mock_client = AsyncMock(spec=EdgarClient)
        mock_client.get_recent_filings.return_value = [filing_10k_aapl()]
        agent = _make_agent(mock_client)

        context = _make_context(["AAPL"], ["10-K"])
        fragments = await agent.process(context)

        assert len(fragments) == 1
        assert fragments[0].data_type == DataType.FILING_10K
        assert fragments[0].validation_status == ValidationStatus.VALID


# ===================================================================
# 8-K Tests
# ===================================================================
class TestPerceptFiling8K:
    """Test PERCEPT-FILING with 8-K filings."""

    @pytest.mark.asyncio
    async def test_valid_8k_produces_fragment(self) -> None:
        """Valid 8-K should produce a FILING_8K fragment with event data."""
        mock_client = AsyncMock(spec=EdgarClient)
        mock_client.get_recent_filings.return_value = [filing_8k_aapl()]
        agent = _make_agent(mock_client)

        context = _make_context(["AAPL"], ["8-K"])
        fragments = await agent.process(context)

        assert len(fragments) == 1
        frag = fragments[0]
        assert frag.data_type == DataType.FILING_8K
        assert frag.validation_status == ValidationStatus.VALID

    @pytest.mark.asyncio
    async def test_8k_payload_has_event_fields(self) -> None:
        """8-K payload should contain event type and description."""
        mock_client = AsyncMock(spec=EdgarClient)
        mock_client.get_recent_filings.return_value = [filing_8k_aapl()]
        agent = _make_agent(mock_client)

        context = _make_context(["AAPL"], ["8-K"])
        fragments = await agent.process(context)

        payload = fragments[0].payload
        assert payload["filing_type"] == "8-K"
        assert payload["event_type"] == "CEO_TRANSITION"
        assert payload["material_impact"] is True
        assert "leadership" in payload["event_description"].lower()


# ===================================================================
# Validation Tests
# ===================================================================
class TestPerceptFilingValidation:
    """Test validation and quarantine behavior."""

    @pytest.mark.asyncio
    async def test_missing_xbrl_partial(self) -> None:
        """Filing with metadata but no XBRL tags → PARTIAL."""
        mock_client = AsyncMock(spec=EdgarClient)
        mock_client.get_recent_filings.return_value = [filing_missing_xbrl()]
        agent = _make_agent(mock_client)

        context = _make_context(["AAPL"], ["10-Q"])
        fragments = await agent.process(context)

        assert len(fragments) == 1
        assert fragments[0].validation_status == ValidationStatus.PARTIAL

    @pytest.mark.asyncio
    async def test_8k_no_event_type_partial(self) -> None:
        """8-K without event_type → PARTIAL."""
        mock_client = AsyncMock(spec=EdgarClient)
        mock_client.get_recent_filings.return_value = [filing_8k_no_event_type()]
        agent = _make_agent(mock_client)

        context = _make_context(["AAPL"], ["8-K"])
        fragments = await agent.process(context)

        assert fragments[0].validation_status == ValidationStatus.PARTIAL

    @pytest.mark.asyncio
    async def test_no_filings_found_quarantined(self) -> None:
        """No filings returned → QUARANTINED fragment."""
        mock_client = AsyncMock(spec=EdgarClient)
        mock_client.get_recent_filings.return_value = []
        agent = _make_agent(mock_client)

        context = _make_context(["AAPL"], ["10-Q"])
        fragments = await agent.process(context)

        assert len(fragments) == 1
        assert fragments[0].validation_status == ValidationStatus.QUARANTINED


# ===================================================================
# Content Hash Tests
# ===================================================================
class TestPerceptFilingContentHash:
    """Test content hash determinism."""

    @pytest.mark.asyncio
    async def test_same_data_same_hash(self) -> None:
        """Same filing data should produce the same content hash."""
        mock_client = AsyncMock(spec=EdgarClient)
        mock_client.get_recent_filings.return_value = [filing_10q_aapl()]
        agent = _make_agent(mock_client)

        context = _make_context(["AAPL"], ["10-Q"])
        f1 = await agent.process(context)

        mock_client.get_recent_filings.return_value = [filing_10q_aapl()]
        f2 = await agent.process(context)

        assert f1[0].version == f2[0].version


# ===================================================================
# Error Handling Tests
# ===================================================================
class TestPerceptFilingErrors:
    """Test error handling."""

    @pytest.mark.asyncio
    async def test_no_tickers_raises(self) -> None:
        """Missing tickers should raise AgentProcessingError."""
        mock_client = AsyncMock(spec=EdgarClient)
        agent = _make_agent(mock_client)

        context = _make_context([])
        with pytest.raises(AgentProcessingError, match="No tickers"):
            await agent.process(context)

    @pytest.mark.asyncio
    async def test_api_error_produces_quarantined(self) -> None:
        """API failure should produce quarantined fragment, not crash."""
        mock_client = AsyncMock(spec=EdgarClient)
        mock_client.get_recent_filings.side_effect = Exception("Connection refused")
        agent = _make_agent(mock_client)

        context = _make_context(["AAPL"], ["10-Q"])
        fragments = await agent.process(context)

        assert len(fragments) == 1
        assert fragments[0].validation_status == ValidationStatus.QUARANTINED

    @pytest.mark.asyncio
    async def test_multiple_filing_types(self) -> None:
        """Processing multiple filing types should produce fragments for each."""
        mock_client = AsyncMock(spec=EdgarClient)
        mock_client.get_recent_filings.side_effect = [
            [filing_10q_aapl()],
            [filing_8k_aapl()],
        ]
        agent = _make_agent(mock_client)

        context = _make_context(["AAPL"], ["10-Q", "8-K"])
        fragments = await agent.process(context)

        assert len(fragments) == 2
        data_types = {f.data_type for f in fragments}
        assert data_types == {DataType.FILING_10Q, DataType.FILING_8K}


# ===================================================================
# Health Tests
# ===================================================================
class TestPerceptFilingHealth:
    """Test health reporting."""

    def test_agent_properties(self) -> None:
        """Agent should have correct identity."""
        mock_client = AsyncMock(spec=EdgarClient)
        agent = _make_agent(mock_client)

        assert agent.agent_id == "PERCEPT-FILING"
        assert agent.agent_type == "perception"

    @pytest.mark.asyncio
    async def test_healthy_after_success(self) -> None:
        """Agent should report HEALTHY after success."""
        mock_client = AsyncMock(spec=EdgarClient)
        mock_client.get_recent_filings.return_value = [filing_10q_aapl()]
        agent = _make_agent(mock_client)

        context = _make_context(["AAPL"], ["10-Q"])
        await agent.process(context)

        health = agent.get_health()
        assert health.status == AgentStatus.HEALTHY
