"""Tests for PERCEPT-NEWS agent.

Uses mocked Polygon client to test the full Perception loop:
FETCH → VALIDATE → NORMALIZE → VERSION → STORE/ALERT

All tests run without real API calls.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch
from uuid import UUID

import pytest

from providence.agents.base import AgentContext, AgentStatus
from providence.agents.perception.news import PerceptNews
from providence.exceptions import AgentProcessingError
from providence.infra.polygon_client import PolygonClient
from providence.schemas.enums import DataType, ValidationStatus
from tests.fixtures.news_responses import (
    news_aapl,
    news_empty,
    news_no_insights,
    news_multi_ticker,
)


# ===================================================================
# Helpers
# ===================================================================
def _make_context(
    tickers: list[str],
    limit: int = 10,
) -> AgentContext:
    """Create an AgentContext for PERCEPT-NEWS testing."""
    return AgentContext(
        agent_id="PERCEPT-NEWS",
        trigger="schedule",
        context_window_hash="test_hash",
        timestamp=datetime.now(timezone.utc),
        metadata={"tickers": tickers, "limit": limit},
    )


def _make_agent(mock_client: AsyncMock) -> PerceptNews:
    """Create a PerceptNews agent with a mocked Polygon client."""
    # We pass the mock directly — it quacks like a PolygonClient
    return PerceptNews(polygon_client=mock_client)


# ===================================================================
# Valid Data Tests
# ===================================================================
class TestPerceptNewsValidData:
    """Test PERCEPT-NEWS with valid Polygon responses."""

    @pytest.mark.asyncio
    async def test_valid_news_produces_fragment(self) -> None:
        """Valid AAPL data should produce a VALID MarketStateFragment."""
        mock_client = AsyncMock(spec=PolygonClient)
        mock_client.get_ticker_news.return_value = news_aapl()
        agent = _make_agent(mock_client)

        context = _make_context(["AAPL"])
        fragments = await agent.process(context)

        assert len(fragments) == 1
        frag = fragments[0]
        assert frag.entity == "AAPL"
        assert frag.data_type == DataType.SENTIMENT_NEWS
        assert frag.validation_status == ValidationStatus.VALID
        assert frag.agent_id == "PERCEPT-NEWS"
        assert isinstance(frag.fragment_id, UUID)

    @pytest.mark.asyncio
    async def test_payload_has_articles(self) -> None:
        """Payload should contain articles, article_count, and avg_sentiment."""
        mock_client = AsyncMock(spec=PolygonClient)
        mock_client.get_ticker_news.return_value = news_aapl()
        agent = _make_agent(mock_client)

        context = _make_context(["AAPL"])
        fragments = await agent.process(context)

        payload = fragments[0].payload
        assert "articles" in payload
        assert "article_count" in payload
        assert "avg_sentiment" in payload
        assert isinstance(payload["articles"], list)
        assert payload["article_count"] == 3
        # Average of [0.5, -0.5, 0.0] = 0.0
        assert payload["avg_sentiment"] == 0.0

    @pytest.mark.asyncio
    async def test_multiple_tickers(self) -> None:
        """Multiple tickers should produce one fragment each."""
        mock_client = AsyncMock(spec=PolygonClient)
        mock_client.get_ticker_news.side_effect = [
            news_aapl(),
            news_multi_ticker(),
        ]
        agent = _make_agent(mock_client)

        context = _make_context(["AAPL", "MSFT"])
        fragments = await agent.process(context)

        assert len(fragments) == 2
        entities = {f.entity for f in fragments}
        assert entities == {"AAPL", "MSFT"}
        assert all(f.validation_status == ValidationStatus.VALID for f in fragments)


# ===================================================================
# Validation & Quarantine Tests
# ===================================================================
class TestPerceptNewsValidation:
    """Test validation and quarantine behavior."""

    @pytest.mark.asyncio
    async def test_empty_results_quarantined(self) -> None:
        """Empty results (e.g., newly listed ticker) should be QUARANTINED."""
        mock_client = AsyncMock(spec=PolygonClient)
        mock_client.get_ticker_news.return_value = news_empty()
        agent = _make_agent(mock_client)

        context = _make_context(["AAPL"])
        fragments = await agent.process(context)

        assert len(fragments) == 1
        assert fragments[0].validation_status == ValidationStatus.QUARANTINED

    @pytest.mark.asyncio
    async def test_no_insights_partial(self) -> None:
        """Articles lacking sentiment insights should be PARTIAL."""
        mock_client = AsyncMock(spec=PolygonClient)
        mock_client.get_ticker_news.return_value = news_no_insights()
        agent = _make_agent(mock_client)

        context = _make_context(["AAPL"])
        fragments = await agent.process(context)

        assert len(fragments) == 1
        assert fragments[0].validation_status == ValidationStatus.PARTIAL


# ===================================================================
# Error Handling Tests
# ===================================================================
class TestPerceptNewsErrorHandling:
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
    async def test_api_error_produces_quarantined_fragment(self) -> None:
        """API failure should produce a quarantined fragment, not crash."""
        mock_client = AsyncMock(spec=PolygonClient)
        mock_client.get_ticker_news.side_effect = Exception("Connection refused")
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
        mock_client.get_ticker_news.side_effect = [
            news_aapl(),                      # AAPL succeeds
            Exception("Timeout"),             # MSFT fails
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
class TestPerceptNewsHealth:
    """Test health reporting."""

    def test_agent_properties(self) -> None:
        """Agent should have correct identity properties."""
        mock_client = AsyncMock(spec=PolygonClient)
        agent = _make_agent(mock_client)

        assert agent.agent_id == "PERCEPT-NEWS"
        assert agent.agent_type == "perception"
        assert agent.version == "1.0.0"

    @pytest.mark.asyncio
    async def test_healthy_after_success(self) -> None:
        """Agent should report HEALTHY after successful processing."""
        mock_client = AsyncMock(spec=PolygonClient)
        mock_client.get_ticker_news.return_value = news_aapl()
        agent = _make_agent(mock_client)

        context = _make_context(["AAPL"])
        await agent.process(context)

        health = agent.get_health()
        assert health.status == AgentStatus.HEALTHY
        assert health.last_run is not None
        assert health.last_success is not None
