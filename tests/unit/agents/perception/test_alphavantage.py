"""Tests for PERCEPT-ALPHAVANTAGE agent.

Uses mocked AlphaVantageClient to test the full Perception loop:
FETCH → VALIDATE → NORMALIZE → VERSION → STORE/ALERT

All tests run without real API calls.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock
from uuid import UUID

import pytest

from providence.agents.base import AgentContext, AgentStatus
from providence.agents.perception.alphavantage import PerceptAlphaVantage
from providence.exceptions import AgentProcessingError
from providence.infra.alphavantage_client import AlphaVantageClient
from providence.schemas.enums import DataType, ValidationStatus


# ===================================================================
# Helpers
# ===================================================================
def _make_context(
    tickers: list[str],
    date: str = "2025-04-16",
    max_quarters: int = 4,
) -> AgentContext:
    """Create an AgentContext for PERCEPT-ALPHAVANTAGE testing."""
    return AgentContext(
        agent_id="PERCEPT-ALPHAVANTAGE",
        trigger="schedule",
        context_window_hash="test_hash",
        timestamp=datetime.now(timezone.utc),
        metadata={
            "tickers": tickers,
            "date": date,
            "max_quarters": max_quarters,
        },
    )


def _make_agent(mock_client: AsyncMock) -> PerceptAlphaVantage:
    """Create a PerceptAlphaVantage agent with a mocked AlphaVantageClient."""
    return PerceptAlphaVantage(alphavantage_client=mock_client)


# ===================================================================
# Test Data Fixtures
# ===================================================================
def _earnings_response_aapl(quarters: int = 4) -> dict:
    """Create mock earnings response for AAPL."""
    quarters_data = [
        {
            "fiscalDateEnding": "2025-03-31",
            "reportedEPS": "1.50",
            "estimatedEPS": "1.45",
            "surprise": "0.05",
            "surprisePercentage": "3.45",
            "reportedDate": "2025-04-15",
        },
        {
            "fiscalDateEnding": "2024-12-31",
            "reportedEPS": "1.30",
            "estimatedEPS": "1.28",
            "surprise": "0.02",
            "surprisePercentage": "1.56",
            "reportedDate": "2025-01-15",
        },
        {
            "fiscalDateEnding": "2024-09-30",
            "reportedEPS": "1.20",
            "estimatedEPS": "1.18",
            "surprise": "0.02",
            "surprisePercentage": "1.69",
            "reportedDate": "2024-10-15",
        },
        {
            "fiscalDateEnding": "2024-06-30",
            "reportedEPS": "1.10",
            "estimatedEPS": "1.08",
            "surprise": "0.02",
            "surprisePercentage": "1.85",
            "reportedDate": "2024-07-15",
        },
    ]
    return {"quarterlyEarnings": quarters_data[:quarters]}


def _income_statement_response_aapl() -> dict:
    """Create mock income statement response for AAPL."""
    return {
        "annualReports": [
            {
                "fiscalDateEnding": "2024-12-31",
                "totalRevenue": "391035000000",
                "grossProfit": "114337000000",
                "ebitda": "130541000000",
                "netIncome": "93736000000",
            }
        ],
        "quarterlyReports": [
            {
                "fiscalDateEnding": "2025-03-31",
                "totalRevenue": "90000000000",
                "grossProfit": "28000000000",
                "ebitda": "32000000000",
                "netIncome": "24000000000",
            },
            {
                "fiscalDateEnding": "2024-12-31",
                "totalRevenue": "123500000000",
                "grossProfit": "40000000000",
                "ebitda": "45000000000",
                "netIncome": "35000000000",
            },
            {
                "fiscalDateEnding": "2024-09-30",
                "totalRevenue": "88500000000",
                "grossProfit": "27000000000",
                "ebitda": "31000000000",
                "netIncome": "22500000000",
            },
            {
                "fiscalDateEnding": "2024-06-30",
                "totalRevenue": "89000000000",
                "grossProfit": "28000000000",
                "ebitda": "32000000000",
                "netIncome": "21500000000",
            },
        ]
    }


# ===================================================================
# Valid Data Tests
# ===================================================================
class TestPerceptAlphaVantageValidData:
    """Test PERCEPT-ALPHAVANTAGE with valid earnings data."""

    @pytest.mark.asyncio
    async def test_single_ticker_valid_earnings_produces_fragments(self) -> None:
        """Valid AAPL earnings data should produce VALID MarketStateFragments."""
        mock_client = AsyncMock(spec=AlphaVantageClient)
        mock_client.get_earnings.return_value = _earnings_response_aapl(2)
        mock_client.get_income_statement.return_value = _income_statement_response_aapl()
        agent = _make_agent(mock_client)

        context = _make_context(["AAPL"], max_quarters=2)
        fragments = await agent.process(context)

        assert len(fragments) == 2
        for frag in fragments:
            assert frag.entity == "AAPL"
            assert frag.data_type == DataType.ALPHAVANTAGE_EARNINGS
            assert frag.validation_status == ValidationStatus.VALID
            assert frag.agent_id == "PERCEPT-ALPHAVANTAGE"
            assert isinstance(frag.fragment_id, UUID)

    @pytest.mark.asyncio
    async def test_payload_has_required_fields(self) -> None:
        """Payload should contain all earnings fields."""
        mock_client = AsyncMock(spec=AlphaVantageClient)
        mock_client.get_earnings.return_value = _earnings_response_aapl(1)
        mock_client.get_income_statement.return_value = _income_statement_response_aapl()
        agent = _make_agent(mock_client)

        context = _make_context(["AAPL"], max_quarters=1)
        fragments = await agent.process(context)

        payload = fragments[0].payload
        assert payload["ticker"] == "AAPL"
        assert payload["fiscal_date_ending"] == "2025-03-31"
        assert payload["reported_eps"] == 1.50
        assert payload["estimated_eps"] == 1.45
        assert payload["surprise"] == 0.05
        assert payload["surprise_pct"] == 3.45
        assert payload["reported_date"] == "2025-04-15"
        assert payload["observation_date"] == "2025-04-16"

    @pytest.mark.asyncio
    async def test_income_statement_data_merged_into_payload(self) -> None:
        """Income statement data (revenue, gross_profit, etc.) merged into payload."""
        mock_client = AsyncMock(spec=AlphaVantageClient)
        mock_client.get_earnings.return_value = _earnings_response_aapl(1)
        mock_client.get_income_statement.return_value = _income_statement_response_aapl()
        agent = _make_agent(mock_client)

        context = _make_context(["AAPL"], max_quarters=1)
        fragments = await agent.process(context)

        payload = fragments[0].payload
        # First quarter matches fiscal_date_ending = "2025-03-31"
        assert payload["revenue"] == 90000000000.0
        assert payload["gross_profit"] == 28000000000.0
        assert payload["ebitda"] == 32000000000.0
        assert payload["net_income"] == 24000000000.0

    @pytest.mark.asyncio
    async def test_max_quarters_limits_fragments(self) -> None:
        """max_quarters parameter limits number of fragments produced."""
        mock_client = AsyncMock(spec=AlphaVantageClient)
        # Provide 4 quarters but limit to 2
        mock_client.get_earnings.return_value = _earnings_response_aapl(4)
        mock_client.get_income_statement.return_value = _income_statement_response_aapl()
        agent = _make_agent(mock_client)

        context = _make_context(["AAPL"], max_quarters=2)
        fragments = await agent.process(context)

        assert len(fragments) == 2
        fiscal_dates = [f.payload["fiscal_date_ending"] for f in fragments]
        assert fiscal_dates == ["2025-03-31", "2024-12-31"]

    @pytest.mark.asyncio
    async def test_multiple_tickers(self) -> None:
        """Multiple tickers produce separate fragments."""
        mock_client = AsyncMock(spec=AlphaVantageClient)
        mock_client.get_earnings.side_effect = [
            _earnings_response_aapl(2),
            _earnings_response_aapl(2),
        ]
        mock_client.get_income_statement.side_effect = [
            _income_statement_response_aapl(),
            _income_statement_response_aapl(),
        ]
        agent = _make_agent(mock_client)

        context = _make_context(["AAPL", "MSFT"], max_quarters=2)
        fragments = await agent.process(context)

        assert len(fragments) == 4  # 2 quarters × 2 tickers
        entities = {f.entity for f in fragments}
        assert entities == {"AAPL", "MSFT"}

    @pytest.mark.asyncio
    async def test_source_hash_computed(self) -> None:
        """Source hash should be computed from raw earnings data."""
        mock_client = AsyncMock(spec=AlphaVantageClient)
        mock_client.get_earnings.return_value = _earnings_response_aapl(1)
        mock_client.get_income_statement.return_value = _income_statement_response_aapl()
        agent = _make_agent(mock_client)

        context = _make_context(["AAPL"], max_quarters=1)
        fragments = await agent.process(context)

        frag = fragments[0]
        assert frag.source_hash is not None
        assert len(frag.source_hash) == 64  # SHA-256 hex digest


# ===================================================================
# Validation Tests
# ===================================================================
class TestPerceptAlphaVantageValidation:
    """Test validation logic."""

    @pytest.mark.asyncio
    async def test_empty_quarterly_earnings_produces_quarantined_fragment(self) -> None:
        """Empty quarterlyEarnings list produces QUARANTINED fragment."""
        mock_client = AsyncMock(spec=AlphaVantageClient)
        mock_client.get_earnings.return_value = {"quarterlyEarnings": []}
        mock_client.get_income_statement.return_value = {"quarterlyReports": []}
        agent = _make_agent(mock_client)

        context = _make_context(["AAPL"])
        fragments = await agent.process(context)

        assert len(fragments) == 1
        assert fragments[0].validation_status == ValidationStatus.QUARANTINED
        assert fragments[0].entity == "AAPL"

    @pytest.mark.asyncio
    async def test_missing_fiscal_date_ending_produces_quarantined(self) -> None:
        """Quarter missing fiscalDateEnding produces QUARANTINED status."""
        mock_client = AsyncMock(spec=AlphaVantageClient)
        mock_client.get_earnings.return_value = {
            "quarterlyEarnings": [
                {
                    # Missing fiscalDateEnding
                    "reportedEPS": "1.50",
                }
            ]
        }
        mock_client.get_income_statement.return_value = {"quarterlyReports": []}
        agent = _make_agent(mock_client)

        context = _make_context(["AAPL"])
        fragments = await agent.process(context)

        assert len(fragments) == 1
        assert fragments[0].validation_status == ValidationStatus.QUARANTINED

    @pytest.mark.asyncio
    async def test_missing_reported_eps_produces_quarantined(self) -> None:
        """Quarter missing reportedEPS produces QUARANTINED status."""
        mock_client = AsyncMock(spec=AlphaVantageClient)
        mock_client.get_earnings.return_value = {
            "quarterlyEarnings": [
                {
                    "fiscalDateEnding": "2025-03-31",
                    # Missing reportedEPS
                }
            ]
        }
        mock_client.get_income_statement.return_value = {"quarterlyReports": []}
        agent = _make_agent(mock_client)

        context = _make_context(["AAPL"])
        fragments = await agent.process(context)

        assert len(fragments) == 1
        assert fragments[0].validation_status == ValidationStatus.QUARANTINED

    @pytest.mark.asyncio
    async def test_partial_data_produces_partial_status(self) -> None:
        """Quarter with one required field produces PARTIAL status."""
        mock_client = AsyncMock(spec=AlphaVantageClient)
        mock_client.get_earnings.return_value = {
            "quarterlyEarnings": [
                {
                    "fiscalDateEnding": "2025-03-31",
                    # Missing reportedEPS, but has fiscalDateEnding
                }
            ]
        }
        mock_client.get_income_statement.return_value = {"quarterlyReports": []}
        agent = _make_agent(mock_client)

        context = _make_context(["AAPL"])
        fragments = await agent.process(context)

        assert len(fragments) == 1
        assert fragments[0].validation_status == ValidationStatus.PARTIAL


# ===================================================================
# Error Handling Tests
# ===================================================================
class TestPerceptAlphaVantageErrorHandling:
    """Test error handling and robustness."""

    @pytest.mark.asyncio
    async def test_no_tickers_raises_agent_processing_error(self) -> None:
        """Empty tickers list raises AgentProcessingError."""
        mock_client = AsyncMock(spec=AlphaVantageClient)
        agent = _make_agent(mock_client)

        context = _make_context([])
        with pytest.raises(AgentProcessingError, match="No tickers"):
            await agent.process(context)

    @pytest.mark.asyncio
    async def test_api_failure_produces_quarantined_fragment(self) -> None:
        """API failure produces QUARANTINED fragment instead of raising."""
        from providence.exceptions import ExternalAPIError

        mock_client = AsyncMock(spec=AlphaVantageClient)
        mock_client.get_earnings.side_effect = ExternalAPIError(
            message="API error",
            service="alphavantage",
        )
        agent = _make_agent(mock_client)

        context = _make_context(["AAPL"])
        fragments = await agent.process(context)

        assert len(fragments) == 1
        assert fragments[0].validation_status == ValidationStatus.QUARANTINED
        assert fragments[0].entity == "AAPL"
        assert "error" in fragments[0].payload

    @pytest.mark.asyncio
    async def test_income_statement_failure_continues_with_earnings(self) -> None:
        """If income statement fetch fails, agent continues with earnings-only data."""
        from providence.exceptions import ExternalAPIError

        mock_client = AsyncMock(spec=AlphaVantageClient)
        mock_client.get_earnings.return_value = _earnings_response_aapl(1)
        mock_client.get_income_statement.side_effect = ExternalAPIError(
            message="Income statement unavailable",
            service="alphavantage",
        )
        agent = _make_agent(mock_client)

        context = _make_context(["AAPL"], max_quarters=1)
        fragments = await agent.process(context)

        # Should still produce a VALID fragment (earnings are present)
        assert len(fragments) == 1
        assert fragments[0].validation_status == ValidationStatus.VALID
        # Income statement fields should be None
        assert fragments[0].payload["revenue"] is None
        assert fragments[0].payload["gross_profit"] is None

    @pytest.mark.asyncio
    async def test_mixed_valid_and_invalid_tickers(self) -> None:
        """One ticker with data and one without produces mixed results."""
        from providence.exceptions import ExternalAPIError

        mock_client = AsyncMock(spec=AlphaVantageClient)
        mock_client.get_earnings.side_effect = [
            _earnings_response_aapl(1),  # AAPL succeeds
            ExternalAPIError(message="Invalid ticker", service="alphavantage"),  # MSFT fails
        ]
        mock_client.get_income_statement.side_effect = [
            _income_statement_response_aapl(),
            None,  # Won't be called if earnings fails
        ]
        agent = _make_agent(mock_client)

        context = _make_context(["AAPL", "MSFT"], max_quarters=1)
        fragments = await agent.process(context)

        # Should have 2 fragments: 1 VALID for AAPL, 1 QUARANTINED for MSFT
        assert len(fragments) == 2
        aapl_frag = [f for f in fragments if f.entity == "AAPL"][0]
        msft_frag = [f for f in fragments if f.entity == "MSFT"][0]
        assert aapl_frag.validation_status == ValidationStatus.VALID
        assert msft_frag.validation_status == ValidationStatus.QUARANTINED

    @pytest.mark.asyncio
    async def test_error_count_incremented_on_failure(self) -> None:
        """_error_count_24h incremented when ticker processing fails."""
        from providence.exceptions import ExternalAPIError

        mock_client = AsyncMock(spec=AlphaVantageClient)
        mock_client.get_earnings.side_effect = ExternalAPIError(
            message="API error",
            service="alphavantage",
        )
        agent = _make_agent(mock_client)

        context = _make_context(["AAPL", "MSFT"])
        fragments = await agent.process(context)

        assert agent._error_count_24h == 2


# ===================================================================
# Numeric Field Handling Tests
# ===================================================================
class TestPerceptAlphaVantageNumericHandling:
    """Test numeric field parsing."""

    @pytest.mark.asyncio
    async def test_string_numeric_fields_converted_to_float(self) -> None:
        """String numeric fields (e.g. "1.50") converted to float."""
        mock_client = AsyncMock(spec=AlphaVantageClient)
        mock_client.get_earnings.return_value = {
            "quarterlyEarnings": [
                {
                    "fiscalDateEnding": "2025-03-31",
                    "reportedEPS": "1.50",  # String
                    "estimatedEPS": "1.45",  # String
                    "surprise": "0.05",  # String
                }
            ]
        }
        mock_client.get_income_statement.return_value = {"quarterlyReports": []}
        agent = _make_agent(mock_client)

        context = _make_context(["AAPL"], max_quarters=1)
        fragments = await agent.process(context)

        payload = fragments[0].payload
        assert payload["reported_eps"] == 1.50
        assert isinstance(payload["reported_eps"], float)
        assert payload["estimated_eps"] == 1.45
        assert isinstance(payload["estimated_eps"], float)
        assert payload["surprise"] == 0.05
        assert isinstance(payload["surprise"], float)

    @pytest.mark.asyncio
    async def test_none_and_empty_string_fields_handled(self) -> None:
        """None and empty string fields converted to None."""
        mock_client = AsyncMock(spec=AlphaVantageClient)
        mock_client.get_earnings.return_value = {
            "quarterlyEarnings": [
                {
                    "fiscalDateEnding": "2025-03-31",
                    "reportedEPS": "1.50",
                    "estimatedEPS": None,  # None
                    "surprise": "",  # Empty string
                }
            ]
        }
        mock_client.get_income_statement.return_value = {"quarterlyReports": []}
        agent = _make_agent(mock_client)

        context = _make_context(["AAPL"], max_quarters=1)
        fragments = await agent.process(context)

        payload = fragments[0].payload
        assert payload["reported_eps"] == 1.50
        assert payload["estimated_eps"] is None
        assert payload["surprise"] is None

    @pytest.mark.asyncio
    async def test_invalid_numeric_fields_become_none(self) -> None:
        """Fields with invalid numeric values become None."""
        mock_client = AsyncMock(spec=AlphaVantageClient)
        mock_client.get_earnings.return_value = {
            "quarterlyEarnings": [
                {
                    "fiscalDateEnding": "2025-03-31",
                    "reportedEPS": "1.50",
                    "estimatedEPS": "invalid_number",  # Non-numeric
                    "surprise": "N/A",
                }
            ]
        }
        mock_client.get_income_statement.return_value = {"quarterlyReports": []}
        agent = _make_agent(mock_client)

        context = _make_context(["AAPL"], max_quarters=1)
        fragments = await agent.process(context)

        payload = fragments[0].payload
        assert payload["estimated_eps"] is None
        assert payload["surprise"] is None


# ===================================================================
# Health Status Tests
# ===================================================================
class TestPerceptAlphaVantageHealth:
    """Test health status reporting."""

    def test_health_status_healthy_on_success(self) -> None:
        """Health status HEALTHY with no errors."""
        mock_client = AsyncMock(spec=AlphaVantageClient)
        agent = _make_agent(mock_client)
        agent._error_count_24h = 0

        health = agent.get_health()

        assert health.agent_id == "PERCEPT-ALPHAVANTAGE"
        assert health.status == AgentStatus.HEALTHY
        assert health.error_count_24h == 0

    def test_health_status_degraded_with_moderate_errors(self) -> None:
        """Health status DEGRADED with 4-10 errors."""
        mock_client = AsyncMock(spec=AlphaVantageClient)
        agent = _make_agent(mock_client)
        agent._error_count_24h = 5

        health = agent.get_health()

        assert health.status == AgentStatus.DEGRADED
        assert health.error_count_24h == 5

    def test_health_status_unhealthy_with_many_errors(self) -> None:
        """Health status UNHEALTHY with >10 errors."""
        mock_client = AsyncMock(spec=AlphaVantageClient)
        agent = _make_agent(mock_client)
        agent._error_count_24h = 15

        health = agent.get_health()

        assert health.status == AgentStatus.UNHEALTHY
        assert health.error_count_24h == 15

    @pytest.mark.asyncio
    async def test_last_success_timestamp_updated_on_valid_fragment(self) -> None:
        """_last_success timestamp updated when processing produces VALID fragments."""
        mock_client = AsyncMock(spec=AlphaVantageClient)
        mock_client.get_earnings.return_value = _earnings_response_aapl(1)
        mock_client.get_income_statement.return_value = _income_statement_response_aapl()
        agent = _make_agent(mock_client)

        assert agent._last_success is None

        context = _make_context(["AAPL"], max_quarters=1)
        await agent.process(context)

        assert agent._last_success is not None
