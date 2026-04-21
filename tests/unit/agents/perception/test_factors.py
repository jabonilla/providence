"""Tests for PERCEPT-FACTORS agent at providence/agents/perception/factors.py.

Tests the Fama-French factor data ingestion pipeline with mocked client.
Validates fragment creation, payload normalization, validation status handling,
momentum fallback behavior, and error handling.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from providence.agents.base import AgentContext, AgentStatus
from providence.agents.perception.factors import PerceptFactors
from providence.exceptions import AgentProcessingError, ExternalAPIError
from providence.infra.famafrench_client import FamaFrenchClient
from providence.schemas.enums import DataType, ValidationStatus
from providence.schemas.market_state import MarketStateFragment


def _make_context(
    date: str = "2026-02-09",
    history_days: int = 7,
) -> AgentContext:
    """Helper to create AgentContext with factor data expectations."""
    return AgentContext(
        run_id=uuid4(),
        timestamp=datetime.now(timezone.utc),
        metadata={
            "date": date,
            "history_days": history_days,
        },
        fragments=[],
        beliefs=[],
    )


class TestPerceptFactorsInit:
    """Test PerceptFactors initialization."""

    def test_init_with_client(self):
        """Agent should initialize with FamaFrenchClient."""
        mock_client = AsyncMock(spec=FamaFrenchClient)
        agent = PerceptFactors(mock_client)

        assert agent.agent_id == "PERCEPT-FACTORS"
        assert agent.agent_type == "perception"
        assert agent.version == "1.0.0"
        assert agent._ff is mock_client
        assert agent._last_run is None
        assert agent._last_success is None
        assert agent._error_count_24h == 0


class TestPerceptFactorsProcess:
    """Test the main process method."""

    @pytest.mark.asyncio
    async def test_process_produces_fragments(self):
        """Should produce one fragment per trading day."""
        mock_client = AsyncMock(spec=FamaFrenchClient)
        mock_client.get_five_factors_daily.return_value = [
            {
                "date": "2026-02-01",
                "mkt_rf": 1.23,
                "smb": 0.12,
                "hml": 0.34,
                "rmw": 0.10,
                "cma": 0.08,
                "rf": 0.02,
            },
            {
                "date": "2026-02-02",
                "mkt_rf": 0.45,
                "smb": -0.05,
                "hml": 0.56,
                "rmw": 0.20,
                "cma": -0.12,
                "rf": 0.02,
            },
        ]
        mock_client.get_momentum_daily.return_value = [
            {"date": "2026-02-01", "mom": 1.5},
            {"date": "2026-02-02", "mom": 2.1},
        ]

        agent = PerceptFactors(mock_client)
        context = _make_context(date="2026-02-02", history_days=2)

        fragments = await agent.process(context)

        assert len(fragments) == 2
        assert all(isinstance(f, MarketStateFragment) for f in fragments)

    @pytest.mark.asyncio
    async def test_fragment_has_correct_data_type(self):
        """All fragments should have FACTOR_RETURNS data_type."""
        mock_client = AsyncMock(spec=FamaFrenchClient)
        mock_client.get_five_factors_daily.return_value = [
            {
                "date": "2026-02-01",
                "mkt_rf": 1.0,
                "smb": 0.1,
                "hml": 0.2,
                "rmw": 0.05,
                "cma": 0.03,
                "rf": 0.01,
            }
        ]
        mock_client.get_momentum_daily.return_value = [
            {"date": "2026-02-01", "mom": 1.0}
        ]

        agent = PerceptFactors(mock_client)
        context = _make_context()

        fragments = await agent.process(context)

        assert all(f.data_type == DataType.FACTOR_RETURNS for f in fragments)

    @pytest.mark.asyncio
    async def test_fragment_entity_is_market(self):
        """All fragments should have entity='MARKET'."""
        mock_client = AsyncMock(spec=FamaFrenchClient)
        mock_client.get_five_factors_daily.return_value = [
            {
                "date": "2026-02-01",
                "mkt_rf": 1.0,
                "smb": 0.1,
                "hml": 0.2,
                "rmw": 0.05,
                "cma": 0.03,
                "rf": 0.01,
            }
        ]
        mock_client.get_momentum_daily.return_value = [
            {"date": "2026-02-01", "mom": 1.0}
        ]

        agent = PerceptFactors(mock_client)
        context = _make_context()

        fragments = await agent.process(context)

        assert all(f.entity == "MARKET" for f in fragments)

    @pytest.mark.asyncio
    async def test_payload_has_all_factor_fields(self):
        """Fragment payload should contain all factor fields."""
        mock_client = AsyncMock(spec=FamaFrenchClient)
        mock_client.get_five_factors_daily.return_value = [
            {
                "date": "2026-02-01",
                "mkt_rf": 1.23,
                "smb": 0.12,
                "hml": 0.34,
                "rmw": 0.10,
                "cma": 0.08,
                "rf": 0.02,
            }
        ]
        mock_client.get_momentum_daily.return_value = [
            {"date": "2026-02-01", "mom": 1.5}
        ]

        agent = PerceptFactors(mock_client)
        context = _make_context()

        fragments = await agent.process(context)

        payload = fragments[0].payload
        assert "date" in payload
        assert "mkt_rf" in payload
        assert "smb" in payload
        assert "hml" in payload
        assert "rmw" in payload
        assert "cma" in payload
        assert "rf" in payload
        assert "mom" in payload

    @pytest.mark.asyncio
    async def test_momentum_data_merged_correctly(self):
        """Momentum data should be merged by date into factor payloads."""
        mock_client = AsyncMock(spec=FamaFrenchClient)
        mock_client.get_five_factors_daily.return_value = [
            {
                "date": "2026-02-01",
                "mkt_rf": 1.0,
                "smb": 0.1,
                "hml": 0.2,
                "rmw": 0.05,
                "cma": 0.03,
                "rf": 0.01,
            },
            {
                "date": "2026-02-02",
                "mkt_rf": 0.5,
                "smb": -0.1,
                "hml": 0.3,
                "rmw": 0.02,
                "cma": -0.01,
                "rf": 0.01,
            },
        ]
        mock_client.get_momentum_daily.return_value = [
            {"date": "2026-02-01", "mom": 1.5},
            {"date": "2026-02-02", "mom": 2.1},
        ]

        agent = PerceptFactors(mock_client)
        context = _make_context()

        fragments = await agent.process(context)

        assert fragments[0].payload["mom"] == 1.5
        assert fragments[1].payload["mom"] == 2.1

    @pytest.mark.asyncio
    async def test_momentum_not_found_sets_none(self):
        """If momentum data not found for a date, mom should be None."""
        mock_client = AsyncMock(spec=FamaFrenchClient)
        mock_client.get_five_factors_daily.return_value = [
            {
                "date": "2026-02-01",
                "mkt_rf": 1.0,
                "smb": 0.1,
                "hml": 0.2,
                "rmw": 0.05,
                "cma": 0.03,
                "rf": 0.01,
            }
        ]
        # No momentum data for 2026-02-01
        mock_client.get_momentum_daily.return_value = [
            {"date": "2026-02-02", "mom": 2.1}
        ]

        agent = PerceptFactors(mock_client)
        context = _make_context()

        fragments = await agent.process(context)

        assert fragments[0].payload["mom"] is None

    @pytest.mark.asyncio
    async def test_failed_momentum_fetch_still_produces_fragments(self):
        """If momentum fetch fails, should still produce fragments from 5-factor data."""
        mock_client = AsyncMock(spec=FamaFrenchClient)
        mock_client.get_five_factors_daily.return_value = [
            {
                "date": "2026-02-01",
                "mkt_rf": 1.0,
                "smb": 0.1,
                "hml": 0.2,
                "rmw": 0.05,
                "cma": 0.03,
                "rf": 0.01,
            }
        ]
        # Momentum fetch fails
        mock_client.get_momentum_daily.side_effect = ExternalAPIError(
            message="Momentum API error",
            service="fama_french",
        )

        agent = PerceptFactors(mock_client)
        context = _make_context()

        fragments = await agent.process(context)

        # Should still produce fragment
        assert len(fragments) == 1
        assert fragments[0].validation_status in (ValidationStatus.VALID, ValidationStatus.PARTIAL)
        assert fragments[0].payload["mom"] is None

    @pytest.mark.asyncio
    async def test_api_failure_produces_quarantined_fragment(self):
        """On API failure, should produce one quarantined fragment."""
        mock_client = AsyncMock(spec=FamaFrenchClient)
        mock_client.get_five_factors_daily.side_effect = ExternalAPIError(
            message="Kenneth French API error",
            service="fama_french",
        )

        agent = PerceptFactors(mock_client)
        context = _make_context()

        fragments = await agent.process(context)

        assert len(fragments) == 1
        assert fragments[0].validation_status == ValidationStatus.QUARANTINED
        assert "error" in fragments[0].payload

    @pytest.mark.asyncio
    async def test_validation_status_valid(self):
        """Fragment should be VALID when all required factors present."""
        mock_client = AsyncMock(spec=FamaFrenchClient)
        mock_client.get_five_factors_daily.return_value = [
            {
                "date": "2026-02-01",
                "mkt_rf": 1.0,
                "smb": 0.1,
                "hml": 0.2,
                "rmw": 0.05,
                "cma": 0.03,
                "rf": 0.01,
            }
        ]
        mock_client.get_momentum_daily.return_value = []

        agent = PerceptFactors(mock_client)
        context = _make_context()

        fragments = await agent.process(context)

        # Should be VALID because required keys (date, mkt_rf, etc.) present
        assert fragments[0].validation_status == ValidationStatus.VALID

    @pytest.mark.asyncio
    async def test_validation_status_partial_missing_optional(self):
        """Fragment should be PARTIAL when optional factors missing."""
        mock_client = AsyncMock(spec=FamaFrenchClient)
        # Missing RMW and CMA (not strictly required)
        mock_client.get_five_factors_daily.return_value = [
            {
                "date": "2026-02-01",
                "mkt_rf": 1.0,
                "smb": 0.1,
                "hml": 0.2,
                # Missing rmw, cma
                "rf": 0.01,
            }
        ]
        mock_client.get_momentum_daily.return_value = []

        agent = PerceptFactors(mock_client)
        context = _make_context()

        fragments = await agent.process(context)

        assert fragments[0].validation_status == ValidationStatus.PARTIAL

    @pytest.mark.asyncio
    async def test_validation_status_quarantined_missing_mkt_rf(self):
        """Fragment should be QUARANTINED when mkt_rf missing."""
        mock_client = AsyncMock(spec=FamaFrenchClient)
        # Missing mkt_rf (critical)
        mock_client.get_five_factors_daily.return_value = [
            {
                "date": "2026-02-01",
                # Missing mkt_rf
                "smb": 0.1,
                "hml": 0.2,
                "rf": 0.01,
            }
        ]
        mock_client.get_momentum_daily.return_value = []

        agent = PerceptFactors(mock_client)
        context = _make_context()

        fragments = await agent.process(context)

        assert fragments[0].validation_status == ValidationStatus.QUARANTINED

    @pytest.mark.asyncio
    async def test_default_date_is_today(self):
        """If date not in context, should default to today."""
        mock_client = AsyncMock(spec=FamaFrenchClient)
        mock_client.get_five_factors_daily.return_value = []
        mock_client.get_momentum_daily.return_value = []

        agent = PerceptFactors(mock_client)
        context = AgentContext(
            run_id=uuid4(),
            timestamp=datetime.now(timezone.utc),
            metadata={"history_days": 1},  # No date key
            fragments=[],
            beliefs=[],
        )

        await agent.process(context)

        # Should have called with today's date
        call_args = mock_client.get_five_factors_daily.call_args
        end_date = call_args[0][1]
        # End date should be close to today
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        assert end_date == today

    @pytest.mark.asyncio
    async def test_health_status_healthy_on_success(self):
        """Health should be HEALTHY after successful run."""
        mock_client = AsyncMock(spec=FamaFrenchClient)
        mock_client.get_five_factors_daily.return_value = [
            {
                "date": "2026-02-01",
                "mkt_rf": 1.0,
                "smb": 0.1,
                "hml": 0.2,
                "rmw": 0.05,
                "cma": 0.03,
                "rf": 0.01,
            }
        ]
        mock_client.get_momentum_daily.return_value = []

        agent = PerceptFactors(mock_client)
        context = _make_context()

        await agent.process(context)

        health = agent.get_health()
        assert health.status == AgentStatus.HEALTHY
        assert health.agent_id == "PERCEPT-FACTORS"
        assert health.error_count_24h == 0

    @pytest.mark.asyncio
    async def test_health_status_degraded_on_multiple_errors(self):
        """Health should be DEGRADED after 3+ errors."""
        mock_client = AsyncMock(spec=FamaFrenchClient)

        agent = PerceptFactors(mock_client)
        agent._error_count_24h = 4

        health = agent.get_health()
        assert health.status == AgentStatus.DEGRADED

    @pytest.mark.asyncio
    async def test_health_status_unhealthy_on_many_errors(self):
        """Health should be UNHEALTHY after 10+ errors."""
        mock_client = AsyncMock(spec=FamaFrenchClient)

        agent = PerceptFactors(mock_client)
        agent._error_count_24h = 11

        health = agent.get_health()
        assert health.status == AgentStatus.UNHEALTHY


class TestPerceptFactorsIntegration:
    """Integration-level tests."""

    @pytest.mark.asyncio
    async def test_multiple_runs_track_success(self):
        """Multiple successful runs should update _last_success."""
        mock_client = AsyncMock(spec=FamaFrenchClient)
        mock_client.get_five_factors_daily.return_value = [
            {
                "date": "2026-02-01",
                "mkt_rf": 1.0,
                "smb": 0.1,
                "hml": 0.2,
                "rmw": 0.05,
                "cma": 0.03,
                "rf": 0.01,
            }
        ]
        mock_client.get_momentum_daily.return_value = []

        agent = PerceptFactors(mock_client)

        context1 = _make_context(date="2026-02-01")
        await agent.process(context1)

        assert agent._last_success is not None
        first_success = agent._last_success

        context2 = _make_context(date="2026-02-02")
        await agent.process(context2)

        assert agent._last_success > first_success

    @pytest.mark.asyncio
    async def test_error_tracking_across_runs(self):
        """Error count should accumulate across runs."""
        mock_client = AsyncMock(spec=FamaFrenchClient)
        mock_client.get_five_factors_daily.side_effect = ExternalAPIError(
            message="API error",
            service="fama_french",
        )

        agent = PerceptFactors(mock_client)

        context1 = _make_context()
        await agent.process(context1)
        assert agent._error_count_24h == 1

        context2 = _make_context()
        await agent.process(context2)
        assert agent._error_count_24h == 2
