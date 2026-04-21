"""Tests for PERCEPT-FUNDFLOW agent at providence/agents/perception/fundflow.py.

Tests the Plaid fund flow data ingestion pipeline with mocked client.
Validates fragment creation, transaction aggregation, account anonymization,
pagination handling, and error handling.
"""

import hashlib
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from providence.agents.base import AgentContext, AgentStatus
from providence.agents.perception.fundflow import PerceptFundFlow
from providence.exceptions import AgentProcessingError, ExternalAPIError
from providence.infra.plaid_client import PlaidClient
from providence.schemas.enums import DataType, ValidationStatus
from providence.schemas.market_state import MarketStateFragment


def _make_context(
    access_tokens: list[str] | None = None,
    date: str = "2026-02-09",
    history_days: int = 30,
) -> AgentContext:
    """Helper to create AgentContext with fund flow data expectations."""
    if access_tokens is None:
        access_tokens = ["test_token_1", "test_token_2"]

    return AgentContext(
        run_id=uuid4(),
        timestamp=datetime.now(timezone.utc),
        metadata={
            "access_tokens": access_tokens,
            "date": date,
            "history_days": history_days,
        },
        fragments=[],
        beliefs=[],
    )


class TestPerceptFundFlowInit:
    """Test PerceptFundFlow initialization."""

    def test_init_with_client(self):
        """Agent should initialize with PlaidClient."""
        mock_client = AsyncMock(spec=PlaidClient)
        agent = PerceptFundFlow(mock_client)

        assert agent.agent_id == "PERCEPT-FUNDFLOW"
        assert agent.agent_type == "perception"
        assert agent.version == "1.0.0"
        assert agent._plaid is mock_client
        assert agent._last_run is None
        assert agent._last_success is None
        assert agent._error_count_24h == 0


class TestPerceptFundFlowProcess:
    """Test the main process method."""

    @pytest.mark.asyncio
    async def test_process_requires_access_tokens(self):
        """Should raise AgentProcessingError when access_tokens not provided."""
        mock_client = AsyncMock(spec=PlaidClient)
        agent = PerceptFundFlow(mock_client)

        context = AgentContext(
            run_id=uuid4(),
            timestamp=datetime.now(timezone.utc),
            metadata={"date": "2026-02-09"},  # No access_tokens
            fragments=[],
            beliefs=[],
        )

        with pytest.raises(AgentProcessingError) as exc_info:
            await agent.process(context)

        assert "No Plaid access_tokens" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_process_produces_fragments(self):
        """Should produce fragments from investment transactions."""
        mock_client = AsyncMock(spec=PlaidClient)
        mock_client.get_investment_transactions.return_value = {
            "investment_transactions": [
                {
                    "investment_transaction_id": "txn1",
                    "account_id": "acct1",
                    "security_id": "sec1",
                    "date": "2026-02-01",
                    "amount": 100.0,  # Buy (positive = outflow)
                },
                {
                    "investment_transaction_id": "txn2",
                    "account_id": "acct1",
                    "security_id": "sec2",
                    "date": "2026-02-01",
                    "amount": -50.0,  # Sell (negative = inflow)
                },
            ],
            "accounts": [{"account_id": "acct1", "name": "Main Brokerage"}],
            "securities": [
                {"security_id": "sec1", "ticker_symbol": "AAPL"},
                {"security_id": "sec2", "ticker_symbol": "MSFT"},
            ],
            "total_investment_transactions": 2,
        }

        agent = PerceptFundFlow(mock_client)
        context = _make_context(access_tokens=["token1"])

        fragments = await agent.process(context)

        assert len(fragments) > 0
        assert all(isinstance(f, MarketStateFragment) for f in fragments)

    @pytest.mark.asyncio
    async def test_fragment_has_correct_data_type(self):
        """All fragments should have FUND_FLOW data_type."""
        mock_client = AsyncMock(spec=PlaidClient)
        mock_client.get_investment_transactions.return_value = {
            "investment_transactions": [
                {
                    "account_id": "acct1",
                    "security_id": "sec1",
                    "date": "2026-02-01",
                    "amount": 100.0,
                }
            ],
            "accounts": [{"account_id": "acct1", "name": "Brokerage"}],
            "securities": [{"security_id": "sec1", "ticker_symbol": "AAPL"}],
            "total_investment_transactions": 1,
        }

        agent = PerceptFundFlow(mock_client)
        context = _make_context(access_tokens=["token1"])

        fragments = await agent.process(context)

        assert all(f.data_type == DataType.FUND_FLOW for f in fragments)

    @pytest.mark.asyncio
    async def test_inflow_outflow_aggregation(self):
        """Should aggregate inflows and outflows correctly."""
        mock_client = AsyncMock(spec=PlaidClient)
        mock_client.get_investment_transactions.return_value = {
            "investment_transactions": [
                {
                    "account_id": "acct1",
                    "security_id": "sec1",
                    "date": "2026-02-01",
                    "amount": 100.0,  # Outflow
                },
                {
                    "account_id": "acct1",
                    "security_id": "sec1",
                    "date": "2026-02-01",
                    "amount": 50.0,  # Outflow
                },
                {
                    "account_id": "acct1",
                    "security_id": "sec1",
                    "date": "2026-02-01",
                    "amount": -30.0,  # Inflow
                },
            ],
            "accounts": [{"account_id": "acct1"}],
            "securities": [{"security_id": "sec1", "ticker_symbol": "AAPL"}],
            "total_investment_transactions": 3,
        }

        agent = PerceptFundFlow(mock_client)
        context = _make_context(access_tokens=["token1"])

        fragments = await agent.process(context)

        payload = fragments[0].payload
        assert payload["inflows"] == 30.0  # abs(-30)
        assert payload["outflows"] == 150.0  # 100 + 50
        assert payload["net_flow"] == -120.0  # inflows - outflows

    @pytest.mark.asyncio
    async def test_transaction_count_aggregation(self):
        """Should count transactions per account per day."""
        mock_client = AsyncMock(spec=PlaidClient)
        mock_client.get_investment_transactions.return_value = {
            "investment_transactions": [
                {"account_id": "acct1", "date": "2026-02-01", "amount": 100.0, "security_id": "sec1"},
                {"account_id": "acct1", "date": "2026-02-01", "amount": 50.0, "security_id": "sec1"},
                {"account_id": "acct1", "date": "2026-02-01", "amount": -30.0, "security_id": "sec1"},
            ],
            "accounts": [{"account_id": "acct1"}],
            "securities": [{"security_id": "sec1", "ticker_symbol": "AAPL"}],
            "total_investment_transactions": 3,
        }

        agent = PerceptFundFlow(mock_client)
        context = _make_context(access_tokens=["token1"])

        fragments = await agent.process(context)

        payload = fragments[0].payload
        assert payload["transaction_count"] == 3

    @pytest.mark.asyncio
    async def test_account_id_anonymized(self):
        """Account IDs should be hashed for privacy."""
        mock_client = AsyncMock(spec=PlaidClient)
        mock_client.get_investment_transactions.return_value = {
            "investment_transactions": [
                {
                    "account_id": "real_account_123",
                    "date": "2026-02-01",
                    "amount": 100.0,
                    "security_id": "sec1",
                }
            ],
            "accounts": [{"account_id": "real_account_123"}],
            "securities": [{"security_id": "sec1", "ticker_symbol": "AAPL"}],
            "total_investment_transactions": 1,
        }

        agent = PerceptFundFlow(mock_client)
        context = _make_context(access_tokens=["token1"])

        fragments = await agent.process(context)

        payload = fragments[0].payload
        expected_hash = hashlib.sha256("real_account_123".encode()).hexdigest()[:16]
        assert payload["account_id"] == expected_hash
        assert payload["account_id"] != "real_account_123"

    @pytest.mark.asyncio
    async def test_top_tickers_extracted(self):
        """Should extract top tickers by flow volume."""
        mock_client = AsyncMock(spec=PlaidClient)
        mock_client.get_investment_transactions.return_value = {
            "investment_transactions": [
                {
                    "account_id": "acct1",
                    "date": "2026-02-01",
                    "amount": 1000.0,
                    "security_id": "sec1",
                },
                {
                    "account_id": "acct1",
                    "date": "2026-02-01",
                    "amount": 500.0,
                    "security_id": "sec2",
                },
                {
                    "account_id": "acct1",
                    "date": "2026-02-01",
                    "amount": 100.0,
                    "security_id": "sec3",
                },
            ],
            "accounts": [{"account_id": "acct1"}],
            "securities": [
                {"security_id": "sec1", "ticker_symbol": "AAPL"},
                {"security_id": "sec2", "ticker_symbol": "MSFT"},
                {"security_id": "sec3", "ticker_symbol": "GOOGL"},
            ],
            "total_investment_transactions": 3,
        }

        agent = PerceptFundFlow(mock_client)
        context = _make_context(access_tokens=["token1"])

        fragments = await agent.process(context)

        payload = fragments[0].payload
        # Top tickers should be in descending order by volume
        assert payload["top_tickers"][0] == "AAPL"
        assert payload["top_tickers"][1] == "MSFT"
        assert payload["top_tickers"][2] == "GOOGL"

    @pytest.mark.asyncio
    async def test_top_tickers_max_10(self):
        """Should limit top tickers to 10."""
        mock_client = AsyncMock(spec=PlaidClient)

        # Create 15 transactions with different tickers
        transactions = []
        securities = []
        for i in range(15):
            ticker = f"TK{i:02d}"
            transactions.append({
                "account_id": "acct1",
                "date": "2026-02-01",
                "amount": 100.0 * (15 - i),  # Descending volumes
                "security_id": f"sec{i}",
            })
            securities.append({
                "security_id": f"sec{i}",
                "ticker_symbol": ticker,
            })

        mock_client.get_investment_transactions.return_value = {
            "investment_transactions": transactions,
            "accounts": [{"account_id": "acct1"}],
            "securities": securities,
            "total_investment_transactions": 15,
        }

        agent = PerceptFundFlow(mock_client)
        context = _make_context(access_tokens=["token1"])

        fragments = await agent.process(context)

        payload = fragments[0].payload
        assert len(payload["top_tickers"]) == 10

    @pytest.mark.asyncio
    async def test_pagination_handling(self):
        """Should handle multiple pages of transactions."""
        mock_client = AsyncMock(spec=PlaidClient)

        # First page has 500 transactions
        first_page = {
            "investment_transactions": [
                {
                    "account_id": "acct1",
                    "date": "2026-02-01",
                    "amount": 10.0,
                    "security_id": "sec1",
                }
                for _ in range(500)
            ],
            "accounts": [{"account_id": "acct1"}],
            "securities": [{"security_id": "sec1", "ticker_symbol": "AAPL"}],
            "total_investment_transactions": 750,
        }

        # Second page has 250 transactions
        second_page = {
            "investment_transactions": [
                {
                    "account_id": "acct1",
                    "date": "2026-02-01",
                    "amount": 10.0,
                    "security_id": "sec1",
                }
                for _ in range(250)
            ],
            "accounts": [{"account_id": "acct1"}],
            "securities": [{"security_id": "sec1", "ticker_symbol": "AAPL"}],
            "total_investment_transactions": 750,
        }

        mock_client.get_investment_transactions.side_effect = [first_page, second_page]

        agent = PerceptFundFlow(mock_client)
        context = _make_context(access_tokens=["token1"])

        fragments = await agent.process(context)

        # Should have called get_investment_transactions twice
        assert mock_client.get_investment_transactions.call_count == 2

        # Payload should reflect all transactions
        payload = fragments[0].payload
        assert payload["transaction_count"] == 750

    @pytest.mark.asyncio
    async def test_no_transactions_returns_quarantined_fragment(self):
        """Should return quarantined fragment if no transactions found."""
        mock_client = AsyncMock(spec=PlaidClient)
        mock_client.get_investment_transactions.return_value = {
            "investment_transactions": [],
            "accounts": [],
            "securities": [],
            "total_investment_transactions": 0,
        }

        agent = PerceptFundFlow(mock_client)
        context = _make_context(access_tokens=["token1"])

        fragments = await agent.process(context)

        assert len(fragments) == 1
        assert fragments[0].validation_status == ValidationStatus.QUARANTINED

    @pytest.mark.asyncio
    async def test_api_failure_produces_quarantined_fragment(self):
        """On API failure, should produce one quarantined fragment."""
        mock_client = AsyncMock(spec=PlaidClient)
        mock_client.get_investment_transactions.side_effect = ExternalAPIError(
            message="Plaid API error",
            service="plaid",
        )

        agent = PerceptFundFlow(mock_client)
        context = _make_context(access_tokens=["token1"])

        fragments = await agent.process(context)

        assert len(fragments) > 0
        # At least one fragment should be quarantined
        assert any(f.validation_status == ValidationStatus.QUARANTINED for f in fragments)

    @pytest.mark.asyncio
    async def test_validation_status_valid(self):
        """Fragment should be VALID when transactions present."""
        mock_client = AsyncMock(spec=PlaidClient)
        mock_client.get_investment_transactions.return_value = {
            "investment_transactions": [
                {
                    "account_id": "acct1",
                    "date": "2026-02-01",
                    "amount": 100.0,
                    "security_id": "sec1",
                }
            ],
            "accounts": [{"account_id": "acct1"}],
            "securities": [{"security_id": "sec1", "ticker_symbol": "AAPL"}],
            "total_investment_transactions": 1,
        }

        agent = PerceptFundFlow(mock_client)
        context = _make_context(access_tokens=["token1"])

        fragments = await agent.process(context)

        assert fragments[0].validation_status == ValidationStatus.VALID

    @pytest.mark.asyncio
    async def test_multiple_access_tokens(self):
        """Should process all access tokens."""
        mock_client = AsyncMock(spec=PlaidClient)
        mock_client.get_investment_transactions.return_value = {
            "investment_transactions": [
                {
                    "account_id": "acct1",
                    "date": "2026-02-01",
                    "amount": 100.0,
                    "security_id": "sec1",
                }
            ],
            "accounts": [{"account_id": "acct1"}],
            "securities": [{"security_id": "sec1", "ticker_symbol": "AAPL"}],
            "total_investment_transactions": 1,
        }

        agent = PerceptFundFlow(mock_client)
        context = _make_context(access_tokens=["token1", "token2", "token3"])

        fragments = await agent.process(context)

        # Should have called get_investment_transactions once per token
        assert mock_client.get_investment_transactions.call_count == 3

    @pytest.mark.asyncio
    async def test_default_date_is_today(self):
        """If date not in context, should default to today."""
        mock_client = AsyncMock(spec=PlaidClient)
        mock_client.get_investment_transactions.return_value = {
            "investment_transactions": [],
            "accounts": [],
            "securities": [],
            "total_investment_transactions": 0,
        }

        agent = PerceptFundFlow(mock_client)
        context = AgentContext(
            run_id=uuid4(),
            timestamp=datetime.now(timezone.utc),
            metadata={
                "access_tokens": ["token1"],
                # No date key
                "history_days": 30,
            },
            fragments=[],
            beliefs=[],
        )

        await agent.process(context)

        # Should have called with today's date
        call_args = mock_client.get_investment_transactions.call_args
        end_date = call_args[1]["end_date"]
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        assert end_date == today

    @pytest.mark.asyncio
    async def test_health_status_healthy_on_success(self):
        """Health should be HEALTHY after successful run."""
        mock_client = AsyncMock(spec=PlaidClient)
        mock_client.get_investment_transactions.return_value = {
            "investment_transactions": [
                {
                    "account_id": "acct1",
                    "date": "2026-02-01",
                    "amount": 100.0,
                    "security_id": "sec1",
                }
            ],
            "accounts": [{"account_id": "acct1"}],
            "securities": [{"security_id": "sec1", "ticker_symbol": "AAPL"}],
            "total_investment_transactions": 1,
        }

        agent = PerceptFundFlow(mock_client)
        context = _make_context(access_tokens=["token1"])

        await agent.process(context)

        health = agent.get_health()
        assert health.status == AgentStatus.HEALTHY
        assert health.agent_id == "PERCEPT-FUNDFLOW"

    @pytest.mark.asyncio
    async def test_health_status_degraded_on_multiple_errors(self):
        """Health should be DEGRADED after 3+ errors."""
        mock_client = AsyncMock(spec=PlaidClient)

        agent = PerceptFundFlow(mock_client)
        agent._error_count_24h = 4

        health = agent.get_health()
        assert health.status == AgentStatus.DEGRADED

    @pytest.mark.asyncio
    async def test_health_status_unhealthy_on_many_errors(self):
        """Health should be UNHEALTHY after 10+ errors."""
        mock_client = AsyncMock(spec=PlaidClient)

        agent = PerceptFundFlow(mock_client)
        agent._error_count_24h = 11

        health = agent.get_health()
        assert health.status == AgentStatus.UNHEALTHY


class TestPerceptFundFlowIntegration:
    """Integration-level tests."""

    @pytest.mark.asyncio
    async def test_multiple_runs_track_success(self):
        """Multiple successful runs should update _last_success."""
        mock_client = AsyncMock(spec=PlaidClient)
        mock_client.get_investment_transactions.return_value = {
            "investment_transactions": [
                {
                    "account_id": "acct1",
                    "date": "2026-02-01",
                    "amount": 100.0,
                    "security_id": "sec1",
                }
            ],
            "accounts": [{"account_id": "acct1"}],
            "securities": [{"security_id": "sec1", "ticker_symbol": "AAPL"}],
            "total_investment_transactions": 1,
        }

        agent = PerceptFundFlow(mock_client)

        context1 = _make_context(access_tokens=["token1"])
        await agent.process(context1)
        assert agent._last_success is not None

        first_success = agent._last_success

        context2 = _make_context(access_tokens=["token1"])
        await agent.process(context2)

        assert agent._last_success > first_success

    @pytest.mark.asyncio
    async def test_error_tracking_across_runs(self):
        """Error count should accumulate across runs."""
        mock_client = AsyncMock(spec=PlaidClient)
        mock_client.get_investment_transactions.side_effect = ExternalAPIError(
            message="API error",
            service="plaid",
        )

        agent = PerceptFundFlow(mock_client)

        context1 = _make_context(access_tokens=["token1"])
        await agent.process(context1)
        assert agent._error_count_24h == 1

        context2 = _make_context(access_tokens=["token1"])
        await agent.process(context2)
        assert agent._error_count_24h == 2
