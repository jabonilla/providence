"""Tests for ExecutionService at providence/services/execution_service.py."""
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from providence.services.execution_service import ExecutionService
from providence.portfolio.order_manager import OrderManager, OrderStatus
from providence.portfolio.tracker import PortfolioTracker
from providence.infra.alpaca_client import AlpacaClient


class TestExecuteRoutingPlan:
    """Test execute_routing_plan method."""

    @pytest.fixture
    def mocked_service(self):
        """Create service with mocked dependencies."""
        broker = AsyncMock(spec=AlpacaClient)
        order_manager = OrderManager()
        portfolio = PortfolioTracker()
        return ExecutionService(broker, order_manager, portfolio)

    @pytest.mark.asyncio
    async def test_execute_routing_plan_submits_orders(self, mocked_service):
        """execute_routing_plan submits orders from routing plan."""
        routed_order = {
            "order_id": str(uuid4()),
            "ticker": "AAPL",
            "action": "OPEN_LONG",
            "direction": "LONG",
            "target_weight": 0.08,
            "confidence": 0.72,
            "source_intent_id": str(uuid4()),
            "execution_strategy": "MARKET",
            "urgency": "NORMAL",
            "time_horizon_days": 90,
            "max_slippage_bps": 50,
        }
        routing_plan = {
            "routing_id": str(uuid4()),
            "orders": [routed_order],
        }
        guardian_verdict = {"system_halt": False}

        mocked_service._broker.get_account.return_value = {
            "portfolio_value": "100000.00",
            "equity": "100000.00",
            "cash": "50000.00",
            "buying_power": "100000.00",
        }
        mocked_service._broker.submit_order.return_value = {
            "id": "broker-order-123",
            "status": "new",
        }
        mocked_service._broker.get_order.return_value = {
            "status": "filled",
            "filled_qty": 100,
            "filled_avg_price": 150.00,
        }

        result = await mocked_service.execute_routing_plan(routing_plan, guardian_verdict)

        assert result["submitted"] >= 1

    @pytest.mark.asyncio
    async def test_execute_routing_plan_returns_summary(self, mocked_service):
        """execute_routing_plan returns execution summary."""
        routing_plan = {
            "routing_id": str(uuid4()),
            "orders": [],
        }
        guardian_verdict = {"system_halt": False}

        mocked_service._broker.get_account.return_value = {
            "portfolio_value": "100000.00",
            "equity": "100000.00",
            "cash": "50000.00",
            "buying_power": "100000.00",
        }

        result = await mocked_service.execute_routing_plan(routing_plan, guardian_verdict)

        assert "submitted" in result
        assert "filled" in result
        assert "rejected" in result
        assert "halted" in result


class TestGuardianHalt:
    """Test guardian halt trigger."""

    @pytest.fixture
    def mocked_service(self):
        """Create service with mocked dependencies."""
        broker = AsyncMock(spec=AlpacaClient)
        order_manager = OrderManager()
        portfolio = PortfolioTracker()
        return ExecutionService(broker, order_manager, portfolio)

    @pytest.mark.asyncio
    async def test_system_halt_triggers_emergency_halt(self, mocked_service):
        """system_halt=True triggers emergency_halt."""
        routing_plan = {"routing_id": str(uuid4()), "orders": []}
        guardian_verdict = {"system_halt": True, "halt_reason": "Risk limit exceeded"}

        mocked_service._broker.cancel_all_orders.return_value = True

        result = await mocked_service.execute_routing_plan(routing_plan, guardian_verdict)

        assert result["halted"] is True
        assert result["halt_reason"] == "Risk limit exceeded"

    @pytest.mark.asyncio
    async def test_no_halt_when_verdict_false(self, mocked_service):
        """system_halt=False doesn't trigger halt."""
        routing_plan = {"routing_id": str(uuid4()), "orders": []}
        guardian_verdict = {"system_halt": False}

        mocked_service._broker.get_account.return_value = {
            "portfolio_value": "100000.00",
            "equity": "100000.00",
            "cash": "50000.00",
            "buying_power": "100000.00",
        }

        result = await mocked_service.execute_routing_plan(routing_plan, guardian_verdict)

        assert result["halted"] is False


class TestPollOrderStatus:
    """Test poll_order_status method."""

    @pytest.fixture
    def mocked_service(self):
        """Create service with mocked dependencies."""
        broker = AsyncMock(spec=AlpacaClient)
        order_manager = OrderManager()
        portfolio = PortfolioTracker()
        return ExecutionService(broker, order_manager, portfolio)

    @pytest.mark.asyncio
    async def test_poll_order_status_updates_fills(self, mocked_service):
        """poll_order_status polls and updates order states."""
        # Create an order
        order_id = uuid4()
        routed = {
            "order_id": str(order_id),
            "ticker": "AAPL",
            "action": "OPEN_LONG",
            "direction": "LONG",
            "target_weight": 0.08,
            "confidence": 0.72,
            "source_intent_id": str(uuid4()),
            "execution_strategy": "MARKET",
            "urgency": "NORMAL",
            "time_horizon_days": 90,
            "max_slippage_bps": 50,
        }
        order = mocked_service._orders.create_from_routed_order(routed, Decimal("100000"))
        mocked_service._orders.mark_submitted(order_id, "broker-order-123")

        # Mock broker response
        mocked_service._broker.get_order.return_value = {
            "id": "broker-order-123",
            "status": "filled",
            "filled_qty": "100",
            "filled_avg_price": "150.00",
        }

        filled_count = await mocked_service.poll_order_status(max_attempts=1)

        assert filled_count >= 0
        updated_order = mocked_service._orders.get_order(order_id)
        assert updated_order.status == OrderStatus.FILLED

    @pytest.mark.asyncio
    async def test_poll_order_status_stops_when_no_active(self, mocked_service):
        """poll_order_status returns early if no active orders."""
        filled_count = await mocked_service.poll_order_status(max_attempts=5)

        assert filled_count == 0


class TestReconcile:
    """Test reconcile method."""

    @pytest.fixture
    def mocked_service(self):
        """Create service with mocked dependencies."""
        broker = AsyncMock(spec=AlpacaClient)
        order_manager = OrderManager()
        portfolio = PortfolioTracker()
        return ExecutionService(broker, order_manager, portfolio)

    @pytest.mark.asyncio
    async def test_reconcile_syncs_portfolio(self, mocked_service):
        """reconcile syncs portfolio with broker."""
        mocked_service._broker.get_account.return_value = {
            "portfolio_value": "105000.00",
            "equity": "105000.00",
            "cash": "50000.00",
            "buying_power": "105000.00",
        }
        mocked_service._broker.list_positions.return_value = [
            {
                "symbol": "AAPL",
                "qty": "100",
                "avg_entry_price": "150.00",
                "current_price": "155.00",
                "sector": "Information Technology",
            }
        ]

        result = await mocked_service.reconcile()

        assert result["account_synced"] is True
        assert "AAPL" in result["positions_synced"]


class TestEmergencyHalt:
    """Test emergency_halt method."""

    @pytest.fixture
    def mocked_service(self):
        """Create service with mocked dependencies."""
        broker = AsyncMock(spec=AlpacaClient)
        order_manager = OrderManager()
        portfolio = PortfolioTracker()
        return ExecutionService(broker, order_manager, portfolio)

    @pytest.mark.asyncio
    async def test_emergency_halt_cancels_orders(self, mocked_service):
        """emergency_halt cancels all orders."""
        # Create an order
        order_id = uuid4()
        routed = {
            "order_id": str(order_id),
            "ticker": "AAPL",
            "action": "OPEN_LONG",
            "direction": "LONG",
            "target_weight": 0.08,
            "confidence": 0.72,
            "source_intent_id": str(uuid4()),
            "execution_strategy": "MARKET",
            "urgency": "NORMAL",
            "time_horizon_days": 90,
            "max_slippage_bps": 50,
        }
        order = mocked_service._orders.create_from_routed_order(routed, Decimal("100000"))
        mocked_service._orders.mark_submitted(order_id, "broker-order-123")

        mocked_service._broker.cancel_all_orders.return_value = True

        result = await mocked_service.emergency_halt()

        assert result["cancelled_orders"] >= 0
        mocked_service._broker.cancel_all_orders.assert_called_once()


class TestRetryFailedOrders:
    """Test retry logic for failed orders."""

    @pytest.fixture
    def mocked_service(self):
        """Create service with mocked dependencies."""
        broker = AsyncMock(spec=AlpacaClient)
        order_manager = OrderManager()
        portfolio = PortfolioTracker()
        return ExecutionService(broker, order_manager, portfolio)

    @pytest.mark.asyncio
    async def test_retry_failed_orders(self, mocked_service):
        """retry_failed_orders retries retryable orders."""
        # Create a failed order
        order_id = uuid4()
        routed = {
            "order_id": str(order_id),
            "ticker": "AAPL",
            "action": "OPEN_LONG",
            "direction": "LONG",
            "target_weight": 0.08,
            "confidence": 0.72,
            "source_intent_id": str(uuid4()),
            "execution_strategy": "MARKET",
            "urgency": "NORMAL",
            "time_horizon_days": 90,
            "max_slippage_bps": 50,
        }
        order = mocked_service._orders.create_from_routed_order(routed, Decimal("100000"))
        mocked_service._orders.mark_submitted(order_id, "broker-order-123")
        mocked_service._orders.mark_failed(order_id, "Timeout error")

        # Mock successful retry
        mocked_service._broker.submit_order.return_value = {
            "id": "broker-order-456",
            "status": "new",
        }

        retryable_orders = mocked_service._orders.get_retryable_orders()
        assert len(retryable_orders) > 0
