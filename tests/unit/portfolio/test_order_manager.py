"""Tests for OrderManager at providence/portfolio/order_manager.py."""
import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import pytest

from providence.portfolio.order_manager import (
    ManagedOrder,
    OrderManager,
    OrderSide,
    OrderStatus,
)


class TestManagedOrder:
    """Test ManagedOrder dataclass."""

    def test_order_creation(self):
        """ManagedOrder can be created."""
        order_id = uuid4()
        order = ManagedOrder(
            order_id=order_id,
            broker_order_id=None,
            client_order_id="prov-12345",
            ticker="AAPL",
            side=OrderSide.BUY,
            order_type="market",
            time_in_force="day",
            qty=Decimal("100"),
            notional=None,
        )
        assert order.order_id == order_id
        assert order.ticker == "AAPL"
        assert order.status == OrderStatus.PENDING

    def test_order_is_terminal(self):
        """is_terminal reflects terminal states."""
        order = ManagedOrder(
            order_id=uuid4(),
            broker_order_id=None,
            client_order_id="prov-12345",
            ticker="AAPL",
            side=OrderSide.BUY,
            order_type="market",
            time_in_force="day",
            qty=Decimal("100"),
            notional=None,
            status=OrderStatus.PENDING,
        )
        assert order.is_terminal is False

        order.status = OrderStatus.FILLED
        assert order.is_terminal is True

    def test_order_is_active(self):
        """is_active is opposite of is_terminal."""
        order = ManagedOrder(
            order_id=uuid4(),
            broker_order_id=None,
            client_order_id="prov-12345",
            ticker="AAPL",
            side=OrderSide.BUY,
            order_type="market",
            time_in_force="day",
            qty=Decimal("100"),
            notional=None,
            status=OrderStatus.SUBMITTED,
        )
        assert order.is_active is True

        order.status = OrderStatus.CANCELLED
        assert order.is_active is False

    def test_order_fill_pct(self):
        """fill_pct calculates fill percentage."""
        order = ManagedOrder(
            order_id=uuid4(),
            broker_order_id=None,
            client_order_id="prov-12345",
            ticker="AAPL",
            side=OrderSide.BUY,
            order_type="market",
            time_in_force="day",
            qty=Decimal("100"),
            notional=None,
            filled_qty=Decimal("50"),
        )
        assert order.fill_pct == pytest.approx(0.5, abs=0.01)

    def test_order_can_retry(self):
        """can_retry checks status and retry count."""
        order = ManagedOrder(
            order_id=uuid4(),
            broker_order_id=None,
            client_order_id="prov-12345",
            ticker="AAPL",
            side=OrderSide.BUY,
            order_type="market",
            time_in_force="day",
            qty=Decimal("100"),
            notional=None,
            status=OrderStatus.FAILED,
            retry_count=0,
        )
        assert order.can_retry is True

        order.retry_count = 3
        assert order.can_retry is False


class TestOrderTransitions:
    """Test order state transitions."""

    def test_valid_transition_pending_to_submitted(self):
        """Valid transition PENDING -> SUBMITTED succeeds."""
        order = ManagedOrder(
            order_id=uuid4(),
            broker_order_id=None,
            client_order_id="prov-12345",
            ticker="AAPL",
            side=OrderSide.BUY,
            order_type="market",
            time_in_force="day",
            qty=Decimal("100"),
            notional=None,
            status=OrderStatus.PENDING,
        )

        order.transition_to(OrderStatus.SUBMITTED, reason="Sent to broker")

        assert order.status == OrderStatus.SUBMITTED
        assert len(order.transitions) == 1

    def test_valid_transition_submitted_to_filled(self):
        """Valid transition SUBMITTED -> FILLED succeeds."""
        order = ManagedOrder(
            order_id=uuid4(),
            broker_order_id="b123",
            client_order_id="prov-12345",
            ticker="AAPL",
            side=OrderSide.BUY,
            order_type="market",
            time_in_force="day",
            qty=Decimal("100"),
            notional=None,
            status=OrderStatus.SUBMITTED,
        )

        order.transition_to(
            OrderStatus.FILLED,
            reason="Fill received from broker",
            broker_data={"filled_qty": "100", "filled_avg_price": "150.00"},
        )

        assert order.status == OrderStatus.FILLED
        assert order.filled_qty == Decimal("100")

    def test_invalid_transition_raises(self):
        """Invalid transition raises ValueError."""
        order = ManagedOrder(
            order_id=uuid4(),
            broker_order_id=None,
            client_order_id="prov-12345",
            ticker="AAPL",
            side=OrderSide.BUY,
            order_type="market",
            time_in_force="day",
            qty=Decimal("100"),
            notional=None,
            status=OrderStatus.FILLED,
        )

        with pytest.raises(ValueError, match="Invalid transition"):
            order.transition_to(OrderStatus.SUBMITTED)

    def test_transition_logging(self):
        """Transitions are logged in transitions list."""
        order = ManagedOrder(
            order_id=uuid4(),
            broker_order_id=None,
            client_order_id="prov-12345",
            ticker="AAPL",
            side=OrderSide.BUY,
            order_type="market",
            time_in_force="day",
            qty=Decimal("100"),
            notional=None,
            status=OrderStatus.PENDING,
        )

        order.transition_to(OrderStatus.SUBMITTED, reason="Test reason")

        assert len(order.transitions) == 1
        transition = order.transitions[0]
        assert transition["from_status"] == "PENDING"
        assert transition["to_status"] == "SUBMITTED"
        assert transition["reason"] == "Test reason"


class TestOrderManagerCreate:
    """Test creating orders from RoutedOrder."""

    def test_create_from_routed_order_open_long(self):
        """create_from_routed_order maps OPEN_LONG to BUY."""
        routed_order = {
            "order_id": str(uuid4()),
            "ticker": "AAPL",
            "action": "OPEN_LONG",
            "direction": "LONG",
            "target_weight": 0.08,
            "confidence": 0.72,
            "source_intent_id": str(uuid4()),
            "execution_strategy": "LIMIT",
            "urgency": "NORMAL",
            "time_horizon_days": 90,
            "max_slippage_bps": 50,
        }
        manager = OrderManager()

        order = manager.create_from_routed_order(routed_order, Decimal("100000"))

        assert order.side == OrderSide.BUY
        assert order.ticker == "AAPL"

    def test_create_from_routed_order_open_short(self):
        """create_from_routed_order maps OPEN_SHORT to SELL."""
        routed_order = {
            "order_id": str(uuid4()),
            "ticker": "AAPL",
            "action": "OPEN_SHORT",
            "direction": "SHORT",
            "target_weight": 0.08,
            "confidence": 0.72,
            "source_intent_id": str(uuid4()),
            "execution_strategy": "MARKET",
            "urgency": "HIGH",
            "time_horizon_days": 90,
            "max_slippage_bps": 50,
        }
        manager = OrderManager()

        order = manager.create_from_routed_order(routed_order, Decimal("100000"))

        assert order.side == OrderSide.SELL

    def test_create_from_routed_order_calculates_notional(self):
        """create_from_routed_order calculates notional from target_weight."""
        routed_order = {
            "order_id": str(uuid4()),
            "ticker": "AAPL",
            "action": "OPEN_LONG",
            "direction": "LONG",
            "target_weight": 0.10,
            "confidence": 0.72,
            "source_intent_id": str(uuid4()),
            "execution_strategy": "MARKET",
            "urgency": "NORMAL",
            "time_horizon_days": 90,
            "max_slippage_bps": 50,
        }
        manager = OrderManager()
        portfolio_equity = Decimal("100000")

        order = manager.create_from_routed_order(routed_order, portfolio_equity)

        # notional should be 10000 (10% of 100000)
        assert order.notional == Decimal("10000")

    def test_create_from_routed_order_maps_execution_strategy(self):
        """create_from_routed_order maps execution_strategy to order_type."""
        routed_order = {
            "order_id": str(uuid4()),
            "ticker": "AAPL",
            "action": "OPEN_LONG",
            "direction": "LONG",
            "target_weight": 0.08,
            "confidence": 0.72,
            "source_intent_id": str(uuid4()),
            "execution_strategy": "LIMIT",
            "urgency": "NORMAL",
            "time_horizon_days": 90,
            "max_slippage_bps": 50,
        }
        manager = OrderManager()

        order = manager.create_from_routed_order(routed_order, Decimal("100000"))

        assert order.order_type == "limit"


class TestOrderManagerSubmit:
    """Test mark_submitted method."""

    def test_mark_submitted_sets_broker_id(self):
        """mark_submitted sets broker_order_id."""
        manager = OrderManager()
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
        order = manager.create_from_routed_order(routed, Decimal("100000"))

        manager.mark_submitted(order_id, "broker-order-123")

        assert order.broker_order_id == "broker-order-123"
        assert order.status == OrderStatus.SUBMITTED

    def test_mark_submitted_updates_index_maps(self):
        """mark_submitted updates internal maps."""
        manager = OrderManager()
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
        order = manager.create_from_routed_order(routed, Decimal("100000"))

        manager.mark_submitted(order_id, "broker-order-123")

        assert manager.get_by_broker_id("broker-order-123") is not None


class TestOrderManagerUpdate:
    """Test update_from_broker method."""

    def test_update_from_broker_filled_status(self):
        """update_from_broker maps filled status."""
        manager = OrderManager()
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
        order = manager.create_from_routed_order(routed, Decimal("100000"))
        manager.mark_submitted(order_id, "broker-123")

        broker_status = {
            "status": "filled",
            "filled_qty": 100,
            "filled_avg_price": 150.00,
        }
        manager.update_from_broker(order_id, broker_status)

        order = manager.get_order(order_id)
        assert order.status == OrderStatus.FILLED
        assert order.filled_qty == Decimal("100")

    def test_update_from_broker_cancelled_status(self):
        """update_from_broker maps cancelled status."""
        manager = OrderManager()
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
        order = manager.create_from_routed_order(routed, Decimal("100000"))
        manager.mark_submitted(order_id, "broker-123")

        broker_status = {"status": "cancelled"}
        manager.update_from_broker(order_id, broker_status)

        order = manager.get_order(order_id)
        assert order.status == OrderStatus.CANCELLED

    def test_update_from_broker_rejected_status(self):
        """update_from_broker maps rejected status."""
        manager = OrderManager()
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
        order = manager.create_from_routed_order(routed, Decimal("100000"))
        manager.mark_submitted(order_id, "broker-123")

        broker_status = {"status": "rejected"}
        manager.update_from_broker(order_id, broker_status)

        order = manager.get_order(order_id)
        assert order.status == OrderStatus.REJECTED


class TestOrderManagerQueries:
    """Test query methods."""

    @pytest.fixture
    def manager_with_orders(self):
        """Create manager with sample orders."""
        manager = OrderManager()
        for i in range(3):
            order_id = uuid4()
            routed = {
                "order_id": str(order_id),
                "ticker": f"TICK{i}",
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
            manager.create_from_routed_order(routed, Decimal("100000"))
        return manager

    def test_get_active_orders(self, manager_with_orders):
        """get_active_orders returns non-terminal orders."""
        active = manager_with_orders.get_active_orders()
        assert len(active) == 3
        assert all(o.is_active for o in active)

    def test_get_pending_orders(self, manager_with_orders):
        """get_pending_orders returns PENDING orders."""
        pending = manager_with_orders.get_pending_orders()
        assert len(pending) == 3
        assert all(o.status == OrderStatus.PENDING for o in pending)

    def test_get_retryable_orders(self, manager_with_orders):
        """get_retryable_orders returns failed/rejected orders."""
        order_id = manager_with_orders.get_pending_orders()[0].order_id
        manager_with_orders.mark_failed(order_id, "Test error")

        retryable = manager_with_orders.get_retryable_orders()
        assert len(retryable) == 1
        assert retryable[0].can_retry

    def test_get_orders_for_ticker(self, manager_with_orders):
        """get_orders_for_ticker returns orders for specific ticker."""
        orders = manager_with_orders.get_orders_for_ticker("TICK0")
        assert len(orders) == 1
        assert orders[0].ticker == "TICK0"


class TestOrderManagerStats:
    """Test stats property."""

    def test_stats_returns_counts(self):
        """stats returns order counts by status."""
        manager = OrderManager()
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
        manager.create_from_routed_order(routed, Decimal("100000"))

        stats = manager.stats

        assert stats["PENDING"] == 1
        assert stats["total"] == 1
        assert stats["active"] == 1


class TestOrderManagerPersistence:
    """Test persistence to JSONL."""

    def test_persistence_saves_orders(self):
        """Orders are persisted to JSONL."""
        with TemporaryDirectory() as tmpdir:
            persist_path = Path(tmpdir) / "orders.jsonl"
            manager = OrderManager(persist_path=persist_path)
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
            manager.create_from_routed_order(routed, Decimal("100000"))

            assert persist_path.exists()
            with open(persist_path, "r") as f:
                lines = f.readlines()
                assert len(lines) > 0

    def test_persistence_loads_on_init(self):
        """OrderManager loads persisted orders on init."""
        with TemporaryDirectory() as tmpdir:
            persist_path = Path(tmpdir) / "orders.jsonl"

            # Create and persist an order
            manager1 = OrderManager(persist_path=persist_path)
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
            manager1.create_from_routed_order(routed, Decimal("100000"))

            # Create new manager from same persist path
            manager2 = OrderManager(persist_path=persist_path)

            # Should have loaded the order
            assert manager2.get_order(order_id) is not None
