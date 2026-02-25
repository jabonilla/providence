"""Order lifecycle management with state machine and error recovery.

Order States:
    PENDING -> SUBMITTED -> PARTIALLY_FILLED -> FILLED
    PENDING -> SUBMITTED -> CANCELLED
    PENDING -> SUBMITTED -> REJECTED
    PENDING -> SUBMITTED -> EXPIRED
    Any state -> FAILED (on unrecoverable error)

Each transition is logged immutably. Failed orders can be retried
up to MAX_RETRIES times with exponential backoff.
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import structlog

logger = structlog.get_logger()


class OrderStatus(str, Enum):
    PENDING = "PENDING"          # Created, not yet submitted to broker
    SUBMITTED = "SUBMITTED"       # Sent to broker, awaiting fill
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"            # Completely filled
    CANCELLED = "CANCELLED"       # Cancelled by us or broker
    REJECTED = "REJECTED"         # Broker rejected
    EXPIRED = "EXPIRED"          # Time-in-force expired
    FAILED = "FAILED"            # Unrecoverable error


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


TERMINAL_STATES = {
    OrderStatus.FILLED,
    OrderStatus.CANCELLED,
    OrderStatus.REJECTED,
    OrderStatus.EXPIRED,
    OrderStatus.FAILED,
}

VALID_TRANSITIONS = {
    OrderStatus.PENDING: {
        OrderStatus.SUBMITTED,
        OrderStatus.FAILED,
        OrderStatus.CANCELLED,
    },
    OrderStatus.SUBMITTED: {
        OrderStatus.PARTIALLY_FILLED,
        OrderStatus.FILLED,
        OrderStatus.CANCELLED,
        OrderStatus.REJECTED,
        OrderStatus.EXPIRED,
        OrderStatus.FAILED,
    },
    OrderStatus.PARTIALLY_FILLED: {
        OrderStatus.FILLED,
        OrderStatus.CANCELLED,
        OrderStatus.FAILED,
    },
}


@dataclass
class ManagedOrder:
    """An order tracked through its full lifecycle."""

    order_id: UUID  # Providence internal ID (matches RoutedOrder.order_id)
    broker_order_id: str | None  # Alpaca's order ID (set after submission)
    client_order_id: str  # Idempotency key for broker
    ticker: str
    side: OrderSide
    order_type: str  # MARKET, LIMIT, STOP, STOP_LIMIT
    time_in_force: str  # day, gtc, ioc, fok
    qty: Decimal | None  # shares (None if using notional)
    notional: Decimal | None  # dollar amount (None if using qty)
    limit_price: Decimal | None
    stop_price: Decimal | None

    # Execution tracking
    status: OrderStatus = OrderStatus.PENDING
    filled_qty: Decimal = Decimal("0")
    filled_avg_price: Decimal = Decimal("0")

    # Metadata from RoutedOrder
    source_intent_id: UUID | None = None
    execution_strategy: str = "MARKET"
    target_weight: float = 0.0
    confidence: float = 0.0
    max_slippage_bps: int = 50

    # Lifecycle tracking
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    submitted_at: datetime | None = None
    filled_at: datetime | None = None
    cancelled_at: datetime | None = None

    # Error recovery
    retry_count: int = 0
    max_retries: int = 3
    last_error: str | None = None

    # Transition log (immutable append-only)
    transitions: list[dict] = field(default_factory=list)

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATES

    @property
    def is_active(self) -> bool:
        return not self.is_terminal

    @property
    def fill_pct(self) -> float:
        if not self.qty or self.qty == 0:
            return 1.0 if self.status == OrderStatus.FILLED else 0.0
        return float(self.filled_qty / self.qty)

    @property
    def can_retry(self) -> bool:
        return (
            self.retry_count < self.max_retries
            and self.status in (OrderStatus.FAILED, OrderStatus.REJECTED)
        )

    def transition_to(
        self,
        new_status: OrderStatus,
        reason: str = "",
        broker_data: dict | None = None,
    ) -> None:
        """Execute a state transition with validation and logging.

        Parameters
        ----------
        new_status : OrderStatus
            Target state
        reason : str
            Human-readable reason for transition
        broker_data : dict | None
            Optional broker metadata (e.g., filled_qty, filled_price)
        """
        # Validate transition is legal
        if self.status not in VALID_TRANSITIONS or new_status not in VALID_TRANSITIONS.get(
            self.status, set()
        ):
            raise ValueError(
                f"Invalid transition: {self.status.value} -> {new_status.value}"
            )

        # Record transition
        transition_record = {
            "from_status": self.status.value,
            "to_status": new_status.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "reason": reason,
            "broker_data": broker_data or {},
        }
        self.transitions.append(transition_record)

        # Update state
        old_status = self.status
        self.status = new_status

        # Update timestamps
        if new_status == OrderStatus.SUBMITTED and not self.submitted_at:
            self.submitted_at = datetime.now(timezone.utc)
        elif new_status == OrderStatus.FILLED and not self.filled_at:
            self.filled_at = datetime.now(timezone.utc)
        elif new_status == OrderStatus.CANCELLED and not self.cancelled_at:
            self.cancelled_at = datetime.now(timezone.utc)

        # Update fill info from broker data if provided
        if broker_data:
            if "filled_qty" in broker_data:
                self.filled_qty = Decimal(str(broker_data["filled_qty"]))
            if "filled_avg_price" in broker_data:
                self.filled_avg_price = Decimal(str(broker_data["filled_avg_price"]))

        logger.info(
            "Order transitioned",
            order_id=str(self.order_id),
            ticker=self.ticker,
            from_status=old_status.value,
            to_status=new_status.value,
            reason=reason,
            filled_qty=str(self.filled_qty),
            fill_pct=self.fill_pct,
        )

    def mark_failed(self, error: str) -> None:
        """Mark order as failed with error message."""
        self.last_error = error
        self.transition_to(OrderStatus.FAILED, reason=f"Error: {error}")

    def to_dict(self) -> dict[str, Any]:
        """Serialize for persistence."""
        return {
            "order_id": str(self.order_id),
            "broker_order_id": self.broker_order_id,
            "client_order_id": self.client_order_id,
            "ticker": self.ticker,
            "side": self.side.value,
            "order_type": self.order_type,
            "time_in_force": self.time_in_force,
            "qty": str(self.qty) if self.qty is not None else None,
            "notional": str(self.notional) if self.notional is not None else None,
            "limit_price": str(self.limit_price) if self.limit_price is not None else None,
            "stop_price": str(self.stop_price) if self.stop_price is not None else None,
            "status": self.status.value,
            "filled_qty": str(self.filled_qty),
            "filled_avg_price": str(self.filled_avg_price),
            "source_intent_id": str(self.source_intent_id) if self.source_intent_id else None,
            "execution_strategy": self.execution_strategy,
            "target_weight": self.target_weight,
            "confidence": self.confidence,
            "max_slippage_bps": self.max_slippage_bps,
            "created_at": self.created_at.isoformat(),
            "submitted_at": self.submitted_at.isoformat() if self.submitted_at else None,
            "filled_at": self.filled_at.isoformat() if self.filled_at else None,
            "cancelled_at": self.cancelled_at.isoformat() if self.cancelled_at else None,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "last_error": self.last_error,
            "transitions": self.transitions,
        }

    @staticmethod
    def from_dict(data: dict) -> ManagedOrder:
        """Deserialize from persistence."""
        return ManagedOrder(
            order_id=UUID(data["order_id"]),
            broker_order_id=data.get("broker_order_id"),
            client_order_id=data["client_order_id"],
            ticker=data["ticker"],
            side=OrderSide(data["side"]),
            order_type=data["order_type"],
            time_in_force=data["time_in_force"],
            qty=Decimal(data["qty"]) if data.get("qty") is not None else None,
            notional=Decimal(data["notional"]) if data.get("notional") is not None else None,
            limit_price=Decimal(data["limit_price"]) if data.get("limit_price") is not None else None,
            stop_price=Decimal(data["stop_price"]) if data.get("stop_price") is not None else None,
            status=OrderStatus(data["status"]),
            filled_qty=Decimal(data.get("filled_qty", "0")),
            filled_avg_price=Decimal(data.get("filled_avg_price", "0")),
            source_intent_id=UUID(data["source_intent_id"]) if data.get("source_intent_id") else None,
            execution_strategy=data.get("execution_strategy", "MARKET"),
            target_weight=data.get("target_weight", 0.0),
            confidence=data.get("confidence", 0.0),
            max_slippage_bps=data.get("max_slippage_bps", 50),
            created_at=datetime.fromisoformat(data["created_at"]),
            submitted_at=datetime.fromisoformat(data["submitted_at"]) if data.get("submitted_at") else None,
            filled_at=datetime.fromisoformat(data["filled_at"]) if data.get("filled_at") else None,
            cancelled_at=datetime.fromisoformat(data["cancelled_at"]) if data.get("cancelled_at") else None,
            retry_count=data.get("retry_count", 0),
            max_retries=data.get("max_retries", 3),
            last_error=data.get("last_error"),
            transitions=data.get("transitions", []),
        )


class OrderManager:
    """Manages order lifecycle with persistence and recovery.

    Thread-safe. All state changes persisted to JSONL.
    Supports idempotent submission via client_order_id.
    """

    MAX_RETRIES = 3

    def __init__(self, *, persist_path: Path | None = None):
        self._lock = threading.RLock()
        self._orders: dict[UUID, ManagedOrder] = {}  # order_id -> ManagedOrder
        self._broker_id_map: dict[str, UUID] = {}  # broker_order_id -> order_id
        self._client_id_map: dict[str, UUID] = {}  # client_order_id -> order_id
        self._persist_path = persist_path

        # Load existing orders from persist_path if exists
        if persist_path and persist_path.exists():
            self._load()

    def create_from_routed_order(
        self,
        routed_order_dict: dict,
        portfolio_equity: Decimal,
    ) -> ManagedOrder:
        """Convert a RoutedOrder (from EXEC-ROUTER) into a ManagedOrder.

        Maps:
        - Action.OPEN_LONG -> side=BUY
        - Action.OPEN_SHORT -> side=SELL (short)
        - Action.CLOSE -> depends on current position (simplified: BUY for now)
        - Action.ADJUST -> depends on direction (BUY for LONG, SELL for SHORT)

        Converts target_weight to notional amount using portfolio_equity.
        Maps execution_strategy to order_type:
        - MARKET -> market
        - LIMIT -> limit (needs limit_price from broker)
        - TWAP -> market (simplified, no native TWAP in Alpaca basic)
        - VWAP -> market (simplified)

        Parameters
        ----------
        routed_order_dict : dict
            Dictionary with keys: order_id, ticker, action, direction,
            target_weight, confidence, source_intent_id, execution_strategy,
            urgency, time_horizon_days, max_slippage_bps
        portfolio_equity : Decimal
            Current portfolio equity for notional calculation

        Returns
        -------
        ManagedOrder
            New managed order instance
        """
        from providence.schemas.enums import Action, Direction

        order_id = UUID(routed_order_dict["order_id"])
        action = Action(routed_order_dict["action"])
        direction = Direction(routed_order_dict["direction"])
        ticker = routed_order_dict["ticker"]
        target_weight = routed_order_dict["target_weight"]
        confidence = routed_order_dict["confidence"]
        source_intent_id = UUID(routed_order_dict["source_intent_id"])
        execution_strategy = routed_order_dict.get("execution_strategy", "TWAP")
        max_slippage_bps = routed_order_dict.get("max_slippage_bps", 50)

        # Determine side and urgency from action/direction
        if action == Action.OPEN_LONG:
            side = OrderSide.BUY
        elif action == Action.OPEN_SHORT:
            side = OrderSide.SELL
        elif action == Action.CLOSE:
            # Simplified: assume closing a long position (BUY to cover short)
            # In practice, need to check current position
            side = OrderSide.BUY
        elif action == Action.ADJUST:
            if direction == Direction.LONG:
                side = OrderSide.BUY
            elif direction == Direction.SHORT:
                side = OrderSide.SELL
            else:
                # NEUTRAL: don't open a position
                side = OrderSide.BUY  # Default fallback
        else:
            side = OrderSide.BUY

        # Map execution strategy to order type
        execution_strategy_upper = execution_strategy.upper()
        if execution_strategy_upper == "LIMIT":
            order_type = "limit"
        elif execution_strategy_upper in ("TWAP", "VWAP"):
            # Simplified: use market orders (Alpaca basic doesn't have TWAP/VWAP native)
            order_type = "market"
        else:  # MARKET or default
            order_type = "market"

        # Determine time-in-force based on urgency
        urgency = routed_order_dict.get("urgency", "NORMAL").upper()
        if urgency == "IMMEDIATE":
            time_in_force = "ioc"  # Immediate or cancel
        elif urgency == "HIGH":
            time_in_force = "day"
        else:  # NORMAL, LOW
            time_in_force = "gtc"  # Good 'til cancelled

        # Calculate notional amount and convert to shares
        notional = portfolio_equity * Decimal(str(target_weight))
        # qty will be determined at execution time based on current price
        # For now, set to None and let broker client calculate

        # Generate idempotency key
        client_order_id = f"prov-{order_id}-{uuid4().hex[:8]}"

        managed_order = ManagedOrder(
            order_id=order_id,
            broker_order_id=None,
            client_order_id=client_order_id,
            ticker=ticker,
            side=side,
            order_type=order_type,
            time_in_force=time_in_force,
            qty=None,  # Will be set at execution
            notional=notional,
            limit_price=None,  # Will be set for LIMIT orders
            stop_price=None,
            source_intent_id=source_intent_id,
            execution_strategy=execution_strategy,
            target_weight=target_weight,
            confidence=confidence,
            max_slippage_bps=max_slippage_bps,
        )

        with self._lock:
            self._orders[order_id] = managed_order
            self._client_id_map[client_order_id] = order_id

        logger.info(
            "Order created from RoutedOrder",
            order_id=str(order_id),
            ticker=ticker,
            action=action.value,
            side=side.value,
            notional=str(notional),
            confidence=confidence,
        )

        if self._persist_path:
            self._persist(managed_order)

        return managed_order

    def mark_submitted(self, order_id: UUID, broker_order_id: str) -> None:
        """Mark order as submitted to broker.

        Parameters
        ----------
        order_id : UUID
            Providence order ID
        broker_order_id : str
            Alpaca's order ID
        """
        with self._lock:
            order = self._orders.get(order_id)
            if not order:
                logger.warning("Order not found for submission", order_id=str(order_id))
                return

            order.broker_order_id = broker_order_id
            self._broker_id_map[broker_order_id] = order_id

            order.transition_to(
                OrderStatus.SUBMITTED,
                reason=f"Submitted to broker: {broker_order_id}",
            )

            if self._persist_path:
                self._persist(order)

    def update_from_broker(self, order_id: UUID, broker_status: dict) -> None:
        """Update order state from broker status response.

        Maps Alpaca statuses:
        - new, accepted, pending_new -> SUBMITTED
        - partially_filled -> PARTIALLY_FILLED
        - filled -> FILLED
        - cancelled, pending_cancel -> CANCELLED
        - rejected -> REJECTED
        - expired -> EXPIRED
        - stopped, suspended -> FAILED

        Parameters
        ----------
        order_id : UUID
            Providence order ID
        broker_status : dict
            Status dict with keys: status, filled_qty, filled_avg_price, etc.
        """
        alpaca_status = broker_status.get("status", "").lower()

        # Map Alpaca status to OrderStatus
        status_map = {
            "new": OrderStatus.SUBMITTED,
            "accepted": OrderStatus.SUBMITTED,
            "pending_new": OrderStatus.SUBMITTED,
            "partially_filled": OrderStatus.PARTIALLY_FILLED,
            "filled": OrderStatus.FILLED,
            "cancelled": OrderStatus.CANCELLED,
            "pending_cancel": OrderStatus.CANCELLED,
            "rejected": OrderStatus.REJECTED,
            "expired": OrderStatus.EXPIRED,
            "stopped": OrderStatus.FAILED,
            "suspended": OrderStatus.FAILED,
        }

        new_status = status_map.get(alpaca_status, OrderStatus.SUBMITTED)

        with self._lock:
            order = self._orders.get(order_id)
            if not order:
                logger.warning(
                    "Order not found for broker update",
                    order_id=str(order_id),
                    broker_status=alpaca_status,
                )
                return

            # Only transition if state changes
            if order.status == new_status:
                logger.debug(
                    "Order status unchanged",
                    order_id=str(order_id),
                    status=new_status.value,
                )
                return

            # Extract broker data
            broker_data = {
                "alpaca_status": alpaca_status,
                "filled_qty": broker_status.get("filled_qty", 0),
                "filled_avg_price": broker_status.get("filled_avg_price", 0),
                "submitted_at": broker_status.get("submitted_at"),
                "filled_at": broker_status.get("filled_at"),
            }

            order.transition_to(
                new_status,
                reason=f"Broker update: {alpaca_status}",
                broker_data=broker_data,
            )

            if self._persist_path:
                self._persist(order)

    def mark_failed(self, order_id: UUID, error: str) -> None:
        """Mark order as failed with error message.

        Parameters
        ----------
        order_id : UUID
            Providence order ID
        error : str
            Error message
        """
        with self._lock:
            order = self._orders.get(order_id)
            if not order:
                logger.warning("Order not found for failure", order_id=str(order_id))
                return

            order.mark_failed(error)

            if self._persist_path:
                self._persist(order)

    def increment_retry_count(self, order_id: UUID) -> None:
        """Increment retry count for a failed order.

        Parameters
        ----------
        order_id : UUID
            Providence order ID
        """
        with self._lock:
            order = self._orders.get(order_id)
            if not order:
                return

            order.retry_count += 1
            logger.info(
                "Retry count incremented",
                order_id=str(order_id),
                retry_count=order.retry_count,
                max_retries=order.max_retries,
            )

            if self._persist_path:
                self._persist(order)

    def get_order(self, order_id: UUID) -> ManagedOrder | None:
        """Get a single order by ID.

        Parameters
        ----------
        order_id : UUID
            Providence order ID

        Returns
        -------
        ManagedOrder | None
            Order instance or None if not found
        """
        with self._lock:
            return self._orders.get(order_id)

    def get_by_broker_id(self, broker_order_id: str) -> ManagedOrder | None:
        """Get order by broker ID.

        Parameters
        ----------
        broker_order_id : str
            Alpaca order ID

        Returns
        -------
        ManagedOrder | None
            Order instance or None if not found
        """
        with self._lock:
            prov_id = self._broker_id_map.get(broker_order_id)
            if prov_id:
                return self._orders.get(prov_id)
            return None

    def get_by_client_id(self, client_order_id: str) -> ManagedOrder | None:
        """Get order by client ID (idempotency key).

        Parameters
        ----------
        client_order_id : str
            Client order ID

        Returns
        -------
        ManagedOrder | None
            Order instance or None if not found
        """
        with self._lock:
            prov_id = self._client_id_map.get(client_order_id)
            if prov_id:
                return self._orders.get(prov_id)
            return None

    def get_active_orders(self) -> list[ManagedOrder]:
        """Get all non-terminal orders.

        Returns
        -------
        list[ManagedOrder]
            Non-terminal orders sorted by created_at (oldest first)
        """
        with self._lock:
            active = [o for o in self._orders.values() if o.is_active]
            active.sort(key=lambda o: o.created_at)
            return active

    def get_pending_orders(self) -> list[ManagedOrder]:
        """Get orders waiting to be submitted.

        Returns
        -------
        list[ManagedOrder]
            PENDING status orders
        """
        with self._lock:
            pending = [o for o in self._orders.values() if o.status == OrderStatus.PENDING]
            pending.sort(key=lambda o: o.created_at)
            return pending

    def get_submitted_orders(self) -> list[ManagedOrder]:
        """Get orders that have been submitted but not filled.

        Returns
        -------
        list[ManagedOrder]
            SUBMITTED and PARTIALLY_FILLED orders
        """
        with self._lock:
            submitted = [
                o
                for o in self._orders.values()
                if o.status in (OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED)
            ]
            submitted.sort(key=lambda o: o.submitted_at or o.created_at)
            return submitted

    def get_retryable_orders(self) -> list[ManagedOrder]:
        """Get failed/rejected orders that can be retried.

        Returns
        -------
        list[ManagedOrder]
            Orders with can_retry == True
        """
        with self._lock:
            retryable = [o for o in self._orders.values() if o.can_retry]
            retryable.sort(key=lambda o: o.created_at)
            return retryable

    def get_orders_for_ticker(self, ticker: str) -> list[ManagedOrder]:
        """Get all orders for a specific ticker.

        Parameters
        ----------
        ticker : str
            Stock ticker symbol

        Returns
        -------
        list[ManagedOrder]
            Orders for that ticker, newest first
        """
        with self._lock:
            ticker_orders = [o for o in self._orders.values() if o.ticker == ticker]
            ticker_orders.sort(key=lambda o: o.created_at, reverse=True)
            return ticker_orders

    def get_recent_fills(self, limit: int = 50) -> list[ManagedOrder]:
        """Get recently filled orders, newest first.

        Parameters
        ----------
        limit : int
            Maximum number to return

        Returns
        -------
        list[ManagedOrder]
            Filled orders sorted by filled_at descending
        """
        with self._lock:
            filled = [o for o in self._orders.values() if o.status == OrderStatus.FILLED]
            filled.sort(key=lambda o: o.filled_at or o.created_at, reverse=True)
            return filled[:limit]

    @property
    def stats(self) -> dict[str, int]:
        """Get order counts by status.

        Returns
        -------
        dict[str, int]
            Counts of orders in each status
        """
        with self._lock:
            counts = {}
            for status in OrderStatus:
                counts[status.value] = sum(
                    1 for o in self._orders.values() if o.status == status
                )
            counts["total"] = len(self._orders)
            counts["active"] = sum(1 for o in self._orders.values() if o.is_active)
            return counts

    def _persist(self, order: ManagedOrder) -> None:
        """Append order state to JSONL.

        Parameters
        ----------
        order : ManagedOrder
            Order to persist
        """
        if not self._persist_path:
            return

        try:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._persist_path, "a") as f:
                f.write(json.dumps(order.to_dict(), default=str) + "\n")
        except OSError as exc:
            logger.error(
                "Failed to persist order",
                order_id=str(order.order_id),
                error=str(exc),
            )

    def _load(self) -> None:
        """Load orders from JSONL on startup. Rebuild index maps."""
        if not self._persist_path or not self._persist_path.exists():
            return

        count = 0
        try:
            with open(self._persist_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    order = ManagedOrder.from_dict(data)

                    if order.order_id not in self._orders:
                        self._orders[order.order_id] = order

                        # Rebuild maps
                        self._client_id_map[order.client_order_id] = order.order_id
                        if order.broker_order_id:
                            self._broker_id_map[order.broker_order_id] = order.order_id

                        count += 1
        except OSError as exc:
            logger.error(
                "Failed to load orders from disk",
                path=str(self._persist_path),
                error=str(exc),
            )
        except json.JSONDecodeError as exc:
            logger.error(
                "Failed to parse order JSON",
                path=str(self._persist_path),
                error=str(exc),
            )

        logger.info("Orders loaded from disk", count=count)
