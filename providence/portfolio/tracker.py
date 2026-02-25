"""Portfolio position tracker — tracks live positions, P&L, and exposure.

Maintains current portfolio state by reconciling with broker positions
and tracking order fills. All state changes are logged immutably.
"""
from __future__ import annotations
import json
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Optional
from uuid import UUID, uuid4

try:
    import structlog
    logger = structlog.get_logger()
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class PositionSide(str, Enum):
    """Position side enumeration."""
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


@dataclass
class Position:
    """A single position in the portfolio."""
    ticker: str
    side: PositionSide
    quantity: Decimal  # signed: positive=long, negative=short
    avg_entry_price: Decimal
    current_price: Decimal
    market_value: Decimal  # quantity * current_price
    unrealized_pnl: Decimal
    unrealized_pnl_pct: float
    realized_pnl: Decimal  # from partial closes
    cost_basis: Decimal  # abs(quantity) * avg_entry_price
    weight: float  # market_value / portfolio_equity
    sector: str
    opened_at: datetime
    last_updated: datetime
    
    @property
    def is_profitable(self) -> bool:
        """Returns True if position is in profit."""
        if self.side == PositionSide.FLAT:
            return False
        if self.side == PositionSide.LONG:
            return self.unrealized_pnl > 0
        else:  # SHORT
            return self.unrealized_pnl > 0
    
    @property
    def days_held(self) -> int:
        """Number of days position has been held."""
        delta = datetime.now(timezone.utc) - self.opened_at
        return delta.days


@dataclass
class PortfolioSnapshot:
    """Point-in-time snapshot of the entire portfolio."""
    snapshot_id: UUID
    timestamp: datetime
    equity: Decimal  # Total account equity
    cash: Decimal
    buying_power: Decimal
    positions: dict[str, Position]  # ticker -> Position
    gross_exposure: float  # sum(abs(weight))
    net_exposure: float  # sum(signed weight)
    long_exposure: float
    short_exposure: float
    sector_exposure: dict[str, float]  # sector -> net weight
    position_count: int
    total_unrealized_pnl: Decimal
    total_realized_pnl: Decimal
    peak_equity: Decimal  # high-water mark
    drawdown_pct: float  # current drawdown from peak
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize for storage."""
        return {
            "snapshot_id": str(self.snapshot_id),
            "timestamp": self.timestamp.isoformat(),
            "equity": str(self.equity),
            "cash": str(self.cash),
            "buying_power": str(self.buying_power),
            "positions": {
                ticker: {
                    "ticker": pos.ticker,
                    "side": pos.side.value,
                    "quantity": str(pos.quantity),
                    "avg_entry_price": str(pos.avg_entry_price),
                    "current_price": str(pos.current_price),
                    "market_value": str(pos.market_value),
                    "unrealized_pnl": str(pos.unrealized_pnl),
                    "unrealized_pnl_pct": pos.unrealized_pnl_pct,
                    "realized_pnl": str(pos.realized_pnl),
                    "cost_basis": str(pos.cost_basis),
                    "weight": pos.weight,
                    "sector": pos.sector,
                    "opened_at": pos.opened_at.isoformat(),
                    "last_updated": pos.last_updated.isoformat(),
                    "days_held": pos.days_held,
                    "is_profitable": pos.is_profitable,
                }
                for ticker, pos in self.positions.items()
            },
            "gross_exposure": self.gross_exposure,
            "net_exposure": self.net_exposure,
            "long_exposure": self.long_exposure,
            "short_exposure": self.short_exposure,
            "sector_exposure": self.sector_exposure,
            "position_count": self.position_count,
            "total_unrealized_pnl": str(self.total_unrealized_pnl),
            "total_realized_pnl": str(self.total_realized_pnl),
            "peak_equity": str(self.peak_equity),
            "drawdown_pct": self.drawdown_pct,
        }


class PortfolioTracker:
    """Tracks portfolio state with thread-safe updates.
    
    State flows:
    1. sync_from_broker() — pull positions from AlpacaClient
    2. update_prices() — mark-to-market with latest prices
    3. record_fill() — process an order fill
    4. snapshot() — capture current state
    """
    
    def __init__(
        self,
        *,
        persist_path: Optional[Path] = None,
        initial_equity: Decimal = Decimal("100000"),
    ):
        """Initialize portfolio tracker.
        
        Args:
            persist_path: Optional JSONL file to append snapshots/fills to
            initial_equity: Starting account equity
        """
        self._lock = threading.RLock()
        
        # Thread-safe state
        self._positions: dict[str, Position] = {}
        self._equity = initial_equity
        self._cash = initial_equity
        self._buying_power = initial_equity
        self._peak_equity = initial_equity
        self._realized_pnl_total = Decimal("0")
        
        # Immutable history
        self._snapshots: list[PortfolioSnapshot] = []
        self._fills: list[dict[str, Any]] = []
        
        # Optional persistence
        self._persist_path = persist_path
        if self._persist_path:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
    
    def sync_from_broker(
        self,
        account: dict[str, Any],
        positions: list[dict[str, Any]],
    ) -> PortfolioSnapshot:
        """Reconcile with broker state.
        
        Args:
            account: AlpacaClient.get_account() response
            positions: AlpacaClient.list_positions() response
            
        Returns:
            Updated snapshot after reconciliation
        """
        with self._lock:
            # Update account-level values
            self._equity = Decimal(str(account.get("equity", self._equity)))
            self._cash = Decimal(str(account.get("cash", self._cash)))
            self._buying_power = Decimal(str(account.get("buying_power", self._buying_power)))
            
            # Update peak equity (high-water mark)
            if self._equity > self._peak_equity:
                self._peak_equity = self._equity
            
            # Reconcile positions
            broker_tickers = set()
            for pos_data in positions:
                ticker = pos_data["symbol"]
                broker_tickers.add(ticker)
                
                qty = Decimal(str(pos_data["qty"]))
                current_price = Decimal(str(pos_data["current_price"]))
                avg_entry = Decimal(str(pos_data["avg_entry_price"]))
                
                # Determine side
                if qty > 0:
                    side = PositionSide.LONG
                elif qty < 0:
                    side = PositionSide.SHORT
                else:
                    side = PositionSide.FLAT
                
                # Calculate P&L
                market_value = qty * current_price
                cost_basis = abs(qty) * avg_entry
                
                if side == PositionSide.LONG:
                    unrealized_pnl = market_value - cost_basis
                elif side == PositionSide.SHORT:
                    unrealized_pnl = cost_basis - market_value
                else:
                    unrealized_pnl = Decimal("0")
                
                unrealized_pnl_pct = float(
                    (unrealized_pnl / cost_basis * 100)
                    if cost_basis > 0
                    else 0
                )
                
                # Create/update position
                now = datetime.now(timezone.utc)
                old_pos = self._positions.get(ticker)
                
                self._positions[ticker] = Position(
                    ticker=ticker,
                    side=side,
                    quantity=qty,
                    avg_entry_price=avg_entry,
                    current_price=current_price,
                    market_value=market_value,
                    unrealized_pnl=unrealized_pnl,
                    unrealized_pnl_pct=unrealized_pnl_pct,
                    realized_pnl=old_pos.realized_pnl if old_pos else Decimal("0"),
                    cost_basis=cost_basis,
                    weight=0.0,  # Calculated in snapshot
                    sector=pos_data.get("sector", ""),
                    opened_at=old_pos.opened_at if old_pos else now,
                    last_updated=now,
                )
            
            # Remove positions that broker no longer reports
            for ticker in list(self._positions.keys()):
                if ticker not in broker_tickers and self._positions[ticker].side == PositionSide.FLAT:
                    del self._positions[ticker]
            
            # Create and return snapshot
            snap = self._create_snapshot()
            self._snapshots.append(snap)
            self._persist_snapshot(snap)
            
            logger.info(
                "portfolio_sync",
                equity=str(self._equity),
                cash=str(self._cash),
                positions=len(self._positions),
                snapshot_id=str(snap.snapshot_id),
            )
            
            return snap
    
    def record_fill(
        self,
        order_id: UUID,
        ticker: str,
        side: str,
        qty: Decimal,
        price: Decimal,
        timestamp: datetime,
    ) -> None:
        """Record an order fill. Updates position and P&L.
        
        Args:
            order_id: Unique order identifier
            ticker: Security ticker
            side: "BUY" or "SELL"
            qty: Absolute quantity filled
            price: Fill price
            timestamp: When fill occurred
        """
        with self._lock:
            qty_signed = qty if side.upper() == "BUY" else -qty
            
            old_pos = self._positions.get(ticker)
            now = datetime.now(timezone.utc)
            
            if not old_pos or old_pos.side == PositionSide.FLAT:
                # Opening new position
                cost_basis = qty * price
                self._positions[ticker] = Position(
                    ticker=ticker,
                    side=PositionSide.LONG if qty_signed > 0 else PositionSide.SHORT,
                    quantity=qty_signed,
                    avg_entry_price=price,
                    current_price=price,
                    market_value=qty_signed * price,
                    unrealized_pnl=Decimal("0"),
                    unrealized_pnl_pct=0.0,
                    realized_pnl=Decimal("0"),
                    cost_basis=cost_basis,
                    weight=0.0,
                    sector="",
                    opened_at=now,
                    last_updated=now,
                )
            else:
                # Modifying existing position
                old_qty = old_pos.quantity
                new_qty = old_qty + qty_signed
                
                # Update realized P&L if closing part of position
                if (old_qty > 0 and qty_signed < 0) or (old_qty < 0 and qty_signed > 0):
                    close_qty = min(abs(old_qty), abs(qty_signed))
                    if old_pos.side == PositionSide.LONG:
                        realized_close = close_qty * (price - old_pos.avg_entry_price)
                    else:
                        realized_close = close_qty * (old_pos.avg_entry_price - price)
                    realized_pnl = old_pos.realized_pnl + realized_close
                else:
                    realized_pnl = old_pos.realized_pnl
                
                # Recalculate average entry price
                if new_qty == 0:
                    avg_entry = old_pos.avg_entry_price
                    new_side = PositionSide.FLAT
                elif new_qty > 0:
                    if old_qty >= 0:
                        # Adding to long
                        total_cost = (old_pos.cost_basis + qty * price)
                        avg_entry = total_cost / new_qty
                    else:
                        # Covering short and going long
                        avg_entry = price
                    new_side = PositionSide.LONG
                else:  # new_qty < 0
                    if old_qty <= 0:
                        # Adding to short
                        total_cost = (old_pos.cost_basis + qty * price)
                        avg_entry = total_cost / abs(new_qty)
                    else:
                        # Selling long and going short
                        avg_entry = price
                    new_side = PositionSide.SHORT
                
                cost_basis = abs(new_qty) * avg_entry
                market_value = new_qty * old_pos.current_price
                
                if new_side == PositionSide.FLAT:
                    unrealized_pnl = Decimal("0")
                    unrealized_pnl_pct = 0.0
                else:
                    unrealized_pnl = market_value - cost_basis if new_side == PositionSide.LONG else cost_basis - market_value
                    unrealized_pnl_pct = float(
                        (unrealized_pnl / cost_basis * 100) if cost_basis > 0 else 0
                    )
                
                self._positions[ticker] = Position(
                    ticker=ticker,
                    side=new_side,
                    quantity=new_qty,
                    avg_entry_price=avg_entry,
                    current_price=old_pos.current_price,
                    market_value=market_value,
                    unrealized_pnl=unrealized_pnl,
                    unrealized_pnl_pct=unrealized_pnl_pct,
                    realized_pnl=realized_pnl,
                    cost_basis=cost_basis,
                    weight=0.0,
                    sector=old_pos.sector,
                    opened_at=old_pos.opened_at,
                    last_updated=now,
                )
            
            # Log fill
            fill_record = {
                "order_id": str(order_id),
                "timestamp": timestamp.isoformat(),
                "ticker": ticker,
                "side": side,
                "qty": str(qty),
                "price": str(price),
            }
            self._fills.append(fill_record)
            self._persist_fill(fill_record)
            
            logger.info(
                "fill_recorded",
                order_id=str(order_id),
                ticker=ticker,
                side=side,
                qty=str(qty),
                price=str(price),
            )
    
    def update_price(self, ticker: str, price: Decimal) -> None:
        """Mark-to-market a single position.
        
        Args:
            ticker: Security ticker
            price: Current market price
        """
        with self._lock:
            if ticker not in self._positions:
                return
            
            pos = self._positions[ticker]
            if pos.side == PositionSide.FLAT:
                return
            
            market_value = pos.quantity * price
            cost_basis = pos.cost_basis
            
            if pos.side == PositionSide.LONG:
                unrealized_pnl = market_value - cost_basis
            else:  # SHORT
                unrealized_pnl = cost_basis - market_value
            
            unrealized_pnl_pct = float(
                (unrealized_pnl / cost_basis * 100) if cost_basis > 0 else 0
            )
            
            self._positions[ticker] = Position(
                ticker=ticker,
                side=pos.side,
                quantity=pos.quantity,
                avg_entry_price=pos.avg_entry_price,
                current_price=price,
                market_value=market_value,
                unrealized_pnl=unrealized_pnl,
                unrealized_pnl_pct=unrealized_pnl_pct,
                realized_pnl=pos.realized_pnl,
                cost_basis=cost_basis,
                weight=0.0,
                sector=pos.sector,
                opened_at=pos.opened_at,
                last_updated=datetime.now(timezone.utc),
            )
    
    def snapshot(self) -> PortfolioSnapshot:
        """Capture current portfolio state."""
        with self._lock:
            snap = self._create_snapshot()
            self._snapshots.append(snap)
            self._persist_snapshot(snap)
            return snap
    
    def get_position(self, ticker: str) -> Optional[Position]:
        """Get position by ticker."""
        with self._lock:
            return self._positions.get(ticker)
    
    @property
    def positions(self) -> dict[str, Position]:
        """Copy of current positions."""
        with self._lock:
            return dict(self._positions)
    
    @property
    def equity(self) -> Decimal:
        """Current account equity."""
        with self._lock:
            return self._equity
    
    @property
    def drawdown_pct(self) -> float:
        """Current drawdown from peak."""
        with self._lock:
            if self._peak_equity <= 0:
                return 0.0
            return float((self._peak_equity - self._equity) / self._peak_equity * 100)
    
    @property
    def gross_exposure(self) -> float:
        """Sum of absolute position weights."""
        with self._lock:
            if self._equity <= 0:
                return 0.0
            return sum(
                abs(pos.weight)
                for pos in self._positions.values()
            )
    
    def sector_exposure(self) -> dict[str, float]:
        """Net exposure by sector."""
        with self._lock:
            exposure: dict[str, float] = {}
            for pos in self._positions.values():
                if pos.sector:
                    exposure[pos.sector] = exposure.get(pos.sector, 0.0) + pos.weight
            return exposure
    
    def _create_snapshot(self) -> PortfolioSnapshot:
        """Create snapshot from current state."""
        if self._equity <= 0:
            equity_for_weight = Decimal("1")  # Avoid div by zero
        else:
            equity_for_weight = self._equity
        
        positions_with_weight = {}
        gross_exp = 0.0
        net_exp = 0.0
        long_exp = 0.0
        short_exp = 0.0
        total_unrealized = Decimal("0")
        
        for pos in self._positions.values():
            if pos.side != PositionSide.FLAT:
                weight = float(pos.market_value / equity_for_weight)
                updated_pos = Position(
                    ticker=pos.ticker,
                    side=pos.side,
                    quantity=pos.quantity,
                    avg_entry_price=pos.avg_entry_price,
                    current_price=pos.current_price,
                    market_value=pos.market_value,
                    unrealized_pnl=pos.unrealized_pnl,
                    unrealized_pnl_pct=pos.unrealized_pnl_pct,
                    realized_pnl=pos.realized_pnl,
                    cost_basis=pos.cost_basis,
                    weight=weight,
                    sector=pos.sector,
                    opened_at=pos.opened_at,
                    last_updated=pos.last_updated,
                )
                positions_with_weight[pos.ticker] = updated_pos
                
                gross_exp += abs(weight)
                net_exp += weight
                if weight > 0:
                    long_exp += weight
                else:
                    short_exp += abs(weight)
                
                total_unrealized += pos.unrealized_pnl
        
        sector_exp = {}
        for pos in positions_with_weight.values():
            if pos.sector:
                sector_exp[pos.sector] = sector_exp.get(pos.sector, 0.0) + pos.weight
        
        drawdown = float(
            (self._peak_equity - self._equity) / self._peak_equity * 100
            if self._peak_equity > 0
            else 0.0
        )
        
        return PortfolioSnapshot(
            snapshot_id=uuid4(),
            timestamp=datetime.now(timezone.utc),
            equity=self._equity,
            cash=self._cash,
            buying_power=self._buying_power,
            positions=positions_with_weight,
            gross_exposure=gross_exp,
            net_exposure=net_exp,
            long_exposure=long_exp,
            short_exposure=short_exp,
            sector_exposure=sector_exp,
            position_count=len(positions_with_weight),
            total_unrealized_pnl=total_unrealized,
            total_realized_pnl=self._realized_pnl_total,
            peak_equity=self._peak_equity,
            drawdown_pct=drawdown,
        )
    
    def _persist_snapshot(self, snap: PortfolioSnapshot) -> None:
        """Append snapshot to JSONL file."""
        if not self._persist_path:
            return
        
        try:
            with open(self._persist_path, "a") as f:
                f.write(json.dumps(snap.to_dict()) + "\n")
        except Exception as e:
            logger.error("snapshot_persist_error", error=str(e))
    
    def _persist_fill(self, fill: dict[str, Any]) -> None:
        """Append fill to JSONL file."""
        if not self._persist_path:
            return
        
        try:
            with open(self._persist_path, "a") as f:
                f.write(json.dumps({"type": "fill", **fill}) + "\n")
        except Exception as e:
            logger.error("fill_persist_error", error=str(e))
