"""Tests for PortfolioTracker at providence/portfolio/tracker.py."""
import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import pytest

from providence.portfolio.tracker import (
    Position,
    PositionSide,
    PortfolioSnapshot,
    PortfolioTracker,
)


class TestPosition:
    """Test Position dataclass."""

    def test_position_creation(self):
        """Position can be created."""
        pos = Position(
            ticker="AAPL",
            side=PositionSide.LONG,
            quantity=Decimal("100"),
            avg_entry_price=Decimal("150.00"),
            current_price=Decimal("155.00"),
            market_value=Decimal("15500.00"),
            unrealized_pnl=Decimal("500.00"),
            unrealized_pnl_pct=0.0333,
            realized_pnl=Decimal("0"),
            cost_basis=Decimal("15000.00"),
            weight=0.15,
            sector="Information Technology",
            opened_at=datetime.now(timezone.utc),
            last_updated=datetime.now(timezone.utc),
        )
        assert pos.ticker == "AAPL"
        assert pos.side == PositionSide.LONG

    def test_position_is_profitable_long(self):
        """is_profitable returns True for profitable long."""
        pos = Position(
            ticker="AAPL",
            side=PositionSide.LONG,
            quantity=Decimal("100"),
            avg_entry_price=Decimal("150.00"),
            current_price=Decimal("155.00"),
            market_value=Decimal("15500.00"),
            unrealized_pnl=Decimal("500.00"),
            unrealized_pnl_pct=0.0333,
            realized_pnl=Decimal("0"),
            cost_basis=Decimal("15000.00"),
            weight=0.15,
            sector="Tech",
            opened_at=datetime.now(timezone.utc),
            last_updated=datetime.now(timezone.utc),
        )
        assert pos.is_profitable is True

    def test_position_is_profitable_short(self):
        """is_profitable works for short positions."""
        pos = Position(
            ticker="AAPL",
            side=PositionSide.SHORT,
            quantity=Decimal("-100"),
            avg_entry_price=Decimal("155.00"),
            current_price=Decimal("150.00"),
            market_value=Decimal("-15000.00"),
            unrealized_pnl=Decimal("500.00"),
            unrealized_pnl_pct=0.0333,
            realized_pnl=Decimal("0"),
            cost_basis=Decimal("15500.00"),
            weight=-0.15,
            sector="Tech",
            opened_at=datetime.now(timezone.utc),
            last_updated=datetime.now(timezone.utc),
        )
        assert pos.is_profitable is True

    def test_position_is_profitable_flat(self):
        """is_profitable returns False for flat positions."""
        pos = Position(
            ticker="AAPL",
            side=PositionSide.FLAT,
            quantity=Decimal("0"),
            avg_entry_price=Decimal("0"),
            current_price=Decimal("0"),
            market_value=Decimal("0"),
            unrealized_pnl=Decimal("0"),
            unrealized_pnl_pct=0.0,
            realized_pnl=Decimal("0"),
            cost_basis=Decimal("0"),
            weight=0.0,
            sector="Tech",
            opened_at=datetime.now(timezone.utc),
            last_updated=datetime.now(timezone.utc),
        )
        assert pos.is_profitable is False

    def test_position_days_held(self):
        """days_held returns days since opened_at."""
        now = datetime.now(timezone.utc)
        # Create a position from 5 days ago
        five_days_ago = now.replace(day=now.day - 5) if now.day > 5 else now.replace(day=1)
        pos = Position(
            ticker="AAPL",
            side=PositionSide.LONG,
            quantity=Decimal("100"),
            avg_entry_price=Decimal("150.00"),
            current_price=Decimal("155.00"),
            market_value=Decimal("15500.00"),
            unrealized_pnl=Decimal("500.00"),
            unrealized_pnl_pct=0.0333,
            realized_pnl=Decimal("0"),
            cost_basis=Decimal("15000.00"),
            weight=0.15,
            sector="Tech",
            opened_at=five_days_ago,
            last_updated=now,
        )
        assert pos.days_held >= 0


class TestPortfolioSnapshot:
    """Test PortfolioSnapshot."""

    def test_snapshot_creation(self):
        """PortfolioSnapshot can be created."""
        snap = PortfolioSnapshot(
            snapshot_id=uuid4(),
            timestamp=datetime.now(timezone.utc),
            equity=Decimal("105000.00"),
            cash=Decimal("50000.00"),
            buying_power=Decimal("100000.00"),
            positions={},
            gross_exposure=0.0,
            net_exposure=0.0,
            long_exposure=0.0,
            short_exposure=0.0,
            sector_exposure={},
            position_count=0,
            total_unrealized_pnl=Decimal("0"),
            total_realized_pnl=Decimal("0"),
            peak_equity=Decimal("105000.00"),
            drawdown_pct=0.0,
        )
        assert snap.equity == Decimal("105000.00")

    def test_snapshot_to_dict(self):
        """to_dict serializes snapshot."""
        snap = PortfolioSnapshot(
            snapshot_id=uuid4(),
            timestamp=datetime.now(timezone.utc),
            equity=Decimal("105000.00"),
            cash=Decimal("50000.00"),
            buying_power=Decimal("100000.00"),
            positions={},
            gross_exposure=0.0,
            net_exposure=0.0,
            long_exposure=0.0,
            short_exposure=0.0,
            sector_exposure={},
            position_count=0,
            total_unrealized_pnl=Decimal("0"),
            total_realized_pnl=Decimal("0"),
            peak_equity=Decimal("105000.00"),
            drawdown_pct=0.0,
        )
        data = snap.to_dict()
        assert data["equity"] == "105000.00"
        assert data["position_count"] == 0


class TestPortfolioTrackerInit:
    """Test PortfolioTracker initialization."""

    def test_init_default_state(self):
        """Tracker initializes with default state."""
        tracker = PortfolioTracker()
        assert tracker.equity == Decimal("100000")
        assert tracker._peak_equity == Decimal("100000")

    def test_init_custom_equity(self):
        """Tracker can be initialized with custom equity."""
        tracker = PortfolioTracker(initial_equity=Decimal("250000"))
        assert tracker.equity == Decimal("250000")

    def test_init_with_persist_path(self):
        """Tracker creates persist path parent directories."""
        with TemporaryDirectory() as tmpdir:
            persist_path = Path(tmpdir) / "subdir" / "tracker.jsonl"
            tracker = PortfolioTracker(persist_path=persist_path)
            assert persist_path.parent.exists()


class TestSyncFromBroker:
    """Test sync_from_broker method."""

    def test_sync_updates_equity(self):
        """sync_from_broker updates equity."""
        tracker = PortfolioTracker(initial_equity=Decimal("100000"))
        account = {
            "equity": "105000.00",
            "cash": "50000.00",
            "buying_power": "100000.00",
        }
        positions = []

        snap = tracker.sync_from_broker(account, positions)

        assert tracker.equity == Decimal("105000.00")
        assert tracker._cash == Decimal("50000.00")

    def test_sync_creates_positions(self):
        """sync_from_broker creates positions from broker data."""
        tracker = PortfolioTracker()
        account = {
            "equity": "105000.00",
            "cash": "50000.00",
            "buying_power": "100000.00",
        }
        positions = [
            {
                "symbol": "AAPL",
                "qty": "100",
                "avg_entry_price": "150.00",
                "current_price": "155.00",
                "sector": "Information Technology",
            }
        ]

        snap = tracker.sync_from_broker(account, positions)

        assert "AAPL" in tracker.positions
        assert tracker.positions["AAPL"].quantity == Decimal("100")

    def test_sync_updates_peak_equity(self):
        """sync_from_broker updates peak equity."""
        tracker = PortfolioTracker(initial_equity=Decimal("100000"))
        account = {
            "equity": "110000.00",
            "cash": "50000.00",
            "buying_power": "110000.00",
        }
        positions = []

        tracker.sync_from_broker(account, positions)

        assert tracker._peak_equity == Decimal("110000.00")

    def test_sync_returns_snapshot(self):
        """sync_from_broker returns snapshot."""
        tracker = PortfolioTracker()
        account = {
            "equity": "105000.00",
            "cash": "50000.00",
            "buying_power": "100000.00",
        }
        positions = []

        snap = tracker.sync_from_broker(account, positions)

        assert isinstance(snap, PortfolioSnapshot)
        assert snap.equity == Decimal("105000.00")


class TestRecordFill:
    """Test record_fill method."""

    def test_record_fill_buy_creates_position(self):
        """record_fill creates position on buy."""
        tracker = PortfolioTracker()
        order_id = uuid4()

        tracker.record_fill(
            order_id=order_id,
            ticker="AAPL",
            side="BUY",
            qty=Decimal("100"),
            price=Decimal("150.00"),
            timestamp=datetime.now(timezone.utc),
        )

        assert "AAPL" in tracker.positions
        assert tracker.positions["AAPL"].quantity == Decimal("100")
        assert tracker.positions["AAPL"].side == PositionSide.LONG

    def test_record_fill_sell_closes_position(self):
        """record_fill closes position on sell."""
        tracker = PortfolioTracker()
        order_id1 = uuid4()
        order_id2 = uuid4()

        # Open position
        tracker.record_fill(
            order_id=order_id1,
            ticker="AAPL",
            side="BUY",
            qty=Decimal("100"),
            price=Decimal("150.00"),
            timestamp=datetime.now(timezone.utc),
        )

        # Close position
        tracker.record_fill(
            order_id=order_id2,
            ticker="AAPL",
            side="SELL",
            qty=Decimal("100"),
            price=Decimal("155.00"),
            timestamp=datetime.now(timezone.utc),
        )

        assert tracker.positions["AAPL"].quantity == Decimal("0")
        assert tracker.positions["AAPL"].side == PositionSide.FLAT

    def test_record_fill_partial_close(self):
        """record_fill handles partial closes."""
        tracker = PortfolioTracker()
        order_id1 = uuid4()
        order_id2 = uuid4()

        # Open position
        tracker.record_fill(
            order_id=order_id1,
            ticker="AAPL",
            side="BUY",
            qty=Decimal("100"),
            price=Decimal("150.00"),
            timestamp=datetime.now(timezone.utc),
        )

        # Partial close
        tracker.record_fill(
            order_id=order_id2,
            ticker="AAPL",
            side="SELL",
            qty=Decimal("30"),
            price=Decimal("155.00"),
            timestamp=datetime.now(timezone.utc),
        )

        assert tracker.positions["AAPL"].quantity == Decimal("70")
        assert tracker.positions["AAPL"].side == PositionSide.LONG

    def test_record_fill_updates_realized_pnl(self):
        """record_fill updates realized P&L on close."""
        tracker = PortfolioTracker()
        order_id1 = uuid4()
        order_id2 = uuid4()

        # Open at 150
        tracker.record_fill(
            order_id=order_id1,
            ticker="AAPL",
            side="BUY",
            qty=Decimal("100"),
            price=Decimal("150.00"),
            timestamp=datetime.now(timezone.utc),
        )

        # Close at 155 (profit of 5 * 100 = 500)
        tracker.record_fill(
            order_id=order_id2,
            ticker="AAPL",
            side="SELL",
            qty=Decimal("100"),
            price=Decimal("155.00"),
            timestamp=datetime.now(timezone.utc),
        )

        assert tracker.positions["AAPL"].realized_pnl == Decimal("500")


class TestUpdatePrice:
    """Test update_price method."""

    def test_update_price_long(self):
        """update_price updates unrealized P&L for long."""
        tracker = PortfolioTracker()
        order_id = uuid4()

        # Open position at 150
        tracker.record_fill(
            order_id=order_id,
            ticker="AAPL",
            side="BUY",
            qty=Decimal("100"),
            price=Decimal("150.00"),
            timestamp=datetime.now(timezone.utc),
        )

        # Update price to 160
        tracker.update_price("AAPL", Decimal("160.00"))

        pos = tracker.positions["AAPL"]
        assert pos.current_price == Decimal("160.00")
        assert pos.unrealized_pnl == Decimal("1000")  # 100 * (160 - 150)

    def test_update_price_short(self):
        """update_price updates unrealized P&L for short."""
        tracker = PortfolioTracker()
        order_id = uuid4()

        # Open short at 150
        tracker.record_fill(
            order_id=order_id,
            ticker="AAPL",
            side="SELL",
            qty=Decimal("100"),
            price=Decimal("150.00"),
            timestamp=datetime.now(timezone.utc),
        )

        # Update price to 145
        tracker.update_price("AAPL", Decimal("145.00"))

        pos = tracker.positions["AAPL"]
        assert pos.current_price == Decimal("145.00")
        assert pos.unrealized_pnl == Decimal("500")  # 100 * (150 - 145)


class TestSnapshot:
    """Test snapshot method."""

    def test_snapshot_captures_state(self):
        """snapshot captures current portfolio state."""
        tracker = PortfolioTracker()
        order_id = uuid4()

        tracker.record_fill(
            order_id=order_id,
            ticker="AAPL",
            side="BUY",
            qty=Decimal("100"),
            price=Decimal("150.00"),
            timestamp=datetime.now(timezone.utc),
        )

        snap = tracker.snapshot()

        assert snap.position_count >= 1
        assert "AAPL" in snap.positions

    def test_snapshot_drawdown_calculation(self):
        """snapshot calculates drawdown correctly."""
        tracker = PortfolioTracker(initial_equity=Decimal("100000"))
        # Peak is 100000

        # Simulate equity drop to 90000
        tracker._equity = Decimal("90000")
        snap = tracker.snapshot()

        expected_drawdown = float((Decimal("100000") - Decimal("90000")) / Decimal("100000") * 100)
        assert snap.drawdown_pct == pytest.approx(expected_drawdown, abs=0.1)


class TestExposure:
    """Test exposure calculation methods."""

    def test_gross_exposure(self):
        """gross_exposure sums absolute position weights."""
        tracker = PortfolioTracker(initial_equity=Decimal("100000"))
        tracker._equity = Decimal("100000")

        # Create position worth 10000 (10% weight)
        pos = Position(
            ticker="AAPL",
            side=PositionSide.LONG,
            quantity=Decimal("100"),
            avg_entry_price=Decimal("100.00"),
            current_price=Decimal("100.00"),
            market_value=Decimal("10000"),
            unrealized_pnl=Decimal("0"),
            unrealized_pnl_pct=0.0,
            realized_pnl=Decimal("0"),
            cost_basis=Decimal("10000"),
            weight=0.10,
            sector="Tech",
            opened_at=datetime.now(timezone.utc),
            last_updated=datetime.now(timezone.utc),
        )
        tracker._positions["AAPL"] = pos

        assert tracker.gross_exposure == pytest.approx(0.10, abs=0.01)

    def test_sector_exposure(self):
        """sector_exposure groups by sector."""
        tracker = PortfolioTracker()
        pos = Position(
            ticker="AAPL",
            side=PositionSide.LONG,
            quantity=Decimal("100"),
            avg_entry_price=Decimal("100.00"),
            current_price=Decimal("100.00"),
            market_value=Decimal("10000"),
            unrealized_pnl=Decimal("0"),
            unrealized_pnl_pct=0.0,
            realized_pnl=Decimal("0"),
            cost_basis=Decimal("10000"),
            weight=0.10,
            sector="Information Technology",
            opened_at=datetime.now(timezone.utc),
            last_updated=datetime.now(timezone.utc),
        )
        tracker._positions["AAPL"] = pos

        sector_exp = tracker.sector_exposure()
        assert "Information Technology" in sector_exp
        assert sector_exp["Information Technology"] == pytest.approx(0.10, abs=0.01)


class TestPersistence:
    """Test persistence to JSONL."""

    def test_persistence_snapshots(self):
        """Snapshots are persisted to JSONL."""
        with TemporaryDirectory() as tmpdir:
            persist_path = Path(tmpdir) / "tracker.jsonl"
            tracker = PortfolioTracker(persist_path=persist_path)

            snap = tracker.snapshot()

            assert persist_path.exists()
            with open(persist_path, "r") as f:
                lines = f.readlines()
                assert len(lines) > 0
                data = json.loads(lines[0])
                assert data["snapshot_id"] == str(snap.snapshot_id)

    def test_persistence_fills(self):
        """Fills are persisted to JSONL."""
        with TemporaryDirectory() as tmpdir:
            persist_path = Path(tmpdir) / "tracker.jsonl"
            tracker = PortfolioTracker(persist_path=persist_path)
            order_id = uuid4()

            tracker.record_fill(
                order_id=order_id,
                ticker="AAPL",
                side="BUY",
                qty=Decimal("100"),
                price=Decimal("150.00"),
                timestamp=datetime.now(timezone.utc),
            )

            assert persist_path.exists()
            with open(persist_path, "r") as f:
                lines = f.readlines()
                # Should have at least 1 fill record
                data = json.loads(lines[-1])
                assert data.get("type") == "fill" or "ticker" in data
