#!/usr/bin/env python3
"""Seed demo data for local testing.

Populates FragmentStore, BeliefStore, RunStore, ShadowSignalStore,
PortfolioTracker, and OrderManager with realistic mock data so the
API and dashboard have something to display.

Usage:
    python scripts/seed_demo_data.py [--data-dir DIR]
    python scripts/seed_demo_data.py  # uses data/demo in cwd
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from uuid import UUID, uuid4

# Ensure providence package is importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from providence.schemas.enums import (
    Action, DataType, Direction, SystemMode, ValidationStatus,
)
from providence.schemas.market_state import MarketStateFragment
from providence.schemas.shadow import ShadowSignal, ShadowRunSummary
from providence.storage.fragment_store import FragmentStore
from providence.storage.belief_store import BeliefStore
from providence.storage.run_store import RunStore
from providence.services.shadow_execution import ShadowSignalStore
from providence.portfolio.tracker import PortfolioTracker, PositionSide, Position
from providence.portfolio.order_manager import OrderManager, ManagedOrder, OrderStatus, OrderSide

# ── Constants ──────────────────────────────────────────────────────

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "JPM"]
SECTORS = {
    "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Communication Services",
    "AMZN": "Consumer Discretionary", "NVDA": "Technology", "TSLA": "Consumer Discretionary",
    "META": "Communication Services", "JPM": "Financials",
}
PRICES = {
    "AAPL": 185.0, "MSFT": 420.0, "GOOGL": 175.0, "AMZN": 195.0,
    "NVDA": 890.0, "TSLA": 175.0, "META": 525.0, "JPM": 210.0,
}
REGIMES = ["LOW_VOL_TRENDING", "HIGH_VOL_MEAN_REVERTING", "TRANSITION_UNCERTAIN", "LOW_VOL_TRENDING"]
RISK_MODES = ["NORMAL", "NORMAL", "CAUTIOUS", "NORMAL"]

NUM_RUNS = 12  # Pipeline runs to simulate


def _hash(data: dict) -> str:
    """SHA-256 content hash."""
    raw = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


def seed_fragments(store: FragmentStore, base_time: datetime) -> None:
    """Create PRICE and TECHNICAL fragments for each ticker across runs."""
    print("  Seeding fragments...")
    count = 0
    for run_idx in range(NUM_RUNS):
        ts = base_time + timedelta(hours=run_idx * 6)
        for ticker in TICKERS:
            base_price = PRICES[ticker]
            drift = random.uniform(-0.03, 0.05) * (run_idx / NUM_RUNS)
            price = round(base_price * (1 + drift + random.gauss(0, 0.01)), 2)

            payload = {
                "open": round(price * 0.998, 2),
                "high": round(price * 1.005, 2),
                "low": round(price * 0.995, 2),
                "close": price,
                "volume": random.randint(5_000_000, 50_000_000),
                "vwap": round(price * 1.001, 2),
            }
            frag = MarketStateFragment(
                agent_id="PERCEPT-PRICE",
                timestamp=ts,
                source_timestamp=ts - timedelta(minutes=5),
                entity=ticker,
                data_type=DataType.PRICE,
                source_hash=_hash(payload),
                payload=payload,
            )
            store.append(frag)
            count += 1

            # Technical fragment
            tech_payload = {
                "sma_20": round(base_price * (1 + drift * 0.8), 2),
                "ema_12": round(base_price * (1 + drift * 0.9), 2),
                "rsi_14": round(random.uniform(30, 70), 1),
                "macd_histogram": round(random.gauss(0, 2), 3),
                "bollinger_position": round(random.uniform(0.1, 0.9), 3),
                "momentum_10d": round(random.gauss(0.01, 0.03), 4),
            }
            tech_frag = MarketStateFragment(
                agent_id="COGNIT-TECHNICAL",
                timestamp=ts,
                source_timestamp=ts - timedelta(minutes=1),
                entity=ticker,
                data_type=DataType.TECHNICAL,
                source_hash=_hash(tech_payload),
                payload=tech_payload,
            )
            store.append(tech_frag)
            count += 1

    print(f"    {count} fragments created")


def seed_beliefs(store: BeliefStore, base_time: datetime) -> None:
    """Create BeliefObjects from various cognition agents."""
    from providence.schemas.belief import BeliefObject, Belief, InvalidationCondition
    from providence.schemas.enums import Magnitude, Operator

    print("  Seeding beliefs...")
    agents = [
        "COGNIT-TECHNICAL", "COGNIT-FUNDAMENTAL", "COGNIT-MACRO",
        "COGNIT-EVENT", "COGNIT-NARRATIVE", "COGNIT-CROSSSEC",
    ]
    count = 0
    for run_idx in range(NUM_RUNS):
        ts = base_time + timedelta(hours=run_idx * 6, minutes=30)
        for agent_id in agents:
            beliefs = []
            for ticker in random.sample(TICKERS, k=random.randint(2, 5)):
                direction = random.choice([Direction.LONG, Direction.SHORT, Direction.NEUTRAL])
                beliefs.append(Belief(
                    thesis_id=uuid4(),
                    ticker=ticker,
                    thesis_summary=f"{agent_id} view on {ticker}: {'bullish' if direction == Direction.LONG else 'bearish' if direction == Direction.SHORT else 'neutral'}",
                    direction=direction,
                    magnitude=random.choice(list(Magnitude)),
                    raw_confidence=round(random.uniform(0.3, 0.9), 2),
                    time_horizon_days=random.choice([5, 10, 20, 60]),
                    evidence=[f"Signal from {agent_id} analysis"],
                    invalidation_conditions=[
                        InvalidationCondition(
                            metric=f"{ticker}_price",
                            operator=Operator.GT if direction == Direction.LONG else Operator.LT,
                            threshold=PRICES[ticker] * (0.9 if direction == Direction.LONG else 1.1),
                        )
                    ],
                ))
            bo = BeliefObject(
                agent_id=agent_id,
                timestamp=ts,
                context_window_hash=_hash({"agent": agent_id, "run": run_idx}),
                beliefs=beliefs,
            )
            store.append(bo)
            count += 1
    print(f"    {count} belief objects created")


def seed_runs(store: RunStore, base_time: datetime) -> None:
    """Create PipelineRun records."""
    from providence.orchestration.models import PipelineRun, StageResult, StageStatus, RunStatus

    print("  Seeding pipeline runs...")
    for run_idx in range(NUM_RUNS):
        ts = base_time + timedelta(hours=run_idx * 6)
        stages = []
        for stage_name in ["COGNIT", "REGIME", "DECIDE", "EXEC"]:
            status = StageStatus.SUCCESS if random.random() > 0.1 else StageStatus.FAILED
            stages.append(StageResult(
                stage_name=stage_name,
                agent_id=f"{stage_name}-STAGE",
                status=status,
                started_at=ts,
                finished_at=ts + timedelta(seconds=random.randint(1, 30)),
                output={"mock": True},
            ))
        succeeded = sum(1 for s in stages if s.status == StageStatus.SUCCESS)
        run = PipelineRun(
            loop_type="main",
            status=RunStatus.COMPLETED if succeeded == len(stages) else RunStatus.PARTIAL,
            started_at=ts,
            finished_at=ts + timedelta(minutes=2),
            stage_results=stages,
        )
        store.append(run)
    print(f"    {NUM_RUNS} pipeline runs created")


def seed_shadow_signals(store: ShadowSignalStore, base_time: datetime) -> None:
    """Create ShadowSignals with realistic returns data."""
    print("  Seeding shadow signals...")
    count = 0
    for run_idx in range(NUM_RUNS):
        run_id = uuid4()
        ts = base_time + timedelta(hours=run_idx * 6, minutes=45)
        signals_in_run = []

        for ticker in random.sample(TICKERS, k=random.randint(2, 5)):
            base_price = PRICES[ticker]
            drift = random.gauss(0.002, 0.01)
            direction = Direction.LONG if random.random() > 0.35 else Direction.SHORT
            action = Action.OPEN_LONG if direction == Direction.LONG else Action.OPEN_SHORT
            approved = random.random() > 0.2
            confidence = round(random.uniform(0.4, 0.9), 2)

            # Simulate returns (direction-aware)
            sign = 1 if direction == Direction.LONG else -1
            ret_1d = round(sign * random.gauss(0.003, 0.015), 4) if run_idx < NUM_RUNS - 2 else None
            ret_5d = round(sign * random.gauss(0.008, 0.025), 4) if run_idx < NUM_RUNS - 4 else None
            ret_20d = round(sign * random.gauss(0.02, 0.04), 4) if run_idx < NUM_RUNS - 8 else None

            entry_price = round(base_price * (1 + random.gauss(0, 0.005)), 2) if approved else None

            signal = ShadowSignal(
                run_id=run_id,
                timestamp=ts,
                ticker=ticker,
                action=action,
                direction=direction,
                target_weight=round(random.uniform(0.02, 0.08), 3),
                confidence=confidence,
                approved=approved,
                rejection_reasons=[] if approved else [random.choice([
                    "Risk mode: CAUTIOUS limits new positions",
                    "Sector concentration exceeded",
                    "Low confidence below threshold",
                ])],
                adjusted_weight=round(random.uniform(0.01, 0.06), 3) if approved else 0.0,
                risk_mode_applied=RISK_MODES[run_idx % len(RISK_MODES)],
                simulated_entry_price=entry_price,
                simulated_fill_qty=random.randint(10, 200) if approved else None,
                simulated_notional=round(random.uniform(2000, 15000), 2) if approved else None,
                price_at_signal=round(base_price * (1 + random.gauss(0, 0.003)), 2),
                price_1d_later=round(base_price * (1 + (ret_1d or 0)), 2) if ret_1d else None,
                price_5d_later=round(base_price * (1 + (ret_5d or 0)), 2) if ret_5d else None,
                price_20d_later=round(base_price * (1 + (ret_20d or 0)), 2) if ret_20d else None,
                realized_return_1d=ret_1d,
                realized_return_5d=ret_5d,
                realized_return_20d=ret_20d,
            )
            store.append(signal)
            signals_in_run.append(signal)
            count += 1

        # Create run summary
        approved_count = sum(1 for s in signals_in_run if s.approved)
        long_count = sum(1 for s in signals_in_run if s.direction == Direction.LONG)
        summary = ShadowRunSummary(
            run_id=run_id,
            timestamp=ts,
            system_mode=SystemMode.SHADOW,
            signals=signals_in_run,
            total_signals=len(signals_in_run),
            approved_signals=approved_count,
            rejected_signals=len(signals_in_run) - approved_count,
            long_signals=long_count,
            short_signals=len(signals_in_run) - long_count,
            risk_mode=RISK_MODES[run_idx % len(RISK_MODES)],
            regime_state=REGIMES[run_idx % len(REGIMES)],
        )
        store.append_summary(summary)

    print(f"    {count} shadow signals + {NUM_RUNS} run summaries created")


def seed_portfolio(tracker: PortfolioTracker) -> None:
    """Seed portfolio with a few open positions."""
    print("  Seeding portfolio positions...")
    now = datetime.now(timezone.utc)
    positions = [
        ("AAPL", PositionSide.LONG, Decimal("50"), Decimal("178.50"), Decimal("185.20")),
        ("NVDA", PositionSide.LONG, Decimal("15"), Decimal("845.00"), Decimal("892.30")),
        ("TSLA", PositionSide.SHORT, Decimal("-20"), Decimal("190.00"), Decimal("174.80")),
        ("META", PositionSide.LONG, Decimal("25"), Decimal("510.00"), Decimal("527.40")),
    ]
    for ticker, side, qty, entry, current in positions:
        cost_basis = abs(qty) * entry
        mv = qty * current
        if side == PositionSide.LONG:
            pnl = abs(qty) * current - cost_basis
        else:
            pnl = cost_basis - abs(qty) * current
        pnl_pct = float(pnl / cost_basis * 100) if cost_basis > 0 else 0.0
        tracker._positions[ticker] = Position(
            ticker=ticker,
            side=side,
            quantity=qty,
            avg_entry_price=entry,
            current_price=current,
            market_value=mv,
            unrealized_pnl=pnl,
            unrealized_pnl_pct=pnl_pct,
            realized_pnl=Decimal("0"),
            cost_basis=cost_basis,
            weight=0.0,
            sector=SECTORS.get(ticker, ""),
            opened_at=now - timedelta(days=random.randint(3, 25)),
            last_updated=now,
        )
    tracker._equity = Decimal("105420.50")
    tracker._cash = Decimal("45200.30")
    tracker._buying_power = Decimal("45200.30")
    tracker._peak_equity = Decimal("108000.00")
    snap = tracker.snapshot()
    print(f"    {len(positions)} positions seeded, equity=${tracker.equity}")


def seed_orders(manager: OrderManager) -> None:
    """Seed some filled and active orders."""
    print("  Seeding orders...")
    now = datetime.now(timezone.utc)
    orders = [
        ("AAPL", OrderSide.BUY, "50", "178.50", OrderStatus.FILLED),
        ("NVDA", OrderSide.BUY, "15", "845.00", OrderStatus.FILLED),
        ("TSLA", OrderSide.SELL, "20", "190.00", OrderStatus.FILLED),
        ("META", OrderSide.BUY, "25", "510.00", OrderStatus.FILLED),
        ("GOOGL", OrderSide.BUY, "30", None, OrderStatus.SUBMITTED),
        ("AMZN", OrderSide.BUY, "10", None, OrderStatus.PENDING),
    ]
    for ticker, side, qty, fill_price, status in orders:
        oid = uuid4()
        order = ManagedOrder(
            order_id=oid,
            broker_order_id=f"alp-{uuid4().hex[:8]}" if status != OrderStatus.PENDING else None,
            client_order_id=f"prov-{oid}-{uuid4().hex[:8]}",
            ticker=ticker,
            side=side,
            order_type="market",
            time_in_force="gtc",
            qty=Decimal(qty),
            notional=None,
            limit_price=None,
            stop_price=None,
            status=OrderStatus.PENDING,  # start at pending
            execution_strategy="MARKET",
            target_weight=round(random.uniform(0.03, 0.08), 3),
            confidence=round(random.uniform(0.5, 0.85), 2),
            created_at=now - timedelta(days=random.randint(1, 20)),
        )
        manager._orders[oid] = order
        manager._client_id_map[order.client_order_id] = oid
        if order.broker_order_id:
            manager._broker_id_map[order.broker_order_id] = oid

        # Transition to target status
        if status in (OrderStatus.SUBMITTED, OrderStatus.FILLED):
            order.status = OrderStatus.SUBMITTED
            order.submitted_at = order.created_at + timedelta(seconds=2)
        if status == OrderStatus.FILLED and fill_price:
            order.status = OrderStatus.FILLED
            order.filled_qty = Decimal(qty)
            order.filled_avg_price = Decimal(fill_price)
            order.filled_at = order.submitted_at + timedelta(seconds=random.randint(1, 30))

    print(f"    {len(orders)} orders seeded")


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed Providence demo data")
    parser.add_argument("--data-dir", type=Path, default=Path("data/demo"),
                        help="Directory for JSONL data files")
    args = parser.parse_args()

    data_dir = args.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*50}")
    print(f" Providence Demo Data Seeder")
    print(f" Data directory: {data_dir.resolve()}")
    print(f"{'='*50}\n")

    random.seed(42)  # Reproducible
    base_time = datetime.now(timezone.utc) - timedelta(days=3)

    # Create stores with persistence
    frag_store = FragmentStore(persist_path=data_dir / "fragments.jsonl")
    belief_store = BeliefStore(persist_path=data_dir / "beliefs.jsonl")
    run_store = RunStore(persist_path=data_dir / "runs.jsonl")
    shadow_store = ShadowSignalStore(persist_path=data_dir / "shadow_signals.jsonl")
    portfolio = PortfolioTracker(persist_path=data_dir / "portfolio.jsonl")
    orders = OrderManager(persist_path=data_dir / "orders.jsonl")

    seed_fragments(frag_store, base_time)
    seed_beliefs(belief_store, base_time)
    seed_runs(run_store, base_time)
    seed_shadow_signals(shadow_store, base_time)
    seed_portfolio(portfolio)
    seed_orders(orders)

    print(f"\n{'='*50}")
    print(f" Seeding complete!")
    print(f" Files written to: {data_dir.resolve()}")
    print(f"")
    print(f" Start the API server:")
    print(f"   python -m providence.api.server --data-dir {data_dir} --skip-perception --skip-adaptive")
    print(f"")
    print(f" Then open the dashboard:")
    print(f"   http://localhost:8000/dashboard")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
