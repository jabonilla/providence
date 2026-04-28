"""Seed demo data endpoint.

POST /api/v1/seed — populates all stores with realistic demo data
so the portal dashboard has something to display.
"""
from __future__ import annotations

import hashlib
import json
import random
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from fastapi import APIRouter, HTTPException

from providence.api.deps import get_state

router = APIRouter(tags=["seed"])

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
AGENTS = [
    "COGNIT-FUNDAMENTAL", "COGNIT-TECHNICAL", "COGNIT-KRONOS",
    "COGNIT-MACRO", "COGNIT-EVENT", "COGNIT-NARRATIVE", "COGNIT-CROSSSEC",
]
NUM_RUNS = 12


def _hash(data: dict) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()


def _seed_fragments(state) -> int:
    """Seed MarketStateFragment objects into FragmentStore."""
    from providence.schemas.enums import DataType, ValidationStatus
    from providence.schemas.market_state import MarketStateFragment

    random.seed(42)
    store = state.fragment_store
    count = 0
    now = datetime.now(timezone.utc)

    for run_idx in range(NUM_RUNS):
        ts = now - timedelta(hours=NUM_RUNS - run_idx)
        for ticker in TICKERS:
            price = PRICES[ticker] * (1 + random.uniform(-0.03, 0.03))
            # PRICE fragment
            payload = {
                "ticker": ticker, "open": round(price * 0.998, 2),
                "high": round(price * 1.01, 2), "low": round(price * 0.99, 2),
                "close": round(price, 2), "volume": random.randint(10_000_000, 80_000_000),
                "vwap": round(price * 0.999, 2),
            }
            frag = MarketStateFragment(
                fragment_id=str(uuid4()), agent_id="PERCEPT-PRICE",
                timestamp=ts, source_timestamp=ts,
                version=_hash(payload), entity=ticker,
                data_type=DataType.PRICE_OHLCV, schema_version="1.0.0",
                source_hash=_hash({"source": "polygon", "ticker": ticker}),
                validation_status=ValidationStatus.VALID, payload=payload,
            )
            store.append(frag)
            count += 1

            # NEWS/SENTIMENT fragment (use SENTIMENT_NEWS type)
            news_payload = {
                "ticker": ticker,
                "headline": f"{ticker} shows {'strong' if random.random() > 0.5 else 'mixed'} momentum",
                "sentiment_score": round(random.uniform(-0.5, 0.8), 2),
                "source": "aggregated",
                "article_count": random.randint(3, 15),
            }
            frag2 = MarketStateFragment(
                fragment_id=str(uuid4()), agent_id="PERCEPT-NEWS",
                timestamp=ts, source_timestamp=ts,
                version=_hash(news_payload), entity=ticker,
                data_type=DataType.SENTIMENT_NEWS, schema_version="1.0.0",
                source_hash=_hash({"source": "news", "ticker": ticker}),
                validation_status=ValidationStatus.VALID, payload=news_payload,
            )
            store.append(frag2)
            count += 1

    return count


def _seed_beliefs(state) -> int:
    """Seed BeliefObject records into BeliefStore."""
    from providence.schemas.belief import BeliefObject, Belief, EvidenceRef, InvalidationCondition, BeliefMetadata
    from providence.schemas.enums import Direction, Magnitude, ComparisonOperator, MarketCapBucket

    random.seed(42)
    store = state.belief_store
    count = 0
    now = datetime.now(timezone.utc)
    directions = [Direction.LONG, Direction.SHORT, Direction.NEUTRAL, Direction.LONG, Direction.LONG]
    magnitudes = [Magnitude.SMALL, Magnitude.MODERATE, Magnitude.LARGE, Magnitude.MODERATE]

    cap_buckets = {
        "AAPL": MarketCapBucket.MEGA, "MSFT": MarketCapBucket.MEGA, "GOOGL": MarketCapBucket.MEGA,
        "AMZN": MarketCapBucket.MEGA, "NVDA": MarketCapBucket.MEGA, "TSLA": MarketCapBucket.LARGE,
        "META": MarketCapBucket.MEGA, "JPM": MarketCapBucket.MEGA,
    }

    for run_idx in range(NUM_RUNS):
        ts = now - timedelta(hours=NUM_RUNS - run_idx)
        for agent_id in AGENTS:
            tickers_for_agent = random.sample(TICKERS, k=random.randint(2, 5))
            beliefs_array = []
            for ticker in tickers_for_agent:
                direction = random.choice(directions)
                frag_id = uuid4()  # mock fragment reference

                belief_entry = Belief(
                    thesis_id=str(uuid4()),
                    ticker=ticker,
                    thesis_summary=f"{agent_id} analysis of {ticker}: "
                                   f"{'bullish momentum' if direction == Direction.LONG else 'bearish signals' if direction == Direction.SHORT else 'neutral outlook'}",
                    direction=direction,
                    magnitude=random.choice(magnitudes),
                    raw_confidence=round(random.uniform(0.3, 0.9), 2),
                    time_horizon_days=random.choice([5, 10, 20, 30, 60]),
                    evidence=[
                        EvidenceRef(
                            source_fragment_id=frag_id,
                            field_path="payload.close",
                            observation=f"Price action for {ticker}",
                            weight=0.8,
                        ),
                    ],
                    invalidation_conditions=[
                        InvalidationCondition(
                            description=f"{ticker} price crosses invalidation threshold",
                            data_source_agent="PERCEPT-PRICE",
                            metric=f"{ticker}_close",
                            operator=ComparisonOperator.CROSSES_BELOW if direction == Direction.LONG else ComparisonOperator.CROSSES_ABOVE,
                            threshold=round(PRICES[ticker] * (0.95 if direction == Direction.LONG else 1.05), 2),
                        ),
                    ],
                    metadata=BeliefMetadata(
                        sector=SECTORS[ticker],
                        market_cap_bucket=cap_buckets.get(ticker, MarketCapBucket.LARGE),
                    ),
                )
                beliefs_array.append(belief_entry)

            context_hash = _hash({"run": run_idx, "agent": agent_id})
            belief_obj = BeliefObject(
                belief_id=str(uuid4()),
                agent_id=agent_id,
                timestamp=ts,
                context_window_hash=context_hash,
                beliefs=beliefs_array,
            )
            store.append(belief_obj)
            count += 1

    return count


def _seed_runs(state) -> int:
    """Seed PipelineRun records into RunStore."""
    from providence.orchestration.models import PipelineRun, StageResult, StageStatus, RunStatus

    random.seed(42)
    store = state.run_store
    count = 0
    now = datetime.now(timezone.utc)

    for run_idx in range(NUM_RUNS):
        ts = now - timedelta(hours=NUM_RUNS - run_idx)
        duration = random.uniform(30, 180)

        # MAIN loop
        stages = []
        agent_ids = [
            "COGNIT-FUNDAMENTAL", "COGNIT-TECHNICAL", "COGNIT-MACRO",
            "COGNIT-EVENT", "COGNIT-NARRATIVE", "COGNIT-CROSSSEC",
            "REGIME-STAT", "REGIME-SECTOR", "REGIME-NARR", "REGIME-MISMATCH",
            "DECIDE-SYNTH", "DECIDE-OPTIM",
            "EXEC-VALIDATE", "EXEC-ROUTER", "EXEC-GUARDIAN", "EXEC-CAPTURE",
        ]
        failed = random.randint(0, 3)
        failed_agents = set(random.sample(agent_ids, k=failed))
        for aid in agent_ids:
            status = StageStatus.FAILED if aid in failed_agents else StageStatus.SUCCEEDED
            stage_dur = random.uniform(0.1, 30000) if status == StageStatus.SUCCEEDED else random.uniform(0.1, 5000)
            stages.append(StageResult(
                stage_name=aid, agent_id=aid, status=status,
                started_at=ts, finished_at=ts + timedelta(milliseconds=stage_dur),
                duration_ms=stage_dur,
                error=f"Agent error in {aid}" if status == StageStatus.FAILED else None,
            ))

        succeeded = len(agent_ids) - failed
        run_status = RunStatus.SUCCEEDED if failed == 0 else RunStatus.PARTIAL_FAILURE

        run = PipelineRun(
            run_id=str(uuid4()), loop_type="MAIN", status=run_status,
            started_at=ts, finished_at=ts + timedelta(seconds=duration),
            stage_results=stages, succeeded_count=succeeded,
            failed_count=failed, skipped_count=0,
            total_duration_ms=duration * 1000,
            content_hash=_hash({"run": run_idx}),
        )
        store.append(run)
        count += 1

        # EXIT loop
        exit_run = PipelineRun(
            run_id=str(uuid4()), loop_type="EXIT", status=RunStatus.SUCCEEDED,
            started_at=ts + timedelta(seconds=duration),
            finished_at=ts + timedelta(seconds=duration + 5),
            stage_results=[], succeeded_count=5, failed_count=0,
            skipped_count=0, total_duration_ms=5000,
            content_hash=_hash({"exit_run": run_idx}),
        )
        store.append(exit_run)
        count += 1

        # GOVERNANCE loop
        gov_run = PipelineRun(
            run_id=str(uuid4()), loop_type="GOVERNANCE", status=RunStatus.SUCCEEDED,
            started_at=ts + timedelta(seconds=duration + 5),
            finished_at=ts + timedelta(seconds=duration + 6),
            stage_results=[], succeeded_count=4, failed_count=0,
            skipped_count=0, total_duration_ms=1000,
            content_hash=_hash({"gov_run": run_idx}),
        )
        store.append(gov_run)
        count += 1

    return count


def _seed_shadow_signals(state) -> int:
    """Seed ShadowSignal records into ShadowSignalStore."""
    from providence.schemas.shadow import ShadowSignal, ShadowRunSummary
    from providence.schemas.enums import Action, Direction

    random.seed(42)
    store = state.shadow_signal_store
    if store is None:
        return 0

    count = 0
    now = datetime.now(timezone.utc)

    for run_idx in range(NUM_RUNS):
        ts = now - timedelta(hours=NUM_RUNS - run_idx)
        run_id = uuid4()
        signals_in_run = []

        tickers_this_run = random.sample(TICKERS, k=random.randint(2, 5))
        for ticker in tickers_this_run:
            price = PRICES[ticker] * (1 + random.uniform(-0.03, 0.03))
            direction = random.choice([Direction.LONG, Direction.SHORT])
            approved = random.random() > 0.3  # 70% approval rate
            confidence = round(random.uniform(0.4, 0.9), 2)
            target_wt = round(random.uniform(0.02, 0.10), 3)

            # Compute simulated returns
            ret_1d = round(random.uniform(-0.02, 0.03), 4) if run_idx < NUM_RUNS - 1 else None
            ret_5d = round(random.uniform(-0.05, 0.08), 4) if run_idx < NUM_RUNS - 3 else None
            ret_20d = round(random.uniform(-0.10, 0.15), 4) if run_idx < NUM_RUNS - 5 else None

            signal = ShadowSignal(
                signal_id=uuid4(), run_id=run_id,
                timestamp=ts, ticker=ticker,
                action=Action.OPEN_LONG if direction == Direction.LONG else Action.OPEN_SHORT,
                direction=direction,
                target_weight=target_wt,
                confidence=confidence,
                approved=approved,
                rejection_reasons=[] if approved else ["Below confidence threshold"],
                adjusted_weight=target_wt * 0.8 if approved else 0.0,
                risk_mode_applied=RISK_MODES[run_idx % len(RISK_MODES)],
                price_at_signal=round(price, 2),
                simulated_entry_price=round(price * 1.001, 2) if approved else None,
                simulated_fill_qty=random.randint(10, 100) if approved else None,
                simulated_notional=round(price * random.randint(10, 100), 2) if approved else None,
                realized_return_1d=ret_1d,
                realized_return_5d=ret_5d,
                realized_return_20d=ret_20d,
                price_1d_later=round(price * (1 + (ret_1d or 0)), 2) if ret_1d else None,
                price_5d_later=round(price * (1 + (ret_5d or 0)), 2) if ret_5d else None,
                price_20d_later=round(price * (1 + (ret_20d or 0)), 2) if ret_20d else None,
            )
            store.append(signal)
            signals_in_run.append(signal)
            count += 1

        # Run summary
        regime_idx = run_idx % len(REGIMES)
        approved_count = sum(1 for s in signals_in_run if s.approved)
        rejected_count = len(signals_in_run) - approved_count
        long_count = sum(1 for s in signals_in_run if s.direction == Direction.LONG)
        short_count = len(signals_in_run) - long_count
        summary = ShadowRunSummary(
            run_id=run_id, timestamp=ts,
            total_signals=len(signals_in_run),
            approved_signals=approved_count,
            rejected_signals=rejected_count,
            long_signals=long_count,
            short_signals=short_count,
            regime_state=REGIMES[regime_idx],
            risk_mode=RISK_MODES[regime_idx],
        )
        store.append_summary(summary)

    return count


def _seed_portfolio(state) -> int:
    """Seed portfolio positions."""
    from decimal import Decimal

    tracker = state.portfolio_tracker
    if tracker is None:
        return 0

    random.seed(42)
    now = datetime.now(timezone.utc)
    count = 0

    positions_data = [
        ("NVDA", 50, 850.0, 890.0, "Technology"),
        ("AAPL", 200, 178.0, 185.0, "Technology"),
        ("AMZN", 100, 188.0, 195.0, "Consumer Discretionary"),
        ("META", 80, 510.0, 525.0, "Communication Services"),
    ]

    for ticker, qty, entry, current, sector in positions_data:
        tracker.record_fill(
            order_id=uuid4(),
            ticker=ticker,
            qty=Decimal(str(qty)),
            price=Decimal(str(entry)),
            side="BUY",
            timestamp=now - timedelta(days=random.randint(3, 20)),
        )
        tracker.update_price(ticker, Decimal(str(current)))
        count += 1

    return count


@router.post("/seed")
async def seed_demo_data():
    """Populate all stores with realistic demo data for portal display."""
    state = get_state()

    # Check if already seeded
    if state.fragment_store.count() > 10:
        return {
            "status": "already_seeded",
            "fragments": state.fragment_store.count(),
            "beliefs": state.belief_store.count(),
        }

    try:
        fragments = _seed_fragments(state)
        beliefs = _seed_beliefs(state)
        runs = _seed_runs(state)
        signals = _seed_shadow_signals(state)
        positions = _seed_portfolio(state)

        return {
            "status": "seeded",
            "fragments": fragments,
            "beliefs": beliefs,
            "pipeline_runs": runs,
            "shadow_signals": signals,
            "portfolio_positions": positions,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Seed failed: {str(e)}")


@router.post("/reset")
async def reset_all_stores():
    """Clear ALL stores — fresh start. Returns empty portal."""
    state = get_state()

    try:
        # Clear FragmentStore
        frag = state.fragment_store
        if frag is not None:
            frag._fragments.clear()
            frag._by_data_type.clear()
            frag._by_entity.clear()
            if hasattr(frag, "_persist_path") and frag._persist_path and frag._persist_path.exists():
                frag._persist_path.write_text("")

        # Clear BeliefStore
        bel = state.belief_store
        if bel is not None:
            bel._beliefs.clear()
            bel._by_agent.clear()
            bel._by_ticker.clear()
            if hasattr(bel, "_persist_path") and bel._persist_path and bel._persist_path.exists():
                bel._persist_path.write_text("")

        # Clear RunStore
        runs = state.run_store
        if runs is not None:
            runs._runs.clear()
            runs._by_loop_type.clear()
            if hasattr(runs, "_persist_path") and runs._persist_path and runs._persist_path.exists():
                runs._persist_path.write_text("")

        # Clear ShadowSignalStore
        shadow = getattr(state, "shadow_signal_store", None)
        if shadow is not None:
            shadow._signals.clear()
            if hasattr(shadow, "_signal_ids"):
                shadow._signal_ids.clear()
            shadow._by_run.clear()
            shadow._by_ticker.clear()
            if hasattr(shadow, "_summaries"):
                shadow._summaries.clear()
            if hasattr(shadow, "_persist_path") and shadow._persist_path and shadow._persist_path.exists():
                shadow._persist_path.write_text("")
            if hasattr(shadow, "_summaries_path") and shadow._summaries_path and shadow._summaries_path.exists():
                shadow._summaries_path.write_text("")

        # Clear portfolio tracker
        tracker = getattr(state, "portfolio_tracker", None)
        if tracker is not None:
            tracker._positions.clear()
            if hasattr(tracker, "_fills"):
                tracker._fills.clear()

        # Clear order manager
        orders = getattr(state, "order_manager", None)
        if orders is not None:
            if hasattr(orders, "_orders"):
                orders._orders.clear()

        return {"status": "reset", "message": "All stores cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")


@router.post("/seed/perception")
async def seed_perception_only():
    """Seed ONLY market data fragments — no beliefs, no runs, no signals.

    This gives the adaptive agents something to analyze when a pipeline
    run is triggered, without pre-populating any results.
    """
    state = get_state()

    if state.fragment_store.count() > 10:
        return {
            "status": "already_seeded",
            "fragments": state.fragment_store.count(),
        }

    try:
        fragments = _seed_fragments(state)
        return {"status": "seeded", "fragments": fragments}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Perception seed failed: {str(e)}")
