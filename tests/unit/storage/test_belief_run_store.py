"""Tests for BeliefStore and RunStore."""

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from providence.orchestration.models import (
    PipelineRun,
    RunStatus,
    StageResult,
    StageStatus,
)
from providence.schemas.belief import Belief, BeliefMetadata, BeliefObject, EvidenceRef, InvalidationCondition
from providence.schemas.enums import (
    ComparisonOperator,
    ConditionStatus,
    Direction,
    Magnitude,
    MarketCapBucket,
)
from providence.storage.belief_store import BeliefStore
from providence.storage.run_store import RunStore


NOW = datetime(2026, 2, 20, 12, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_belief(
    *,
    agent_id: str = "COGNIT-FUNDAMENTAL",
    ticker: str = "AAPL",
    timestamp: datetime | None = None,
) -> BeliefObject:
    return BeliefObject(
        belief_id=uuid4(),
        agent_id=agent_id,
        timestamp=timestamp or NOW,
        context_window_hash="sha256-test-" + uuid4().hex[:8],
        beliefs=[
            Belief(
                thesis_id=f"TEST-{ticker}-001",
                ticker=ticker,
                thesis_summary=f"Test thesis for {ticker}",
                direction=Direction.LONG,
                magnitude=Magnitude.MODERATE,
                raw_confidence=0.7,
                time_horizon_days=90,
                evidence=[
                    EvidenceRef(
                        source_fragment_id=uuid4(),
                        field_path="payload.close",
                        observation="Price data",
                        weight=0.8,
                    )
                ],
                invalidation_conditions=[
                    InvalidationCondition(
                        condition_id=uuid4(),
                        description="Price drops below threshold",
                        data_source_agent="PERCEPT-PRICE",
                        metric="close",
                        operator=ComparisonOperator.LT,
                        threshold=100.0,
                        status=ConditionStatus.ACTIVE,
                    )
                ],
                correlated_beliefs=[],
                metadata=BeliefMetadata(
                    sector="Technology",
                    market_cap_bucket=MarketCapBucket.MEGA,
                ),
            )
        ],
    )


def _make_run(
    *,
    loop_type: str = "MAIN",
    status: RunStatus = RunStatus.SUCCEEDED,
    started_at: datetime | None = None,
) -> PipelineRun:
    start = started_at or NOW
    return PipelineRun(
        run_id=uuid4(),
        loop_type=loop_type,
        status=status,
        started_at=start,
        finished_at=start + timedelta(seconds=5),
        stage_results=[
            StageResult(
                stage_name="TEST-STAGE",
                agent_id="TEST-AGENT",
                status=StageStatus.SUCCEEDED,
                started_at=start,
                finished_at=start + timedelta(seconds=3),
                duration_ms=3000.0,
                output={"test": True},
                error=None,
            )
        ],
        metadata={},
    )


# ===========================================================================
# BeliefStore — Append and Get
# ===========================================================================


class TestBeliefAppendAndGet:
    def test_append_new(self):
        store = BeliefStore()
        belief = _make_belief()
        assert store.append(belief) is True
        assert store.count() == 1

    def test_append_duplicate(self):
        store = BeliefStore()
        belief = _make_belief()
        store.append(belief)
        assert store.append(belief) is False
        assert store.count() == 1

    def test_get_by_id(self):
        store = BeliefStore()
        belief = _make_belief()
        store.append(belief)
        assert store.get(belief.belief_id) == belief

    def test_get_missing(self):
        store = BeliefStore()
        assert store.get(uuid4()) is None

    def test_append_many(self):
        store = BeliefStore()
        beliefs = [_make_belief(ticker=f"T{i}") for i in range(5)]
        added = store.append_many(beliefs)
        assert added == 5


# ===========================================================================
# BeliefStore — Query
# ===========================================================================


class TestBeliefQuery:
    def test_filter_by_agent(self):
        store = BeliefStore()
        store.append(_make_belief(agent_id="COGNIT-FUNDAMENTAL"))
        store.append(_make_belief(agent_id="COGNIT-TECHNICAL"))
        store.append(_make_belief(agent_id="COGNIT-FUNDAMENTAL"))

        results = store.query(agent_ids={"COGNIT-FUNDAMENTAL"})
        assert len(results) == 2

    def test_filter_by_ticker(self):
        store = BeliefStore()
        store.append(_make_belief(ticker="AAPL"))
        store.append(_make_belief(ticker="NVDA"))
        store.append(_make_belief(ticker="AAPL"))

        results = store.query(tickers={"AAPL"})
        assert len(results) == 2

    def test_filter_by_agent_and_ticker(self):
        store = BeliefStore()
        store.append(_make_belief(agent_id="COGNIT-FUNDAMENTAL", ticker="AAPL"))
        store.append(_make_belief(agent_id="COGNIT-FUNDAMENTAL", ticker="NVDA"))
        store.append(_make_belief(agent_id="COGNIT-TECHNICAL", ticker="AAPL"))

        results = store.query(
            agent_ids={"COGNIT-FUNDAMENTAL"}, tickers={"AAPL"}
        )
        assert len(results) == 1

    def test_since_filter(self):
        store = BeliefStore()
        store.append(_make_belief(timestamp=NOW - timedelta(hours=48)))
        store.append(_make_belief(timestamp=NOW))

        results = store.query(since=NOW - timedelta(hours=24))
        assert len(results) == 1

    def test_limit(self):
        store = BeliefStore()
        for i in range(10):
            store.append(_make_belief(timestamp=NOW - timedelta(hours=i)))
        results = store.query(limit=3)
        assert len(results) == 3

    def test_ordered_newest_first(self):
        store = BeliefStore()
        old = _make_belief(timestamp=NOW - timedelta(hours=5))
        new = _make_belief(timestamp=NOW)
        store.append(old)
        store.append(new)

        results = store.query()
        assert results[0].timestamp >= results[1].timestamp


# ===========================================================================
# BeliefStore — Convenience
# ===========================================================================


class TestBeliefConvenience:
    def test_get_latest_by_agent(self):
        store = BeliefStore()
        old = _make_belief(agent_id="COGNIT-FUNDAMENTAL", timestamp=NOW - timedelta(hours=5))
        new = _make_belief(agent_id="COGNIT-FUNDAMENTAL", timestamp=NOW)
        store.append(old)
        store.append(new)

        latest = store.get_latest_by_agent("COGNIT-FUNDAMENTAL")
        assert latest.belief_id == new.belief_id

    def test_get_latest_by_ticker(self):
        store = BeliefStore()
        store.append(_make_belief(agent_id="COGNIT-FUNDAMENTAL", ticker="AAPL", timestamp=NOW))
        store.append(_make_belief(agent_id="COGNIT-TECHNICAL", ticker="AAPL", timestamp=NOW - timedelta(hours=1)))

        results = store.get_latest_by_ticker("AAPL")
        assert len(results) == 2  # One from each agent

    def test_all_agents(self):
        store = BeliefStore()
        store.append(_make_belief(agent_id="COGNIT-FUNDAMENTAL"))
        store.append(_make_belief(agent_id="COGNIT-TECHNICAL"))
        assert store.all_agents() == {"COGNIT-FUNDAMENTAL", "COGNIT-TECHNICAL"}

    def test_all_tickers(self):
        store = BeliefStore()
        store.append(_make_belief(ticker="AAPL"))
        store.append(_make_belief(ticker="NVDA"))
        assert store.all_tickers() == {"AAPL", "NVDA"}


# ===========================================================================
# BeliefStore — Persistence
# ===========================================================================


class TestBeliefPersistence:
    def test_roundtrip(self, tmp_path):
        path = tmp_path / "beliefs.jsonl"

        store1 = BeliefStore(persist_path=path)
        beliefs = [_make_belief(ticker=f"T{i}") for i in range(3)]
        store1.append_many(beliefs)

        store2 = BeliefStore(persist_path=path)
        assert store2.count() == 3
        for b in beliefs:
            assert store2.get(b.belief_id) is not None


# ===========================================================================
# RunStore — Append and Get
# ===========================================================================


class TestRunAppendAndGet:
    def test_append_new(self):
        store = RunStore()
        run = _make_run()
        assert store.append(run) is True

    def test_append_duplicate(self):
        store = RunStore()
        run = _make_run()
        store.append(run)
        assert store.append(run) is False

    def test_get_by_id(self):
        store = RunStore()
        run = _make_run()
        store.append(run)
        assert store.get(run.run_id) == run

    def test_count(self):
        store = RunStore()
        store.append(_make_run())
        store.append(_make_run())
        assert store.count() == 2

    def test_count_by_loop_type(self):
        store = RunStore()
        store.append(_make_run(loop_type="MAIN"))
        store.append(_make_run(loop_type="MAIN"))
        store.append(_make_run(loop_type="EXIT"))
        assert store.count(loop_type="MAIN") == 2
        assert store.count(loop_type="EXIT") == 1


# ===========================================================================
# RunStore — Query
# ===========================================================================


class TestRunQuery:
    def test_filter_by_loop_type(self):
        store = RunStore()
        store.append(_make_run(loop_type="MAIN"))
        store.append(_make_run(loop_type="EXIT"))
        store.append(_make_run(loop_type="MAIN"))

        results = store.query(loop_type="MAIN")
        assert len(results) == 2

    def test_filter_by_status(self):
        store = RunStore()
        store.append(_make_run(status=RunStatus.SUCCEEDED))
        store.append(_make_run(status=RunStatus.FAILED))

        results = store.query(status=RunStatus.SUCCEEDED)
        assert len(results) == 1

    def test_since_filter(self):
        store = RunStore()
        store.append(_make_run(started_at=NOW - timedelta(hours=48)))
        store.append(_make_run(started_at=NOW))

        results = store.query(since=NOW - timedelta(hours=24))
        assert len(results) == 1

    def test_get_latest(self):
        store = RunStore()
        old = _make_run(started_at=NOW - timedelta(hours=5))
        new = _make_run(started_at=NOW)
        store.append(old)
        store.append(new)

        latest = store.get_latest()
        assert latest.run_id == new.run_id

    def test_get_latest_by_loop_type(self):
        store = RunStore()
        store.append(_make_run(loop_type="MAIN", started_at=NOW))
        store.append(_make_run(loop_type="EXIT", started_at=NOW + timedelta(minutes=1)))

        latest = store.get_latest(loop_type="MAIN")
        assert latest.loop_type == "MAIN"


# ===========================================================================
# RunStore — Success Rate
# ===========================================================================


class TestRunSuccessRate:
    def test_all_succeeded(self):
        store = RunStore()
        for _ in range(5):
            store.append(_make_run(status=RunStatus.SUCCEEDED))
        assert store.success_rate() == 1.0

    def test_mixed_results(self):
        store = RunStore()
        store.append(_make_run(status=RunStatus.SUCCEEDED))
        store.append(_make_run(status=RunStatus.FAILED))
        assert store.success_rate() == 0.5

    def test_empty_store(self):
        store = RunStore()
        assert store.success_rate() == 0.0

    def test_by_loop_type(self):
        store = RunStore()
        store.append(_make_run(loop_type="MAIN", status=RunStatus.SUCCEEDED))
        store.append(_make_run(loop_type="MAIN", status=RunStatus.FAILED))
        store.append(_make_run(loop_type="EXIT", status=RunStatus.SUCCEEDED))

        assert store.success_rate(loop_type="MAIN") == 0.5
        assert store.success_rate(loop_type="EXIT") == 1.0


# ===========================================================================
# RunStore — Persistence
# ===========================================================================


class TestRunPersistence:
    def test_roundtrip(self, tmp_path):
        path = tmp_path / "runs.jsonl"

        store1 = RunStore(persist_path=path)
        runs = [_make_run() for _ in range(3)]
        for r in runs:
            store1.append(r)

        store2 = RunStore(persist_path=path)
        assert store2.count() == 3
        for r in runs:
            assert store2.get(r.run_id) is not None
