"""Tests for FragmentStore — append-only MarketStateFragment storage."""

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from providence.schemas.enums import DataType, ValidationStatus
from providence.schemas.market_state import MarketStateFragment
from providence.storage.fragment_store import FragmentStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NOW = datetime(2026, 2, 20, 12, 0, 0, tzinfo=timezone.utc)


def _make_fragment(
    *,
    entity: str = "AAPL",
    data_type: DataType = DataType.PRICE_OHLCV,
    timestamp: datetime | None = None,
    validation_status: ValidationStatus = ValidationStatus.VALID,
) -> MarketStateFragment:
    return MarketStateFragment(
        fragment_id=uuid4(),
        agent_id="PERCEPT-PRICE",
        timestamp=timestamp or NOW,
        source_timestamp=NOW - timedelta(hours=1),
        entity=entity,
        data_type=data_type,
        schema_version="1.0.0",
        source_hash=f"test-{entity}-{uuid4().hex[:8]}",
        validation_status=validation_status,
        payload={"close": 100.0, "volume": 1000},
    )


# ===========================================================================
# Append and Get
# ===========================================================================


class TestAppendAndGet:
    def test_append_returns_true_for_new(self):
        store = FragmentStore()
        frag = _make_fragment()
        assert store.append(frag) is True

    def test_append_returns_false_for_duplicate(self):
        store = FragmentStore()
        frag = _make_fragment()
        store.append(frag)
        assert store.append(frag) is False

    def test_get_by_id(self):
        store = FragmentStore()
        frag = _make_fragment()
        store.append(frag)
        assert store.get(frag.fragment_id) == frag

    def test_get_returns_none_for_missing(self):
        store = FragmentStore()
        assert store.get(uuid4()) is None

    def test_append_many(self):
        store = FragmentStore()
        frags = [_make_fragment(entity=f"TICK{i}") for i in range(5)]
        added = store.append_many(frags)
        assert added == 5
        assert store.count() == 5

    def test_append_many_deduplicates(self):
        store = FragmentStore()
        frag = _make_fragment()
        added = store.append_many([frag, frag, frag])
        assert added == 1

    def test_count(self):
        store = FragmentStore()
        assert store.count() == 0
        store.append(_make_fragment())
        assert store.count() == 1


# ===========================================================================
# Query — Data Type
# ===========================================================================


class TestQueryDataType:
    def test_filter_by_data_type(self):
        store = FragmentStore()
        store.append(_make_fragment(data_type=DataType.PRICE_OHLCV))
        store.append(_make_fragment(data_type=DataType.FILING_10Q))
        store.append(_make_fragment(data_type=DataType.PRICE_OHLCV))

        results = store.query(data_types={DataType.PRICE_OHLCV})
        assert len(results) == 2
        assert all(f.data_type == DataType.PRICE_OHLCV for f in results)

    def test_filter_multiple_data_types(self):
        store = FragmentStore()
        store.append(_make_fragment(data_type=DataType.PRICE_OHLCV))
        store.append(_make_fragment(data_type=DataType.FILING_10Q))
        store.append(_make_fragment(data_type=DataType.SENTIMENT_NEWS))

        results = store.query(
            data_types={DataType.PRICE_OHLCV, DataType.FILING_10Q}
        )
        assert len(results) == 2


# ===========================================================================
# Query — Entity
# ===========================================================================


class TestQueryEntity:
    def test_filter_by_entity(self):
        store = FragmentStore()
        store.append(_make_fragment(entity="AAPL"))
        store.append(_make_fragment(entity="NVDA"))
        store.append(_make_fragment(entity="AAPL"))

        results = store.query(entities={"AAPL"})
        assert len(results) == 2
        assert all(f.entity == "AAPL" for f in results)

    def test_filter_multiple_entities(self):
        store = FragmentStore()
        store.append(_make_fragment(entity="AAPL"))
        store.append(_make_fragment(entity="NVDA"))
        store.append(_make_fragment(entity="JPM"))

        results = store.query(entities={"AAPL", "NVDA"})
        assert len(results) == 2

    def test_all_entities(self):
        store = FragmentStore()
        store.append(_make_fragment(entity="AAPL"))
        store.append(_make_fragment(entity="NVDA"))
        assert store.all_entities() == {"AAPL", "NVDA"}


# ===========================================================================
# Query — Combined Filters
# ===========================================================================


class TestQueryCombined:
    def test_data_type_and_entity_intersection(self):
        store = FragmentStore()
        store.append(_make_fragment(entity="AAPL", data_type=DataType.PRICE_OHLCV))
        store.append(_make_fragment(entity="AAPL", data_type=DataType.FILING_10Q))
        store.append(_make_fragment(entity="NVDA", data_type=DataType.PRICE_OHLCV))

        results = store.query(
            data_types={DataType.PRICE_OHLCV}, entities={"AAPL"}
        )
        assert len(results) == 1
        assert results[0].entity == "AAPL"
        assert results[0].data_type == DataType.PRICE_OHLCV

    def test_exclude_quarantined(self):
        store = FragmentStore()
        store.append(_make_fragment(validation_status=ValidationStatus.VALID))
        store.append(
            _make_fragment(validation_status=ValidationStatus.QUARANTINED)
        )
        results = store.query(exclude_quarantined=True)
        assert len(results) == 1

    def test_include_quarantined(self):
        store = FragmentStore()
        store.append(_make_fragment(validation_status=ValidationStatus.VALID))
        store.append(
            _make_fragment(validation_status=ValidationStatus.QUARANTINED)
        )
        results = store.query(exclude_quarantined=False)
        assert len(results) == 2


# ===========================================================================
# Query — Timestamp
# ===========================================================================


class TestQueryTimestamp:
    def test_since_filter(self):
        store = FragmentStore()
        store.append(_make_fragment(timestamp=NOW - timedelta(hours=48)))
        store.append(_make_fragment(timestamp=NOW - timedelta(hours=12)))
        store.append(_make_fragment(timestamp=NOW))

        results = store.query(since=NOW - timedelta(hours=24))
        assert len(results) == 2

    def test_until_filter(self):
        store = FragmentStore()
        store.append(_make_fragment(timestamp=NOW - timedelta(hours=48)))
        store.append(_make_fragment(timestamp=NOW))

        results = store.query(until=NOW - timedelta(hours=24))
        assert len(results) == 1

    def test_results_ordered_newest_first(self):
        store = FragmentStore()
        old = _make_fragment(timestamp=NOW - timedelta(hours=5))
        new = _make_fragment(timestamp=NOW)
        store.append(old)
        store.append(new)

        results = store.query()
        assert results[0].timestamp >= results[1].timestamp

    def test_limit(self):
        store = FragmentStore()
        for i in range(10):
            store.append(_make_fragment(timestamp=NOW - timedelta(hours=i)))
        results = store.query(limit=3)
        assert len(results) == 3


# ===========================================================================
# Convenience Methods
# ===========================================================================


class TestConvenience:
    def test_get_latest_by_entity(self):
        store = FragmentStore()
        old = _make_fragment(entity="AAPL", timestamp=NOW - timedelta(hours=5))
        new = _make_fragment(entity="AAPL", timestamp=NOW)
        store.append(old)
        store.append(new)

        latest = store.get_latest_by_entity("AAPL")
        assert latest.fragment_id == new.fragment_id

    def test_get_latest_by_entity_with_data_type(self):
        store = FragmentStore()
        store.append(
            _make_fragment(entity="AAPL", data_type=DataType.PRICE_OHLCV, timestamp=NOW)
        )
        filing = _make_fragment(
            entity="AAPL",
            data_type=DataType.FILING_10Q,
            timestamp=NOW - timedelta(hours=1),
        )
        store.append(filing)

        latest = store.get_latest_by_entity("AAPL", DataType.FILING_10Q)
        assert latest.data_type == DataType.FILING_10Q

    def test_get_latest_by_entity_returns_none(self):
        store = FragmentStore()
        assert store.get_latest_by_entity("AAPL") is None


# ===========================================================================
# Persistence
# ===========================================================================


class TestPersistence:
    def test_roundtrip_to_disk(self, tmp_path):
        path = tmp_path / "fragments.jsonl"

        # Write
        store1 = FragmentStore(persist_path=path)
        frags = [_make_fragment(entity=f"T{i}") for i in range(5)]
        store1.append_many(frags)
        assert path.exists()

        # Reload
        store2 = FragmentStore(persist_path=path)
        assert store2.count() == 5
        for f in frags:
            assert store2.get(f.fragment_id) is not None

    def test_incremental_persistence(self, tmp_path):
        path = tmp_path / "fragments.jsonl"
        store = FragmentStore(persist_path=path)
        store.append(_make_fragment(entity="AAPL"))
        store.append(_make_fragment(entity="NVDA"))

        # Count lines
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2

    def test_no_persist_without_path(self, tmp_path):
        store = FragmentStore()  # No persist_path
        store.append(_make_fragment())
        # Just verify it doesn't crash
        assert store.count() == 1
