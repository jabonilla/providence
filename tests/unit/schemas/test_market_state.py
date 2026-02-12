"""Tests for MarketStateFragment schema.

Validates: creation, content hashing, immutability, validation rules.
"""

import json
from datetime import datetime, timezone
from uuid import uuid4

import pytest
from pydantic import ValidationError

from providence.schemas.enums import DataType, ValidationStatus
from providence.schemas.market_state import MarketStateFragment


class TestMarketStateFragmentCreation:
    """Test valid MarketStateFragment creation."""

    def test_valid_fragment_creation(self, sample_fragment: MarketStateFragment) -> None:
        """A properly constructed fragment should have all fields set."""
        assert sample_fragment.agent_id == "PERCEPT-PRICE"
        assert sample_fragment.entity == "AAPL"
        assert sample_fragment.data_type == DataType.PRICE_OHLCV
        assert sample_fragment.validation_status == ValidationStatus.VALID
        assert sample_fragment.schema_version == "1.0.0"

    def test_fragment_with_all_data_types(self, sample_price_payload: dict) -> None:
        """Fragments can be created with any valid DataType."""
        for dt in DataType:
            fragment = MarketStateFragment(
                agent_id=f"PERCEPT-{dt.value}",
                timestamp=datetime(2026, 2, 9, 12, 0, 0, tzinfo=timezone.utc),
                source_timestamp=datetime(2026, 2, 9, 11, 0, 0, tzinfo=timezone.utc),
                entity="AAPL",
                data_type=dt,
                source_hash="test-hash",
                payload=sample_price_payload,
            )
            assert fragment.data_type == dt

    def test_fragment_entity_nullable(self, sample_price_payload: dict) -> None:
        """Entity can be explicitly None for macro data types."""
        fragment = MarketStateFragment(
            agent_id="PERCEPT-MACRO",
            timestamp=datetime(2026, 2, 9, 12, 0, 0, tzinfo=timezone.utc),
            source_timestamp=datetime(2026, 2, 9, 11, 0, 0, tzinfo=timezone.utc),
            entity=None,
            data_type=DataType.MACRO_YIELD_CURVE,
            source_hash="macro-hash",
            payload=sample_price_payload,
        )
        assert fragment.entity is None

    def test_fragment_default_values(self, sample_price_payload: dict) -> None:
        """Defaults are applied for optional fields."""
        fragment = MarketStateFragment(
            agent_id="PERCEPT-PRICE",
            timestamp=datetime(2026, 2, 9, 12, 0, 0, tzinfo=timezone.utc),
            source_timestamp=datetime(2026, 2, 9, 11, 0, 0, tzinfo=timezone.utc),
            entity="AAPL",
            data_type=DataType.PRICE_OHLCV,
            source_hash="test-hash",
            payload=sample_price_payload,
        )
        assert fragment.validation_status == ValidationStatus.VALID
        assert fragment.schema_version == "1.0.0"
        assert fragment.fragment_id is not None

    def test_entity_required(self, sample_price_payload: dict) -> None:
        """Entity must be explicitly provided (no default)."""
        with pytest.raises(ValidationError):
            MarketStateFragment(
                agent_id="PERCEPT-PRICE",
                timestamp=datetime(2026, 2, 9, 12, 0, 0, tzinfo=timezone.utc),
                source_timestamp=datetime(2026, 2, 9, 11, 0, 0, tzinfo=timezone.utc),
                data_type=DataType.PRICE_OHLCV,
                source_hash="test-hash",
                payload=sample_price_payload,
            )

    def test_source_hash_required(self, sample_price_payload: dict) -> None:
        """Source hash must be explicitly provided (no default)."""
        with pytest.raises(ValidationError):
            MarketStateFragment(
                agent_id="PERCEPT-PRICE",
                timestamp=datetime(2026, 2, 9, 12, 0, 0, tzinfo=timezone.utc),
                source_timestamp=datetime(2026, 2, 9, 11, 0, 0, tzinfo=timezone.utc),
                entity="AAPL",
                data_type=DataType.PRICE_OHLCV,
                payload=sample_price_payload,
            )


class TestMarketStateFragmentValidation:
    """Test validation rules for MarketStateFragment."""

    def test_timestamp_requires_timezone(self, sample_price_payload: dict) -> None:
        """Timestamp without timezone must be rejected."""
        with pytest.raises(ValidationError, match="timezone"):
            MarketStateFragment(
                agent_id="PERCEPT-PRICE",
                timestamp=datetime(2026, 2, 9, 12, 0, 0),  # no tzinfo
                source_timestamp=datetime(2026, 2, 9, 11, 0, 0, tzinfo=timezone.utc),
                entity="AAPL",
                data_type=DataType.PRICE_OHLCV,
                source_hash="test-hash",
                payload=sample_price_payload,
            )

    def test_source_timestamp_requires_timezone(self, sample_price_payload: dict) -> None:
        """Source timestamp without timezone must be rejected."""
        with pytest.raises(ValidationError, match="timezone"):
            MarketStateFragment(
                agent_id="PERCEPT-PRICE",
                timestamp=datetime(2026, 2, 9, 12, 0, 0, tzinfo=timezone.utc),
                source_timestamp=datetime(2026, 2, 9, 11, 0, 0),  # no tzinfo
                entity="AAPL",
                data_type=DataType.PRICE_OHLCV,
                source_hash="test-hash",
                payload=sample_price_payload,
            )

    def test_invalid_data_type_rejected(self, sample_price_payload: dict) -> None:
        """Invalid data_type string must be rejected."""
        with pytest.raises(ValidationError):
            MarketStateFragment(
                agent_id="PERCEPT-PRICE",
                timestamp=datetime(2026, 2, 9, 12, 0, 0, tzinfo=timezone.utc),
                source_timestamp=datetime(2026, 2, 9, 11, 0, 0, tzinfo=timezone.utc),
                entity="AAPL",
                data_type="INVALID_TYPE",  # type: ignore[arg-type]
                source_hash="test-hash",
                payload=sample_price_payload,
            )

    def test_invalid_validation_status_rejected(self, sample_price_payload: dict) -> None:
        """Invalid validation_status must be rejected."""
        with pytest.raises(ValidationError):
            MarketStateFragment(
                agent_id="PERCEPT-PRICE",
                timestamp=datetime(2026, 2, 9, 12, 0, 0, tzinfo=timezone.utc),
                source_timestamp=datetime(2026, 2, 9, 11, 0, 0, tzinfo=timezone.utc),
                entity="AAPL",
                data_type=DataType.PRICE_OHLCV,
                source_hash="test-hash",
                validation_status="UNKNOWN",  # type: ignore[arg-type]
                payload=sample_price_payload,
            )

    def test_invalid_semver_rejected(self, sample_price_payload: dict) -> None:
        """Invalid SemVer schema_version must be rejected."""
        with pytest.raises(ValidationError, match="SemVer"):
            MarketStateFragment(
                agent_id="PERCEPT-PRICE",
                timestamp=datetime(2026, 2, 9, 12, 0, 0, tzinfo=timezone.utc),
                source_timestamp=datetime(2026, 2, 9, 11, 0, 0, tzinfo=timezone.utc),
                entity="AAPL",
                data_type=DataType.PRICE_OHLCV,
                source_hash="test-hash",
                schema_version="1.0",  # missing patch version
                payload=sample_price_payload,
            )

    def test_missing_required_fields(self) -> None:
        """Missing required fields must raise ValidationError."""
        with pytest.raises(ValidationError):
            MarketStateFragment()  # type: ignore[call-arg]


class TestMarketStateFragmentContentHash:
    """Test content hash computation and determinism."""

    def test_content_hash_computed(self, sample_fragment: MarketStateFragment) -> None:
        """Content hash (version field) should be computed automatically."""
        assert sample_fragment.version != ""
        assert len(sample_fragment.version) == 64  # SHA-256 hex digest length

    def test_content_hash_deterministic(self, sample_price_payload: dict) -> None:
        """Same payload must always produce the same content hash."""
        ts = datetime(2026, 2, 9, 12, 0, 0, tzinfo=timezone.utc)
        src_ts = datetime(2026, 2, 9, 11, 0, 0, tzinfo=timezone.utc)

        f1 = MarketStateFragment(
            agent_id="PERCEPT-PRICE",
            timestamp=ts,
            source_timestamp=src_ts,
            entity="AAPL",
            data_type=DataType.PRICE_OHLCV,
            source_hash="test-hash",
            payload=sample_price_payload,
        )
        f2 = MarketStateFragment(
            agent_id="PERCEPT-PRICE",
            timestamp=ts,
            source_timestamp=src_ts,
            entity="AAPL",
            data_type=DataType.PRICE_OHLCV,
            source_hash="test-hash",
            payload=sample_price_payload,
        )
        assert f1.version == f2.version

    def test_different_payloads_different_hashes(self) -> None:
        """Different payloads must produce different content hashes."""
        ts = datetime(2026, 2, 9, 12, 0, 0, tzinfo=timezone.utc)
        src_ts = datetime(2026, 2, 9, 11, 0, 0, tzinfo=timezone.utc)

        f1 = MarketStateFragment(
            agent_id="PERCEPT-PRICE",
            timestamp=ts,
            source_timestamp=src_ts,
            entity="AAPL",
            data_type=DataType.PRICE_OHLCV,
            source_hash="test-hash",
            payload={"close": 185.0},
        )
        f2 = MarketStateFragment(
            agent_id="PERCEPT-PRICE",
            timestamp=ts,
            source_timestamp=src_ts,
            entity="AAPL",
            data_type=DataType.PRICE_OHLCV,
            source_hash="test-hash",
            payload={"close": 186.0},
        )
        assert f1.version != f2.version

    def test_key_order_does_not_affect_hash(self) -> None:
        """Payload key order should not affect content hash (sorted serialization)."""
        ts = datetime(2026, 2, 9, 12, 0, 0, tzinfo=timezone.utc)
        src_ts = datetime(2026, 2, 9, 11, 0, 0, tzinfo=timezone.utc)

        f1 = MarketStateFragment(
            agent_id="PERCEPT-PRICE",
            timestamp=ts,
            source_timestamp=src_ts,
            entity="AAPL",
            data_type=DataType.PRICE_OHLCV,
            source_hash="test-hash",
            payload={"a": 1, "b": 2, "c": 3},
        )
        f2 = MarketStateFragment(
            agent_id="PERCEPT-PRICE",
            timestamp=ts,
            source_timestamp=src_ts,
            entity="AAPL",
            data_type=DataType.PRICE_OHLCV,
            source_hash="test-hash",
            payload={"c": 3, "a": 1, "b": 2},
        )
        assert f1.version == f2.version


class TestMarketStateFragmentImmutability:
    """Test that MarketStateFragment is truly immutable."""

    def test_cannot_modify_fields(self, sample_fragment: MarketStateFragment) -> None:
        """Attempting to modify any field should raise an error."""
        with pytest.raises(ValidationError):
            sample_fragment.entity = "MSFT"

    def test_cannot_modify_payload(self, sample_fragment: MarketStateFragment) -> None:
        """Attempting to replace the payload should raise an error."""
        with pytest.raises(ValidationError):
            sample_fragment.payload = {"new": "data"}

    def test_cannot_modify_validation_status(self, sample_fragment: MarketStateFragment) -> None:
        """Attempting to change validation_status should raise an error."""
        with pytest.raises(ValidationError):
            sample_fragment.validation_status = ValidationStatus.QUARANTINED

    def test_cannot_modify_version(self, sample_fragment: MarketStateFragment) -> None:
        """Attempting to change the content hash should raise an error."""
        with pytest.raises(ValidationError):
            sample_fragment.version = "tampered"


class TestMarketStateFragmentSerialization:
    """Test JSON serialization/deserialization round-trip."""

    def test_round_trip_json(self, sample_fragment: MarketStateFragment) -> None:
        """Fragment should survive JSON serialization round-trip."""
        json_str = sample_fragment.model_dump_json()
        restored = MarketStateFragment.model_validate_json(json_str)

        assert restored.fragment_id == sample_fragment.fragment_id
        assert restored.agent_id == sample_fragment.agent_id
        assert restored.version == sample_fragment.version
        assert restored.payload == sample_fragment.payload

    def test_round_trip_dict(self, sample_fragment: MarketStateFragment) -> None:
        """Fragment should survive dict serialization round-trip."""
        data = sample_fragment.model_dump()
        restored = MarketStateFragment.model_validate(data)

        assert restored.version == sample_fragment.version
        assert restored.entity == sample_fragment.entity
