"""Tests for BeliefObject and nested types.

Validates: creation, confidence bounds, invalidation condition validation,
evidence ref linking, content hash computation.
"""

from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from providence.schemas.enums import (
    CatalystType,
    ComparisonOperator,
    ConditionStatus,
    Direction,
    Magnitude,
    MarketCapBucket,
)
from providence.schemas.belief import (
    Belief,
    BeliefMetadata,
    BeliefObject,
    EvidenceRef,
    InvalidationCondition,
)


# ===================================================================
# EvidenceRef Tests
# ===================================================================
class TestEvidenceRef:
    """Tests for EvidenceRef schema."""

    def test_valid_evidence_ref(self, sample_evidence_ref: EvidenceRef) -> None:
        """Valid EvidenceRef should be created successfully."""
        assert sample_evidence_ref.field_path == "payload.revenue"
        assert sample_evidence_ref.weight == 0.8

    def test_weight_lower_bound(self) -> None:
        """Weight below 0.0 must be rejected."""
        with pytest.raises(ValidationError):
            EvidenceRef(
                source_fragment_id=uuid4(),
                field_path="payload.close",
                observation="Price dropped",
                weight=-0.1,
            )

    def test_weight_upper_bound(self) -> None:
        """Weight above 1.0 must be rejected."""
        with pytest.raises(ValidationError):
            EvidenceRef(
                source_fragment_id=uuid4(),
                field_path="payload.close",
                observation="Price dropped",
                weight=1.1,
            )

    def test_weight_at_boundaries(self) -> None:
        """Weight at exactly 0.0 and 1.0 should be accepted."""
        ref_zero = EvidenceRef(
            source_fragment_id=uuid4(),
            field_path="payload.x",
            observation="obs",
            weight=0.0,
        )
        ref_one = EvidenceRef(
            source_fragment_id=uuid4(),
            field_path="payload.x",
            observation="obs",
            weight=1.0,
        )
        assert ref_zero.weight == 0.0
        assert ref_one.weight == 1.0

    def test_immutability(self, sample_evidence_ref: EvidenceRef) -> None:
        """EvidenceRef should be frozen."""
        with pytest.raises(ValidationError):
            sample_evidence_ref.weight = 0.5


# ===================================================================
# InvalidationCondition Tests
# ===================================================================
class TestInvalidationCondition:
    """Tests for InvalidationCondition schema."""

    def test_valid_condition(self, sample_invalidation_condition: InvalidationCondition) -> None:
        """Valid condition should have all fields set correctly."""
        assert sample_invalidation_condition.metric == "revenue_growth_yoy"
        assert sample_invalidation_condition.operator == ComparisonOperator.LT
        assert sample_invalidation_condition.threshold == 0.05
        assert sample_invalidation_condition.status == ConditionStatus.ACTIVE

    def test_all_operators_accepted(self) -> None:
        """All ComparisonOperator values should be valid."""
        for op in ComparisonOperator:
            cond = InvalidationCondition(
                description=f"Test {op.value}",
                data_source_agent="PERCEPT-PRICE",
                metric="price",
                operator=op,
                threshold=100.0,
            )
            assert cond.operator == op

    def test_empty_metric_rejected(self) -> None:
        """Empty or whitespace-only metric must be rejected."""
        with pytest.raises(ValidationError, match="non-empty"):
            InvalidationCondition(
                description="Bad condition",
                data_source_agent="PERCEPT-PRICE",
                metric="   ",
                operator=ComparisonOperator.GT,
                threshold=100.0,
            )

    def test_breach_magnitude_non_negative(self) -> None:
        """breach_magnitude must be >= 0.0."""
        with pytest.raises(ValidationError):
            InvalidationCondition(
                description="Test",
                data_source_agent="PERCEPT-PRICE",
                metric="price",
                operator=ComparisonOperator.GT,
                threshold=100.0,
                breach_magnitude=-0.1,
            )

    def test_breach_magnitude_valid(self) -> None:
        """Valid breach_magnitude should be accepted."""
        cond = InvalidationCondition(
            description="Test",
            data_source_agent="PERCEPT-PRICE",
            metric="price",
            operator=ComparisonOperator.GT,
            threshold=100.0,
            status=ConditionStatus.TRIGGERED,
            current_value=120.0,
            breach_magnitude=0.20,
            breach_velocity=0.05,
        )
        assert cond.breach_magnitude == 0.20

    def test_immutability(self, sample_invalidation_condition: InvalidationCondition) -> None:
        """InvalidationCondition should be frozen."""
        with pytest.raises(ValidationError):
            sample_invalidation_condition.status = ConditionStatus.TRIGGERED


# ===================================================================
# Belief Tests
# ===================================================================
class TestBelief:
    """Tests for Belief schema."""

    def test_valid_belief(self, sample_belief: Belief) -> None:
        """Valid belief should have all fields set correctly."""
        assert sample_belief.ticker == "AAPL"
        assert sample_belief.direction == Direction.LONG
        assert sample_belief.magnitude == Magnitude.MODERATE
        assert sample_belief.raw_confidence == 0.72
        assert sample_belief.time_horizon_days == 90

    def test_confidence_below_zero_rejected(
        self,
        sample_evidence_ref: EvidenceRef,
        sample_invalidation_condition: InvalidationCondition,
        sample_belief_metadata: BeliefMetadata,
    ) -> None:
        """raw_confidence below 0.0 must be rejected."""
        with pytest.raises(ValidationError):
            Belief(
                thesis_id="TEST",
                ticker="AAPL",
                thesis_summary="Test",
                direction=Direction.LONG,
                magnitude=Magnitude.SMALL,
                raw_confidence=-0.01,
                time_horizon_days=30,
                evidence=[sample_evidence_ref],
                invalidation_conditions=[sample_invalidation_condition],
                metadata=sample_belief_metadata,
            )

    def test_confidence_above_one_rejected(
        self,
        sample_evidence_ref: EvidenceRef,
        sample_invalidation_condition: InvalidationCondition,
        sample_belief_metadata: BeliefMetadata,
    ) -> None:
        """raw_confidence above 1.0 must be rejected."""
        with pytest.raises(ValidationError):
            Belief(
                thesis_id="TEST",
                ticker="AAPL",
                thesis_summary="Test",
                direction=Direction.LONG,
                magnitude=Magnitude.SMALL,
                raw_confidence=1.01,
                time_horizon_days=30,
                evidence=[sample_evidence_ref],
                invalidation_conditions=[sample_invalidation_condition],
                metadata=sample_belief_metadata,
            )

    def test_confidence_at_boundaries(
        self,
        sample_evidence_ref: EvidenceRef,
        sample_invalidation_condition: InvalidationCondition,
        sample_belief_metadata: BeliefMetadata,
    ) -> None:
        """raw_confidence at 0.0 and 1.0 should be accepted."""
        for conf in [0.0, 1.0]:
            belief = Belief(
                thesis_id="TEST",
                ticker="AAPL",
                thesis_summary="Test",
                direction=Direction.LONG,
                magnitude=Magnitude.SMALL,
                raw_confidence=conf,
                time_horizon_days=30,
                evidence=[sample_evidence_ref],
                invalidation_conditions=[sample_invalidation_condition],
                metadata=sample_belief_metadata,
            )
            assert belief.raw_confidence == conf

    def test_time_horizon_must_be_positive(
        self,
        sample_evidence_ref: EvidenceRef,
        sample_invalidation_condition: InvalidationCondition,
        sample_belief_metadata: BeliefMetadata,
    ) -> None:
        """time_horizon_days must be > 0."""
        with pytest.raises(ValidationError):
            Belief(
                thesis_id="TEST",
                ticker="AAPL",
                thesis_summary="Test",
                direction=Direction.LONG,
                magnitude=Magnitude.SMALL,
                raw_confidence=0.5,
                time_horizon_days=0,
                evidence=[sample_evidence_ref],
                invalidation_conditions=[sample_invalidation_condition],
                metadata=sample_belief_metadata,
            )

    def test_all_directions_accepted(
        self,
        sample_evidence_ref: EvidenceRef,
        sample_invalidation_condition: InvalidationCondition,
        sample_belief_metadata: BeliefMetadata,
    ) -> None:
        """All Direction enum values should be valid."""
        for direction in Direction:
            belief = Belief(
                thesis_id="TEST",
                ticker="AAPL",
                thesis_summary="Test",
                direction=direction,
                magnitude=Magnitude.SMALL,
                raw_confidence=0.5,
                time_horizon_days=30,
                evidence=[sample_evidence_ref],
                invalidation_conditions=[sample_invalidation_condition],
                metadata=sample_belief_metadata,
            )
            assert belief.direction == direction

    def test_all_magnitudes_accepted(
        self,
        sample_evidence_ref: EvidenceRef,
        sample_invalidation_condition: InvalidationCondition,
        sample_belief_metadata: BeliefMetadata,
    ) -> None:
        """All Magnitude enum values should be valid."""
        for mag in Magnitude:
            belief = Belief(
                thesis_id="TEST",
                ticker="AAPL",
                thesis_summary="Test",
                direction=Direction.LONG,
                magnitude=mag,
                raw_confidence=0.5,
                time_horizon_days=30,
                evidence=[sample_evidence_ref],
                invalidation_conditions=[sample_invalidation_condition],
                metadata=sample_belief_metadata,
            )
            assert belief.magnitude == mag

    def test_immutability(self, sample_belief: Belief) -> None:
        """Belief should be frozen."""
        with pytest.raises(ValidationError):
            sample_belief.raw_confidence = 0.99


# ===================================================================
# BeliefObject Tests
# ===================================================================
class TestBeliefObject:
    """Tests for BeliefObject schema."""

    def test_valid_belief_object(self, sample_belief_object: BeliefObject) -> None:
        """Valid BeliefObject should have all fields set correctly."""
        assert sample_belief_object.agent_id == "COGNIT-FUNDAMENTAL"
        assert len(sample_belief_object.beliefs) == 1
        assert sample_belief_object.beliefs[0].ticker == "AAPL"

    def test_content_hash_computed(self, sample_belief_object: BeliefObject) -> None:
        """Content hash should be computed automatically."""
        assert sample_belief_object.content_hash != ""
        assert len(sample_belief_object.content_hash) == 64

    def test_content_hash_deterministic(self, sample_belief: Belief) -> None:
        """Same beliefs must produce the same content hash."""
        ts = datetime(2026, 2, 9, 12, 0, 0, tzinfo=timezone.utc)
        ctx_hash = "abc123"

        b1 = BeliefObject(
            agent_id="COGNIT-FUNDAMENTAL",
            timestamp=ts,
            context_window_hash=ctx_hash,
            beliefs=[sample_belief],
        )
        b2 = BeliefObject(
            agent_id="COGNIT-FUNDAMENTAL",
            timestamp=ts,
            context_window_hash=ctx_hash,
            beliefs=[sample_belief],
        )
        assert b1.content_hash == b2.content_hash

    def test_different_beliefs_different_hash(
        self,
        sample_belief: Belief,
        sample_evidence_ref: EvidenceRef,
        sample_invalidation_condition: InvalidationCondition,
        sample_belief_metadata: BeliefMetadata,
    ) -> None:
        """Different beliefs must produce different content hashes."""
        ts = datetime(2026, 2, 9, 12, 0, 0, tzinfo=timezone.utc)

        different_belief = Belief(
            thesis_id="DIFFERENT-THESIS",
            ticker="MSFT",
            thesis_summary="Different thesis",
            direction=Direction.SHORT,
            magnitude=Magnitude.LARGE,
            raw_confidence=0.85,
            time_horizon_days=60,
            evidence=[sample_evidence_ref],
            invalidation_conditions=[sample_invalidation_condition],
            metadata=sample_belief_metadata,
        )

        b1 = BeliefObject(
            agent_id="COGNIT-FUNDAMENTAL",
            timestamp=ts,
            context_window_hash="hash1",
            beliefs=[sample_belief],
        )
        b2 = BeliefObject(
            agent_id="COGNIT-FUNDAMENTAL",
            timestamp=ts,
            context_window_hash="hash1",
            beliefs=[different_belief],
        )
        assert b1.content_hash != b2.content_hash

    def test_timestamp_requires_timezone(self, sample_belief: Belief) -> None:
        """BeliefObject timestamp must include timezone."""
        with pytest.raises(ValidationError, match="timezone"):
            BeliefObject(
                agent_id="COGNIT-FUNDAMENTAL",
                timestamp=datetime(2026, 2, 9, 12, 0, 0),  # no tzinfo
                context_window_hash="abc",
                beliefs=[sample_belief],
            )

    def test_empty_beliefs_accepted(self) -> None:
        """BeliefObject with empty beliefs list should be valid."""
        bo = BeliefObject(
            agent_id="COGNIT-FUNDAMENTAL",
            timestamp=datetime(2026, 2, 9, 12, 0, 0, tzinfo=timezone.utc),
            context_window_hash="abc",
            beliefs=[],
        )
        assert len(bo.beliefs) == 0
        assert bo.content_hash != ""  # hash of empty list

    def test_multiple_beliefs(
        self,
        sample_belief: Belief,
        sample_evidence_ref: EvidenceRef,
        sample_invalidation_condition: InvalidationCondition,
        sample_belief_metadata: BeliefMetadata,
    ) -> None:
        """BeliefObject can contain multiple beliefs."""
        second_belief = Belief(
            thesis_id="FUND-MSFT-2026Q1-CLOUD-GROWTH",
            ticker="MSFT",
            thesis_summary="Azure growth re-accelerating",
            direction=Direction.LONG,
            magnitude=Magnitude.LARGE,
            raw_confidence=0.81,
            time_horizon_days=120,
            evidence=[sample_evidence_ref],
            invalidation_conditions=[sample_invalidation_condition],
            metadata=BeliefMetadata(
                sector="Technology",
                market_cap_bucket=MarketCapBucket.MEGA,
                catalyst_type=CatalystType.EARNINGS,
            ),
        )
        bo = BeliefObject(
            agent_id="COGNIT-FUNDAMENTAL",
            timestamp=datetime(2026, 2, 9, 12, 0, 0, tzinfo=timezone.utc),
            context_window_hash="abc",
            beliefs=[sample_belief, second_belief],
        )
        assert len(bo.beliefs) == 2
        tickers = {b.ticker for b in bo.beliefs}
        assert tickers == {"AAPL", "MSFT"}

    def test_evidence_ref_links_to_fragment(
        self,
        sample_belief_object: BeliefObject,
    ) -> None:
        """Evidence refs should contain valid UUIDs linking to fragments."""
        for belief in sample_belief_object.beliefs:
            for evidence in belief.evidence:
                assert isinstance(evidence.source_fragment_id, UUID)
                assert evidence.field_path != ""
                assert evidence.observation != ""

    def test_immutability(self, sample_belief_object: BeliefObject) -> None:
        """BeliefObject should be frozen."""
        with pytest.raises(ValidationError):
            sample_belief_object.agent_id = "TAMPERED"

    def test_serialization_round_trip(self, sample_belief_object: BeliefObject) -> None:
        """BeliefObject should survive JSON serialization round-trip."""
        json_str = sample_belief_object.model_dump_json()
        restored = BeliefObject.model_validate_json(json_str)

        assert restored.belief_id == sample_belief_object.belief_id
        assert restored.content_hash == sample_belief_object.content_hash
        assert len(restored.beliefs) == len(sample_belief_object.beliefs)
        assert restored.beliefs[0].ticker == sample_belief_object.beliefs[0].ticker


# ===================================================================
# BeliefMetadata Tests
# ===================================================================
class TestBeliefMetadata:
    """Tests for BeliefMetadata schema."""

    def test_valid_metadata(self, sample_belief_metadata: BeliefMetadata) -> None:
        """Valid metadata should be created successfully."""
        assert sample_belief_metadata.sector == "Technology"
        assert sample_belief_metadata.market_cap_bucket == MarketCapBucket.MEGA
        assert sample_belief_metadata.catalyst_type == CatalystType.EARNINGS

    def test_catalyst_type_nullable(self) -> None:
        """catalyst_type can be None."""
        meta = BeliefMetadata(
            sector="Healthcare",
            market_cap_bucket=MarketCapBucket.LARGE,
            catalyst_type=None,
        )
        assert meta.catalyst_type is None

    def test_all_market_cap_buckets(self) -> None:
        """All MarketCapBucket values should be valid."""
        for bucket in MarketCapBucket:
            meta = BeliefMetadata(
                sector="Technology",
                market_cap_bucket=bucket,
            )
            assert meta.market_cap_bucket == bucket
