"""Shared test fixtures for Providence tests.

Provides sample MarketStateFragments and BeliefObjects that can be
reused across test modules.
"""

from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest

from providence.schemas.enums import (
    CatalystType,
    ComparisonOperator,
    ConditionStatus,
    DataType,
    Direction,
    Magnitude,
    MarketCapBucket,
    ValidationStatus,
)
from providence.schemas.market_state import MarketStateFragment
from providence.schemas.belief import (
    Belief,
    BeliefMetadata,
    BeliefObject,
    EvidenceRef,
    InvalidationCondition,
)


# ---------------------------------------------------------------------------
# Fixed UUIDs for deterministic testing
# ---------------------------------------------------------------------------
SAMPLE_FRAGMENT_ID = UUID("12345678-1234-5678-1234-567812345678")
SAMPLE_BELIEF_ID = UUID("abcdef01-abcd-abcd-abcd-abcdef012345")
SAMPLE_CONDITION_ID = UUID("cccccccc-cccc-cccc-cccc-cccccccccccc")
SAMPLE_EVIDENCE_FRAGMENT_ID = UUID("eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee")


# ---------------------------------------------------------------------------
# Timestamps
# ---------------------------------------------------------------------------
SAMPLE_TIMESTAMP = datetime(2026, 2, 9, 14, 30, 0, tzinfo=timezone.utc)
SAMPLE_SOURCE_TIMESTAMP = datetime(2026, 2, 9, 14, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# MarketStateFragment fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_price_payload() -> dict:
    """Sample OHLCV price payload."""
    return {
        "open": 185.50,
        "high": 187.20,
        "low": 184.80,
        "close": 186.90,
        "volume": 52_345_678,
        "vwap": 186.15,
        "timeframe": "1D",
    }


@pytest.fixture
def sample_fragment(sample_price_payload: dict) -> MarketStateFragment:
    """A valid MarketStateFragment with PRICE_OHLCV data."""
    return MarketStateFragment(
        fragment_id=SAMPLE_FRAGMENT_ID,
        agent_id="PERCEPT-PRICE",
        timestamp=SAMPLE_TIMESTAMP,
        source_timestamp=SAMPLE_SOURCE_TIMESTAMP,
        entity="AAPL",
        data_type=DataType.PRICE_OHLCV,
        schema_version="1.0.0",
        source_hash="abc123def456",
        validation_status=ValidationStatus.VALID,
        payload=sample_price_payload,
    )


@pytest.fixture
def sample_filing_payload() -> dict:
    """Sample 10-Q filing payload."""
    return {
        "filing_type": "10-Q",
        "company_name": "Apple Inc.",
        "cik": "0000320193",
        "ticker": "AAPL",
        "filed_date": "2026-01-30",
        "period_of_report": "2025-12-31",
        "revenue": 124_300_000_000,
        "net_income": 33_900_000_000,
        "eps": 2.18,
        "total_assets": 352_600_000_000,
        "total_liabilities": 274_800_000_000,
        "operating_cash_flow": 39_100_000_000,
    }


@pytest.fixture
def sample_filing_fragment(sample_filing_payload: dict) -> MarketStateFragment:
    """A valid MarketStateFragment with FILING_10Q data."""
    return MarketStateFragment(
        fragment_id=uuid4(),
        agent_id="PERCEPT-FILING",
        timestamp=SAMPLE_TIMESTAMP,
        source_timestamp=SAMPLE_SOURCE_TIMESTAMP,
        entity="AAPL",
        data_type=DataType.FILING_10Q,
        schema_version="1.0.0",
        source_hash="filing_hash_abc",
        validation_status=ValidationStatus.VALID,
        payload=sample_filing_payload,
    )


# ---------------------------------------------------------------------------
# Belief fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_evidence_ref() -> EvidenceRef:
    """A valid EvidenceRef linking to a MarketStateFragment."""
    return EvidenceRef(
        source_fragment_id=SAMPLE_EVIDENCE_FRAGMENT_ID,
        field_path="payload.revenue",
        observation="Revenue grew 12% YoY, exceeding consensus by 3%",
        weight=0.8,
    )


@pytest.fixture
def sample_invalidation_condition() -> InvalidationCondition:
    """A valid machine-evaluable InvalidationCondition."""
    return InvalidationCondition(
        condition_id=SAMPLE_CONDITION_ID,
        description="Revenue growth drops below 5% YoY",
        data_source_agent="PERCEPT-FILING",
        metric="revenue_growth_yoy",
        operator=ComparisonOperator.LT,
        threshold=0.05,
        status=ConditionStatus.ACTIVE,
        current_value=0.12,
    )


@pytest.fixture
def sample_belief_metadata() -> BeliefMetadata:
    """Sample belief metadata."""
    return BeliefMetadata(
        sector="Technology",
        market_cap_bucket=MarketCapBucket.MEGA,
        catalyst_type=CatalystType.EARNINGS,
    )


@pytest.fixture
def sample_belief(
    sample_evidence_ref: EvidenceRef,
    sample_invalidation_condition: InvalidationCondition,
    sample_belief_metadata: BeliefMetadata,
) -> Belief:
    """A valid Belief with evidence and invalidation conditions."""
    return Belief(
        thesis_id="FUND-AAPL-2026Q1-MARGIN-EXPANSION",
        ticker="AAPL",
        thesis_summary="Apple margin expansion driven by services mix shift",
        direction=Direction.LONG,
        magnitude=Magnitude.MODERATE,
        raw_confidence=0.72,
        time_horizon_days=90,
        evidence=[sample_evidence_ref],
        invalidation_conditions=[sample_invalidation_condition],
        correlated_beliefs=[],
        metadata=sample_belief_metadata,
    )


@pytest.fixture
def sample_belief_object(sample_belief: Belief) -> BeliefObject:
    """A valid BeliefObject containing one belief."""
    return BeliefObject(
        belief_id=SAMPLE_BELIEF_ID,
        agent_id="COGNIT-FUNDAMENTAL",
        timestamp=SAMPLE_TIMESTAMP,
        context_window_hash="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        beliefs=[sample_belief],
    )
