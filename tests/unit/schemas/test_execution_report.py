"""Tests for ExecutionReport schema.

Validates creation, immutability, content hashing, field constraints,
and ExecutionStatus enum values.
"""

from datetime import datetime, timezone
from uuid import uuid4

import pytest
from pydantic import ValidationError

from providence.schemas.enums import ExecutionStatus
from providence.schemas.execution import ExecutionReport


NOW = datetime.now(timezone.utc)
PROPOSAL_ID = uuid4()


def _make_execution_report(**overrides) -> ExecutionReport:
    """Create a test ExecutionReport with sensible defaults."""
    defaults = {
        "proposal_id": PROPOSAL_ID,
        "ticker": "AAPL",
        "status": ExecutionStatus.FILLED,
        "requested_weight": 0.05,
        "achieved_weight": 0.05,
        "avg_fill_price": 150.25,
        "benchmark_price": 150.00,
        "slippage_bps": 16.67,
        "market_impact_bps": 5.0,
        "execution_algo": "TWAP_15",
        "execution_duration_ms": 900000,
        "venue": "NASDAQ",
        "fees_bps": 1.0,
        "timestamp_start": NOW,
        "timestamp_end": NOW,
    }
    defaults.update(overrides)
    return ExecutionReport(**defaults)


class TestExecutionReportCreation:
    """Test ExecutionReport creation and field validation."""

    def test_create_valid_filled(self):
        """Valid filled execution report creates successfully."""
        report = _make_execution_report()
        assert report.ticker == "AAPL"
        assert report.status == ExecutionStatus.FILLED
        assert report.achieved_weight == 0.05
        assert report.venue == "NASDAQ"

    def test_execution_id_auto_generated(self):
        """execution_id is a UUID generated automatically."""
        report = _make_execution_report()
        assert report.execution_id is not None

    def test_content_hash_computed(self):
        """Content hash is computed automatically."""
        report = _make_execution_report()
        assert report.content_hash
        assert len(report.content_hash) == 64  # SHA-256 hex

    def test_same_data_same_hash(self):
        """Identical data produces identical content hash."""
        report1 = _make_execution_report(slippage_bps=10.0)
        report2 = _make_execution_report(slippage_bps=10.0)
        assert report1.content_hash == report2.content_hash

    def test_different_data_different_hash(self):
        """Different data produces different content hash."""
        report1 = _make_execution_report(slippage_bps=10.0)
        report2 = _make_execution_report(slippage_bps=20.0)
        assert report1.content_hash != report2.content_hash

    def test_immutability(self):
        """Frozen model rejects attribute mutation."""
        report = _make_execution_report()
        with pytest.raises(ValidationError):
            report.achieved_weight = 0.10

    def test_timestamp_start_requires_timezone(self):
        """timestamp_start without timezone is rejected."""
        with pytest.raises(ValidationError):
            _make_execution_report(timestamp_start=datetime(2026, 1, 1, 12, 0, 0))

    def test_timestamp_end_requires_timezone(self):
        """timestamp_end without timezone is rejected."""
        with pytest.raises(ValidationError):
            _make_execution_report(timestamp_end=datetime(2026, 1, 1, 12, 0, 0))

    def test_requested_weight_bounds(self):
        """Requested weight must be between 0.0 and 0.20."""
        with pytest.raises(ValidationError):
            _make_execution_report(requested_weight=-0.01)
        with pytest.raises(ValidationError):
            _make_execution_report(requested_weight=0.25)

    def test_achieved_weight_bounds(self):
        """Achieved weight must be between 0.0 and 0.20."""
        with pytest.raises(ValidationError):
            _make_execution_report(achieved_weight=-0.01)
        with pytest.raises(ValidationError):
            _make_execution_report(achieved_weight=0.25)

    def test_slippage_bps_non_negative(self):
        """Slippage must be non-negative."""
        with pytest.raises(ValidationError):
            _make_execution_report(slippage_bps=-1.0)

    def test_market_impact_bps_non_negative(self):
        """Market impact must be non-negative."""
        with pytest.raises(ValidationError):
            _make_execution_report(market_impact_bps=-1.0)

    def test_fees_bps_non_negative(self):
        """Fees must be non-negative."""
        with pytest.raises(ValidationError):
            _make_execution_report(fees_bps=-0.5)

    def test_avg_fill_price_positive(self):
        """Average fill price must be positive."""
        with pytest.raises(ValidationError):
            _make_execution_report(avg_fill_price=0.0)
        with pytest.raises(ValidationError):
            _make_execution_report(avg_fill_price=-150.0)

    def test_benchmark_price_positive(self):
        """Benchmark price must be positive."""
        with pytest.raises(ValidationError):
            _make_execution_report(benchmark_price=0.0)
        with pytest.raises(ValidationError):
            _make_execution_report(benchmark_price=-150.0)

    def test_execution_duration_non_negative(self):
        """Execution duration must be non-negative."""
        with pytest.raises(ValidationError):
            _make_execution_report(execution_duration_ms=-1)

    def test_constraint_violations_empty_default(self):
        """Constraint violations default to empty list."""
        report = _make_execution_report()
        assert report.constraint_violations == []

    def test_constraint_violations_with_items(self):
        """Constraint violations can contain error messages."""
        report = _make_execution_report(
            constraint_violations=["Position limit exceeded", "Daily volume limit hit"]
        )
        assert len(report.constraint_violations) == 2


class TestExecutionReportStatus:
    """Test ExecutionStatus enum and report status field."""

    def test_status_filled(self):
        """FILLED status is valid."""
        report = _make_execution_report(status=ExecutionStatus.FILLED)
        assert report.status == ExecutionStatus.FILLED

    def test_status_partial(self):
        """PARTIAL status is valid."""
        report = _make_execution_report(status=ExecutionStatus.PARTIAL, achieved_weight=0.03)
        assert report.status == ExecutionStatus.PARTIAL

    def test_status_rejected(self):
        """REJECTED status is valid."""
        report = _make_execution_report(
            status=ExecutionStatus.REJECTED, 
            achieved_weight=0.0,
            constraint_violations=["Insufficient liquidity"]
        )
        assert report.status == ExecutionStatus.REJECTED

    def test_status_cancelled(self):
        """CANCELLED status is valid."""
        report = _make_execution_report(
            status=ExecutionStatus.CANCELLED,
            achieved_weight=0.0
        )
        assert report.status == ExecutionStatus.CANCELLED

    def test_all_execution_status_values(self):
        """All ExecutionStatus enum values are accessible."""
        assert ExecutionStatus.FILLED.value == "FILLED"
        assert ExecutionStatus.PARTIAL.value == "PARTIAL"
        assert ExecutionStatus.REJECTED.value == "REJECTED"
        assert ExecutionStatus.CANCELLED.value == "CANCELLED"


class TestExecutionReportAlgos:
    """Test execution algorithm field."""

    def test_twap_algo(self):
        """TWAP_15 algorithm is accepted."""
        report = _make_execution_report(execution_algo="TWAP_15")
        assert report.execution_algo == "TWAP_15"

    def test_vwap_algo(self):
        """VWAP_30 algorithm is accepted."""
        report = _make_execution_report(execution_algo="VWAP_30")
        assert report.execution_algo == "VWAP_30"

    def test_limit_algo(self):
        """LIMIT algorithm is accepted."""
        report = _make_execution_report(execution_algo="LIMIT")
        assert report.execution_algo == "LIMIT"

    def test_market_algo(self):
        """MARKET algorithm is accepted."""
        report = _make_execution_report(execution_algo="MARKET")
        assert report.execution_algo == "MARKET"


class TestExecutionReportPartialFill:
    """Test partial fill scenarios."""

    def test_partial_fill_less_than_requested(self):
        """Partial fill with less weight than requested."""
        report = _make_execution_report(
            status=ExecutionStatus.PARTIAL,
            requested_weight=0.10,
            achieved_weight=0.06,
            constraint_violations=["Partial fill due to liquidity"]
        )
        assert report.requested_weight == 0.10
        assert report.achieved_weight == 0.06
        assert len(report.constraint_violations) == 1

    def test_partial_fill_zero_achieved(self):
        """Partial fill can result in zero achieved weight if timing issue."""
        report = _make_execution_report(
            status=ExecutionStatus.PARTIAL,
            achieved_weight=0.0,
            constraint_violations=["No shares filled"]
        )
        assert report.achieved_weight == 0.0
