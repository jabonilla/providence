"""Tests for AgentScorecard schema.

Validates creation, immutability, content hashing, nested model validation,
and AgentRecommendation enum values.
"""

import pytest
from pydantic import ValidationError

from providence.schemas.enums import AgentRecommendation, StatisticalRegime
from providence.schemas.learning import (
    AgentScorecard,
    ScorecardCalibration,
    RegimePerformance,
    ConflictRecord,
    RenewalRecord,
    InvalidationRecord,
)


def _make_scorecard(**overrides) -> AgentScorecard:
    """Create a test AgentScorecard with sensible defaults."""
    defaults = {
        "agent_id": "agent_123",
        "review_period": "2026-Q1",
        "beliefs_generated": 150,
        "beliefs_acted_on": 120,
        "hit_rate_raw": 0.65,
        "hit_rate_calibrated": 0.62,
        "avg_pnl_per_belief_bps": 25.5,
        "sharpe_contribution": 0.45,
        "information_ratio": 1.25,
        "max_drawdown_contribution_bps": 50.0,
        "calibration": ScorecardCalibration(
            overall_brier_score=0.18,
            calibration_curve={
                "0.5-0.6": {"stated": 0.55, "realized": 0.58},
                "0.6-0.7": {"stated": 0.65, "realized": 0.62},
            },
            is_overconfident=False,
            recommended_adjustment=0.98,
        ),
        "recommendation": AgentRecommendation.MAINTAIN,
        "recommendation_rationale": "Solid performance, within expected bounds",
    }
    defaults.update(overrides)
    return AgentScorecard(**defaults)


class TestAgentScorecardCreation:
    """Test AgentScorecard creation and field validation."""

    def test_create_valid(self):
        """Valid scorecard creates successfully."""
        scorecard = _make_scorecard()
        assert scorecard.agent_id == "agent_123"
        assert scorecard.review_period == "2026-Q1"
        assert scorecard.hit_rate_calibrated == 0.62
        assert scorecard.recommendation == AgentRecommendation.MAINTAIN

    def test_content_hash_computed(self):
        """Content hash is computed automatically."""
        scorecard = _make_scorecard()
        assert scorecard.content_hash
        assert len(scorecard.content_hash) == 64  # SHA-256 hex

    def test_same_data_same_hash(self):
        """Identical data produces identical content hash."""
        scorecard1 = _make_scorecard(hit_rate_calibrated=0.60)
        scorecard2 = _make_scorecard(hit_rate_calibrated=0.60)
        assert scorecard1.content_hash == scorecard2.content_hash

    def test_different_data_different_hash(self):
        """Different data produces different content hash."""
        scorecard1 = _make_scorecard(hit_rate_calibrated=0.60)
        scorecard2 = _make_scorecard(hit_rate_calibrated=0.70)
        assert scorecard1.content_hash != scorecard2.content_hash

    def test_immutability(self):
        """Frozen model rejects attribute mutation."""
        scorecard = _make_scorecard()
        with pytest.raises(ValidationError):
            scorecard.hit_rate_calibrated = 0.99

    def test_beliefs_generated_non_negative(self):
        """Beliefs generated must be non-negative."""
        with pytest.raises(ValidationError):
            _make_scorecard(beliefs_generated=-1)

    def test_beliefs_acted_on_non_negative(self):
        """Beliefs acted on must be non-negative."""
        with pytest.raises(ValidationError):
            _make_scorecard(beliefs_acted_on=-1)

    def test_hit_rate_raw_bounds(self):
        """Raw hit rate must be between 0.0 and 1.0."""
        with pytest.raises(ValidationError):
            _make_scorecard(hit_rate_raw=-0.1)
        with pytest.raises(ValidationError):
            _make_scorecard(hit_rate_raw=1.5)

    def test_hit_rate_calibrated_bounds(self):
        """Calibrated hit rate must be between 0.0 and 1.0."""
        with pytest.raises(ValidationError):
            _make_scorecard(hit_rate_calibrated=-0.1)
        with pytest.raises(ValidationError):
            _make_scorecard(hit_rate_calibrated=1.5)

    def test_max_drawdown_contribution_non_negative(self):
        """Max drawdown contribution must be non-negative."""
        with pytest.raises(ValidationError):
            _make_scorecard(max_drawdown_contribution_bps=-1.0)


class TestAgentScorecardRecommendation:
    """Test AgentRecommendation enum and recommendation field."""

    def test_recommendation_promote(self):
        """PROMOTE recommendation is valid."""
        scorecard = _make_scorecard(
            recommendation=AgentRecommendation.PROMOTE,
            recommendation_rationale="Excellent outperformance",
        )
        assert scorecard.recommendation == AgentRecommendation.PROMOTE

    def test_recommendation_maintain(self):
        """MAINTAIN recommendation is valid."""
        scorecard = _make_scorecard(recommendation=AgentRecommendation.MAINTAIN)
        assert scorecard.recommendation == AgentRecommendation.MAINTAIN

    def test_recommendation_retrain(self):
        """RETRAIN recommendation is valid."""
        scorecard = _make_scorecard(
            recommendation=AgentRecommendation.RETRAIN,
            recommendation_rationale="Performance degradation detected",
        )
        assert scorecard.recommendation == AgentRecommendation.RETRAIN

    def test_recommendation_retire(self):
        """RETIRE recommendation is valid."""
        scorecard = _make_scorecard(
            recommendation=AgentRecommendation.RETIRE,
            recommendation_rationale="Consistent underperformance",
        )
        assert scorecard.recommendation == AgentRecommendation.RETIRE

    def test_all_agent_recommendation_values(self):
        """All AgentRecommendation enum values are accessible."""
        assert AgentRecommendation.PROMOTE.value == "PROMOTE"
        assert AgentRecommendation.MAINTAIN.value == "MAINTAIN"
        assert AgentRecommendation.RETRAIN.value == "RETRAIN"
        assert AgentRecommendation.RETIRE.value == "RETIRE"


class TestScorecardCalibration:
    """Test nested ScorecardCalibration model."""

    def test_calibration_creation(self):
        """Calibration object creates with valid data."""
        calib = ScorecardCalibration(
            overall_brier_score=0.20,
            is_overconfident=True,
            recommended_adjustment=0.85,
        )
        assert calib.overall_brier_score == 0.20
        assert calib.is_overconfident is True
        assert calib.recommended_adjustment == 0.85

    def test_calibration_immutable(self):
        """Calibration object is frozen."""
        calib = ScorecardCalibration()
        with pytest.raises(ValidationError):
            calib.overall_brier_score = 0.50

    def test_calibration_brier_score_non_negative(self):
        """Brier score must be non-negative."""
        with pytest.raises(ValidationError):
            ScorecardCalibration(overall_brier_score=-0.1)

    def test_calibration_adjustment_positive(self):
        """Recommended adjustment must be positive."""
        with pytest.raises(ValidationError):
            ScorecardCalibration(recommended_adjustment=0.0)
        with pytest.raises(ValidationError):
            ScorecardCalibration(recommended_adjustment=-0.5)


class TestRegimePerformance:
    """Test nested RegimePerformance model."""

    def test_regime_performance_creation(self):
        """RegimePerformance object creates with valid data."""
        perf = RegimePerformance(
            hit_rate=0.68,
            avg_pnl=35.0,
            sharpe=0.55,
        )
        assert perf.hit_rate == 0.68
        assert perf.avg_pnl == 35.0
        assert perf.sharpe == 0.55

    def test_regime_performance_immutable(self):
        """RegimePerformance object is frozen."""
        perf = RegimePerformance()
        with pytest.raises(ValidationError):
            perf.hit_rate = 0.80

    def test_regime_performance_hit_rate_bounds(self):
        """Hit rate must be between 0.0 and 1.0."""
        with pytest.raises(ValidationError):
            RegimePerformance(hit_rate=-0.1)
        with pytest.raises(ValidationError):
            RegimePerformance(hit_rate=1.5)


class TestConflictRecord:
    """Test nested ConflictRecord model."""

    def test_conflict_record_creation(self):
        """ConflictRecord creates with valid data."""
        record = ConflictRecord(
            times_in_conflict=10,
            times_conflict_won=7,
            conflict_win_rate=0.70,
            common_opponents=["agent_A", "agent_B"],
        )
        assert record.times_in_conflict == 10
        assert record.times_conflict_won == 7
        assert record.conflict_win_rate == 0.70
        assert len(record.common_opponents) == 2

    def test_conflict_record_immutable(self):
        """ConflictRecord is frozen."""
        record = ConflictRecord()
        with pytest.raises(ValidationError):
            record.times_in_conflict = 5

    def test_conflict_counts_non_negative(self):
        """Conflict counts must be non-negative."""
        with pytest.raises(ValidationError):
            ConflictRecord(times_in_conflict=-1)
        with pytest.raises(ValidationError):
            ConflictRecord(times_conflict_won=-1)


class TestRenewalRecord:
    """Test nested RenewalRecord model."""

    def test_renewal_record_creation(self):
        """RenewalRecord creates with valid data."""
        record = RenewalRecord(
            renewals_issued=25,
            renewals_value_added=20,
            avg_renewal_pnl_bps=30.0,
            renewal_quality_score=0.85,
        )
        assert record.renewals_issued == 25
        assert record.renewals_value_added == 20
        assert record.avg_renewal_pnl_bps == 30.0
        assert record.renewal_quality_score == 0.85

    def test_renewal_record_immutable(self):
        """RenewalRecord is frozen."""
        record = RenewalRecord()
        with pytest.raises(ValidationError):
            record.renewals_issued = 10

    def test_renewal_counts_non_negative(self):
        """Renewal counts must be non-negative."""
        with pytest.raises(ValidationError):
            RenewalRecord(renewals_issued=-1)
        with pytest.raises(ValidationError):
            RenewalRecord(renewals_value_added=-1)

    def test_renewal_quality_score_bounds(self):
        """Renewal quality score must be between 0.0 and 1.0."""
        with pytest.raises(ValidationError):
            RenewalRecord(renewal_quality_score=-0.1)
        with pytest.raises(ValidationError):
            RenewalRecord(renewal_quality_score=1.5)


class TestInvalidationRecord:
    """Test nested InvalidationRecord model."""

    def test_invalidation_record_creation(self):
        """InvalidationRecord creates with valid data."""
        record = InvalidationRecord(
            invalidations_triggered=15,
            invalidations_correct=12,
            invalidation_accuracy=0.80,
        )
        assert record.invalidations_triggered == 15
        assert record.invalidations_correct == 12
        assert record.invalidation_accuracy == 0.80

    def test_invalidation_record_immutable(self):
        """InvalidationRecord is frozen."""
        record = InvalidationRecord()
        with pytest.raises(ValidationError):
            record.invalidations_triggered = 5

    def test_invalidation_counts_non_negative(self):
        """Invalidation counts must be non-negative."""
        with pytest.raises(ValidationError):
            InvalidationRecord(invalidations_triggered=-1)
        with pytest.raises(ValidationError):
            InvalidationRecord(invalidations_correct=-1)

    def test_invalidation_accuracy_bounds(self):
        """Invalidation accuracy must be between 0.0 and 1.0."""
        with pytest.raises(ValidationError):
            InvalidationRecord(invalidation_accuracy=-0.1)
        with pytest.raises(ValidationError):
            InvalidationRecord(invalidation_accuracy=1.5)


class TestAgentScorecardNested:
    """Test AgentScorecard with nested models."""

    def test_regime_performance_map(self):
        """Regime performance can be a dict of regime -> performance."""
        scorecard = _make_scorecard(
            regime_performance={
                "LOW_VOL_TRENDING": RegimePerformance(hit_rate=0.70, avg_pnl=40.0, sharpe=0.60),
                "HIGH_VOL_MEAN_REVERTING": RegimePerformance(hit_rate=0.55, avg_pnl=15.0, sharpe=0.25),
            }
        )
        assert len(scorecard.regime_performance) == 2
        assert scorecard.regime_performance["LOW_VOL_TRENDING"].hit_rate == 0.70

    def test_default_nested_models(self):
        """Nested models have sensible defaults when not provided."""
        scorecard = _make_scorecard(
            calibration=ScorecardCalibration(),
            conflict_record=ConflictRecord(),
            renewal_record=RenewalRecord(),
            invalidation_record=InvalidationRecord(),
        )
        assert scorecard.calibration.overall_brier_score == 0.0
        assert scorecard.conflict_record.times_in_conflict == 0
        assert scorecard.renewal_record.renewals_issued == 0
        assert scorecard.invalidation_record.invalidations_triggered == 0

    def test_comprehensive_scorecard(self):
        """Full scorecard with all nested models populated."""
        scorecard = AgentScorecard(
            agent_id="agent_comprehensive",
            review_period="2026-Q1",
            beliefs_generated=500,
            beliefs_acted_on=400,
            hit_rate_raw=0.72,
            hit_rate_calibrated=0.70,
            avg_pnl_per_belief_bps=45.0,
            sharpe_contribution=0.75,
            information_ratio=1.85,
            max_drawdown_contribution_bps=75.0,
            calibration=ScorecardCalibration(
                overall_brier_score=0.15,
                is_overconfident=False,
                recommended_adjustment=1.02,
            ),
            regime_performance={
                "LOW_VOL_TRENDING": RegimePerformance(hit_rate=0.75, avg_pnl=55.0, sharpe=0.85),
                "HIGH_VOL_MEAN_REVERTING": RegimePerformance(hit_rate=0.65, avg_pnl=30.0, sharpe=0.55),
                "CRISIS_DISLOCATION": RegimePerformance(hit_rate=0.60, avg_pnl=20.0, sharpe=0.35),
            },
            conflict_record=ConflictRecord(
                times_in_conflict=8,
                times_conflict_won=5,
                conflict_win_rate=0.625,
                common_opponents=["agent_A", "agent_D"],
            ),
            renewal_record=RenewalRecord(
                renewals_issued=35,
                renewals_value_added=28,
                avg_renewal_pnl_bps=50.0,
                renewal_quality_score=0.92,
            ),
            invalidation_record=InvalidationRecord(
                invalidations_triggered=12,
                invalidations_correct=11,
                invalidation_accuracy=0.917,
            ),
            recommendation=AgentRecommendation.PROMOTE,
            recommendation_rationale="Consistently strong performance across all regimes",
        )
        assert scorecard.agent_id == "agent_comprehensive"
        assert scorecard.recommendation == AgentRecommendation.PROMOTE
        assert len(scorecard.regime_performance) == 3
        assert scorecard.renewal_record.renewals_issued == 35
