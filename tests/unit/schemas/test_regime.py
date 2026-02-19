"""Tests for RegimeStateObject schema.

Validates creation, immutability, content hashing, and field constraints.
"""

from datetime import datetime, timezone
from uuid import uuid4

import pytest
from pydantic import ValidationError

from providence.schemas.enums import StatisticalRegime, SystemRiskMode
from providence.schemas.regime import RegimeStateObject


NOW = datetime.now(timezone.utc)

SAMPLE_PROBS = {
    "LOW_VOL_TRENDING": 0.60,
    "HIGH_VOL_MEAN_REVERTING": 0.25,
    "CRISIS_DISLOCATION": 0.05,
    "TRANSITION_UNCERTAIN": 0.10,
}


def _make_regime_state(**overrides) -> RegimeStateObject:
    """Create a test RegimeStateObject with sensible defaults."""
    defaults = {
        "agent_id": "REGIME-STAT",
        "timestamp": NOW,
        "context_window_hash": "test_regime_hash",
        "statistical_regime": StatisticalRegime.LOW_VOL_TRENDING,
        "regime_confidence": 0.60,
        "regime_probabilities": SAMPLE_PROBS,
        "system_risk_mode": SystemRiskMode.NORMAL,
        "features_used": {"realized_vol_20d": 0.12, "composite_score": 0.25},
    }
    defaults.update(overrides)
    return RegimeStateObject(**defaults)


class TestRegimeStateObjectCreation:
    """Test RegimeStateObject creation and field validation."""

    def test_create_valid(self):
        """Valid regime state object creates successfully."""
        obj = _make_regime_state()
        assert obj.agent_id == "REGIME-STAT"
        assert obj.statistical_regime == StatisticalRegime.LOW_VOL_TRENDING
        assert obj.regime_confidence == 0.60
        assert obj.system_risk_mode == SystemRiskMode.NORMAL

    def test_regime_id_auto_generated(self):
        """regime_id is a UUID generated automatically."""
        obj = _make_regime_state()
        assert obj.regime_id is not None

    def test_content_hash_computed(self):
        """Content hash is computed automatically."""
        obj = _make_regime_state()
        assert obj.content_hash
        assert len(obj.content_hash) == 64  # SHA-256 hex

    def test_same_data_same_hash(self):
        """Identical data produces identical content hash."""
        obj1 = _make_regime_state(regime_confidence=0.75)
        obj2 = _make_regime_state(regime_confidence=0.75)
        assert obj1.content_hash == obj2.content_hash

    def test_different_data_different_hash(self):
        """Different data produces different content hash."""
        obj1 = _make_regime_state(regime_confidence=0.60)
        obj2 = _make_regime_state(regime_confidence=0.80)
        assert obj1.content_hash != obj2.content_hash

    def test_immutability(self):
        """Frozen model rejects attribute mutation."""
        obj = _make_regime_state()
        with pytest.raises(ValidationError):
            obj.regime_confidence = 0.99

    def test_confidence_bounds(self):
        """Confidence must be between 0.0 and 1.0."""
        with pytest.raises(ValidationError):
            _make_regime_state(regime_confidence=1.5)
        with pytest.raises(ValidationError):
            _make_regime_state(regime_confidence=-0.1)

    def test_negative_probability_rejected(self):
        """Negative probabilities are rejected."""
        bad_probs = {
            "LOW_VOL_TRENDING": -0.1,
            "HIGH_VOL_MEAN_REVERTING": 0.5,
            "CRISIS_DISLOCATION": 0.3,
            "TRANSITION_UNCERTAIN": 0.3,
        }
        with pytest.raises(ValidationError):
            _make_regime_state(regime_probabilities=bad_probs)
