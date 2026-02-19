"""Tests for REGIME-MISMATCH Regime Divergence Detection Agent.

Tests mismatch computation between statistical, narrative, and sector
regime classifications. Validates risk mode escalation, divergence
counting, mismatch signal generation, and error handling.

REGIME-MISMATCH is FROZEN: zero LLM calls. Pure computation.
"""

from datetime import datetime, timezone

import pytest

from providence.agents.base import AgentContext, AgentStatus
from providence.agents.regime.regime_mismatch import (
    RegimeMismatch,
    compute_mismatch_score,
    compute_regime_distance,
    count_sector_divergences,
    escalate_risk_mode,
    identify_mismatch_signals,
)
from providence.schemas.enums import StatisticalRegime, SystemRiskMode
from providence.schemas.regime import (
    NarrativeRegimeOverlay,
    RegimeStateObject,
    SectorRegimeOverlay,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
NOW = datetime.now(timezone.utc)

DEFAULT_PROBS = {
    StatisticalRegime.LOW_VOL_TRENDING.value: 0.60,
    StatisticalRegime.HIGH_VOL_MEAN_REVERTING.value: 0.20,
    StatisticalRegime.CRISIS_DISLOCATION.value: 0.05,
    StatisticalRegime.TRANSITION_UNCERTAIN.value: 0.15,
}


def _make_stat_metadata(
    regime: str = "LOW_VOL_TRENDING",
    confidence: float = 0.70,
    risk_mode: str = "NORMAL",
    features: dict | None = None,
) -> dict:
    """Create stat_regime metadata dict."""
    return {
        "statistical_regime": regime,
        "regime_confidence": confidence,
        "system_risk_mode": risk_mode,
        "regime_probabilities": DEFAULT_PROBS,
        "features_used": features or {"composite_score": 0.25},
    }


def _make_narr_metadata(
    label: str = "AI-driven tech euphoria",
    confidence: float = 0.65,
    alignment: str = "CONFIRMS",
    key_signals: list | None = None,
    affected_sectors: list | None = None,
) -> dict:
    """Create narr_regime metadata dict."""
    return {
        "narrative_overlay": {
            "label": label,
            "confidence": confidence,
            "regime_alignment": alignment,
            "key_signals": key_signals or ["signal 1"],
            "affected_sectors": affected_sectors or ["Technology"],
            "summary": "Test narrative summary.",
        },
    }


def _make_sector_metadata(
    sectors: dict[str, str] | None = None,
) -> dict:
    """Create sector_regime metadata dict.

    Args:
        sectors: Dict of sector_name → regime_value. Defaults to
                 Technology=LOW_VOL, Financials=LOW_VOL.
    """
    if sectors is None:
        sectors = {
            "Technology": "LOW_VOL_TRENDING",
            "Financials": "LOW_VOL_TRENDING",
        }

    overlays = {}
    for sector_name, regime_val in sectors.items():
        overlays[sector_name] = {
            "regime": regime_val,
            "regime_confidence": 0.60,
            "regime_probabilities": DEFAULT_PROBS,
            "relative_stress": 0.0,
            "key_signals": [],
            "ticker_count": 3,
        }

    return {"sector_overlays": overlays}


def _make_context(
    stat: dict | None = None,
    narr: dict | None = None,
    sector: dict | None = None,
) -> AgentContext:
    """Create a test AgentContext for REGIME-MISMATCH."""
    metadata = {}
    if stat is not None:
        metadata["stat_regime"] = stat
    if narr is not None:
        metadata["narr_regime"] = narr
    if sector is not None:
        metadata["sector_regime"] = sector

    return AgentContext(
        agent_id="REGIME-MISMATCH",
        trigger="schedule",
        fragments=[],
        context_window_hash="mismatch_test_hash",
        timestamp=NOW,
        metadata=metadata,
    )


def _make_sector_overlay(
    sector: str,
    regime: StatisticalRegime = StatisticalRegime.LOW_VOL_TRENDING,
) -> SectorRegimeOverlay:
    """Create a SectorRegimeOverlay for unit tests."""
    return SectorRegimeOverlay(
        sector=sector,
        regime=regime,
        regime_confidence=0.60,
        regime_probabilities=DEFAULT_PROBS,
    )


# ---------------------------------------------------------------------------
# Regime distance tests
# ---------------------------------------------------------------------------
class TestRegimeDistance:
    """Tests for regime severity distance computation."""

    def test_same_regime_zero_distance(self):
        """Same regime → distance 0."""
        d = compute_regime_distance(
            StatisticalRegime.LOW_VOL_TRENDING,
            StatisticalRegime.LOW_VOL_TRENDING,
        )
        assert d == 0

    def test_adjacent_distance_one(self):
        """Adjacent regimes → distance 1."""
        d = compute_regime_distance(
            StatisticalRegime.LOW_VOL_TRENDING,
            StatisticalRegime.TRANSITION_UNCERTAIN,
        )
        assert d == 1

    def test_opposite_extremes_distance_three(self):
        """LOW_VOL vs CRISIS → distance 3."""
        d = compute_regime_distance(
            StatisticalRegime.LOW_VOL_TRENDING,
            StatisticalRegime.CRISIS_DISLOCATION,
        )
        assert d == 3

    def test_distance_is_symmetric(self):
        """Distance is the same regardless of order."""
        d1 = compute_regime_distance(
            StatisticalRegime.HIGH_VOL_MEAN_REVERTING,
            StatisticalRegime.CRISIS_DISLOCATION,
        )
        d2 = compute_regime_distance(
            StatisticalRegime.CRISIS_DISLOCATION,
            StatisticalRegime.HIGH_VOL_MEAN_REVERTING,
        )
        assert d1 == d2


# ---------------------------------------------------------------------------
# Mismatch score tests
# ---------------------------------------------------------------------------
class TestMismatchScore:
    """Tests for composite mismatch score computation."""

    def test_no_mismatch(self):
        """All signals confirm → low mismatch score."""
        score = compute_mismatch_score(
            stat_regime=StatisticalRegime.LOW_VOL_TRENDING,
            stat_confidence=0.70,
            narrative_alignment="CONFIRMS",
            narrative_confidence=0.65,
            sector_divergence_count=0,
            total_sectors=3,
        )
        assert score < 0.10

    def test_narrative_diverges_high_mismatch(self):
        """Narrative DIVERGES → elevated mismatch score."""
        score = compute_mismatch_score(
            stat_regime=StatisticalRegime.LOW_VOL_TRENDING,
            stat_confidence=0.70,
            narrative_alignment="DIVERGES",
            narrative_confidence=0.70,
            sector_divergence_count=0,
            total_sectors=0,
        )
        assert score > 0.40

    def test_sector_divergence_increases_score(self):
        """Sector divergences increase mismatch score."""
        score_no_diverge = compute_mismatch_score(
            stat_regime=StatisticalRegime.LOW_VOL_TRENDING,
            stat_confidence=0.70,
            narrative_alignment="CONFIRMS",
            narrative_confidence=0.65,
            sector_divergence_count=0,
            total_sectors=5,
        )
        score_with_diverge = compute_mismatch_score(
            stat_regime=StatisticalRegime.LOW_VOL_TRENDING,
            stat_confidence=0.70,
            narrative_alignment="CONFIRMS",
            narrative_confidence=0.65,
            sector_divergence_count=4,
            total_sectors=5,
        )
        assert score_with_diverge > score_no_diverge

    def test_neutral_alignment_moderate_score(self):
        """NEUTRAL alignment → moderate mismatch contribution."""
        score = compute_mismatch_score(
            stat_regime=StatisticalRegime.LOW_VOL_TRENDING,
            stat_confidence=0.70,
            narrative_alignment="NEUTRAL",
            narrative_confidence=0.50,
            sector_divergence_count=0,
            total_sectors=0,
        )
        assert 0.10 < score < 0.40

    def test_all_diverging_maximum_mismatch(self):
        """All signals diverging → near-maximum mismatch."""
        score = compute_mismatch_score(
            stat_regime=StatisticalRegime.LOW_VOL_TRENDING,
            stat_confidence=0.80,
            narrative_alignment="DIVERGES",
            narrative_confidence=0.80,
            sector_divergence_count=5,
            total_sectors=5,
        )
        assert score > 0.70

    def test_score_clamped_to_unit_interval(self):
        """Score is always in [0, 1]."""
        score = compute_mismatch_score(
            stat_regime=StatisticalRegime.CRISIS_DISLOCATION,
            stat_confidence=1.0,
            narrative_alignment="DIVERGES",
            narrative_confidence=1.0,
            sector_divergence_count=10,
            total_sectors=10,
        )
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Sector divergence counting tests
# ---------------------------------------------------------------------------
class TestSectorDivergence:
    """Tests for sector divergence counting."""

    def test_no_divergence(self):
        """All sectors match global → 0 divergences."""
        overlays = {
            "Technology": _make_sector_overlay("Technology", StatisticalRegime.LOW_VOL_TRENDING),
            "Financials": _make_sector_overlay("Financials", StatisticalRegime.LOW_VOL_TRENDING),
        }
        count = count_sector_divergences(StatisticalRegime.LOW_VOL_TRENDING, overlays)
        assert count == 0

    def test_one_diverging_sector(self):
        """One sector in CRISIS vs global LOW_VOL → 1 divergence."""
        overlays = {
            "Technology": _make_sector_overlay("Technology", StatisticalRegime.LOW_VOL_TRENDING),
            "Financials": _make_sector_overlay("Financials", StatisticalRegime.CRISIS_DISLOCATION),
        }
        count = count_sector_divergences(StatisticalRegime.LOW_VOL_TRENDING, overlays)
        assert count == 1

    def test_adjacent_regime_not_divergence(self):
        """Adjacent regime (distance 1) is NOT counted as divergence."""
        overlays = {
            "Technology": _make_sector_overlay("Technology", StatisticalRegime.TRANSITION_UNCERTAIN),
        }
        count = count_sector_divergences(StatisticalRegime.LOW_VOL_TRENDING, overlays)
        assert count == 0

    def test_empty_overlays(self):
        """Empty overlays → 0 divergences."""
        count = count_sector_divergences(StatisticalRegime.LOW_VOL_TRENDING, {})
        assert count == 0


# ---------------------------------------------------------------------------
# Risk mode escalation tests
# ---------------------------------------------------------------------------
class TestRiskModeEscalation:
    """Tests for risk mode escalation based on mismatch score."""

    def test_no_escalation_low_mismatch(self):
        """Low mismatch score → no escalation."""
        result = escalate_risk_mode(SystemRiskMode.NORMAL, 0.15)
        assert result == SystemRiskMode.NORMAL

    def test_escalate_one_level(self):
        """Moderate mismatch → escalate by 1 level."""
        result = escalate_risk_mode(SystemRiskMode.NORMAL, 0.45)
        assert result == SystemRiskMode.CAUTIOUS

    def test_escalate_two_levels(self):
        """High mismatch → escalate by 2 levels."""
        result = escalate_risk_mode(SystemRiskMode.NORMAL, 0.75)
        assert result == SystemRiskMode.DEFENSIVE

    def test_escalation_caps_at_halted(self):
        """Escalation cannot exceed HALTED."""
        result = escalate_risk_mode(SystemRiskMode.DEFENSIVE, 0.75)
        assert result == SystemRiskMode.HALTED

    def test_already_halted_stays_halted(self):
        """Already HALTED stays HALTED regardless of score."""
        result = escalate_risk_mode(SystemRiskMode.HALTED, 0.80)
        assert result == SystemRiskMode.HALTED

    def test_boundary_030(self):
        """Exactly 0.30 → no escalation (threshold is exclusive)."""
        result = escalate_risk_mode(SystemRiskMode.NORMAL, 0.29)
        assert result == SystemRiskMode.NORMAL

    def test_boundary_060(self):
        """Just above 0.60 → 2-level escalation."""
        result = escalate_risk_mode(SystemRiskMode.NORMAL, 0.61)
        assert result == SystemRiskMode.DEFENSIVE


# ---------------------------------------------------------------------------
# Mismatch signals tests
# ---------------------------------------------------------------------------
class TestMismatchSignals:
    """Tests for mismatch signal generation."""

    def test_diverges_signal(self):
        """DIVERGES alignment produces divergence signal."""
        signals = identify_mismatch_signals(
            stat_regime=StatisticalRegime.LOW_VOL_TRENDING,
            narrative_alignment="DIVERGES",
            narrative_label="banking contagion fear",
            sector_divergence_count=0,
            total_sectors=0,
            mismatch_score=0.50,
        )
        assert any("stat-narrative divergence" in s for s in signals)

    def test_sector_divergence_signal(self):
        """Sector divergences produce sector signal."""
        signals = identify_mismatch_signals(
            stat_regime=StatisticalRegime.LOW_VOL_TRENDING,
            narrative_alignment="CONFIRMS",
            narrative_label="calm markets",
            sector_divergence_count=2,
            total_sectors=5,
            mismatch_score=0.20,
        )
        assert any("2/5 sectors" in s for s in signals)

    def test_no_mismatch_signal(self):
        """No mismatch produces 'no significant mismatch' signal."""
        signals = identify_mismatch_signals(
            stat_regime=StatisticalRegime.LOW_VOL_TRENDING,
            narrative_alignment="CONFIRMS",
            narrative_label="calm markets",
            sector_divergence_count=0,
            total_sectors=3,
            mismatch_score=0.05,
        )
        assert any("no significant" in s for s in signals)

    def test_severe_mismatch_signal(self):
        """High mismatch score → severe escalation signal."""
        signals = identify_mismatch_signals(
            stat_regime=StatisticalRegime.LOW_VOL_TRENDING,
            narrative_alignment="DIVERGES",
            narrative_label="crisis",
            sector_divergence_count=3,
            total_sectors=5,
            mismatch_score=0.75,
        )
        assert any("severe mismatch" in s for s in signals)


# ---------------------------------------------------------------------------
# RegimeMismatch agent tests
# ---------------------------------------------------------------------------
class TestRegimeMismatch:
    """Tests for the RegimeMismatch agent."""

    @pytest.mark.asyncio
    async def test_process_returns_regime_state_object(self):
        """Agent process() returns a valid RegimeStateObject."""
        agent = RegimeMismatch()
        context = _make_context(
            stat=_make_stat_metadata(),
            narr=_make_narr_metadata(),
        )

        result = await agent.process(context)

        assert isinstance(result, RegimeStateObject)
        assert result.agent_id == "REGIME-MISMATCH"

    @pytest.mark.asyncio
    async def test_no_mismatch_preserves_risk_mode(self):
        """Confirming narrative → no risk mode escalation."""
        agent = RegimeMismatch()
        context = _make_context(
            stat=_make_stat_metadata(regime="LOW_VOL_TRENDING", risk_mode="NORMAL"),
            narr=_make_narr_metadata(alignment="CONFIRMS"),
            sector=_make_sector_metadata(),
        )

        result = await agent.process(context)

        assert result.system_risk_mode == SystemRiskMode.NORMAL

    @pytest.mark.asyncio
    async def test_narrative_divergence_escalates(self):
        """Diverging narrative with high confidence → risk mode escalation."""
        agent = RegimeMismatch()
        context = _make_context(
            stat=_make_stat_metadata(regime="LOW_VOL_TRENDING", confidence=0.75, risk_mode="NORMAL"),
            narr=_make_narr_metadata(alignment="DIVERGES", confidence=0.70),
        )

        result = await agent.process(context)

        # Mismatch score should be high enough to escalate
        assert RISK_MODE_SEVERITY[result.system_risk_mode] > RISK_MODE_SEVERITY[SystemRiskMode.NORMAL]

    @pytest.mark.asyncio
    async def test_sector_divergence_escalates(self):
        """Multiple sectors diverging → risk mode escalation."""
        agent = RegimeMismatch()
        context = _make_context(
            stat=_make_stat_metadata(regime="LOW_VOL_TRENDING", risk_mode="NORMAL"),
            narr=_make_narr_metadata(alignment="NEUTRAL", confidence=0.40),
            sector=_make_sector_metadata(sectors={
                "Technology": "CRISIS_DISLOCATION",
                "Financials": "CRISIS_DISLOCATION",
                "Healthcare": "LOW_VOL_TRENDING",
            }),
        )

        result = await agent.process(context)

        # With NEUTRAL narrative + 2/3 sectors diverging, should escalate
        assert result.system_risk_mode != SystemRiskMode.NORMAL

    @pytest.mark.asyncio
    async def test_features_include_mismatch_metrics(self):
        """Output features_used includes mismatch metrics."""
        agent = RegimeMismatch()
        context = _make_context(
            stat=_make_stat_metadata(),
            narr=_make_narr_metadata(),
        )

        result = await agent.process(context)

        assert "mismatch_score" in result.features_used
        assert "sector_divergence_count" in result.features_used
        assert "total_sectors" in result.features_used

    @pytest.mark.asyncio
    async def test_narrative_overlay_preserved(self):
        """Narrative overlay from upstream is preserved in output."""
        agent = RegimeMismatch()
        context = _make_context(
            stat=_make_stat_metadata(),
            narr=_make_narr_metadata(label="Test narrative label"),
        )

        result = await agent.process(context)

        assert result.narrative_overlay is not None
        assert result.narrative_overlay.label == "Test narrative label"

    @pytest.mark.asyncio
    async def test_sector_overlays_preserved(self):
        """Sector overlays from upstream are preserved in output."""
        agent = RegimeMismatch()
        context = _make_context(
            stat=_make_stat_metadata(),
            sector=_make_sector_metadata(sectors={"Technology": "LOW_VOL_TRENDING"}),
        )

        result = await agent.process(context)

        assert "Technology" in result.sector_overlays

    @pytest.mark.asyncio
    async def test_stat_only_no_escalation(self):
        """Stat regime only (no narrative/sector) → no escalation."""
        agent = RegimeMismatch()
        context = _make_context(
            stat=_make_stat_metadata(regime="HIGH_VOL_MEAN_REVERTING", risk_mode="CAUTIOUS"),
        )

        result = await agent.process(context)

        # NEUTRAL narrative alignment with no sectors → low mismatch
        assert result.statistical_regime == StatisticalRegime.HIGH_VOL_MEAN_REVERTING

    @pytest.mark.asyncio
    async def test_empty_metadata_uses_defaults(self):
        """Empty metadata uses safe defaults."""
        agent = RegimeMismatch()
        context = _make_context()

        result = await agent.process(context)

        assert isinstance(result, RegimeStateObject)
        assert result.statistical_regime == StatisticalRegime.LOW_VOL_TRENDING

    @pytest.mark.asyncio
    async def test_context_window_hash_preserved(self):
        """Output preserves context_window_hash."""
        agent = RegimeMismatch()
        context = _make_context(stat=_make_stat_metadata())

        result = await agent.process(context)

        assert result.context_window_hash == "mismatch_test_hash"

    @pytest.mark.asyncio
    async def test_content_hash_computed(self):
        """Output has a valid content hash."""
        agent = RegimeMismatch()
        context = _make_context(stat=_make_stat_metadata())

        result = await agent.process(context)

        assert result.content_hash
        assert len(result.content_hash) == 64

    @pytest.mark.asyncio
    async def test_probabilities_sum_to_one(self):
        """Regime probabilities sum to ~1.0."""
        agent = RegimeMismatch()
        context = _make_context(stat=_make_stat_metadata())

        result = await agent.process(context)

        total = sum(result.regime_probabilities.values())
        assert total == pytest.approx(1.0, abs=1e-4)


# ---------------------------------------------------------------------------
# Health status tests
# ---------------------------------------------------------------------------
class TestRegimeMismatchHealth:
    """Tests for agent health reporting."""

    def test_initial_health_is_healthy(self):
        """Agent starts in HEALTHY state."""
        agent = RegimeMismatch()
        health = agent.get_health()
        assert health.agent_id == "REGIME-MISMATCH"
        assert health.status == AgentStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_health_after_success(self):
        """Health reports last_success after successful run."""
        agent = RegimeMismatch()
        context = _make_context(stat=_make_stat_metadata())

        await agent.process(context)

        health = agent.get_health()
        assert health.last_run is not None
        assert health.last_success is not None

    def test_health_degraded(self):
        agent = RegimeMismatch()
        agent._error_count_24h = 4
        assert agent.get_health().status == AgentStatus.DEGRADED

    def test_health_unhealthy(self):
        agent = RegimeMismatch()
        agent._error_count_24h = 11
        assert agent.get_health().status == AgentStatus.UNHEALTHY


# ---------------------------------------------------------------------------
# Agent properties
# ---------------------------------------------------------------------------
class TestRegimeMismatchProperties:
    """Test agent identity properties."""

    def test_agent_id(self):
        assert RegimeMismatch().agent_id == "REGIME-MISMATCH"

    def test_agent_type(self):
        assert RegimeMismatch().agent_type == "regime"

    def test_version(self):
        assert RegimeMismatch().version == "1.0.0"

    def test_consumed_data_types_empty(self):
        """REGIME-MISMATCH doesn't consume raw market data."""
        assert RegimeMismatch.CONSUMED_DATA_TYPES == set()


# Need this import for the test that checks escalation
RISK_MODE_SEVERITY = {
    SystemRiskMode.NORMAL: 0,
    SystemRiskMode.CAUTIOUS: 1,
    SystemRiskMode.DEFENSIVE: 2,
    SystemRiskMode.HALTED: 3,
}
