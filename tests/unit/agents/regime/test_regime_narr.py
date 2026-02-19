"""Tests for REGIME-NARR Narrative Regime Classification Agent.

Tests the full narrative regime pipeline with mocked LLM client:
response parsing, narrative overlay construction, regime alignment,
prompt building, error handling, and health reporting.

REGIME-NARR is ADAPTIVE: uses Claude Sonnet 4 for narrative analysis.
"""

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest

from providence.agents.base import AgentContext, AgentStatus
from providence.agents.regime.regime_narr import RegimeNarr, parse_narrative_response
from providence.exceptions import AgentProcessingError
from providence.schemas.enums import (
    DataType,
    StatisticalRegime,
    SystemRiskMode,
    ValidationStatus,
)
from providence.schemas.market_state import MarketStateFragment
from providence.schemas.regime import NarrativeRegimeOverlay, RegimeStateObject

from tests.fixtures.llm_responses import (
    CRISIS_NARRATIVE_REGIME_RESPONSE,
    MALFORMED_NARRATIVE_REGIME_RESPONSE,
    VALID_NARRATIVE_REGIME_RESPONSE,
    WEAK_NARRATIVE_REGIME_RESPONSE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
NOW = datetime.now(timezone.utc)

NEWS_FRAG_ID = UUID("eeeeee01-eeee-eeee-eeee-eeeeeeeeeeee")
EARNINGS_FRAG_ID = UUID("eeeeee02-eeee-eeee-eeee-eeeeeeeeeeee")
PRICE_FRAG_ID = UUID("eeeeee03-eeee-eeee-eeee-eeeeeeeeeeee")
YIELD_FRAG_ID = UUID("eeeeee04-eeee-eeee-eeee-eeeeeeeeeeee")


def _make_news_fragment(
    fragment_id: UUID | None = None,
    hours_ago: float = 1,
) -> MarketStateFragment:
    """Create a SENTIMENT_NEWS test fragment."""
    ts = NOW - timedelta(hours=hours_ago)
    return MarketStateFragment(
        fragment_id=fragment_id or NEWS_FRAG_ID,
        agent_id="PERCEPT-NEWS",
        timestamp=ts,
        source_timestamp=ts,
        entity="NVDA",
        data_type=DataType.SENTIMENT_NEWS,
        schema_version="1.0.0",
        source_hash="hash-news-nvda",
        validation_status=ValidationStatus.VALID,
        payload={
            "headline": "NVIDIA beats earnings expectations, AI demand surges",
            "sentiment_score": 0.85,
            "source": "Reuters",
            "category": "earnings",
        },
    )


def _make_earnings_fragment(
    fragment_id: UUID | None = None,
    hours_ago: float = 1,
) -> MarketStateFragment:
    """Create an EARNINGS_CALL test fragment."""
    ts = NOW - timedelta(hours=hours_ago)
    return MarketStateFragment(
        fragment_id=fragment_id or EARNINGS_FRAG_ID,
        agent_id="PERCEPT-FILING",
        timestamp=ts,
        source_timestamp=ts,
        entity="MSFT",
        data_type=DataType.EARNINGS_CALL,
        schema_version="1.0.0",
        source_hash="hash-earnings-msft",
        validation_status=ValidationStatus.VALID,
        payload={
            "quarter": "Q1-2026",
            "key_quotes": [
                "AI cloud revenue grew 45% year-over-year",
                "We are seeing unprecedented enterprise AI adoption",
            ],
            "guidance": "raised",
            "eps_surprise_pct": 8.2,
        },
    )


def _make_price_fragment(
    close: float = 500.0,
    entity: str = "SPY",
    hours_ago: float = 1,
) -> MarketStateFragment:
    """Create a PRICE_OHLCV test fragment."""
    ts = NOW - timedelta(hours=hours_ago)
    return MarketStateFragment(
        fragment_id=PRICE_FRAG_ID if entity == "SPY" else uuid4(),
        agent_id="PERCEPT-PRICE",
        timestamp=ts,
        source_timestamp=ts,
        entity=entity,
        data_type=DataType.PRICE_OHLCV,
        schema_version="1.0.0",
        source_hash=f"hash-price-{entity}",
        validation_status=ValidationStatus.VALID,
        payload={"open": close * 0.99, "high": close * 1.01, "low": close * 0.98, "close": close, "volume": 50000000},
    )


def _make_yield_curve_fragment(
    spread_2s10s: float = 50.0,
    hours_ago: float = 1,
) -> MarketStateFragment:
    """Create a MACRO_YIELD_CURVE test fragment."""
    ts = NOW - timedelta(hours=hours_ago)
    return MarketStateFragment(
        fragment_id=YIELD_FRAG_ID,
        agent_id="PERCEPT-MACRO",
        timestamp=ts,
        source_timestamp=ts,
        entity=None,
        data_type=DataType.MACRO_YIELD_CURVE,
        schema_version="1.0.0",
        source_hash="hash-yield",
        validation_status=ValidationStatus.VALID,
        payload={"spreads": {"2s10s": spread_2s10s}, "curve_date": "2026-02-16"},
    )


def _make_context(
    fragments: list[MarketStateFragment] | None = None,
) -> AgentContext:
    """Create a test AgentContext for REGIME-NARR."""
    if fragments is None:
        fragments = [
            _make_news_fragment(),
            _make_earnings_fragment(),
            _make_price_fragment(),
            _make_yield_curve_fragment(),
        ]
    return AgentContext(
        agent_id="REGIME-NARR",
        trigger="schedule",
        fragments=fragments,
        context_window_hash="narr_test_hash",
        timestamp=NOW,
        metadata={},
    )


def _mock_llm_client(response_dict: dict) -> AsyncMock:
    """Create a mock LLM client that returns the given response as JSON."""
    mock = AsyncMock()
    mock.complete = AsyncMock(return_value=json.dumps(response_dict))
    return mock


# ---------------------------------------------------------------------------
# Response parser tests
# ---------------------------------------------------------------------------
class TestParseNarrativeResponse:
    """Tests for the narrative response parser."""

    def test_valid_response(self):
        """Parser extracts all fields from valid response."""
        raw = json.dumps(VALID_NARRATIVE_REGIME_RESPONSE)
        parsed = parse_narrative_response(raw)

        assert parsed is not None
        assert parsed["narrative_label"] == "AI-driven tech euphoria with rotation risk"
        assert parsed["narrative_confidence"] == 0.68
        assert len(parsed["key_signals"]) == 4
        assert "Technology" in parsed["affected_sectors"]
        assert parsed["regime_alignment"] == "DIVERGES"
        assert parsed["statistical_regime"] == StatisticalRegime.LOW_VOL_TRENDING
        assert parsed["risk_mode"] == SystemRiskMode.CAUTIOUS

    def test_crisis_response(self):
        """Parser handles crisis narrative response."""
        raw = json.dumps(CRISIS_NARRATIVE_REGIME_RESPONSE)
        parsed = parse_narrative_response(raw)

        assert parsed is not None
        assert parsed["narrative_label"] == "Banking contagion fear and credit stress"
        assert parsed["narrative_confidence"] == 0.75
        assert parsed["regime_alignment"] == "CONFIRMS"
        assert parsed["statistical_regime"] == StatisticalRegime.CRISIS_DISLOCATION

    def test_weak_response(self):
        """Parser handles weak/ambiguous narrative."""
        raw = json.dumps(WEAK_NARRATIVE_REGIME_RESPONSE)
        parsed = parse_narrative_response(raw)

        assert parsed is not None
        assert parsed["narrative_confidence"] == 0.28
        assert parsed["regime_alignment"] == "NEUTRAL"
        assert len(parsed["affected_sectors"]) == 0

    def test_malformed_response_returns_none(self):
        """Missing narrative_label returns None."""
        raw = json.dumps(MALFORMED_NARRATIVE_REGIME_RESPONSE)
        parsed = parse_narrative_response(raw)
        assert parsed is None

    def test_non_json_returns_none(self):
        """Non-JSON response returns None."""
        parsed = parse_narrative_response("This is not JSON at all.")
        assert parsed is None

    def test_empty_string_returns_none(self):
        """Empty string returns None."""
        parsed = parse_narrative_response("")
        assert parsed is None

    def test_confidence_clamped_at_080(self):
        """Confidence above 0.80 is clamped."""
        response = dict(VALID_NARRATIVE_REGIME_RESPONSE)
        response["narrative_confidence"] = 0.95
        raw = json.dumps(response)
        parsed = parse_narrative_response(raw)

        assert parsed is not None
        assert parsed["narrative_confidence"] == 0.80

    def test_confidence_clamped_at_zero(self):
        """Negative confidence is clamped to 0.0."""
        response = dict(VALID_NARRATIVE_REGIME_RESPONSE)
        response["narrative_confidence"] = -0.5
        raw = json.dumps(response)
        parsed = parse_narrative_response(raw)

        assert parsed is not None
        assert parsed["narrative_confidence"] == 0.0

    def test_json_in_markdown_fences(self):
        """Parser extracts JSON from markdown code fences."""
        raw = "```json\n" + json.dumps(VALID_NARRATIVE_REGIME_RESPONSE) + "\n```"
        parsed = parse_narrative_response(raw)
        assert parsed is not None
        assert parsed["narrative_label"] == "AI-driven tech euphoria with rotation risk"

    def test_json_with_surrounding_text(self):
        """Parser extracts JSON from surrounding text."""
        raw = "Here is my analysis:\n" + json.dumps(VALID_NARRATIVE_REGIME_RESPONSE) + "\nEnd."
        parsed = parse_narrative_response(raw)
        assert parsed is not None

    def test_invalid_alignment_defaults_to_neutral(self):
        """Invalid regime_alignment defaults to NEUTRAL."""
        response = dict(VALID_NARRATIVE_REGIME_RESPONSE)
        response["regime_alignment"] = "INVALID_VALUE"
        raw = json.dumps(response)
        parsed = parse_narrative_response(raw)

        assert parsed is not None
        assert parsed["regime_alignment"] == "NEUTRAL"

    def test_missing_optional_fields(self):
        """Missing optional fields get defaults."""
        minimal = {
            "narrative_label": "Test label",
            "narrative_confidence": 0.50,
        }
        raw = json.dumps(minimal)
        parsed = parse_narrative_response(raw)

        assert parsed is not None
        assert parsed["key_signals"] == []
        assert parsed["affected_sectors"] == []
        assert parsed["regime_alignment"] == "NEUTRAL"
        assert parsed["summary"] == ""
        assert parsed["statistical_regime"] is None
        assert parsed["risk_mode"] is None

    def test_label_truncated_at_200(self):
        """Long labels are truncated."""
        response = dict(VALID_NARRATIVE_REGIME_RESPONSE)
        response["narrative_label"] = "A" * 300
        raw = json.dumps(response)
        parsed = parse_narrative_response(raw)

        assert parsed is not None
        assert len(parsed["narrative_label"]) == 200


# ---------------------------------------------------------------------------
# RegimeNarr agent tests
# ---------------------------------------------------------------------------
class TestRegimeNarr:
    """Tests for the RegimeNarr agent."""

    @pytest.mark.asyncio
    async def test_process_returns_regime_state_object(self):
        """Agent process() returns a valid RegimeStateObject."""
        mock_llm = _mock_llm_client(VALID_NARRATIVE_REGIME_RESPONSE)
        agent = RegimeNarr(llm_client=mock_llm)
        context = _make_context()

        result = await agent.process(context)

        assert isinstance(result, RegimeStateObject)
        assert result.agent_id == "REGIME-NARR"

    @pytest.mark.asyncio
    async def test_process_populates_narrative_overlay(self):
        """Output has narrative overlay populated."""
        mock_llm = _mock_llm_client(VALID_NARRATIVE_REGIME_RESPONSE)
        agent = RegimeNarr(llm_client=mock_llm)
        context = _make_context()

        result = await agent.process(context)

        assert result.narrative_overlay is not None
        assert isinstance(result.narrative_overlay, NarrativeRegimeOverlay)
        assert result.narrative_overlay.label == "AI-driven tech euphoria with rotation risk"
        assert result.narrative_overlay.confidence == 0.68

    @pytest.mark.asyncio
    async def test_process_narrative_key_signals(self):
        """Narrative overlay has key signals from LLM."""
        mock_llm = _mock_llm_client(VALID_NARRATIVE_REGIME_RESPONSE)
        agent = RegimeNarr(llm_client=mock_llm)
        context = _make_context()

        result = await agent.process(context)

        assert len(result.narrative_overlay.key_signals) == 4

    @pytest.mark.asyncio
    async def test_process_narrative_affected_sectors(self):
        """Narrative overlay has affected sectors."""
        mock_llm = _mock_llm_client(VALID_NARRATIVE_REGIME_RESPONSE)
        agent = RegimeNarr(llm_client=mock_llm)
        context = _make_context()

        result = await agent.process(context)

        assert "Technology" in result.narrative_overlay.affected_sectors

    @pytest.mark.asyncio
    async def test_process_regime_alignment(self):
        """Narrative overlay has correct regime alignment."""
        mock_llm = _mock_llm_client(VALID_NARRATIVE_REGIME_RESPONSE)
        agent = RegimeNarr(llm_client=mock_llm)
        context = _make_context()

        result = await agent.process(context)

        assert result.narrative_overlay.regime_alignment == "DIVERGES"

    @pytest.mark.asyncio
    async def test_process_uses_llm_regime_assessment(self):
        """Agent uses LLM's regime assessment when available."""
        mock_llm = _mock_llm_client(VALID_NARRATIVE_REGIME_RESPONSE)
        agent = RegimeNarr(llm_client=mock_llm)
        context = _make_context()

        result = await agent.process(context)

        assert result.statistical_regime == StatisticalRegime.LOW_VOL_TRENDING

    @pytest.mark.asyncio
    async def test_process_uses_llm_risk_mode(self):
        """Agent uses LLM's risk mode suggestion when available."""
        mock_llm = _mock_llm_client(VALID_NARRATIVE_REGIME_RESPONSE)
        agent = RegimeNarr(llm_client=mock_llm)
        context = _make_context()

        result = await agent.process(context)

        assert result.system_risk_mode == SystemRiskMode.CAUTIOUS

    @pytest.mark.asyncio
    async def test_process_crisis_narrative(self):
        """Crisis narrative populates correctly."""
        mock_llm = _mock_llm_client(CRISIS_NARRATIVE_REGIME_RESPONSE)
        agent = RegimeNarr(llm_client=mock_llm)
        context = _make_context()

        result = await agent.process(context)

        assert result.narrative_overlay.label == "Banking contagion fear and credit stress"
        assert result.narrative_overlay.regime_alignment == "CONFIRMS"
        assert result.statistical_regime == StatisticalRegime.CRISIS_DISLOCATION
        assert result.system_risk_mode == SystemRiskMode.DEFENSIVE

    @pytest.mark.asyncio
    async def test_process_context_window_hash_preserved(self):
        """Output carries the context_window_hash."""
        mock_llm = _mock_llm_client(VALID_NARRATIVE_REGIME_RESPONSE)
        agent = RegimeNarr(llm_client=mock_llm)
        context = _make_context()

        result = await agent.process(context)

        assert result.context_window_hash == "narr_test_hash"

    @pytest.mark.asyncio
    async def test_process_content_hash_computed(self):
        """Output has a non-empty content hash."""
        mock_llm = _mock_llm_client(VALID_NARRATIVE_REGIME_RESPONSE)
        agent = RegimeNarr(llm_client=mock_llm)
        context = _make_context()

        result = await agent.process(context)

        assert result.content_hash
        assert len(result.content_hash) == 64

    @pytest.mark.asyncio
    async def test_process_malformed_response_raises(self):
        """Malformed LLM response raises AgentProcessingError."""
        mock_llm = _mock_llm_client(MALFORMED_NARRATIVE_REGIME_RESPONSE)
        agent = RegimeNarr(llm_client=mock_llm)
        context = _make_context()

        with pytest.raises(AgentProcessingError, match="Failed to parse"):
            await agent.process(context)

    @pytest.mark.asyncio
    async def test_process_llm_exception_raises(self):
        """LLM client exception raises AgentProcessingError."""
        mock_llm = AsyncMock()
        mock_llm.complete = AsyncMock(side_effect=RuntimeError("API timeout"))
        agent = RegimeNarr(llm_client=mock_llm)
        context = _make_context()

        with pytest.raises(AgentProcessingError, match="REGIME-NARR processing failed"):
            await agent.process(context)

    @pytest.mark.asyncio
    async def test_process_empty_fragments(self):
        """Agent handles empty fragment list."""
        mock_llm = _mock_llm_client(WEAK_NARRATIVE_REGIME_RESPONSE)
        agent = RegimeNarr(llm_client=mock_llm)
        context = _make_context(fragments=[])

        result = await agent.process(context)

        assert isinstance(result, RegimeStateObject)
        assert result.narrative_overlay is not None

    @pytest.mark.asyncio
    async def test_process_probabilities_sum_to_one(self):
        """Regime probabilities sum to ~1.0."""
        mock_llm = _mock_llm_client(VALID_NARRATIVE_REGIME_RESPONSE)
        agent = RegimeNarr(llm_client=mock_llm)
        context = _make_context()

        result = await agent.process(context)

        total = sum(result.regime_probabilities.values())
        assert total == pytest.approx(1.0, abs=1e-4)

    @pytest.mark.asyncio
    async def test_llm_called_with_system_and_user_prompt(self):
        """LLM client receives system and user prompts."""
        mock_llm = _mock_llm_client(VALID_NARRATIVE_REGIME_RESPONSE)
        agent = RegimeNarr(llm_client=mock_llm)
        context = _make_context()

        await agent.process(context)

        mock_llm.complete.assert_called_once()
        call_kwargs = mock_llm.complete.call_args
        assert "system_prompt" in call_kwargs.kwargs or len(call_kwargs.args) >= 1
        assert "user_prompt" in call_kwargs.kwargs or len(call_kwargs.args) >= 2


# ---------------------------------------------------------------------------
# Prompt building tests
# ---------------------------------------------------------------------------
class TestRegimeNarrPromptBuilding:
    """Tests for prompt construction."""

    @pytest.mark.asyncio
    async def test_prompt_includes_news_data(self):
        """User prompt includes news fragment data."""
        mock_llm = _mock_llm_client(VALID_NARRATIVE_REGIME_RESPONSE)
        agent = RegimeNarr(llm_client=mock_llm)
        context = _make_context()

        await agent.process(context)

        user_prompt = mock_llm.complete.call_args.kwargs.get(
            "user_prompt", mock_llm.complete.call_args.args[1] if len(mock_llm.complete.call_args.args) > 1 else ""
        )
        assert "NVDA" in user_prompt or "SENTIMENT_NEWS" in user_prompt

    @pytest.mark.asyncio
    async def test_prompt_includes_earnings_data(self):
        """User prompt includes earnings fragment data."""
        mock_llm = _mock_llm_client(VALID_NARRATIVE_REGIME_RESPONSE)
        agent = RegimeNarr(llm_client=mock_llm)
        context = _make_context()

        await agent.process(context)

        user_prompt = mock_llm.complete.call_args.kwargs.get(
            "user_prompt", mock_llm.complete.call_args.args[1] if len(mock_llm.complete.call_args.args) > 1 else ""
        )
        assert "MSFT" in user_prompt or "EARNINGS_CALL" in user_prompt

    @pytest.mark.asyncio
    async def test_prompt_includes_statistical_regime(self):
        """User prompt includes current statistical regime."""
        mock_llm = _mock_llm_client(VALID_NARRATIVE_REGIME_RESPONSE)
        agent = RegimeNarr(llm_client=mock_llm)
        context = _make_context()

        await agent.process(context)

        user_prompt = mock_llm.complete.call_args.kwargs.get(
            "user_prompt", mock_llm.complete.call_args.args[1] if len(mock_llm.complete.call_args.args) > 1 else ""
        )
        # Should contain one of the regime state values
        has_regime = any(
            r.value in user_prompt for r in StatisticalRegime
        )
        assert has_regime

    @pytest.mark.asyncio
    async def test_prompt_handles_missing_data_types(self):
        """Prompt handles missing news/earnings gracefully."""
        mock_llm = _mock_llm_client(WEAK_NARRATIVE_REGIME_RESPONSE)
        agent = RegimeNarr(llm_client=mock_llm)
        # Only price data, no news or earnings
        context = _make_context(fragments=[_make_price_fragment()])

        result = await agent.process(context)

        assert isinstance(result, RegimeStateObject)
        user_prompt = mock_llm.complete.call_args.kwargs.get(
            "user_prompt", mock_llm.complete.call_args.args[1] if len(mock_llm.complete.call_args.args) > 1 else ""
        )
        assert "No news sentiment data available" in user_prompt
        assert "No earnings call data available" in user_prompt


# ---------------------------------------------------------------------------
# Health status tests
# ---------------------------------------------------------------------------
class TestRegimeNarrHealth:
    """Tests for agent health reporting."""

    def test_initial_health_is_healthy(self):
        """Agent starts in HEALTHY state."""
        mock_llm = _mock_llm_client(VALID_NARRATIVE_REGIME_RESPONSE)
        agent = RegimeNarr(llm_client=mock_llm)
        health = agent.get_health()
        assert health.agent_id == "REGIME-NARR"
        assert health.status == AgentStatus.HEALTHY
        assert health.error_count_24h == 0

    @pytest.mark.asyncio
    async def test_health_after_success(self):
        """Health reports last_success after successful run."""
        mock_llm = _mock_llm_client(VALID_NARRATIVE_REGIME_RESPONSE)
        agent = RegimeNarr(llm_client=mock_llm)
        context = _make_context()

        await agent.process(context)

        health = agent.get_health()
        assert health.last_run is not None
        assert health.last_success is not None
        assert health.status == AgentStatus.HEALTHY

    def test_health_degraded_after_errors(self):
        """Health degrades after repeated errors."""
        mock_llm = _mock_llm_client(VALID_NARRATIVE_REGIME_RESPONSE)
        agent = RegimeNarr(llm_client=mock_llm)
        agent._error_count_24h = 4
        health = agent.get_health()
        assert health.status == AgentStatus.DEGRADED

    def test_health_unhealthy_after_many_errors(self):
        """Health becomes UNHEALTHY after 11+ errors."""
        mock_llm = _mock_llm_client(VALID_NARRATIVE_REGIME_RESPONSE)
        agent = RegimeNarr(llm_client=mock_llm)
        agent._error_count_24h = 11
        health = agent.get_health()
        assert health.status == AgentStatus.UNHEALTHY


# ---------------------------------------------------------------------------
# Agent properties
# ---------------------------------------------------------------------------
class TestRegimeNarrProperties:
    """Test agent identity properties."""

    def test_agent_id(self):
        mock_llm = _mock_llm_client(VALID_NARRATIVE_REGIME_RESPONSE)
        agent = RegimeNarr(llm_client=mock_llm)
        assert agent.agent_id == "REGIME-NARR"

    def test_agent_type(self):
        mock_llm = _mock_llm_client(VALID_NARRATIVE_REGIME_RESPONSE)
        agent = RegimeNarr(llm_client=mock_llm)
        assert agent.agent_type == "regime"

    def test_version(self):
        mock_llm = _mock_llm_client(VALID_NARRATIVE_REGIME_RESPONSE)
        agent = RegimeNarr(llm_client=mock_llm)
        assert agent.version == "1.0.0"

    def test_consumed_data_types(self):
        """Agent consumes the correct data types."""
        expected = {
            DataType.SENTIMENT_NEWS,
            DataType.EARNINGS_CALL,
            DataType.MACRO_YIELD_CURVE,
            DataType.MACRO_CDS,
            DataType.MACRO_ECONOMIC,
            DataType.PRICE_OHLCV,
        }
        assert RegimeNarr.CONSUMED_DATA_TYPES == expected
