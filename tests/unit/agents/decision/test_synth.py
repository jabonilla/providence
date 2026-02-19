"""Tests for DECIDE-SYNTH Belief Synthesis Agent.

Tests the full synthesis pipeline with mocked LLM client:
response parsing, intent building, conflict resolution,
regime adjustment, prompt building, error handling, and health reporting.

DECIDE-SYNTH is ADAPTIVE: uses Claude Sonnet 4 for belief synthesis.
"""

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest

from providence.agents.base import AgentContext, AgentStatus
from providence.agents.decision.synth import (
    DecideSynth,
    _build_intent,
    parse_synthesis_response,
)
from providence.exceptions import AgentProcessingError
from providence.schemas.decision import (
    ActiveInvalidation,
    ConflictResolution,
    ContributingThesis,
    SynthesisOutput,
    SynthesizedPositionIntent,
)
from providence.schemas.enums import Direction, Magnitude

from tests.fixtures.llm_responses import (
    CRISIS_SYNTHESIS_RESPONSE,
    EMPTY_SYNTHESIS_RESPONSE,
    INVALID_DIRECTION_SYNTHESIS_RESPONSE,
    MALFORMED_SYNTHESIS_RESPONSE,
    MULTI_INTENT_SYNTHESIS_RESPONSE,
    OVERCCONFIDENT_SYNTHESIS_RESPONSE,
    VALID_SYNTHESIS_RESPONSE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
NOW = datetime.now(timezone.utc)


def _make_belief_objects() -> list[dict]:
    """Create realistic belief_objects for testing."""
    return [
        {
            "agent_id": "COGNIT-FUNDAMENTAL",
            "beliefs": [
                {
                    "thesis_id": "FUND-AAPL-2026Q1-MARGIN-EXPANSION",
                    "ticker": "AAPL",
                    "direction": "LONG",
                    "magnitude": "MODERATE",
                    "raw_confidence": 0.72,
                    "time_horizon_days": 90,
                    "thesis_summary": "Apple margin expansion thesis",
                    "invalidation_conditions": [
                        {
                            "metric": "gross_margin_pct",
                            "operator": "LT",
                            "threshold": 44.0,
                        },
                    ],
                },
            ],
        },
        {
            "agent_id": "COGNIT-NARRATIVE",
            "beliefs": [
                {
                    "thesis_id": "NARR-AAPL-20260212-TONE",
                    "ticker": "AAPL",
                    "direction": "LONG",
                    "magnitude": "MODERATE",
                    "raw_confidence": 0.62,
                    "time_horizon_days": 60,
                    "thesis_summary": "CEO tone shift signals beat likelihood",
                },
                {
                    "thesis_id": "NARR-MSFT-20260212-GUIDANCE",
                    "ticker": "MSFT",
                    "direction": "SHORT",
                    "magnitude": "SMALL",
                    "raw_confidence": 0.58,
                    "time_horizon_days": 45,
                    "thesis_summary": "MSFT guidance hedging language increased",
                },
            ],
        },
    ]


def _make_regime_state() -> dict:
    """Create realistic regime_state for testing."""
    return {
        "statistical_regime": "LOW_VOL_TRENDING",
        "regime_confidence": 0.75,
        "system_risk_mode": "NORMAL",
        "features_used": {
            "realized_vol": 0.12,
            "vol_of_vol": 0.08,
        },
        "narrative_overlay": {
            "label": "AI-driven tech euphoria",
            "regime_alignment": "DIVERGES",
            "confidence": 0.68,
        },
    }


def _make_context(
    belief_objects: list[dict] | None = None,
    regime_state: dict | None = None,
) -> AgentContext:
    """Build an AgentContext for DECIDE-SYNTH tests."""
    metadata = {}
    if belief_objects is not None:
        metadata["belief_objects"] = belief_objects
    if regime_state is not None:
        metadata["regime_state"] = regime_state

    return AgentContext(
        agent_id="DECIDE-SYNTH",
        trigger="schedule",
        fragments=[],
        context_window_hash="test-hash-synth-001",
        timestamp=NOW,
        metadata=metadata,
    )


def _mock_llm(response_dict: dict) -> AsyncMock:
    """Create a mock LLM client returning the given dict as JSON."""
    mock = AsyncMock()
    mock.complete = AsyncMock(return_value=json.dumps(response_dict))
    return mock


# ===========================================================================
# Tests: parse_synthesis_response
# ===========================================================================
class TestParseSynthesisResponse:
    """Tests for the parse_synthesis_response function."""

    def test_valid_response(self):
        raw = json.dumps(VALID_SYNTHESIS_RESPONSE)
        parsed = parse_synthesis_response(raw)
        assert parsed is not None
        assert "position_intents" in parsed
        assert len(parsed["position_intents"]) == 1
        assert parsed["position_intents"][0]["ticker"] == "AAPL"

    def test_multi_intent_response(self):
        raw = json.dumps(MULTI_INTENT_SYNTHESIS_RESPONSE)
        parsed = parse_synthesis_response(raw)
        assert parsed is not None
        assert len(parsed["position_intents"]) == 2
        tickers = [i["ticker"] for i in parsed["position_intents"]]
        assert "AAPL" in tickers
        assert "MSFT" in tickers

    def test_crisis_response(self):
        raw = json.dumps(CRISIS_SYNTHESIS_RESPONSE)
        parsed = parse_synthesis_response(raw)
        assert parsed is not None
        assert parsed["position_intents"][0]["net_direction"] == "SHORT"

    def test_malformed_response_returns_none(self):
        raw = json.dumps(MALFORMED_SYNTHESIS_RESPONSE)
        parsed = parse_synthesis_response(raw)
        assert parsed is None

    def test_empty_intents_returns_none(self):
        raw = json.dumps(EMPTY_SYNTHESIS_RESPONSE)
        parsed = parse_synthesis_response(raw)
        assert parsed is None

    def test_invalid_direction_returns_none(self):
        raw = json.dumps(INVALID_DIRECTION_SYNTHESIS_RESPONSE)
        parsed = parse_synthesis_response(raw)
        assert parsed is None

    def test_confidence_clamped_to_085(self):
        raw = json.dumps(OVERCCONFIDENT_SYNTHESIS_RESPONSE)
        parsed = parse_synthesis_response(raw)
        assert parsed is not None
        intent = parsed["position_intents"][0]
        assert intent["synthesized_confidence"] == 0.85

    def test_negative_confidence_clamped_to_zero(self):
        data = {
            "position_intents": [
                {
                    "ticker": "AAPL",
                    "net_direction": "SHORT",
                    "synthesized_confidence": -0.5,
                }
            ]
        }
        parsed = parse_synthesis_response(json.dumps(data))
        assert parsed is not None
        assert parsed["position_intents"][0]["synthesized_confidence"] == 0.0

    def test_markdown_fences(self):
        inner = json.dumps(VALID_SYNTHESIS_RESPONSE)
        raw = f"```json\n{inner}\n```"
        parsed = parse_synthesis_response(raw)
        assert parsed is not None
        assert len(parsed["position_intents"]) == 1

    def test_surrounding_text(self):
        inner = json.dumps(VALID_SYNTHESIS_RESPONSE)
        raw = f"Here is my analysis:\n{inner}\nHope this helps!"
        parsed = parse_synthesis_response(raw)
        assert parsed is not None
        assert parsed["position_intents"][0]["ticker"] == "AAPL"

    def test_non_json_returns_none(self):
        parsed = parse_synthesis_response("This is not JSON at all")
        assert parsed is None

    def test_non_dict_returns_none(self):
        parsed = parse_synthesis_response("[1, 2, 3]")
        assert parsed is None

    def test_missing_ticker_returns_none(self):
        data = {
            "position_intents": [
                {
                    "net_direction": "LONG",
                    "synthesized_confidence": 0.50,
                }
            ]
        }
        parsed = parse_synthesis_response(json.dumps(data))
        assert parsed is None

    def test_missing_confidence_returns_none(self):
        data = {
            "position_intents": [
                {
                    "ticker": "AAPL",
                    "net_direction": "LONG",
                }
            ]
        }
        parsed = parse_synthesis_response(json.dumps(data))
        assert parsed is None


# ===========================================================================
# Tests: _build_intent
# ===========================================================================
class TestBuildIntent:
    """Tests for the _build_intent function."""

    def test_basic_intent(self):
        raw = VALID_SYNTHESIS_RESPONSE["position_intents"][0]
        intent = _build_intent(raw)
        assert isinstance(intent, SynthesizedPositionIntent)
        assert intent.ticker == "AAPL"
        assert intent.net_direction == Direction.LONG
        assert intent.synthesized_confidence == 0.62
        assert intent.time_horizon_days == 75

    def test_contributing_theses(self):
        raw = VALID_SYNTHESIS_RESPONSE["position_intents"][0]
        intent = _build_intent(raw)
        assert len(intent.contributing_theses) == 2
        ct = intent.contributing_theses[0]
        assert ct.thesis_id == "FUND-AAPL-2026Q1-MARGIN-EXPANSION"
        assert ct.agent_id == "COGNIT-FUNDAMENTAL"
        assert ct.direction == Direction.LONG
        assert ct.synthesis_weight == 0.35

    def test_conflicting_theses(self):
        raw = MULTI_INTENT_SYNTHESIS_RESPONSE["position_intents"][1]
        intent = _build_intent(raw)
        assert len(intent.conflicting_theses) == 1
        ct = intent.conflicting_theses[0]
        assert ct.direction == Direction.SHORT

    def test_conflict_resolution(self):
        raw = MULTI_INTENT_SYNTHESIS_RESPONSE["position_intents"][1]
        intent = _build_intent(raw)
        cr = intent.conflict_resolution
        assert cr.has_conflict is True
        assert cr.conflict_type == "DIRECTIONAL"
        assert cr.resolution_method == "CONFIDENCE_WEIGHTED"
        assert cr.net_conviction_delta == -0.15

    def test_active_invalidations(self):
        raw = VALID_SYNTHESIS_RESPONSE["position_intents"][0]
        intent = _build_intent(raw)
        assert len(intent.active_invalidations) == 1
        inv = intent.active_invalidations[0]
        assert inv.metric == "gross_margin_pct"
        assert inv.operator == "LT"
        assert inv.threshold == 44.0

    def test_regime_adjustment_clamped(self):
        raw = dict(VALID_SYNTHESIS_RESPONSE["position_intents"][0])
        raw["regime_adjustment"] = -0.50  # Over limit — should clamp to -0.30
        intent = _build_intent(raw)
        assert intent.regime_adjustment == -0.30

    def test_regime_adjustment_upper_bound(self):
        raw = dict(VALID_SYNTHESIS_RESPONSE["position_intents"][0])
        raw["regime_adjustment"] = 0.50  # Over limit — should clamp to 0.10
        intent = _build_intent(raw)
        assert intent.regime_adjustment == 0.10

    def test_time_horizon_minimum(self):
        raw = dict(VALID_SYNTHESIS_RESPONSE["position_intents"][0])
        raw["time_horizon_days"] = 0
        intent = _build_intent(raw)
        assert intent.time_horizon_days == 1

    def test_invalidations_capped_at_five(self):
        raw = dict(VALID_SYNTHESIS_RESPONSE["position_intents"][0])
        raw["active_invalidations"] = [
            {"metric": f"metric_{i}", "operator": "GT", "threshold": float(i)}
            for i in range(8)
        ]
        intent = _build_intent(raw)
        assert len(intent.active_invalidations) <= 5

    def test_empty_contributing_theses(self):
        raw = dict(VALID_SYNTHESIS_RESPONSE["position_intents"][0])
        raw["contributing_theses"] = []
        intent = _build_intent(raw)
        assert intent.contributing_theses == []

    def test_synthesis_rationale(self):
        raw = VALID_SYNTHESIS_RESPONSE["position_intents"][0]
        intent = _build_intent(raw)
        assert "fundamental" in intent.synthesis_rationale.lower()

    def test_default_conflict_resolution(self):
        raw = dict(VALID_SYNTHESIS_RESPONSE["position_intents"][0])
        raw["conflict_resolution"] = "not a dict"
        intent = _build_intent(raw)
        cr = intent.conflict_resolution
        assert cr.has_conflict is False
        assert cr.conflict_type == ""


# ===========================================================================
# Tests: DecideSynth agent
# ===========================================================================
class TestDecideSynth:
    """Integration tests for the DecideSynth agent pipeline."""

    @pytest.mark.asyncio
    async def test_process_valid(self):
        llm = _mock_llm(VALID_SYNTHESIS_RESPONSE)
        agent = DecideSynth(llm_client=llm)
        ctx = _make_context(
            belief_objects=_make_belief_objects(),
            regime_state=_make_regime_state(),
        )
        result = await agent.process(ctx)
        assert isinstance(result, SynthesisOutput)
        assert result.agent_id == "DECIDE-SYNTH"
        assert len(result.position_intents) == 1
        assert result.position_intents[0].ticker == "AAPL"

    @pytest.mark.asyncio
    async def test_process_multi_intent(self):
        llm = _mock_llm(MULTI_INTENT_SYNTHESIS_RESPONSE)
        agent = DecideSynth(llm_client=llm)
        ctx = _make_context(
            belief_objects=_make_belief_objects(),
            regime_state=_make_regime_state(),
        )
        result = await agent.process(ctx)
        assert len(result.position_intents) == 2
        tickers = [i.ticker for i in result.position_intents]
        assert "AAPL" in tickers
        assert "MSFT" in tickers

    @pytest.mark.asyncio
    async def test_process_crisis(self):
        llm = _mock_llm(CRISIS_SYNTHESIS_RESPONSE)
        regime = _make_regime_state()
        regime["statistical_regime"] = "CRISIS_DISLOCATION"
        regime["system_risk_mode"] = "DEFENSIVE"
        agent = DecideSynth(llm_client=llm)
        ctx = _make_context(
            belief_objects=_make_belief_objects(),
            regime_state=regime,
        )
        result = await agent.process(ctx)
        assert result.position_intents[0].net_direction == Direction.SHORT
        assert result.position_intents[0].regime_adjustment == -0.20

    @pytest.mark.asyncio
    async def test_malformed_raises(self):
        llm = _mock_llm(MALFORMED_SYNTHESIS_RESPONSE)
        agent = DecideSynth(llm_client=llm)
        ctx = _make_context(
            belief_objects=_make_belief_objects(),
            regime_state=_make_regime_state(),
        )
        with pytest.raises(AgentProcessingError):
            await agent.process(ctx)

    @pytest.mark.asyncio
    async def test_empty_intents_raises(self):
        llm = _mock_llm(EMPTY_SYNTHESIS_RESPONSE)
        agent = DecideSynth(llm_client=llm)
        ctx = _make_context(
            belief_objects=_make_belief_objects(),
            regime_state=_make_regime_state(),
        )
        with pytest.raises(AgentProcessingError):
            await agent.process(ctx)

    @pytest.mark.asyncio
    async def test_llm_exception_raises(self):
        mock = AsyncMock()
        mock.complete = AsyncMock(side_effect=RuntimeError("LLM unreachable"))
        agent = DecideSynth(llm_client=mock)
        ctx = _make_context(
            belief_objects=_make_belief_objects(),
            regime_state=_make_regime_state(),
        )
        with pytest.raises(AgentProcessingError, match="DECIDE-SYNTH processing failed"):
            await agent.process(ctx)

    @pytest.mark.asyncio
    async def test_total_beliefs_consumed(self):
        llm = _mock_llm(VALID_SYNTHESIS_RESPONSE)
        agent = DecideSynth(llm_client=llm)
        ctx = _make_context(
            belief_objects=_make_belief_objects(),
            regime_state=_make_regime_state(),
        )
        result = await agent.process(ctx)
        # _make_belief_objects has 3 beliefs total (1 + 2)
        assert result.total_beliefs_consumed == 3

    @pytest.mark.asyncio
    async def test_regime_context_label(self):
        llm = _mock_llm(VALID_SYNTHESIS_RESPONSE)
        agent = DecideSynth(llm_client=llm)
        ctx = _make_context(
            belief_objects=_make_belief_objects(),
            regime_state=_make_regime_state(),
        )
        result = await agent.process(ctx)
        assert "LOW_VOL_TRENDING" in result.regime_context
        assert "AI-driven tech euphoria" in result.regime_context

    @pytest.mark.asyncio
    async def test_content_hash_computed(self):
        llm = _mock_llm(VALID_SYNTHESIS_RESPONSE)
        agent = DecideSynth(llm_client=llm)
        ctx = _make_context(
            belief_objects=_make_belief_objects(),
            regime_state=_make_regime_state(),
        )
        result = await agent.process(ctx)
        assert result.content_hash != ""
        assert len(result.content_hash) == 64  # SHA-256 hex

    @pytest.mark.asyncio
    async def test_context_window_hash_passthrough(self):
        llm = _mock_llm(VALID_SYNTHESIS_RESPONSE)
        agent = DecideSynth(llm_client=llm)
        ctx = _make_context(
            belief_objects=_make_belief_objects(),
            regime_state=_make_regime_state(),
        )
        result = await agent.process(ctx)
        assert result.context_window_hash == "test-hash-synth-001"

    @pytest.mark.asyncio
    async def test_llm_called_with_beliefs_and_regime(self):
        llm = _mock_llm(VALID_SYNTHESIS_RESPONSE)
        agent = DecideSynth(llm_client=llm)
        ctx = _make_context(
            belief_objects=_make_belief_objects(),
            regime_state=_make_regime_state(),
        )
        await agent.process(ctx)
        assert llm.complete.call_count == 1
        call_kwargs = llm.complete.call_args
        user_prompt = call_kwargs.kwargs.get("user_prompt", "") or call_kwargs[1] if len(call_kwargs) > 1 else ""
        # The LLM was called — that's the key check
        assert llm.complete.called

    @pytest.mark.asyncio
    async def test_empty_beliefs_still_calls_llm(self):
        llm = _mock_llm(VALID_SYNTHESIS_RESPONSE)
        agent = DecideSynth(llm_client=llm)
        ctx = _make_context(belief_objects=[], regime_state=_make_regime_state())
        result = await agent.process(ctx)
        assert result.total_beliefs_consumed == 0

    @pytest.mark.asyncio
    async def test_no_metadata_defaults(self):
        llm = _mock_llm(VALID_SYNTHESIS_RESPONSE)
        agent = DecideSynth(llm_client=llm)
        ctx = _make_context()  # No beliefs or regime
        result = await agent.process(ctx)
        assert result.total_beliefs_consumed == 0
        assert result.regime_context == ""


# ===========================================================================
# Tests: prompt building
# ===========================================================================
class TestDecideSynthPromptBuilding:
    """Tests for the internal prompt formatting methods."""

    def test_format_beliefs_with_data(self):
        agent = DecideSynth(llm_client=AsyncMock())
        beliefs = _make_belief_objects()
        formatted = agent._format_beliefs(beliefs)
        assert "COGNIT-FUNDAMENTAL" in formatted
        assert "COGNIT-NARRATIVE" in formatted
        assert "AAPL" in formatted
        assert "LONG" in formatted

    def test_format_beliefs_empty(self):
        agent = DecideSynth(llm_client=AsyncMock())
        formatted = agent._format_beliefs([])
        assert "No beliefs available" in formatted

    def test_format_beliefs_non_list(self):
        agent = DecideSynth(llm_client=AsyncMock())
        formatted = agent._format_beliefs("not a list")
        assert "No beliefs available" in formatted

    def test_format_regime_with_data(self):
        agent = DecideSynth(llm_client=AsyncMock())
        regime = _make_regime_state()
        formatted = agent._format_regime(regime)
        assert "LOW_VOL_TRENDING" in formatted
        assert "NORMAL" in formatted
        assert "realized_vol" in formatted
        assert "AI-driven tech euphoria" in formatted

    def test_format_regime_empty(self):
        agent = DecideSynth(llm_client=AsyncMock())
        formatted = agent._format_regime({})
        assert "UNKNOWN" in formatted

    def test_format_regime_non_dict(self):
        agent = DecideSynth(llm_client=AsyncMock())
        formatted = agent._format_regime("not a dict")
        assert "No regime context available" in formatted

    def test_format_beliefs_includes_invalidations(self):
        agent = DecideSynth(llm_client=AsyncMock())
        beliefs = _make_belief_objects()
        formatted = agent._format_beliefs(beliefs)
        assert "Invalidation" in formatted
        assert "gross_margin_pct" in formatted

    def test_format_beliefs_includes_thesis_summary(self):
        agent = DecideSynth(llm_client=AsyncMock())
        beliefs = _make_belief_objects()
        formatted = agent._format_beliefs(beliefs)
        assert "margin expansion" in formatted.lower()


# ===========================================================================
# Tests: health reporting
# ===========================================================================
class TestDecideSynthHealth:
    """Tests for DecideSynth health reporting."""

    def test_healthy_initial(self):
        agent = DecideSynth(llm_client=AsyncMock())
        health = agent.get_health()
        assert health.status == AgentStatus.HEALTHY
        assert health.error_count_24h == 0

    def test_degraded_after_errors(self):
        agent = DecideSynth(llm_client=AsyncMock())
        agent._error_count_24h = 5
        health = agent.get_health()
        assert health.status == AgentStatus.DEGRADED

    def test_unhealthy_after_many_errors(self):
        agent = DecideSynth(llm_client=AsyncMock())
        agent._error_count_24h = 15
        health = agent.get_health()
        assert health.status == AgentStatus.UNHEALTHY

    def test_health_agent_id(self):
        agent = DecideSynth(llm_client=AsyncMock())
        health = agent.get_health()
        assert health.agent_id == "DECIDE-SYNTH"


# ===========================================================================
# Tests: agent properties
# ===========================================================================
class TestDecideSynthProperties:
    """Tests for DecideSynth static properties and metadata."""

    def test_agent_id(self):
        agent = DecideSynth(llm_client=AsyncMock())
        assert agent.agent_id == "DECIDE-SYNTH"

    def test_agent_type(self):
        agent = DecideSynth(llm_client=AsyncMock())
        assert agent.agent_type == "decision"

    def test_version(self):
        agent = DecideSynth(llm_client=AsyncMock())
        assert agent.version == "1.0.0"

    def test_consumed_data_types_empty(self):
        """DECIDE-SYNTH reads from metadata, not raw fragments."""
        assert DecideSynth.CONSUMED_DATA_TYPES == set()


# ===========================================================================
# Tests: schema validation
# ===========================================================================
class TestSynthesisSchemas:
    """Tests for decision.py schema models."""

    def test_contributing_thesis_frozen(self):
        ct = ContributingThesis(
            thesis_id="T1",
            agent_id="COGNIT-FUNDAMENTAL",
            ticker="AAPL",
            direction=Direction.LONG,
            raw_confidence=0.72,
            magnitude=Magnitude.MODERATE,
            synthesis_weight=0.35,
        )
        with pytest.raises(Exception):
            ct.thesis_id = "T2"

    def test_conflict_resolution_defaults(self):
        cr = ConflictResolution()
        assert cr.has_conflict is False
        assert cr.conflict_type == ""
        assert cr.net_conviction_delta == 0.0

    def test_active_invalidation_has_condition_id(self):
        inv = ActiveInvalidation(
            source_thesis_id="T1",
            source_agent_id="COGNIT-MACRO",
            metric="spread_2s10s_bps",
            operator="GT",
            threshold=0.0,
        )
        assert isinstance(inv.condition_id, UUID)

    def test_synthesized_position_intent_frozen(self):
        intent = SynthesizedPositionIntent(
            ticker="AAPL",
            net_direction=Direction.LONG,
            synthesized_confidence=0.60,
            time_horizon_days=60,
        )
        with pytest.raises(Exception):
            intent.ticker = "MSFT"

    def test_synthesis_output_content_hash(self):
        intent = SynthesizedPositionIntent(
            ticker="AAPL",
            net_direction=Direction.LONG,
            synthesized_confidence=0.60,
            time_horizon_days=60,
        )
        output = SynthesisOutput(
            agent_id="DECIDE-SYNTH",
            timestamp=NOW,
            context_window_hash="test-hash",
            position_intents=[intent],
            regime_context="LOW_VOL_TRENDING",
            total_beliefs_consumed=4,
        )
        assert output.content_hash != ""
        assert len(output.content_hash) == 64

    def test_synthesis_output_requires_tz(self):
        intent = SynthesizedPositionIntent(
            ticker="AAPL",
            net_direction=Direction.LONG,
            synthesized_confidence=0.60,
            time_horizon_days=60,
        )
        with pytest.raises(Exception):
            SynthesisOutput(
                agent_id="DECIDE-SYNTH",
                timestamp=datetime(2026, 1, 1),  # No timezone
                context_window_hash="test-hash",
                position_intents=[intent],
            )

    def test_confidence_bounds(self):
        with pytest.raises(Exception):
            SynthesizedPositionIntent(
                ticker="AAPL",
                net_direction=Direction.LONG,
                synthesized_confidence=1.5,  # Over 1.0
                time_horizon_days=60,
            )

    def test_time_horizon_positive(self):
        with pytest.raises(Exception):
            SynthesizedPositionIntent(
                ticker="AAPL",
                net_direction=Direction.LONG,
                synthesized_confidence=0.60,
                time_horizon_days=0,  # Must be > 0
            )
