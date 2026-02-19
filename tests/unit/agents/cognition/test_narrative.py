"""Tests for COGNIT-NARRATIVE Research Agent.

Tests the full Research Agent loop (7 steps) with mocked LLM client.
Validates narrative-specific prompt building with peer comparison,
belief parsing, evidence linking, invalidation condition
machine-evaluability, and error handling.

COGNIT-NARRATIVE is ADAPTIVE: uses Claude Sonnet 4 for narrative analysis.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest

from providence.agents.base import AgentContext, AgentStatus
from providence.agents.cognition.narrative import CognitNarrative
from providence.agents.cognition.response_parser import parse_llm_response
from providence.exceptions import AgentProcessingError
from providence.schemas.belief import BeliefObject
from providence.schemas.enums import (
    ComparisonOperator,
    DataType,
    Direction,
    ValidationStatus,
)
from providence.schemas.market_state import MarketStateFragment

from tests.fixtures.llm_responses import (
    MULTI_NARRATIVE_BELIEF_RESPONSE,
    NARRATIVE_EMPTY_BELIEFS_RESPONSE,
    NARRATIVE_VAGUE_INVALIDATION_RESPONSE,
    VALID_NARRATIVE_RESPONSE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
NOW = datetime.now(timezone.utc)

EARNINGS_FRAG_ID = UUID("cccccc01-cccc-cccc-cccc-cccccccccccc")
NEWS_FRAG_ID = UUID("cccccc02-cccc-cccc-cccc-cccccccccccc")
FILING_FRAG_ID = UUID("cccccc03-cccc-cccc-cccc-cccccccccccc")
PEER_FRAG_ID = UUID("cccccc04-cccc-cccc-cccc-cccccccccccc")


def _make_earnings_call_fragment(
    fragment_id: UUID | None = None,
    ticker: str = "AAPL",
    tone_score: float = 0.74,
    hours_ago: float = 1,
) -> MarketStateFragment:
    """Create an EARNINGS_CALL test fragment."""
    ts = NOW - timedelta(hours=hours_ago)
    return MarketStateFragment(
        fragment_id=fragment_id or EARNINGS_FRAG_ID,
        agent_id="PERCEPT-EARNINGS",
        timestamp=ts,
        source_timestamp=ts,
        entity=ticker,
        data_type=DataType.EARNINGS_CALL,
        schema_version="1.0.0",
        source_hash=f"hash-earnings-{ticker}",
        validation_status=ValidationStatus.VALID,
        payload={
            "quarter": "2026-Q1",
            "tone_score": tone_score,
            "hedge_word_count": 12,
            "forward_guidance_mentions": 8,
            "transcript_length_words": 8500,
            "key_phrases": ["services growth", "margin expansion", "capital return"],
            "ceo_sentiment": "bullish",
        },
    )


def _make_news_fragment(
    fragment_id: UUID | None = None,
    ticker: str = "AAPL",
    headline: str = "Apple beats earnings estimates, raises services guidance",
    sentiment_score: float = 0.65,
    hours_ago: float = 1,
) -> MarketStateFragment:
    """Create a SENTIMENT_NEWS test fragment."""
    ts = NOW - timedelta(hours=hours_ago)
    return MarketStateFragment(
        fragment_id=fragment_id or NEWS_FRAG_ID,
        agent_id="PERCEPT-NEWS",
        timestamp=ts,
        source_timestamp=ts,
        entity=ticker,
        data_type=DataType.SENTIMENT_NEWS,
        schema_version="1.0.0",
        source_hash=f"hash-news-{ticker}",
        validation_status=ValidationStatus.VALID,
        payload={
            "headline": headline,
            "sentiment_score": sentiment_score,
            "source": "Bloomberg",
            "published_at": "2026-02-12T08:00:00Z",
            "article_body": f"Analyst commentary on {ticker} earnings...",
        },
    )


def _make_filing_8k_fragment(
    fragment_id: UUID | None = None,
    ticker: str = "AAPL",
    event_type: str = "Buyback Authorization",
    hours_ago: float = 1,
) -> MarketStateFragment:
    """Create a FILING_8K test fragment."""
    ts = NOW - timedelta(hours=hours_ago)
    return MarketStateFragment(
        fragment_id=fragment_id or FILING_FRAG_ID,
        agent_id="PERCEPT-FILING",
        timestamp=ts,
        source_timestamp=ts,
        entity=ticker,
        data_type=DataType.FILING_8K,
        schema_version="1.0.0",
        source_hash=f"hash-8k-{ticker}",
        validation_status=ValidationStatus.VALID,
        payload={
            "event_type": event_type,
            "description": f"{event_type} announced for {ticker}",
            "filing_date": "2026-02-12",
            "sec_item_number": "5.03",
        },
    )


def _make_peer_earnings_fragment(
    fragment_id: UUID | None = None,
    ticker: str = "GOOG",
    tone_score: float = 0.38,
    hours_ago: float = 2,
) -> MarketStateFragment:
    """Create a peer's EARNINGS_CALL fragment (different entity)."""
    ts = NOW - timedelta(hours=hours_ago)
    return MarketStateFragment(
        fragment_id=fragment_id or PEER_FRAG_ID,
        agent_id="PERCEPT-EARNINGS",
        timestamp=ts,
        source_timestamp=ts,
        entity=ticker,
        data_type=DataType.EARNINGS_CALL,
        schema_version="1.0.0",
        source_hash=f"hash-earnings-{ticker}",
        validation_status=ValidationStatus.VALID,
        payload={
            "quarter": "2026-Q1",
            "tone_score": tone_score,
            "hedge_word_count": 28,
            "forward_guidance_mentions": 3,
            "transcript_length_words": 7200,
            "key_phrases": ["cost discipline", "uncertain macro", "cautious outlook"],
            "ceo_sentiment": "cautious",
        },
    )


def _make_full_narrative_fragments() -> list[MarketStateFragment]:
    """Create a full set: primary entity (3 types) + 1 peer fragment."""
    return [
        _make_earnings_call_fragment(),
        _make_news_fragment(),
        _make_filing_8k_fragment(),
        _make_peer_earnings_fragment(),
    ]


def _make_context(
    fragments: list[MarketStateFragment] | None = None,
) -> AgentContext:
    """Create a test AgentContext for COGNIT-NARRATIVE."""
    return AgentContext(
        agent_id="COGNIT-NARRATIVE",
        trigger="schedule",
        fragments=fragments if fragments is not None else _make_full_narrative_fragments(),
        context_window_hash="narrative_test_hash",
        timestamp=NOW,
        metadata={},
    )


def _make_mock_llm(response: dict) -> AsyncMock:
    """Create a mock LLM client that returns a fixed response."""
    mock = AsyncMock()
    mock.complete = AsyncMock(return_value=response)
    return mock


# ---------------------------------------------------------------------------
# Response parser tests (with narrative-specific responses)
# ---------------------------------------------------------------------------
class TestNarrativeResponseParsing:
    """Tests for parsing narrative LLM responses via the shared response_parser."""

    def test_valid_narrative_response_produces_belief_object(self):
        """A well-formed narrative LLM response parses into a valid BeliefObject."""
        result = parse_llm_response(
            raw=VALID_NARRATIVE_RESPONSE,
            agent_id="COGNIT-NARRATIVE",
            context_window_hash="test_hash",
        )
        assert result is not None
        assert isinstance(result, BeliefObject)
        assert result.agent_id == "COGNIT-NARRATIVE"
        assert len(result.beliefs) == 1

    def test_narrative_belief_fields(self):
        """Parsed narrative belief has correct direction, magnitude, confidence."""
        result = parse_llm_response(
            raw=VALID_NARRATIVE_RESPONSE,
            agent_id="COGNIT-NARRATIVE",
            context_window_hash="test_hash",
        )
        belief = result.beliefs[0]
        assert belief.direction == Direction.LONG
        assert belief.ticker == "AAPL"
        assert belief.raw_confidence == 0.62
        assert belief.time_horizon_days == 60

    def test_narrative_evidence_refs_parsed(self):
        """Evidence refs point to narrative data fragments."""
        result = parse_llm_response(
            raw=VALID_NARRATIVE_RESPONSE,
            agent_id="COGNIT-NARRATIVE",
            context_window_hash="test_hash",
        )
        evidence = result.beliefs[0].evidence
        assert len(evidence) == 3
        assert evidence[0].source_fragment_id == EARNINGS_FRAG_ID
        assert evidence[0].field_path == "payload.tone_score"
        assert evidence[1].source_fragment_id == NEWS_FRAG_ID
        assert evidence[2].source_fragment_id == FILING_FRAG_ID

    def test_narrative_invalidation_conditions_machine_evaluable(self):
        """All parsed conditions have metric, operator, and threshold."""
        result = parse_llm_response(
            raw=VALID_NARRATIVE_RESPONSE,
            agent_id="COGNIT-NARRATIVE",
            context_window_hash="test_hash",
        )
        conditions = result.beliefs[0].invalidation_conditions
        assert len(conditions) >= 2
        for cond in conditions:
            assert cond.metric
            assert cond.operator in ComparisonOperator
            assert cond.threshold is not None

    def test_multi_narrative_beliefs(self):
        """Multiple narrative beliefs parse correctly."""
        result = parse_llm_response(
            raw=MULTI_NARRATIVE_BELIEF_RESPONSE,
            agent_id="COGNIT-NARRATIVE",
            context_window_hash="test_hash",
        )
        assert result is not None
        assert len(result.beliefs) == 2
        directions = {b.direction for b in result.beliefs}
        assert Direction.SHORT in directions
        assert Direction.LONG in directions

    def test_narrative_empty_beliefs_returns_none(self):
        """Empty beliefs array returns None."""
        result = parse_llm_response(
            raw=NARRATIVE_EMPTY_BELIEFS_RESPONSE,
            agent_id="COGNIT-NARRATIVE",
            context_window_hash="test_hash",
        )
        assert result is None

    def test_narrative_vague_invalidation_drops_belief(self):
        """Beliefs with < 2 valid invalidation conditions are dropped."""
        result = parse_llm_response(
            raw=NARRATIVE_VAGUE_INVALIDATION_RESPONSE,
            agent_id="COGNIT-NARRATIVE",
            context_window_hash="test_hash",
        )
        # Only 1 valid condition → belief dropped → None
        assert result is None


# ---------------------------------------------------------------------------
# Prompt building tests (with peer comparison)
# ---------------------------------------------------------------------------
class TestNarrativePromptBuilding:
    """Tests for _build_user_prompt() with peer narrative comparison."""

    def test_earnings_fragments_in_earnings_section(self):
        """Primary entity's earnings call fragments appear in earnings section."""
        agent = CognitNarrative(llm_client=_make_mock_llm(VALID_NARRATIVE_RESPONSE))
        context = _make_context(fragments=[_make_earnings_call_fragment()])

        prompt = agent._build_user_prompt(context)

        assert "EARNINGS_CALL" in prompt
        assert "tone_score" in prompt
        assert "0.74" in prompt

    def test_news_fragments_in_news_section(self):
        """Primary entity's news fragments appear in news section."""
        agent = CognitNarrative(llm_client=_make_mock_llm(VALID_NARRATIVE_RESPONSE))
        context = _make_context(fragments=[_make_news_fragment()])

        prompt = agent._build_user_prompt(context)

        assert "SENTIMENT_NEWS" in prompt
        assert "0.65" in prompt
        assert "Bloomberg" in prompt

    def test_filing_fragments_in_filing_section(self):
        """Primary entity's 8K filing fragments appear in filing section."""
        agent = CognitNarrative(llm_client=_make_mock_llm(VALID_NARRATIVE_RESPONSE))
        context = _make_context(fragments=[_make_filing_8k_fragment()])

        prompt = agent._build_user_prompt(context)

        assert "FILING_8K" in prompt
        assert "Buyback Authorization" in prompt

    def test_peer_fragments_in_peer_section(self):
        """Non-primary entity fragments go to the peer section."""
        agent = CognitNarrative(llm_client=_make_mock_llm(VALID_NARRATIVE_RESPONSE))
        # 2 AAPL fragments (primary) + 1 GOOG fragment (peer)
        frags = [
            _make_earnings_call_fragment(ticker="AAPL"),
            _make_news_fragment(fragment_id=uuid4(), ticker="AAPL"),
            _make_peer_earnings_fragment(ticker="GOOG"),
        ]
        context = _make_context(fragments=frags)

        prompt = agent._build_user_prompt(context)

        # GOOG should be in the peer section, not in earnings section
        # The prompt should contain both entities
        assert "AAPL" in prompt
        assert "GOOG" in prompt

    def test_primary_ticker_identified_from_entities(self):
        """Primary ticker is the most common entity in fragments."""
        agent = CognitNarrative(llm_client=_make_mock_llm(VALID_NARRATIVE_RESPONSE))
        # 3 AAPL fragments, 1 GOOG → primary = AAPL
        frags = [
            _make_earnings_call_fragment(fragment_id=uuid4(), ticker="AAPL"),
            _make_news_fragment(fragment_id=uuid4(), ticker="AAPL"),
            _make_filing_8k_fragment(fragment_id=uuid4(), ticker="AAPL"),
            _make_peer_earnings_fragment(fragment_id=uuid4(), ticker="GOOG"),
        ]
        context = _make_context(fragments=frags)

        prompt = agent._build_user_prompt(context)

        # AAPL should be the primary ticker in the template
        assert "AAPL" in prompt

    def test_missing_earnings_shows_default(self):
        """When no earnings fragments for primary entity, default message appears."""
        agent = CognitNarrative(llm_client=_make_mock_llm(VALID_NARRATIVE_RESPONSE))
        # Only news — no earnings calls
        frags = [_make_news_fragment()]
        context = _make_context(fragments=frags)

        prompt = agent._build_user_prompt(context)

        assert "No earnings call data available" in prompt

    def test_missing_peer_shows_default(self):
        """When no peer fragments exist, default message appears."""
        agent = CognitNarrative(llm_client=_make_mock_llm(VALID_NARRATIVE_RESPONSE))
        # All fragments for same entity — no peers
        frags = [
            _make_earnings_call_fragment(ticker="AAPL"),
            _make_news_fragment(fragment_id=uuid4(), ticker="AAPL"),
        ]
        context = _make_context(fragments=frags)

        prompt = agent._build_user_prompt(context)

        assert "No peer narrative data available" in prompt

    def test_all_fragment_ids_listed(self):
        """All fragment IDs appear in the fragment_ids section."""
        agent = CognitNarrative(llm_client=_make_mock_llm(VALID_NARRATIVE_RESPONSE))
        frags = _make_full_narrative_fragments()
        context = _make_context(fragments=frags)

        prompt = agent._build_user_prompt(context)

        for frag in frags:
            assert str(frag.fragment_id) in prompt

    def test_non_primary_entity_routed_to_peers(self):
        """Fragments from non-primary entities go to peer section regardless of type."""
        agent = CognitNarrative(llm_client=_make_mock_llm(VALID_NARRATIVE_RESPONSE))
        # 2 AAPL (primary) + 1 MSFT news + 1 GOOG earnings (both peers)
        frags = [
            _make_earnings_call_fragment(fragment_id=uuid4(), ticker="AAPL"),
            _make_news_fragment(fragment_id=uuid4(), ticker="AAPL"),
            _make_news_fragment(fragment_id=uuid4(), ticker="MSFT", headline="MSFT earnings solid"),
            _make_peer_earnings_fragment(fragment_id=uuid4(), ticker="GOOG"),
        ]
        context = _make_context(fragments=frags)

        prompt = agent._build_user_prompt(context)

        # Both peer entities should appear, and primary AAPL should have earnings
        assert "AAPL" in prompt
        assert "MSFT" in prompt
        assert "GOOG" in prompt


# ---------------------------------------------------------------------------
# CognitNarrative agent tests (mocked LLM)
# ---------------------------------------------------------------------------
class TestCognitNarrative:
    """Tests for the CognitNarrative agent with mocked LLM."""

    @pytest.mark.asyncio
    async def test_process_returns_belief_object(self):
        """Agent process() returns a valid BeliefObject."""
        mock_llm = _make_mock_llm(VALID_NARRATIVE_RESPONSE)
        agent = CognitNarrative(llm_client=mock_llm)
        context = _make_context()

        result = await agent.process(context)

        assert isinstance(result, BeliefObject)
        assert result.agent_id == "COGNIT-NARRATIVE"
        assert len(result.beliefs) == 1
        mock_llm.complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_multi_belief(self):
        """Agent handles multi-belief LLM responses."""
        mock_llm = _make_mock_llm(MULTI_NARRATIVE_BELIEF_RESPONSE)
        agent = CognitNarrative(llm_client=mock_llm)
        context = _make_context()

        result = await agent.process(context)

        assert len(result.beliefs) == 2

    @pytest.mark.asyncio
    async def test_process_malformed_raises(self):
        """Agent raises AgentProcessingError on malformed LLM response."""
        mock_llm = _make_mock_llm({"analysis": "Tone seems positive"})
        agent = CognitNarrative(llm_client=mock_llm)
        context = _make_context()

        with pytest.raises(AgentProcessingError):
            await agent.process(context)

    @pytest.mark.asyncio
    async def test_process_empty_beliefs_raises(self):
        """Agent raises AgentProcessingError when no beliefs parsed."""
        mock_llm = _make_mock_llm(NARRATIVE_EMPTY_BELIEFS_RESPONSE)
        agent = CognitNarrative(llm_client=mock_llm)
        context = _make_context()

        with pytest.raises(AgentProcessingError):
            await agent.process(context)

    @pytest.mark.asyncio
    async def test_process_vague_conditions_raises(self):
        """Agent raises when all beliefs are dropped due to vague conditions."""
        mock_llm = _make_mock_llm(NARRATIVE_VAGUE_INVALIDATION_RESPONSE)
        agent = CognitNarrative(llm_client=mock_llm)
        context = _make_context()

        with pytest.raises(AgentProcessingError):
            await agent.process(context)

    @pytest.mark.asyncio
    async def test_process_llm_error_raises(self):
        """Agent raises AgentProcessingError on LLM client error."""
        mock_llm = AsyncMock()
        mock_llm.complete = AsyncMock(side_effect=Exception("API down"))
        agent = CognitNarrative(llm_client=mock_llm)
        context = _make_context()

        with pytest.raises(AgentProcessingError):
            await agent.process(context)

    @pytest.mark.asyncio
    async def test_context_window_hash_preserved(self):
        """BeliefObject carries the context_window_hash from input."""
        mock_llm = _make_mock_llm(VALID_NARRATIVE_RESPONSE)
        agent = CognitNarrative(llm_client=mock_llm)
        context = _make_context()

        result = await agent.process(context)

        assert result.context_window_hash == "narrative_test_hash"

    @pytest.mark.asyncio
    async def test_content_hash_computed(self):
        """BeliefObject has a non-empty content hash."""
        mock_llm = _make_mock_llm(VALID_NARRATIVE_RESPONSE)
        agent = CognitNarrative(llm_client=mock_llm)
        context = _make_context()

        result = await agent.process(context)

        assert result.content_hash
        assert len(result.content_hash) == 64  # SHA-256 hex

    @pytest.mark.asyncio
    async def test_llm_receives_system_and_user_prompt(self):
        """LLM client receives both system and user prompt."""
        mock_llm = _make_mock_llm(VALID_NARRATIVE_RESPONSE)
        agent = CognitNarrative(llm_client=mock_llm)
        context = _make_context()

        await agent.process(context)

        call_args = mock_llm.complete.call_args
        assert "system_prompt" in call_args.kwargs or len(call_args.args) >= 1
        assert "user_prompt" in call_args.kwargs or len(call_args.args) >= 2


# ---------------------------------------------------------------------------
# Health status tests
# ---------------------------------------------------------------------------
class TestCognitNarrativeHealth:
    """Tests for agent health reporting."""

    def test_initial_health_is_healthy(self):
        """Agent starts in HEALTHY state."""
        agent = CognitNarrative(llm_client=AsyncMock())
        health = agent.get_health()
        assert health.agent_id == "COGNIT-NARRATIVE"
        assert health.status == AgentStatus.HEALTHY
        assert health.error_count_24h == 0

    @pytest.mark.asyncio
    async def test_health_after_success(self):
        """Health reports last_success after successful run."""
        mock_llm = _make_mock_llm(VALID_NARRATIVE_RESPONSE)
        agent = CognitNarrative(llm_client=mock_llm)
        context = _make_context()

        await agent.process(context)

        health = agent.get_health()
        assert health.last_run is not None
        assert health.last_success is not None
        assert health.status == AgentStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_health_degraded_after_errors(self):
        """Health degrades after repeated errors."""
        mock_llm = _make_mock_llm(NARRATIVE_EMPTY_BELIEFS_RESPONSE)
        agent = CognitNarrative(llm_client=mock_llm)
        context = _make_context()

        for _ in range(4):
            try:
                await agent.process(context)
            except AgentProcessingError:
                pass

        health = agent.get_health()
        assert health.status == AgentStatus.DEGRADED
        assert health.error_count_24h == 4

    @pytest.mark.asyncio
    async def test_health_unhealthy_after_many_errors(self):
        """Health becomes UNHEALTHY after 11+ errors."""
        mock_llm = _make_mock_llm(NARRATIVE_EMPTY_BELIEFS_RESPONSE)
        agent = CognitNarrative(llm_client=mock_llm)
        context = _make_context()

        for _ in range(11):
            try:
                await agent.process(context)
            except AgentProcessingError:
                pass

        health = agent.get_health()
        assert health.status == AgentStatus.UNHEALTHY


# ---------------------------------------------------------------------------
# Agent properties
# ---------------------------------------------------------------------------
class TestCognitNarrativeProperties:
    """Test agent identity properties."""

    def test_agent_id(self):
        agent = CognitNarrative(llm_client=AsyncMock())
        assert agent.agent_id == "COGNIT-NARRATIVE"

    def test_agent_type(self):
        agent = CognitNarrative(llm_client=AsyncMock())
        assert agent.agent_type == "cognition"

    def test_version(self):
        agent = CognitNarrative(llm_client=AsyncMock())
        assert agent.version == "1.0.0"

    def test_consumed_data_types(self):
        """Agent consumes the correct data types."""
        expected = {
            DataType.EARNINGS_CALL,
            DataType.SENTIMENT_NEWS,
            DataType.FILING_8K,
        }
        assert CognitNarrative.CONSUMED_DATA_TYPES == expected
