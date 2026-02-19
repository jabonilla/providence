"""Tests for COGNIT-EVENT Research Agent.

Tests the full Research Agent loop (7 steps) with mocked LLM client.
Validates event-specific prompt building, belief parsing, evidence linking,
invalidation condition machine-evaluability, and error handling.

COGNIT-EVENT is ADAPTIVE: uses Claude Sonnet 4 for event-driven analysis.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest

from providence.agents.base import AgentContext, AgentStatus
from providence.agents.cognition.event import CognitEvent
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
    EVENT_EMPTY_BELIEFS_RESPONSE,
    EVENT_VAGUE_INVALIDATION_RESPONSE,
    MULTI_EVENT_BELIEF_RESPONSE,
    VALID_EVENT_RESPONSE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
NOW = datetime.now(timezone.utc)

FILING_FRAG_ID = UUID("bbbbbb01-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
NEWS_FRAG_ID = UUID("bbbbbb02-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
PRICE_FRAG_ID = UUID("bbbbbb03-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
OPTIONS_FRAG_ID = UUID("bbbbbb04-bbbb-bbbb-bbbb-bbbbbbbbbbbb")


def _make_filing_8k_fragment(
    fragment_id: UUID | None = None,
    ticker: str = "AAPL",
    event_type: str = "M&A",
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
            "description": f"{event_type} event for {ticker}",
            "filing_date": "2026-02-12",
            "sec_item_number": "1.01",
            "deal_value": 185.0,
            "acquirer": "BuyerCo",
            "target": ticker,
        },
    )


def _make_news_fragment(
    fragment_id: UUID | None = None,
    ticker: str = "AAPL",
    headline: str = "Apple announces major acquisition at 18% premium",
    sentiment_score: float = 0.8,
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
            "source": "Reuters",
            "published_at": "2026-02-12T10:30:00Z",
            "article_body": f"Details about {ticker} news event...",
        },
    )


def _make_price_fragment(
    fragment_id: UUID | None = None,
    entity: str = "AAPL",
    close: float = 230.0,
    hours_ago: float = 1,
) -> MarketStateFragment:
    """Create a PRICE_OHLCV test fragment."""
    ts = NOW - timedelta(hours=hours_ago)
    return MarketStateFragment(
        fragment_id=fragment_id or PRICE_FRAG_ID,
        agent_id="PERCEPT-PRICE",
        timestamp=ts,
        source_timestamp=ts,
        entity=entity,
        data_type=DataType.PRICE_OHLCV,
        schema_version="1.0.0",
        source_hash=f"hash-price-{entity}",
        validation_status=ValidationStatus.VALID,
        payload={
            "open": close - 3.0,
            "high": close + 2.0,
            "low": close - 5.0,
            "close": close,
            "volume": 89_345_200,
            "vwap": close - 1.0,
            "timeframe": "1D",
        },
    )


def _make_options_fragment(
    fragment_id: UUID | None = None,
    ticker: str = "AAPL",
    implied_vol_atm: float = 0.32,
    put_call_ratio: float = 0.85,
    hours_ago: float = 1,
) -> MarketStateFragment:
    """Create an OPTIONS_CHAIN test fragment."""
    ts = NOW - timedelta(hours=hours_ago)
    return MarketStateFragment(
        fragment_id=fragment_id or OPTIONS_FRAG_ID,
        agent_id="PERCEPT-OPTIONS",
        timestamp=ts,
        source_timestamp=ts,
        entity=ticker,
        data_type=DataType.OPTIONS_CHAIN,
        schema_version="1.0.0",
        source_hash=f"hash-options-{ticker}",
        validation_status=ValidationStatus.VALID,
        payload={
            "implied_vol_atm": implied_vol_atm,
            "implied_vol_skew": 0.05,
            "put_call_ratio": put_call_ratio,
            "total_volume": 145_320,
            "largest_block_strike": 235.0,
            "largest_block_volume": 12_500,
            "days_to_expiration": 14,
        },
    )


def _make_full_event_fragments() -> list[MarketStateFragment]:
    """Create a full set of event fragments (all 4 data types)."""
    return [
        _make_filing_8k_fragment(),
        _make_news_fragment(),
        _make_price_fragment(),
        _make_options_fragment(),
    ]


def _make_context(
    fragments: list[MarketStateFragment] | None = None,
) -> AgentContext:
    """Create a test AgentContext for COGNIT-EVENT."""
    return AgentContext(
        agent_id="COGNIT-EVENT",
        trigger="event",
        fragments=fragments if fragments is not None else _make_full_event_fragments(),
        context_window_hash="event_test_hash",
        timestamp=NOW,
        metadata={},
    )


def _make_mock_llm(response: dict) -> AsyncMock:
    """Create a mock LLM client that returns a fixed response."""
    mock = AsyncMock()
    mock.complete = AsyncMock(return_value=response)
    return mock


# ---------------------------------------------------------------------------
# Response parser tests (with event-specific responses)
# ---------------------------------------------------------------------------
class TestEventResponseParsing:
    """Tests for parsing event LLM responses via the shared response_parser."""

    def test_valid_event_response_produces_belief_object(self):
        """A well-formed event LLM response parses into a valid BeliefObject."""
        result = parse_llm_response(
            raw=VALID_EVENT_RESPONSE,
            agent_id="COGNIT-EVENT",
            context_window_hash="test_hash",
        )
        assert result is not None
        assert isinstance(result, BeliefObject)
        assert result.agent_id == "COGNIT-EVENT"
        assert len(result.beliefs) == 1

    def test_event_belief_fields(self):
        """Parsed event belief has correct direction, magnitude, confidence."""
        result = parse_llm_response(
            raw=VALID_EVENT_RESPONSE,
            agent_id="COGNIT-EVENT",
            context_window_hash="test_hash",
        )
        belief = result.beliefs[0]
        assert belief.direction == Direction.LONG
        assert belief.ticker == "AAPL"
        assert belief.raw_confidence == 0.72
        assert belief.time_horizon_days == 21

    def test_event_evidence_refs_parsed(self):
        """Evidence refs point to event data fragments."""
        result = parse_llm_response(
            raw=VALID_EVENT_RESPONSE,
            agent_id="COGNIT-EVENT",
            context_window_hash="test_hash",
        )
        evidence = result.beliefs[0].evidence
        assert len(evidence) == 3
        assert evidence[0].source_fragment_id == FILING_FRAG_ID
        assert evidence[0].field_path == "payload.deal_value"
        assert evidence[1].source_fragment_id == PRICE_FRAG_ID
        assert evidence[2].source_fragment_id == NEWS_FRAG_ID

    def test_event_invalidation_conditions_machine_evaluable(self):
        """All parsed conditions have metric, operator, and threshold."""
        result = parse_llm_response(
            raw=VALID_EVENT_RESPONSE,
            agent_id="COGNIT-EVENT",
            context_window_hash="test_hash",
        )
        conditions = result.beliefs[0].invalidation_conditions
        assert len(conditions) >= 2
        for cond in conditions:
            assert cond.metric
            assert cond.operator in ComparisonOperator
            assert cond.threshold is not None

    def test_multi_event_beliefs(self):
        """Multiple event beliefs parse correctly."""
        result = parse_llm_response(
            raw=MULTI_EVENT_BELIEF_RESPONSE,
            agent_id="COGNIT-EVENT",
            context_window_hash="test_hash",
        )
        assert result is not None
        assert len(result.beliefs) == 2
        directions = {b.direction for b in result.beliefs}
        assert Direction.SHORT in directions
        assert Direction.LONG in directions

    def test_event_empty_beliefs_returns_none(self):
        """Empty beliefs array returns None."""
        result = parse_llm_response(
            raw=EVENT_EMPTY_BELIEFS_RESPONSE,
            agent_id="COGNIT-EVENT",
            context_window_hash="test_hash",
        )
        assert result is None

    def test_event_vague_invalidation_drops_belief(self):
        """Beliefs with < 2 valid invalidation conditions are dropped."""
        result = parse_llm_response(
            raw=EVENT_VAGUE_INVALIDATION_RESPONSE,
            agent_id="COGNIT-EVENT",
            context_window_hash="test_hash",
        )
        # Only 1 valid condition → belief dropped → None
        assert result is None


# ---------------------------------------------------------------------------
# Prompt building tests
# ---------------------------------------------------------------------------
class TestEventPromptBuilding:
    """Tests for _build_user_prompt() event-specific fragment classification."""

    def test_filing_fragments_in_filing_section(self):
        """8K filing fragments appear in the filing events section."""
        agent = CognitEvent(llm_client=_make_mock_llm(VALID_EVENT_RESPONSE))
        context = _make_context(fragments=[_make_filing_8k_fragment()])

        prompt = agent._build_user_prompt(context)

        assert "FILING_8K" in prompt
        assert "M&A" in prompt
        assert "deal_value" in prompt

    def test_news_fragments_in_news_section(self):
        """News sentiment fragments appear in the news section."""
        agent = CognitEvent(llm_client=_make_mock_llm(VALID_EVENT_RESPONSE))
        context = _make_context(fragments=[_make_news_fragment()])

        prompt = agent._build_user_prompt(context)

        assert "SENTIMENT_NEWS" in prompt
        assert "0.8" in prompt
        assert "Reuters" in prompt

    def test_price_fragments_in_price_section(self):
        """Price fragments appear in the price action section."""
        agent = CognitEvent(llm_client=_make_mock_llm(VALID_EVENT_RESPONSE))
        context = _make_context(fragments=[_make_price_fragment()])

        prompt = agent._build_user_prompt(context)

        assert "PRICE_OHLCV" in prompt
        assert "230.0" in prompt

    def test_options_fragments_in_options_section(self):
        """Options fragments appear in the options activity section."""
        agent = CognitEvent(llm_client=_make_mock_llm(VALID_EVENT_RESPONSE))
        context = _make_context(fragments=[_make_options_fragment()])

        prompt = agent._build_user_prompt(context)

        assert "OPTIONS_CHAIN" in prompt
        assert "0.32" in prompt
        assert "put_call_ratio" in prompt

    def test_primary_ticker_identified_from_entities(self):
        """Primary ticker is identified from most common entity."""
        agent = CognitEvent(llm_client=_make_mock_llm(VALID_EVENT_RESPONSE))
        # 3 AAPL fragments, 1 MSFT fragment → primary = AAPL
        frags = [
            _make_filing_8k_fragment(fragment_id=uuid4(), ticker="AAPL"),
            _make_news_fragment(fragment_id=uuid4(), ticker="AAPL"),
            _make_price_fragment(fragment_id=uuid4(), entity="AAPL"),
            _make_options_fragment(fragment_id=uuid4(), ticker="MSFT"),
        ]
        context = _make_context(fragments=frags)

        prompt = agent._build_user_prompt(context)

        # AAPL should be the primary ticker used in the template
        assert "AAPL" in prompt

    def test_missing_news_shows_default(self):
        """When no news fragments exist, default message appears."""
        agent = CognitEvent(llm_client=_make_mock_llm(VALID_EVENT_RESPONSE))
        # Only filing and price — no news
        frags = [_make_filing_8k_fragment(), _make_price_fragment()]
        context = _make_context(fragments=frags)

        prompt = agent._build_user_prompt(context)

        assert "No news sentiment data available" in prompt

    def test_missing_options_shows_default(self):
        """When no options fragments exist, default message appears."""
        agent = CognitEvent(llm_client=_make_mock_llm(VALID_EVENT_RESPONSE))
        frags = [_make_filing_8k_fragment(), _make_news_fragment()]
        context = _make_context(fragments=frags)

        prompt = agent._build_user_prompt(context)

        assert "No options activity data available" in prompt

    def test_all_fragment_ids_listed(self):
        """All fragment IDs appear in the fragment_ids section."""
        agent = CognitEvent(llm_client=_make_mock_llm(VALID_EVENT_RESPONSE))
        frags = _make_full_event_fragments()
        context = _make_context(fragments=frags)

        prompt = agent._build_user_prompt(context)

        for frag in frags:
            assert str(frag.fragment_id) in prompt


# ---------------------------------------------------------------------------
# CognitEvent agent tests (mocked LLM)
# ---------------------------------------------------------------------------
class TestCognitEvent:
    """Tests for the CognitEvent agent with mocked LLM."""

    @pytest.mark.asyncio
    async def test_process_returns_belief_object(self):
        """Agent process() returns a valid BeliefObject."""
        mock_llm = _make_mock_llm(VALID_EVENT_RESPONSE)
        agent = CognitEvent(llm_client=mock_llm)
        context = _make_context()

        result = await agent.process(context)

        assert isinstance(result, BeliefObject)
        assert result.agent_id == "COGNIT-EVENT"
        assert len(result.beliefs) == 1
        mock_llm.complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_multi_belief(self):
        """Agent handles multi-belief LLM responses."""
        mock_llm = _make_mock_llm(MULTI_EVENT_BELIEF_RESPONSE)
        agent = CognitEvent(llm_client=mock_llm)
        context = _make_context()

        result = await agent.process(context)

        assert len(result.beliefs) == 2

    @pytest.mark.asyncio
    async def test_process_malformed_raises(self):
        """Agent raises AgentProcessingError on malformed LLM response."""
        mock_llm = _make_mock_llm({"analysis": "Events are complex"})
        agent = CognitEvent(llm_client=mock_llm)
        context = _make_context()

        with pytest.raises(AgentProcessingError):
            await agent.process(context)

    @pytest.mark.asyncio
    async def test_process_empty_beliefs_raises(self):
        """Agent raises AgentProcessingError when no beliefs parsed."""
        mock_llm = _make_mock_llm(EVENT_EMPTY_BELIEFS_RESPONSE)
        agent = CognitEvent(llm_client=mock_llm)
        context = _make_context()

        with pytest.raises(AgentProcessingError):
            await agent.process(context)

    @pytest.mark.asyncio
    async def test_process_vague_conditions_raises(self):
        """Agent raises when all beliefs are dropped due to vague conditions."""
        mock_llm = _make_mock_llm(EVENT_VAGUE_INVALIDATION_RESPONSE)
        agent = CognitEvent(llm_client=mock_llm)
        context = _make_context()

        with pytest.raises(AgentProcessingError):
            await agent.process(context)

    @pytest.mark.asyncio
    async def test_process_llm_error_raises(self):
        """Agent raises AgentProcessingError on LLM client error."""
        mock_llm = AsyncMock()
        mock_llm.complete = AsyncMock(side_effect=Exception("API down"))
        agent = CognitEvent(llm_client=mock_llm)
        context = _make_context()

        with pytest.raises(AgentProcessingError):
            await agent.process(context)

    @pytest.mark.asyncio
    async def test_context_window_hash_preserved(self):
        """BeliefObject carries the context_window_hash from input."""
        mock_llm = _make_mock_llm(VALID_EVENT_RESPONSE)
        agent = CognitEvent(llm_client=mock_llm)
        context = _make_context()

        result = await agent.process(context)

        assert result.context_window_hash == "event_test_hash"

    @pytest.mark.asyncio
    async def test_content_hash_computed(self):
        """BeliefObject has a non-empty content hash."""
        mock_llm = _make_mock_llm(VALID_EVENT_RESPONSE)
        agent = CognitEvent(llm_client=mock_llm)
        context = _make_context()

        result = await agent.process(context)

        assert result.content_hash
        assert len(result.content_hash) == 64  # SHA-256 hex

    @pytest.mark.asyncio
    async def test_llm_receives_system_and_user_prompt(self):
        """LLM client receives both system and user prompt."""
        mock_llm = _make_mock_llm(VALID_EVENT_RESPONSE)
        agent = CognitEvent(llm_client=mock_llm)
        context = _make_context()

        await agent.process(context)

        call_args = mock_llm.complete.call_args
        assert "system_prompt" in call_args.kwargs or len(call_args.args) >= 1
        assert "user_prompt" in call_args.kwargs or len(call_args.args) >= 2

    @pytest.mark.asyncio
    async def test_partial_data_still_works(self):
        """Agent works with only a subset of data types present."""
        mock_llm = _make_mock_llm(VALID_EVENT_RESPONSE)
        agent = CognitEvent(llm_client=mock_llm)
        # Only filing — no news, no price, no options
        context = _make_context(fragments=[_make_filing_8k_fragment()])

        result = await agent.process(context)

        assert isinstance(result, BeliefObject)
        assert len(result.beliefs) == 1

    @pytest.mark.asyncio
    async def test_empty_fragments_raises(self):
        """Agent handles empty fragment list via LLM (may produce no beliefs)."""
        mock_llm = _make_mock_llm(EVENT_EMPTY_BELIEFS_RESPONSE)
        agent = CognitEvent(llm_client=mock_llm)
        context = _make_context(fragments=[])

        with pytest.raises(AgentProcessingError):
            await agent.process(context)


# ---------------------------------------------------------------------------
# Health status tests
# ---------------------------------------------------------------------------
class TestCognitEventHealth:
    """Tests for agent health reporting."""

    def test_initial_health_is_healthy(self):
        """Agent starts in HEALTHY state."""
        agent = CognitEvent(llm_client=AsyncMock())
        health = agent.get_health()
        assert health.agent_id == "COGNIT-EVENT"
        assert health.status == AgentStatus.HEALTHY
        assert health.error_count_24h == 0

    @pytest.mark.asyncio
    async def test_health_after_success(self):
        """Health reports last_success after successful run."""
        mock_llm = _make_mock_llm(VALID_EVENT_RESPONSE)
        agent = CognitEvent(llm_client=mock_llm)
        context = _make_context()

        await agent.process(context)

        health = agent.get_health()
        assert health.last_run is not None
        assert health.last_success is not None
        assert health.status == AgentStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_health_degraded_after_errors(self):
        """Health degrades after repeated errors."""
        mock_llm = _make_mock_llm(EVENT_EMPTY_BELIEFS_RESPONSE)
        agent = CognitEvent(llm_client=mock_llm)
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
        mock_llm = _make_mock_llm(EVENT_EMPTY_BELIEFS_RESPONSE)
        agent = CognitEvent(llm_client=mock_llm)
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
class TestCognitEventProperties:
    """Test agent identity properties."""

    def test_agent_id(self):
        agent = CognitEvent(llm_client=AsyncMock())
        assert agent.agent_id == "COGNIT-EVENT"

    def test_agent_type(self):
        agent = CognitEvent(llm_client=AsyncMock())
        assert agent.agent_type == "cognition"

    def test_version(self):
        agent = CognitEvent(llm_client=AsyncMock())
        assert agent.version == "1.0.0"

    def test_consumed_data_types(self):
        """Agent consumes the correct data types."""
        expected = {
            DataType.FILING_8K,
            DataType.SENTIMENT_NEWS,
            DataType.PRICE_OHLCV,
            DataType.OPTIONS_CHAIN,
        }
        assert CognitEvent.CONSUMED_DATA_TYPES == expected
