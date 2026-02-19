"""Tests for COGNIT-CROSSSEC Research Agent.

Tests the full Research Agent loop (7 steps) with mocked LLM client.
Validates cross-sectional prompt building with peer comparison,
belief parsing, evidence linking, invalidation condition
machine-evaluability, and error handling.

COGNIT-CROSSSEC is ADAPTIVE: uses Claude Sonnet 4 for cross-sectional analysis.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest

from providence.agents.base import AgentContext, AgentStatus
from providence.agents.cognition.crosssec import CognitCrossSec
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
    CROSSSEC_EMPTY_BELIEFS_RESPONSE,
    CROSSSEC_VAGUE_INVALIDATION_RESPONSE,
    MULTI_CROSSSEC_BELIEF_RESPONSE,
    VALID_CROSSSEC_RESPONSE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
NOW = datetime.now(timezone.utc)

FILING_10Q_FRAG_ID = UUID("dddddd01-dddd-dddd-dddd-dddddddddddd")
PRICE_FRAG_ID = UUID("dddddd02-dddd-dddd-dddd-dddddddddddd")
EARNINGS_FRAG_ID = UUID("dddddd03-dddd-dddd-dddd-dddddddddddd")
PEER_FRAG_ID = UUID("dddddd04-dddd-dddd-dddd-dddddddddddd")


def _make_filing_10q_fragment(
    fragment_id: UUID | None = None,
    ticker: str = "AAPL",
    revenue: float = 94_800_000_000,
    hours_ago: float = 1,
) -> MarketStateFragment:
    """Create a FILING_10Q test fragment."""
    ts = NOW - timedelta(hours=hours_ago)
    return MarketStateFragment(
        fragment_id=fragment_id or FILING_10Q_FRAG_ID,
        agent_id="PERCEPT-FILING",
        timestamp=ts,
        source_timestamp=ts,
        entity=ticker,
        data_type=DataType.FILING_10Q,
        schema_version="1.0.0",
        source_hash=f"hash-10q-{ticker}",
        validation_status=ValidationStatus.VALID,
        payload={
            "quarter": "2026-Q1",
            "revenue": revenue,
            "operating_income": 30_800_000_000,
            "operating_margin": 32.5,
            "pe_ratio": 22.0,
            "ev_ebitda": 18.5,
            "eps": 1.65,
            "cash_conversion_ratio": 1.1,
            "accruals_ratio": 0.04,
        },
    )


def _make_price_fragment(
    fragment_id: UUID | None = None,
    entity: str = "AAPL",
    close: float = 195.0,
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
            "open": 193.0,
            "high": 197.0,
            "low": 192.0,
            "close": close,
            "volume": 45_000_000,
            "vwap": 194.5,
            "return_30d_pct": -2.5,
        },
    )


def _make_earnings_call_fragment(
    fragment_id: UUID | None = None,
    ticker: str = "AAPL",
    tone_score: float = 0.72,
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
            "hedge_word_count": 8,
            "forward_guidance_mentions": 12,
            "transcript_length_words": 9200,
            "key_phrases": ["margin expansion", "peer-leading growth", "relative value"],
            "ceo_sentiment": "bullish",
        },
    )


def _make_peer_filing_fragment(
    fragment_id: UUID | None = None,
    ticker: str = "GOOG",
    hours_ago: float = 2,
) -> MarketStateFragment:
    """Create a peer's FILING_10Q fragment (different entity)."""
    ts = NOW - timedelta(hours=hours_ago)
    return MarketStateFragment(
        fragment_id=fragment_id or PEER_FRAG_ID,
        agent_id="PERCEPT-FILING",
        timestamp=ts,
        source_timestamp=ts,
        entity=ticker,
        data_type=DataType.FILING_10Q,
        schema_version="1.0.0",
        source_hash=f"hash-10q-{ticker}",
        validation_status=ValidationStatus.VALID,
        payload={
            "quarter": "2026-Q1",
            "revenue": 86_300_000_000,
            "operating_income": 24_200_000_000,
            "operating_margin": 28.0,
            "pe_ratio": 28.0,
            "ev_ebitda": 22.0,
            "eps": 1.42,
            "cash_conversion_ratio": 0.85,
            "accruals_ratio": 0.07,
        },
    )


def _make_full_crosssec_fragments() -> list[MarketStateFragment]:
    """Create a full set: primary entity (3 types) + 1 peer fragment."""
    return [
        _make_filing_10q_fragment(),
        _make_price_fragment(),
        _make_earnings_call_fragment(),
        _make_peer_filing_fragment(),
    ]


def _make_context(
    fragments: list[MarketStateFragment] | None = None,
) -> AgentContext:
    """Create a test AgentContext for COGNIT-CROSSSEC."""
    return AgentContext(
        agent_id="COGNIT-CROSSSEC",
        trigger="schedule",
        fragments=fragments if fragments is not None else _make_full_crosssec_fragments(),
        context_window_hash="crosssec_test_hash",
        timestamp=NOW,
        metadata={},
    )


def _make_mock_llm(response: dict) -> AsyncMock:
    """Create a mock LLM client that returns a fixed response."""
    mock = AsyncMock()
    mock.complete = AsyncMock(return_value=response)
    return mock


# ---------------------------------------------------------------------------
# Response parser tests (with crosssec-specific responses)
# ---------------------------------------------------------------------------
class TestCrossSecResponseParsing:
    """Tests for parsing crosssec LLM responses via the shared response_parser."""

    def test_valid_crosssec_response_produces_belief_object(self):
        """A well-formed crosssec LLM response parses into a valid BeliefObject."""
        result = parse_llm_response(
            raw=VALID_CROSSSEC_RESPONSE,
            agent_id="COGNIT-CROSSSEC",
            context_window_hash="test_hash",
        )
        assert result is not None
        assert isinstance(result, BeliefObject)
        assert result.agent_id == "COGNIT-CROSSSEC"
        assert len(result.beliefs) == 1

    def test_crosssec_belief_fields(self):
        """Parsed crosssec belief has correct direction, magnitude, confidence."""
        result = parse_llm_response(
            raw=VALID_CROSSSEC_RESPONSE,
            agent_id="COGNIT-CROSSSEC",
            context_window_hash="test_hash",
        )
        belief = result.beliefs[0]
        assert belief.direction == Direction.LONG
        assert belief.ticker == "AAPL"
        assert belief.raw_confidence == 0.65
        assert belief.time_horizon_days == 60

    def test_crosssec_evidence_refs_parsed(self):
        """Evidence refs point to crosssec data fragments."""
        result = parse_llm_response(
            raw=VALID_CROSSSEC_RESPONSE,
            agent_id="COGNIT-CROSSSEC",
            context_window_hash="test_hash",
        )
        evidence = result.beliefs[0].evidence
        assert len(evidence) == 3
        assert evidence[0].source_fragment_id == FILING_10Q_FRAG_ID
        assert evidence[0].field_path == "payload.pe_ratio"
        assert evidence[1].source_fragment_id == PRICE_FRAG_ID
        assert evidence[2].source_fragment_id == EARNINGS_FRAG_ID

    def test_crosssec_invalidation_conditions_machine_evaluable(self):
        """All parsed conditions have metric, operator, and threshold."""
        result = parse_llm_response(
            raw=VALID_CROSSSEC_RESPONSE,
            agent_id="COGNIT-CROSSSEC",
            context_window_hash="test_hash",
        )
        conditions = result.beliefs[0].invalidation_conditions
        assert len(conditions) >= 2
        for cond in conditions:
            assert cond.metric
            assert cond.operator in ComparisonOperator
            assert cond.threshold is not None

    def test_multi_crosssec_beliefs(self):
        """Multiple crosssec beliefs parse correctly."""
        result = parse_llm_response(
            raw=MULTI_CROSSSEC_BELIEF_RESPONSE,
            agent_id="COGNIT-CROSSSEC",
            context_window_hash="test_hash",
        )
        assert result is not None
        assert len(result.beliefs) == 2
        directions = {b.direction for b in result.beliefs}
        assert Direction.LONG in directions
        assert Direction.SHORT in directions

    def test_crosssec_empty_beliefs_returns_none(self):
        """Empty beliefs array returns None."""
        result = parse_llm_response(
            raw=CROSSSEC_EMPTY_BELIEFS_RESPONSE,
            agent_id="COGNIT-CROSSSEC",
            context_window_hash="test_hash",
        )
        assert result is None

    def test_crosssec_vague_invalidation_drops_belief(self):
        """Beliefs with < 2 valid invalidation conditions are dropped."""
        result = parse_llm_response(
            raw=CROSSSEC_VAGUE_INVALIDATION_RESPONSE,
            agent_id="COGNIT-CROSSSEC",
            context_window_hash="test_hash",
        )
        # Only 1 valid condition → belief dropped → None
        assert result is None


# ---------------------------------------------------------------------------
# Prompt building tests (with peer comparison)
# ---------------------------------------------------------------------------
class TestCrossSecPromptBuilding:
    """Tests for _build_user_prompt() with cross-sectional peer comparison."""

    def test_quarterly_fragments_in_quarterly_section(self):
        """Primary entity's 10-Q fragments appear in quarterly section."""
        agent = CognitCrossSec(llm_client=_make_mock_llm(VALID_CROSSSEC_RESPONSE))
        context = _make_context(fragments=[_make_filing_10q_fragment()])

        prompt = agent._build_user_prompt(context)

        assert "FILING_10Q" in prompt
        assert "pe_ratio" in prompt
        assert "22.0" in prompt

    def test_price_fragments_in_price_section(self):
        """Primary entity's price fragments appear in price section."""
        agent = CognitCrossSec(llm_client=_make_mock_llm(VALID_CROSSSEC_RESPONSE))
        context = _make_context(fragments=[_make_price_fragment()])

        prompt = agent._build_user_prompt(context)

        assert "PRICE_OHLCV" in prompt
        assert "195.0" in prompt

    def test_earnings_fragments_in_earnings_section(self):
        """Primary entity's earnings call fragments appear in earnings section."""
        agent = CognitCrossSec(llm_client=_make_mock_llm(VALID_CROSSSEC_RESPONSE))
        context = _make_context(fragments=[_make_earnings_call_fragment()])

        prompt = agent._build_user_prompt(context)

        assert "EARNINGS_CALL" in prompt
        assert "tone_score" in prompt
        assert "0.72" in prompt

    def test_peer_fragments_in_peer_section(self):
        """Non-primary entity fragments go to the peer section."""
        agent = CognitCrossSec(llm_client=_make_mock_llm(VALID_CROSSSEC_RESPONSE))
        # 2 AAPL fragments (primary) + 1 GOOG fragment (peer)
        frags = [
            _make_filing_10q_fragment(ticker="AAPL"),
            _make_price_fragment(fragment_id=uuid4(), entity="AAPL"),
            _make_peer_filing_fragment(ticker="GOOG"),
        ]
        context = _make_context(fragments=frags)

        prompt = agent._build_user_prompt(context)

        # GOOG should be in the peer section, not in quarterly section
        assert "AAPL" in prompt
        assert "GOOG" in prompt

    def test_primary_ticker_identified_from_entities(self):
        """Primary ticker is the most common entity in fragments."""
        agent = CognitCrossSec(llm_client=_make_mock_llm(VALID_CROSSSEC_RESPONSE))
        # 3 AAPL fragments, 1 GOOG → primary = AAPL
        frags = [
            _make_filing_10q_fragment(fragment_id=uuid4(), ticker="AAPL"),
            _make_price_fragment(fragment_id=uuid4(), entity="AAPL"),
            _make_earnings_call_fragment(fragment_id=uuid4(), ticker="AAPL"),
            _make_peer_filing_fragment(fragment_id=uuid4(), ticker="GOOG"),
        ]
        context = _make_context(fragments=frags)

        prompt = agent._build_user_prompt(context)

        # AAPL should be the primary ticker in the template
        assert "AAPL" in prompt

    def test_missing_quarterly_shows_default(self):
        """When no 10-Q fragments for primary entity, default message appears."""
        agent = CognitCrossSec(llm_client=_make_mock_llm(VALID_CROSSSEC_RESPONSE))
        # Only price — no quarterly filings
        frags = [_make_price_fragment()]
        context = _make_context(fragments=frags)

        prompt = agent._build_user_prompt(context)

        assert "No quarterly filing data available" in prompt

    def test_missing_peer_shows_default(self):
        """When no peer fragments exist, default message appears."""
        agent = CognitCrossSec(llm_client=_make_mock_llm(VALID_CROSSSEC_RESPONSE))
        # All fragments for same entity — no peers
        frags = [
            _make_filing_10q_fragment(ticker="AAPL"),
            _make_price_fragment(fragment_id=uuid4(), entity="AAPL"),
        ]
        context = _make_context(fragments=frags)

        prompt = agent._build_user_prompt(context)

        assert "No peer data available" in prompt

    def test_all_fragment_ids_listed(self):
        """All fragment IDs appear in the fragment_ids section."""
        agent = CognitCrossSec(llm_client=_make_mock_llm(VALID_CROSSSEC_RESPONSE))
        frags = _make_full_crosssec_fragments()
        context = _make_context(fragments=frags)

        prompt = agent._build_user_prompt(context)

        for frag in frags:
            assert str(frag.fragment_id) in prompt

    def test_non_primary_entity_routed_to_peers(self):
        """Fragments from non-primary entities go to peer section regardless of type."""
        agent = CognitCrossSec(llm_client=_make_mock_llm(VALID_CROSSSEC_RESPONSE))
        # 2 AAPL (primary) + 1 MSFT price + 1 GOOG 10-Q (both peers)
        frags = [
            _make_filing_10q_fragment(fragment_id=uuid4(), ticker="AAPL"),
            _make_price_fragment(fragment_id=uuid4(), entity="AAPL"),
            _make_price_fragment(fragment_id=uuid4(), entity="MSFT", close=420.0),
            _make_peer_filing_fragment(fragment_id=uuid4(), ticker="GOOG"),
        ]
        context = _make_context(fragments=frags)

        prompt = agent._build_user_prompt(context)

        # Both peer entities should appear
        assert "AAPL" in prompt
        assert "MSFT" in prompt
        assert "GOOG" in prompt


# ---------------------------------------------------------------------------
# CognitCrossSec agent tests (mocked LLM)
# ---------------------------------------------------------------------------
class TestCognitCrossSec:
    """Tests for the CognitCrossSec agent with mocked LLM."""

    @pytest.mark.asyncio
    async def test_process_returns_belief_object(self):
        """Agent process() returns a valid BeliefObject."""
        mock_llm = _make_mock_llm(VALID_CROSSSEC_RESPONSE)
        agent = CognitCrossSec(llm_client=mock_llm)
        context = _make_context()

        result = await agent.process(context)

        assert isinstance(result, BeliefObject)
        assert result.agent_id == "COGNIT-CROSSSEC"
        assert len(result.beliefs) == 1
        mock_llm.complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_multi_belief(self):
        """Agent handles multi-belief LLM responses."""
        mock_llm = _make_mock_llm(MULTI_CROSSSEC_BELIEF_RESPONSE)
        agent = CognitCrossSec(llm_client=mock_llm)
        context = _make_context()

        result = await agent.process(context)

        assert len(result.beliefs) == 2

    @pytest.mark.asyncio
    async def test_process_malformed_raises(self):
        """Agent raises AgentProcessingError on malformed LLM response."""
        mock_llm = _make_mock_llm({"analysis": "Peers look cheap"})
        agent = CognitCrossSec(llm_client=mock_llm)
        context = _make_context()

        with pytest.raises(AgentProcessingError):
            await agent.process(context)

    @pytest.mark.asyncio
    async def test_process_empty_beliefs_raises(self):
        """Agent raises AgentProcessingError when no beliefs parsed."""
        mock_llm = _make_mock_llm(CROSSSEC_EMPTY_BELIEFS_RESPONSE)
        agent = CognitCrossSec(llm_client=mock_llm)
        context = _make_context()

        with pytest.raises(AgentProcessingError):
            await agent.process(context)

    @pytest.mark.asyncio
    async def test_process_vague_conditions_raises(self):
        """Agent raises when all beliefs are dropped due to vague conditions."""
        mock_llm = _make_mock_llm(CROSSSEC_VAGUE_INVALIDATION_RESPONSE)
        agent = CognitCrossSec(llm_client=mock_llm)
        context = _make_context()

        with pytest.raises(AgentProcessingError):
            await agent.process(context)

    @pytest.mark.asyncio
    async def test_process_llm_error_raises(self):
        """Agent raises AgentProcessingError on LLM client error."""
        mock_llm = AsyncMock()
        mock_llm.complete = AsyncMock(side_effect=Exception("API down"))
        agent = CognitCrossSec(llm_client=mock_llm)
        context = _make_context()

        with pytest.raises(AgentProcessingError):
            await agent.process(context)

    @pytest.mark.asyncio
    async def test_context_window_hash_preserved(self):
        """BeliefObject carries the context_window_hash from input."""
        mock_llm = _make_mock_llm(VALID_CROSSSEC_RESPONSE)
        agent = CognitCrossSec(llm_client=mock_llm)
        context = _make_context()

        result = await agent.process(context)

        assert result.context_window_hash == "crosssec_test_hash"

    @pytest.mark.asyncio
    async def test_content_hash_computed(self):
        """BeliefObject has a non-empty content hash."""
        mock_llm = _make_mock_llm(VALID_CROSSSEC_RESPONSE)
        agent = CognitCrossSec(llm_client=mock_llm)
        context = _make_context()

        result = await agent.process(context)

        assert result.content_hash
        assert len(result.content_hash) == 64  # SHA-256 hex

    @pytest.mark.asyncio
    async def test_llm_receives_system_and_user_prompt(self):
        """LLM client receives both system and user prompt."""
        mock_llm = _make_mock_llm(VALID_CROSSSEC_RESPONSE)
        agent = CognitCrossSec(llm_client=mock_llm)
        context = _make_context()

        await agent.process(context)

        call_args = mock_llm.complete.call_args
        assert "system_prompt" in call_args.kwargs or len(call_args.args) >= 1
        assert "user_prompt" in call_args.kwargs or len(call_args.args) >= 2


# ---------------------------------------------------------------------------
# Health status tests
# ---------------------------------------------------------------------------
class TestCognitCrossSecHealth:
    """Tests for agent health reporting."""

    def test_initial_health_is_healthy(self):
        """Agent starts in HEALTHY state."""
        agent = CognitCrossSec(llm_client=AsyncMock())
        health = agent.get_health()
        assert health.agent_id == "COGNIT-CROSSSEC"
        assert health.status == AgentStatus.HEALTHY
        assert health.error_count_24h == 0

    @pytest.mark.asyncio
    async def test_health_after_success(self):
        """Health reports last_success after successful run."""
        mock_llm = _make_mock_llm(VALID_CROSSSEC_RESPONSE)
        agent = CognitCrossSec(llm_client=mock_llm)
        context = _make_context()

        await agent.process(context)

        health = agent.get_health()
        assert health.last_run is not None
        assert health.last_success is not None
        assert health.status == AgentStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_health_degraded_after_errors(self):
        """Health degrades after repeated errors."""
        mock_llm = _make_mock_llm(CROSSSEC_EMPTY_BELIEFS_RESPONSE)
        agent = CognitCrossSec(llm_client=mock_llm)
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
        mock_llm = _make_mock_llm(CROSSSEC_EMPTY_BELIEFS_RESPONSE)
        agent = CognitCrossSec(llm_client=mock_llm)
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
class TestCognitCrossSecProperties:
    """Test agent identity properties."""

    def test_agent_id(self):
        agent = CognitCrossSec(llm_client=AsyncMock())
        assert agent.agent_id == "COGNIT-CROSSSEC"

    def test_agent_type(self):
        agent = CognitCrossSec(llm_client=AsyncMock())
        assert agent.agent_type == "cognition"

    def test_version(self):
        agent = CognitCrossSec(llm_client=AsyncMock())
        assert agent.version == "1.0.0"

    def test_consumed_data_types(self):
        """Agent consumes the correct data types."""
        expected = {
            DataType.FILING_10Q,
            DataType.PRICE_OHLCV,
            DataType.EARNINGS_CALL,
        }
        assert CognitCrossSec.CONSUMED_DATA_TYPES == expected
