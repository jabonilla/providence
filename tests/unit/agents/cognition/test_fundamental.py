"""Tests for COGNIT-FUNDAMENTAL Research Agent.

Tests the full Research Agent loop (7 steps) with mocked LLM client.
Validates belief parsing, evidence linking, invalidation condition
machine-evaluability, and error handling.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from providence.agents.base import AgentContext
from providence.agents.cognition.fundamental import CognitFundamental
from providence.agents.cognition.response_parser import (
    parse_llm_response,
    _parse_invalidation_conditions,
)
from providence.exceptions import AgentProcessingError
from providence.schemas.belief import BeliefObject
from providence.schemas.enums import (
    CatalystType,
    ComparisonOperator,
    DataType,
    Direction,
    Magnitude,
    MarketCapBucket,
    ValidationStatus,
)
from providence.schemas.market_state import MarketStateFragment

from tests.fixtures.llm_responses import (
    BAD_EVIDENCE_REFS_RESPONSE,
    EMPTY_BELIEFS_RESPONSE,
    MALFORMED_RESPONSE_NO_BELIEFS,
    MULTI_BELIEF_RESPONSE,
    OUT_OF_RANGE_CONFIDENCE_RESPONSE,
    VALID_FUNDAMENTAL_RESPONSE,
    VAGUE_INVALIDATION_RESPONSE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
NOW = datetime.now(timezone.utc)

FRAG_ID_1 = UUID("12345678-1234-5678-1234-567812345678")
FRAG_ID_2 = UUID("eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee")


def _make_fragment(
    fragment_id: UUID | None = None,
    entity: str = "AAPL",
    data_type: DataType = DataType.PRICE_OHLCV,
    hours_ago: float = 1,
    payload: dict | None = None,
) -> MarketStateFragment:
    """Create a test fragment."""
    ts = NOW - timedelta(hours=hours_ago)
    return MarketStateFragment(
        fragment_id=fragment_id or uuid4(),
        agent_id="PERCEPT-TEST",
        timestamp=ts,
        source_timestamp=ts,
        entity=entity,
        data_type=data_type,
        schema_version="1.0.0",
        source_hash=f"hash-{entity}-{data_type.value}",
        validation_status=ValidationStatus.VALID,
        payload=payload or {"close": 186.90, "volume": 52345678},
    )


def _make_context(
    fragments: list[MarketStateFragment] | None = None,
) -> AgentContext:
    """Create a test AgentContext."""
    frags = fragments or [
        _make_fragment(fragment_id=FRAG_ID_1, data_type=DataType.PRICE_OHLCV),
        _make_fragment(
            fragment_id=FRAG_ID_2,
            data_type=DataType.FILING_10Q,
            payload={"revenue": 124_300_000_000, "net_income": 33_900_000_000},
        ),
    ]
    return AgentContext(
        agent_id="COGNIT-FUNDAMENTAL",
        trigger="schedule",
        fragments=frags,
        context_window_hash="abc123hash",
        timestamp=NOW,
        metadata={},
    )


def _make_mock_llm(response: dict) -> AsyncMock:
    """Create a mock LLM client that returns a fixed response."""
    mock = AsyncMock()
    mock.complete = AsyncMock(return_value=response)
    return mock


# ---------------------------------------------------------------------------
# Response Parser tests
# ---------------------------------------------------------------------------
class TestResponseParser:
    """Tests for the response_parser module."""

    def test_valid_response_produces_belief_object(self):
        """A well-formed LLM response parses into a valid BeliefObject."""
        result = parse_llm_response(
            raw=VALID_FUNDAMENTAL_RESPONSE,
            agent_id="COGNIT-FUNDAMENTAL",
            context_window_hash="test_hash",
        )
        assert result is not None
        assert isinstance(result, BeliefObject)
        assert result.agent_id == "COGNIT-FUNDAMENTAL"
        assert result.context_window_hash == "test_hash"
        assert len(result.beliefs) == 1

    def test_valid_belief_fields(self):
        """Parsed belief has correct direction, magnitude, confidence."""
        result = parse_llm_response(
            raw=VALID_FUNDAMENTAL_RESPONSE,
            agent_id="COGNIT-FUNDAMENTAL",
            context_window_hash="test_hash",
        )
        belief = result.beliefs[0]
        assert belief.direction == Direction.LONG
        assert belief.magnitude == Magnitude.MODERATE
        assert belief.raw_confidence == 0.72
        assert belief.time_horizon_days == 90
        assert belief.ticker == "AAPL"

    def test_evidence_refs_parsed(self):
        """Evidence refs are properly parsed with valid UUIDs."""
        result = parse_llm_response(
            raw=VALID_FUNDAMENTAL_RESPONSE,
            agent_id="COGNIT-FUNDAMENTAL",
            context_window_hash="test_hash",
        )
        evidence = result.beliefs[0].evidence
        assert len(evidence) == 2
        assert evidence[0].source_fragment_id == FRAG_ID_1
        assert evidence[0].weight == 0.8
        assert "Revenue grew" in evidence[0].observation

    def test_invalidation_conditions_machine_evaluable(self):
        """All parsed invalidation conditions have metric, operator, threshold."""
        result = parse_llm_response(
            raw=VALID_FUNDAMENTAL_RESPONSE,
            agent_id="COGNIT-FUNDAMENTAL",
            context_window_hash="test_hash",
        )
        conditions = result.beliefs[0].invalidation_conditions
        assert len(conditions) >= 2

        for cond in conditions:
            assert cond.metric  # non-empty
            assert cond.operator in ComparisonOperator
            assert cond.threshold is not None

    def test_multi_belief_response(self):
        """Multiple beliefs are parsed from a single response."""
        result = parse_llm_response(
            raw=MULTI_BELIEF_RESPONSE,
            agent_id="COGNIT-FUNDAMENTAL",
            context_window_hash="test_hash",
        )
        assert result is not None
        assert len(result.beliefs) == 2
        assert result.beliefs[0].direction == Direction.LONG
        assert result.beliefs[1].direction == Direction.NEUTRAL

    def test_malformed_no_beliefs_returns_none(self):
        """Response without beliefs array returns None."""
        result = parse_llm_response(
            raw=MALFORMED_RESPONSE_NO_BELIEFS,
            agent_id="COGNIT-FUNDAMENTAL",
            context_window_hash="test_hash",
        )
        assert result is None

    def test_empty_beliefs_returns_none(self):
        """Response with empty beliefs array returns None."""
        result = parse_llm_response(
            raw=EMPTY_BELIEFS_RESPONSE,
            agent_id="COGNIT-FUNDAMENTAL",
            context_window_hash="test_hash",
        )
        assert result is None

    def test_vague_invalidation_conditions_drops_belief(self):
        """Beliefs with fewer than 2 valid invalidation conditions are dropped."""
        result = parse_llm_response(
            raw=VAGUE_INVALIDATION_RESPONSE,
            agent_id="COGNIT-FUNDAMENTAL",
            context_window_hash="test_hash",
        )
        # The belief only has 1 valid condition after filtering vague ones,
        # which is below the spec minimum of 2, so the entire belief is dropped
        assert result is None

    def test_bad_evidence_refs_filtered(self):
        """Non-UUID evidence refs are filtered out."""
        result = parse_llm_response(
            raw=BAD_EVIDENCE_REFS_RESPONSE,
            agent_id="COGNIT-FUNDAMENTAL",
            context_window_hash="test_hash",
        )
        assert result is not None
        evidence = result.beliefs[0].evidence
        # Only the valid UUID ref should survive
        assert len(evidence) == 1
        assert evidence[0].source_fragment_id == FRAG_ID_1

    def test_confidence_clamped_to_range(self):
        """Confidence > 1.0 is clamped to 1.0."""
        result = parse_llm_response(
            raw=OUT_OF_RANGE_CONFIDENCE_RESPONSE,
            agent_id="COGNIT-FUNDAMENTAL",
            context_window_hash="test_hash",
        )
        assert result is not None
        assert result.beliefs[0].raw_confidence == 1.0

    def test_metadata_parsed(self):
        """Belief metadata (sector, market_cap, catalyst) parsed correctly."""
        result = parse_llm_response(
            raw=VALID_FUNDAMENTAL_RESPONSE,
            agent_id="COGNIT-FUNDAMENTAL",
            context_window_hash="test_hash",
        )
        meta = result.beliefs[0].metadata
        assert meta.sector == "Technology"
        assert meta.market_cap_bucket == MarketCapBucket.MEGA
        assert meta.catalyst_type == CatalystType.EARNINGS

    def test_content_hash_computed(self):
        """BeliefObject has a non-empty content_hash."""
        result = parse_llm_response(
            raw=VALID_FUNDAMENTAL_RESPONSE,
            agent_id="COGNIT-FUNDAMENTAL",
            context_window_hash="test_hash",
        )
        assert result.content_hash  # auto-computed, should be non-empty


# ---------------------------------------------------------------------------
# Invalidation condition parsing edge cases
# ---------------------------------------------------------------------------
class TestInvalidationParsing:
    """Edge cases for invalidation condition parsing."""

    def test_operator_aliases(self):
        """Common operator aliases are mapped correctly."""
        raw = [
            {
                "metric": "price",
                "operator": ">",
                "threshold": 100,
                "description": "Price above 100",
            },
            {
                "metric": "margin",
                "operator": "LESS_THAN",
                "threshold": 0.30,
                "description": "Margin below 30%",
            },
        ]
        conditions = _parse_invalidation_conditions(raw)
        assert len(conditions) == 2
        assert conditions[0].operator == ComparisonOperator.GT
        assert conditions[1].operator == ComparisonOperator.LT

    def test_missing_threshold_skipped(self):
        """Conditions without threshold are skipped."""
        raw = [
            {
                "metric": "revenue",
                "operator": "GT",
                # No threshold
                "description": "Revenue goes up",
            },
        ]
        conditions = _parse_invalidation_conditions(raw)
        assert len(conditions) == 0

    def test_empty_metric_skipped(self):
        """Conditions with empty metric are skipped."""
        raw = [
            {
                "metric": "",
                "operator": "GT",
                "threshold": 100,
                "description": "Some condition",
            },
        ]
        conditions = _parse_invalidation_conditions(raw)
        assert len(conditions) == 0


# ---------------------------------------------------------------------------
# CognitFundamental agent tests (mocked LLM)
# ---------------------------------------------------------------------------
class TestCognitFundamental:
    """Tests for the CognitFundamental agent with mocked LLM."""

    @pytest.mark.asyncio
    async def test_process_returns_belief_object(self):
        """Agent process() returns a valid BeliefObject."""
        mock_llm = _make_mock_llm(VALID_FUNDAMENTAL_RESPONSE)
        agent = CognitFundamental(llm_client=mock_llm)
        context = _make_context()

        result = await agent.process(context)

        assert isinstance(result, BeliefObject)
        assert result.agent_id == "COGNIT-FUNDAMENTAL"
        assert len(result.beliefs) == 1
        mock_llm.complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_multi_belief(self):
        """Agent handles multi-belief LLM responses."""
        mock_llm = _make_mock_llm(MULTI_BELIEF_RESPONSE)
        agent = CognitFundamental(llm_client=mock_llm)
        context = _make_context()

        result = await agent.process(context)

        assert len(result.beliefs) == 2

    @pytest.mark.asyncio
    async def test_process_malformed_raises(self):
        """Agent raises AgentProcessingError on malformed LLM response."""
        mock_llm = _make_mock_llm(MALFORMED_RESPONSE_NO_BELIEFS)
        agent = CognitFundamental(llm_client=mock_llm)
        context = _make_context()

        with pytest.raises(AgentProcessingError):
            await agent.process(context)

    @pytest.mark.asyncio
    async def test_process_empty_beliefs_raises(self):
        """Agent raises AgentProcessingError when no beliefs parsed."""
        mock_llm = _make_mock_llm(EMPTY_BELIEFS_RESPONSE)
        agent = CognitFundamental(llm_client=mock_llm)
        context = _make_context()

        with pytest.raises(AgentProcessingError):
            await agent.process(context)

    @pytest.mark.asyncio
    async def test_process_llm_error_raises(self):
        """Agent raises AgentProcessingError on LLM client error."""
        mock_llm = AsyncMock()
        mock_llm.complete = AsyncMock(side_effect=Exception("API down"))
        agent = CognitFundamental(llm_client=mock_llm)
        context = _make_context()

        with pytest.raises(AgentProcessingError):
            await agent.process(context)

    @pytest.mark.asyncio
    async def test_context_window_hash_preserved(self):
        """BeliefObject carries the context_window_hash from input."""
        mock_llm = _make_mock_llm(VALID_FUNDAMENTAL_RESPONSE)
        agent = CognitFundamental(llm_client=mock_llm)
        context = _make_context()

        result = await agent.process(context)

        assert result.context_window_hash == "abc123hash"

    @pytest.mark.asyncio
    async def test_prompt_includes_fragment_ids(self):
        """User prompt sent to LLM includes fragment IDs."""
        mock_llm = _make_mock_llm(VALID_FUNDAMENTAL_RESPONSE)
        agent = CognitFundamental(llm_client=mock_llm)
        context = _make_context()

        await agent.process(context)

        # Check the user_prompt argument
        call_args = mock_llm.complete.call_args
        user_prompt = call_args.kwargs.get("user_prompt") or call_args[1] if len(call_args) > 1 else ""
        # The prompt should contain the fragment UUIDs
        if isinstance(user_prompt, str):
            assert str(FRAG_ID_1) in user_prompt

    @pytest.mark.asyncio
    async def test_prompt_includes_financial_data(self):
        """User prompt includes filing data from fragments."""
        filing_frag = _make_fragment(
            fragment_id=FRAG_ID_2,
            data_type=DataType.FILING_10Q,
            payload={"revenue": 124_300_000_000, "net_income": 33_900_000_000},
        )
        mock_llm = _make_mock_llm(VALID_FUNDAMENTAL_RESPONSE)
        agent = CognitFundamental(llm_client=mock_llm)
        context = _make_context(fragments=[
            _make_fragment(fragment_id=FRAG_ID_1),
            filing_frag,
        ])

        await agent.process(context)

        call_args = mock_llm.complete.call_args
        user_prompt = call_args.kwargs.get("user_prompt") or call_args[1] if len(call_args) > 1 else ""
        if isinstance(user_prompt, str):
            assert "124300000000" in user_prompt


# ---------------------------------------------------------------------------
# Health status tests
# ---------------------------------------------------------------------------
class TestCognitFundamentalHealth:
    """Tests for agent health reporting."""

    def test_initial_health_is_healthy(self):
        """Agent starts in HEALTHY state."""
        agent = CognitFundamental(llm_client=AsyncMock())
        health = agent.get_health()
        assert health.agent_id == "COGNIT-FUNDAMENTAL"
        assert health.status.value == "HEALTHY"

    @pytest.mark.asyncio
    async def test_health_after_success(self):
        """Health reports last_success after successful run."""
        mock_llm = _make_mock_llm(VALID_FUNDAMENTAL_RESPONSE)
        agent = CognitFundamental(llm_client=mock_llm)
        context = _make_context()

        await agent.process(context)

        health = agent.get_health()
        assert health.last_run is not None
        assert health.last_success is not None

    @pytest.mark.asyncio
    async def test_health_degraded_after_errors(self):
        """Health degrades after repeated errors."""
        mock_llm = _make_mock_llm(MALFORMED_RESPONSE_NO_BELIEFS)
        agent = CognitFundamental(llm_client=mock_llm)
        context = _make_context()

        for _ in range(4):
            try:
                await agent.process(context)
            except AgentProcessingError:
                pass

        health = agent.get_health()
        assert health.status.value == "DEGRADED"
        assert health.error_count_24h == 4


# ---------------------------------------------------------------------------
# Agent properties
# ---------------------------------------------------------------------------
class TestCognitFundamentalProperties:
    """Test agent identity properties."""

    def test_agent_id(self):
        agent = CognitFundamental(llm_client=AsyncMock())
        assert agent.agent_id == "COGNIT-FUNDAMENTAL"

    def test_agent_type(self):
        agent = CognitFundamental(llm_client=AsyncMock())
        assert agent.agent_type == "cognition"

    def test_version(self):
        agent = CognitFundamental(llm_client=AsyncMock())
        assert agent.version == "1.0.0"
