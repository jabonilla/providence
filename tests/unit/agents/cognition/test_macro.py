"""Tests for COGNIT-MACRO Research Agent.

Tests the full Research Agent loop (7 steps) with mocked LLM client.
Validates macro-specific prompt building, belief parsing, evidence linking,
invalidation condition machine-evaluability, and error handling.

COGNIT-MACRO is ADAPTIVE: uses Claude Sonnet 4 for macroeconomic analysis.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest

from providence.agents.base import AgentContext, AgentStatus
from providence.agents.cognition.macro import CognitMacro
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
    MACRO_EMPTY_BELIEFS_RESPONSE,
    MACRO_VAGUE_INVALIDATION_RESPONSE,
    MULTI_MACRO_BELIEF_RESPONSE,
    VALID_MACRO_RESPONSE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
NOW = datetime.now(timezone.utc)

YIELD_FRAG_ID = UUID("aaaaaa01-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
CDS_FRAG_ID = UUID("aaaaaa02-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
ECON_FRAG_ID = UUID("aaaaaa03-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
PRICE_FRAG_ID = UUID("aaaaaa04-aaaa-aaaa-aaaa-aaaaaaaaaaaa")


def _make_yield_curve_fragment(
    fragment_id: UUID | None = None,
    hours_ago: float = 1,
) -> MarketStateFragment:
    """Create a MACRO_YIELD_CURVE test fragment."""
    ts = NOW - timedelta(hours=hours_ago)
    return MarketStateFragment(
        fragment_id=fragment_id or YIELD_FRAG_ID,
        agent_id="PERCEPT-MACRO",
        timestamp=ts,
        source_timestamp=ts,
        entity="US",
        data_type=DataType.MACRO_YIELD_CURVE,
        schema_version="1.0.0",
        source_hash="hash-yield-curve",
        validation_status=ValidationStatus.VALID,
        payload={
            "curve_date": "2026-02-12",
            "tenors": {
                "1M": 5.42, "3M": 5.38, "6M": 5.25,
                "1Y": 4.95, "2Y": 4.35, "5Y": 3.88,
                "10Y": 3.65, "20Y": 3.80, "30Y": 3.92,
            },
            "spread_2s10s": -70.0,
            "spread_3m10y": -173.0,
            "curve_source": "FRED",
            "previous_tenors": {
                "1M": 5.40, "3M": 5.36, "10Y": 3.60,
            },
        },
    )


def _make_cds_fragment(
    fragment_id: UUID | None = None,
    hours_ago: float = 1,
) -> MarketStateFragment:
    """Create a MACRO_CDS test fragment."""
    ts = NOW - timedelta(hours=hours_ago)
    return MarketStateFragment(
        fragment_id=fragment_id or CDS_FRAG_ID,
        agent_id="PERCEPT-CDS",
        timestamp=ts,
        source_timestamp=ts,
        entity="IG_CDX",
        data_type=DataType.MACRO_CDS,
        schema_version="1.0.0",
        source_hash="hash-cds-ig",
        validation_status=ValidationStatus.VALID,
        payload={
            "reference_entity": "US IG",
            "tenor": "5Y",
            "spread_bps": 185.5,
            "previous_spread_bps": 165.0,
            "spread_change_bps": 20.5,
            "recovery_rate": 0.4,
            "currency": "USD",
            "observation_date": "2026-02-12",
        },
    )


def _make_economic_fragment(
    fragment_id: UUID | None = None,
    indicator: str = "CPI",
    value: float = 2.8,
    previous_value: float = 2.9,
    hours_ago: float = 1,
) -> MarketStateFragment:
    """Create a MACRO_ECONOMIC test fragment."""
    ts = NOW - timedelta(hours=hours_ago)
    return MarketStateFragment(
        fragment_id=fragment_id or ECON_FRAG_ID,
        agent_id="PERCEPT-MACRO",
        timestamp=ts,
        source_timestamp=ts,
        entity="US",
        data_type=DataType.MACRO_ECONOMIC,
        schema_version="1.0.0",
        source_hash=f"hash-econ-{indicator}",
        validation_status=ValidationStatus.VALID,
        payload={
            "indicator": indicator,
            "value": value,
            "previous_value": previous_value,
            "period": "2026-01",
            "frequency": "MONTHLY",
            "unit": "PERCENT",
            "source_series_id": "CPIAUCSL",
            "observation_date": "2026-02-12",
            "revision_number": 0,
        },
    )


def _make_price_fragment(
    fragment_id: UUID | None = None,
    entity: str = "SPY",
    close: float = 506.45,
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
            "open": close - 2.0,
            "high": close + 1.5,
            "low": close - 4.0,
            "close": close,
            "volume": 2_145_678_900,
            "vwap": close - 0.5,
            "timeframe": "1D",
        },
    )


def _make_full_macro_fragments() -> list[MarketStateFragment]:
    """Create a full set of macro fragments (all 4 data types)."""
    return [
        _make_yield_curve_fragment(),
        _make_cds_fragment(),
        _make_economic_fragment(),
        _make_price_fragment(),
    ]


def _make_context(
    fragments: list[MarketStateFragment] | None = None,
) -> AgentContext:
    """Create a test AgentContext for COGNIT-MACRO."""
    return AgentContext(
        agent_id="COGNIT-MACRO",
        trigger="schedule",
        fragments=fragments or _make_full_macro_fragments(),
        context_window_hash="macro_test_hash",
        timestamp=NOW,
        metadata={},
    )


def _make_mock_llm(response: dict) -> AsyncMock:
    """Create a mock LLM client that returns a fixed response."""
    mock = AsyncMock()
    mock.complete = AsyncMock(return_value=response)
    return mock


# ---------------------------------------------------------------------------
# Response parser tests (with macro-specific responses)
# ---------------------------------------------------------------------------
class TestMacroResponseParsing:
    """Tests for parsing macro LLM responses via the shared response_parser."""

    def test_valid_macro_response_produces_belief_object(self):
        """A well-formed macro LLM response parses into a valid BeliefObject."""
        result = parse_llm_response(
            raw=VALID_MACRO_RESPONSE,
            agent_id="COGNIT-MACRO",
            context_window_hash="test_hash",
        )
        assert result is not None
        assert isinstance(result, BeliefObject)
        assert result.agent_id == "COGNIT-MACRO"
        assert len(result.beliefs) == 1

    def test_macro_belief_fields(self):
        """Parsed macro belief has correct direction, magnitude, confidence."""
        result = parse_llm_response(
            raw=VALID_MACRO_RESPONSE,
            agent_id="COGNIT-MACRO",
            context_window_hash="test_hash",
        )
        belief = result.beliefs[0]
        assert belief.direction == Direction.SHORT
        assert belief.ticker == "SPY"
        assert belief.raw_confidence == 0.68
        assert belief.time_horizon_days == 120

    def test_macro_evidence_refs_parsed(self):
        """Evidence refs point to macro data fragments."""
        result = parse_llm_response(
            raw=VALID_MACRO_RESPONSE,
            agent_id="COGNIT-MACRO",
            context_window_hash="test_hash",
        )
        evidence = result.beliefs[0].evidence
        assert len(evidence) == 2
        assert evidence[0].source_fragment_id == YIELD_FRAG_ID
        assert evidence[0].field_path == "payload.spread_2s10s"
        assert evidence[1].source_fragment_id == CDS_FRAG_ID

    def test_macro_invalidation_conditions_machine_evaluable(self):
        """All parsed conditions have metric, operator, and threshold."""
        result = parse_llm_response(
            raw=VALID_MACRO_RESPONSE,
            agent_id="COGNIT-MACRO",
            context_window_hash="test_hash",
        )
        conditions = result.beliefs[0].invalidation_conditions
        assert len(conditions) >= 2
        for cond in conditions:
            assert cond.metric
            assert cond.operator in ComparisonOperator
            assert cond.threshold is not None

    def test_multi_macro_beliefs(self):
        """Multiple macro beliefs parse correctly."""
        result = parse_llm_response(
            raw=MULTI_MACRO_BELIEF_RESPONSE,
            agent_id="COGNIT-MACRO",
            context_window_hash="test_hash",
        )
        assert result is not None
        assert len(result.beliefs) == 2
        tickers = {b.ticker for b in result.beliefs}
        assert "XLU" in tickers
        assert "HYG" in tickers

    def test_macro_empty_beliefs_returns_none(self):
        """Empty beliefs array returns None."""
        result = parse_llm_response(
            raw=MACRO_EMPTY_BELIEFS_RESPONSE,
            agent_id="COGNIT-MACRO",
            context_window_hash="test_hash",
        )
        assert result is None

    def test_macro_vague_invalidation_drops_belief(self):
        """Beliefs with < 2 valid invalidation conditions are dropped."""
        result = parse_llm_response(
            raw=MACRO_VAGUE_INVALIDATION_RESPONSE,
            agent_id="COGNIT-MACRO",
            context_window_hash="test_hash",
        )
        # Only 1 valid condition → belief dropped → None
        assert result is None


# ---------------------------------------------------------------------------
# Prompt building tests
# ---------------------------------------------------------------------------
class TestMacroPromptBuilding:
    """Tests for _build_user_prompt() macro-specific fragment classification."""

    def test_yield_curve_fragments_in_yield_section(self):
        """Yield curve fragments appear in the yield curve section."""
        agent = CognitMacro(llm_client=_make_mock_llm(VALID_MACRO_RESPONSE))
        context = _make_context(fragments=[_make_yield_curve_fragment()])

        prompt = agent._build_user_prompt(context)

        assert "MACRO_YIELD_CURVE" in prompt
        assert "spread_2s10s" in prompt
        assert "-70.0" in prompt

    def test_cds_fragments_in_credit_section(self):
        """CDS fragments appear in the credit section."""
        agent = CognitMacro(llm_client=_make_mock_llm(VALID_MACRO_RESPONSE))
        context = _make_context(fragments=[_make_cds_fragment()])

        prompt = agent._build_user_prompt(context)

        assert "MACRO_CDS" in prompt
        assert "185.5" in prompt
        assert "spread_bps" in prompt

    def test_economic_fragments_in_economic_section(self):
        """Economic indicator fragments appear in the economic section."""
        agent = CognitMacro(llm_client=_make_mock_llm(VALID_MACRO_RESPONSE))
        context = _make_context(fragments=[_make_economic_fragment()])

        prompt = agent._build_user_prompt(context)

        assert "MACRO_ECONOMIC" in prompt
        assert "CPI" in prompt
        assert "2.8" in prompt

    def test_price_fragments_in_price_section(self):
        """Price fragments appear in the broad market section."""
        agent = CognitMacro(llm_client=_make_mock_llm(VALID_MACRO_RESPONSE))
        context = _make_context(fragments=[_make_price_fragment()])

        prompt = agent._build_user_prompt(context)

        assert "PRICE_OHLCV" in prompt
        assert "506.45" in prompt

    def test_missing_cds_shows_default(self):
        """When no CDS fragments exist, default message appears."""
        agent = CognitMacro(llm_client=_make_mock_llm(VALID_MACRO_RESPONSE))
        # Only yield curve and price — no CDS
        frags = [_make_yield_curve_fragment(), _make_price_fragment()]
        context = _make_context(fragments=frags)

        prompt = agent._build_user_prompt(context)

        assert "No CDS data available" in prompt

    def test_missing_economic_shows_default(self):
        """When no economic fragments exist, default message appears."""
        agent = CognitMacro(llm_client=_make_mock_llm(VALID_MACRO_RESPONSE))
        frags = [_make_yield_curve_fragment(), _make_cds_fragment()]
        context = _make_context(fragments=frags)

        prompt = agent._build_user_prompt(context)

        assert "No economic indicator data available" in prompt

    def test_all_fragment_ids_listed(self):
        """All fragment IDs appear in the fragment_ids section."""
        agent = CognitMacro(llm_client=_make_mock_llm(VALID_MACRO_RESPONSE))
        frags = _make_full_macro_fragments()
        context = _make_context(fragments=frags)

        prompt = agent._build_user_prompt(context)

        for frag in frags:
            assert str(frag.fragment_id) in prompt

    def test_multiple_economic_indicators_all_present(self):
        """Multiple economic indicators all appear in the prompt."""
        agent = CognitMacro(llm_client=_make_mock_llm(VALID_MACRO_RESPONSE))
        cpi = _make_economic_fragment(
            fragment_id=uuid4(), indicator="CPI", value=2.8
        )
        gdp = _make_economic_fragment(
            fragment_id=uuid4(), indicator="GDP", value=2.1
        )
        unemployment = _make_economic_fragment(
            fragment_id=uuid4(), indicator="UNEMPLOYMENT_RATE", value=3.9
        )
        context = _make_context(fragments=[cpi, gdp, unemployment])

        prompt = agent._build_user_prompt(context)

        assert "CPI" in prompt
        assert "GDP" in prompt
        assert "UNEMPLOYMENT_RATE" in prompt

    def test_no_primary_ticker_concept(self):
        """Macro agent does NOT identify a primary ticker (unlike FUNDAMENTAL)."""
        agent = CognitMacro(llm_client=_make_mock_llm(VALID_MACRO_RESPONSE))
        frags = _make_full_macro_fragments()
        context = _make_context(fragments=frags)

        prompt = agent._build_user_prompt(context)

        # All data types appear regardless of entity
        assert "MACRO_YIELD_CURVE" in prompt
        assert "MACRO_CDS" in prompt
        assert "MACRO_ECONOMIC" in prompt
        assert "PRICE_OHLCV" in prompt

    def test_non_consumed_fragments_excluded(self):
        """Fragments outside CONSUMED_DATA_TYPES are not included in sections."""
        agent = CognitMacro(llm_client=_make_mock_llm(VALID_MACRO_RESPONSE))
        filing_frag = MarketStateFragment(
            fragment_id=uuid4(),
            agent_id="PERCEPT-FILING",
            timestamp=NOW,
            source_timestamp=NOW,
            entity="AAPL",
            data_type=DataType.FILING_10Q,
            schema_version="1.0.0",
            source_hash="filing-hash",
            validation_status=ValidationStatus.VALID,
            payload={"revenue": 124_300_000_000},
        )
        frags = [_make_yield_curve_fragment(), filing_frag]
        context = _make_context(fragments=frags)

        prompt = agent._build_user_prompt(context)

        # Filing fragment appears in fragment_ids list but not in any data section
        assert str(filing_frag.fragment_id) in prompt  # listed in fragment IDs
        assert "No CDS data available" in prompt  # CDS section empty
        assert "No economic indicator data available" in prompt  # Economic section empty


# ---------------------------------------------------------------------------
# CognitMacro agent tests (mocked LLM)
# ---------------------------------------------------------------------------
class TestCognitMacro:
    """Tests for the CognitMacro agent with mocked LLM."""

    @pytest.mark.asyncio
    async def test_process_returns_belief_object(self):
        """Agent process() returns a valid BeliefObject."""
        mock_llm = _make_mock_llm(VALID_MACRO_RESPONSE)
        agent = CognitMacro(llm_client=mock_llm)
        context = _make_context()

        result = await agent.process(context)

        assert isinstance(result, BeliefObject)
        assert result.agent_id == "COGNIT-MACRO"
        assert len(result.beliefs) == 1
        mock_llm.complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_multi_belief(self):
        """Agent handles multi-belief LLM responses."""
        mock_llm = _make_mock_llm(MULTI_MACRO_BELIEF_RESPONSE)
        agent = CognitMacro(llm_client=mock_llm)
        context = _make_context()

        result = await agent.process(context)

        assert len(result.beliefs) == 2

    @pytest.mark.asyncio
    async def test_process_malformed_raises(self):
        """Agent raises AgentProcessingError on malformed LLM response."""
        mock_llm = _make_mock_llm({"analysis": "Economy is complex"})
        agent = CognitMacro(llm_client=mock_llm)
        context = _make_context()

        with pytest.raises(AgentProcessingError):
            await agent.process(context)

    @pytest.mark.asyncio
    async def test_process_empty_beliefs_raises(self):
        """Agent raises AgentProcessingError when no beliefs parsed."""
        mock_llm = _make_mock_llm(MACRO_EMPTY_BELIEFS_RESPONSE)
        agent = CognitMacro(llm_client=mock_llm)
        context = _make_context()

        with pytest.raises(AgentProcessingError):
            await agent.process(context)

    @pytest.mark.asyncio
    async def test_process_vague_conditions_raises(self):
        """Agent raises when all beliefs are dropped due to vague conditions."""
        mock_llm = _make_mock_llm(MACRO_VAGUE_INVALIDATION_RESPONSE)
        agent = CognitMacro(llm_client=mock_llm)
        context = _make_context()

        with pytest.raises(AgentProcessingError):
            await agent.process(context)

    @pytest.mark.asyncio
    async def test_process_llm_error_raises(self):
        """Agent raises AgentProcessingError on LLM client error."""
        mock_llm = AsyncMock()
        mock_llm.complete = AsyncMock(side_effect=Exception("API down"))
        agent = CognitMacro(llm_client=mock_llm)
        context = _make_context()

        with pytest.raises(AgentProcessingError):
            await agent.process(context)

    @pytest.mark.asyncio
    async def test_context_window_hash_preserved(self):
        """BeliefObject carries the context_window_hash from input."""
        mock_llm = _make_mock_llm(VALID_MACRO_RESPONSE)
        agent = CognitMacro(llm_client=mock_llm)
        context = _make_context()

        result = await agent.process(context)

        assert result.context_window_hash == "macro_test_hash"

    @pytest.mark.asyncio
    async def test_content_hash_computed(self):
        """BeliefObject has a non-empty content hash."""
        mock_llm = _make_mock_llm(VALID_MACRO_RESPONSE)
        agent = CognitMacro(llm_client=mock_llm)
        context = _make_context()

        result = await agent.process(context)

        assert result.content_hash
        assert len(result.content_hash) == 64  # SHA-256 hex

    @pytest.mark.asyncio
    async def test_llm_receives_system_and_user_prompt(self):
        """LLM client receives both system and user prompt."""
        mock_llm = _make_mock_llm(VALID_MACRO_RESPONSE)
        agent = CognitMacro(llm_client=mock_llm)
        context = _make_context()

        await agent.process(context)

        call_args = mock_llm.complete.call_args
        assert "system_prompt" in call_args.kwargs or len(call_args.args) >= 1
        assert "user_prompt" in call_args.kwargs or len(call_args.args) >= 2

    @pytest.mark.asyncio
    async def test_partial_data_still_works(self):
        """Agent works with only a subset of data types present."""
        mock_llm = _make_mock_llm(VALID_MACRO_RESPONSE)
        agent = CognitMacro(llm_client=mock_llm)
        # Only yield curve — no CDS, no economic, no price
        context = _make_context(fragments=[_make_yield_curve_fragment()])

        result = await agent.process(context)

        assert isinstance(result, BeliefObject)
        assert len(result.beliefs) == 1


# ---------------------------------------------------------------------------
# Health status tests
# ---------------------------------------------------------------------------
class TestCognitMacroHealth:
    """Tests for agent health reporting."""

    def test_initial_health_is_healthy(self):
        """Agent starts in HEALTHY state."""
        agent = CognitMacro(llm_client=AsyncMock())
        health = agent.get_health()
        assert health.agent_id == "COGNIT-MACRO"
        assert health.status == AgentStatus.HEALTHY
        assert health.error_count_24h == 0

    @pytest.mark.asyncio
    async def test_health_after_success(self):
        """Health reports last_success after successful run."""
        mock_llm = _make_mock_llm(VALID_MACRO_RESPONSE)
        agent = CognitMacro(llm_client=mock_llm)
        context = _make_context()

        await agent.process(context)

        health = agent.get_health()
        assert health.last_run is not None
        assert health.last_success is not None
        assert health.status == AgentStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_health_degraded_after_errors(self):
        """Health degrades after repeated errors."""
        mock_llm = _make_mock_llm(MACRO_EMPTY_BELIEFS_RESPONSE)
        agent = CognitMacro(llm_client=mock_llm)
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
        mock_llm = _make_mock_llm(MACRO_EMPTY_BELIEFS_RESPONSE)
        agent = CognitMacro(llm_client=mock_llm)
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
class TestCognitMacroProperties:
    """Test agent identity properties."""

    def test_agent_id(self):
        agent = CognitMacro(llm_client=AsyncMock())
        assert agent.agent_id == "COGNIT-MACRO"

    def test_agent_type(self):
        agent = CognitMacro(llm_client=AsyncMock())
        assert agent.agent_type == "cognition"

    def test_version(self):
        agent = CognitMacro(llm_client=AsyncMock())
        assert agent.version == "1.0.0"

    def test_consumed_data_types(self):
        """Agent consumes the correct data types."""
        expected = {
            DataType.MACRO_YIELD_CURVE,
            DataType.MACRO_CDS,
            DataType.MACRO_ECONOMIC,
            DataType.PRICE_OHLCV,
        }
        assert CognitMacro.CONSUMED_DATA_TYPES == expected
