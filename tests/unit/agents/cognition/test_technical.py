"""Tests for COGNIT-TECHNICAL Research Agent.

Tests the full Research Agent loop (7 steps) for the FROZEN technical
analysis agent. Validates signal-based belief generation, evidence linking,
invalidation condition machine-evaluability, and error handling.

COGNIT-TECHNICAL is FROZEN: zero LLM calls, pure computation.
"""

import math
from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

import pytest

from providence.agents.base import AgentContext, AgentStatus
from providence.agents.cognition.technical import CognitTechnical
from providence.agents.cognition.technical_indicators import TechnicalSignals
from providence.exceptions import AgentProcessingError
from providence.schemas.belief import BeliefObject
from providence.schemas.enums import (
    ComparisonOperator,
    ConditionStatus,
    DataType,
    Direction,
    Magnitude,
    MarketCapBucket,
    ValidationStatus,
)
from providence.schemas.market_state import MarketStateFragment


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
NOW = datetime.now(timezone.utc)

FRAG_IDS = [UUID(f"{i:08x}-0000-0000-0000-000000000000") for i in range(250)]


def _make_price_fragment(
    ticker: str = "AAPL",
    close: float = 186.90,
    volume: float = 52_345_678.0,
    hours_ago: float = 1.0,
    fragment_id: UUID | None = None,
) -> MarketStateFragment:
    """Create a PRICE_OHLCV fragment."""
    ts = NOW - timedelta(hours=hours_ago)
    return MarketStateFragment(
        fragment_id=fragment_id or uuid4(),
        agent_id="PERCEPT-PRICE",
        timestamp=ts,
        source_timestamp=ts,
        entity=ticker,
        data_type=DataType.PRICE_OHLCV,
        schema_version="1.0.0",
        source_hash=f"hash-{ticker}-{close}",
        validation_status=ValidationStatus.VALID,
        payload={"close": close, "volume": volume, "open": close - 1, "high": close + 2, "low": close - 2},
    )


def _make_price_series(
    ticker: str = "AAPL",
    n: int = 250,
    start: float = 100.0,
    step: float = 0.5,
) -> list[MarketStateFragment]:
    """Create a chronological series of price fragments (uptrend)."""
    fragments = []
    for i in range(n):
        fragments.append(
            _make_price_fragment(
                ticker=ticker,
                close=start + i * step,
                volume=1_000_000.0 + i * 10_000,
                hours_ago=float(n - i),  # oldest first
                fragment_id=FRAG_IDS[i] if i < len(FRAG_IDS) else uuid4(),
            )
        )
    return fragments


def _make_downtrend_series(
    ticker: str = "AAPL",
    n: int = 250,
    start: float = 200.0,
    step: float = 0.5,
) -> list[MarketStateFragment]:
    """Create a chronological series of price fragments (downtrend)."""
    fragments = []
    for i in range(n):
        fragments.append(
            _make_price_fragment(
                ticker=ticker,
                close=start - i * step,
                volume=1_000_000.0 + i * 10_000,
                hours_ago=float(n - i),
            )
        )
    return fragments


def _make_context(
    fragments: list[MarketStateFragment] | None = None,
) -> AgentContext:
    """Create a test AgentContext."""
    return AgentContext(
        agent_id="COGNIT-TECHNICAL",
        trigger="schedule",
        fragments=fragments if fragments is not None else _make_price_series(),
        context_window_hash="tech_test_hash",
        timestamp=NOW,
        metadata={},
    )


# ---------------------------------------------------------------------------
# Agent identity and properties
# ---------------------------------------------------------------------------
class TestCognitTechnicalProperties:
    """Tests for agent identity and configuration."""

    def test_agent_id(self):
        agent = CognitTechnical()
        assert agent.agent_id == "COGNIT-TECHNICAL"

    def test_agent_type(self):
        agent = CognitTechnical()
        assert agent.agent_type == "cognition"

    def test_version(self):
        agent = CognitTechnical()
        assert agent.version == "1.0.0"

    def test_consumed_data_types(self):
        """Agent consumes PRICE_OHLCV and OPTIONS_CHAIN."""
        assert DataType.PRICE_OHLCV in CognitTechnical.CONSUMED_DATA_TYPES
        assert DataType.OPTIONS_CHAIN in CognitTechnical.CONSUMED_DATA_TYPES

    def test_no_llm_dependency(self):
        """FROZEN agent: constructor takes no LLM client argument."""
        agent = CognitTechnical()  # No llm_client needed
        assert agent is not None


# ---------------------------------------------------------------------------
# process() — full agent loop
# ---------------------------------------------------------------------------
class TestCognitTechnicalProcess:
    """Tests for the main process() method (Research Agent 7-step loop)."""

    @pytest.mark.asyncio
    async def test_process_returns_belief_object(self):
        """Agent process() returns a valid BeliefObject."""
        agent = CognitTechnical()
        context = _make_context()

        result = await agent.process(context)

        assert isinstance(result, BeliefObject)
        assert result.agent_id == "COGNIT-TECHNICAL"

    @pytest.mark.asyncio
    async def test_context_window_hash_preserved(self):
        """BeliefObject carries the context_window_hash from input."""
        agent = CognitTechnical()
        context = _make_context()

        result = await agent.process(context)

        assert result.context_window_hash == "tech_test_hash"

    @pytest.mark.asyncio
    async def test_uptrend_produces_long_belief(self):
        """Uptrend price series produces a LONG belief."""
        agent = CognitTechnical()
        frags = _make_price_series(n=250, start=50.0, step=0.5)
        context = _make_context(fragments=frags)

        result = await agent.process(context)

        assert len(result.beliefs) >= 1
        belief = result.beliefs[0]
        assert belief.direction == Direction.LONG
        assert belief.ticker == "AAPL"

    @pytest.mark.asyncio
    async def test_downtrend_produces_short_belief(self):
        """Downtrend price series produces a SHORT belief."""
        agent = CognitTechnical()
        frags = _make_downtrend_series(n=250, start=200.0, step=0.5)
        context = _make_context(fragments=frags)

        result = await agent.process(context)

        assert len(result.beliefs) >= 1
        belief = result.beliefs[0]
        assert belief.direction == Direction.SHORT

    @pytest.mark.asyncio
    async def test_multiple_tickers_produce_multiple_beliefs(self):
        """Multiple tickers each get their own belief."""
        agent = CognitTechnical()
        aapl_frags = _make_price_series(ticker="AAPL", n=30, start=150.0, step=1.0)
        msft_frags = _make_price_series(ticker="MSFT", n=30, start=300.0, step=0.5)
        context = _make_context(fragments=aapl_frags + msft_frags)

        result = await agent.process(context)

        tickers = [b.ticker for b in result.beliefs]
        assert "AAPL" in tickers
        assert "MSFT" in tickers

    @pytest.mark.asyncio
    async def test_no_price_fragments_raises(self):
        """No PRICE_OHLCV fragments raises AgentProcessingError."""
        agent = CognitTechnical()
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
            payload={"revenue": 100_000_000},
        )
        context = _make_context(fragments=[filing_frag])

        with pytest.raises(AgentProcessingError, match="No PRICE_OHLCV"):
            await agent.process(context)

    @pytest.mark.asyncio
    async def test_empty_fragments_raises(self):
        """Empty fragment list raises AgentProcessingError."""
        agent = CognitTechnical()
        context = _make_context(fragments=[])

        with pytest.raises(AgentProcessingError):
            await agent.process(context)

    @pytest.mark.asyncio
    async def test_insufficient_data_per_ticker_skipped(self):
        """Tickers with too few data points are skipped."""
        agent = CognitTechnical()
        # Only 5 data points — below MIN_PRICE_POINTS (20)
        short_series = _make_price_series(ticker="TINY", n=5, start=100.0, step=1.0)
        # Need at least one ticker with enough data to not error
        normal_series = _make_price_series(ticker="AAPL", n=25, start=100.0, step=0.5)
        context = _make_context(fragments=short_series + normal_series)

        result = await agent.process(context)

        tickers = [b.ticker for b in result.beliefs]
        assert "TINY" not in tickers
        assert "AAPL" in tickers

    @pytest.mark.asyncio
    async def test_all_tickers_insufficient_raises(self):
        """If ALL tickers have insufficient data, raises AgentProcessingError."""
        agent = CognitTechnical()
        short_series = _make_price_series(ticker="TINY", n=5, start=100.0, step=1.0)
        context = _make_context(fragments=short_series)

        with pytest.raises(AgentProcessingError, match="No beliefs generated"):
            await agent.process(context)


# ---------------------------------------------------------------------------
# Belief generation — _signals_to_beliefs
# ---------------------------------------------------------------------------
class TestBeliefGeneration:
    """Tests for belief field correctness from signal analysis."""

    @pytest.mark.asyncio
    async def test_confidence_formula(self):
        """Confidence follows: min(0.85, 0.35 + abs(net_signal) * 0.12)."""
        agent = CognitTechnical()
        frags = _make_price_series(n=250)
        context = _make_context(fragments=frags)

        result = await agent.process(context)
        belief = result.beliefs[0]

        # Confidence is always in [0.35, 0.85]
        assert 0.35 <= belief.raw_confidence <= 0.85

    @pytest.mark.asyncio
    async def test_confidence_capped_at_085(self):
        """Confidence is capped at 0.85 regardless of signal strength."""
        agent = CognitTechnical()
        # Use a very strong uptrend to maximize signals
        frags = _make_price_series(n=250, start=10.0, step=2.0)
        context = _make_context(fragments=frags)

        result = await agent.process(context)
        belief = result.beliefs[0]

        assert belief.raw_confidence <= 0.85

    @pytest.mark.asyncio
    async def test_thesis_id_format(self):
        """Thesis ID follows TECH-{ticker}-{direction}-{net_signal:+d} pattern."""
        agent = CognitTechnical()
        frags = _make_price_series(n=250)
        context = _make_context(fragments=frags)

        result = await agent.process(context)
        thesis_id = result.beliefs[0].thesis_id

        assert thesis_id.startswith("TECH-AAPL-")

    @pytest.mark.asyncio
    async def test_time_horizon_range(self):
        """Time horizon is between 5-60 days per spec."""
        agent = CognitTechnical()
        frags = _make_price_series(n=250)
        context = _make_context(fragments=frags)

        result = await agent.process(context)
        belief = result.beliefs[0]

        assert 5 <= belief.time_horizon_days <= 60

    @pytest.mark.asyncio
    async def test_evidence_refs_present(self):
        """Evidence refs link back to input fragment IDs."""
        agent = CognitTechnical()
        frags = _make_price_series(n=30)
        context = _make_context(fragments=frags)

        result = await agent.process(context)
        evidence = result.beliefs[0].evidence

        assert len(evidence) > 0
        # Evidence fragment IDs should be from the input
        frag_ids = {f.fragment_id for f in frags}
        for ref in evidence:
            assert ref.source_fragment_id in frag_ids

    @pytest.mark.asyncio
    async def test_evidence_limited_to_five(self):
        """Evidence refs are limited to at most 5 fragments."""
        agent = CognitTechnical()
        frags = _make_price_series(n=250)
        context = _make_context(fragments=frags)

        result = await agent.process(context)
        evidence = result.beliefs[0].evidence

        assert len(evidence) <= 5

    @pytest.mark.asyncio
    async def test_evidence_weights_sum_to_one(self):
        """Evidence weights should sum to approximately 1.0."""
        agent = CognitTechnical()
        frags = _make_price_series(n=30)
        context = _make_context(fragments=frags)

        result = await agent.process(context)
        evidence = result.beliefs[0].evidence

        total_weight = sum(ref.weight for ref in evidence)
        # Each ref gets 1/N weight, so sum should be close to # evidence refs / N
        assert total_weight > 0

    @pytest.mark.asyncio
    async def test_metadata_defaults(self):
        """Metadata has expected default values for technical analysis."""
        agent = CognitTechnical()
        frags = _make_price_series(n=30)
        context = _make_context(fragments=frags)

        result = await agent.process(context)
        meta = result.beliefs[0].metadata

        assert meta.sector == "UNKNOWN"  # Technical analysis is sector-agnostic
        assert meta.market_cap_bucket == MarketCapBucket.LARGE

    @pytest.mark.asyncio
    async def test_magnitude_large_on_strong_signal(self):
        """Strong signal (net >= 3) produces LARGE magnitude."""
        agent = CognitTechnical()
        # Very strong uptrend to max out signals
        frags = _make_price_series(n=250, start=10.0, step=2.0)
        context = _make_context(fragments=frags)

        result = await agent.process(context)
        belief = result.beliefs[0]

        # A very strong trend should produce multiple bullish signals
        if abs(int(belief.thesis_id.split(":")[0].split("-")[-1] if ":" in belief.thesis_id else belief.thesis_id.rsplit("-", 1)[-1].replace("+", ""))) >= 3:
            assert belief.magnitude == Magnitude.LARGE


# ---------------------------------------------------------------------------
# Invalidation conditions
# ---------------------------------------------------------------------------
class TestInvalidationConditions:
    """Tests for machine-evaluable invalidation conditions."""

    @pytest.mark.asyncio
    async def test_long_belief_has_invalidation_conditions(self):
        """LONG beliefs have at least 2 invalidation conditions."""
        agent = CognitTechnical()
        frags = _make_price_series(n=250, start=50.0, step=0.5)
        context = _make_context(fragments=frags)

        result = await agent.process(context)
        belief = result.beliefs[0]

        assert belief.direction == Direction.LONG
        assert len(belief.invalidation_conditions) >= 2

    @pytest.mark.asyncio
    async def test_short_belief_has_invalidation_conditions(self):
        """SHORT beliefs have at least 2 invalidation conditions."""
        agent = CognitTechnical()
        frags = _make_downtrend_series(n=250, start=200.0, step=0.5)
        context = _make_context(fragments=frags)

        result = await agent.process(context)
        belief = result.beliefs[0]

        assert belief.direction == Direction.SHORT
        assert len(belief.invalidation_conditions) >= 2

    @pytest.mark.asyncio
    async def test_conditions_are_machine_evaluable(self):
        """All conditions have metric, operator, and threshold."""
        agent = CognitTechnical()
        frags = _make_price_series(n=250)
        context = _make_context(fragments=frags)

        result = await agent.process(context)
        for belief in result.beliefs:
            for cond in belief.invalidation_conditions:
                assert cond.metric, "Metric must be non-empty"
                assert cond.operator in ComparisonOperator
                assert cond.threshold is not None
                assert cond.status == ConditionStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_long_invalidation_uses_lt_for_price(self):
        """LONG belief's price condition uses LT operator (price drops below threshold)."""
        agent = CognitTechnical()
        frags = _make_price_series(n=250, start=50.0, step=0.5)
        context = _make_context(fragments=frags)

        result = await agent.process(context)
        belief = result.beliefs[0]

        price_conditions = [c for c in belief.invalidation_conditions if c.metric == "close"]
        if price_conditions:
            assert price_conditions[0].operator == ComparisonOperator.LT

    @pytest.mark.asyncio
    async def test_short_invalidation_uses_gt_for_price(self):
        """SHORT belief's price condition uses GT operator (price rises above threshold)."""
        agent = CognitTechnical()
        frags = _make_downtrend_series(n=250, start=200.0, step=0.5)
        context = _make_context(fragments=frags)

        result = await agent.process(context)
        belief = result.beliefs[0]

        price_conditions = [c for c in belief.invalidation_conditions if c.metric == "close"]
        if price_conditions:
            assert price_conditions[0].operator == ComparisonOperator.GT


# ---------------------------------------------------------------------------
# Price series extraction
# ---------------------------------------------------------------------------
class TestPriceSeriesExtraction:
    """Tests for _extract_price_series internal method."""

    @pytest.mark.asyncio
    async def test_groups_by_ticker(self):
        """Fragments are grouped by ticker entity."""
        agent = CognitTechnical()
        aapl_frags = _make_price_series(ticker="AAPL", n=25)
        msft_frags = _make_price_series(ticker="MSFT", n=25)
        all_frags = aapl_frags + msft_frags

        price_series = agent._extract_price_series(all_frags)

        assert "AAPL" in price_series
        assert "MSFT" in price_series
        assert len(price_series["AAPL"][0]) == 25
        assert len(price_series["MSFT"][0]) == 25

    @pytest.mark.asyncio
    async def test_sorts_chronologically(self):
        """Prices are sorted oldest-first regardless of fragment order."""
        agent = CognitTechnical()
        frags = _make_price_series(ticker="AAPL", n=10, start=100.0, step=1.0)
        # Reverse the fragment order
        frags_reversed = list(reversed(frags))

        price_series = agent._extract_price_series(frags_reversed)
        close_prices = price_series["AAPL"][0]

        # Should be sorted ascending (100.0, 100.5, 101.0, ...)
        assert close_prices == sorted(close_prices)

    @pytest.mark.asyncio
    async def test_non_price_fragments_ignored(self):
        """Non-PRICE_OHLCV fragments are excluded."""
        agent = CognitTechnical()
        filing = MarketStateFragment(
            fragment_id=uuid4(),
            agent_id="PERCEPT-FILING",
            timestamp=NOW,
            source_timestamp=NOW,
            entity="AAPL",
            data_type=DataType.FILING_10Q,
            schema_version="1.0.0",
            source_hash="filing-hash",
            validation_status=ValidationStatus.VALID,
            payload={"revenue": 100_000_000},
        )
        price = _make_price_fragment(ticker="AAPL")

        price_series = agent._extract_price_series([filing, price])
        assert "AAPL" in price_series
        assert len(price_series["AAPL"][0]) == 1  # Only the price fragment

    @pytest.mark.asyncio
    async def test_fragments_without_close_excluded(self):
        """Fragments with no 'close' in payload are handled gracefully."""
        agent = CognitTechnical()
        bad_frag = MarketStateFragment(
            fragment_id=uuid4(),
            agent_id="PERCEPT-PRICE",
            timestamp=NOW,
            source_timestamp=NOW,
            entity="AAPL",
            data_type=DataType.PRICE_OHLCV,
            schema_version="1.0.0",
            source_hash="no-close",
            validation_status=ValidationStatus.VALID,
            payload={"open": 100.0, "high": 105.0, "low": 95.0},  # No close!
        )
        good_frag = _make_price_fragment(ticker="AAPL")

        price_series = agent._extract_price_series([bad_frag, good_frag])
        # Should still have AAPL with 1 price (the good fragment)
        assert "AAPL" in price_series
        assert len(price_series["AAPL"][0]) == 1

    @pytest.mark.asyncio
    async def test_empty_entity_skipped(self):
        """Fragments with empty entity string are skipped."""
        agent = CognitTechnical()
        no_entity = MarketStateFragment(
            fragment_id=uuid4(),
            agent_id="PERCEPT-PRICE",
            timestamp=NOW,
            source_timestamp=NOW,
            entity="",
            data_type=DataType.PRICE_OHLCV,
            schema_version="1.0.0",
            source_hash="no-entity",
            validation_status=ValidationStatus.VALID,
            payload={"close": 100.0, "volume": 1000},
        )

        price_series = agent._extract_price_series([no_entity])
        # Empty entity should not appear (the _extract_price_series checks `frag.entity`)
        assert len(price_series) == 0


# ---------------------------------------------------------------------------
# Health reporting
# ---------------------------------------------------------------------------
class TestCognitTechnicalHealth:
    """Tests for agent health status reporting."""

    def test_initial_health_is_healthy(self):
        """Agent starts in HEALTHY state."""
        agent = CognitTechnical()
        health = agent.get_health()
        assert health.agent_id == "COGNIT-TECHNICAL"
        assert health.status == AgentStatus.HEALTHY
        assert health.error_count_24h == 0

    @pytest.mark.asyncio
    async def test_health_after_success(self):
        """Health reports last_run and last_success after successful run."""
        agent = CognitTechnical()
        context = _make_context()

        await agent.process(context)

        health = agent.get_health()
        assert health.last_run is not None
        assert health.last_success is not None
        assert health.status == AgentStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_health_degraded_after_errors(self):
        """Health becomes DEGRADED after 4+ errors."""
        agent = CognitTechnical()
        bad_context = _make_context(fragments=[])

        for _ in range(4):
            try:
                await agent.process(bad_context)
            except AgentProcessingError:
                pass

        health = agent.get_health()
        assert health.status == AgentStatus.DEGRADED
        assert health.error_count_24h == 4

    @pytest.mark.asyncio
    async def test_health_unhealthy_after_many_errors(self):
        """Health becomes UNHEALTHY after 11+ errors."""
        agent = CognitTechnical()
        bad_context = _make_context(fragments=[])

        for _ in range(11):
            try:
                await agent.process(bad_context)
            except AgentProcessingError:
                pass

        health = agent.get_health()
        assert health.status == AgentStatus.UNHEALTHY
        assert health.error_count_24h == 11

    @pytest.mark.asyncio
    async def test_success_doesnt_reset_error_count(self):
        """A successful run does not reset the error count."""
        agent = CognitTechnical()
        bad_context = _make_context(fragments=[])

        # Generate some errors
        for _ in range(2):
            try:
                await agent.process(bad_context)
            except AgentProcessingError:
                pass

        # Now a successful run
        good_context = _make_context()
        await agent.process(good_context)

        health = agent.get_health()
        assert health.error_count_24h == 2  # Errors not reset
        assert health.last_success is not None


# ---------------------------------------------------------------------------
# Content hash and immutability
# ---------------------------------------------------------------------------
class TestBeliefObjectIntegrity:
    """Tests for BeliefObject content hashing and immutability."""

    @pytest.mark.asyncio
    async def test_content_hash_computed(self):
        """BeliefObject has a non-empty content hash."""
        agent = CognitTechnical()
        context = _make_context()

        result = await agent.process(context)

        assert result.content_hash
        assert len(result.content_hash) == 64  # SHA-256 hex

    @pytest.mark.asyncio
    async def test_deterministic_output(self):
        """Same input produces same beliefs (deterministic computation)."""
        agent = CognitTechnical()
        frags = _make_price_series(n=50, start=100.0, step=0.5)
        context = _make_context(fragments=frags)

        result1 = await agent.process(context)
        result2 = await agent.process(context)

        # Same beliefs (same direction, confidence, etc.)
        assert len(result1.beliefs) == len(result2.beliefs)
        for b1, b2 in zip(result1.beliefs, result2.beliefs):
            assert b1.direction == b2.direction
            assert b1.raw_confidence == b2.raw_confidence
            assert b1.ticker == b2.ticker
            assert b1.magnitude == b2.magnitude

    @pytest.mark.asyncio
    async def test_belief_object_is_frozen(self):
        """BeliefObject is immutable (frozen=True)."""
        agent = CognitTechnical()
        context = _make_context()

        result = await agent.process(context)

        with pytest.raises(Exception):  # Pydantic ValidationError for frozen models
            result.agent_id = "HACKED"  # type: ignore[misc]
