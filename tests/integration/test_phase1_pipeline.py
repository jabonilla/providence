"""End-to-end integration test for Phase 1 pipeline.

Validates the full pipeline:
  PERCEPT-PRICE → MarketStateFragment
  PERCEPT-FILING → MarketStateFragment
  CONTEXT-SVC → AgentContext
  COGNIT-FUNDAMENTAL → BeliefObject

Tests 5 stocks: AAPL, NVDA, JPM, JNJ, TSLA
Uses mocked LLM (no real API calls) with realistic sample responses.

Spec Reference: Technical Spec v2.3, Phase 1 (Sessions 1-7)
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock
from uuid import UUID

import pytest

from providence.agents.base import AgentContext
from providence.agents.cognition.fundamental import CognitFundamental
from providence.agents.cognition.response_parser import parse_llm_response
from providence.config.agent_config import AgentConfigRegistry
from providence.schemas.belief import BeliefObject
from providence.schemas.enums import (
    ComparisonOperator,
    DataType,
    Direction,
    Magnitude,
    ValidationStatus,
)
from providence.schemas.market_state import MarketStateFragment
from providence.services.context_svc import ContextService
from providence.utils.hashing import compute_context_window_hash

from tests.fixtures.sample_beliefs import BELIEF_RESPONSES_BY_TICKER
from tests.fixtures.sample_fragments import (
    ALL_FRAGMENTS,
    FRAGMENTS_BY_TICKER,
    FRAGMENT_ID_MAP,
)


# ---------------------------------------------------------------------------
# Test config
# ---------------------------------------------------------------------------
TICKERS = ["AAPL", "NVDA", "JPM", "JNJ", "TSLA"]

AGENT_CONFIG = {
    "COGNIT-FUNDAMENTAL": {
        "consumes": [
            DataType.PRICE_OHLCV,
            DataType.FILING_10K,
            DataType.FILING_10Q,
            DataType.FILING_8K,
        ],
        "max_token_budget": 100_000,
        "entity_scope": [],
        "peer_count": 0,
        "priority_window_hours": 72,
    }
}


def _make_registry() -> AgentConfigRegistry:
    return AgentConfigRegistry.from_dict(AGENT_CONFIG)


def _make_context_svc() -> ContextService:
    return ContextService(_make_registry())


def _make_mock_llm(ticker: str) -> AsyncMock:
    """Create mock LLM that returns the golden response for a ticker."""
    mock = AsyncMock()
    mock.complete = AsyncMock(return_value=BELIEF_RESPONSES_BY_TICKER[ticker])
    return mock


# ---------------------------------------------------------------------------
# Pipeline step tests
# ---------------------------------------------------------------------------
class TestPerceptionFragments:
    """Validate that sample fragments are well-formed."""

    @pytest.mark.parametrize("ticker", TICKERS)
    def test_fragments_are_valid(self, ticker: str):
        """Each ticker's fragments have valid schema and status."""
        fragments = FRAGMENTS_BY_TICKER[ticker]
        assert len(fragments) >= 2  # at least price + filing

        for frag in fragments:
            assert isinstance(frag, MarketStateFragment)
            assert frag.validation_status == ValidationStatus.VALID
            assert frag.entity == ticker
            assert frag.version  # content hash computed

    @pytest.mark.parametrize("ticker", TICKERS)
    def test_fragment_content_hashes_unique(self, ticker: str):
        """Each fragment has a unique content hash."""
        fragments = FRAGMENTS_BY_TICKER[ticker]
        hashes = [f.version for f in fragments]
        assert len(hashes) == len(set(hashes))

    def test_all_fragments_count(self):
        """We have 10 total fragments across 5 stocks."""
        assert len(ALL_FRAGMENTS) == 10

    @pytest.mark.parametrize("ticker", TICKERS)
    def test_fragments_have_price_data(self, ticker: str):
        """Each ticker has at least one PRICE_OHLCV fragment."""
        fragments = FRAGMENTS_BY_TICKER[ticker]
        price_frags = [f for f in fragments if f.data_type == DataType.PRICE_OHLCV]
        assert len(price_frags) >= 1

    @pytest.mark.parametrize("ticker", TICKERS)
    def test_fragments_have_filing_data(self, ticker: str):
        """Each ticker has at least one filing fragment."""
        fragments = FRAGMENTS_BY_TICKER[ticker]
        filing_types = {DataType.FILING_10K, DataType.FILING_10Q, DataType.FILING_8K}
        filing_frags = [f for f in fragments if f.data_type in filing_types]
        assert len(filing_frags) >= 1


class TestContextAssembly:
    """Validate CONTEXT-SVC assembles correct contexts."""

    @pytest.mark.parametrize("ticker", TICKERS)
    def test_context_assembly_per_ticker(self, ticker: str):
        """CONTEXT-SVC produces valid AgentContext for each ticker."""
        svc = _make_context_svc()
        fragments = FRAGMENTS_BY_TICKER[ticker]

        ctx = svc.assemble_context(
            agent_id="COGNIT-FUNDAMENTAL",
            trigger="schedule",
            available_fragments=fragments,
        )

        assert isinstance(ctx, AgentContext)
        assert ctx.agent_id == "COGNIT-FUNDAMENTAL"
        assert len(ctx.fragments) > 0
        assert ctx.context_window_hash

    def test_context_hash_deterministic(self):
        """Same fragments produce same context_window_hash."""
        svc = _make_context_svc()
        fragments = FRAGMENTS_BY_TICKER["AAPL"]

        ctx1 = svc.assemble_context("COGNIT-FUNDAMENTAL", "schedule", fragments)
        ctx2 = svc.assemble_context("COGNIT-FUNDAMENTAL", "schedule", fragments)

        assert ctx1.context_window_hash == ctx2.context_window_hash

    def test_context_includes_all_data_types(self):
        """Context includes both price and filing fragments."""
        svc = _make_context_svc()
        fragments = FRAGMENTS_BY_TICKER["AAPL"]

        ctx = svc.assemble_context("COGNIT-FUNDAMENTAL", "schedule", fragments)

        data_types = {f.data_type for f in ctx.fragments}
        assert DataType.PRICE_OHLCV in data_types
        assert DataType.FILING_10Q in data_types

    def test_context_with_all_fragments(self):
        """Context built from all 10 fragments works correctly."""
        svc = _make_context_svc()

        ctx = svc.assemble_context(
            agent_id="COGNIT-FUNDAMENTAL",
            trigger="schedule",
            available_fragments=ALL_FRAGMENTS,
        )

        assert len(ctx.fragments) == 10
        assert ctx.context_window_hash


class TestCognitFundamentalPipeline:
    """Validate COGNIT-FUNDAMENTAL produces valid BeliefObjects."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("ticker", TICKERS)
    async def test_full_pipeline_per_ticker(self, ticker: str):
        """Full pipeline: fragments → context → COGNIT-FUNDAMENTAL → BeliefObject."""
        svc = _make_context_svc()
        fragments = FRAGMENTS_BY_TICKER[ticker]

        # Step 1: Assemble context
        ctx = svc.assemble_context(
            agent_id="COGNIT-FUNDAMENTAL",
            trigger="schedule",
            available_fragments=fragments,
        )

        # Step 2: Run agent (mocked LLM)
        mock_llm = _make_mock_llm(ticker)
        agent = CognitFundamental(llm_client=mock_llm)
        belief_obj = await agent.process(ctx)

        # Step 3: Validate output
        assert isinstance(belief_obj, BeliefObject)
        assert belief_obj.agent_id == "COGNIT-FUNDAMENTAL"
        assert belief_obj.context_window_hash == ctx.context_window_hash

    @pytest.mark.asyncio
    @pytest.mark.parametrize("ticker", TICKERS)
    async def test_belief_has_valid_direction(self, ticker: str):
        """Each belief has a valid direction enum."""
        svc = _make_context_svc()
        ctx = svc.assemble_context(
            "COGNIT-FUNDAMENTAL", "schedule", FRAGMENTS_BY_TICKER[ticker]
        )
        agent = CognitFundamental(llm_client=_make_mock_llm(ticker))
        belief_obj = await agent.process(ctx)

        for belief in belief_obj.beliefs:
            assert belief.direction in (Direction.LONG, Direction.SHORT, Direction.NEUTRAL)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("ticker", TICKERS)
    async def test_belief_has_valid_magnitude(self, ticker: str):
        """Each belief has a valid magnitude enum."""
        svc = _make_context_svc()
        ctx = svc.assemble_context(
            "COGNIT-FUNDAMENTAL", "schedule", FRAGMENTS_BY_TICKER[ticker]
        )
        agent = CognitFundamental(llm_client=_make_mock_llm(ticker))
        belief_obj = await agent.process(ctx)

        for belief in belief_obj.beliefs:
            assert belief.magnitude in (Magnitude.SMALL, Magnitude.MODERATE, Magnitude.LARGE)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("ticker", TICKERS)
    async def test_confidence_is_reasonable(self, ticker: str):
        """Confidence scores are differentiated (not all 0.5 or all 0.95)."""
        svc = _make_context_svc()
        ctx = svc.assemble_context(
            "COGNIT-FUNDAMENTAL", "schedule", FRAGMENTS_BY_TICKER[ticker]
        )
        agent = CognitFundamental(llm_client=_make_mock_llm(ticker))
        belief_obj = await agent.process(ctx)

        for belief in belief_obj.beliefs:
            assert 0.0 <= belief.raw_confidence <= 1.0
            # Should not be the degenerate extremes
            assert belief.raw_confidence != 0.0
            assert belief.raw_confidence != 1.0

    @pytest.mark.asyncio
    @pytest.mark.parametrize("ticker", TICKERS)
    async def test_has_machine_evaluable_invalidation_conditions(self, ticker: str):
        """Each belief has at least 2 machine-evaluable invalidation conditions."""
        svc = _make_context_svc()
        ctx = svc.assemble_context(
            "COGNIT-FUNDAMENTAL", "schedule", FRAGMENTS_BY_TICKER[ticker]
        )
        agent = CognitFundamental(llm_client=_make_mock_llm(ticker))
        belief_obj = await agent.process(ctx)

        for belief in belief_obj.beliefs:
            conditions = belief.invalidation_conditions
            assert len(conditions) >= 2, (
                f"Belief {belief.thesis_id} has {len(conditions)} invalidation "
                f"conditions, expected at least 2"
            )

            for cond in conditions:
                assert cond.metric, f"Condition {cond.condition_id} missing metric"
                assert cond.operator in ComparisonOperator
                assert cond.threshold is not None

    @pytest.mark.asyncio
    @pytest.mark.parametrize("ticker", TICKERS)
    async def test_evidence_refs_point_to_context_fragments(self, ticker: str):
        """Evidence refs should reference fragment IDs that exist in context."""
        svc = _make_context_svc()
        ctx = svc.assemble_context(
            "COGNIT-FUNDAMENTAL", "schedule", FRAGMENTS_BY_TICKER[ticker]
        )
        context_fragment_ids = {f.fragment_id for f in ctx.fragments}

        agent = CognitFundamental(llm_client=_make_mock_llm(ticker))
        belief_obj = await agent.process(ctx)

        for belief in belief_obj.beliefs:
            for ref in belief.evidence:
                assert ref.source_fragment_id in context_fragment_ids, (
                    f"Evidence ref {ref.source_fragment_id} not found in "
                    f"context fragments for {ticker}"
                )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("ticker", TICKERS)
    async def test_time_horizon_in_range(self, ticker: str):
        """Time horizon is within 30-180 days per spec."""
        svc = _make_context_svc()
        ctx = svc.assemble_context(
            "COGNIT-FUNDAMENTAL", "schedule", FRAGMENTS_BY_TICKER[ticker]
        )
        agent = CognitFundamental(llm_client=_make_mock_llm(ticker))
        belief_obj = await agent.process(ctx)

        for belief in belief_obj.beliefs:
            assert 30 <= belief.time_horizon_days <= 180

    @pytest.mark.asyncio
    @pytest.mark.parametrize("ticker", TICKERS)
    async def test_belief_has_content_hash(self, ticker: str):
        """BeliefObject has a computed content_hash."""
        svc = _make_context_svc()
        ctx = svc.assemble_context(
            "COGNIT-FUNDAMENTAL", "schedule", FRAGMENTS_BY_TICKER[ticker]
        )
        agent = CognitFundamental(llm_client=_make_mock_llm(ticker))
        belief_obj = await agent.process(ctx)

        assert belief_obj.content_hash
        assert len(belief_obj.content_hash) == 64  # SHA-256 hex


class TestConfidenceDiversity:
    """Validate confidence scores are differentiated across stocks."""

    @pytest.mark.asyncio
    async def test_confidence_scores_vary_across_tickers(self):
        """Confidence scores across 5 stocks are not all the same."""
        svc = _make_context_svc()
        confidences = []

        for ticker in TICKERS:
            ctx = svc.assemble_context(
                "COGNIT-FUNDAMENTAL", "schedule", FRAGMENTS_BY_TICKER[ticker]
            )
            agent = CognitFundamental(llm_client=_make_mock_llm(ticker))
            belief_obj = await agent.process(ctx)

            for belief in belief_obj.beliefs:
                confidences.append(belief.raw_confidence)

        # Should have at least 3 distinct confidence values
        unique_confidences = set(confidences)
        assert len(unique_confidences) >= 3, (
            f"Only {len(unique_confidences)} distinct confidence values: {unique_confidences}"
        )

    @pytest.mark.asyncio
    async def test_directions_vary_across_tickers(self):
        """Not all beliefs should have the same direction."""
        svc = _make_context_svc()
        directions = []

        for ticker in TICKERS:
            ctx = svc.assemble_context(
                "COGNIT-FUNDAMENTAL", "schedule", FRAGMENTS_BY_TICKER[ticker]
            )
            agent = CognitFundamental(llm_client=_make_mock_llm(ticker))
            belief_obj = await agent.process(ctx)

            for belief in belief_obj.beliefs:
                directions.append(belief.direction)

        unique_directions = set(directions)
        assert len(unique_directions) >= 2, (
            f"Only {len(unique_directions)} direction: {unique_directions}"
        )


class TestContextHashReproducibility:
    """Validate context_window_hash guarantees input reproducibility."""

    @pytest.mark.asyncio
    async def test_same_input_same_context_hash(self):
        """Running twice with same fragments produces same context hash."""
        svc = _make_context_svc()

        for ticker in TICKERS:
            fragments = FRAGMENTS_BY_TICKER[ticker]
            ctx1 = svc.assemble_context("COGNIT-FUNDAMENTAL", "schedule", fragments)
            ctx2 = svc.assemble_context("COGNIT-FUNDAMENTAL", "schedule", fragments)
            assert ctx1.context_window_hash == ctx2.context_window_hash

    @pytest.mark.asyncio
    async def test_different_tickers_different_hashes(self):
        """Different tickers produce different context hashes."""
        svc = _make_context_svc()
        hashes = set()

        for ticker in TICKERS:
            ctx = svc.assemble_context(
                "COGNIT-FUNDAMENTAL", "schedule", FRAGMENTS_BY_TICKER[ticker]
            )
            hashes.add(ctx.context_window_hash)

        assert len(hashes) == 5  # All 5 tickers should produce different hashes
