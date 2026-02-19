"""COGNIT-CROSSSEC: Cross-Sectional Peer Comparison Research Agent.

Analyzes quarterly financials, price action, and earnings calls across
a company and its sector peers to detect relative value dislocations,
fundamental divergences, and sector rotation signals.

Spec Reference: Technical Spec v2.3, Section 4.2

Classification: ADAPTIVE — uses Claude Sonnet 4 via Anthropic API.
Subject to offline retraining. Prompt template is version-controlled.

Thesis types:
  1. Relative valuation dislocation — P/E, EV/EBITDA vs peers
  2. Fundamental divergence — margin/growth trends diverging from peers
  3. Sector rotation signal — capital flowing between subsectors
  4. Earnings quality spread — quality gap widening vs peers
  5. Mean reversion opportunity — peer spread at historical extremes

Time horizon: 30-120 days

Research Agent Common Loop:
  1. RECEIVE CONTEXT  → AgentContext from CONTEXT-SVC
  2. ANALYZE          → Send context + prompt to LLM
  3. HYPOTHESIZE      → Parse LLM response into thesis with direction + magnitude
  4. EVIDENCE LINK    → Attach source fragment references
  5. SCORE            → Extract raw_confidence from LLM output
  6. INVALIDATE       → Extract machine-readable invalidation conditions
  7. EMIT             → Return validated BeliefObject
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import structlog
import yaml

from providence.agents.base import AgentContext, AgentStatus, BaseAgent, HealthStatus
from providence.agents.cognition.response_parser import parse_llm_response
from providence.exceptions import AgentProcessingError
from providence.infra.llm_client import AnthropicClient, LLMClient
from providence.schemas.belief import BeliefObject
from providence.schemas.enums import DataType
from providence.schemas.market_state import MarketStateFragment

logger = structlog.get_logger()

# Path to prompt templates
PROMPT_DIR = Path(__file__).parent.parent.parent / "prompts"
DEFAULT_PROMPT_VERSION = "cognit_crosssec_v1.0.yaml"


class CognitCrossSec(BaseAgent[BeliefObject]):
    """Cross-sectional peer comparison Research Agent.

    Consumes FILING_10Q, PRICE_OHLCV, and EARNINGS_CALL
    MarketStateFragments.

    Produces BeliefObjects with investment theses driven by cross-sectional
    analysis — relative valuation dislocations, fundamental divergences,
    sector rotation signals, earnings quality spreads, and mean reversion
    opportunities.

    Like COGNIT-FUNDAMENTAL and COGNIT-NARRATIVE, this agent identifies
    a primary ticker and separates peer data into a dedicated section for
    explicit cross-sectional comparison.
    """

    CONSUMED_DATA_TYPES = {
        DataType.FILING_10Q,
        DataType.PRICE_OHLCV,
        DataType.EARNINGS_CALL,
    }

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        prompt_version: str = DEFAULT_PROMPT_VERSION,
    ) -> None:
        super().__init__(
            agent_id="COGNIT-CROSSSEC",
            agent_type="cognition",
            version="1.0.0",
        )
        self._llm_client = llm_client or AnthropicClient()
        self._prompt_config = self._load_prompt(prompt_version)
        self._last_run: Optional[datetime] = None
        self._last_success: Optional[datetime] = None
        self._error_count_24h: int = 0

    def _load_prompt(self, version: str) -> dict[str, Any]:
        """Load prompt template from YAML."""
        path = PROMPT_DIR / version
        if not path.exists():
            logger.warning(
                "Prompt file not found, using defaults",
                path=str(path),
            )
            return {
                "system_prompt": "You are a cross-sectional equity analyst.",
                "user_prompt_template": (
                    "Analyze the cross-sectional data for {ticker}.\n"
                    "Quarterly Financials:\n{quarterly_data}\n"
                    "Price Action:\n{price_data}\n"
                    "Earnings Calls:\n{earnings_data}\n"
                    "Peer Data:\n{peer_data}\n"
                    "Fragment IDs:\n{fragment_ids}"
                ),
            }

        with open(path) as f:
            return yaml.safe_load(f)

    async def process(self, context: AgentContext) -> BeliefObject:
        """Execute the Research Agent common loop.

        Steps:
          1. RECEIVE CONTEXT  → Validated by type
          2. ANALYZE          → Build prompt from cross-sectional context, send to LLM
          3. HYPOTHESIZE      → Parse LLM response into beliefs
          4. EVIDENCE LINK    → Included in LLM response parsing
          5. SCORE            → Included in LLM response parsing
          6. INVALIDATE       → Included in LLM response parsing
          7. EMIT             → Return validated BeliefObject

        Args:
            context: AgentContext assembled by CONTEXT-SVC.

        Returns:
            BeliefObject with cross-sectional investment theses.

        Raises:
            AgentProcessingError: If processing fails.
        """
        self._last_run = datetime.now(timezone.utc)

        try:
            # Step 1: RECEIVE CONTEXT — already validated by caller
            log = logger.bind(
                agent_id=self.agent_id,
                context_hash=context.context_window_hash,
                fragment_count=len(context.fragments),
            )
            log.info("Starting cross-sectional analysis")

            # Step 2: ANALYZE — build prompts and call LLM
            system_prompt = self._prompt_config.get(
                "system_prompt",
                "You are a cross-sectional equity analyst.",
            )
            user_prompt = self._build_user_prompt(context)

            log.info("Sending cross-sectional context to LLM")
            raw_response = await self._llm_client.complete(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )

            # Steps 3-6: HYPOTHESIZE + EVIDENCE LINK + SCORE + INVALIDATE
            available_fragment_ids = {f.fragment_id for f in context.fragments}
            belief_object = parse_llm_response(
                raw=raw_response,
                agent_id=self.agent_id,
                context_window_hash=context.context_window_hash,
                available_fragment_ids=available_fragment_ids,
            )

            if belief_object is None:
                self._error_count_24h += 1
                raise AgentProcessingError(
                    message="Failed to parse LLM response into BeliefObject",
                    agent_id=self.agent_id,
                )

            # Step 7: EMIT
            self._last_success = datetime.now(timezone.utc)
            log.info(
                "Cross-sectional analysis complete",
                belief_count=len(belief_object.beliefs),
                tickers=[b.ticker for b in belief_object.beliefs],
            )
            return belief_object

        except AgentProcessingError:
            raise
        except Exception as e:
            self._error_count_24h += 1
            raise AgentProcessingError(
                message=f"COGNIT-CROSSSEC processing failed: {e}",
                agent_id=self.agent_id,
            ) from e

    def _build_user_prompt(self, context: AgentContext) -> str:
        """Build the user prompt from context fragments.

        Routes fragments by data type AND entity into sections:
        - Primary entity's quarterly filings → quarterly_data
        - Primary entity's price data → price_data
        - Primary entity's earnings calls → earnings_data
        - Non-primary entity fragments → peer_data (for cross-sectional comparison)

        Like COGNIT-FUNDAMENTAL and COGNIT-NARRATIVE, identifies a primary
        ticker from entity counts and separates peer data. Cross-sectional
        analysis is fundamentally comparative — the peer section is central.
        """
        template = self._prompt_config.get(
            "user_prompt_template",
            (
                "Analyze the cross-sectional data for {ticker}.\n"
                "Quarterly Financials:\n{quarterly_data}\n"
                "Price Action:\n{price_data}\n"
                "Earnings Calls:\n{earnings_data}\n"
                "Peer Data:\n{peer_data}\n"
                "Fragment IDs:\n{fragment_ids}"
            ),
        )

        # Classify fragments by type and entity
        quarterly_fragments: list[str] = []
        price_fragments: list[str] = []
        earnings_fragments: list[str] = []
        peer_fragments: list[str] = []
        fragment_ids: list[str] = []

        # Identify the primary ticker (most common entity)
        entity_counts: dict[str, int] = {}
        for frag in context.fragments:
            if frag.entity:
                entity_counts[frag.entity] = entity_counts.get(frag.entity, 0) + 1

        primary_ticker = (
            max(entity_counts, key=entity_counts.get)
            if entity_counts
            else "UNKNOWN"
        )

        for frag in context.fragments:
            fragment_ids.append(
                f"- {frag.fragment_id} ({frag.data_type.value}, {frag.entity or 'N/A'})"
            )

            is_primary = (frag.entity == primary_ticker)

            if frag.data_type == DataType.FILING_10Q:
                if is_primary:
                    quarterly_fragments.append(self._format_fragment(frag))
                else:
                    peer_fragments.append(self._format_fragment(frag))
            elif frag.data_type == DataType.PRICE_OHLCV:
                if is_primary:
                    price_fragments.append(self._format_fragment(frag))
                else:
                    peer_fragments.append(self._format_fragment(frag))
            elif frag.data_type == DataType.EARNINGS_CALL:
                if is_primary:
                    earnings_fragments.append(self._format_fragment(frag))
                else:
                    peer_fragments.append(self._format_fragment(frag))

        # Build formatted sections — provide informative defaults for missing data
        quarterly_data = (
            "\n\n".join(quarterly_fragments)
            if quarterly_fragments
            else "No quarterly filing data available."
        )
        price_data = (
            "\n\n".join(price_fragments)
            if price_fragments
            else "No price data available."
        )
        earnings_data = (
            "\n\n".join(earnings_fragments)
            if earnings_fragments
            else "No earnings call data available."
        )
        peer_data = (
            "\n\n".join(peer_fragments)
            if peer_fragments
            else "No peer data available."
        )
        fragment_id_list = (
            "\n".join(fragment_ids)
            if fragment_ids
            else "No fragments available."
        )

        return template.format(
            ticker=primary_ticker,
            quarterly_data=quarterly_data,
            price_data=price_data,
            earnings_data=earnings_data,
            peer_data=peer_data,
            fragment_ids=fragment_id_list,
        )

    def _format_fragment(self, frag: MarketStateFragment) -> str:
        """Format a single fragment for prompt inclusion."""
        header = (
            f"[{frag.data_type.value}] {frag.entity or 'N/A'} "
            f"(fragment_id: {frag.fragment_id}, "
            f"timestamp: {frag.timestamp.isoformat()})"
        )
        payload_str = json.dumps(frag.payload, indent=2, default=str)
        return f"{header}\n{payload_str}"

    def get_health(self) -> HealthStatus:
        """Report health status."""
        if self._error_count_24h > 10:
            status = AgentStatus.UNHEALTHY
        elif self._error_count_24h > 3:
            status = AgentStatus.DEGRADED
        else:
            status = AgentStatus.HEALTHY

        return HealthStatus(
            agent_id=self.agent_id,
            status=status,
            last_run=self._last_run,
            last_success=self._last_success,
            error_count_24h=self._error_count_24h,
        )
