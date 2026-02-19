"""COGNIT-EVENT: Event-Driven Research Agent.

Analyzes corporate catalysts (8K filings), news sentiment, price action,
and options activity to generate short-term event-driven investment theses.

Spec Reference: Technical Spec v2.3, Section 4.2

Classification: ADAPTIVE — uses Claude Sonnet 4 via Anthropic API.
Subject to offline retraining. Prompt template is version-controlled.

Thesis types:
  1. Corporate action catalyst — M&A, spin-offs, restructuring from 8K
  2. Earnings event surprise — guidance changes, beats/misses from 8K + news
  3. Sentiment reversal — news-driven shift confirmed by price action
  4. Unusual options activity — IV spike, put/call extremes, large blocks
  5. Regulatory/legal catalyst — SEC actions, lawsuits from 8K

Time horizon: 1-30 days (shortest among cognition agents)

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
DEFAULT_PROMPT_VERSION = "cognit_event_v1.0.yaml"


class CognitEvent(BaseAgent[BeliefObject]):
    """Event-driven Research Agent.

    Consumes FILING_8K, SENTIMENT_NEWS, PRICE_OHLCV, and OPTIONS_CHAIN
    MarketStateFragments.

    Produces BeliefObjects with short-term investment theses driven by
    corporate events, sentiment shifts, and unusual options activity.

    Like COGNIT-FUNDAMENTAL, this agent identifies a primary ticker from
    entity counts — events are entity-specific, not cross-asset like
    COGNIT-MACRO. The 12-hour priority window reflects the urgency of
    event-driven catalysts.
    """

    CONSUMED_DATA_TYPES = {
        DataType.FILING_8K,
        DataType.SENTIMENT_NEWS,
        DataType.PRICE_OHLCV,
        DataType.OPTIONS_CHAIN,
    }

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        prompt_version: str = DEFAULT_PROMPT_VERSION,
    ) -> None:
        super().__init__(
            agent_id="COGNIT-EVENT",
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
                "system_prompt": "You are an event-driven equity analyst.",
                "user_prompt_template": (
                    "Analyze event-driven signals for {ticker}.\n"
                    "Filing Events:\n{filing_events}\n"
                    "News Sentiment:\n{news_data}\n"
                    "Price Action:\n{price_data}\n"
                    "Options Activity:\n{options_data}\n"
                    "Fragment IDs:\n{fragment_ids}"
                ),
            }

        with open(path) as f:
            return yaml.safe_load(f)

    async def process(self, context: AgentContext) -> BeliefObject:
        """Execute the Research Agent common loop.

        Steps:
          1. RECEIVE CONTEXT  → Validated by type
          2. ANALYZE          → Build prompt from event context, send to LLM
          3. HYPOTHESIZE      → Parse LLM response into beliefs
          4. EVIDENCE LINK    → Included in LLM response parsing
          5. SCORE            → Included in LLM response parsing
          6. INVALIDATE       → Included in LLM response parsing
          7. EMIT             → Return validated BeliefObject

        Args:
            context: AgentContext assembled by CONTEXT-SVC.

        Returns:
            BeliefObject with event-driven investment theses.

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
            log.info("Starting event-driven analysis")

            # Step 2: ANALYZE — build prompts and call LLM
            system_prompt = self._prompt_config.get(
                "system_prompt",
                "You are an event-driven equity analyst.",
            )
            user_prompt = self._build_user_prompt(context)

            log.info("Sending event context to LLM")
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
                "Event-driven analysis complete",
                belief_count=len(belief_object.beliefs),
                tickers=[b.ticker for b in belief_object.beliefs],
            )
            return belief_object

        except AgentProcessingError:
            raise
        except Exception as e:
            self._error_count_24h += 1
            raise AgentProcessingError(
                message=f"COGNIT-EVENT processing failed: {e}",
                agent_id=self.agent_id,
            ) from e

    def _build_user_prompt(self, context: AgentContext) -> str:
        """Build the user prompt from context fragments.

        Routes fragments by data type into four event-driven sections:
        - Corporate action events (FILING_8K)
        - News sentiment snapshot (SENTIMENT_NEWS)
        - Price action context (PRICE_OHLCV)
        - Options market activity (OPTIONS_CHAIN)

        Like COGNIT-FUNDAMENTAL, identifies a primary ticker from entity
        counts — events are entity-specific (unlike COGNIT-MACRO which
        is cross-asset).
        """
        template = self._prompt_config.get(
            "user_prompt_template",
            (
                "Analyze event-driven signals for {ticker}.\n"
                "Filing Events:\n{filing_events}\n"
                "News Sentiment:\n{news_data}\n"
                "Price Action:\n{price_data}\n"
                "Options Activity:\n{options_data}\n"
                "Fragment IDs:\n{fragment_ids}"
            ),
        )

        # Classify fragments by data type
        filing_fragments: list[str] = []
        news_fragments: list[str] = []
        price_fragments: list[str] = []
        options_fragments: list[str] = []
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

            if frag.data_type == DataType.FILING_8K:
                filing_fragments.append(self._format_fragment(frag))
            elif frag.data_type == DataType.SENTIMENT_NEWS:
                news_fragments.append(self._format_fragment(frag))
            elif frag.data_type == DataType.PRICE_OHLCV:
                price_fragments.append(self._format_fragment(frag))
            elif frag.data_type == DataType.OPTIONS_CHAIN:
                options_fragments.append(self._format_fragment(frag))

        # Build formatted sections — provide informative defaults for missing data
        filing_events = (
            "\n\n".join(filing_fragments)
            if filing_fragments
            else "No 8K filing events available."
        )
        news_data = (
            "\n\n".join(news_fragments)
            if news_fragments
            else "No news sentiment data available."
        )
        price_data = (
            "\n\n".join(price_fragments)
            if price_fragments
            else "No price action data available."
        )
        options_data = (
            "\n\n".join(options_fragments)
            if options_fragments
            else "No options activity data available."
        )
        fragment_id_list = (
            "\n".join(fragment_ids)
            if fragment_ids
            else "No fragments available."
        )

        return template.format(
            ticker=primary_ticker,
            filing_events=filing_events,
            news_data=news_data,
            price_data=price_data,
            options_data=options_data,
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
