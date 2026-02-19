"""COGNIT-NARRATIVE: Narrative Analysis Research Agent.

Analyzes earnings call transcripts, news sentiment, and 8K filings to
detect management tone shifts, language pattern changes, guidance
divergences, and narrative momentum across the company and its peers.

Spec Reference: Technical Spec v2.3, Section 4.2

Classification: ADAPTIVE — uses Claude Sonnet 4 via Anthropic API.
Subject to offline retraining. Prompt template is version-controlled.

Thesis types:
  1. Management tone shift — language/confidence changes in earnings calls
  2. Guidance divergence — gap between stated guidance and tone/hedging language
  3. Narrative momentum — accelerating sentiment in news + calls
  4. Peer narrative divergence — company narrative deviates from sector peers
  5. Disclosure anomaly — unusual 8K filing patterns, timing, or language

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
DEFAULT_PROMPT_VERSION = "cognit_narrative_v1.0.yaml"


class CognitNarrative(BaseAgent[BeliefObject]):
    """Narrative analysis Research Agent.

    Consumes EARNINGS_CALL, SENTIMENT_NEWS, and FILING_8K
    MarketStateFragments.

    Produces BeliefObjects with investment theses driven by narrative
    analysis — management tone shifts, guidance divergences, sentiment
    momentum, and peer narrative comparison.

    Like COGNIT-FUNDAMENTAL, this agent identifies a primary ticker and
    separates peer data into a dedicated section. Narrative analysis
    benefits from comparing a company's management language against
    sector peers to detect divergences.
    """

    CONSUMED_DATA_TYPES = {
        DataType.EARNINGS_CALL,
        DataType.SENTIMENT_NEWS,
        DataType.FILING_8K,
    }

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        prompt_version: str = DEFAULT_PROMPT_VERSION,
    ) -> None:
        super().__init__(
            agent_id="COGNIT-NARRATIVE",
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
                "system_prompt": "You are a narrative analysis equity analyst.",
                "user_prompt_template": (
                    "Analyze the narrative signals for {ticker}.\n"
                    "Earnings Call Transcripts:\n{earnings_data}\n"
                    "News Sentiment:\n{news_data}\n"
                    "8K Filings:\n{filing_data}\n"
                    "Peer Narratives:\n{peer_data}\n"
                    "Fragment IDs:\n{fragment_ids}"
                ),
            }

        with open(path) as f:
            return yaml.safe_load(f)

    async def process(self, context: AgentContext) -> BeliefObject:
        """Execute the Research Agent common loop.

        Steps:
          1. RECEIVE CONTEXT  → Validated by type
          2. ANALYZE          → Build prompt from narrative context, send to LLM
          3. HYPOTHESIZE      → Parse LLM response into beliefs
          4. EVIDENCE LINK    → Included in LLM response parsing
          5. SCORE            → Included in LLM response parsing
          6. INVALIDATE       → Included in LLM response parsing
          7. EMIT             → Return validated BeliefObject

        Args:
            context: AgentContext assembled by CONTEXT-SVC.

        Returns:
            BeliefObject with narrative-driven investment theses.

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
            log.info("Starting narrative analysis")

            # Step 2: ANALYZE — build prompts and call LLM
            system_prompt = self._prompt_config.get(
                "system_prompt",
                "You are a narrative analysis equity analyst.",
            )
            user_prompt = self._build_user_prompt(context)

            log.info("Sending narrative context to LLM")
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
                "Narrative analysis complete",
                belief_count=len(belief_object.beliefs),
                tickers=[b.ticker for b in belief_object.beliefs],
            )
            return belief_object

        except AgentProcessingError:
            raise
        except Exception as e:
            self._error_count_24h += 1
            raise AgentProcessingError(
                message=f"COGNIT-NARRATIVE processing failed: {e}",
                agent_id=self.agent_id,
            ) from e

    def _build_user_prompt(self, context: AgentContext) -> str:
        """Build the user prompt from context fragments.

        Routes fragments by data type AND entity into sections:
        - Primary entity's earnings call transcripts → earnings_data
        - Primary entity's news sentiment → news_data
        - Primary entity's 8K filings → filing_data
        - Non-primary entity fragments → peer_data (for comparison)

        Like COGNIT-FUNDAMENTAL, identifies a primary ticker from entity
        counts and separates peer data. Narrative analysis benefits from
        comparing management tone across sector peers.
        """
        template = self._prompt_config.get(
            "user_prompt_template",
            (
                "Analyze the narrative signals for {ticker}.\n"
                "Earnings Call Transcripts:\n{earnings_data}\n"
                "News Sentiment:\n{news_data}\n"
                "8K Filings:\n{filing_data}\n"
                "Peer Narratives:\n{peer_data}\n"
                "Fragment IDs:\n{fragment_ids}"
            ),
        )

        # Classify fragments by type and entity
        earnings_fragments: list[str] = []
        news_fragments: list[str] = []
        filing_fragments: list[str] = []
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

            if frag.data_type == DataType.EARNINGS_CALL:
                if is_primary:
                    earnings_fragments.append(self._format_fragment(frag))
                else:
                    peer_fragments.append(self._format_fragment(frag))
            elif frag.data_type == DataType.SENTIMENT_NEWS:
                if is_primary:
                    news_fragments.append(self._format_fragment(frag))
                else:
                    peer_fragments.append(self._format_fragment(frag))
            elif frag.data_type == DataType.FILING_8K:
                if is_primary:
                    filing_fragments.append(self._format_fragment(frag))
                else:
                    peer_fragments.append(self._format_fragment(frag))

        # Build formatted sections — provide informative defaults for missing data
        earnings_data = (
            "\n\n".join(earnings_fragments)
            if earnings_fragments
            else "No earnings call data available."
        )
        news_data = (
            "\n\n".join(news_fragments)
            if news_fragments
            else "No news sentiment data available."
        )
        filing_data = (
            "\n\n".join(filing_fragments)
            if filing_fragments
            else "No 8K filing data available."
        )
        peer_data = (
            "\n\n".join(peer_fragments)
            if peer_fragments
            else "No peer narrative data available."
        )
        fragment_id_list = (
            "\n".join(fragment_ids)
            if fragment_ids
            else "No fragments available."
        )

        return template.format(
            ticker=primary_ticker,
            earnings_data=earnings_data,
            news_data=news_data,
            filing_data=filing_data,
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
