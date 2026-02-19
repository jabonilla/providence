"""COGNIT-MACRO: Macroeconomic Research Agent.

Analyzes yield curve dynamics, credit conditions (CDS), economic indicators,
and broad market price action to generate macro-driven investment theses.

Spec Reference: Technical Spec v2.3, Section 4.2

Classification: ADAPTIVE — uses Claude Sonnet 4 via Anthropic API.
Subject to offline retraining. Prompt template is version-controlled.

Thesis types:
  1. Yield curve signal — inversion/steepening → sector rotation
  2. Credit stress — CDS spread widening → risk-off positioning
  3. Macro momentum — GDP/CPI/employment trends → cyclical vs defensive
  4. Rate regime change — Fed policy implications from rate movements
  5. Macro divergence — conflicting signals across indicators

Time horizon: 30-180 days

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
DEFAULT_PROMPT_VERSION = "cognit_macro_v1.0.yaml"


class CognitMacro(BaseAgent[BeliefObject]):
    """Macroeconomic analysis Research Agent.

    Consumes MACRO_YIELD_CURVE, MACRO_CDS, MACRO_ECONOMIC, and PRICE_OHLCV
    MarketStateFragments.

    Produces BeliefObjects with investment theses driven by macroeconomic
    analysis — yield curve signals, credit conditions, economic trends,
    and rate regime shifts.

    Unlike COGNIT-FUNDAMENTAL, this agent has no "primary ticker" concept.
    Macro analysis is cross-asset; theses target broad market indices or
    sector ETFs (SPY, XLF, IYR, etc.) rather than individual stocks.
    """

    CONSUMED_DATA_TYPES = {
        DataType.MACRO_YIELD_CURVE,
        DataType.MACRO_CDS,
        DataType.MACRO_ECONOMIC,
        DataType.PRICE_OHLCV,
    }

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        prompt_version: str = DEFAULT_PROMPT_VERSION,
    ) -> None:
        super().__init__(
            agent_id="COGNIT-MACRO",
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
                "system_prompt": "You are a macroeconomic research analyst.",
                "user_prompt_template": (
                    "Analyze the macroeconomic environment.\n"
                    "Yield Curve:\n{yield_curve_data}\n"
                    "Credit:\n{cds_data}\n"
                    "Economic Indicators:\n{economic_data}\n"
                    "Price Data:\n{price_data}\n"
                    "Fragment IDs:\n{fragment_ids}"
                ),
            }

        with open(path) as f:
            return yaml.safe_load(f)

    async def process(self, context: AgentContext) -> BeliefObject:
        """Execute the Research Agent common loop.

        Steps:
          1. RECEIVE CONTEXT  → Validated by type
          2. ANALYZE          → Build prompt from macro context, send to LLM
          3. HYPOTHESIZE      → Parse LLM response into beliefs
          4. EVIDENCE LINK    → Included in LLM response parsing
          5. SCORE            → Included in LLM response parsing
          6. INVALIDATE       → Included in LLM response parsing
          7. EMIT             → Return validated BeliefObject

        Args:
            context: AgentContext assembled by CONTEXT-SVC.

        Returns:
            BeliefObject with macro-driven investment theses.

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
            log.info("Starting macro analysis")

            # Step 2: ANALYZE — build prompts and call LLM
            system_prompt = self._prompt_config.get(
                "system_prompt",
                "You are a macroeconomic research analyst.",
            )
            user_prompt = self._build_user_prompt(context)

            log.info("Sending macro context to LLM")
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
                "Macro analysis complete",
                belief_count=len(belief_object.beliefs),
                tickers=[b.ticker for b in belief_object.beliefs],
            )
            return belief_object

        except AgentProcessingError:
            raise
        except Exception as e:
            self._error_count_24h += 1
            raise AgentProcessingError(
                message=f"COGNIT-MACRO processing failed: {e}",
                agent_id=self.agent_id,
            ) from e

    def _build_user_prompt(self, context: AgentContext) -> str:
        """Build the user prompt from context fragments.

        Routes fragments by data type into four sections:
        - Yield curve context (MACRO_YIELD_CURVE)
        - Credit market snapshot (MACRO_CDS)
        - Economic indicators (MACRO_ECONOMIC)
        - Broad market price action (PRICE_OHLCV)

        Unlike COGNIT-FUNDAMENTAL, there is no "primary ticker" concept.
        Macro analysis is cross-asset — all fragments are grouped by data
        type regardless of entity.
        """
        template = self._prompt_config.get(
            "user_prompt_template",
            (
                "Analyze the macroeconomic environment.\n"
                "Yield Curve:\n{yield_curve_data}\n"
                "Credit:\n{cds_data}\n"
                "Economic Indicators:\n{economic_data}\n"
                "Price Data:\n{price_data}\n"
                "Fragment IDs:\n{fragment_ids}"
            ),
        )

        # Classify fragments by data type (not by entity)
        yield_curve_fragments: list[str] = []
        cds_fragments: list[str] = []
        economic_fragments: list[str] = []
        price_fragments: list[str] = []
        fragment_ids: list[str] = []

        for frag in context.fragments:
            fragment_ids.append(
                f"- {frag.fragment_id} ({frag.data_type.value}, {frag.entity or 'N/A'})"
            )

            if frag.data_type == DataType.MACRO_YIELD_CURVE:
                yield_curve_fragments.append(self._format_fragment(frag))
            elif frag.data_type == DataType.MACRO_CDS:
                cds_fragments.append(self._format_fragment(frag))
            elif frag.data_type == DataType.MACRO_ECONOMIC:
                economic_fragments.append(self._format_fragment(frag))
            elif frag.data_type == DataType.PRICE_OHLCV:
                price_fragments.append(self._format_fragment(frag))

        # Build formatted sections — provide informative defaults for missing data
        yield_curve_data = (
            "\n\n".join(yield_curve_fragments)
            if yield_curve_fragments
            else "No yield curve data available."
        )
        cds_data = (
            "\n\n".join(cds_fragments)
            if cds_fragments
            else "No CDS data available."
        )
        economic_data = (
            "\n\n".join(economic_fragments)
            if economic_fragments
            else "No economic indicator data available."
        )
        price_data = (
            "\n\n".join(price_fragments)
            if price_fragments
            else "No broad market price data available."
        )
        fragment_id_list = (
            "\n".join(fragment_ids)
            if fragment_ids
            else "No fragments available."
        )

        return template.format(
            yield_curve_data=yield_curve_data,
            cds_data=cds_data,
            economic_data=economic_data,
            price_data=price_data,
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
