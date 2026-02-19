"""REGIME-NARR: Narrative Regime Classification Agent.

Uses an LLM to analyze news sentiment, earnings narratives, and market
commentary to classify the dominant market narrative driving current
price action and positioning.

Spec Reference: Technical Spec v2.3, Section 4.3

Classification: ADAPTIVE — uses Claude Sonnet 4 via Anthropic API.
Subject to offline retraining. Prompt template is version-controlled.

The output is a RegimeStateObject with a populated narrative_overlay
containing the LLM-generated narrative label, confidence, key signals,
affected sectors, and regime alignment assessment.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import structlog
import yaml

from providence.agents.base import AgentContext, AgentStatus, BaseAgent, HealthStatus
from providence.agents.regime.hmm_model import (
    classify_regime,
    derive_risk_mode,
    features_to_composite_score,
)
from providence.agents.regime.regime_features import extract_regime_features
from providence.exceptions import AgentProcessingError
from providence.infra.llm_client import AnthropicClient, LLMClient
from providence.schemas.enums import DataType, StatisticalRegime, SystemRiskMode
from providence.schemas.market_state import MarketStateFragment
from providence.schemas.regime import NarrativeRegimeOverlay, RegimeStateObject

logger = structlog.get_logger()

# Path to prompt templates
PROMPT_DIR = Path(__file__).parent.parent.parent / "prompts"
DEFAULT_PROMPT_VERSION = "regime_narr_v1.0.yaml"

# Valid regime alignment values
VALID_ALIGNMENTS = {"CONFIRMS", "DIVERGES", "NEUTRAL"}

# Map from string to enum for statistical regime
REGIME_MAP = {r.value: r for r in StatisticalRegime}
RISK_MODE_MAP = {r.value: r for r in SystemRiskMode}


def parse_narrative_response(
    raw: str,
) -> dict[str, Any] | None:
    """Parse the LLM response into a structured narrative assessment.

    Extracts the JSON from the raw LLM response and validates
    required fields.

    Args:
        raw: Raw LLM response string (should be JSON).

    Returns:
        Parsed dict with narrative fields, or None if parsing fails.
    """
    # Try to extract JSON from the response
    text = raw.strip()

    # Handle markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last lines (fences)
        json_lines = []
        in_block = False
        for line in lines:
            if line.strip().startswith("```") and not in_block:
                in_block = True
                continue
            elif line.strip() == "```" and in_block:
                break
            elif in_block:
                json_lines.append(line)
        text = "\n".join(json_lines)

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object within the text
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            return None
        try:
            parsed = json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return None

    if not isinstance(parsed, dict):
        return None

    # Validate required fields
    label = parsed.get("narrative_label")
    if not label or not isinstance(label, str):
        return None

    confidence = parsed.get("narrative_confidence")
    if confidence is None:
        return None
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        return None

    # Clamp confidence to [0, 0.80]
    confidence = max(0.0, min(0.80, confidence))

    # Optional fields with defaults
    key_signals = parsed.get("key_signals", [])
    if not isinstance(key_signals, list):
        key_signals = []
    key_signals = [str(s) for s in key_signals if s]

    affected_sectors = parsed.get("affected_sectors", [])
    if not isinstance(affected_sectors, list):
        affected_sectors = []
    affected_sectors = [str(s) for s in affected_sectors if s]

    alignment = parsed.get("regime_alignment", "NEUTRAL")
    if alignment not in VALID_ALIGNMENTS:
        alignment = "NEUTRAL"

    summary = parsed.get("summary", "")
    if not isinstance(summary, str):
        summary = ""

    regime_str = parsed.get("statistical_regime_assessment", "")
    regime = REGIME_MAP.get(regime_str)

    risk_mode_str = parsed.get("risk_mode_suggestion", "")
    risk_mode = RISK_MODE_MAP.get(risk_mode_str)

    return {
        "narrative_label": label.strip()[:200],
        "narrative_confidence": round(confidence, 4),
        "key_signals": key_signals[:10],
        "affected_sectors": affected_sectors[:11],
        "regime_alignment": alignment,
        "summary": summary[:1000],
        "statistical_regime": regime,
        "risk_mode": risk_mode,
    }


class RegimeNarr(BaseAgent[RegimeStateObject]):
    """Narrative regime classification agent.

    Consumes SENTIMENT_NEWS, EARNINGS_CALL, and macro/price
    MarketStateFragments. Sends them to an LLM to classify the
    dominant market narrative.

    Produces a RegimeStateObject with populated narrative_overlay.

    ADAPTIVE: Uses Claude Sonnet 4 for narrative regime classification.
    """

    CONSUMED_DATA_TYPES = {
        DataType.SENTIMENT_NEWS,
        DataType.EARNINGS_CALL,
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
            agent_id="REGIME-NARR",
            agent_type="regime",
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
                "system_prompt": "You are a market regime analyst.",
                "user_prompt_template": (
                    "Analyze the market narrative.\n"
                    "News:\n{news_data}\n"
                    "Earnings:\n{earnings_data}\n"
                    "Macro:\n{macro_data}\n"
                    "Prices:\n{price_data}\n"
                    "Statistical Regime:\n{statistical_regime}\n"
                    "Fragment IDs:\n{fragment_ids}"
                ),
            }

        with open(path) as f:
            return yaml.safe_load(f)

    async def process(self, context: AgentContext) -> RegimeStateObject:
        """Execute the narrative regime classification pipeline.

        Steps:
          1. RECEIVE CONTEXT   → Validated by caller
          2. COMPUTE BASELINE  → Extract features for statistical baseline
          3. BUILD PROMPT      → Route fragments by type into prompt sections
          4. CALL LLM          → Send prompt to Claude Sonnet 4
          5. PARSE RESPONSE    → Extract narrative assessment
          6. BUILD OVERLAY     → Create NarrativeRegimeOverlay
          7. EMIT              → Return RegimeStateObject with narrative_overlay

        Args:
            context: AgentContext assembled by CONTEXT-SVC.

        Returns:
            RegimeStateObject with narrative_overlay populated.

        Raises:
            AgentProcessingError: If processing fails.
        """
        self._last_run = datetime.now(timezone.utc)

        try:
            log = logger.bind(
                agent_id=self.agent_id,
                context_hash=context.context_window_hash,
                fragment_count=len(context.fragments),
            )
            log.info("Starting narrative regime classification")

            # Step 2: COMPUTE BASELINE — statistical regime for context
            global_features = extract_regime_features(context.fragments)
            stat_regime, stat_conf, stat_probs = classify_regime(global_features)
            stat_risk_mode = derive_risk_mode(stat_regime, stat_conf)

            # Step 3: BUILD PROMPT
            system_prompt = self._prompt_config.get(
                "system_prompt",
                "You are a market regime analyst.",
            )
            user_prompt = self._build_user_prompt(context, stat_regime)

            # Step 4: CALL LLM
            log.info("Sending narrative context to LLM")
            raw_response = await self._llm_client.complete(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )

            # Step 5: PARSE RESPONSE
            parsed = parse_narrative_response(raw_response)
            if parsed is None:
                self._error_count_24h += 1
                raise AgentProcessingError(
                    message="Failed to parse LLM response into narrative assessment",
                    agent_id=self.agent_id,
                )

            # Step 6: BUILD OVERLAY
            narrative_overlay = NarrativeRegimeOverlay(
                label=parsed["narrative_label"],
                confidence=parsed["narrative_confidence"],
                key_signals=parsed["key_signals"],
                affected_sectors=parsed["affected_sectors"],
                regime_alignment=parsed["regime_alignment"],
                summary=parsed["summary"],
            )

            # Use LLM's regime assessment if available, else use statistical
            final_regime = parsed["statistical_regime"] or stat_regime
            final_risk_mode = parsed["risk_mode"] or stat_risk_mode

            # Build features dict
            composite = features_to_composite_score(global_features)
            features_dict: dict[str, float] = {"composite_score": composite}
            if global_features.realized_vol_20d is not None:
                features_dict["realized_vol_20d"] = global_features.realized_vol_20d
            if global_features.yield_spread_2s10s is not None:
                features_dict["yield_spread_2s10s"] = global_features.yield_spread_2s10s
            if global_features.cds_ig_spread is not None:
                features_dict["cds_ig_spread"] = global_features.cds_ig_spread
            if global_features.vix_proxy is not None:
                features_dict["vix_proxy"] = global_features.vix_proxy

            # Step 7: EMIT
            regime_state = RegimeStateObject(
                agent_id=self.agent_id,
                timestamp=datetime.now(timezone.utc),
                context_window_hash=context.context_window_hash,
                statistical_regime=final_regime,
                regime_confidence=round(stat_conf, 6),
                regime_probabilities=stat_probs,
                system_risk_mode=final_risk_mode,
                features_used=features_dict,
                narrative_overlay=narrative_overlay,
            )

            self._last_success = datetime.now(timezone.utc)
            log.info(
                "Narrative regime classification complete",
                narrative_label=parsed["narrative_label"],
                narrative_confidence=parsed["narrative_confidence"],
                regime_alignment=parsed["regime_alignment"],
            )
            return regime_state

        except AgentProcessingError:
            raise
        except Exception as e:
            self._error_count_24h += 1
            raise AgentProcessingError(
                message=f"REGIME-NARR processing failed: {e}",
                agent_id=self.agent_id,
            ) from e

    def _build_user_prompt(
        self,
        context: AgentContext,
        statistical_regime: StatisticalRegime,
    ) -> str:
        """Build the user prompt from context fragments.

        Routes fragments by data type into four sections:
        - News sentiment (SENTIMENT_NEWS)
        - Earnings (EARNINGS_CALL)
        - Macro context (MACRO_*)
        - Price action (PRICE_OHLCV)
        """
        template = self._prompt_config.get(
            "user_prompt_template",
            (
                "Analyze the market narrative.\n"
                "News:\n{news_data}\n"
                "Earnings:\n{earnings_data}\n"
                "Macro:\n{macro_data}\n"
                "Prices:\n{price_data}\n"
                "Statistical Regime:\n{statistical_regime}\n"
                "Fragment IDs:\n{fragment_ids}"
            ),
        )

        news_fragments: list[str] = []
        earnings_fragments: list[str] = []
        macro_fragments: list[str] = []
        price_fragments: list[str] = []
        fragment_ids: list[str] = []

        macro_types = {
            DataType.MACRO_YIELD_CURVE,
            DataType.MACRO_CDS,
            DataType.MACRO_ECONOMIC,
        }

        for frag in context.fragments:
            fragment_ids.append(
                f"- {frag.fragment_id} ({frag.data_type.value}, {frag.entity or 'N/A'})"
            )

            if frag.data_type == DataType.SENTIMENT_NEWS:
                news_fragments.append(self._format_fragment(frag))
            elif frag.data_type == DataType.EARNINGS_CALL:
                earnings_fragments.append(self._format_fragment(frag))
            elif frag.data_type in macro_types:
                macro_fragments.append(self._format_fragment(frag))
            elif frag.data_type == DataType.PRICE_OHLCV:
                price_fragments.append(self._format_fragment(frag))

        news_data = (
            "\n\n".join(news_fragments)
            if news_fragments
            else "No news sentiment data available."
        )
        earnings_data = (
            "\n\n".join(earnings_fragments)
            if earnings_fragments
            else "No earnings call data available."
        )
        macro_data = (
            "\n\n".join(macro_fragments)
            if macro_fragments
            else "No macro context data available."
        )
        price_data = (
            "\n\n".join(price_fragments)
            if price_fragments
            else "No price action data available."
        )
        fragment_id_list = (
            "\n".join(fragment_ids)
            if fragment_ids
            else "No fragments available."
        )

        return template.format(
            news_data=news_data,
            earnings_data=earnings_data,
            macro_data=macro_data,
            price_data=price_data,
            statistical_regime=statistical_regime.value,
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
