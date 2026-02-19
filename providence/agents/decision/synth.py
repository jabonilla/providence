"""DECIDE-SYNTH: Belief Synthesis Agent.

Synthesizes beliefs from independent research agents into unified
position intents. Resolves conflicts between agents, applies regime
adjustments, and aggregates invalidation conditions.

Spec Reference: Technical Spec v2.3, Section 4.4

Classification: ADAPTIVE — uses Claude Sonnet 4 via Anthropic API.
Subject to offline retraining. Prompt template is version-controlled.

Input: AgentContext with beliefs and regime data in metadata:
  - metadata["belief_objects"]: list of serialized BeliefObject dicts
  - metadata["regime_state"]: serialized RegimeStateObject dict

Output: SynthesisOutput containing SynthesizedPositionIntents.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

import structlog
import yaml

from providence.agents.base import AgentContext, AgentStatus, BaseAgent, HealthStatus
from providence.exceptions import AgentProcessingError
from providence.infra.llm_client import AnthropicClient, LLMClient
from providence.schemas.decision import (
    ActiveInvalidation,
    ConflictResolution,
    ContributingThesis,
    SynthesisOutput,
    SynthesizedPositionIntent,
)
from providence.schemas.enums import Direction, Magnitude

logger = structlog.get_logger()

PROMPT_DIR = Path(__file__).parent.parent.parent / "prompts"
DEFAULT_PROMPT_VERSION = "decide_synth_v1.0.yaml"

# Valid direction and magnitude values for parsing
DIRECTION_MAP = {d.value: d for d in Direction}
MAGNITUDE_MAP = {m.value: m for m in Magnitude}


def parse_synthesis_response(raw: str) -> dict[str, Any] | None:
    """Parse LLM response into synthesis data.

    Args:
        raw: Raw LLM response string (should be JSON).

    Returns:
        Parsed dict with position_intents, or None if parsing fails.
    """
    text = raw.strip()

    # Handle markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
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

    intents = parsed.get("position_intents")
    if not isinstance(intents, list) or len(intents) == 0:
        return None

    # Validate each intent has required fields
    for intent in intents:
        if not isinstance(intent, dict):
            return None
        if not intent.get("ticker"):
            return None
        if intent.get("net_direction") not in DIRECTION_MAP:
            return None
        conf = intent.get("synthesized_confidence")
        if conf is None:
            return None
        try:
            conf = float(conf)
        except (TypeError, ValueError):
            return None
        # Clamp to [0, 0.85]
        intent["synthesized_confidence"] = max(0.0, min(0.85, conf))

    return parsed


def _build_intent(raw_intent: dict) -> SynthesizedPositionIntent:
    """Build a SynthesizedPositionIntent from parsed LLM data.

    Args:
        raw_intent: Dict from parsed LLM response.

    Returns:
        SynthesizedPositionIntent.
    """
    # Contributing theses
    contributing = []
    for ct in raw_intent.get("contributing_theses", []):
        if isinstance(ct, dict) and ct.get("thesis_id"):
            contributing.append(ContributingThesis(
                thesis_id=ct["thesis_id"],
                agent_id=ct.get("agent_id", "UNKNOWN"),
                ticker=ct.get("ticker", raw_intent["ticker"]),
                direction=DIRECTION_MAP.get(ct.get("direction", "NEUTRAL"), Direction.NEUTRAL),
                raw_confidence=max(0.0, min(1.0, float(ct.get("raw_confidence", 0.5)))),
                magnitude=MAGNITUDE_MAP.get(ct.get("magnitude", "MODERATE"), Magnitude.MODERATE),
                synthesis_weight=max(0.0, min(1.0, float(ct.get("synthesis_weight", 0.2)))),
            ))

    # Conflicting theses
    conflicting = []
    for ct in raw_intent.get("conflicting_theses", []):
        if isinstance(ct, dict) and ct.get("thesis_id"):
            conflicting.append(ContributingThesis(
                thesis_id=ct["thesis_id"],
                agent_id=ct.get("agent_id", "UNKNOWN"),
                ticker=ct.get("ticker", raw_intent["ticker"]),
                direction=DIRECTION_MAP.get(ct.get("direction", "NEUTRAL"), Direction.NEUTRAL),
                raw_confidence=max(0.0, min(1.0, float(ct.get("raw_confidence", 0.5)))),
                magnitude=MAGNITUDE_MAP.get(ct.get("magnitude", "MODERATE"), Magnitude.MODERATE),
                synthesis_weight=max(0.0, min(1.0, float(ct.get("synthesis_weight", 0.1)))),
            ))

    # Conflict resolution
    cr_data = raw_intent.get("conflict_resolution", {})
    if isinstance(cr_data, dict):
        conflict_resolution = ConflictResolution(
            has_conflict=bool(cr_data.get("has_conflict", False)),
            conflict_type=str(cr_data.get("conflict_type", "NONE")),
            resolution_method=str(cr_data.get("resolution_method", "")),
            resolution_rationale=str(cr_data.get("resolution_rationale", "")),
            net_conviction_delta=max(-1.0, min(0.0, float(cr_data.get("net_conviction_delta", 0.0)))),
        )
    else:
        conflict_resolution = ConflictResolution()

    # Active invalidations
    invalidations = []
    for inv in raw_intent.get("active_invalidations", [])[:5]:
        if isinstance(inv, dict) and inv.get("metric"):
            invalidations.append(ActiveInvalidation(
                source_thesis_id=str(inv.get("source_thesis_id", "")),
                source_agent_id=str(inv.get("source_agent_id", "")),
                metric=str(inv["metric"]),
                operator=str(inv.get("operator", "GT")),
                threshold=float(inv.get("threshold", 0.0)),
                description=str(inv.get("description", "")),
            ))

    # Time horizon
    time_horizon = int(raw_intent.get("time_horizon_days", 60))
    if time_horizon < 1:
        time_horizon = 1

    # Regime adjustment
    regime_adj = float(raw_intent.get("regime_adjustment", 0.0))
    regime_adj = max(-0.30, min(0.10, regime_adj))

    return SynthesizedPositionIntent(
        ticker=raw_intent["ticker"],
        net_direction=DIRECTION_MAP[raw_intent["net_direction"]],
        synthesized_confidence=raw_intent["synthesized_confidence"],
        contributing_theses=contributing,
        conflicting_theses=conflicting,
        conflict_resolution=conflict_resolution,
        time_horizon_days=time_horizon,
        regime_adjustment=round(regime_adj, 4),
        active_invalidations=invalidations,
        synthesis_rationale=str(raw_intent.get("synthesis_rationale", "")),
    )


class DecideSynth(BaseAgent[SynthesisOutput]):
    """Belief synthesis agent.

    Consumes BeliefObjects from all research agents and the current
    RegimeStateObject. Synthesizes them into unified position intents
    using an LLM for conflict resolution and regime adjustment.

    ADAPTIVE: Uses Claude Sonnet 4 for synthesis.
    """

    # DECIDE-SYNTH doesn't consume raw market data.
    # It reads beliefs and regime from context metadata.
    CONSUMED_DATA_TYPES: set = set()

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        prompt_version: str = DEFAULT_PROMPT_VERSION,
    ) -> None:
        super().__init__(
            agent_id="DECIDE-SYNTH",
            agent_type="decision",
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
            logger.warning("Prompt file not found, using defaults", path=str(path))
            return {
                "system_prompt": "You are a belief synthesis agent.",
                "user_prompt_template": (
                    "Synthesize beliefs:\n{beliefs_data}\n"
                    "Regime:\n{regime_context}\n"
                    "Risk mode:\n{risk_mode}"
                ),
            }
        with open(path) as f:
            return yaml.safe_load(f)

    async def process(self, context: AgentContext) -> SynthesisOutput:
        """Execute the belief synthesis pipeline.

        Steps:
          1. RECEIVE CONTEXT   → Extract beliefs and regime from metadata
          2. FORMAT BELIEFS    → Prepare belief summaries for LLM
          3. CALL LLM          → Send beliefs + regime to Claude Sonnet 4
          4. PARSE RESPONSE    → Extract position intents
          5. BUILD INTENTS     → Construct SynthesizedPositionIntents
          6. EMIT              → Return SynthesisOutput

        Args:
            context: AgentContext with beliefs and regime in metadata.

        Returns:
            SynthesisOutput with synthesized position intents.

        Raises:
            AgentProcessingError: If processing fails.
        """
        self._last_run = datetime.now(timezone.utc)

        try:
            log = logger.bind(
                agent_id=self.agent_id,
                context_hash=context.context_window_hash,
            )
            log.info("Starting belief synthesis")

            # Step 1: EXTRACT UPSTREAM DATA
            belief_objects = context.metadata.get("belief_objects", [])
            regime_state = context.metadata.get("regime_state", {})

            total_beliefs = 0
            if isinstance(belief_objects, list):
                for bo in belief_objects:
                    if isinstance(bo, dict):
                        beliefs = bo.get("beliefs", [])
                        if isinstance(beliefs, list):
                            total_beliefs += len(beliefs)

            log.info(
                "Beliefs extracted",
                belief_object_count=len(belief_objects) if isinstance(belief_objects, list) else 0,
                total_beliefs=total_beliefs,
            )

            # Step 2: FORMAT BELIEFS
            beliefs_data = self._format_beliefs(belief_objects)
            regime_context = self._format_regime(regime_state)
            risk_mode = regime_state.get("system_risk_mode", "NORMAL") if isinstance(regime_state, dict) else "NORMAL"

            # Step 3: CALL LLM
            system_prompt = self._prompt_config.get(
                "system_prompt",
                "You are a belief synthesis agent.",
            )
            template = self._prompt_config.get(
                "user_prompt_template",
                "Synthesize beliefs:\n{beliefs_data}\nRegime:\n{regime_context}\nRisk mode:\n{risk_mode}",
            )
            user_prompt = template.format(
                beliefs_data=beliefs_data,
                regime_context=regime_context,
                risk_mode=risk_mode,
            )

            log.info("Sending beliefs to LLM for synthesis")
            raw_response = await self._llm_client.complete(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )

            # Step 4: PARSE RESPONSE
            parsed = parse_synthesis_response(raw_response)
            if parsed is None:
                self._error_count_24h += 1
                raise AgentProcessingError(
                    message="Failed to parse LLM response into synthesis output",
                    agent_id=self.agent_id,
                )

            # Step 5: BUILD INTENTS
            intents = []
            for raw_intent in parsed.get("position_intents", []):
                try:
                    intent = _build_intent(raw_intent)
                    intents.append(intent)
                except (ValueError, TypeError, KeyError) as e:
                    log.warning("Skipping malformed intent", error=str(e))
                    continue

            if not intents:
                self._error_count_24h += 1
                raise AgentProcessingError(
                    message="No valid position intents produced from LLM response",
                    agent_id=self.agent_id,
                )

            # Step 6: EMIT
            regime_label = ""
            if isinstance(regime_state, dict):
                regime_label = regime_state.get("statistical_regime", "")
                narr = regime_state.get("narrative_overlay", {})
                if isinstance(narr, dict) and narr.get("label"):
                    regime_label = f"{regime_label} / {narr['label']}"

            synthesis_output = SynthesisOutput(
                agent_id=self.agent_id,
                timestamp=datetime.now(timezone.utc),
                context_window_hash=context.context_window_hash,
                position_intents=intents,
                regime_context=regime_label,
                total_beliefs_consumed=total_beliefs,
            )

            self._last_success = datetime.now(timezone.utc)
            log.info(
                "Belief synthesis complete",
                intent_count=len(intents),
                tickers=[i.ticker for i in intents],
                total_beliefs=total_beliefs,
            )
            return synthesis_output

        except AgentProcessingError:
            raise
        except Exception as e:
            self._error_count_24h += 1
            raise AgentProcessingError(
                message=f"DECIDE-SYNTH processing failed: {e}",
                agent_id=self.agent_id,
            ) from e

    def _format_beliefs(self, belief_objects: Any) -> str:
        """Format belief objects for prompt inclusion."""
        if not isinstance(belief_objects, list) or not belief_objects:
            return "No beliefs available."

        sections: list[str] = []
        for bo in belief_objects:
            if not isinstance(bo, dict):
                continue
            agent_id = bo.get("agent_id", "UNKNOWN")
            beliefs = bo.get("beliefs", [])
            if not isinstance(beliefs, list):
                continue

            lines = [f"### {agent_id} ({len(beliefs)} beliefs)"]
            for b in beliefs:
                if not isinstance(b, dict):
                    continue
                lines.append(
                    f"  - [{b.get('thesis_id', '?')}] {b.get('ticker', '?')} "
                    f"{b.get('direction', '?')} {b.get('magnitude', '?')} "
                    f"conf={b.get('raw_confidence', '?')} "
                    f"horizon={b.get('time_horizon_days', '?')}d"
                )
                summary = b.get("thesis_summary", "")
                if summary:
                    lines.append(f"    Summary: {summary}")

                # Include invalidation conditions
                invs = b.get("invalidation_conditions", [])
                if isinstance(invs, list) and invs:
                    for inv in invs[:3]:
                        if isinstance(inv, dict):
                            lines.append(
                                f"    Invalidation: {inv.get('metric', '?')} "
                                f"{inv.get('operator', '?')} {inv.get('threshold', '?')}"
                            )

            sections.append("\n".join(lines))

        return "\n\n".join(sections) if sections else "No beliefs available."

    def _format_regime(self, regime_state: Any) -> str:
        """Format regime state for prompt inclusion."""
        if not isinstance(regime_state, dict):
            return "No regime context available."

        lines = [
            f"Statistical Regime: {regime_state.get('statistical_regime', 'UNKNOWN')}",
            f"Regime Confidence: {regime_state.get('regime_confidence', 'N/A')}",
            f"Risk Mode: {regime_state.get('system_risk_mode', 'NORMAL')}",
        ]

        features = regime_state.get("features_used", {})
        if isinstance(features, dict) and features:
            feature_strs = [f"{k}={v}" for k, v in features.items()]
            lines.append(f"Features: {', '.join(feature_strs)}")

        narr = regime_state.get("narrative_overlay", {})
        if isinstance(narr, dict) and narr.get("label"):
            lines.append(f"Narrative: {narr['label']} (alignment: {narr.get('regime_alignment', 'N/A')})")

        return "\n".join(lines)

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
