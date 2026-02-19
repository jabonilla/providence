"""COGNIT-EXIT: Exit Assessment Agent.

Evaluates active positions for exit signals by analyzing thesis health,
invalidation condition status, PnL dynamics, and regime context.

Spec Reference: Technical Spec v2.3, Section 2.8 (ExitAssessment)

Classification: ADAPTIVE — uses Claude Sonnet 4 via Anthropic API.
Subject to offline retraining. Prompt template is version-controlled.

Critical Rules:
  - COGNIT-EXIT defers CLOSE if renewal_pending AND asymmetry > 0.5
  - EXEC-CAPTURE has SUPREMACY — trailing stop overrides everything
  - exit_confidence [0-1], regret in basis points
  - regret_direction: MISSED_UPSIDE or SUFFERED_GIVEBACK

Input: AgentContext with metadata:
  - metadata["active_positions"]: list of position dicts with PnL data
  - metadata["active_beliefs"]: list of belief dicts with invalidation state
  - metadata["regime_state"]: serialized RegimeStateObject dict
  - metadata["renewal_state"]: dict of thesis_id -> renewal info

Output: ExitOutput with per-position ExitAssessments.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import structlog
import yaml

from providence.agents.base import AgentContext, AgentStatus, BaseAgent, HealthStatus
from providence.exceptions import AgentProcessingError
from providence.infra.llm_client import AnthropicClient, LLMClient
from providence.schemas.exit import ExitAssessment, ExitOutput

logger = structlog.get_logger()

# Path to prompt templates
PROMPT_DIR = Path(__file__).parent.parent.parent / "prompts"
DEFAULT_PROMPT_VERSION = "cognit_exit_v1.0.yaml"

# Valid exit actions
VALID_EXIT_ACTIONS = {"HOLD", "REDUCE", "EXIT"}
VALID_REGRET_DIRECTIONS = {"MISSED_UPSIDE", "SUFFERED_GIVEBACK"}

# Confidence bounds
MAX_EXIT_CONFIDENCE = 0.95
MIN_EXIT_CONFIDENCE = 0.0

# Thesis health thresholds
HEALTH_SCORE_EXIT_THRESHOLD = 0.30  # Below this → strong exit signal
HEALTH_SCORE_REDUCE_THRESHOLD = 0.50  # Below this → reduce signal


def parse_exit_response(raw: str) -> list[dict[str, Any]] | None:
    """Parse the LLM response into exit assessments.

    Args:
        raw: Raw LLM response string (should be JSON).

    Returns:
        List of assessment dicts, or None if parsing fails.
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

    assessments = parsed.get("assessments")
    if not isinstance(assessments, list):
        return None

    results = []
    for a in assessments:
        if not isinstance(a, dict):
            continue

        ticker = a.get("ticker")
        if not ticker or not isinstance(ticker, str):
            continue

        exit_action = a.get("exit_action", "HOLD")
        if exit_action not in VALID_EXIT_ACTIONS:
            exit_action = "HOLD"

        try:
            exit_conf = float(a.get("exit_confidence", 0.0))
        except (TypeError, ValueError):
            exit_conf = 0.0
        exit_conf = max(MIN_EXIT_CONFIDENCE, min(MAX_EXIT_CONFIDENCE, exit_conf))

        try:
            regret_bps = max(0.0, float(a.get("regret_estimate_bps", 0.0)))
        except (TypeError, ValueError):
            regret_bps = 0.0

        regret_dir = a.get("regret_direction", "MISSED_UPSIDE")
        if regret_dir not in VALID_REGRET_DIRECTIONS:
            regret_dir = "MISSED_UPSIDE"

        try:
            health = float(a.get("thesis_health_score", 1.0))
        except (TypeError, ValueError):
            health = 1.0
        health = max(0.0, min(1.0, health))

        rationale = str(a.get("rationale", ""))[:1000]

        results.append({
            "ticker": ticker.strip().upper(),
            "exit_action": exit_action,
            "exit_confidence": round(exit_conf, 4),
            "regret_estimate_bps": round(regret_bps, 2),
            "regret_direction": regret_dir,
            "thesis_health_score": round(health, 4),
            "rationale": rationale,
        })

    return results if results else None


def apply_renewal_deferral(
    assessment: dict[str, Any],
    renewal_state: dict[str, Any],
    active_beliefs: list[dict[str, Any]],
) -> dict[str, Any]:
    """Apply THESIS-RENEW deferral rule.

    Critical Rule: COGNIT-EXIT defers CLOSE if renewal pending AND asymmetry > 0.5.

    Args:
        assessment: Parsed exit assessment dict.
        renewal_state: Dict of thesis_id -> renewal info.
        active_beliefs: List of active belief dicts.

    Returns:
        Modified assessment dict with renewal fields populated.
    """
    ticker = assessment["ticker"]

    # Find beliefs for this ticker
    ticker_beliefs = [
        b for b in active_beliefs
        if isinstance(b, dict) and b.get("ticker") == ticker
    ]

    # Check if any thesis for this ticker has a pending renewal
    renewal_pending = False
    max_asymmetry = 0.0

    for belief in ticker_beliefs:
        thesis_id = belief.get("thesis_id", "")
        if thesis_id in renewal_state:
            renewal_info = renewal_state[thesis_id]
            if isinstance(renewal_info, dict) and renewal_info.get("is_renewed", False):
                renewal_pending = True
                asymmetry = float(renewal_info.get("asymmetry_score", 0.0))
                max_asymmetry = max(max_asymmetry, asymmetry)

    assessment["renewal_pending"] = renewal_pending
    assessment["renewal_asymmetry"] = round(max_asymmetry, 4)

    # Apply deferral rule: defer EXIT if renewal pending AND asymmetry > 0.5
    if assessment["exit_action"] == "EXIT" and renewal_pending and max_asymmetry > 0.5:
        assessment["exit_action"] = "REDUCE"
        assessment["rationale"] = (
            f"[DEFERRED] Exit deferred due to pending renewal with "
            f"asymmetry {max_asymmetry:.2f} > 0.5. " + assessment["rationale"]
        )

    return assessment


def compute_thesis_health(
    ticker: str,
    active_beliefs: list[dict[str, Any]],
) -> tuple[float, int, int]:
    """Compute thesis health score from invalidation condition state.

    Args:
        ticker: Ticker to compute health for.
        active_beliefs: List of active belief dicts.

    Returns:
        Tuple of (health_score, conditions_triggered, conditions_total).
    """
    conditions_total = 0
    conditions_triggered = 0

    for belief in active_beliefs:
        if not isinstance(belief, dict) or belief.get("ticker") != ticker:
            continue

        conditions = belief.get("invalidation_conditions", [])
        if not isinstance(conditions, list):
            continue

        for cond in conditions:
            if not isinstance(cond, dict):
                continue
            conditions_total += 1
            status = cond.get("status", "ACTIVE")
            if status == "TRIGGERED":
                conditions_triggered += 1

    if conditions_total == 0:
        return 1.0, 0, 0

    health = 1.0 - (conditions_triggered / conditions_total)
    return round(health, 4), conditions_triggered, conditions_total


class CognitExit(BaseAgent[ExitOutput]):
    """Exit assessment agent.

    Evaluates active positions for exit signals using LLM analysis
    of thesis health, invalidation conditions, PnL dynamics, and
    regime context.

    Critical Rules:
      - Defers EXIT if renewal_pending AND asymmetry > 0.5
      - Subject to EXEC-CAPTURE supremacy (trailing stop overrides)

    ADAPTIVE: Uses Claude Sonnet 4 for exit assessment.
    """

    CONSUMED_DATA_TYPES = set()  # Reads from metadata

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        prompt_version: str = DEFAULT_PROMPT_VERSION,
    ) -> None:
        super().__init__(
            agent_id="COGNIT-EXIT",
            agent_type="exit",
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
                "system_prompt": (
                    "You are an exit assessment analyst for an algorithmic trading system. "
                    "Evaluate active positions and determine whether they should be held, "
                    "reduced, or fully exited. Consider thesis health, invalidation "
                    "conditions, PnL dynamics, and regime context.\n\n"
                    "Respond with ONLY valid JSON. No markdown, no commentary."
                ),
                "user_prompt_template": (
                    "Evaluate the following active positions for exit:\n\n"
                    "Active Positions:\n{positions_data}\n\n"
                    "Thesis Health:\n{thesis_data}\n\n"
                    "Regime Context:\n{regime_data}\n\n"
                    "Respond with JSON in this format:\n"
                    '{{"assessments": [{{"ticker": "...", "exit_action": "HOLD|REDUCE|EXIT", '
                    '"exit_confidence": 0.0-1.0, "regret_estimate_bps": 0, '
                    '"regret_direction": "MISSED_UPSIDE|SUFFERED_GIVEBACK", '
                    '"thesis_health_score": 0.0-1.0, "rationale": "..."}}]}}'
                ),
            }

        with open(path) as f:
            return yaml.safe_load(f)

    async def process(self, context: AgentContext) -> ExitOutput:
        """Execute the exit assessment pipeline.

        Steps:
          1. EXTRACT active positions, beliefs, regime, renewal state
          2. COMPUTE thesis health per position
          3. BUILD PROMPT with position and thesis data
          4. CALL LLM for exit assessment
          5. PARSE RESPONSE and validate
          6. APPLY renewal deferral rule
          7. EMIT ExitOutput

        Args:
            context: AgentContext with position/belief data in metadata.

        Returns:
            ExitOutput with per-position exit assessments.

        Raises:
            AgentProcessingError: If processing fails.
        """
        self._last_run = datetime.now(timezone.utc)

        try:
            log = logger.bind(
                agent_id=self.agent_id,
                context_hash=context.context_window_hash,
            )
            log.info("Starting exit assessment")

            # Step 1: EXTRACT
            active_positions = context.metadata.get("active_positions", [])
            active_beliefs = context.metadata.get("active_beliefs", [])
            regime_state = context.metadata.get("regime_state", {})
            renewal_state = context.metadata.get("renewal_state", {})

            if not isinstance(active_positions, list):
                active_positions = []
            if not isinstance(active_beliefs, list):
                active_beliefs = []
            if not isinstance(regime_state, dict):
                regime_state = {}
            if not isinstance(renewal_state, dict):
                renewal_state = {}

            if not active_positions:
                log.info("No active positions to assess")
                return ExitOutput(
                    agent_id=self.agent_id,
                    timestamp=datetime.now(timezone.utc),
                    context_window_hash=context.context_window_hash,
                    assessments=[],
                    positions_hold=0,
                    positions_reduce=0,
                    positions_exit=0,
                )

            # Step 2: COMPUTE thesis health
            health_by_ticker: dict[str, tuple[float, int, int]] = {}
            for pos in active_positions:
                if isinstance(pos, dict):
                    ticker = pos.get("ticker", "")
                    if ticker and ticker not in health_by_ticker:
                        health_by_ticker[ticker] = compute_thesis_health(
                            ticker, active_beliefs,
                        )

            # Step 3: BUILD PROMPT
            system_prompt = self._prompt_config.get(
                "system_prompt",
                "You are an exit assessment analyst.",
            )
            user_prompt = self._build_user_prompt(
                active_positions, active_beliefs, regime_state, health_by_ticker,
            )

            # Step 4: CALL LLM
            log.info("Sending exit context to LLM")
            raw_response = await self._llm_client.complete(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )

            # Step 5: PARSE RESPONSE
            parsed = parse_exit_response(raw_response)
            if parsed is None:
                self._error_count_24h += 1
                raise AgentProcessingError(
                    message="Failed to parse LLM response into exit assessments",
                    agent_id=self.agent_id,
                )

            # Step 6: APPLY renewal deferral + thesis health enrichment
            assessments: list[ExitAssessment] = []
            hold_count = 0
            reduce_count = 0
            exit_count = 0

            for assessment_dict in parsed:
                ticker = assessment_dict["ticker"]

                # Enrich with thesis health
                health, triggered, total = health_by_ticker.get(
                    ticker, (1.0, 0, 0),
                )
                assessment_dict["conditions_triggered"] = triggered
                assessment_dict["conditions_total"] = total
                # Use LLM health if provided, else computed
                if assessment_dict.get("thesis_health_score", 1.0) == 1.0 and health < 1.0:
                    assessment_dict["thesis_health_score"] = health

                # Apply renewal deferral
                assessment_dict = apply_renewal_deferral(
                    assessment_dict, renewal_state, active_beliefs,
                )

                assessments.append(ExitAssessment(**assessment_dict))

                if assessment_dict["exit_action"] == "HOLD":
                    hold_count += 1
                elif assessment_dict["exit_action"] == "REDUCE":
                    reduce_count += 1
                elif assessment_dict["exit_action"] == "EXIT":
                    exit_count += 1

            # Step 7: EMIT
            output = ExitOutput(
                agent_id=self.agent_id,
                timestamp=datetime.now(timezone.utc),
                context_window_hash=context.context_window_hash,
                assessments=assessments,
                positions_hold=hold_count,
                positions_reduce=reduce_count,
                positions_exit=exit_count,
            )

            self._last_success = datetime.now(timezone.utc)
            log.info(
                "Exit assessment complete",
                hold=hold_count,
                reduce=reduce_count,
                exit=exit_count,
            )
            return output

        except AgentProcessingError:
            raise
        except Exception as e:
            self._error_count_24h += 1
            raise AgentProcessingError(
                message=f"COGNIT-EXIT processing failed: {e}",
                agent_id=self.agent_id,
            ) from e

    def _build_user_prompt(
        self,
        positions: list[dict[str, Any]],
        beliefs: list[dict[str, Any]],
        regime: dict[str, Any],
        health_by_ticker: dict[str, tuple[float, int, int]],
    ) -> str:
        """Build the user prompt from context data."""
        template = self._prompt_config.get(
            "user_prompt_template",
            (
                "Evaluate the following active positions for exit:\n\n"
                "Active Positions:\n{positions_data}\n\n"
                "Thesis Health:\n{thesis_data}\n\n"
                "Regime Context:\n{regime_data}"
            ),
        )

        positions_data = json.dumps(positions, indent=2, default=str)
        thesis_data = json.dumps(
            {
                ticker: {
                    "health_score": h[0],
                    "conditions_triggered": h[1],
                    "conditions_total": h[2],
                }
                for ticker, h in health_by_ticker.items()
            },
            indent=2,
        )
        regime_data = json.dumps(regime, indent=2, default=str)

        return template.format(
            positions_data=positions_data,
            thesis_data=thesis_data,
            regime_data=regime_data,
        )

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
