"""SHADOW-EXIT: Shadow Exit Tracking Agent.

Compares COGNIT-EXIT recommendations against EXEC-CAPTURE decisions
to detect agreement/divergence between the two exit systems. Tracks
positions where exit signals are developing but not yet triggered.

Spec Reference: Technical Spec v2.3, Phase 3

Classification: FROZEN — zero LLM calls. Pure computation.

Key concepts:
  - EXEC-CAPTURE has SUPREMACY: its decisions override COGNIT-EXIT
  - SHADOW-EXIT provides early warning when exit signals are developing
  - Agreement between COGNIT-EXIT and EXEC-CAPTURE increases exit probability
  - Divergence is flagged for governance oversight

Input: AgentContext with metadata:
  - metadata["exit_assessments"]: list of ExitAssessment dicts from COGNIT-EXIT
  - metadata["capture_decisions"]: list of CaptureDecision dicts from EXEC-CAPTURE
  - metadata["shadow_history"]: dict of ticker -> days_in_shadow (prior state)

Output: ShadowExitOutput with per-position shadow tracking.
"""

from datetime import datetime, timezone
from typing import Any, Optional

import structlog

from providence.agents.base import AgentContext, AgentStatus, BaseAgent, HealthStatus
from providence.exceptions import AgentProcessingError
from providence.schemas.exit import ShadowExitOutput, ShadowExitSignal

logger = structlog.get_logger()

# Exit action mapping for agreement detection
# COGNIT-EXIT actions: HOLD, REDUCE, EXIT
# EXEC-CAPTURE actions: HOLD, TRIM, CLOSE
EXIT_DIRECTION_MAP = {
    "HOLD": 0,    # No exit signal
    "REDUCE": 1,  # Partial exit
    "TRIM": 1,    # Partial exit (CAPTURE equivalent)
    "EXIT": 2,    # Full exit
    "CLOSE": 2,   # Full exit (CAPTURE equivalent)
}

# Exit probability weights
COGNIT_EXIT_WEIGHT = 0.35  # COGNIT-EXIT contribution to exit probability
CAPTURE_WEIGHT = 0.65  # EXEC-CAPTURE contribution (higher due to supremacy)

# Shadow day tracking
MAX_SHADOW_DAYS = 30  # Cap shadow days


def compute_agreement(
    cognit_action: str,
    capture_action: str,
) -> tuple[bool, str]:
    """Determine if COGNIT-EXIT and EXEC-CAPTURE agree on exit direction.

    Args:
        cognit_action: COGNIT-EXIT recommendation (HOLD, REDUCE, EXIT).
        capture_action: EXEC-CAPTURE decision (HOLD, TRIM, CLOSE).

    Returns:
        Tuple of (signals_agree, divergence_reason).
    """
    cognit_level = EXIT_DIRECTION_MAP.get(cognit_action, 0)
    capture_level = EXIT_DIRECTION_MAP.get(capture_action, 0)

    # Agreement: both at same level
    if cognit_level == capture_level:
        return True, ""

    # Divergence
    if cognit_level > capture_level:
        return False, (
            f"COGNIT-EXIT recommends {cognit_action} but "
            f"EXEC-CAPTURE says {capture_action} (CAPTURE has supremacy)"
        )
    else:
        return False, (
            f"EXEC-CAPTURE says {capture_action} but "
            f"COGNIT-EXIT only recommends {cognit_action}"
        )


def compute_exit_probability(
    cognit_action: str,
    cognit_confidence: float,
    capture_action: str,
    capture_confidence: float,
) -> float:
    """Compute combined exit probability from both systems.

    EXEC-CAPTURE has higher weight due to supremacy.

    Args:
        cognit_action: COGNIT-EXIT action.
        cognit_confidence: COGNIT-EXIT exit confidence.
        capture_action: EXEC-CAPTURE action.
        capture_confidence: EXEC-CAPTURE exit confidence.

    Returns:
        Combined exit probability [0, 1].
    """
    cognit_level = EXIT_DIRECTION_MAP.get(cognit_action, 0) / 2.0
    capture_level = EXIT_DIRECTION_MAP.get(capture_action, 0) / 2.0

    # Weight by action level and confidence
    cognit_signal = cognit_level * max(cognit_confidence, 0.1)
    capture_signal = capture_level * max(capture_confidence, 0.1)

    combined = (
        COGNIT_EXIT_WEIGHT * cognit_signal
        + CAPTURE_WEIGHT * capture_signal
    )

    return round(max(0.0, min(1.0, combined)), 4)


def build_shadow_signal(
    ticker: str,
    exit_assessment: dict[str, Any] | None,
    capture_decision: dict[str, Any] | None,
    shadow_history: dict[str, int],
) -> dict[str, Any]:
    """Build a shadow exit signal for a single position.

    Args:
        ticker: Ticker symbol.
        exit_assessment: COGNIT-EXIT assessment dict (or None).
        capture_decision: EXEC-CAPTURE decision dict (or None).
        shadow_history: Dict of ticker -> prior days_in_shadow.

    Returns:
        Dict with shadow signal fields.
    """
    cognit_action = "HOLD"
    cognit_confidence = 0.0
    capture_action = "HOLD"
    capture_confidence = 0.0

    if exit_assessment and isinstance(exit_assessment, dict):
        cognit_action = exit_assessment.get("exit_action", "HOLD")
        cognit_confidence = float(exit_assessment.get("exit_confidence", 0.0))

    if capture_decision and isinstance(capture_decision, dict):
        capture_action = capture_decision.get("action", "HOLD")
        capture_confidence = float(capture_decision.get("exit_confidence", 0.0))

    signals_agree, divergence_reason = compute_agreement(
        cognit_action, capture_action,
    )
    exit_prob = compute_exit_probability(
        cognit_action, cognit_confidence,
        capture_action, capture_confidence,
    )

    # Track shadow days: increment if any exit signal, reset if both HOLD
    prior_days = shadow_history.get(ticker, 0)
    any_signal = (cognit_action != "HOLD" or capture_action != "HOLD")
    if any_signal:
        days_in_shadow = min(prior_days + 1, MAX_SHADOW_DAYS)
    else:
        days_in_shadow = 0

    return {
        "ticker": ticker,
        "cognit_exit_action": cognit_action,
        "capture_action": capture_action,
        "signals_agree": signals_agree,
        "exit_probability": exit_prob,
        "divergence_reason": divergence_reason,
        "days_in_shadow": days_in_shadow,
    }


class ShadowExit(BaseAgent[ShadowExitOutput]):
    """Shadow exit tracking agent.

    Compares COGNIT-EXIT and EXEC-CAPTURE decisions to detect agreement,
    divergence, and developing exit signals. Provides early warning
    for governance oversight.

    EXEC-CAPTURE supremacy: its decisions always override.

    FROZEN: Zero LLM calls.
    """

    CONSUMED_DATA_TYPES: set = set()

    def __init__(self) -> None:
        super().__init__(
            agent_id="SHADOW-EXIT",
            agent_type="exit",
            version="1.0.0",
        )
        self._last_run: Optional[datetime] = None
        self._last_success: Optional[datetime] = None
        self._error_count_24h: int = 0

    async def process(self, context: AgentContext) -> ShadowExitOutput:
        """Track shadow exit signals across positions.

        Steps:
          1. EXTRACT exit assessments, capture decisions, history
          2. MATCH positions across both systems
          3. COMPUTE agreement, exit probability, shadow tracking
          4. EMIT ShadowExitOutput

        Args:
            context: AgentContext with exit and capture data.

        Returns:
            ShadowExitOutput with per-position shadow signals.
        """
        self._last_run = datetime.now(timezone.utc)

        try:
            log = logger.bind(
                agent_id=self.agent_id,
                context_hash=context.context_window_hash,
            )
            log.info("Starting shadow exit tracking")

            exit_assessments = context.metadata.get("exit_assessments", [])
            capture_decisions = context.metadata.get("capture_decisions", [])
            shadow_history = context.metadata.get("shadow_history", {})

            if not isinstance(exit_assessments, list):
                exit_assessments = []
            if not isinstance(capture_decisions, list):
                capture_decisions = []
            if not isinstance(shadow_history, dict):
                shadow_history = {}

            # Index by ticker
            exit_by_ticker: dict[str, dict] = {}
            for ea in exit_assessments:
                if isinstance(ea, dict) and "ticker" in ea:
                    exit_by_ticker[ea["ticker"]] = ea

            capture_by_ticker: dict[str, dict] = {}
            for cd in capture_decisions:
                if isinstance(cd, dict) and "ticker" in cd:
                    capture_by_ticker[cd["ticker"]] = cd

            # Union of all tickers
            all_tickers = set(exit_by_ticker.keys()) | set(capture_by_ticker.keys())

            signals: list[ShadowExitSignal] = []
            agreeing = 0
            diverging = 0

            for ticker in sorted(all_tickers):
                signal_dict = build_shadow_signal(
                    ticker,
                    exit_by_ticker.get(ticker),
                    capture_by_ticker.get(ticker),
                    shadow_history,
                )
                signals.append(ShadowExitSignal(**signal_dict))

                if signal_dict["signals_agree"]:
                    agreeing += 1
                else:
                    diverging += 1

            output = ShadowExitOutput(
                agent_id=self.agent_id,
                timestamp=datetime.now(timezone.utc),
                context_window_hash=context.context_window_hash,
                signals=signals,
                total_positions=len(signals),
                positions_agreeing=agreeing,
                positions_diverging=diverging,
            )

            self._last_success = datetime.now(timezone.utc)
            log.info(
                "Shadow exit tracking complete",
                total=len(signals),
                agreeing=agreeing,
                diverging=diverging,
            )
            return output

        except AgentProcessingError:
            raise
        except Exception as e:
            self._error_count_24h += 1
            raise AgentProcessingError(
                message=f"SHADOW-EXIT processing failed: {e}",
                agent_id=self.agent_id,
            ) from e

    def get_health(self) -> HealthStatus:
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
