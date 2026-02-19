"""EXEC-CAPTURE: Trailing Stop Management Agent.

Manages trailing stops, trim stages, and exit decisions for active
positions. EXEC-CAPTURE has SUPREMACY: its decisions override all
upstream agents including COGNIT-EXIT.

Spec Reference: Technical Spec v2.3, Section 2.9 (TrailingStopState)

Classification: FROZEN — zero LLM calls. Pure computation.

Key rules:
  - Activation: unrealized PnL > 2.0x expected_return
  - Trail: 30% NORMAL, 20% CAUTIOUS/DEFENSIVE
  - Hard giveback: > 50% of peak → mandatory CLOSE
  - Minimum hold: 5 days
  - Max 3 trim stages, then mandatory CLOSE
  - trim_pct applies to REMAINING position, not original

Input: AgentContext with active positions in metadata:
  - metadata["active_positions"]: list of position dicts
  - metadata["regime_state"]: serialized RegimeStateObject dict

Output: CaptureOutput with per-position decisions.
"""

from datetime import datetime, timezone
from typing import Any, Optional

import structlog

from providence.agents.base import AgentContext, AgentStatus, BaseAgent, HealthStatus
from providence.exceptions import AgentProcessingError
from providence.schemas.execution import CaptureDecision, CaptureOutput, TrailingStopState

logger = structlog.get_logger()

# Trailing stop parameters by risk mode
TRAIL_PARAMS: dict[str, dict[str, float]] = {
    "NORMAL": {
        "trail_pct": 0.30,
        "activation_multiplier": 2.0,
        "hard_giveback_pct": 0.50,
        "min_hold_days": 5,
    },
    "CAUTIOUS": {
        "trail_pct": 0.20,
        "activation_multiplier": 2.0,
        "hard_giveback_pct": 0.50,
        "min_hold_days": 5,
    },
    "DEFENSIVE": {
        "trail_pct": 0.20,
        "activation_multiplier": 2.0,
        "hard_giveback_pct": 0.50,
        "min_hold_days": 5,
    },
    "HALTED": {
        "trail_pct": 0.0,
        "activation_multiplier": 0.0,
        "hard_giveback_pct": 0.0,
        "min_hold_days": 0,
    },
}

# Trim stage defaults
MAX_TRIM_STAGES = 3
DEFAULT_TRIM_PCTS = [0.30, 0.40, 1.0]  # Stage 0, 1, 2+ → full close


def compute_trailing_stop(
    position: dict,
    params: dict[str, float],
) -> TrailingStopState:
    """Compute trailing stop state for a single position.

    Args:
        position: Active position dict with PnL data.
        params: Risk-mode-specific trail parameters.

    Returns:
        TrailingStopState for this position.
    """
    ticker = position.get("ticker", "UNKNOWN")
    days_held = int(position.get("days_held", 0))
    current_pnl = float(position.get("unrealized_pnl", 0.0))
    peak_pnl = float(position.get("peak_unrealized_pnl", 0.0))
    expected_return = float(position.get("expected_return", 0.0))
    trim_stage = int(position.get("trim_stage", 0))

    # Update peak
    peak_pnl = max(peak_pnl, current_pnl)

    # Check activation
    activation_threshold = expected_return * params["activation_multiplier"]
    is_active = peak_pnl > activation_threshold and activation_threshold > 0

    # Compute trigger level
    trail_pct = params["trail_pct"]
    trigger_level = peak_pnl * (1.0 - trail_pct) if is_active else 0.0

    return TrailingStopState(
        ticker=ticker,
        is_active=is_active,
        peak_unrealized_pnl=round(peak_pnl, 4),
        current_unrealized_pnl=round(current_pnl, 4),
        trail_pct=trail_pct,
        trigger_level=round(trigger_level, 4),
        days_held=days_held,
        trim_stage=min(trim_stage, MAX_TRIM_STAGES),
    )


def evaluate_position(
    position: dict,
    stop_state: TrailingStopState,
    params: dict[str, float],
) -> tuple[str, float, float, str]:
    """Evaluate whether a position should be held, trimmed, or closed.

    EXEC-CAPTURE supremacy: these decisions override all upstream agents.

    Args:
        position: Active position dict.
        stop_state: Computed trailing stop state.
        params: Risk-mode-specific trail parameters.

    Returns:
        Tuple of (action, trim_pct, exit_confidence, reason).
    """
    days_held = stop_state.days_held
    current_pnl = stop_state.current_unrealized_pnl
    peak_pnl = stop_state.peak_unrealized_pnl
    trim_stage = stop_state.trim_stage

    # Rule: Max 3 trim stages → mandatory CLOSE
    if trim_stage >= MAX_TRIM_STAGES:
        return "CLOSE", 1.0, 0.95, f"Max trim stages ({MAX_TRIM_STAGES}) reached — mandatory close"

    # Rule: System HALTED → close everything
    # (Caller should check, but enforce here too)

    # Rule: Hard giveback — lost > 50% of peak unrealized gain
    if peak_pnl > 0 and current_pnl < peak_pnl * (1.0 - params["hard_giveback_pct"]):
        return (
            "CLOSE", 1.0, 0.90,
            f"Hard giveback: current PnL {current_pnl:.4f} lost >"
            f" {params['hard_giveback_pct']*100:.0f}% of peak {peak_pnl:.4f}",
        )

    # Rule: Minimum hold period — don't trigger trailing stop before min_hold_days
    if days_held < params["min_hold_days"]:
        return "HOLD", 0.0, 0.0, f"Minimum hold period ({params['min_hold_days']} days) not reached"

    # Rule: Trailing stop activated and breached
    if stop_state.is_active and current_pnl <= stop_state.trigger_level:
        # Determine trim amount based on stage
        if trim_stage < len(DEFAULT_TRIM_PCTS):
            trim_pct = DEFAULT_TRIM_PCTS[trim_stage]
        else:
            trim_pct = 1.0  # Full close

        if trim_pct >= 1.0:
            return (
                "CLOSE", 1.0, 0.85,
                f"Trailing stop breached at stage {trim_stage} — full close",
            )
        else:
            return (
                "TRIM", trim_pct, 0.70,
                f"Trailing stop breached — trim {trim_pct*100:.0f}% of remaining (stage {trim_stage})",
            )

    # Rule: Trailing stop active but not breached
    if stop_state.is_active:
        return (
            "HOLD", 0.0, 0.0,
            f"Trailing stop active, PnL {current_pnl:.4f} above trigger {stop_state.trigger_level:.4f}",
        )

    # Default: hold
    return "HOLD", 0.0, 0.0, "Position within parameters"


class ExecCapture(BaseAgent[CaptureOutput]):
    """Trailing stop management agent.

    Manages trailing stops, trim stages, and exit decisions.
    EXEC-CAPTURE has SUPREMACY — its decisions override all upstream
    agents including COGNIT-EXIT.

    Key rules:
      - Activation at 2.0x expected return
      - Trail: 30% NORMAL, 20% CAUTIOUS/DEFENSIVE
      - Hard giveback > 50% of peak → close
      - Min hold: 5 days
      - Max 3 trim stages, then close
      - trim_pct applies to REMAINING position

    FROZEN: Zero LLM calls.
    """

    CONSUMED_DATA_TYPES: set = set()

    def __init__(self) -> None:
        super().__init__(
            agent_id="EXEC-CAPTURE",
            agent_type="execution",
            version="1.0.0",
        )
        self._last_run: Optional[datetime] = None
        self._last_success: Optional[datetime] = None
        self._error_count_24h: int = 0

    async def process(self, context: AgentContext) -> CaptureOutput:
        """Evaluate active positions for trailing stop / trim / close.

        Steps:
          1. EXTRACT active positions and regime from metadata
          2. COMPUTE trailing stop state per position
          3. EVALUATE each position for hold/trim/close
          4. EMIT CaptureOutput

        Args:
            context: AgentContext with active_positions in metadata.

        Returns:
            CaptureOutput with per-position decisions.
        """
        self._last_run = datetime.now(timezone.utc)

        try:
            log = logger.bind(agent_id=self.agent_id, context_hash=context.context_window_hash)
            log.info("Starting capture evaluation")

            active_positions = context.metadata.get("active_positions", [])
            regime_state = context.metadata.get("regime_state", {})
            risk_mode = "NORMAL"
            if isinstance(regime_state, dict):
                risk_mode = regime_state.get("system_risk_mode", "NORMAL")

            params = TRAIL_PARAMS.get(risk_mode, TRAIL_PARAMS["NORMAL"])

            if not isinstance(active_positions, list):
                active_positions = []

            # HALTED mode: close everything
            if risk_mode == "HALTED":
                log.warning("System HALTED — closing all positions")
                decisions = []
                for pos in active_positions:
                    if not isinstance(pos, dict):
                        continue
                    ticker = pos.get("ticker", "UNKNOWN")
                    stop_state = TrailingStopState(
                        ticker=ticker,
                        is_active=False,
                        days_held=int(pos.get("days_held", 0)),
                        trim_stage=int(pos.get("trim_stage", 0)),
                    )
                    decisions.append(CaptureDecision(
                        ticker=ticker,
                        action="CLOSE",
                        trim_pct=1.0,
                        exit_confidence=1.0,
                        reason="System HALTED — mandatory close",
                        trailing_stop=stop_state,
                    ))
                return CaptureOutput(
                    agent_id=self.agent_id,
                    timestamp=datetime.now(timezone.utc),
                    context_window_hash=context.context_window_hash,
                    decisions=decisions,
                    positions_held=0,
                    positions_trimmed=0,
                    positions_closed=len(decisions),
                )

            # Normal processing
            decisions: list[CaptureDecision] = []
            held = 0
            trimmed = 0
            closed = 0

            for pos in active_positions:
                if not isinstance(pos, dict):
                    continue

                stop_state = compute_trailing_stop(pos, params)
                action, trim_pct, exit_conf, reason = evaluate_position(
                    pos, stop_state, params,
                )

                decisions.append(CaptureDecision(
                    ticker=stop_state.ticker,
                    action=action,
                    trim_pct=round(trim_pct, 4),
                    exit_confidence=round(exit_conf, 4),
                    reason=reason,
                    trailing_stop=stop_state,
                ))

                if action == "HOLD":
                    held += 1
                elif action == "TRIM":
                    trimmed += 1
                elif action == "CLOSE":
                    closed += 1

            output = CaptureOutput(
                agent_id=self.agent_id,
                timestamp=datetime.now(timezone.utc),
                context_window_hash=context.context_window_hash,
                decisions=decisions,
                positions_held=held,
                positions_trimmed=trimmed,
                positions_closed=closed,
            )

            self._last_success = datetime.now(timezone.utc)
            log.info(
                "Capture evaluation complete",
                held=held,
                trimmed=trimmed,
                closed=closed,
            )
            return output

        except AgentProcessingError:
            raise
        except Exception as e:
            self._error_count_24h += 1
            raise AgentProcessingError(
                message=f"EXEC-CAPTURE processing failed: {e}",
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
