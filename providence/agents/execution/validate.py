"""EXEC-VALIDATE: Pre-Trade Validation Agent.

Validates PositionProposals against constraints before routing.
Checks position limits, sector concentrations, exposure caps,
and risk mode compatibility.

Spec Reference: Technical Spec v2.3, Section 5.1

Classification: FROZEN — zero LLM calls. Pure computation.

Input: AgentContext with proposal and regime in metadata:
  - metadata["proposal"]: serialized PositionProposal dict
  - metadata["regime_state"]: serialized RegimeStateObject dict

Output: ValidatedProposal with per-position approval/rejection.
"""

from datetime import datetime, timezone
from typing import Any, Optional
from uuid import UUID, uuid4

import structlog

from providence.agents.base import AgentContext, AgentStatus, BaseAgent, HealthStatus
from providence.exceptions import AgentProcessingError
from providence.schemas.enums import Action, Direction
from providence.schemas.execution import ValidatedProposal, ValidationResult

logger = structlog.get_logger()

# Per-risk-mode validation limits
VALIDATION_LIMITS: dict[str, dict[str, float]] = {
    "NORMAL": {
        "max_position_weight": 0.10,
        "max_gross_exposure": 1.60,
        "max_sector_pct": 0.35,
        "min_confidence": 0.20,
    },
    "CAUTIOUS": {
        "max_position_weight": 0.08,
        "max_gross_exposure": 1.20,
        "max_sector_pct": 0.30,
        "min_confidence": 0.25,
    },
    "DEFENSIVE": {
        "max_position_weight": 0.05,
        "max_gross_exposure": 0.80,
        "max_sector_pct": 0.25,
        "min_confidence": 0.30,
    },
    "HALTED": {
        "max_position_weight": 0.0,
        "max_gross_exposure": 0.0,
        "max_sector_pct": 0.0,
        "min_confidence": 1.0,
    },
}


def validate_position(
    pos: dict,
    limits: dict[str, float],
    sector_totals: dict[str, float],
    cumulative_gross: float,
) -> tuple[bool, list[str], float]:
    """Validate a single proposed position against constraints.

    Args:
        pos: Serialized ProposedPosition dict.
        limits: Risk-mode-specific limits.
        sector_totals: Running sector concentration totals.
        cumulative_gross: Running gross exposure total.

    Returns:
        Tuple of (approved, rejection_reasons, adjusted_weight).
    """
    reasons: list[str] = []
    weight = float(pos.get("target_weight", 0.0))
    confidence = float(pos.get("confidence", 0.0))
    ticker = pos.get("ticker", "UNKNOWN")
    sector = pos.get("sector", "Unknown")
    action = pos.get("action", "")

    # Check: ticker must be present
    if not ticker or ticker == "UNKNOWN":
        reasons.append("Missing ticker")
        return False, reasons, 0.0

    # Check: valid action
    valid_actions = {a.value for a in Action}
    if action not in valid_actions:
        reasons.append(f"Invalid action: {action}")
        return False, reasons, 0.0

    # Check: minimum confidence
    if confidence < limits["min_confidence"]:
        reasons.append(
            f"Confidence {confidence:.2f} below minimum {limits['min_confidence']:.2f}"
        )
        return False, reasons, 0.0

    # Check: position weight
    max_weight = limits["max_position_weight"]
    adjusted = min(weight, max_weight)
    if weight > max_weight:
        reasons.append(
            f"Weight {weight:.4f} exceeds limit {max_weight:.4f}, adjusted to {adjusted:.4f}"
        )
        # Adjusted, not rejected

    # Check: gross exposure
    if cumulative_gross + adjusted > limits["max_gross_exposure"]:
        remaining = max(0.0, limits["max_gross_exposure"] - cumulative_gross)
        if remaining < 0.001:
            reasons.append("Gross exposure limit reached")
            return False, reasons, 0.0
        adjusted = min(adjusted, remaining)
        reasons.append(f"Weight reduced to {adjusted:.4f} for gross exposure limit")

    # Check: sector concentration
    current_sector = sector_totals.get(sector, 0.0)
    if current_sector + adjusted > limits["max_sector_pct"]:
        sector_remaining = max(0.0, limits["max_sector_pct"] - current_sector)
        if sector_remaining < 0.001:
            reasons.append(f"Sector {sector} concentration limit reached")
            return False, reasons, 0.0
        adjusted = min(adjusted, sector_remaining)
        reasons.append(f"Weight reduced to {adjusted:.4f} for sector limit")

    # If weight was reduced but still positive, approve with adjustment
    approved = adjusted >= 0.001
    if not approved:
        reasons.append("Adjusted weight too small")

    return approved, reasons, round(adjusted, 6)


class ExecValidate(BaseAgent[ValidatedProposal]):
    """Pre-trade validation agent.

    Validates each proposed position against risk-mode-specific
    constraints: position limits, sector caps, gross exposure,
    minimum confidence thresholds.

    FROZEN: Zero LLM calls.
    """

    CONSUMED_DATA_TYPES: set = set()

    def __init__(self) -> None:
        super().__init__(
            agent_id="EXEC-VALIDATE",
            agent_type="execution",
            version="1.0.0",
        )
        self._last_run: Optional[datetime] = None
        self._last_success: Optional[datetime] = None
        self._error_count_24h: int = 0

    async def process(self, context: AgentContext) -> ValidatedProposal:
        """Validate proposed positions against constraints.

        Steps:
          1. EXTRACT proposal and regime from metadata
          2. DETERMINE risk mode limits
          3. VALIDATE each position sequentially
          4. EMIT ValidatedProposal

        Args:
            context: AgentContext with proposal in metadata.

        Returns:
            ValidatedProposal with per-position results.
        """
        self._last_run = datetime.now(timezone.utc)

        try:
            log = logger.bind(agent_id=self.agent_id, context_hash=context.context_window_hash)
            log.info("Starting pre-trade validation")

            proposal = context.metadata.get("proposal", {})
            regime_state = context.metadata.get("regime_state", {})
            risk_mode = "NORMAL"
            if isinstance(regime_state, dict):
                risk_mode = regime_state.get("system_risk_mode", "NORMAL")

            limits = VALIDATION_LIMITS.get(risk_mode, VALIDATION_LIMITS["NORMAL"])

            positions = []
            if isinstance(proposal, dict):
                positions = proposal.get("proposals", [])
            if not isinstance(positions, list):
                positions = []

            # Validate sequentially, tracking running totals
            results: list[ValidationResult] = []
            sector_totals: dict[str, float] = {}
            cumulative_gross = 0.0
            approved_count = 0
            rejected_count = 0

            for pos in positions:
                if not isinstance(pos, dict):
                    continue

                approved, reasons, adjusted = validate_position(
                    pos, limits, sector_totals, cumulative_gross,
                )

                ticker = pos.get("ticker", "UNKNOWN")
                sector = pos.get("sector", "Unknown")
                action_str = pos.get("action", "CLOSE")
                direction_str = pos.get("direction", "NEUTRAL")

                try:
                    action = Action(action_str)
                except ValueError:
                    action = Action.CLOSE
                try:
                    direction = Direction(direction_str)
                except ValueError:
                    direction = Direction.NEUTRAL

                intent_id = pos.get("source_intent_id", str(uuid4()))
                try:
                    source_uuid = UUID(str(intent_id))
                except (ValueError, TypeError):
                    source_uuid = uuid4()

                results.append(ValidationResult(
                    ticker=ticker,
                    action=action,
                    direction=direction,
                    target_weight=float(pos.get("target_weight", 0.0)),
                    confidence=float(pos.get("confidence", 0.0)),
                    source_intent_id=source_uuid,
                    approved=approved,
                    rejection_reasons=reasons,
                    adjusted_weight=adjusted,
                ))

                if approved:
                    approved_count += 1
                    cumulative_gross += adjusted
                    sector_totals[sector] = sector_totals.get(sector, 0.0) + adjusted
                else:
                    rejected_count += 1

            output = ValidatedProposal(
                agent_id=self.agent_id,
                timestamp=datetime.now(timezone.utc),
                context_window_hash=context.context_window_hash,
                results=results,
                approved_count=approved_count,
                rejected_count=rejected_count,
                risk_mode_applied=risk_mode,
            )

            self._last_success = datetime.now(timezone.utc)
            log.info(
                "Validation complete",
                approved=approved_count,
                rejected=rejected_count,
            )
            return output

        except AgentProcessingError:
            raise
        except Exception as e:
            self._error_count_24h += 1
            raise AgentProcessingError(
                message=f"EXEC-VALIDATE processing failed: {e}",
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
