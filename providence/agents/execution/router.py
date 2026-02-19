"""EXEC-ROUTER: Order Routing Agent.

Converts validated positions into routed orders with execution strategy,
urgency, and slippage parameters based on position characteristics and
market regime.

Spec Reference: Technical Spec v2.3, Section 5.2

Classification: FROZEN — zero LLM calls. Pure computation.

Input: AgentContext with validated_proposal and regime in metadata:
  - metadata["validated_proposal"]: serialized ValidatedProposal dict
  - metadata["regime_state"]: serialized RegimeStateObject dict

Output: RoutingPlan with RoutedOrders.
"""

from datetime import datetime, timezone
from typing import Any, Optional
from uuid import UUID, uuid4

import structlog

from providence.agents.base import AgentContext, AgentStatus, BaseAgent, HealthStatus
from providence.exceptions import AgentProcessingError
from providence.schemas.enums import Action, Direction
from providence.schemas.execution import RoutedOrder, RoutingPlan

logger = structlog.get_logger()

# Execution strategy selection rules
STRATEGY_RULES: dict[str, dict[str, Any]] = {
    "NORMAL": {
        "default_strategy": "TWAP",
        "high_confidence_strategy": "LIMIT",
        "max_slippage_bps": 50,
        "urgency_threshold": 0.70,
    },
    "CAUTIOUS": {
        "default_strategy": "TWAP",
        "high_confidence_strategy": "TWAP",
        "max_slippage_bps": 30,
        "urgency_threshold": 0.75,
    },
    "DEFENSIVE": {
        "default_strategy": "VWAP",
        "high_confidence_strategy": "VWAP",
        "max_slippage_bps": 20,
        "urgency_threshold": 0.80,
    },
    "HALTED": {
        "default_strategy": "MARKET",
        "high_confidence_strategy": "MARKET",
        "max_slippage_bps": 0,
        "urgency_threshold": 1.0,
    },
}


def determine_strategy(
    action: str,
    confidence: float,
    time_horizon: int,
    rules: dict[str, Any],
) -> str:
    """Determine execution strategy based on position characteristics.

    Args:
        action: Position action (OPEN_LONG, OPEN_SHORT, CLOSE, ADJUST).
        confidence: Position confidence [0, 1].
        time_horizon: Time horizon in days.
        rules: Risk-mode-specific strategy rules.

    Returns:
        Execution strategy string.
    """
    # CLOSE actions: use MARKET for speed
    if action == "CLOSE":
        return "MARKET"

    # Short time horizon: more urgent
    if time_horizon <= 5:
        return "MARKET"

    # High confidence: use tighter execution
    if confidence >= rules["urgency_threshold"]:
        return rules["high_confidence_strategy"]

    return rules["default_strategy"]


def determine_urgency(
    action: str,
    confidence: float,
    time_horizon: int,
) -> str:
    """Determine execution urgency.

    Args:
        action: Position action.
        confidence: Position confidence.
        time_horizon: Time horizon in days.

    Returns:
        Urgency level: LOW, NORMAL, HIGH, or IMMEDIATE.
    """
    if action == "CLOSE":
        return "IMMEDIATE"
    if time_horizon <= 3:
        return "HIGH"
    if confidence >= 0.75:
        return "HIGH"
    if time_horizon >= 60:
        return "LOW"
    return "NORMAL"


def compute_max_slippage(
    action: str,
    confidence: float,
    base_slippage: int,
) -> int:
    """Compute maximum acceptable slippage for an order.

    Args:
        action: Position action.
        confidence: Position confidence.
        base_slippage: Base max slippage from risk mode.

    Returns:
        Max slippage in basis points.
    """
    if action == "CLOSE":
        return base_slippage * 2  # Allow more slippage for exits

    # Higher confidence → tighter slippage tolerance
    if confidence >= 0.70:
        return max(10, int(base_slippage * 0.6))

    return base_slippage


class ExecRouter(BaseAgent[RoutingPlan]):
    """Order routing agent.

    Converts validated positions into routed orders with execution
    strategy, urgency, and slippage parameters.

    FROZEN: Zero LLM calls.
    """

    CONSUMED_DATA_TYPES: set = set()

    def __init__(self) -> None:
        super().__init__(
            agent_id="EXEC-ROUTER",
            agent_type="execution",
            version="1.0.0",
        )
        self._last_run: Optional[datetime] = None
        self._last_success: Optional[datetime] = None
        self._error_count_24h: int = 0

    async def process(self, context: AgentContext) -> RoutingPlan:
        """Route validated positions into execution orders.

        Steps:
          1. EXTRACT validated proposal and regime from metadata
          2. DETERMINE strategy rules from risk mode
          3. ROUTE each approved position
          4. EMIT RoutingPlan

        Args:
            context: AgentContext with validated_proposal in metadata.

        Returns:
            RoutingPlan with routed orders.
        """
        self._last_run = datetime.now(timezone.utc)

        try:
            log = logger.bind(agent_id=self.agent_id, context_hash=context.context_window_hash)
            log.info("Starting order routing")

            validated = context.metadata.get("validated_proposal", {})
            regime_state = context.metadata.get("regime_state", {})
            risk_mode = "NORMAL"
            if isinstance(regime_state, dict):
                risk_mode = regime_state.get("system_risk_mode", "NORMAL")

            rules = STRATEGY_RULES.get(risk_mode, STRATEGY_RULES["NORMAL"])

            results = []
            if isinstance(validated, dict):
                results = validated.get("results", [])
            if not isinstance(results, list):
                results = []

            orders: list[RoutedOrder] = []
            for result in results:
                if not isinstance(result, dict):
                    continue
                if not result.get("approved", False):
                    continue

                ticker = result.get("ticker", "")
                action_str = result.get("action", "CLOSE")
                direction_str = result.get("direction", "NEUTRAL")
                confidence = float(result.get("confidence", 0.0))
                weight = float(result.get("adjusted_weight", result.get("target_weight", 0.0)))
                time_horizon = int(result.get("time_horizon_days", 60))

                try:
                    action = Action(action_str)
                except ValueError:
                    action = Action.CLOSE
                try:
                    direction = Direction(direction_str)
                except ValueError:
                    direction = Direction.NEUTRAL

                intent_id = result.get("source_intent_id", str(uuid4()))
                try:
                    source_uuid = UUID(str(intent_id))
                except (ValueError, TypeError):
                    source_uuid = uuid4()

                strategy = determine_strategy(
                    action_str, confidence, time_horizon, rules,
                )
                urgency = determine_urgency(action_str, confidence, time_horizon)
                slippage = compute_max_slippage(
                    action_str, confidence, rules["max_slippage_bps"],
                )

                orders.append(RoutedOrder(
                    ticker=ticker,
                    action=action,
                    direction=direction,
                    target_weight=weight,
                    confidence=confidence,
                    source_intent_id=source_uuid,
                    execution_strategy=strategy,
                    urgency=urgency,
                    time_horizon_days=time_horizon,
                    max_slippage_bps=slippage,
                ))

            output = RoutingPlan(
                agent_id=self.agent_id,
                timestamp=datetime.now(timezone.utc),
                context_window_hash=context.context_window_hash,
                orders=orders,
                total_orders=len(orders),
                risk_mode_applied=risk_mode,
            )

            self._last_success = datetime.now(timezone.utc)
            log.info("Routing complete", order_count=len(orders))
            return output

        except AgentProcessingError:
            raise
        except Exception as e:
            self._error_count_24h += 1
            raise AgentProcessingError(
                message=f"EXEC-ROUTER processing failed: {e}",
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
