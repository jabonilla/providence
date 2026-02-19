"""EXEC-GUARDIAN: Kill Switch / Circuit Breaker Agent.

Final safety check before order execution. Enforces system-wide halts,
per-order circuit breakers, and drawdown proximity checks.

Spec Reference: Technical Spec v2.3, Section 5.3

Classification: FROZEN — zero LLM calls. Pure computation.

Input: AgentContext with routing_plan and regime in metadata:
  - metadata["routing_plan"]: serialized RoutingPlan dict
  - metadata["regime_state"]: serialized RegimeStateObject dict
  - metadata["portfolio_state"]: optional current portfolio state dict

Output: GuardianVerdict with per-order approval/halt decisions.
"""

from datetime import datetime, timezone
from typing import Any, Optional
from uuid import UUID, uuid4

import structlog

from providence.agents.base import AgentContext, AgentStatus, BaseAgent, HealthStatus
from providence.exceptions import AgentProcessingError
from providence.schemas.execution import GuardianCheck, GuardianVerdict

logger = structlog.get_logger()

# Circuit breaker thresholds
CIRCUIT_BREAKERS: dict[str, dict[str, float]] = {
    "NORMAL": {
        "max_daily_trades": 50,
        "max_daily_turnover_pct": 40.0,
        "drawdown_halt_pct": 5.0,
    },
    "CAUTIOUS": {
        "max_daily_trades": 30,
        "max_daily_turnover_pct": 25.0,
        "drawdown_halt_pct": 3.0,
    },
    "DEFENSIVE": {
        "max_daily_trades": 15,
        "max_daily_turnover_pct": 15.0,
        "drawdown_halt_pct": 2.0,
    },
    "HALTED": {
        "max_daily_trades": 0,
        "max_daily_turnover_pct": 0.0,
        "drawdown_halt_pct": 0.0,
    },
}


def check_system_halt(
    risk_mode: str,
    portfolio_state: dict,
    breakers: dict[str, float],
) -> tuple[bool, str]:
    """Check whether a system-wide halt should be triggered.

    Args:
        risk_mode: Current SystemRiskMode.
        portfolio_state: Current portfolio state with drawdown info.
        breakers: Circuit breaker thresholds.

    Returns:
        Tuple of (should_halt, reason).
    """
    if risk_mode == "HALTED":
        return True, "System in HALTED risk mode — all trading suspended"

    # Check portfolio drawdown
    drawdown_pct = float(portfolio_state.get("daily_drawdown_pct", 0.0))
    if drawdown_pct >= breakers["drawdown_halt_pct"]:
        return True, (
            f"Daily drawdown {drawdown_pct:.2f}% exceeds "
            f"halt threshold {breakers['drawdown_halt_pct']:.2f}%"
        )

    # Check daily trade count
    daily_trades = int(portfolio_state.get("daily_trade_count", 0))
    if daily_trades >= breakers["max_daily_trades"]:
        return True, (
            f"Daily trade count {daily_trades} exceeds "
            f"limit {int(breakers['max_daily_trades'])}"
        )

    return False, ""


def check_order(
    order: dict,
    portfolio_state: dict,
    breakers: dict[str, float],
    daily_turnover: float,
) -> tuple[bool, str]:
    """Check whether a single order should be halted.

    Args:
        order: Serialized RoutedOrder dict.
        portfolio_state: Current portfolio state.
        breakers: Circuit breaker thresholds.
        daily_turnover: Running daily turnover percentage.

    Returns:
        Tuple of (approved, halt_reason).
    """
    weight = float(order.get("target_weight", 0.0))

    # Check turnover
    new_turnover = daily_turnover + weight * 100.0
    if new_turnover > breakers["max_daily_turnover_pct"]:
        return False, (
            f"Daily turnover {new_turnover:.1f}% would exceed "
            f"limit {breakers['max_daily_turnover_pct']:.1f}%"
        )

    return True, ""


class ExecGuardian(BaseAgent[GuardianVerdict]):
    """Kill switch / circuit breaker agent.

    Final safety check before execution. Enforces system-wide halts
    and per-order circuit breakers.

    FROZEN: Zero LLM calls.
    """

    CONSUMED_DATA_TYPES: set = set()

    def __init__(self) -> None:
        super().__init__(
            agent_id="EXEC-GUARDIAN",
            agent_type="execution",
            version="1.0.0",
        )
        self._last_run: Optional[datetime] = None
        self._last_success: Optional[datetime] = None
        self._error_count_24h: int = 0

    async def process(self, context: AgentContext) -> GuardianVerdict:
        """Run kill-switch and circuit breaker checks.

        Steps:
          1. EXTRACT routing plan, regime, portfolio state from metadata
          2. CHECK system-wide halt conditions
          3. CHECK per-order circuit breakers
          4. EMIT GuardianVerdict

        Args:
            context: AgentContext with routing_plan in metadata.

        Returns:
            GuardianVerdict with per-order checks.
        """
        self._last_run = datetime.now(timezone.utc)

        try:
            log = logger.bind(agent_id=self.agent_id, context_hash=context.context_window_hash)
            log.info("Starting guardian checks")

            routing_plan = context.metadata.get("routing_plan", {})
            regime_state = context.metadata.get("regime_state", {})
            portfolio_state = context.metadata.get("portfolio_state", {})
            if not isinstance(portfolio_state, dict):
                portfolio_state = {}

            risk_mode = "NORMAL"
            if isinstance(regime_state, dict):
                risk_mode = regime_state.get("system_risk_mode", "NORMAL")

            breakers = CIRCUIT_BREAKERS.get(risk_mode, CIRCUIT_BREAKERS["NORMAL"])

            # Step 2: System-wide halt check
            system_halt, halt_reason = check_system_halt(
                risk_mode, portfolio_state, breakers,
            )

            orders = []
            if isinstance(routing_plan, dict):
                orders = routing_plan.get("orders", [])
            if not isinstance(orders, list):
                orders = []

            # Step 3: Per-order checks
            checks: list[GuardianCheck] = []
            approved_count = 0
            halted_count = 0
            daily_turnover = float(portfolio_state.get("daily_turnover_pct", 0.0))

            for order in orders:
                if not isinstance(order, dict):
                    continue

                ticker = order.get("ticker", "UNKNOWN")
                order_id = order.get("order_id", str(uuid4()))
                try:
                    order_uuid = UUID(str(order_id))
                except (ValueError, TypeError):
                    order_uuid = uuid4()

                if system_halt:
                    checks.append(GuardianCheck(
                        order_id=order_uuid,
                        ticker=ticker,
                        approved=False,
                        halt_reason=halt_reason,
                    ))
                    halted_count += 1
                else:
                    approved, reason = check_order(
                        order, portfolio_state, breakers, daily_turnover,
                    )
                    checks.append(GuardianCheck(
                        order_id=order_uuid,
                        ticker=ticker,
                        approved=approved,
                        halt_reason=reason,
                    ))
                    if approved:
                        approved_count += 1
                        daily_turnover += float(order.get("target_weight", 0.0)) * 100.0
                    else:
                        halted_count += 1

            output = GuardianVerdict(
                agent_id=self.agent_id,
                timestamp=datetime.now(timezone.utc),
                context_window_hash=context.context_window_hash,
                checks=checks,
                system_halt=system_halt,
                halt_reason=halt_reason if system_halt else "",
                approved_count=approved_count,
                halted_count=halted_count,
                risk_mode_applied=risk_mode,
            )

            self._last_success = datetime.now(timezone.utc)
            log.info(
                "Guardian checks complete",
                system_halt=system_halt,
                approved=approved_count,
                halted=halted_count,
            )
            return output

        except AgentProcessingError:
            raise
        except Exception as e:
            self._error_count_24h += 1
            raise AgentProcessingError(
                message=f"EXEC-GUARDIAN processing failed: {e}",
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
