"""GOVERN-POLICY: Policy Enforcement Agent.

Enforces capital constraints, position limits, maturity compliance,
and risk mode rules. Detects violations and flags them for human review.

Spec Reference: Technical Spec v2.3, Phase 4 (Governance)

Classification: FROZEN — zero LLM calls. Pure computation.

Policies enforced:
  1. SEED tier cannot have live execution
  2. Gross exposure must not exceed tier limit
  3. Single sector concentration must not exceed tier limit
  4. Position count must not exceed tier limit
  5. SHADOW agents must not contribute to live positions
  6. HALTED risk mode must have zero new orders

Input: AgentContext with metadata:
  - metadata["current_tier"]: str
  - metadata["tier_constraints"]: dict of constraints
  - metadata["active_positions"]: list of position dicts
  - metadata["agent_maturity_records"]: list of maturity record dicts
  - metadata["current_risk_mode"]: str
  - metadata["pending_orders"]: list of pending order dicts
  - metadata["gross_exposure_pct"]: float
  - metadata["sector_exposures"]: dict of sector -> float

Output: PolicyOutput with violations and enforcement results.
"""

from datetime import datetime, timezone
from typing import Any, Optional

import structlog

from providence.agents.base import AgentContext, AgentStatus, BaseAgent, HealthStatus
from providence.exceptions import AgentProcessingError
from providence.schemas.enums import IncidentSeverity
from providence.schemas.governance import PolicyOutput, PolicyViolation

logger = structlog.get_logger()


def check_seed_execution(
    tier: str,
    pending_orders: list[dict[str, Any]],
) -> list[PolicyViolation]:
    """Check that SEED tier has no live execution.

    Args:
        tier: Current capital tier.
        pending_orders: List of pending order dicts.

    Returns:
        List of violations if SEED tier has pending orders.
    """
    violations: list[PolicyViolation] = []
    if tier == "SEED" and len(pending_orders) > 0:
        violations.append(PolicyViolation(
            policy_name="SEED_NO_EXECUTION",
            severity=IncidentSeverity.CRITICAL,
            description=f"SEED tier has {len(pending_orders)} pending orders — live execution prohibited",
            auto_enforced=True,
        ))
    return violations


def check_exposure_limits(
    gross_exposure: float,
    max_gross: float,
) -> list[PolicyViolation]:
    """Check gross exposure against tier limit.

    Args:
        gross_exposure: Current gross exposure percentage.
        max_gross: Maximum allowed gross exposure.

    Returns:
        List of violations if exposure exceeds limit.
    """
    violations: list[PolicyViolation] = []
    if max_gross > 0 and gross_exposure > max_gross:
        violations.append(PolicyViolation(
            policy_name="GROSS_EXPOSURE_LIMIT",
            severity=IncidentSeverity.CRITICAL,
            description=f"Gross exposure {gross_exposure:.2%} exceeds tier limit {max_gross:.2%}",
            auto_enforced=False,
        ))
    return violations


def check_sector_concentration(
    sector_exposures: dict[str, float],
    max_sector: float,
) -> list[PolicyViolation]:
    """Check sector concentrations against tier limit.

    Args:
        sector_exposures: Dict of sector -> exposure percentage.
        max_sector: Maximum allowed single sector exposure.

    Returns:
        List of violations for sectors exceeding limit.
    """
    violations: list[PolicyViolation] = []
    if max_sector <= 0:
        return violations

    for sector, exposure in sector_exposures.items():
        if exposure > max_sector:
            violations.append(PolicyViolation(
                policy_name="SECTOR_CONCENTRATION_LIMIT",
                severity=IncidentSeverity.WARNING,
                description=f"Sector {sector} exposure {exposure:.2%} exceeds limit {max_sector:.2%}",
                auto_enforced=False,
            ))
    return violations


def check_position_count(
    active_count: int,
    max_positions: int,
) -> list[PolicyViolation]:
    """Check position count against tier limit.

    Args:
        active_count: Current number of active positions.
        max_positions: Maximum allowed positions.

    Returns:
        List of violations if position count exceeds limit.
    """
    violations: list[PolicyViolation] = []
    if max_positions > 0 and active_count > max_positions:
        violations.append(PolicyViolation(
            policy_name="POSITION_COUNT_LIMIT",
            severity=IncidentSeverity.WARNING,
            description=f"{active_count} positions exceed tier limit of {max_positions}",
            auto_enforced=False,
        ))
    return violations


def check_shadow_agent_leakage(
    maturity_records: list[dict[str, Any]],
    active_positions: list[dict[str, Any]],
) -> list[PolicyViolation]:
    """Check that SHADOW agents don't contribute to live positions.

    Args:
        maturity_records: List of agent maturity record dicts.
        active_positions: List of active position dicts.

    Returns:
        List of violations if shadow agent outputs appear in live positions.
    """
    violations: list[PolicyViolation] = []

    shadow_agents = set()
    for rec in maturity_records:
        if isinstance(rec, dict) and rec.get("current_stage") == "SHADOW":
            shadow_agents.add(rec.get("agent_id", ""))

    if not shadow_agents:
        return violations

    for pos in active_positions:
        if not isinstance(pos, dict):
            continue
        contributing = pos.get("contributing_agents", [])
        if not isinstance(contributing, list):
            continue
        for agent_id in contributing:
            if agent_id in shadow_agents:
                ticker = pos.get("ticker", "UNKNOWN")
                violations.append(PolicyViolation(
                    policy_name="SHADOW_AGENT_LEAKAGE",
                    severity=IncidentSeverity.CRITICAL,
                    description=f"SHADOW agent {agent_id} contributed to live position {ticker}",
                    violating_agent_id=agent_id,
                    auto_enforced=True,
                ))
    return violations


def check_halted_orders(
    risk_mode: str,
    pending_orders: list[dict[str, Any]],
) -> list[PolicyViolation]:
    """Check that HALTED mode has no new orders.

    Args:
        risk_mode: Current risk mode.
        pending_orders: List of pending order dicts.

    Returns:
        List of violations if HALTED mode has pending orders.
    """
    violations: list[PolicyViolation] = []
    if risk_mode == "HALTED" and len(pending_orders) > 0:
        violations.append(PolicyViolation(
            policy_name="HALTED_NO_ORDERS",
            severity=IncidentSeverity.CRITICAL,
            description=f"HALTED mode has {len(pending_orders)} pending orders — must be zero",
            auto_enforced=True,
        ))
    return violations


class GovernPolicy(BaseAgent[PolicyOutput]):
    """Policy enforcement agent.

    Enforces capital constraints, position limits, maturity compliance,
    and risk mode rules. Detects violations and flags for human review.

    FROZEN: Zero LLM calls. Pure computation.
    """

    CONSUMED_DATA_TYPES: set = set()

    def __init__(self) -> None:
        super().__init__(
            agent_id="GOVERN-POLICY",
            agent_type="governance",
            version="1.0.0",
        )
        self._last_run: Optional[datetime] = None
        self._last_success: Optional[datetime] = None
        self._error_count_24h: int = 0

    async def process(self, context: AgentContext) -> PolicyOutput:
        """Evaluate all policies and detect violations.

        Steps:
          1. EXTRACT system state from metadata
          2. RUN each policy check
          3. AGGREGATE violations
          4. EMIT PolicyOutput

        Args:
            context: AgentContext with system state in metadata.

        Returns:
            PolicyOutput with violations and enforcement results.
        """
        self._last_run = datetime.now(timezone.utc)

        try:
            log = logger.bind(agent_id=self.agent_id, context_hash=context.context_window_hash)
            log.info("Starting policy enforcement")

            tier = str(context.metadata.get("current_tier", "SEED"))
            constraints = context.metadata.get("tier_constraints", {})
            if not isinstance(constraints, dict):
                constraints = {}
            active_positions = context.metadata.get("active_positions", [])
            if not isinstance(active_positions, list):
                active_positions = []
            maturity_records = context.metadata.get("agent_maturity_records", [])
            if not isinstance(maturity_records, list):
                maturity_records = []
            risk_mode = str(context.metadata.get("current_risk_mode", "NORMAL"))
            pending_orders = context.metadata.get("pending_orders", [])
            if not isinstance(pending_orders, list):
                pending_orders = []
            gross_exposure = float(context.metadata.get("gross_exposure_pct", 0.0))
            sector_exposures = context.metadata.get("sector_exposures", {})
            if not isinstance(sector_exposures, dict):
                sector_exposures = {}

            max_gross = float(constraints.get("max_gross_exposure", 0.0))
            max_sector = float(constraints.get("max_single_sector_pct", 0.0))
            max_positions = int(constraints.get("max_positions", 0))

            # Run all policy checks
            all_violations: list[PolicyViolation] = []
            policies_checked = 0

            policies_checked += 1
            all_violations.extend(check_seed_execution(tier, pending_orders))

            policies_checked += 1
            all_violations.extend(check_exposure_limits(gross_exposure, max_gross))

            policies_checked += 1
            all_violations.extend(check_sector_concentration(sector_exposures, max_sector))

            policies_checked += 1
            all_violations.extend(check_position_count(len(active_positions), max_positions))

            policies_checked += 1
            all_violations.extend(check_shadow_agent_leakage(maturity_records, active_positions))

            policies_checked += 1
            all_violations.extend(check_halted_orders(risk_mode, pending_orders))

            auto_enforced = sum(1 for v in all_violations if v.auto_enforced)
            needs_review = any(
                v.severity == IncidentSeverity.CRITICAL and not v.auto_enforced
                for v in all_violations
            )

            output = PolicyOutput(
                agent_id=self.agent_id,
                timestamp=datetime.now(timezone.utc),
                context_window_hash=context.context_window_hash,
                violations=all_violations,
                total_policies_checked=policies_checked,
                total_violations=len(all_violations),
                auto_enforced_count=auto_enforced,
                requires_human_review=needs_review,
            )

            self._last_success = datetime.now(timezone.utc)
            log.info(
                "Policy enforcement complete",
                policies=policies_checked,
                violations=len(all_violations),
                auto_enforced=auto_enforced,
                needs_review=needs_review,
            )
            return output

        except AgentProcessingError:
            raise
        except Exception as e:
            self._error_count_24h += 1
            raise AgentProcessingError(
                message=f"GOVERN-POLICY processing failed: {e}",
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
