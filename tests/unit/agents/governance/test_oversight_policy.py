"""Tests for GOVERN-OVERSIGHT and GOVERN-POLICY agents.

Tests cover:
  - GOVERN-OVERSIGHT: Health aggregation, incident detection, dashboard data
  - GOVERN-POLICY: Policy checks, violation detection, enforcement
"""

from datetime import datetime, timezone

import pytest

from providence.agents.base import AgentContext, AgentStatus
from providence.agents.governance.oversight import (
    DEGRADED_THRESHOLD,
    ERROR_THRESHOLD_24H,
    GROSS_EXPOSURE_WARNING,
    RETRAIN_QUEUE_WARNING,
    SHADOW_DIVERGENCE_WARNING,
    UNHEALTHY_THRESHOLD,
    GovernOversight,
    aggregate_health,
    detect_incidents,
)
from providence.agents.governance.policy import (
    GovernPolicy,
    check_exposure_limits,
    check_halted_orders,
    check_position_count,
    check_sector_concentration,
    check_seed_execution,
    check_shadow_agent_leakage,
)
from providence.schemas.enums import CapitalTier, IncidentSeverity, SystemRiskMode
from providence.schemas.governance import (
    GovernanceIncident,
    OversightOutput,
    PolicyOutput,
    PolicyViolation,
    SystemHealthSummary,
)

NOW = datetime.now(timezone.utc)


def _make_context(**metadata_kwargs) -> AgentContext:
    return AgentContext(
        agent_id="TEST",
        trigger="manual",
        fragments=[],
        context_window_hash="test-hash-gov-002",
        timestamp=NOW,
        metadata=metadata_kwargs,
    )


def _make_health_report(
    agent_id: str = "AGENT-A",
    status: str = "HEALTHY",
    error_count_24h: int = 0,
) -> dict:
    return {
        "agent_id": agent_id,
        "status": status,
        "error_count_24h": error_count_24h,
    }


# ===========================================================================
# aggregate_health Tests
# ===========================================================================

class TestAggregateHealth:
    def test_all_healthy(self):
        reports = [
            _make_health_report("A", "HEALTHY"),
            _make_health_report("B", "HEALTHY"),
        ]
        h = aggregate_health(reports)
        assert h.total_agents == 2
        assert h.healthy_count == 2
        assert h.degraded_count == 0

    def test_mixed_status(self):
        reports = [
            _make_health_report("A", "HEALTHY"),
            _make_health_report("B", "DEGRADED", 5),
            _make_health_report("C", "UNHEALTHY", 20),
            _make_health_report("D", "OFFLINE"),
        ]
        h = aggregate_health(reports)
        assert h.total_agents == 4
        assert h.healthy_count == 1
        assert h.degraded_count == 1
        assert h.unhealthy_count == 1
        assert h.offline_count == 1
        assert h.total_errors_24h == 25

    def test_empty(self):
        h = aggregate_health([])
        assert h.total_agents == 0


# ===========================================================================
# detect_incidents Tests
# ===========================================================================

class TestDetectIncidents:
    def test_healthy_system_no_incidents(self):
        health = SystemHealthSummary(
            total_agents=10, healthy_count=10,
            degraded_count=0, unhealthy_count=0, offline_count=0,
            total_errors_24h=5,
        )
        incidents = detect_incidents(health, "NORMAL", 0, 0, 0.50)
        assert len(incidents) == 0

    def test_unhealthy_agent_critical(self):
        health = SystemHealthSummary(
            total_agents=10, healthy_count=9,
            unhealthy_count=1, total_errors_24h=0,
        )
        incidents = detect_incidents(health, "NORMAL", 0, 0, 0.50)
        assert any(i.severity == IncidentSeverity.CRITICAL for i in incidents)
        assert any("unhealthy" in i.title.lower() for i in incidents)

    def test_offline_agent_critical(self):
        health = SystemHealthSummary(
            total_agents=10, healthy_count=9, offline_count=1,
        )
        incidents = detect_incidents(health, "NORMAL", 0, 0, 0.50)
        assert any("offline" in i.title.lower() for i in incidents)

    def test_many_degraded_warning(self):
        health = SystemHealthSummary(
            total_agents=10, healthy_count=7, degraded_count=3,
        )
        incidents = detect_incidents(health, "NORMAL", 0, 0, 0.50)
        assert any(i.severity == IncidentSeverity.WARNING for i in incidents)

    def test_halted_mode_critical(self):
        health = SystemHealthSummary(total_agents=10, healthy_count=10)
        incidents = detect_incidents(health, "HALTED", 0, 0, 0.50)
        assert any("HALTED" in i.title for i in incidents)

    def test_retrain_backlog(self):
        health = SystemHealthSummary(total_agents=10, healthy_count=10)
        incidents = detect_incidents(health, "NORMAL", 5, 0, 0.50)
        assert any("retrain" in i.title.lower() for i in incidents)

    def test_shadow_divergences(self):
        health = SystemHealthSummary(total_agents=10, healthy_count=10)
        incidents = detect_incidents(health, "NORMAL", 0, 6, 0.50)
        assert any("divergence" in i.title.lower() for i in incidents)

    def test_gross_exposure_breach(self):
        health = SystemHealthSummary(total_agents=10, healthy_count=10)
        incidents = detect_incidents(health, "NORMAL", 0, 0, 1.60)
        assert any("exposure" in i.title.lower() for i in incidents)

    def test_high_error_count(self):
        health = SystemHealthSummary(
            total_agents=10, healthy_count=10, total_errors_24h=60,
        )
        incidents = detect_incidents(health, "NORMAL", 0, 0, 0.50)
        assert any("error" in i.title.lower() for i in incidents)


# ===========================================================================
# GovernOversight Integration Tests
# ===========================================================================

class TestGovernOversight:
    @pytest.mark.asyncio
    async def test_process_healthy(self):
        agent = GovernOversight()
        ctx = _make_context(
            agent_health_reports=[
                _make_health_report("A", "HEALTHY"),
                _make_health_report("B", "HEALTHY"),
            ],
            current_risk_mode="NORMAL",
            current_tier="GROWTH",
            active_positions_count=5,
            gross_exposure_pct=0.80,
            net_exposure_pct=0.50,
        )
        result = await agent.process(ctx)
        assert isinstance(result, OversightOutput)
        assert result.health_summary.total_agents == 2
        assert result.current_risk_mode == SystemRiskMode.NORMAL
        assert result.current_tier == CapitalTier.GROWTH
        assert len(result.incidents) == 0

    @pytest.mark.asyncio
    async def test_process_with_incidents(self):
        agent = GovernOversight()
        ctx = _make_context(
            agent_health_reports=[
                _make_health_report("A", "UNHEALTHY", 20),
            ],
            current_risk_mode="HALTED",
        )
        result = await agent.process(ctx)
        assert len(result.incidents) >= 2  # UNHEALTHY + HALTED

    @pytest.mark.asyncio
    async def test_process_empty(self):
        agent = GovernOversight()
        ctx = _make_context()
        result = await agent.process(ctx)
        assert result.health_summary.total_agents == 0

    @pytest.mark.asyncio
    async def test_content_hash(self):
        agent = GovernOversight()
        ctx = _make_context(
            agent_health_reports=[_make_health_report()],
            current_risk_mode="NORMAL",
        )
        result = await agent.process(ctx)
        assert len(result.content_hash) == 64

    @pytest.mark.asyncio
    async def test_invalid_enums_default(self):
        agent = GovernOversight()
        ctx = _make_context(
            current_risk_mode="INVALID",
            current_tier="INVALID",
        )
        result = await agent.process(ctx)
        assert result.current_risk_mode == SystemRiskMode.NORMAL
        assert result.current_tier == CapitalTier.SEED


class TestGovernOversightHealth:
    def test_agent_id(self):
        assert GovernOversight().agent_id == "GOVERN-OVERSIGHT"

    def test_healthy(self):
        assert GovernOversight().get_health().status == AgentStatus.HEALTHY


# ===========================================================================
# Policy Check Tests
# ===========================================================================

class TestCheckSeedExecution:
    def test_seed_with_orders(self):
        violations = check_seed_execution("SEED", [{"order": 1}])
        assert len(violations) == 1
        assert violations[0].policy_name == "SEED_NO_EXECUTION"
        assert violations[0].auto_enforced is True

    def test_seed_no_orders(self):
        assert len(check_seed_execution("SEED", [])) == 0

    def test_non_seed_with_orders(self):
        assert len(check_seed_execution("GROWTH", [{"order": 1}])) == 0


class TestCheckExposureLimits:
    def test_within_limits(self):
        assert len(check_exposure_limits(0.80, 1.20)) == 0

    def test_exceeds_limits(self):
        violations = check_exposure_limits(1.30, 1.20)
        assert len(violations) == 1
        assert violations[0].severity == IncidentSeverity.CRITICAL

    def test_zero_limit(self):
        assert len(check_exposure_limits(0.50, 0.0)) == 0


class TestCheckSectorConcentration:
    def test_within_limits(self):
        sectors = {"tech": 0.20, "health": 0.15}
        assert len(check_sector_concentration(sectors, 0.30)) == 0

    def test_exceeds_limit(self):
        sectors = {"tech": 0.40, "health": 0.15}
        violations = check_sector_concentration(sectors, 0.30)
        assert len(violations) == 1
        assert "tech" in violations[0].description

    def test_multiple_breaches(self):
        sectors = {"tech": 0.40, "health": 0.35}
        violations = check_sector_concentration(sectors, 0.30)
        assert len(violations) == 2


class TestCheckPositionCount:
    def test_within_limits(self):
        assert len(check_position_count(10, 30)) == 0

    def test_exceeds_limit(self):
        violations = check_position_count(35, 30)
        assert len(violations) == 1


class TestCheckShadowAgentLeakage:
    def test_no_shadow_agents(self):
        maturity = [{"agent_id": "A", "current_stage": "FULL"}]
        positions = [{"ticker": "AAPL", "contributing_agents": ["A"]}]
        assert len(check_shadow_agent_leakage(maturity, positions)) == 0

    def test_shadow_leakage_detected(self):
        maturity = [{"agent_id": "A", "current_stage": "SHADOW"}]
        positions = [{"ticker": "AAPL", "contributing_agents": ["A"]}]
        violations = check_shadow_agent_leakage(maturity, positions)
        assert len(violations) == 1
        assert violations[0].policy_name == "SHADOW_AGENT_LEAKAGE"
        assert violations[0].violating_agent_id == "A"

    def test_no_leakage(self):
        maturity = [{"agent_id": "A", "current_stage": "SHADOW"}]
        positions = [{"ticker": "AAPL", "contributing_agents": ["B"]}]
        assert len(check_shadow_agent_leakage(maturity, positions)) == 0


class TestCheckHaltedOrders:
    def test_halted_with_orders(self):
        violations = check_halted_orders("HALTED", [{"order": 1}])
        assert len(violations) == 1
        assert violations[0].auto_enforced is True

    def test_halted_no_orders(self):
        assert len(check_halted_orders("HALTED", [])) == 0

    def test_normal_with_orders(self):
        assert len(check_halted_orders("NORMAL", [{"order": 1}])) == 0


# ===========================================================================
# GovernPolicy Integration Tests
# ===========================================================================

class TestGovernPolicy:
    @pytest.mark.asyncio
    async def test_process_no_violations(self):
        agent = GovernPolicy()
        ctx = _make_context(
            current_tier="GROWTH",
            tier_constraints={"max_gross_exposure": 1.20, "max_single_sector_pct": 0.30, "max_positions": 15},
            active_positions=[{"ticker": "AAPL"}],
            agent_maturity_records=[{"agent_id": "A", "current_stage": "FULL"}],
            current_risk_mode="NORMAL",
            pending_orders=[],
            gross_exposure_pct=0.50,
            sector_exposures={"tech": 0.15},
        )
        result = await agent.process(ctx)
        assert isinstance(result, PolicyOutput)
        assert result.total_violations == 0
        assert result.total_policies_checked == 6

    @pytest.mark.asyncio
    async def test_process_seed_violation(self):
        agent = GovernPolicy()
        ctx = _make_context(
            current_tier="SEED",
            pending_orders=[{"order": 1}],
        )
        result = await agent.process(ctx)
        assert result.total_violations >= 1
        assert result.auto_enforced_count >= 1

    @pytest.mark.asyncio
    async def test_process_multiple_violations(self):
        agent = GovernPolicy()
        ctx = _make_context(
            current_tier="GROWTH",
            tier_constraints={"max_gross_exposure": 0.80, "max_single_sector_pct": 0.25, "max_positions": 5},
            active_positions=[{"ticker": t} for t in ["A", "B", "C", "D", "E", "F"]],
            gross_exposure_pct=1.00,
            sector_exposures={"tech": 0.40},
            current_risk_mode="NORMAL",
            pending_orders=[],
        )
        result = await agent.process(ctx)
        assert result.total_violations >= 3  # Exposure + sector + position count

    @pytest.mark.asyncio
    async def test_requires_human_review(self):
        agent = GovernPolicy()
        ctx = _make_context(
            current_tier="GROWTH",
            tier_constraints={"max_gross_exposure": 0.80},
            gross_exposure_pct=1.00,
            current_risk_mode="NORMAL",
        )
        result = await agent.process(ctx)
        # Gross exposure violation is CRITICAL and not auto-enforced
        assert result.requires_human_review is True

    @pytest.mark.asyncio
    async def test_content_hash(self):
        agent = GovernPolicy()
        ctx = _make_context(current_tier="GROWTH")
        result = await agent.process(ctx)
        assert len(result.content_hash) == 64

    @pytest.mark.asyncio
    async def test_process_empty(self):
        agent = GovernPolicy()
        ctx = _make_context()
        result = await agent.process(ctx)
        assert result.total_policies_checked == 6


class TestGovernPolicyHealth:
    def test_agent_id(self):
        assert GovernPolicy().agent_id == "GOVERN-POLICY"

    def test_healthy(self):
        assert GovernPolicy().get_health().status == AgentStatus.HEALTHY

    def test_degraded(self):
        agent = GovernPolicy()
        agent._error_count_24h = 5
        assert agent.get_health().status == AgentStatus.DEGRADED


# ===========================================================================
# Schema Tests
# ===========================================================================

class TestGovernanceOversightSchemas:
    def test_incident_frozen(self):
        gi = GovernanceIncident(
            severity=IncidentSeverity.CRITICAL,
            source_agent_id="TEST",
            title="Test",
        )
        with pytest.raises(Exception):
            gi.title = "Modified"

    def test_health_summary_frozen(self):
        hs = SystemHealthSummary(total_agents=10, healthy_count=10)
        with pytest.raises(Exception):
            hs.total_agents = 20

    def test_oversight_output_hash(self):
        oo = OversightOutput(
            agent_id="TEST",
            timestamp=NOW,
            context_window_hash="test",
            health_summary=SystemHealthSummary(total_agents=1, healthy_count=1),
            current_risk_mode=SystemRiskMode.NORMAL,
            current_tier=CapitalTier.SEED,
        )
        assert oo.content_hash != ""

    def test_violation_frozen(self):
        pv = PolicyViolation(
            policy_name="TEST",
            severity=IncidentSeverity.WARNING,
        )
        with pytest.raises(Exception):
            pv.policy_name = "MODIFIED"

    def test_policy_output_timezone(self):
        with pytest.raises(Exception):
            PolicyOutput(
                agent_id="TEST",
                timestamp=datetime(2026, 1, 1),  # No tz
                context_window_hash="test",
            )

    def test_incident_severity_values(self):
        assert IncidentSeverity.INFO.value == "INFO"
        assert IncidentSeverity.WARNING.value == "WARNING"
        assert IncidentSeverity.CRITICAL.value == "CRITICAL"
