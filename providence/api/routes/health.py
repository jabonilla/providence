"""Health & status endpoints."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from providence.api.deps import get_state
from providence.api.schemas import (
    AgentHealthResponse,
    SystemHealthResponse,
)

router = APIRouter(prefix="/health", tags=["health"])


@router.get("", response_model=SystemHealthResponse)
async def get_system_health(include_agents: bool = False) -> SystemHealthResponse:
    """Get aggregated system health.

    Query params:
        include_agents: if True, include per-agent health details
    """
    state = get_state()
    if state.health_service is None:
        raise HTTPException(status_code=503, detail="Health service not initialized")

    report = state.health_service.check()
    summary = report.summary()

    agent_details = None
    if include_agents:
        agent_details = {}
        for agent_id, health in report.agent_health.items():
            agent_details[agent_id] = AgentHealthResponse(
                agent_id=health.agent_id,
                status=health.status if isinstance(health.status, str) else health.status.value,
                last_run=health.last_run,
                last_success=health.last_success,
                error_count_24h=health.error_count_24h,
                avg_latency_ms=health.avg_latency_ms,
                message=health.message,
            )

    return SystemHealthResponse(
        timestamp=report.timestamp,
        system_status=report.system_status,
        agents=summary["agents"],
        pipeline=summary["pipeline"],
        agent_details=agent_details,
    )


@router.get("/ready")
async def readiness_check() -> dict[str, str]:
    """Lightweight readiness probe for load balancers."""
    state = get_state()
    if not state.agent_registry:
        raise HTTPException(status_code=503, detail="No agents registered")
    return {"status": "ready"}


@router.get("/live")
async def liveness_check() -> dict[str, str]:
    """Lightweight liveness probe for orchestrators."""
    return {"status": "alive"}


@router.get("/subsystems")
async def subsystem_status() -> dict:
    """Detailed subsystem-level health status for monitoring dashboards.

    Returns store counts, agent registration, and data freshness.
    """
    state = get_state()
    now = datetime.now(timezone.utc)

    # Store stats
    stores = {}
    if state.fragment_store:
        stores["fragments"] = {
            "count": state.fragment_store.count(),
            "status": "ok" if state.fragment_store.count() > 0 else "empty",
        }
    if state.belief_store:
        stores["beliefs"] = {
            "count": state.belief_store.count(),
            "status": "ok" if state.belief_store.count() > 0 else "empty",
        }
    if state.run_store:
        runs = state.run_store
        stores["runs"] = {
            "count": runs.count(),
            "success_rate": round(runs.success_rate(), 2) if runs.count() > 0 else None,
            "status": "ok" if runs.count() > 0 else "empty",
        }
    if state.shadow_signal_store:
        stores["shadow_signals"] = {
            "count": state.shadow_signal_store.count(),
            "status": "ok" if state.shadow_signal_store.count() > 0 else "empty",
        }

    # Agent registry
    agents_info = {}
    if state.agent_registry:
        total = len(state.agent_registry)
        subsystems = {}
        for agent_id in state.agent_registry:
            sub = agent_id.split("-")[0] if "-" in agent_id else "unknown"
            subsystems[sub] = subsystems.get(sub, 0) + 1
        agents_info = {
            "total_registered": total,
            "by_subsystem": subsystems,
            "status": "ok" if total > 0 else "degraded",
        }

    # Portfolio
    portfolio_info = {}
    if state.portfolio_tracker:
        snap = state.portfolio_tracker.snapshot()
        portfolio_info = {
            "positions": len(snap.positions) if snap else 0,
            "equity": float(snap.equity) if snap and snap.equity else None,
            "status": "ok",
        }

    return {
        "timestamp": now.isoformat(),
        "stores": stores,
        "agents": agents_info,
        "portfolio": portfolio_info,
        "system_mode": "SHADOW",
    }
