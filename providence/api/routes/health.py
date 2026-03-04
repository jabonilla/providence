"""Health & status endpoints."""

from __future__ import annotations

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
