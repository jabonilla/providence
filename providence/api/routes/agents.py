"""Agent registry and health endpoints."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from providence.api.deps import get_state
from providence.api.schemas import AgentHealthResponse, AgentInfoResponse

router = APIRouter(prefix="/agents", tags=["agents"])

# Valid filter values
_VALID_SUBSYSTEMS = {
    "perception", "cognition", "regime", "decision",
    "execution", "exit", "learning", "governance",
}
_VALID_CLASSIFICATIONS = {"FROZEN", "ADAPTIVE", "PERCEPTION"}

# Classification maps — mirrors factory.py constants
_ADAPTIVE_IDS = {
    "COGNIT-FUNDAMENTAL", "COGNIT-MACRO", "COGNIT-EVENT",
    "COGNIT-NARRATIVE", "COGNIT-CROSSSEC", "COGNIT-EXIT",
    "REGIME-NARR", "DECIDE-SYNTH",
}

_PERCEPTION_IDS = {
    "PERCEPT-PRICE", "PERCEPT-FILING", "PERCEPT-NEWS",
    "PERCEPT-OPTIONS", "PERCEPT-CDS", "PERCEPT-MACRO",
}

_SUBSYSTEM_MAP = {
    "PERCEPT-": "perception",
    "COGNIT-": "cognition",
    "REGIME-": "regime",
    "DECIDE-": "decision",
    "EXEC-": "execution",
    "INVALID-": "exit",
    "THESIS-": "exit",
    "SHADOW-": "exit",
    "RENEW-": "exit",
    "LEARN-": "learning",
    "GOVERN-": "governance",
}


def _classify_agent(agent_id: str) -> tuple[str, str]:
    """Return (subsystem, classification) for an agent ID."""
    classification = "FROZEN"
    if agent_id in _ADAPTIVE_IDS:
        classification = "ADAPTIVE"
    elif agent_id in _PERCEPTION_IDS:
        classification = "PERCEPTION"

    subsystem = "unknown"
    for prefix, sub in _SUBSYSTEM_MAP.items():
        if agent_id.startswith(prefix):
            subsystem = sub
            break

    return subsystem, classification


@router.get("", response_model=list[AgentInfoResponse])
async def list_agents(
    subsystem: Optional[str] = Query(None, description="Filter by subsystem"),
    classification: Optional[str] = Query(None, description="Filter by classification"),
) -> list[AgentInfoResponse]:
    """List all registered agents with optional filtering."""
    # Validate filter values
    if subsystem and subsystem not in _VALID_SUBSYSTEMS:
        raise HTTPException(status_code=400, detail="Invalid subsystem parameter")
    if classification and classification not in _VALID_CLASSIFICATIONS:
        raise HTTPException(status_code=400, detail="Invalid classification parameter")

    state = get_state()
    agents = []

    for agent_id, agent in sorted(state.agent_registry.items()):
        sub, cls = _classify_agent(agent_id)

        if subsystem and sub != subsystem:
            continue
        if classification and cls != classification:
            continue

        agents.append(AgentInfoResponse(
            agent_id=agent.agent_id,
            agent_type=agent.agent_type,
            version=agent.version,
            subsystem=sub,
            classification=cls,
        ))

    return agents


@router.get("/{agent_id}", response_model=AgentInfoResponse)
async def get_agent(agent_id: str) -> AgentInfoResponse:
    """Get info about a specific agent."""
    state = get_state()
    agent = state.agent_registry.get(agent_id)
    if agent is None:
        raise HTTPException(status_code=404, detail="Agent not found")

    sub, cls = _classify_agent(agent_id)
    return AgentInfoResponse(
        agent_id=agent.agent_id,
        agent_type=agent.agent_type,
        version=agent.version,
        subsystem=sub,
        classification=cls,
    )


@router.get("/{agent_id}/health", response_model=AgentHealthResponse)
async def get_agent_health(agent_id: str) -> AgentHealthResponse:
    """Get health status for a specific agent."""
    state = get_state()
    agent = state.agent_registry.get(agent_id)
    if agent is None:
        raise HTTPException(status_code=404, detail="Agent not found")

    health = agent.get_health()
    return AgentHealthResponse(
        agent_id=health.agent_id,
        status=health.status if isinstance(health.status, str) else health.status.value,
        last_run=health.last_run,
        last_success=health.last_success,
        error_count_24h=health.error_count_24h,
        avg_latency_ms=health.avg_latency_ms,
        message=health.message,
    )
