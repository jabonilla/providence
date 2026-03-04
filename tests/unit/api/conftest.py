"""Shared fixtures for API tests.

Creates a test AppState with minimal in-memory stores and mock agents,
then builds a FastAPI TestClient.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from typing import Any, Optional
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

# Guard: skip all API tests if fastapi not installed
fastapi = pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from providence.agents.base import AgentStatus, BaseAgent, HealthStatus
from providence.api.app import create_app
from providence.api.deps import AppState, set_state
from providence.orchestration.models import PipelineRun, RunStatus, StageResult, StageStatus
from providence.schemas.enums import DataType, ValidationStatus
from providence.schemas.market_state import MarketStateFragment
from providence.services.health import HealthService
from providence.storage.belief_store import BeliefStore
from providence.storage.fragment_store import FragmentStore
from providence.storage.run_store import RunStore


class MockAgent(BaseAgent):
    """Minimal agent for testing."""

    def __init__(self, agent_id: str, agent_type: str = "mock", version: str = "0.1.0"):
        super().__init__(agent_id=agent_id, agent_type=agent_type, version=version)

    async def process(self, context: Any) -> dict:
        return {"status": "ok"}


def _make_fragment(
    entity: str = "AAPL",
    data_type: DataType = DataType.PRICE_OHLCV,
    status: ValidationStatus = ValidationStatus.VALID,
) -> MarketStateFragment:
    """Create a test fragment."""
    return MarketStateFragment(
        fragment_id=uuid4(),
        agent_id="PERCEPT-PRICE",
        timestamp=datetime.now(timezone.utc),
        source_timestamp=datetime.now(timezone.utc),
        version="test",
        entity=entity,
        data_type=data_type,
        schema_version="1.0.0",
        source_hash="abc123",
        validation_status=status,
        payload={"test": True, "entity": entity},
    )


def _make_run(
    loop_type: str = "MAIN",
    status: RunStatus = RunStatus.SUCCEEDED,
    num_stages: int = 3,
) -> PipelineRun:
    """Create a test pipeline run."""
    now = datetime.now(timezone.utc)
    stages = []
    for i in range(num_stages):
        stages.append(StageResult(
            stage_name=f"stage-{i}",
            agent_id=f"AGENT-{i}",
            status=StageStatus.SUCCEEDED,
            started_at=now,
            finished_at=now,
            duration_ms=100.0 * (i + 1),
        ))
    return PipelineRun(
        run_id=uuid4(),
        loop_type=loop_type,
        status=status,
        started_at=now,
        finished_at=now,
        stage_results=stages,
    )


@pytest.fixture
def test_stores():
    """Create in-memory stores with sample data."""
    frag_store = FragmentStore()
    belief_store = BeliefStore()
    run_store = RunStore()

    # Seed fragments
    for ticker in ["AAPL", "MSFT", "GOOG"]:
        frag_store.append(_make_fragment(entity=ticker))

    # Seed runs
    run_store.append(_make_run(loop_type="MAIN", status=RunStatus.SUCCEEDED))
    run_store.append(_make_run(loop_type="EXIT", status=RunStatus.SUCCEEDED))
    run_store.append(_make_run(loop_type="MAIN", status=RunStatus.PARTIAL_FAILURE))

    return frag_store, belief_store, run_store


@pytest.fixture
def test_agents():
    """Create a minimal agent registry."""
    agents = {}
    # A few representative agents from each subsystem
    for aid in [
        "PERCEPT-PRICE", "COGNIT-TECHNICAL", "COGNIT-FUNDAMENTAL",
        "REGIME-STAT", "DECIDE-OPTIM", "EXEC-VALIDATE",
        "LEARN-ATTRIB", "GOVERN-CAPITAL",
    ]:
        agents[aid] = MockAgent(agent_id=aid)
    return agents


@pytest.fixture
def test_state(test_stores, test_agents):
    """Build a complete AppState for testing."""
    frag_store, belief_store, run_store = test_stores

    health_svc = HealthService(
        agent_registry=test_agents,
        run_store=run_store,
    )

    return AppState(
        fragment_store=frag_store,
        belief_store=belief_store,
        run_store=run_store,
        agent_registry=test_agents,
        health_service=health_svc,
    )


@pytest.fixture
def client(test_state) -> TestClient:
    """Create a FastAPI TestClient with test state."""
    app = create_app(state=test_state, title="Providence Test API")
    return TestClient(app)
