"""Tests for BaseAgent, AgentContext, and HealthStatus.

BaseAgent is tested via a MockAgent implementation since it's abstract.
"""

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from providence.agents.base import (
    AgentContext,
    AgentStatus,
    BaseAgent,
    HealthStatus,
)
from providence.schemas.enums import DataType, ValidationStatus
from providence.schemas.market_state import MarketStateFragment


# ===================================================================
# MockAgent for testing BaseAgent ABC
# ===================================================================
class MockAgent(BaseAgent[dict]):
    """Concrete implementation of BaseAgent for testing."""

    def __init__(
        self,
        agent_id: str = "MOCK-AGENT",
        agent_type: str = "test",
        version: str = "1.0.0",
    ) -> None:
        super().__init__(agent_id, agent_type, version)
        self._last_context: AgentContext | None = None
        self._should_fail: bool = False

    async def process(self, context: AgentContext) -> dict:
        """Store context and return a simple result."""
        if self._should_fail:
            raise RuntimeError("Mock processing failure")
        self._last_context = context
        return {"processed": True, "agent_id": self.agent_id}

    def get_health(self) -> HealthStatus:
        """Return a healthy status."""
        return HealthStatus(
            agent_id=self.agent_id,
            status=AgentStatus.HEALTHY,
            last_run=datetime.now(timezone.utc),
            last_success=datetime.now(timezone.utc),
            error_count_24h=0,
            avg_latency_ms=42.5,
            message="All systems nominal",
        )


# ===================================================================
# BaseAgent Tests
# ===================================================================
class TestBaseAgent:
    """Tests for BaseAgent ABC via MockAgent."""

    def test_agent_properties(self) -> None:
        """Agent properties should be set correctly."""
        agent = MockAgent("TEST-01", "perception", "2.1.0")
        assert agent.agent_id == "TEST-01"
        assert agent.agent_type == "perception"
        assert agent.version == "2.1.0"

    @pytest.mark.asyncio
    async def test_process_returns_result(self) -> None:
        """process() should return the agent's output."""
        agent = MockAgent()
        context = AgentContext(
            agent_id="MOCK-AGENT",
            trigger="manual",
            context_window_hash="abc123",
            timestamp=datetime.now(timezone.utc),
        )
        result = await agent.process(context)
        assert result["processed"] is True
        assert result["agent_id"] == "MOCK-AGENT"

    @pytest.mark.asyncio
    async def test_process_receives_context(self) -> None:
        """process() should receive the full AgentContext."""
        agent = MockAgent()
        ts = datetime.now(timezone.utc)
        context = AgentContext(
            agent_id="MOCK-AGENT",
            trigger="schedule",
            context_window_hash="xyz789",
            timestamp=ts,
            metadata={"test_key": "test_value"},
        )
        await agent.process(context)
        assert agent._last_context is not None
        assert agent._last_context.trigger == "schedule"
        assert agent._last_context.metadata["test_key"] == "test_value"

    @pytest.mark.asyncio
    async def test_process_failure_propagates(self) -> None:
        """Errors during process() should propagate."""
        agent = MockAgent()
        agent._should_fail = True
        context = AgentContext(
            agent_id="MOCK-AGENT",
            trigger="manual",
            context_window_hash="abc",
            timestamp=datetime.now(timezone.utc),
        )
        with pytest.raises(RuntimeError, match="Mock processing failure"):
            await agent.process(context)

    def test_compute_content_hash(self) -> None:
        """Static compute_content_hash should return consistent SHA-256."""
        data = {"ticker": "AAPL", "close": 186.90}
        hash1 = MockAgent.compute_content_hash(data)
        hash2 = MockAgent.compute_content_hash(data)
        assert hash1 == hash2
        assert len(hash1) == 64

    def test_compute_content_hash_key_order_invariant(self) -> None:
        """Hash should be the same regardless of key order."""
        hash1 = MockAgent.compute_content_hash({"a": 1, "b": 2})
        hash2 = MockAgent.compute_content_hash({"b": 2, "a": 1})
        assert hash1 == hash2

    def test_compute_content_hash_different_data(self) -> None:
        """Different data should produce different hashes."""
        hash1 = MockAgent.compute_content_hash({"value": 1})
        hash2 = MockAgent.compute_content_hash({"value": 2})
        assert hash1 != hash2

    def test_get_health(self) -> None:
        """get_health() should return a valid HealthStatus."""
        agent = MockAgent()
        health = agent.get_health()
        assert health.agent_id == "MOCK-AGENT"
        assert health.status == AgentStatus.HEALTHY
        assert health.error_count_24h == 0
        assert health.avg_latency_ms == 42.5


# ===================================================================
# AgentContext Tests
# ===================================================================
class TestAgentContext:
    """Tests for AgentContext model."""

    def test_valid_context_creation(self) -> None:
        """Valid context should be created successfully."""
        ctx = AgentContext(
            agent_id="COGNIT-FUNDAMENTAL",
            trigger="schedule",
            context_window_hash="abc123",
            timestamp=datetime.now(timezone.utc),
        )
        assert ctx.agent_id == "COGNIT-FUNDAMENTAL"
        assert ctx.trigger == "schedule"
        assert len(ctx.fragments) == 0

    def test_context_with_fragments(self, sample_fragment: MarketStateFragment) -> None:
        """Context can contain MarketStateFragments."""
        ctx = AgentContext(
            agent_id="COGNIT-FUNDAMENTAL",
            trigger="schedule",
            fragments=[sample_fragment],
            context_window_hash="abc123",
            timestamp=datetime.now(timezone.utc),
        )
        assert len(ctx.fragments) == 1
        assert ctx.fragments[0].entity == "AAPL"

    def test_context_with_metadata(self) -> None:
        """Context can carry arbitrary metadata."""
        ctx = AgentContext(
            agent_id="COGNIT-FUNDAMENTAL",
            trigger="event",
            context_window_hash="abc123",
            timestamp=datetime.now(timezone.utc),
            metadata={"urgency": "high", "event_type": "earnings"},
        )
        assert ctx.metadata["urgency"] == "high"

    def test_context_is_immutable(self) -> None:
        """AgentContext should be frozen."""
        ctx = AgentContext(
            agent_id="COGNIT-FUNDAMENTAL",
            trigger="schedule",
            context_window_hash="abc123",
            timestamp=datetime.now(timezone.utc),
        )
        with pytest.raises(Exception):
            ctx.agent_id = "TAMPERED"


# ===================================================================
# HealthStatus Tests
# ===================================================================
class TestHealthStatus:
    """Tests for HealthStatus model."""

    def test_valid_health_status(self) -> None:
        """Valid health status should be created successfully."""
        health = HealthStatus(
            agent_id="PERCEPT-PRICE",
            status=AgentStatus.HEALTHY,
        )
        assert health.agent_id == "PERCEPT-PRICE"
        assert health.error_count_24h == 0
        assert health.avg_latency_ms == 0.0
        assert health.message is None

    def test_all_status_values(self) -> None:
        """All AgentStatus values should be valid."""
        for status in AgentStatus:
            health = HealthStatus(agent_id="TEST", status=status)
            assert health.status == status

    def test_degraded_status_with_errors(self) -> None:
        """Degraded status can include error counts and messages."""
        health = HealthStatus(
            agent_id="PERCEPT-PRICE",
            status=AgentStatus.DEGRADED,
            error_count_24h=5,
            avg_latency_ms=2500.0,
            message="Polygon.io intermittent timeouts",
        )
        assert health.status == AgentStatus.DEGRADED
        assert health.error_count_24h == 5
        assert "timeout" in health.message.lower()

    def test_error_count_non_negative(self) -> None:
        """error_count_24h must be >= 0."""
        with pytest.raises(Exception):
            HealthStatus(
                agent_id="TEST",
                status=AgentStatus.HEALTHY,
                error_count_24h=-1,
            )

    def test_latency_non_negative(self) -> None:
        """avg_latency_ms must be >= 0."""
        with pytest.raises(Exception):
            HealthStatus(
                agent_id="TEST",
                status=AgentStatus.HEALTHY,
                avg_latency_ms=-10.0,
            )
