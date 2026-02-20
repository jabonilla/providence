"""Tests for PipelineStage — isolated async executor.

Tests cover:
  - Successful execution with output serialization
  - Timeout handling
  - AgentProcessingError handling
  - Unexpected exception handling
  - Skipped stage creation
  - Timing accuracy
"""

import asyncio
from datetime import datetime, timezone
from typing import Any, Optional
from unittest.mock import AsyncMock

import pytest

from providence.agents.base import AgentContext, AgentStatus, BaseAgent, HealthStatus
from providence.exceptions import AgentProcessingError
from providence.orchestration.models import StageStatus
from providence.orchestration.stage import PipelineStage

NOW = datetime.now(timezone.utc)


class MockOutput:
    """Mock Pydantic-like output with model_dump."""

    def __init__(self, data: dict):
        self._data = data

    def model_dump(self, mode: str = "python") -> dict:
        return self._data


class MockAgent(BaseAgent):
    """Mock agent for testing PipelineStage."""

    CONSUMED_DATA_TYPES: set = set()

    def __init__(
        self,
        agent_id: str = "MOCK-AGENT",
        output: Any = None,
        error: Exception | None = None,
        delay: float = 0.0,
    ):
        super().__init__(agent_id=agent_id, agent_type="test", version="1.0.0")
        self._output = output
        self._error = error
        self._delay = delay

    async def process(self, context: AgentContext) -> Any:
        if self._delay > 0:
            await asyncio.sleep(self._delay)
        if self._error:
            raise self._error
        return self._output

    def get_health(self) -> HealthStatus:
        return HealthStatus(agent_id=self.agent_id, status=AgentStatus.HEALTHY)


def _make_context(agent_id: str = "MOCK-AGENT") -> AgentContext:
    return AgentContext(
        agent_id=agent_id,
        trigger="manual",
        fragments=[],
        context_window_hash="test-hash-stage",
        timestamp=NOW,
        metadata={},
    )


# ===========================================================================
# Successful Execution Tests
# ===========================================================================

class TestStageSuccess:
    @pytest.mark.asyncio
    async def test_basic_success(self):
        output = MockOutput({"ticker": "AAPL", "direction": "LONG"})
        agent = MockAgent(output=output)
        stage = PipelineStage("COGNIT-FUND", agent)
        ctx = _make_context()

        result = await stage.execute(ctx)

        assert result.status == StageStatus.SUCCEEDED
        assert result.agent_id == "MOCK-AGENT"
        assert result.stage_name == "COGNIT-FUND"
        assert result.output == {"ticker": "AAPL", "direction": "LONG"}
        assert result.error is None
        assert result.duration_ms > 0

    @pytest.mark.asyncio
    async def test_dict_output(self):
        agent = MockAgent(output={"key": "value"})
        stage = PipelineStage("TEST", agent)
        result = await stage.execute(_make_context())
        assert result.output == {"key": "value"}

    @pytest.mark.asyncio
    async def test_non_serializable_output(self):
        agent = MockAgent(output="raw_string")
        stage = PipelineStage("TEST", agent)
        result = await stage.execute(_make_context())
        assert result.output == {"raw": "raw_string"}

    @pytest.mark.asyncio
    async def test_timing(self):
        agent = MockAgent(output={"ok": True}, delay=0.05)
        stage = PipelineStage("TEST", agent)
        result = await stage.execute(_make_context())
        assert result.duration_ms >= 40  # At least ~50ms


# ===========================================================================
# Timeout Tests
# ===========================================================================

class TestStageTimeout:
    @pytest.mark.asyncio
    async def test_timeout(self):
        agent = MockAgent(output={"ok": True}, delay=5.0)
        stage = PipelineStage("SLOW-AGENT", agent, timeout_seconds=0.1)
        result = await stage.execute(_make_context())
        assert result.status == StageStatus.FAILED
        assert "timed out" in result.error.lower()


# ===========================================================================
# Error Handling Tests
# ===========================================================================

class TestStageErrors:
    @pytest.mark.asyncio
    async def test_agent_processing_error(self):
        error = AgentProcessingError("LLM call failed", agent_id="MOCK")
        agent = MockAgent(error=error)
        stage = PipelineStage("FAILING", agent)
        result = await stage.execute(_make_context())
        assert result.status == StageStatus.FAILED
        assert "LLM call failed" in result.error

    @pytest.mark.asyncio
    async def test_unexpected_error(self):
        agent = MockAgent(error=ValueError("bad value"))
        stage = PipelineStage("BROKEN", agent)
        result = await stage.execute(_make_context())
        assert result.status == StageStatus.FAILED
        assert "bad value" in result.error

    @pytest.mark.asyncio
    async def test_error_preserves_timing(self):
        agent = MockAgent(error=RuntimeError("crash"), delay=0.05)
        stage = PipelineStage("CRASH", agent)
        result = await stage.execute(_make_context())
        assert result.status == StageStatus.FAILED
        assert result.duration_ms >= 40


# ===========================================================================
# Skipped Stage Tests
# ===========================================================================

class TestStageSkipped:
    def test_make_skipped(self):
        result = PipelineStage.make_skipped(
            "EXEC-ROUTER", "EXEC-ROUTER", "upstream EXEC-VALIDATE failed",
        )
        assert result.status == StageStatus.SKIPPED
        assert result.agent_id == "EXEC-ROUTER"
        assert "upstream" in result.error
        assert result.duration_ms == 0.0
        assert result.output is None


# ===========================================================================
# Properties Tests
# ===========================================================================

class TestStageProperties:
    def test_stage_name(self):
        agent = MockAgent(agent_id="MY-AGENT")
        stage = PipelineStage("MY-STAGE", agent)
        assert stage.stage_name == "MY-STAGE"
        assert stage.agent_id == "MY-AGENT"
