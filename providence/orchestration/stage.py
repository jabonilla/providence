"""PipelineStage — isolated async executor for a single agent.

Wraps a BaseAgent with timeout, error isolation, and result tracking.
If an agent fails, the stage returns a FAILED StageResult instead of
crashing the pipeline.
"""

import asyncio
import time
from datetime import datetime, timezone
from typing import Any

import structlog

from providence.agents.base import AgentContext, BaseAgent
from providence.exceptions import AgentProcessingError
from providence.orchestration.models import StageResult, StageStatus

logger = structlog.get_logger()

DEFAULT_TIMEOUT_SECONDS = 120.0


class PipelineStage:
    """Isolated async executor for a single agent.

    Wraps agent.process() with:
      - Timeout (asyncio.wait_for)
      - Error isolation (catches exceptions → FAILED result)
      - Timing (duration_ms)
      - Structured logging
    """

    def __init__(
        self,
        stage_name: str,
        agent: BaseAgent,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._stage_name = stage_name
        self._agent = agent
        self._timeout = timeout_seconds

    @property
    def stage_name(self) -> str:
        return self._stage_name

    @property
    def agent_id(self) -> str:
        return self._agent.agent_id

    async def execute(self, context: AgentContext) -> StageResult:
        """Execute the agent with timeout and error isolation.

        Args:
            context: AgentContext assembled for this agent.

        Returns:
            StageResult with status, output, timing, and errors.
            Never raises — all errors become FAILED results.
        """
        log = logger.bind(
            stage=self._stage_name,
            agent_id=self._agent.agent_id,
        )
        started_at = datetime.now(timezone.utc)
        start_time = time.monotonic()

        try:
            log.info("Stage starting")
            result = await asyncio.wait_for(
                self._agent.process(context),
                timeout=self._timeout,
            )

            duration_ms = (time.monotonic() - start_time) * 1000.0
            finished_at = datetime.now(timezone.utc)

            # Serialize output to dict
            output_dict = self._serialize_output(result)

            log.info("Stage succeeded", duration_ms=round(duration_ms, 1))
            return StageResult(
                stage_name=self._stage_name,
                agent_id=self._agent.agent_id,
                status=StageStatus.SUCCEEDED,
                started_at=started_at,
                finished_at=finished_at,
                duration_ms=round(duration_ms, 2),
                output=output_dict,
            )

        except asyncio.TimeoutError:
            duration_ms = (time.monotonic() - start_time) * 1000.0
            finished_at = datetime.now(timezone.utc)
            error_msg = f"Stage timed out after {self._timeout}s"
            log.error("Stage timeout", duration_ms=round(duration_ms, 1))
            return StageResult(
                stage_name=self._stage_name,
                agent_id=self._agent.agent_id,
                status=StageStatus.FAILED,
                started_at=started_at,
                finished_at=finished_at,
                duration_ms=round(duration_ms, 2),
                error=error_msg,
            )

        except AgentProcessingError as e:
            duration_ms = (time.monotonic() - start_time) * 1000.0
            finished_at = datetime.now(timezone.utc)
            log.error("Stage failed (agent error)", error=str(e))
            return StageResult(
                stage_name=self._stage_name,
                agent_id=self._agent.agent_id,
                status=StageStatus.FAILED,
                started_at=started_at,
                finished_at=finished_at,
                duration_ms=round(duration_ms, 2),
                error=str(e),
            )

        except Exception as e:
            duration_ms = (time.monotonic() - start_time) * 1000.0
            finished_at = datetime.now(timezone.utc)
            log.error("Stage failed (unexpected)", error=str(e), exc_info=True)
            return StageResult(
                stage_name=self._stage_name,
                agent_id=self._agent.agent_id,
                status=StageStatus.FAILED,
                started_at=started_at,
                finished_at=finished_at,
                duration_ms=round(duration_ms, 2),
                error=f"Unexpected error: {e}",
            )

    @staticmethod
    def make_skipped(stage_name: str, agent_id: str, reason: str) -> StageResult:
        """Create a SKIPPED result for a stage that was not executed.

        Args:
            stage_name: Name of the skipped stage.
            agent_id: Agent that was not executed.
            reason: Why the stage was skipped.

        Returns:
            StageResult with SKIPPED status.
        """
        now = datetime.now(timezone.utc)
        return StageResult(
            stage_name=stage_name,
            agent_id=agent_id,
            status=StageStatus.SKIPPED,
            started_at=now,
            finished_at=now,
            duration_ms=0.0,
            error=reason,
        )

    def _serialize_output(self, result: Any) -> dict[str, Any]:
        """Serialize agent output to a dict for metadata passing.

        Args:
            result: Agent output (Pydantic model or dict).

        Returns:
            Dict representation of the output.
        """
        if hasattr(result, "model_dump"):
            return result.model_dump(mode="json")
        if isinstance(result, dict):
            return result
        return {"raw": str(result)}
