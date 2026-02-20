"""Orchestration models — pipeline run tracking and observability.

Immutable records for tracking pipeline execution: per-stage results,
aggregate run status, and timing metrics.
"""

import hashlib
import json
from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StageStatus(str, Enum):
    """Execution status for a single pipeline stage."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class RunStatus(str, Enum):
    """Aggregate status for a pipeline run."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    PARTIAL_FAILURE = "PARTIAL_FAILURE"
    FAILED = "FAILED"


class StageResult(BaseModel):
    """Result of executing a single pipeline stage (one agent).

    Immutable after creation. Captures timing, status, output, and errors.
    """

    model_config = ConfigDict(frozen=True)

    stage_name: str = Field(...)
    agent_id: str = Field(...)
    status: StageStatus = Field(...)
    started_at: datetime = Field(...)
    finished_at: datetime = Field(...)
    duration_ms: float = Field(default=0.0, ge=0.0)
    output: Optional[dict[str, Any]] = Field(
        default=None,
        description="Serialized agent output (None if failed/skipped)",
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if stage failed",
    )


class PipelineRun(BaseModel):
    """Complete record of a pipeline execution.

    Tracks all stages, aggregate status, and timing for a single
    loop invocation (MAIN, EXIT, LEARNING, or GOVERNANCE).
    """

    model_config = ConfigDict(frozen=True)

    run_id: UUID = Field(default_factory=uuid4)
    loop_type: str = Field(..., description="MAIN, EXIT, LEARNING, or GOVERNANCE")
    status: RunStatus = Field(default=RunStatus.PENDING)
    started_at: datetime = Field(...)
    finished_at: Optional[datetime] = Field(default=None)
    stage_results: list[StageResult] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Accumulated metadata from all stages",
    )
    content_hash: str = Field(default="")

    @model_validator(mode="after")
    def _compute_content_hash(self) -> "PipelineRun":
        if not self.content_hash:
            data = {
                "run_id": str(self.run_id),
                "loop_type": self.loop_type,
                "status": self.status.value,
                "stages": [
                    {"stage_name": s.stage_name, "status": s.status.value}
                    for s in self.stage_results
                ],
            }
            serialized = json.dumps(data, sort_keys=True, default=str).encode("utf-8")
            object.__setattr__(
                self, "content_hash", hashlib.sha256(serialized).hexdigest(),
            )
        return self

    @property
    def succeeded_count(self) -> int:
        return sum(1 for s in self.stage_results if s.status == StageStatus.SUCCEEDED)

    @property
    def failed_count(self) -> int:
        return sum(1 for s in self.stage_results if s.status == StageStatus.FAILED)

    @property
    def skipped_count(self) -> int:
        return sum(1 for s in self.stage_results if s.status == StageStatus.SKIPPED)

    @property
    def total_duration_ms(self) -> float:
        return sum(s.duration_ms for s in self.stage_results)
