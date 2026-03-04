"""Pipeline execution and run history endpoints."""

from __future__ import annotations

import asyncio
from typing import Optional
from uuid import UUID

import structlog
from fastapi import APIRouter, HTTPException, Query

logger = structlog.get_logger(__name__)

from providence.api.deps import get_state
from providence.api.schemas import (
    PipelineRunResponse,
    RunStoreStatsResponse,
    RunTriggerRequest,
    RunTriggerResponse,
    StageResultResponse,
)
from providence.orchestration.models import RunStatus

router = APIRouter(prefix="/pipeline", tags=["pipeline"])


def _run_to_response(run) -> PipelineRunResponse:
    """Convert a PipelineRun to API response."""
    stages = [
        StageResultResponse(
            stage_name=s.stage_name,
            agent_id=s.agent_id,
            status=s.status.value,
            started_at=s.started_at,
            finished_at=s.finished_at,
            duration_ms=s.duration_ms,
            error=s.error,
        )
        for s in run.stage_results
    ]
    return PipelineRunResponse(
        run_id=run.run_id,
        loop_type=run.loop_type,
        status=run.status.value,
        started_at=run.started_at,
        finished_at=run.finished_at,
        stage_results=stages,
        succeeded_count=run.succeeded_count,
        failed_count=run.failed_count,
        skipped_count=run.skipped_count,
        total_duration_ms=run.total_duration_ms,
        content_hash=run.content_hash,
    )


# ── Trigger endpoints ───────────────────────────────────────────────

@router.post("/run", response_model=RunTriggerResponse)
async def trigger_run(request: RunTriggerRequest | None = None) -> RunTriggerResponse:
    """Trigger a single pipeline cycle (main + optional exit/governance).

    This is the primary endpoint for on-demand pipeline execution.
    Equivalent to `python -m providence run-once`.
    """
    state = get_state()
    if state.runner is None:
        raise HTTPException(status_code=503, detail="Runner not initialized")

    req = request or RunTriggerRequest()

    try:
        results = await state.runner.run_once(
            run_exit=req.run_exit,
            run_governance=req.run_governance,
        )
    except Exception as exc:
        logger.error("pipeline_run_failed", error=str(exc))
        raise HTTPException(status_code=500, detail="Pipeline run failed")

    runs = {}
    summary_parts = []
    for loop_name, run in results.items():
        runs[loop_name] = _run_to_response(run)
        summary_parts.append(
            f"{loop_name}: {run.status.value} "
            f"({run.succeeded_count}/{len(run.stage_results)} stages passed)"
        )

    return RunTriggerResponse(
        runs=runs,
        summary=" | ".join(summary_parts),
    )


@router.post("/run/learning", response_model=PipelineRunResponse)
async def trigger_learning() -> PipelineRunResponse:
    """Trigger an offline learning batch.

    Equivalent to `python -m providence run-learning`.
    """
    state = get_state()
    if state.runner is None:
        raise HTTPException(status_code=503, detail="Runner not initialized")

    try:
        run = await state.runner.run_learning_batch()
    except Exception as exc:
        logger.error("learning_run_failed", error=str(exc))
        raise HTTPException(status_code=500, detail="Learning run failed")

    return _run_to_response(run)


# ── Run history ─────────────────────────────────────────────────────

@router.get("/runs", response_model=list[PipelineRunResponse])
async def list_runs(
    loop_type: Optional[str] = Query(None, description="Filter by loop type: MAIN, EXIT, LEARNING, GOVERNANCE"),
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(20, ge=1, le=200),
) -> list[PipelineRunResponse]:
    """List pipeline runs, newest first."""
    state = get_state()

    status_filter = None
    if status:
        try:
            status_filter = RunStatus(status)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="Invalid status parameter",
            )

    runs = state.run_store.query(
        loop_type=loop_type,
        status=status_filter,
        limit=limit,
    )
    return [_run_to_response(r) for r in runs]


@router.get("/runs/latest", response_model=Optional[PipelineRunResponse])
async def get_latest_run(
    loop_type: Optional[str] = Query(None),
) -> PipelineRunResponse | None:
    """Get the most recent pipeline run."""
    state = get_state()
    run = state.run_store.get_latest(loop_type=loop_type)
    if run is None:
        return None
    return _run_to_response(run)


@router.get("/runs/{run_id}", response_model=PipelineRunResponse)
async def get_run(run_id: UUID) -> PipelineRunResponse:
    """Get a specific pipeline run by ID."""
    state = get_state()
    run = state.run_store.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return _run_to_response(run)


# ── Statistics ──────────────────────────────────────────────────────

@router.get("/stats", response_model=RunStoreStatsResponse)
async def get_pipeline_stats() -> RunStoreStatsResponse:
    """Get pipeline execution statistics."""
    state = get_state()
    store = state.run_store

    loop_types = ["MAIN", "EXIT", "LEARNING", "GOVERNANCE"]
    by_loop = {lt: store.count(loop_type=lt) for lt in loop_types}
    success_by_loop = {lt: store.success_rate(loop_type=lt) for lt in loop_types}

    return RunStoreStatsResponse(
        total_count=store.count(),
        by_loop_type=by_loop,
        success_rate=store.success_rate(),
        success_rate_by_loop=success_by_loop,
    )
