"""RunStore — append-only storage for PipelineRun results.

Stores pipeline execution history for audit, debugging, and learning.
"""

from __future__ import annotations

import threading
from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import UUID

import structlog

from providence.orchestration.models import PipelineRun, RunStatus

logger = structlog.get_logger()


class RunStore:
    """Append-only store for PipelineRun history.

    Thread-safe. Indexed by run_id and loop_type.
    """

    def __init__(self, persist_path: Path | None = None) -> None:
        self._lock = threading.RLock()
        self._runs: dict[UUID, PipelineRun] = {}

        # Secondary indexes
        self._by_loop_type: dict[str, list[UUID]] = {}

        self._persist_path = persist_path
        if persist_path and persist_path.exists():
            self._load_from_disk()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def append(self, run: PipelineRun) -> bool:
        """Append a pipeline run. Returns True if new, False if duplicate."""
        with self._lock:
            if run.run_id in self._runs:
                return False

            self._runs[run.run_id] = run

            if run.loop_type not in self._by_loop_type:
                self._by_loop_type[run.loop_type] = []
            self._by_loop_type[run.loop_type].append(run.run_id)

            if self._persist_path:
                self._persist_one(run)

            logger.debug(
                "Run stored",
                run_id=str(run.run_id),
                loop_type=run.loop_type,
                status=run.status.value,
            )
            return True

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get(self, run_id: UUID) -> PipelineRun | None:
        """Get a single run by ID."""
        with self._lock:
            return self._runs.get(run_id)

    def query(
        self,
        *,
        loop_type: str | None = None,
        status: RunStatus | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int | None = None,
    ) -> list[PipelineRun]:
        """Query runs with filters. Returns newest-first."""
        with self._lock:
            if loop_type is not None:
                candidate_ids = set(self._by_loop_type.get(loop_type, []))
            else:
                candidate_ids = set(self._runs.keys())

            results: list[PipelineRun] = []
            for rid in candidate_ids:
                run = self._runs[rid]

                if status is not None and run.status != status:
                    continue
                if since and run.started_at < since:
                    continue
                if until and run.started_at > until:
                    continue

                results.append(run)

            results.sort(key=lambda r: r.started_at, reverse=True)

            if limit:
                results = results[:limit]

            return results

    def get_latest(self, loop_type: str | None = None) -> PipelineRun | None:
        """Get the most recent run, optionally filtered by loop type."""
        results = self.query(loop_type=loop_type, limit=1)
        return results[0] if results else None

    def success_rate(self, loop_type: str | None = None, last_n: int = 100) -> float:
        """Compute success rate over the last N runs."""
        runs = self.query(loop_type=loop_type, limit=last_n)
        if not runs:
            return 0.0
        succeeded = sum(1 for r in runs if r.status == RunStatus.SUCCEEDED)
        return succeeded / len(runs)

    def count(self, loop_type: str | None = None) -> int:
        """Total number of stored runs."""
        with self._lock:
            if loop_type is not None:
                return len(self._by_loop_type.get(loop_type, []))
            return len(self._runs)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _persist_one(self, run: PipelineRun) -> None:
        """Append a single run to the JSON-lines file."""
        try:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._persist_path, "a") as f:
                f.write(run.model_dump_json() + "\n")
        except OSError as exc:
            logger.error(
                "Failed to persist run",
                run_id=str(run.run_id),
                error=str(exc),
            )

    def _load_from_disk(self) -> None:
        """Load runs from JSON-lines file."""
        count = 0
        try:
            with open(self._persist_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    run = PipelineRun.model_validate_json(line)
                    if run.run_id not in self._runs:
                        self._runs[run.run_id] = run
                        if run.loop_type not in self._by_loop_type:
                            self._by_loop_type[run.loop_type] = []
                        self._by_loop_type[run.loop_type].append(run.run_id)
                        count += 1
        except OSError as exc:
            logger.error(
                "Failed to load runs from disk",
                path=str(self._persist_path),
                error=str(exc),
            )
        logger.info("Runs loaded from disk", count=count)
