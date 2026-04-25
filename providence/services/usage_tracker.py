"""Usage tracking service — records API calls, pipeline runs, and resource consumption per user.

Thread-safe, JSONL-backed, append-only store following the same pattern as
FragmentStore, BeliefStore, RunStore, etc.
"""

from __future__ import annotations

import json
import threading
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class UsageEvent(BaseModel):
    """Single usage event record."""

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = "default"
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )
    event_type: str = Field(
        description="api_call, pipeline_run, signal_generated, doc_upload, chat_message",
    )
    detail: dict = Field(default_factory=dict)


class UsageSummary(BaseModel):
    """Aggregated usage summary for a user."""

    user_id: str
    period_start: str
    period_end: str
    api_calls: int = 0
    pipeline_runs: int = 0
    signals_generated: int = 0
    doc_uploads: int = 0
    chat_messages: int = 0
    beliefs_created: int = 0
    agent_invocations: int = 0


class DailyUsage(BaseModel):
    """Usage counts for a single day."""

    date: str
    api_calls: int = 0
    pipeline_runs: int = 0
    signals_generated: int = 0
    doc_uploads: int = 0
    chat_messages: int = 0


class UsageTracker:
    """Thread-safe usage event tracker with JSONL persistence."""

    def __init__(self, persist_path: Optional[Path] = None) -> None:
        self._lock = threading.RLock()
        self._events: list[UsageEvent] = []
        self._by_user: dict[str, list[UsageEvent]] = defaultdict(list)
        self._by_type: dict[str, list[UsageEvent]] = defaultdict(list)
        self._persist_path = persist_path

        if persist_path and persist_path.exists():
            self._load(persist_path)

    def _load(self, path: Path) -> None:
        """Load events from JSONL file."""
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    event = UsageEvent(**data)
                    self._events.append(event)
                    self._by_user[event.user_id].append(event)
                    self._by_type[event.event_type].append(event)
                except (json.JSONDecodeError, Exception):
                    continue

    def _persist(self, event: UsageEvent) -> None:
        """Append a single event to JSONL."""
        if self._persist_path:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._persist_path, "a") as f:
                f.write(event.model_dump_json() + "\n")

    def record(
        self,
        event_type: str,
        user_id: str = "default",
        detail: dict | None = None,
    ) -> UsageEvent:
        """Record a usage event."""
        event = UsageEvent(
            user_id=user_id,
            event_type=event_type,
            detail=detail or {},
        )
        with self._lock:
            self._events.append(event)
            self._by_user[event.user_id].append(event)
            self._by_type[event.event_type].append(event)
            self._persist(event)
        return event

    def count(self, user_id: str | None = None) -> int:
        """Total event count, optionally filtered by user."""
        with self._lock:
            if user_id:
                return len(self._by_user.get(user_id, []))
            return len(self._events)

    def get_summary(
        self,
        user_id: str = "default",
        days: int = 30,
    ) -> UsageSummary:
        """Get aggregated usage summary for a user over N days."""
        now = datetime.now(timezone.utc)
        cutoff = datetime(
            now.year, now.month, now.day, tzinfo=timezone.utc
        )
        # Go back N days
        from datetime import timedelta

        cutoff = cutoff - timedelta(days=days)
        cutoff_iso = cutoff.isoformat()

        with self._lock:
            events = [
                e
                for e in self._by_user.get(user_id, [])
                if e.timestamp >= cutoff_iso
            ]

        summary = UsageSummary(
            user_id=user_id,
            period_start=cutoff_iso,
            period_end=now.isoformat(),
        )

        for e in events:
            if e.event_type == "api_call":
                summary.api_calls += 1
            elif e.event_type == "pipeline_run":
                summary.pipeline_runs += 1
            elif e.event_type == "signal_generated":
                summary.signals_generated += 1
            elif e.event_type == "doc_upload":
                summary.doc_uploads += 1
            elif e.event_type == "chat_message":
                summary.chat_messages += 1
            elif e.event_type == "belief_created":
                summary.beliefs_created += 1
            elif e.event_type == "agent_invocation":
                summary.agent_invocations += 1

        return summary

    def get_daily_breakdown(
        self,
        user_id: str = "default",
        days: int = 30,
    ) -> list[DailyUsage]:
        """Get per-day usage breakdown."""
        from datetime import timedelta

        now = datetime.now(timezone.utc)
        start = now - timedelta(days=days)
        start_iso = start.isoformat()

        with self._lock:
            events = [
                e
                for e in self._by_user.get(user_id, [])
                if e.timestamp >= start_iso
            ]

        # Group by date
        by_date: dict[str, DailyUsage] = {}
        for i in range(days + 1):
            d = (start + timedelta(days=i)).strftime("%Y-%m-%d")
            by_date[d] = DailyUsage(date=d)

        for e in events:
            date_str = e.timestamp[:10]  # YYYY-MM-DD
            if date_str not in by_date:
                by_date[date_str] = DailyUsage(date=date_str)
            du = by_date[date_str]
            if e.event_type == "api_call":
                du.api_calls += 1
            elif e.event_type == "pipeline_run":
                du.pipeline_runs += 1
            elif e.event_type == "signal_generated":
                du.signals_generated += 1
            elif e.event_type == "doc_upload":
                du.doc_uploads += 1
            elif e.event_type == "chat_message":
                du.chat_messages += 1

        return sorted(by_date.values(), key=lambda x: x.date)

    def get_type_breakdown(
        self,
        user_id: str = "default",
        days: int = 30,
    ) -> dict[str, int]:
        """Get counts by event type."""
        from datetime import timedelta

        now = datetime.now(timezone.utc)
        cutoff = (now - timedelta(days=days)).isoformat()

        with self._lock:
            events = [
                e
                for e in self._by_user.get(user_id, [])
                if e.timestamp >= cutoff
            ]

        counts: dict[str, int] = defaultdict(int)
        for e in events:
            counts[e.event_type] += 1
        return dict(counts)

    def check_limit(
        self,
        user_id: str,
        event_type: str,
        max_per_day: int,
    ) -> tuple[bool, int]:
        """Check if a user has exceeded their daily limit for an event type.

        Returns (allowed, remaining).
        """
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        with self._lock:
            count = sum(
                1
                for e in self._by_user.get(user_id, [])
                if e.event_type == event_type and e.timestamp[:10] == today
            )

        remaining = max(0, max_per_day - count)
        return count < max_per_day, remaining
