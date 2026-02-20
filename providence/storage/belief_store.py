"""BeliefStore — append-only storage for BeliefObjects.

Supports in-memory indexing with optional JSON-lines file persistence.
Query patterns: filter by agent_id, ticker (across beliefs), timestamp range.
"""

from __future__ import annotations

import threading
from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import UUID

import structlog

from providence.schemas.belief import BeliefObject

logger = structlog.get_logger()


class BeliefStore:
    """Append-only store for BeliefObjects.

    Thread-safe. Indexed by belief_id and agent_id for fast lookups.
    """

    def __init__(self, persist_path: Path | None = None) -> None:
        self._lock = threading.RLock()
        self._beliefs: dict[UUID, BeliefObject] = {}

        # Secondary indexes
        self._by_agent: dict[str, list[UUID]] = {}
        self._by_ticker: dict[str, list[UUID]] = {}

        self._persist_path = persist_path
        if persist_path and persist_path.exists():
            self._load_from_disk()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def append(self, belief: BeliefObject) -> bool:
        """Append a belief object. Returns True if new, False if duplicate."""
        with self._lock:
            if belief.belief_id in self._beliefs:
                logger.debug(
                    "Duplicate belief skipped",
                    belief_id=str(belief.belief_id),
                )
                return False

            self._beliefs[belief.belief_id] = belief
            self._index(belief)

            if self._persist_path:
                self._persist_one(belief)

            logger.debug(
                "Belief stored",
                belief_id=str(belief.belief_id),
                agent_id=belief.agent_id,
                num_beliefs=len(belief.beliefs),
            )
            return True

    def append_many(self, beliefs: list[BeliefObject]) -> int:
        """Append multiple beliefs. Returns count of new beliefs added."""
        added = 0
        for b in beliefs:
            if self.append(b):
                added += 1
        return added

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get(self, belief_id: UUID) -> BeliefObject | None:
        """Get a single belief object by ID."""
        with self._lock:
            return self._beliefs.get(belief_id)

    def query(
        self,
        *,
        agent_ids: set[str] | None = None,
        tickers: set[str] | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int | None = None,
    ) -> list[BeliefObject]:
        """Query beliefs with filters. Returns newest-first.

        Parameters
        ----------
        agent_ids : filter by producing agent (None = all)
        tickers : filter by ticker appearing in any belief (None = all)
        since : only beliefs with timestamp >= since
        until : only beliefs with timestamp <= until
        limit : max number of results
        """
        with self._lock:
            candidate_ids = self._resolve_candidates(agent_ids, tickers)
            results: list[BeliefObject] = []

            for bid in candidate_ids:
                belief = self._beliefs[bid]

                if since and belief.timestamp < since:
                    continue
                if until and belief.timestamp > until:
                    continue

                results.append(belief)

            results.sort(key=lambda b: b.timestamp, reverse=True)

            if limit:
                results = results[:limit]

            return results

    def get_latest_by_agent(self, agent_id: str) -> BeliefObject | None:
        """Get the most recent belief object from a specific agent."""
        results = self.query(agent_ids={agent_id}, limit=1)
        return results[0] if results else None

    def get_latest_by_ticker(self, ticker: str) -> list[BeliefObject]:
        """Get the most recent belief from each agent for a ticker."""
        with self._lock:
            candidate_ids = set(self._by_ticker.get(ticker, []))

        # Group by agent, take latest from each
        by_agent: dict[str, BeliefObject] = {}
        for bid in candidate_ids:
            belief = self._beliefs[bid]
            existing = by_agent.get(belief.agent_id)
            if existing is None or belief.timestamp > existing.timestamp:
                by_agent[belief.agent_id] = belief

        return sorted(by_agent.values(), key=lambda b: b.timestamp, reverse=True)

    def all_agents(self) -> set[str]:
        """Return all unique agent IDs that have produced beliefs."""
        with self._lock:
            return set(self._by_agent.keys())

    def all_tickers(self) -> set[str]:
        """Return all unique tickers mentioned in beliefs."""
        with self._lock:
            return set(self._by_ticker.keys())

    def count(self) -> int:
        """Total number of stored belief objects."""
        with self._lock:
            return len(self._beliefs)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _index(self, belief: BeliefObject) -> None:
        """Update secondary indexes."""
        # By agent
        if belief.agent_id not in self._by_agent:
            self._by_agent[belief.agent_id] = []
        self._by_agent[belief.agent_id].append(belief.belief_id)

        # By ticker (from individual beliefs)
        tickers_seen: set[str] = set()
        for b in belief.beliefs:
            if b.ticker not in tickers_seen:
                tickers_seen.add(b.ticker)
                if b.ticker not in self._by_ticker:
                    self._by_ticker[b.ticker] = []
                self._by_ticker[b.ticker].append(belief.belief_id)

    def _resolve_candidates(
        self,
        agent_ids: set[str] | None,
        tickers: set[str] | None,
    ) -> set[UUID]:
        """Resolve candidate belief IDs from index intersection."""
        if agent_ids is None and tickers is None:
            return set(self._beliefs.keys())

        sets: list[set[UUID]] = []

        if agent_ids is not None:
            aid_ids: set[UUID] = set()
            for aid in agent_ids:
                aid_ids.update(self._by_agent.get(aid, []))
            sets.append(aid_ids)

        if tickers is not None:
            tick_ids: set[UUID] = set()
            for ticker in tickers:
                tick_ids.update(self._by_ticker.get(ticker, []))
            sets.append(tick_ids)

        if not sets:
            return set()
        result = sets[0]
        for s in sets[1:]:
            result = result & s
        return result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _persist_one(self, belief: BeliefObject) -> None:
        """Append a single belief to the JSON-lines file."""
        try:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._persist_path, "a") as f:
                f.write(belief.model_dump_json() + "\n")
        except OSError as exc:
            logger.error(
                "Failed to persist belief",
                belief_id=str(belief.belief_id),
                error=str(exc),
            )

    def _load_from_disk(self) -> None:
        """Load beliefs from JSON-lines file."""
        count = 0
        try:
            with open(self._persist_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    belief = BeliefObject.model_validate_json(line)
                    if belief.belief_id not in self._beliefs:
                        self._beliefs[belief.belief_id] = belief
                        self._index(belief)
                        count += 1
        except OSError as exc:
            logger.error(
                "Failed to load beliefs from disk",
                path=str(self._persist_path),
                error=str(exc),
            )
        logger.info("Beliefs loaded from disk", count=count)
