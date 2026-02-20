"""FragmentStore — append-only storage for MarketStateFragments.

Supports in-memory indexing with optional JSON-lines file persistence.
Query patterns mirror ContextService needs: filter by data_type, entity,
validation_status, and timestamp range.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from uuid import UUID

import structlog

from providence.schemas.enums import DataType, ValidationStatus
from providence.schemas.market_state import MarketStateFragment

logger = structlog.get_logger()


class FragmentStore:
    """Append-only store for MarketStateFragments.

    Thread-safe via a read-write lock. Fragments are indexed in memory
    by fragment_id, data_type, and entity for fast lookups.

    If a persist_path is provided, fragments are written as JSON lines
    on append and reloaded on init.
    """

    def __init__(self, persist_path: Path | None = None) -> None:
        self._lock = threading.RLock()
        self._fragments: dict[UUID, MarketStateFragment] = {}

        # Secondary indexes
        self._by_data_type: dict[DataType, list[UUID]] = {}
        self._by_entity: dict[str | None, list[UUID]] = {}

        self._persist_path = persist_path
        if persist_path and persist_path.exists():
            self._load_from_disk()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def append(self, fragment: MarketStateFragment) -> bool:
        """Append a fragment. Returns True if new, False if duplicate."""
        with self._lock:
            if fragment.fragment_id in self._fragments:
                logger.debug(
                    "Duplicate fragment skipped",
                    fragment_id=str(fragment.fragment_id),
                )
                return False

            self._fragments[fragment.fragment_id] = fragment
            self._index(fragment)

            if self._persist_path:
                self._persist_one(fragment)

            logger.debug(
                "Fragment stored",
                fragment_id=str(fragment.fragment_id),
                data_type=fragment.data_type.value,
                entity=fragment.entity,
            )
            return True

    def append_many(self, fragments: list[MarketStateFragment]) -> int:
        """Append multiple fragments. Returns count of new fragments added."""
        added = 0
        for frag in fragments:
            if self.append(frag):
                added += 1
        return added

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get(self, fragment_id: UUID) -> MarketStateFragment | None:
        """Get a single fragment by ID."""
        with self._lock:
            return self._fragments.get(fragment_id)

    def query(
        self,
        *,
        data_types: set[DataType] | None = None,
        entities: set[str] | None = None,
        exclude_quarantined: bool = True,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int | None = None,
    ) -> list[MarketStateFragment]:
        """Query fragments with filters. Returns newest-first.

        Parameters
        ----------
        data_types : filter by data type (None = all)
        entities : filter by entity ticker (None = all)
        exclude_quarantined : skip QUARANTINED fragments (default True)
        since : only fragments with timestamp >= since
        until : only fragments with timestamp <= until
        limit : max number of results
        """
        with self._lock:
            # Start with candidate IDs from indexes
            candidate_ids = self._resolve_candidates(data_types, entities)
            results: list[MarketStateFragment] = []

            for fid in candidate_ids:
                frag = self._fragments[fid]

                if exclude_quarantined and frag.validation_status == ValidationStatus.QUARANTINED:
                    continue
                if since and frag.timestamp < since:
                    continue
                if until and frag.timestamp > until:
                    continue

                results.append(frag)

            # Sort newest first
            results.sort(key=lambda f: f.timestamp, reverse=True)

            if limit:
                results = results[:limit]

            return results

    def get_latest_by_entity(
        self,
        entity: str,
        data_type: DataType | None = None,
    ) -> MarketStateFragment | None:
        """Get the most recent fragment for an entity."""
        results = self.query(
            entities={entity},
            data_types={data_type} if data_type else None,
            limit=1,
        )
        return results[0] if results else None

    def all_entities(self) -> set[str]:
        """Return all unique entity values (excluding None)."""
        with self._lock:
            return {e for e in self._by_entity if e is not None}

    def count(self) -> int:
        """Total number of stored fragments."""
        with self._lock:
            return len(self._fragments)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _index(self, fragment: MarketStateFragment) -> None:
        """Update secondary indexes."""
        # By data type
        if fragment.data_type not in self._by_data_type:
            self._by_data_type[fragment.data_type] = []
        self._by_data_type[fragment.data_type].append(fragment.fragment_id)

        # By entity
        if fragment.entity not in self._by_entity:
            self._by_entity[fragment.entity] = []
        self._by_entity[fragment.entity].append(fragment.fragment_id)

    def _resolve_candidates(
        self,
        data_types: set[DataType] | None,
        entities: set[str] | None,
    ) -> set[UUID]:
        """Resolve candidate fragment IDs from index intersection."""
        if data_types is None and entities is None:
            return set(self._fragments.keys())

        sets: list[set[UUID]] = []

        if data_types is not None:
            dt_ids: set[UUID] = set()
            for dt in data_types:
                dt_ids.update(self._by_data_type.get(dt, []))
            sets.append(dt_ids)

        if entities is not None:
            ent_ids: set[UUID] = set()
            for ent in entities:
                ent_ids.update(self._by_entity.get(ent, []))
            sets.append(ent_ids)

        if not sets:
            return set()
        result = sets[0]
        for s in sets[1:]:
            result = result & s
        return result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _persist_one(self, fragment: MarketStateFragment) -> None:
        """Append a single fragment to the JSON-lines file."""
        try:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._persist_path, "a") as f:
                f.write(fragment.model_dump_json() + "\n")
        except OSError as exc:
            logger.error(
                "Failed to persist fragment",
                fragment_id=str(fragment.fragment_id),
                error=str(exc),
            )

    def _load_from_disk(self) -> None:
        """Load fragments from JSON-lines file."""
        count = 0
        try:
            with open(self._persist_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    frag = MarketStateFragment.model_validate_json(line)
                    if frag.fragment_id not in self._fragments:
                        self._fragments[frag.fragment_id] = frag
                        self._index(frag)
                        count += 1
        except OSError as exc:
            logger.error(
                "Failed to load fragments from disk",
                path=str(self._persist_path),
                error=str(exc),
            )
        logger.info("Fragments loaded from disk", count=count)
