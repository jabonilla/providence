"""Per-user agent preference configuration.

Controls per-agent behavior parameters such as time horizon,
risk threshold, regime sensitivity, and sector filters.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import structlog
from pydantic import BaseModel, Field, field_validator

logger = structlog.get_logger()


class AgentPreferences(BaseModel):
    """Configuration preferences for a single agent.

    These preferences influence how an agent processes signals
    and generates beliefs.
    """

    agent_id: str = Field(description="Agent identifier (e.g. COGNIT-FUNDAMENTAL)")
    time_horizon_days: int = Field(
        default=30,
        ge=1,
        le=365,
        description="Investment time horizon in days",
    )
    risk_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Maximum risk tolerance (0.0=conservative, 1.0=aggressive)",
    )
    regime_sensitivity: float = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        description="Multiplier on regime adjustments (0.0=ignore, 2.0=double)",
    )
    sector_filters: list[str] = Field(
        default_factory=list,
        description="Sector filter list. Empty = all sectors allowed.",
    )
    enabled: bool = Field(
        default=True,
        description="Whether this agent is active",
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "time_horizon_days": self.time_horizon_days,
            "risk_threshold": self.risk_threshold,
            "regime_sensitivity": self.regime_sensitivity,
            "sector_filters": self.sector_filters,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentPreferences:
        return cls(**data)


class AgentPreferencesStore:
    """Thread-safe, JSONL-backed store for per-user agent preferences.

    Keyed by (user_id, agent_id). Follows the same pattern as
    ConversationStore/FragmentStore: in-memory dict with RLock,
    optional JSONL persistence.
    """

    def __init__(self, persist_path: Path | None = None) -> None:
        self._lock = threading.RLock()
        # _prefs[user_id][agent_id] = AgentPreferences
        self._prefs: dict[str, dict[str, AgentPreferences]] = {}
        self._persist_path = persist_path
        if persist_path and persist_path.exists():
            self._load_from_disk()

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get(self, user_id: str, agent_id: str) -> AgentPreferences | None:
        """Get preferences for a specific agent. Returns None if not set."""
        with self._lock:
            user_prefs = self._prefs.get(user_id)
            if user_prefs is None:
                return None
            return user_prefs.get(agent_id)

    def get_all(self, user_id: str) -> dict[str, AgentPreferences]:
        """Get all agent preferences for a user."""
        with self._lock:
            user_prefs = self._prefs.get(user_id)
            if user_prefs is None:
                return {}
            return dict(user_prefs)

    def list_users(self) -> list[str]:
        """List all user IDs with stored preferences."""
        with self._lock:
            return list(self._prefs.keys())

    def count(self, user_id: str | None = None) -> int:
        """Total number of stored preference records.

        If user_id is given, count only that user's prefs.
        """
        with self._lock:
            if user_id is not None:
                return len(self._prefs.get(user_id, {}))
            return sum(len(v) for v in self._prefs.values())

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def set(
        self,
        user_id: str,
        agent_id: str,
        prefs: AgentPreferences,
    ) -> AgentPreferences:
        """Set preferences for a specific agent. Returns the stored prefs."""
        # Ensure agent_id matches
        if prefs.agent_id != agent_id:
            prefs = prefs.model_copy(update={"agent_id": agent_id})
        with self._lock:
            if user_id not in self._prefs:
                self._prefs[user_id] = {}
            self._prefs[user_id][agent_id] = prefs
            self._re_persist_all()
        logger.info(
            "Agent preferences updated",
            user_id=user_id,
            agent_id=agent_id,
        )
        return prefs

    def reset(self, user_id: str = "default") -> None:
        """Remove all custom preferences for a user."""
        with self._lock:
            if user_id in self._prefs:
                del self._prefs[user_id]
                self._re_persist_all()
        logger.info("Agent preferences reset", user_id=user_id)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _re_persist_all(self) -> None:
        """Rewrite entire JSONL file from in-memory state."""
        if not self._persist_path:
            return
        try:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._persist_path, "w") as f:
                for user_id, agent_prefs in self._prefs.items():
                    for agent_id, prefs in agent_prefs.items():
                        record = {
                            "user_id": user_id,
                            "prefs": prefs.to_dict(),
                        }
                        f.write(json.dumps(record) + "\n")
        except OSError as exc:
            logger.error("Failed to persist agent preferences", error=str(exc))

    def _load_from_disk(self) -> None:
        """Load preferences from JSONL file."""
        count = 0
        try:
            with open(self._persist_path) as f:  # type: ignore[arg-type]
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    user_id = data.get("user_id", "default")
                    prefs = AgentPreferences.from_dict(data["prefs"])
                    if user_id not in self._prefs:
                        self._prefs[user_id] = {}
                    if prefs.agent_id not in self._prefs[user_id]:
                        self._prefs[user_id][prefs.agent_id] = prefs
                        count += 1
        except OSError as exc:
            logger.error(
                "Failed to load agent preferences from disk",
                path=str(self._persist_path),
                error=str(exc),
            )
        logger.info("Agent preferences loaded from disk", count=count)
