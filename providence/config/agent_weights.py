"""Per-user agent weight configuration for belief synthesis.

Default weights match DECIDE-SYNTH prompt template v1.0:
  COGNIT-FUNDAMENTAL: 0.25
  COGNIT-MACRO: 0.20
  COGNIT-TECHNICAL: 0.15
  COGNIT-NARRATIVE: 0.15
  COGNIT-EVENT: 0.15
  COGNIT-CROSSSEC: 0.10
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import structlog
from pydantic import BaseModel, Field, field_validator

logger = structlog.get_logger()


DEFAULT_WEIGHTS: dict[str, float] = {
    "COGNIT-FUNDAMENTAL": 0.25,
    "COGNIT-MACRO": 0.20,
    "COGNIT-TECHNICAL": 0.15,
    "COGNIT-NARRATIVE": 0.15,
    "COGNIT-EVENT": 0.15,
    "COGNIT-CROSSSEC": 0.10,
}


class AgentWeightConfig(BaseModel):
    """Agent weight configuration for a single user.

    Weights map agent_id -> float in [0.0, 1.0].
    Use ``normalized()`` to get a copy scaled to sum to 1.0.
    """

    user_id: str = Field(default="default", description="Owner user ID")
    weights: dict[str, float] = Field(
        default_factory=lambda: dict(DEFAULT_WEIGHTS),
        description="Agent ID to weight mapping",
    )

    @field_validator("weights")
    @classmethod
    def _validate_weights(cls, v: dict[str, float]) -> dict[str, float]:
        for agent_id, w in v.items():
            if not 0.0 <= w <= 1.0:
                raise ValueError(
                    f"Weight for {agent_id} must be in [0.0, 1.0], got {w}"
                )
        return v

    def normalized(self) -> dict[str, float]:
        """Return a copy of weights scaled to sum to 1.0.

        If all weights are zero, returns equal weights.
        """
        total = sum(self.weights.values())
        if total == 0.0:
            n = len(self.weights)
            if n == 0:
                return {}
            return {k: 1.0 / n for k in self.weights}
        return {k: v / total for k, v in self.weights.items()}

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "weights": self.weights,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentWeightConfig:
        return cls(
            user_id=data.get("user_id", "default"),
            weights=data.get("weights", dict(DEFAULT_WEIGHTS)),
        )


class AgentWeightStore:
    """Thread-safe, JSONL-backed store for per-user agent weight configs.

    Follows the same pattern as ConversationStore/FragmentStore:
    in-memory dict with RLock, optional JSONL persistence.
    """

    def __init__(self, persist_path: Path | None = None) -> None:
        self._lock = threading.RLock()
        self._configs: dict[str, AgentWeightConfig] = {}
        self._persist_path = persist_path
        if persist_path and persist_path.exists():
            self._load_from_disk()

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get(self, user_id: str = "default") -> AgentWeightConfig:
        """Get weight config for a user. Returns defaults if not set."""
        with self._lock:
            config = self._configs.get(user_id)
            if config is None:
                return AgentWeightConfig(user_id=user_id)
            return config

    def list_users(self) -> list[str]:
        """List all user IDs with custom weight configs."""
        with self._lock:
            return list(self._configs.keys())

    def count(self) -> int:
        """Total number of stored configs."""
        with self._lock:
            return len(self._configs)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def set(self, user_id: str, weights: dict[str, float]) -> AgentWeightConfig:
        """Set weights for a user. Returns the stored config."""
        config = AgentWeightConfig(user_id=user_id, weights=weights)
        with self._lock:
            self._configs[user_id] = config
            self._re_persist_all()
        logger.info("Agent weights updated", user_id=user_id)
        return config

    def reset(self, user_id: str = "default") -> AgentWeightConfig:
        """Reset a user's weights to defaults."""
        config = AgentWeightConfig(
            user_id=user_id,
            weights=dict(DEFAULT_WEIGHTS),
        )
        with self._lock:
            self._configs[user_id] = config
            self._re_persist_all()
        logger.info("Agent weights reset to defaults", user_id=user_id)
        return config

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
                for config in self._configs.values():
                    f.write(json.dumps(config.to_dict()) + "\n")
        except OSError as exc:
            logger.error("Failed to persist agent weights", error=str(exc))

    def _load_from_disk(self) -> None:
        """Load configs from JSONL file."""
        count = 0
        try:
            with open(self._persist_path) as f:  # type: ignore[arg-type]
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    config = AgentWeightConfig.from_dict(data)
                    if config.user_id not in self._configs:
                        self._configs[config.user_id] = config
                        count += 1
        except OSError as exc:
            logger.error(
                "Failed to load agent weights from disk",
                path=str(self._persist_path),
                error=str(exc),
            )
        logger.info("Agent weights loaded from disk", count=count)
