"""Per-agent configuration for CONTEXT-SVC.

Defines what data types each agent consumes, its token budget,
entity scope, and peer context settings. Loaded from YAML config.

Spec Reference: Technical Spec v2.3, Section 4.6 (CONTEXT-SVC)
"""

from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field

from providence.schemas.enums import DataType


class AgentConfig(BaseModel):
    """Configuration for a single agent's context assembly."""

    model_config = ConfigDict(frozen=True)

    agent_id: str = Field(..., description="Agent identifier")
    consumes: list[DataType] = Field(
        ...,
        description="Data types this agent consumes from Perception",
    )
    max_token_budget: int = Field(
        default=100_000,
        gt=0,
        description="Maximum token budget for context window",
    )
    entity_scope: list[str] = Field(
        default_factory=list,
        description="Specific entities this agent covers (empty = all)",
    )
    peer_count: int = Field(
        default=0,
        ge=0,
        description="Number of peer entities to include in context",
    )
    peer_group: Optional[str] = Field(
        default=None,
        description="Peer group key from peer_groups.yaml",
    )
    priority_window_hours: int = Field(
        default=24,
        ge=1,
        description="Hours of data that are never dropped from context",
    )


class AgentConfigRegistry:
    """Registry of agent configurations loaded from YAML.

    Provides lookup by agent_id and validates all configs on load.
    """

    def __init__(self, configs: dict[str, AgentConfig] | None = None) -> None:
        self._configs: dict[str, AgentConfig] = configs or {}

    def get(self, agent_id: str) -> AgentConfig | None:
        """Get configuration for a specific agent."""
        return self._configs.get(agent_id)

    def get_or_raise(self, agent_id: str) -> AgentConfig:
        """Get configuration, raising if not found."""
        config = self._configs.get(agent_id)
        if config is None:
            raise KeyError(f"No configuration found for agent: {agent_id}")
        return config

    @property
    def agent_ids(self) -> list[str]:
        """List all configured agent IDs."""
        return list(self._configs.keys())

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AgentConfigRegistry":
        """Load agent configurations from a YAML file.

        Expected YAML structure:
            agents:
              COGNIT-FUNDAMENTAL:
                consumes: [FILING_10K, FILING_10Q, PRICE_OHLCV]
                max_token_budget: 100000
                peer_count: 3
                ...
        """
        path = Path(path)
        if not path.exists():
            return cls({})

        with open(path) as f:
            raw = yaml.safe_load(f) or {}

        agents_raw: dict[str, Any] = raw.get("agents", {})
        configs: dict[str, AgentConfig] = {}

        for agent_id, agent_data in agents_raw.items():
            # Convert string data type names to DataType enums
            consumes_raw = agent_data.get("consumes", [])
            consumes = [DataType(dt) for dt in consumes_raw]

            configs[agent_id] = AgentConfig(
                agent_id=agent_id,
                consumes=consumes,
                max_token_budget=agent_data.get("max_token_budget", 100_000),
                entity_scope=agent_data.get("entity_scope", []),
                peer_count=agent_data.get("peer_count", 0),
                peer_group=agent_data.get("peer_group"),
                priority_window_hours=agent_data.get("priority_window_hours", 24),
            )

        return cls(configs)

    @classmethod
    def from_dict(cls, agents: dict[str, dict[str, Any]]) -> "AgentConfigRegistry":
        """Create registry from a plain dictionary (useful for testing)."""
        configs: dict[str, AgentConfig] = {}
        for agent_id, agent_data in agents.items():
            consumes_raw = agent_data.get("consumes", [])
            consumes = [
                DataType(dt) if isinstance(dt, str) else dt
                for dt in consumes_raw
            ]
            configs[agent_id] = AgentConfig(
                agent_id=agent_id,
                consumes=consumes,
                max_token_budget=agent_data.get("max_token_budget", 100_000),
                entity_scope=agent_data.get("entity_scope", []),
                peer_count=agent_data.get("peer_count", 0),
                peer_group=agent_data.get("peer_group"),
                priority_window_hours=agent_data.get("priority_window_hours", 24),
            )
        return cls(configs)
