"""Base agent infrastructure for the Providence system.

All agents in the system inherit from BaseAgent, which defines the
common interface, health reporting, and content hashing utility.

AgentContext is the standard input for all agents — assembled by
CONTEXT-SVC from MarketStateFragments.
"""

import hashlib
import json
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Generic, Optional, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from providence.schemas.market_state import MarketStateFragment

T = TypeVar("T")


class AgentStatus(str, Enum):
    """Health status classification for an agent."""
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    UNHEALTHY = "UNHEALTHY"
    OFFLINE = "OFFLINE"


class AgentContext(BaseModel):
    """Input context assembled by CONTEXT-SVC for a Research Agent.

    Contains the relevant MarketStateFragments and a context window hash
    for exact input reproducibility.
    """

    model_config = ConfigDict(frozen=True)

    agent_id: str = Field(..., description="ID of the agent this context is for")
    trigger: str = Field(
        ...,
        description="What triggered this run: 'schedule', 'event', or 'manual'",
    )
    fragments: list[MarketStateFragment] = Field(
        default_factory=list,
        description="MarketStateFragments assembled by CONTEXT-SVC",
    )
    context_window_hash: str = Field(
        ...,
        description="SHA-256 of sorted fragment content hashes",
    )
    timestamp: datetime = Field(..., description="When this context was assembled")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata for the agent run",
    )


class HealthStatus(BaseModel):
    """Health status report for an agent.

    Used for monitoring, alerting, and governance oversight.
    """

    agent_id: str = Field(..., description="ID of the reporting agent")
    status: AgentStatus = Field(..., description="Current health status")
    last_run: Optional[datetime] = Field(default=None, description="Timestamp of last run")
    last_success: Optional[datetime] = Field(default=None, description="Timestamp of last successful run")
    error_count_24h: int = Field(default=0, ge=0, description="Errors in the last 24 hours")
    avg_latency_ms: float = Field(default=0.0, ge=0.0, description="Average processing latency in ms")
    message: Optional[str] = Field(default=None, description="Optional status message")


class BaseAgent(ABC, Generic[T]):
    """Abstract base class for all Providence agents.

    Every agent in the system — Perception, Cognition, Regime, Decision,
    and Execution — inherits from this class. It defines the common
    interface for processing and health reporting.

    Type parameter T is the output type of the agent (e.g., MarketStateFragment,
    BeliefObject, RegimeStateObject, etc.).
    """

    def __init__(self, agent_id: str, agent_type: str, version: str) -> None:
        self._agent_id = agent_id
        self._agent_type = agent_type
        self._version = version

    @property
    def agent_id(self) -> str:
        """Unique identifier for this agent instance."""
        return self._agent_id

    @property
    def agent_type(self) -> str:
        """Agent subsystem type (perception, cognition, regime, decision, execution)."""
        return self._agent_type

    @property
    def version(self) -> str:
        """Version string for this agent."""
        return self._version

    @abstractmethod
    async def process(self, context: AgentContext) -> T:
        """Process the given context and produce output.

        This is the core method that each agent implements. It receives
        an assembled context from CONTEXT-SVC and returns a typed output.

        Args:
            context: AgentContext with relevant MarketStateFragments.

        Returns:
            Agent-specific output (MarketStateFragment, BeliefObject, etc.).
        """
        ...

    @abstractmethod
    def get_health(self) -> HealthStatus:
        """Report current health status of this agent.

        Returns:
            HealthStatus with current metrics and status.
        """
        ...

    @staticmethod
    def compute_content_hash(data: dict[str, Any]) -> str:
        """Compute SHA-256 content hash from a dictionary.

        Uses deterministic serialization (sorted keys) so the same
        data always produces the same hash.

        Args:
            data: Dictionary to hash.

        Returns:
            Hex-encoded SHA-256 hash string.
        """
        serialized = json.dumps(data, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(serialized).hexdigest()
