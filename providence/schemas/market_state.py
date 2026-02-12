"""MarketStateFragment schema — the core data object of the Perception System.

Spec Reference: Technical Spec v2.3, Section 2.1

Produced by: Perception System (Data Agents)
Consumed by: Cognition System, Regime System, Learning System

Immutability Rules:
- Fragments are append-only. Once written, they are never modified or deleted.
- Late corrections are issued as new fragments with a 'supersedes' reference.
- Quarantined fragments are flagged but retained for audit purposes.
"""

import hashlib
import json
from datetime import datetime
from typing import Any, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from providence.schemas.enums import DataType, ValidationStatus


class MarketStateFragment(BaseModel):
    """A single unit of market data ingested by a Perception agent.

    Content-addressable via SHA-256 hash of the serialized payload.
    Immutable after creation (frozen=True).
    """

    model_config = ConfigDict(frozen=True)

    fragment_id: UUID = Field(default_factory=uuid4, description="Unique identifier for this fragment")
    agent_id: str = Field(..., description="ID of the Perception agent that produced this fragment")
    timestamp: datetime = Field(..., description="Ingestion time (must include timezone)")
    source_timestamp: datetime = Field(..., description="Original data publication time")
    version: str = Field(default="", description="SHA-256 content hash, computed automatically")
    entity: Optional[str] = Field(..., description="Ticker, index, or null for macro data")
    data_type: DataType = Field(..., description="Data type from the Data Type Registry")
    schema_version: str = Field(default="1.0.0", description="SemVer of the payload schema")
    source_hash: str = Field(..., description="SHA-256 hash of raw source data for provenance")
    validation_status: ValidationStatus = Field(
        default=ValidationStatus.VALID,
        description="Validation status: VALID, QUARANTINED, or PARTIAL",
    )
    payload: dict[str, Any] = Field(..., description="Typed payload per data_type")

    @field_validator("timestamp", "source_timestamp")
    @classmethod
    def must_have_timezone(cls, v: datetime) -> datetime:
        """Timestamps must include timezone information."""
        if v.tzinfo is None:
            raise ValueError("Timestamp must include timezone information (tzinfo cannot be None)")
        return v

    @field_validator("schema_version")
    @classmethod
    def must_be_semver(cls, v: str) -> str:
        """Schema version must follow SemVer format (X.Y.Z)."""
        parts = v.split(".")
        if len(parts) != 3:
            raise ValueError(f"schema_version must be SemVer format (X.Y.Z), got: {v}")
        for part in parts:
            if not part.isdigit():
                raise ValueError(f"schema_version components must be numeric, got: {v}")
        return v

    @model_validator(mode="after")
    def compute_content_hash(self) -> "MarketStateFragment":
        """Compute SHA-256 content hash from deterministically serialized payload.

        Uses sorted keys for deterministic serialization so the same
        payload always produces the same hash.
        """
        payload_bytes = json.dumps(self.payload, sort_keys=True, default=str).encode("utf-8")
        content_hash = hashlib.sha256(payload_bytes).hexdigest()
        # Use object.__setattr__ because model is frozen
        object.__setattr__(self, "version", content_hash)
        return self
