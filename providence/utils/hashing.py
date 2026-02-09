"""Deterministic hashing utilities for Providence.

All hashing uses SHA-256 with deterministic serialization (sorted keys)
to guarantee that identical data always produces identical hashes.
"""

import hashlib
import json
from typing import Any

from providence.schemas.market_state import MarketStateFragment


def compute_content_hash(data: Any) -> str:
    """Compute SHA-256 hash of arbitrary data.

    Serializes the data deterministically using sorted keys and
    computes the SHA-256 hex digest.

    Args:
        data: Any JSON-serializable data (dict, list, primitive, or
              Pydantic model with .model_dump()).

    Returns:
        Hex-encoded SHA-256 hash string (64 characters).
    """
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    else:
        serializable = data

    serialized = json.dumps(serializable, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def compute_context_window_hash(fragments: list[MarketStateFragment]) -> str:
    """Compute SHA-256 hash of a list of MarketStateFragments.

    Sorts fragment content hashes alphabetically, concatenates them,
    and hashes the result. This ensures the same set of fragments
    always produces the same context window hash regardless of
    input ordering.

    Args:
        fragments: List of MarketStateFragments to hash.

    Returns:
        Hex-encoded SHA-256 hash string (64 characters).
    """
    if not fragments:
        # Hash of empty input — consistent sentinel value
        return hashlib.sha256(b"").hexdigest()

    sorted_hashes = sorted(f.version for f in fragments)
    combined = "|".join(sorted_hashes).encode("utf-8")
    return hashlib.sha256(combined).hexdigest()
