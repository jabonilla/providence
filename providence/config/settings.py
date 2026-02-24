"""Centralized environment-based settings for Providence.

Reads configuration from environment variables with sensible defaults.
All API clients already read their own keys from env, so this module
provides the system-level settings that aren't client-specific.

Usage:
    from providence.config.settings import get_settings
    settings = get_settings()
"""

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ProvidenceSettings:
    """Immutable application settings loaded from environment."""

    # Logging
    log_level: str = "INFO"

    # Storage
    data_dir: Path = Path("data")

    # Pipeline timing
    interval_seconds: int = 300
    default_timeout: float = 120.0

    # Agent filtering
    skip_perception: bool = False
    skip_adaptive: bool = False

    # LLM model override (empty = use client default)
    llm_model: str = ""
    llm_max_tokens: int = 4096

    # Feature flags
    persist_storage: bool = True
    extract_beliefs: bool = True

    def has_polygon_key(self) -> bool:
        return bool(os.environ.get("POLYGON_API_KEY"))

    def has_edgar_agent(self) -> bool:
        return bool(os.environ.get("EDGAR_USER_AGENT"))

    def has_fred_key(self) -> bool:
        return bool(os.environ.get("FRED_API_KEY"))

    def has_anthropic_key(self) -> bool:
        return bool(os.environ.get("ANTHROPIC_API_KEY"))

    def available_api_summary(self) -> dict[str, bool]:
        """Return which APIs are configured."""
        return {
            "polygon": self.has_polygon_key(),
            "edgar": self.has_edgar_agent(),
            "fred": self.has_fred_key(),
            "anthropic": self.has_anthropic_key(),
        }


def get_settings() -> ProvidenceSettings:
    """Load settings from environment variables.

    Environment variables (all optional):
        PROVIDENCE_LOG_LEVEL: Logging level (default: INFO)
        PROVIDENCE_DATA_DIR: Storage directory (default: data)
        PROVIDENCE_INTERVAL_SECONDS: Continuous mode interval (default: 300)
        PROVIDENCE_DEFAULT_TIMEOUT: Stage timeout in seconds (default: 120)
        PROVIDENCE_SKIP_PERCEPTION: Skip perception agents (default: false)
        PROVIDENCE_SKIP_ADAPTIVE: Skip adaptive agents (default: false)
        PROVIDENCE_LLM_MODEL: Override LLM model name
        PROVIDENCE_LLM_MAX_TOKENS: Override max tokens
        PROVIDENCE_PERSIST_STORAGE: Enable JSONL persistence (default: true)
        PROVIDENCE_EXTRACT_BELIEFS: Extract beliefs from cognition (default: true)
    """
    def _bool(key: str, default: bool = False) -> bool:
        val = os.environ.get(key, "").lower()
        if val in ("1", "true", "yes"):
            return True
        if val in ("0", "false", "no"):
            return False
        return default

    return ProvidenceSettings(
        log_level=os.environ.get("PROVIDENCE_LOG_LEVEL", "INFO").upper(),
        data_dir=Path(os.environ.get("PROVIDENCE_DATA_DIR", "data")),
        interval_seconds=int(os.environ.get("PROVIDENCE_INTERVAL_SECONDS", "300")),
        default_timeout=float(os.environ.get("PROVIDENCE_DEFAULT_TIMEOUT", "120")),
        skip_perception=_bool("PROVIDENCE_SKIP_PERCEPTION", False),
        skip_adaptive=_bool("PROVIDENCE_SKIP_ADAPTIVE", False),
        llm_model=os.environ.get("PROVIDENCE_LLM_MODEL", ""),
        llm_max_tokens=int(os.environ.get("PROVIDENCE_LLM_MAX_TOKENS", "4096")),
        persist_storage=_bool("PROVIDENCE_PERSIST_STORAGE", True),
        extract_beliefs=_bool("PROVIDENCE_EXTRACT_BELIEFS", True),
    )
