"""API key management endpoints.

Allows users to store, update, and validate their own API keys
(Polygon, Anthropic, FRED, EDGAR, Alpaca) so each trader
doesn't rely on a shared set of credentials.

Keys are stored in a JSON file on disk and loaded into env vars
at runtime. The server can be hot-reloaded with new keys.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import structlog
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from providence.api.deps import get_state

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/config/keys", tags=["config"])

# Key definitions — what we support and how to validate them
SUPPORTED_KEYS = {
    "POLYGON_API_KEY": {
        "label": "Polygon.io",
        "description": "Market data — prices, options, news",
        "required": True,
        "prefix": None,
    },
    "ANTHROPIC_API_KEY": {
        "label": "Anthropic",
        "description": "Claude LLM for adaptive agents",
        "required": True,
        "prefix": "sk-ant-",
    },
    "FRED_API_KEY": {
        "label": "FRED (Federal Reserve)",
        "description": "Macro economic data — yields, GDP, CPI",
        "required": True,
        "prefix": None,
    },
    "EDGAR_USER_AGENT": {
        "label": "SEC EDGAR",
        "description": "SEC filings — 10-K, 10-Q, 8-K (format: App/Version email)",
        "required": True,
        "prefix": None,
    },
    "ALPACA_API_KEY": {
        "label": "Alpaca (API Key)",
        "description": "Paper/live trading — order execution",
        "required": False,
        "prefix": None,
    },
    "ALPACA_SECRET_KEY": {
        "label": "Alpaca (Secret Key)",
        "description": "Alpaca secret key — paired with API key",
        "required": False,
        "prefix": None,
    },
}


class KeyUpdate(BaseModel):
    """Request to set one or more API keys."""
    keys: dict[str, str]


class KeyStatus(BaseModel):
    """Status of a single API key."""
    key_name: str
    label: str
    description: str
    required: bool
    is_set: bool
    masked_value: Optional[str] = None  # e.g. "sk-ant-...3xYz"


class KeysStatusResponse(BaseModel):
    """Status of all API keys."""
    keys: list[KeyStatus]
    all_required_set: bool
    paper_trading_ready: bool


def _get_keys_path() -> Path:
    """Get the path to the stored keys file."""
    state = get_state()
    data_dir = state.extra.get("data_dir")
    if data_dir:
        return Path(data_dir) / "api_keys.json"
    return Path("data") / "api_keys.json"


def _load_stored_keys() -> dict[str, str]:
    """Load stored keys from disk."""
    path = _get_keys_path()
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_keys(keys: dict[str, str]) -> None:
    """Save keys to disk."""
    path = _get_keys_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(keys, f, indent=2)
    # Also chmod to owner-only read/write
    try:
        path.chmod(0o600)
    except OSError:
        pass


def _mask_key(value: str) -> str:
    """Mask a key value for display."""
    if not value:
        return ""
    if len(value) <= 8:
        return "***"
    return f"{value[:6]}...{value[-4:]}"


def _get_effective_value(key_name: str) -> str:
    """Get the effective value for a key (stored file > env var)."""
    stored = _load_stored_keys()
    return stored.get(key_name, "") or os.environ.get(key_name, "")


@router.get("", response_model=KeysStatusResponse)
async def get_keys_status() -> KeysStatusResponse:
    """Get the status of all API keys (masked values, not full keys)."""
    statuses = []
    all_required_set = True
    alpaca_ready = False

    for key_name, meta in SUPPORTED_KEYS.items():
        value = _get_effective_value(key_name)
        is_set = bool(value)

        if meta["required"] and not is_set:
            all_required_set = False

        statuses.append(KeyStatus(
            key_name=key_name,
            label=meta["label"],
            description=meta["description"],
            required=meta["required"],
            is_set=is_set,
            masked_value=_mask_key(value) if is_set else None,
        ))

    # Paper trading needs both Alpaca keys
    alpaca_key = _get_effective_value("ALPACA_API_KEY")
    alpaca_secret = _get_effective_value("ALPACA_SECRET_KEY")
    alpaca_ready = bool(alpaca_key and alpaca_secret)

    return KeysStatusResponse(
        keys=statuses,
        all_required_set=all_required_set,
        paper_trading_ready=alpaca_ready,
    )


@router.put("")
async def update_keys(request: KeyUpdate) -> dict:
    """Update one or more API keys.

    Keys are validated for basic format, stored to disk,
    and applied to the current process environment.
    """
    errors = []

    for key_name, value in request.keys.items():
        if key_name not in SUPPORTED_KEYS:
            errors.append(f"Unknown key: {key_name}")
            continue

        # Basic validation
        meta = SUPPORTED_KEYS[key_name]
        if meta["prefix"] and value and not value.startswith(meta["prefix"]):
            errors.append(
                f"{meta['label']} key should start with '{meta['prefix']}'"
            )
            continue

    if errors:
        raise HTTPException(status_code=400, detail="; ".join(errors))

    # Load existing, merge, save
    stored = _load_stored_keys()
    for key_name, value in request.keys.items():
        if value:  # Don't store empty strings
            stored[key_name] = value
            # Also update the live environment
            os.environ[key_name] = value
        elif key_name in stored:
            # Empty value = remove
            del stored[key_name]
            if key_name in os.environ:
                del os.environ[key_name]

    _save_keys(stored)

    logger.info(
        "API keys updated",
        keys_updated=list(request.keys.keys()),
        total_stored=len(stored),
    )

    return {
        "status": "ok",
        "keys_updated": list(request.keys.keys()),
        "message": "Keys saved. Restart server or trigger rebuild for full effect.",
    }


@router.post("/validate")
async def validate_keys() -> dict:
    """Quick validation of stored keys (checks format, not liveness)."""
    results = {}
    for key_name, meta in SUPPORTED_KEYS.items():
        value = _get_effective_value(key_name)
        if not value:
            results[key_name] = {"valid": False, "reason": "Not set"}
        elif meta["prefix"] and not value.startswith(meta["prefix"]):
            results[key_name] = {
                "valid": False,
                "reason": f"Should start with '{meta['prefix']}'",
            }
        else:
            results[key_name] = {"valid": True, "reason": "Format OK"}

    all_valid = all(
        results[k]["valid"]
        for k in SUPPORTED_KEYS
        if SUPPORTED_KEYS[k]["required"]
    )

    return {"results": results, "all_required_valid": all_valid}
