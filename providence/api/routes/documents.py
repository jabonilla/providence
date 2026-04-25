"""Document management endpoints — list, detail, delete uploaded documents.

Uploaded documents are stored as MarketStateFragments with
data_type=USER_DOCUMENT and agent_id=USER-UPLOAD.
"""

from __future__ import annotations

from typing import Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, UploadFile, File

from providence.api.deps import get_state

router = APIRouter(prefix="/documents", tags=["documents"])


# ── List Documents ────────────────────────────────────────────────

@router.get("")
async def list_documents(
    limit: int = Query(100, ge=1, le=500),
) -> list[dict]:
    """List all uploaded documents with metadata."""
    state = get_state()
    store = state.fragment_store

    from providence.schemas.enums import DataType

    # Query fragments with USER_DOCUMENT data type
    fragments = store.query(
        data_types={DataType.USER_DOCUMENT},
        limit=limit,
    )

    return [
        {
            "fragment_id": str(f.fragment_id),
            "filename": f.payload.get("filename", f.entity) if isinstance(f.payload, dict) else f.entity,
            "content_type": f.payload.get("content_type", "unknown") if isinstance(f.payload, dict) else "unknown",
            "size_bytes": f.payload.get("size_bytes", 0) if isinstance(f.payload, dict) else 0,
            "text_length": len(f.payload.get("text", "")) if isinstance(f.payload, dict) else 0,
            "uploaded_at": f.timestamp.isoformat() if hasattr(f.timestamp, "isoformat") else str(f.timestamp),
            "validation_status": f.validation_status.value if hasattr(f.validation_status, "value") else str(f.validation_status),
        }
        for f in fragments
    ]


# ── Document Detail ───────────────────────────────────────────────

@router.get("/{fragment_id}")
async def get_document(fragment_id: UUID) -> dict:
    """Get a single uploaded document with full metadata."""
    state = get_state()
    fragment = state.fragment_store.get(fragment_id)

    if fragment is None:
        raise HTTPException(status_code=404, detail="Document not found")

    from providence.schemas.enums import DataType

    if fragment.data_type != DataType.USER_DOCUMENT:
        raise HTTPException(status_code=404, detail="Document not found")

    payload = fragment.payload if isinstance(fragment.payload, dict) else {}

    return {
        "fragment_id": str(fragment.fragment_id),
        "filename": payload.get("filename", fragment.entity),
        "content_type": payload.get("content_type", "unknown"),
        "size_bytes": payload.get("size_bytes", 0),
        "text_length": len(payload.get("text", "")),
        "text_preview": payload.get("text", "")[:500],
        "uploaded_at": fragment.timestamp.isoformat() if hasattr(fragment.timestamp, "isoformat") else str(fragment.timestamp),
        "validation_status": fragment.validation_status.value if hasattr(fragment.validation_status, "value") else str(fragment.validation_status),
        "source_hash": fragment.source_hash,
    }


# ── Upload Document ───────────────────────────────────────────────

@router.post("")
async def upload_document(
    file: UploadFile = File(...),
) -> dict:
    """Upload a research document and store as a MarketStateFragment.

    This is an alias for /chat/upload that returns the same data
    but lives under the /documents namespace.
    """
    import hashlib
    import json
    from datetime import datetime, timezone

    from providence.schemas.enums import DataType, ValidationStatus
    from providence.schemas.market_state import MarketStateFragment

    state = get_state()

    filename = file.filename or "unnamed"
    content_type = file.content_type or "application/octet-stream"
    raw_bytes = await file.read()
    size_bytes = len(raw_bytes)

    ext = ""
    if "." in filename:
        ext = "." + filename.rsplit(".", 1)[-1].lower()

    _TEXT_EXTENSIONS = {".txt", ".md", ".csv", ".tsv", ".log", ".html", ".xml"}
    _JSON_EXTENSIONS = {".json", ".jsonl"}
    _PDF_EXTENSIONS = {".pdf"}

    text = ""
    if ext in _TEXT_EXTENSIONS:
        try:
            text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            text = raw_bytes.decode("latin-1", errors="replace")
    elif ext in _JSON_EXTENSIONS:
        try:
            text = raw_bytes.decode("utf-8")
            json.loads(text)
        except (UnicodeDecodeError, json.JSONDecodeError):
            text = ""
    elif ext in _PDF_EXTENSIONS:
        try:
            text = raw_bytes.decode("utf-8", errors="ignore")
            printable_ratio = sum(
                1 for c in text[:1000] if c.isprintable() or c.isspace()
            ) / max(len(text[:1000]), 1)
            if printable_ratio < 0.5:
                text = f"[Binary PDF: {filename}, {size_bytes} bytes]"
        except Exception:
            text = f"[Binary PDF: {filename}, {size_bytes} bytes]"
    else:
        text = f"[File: {filename}, {size_bytes} bytes, type: {content_type}]"

    payload = {
        "filename": filename,
        "content_type": content_type,
        "text": text,
        "size_bytes": size_bytes,
    }

    source_hash = hashlib.sha256(raw_bytes).hexdigest()
    now = datetime.now(timezone.utc)

    fragment = MarketStateFragment(
        agent_id="USER-UPLOAD",
        timestamp=now,
        source_timestamp=now,
        entity=filename,
        data_type=DataType.USER_DOCUMENT,
        schema_version="1.0.0",
        source_hash=source_hash,
        validation_status=ValidationStatus.VALID,
        payload=payload,
    )

    state.fragment_store.append(fragment)

    # Track usage
    if state.usage_tracker:
        state.usage_tracker.record("doc_upload", detail={"filename": filename})

    return {
        "status": "success",
        "fragment_id": str(fragment.fragment_id),
        "filename": filename,
        "content_type": content_type,
        "size_bytes": size_bytes,
        "text_length": len(text),
    }


# ── Delete Document ───────────────────────────────────────────────

@router.delete("/{fragment_id}")
async def delete_document(fragment_id: UUID) -> dict:
    """Mark a document fragment as deleted (soft delete via quarantine).

    Since fragment stores are append-only, we mark the validation_status
    as QUARANTINED rather than physically deleting.
    """
    state = get_state()
    fragment = state.fragment_store.get(fragment_id)

    if fragment is None:
        raise HTTPException(status_code=404, detail="Document not found")

    from providence.schemas.enums import DataType, ValidationStatus

    if fragment.data_type != DataType.USER_DOCUMENT:
        raise HTTPException(status_code=404, detail="Document not found")

    # Soft delete: create a replacement fragment with QUARANTINED status
    from datetime import datetime, timezone
    from providence.schemas.market_state import MarketStateFragment

    quarantined = MarketStateFragment(
        fragment_id=fragment.fragment_id,
        agent_id=fragment.agent_id,
        timestamp=fragment.timestamp,
        source_timestamp=fragment.source_timestamp,
        entity=fragment.entity,
        data_type=fragment.data_type,
        schema_version=fragment.schema_version,
        source_hash=fragment.source_hash,
        validation_status=ValidationStatus.QUARANTINED,
        payload={"deleted": True, "original_filename": fragment.payload.get("filename", "") if isinstance(fragment.payload, dict) else ""},
    )

    # Replace in store (overwrite the in-memory entry)
    state.fragment_store._fragments[fragment.fragment_id] = quarantined

    return {
        "status": "deleted",
        "fragment_id": str(fragment_id),
    }
