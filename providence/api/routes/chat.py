"""Chat endpoints — conversational interface to Providence data."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile

from providence.api.deps import get_state
from providence.api.schemas import (
    ChatCitation,
    ChatMessageInResponse,
    ChatMessageRequest,
    ChatSendResponse,
    ConversationDetail,
    ConversationMessage,
    ConversationSummary,
    DocumentUploadResponse,
)
from providence.services.conversation_store import ChatMessage

router = APIRouter(prefix="/chat", tags=["chat"])


# ── Allowed file extensions for text extraction ──────────────────────
_TEXT_EXTENSIONS = {".txt", ".md", ".csv"}
_JSON_EXTENSIONS = {".json"}
_PDF_EXTENSIONS = {".pdf"}


@router.post("/send", response_model=ChatSendResponse)
async def send_message(req: ChatMessageRequest) -> ChatSendResponse:
    """Send a message and receive a response.

    If conversation_id is null, a new conversation is created.
    Otherwise the message is appended to the existing conversation.
    """
    state = get_state()

    if state.chat_engine is None or state.conversation_store is None:
        raise HTTPException(status_code=503, detail="Chat service not initialized")

    # Resolve or create conversation
    if req.conversation_id:
        try:
            conv_id = UUID(req.conversation_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid conversation ID format")
        conv = state.conversation_store.get(conv_id)
        if conv is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
    else:
        conv = state.conversation_store.create()

    now = datetime.now(timezone.utc)

    # Record user message
    user_msg = ChatMessage(role="user", content=req.message, timestamp=now)
    state.conversation_store.add_message(conv.conversation_id, user_msg)

    # Build conversation history from the last few messages for context
    conversation_history: list[dict[str, Any]] = []
    recent_messages = conv.messages[-10:]  # Last 10 messages for context
    for m in recent_messages:
        conversation_history.append({
            "role": m.role,
            "content": m.content,
            "timestamp": m.timestamp.isoformat(),
        })

    # Process through chat engine with conversation context
    response_text, citations_raw = state.chat_engine.process(
        req.message,
        conversation_history=conversation_history,
    )

    # Convert citations
    citations = [
        ChatCitation(
            type=c["type"],
            id=str(c["id"]),
            label=c["label"],
            url=c["url"],
        )
        for c in citations_raw
    ]

    # Record assistant message
    assistant_msg = ChatMessage(
        role="assistant",
        content=response_text,
        citations=[c.model_dump() for c in citations],
        timestamp=datetime.now(timezone.utc),
    )
    state.conversation_store.add_message(conv.conversation_id, assistant_msg)

    # Build response in the shape the frontend expects
    message_id = str(uuid4())
    return ChatSendResponse(
        message=ChatMessageInResponse(
            id=message_id,
            role="assistant",
            content=response_text,
            citations=citations,
            timestamp=assistant_msg.timestamp,
        ),
        conversation_id=str(conv.conversation_id),
    )


@router.get("/conversations", response_model=list[ConversationSummary])
async def list_conversations(
    limit: int = 50,
    offset: int = 0,
) -> list[ConversationSummary]:
    """List conversations, newest first."""
    state = get_state()

    if state.conversation_store is None:
        raise HTTPException(status_code=503, detail="Chat service not initialized")

    convs = state.conversation_store.list_conversations(limit=limit, offset=offset)
    return [
        ConversationSummary(
            id=str(c.conversation_id),
            title=c.title,
            message_count=c.message_count,
            last_message_at=c.last_message_at,
            created_at=c.created_at,
        )
        for c in convs
    ]


@router.get(
    "/conversations/{conversation_id}/messages",
    response_model=list[ConversationMessage],
)
async def get_conversation_messages(conversation_id: str) -> list[ConversationMessage]:
    """Get all messages for a conversation."""
    state = get_state()

    if state.conversation_store is None:
        raise HTTPException(status_code=503, detail="Chat service not initialized")

    try:
        conv_id = UUID(conversation_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid conversation ID format")

    conv = state.conversation_store.get(conv_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return [
        ConversationMessage(
            role=m.role,
            content=m.content,
            citations=[
                ChatCitation(**c) if isinstance(c, dict) else c
                for c in m.citations
            ],
            timestamp=m.timestamp,
        )
        for m in conv.messages
    ]


@router.get("/conversations/{conversation_id}", response_model=ConversationDetail)
async def get_conversation(conversation_id: str) -> ConversationDetail:
    """Get full conversation history."""
    state = get_state()

    if state.conversation_store is None:
        raise HTTPException(status_code=503, detail="Chat service not initialized")

    try:
        conv_id = UUID(conversation_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid conversation ID format")

    conv = state.conversation_store.get(conv_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    messages = [
        ConversationMessage(
            role=m.role,
            content=m.content,
            citations=[
                ChatCitation(**c) if isinstance(c, dict) else c
                for c in m.citations
            ],
            timestamp=m.timestamp,
        )
        for m in conv.messages
    ]

    return ConversationDetail(
        id=str(conv.conversation_id),
        title=conv.title,
        messages=messages,
        created_at=conv.created_at,
    )


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
) -> DocumentUploadResponse:
    """Upload a research document and store it as a MarketStateFragment.

    Accepts: .txt, .md, .csv, .json, .pdf (text-only), and others (metadata only).
    """
    state = get_state()

    filename = file.filename or "unnamed"
    content_type = file.content_type or "application/octet-stream"
    raw_bytes = await file.read()
    size_bytes = len(raw_bytes)

    # Determine file extension
    ext = ""
    if "." in filename:
        ext = "." + filename.rsplit(".", 1)[-1].lower()

    # Extract text based on file type
    text = ""
    if ext in _TEXT_EXTENSIONS:
        try:
            text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            text = raw_bytes.decode("latin-1", errors="replace")
    elif ext in _JSON_EXTENSIONS:
        try:
            text = raw_bytes.decode("utf-8")
            # Validate JSON
            json.loads(text)
        except (UnicodeDecodeError, json.JSONDecodeError):
            text = ""
    elif ext in _PDF_EXTENSIONS:
        # Best-effort text extraction for text-based PDFs
        try:
            text = raw_bytes.decode("utf-8", errors="ignore")
            # If mostly non-printable, treat as binary
            printable_ratio = sum(1 for c in text[:1000] if c.isprintable() or c.isspace()) / max(len(text[:1000]), 1)
            if printable_ratio < 0.5:
                text = f"[Binary PDF uploaded: {filename}, {size_bytes} bytes]"
        except Exception:
            text = f"[Binary PDF uploaded: {filename}, {size_bytes} bytes]"
    else:
        text = f"[File uploaded: {filename}, {size_bytes} bytes, type: {content_type}]"

    # Build payload
    payload = {
        "filename": filename,
        "content_type": content_type,
        "text": text,
        "size_bytes": size_bytes,
    }

    # Compute source hash from raw bytes
    source_hash = hashlib.sha256(raw_bytes).hexdigest()

    # Create MarketStateFragment
    from providence.schemas.enums import DataType, ValidationStatus
    from providence.schemas.market_state import MarketStateFragment

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

    # Store in FragmentStore
    state.fragment_store.append(fragment)

    return DocumentUploadResponse(
        status="success",
        filename=filename,
        fragment_id=str(fragment.fragment_id),
        content_type=content_type,
        text_length=len(text),
        message=f"Document '{filename}' uploaded and stored as fragment {fragment.fragment_id}.",
    )
