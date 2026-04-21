"""Chat endpoints — conversational interface to Providence data."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from fastapi import APIRouter, HTTPException

from providence.api.deps import get_state
from providence.api.schemas import (
    ChatCitation,
    ChatMessageRequest,
    ChatMessageResponse,
    ConversationDetail,
    ConversationMessage,
    ConversationSummary,
)
from providence.services.conversation_store import ChatMessage

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatMessageResponse)
async def send_message(req: ChatMessageRequest) -> ChatMessageResponse:
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

    # Process through chat engine
    response_text, citations_raw = state.chat_engine.process(req.message)

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

    return ChatMessageResponse(
        response=response_text,
        citations=citations,
        conversation_id=str(conv.conversation_id),
        timestamp=assistant_msg.timestamp,
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
