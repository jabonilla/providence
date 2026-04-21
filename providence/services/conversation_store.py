"""ConversationStore — append-only, JSONL-backed conversation persistence.

Follows the same pattern as FragmentStore/BeliefStore/RunStore:
thread-safe via RLock, in-memory indexing, optional JSONL persistence.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from uuid import UUID, uuid4

import structlog

logger = structlog.get_logger()


class ChatMessage:
    """A single message in a conversation."""

    __slots__ = ("role", "content", "citations", "timestamp")

    def __init__(
        self,
        *,
        role: str,
        content: str,
        citations: list[dict[str, Any]] | None = None,
        timestamp: datetime | None = None,
    ) -> None:
        self.role = role
        self.content = content
        self.citations = citations or []
        self.timestamp = timestamp or datetime.now(timezone.utc)

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "citations": self.citations,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChatMessage:
        ts = data.get("timestamp")
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        return cls(
            role=data["role"],
            content=data["content"],
            citations=data.get("citations", []),
            timestamp=ts,
        )


class Conversation:
    """A conversation thread containing ordered messages."""

    __slots__ = ("conversation_id", "title", "messages", "created_at", "updated_at")

    def __init__(
        self,
        *,
        conversation_id: UUID | None = None,
        title: str = "New conversation",
        messages: list[ChatMessage] | None = None,
        created_at: datetime | None = None,
        updated_at: datetime | None = None,
    ) -> None:
        self.conversation_id = conversation_id or uuid4()
        self.title = title
        self.messages = messages or []
        self.created_at = created_at or datetime.now(timezone.utc)
        self.updated_at = updated_at or datetime.now(timezone.utc)

    @property
    def message_count(self) -> int:
        return len(self.messages)

    @property
    def last_message_at(self) -> datetime | None:
        if self.messages:
            return self.messages[-1].timestamp
        return None

    def add_message(self, msg: ChatMessage) -> None:
        self.messages.append(msg)
        self.updated_at = msg.timestamp

    def to_dict(self) -> dict[str, Any]:
        return {
            "conversation_id": str(self.conversation_id),
            "title": self.title,
            "messages": [m.to_dict() for m in self.messages],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Conversation:
        created = data.get("created_at")
        if isinstance(created, str):
            created = datetime.fromisoformat(created)
        updated = data.get("updated_at")
        if isinstance(updated, str):
            updated = datetime.fromisoformat(updated)
        messages = [ChatMessage.from_dict(m) for m in data.get("messages", [])]
        cid = data.get("conversation_id")
        if isinstance(cid, str):
            cid = UUID(cid)
        return cls(
            conversation_id=cid,
            title=data.get("title", "New conversation"),
            messages=messages,
            created_at=created,
            updated_at=updated,
        )


class ConversationStore:
    """Append-only store for conversations.

    Thread-safe. Indexed by conversation_id.
    Persists entire conversation state to JSONL (one line per conversation,
    re-persisted on each update since conversations are mutable).
    """

    def __init__(self, persist_path: Path | None = None) -> None:
        self._lock = threading.RLock()
        self._conversations: dict[UUID, Conversation] = {}
        self._persist_path = persist_path
        if persist_path and persist_path.exists():
            self._load_from_disk()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def create(self, title: str = "New conversation") -> Conversation:
        """Create a new conversation and return it."""
        conv = Conversation(title=title)
        with self._lock:
            self._conversations[conv.conversation_id] = conv
            self._re_persist_all()
        logger.debug("Conversation created", conversation_id=str(conv.conversation_id))
        return conv

    def add_message(
        self,
        conversation_id: UUID,
        msg: ChatMessage,
    ) -> bool:
        """Add a message to a conversation. Returns True if successful."""
        with self._lock:
            conv = self._conversations.get(conversation_id)
            if conv is None:
                return False
            conv.add_message(msg)
            # Auto-title from first user message
            if conv.message_count == 1 and msg.role == "user":
                conv.title = msg.content[:80].strip()
            self._re_persist_all()
        return True

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get(self, conversation_id: UUID) -> Conversation | None:
        """Get a conversation by ID."""
        with self._lock:
            return self._conversations.get(conversation_id)

    def list_conversations(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Conversation]:
        """List conversations, newest-first."""
        with self._lock:
            convs = sorted(
                self._conversations.values(),
                key=lambda c: c.updated_at,
                reverse=True,
            )
            return convs[offset : offset + limit]

    def count(self) -> int:
        """Total number of conversations."""
        with self._lock:
            return len(self._conversations)

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
                for conv in self._conversations.values():
                    f.write(json.dumps(conv.to_dict()) + "\n")
        except OSError as exc:
            logger.error("Failed to persist conversations", error=str(exc))

    def _load_from_disk(self) -> None:
        """Load conversations from JSONL file."""
        count = 0
        try:
            with open(self._persist_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    conv = Conversation.from_dict(data)
                    if conv.conversation_id not in self._conversations:
                        self._conversations[conv.conversation_id] = conv
                        count += 1
        except OSError as exc:
            logger.error(
                "Failed to load conversations from disk",
                path=str(self._persist_path),
                error=str(exc),
            )
        logger.info("Conversations loaded from disk", count=count)
