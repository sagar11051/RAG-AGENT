"""
Session memory management for the LangGraph RAG agent.

Provides:
- Session creation and retrieval
- Conversation history management
- Async persistence to Supabase
- In-memory caching for fast access
"""

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.agent.cache import get_caches
from src.database.operations import DatabaseOperations
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Session:
    """
    In-memory session representation.

    Caches conversation history for fast access while
    persisting to Supabase asynchronously.
    """

    thread_id: str
    user_id: Optional[str] = None
    title: Optional[str] = None
    messages: List[Dict[str, str]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_message(self, role: str, content: str) -> Dict[str, str]:
        """Add a message to the session."""
        message = {"role": role, "content": content}
        self.messages.append(message)
        self.updated_at = datetime.now()
        return message

    def get_history(self, max_messages: int = 20) -> List[Dict[str, str]]:
        """Get conversation history, limited to recent messages."""
        return self.messages[-max_messages:] if self.messages else []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for caching."""
        return {
            "thread_id": self.thread_id,
            "user_id": self.user_id,
            "title": self.title,
            "messages": self.messages,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Session":
        """Create Session from dictionary."""
        return cls(
            thread_id=data["thread_id"],
            user_id=data.get("user_id"),
            title=data.get("title"),
            messages=data.get("messages", []),
            created_at=datetime.fromisoformat(data["created_at"])
            if isinstance(data.get("created_at"), str)
            else data.get("created_at", datetime.now()),
            updated_at=datetime.fromisoformat(data["updated_at"])
            if isinstance(data.get("updated_at"), str)
            else data.get("updated_at", datetime.now()),
            metadata=data.get("metadata", {}),
        )


class MemoryManager:
    """
    Manages session memory with caching and async persistence.

    Features:
    - In-memory session cache for fast access
    - Async persistence to Supabase (fire-and-forget)
    - Session creation and retrieval
    - Conversation history management
    """

    def __init__(self, db_ops: Optional[DatabaseOperations] = None):
        """
        Initialize memory manager.

        Args:
            db_ops: Database operations instance (optional)
        """
        self._db_ops = db_ops
        self._caches = get_caches()
        self._sessions: Dict[str, Session] = {}  # In-memory sessions

        logger.info("Initialized MemoryManager")

    @property
    def db_ops(self) -> DatabaseOperations:
        """Get database operations (lazy initialized)."""
        if self._db_ops is None:
            self._db_ops = DatabaseOperations()
        return self._db_ops

    def get_or_create_session(
        self,
        thread_id: Optional[str] = None,
        user_id: Optional[str] = None,
        title: Optional[str] = None,
    ) -> Session:
        """
        Get existing session or create new one.

        Args:
            thread_id: Session thread ID (None to create new)
            user_id: User identifier
            title: Session title

        Returns:
            Session instance
        """
        # Try to get existing session
        if thread_id:
            # Check in-memory first
            if thread_id in self._sessions:
                return self._sessions[thread_id]

            # Check cache
            cached = self._caches.get_session(thread_id)
            if cached:
                session = Session.from_dict(cached)
                self._sessions[thread_id] = session
                return session

            # Try to load from database
            try:
                messages = self.db_ops.get_conversation_history(
                    thread_id, format_for_llm=True
                )
                if messages:
                    session = Session(
                        thread_id=thread_id,
                        user_id=user_id,
                        messages=messages,
                    )
                    self._sessions[thread_id] = session
                    self._caches.set_session(thread_id, session.to_dict())
                    return session
            except Exception as e:
                logger.warning(f"Failed to load session from DB: {e}")

        # Create new session
        new_thread_id = thread_id or str(uuid.uuid4())
        session = Session(
            thread_id=new_thread_id,
            user_id=user_id,
            title=title or f"Session {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        )

        # Store in memory and cache
        self._sessions[new_thread_id] = session
        self._caches.set_session(new_thread_id, session.to_dict())

        # Persist to database (fire-and-forget in background)
        try:
            self.db_ops.start_conversation(
                user_id=user_id,
                title=session.title,
                thread_id=new_thread_id,
            )
        except Exception as e:
            logger.warning(f"Failed to persist new session to DB: {e}")

        logger.info(f"Created new session: {new_thread_id}")
        return session

    def get_session(self, thread_id: str) -> Optional[Session]:
        """
        Get session by thread ID.

        Args:
            thread_id: Session thread ID

        Returns:
            Session or None if not found
        """
        # Check in-memory
        if thread_id in self._sessions:
            return self._sessions[thread_id]

        # Check cache
        cached = self._caches.get_session(thread_id)
        if cached:
            session = Session.from_dict(cached)
            self._sessions[thread_id] = session
            return session

        return None

    def add_turn(
        self,
        thread_id: str,
        user_message: str,
        assistant_response: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add a conversation turn to the session.

        Args:
            thread_id: Session thread ID
            user_message: User's message
            assistant_response: Assistant's response
            metadata: Optional metadata
        """
        session = self.get_session(thread_id)
        if not session:
            session = self.get_or_create_session(thread_id)

        # Add to in-memory session
        session.add_message("user", user_message)
        session.add_message("assistant", assistant_response)

        # Update cache
        self._caches.set_session(thread_id, session.to_dict())

        # Persist to database asynchronously
        try:
            self.db_ops.add_conversation_turn(
                thread_id=thread_id,
                user_message=user_message,
                assistant_response=assistant_response,
                metadata=metadata,
            )
        except Exception as e:
            logger.warning(f"Failed to persist turn to DB: {e}")

    async def add_turn_async(
        self,
        thread_id: str,
        user_message: str,
        assistant_response: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add conversation turn asynchronously (non-blocking).

        Args:
            thread_id: Session thread ID
            user_message: User's message
            assistant_response: Assistant's response
            metadata: Optional metadata
        """
        session = self.get_session(thread_id)
        if not session:
            session = self.get_or_create_session(thread_id)

        # Add to in-memory session
        session.add_message("user", user_message)
        session.add_message("assistant", assistant_response)

        # Update cache
        self._caches.set_session(thread_id, session.to_dict())

        # Persist in background (fire-and-forget)
        loop = asyncio.get_event_loop()
        loop.run_in_executor(
            None,
            lambda: self._persist_turn_sync(
                thread_id, user_message, assistant_response, metadata
            ),
        )

    def _persist_turn_sync(
        self,
        thread_id: str,
        user_message: str,
        assistant_response: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Synchronous helper for background persistence."""
        try:
            self.db_ops.add_conversation_turn(
                thread_id=thread_id,
                user_message=user_message,
                assistant_response=assistant_response,
                metadata=metadata,
            )
            logger.debug(f"Persisted turn for session {thread_id}")
        except Exception as e:
            logger.error(f"Background persistence failed: {e}")

    def get_history(
        self,
        thread_id: str,
        max_messages: int = 20,
    ) -> List[Dict[str, str]]:
        """
        Get conversation history for a session.

        Args:
            thread_id: Session thread ID
            max_messages: Maximum messages to return

        Returns:
            List of message dicts with role and content
        """
        session = self.get_session(thread_id)
        if session:
            return session.get_history(max_messages)
        return []

    async def load_memory_async(
        self,
        thread_id: str,
        max_messages: int = 20,
    ) -> List[Dict[str, str]]:
        """
        Load conversation history asynchronously.

        Args:
            thread_id: Session thread ID
            max_messages: Maximum messages to return

        Returns:
            List of message dicts
        """
        # Check in-memory first (fast path)
        session = self.get_session(thread_id)
        if session and session.messages:
            return session.get_history(max_messages)

        # Load from DB in background
        loop = asyncio.get_event_loop()
        messages = await loop.run_in_executor(
            None,
            lambda: self.db_ops.get_conversation_history(
                thread_id, max_messages=max_messages, format_for_llm=True
            ),
        )

        # Cache the session
        if messages:
            session = Session(thread_id=thread_id, messages=messages)
            self._sessions[thread_id] = session
            self._caches.set_session(thread_id, session.to_dict())

        return messages or []

    def clear_session(self, thread_id: str) -> None:
        """
        Clear session from memory and cache.

        Args:
            thread_id: Session thread ID
        """
        self._sessions.pop(thread_id, None)
        self._caches.invalidate_session(thread_id)
        logger.info(f"Cleared session: {thread_id}")


# Singleton instance
_memory_manager: Optional[MemoryManager] = None


def get_memory_manager() -> MemoryManager:
    """Get singleton MemoryManager instance."""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager()
    return _memory_manager


def reset_memory_manager() -> None:
    """Reset singleton (for testing)."""
    global _memory_manager
    _memory_manager = None
