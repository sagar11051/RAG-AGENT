"""
Supabase client wrapper with connection management and helper methods.

Provides a singleton client for interacting with Supabase database,
including vector operations via pgvector.
"""

from functools import lru_cache
from typing import Any, Dict, List, Optional

from supabase import Client, create_client

from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SupabaseClient:
    """
    Supabase client wrapper with helper methods for RAG operations.

    Provides methods for:
    - Document and chunk CRUD operations
    - Vector similarity search
    - Session management
    - Message history
    """

    def __init__(self):
        """Initialize Supabase client."""
        self._client: Optional[Client] = None
        self._initialized = False

    @property
    def client(self) -> Client:
        """Get or create Supabase client."""
        if self._client is None:
            self._initialize()
        return self._client

    def _initialize(self) -> None:
        """Initialize the Supabase client connection."""
        if self._initialized:
            return

        url = settings.supabase.url
        key = settings.supabase.key

        if not url or not key:
            raise ValueError(
                "Supabase URL and key must be set in environment variables. "
                "Please set SUPABASE_URL and SUPABASE_KEY in your .env file."
            )

        self._client = create_client(url, key)
        self._initialized = True
        logger.info("Supabase client initialized successfully")

    def is_configured(self) -> bool:
        """Check if Supabase credentials are configured."""
        return bool(settings.supabase.url and settings.supabase.key)

    async def test_connection(self) -> bool:
        """
        Test the database connection.

        Returns:
            True if connection is successful, False otherwise
        """
        try:
            # Try a simple query to test connection
            result = self.client.table("documents").select("id").limit(1).execute()
            logger.info("Database connection test successful")
            return True
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False

    # =========================================================================
    # Document Operations (Level 1)
    # =========================================================================

    def insert_document(
        self,
        url: str,
        title: str,
        content: str,
        content_hash: str,
        total_tokens: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Insert a new document.

        Args:
            url: Document URL (must be unique)
            title: Document title
            content: Full document content
            content_hash: SHA256 hash of content
            total_tokens: Token count
            metadata: Additional metadata

        Returns:
            Inserted document record
        """
        data = {
            "url": url,
            "title": title,
            "content": content,
            "content_hash": content_hash,
            "total_tokens": total_tokens,
            "metadata": metadata or {},
        }

        result = self.client.table("documents").upsert(
            data, on_conflict="url"
        ).execute()

        logger.debug(f"Inserted/updated document: {url}")
        return result.data[0] if result.data else None

    def get_document_by_url(self, url: str) -> Optional[Dict[str, Any]]:
        """Get a document by its URL."""
        result = (
            self.client.table("documents")
            .select("*")
            .eq("url", url)
            .single()
            .execute()
        )
        return result.data

    def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get a document by its ID."""
        result = (
            self.client.table("documents")
            .select("*")
            .eq("id", doc_id)
            .single()
            .execute()
        )
        return result.data

    def document_exists(self, content_hash: str) -> bool:
        """Check if a document with the given content hash exists."""
        result = (
            self.client.table("documents")
            .select("id")
            .eq("content_hash", content_hash)
            .execute()
        )
        return len(result.data) > 0

    def get_all_documents(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """Get all documents with pagination."""
        result = (
            self.client.table("documents")
            .select("*")
            .limit(limit)
            .execute()
        )
        return result.data

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document and its chunks."""
        try:
            self.client.table("documents").delete().eq("id", doc_id).execute()
            logger.info(f"Deleted document: {doc_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete document {doc_id}: {e}")
            return False

    # =========================================================================
    # Chunk Operations (Level 2)
    # =========================================================================

    def insert_chunk(
        self,
        parent_doc_id: str,
        chunk_index: int,
        content: str,
        embedding: List[float],
        tokens: int,
        has_code: bool = False,
        section_title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Insert a new chunk with embedding.

        Args:
            parent_doc_id: Reference to parent document
            chunk_index: Position in document (0-indexed)
            content: Chunk text content
            embedding: Vector embedding (1024 dimensions for BGE-M3)
            tokens: Token count
            has_code: Whether chunk contains code
            section_title: Section header if available
            metadata: Additional metadata

        Returns:
            Inserted chunk record
        """
        data = {
            "parent_doc_id": parent_doc_id,
            "chunk_index": chunk_index,
            "content": content,
            "embedding": embedding,
            "tokens": tokens,
            "has_code": has_code,
            "section_title": section_title,
            "metadata": metadata or {},
        }

        result = self.client.table("chunks").insert(data).execute()
        logger.debug(f"Inserted chunk {chunk_index} for document {parent_doc_id}")
        return result.data[0] if result.data else None

    def insert_chunks_batch(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Insert multiple chunks in a batch.

        Args:
            chunks: List of chunk data dictionaries

        Returns:
            List of inserted chunk records
        """
        if not chunks:
            return []

        result = self.client.table("chunks").insert(chunks).execute()
        logger.info(f"Inserted {len(chunks)} chunks in batch")
        return result.data

    def get_chunks_by_document(self, doc_id: str) -> List[Dict[str, Any]]:
        """Get all chunks for a document, ordered by index."""
        result = (
            self.client.table("chunks")
            .select("*")
            .eq("parent_doc_id", doc_id)
            .order("chunk_index")
            .execute()
        )
        return result.data

    def delete_chunks_by_document(self, doc_id: str) -> bool:
        """Delete all chunks for a document."""
        try:
            self.client.table("chunks").delete().eq("parent_doc_id", doc_id).execute()
            logger.info(f"Deleted chunks for document: {doc_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete chunks for document {doc_id}: {e}")
            return False

    # =========================================================================
    # Vector Search Operations
    # =========================================================================

    def semantic_search(
        self,
        query_embedding: List[float],
        match_count: int = 5,
        filter_has_code: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic search using vector similarity.

        Args:
            query_embedding: Query vector (1024 dimensions)
            match_count: Number of results to return
            filter_has_code: Optional filter for code chunks

        Returns:
            List of matching chunks with similarity scores
        """
        # Use the RPC function we defined in schema.sql
        result = self.client.rpc(
            "match_chunks",
            {
                "query_embedding": query_embedding,
                "match_count": match_count,
                "filter_has_code": filter_has_code,
            },
        ).execute()

        return result.data

    def full_text_search(
        self,
        query: str,
        match_count: int = 5,
        filter_has_code: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform BM25-style full-text search using PostgreSQL tsvector.

        Args:
            query: Search query string
            match_count: Number of results to return
            filter_has_code: Optional filter for code chunks

        Returns:
            List of matching chunks with rank scores
        """
        result = self.client.rpc(
            "match_chunks_fts",
            {
                "search_query": query,
                "match_count": match_count,
                "filter_has_code": filter_has_code,
            },
        ).execute()

        return result.data

    def hybrid_search(
        self,
        query_embedding: List[float],
        query: str,
        match_count: int = 10,
        filter_has_code: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining vector similarity and full-text search.

        Args:
            query_embedding: Query vector (1024 dimensions)
            query: Search query string for FTS
            match_count: Number of results to return
            filter_has_code: Optional filter for code chunks

        Returns:
            List of matching chunks with both similarity and rank scores
        """
        result = self.client.rpc(
            "match_chunks_hybrid",
            {
                "query_embedding": query_embedding,
                "search_query": query,
                "match_count": match_count,
                "filter_has_code": filter_has_code,
            },
        ).execute()

        return result.data

    # =========================================================================
    # Session Operations
    # =========================================================================

    def create_session(
        self,
        user_id: Optional[str] = None,
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        thread_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create a new chat session.

        Args:
            user_id: Optional user identifier
            title: Optional session title
            metadata: Additional metadata
            thread_id: Optional thread ID (generated if not provided)

        Returns:
            Created session record
        """
        data = {
            "user_id": user_id,
            "title": title,
            "metadata": metadata or {},
        }
        if thread_id:
            data["thread_id"] = thread_id

        result = self.client.table("sessions").insert(data).execute()
        session = result.data[0] if result.data else None
        if session:
            logger.info(f"Created session: {session['thread_id']}")
        return session

    def get_session(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """Get a session by thread ID."""
        result = (
            self.client.table("sessions")
            .select("*")
            .eq("thread_id", thread_id)
            .single()
            .execute()
        )
        return result.data

    def update_session(
        self, thread_id: str, updates: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Update a session."""
        result = (
            self.client.table("sessions")
            .update(updates)
            .eq("thread_id", thread_id)
            .execute()
        )
        return result.data[0] if result.data else None

    def delete_session(self, thread_id: str) -> bool:
        """Delete a session and all its messages."""
        try:
            self.client.table("sessions").delete().eq("thread_id", thread_id).execute()
            logger.info(f"Deleted session: {thread_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete session {thread_id}: {e}")
            return False

    def get_user_sessions(
        self, user_id: str, limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Get all sessions for a user."""
        result = (
            self.client.table("sessions")
            .select("*")
            .eq("user_id", user_id)
            .order("updated_at", desc=True)
            .limit(limit)
            .execute()
        )
        return result.data

    # =========================================================================
    # Message Operations
    # =========================================================================

    def add_message(
        self,
        thread_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Add a message to a session.

        Args:
            thread_id: Session thread ID
            role: Message role (user, assistant, system)
            content: Message content
            metadata: Additional metadata

        Returns:
            Created message record
        """
        data = {
            "thread_id": thread_id,
            "role": role,
            "content": content,
            "metadata": metadata or {},
        }

        result = self.client.table("messages").insert(data).execute()
        return result.data[0] if result.data else None

    def get_messages(
        self,
        thread_id: str,
        limit: int = 20,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Get messages for a session.

        Args:
            thread_id: Session thread ID
            limit: Maximum messages to return
            offset: Offset for pagination

        Returns:
            List of messages ordered by creation time
        """
        result = (
            self.client.table("messages")
            .select("*")
            .eq("thread_id", thread_id)
            .order("created_at")
            .range(offset, offset + limit - 1)
            .execute()
        )
        return result.data

    def get_message_count(self, thread_id: str) -> int:
        """Get total message count for a session."""
        result = (
            self.client.table("messages")
            .select("id", count="exact")
            .eq("thread_id", thread_id)
            .execute()
        )
        return result.count or 0

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def get_stats(self) -> Dict[str, int]:
        """Get database statistics."""
        docs = self.client.table("documents").select("id", count="exact").execute()
        chunks = self.client.table("chunks").select("id", count="exact").execute()
        sessions = self.client.table("sessions").select("thread_id", count="exact").execute()

        return {
            "documents_count": docs.count or 0,
            "chunks_count": chunks.count or 0,
            "sessions_count": sessions.count or 0,
        }


@lru_cache()
def get_supabase_client() -> SupabaseClient:
    """Get cached Supabase client instance."""
    return SupabaseClient()
