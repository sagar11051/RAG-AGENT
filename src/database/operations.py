"""
High-level database operations for the RAG agent.

Provides a simplified interface for common database operations,
combining multiple low-level operations into logical workflows.
"""

from typing import Any, Dict, List, Optional, Tuple
import hashlib

from src.database.supabase_client import SupabaseClient, get_supabase_client
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DatabaseOperations:
    """
    High-level database operations for RAG workflows.

    Provides methods for:
    - Document ingestion with deduplication
    - Chunk management
    - Retrieval operations
    - Session and conversation management
    """

    def __init__(self, client: Optional[SupabaseClient] = None):
        """
        Initialize database operations.

        Args:
            client: Optional SupabaseClient instance. If not provided,
                   uses the singleton instance.
        """
        self._client = client or get_supabase_client()

    @property
    def client(self) -> SupabaseClient:
        """Get the Supabase client."""
        return self._client

    # =========================================================================
    # Document Ingestion
    # =========================================================================

    @staticmethod
    def compute_content_hash(content: str) -> str:
        """
        Compute SHA256 hash of content for deduplication.

        Args:
            content: Text content to hash

        Returns:
            Hex digest of SHA256 hash
        """
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def ingest_document(
        self,
        url: str,
        title: str,
        content: str,
        total_tokens: int,
        metadata: Optional[Dict[str, Any]] = None,
        force_update: bool = False,
    ) -> Tuple[Dict[str, Any], bool]:
        """
        Ingest a document with deduplication.

        Args:
            url: Document URL
            title: Document title
            content: Full document content
            total_tokens: Token count
            metadata: Additional metadata
            force_update: If True, update even if content hash matches

        Returns:
            Tuple of (document record, is_new) where is_new indicates
            if this was a new document or update
        """
        content_hash = self.compute_content_hash(content)

        # Check if document with same content exists
        if not force_update and self.client.document_exists(content_hash):
            existing = self.client.get_document_by_url(url)
            if existing:
                logger.info(f"Document unchanged, skipping: {url}")
                return existing, False

        # Insert or update document
        doc = self.client.insert_document(
            url=url,
            title=title,
            content=content,
            content_hash=content_hash,
            total_tokens=total_tokens,
            metadata=metadata,
        )

        logger.info(f"Ingested document: {url}")
        return doc, True

    def ingest_chunks(
        self,
        doc_id: str,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]],
        replace_existing: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Ingest chunks with embeddings for a document.

        Args:
            doc_id: Parent document ID
            chunks: List of chunk data with keys:
                   - content: Chunk text
                   - tokens: Token count
                   - has_code: Whether contains code
                   - section_title: Optional section header
                   - metadata: Optional additional metadata
            embeddings: List of embedding vectors (same order as chunks)
            replace_existing: If True, delete existing chunks first

        Returns:
            List of inserted chunk records
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunks ({len(chunks)}) and embeddings ({len(embeddings)}) "
                "must have same length"
            )

        # Delete existing chunks if requested
        if replace_existing:
            self.client.delete_chunks_by_document(doc_id)

        # Prepare batch data
        chunk_data = []
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_data.append({
                "parent_doc_id": doc_id,
                "chunk_index": idx,
                "content": chunk["content"],
                "embedding": embedding,
                "tokens": chunk.get("tokens", 0),
                "has_code": chunk.get("has_code", False),
                "section_title": chunk.get("section_title"),
                "metadata": chunk.get("metadata", {}),
            })

        # Insert in batch
        result = self.client.insert_chunks_batch(chunk_data)
        logger.info(f"Ingested {len(result)} chunks for document {doc_id}")
        return result

    # =========================================================================
    # Retrieval Operations
    # =========================================================================

    def search_similar_chunks(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        include_code_only: Optional[bool] = None,
        include_parent_docs: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using vector similarity.

        Args:
            query_embedding: Query vector (1024 dimensions)
            top_k: Number of results
            include_code_only: If True, only return code chunks;
                              If False, only non-code; If None, all
            include_parent_docs: If True, include parent document info

        Returns:
            List of matching chunks with similarity scores
        """
        results = self.client.semantic_search(
            query_embedding=query_embedding,
            match_count=top_k,
            filter_has_code=include_code_only,
        )

        if include_parent_docs and results:
            # Enrich with parent document info
            for result in results:
                doc = self.client.get_document_by_id(result["parent_doc_id"])
                if doc:
                    result["parent_doc_url"] = doc["url"]
                    result["parent_doc_title"] = doc["title"]

        return results

    def get_document_for_backtracking(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the full parent document for a chunk (backtracking retrieval).

        Args:
            chunk_id: Chunk ID to backtrack from

        Returns:
            Parent document with full content, or None if not found
        """
        # First get the chunk to find parent doc ID
        result = (
            self.client.client.table("chunks")
            .select("parent_doc_id")
            .eq("id", chunk_id)
            .single()
            .execute()
        )

        if not result.data:
            return None

        return self.client.get_document_by_id(result.data["parent_doc_id"])

    def get_adjacent_chunks(
        self, chunk_id: str, before: int = 1, after: int = 1
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get chunks adjacent to a given chunk (context expansion).

        Args:
            chunk_id: Center chunk ID
            before: Number of chunks before
            after: Number of chunks after

        Returns:
            Dict with 'before', 'center', and 'after' chunk lists
        """
        # Get the chunk details
        result = (
            self.client.client.table("chunks")
            .select("*")
            .eq("id", chunk_id)
            .single()
            .execute()
        )

        if not result.data:
            return {"before": [], "center": [], "after": []}

        chunk = result.data
        doc_id = chunk["parent_doc_id"]
        chunk_index = chunk["chunk_index"]

        # Get adjacent chunks
        chunks_before = (
            self.client.client.table("chunks")
            .select("*")
            .eq("parent_doc_id", doc_id)
            .lt("chunk_index", chunk_index)
            .order("chunk_index", desc=True)
            .limit(before)
            .execute()
        )

        chunks_after = (
            self.client.client.table("chunks")
            .select("*")
            .eq("parent_doc_id", doc_id)
            .gt("chunk_index", chunk_index)
            .order("chunk_index")
            .limit(after)
            .execute()
        )

        return {
            "before": list(reversed(chunks_before.data)),
            "center": [chunk],
            "after": chunks_after.data,
        }

    # =========================================================================
    # Session Management
    # =========================================================================

    def start_conversation(
        self,
        user_id: Optional[str] = None,
        title: Optional[str] = None,
        system_message: Optional[str] = None,
        thread_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Start a new conversation session.

        Args:
            user_id: Optional user identifier
            title: Optional session title
            system_message: Optional system message to start with
            thread_id: Optional thread ID (generated if not provided)

        Returns:
            Created session record
        """
        session = self.client.create_session(user_id=user_id, title=title, thread_id=thread_id)

        if system_message and session:
            self.client.add_message(
                thread_id=session["thread_id"],
                role="system",
                content=system_message,
            )

        return session

    def add_conversation_turn(
        self,
        thread_id: str,
        user_message: str,
        assistant_response: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Add a complete conversation turn (user + assistant).

        Args:
            thread_id: Session thread ID
            user_message: User's message
            assistant_response: Assistant's response
            metadata: Optional metadata for assistant message

        Returns:
            Tuple of (user_message_record, assistant_message_record)
        """
        user_msg = self.client.add_message(
            thread_id=thread_id,
            role="user",
            content=user_message,
        )

        assistant_msg = self.client.add_message(
            thread_id=thread_id,
            role="assistant",
            content=assistant_response,
            metadata=metadata,
        )

        # Update session timestamp
        self.client.update_session(thread_id, {})

        return user_msg, assistant_msg

    def get_conversation_history(
        self,
        thread_id: str,
        max_messages: int = 20,
        format_for_llm: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Get conversation history for a session.

        Args:
            thread_id: Session thread ID
            max_messages: Maximum messages to retrieve
            format_for_llm: If True, format as simple role/content dicts

        Returns:
            List of messages
        """
        messages = self.client.get_messages(thread_id, limit=max_messages)

        if format_for_llm:
            return [
                {"role": msg["role"], "content": msg["content"]}
                for msg in messages
            ]

        return messages

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def clear_all_data(self, confirm: bool = False) -> bool:
        """
        Delete all data from all tables.

        WARNING: This is destructive and cannot be undone!

        Args:
            confirm: Must be True to proceed

        Returns:
            True if successful
        """
        if not confirm:
            logger.warning("clear_all_data called without confirmation")
            return False

        try:
            # Delete in order respecting foreign keys
            # Use gt with minimum UUID to match all valid UUIDs
            min_uuid = "00000000-0000-0000-0000-000000000000"
            self.client.client.table("messages").delete().gt("id", min_uuid).execute()
            self.client.client.table("sessions").delete().gt("thread_id", min_uuid).execute()
            self.client.client.table("chunks").delete().gt("id", min_uuid).execute()
            self.client.client.table("documents").delete().gt("id", min_uuid).execute()

            logger.warning("All data cleared from database")
            return True
        except Exception as e:
            logger.error(f"Failed to clear data: {e}")
            return False
