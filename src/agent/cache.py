"""
Caching layer for the LangGraph RAG agent.

Provides:
- Embedding cache: Avoid redundant embedding API calls
- Semantic response cache: Return cached responses for similar queries
- Session cache: Fast in-memory access to session data
"""

import hashlib
from typing import List, Optional, Tuple

import numpy as np
from cachetools import LRUCache, TTLCache

from src.embeddings.ovh_embeddings import OVHEmbeddings, get_embeddings_client
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _ensure_string(text) -> str:
    """Ensure text is a string, extracting from complex types if needed."""
    if isinstance(text, str):
        return text
    if not text:
        return ""
    # Handle list (e.g., messages list or multimodal content)
    if isinstance(text, list):
        parts = []
        for item in text:
            parts.append(_ensure_string(item))
        return " ".join(parts)
    # Handle dict
    if isinstance(text, dict):
        return _ensure_string(text.get("content", "") or text.get("text", ""))
    # Handle message objects
    if hasattr(text, "content"):
        return _ensure_string(text.content)
    # Fallback
    return str(text)


class AgentCaches:
    """
    Centralized caching for all agent operations.

    Provides three cache types:
    1. Embedding cache (TTL): Avoid redundant embedding API calls
    2. Response cache (LRU): Cache responses with semantic similarity matching
    3. Session cache (TTL): In-memory cache for fast session access

    Thread-safe through cachetools implementations.
    """

    def __init__(
        self,
        embedding_cache_size: int = 10000,
        embedding_ttl: int = 3600,
        response_cache_size: int = 1000,
        session_cache_size: int = 1000,
        session_ttl: int = 3600,
        similarity_threshold: float = 0.92,
    ):
        """
        Initialize agent caches.

        Args:
            embedding_cache_size: Max embeddings to cache
            embedding_ttl: Embedding cache TTL in seconds
            response_cache_size: Max responses to cache
            session_cache_size: Max sessions to cache
            session_ttl: Session cache TTL in seconds
            similarity_threshold: Min similarity for cache hit (0.0-1.0)
        """
        # Embedding cache: avoid redundant API calls
        self.embedding_cache: TTLCache = TTLCache(
            maxsize=embedding_cache_size, ttl=embedding_ttl
        )

        # Semantic response cache: return cached for similar queries
        self.response_cache: LRUCache = LRUCache(maxsize=response_cache_size)
        self.response_embeddings: dict = {}  # query_key -> embedding

        # Session cache: in-memory for fast access
        self.session_cache: TTLCache = TTLCache(
            maxsize=session_cache_size, ttl=session_ttl
        )

        self.similarity_threshold = similarity_threshold
        self._embeddings_client: Optional[OVHEmbeddings] = None

        # Stats
        self.stats = {
            "embedding_hits": 0,
            "embedding_misses": 0,
            "response_hits": 0,
            "response_misses": 0,
        }

        logger.info(
            f"Initialized AgentCaches "
            f"(embedding_size={embedding_cache_size}, "
            f"response_size={response_cache_size}, "
            f"threshold={similarity_threshold})"
        )

    @property
    def embeddings_client(self) -> OVHEmbeddings:
        """Get embeddings client (lazy initialized)."""
        if self._embeddings_client is None:
            self._embeddings_client = get_embeddings_client()
        return self._embeddings_client

    @staticmethod
    def _hash_text(text) -> str:
        """Generate MD5 hash for cache key."""
        # Ensure text is a string
        if not isinstance(text, str):
            text = _ensure_string(text)
        return hashlib.md5(text.encode()).hexdigest()

    def get_cached_embedding(self, text: str) -> Optional[List[float]]:
        """
        Get embedding from cache if exists.

        Args:
            text: Text to get embedding for

        Returns:
            Cached embedding or None
        """
        key = self._hash_text(text)
        embedding = self.embedding_cache.get(key)
        if embedding is not None:
            self.stats["embedding_hits"] += 1
            return embedding
        self.stats["embedding_misses"] += 1
        return None

    def set_cached_embedding(self, text: str, embedding: List[float]) -> None:
        """
        Store embedding in cache.

        Args:
            text: Original text
            embedding: Embedding vector to cache
        """
        key = self._hash_text(text)
        self.embedding_cache[key] = embedding

    async def get_or_compute_embedding(self, text) -> List[float]:
        """
        Get embedding from cache or compute it (async).

        Args:
            text: Text to embed (will be converted to string if needed)

        Returns:
            Embedding vector (cached or freshly computed)
        """
        # Ensure text is a string
        if not isinstance(text, str):
            text = _ensure_string(text)
            logger.debug(f"Converted non-string text to: {text[:50]}...")

        if not text:
            logger.warning("Empty text passed to get_or_compute_embedding")
            return []

        # Check cache first
        cached = self.get_cached_embedding(text)
        if cached is not None:
            return cached

        # Compute and cache
        embedding = await self.embeddings_client.aget_text_embedding(text)
        self.set_cached_embedding(text, embedding)
        return embedding

    def get_or_compute_embedding_sync(self, text) -> List[float]:
        """
        Get embedding from cache or compute it (sync).

        Args:
            text: Text to embed (will be converted to string if needed)

        Returns:
            Embedding vector (cached or freshly computed)
        """
        # Ensure text is a string
        if not isinstance(text, str):
            text = _ensure_string(text)
            logger.debug(f"Converted non-string text to: {text[:50]}...")

        if not text:
            logger.warning("Empty text passed to get_or_compute_embedding_sync")
            return []

        # Check cache first
        cached = self.get_cached_embedding(text)
        if cached is not None:
            return cached

        # Compute and cache
        embedding = self.embeddings_client.get_text_embedding(text)
        self.set_cached_embedding(text, embedding)
        return embedding

    @staticmethod
    def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        arr1 = np.array(vec1)
        arr2 = np.array(vec2)
        dot_product = np.dot(arr1, arr2)
        norm1 = np.linalg.norm(arr1)
        norm2 = np.linalg.norm(arr2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(dot_product / (norm1 * norm2))

    def check_semantic_cache(
        self,
        query_embedding: List[float],
        threshold: Optional[float] = None,
    ) -> Optional[Tuple[str, float]]:
        """
        Check if a similar query exists in response cache.

        Args:
            query_embedding: Embedding of current query
            threshold: Override default similarity threshold

        Returns:
            Tuple of (cached_response, similarity_score) or None
        """
        threshold = threshold or self.similarity_threshold

        best_match = None
        best_similarity = 0.0

        for cached_key, cached_embedding in self.response_embeddings.items():
            similarity = self._cosine_similarity(query_embedding, cached_embedding)
            if similarity >= threshold and similarity > best_similarity:
                best_similarity = similarity
                cached_response = self.response_cache.get(cached_key)
                if cached_response is not None:
                    best_match = (cached_response, similarity)

        if best_match:
            self.stats["response_hits"] += 1
            logger.debug(f"Semantic cache hit (similarity={best_similarity:.3f})")
            return best_match

        self.stats["response_misses"] += 1
        return None

    def update_response_cache(
        self,
        query: str,
        query_embedding: List[float],
        response: str,
    ) -> None:
        """
        Cache a response for future similar queries.

        Args:
            query: Original query text
            query_embedding: Query embedding vector
            response: Response to cache
        """
        key = self._hash_text(query)
        self.response_cache[key] = response
        self.response_embeddings[key] = query_embedding
        logger.debug(f"Cached response for query: {query[:50]}...")

    def get_session(self, thread_id: str) -> Optional[dict]:
        """
        Get session data from cache.

        Args:
            thread_id: Session thread ID

        Returns:
            Cached session data or None
        """
        return self.session_cache.get(thread_id)

    def set_session(self, thread_id: str, session_data: dict) -> None:
        """
        Cache session data.

        Args:
            thread_id: Session thread ID
            session_data: Session data to cache
        """
        self.session_cache[thread_id] = session_data

    def invalidate_session(self, thread_id: str) -> None:
        """
        Remove session from cache.

        Args:
            thread_id: Session thread ID
        """
        self.session_cache.pop(thread_id, None)

    def get_stats(self) -> dict:
        """Get cache statistics."""
        return {
            **self.stats,
            "embedding_cache_size": len(self.embedding_cache),
            "response_cache_size": len(self.response_cache),
            "session_cache_size": len(self.session_cache),
        }

    def clear_all(self) -> None:
        """Clear all caches."""
        self.embedding_cache.clear()
        self.response_cache.clear()
        self.response_embeddings.clear()
        self.session_cache.clear()
        logger.info("All caches cleared")


# Singleton instance
_caches: Optional[AgentCaches] = None


def get_caches() -> AgentCaches:
    """Get singleton AgentCaches instance."""
    global _caches
    if _caches is None:
        _caches = AgentCaches()
    return _caches


def reset_caches() -> None:
    """Reset singleton caches (for testing)."""
    global _caches
    if _caches is not None:
        _caches.clear_all()
    _caches = None
