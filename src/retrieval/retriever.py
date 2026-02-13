"""
RAG Retriever - Unified retrieval interface for the agent.

This module provides:
1. Multiple retrieval strategies (Semantic, BM25, Hybrid)
2. Context expansion (adjacent chunks)
3. Parent document backtracking
4. Agent-friendly tool interface with backtracking hints

The agent should use backtracking when:
- Retrieved chunks mention "see full example" or reference other sections
- The chunk appears to be part of a larger code block
- User asks for complete implementation or full context
- Retrieved chunks have low similarity but high relevance indicators
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from src.config.settings import settings
from src.database.operations import DatabaseOperations
from src.embeddings.ovh_embeddings import OVHEmbeddings, get_embeddings_client
from src.utils.logger import get_logger

# Import from new retrievers module
from src.retrieval.retrievers import (
    BaseRetriever,
    BM25Retriever,
    HybridRetriever,
    RetrievedChunk,
    RetrieverType,
    SemanticRetriever,
    format_chunks_for_llm,
    get_retriever,
)

logger = get_logger(__name__)

# Re-export for backwards compatibility
__all__ = [
    "RAGRetriever",
    "RetrievedChunk",
    "RetrieverType",
    "SemanticRetriever",
    "BM25Retriever",
    "HybridRetriever",
    "get_retriever",
    "format_chunks_for_llm",
    "BacktrackingHint",
]


class BacktrackingHint(Enum):
    """Hints for when the agent should consider backtracking to full document."""

    NONE = "none"
    SUGGESTED = "suggested"  # Might benefit from full context
    RECOMMENDED = "recommended"  # Likely needs full context
    REQUIRED = "required"  # Definitely needs full context


@dataclass
class RetrievalResult:
    """
    Complete retrieval result with chunks and metadata for agent decision-making.

    Includes backtracking hints to help the agent decide whether to
    fetch full parent documents for better context.
    """

    chunks: List[RetrievedChunk]
    query: str
    retriever_type: str
    backtracking_hints: Dict[str, BacktrackingHint]  # chunk_id -> hint

    @property
    def needs_backtracking(self) -> bool:
        """Check if any chunks suggest backtracking."""
        return any(
            hint in (BacktrackingHint.RECOMMENDED, BacktrackingHint.REQUIRED)
            for hint in self.backtracking_hints.values()
        )

    @property
    def chunks_needing_backtracking(self) -> List[str]:
        """Get chunk IDs that suggest backtracking."""
        return [
            chunk_id
            for chunk_id, hint in self.backtracking_hints.items()
            if hint in (BacktrackingHint.RECOMMENDED, BacktrackingHint.REQUIRED)
        ]


class RAGRetriever:
    """
    Unified RAG Retriever supporting multiple retrieval strategies.

    This is the main interface for the LangGraph agent to use.
    It provides:
    - Multiple retrieval strategies (semantic, BM25, hybrid)
    - Automatic backtracking hint detection
    - Context expansion capabilities
    - LLM-ready formatting

    Backtracking Strategy for Agent:
    --------------------------------
    The agent should consider fetching the full parent document when:

    1. REQUIRED: Chunk contains incomplete code (unclosed brackets, "..." truncation)
    2. RECOMMENDED: Chunk references "see below", "full example", or is part of tutorial
    3. SUGGESTED: Chunk mentions multiple sections or has multiple code blocks
    4. NONE: Chunk is self-contained

    The agent can use get_full_document(chunk_id) to fetch the complete page
    when backtracking is needed.
    """

    # Patterns that suggest backtracking would help
    BACKTRACK_PATTERNS = {
        "required": [
            "...",  # Truncated content
            "# continued",
            "see full",
            "complete example",
        ],
        "recommended": [
            "see below",
            "as shown above",
            "following example",
            "previous section",
            "next section",
            "part 1",
            "part 2",
            "step 1",
            "step 2",
        ],
        "suggested": [
            "tutorial",
            "guide",
            "walkthrough",
            "```python",  # Code blocks might need full context
        ],
    }

    def __init__(
        self,
        retriever_type: RetrieverType = RetrieverType.HYBRID,
        embeddings_client: Optional[OVHEmbeddings] = None,
        db_ops: Optional[DatabaseOperations] = None,
        top_k: int = None,
    ):
        """
        Initialize RAG retriever.

        Args:
            retriever_type: Which retriever strategy to use
            embeddings_client: OVH embeddings client (for semantic/hybrid)
            db_ops: Database operations instance
            top_k: Default number of chunks to retrieve
        """
        self.embeddings = embeddings_client or get_embeddings_client()
        self.db_ops = db_ops or DatabaseOperations()
        self.top_k = top_k or settings.retrieval.top_k
        self.retriever_type = retriever_type

        # Initialize the underlying retriever
        if retriever_type == RetrieverType.SEMANTIC:
            self._retriever = SemanticRetriever(
                embeddings_client=self.embeddings,
                db_ops=self.db_ops,
                top_k=self.top_k,
            )
        elif retriever_type == RetrieverType.BM25:
            self._retriever = BM25Retriever(
                db_ops=self.db_ops,
                top_k=self.top_k,
            )
        else:  # HYBRID
            self._retriever = HybridRetriever(
                embeddings_client=self.embeddings,
                db_ops=self.db_ops,
                top_k=self.top_k,
            )

        logger.info(
            f"Initialized RAGRetriever (type: {retriever_type.value}, top_k: {self.top_k})"
        )

    def _analyze_backtracking_need(self, chunk: RetrievedChunk) -> BacktrackingHint:
        """
        Analyze a chunk to determine if backtracking to full document is needed.

        Args:
            chunk: Retrieved chunk to analyze

        Returns:
            BacktrackingHint indicating need level
        """
        content_lower = chunk.content.lower()

        # Check for required patterns
        for pattern in self.BACKTRACK_PATTERNS["required"]:
            if pattern.lower() in content_lower:
                return BacktrackingHint.REQUIRED

        # Check for recommended patterns
        for pattern in self.BACKTRACK_PATTERNS["recommended"]:
            if pattern.lower() in content_lower:
                return BacktrackingHint.RECOMMENDED

        # Check for suggested patterns
        for pattern in self.BACKTRACK_PATTERNS["suggested"]:
            if pattern.lower() in content_lower:
                return BacktrackingHint.SUGGESTED

        # Check if chunk appears to be incomplete code
        if chunk.has_code:
            # Count brackets to detect incomplete code
            open_brackets = chunk.content.count("{") + chunk.content.count("[")
            close_brackets = chunk.content.count("}") + chunk.content.count("]")
            if abs(open_brackets - close_brackets) > 2:
                return BacktrackingHint.RECOMMENDED

        return BacktrackingHint.NONE

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        include_code_only: Optional[bool] = None,
        include_parent_docs: bool = True,
        analyze_backtracking: bool = True,
    ) -> RetrievalResult:
        """
        Retrieve relevant chunks with backtracking analysis.

        Args:
            query: Search query
            top_k: Number of chunks to retrieve
            include_code_only: Filter for code chunks only
            include_parent_docs: Include parent document info
            analyze_backtracking: Analyze chunks for backtracking hints

        Returns:
            RetrievalResult with chunks and backtracking hints
        """
        top_k = top_k or self.top_k

        # Use underlying retriever
        chunks = self._retriever.retrieve(
            query=query,
            top_k=top_k,
            include_code_only=include_code_only,
            include_parent_docs=include_parent_docs,
        )

        # Analyze backtracking needs
        backtracking_hints = {}
        if analyze_backtracking:
            for chunk in chunks:
                backtracking_hints[chunk.id] = self._analyze_backtracking_need(chunk)

        return RetrievalResult(
            chunks=chunks,
            query=query,
            retriever_type=self.retriever_type.value,
            backtracking_hints=backtracking_hints,
        )

    def retrieve_sync(
        self,
        query: str,
        top_k: Optional[int] = None,
        include_code_only: Optional[bool] = None,
        include_parent_docs: bool = True,
        min_similarity: float = 0.0,
    ) -> List[RetrievedChunk]:
        """
        Simple synchronous retrieve (backwards compatible).

        Returns just the chunks without backtracking analysis.
        """
        return self._retriever.retrieve(
            query=query,
            top_k=top_k or self.top_k,
            include_code_only=include_code_only,
            include_parent_docs=include_parent_docs,
        )

    def get_expanded_context(
        self,
        chunk_id: str,
        before: int = 1,
        after: int = 1,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get expanded context by retrieving adjacent chunks.

        Args:
            chunk_id: Center chunk ID
            before: Number of chunks before
            after: Number of chunks after

        Returns:
            Dict with 'before', 'center', and 'after' chunks
        """
        return self._retriever.get_expanded_context(chunk_id, before=before, after=after)

    def get_full_document(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the full parent document (backtracking retrieval).

        Use this when:
        - RetrievalResult.needs_backtracking is True
        - User asks for complete/full implementation
        - Retrieved chunk seems incomplete

        Args:
            chunk_id: Chunk ID to backtrack from

        Returns:
            Full parent document with complete content, or None
        """
        return self._retriever.get_full_document(chunk_id)

    def format_context_for_llm(
        self,
        chunks: List[RetrievedChunk],
        max_tokens: int = 4000,
        include_sources: bool = True,
    ) -> str:
        """
        Format retrieved chunks as context for LLM.

        Args:
            chunks: List of retrieved chunks
            max_tokens: Maximum tokens for context
            include_sources: Include source URLs

        Returns:
            Formatted context string
        """
        return format_chunks_for_llm(
            chunks=chunks,
            max_tokens=max_tokens,
            include_sources=include_sources,
        )

    def switch_retriever(self, retriever_type: RetrieverType) -> None:
        """
        Switch to a different retriever strategy.

        Args:
            retriever_type: New retriever type to use
        """
        if retriever_type == self.retriever_type:
            return

        if retriever_type == RetrieverType.SEMANTIC:
            self._retriever = SemanticRetriever(
                embeddings_client=self.embeddings,
                db_ops=self.db_ops,
                top_k=self.top_k,
            )
        elif retriever_type == RetrieverType.BM25:
            self._retriever = BM25Retriever(
                db_ops=self.db_ops,
                top_k=self.top_k,
            )
        else:
            self._retriever = HybridRetriever(
                embeddings_client=self.embeddings,
                db_ops=self.db_ops,
                top_k=self.top_k,
            )

        self.retriever_type = retriever_type
        logger.info(f"Switched to {retriever_type.value} retriever")


# Singleton retriever instance
_retriever: Optional[RAGRetriever] = None


def get_rag_retriever(
    retriever_type: RetrieverType = RetrieverType.HYBRID,
) -> RAGRetriever:
    """
    Get cached RAG retriever instance.

    Args:
        retriever_type: Which retriever strategy to use

    Returns:
        Configured RAGRetriever instance
    """
    global _retriever
    if _retriever is None or _retriever.retriever_type != retriever_type:
        _retriever = RAGRetriever(retriever_type=retriever_type)
    return _retriever
