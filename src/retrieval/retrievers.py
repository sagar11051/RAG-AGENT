"""
Multi-Strategy Retrievers for RAG System.

Provides multiple retrieval strategies:
- SemanticRetriever: Vector similarity search using embeddings
- BM25Retriever: Full-text search using PostgreSQL tsvector
- HybridRetriever: Combines semantic + BM25 with RRF reranking

Each retriever implements a common interface for easy swapping.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from src.config.settings import settings
from src.database.operations import DatabaseOperations
from src.embeddings.ovh_embeddings import OVHEmbeddings, get_embeddings_client
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RetrieverType(Enum):
    """Available retriever types."""
    SEMANTIC = "semantic"
    BM25 = "bm25"
    HYBRID = "hybrid"


@dataclass
class RetrievedChunk:
    """Represents a retrieved chunk with context and scoring info."""

    id: str
    content: str
    tokens: int
    has_code: bool
    section_title: Optional[str]
    parent_doc_id: str
    parent_doc_url: Optional[str] = None
    parent_doc_title: Optional[str] = None
    chunk_index: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Scoring fields
    similarity: float = 0.0  # Vector similarity score (0-1)
    bm25_rank: float = 0.0   # BM25/FTS rank score
    rrf_score: float = 0.0   # Reciprocal Rank Fusion score
    retriever_type: str = "semantic"  # Which retriever found this chunk

    def to_context_string(self, include_source: bool = True) -> str:
        """Format chunk as context string for LLM."""
        parts = []
        if self.section_title:
            parts.append(f"## {self.section_title}")
        parts.append(self.content)
        if include_source and self.parent_doc_url:
            parts.append(f"\n[Source: {self.parent_doc_url}]")
        return "\n".join(parts)


class BaseRetriever(ABC):
    """Abstract base class for retrievers."""

    def __init__(
        self,
        db_ops: Optional[DatabaseOperations] = None,
        top_k: int = None,
    ):
        self.db_ops = db_ops or DatabaseOperations()
        self.top_k = top_k or settings.retrieval.top_k

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        include_code_only: Optional[bool] = None,
        include_parent_docs: bool = True,
    ) -> List[RetrievedChunk]:
        """Retrieve relevant chunks for a query."""
        pass

    def _enrich_with_parent_docs(
        self, results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Add parent document info to results."""
        for result in results:
            doc = self.db_ops.client.get_document_by_id(result["parent_doc_id"])
            if doc:
                result["parent_doc_url"] = doc["url"]
                result["parent_doc_title"] = doc["title"]
        return results

    def get_expanded_context(
        self,
        chunk_id: str,
        before: int = 1,
        after: int = 1,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Get expanded context by retrieving adjacent chunks."""
        return self.db_ops.get_adjacent_chunks(chunk_id, before=before, after=after)

    def get_full_document(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Get the full parent document (backtracking retrieval)."""
        return self.db_ops.get_document_for_backtracking(chunk_id)


class SemanticRetriever(BaseRetriever):
    """
    Semantic retriever using vector similarity search.

    Uses OVH BGE-M3 embeddings for query encoding and pgvector
    for cosine similarity search.
    """

    def __init__(
        self,
        embeddings_client: Optional[OVHEmbeddings] = None,
        db_ops: Optional[DatabaseOperations] = None,
        top_k: int = None,
    ):
        super().__init__(db_ops, top_k)
        self.embeddings = embeddings_client or get_embeddings_client()
        logger.info(f"Initialized SemanticRetriever (top_k: {self.top_k})")

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        include_code_only: Optional[bool] = None,
        include_parent_docs: bool = True,
        min_similarity: float = 0.0,
    ) -> List[RetrievedChunk]:
        """
        Retrieve chunks using vector similarity.

        Args:
            query: Search query
            top_k: Number of results
            include_code_only: Filter for code chunks
            include_parent_docs: Include parent doc info
            min_similarity: Minimum similarity threshold

        Returns:
            List of RetrievedChunk objects sorted by similarity
        """
        top_k = top_k or self.top_k
        logger.info(f"[Semantic] Searching: {query[:50]}...")

        # Generate query embedding
        query_embedding = self.embeddings.get_text_embedding(query)

        # Search
        results = self.db_ops.client.semantic_search(
            query_embedding=query_embedding,
            match_count=top_k,
            filter_has_code=include_code_only,
        )

        if include_parent_docs:
            results = self._enrich_with_parent_docs(results)

        # Convert to RetrievedChunk objects
        chunks = []
        for result in results:
            similarity = result.get("similarity", 0)
            if similarity < min_similarity:
                continue

            chunk = RetrievedChunk(
                id=result["id"],
                content=result["content"],
                similarity=similarity,
                tokens=result.get("tokens", 0),
                has_code=result.get("has_code", False),
                section_title=result.get("section_title"),
                parent_doc_id=result["parent_doc_id"],
                parent_doc_url=result.get("parent_doc_url"),
                parent_doc_title=result.get("parent_doc_title"),
                chunk_index=result.get("chunk_index", 0),
                retriever_type="semantic",
            )
            chunks.append(chunk)

        logger.info(f"[Semantic] Retrieved {len(chunks)} chunks")
        return chunks


class BM25Retriever(BaseRetriever):
    """
    BM25-style retriever using PostgreSQL full-text search.

    Uses tsvector with ts_rank_cd for TF-IDF style ranking.
    Good for exact keyword matching and technical terms.
    """

    def __init__(
        self,
        db_ops: Optional[DatabaseOperations] = None,
        top_k: int = None,
    ):
        super().__init__(db_ops, top_k)
        logger.info(f"Initialized BM25Retriever (top_k: {self.top_k})")

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        include_code_only: Optional[bool] = None,
        include_parent_docs: bool = True,
        min_rank: float = 0.0,
    ) -> List[RetrievedChunk]:
        """
        Retrieve chunks using BM25-style full-text search.

        Args:
            query: Search query
            top_k: Number of results
            include_code_only: Filter for code chunks
            include_parent_docs: Include parent doc info
            min_rank: Minimum rank threshold

        Returns:
            List of RetrievedChunk objects sorted by rank
        """
        top_k = top_k or self.top_k
        logger.info(f"[BM25] Searching: {query[:50]}...")

        # Search
        results = self.db_ops.client.full_text_search(
            query=query,
            match_count=top_k,
            filter_has_code=include_code_only,
        )

        if include_parent_docs:
            results = self._enrich_with_parent_docs(results)

        # Convert to RetrievedChunk objects
        chunks = []
        for result in results:
            rank = result.get("rank", 0)
            if rank < min_rank:
                continue

            chunk = RetrievedChunk(
                id=result["id"],
                content=result["content"],
                bm25_rank=rank,
                tokens=result.get("tokens", 0),
                has_code=result.get("has_code", False),
                section_title=result.get("section_title"),
                parent_doc_id=result["parent_doc_id"],
                parent_doc_url=result.get("parent_doc_url"),
                parent_doc_title=result.get("parent_doc_title"),
                chunk_index=result.get("chunk_index", 0),
                retriever_type="bm25",
            )
            chunks.append(chunk)

        logger.info(f"[BM25] Retrieved {len(chunks)} chunks")
        return chunks


class HybridRetriever(BaseRetriever):
    """
    Hybrid retriever combining semantic and BM25 search with RRF reranking.

    Uses Reciprocal Rank Fusion (RRF) with k=60 to combine results:
    RRF_score = sum(1 / (k + rank_i)) for each ranking

    This approach balances semantic understanding with exact keyword matching.
    """

    # RRF constant - standard value that works well in practice
    RRF_K = 60

    def __init__(
        self,
        embeddings_client: Optional[OVHEmbeddings] = None,
        db_ops: Optional[DatabaseOperations] = None,
        top_k: int = None,
        semantic_weight: float = None,
        bm25_weight: float = None,
    ):
        super().__init__(db_ops, top_k)
        self.embeddings = embeddings_client or get_embeddings_client()
        self.semantic_weight = semantic_weight or settings.retrieval.semantic_weight
        self.bm25_weight = bm25_weight or settings.retrieval.keyword_weight
        logger.info(
            f"Initialized HybridRetriever (top_k: {self.top_k}, "
            f"semantic_weight: {self.semantic_weight}, bm25_weight: {self.bm25_weight})"
        )

    def _compute_rrf_scores(
        self,
        semantic_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """
        Compute RRF scores for all chunks.

        RRF formula: score = sum(1 / (k + rank)) where k=60

        Args:
            semantic_results: Results from semantic search (ordered by similarity)
            bm25_results: Results from BM25 search (ordered by rank)

        Returns:
            Dict mapping chunk_id to RRF score
        """
        rrf_scores: Dict[str, float] = {}

        # Score semantic results (weighted)
        for rank, result in enumerate(semantic_results):
            chunk_id = result["id"]
            score = self.semantic_weight * (1.0 / (self.RRF_K + rank + 1))
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + score

        # Score BM25 results (weighted)
        for rank, result in enumerate(bm25_results):
            chunk_id = result["id"]
            score = self.bm25_weight * (1.0 / (self.RRF_K + rank + 1))
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + score

        return rrf_scores

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        include_code_only: Optional[bool] = None,
        include_parent_docs: bool = True,
        fetch_multiplier: int = 2,
    ) -> List[RetrievedChunk]:
        """
        Retrieve chunks using hybrid search with RRF reranking.

        Fetches top_k * fetch_multiplier from each retriever, then
        combines using RRF and returns top_k results.

        Args:
            query: Search query
            top_k: Number of final results
            include_code_only: Filter for code chunks
            include_parent_docs: Include parent doc info
            fetch_multiplier: Multiplier for initial fetch count

        Returns:
            List of RetrievedChunk objects sorted by RRF score
        """
        top_k = top_k or self.top_k
        fetch_count = top_k * fetch_multiplier

        logger.info(f"[Hybrid] Searching: {query[:50]}...")

        # Generate query embedding for semantic search
        query_embedding = self.embeddings.get_text_embedding(query)

        # Get results from both retrievers
        semantic_results = self.db_ops.client.semantic_search(
            query_embedding=query_embedding,
            match_count=fetch_count,
            filter_has_code=include_code_only,
        )

        bm25_results = self.db_ops.client.full_text_search(
            query=query,
            match_count=fetch_count,
            filter_has_code=include_code_only,
        )

        logger.info(
            f"[Hybrid] Fetched {len(semantic_results)} semantic, "
            f"{len(bm25_results)} BM25 results"
        )

        # Compute RRF scores
        rrf_scores = self._compute_rrf_scores(semantic_results, bm25_results)

        # Merge results (deduplicate by chunk_id)
        all_results: Dict[str, Dict[str, Any]] = {}

        for result in semantic_results:
            chunk_id = result["id"]
            if chunk_id not in all_results:
                all_results[chunk_id] = result.copy()
                all_results[chunk_id]["retriever_sources"] = set()
            all_results[chunk_id]["similarity"] = result.get("similarity", 0)
            all_results[chunk_id]["retriever_sources"].add("semantic")

        for result in bm25_results:
            chunk_id = result["id"]
            if chunk_id not in all_results:
                all_results[chunk_id] = result.copy()
                all_results[chunk_id]["retriever_sources"] = set()
            all_results[chunk_id]["bm25_rank"] = result.get("rank", 0)
            all_results[chunk_id]["retriever_sources"].add("bm25")

        # Sort by RRF score and take top_k
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        top_ids = sorted_ids[:top_k]

        # Enrich with parent docs if requested
        if include_parent_docs:
            for chunk_id in top_ids:
                if chunk_id in all_results:
                    doc = self.db_ops.client.get_document_by_id(
                        all_results[chunk_id]["parent_doc_id"]
                    )
                    if doc:
                        all_results[chunk_id]["parent_doc_url"] = doc["url"]
                        all_results[chunk_id]["parent_doc_title"] = doc["title"]

        # Convert to RetrievedChunk objects
        chunks = []
        for chunk_id in top_ids:
            result = all_results[chunk_id]
            sources = result.get("retriever_sources", set())

            # Determine primary retriever type
            if "semantic" in sources and "bm25" in sources:
                retriever_type = "hybrid"
            elif "bm25" in sources:
                retriever_type = "bm25"
            else:
                retriever_type = "semantic"

            chunk = RetrievedChunk(
                id=result["id"],
                content=result["content"],
                similarity=result.get("similarity", 0),
                bm25_rank=result.get("bm25_rank", 0),
                rrf_score=rrf_scores[chunk_id],
                tokens=result.get("tokens", 0),
                has_code=result.get("has_code", False),
                section_title=result.get("section_title"),
                parent_doc_id=result["parent_doc_id"],
                parent_doc_url=result.get("parent_doc_url"),
                parent_doc_title=result.get("parent_doc_title"),
                chunk_index=result.get("chunk_index", 0),
                retriever_type=retriever_type,
            )
            chunks.append(chunk)

        logger.info(f"[Hybrid] Final result: {len(chunks)} chunks after RRF fusion")
        return chunks


def get_retriever(
    retriever_type: RetrieverType = RetrieverType.SEMANTIC,
    **kwargs
) -> BaseRetriever:
    """
    Factory function to get a retriever by type.

    Args:
        retriever_type: Type of retriever to create
        **kwargs: Additional arguments passed to retriever constructor

    Returns:
        Configured retriever instance
    """
    if retriever_type == RetrieverType.SEMANTIC:
        return SemanticRetriever(**kwargs)
    elif retriever_type == RetrieverType.BM25:
        return BM25Retriever(**kwargs)
    elif retriever_type == RetrieverType.HYBRID:
        return HybridRetriever(**kwargs)
    else:
        raise ValueError(f"Unknown retriever type: {retriever_type}")


# Convenience function to format multiple chunks for LLM
def format_chunks_for_llm(
    chunks: List[RetrievedChunk],
    max_tokens: int = 4000,
    include_sources: bool = True,
    include_scores: bool = False,
) -> str:
    """
    Format retrieved chunks as context for LLM.

    Args:
        chunks: List of retrieved chunks
        max_tokens: Maximum tokens for context
        include_sources: Include source URLs
        include_scores: Include retrieval scores

    Returns:
        Formatted context string
    """
    context_parts = []
    current_tokens = 0

    for i, chunk in enumerate(chunks):
        chunk_text = chunk.to_context_string(include_source=include_sources)

        if include_scores:
            scores = []
            if chunk.similarity > 0:
                scores.append(f"sim: {chunk.similarity:.3f}")
            if chunk.bm25_rank > 0:
                scores.append(f"bm25: {chunk.bm25_rank:.4f}")
            if chunk.rrf_score > 0:
                scores.append(f"rrf: {chunk.rrf_score:.4f}")
            if scores:
                chunk_text = f"[{', '.join(scores)}]\n{chunk_text}"

        chunk_tokens = len(chunk_text) // 4
        if current_tokens + chunk_tokens > max_tokens:
            break

        context_parts.append(f"[Chunk {i + 1}]\n{chunk_text}")
        current_tokens += chunk_tokens

    return "\n\n---\n\n".join(context_parts)
