"""
Retrieval module for multi-strategy RAG search.

Provides three retrieval strategies:
- SemanticRetriever: Vector similarity using embeddings
- BM25Retriever: Full-text search using PostgreSQL tsvector
- HybridRetriever: Combines both with RRF (Reciprocal Rank Fusion)

The RAGRetriever is the unified interface that supports all strategies
and provides backtracking hints for when full documents are needed.
"""

from src.retrieval.ingestion import IngestionPipeline, run_ingestion
from src.retrieval.retriever import (
    RAGRetriever,
    RetrievalResult,
    BacktrackingHint,
    get_rag_retriever,
)
from src.retrieval.retrievers import (
    RetrievedChunk,
    RetrieverType,
    SemanticRetriever,
    BM25Retriever,
    HybridRetriever,
    BaseRetriever,
    get_retriever,
    format_chunks_for_llm,
)

__all__ = [
    # Main interface
    "RAGRetriever",
    "RetrievalResult",
    "BacktrackingHint",
    "get_rag_retriever",
    # Individual retrievers
    "SemanticRetriever",
    "BM25Retriever",
    "HybridRetriever",
    "BaseRetriever",
    # Types and utilities
    "RetrievedChunk",
    "RetrieverType",
    "get_retriever",
    "format_chunks_for_llm",
    # Ingestion
    "IngestionPipeline",
    "run_ingestion",
]
