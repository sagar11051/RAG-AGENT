"""Chunker module for smart text chunking and content cleaning."""

# Use LlamaIndex-based chunker for better semantic chunking
from src.chunker.llamaindex_chunker import (
    Chunk,
    LlamaIndexChunker,
    SmartChunker,  # Alias for backward compatibility
    chunk_documents,
)
from src.chunker.content_cleaner import (
    ContentCleaner,
    ContentCleanerConfig,
    CleaningStats,
    get_content_cleaner,
)

__all__ = [
    "Chunk",
    "LlamaIndexChunker",
    "SmartChunker",
    "chunk_documents",
    "ContentCleaner",
    "ContentCleanerConfig",
    "CleaningStats",
    "get_content_cleaner",
]
