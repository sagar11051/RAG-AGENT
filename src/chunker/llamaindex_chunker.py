"""
LlamaIndex-based semantic chunker for markdown documentation.

Uses LlamaIndex's SentenceSplitter with markdown-aware chunking
and code block preservation.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from uuid import uuid4

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document as LlamaDocument

from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Chunk:
    """Represents a single chunk of content."""

    id: str
    content: str
    token_count: int
    chunk_index: int
    total_chunks: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def document_url(self) -> Optional[str]:
        """Get source document URL from metadata."""
        return self.metadata.get("url")

    @property
    def document_title(self) -> Optional[str]:
        """Get source document title from metadata."""
        return self.metadata.get("title")


class LlamaIndexChunker:
    """
    Semantic chunker using LlamaIndex's SentenceSplitter.

    Features:
    - Sentence-boundary aware splitting
    - Code block preservation
    - Configurable chunk size and overlap
    - Metadata preservation
    """

    # Pattern to match fenced code blocks
    CODE_BLOCK_PATTERN = re.compile(r'(```[\s\S]*?```|~~~[\s\S]*?~~~)', re.MULTILINE)

    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
    ):
        """
        Initialize the LlamaIndex chunker.

        Args:
            chunk_size: Target chunk size in tokens (default from settings)
            chunk_overlap: Overlap between chunks in tokens (default from settings)
        """
        self.chunk_size = chunk_size or settings.chunker.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunker.chunk_overlap

        # Initialize the LlamaIndex sentence splitter
        self.splitter = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            paragraph_separator="\n\n",
            secondary_chunking_regex="[^,.;。？！]+[,.;。？！]?",
        )

        logger.info(
            f"Initialized LlamaIndexChunker (size: {self.chunk_size}, overlap: {self.chunk_overlap})"
        )

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (roughly 4 chars per token for English)."""
        return len(text) // 4

    def _extract_code_blocks(self, text: str) -> tuple[str, list[tuple[str, str]]]:
        """
        Extract code blocks and replace with placeholders.

        Returns:
            Tuple of (text with placeholders, list of (placeholder, original_code) tuples)
        """
        code_blocks = []
        counter = [0]  # Use list to allow modification in nested function

        def replace_code_block(match):
            placeholder = f"__CODEBLOCK_{counter[0]}__"
            code_blocks.append((placeholder, match.group(0)))
            counter[0] += 1
            return placeholder

        text_with_placeholders = self.CODE_BLOCK_PATTERN.sub(replace_code_block, text)
        return text_with_placeholders, code_blocks

    def _restore_code_blocks(self, text: str, code_blocks: list[tuple[str, str]]) -> str:
        """Restore code blocks from placeholders."""
        for placeholder, original_code in code_blocks:
            text = text.replace(placeholder, original_code)
        return text

    def _split_preserving_code_blocks(self, text: str) -> List[str]:
        """
        Split text while preserving code blocks intact.

        This method:
        1. Extracts code blocks and replaces with placeholders
        2. Splits the text using LlamaIndex
        3. Restores code blocks in each chunk
        4. Ensures code blocks are not split across chunks
        """
        # Extract code blocks
        text_with_placeholders, code_blocks = self._extract_code_blocks(text)

        # Create a LlamaIndex document
        doc = LlamaDocument(text=text_with_placeholders)

        # Split using LlamaIndex
        nodes = self.splitter.get_nodes_from_documents([doc])

        # Process each chunk
        chunks = []
        for node in nodes:
            chunk_text = node.get_content()

            # Restore code blocks in this chunk
            chunk_text = self._restore_code_blocks(chunk_text, code_blocks)

            # Skip empty chunks
            if chunk_text.strip():
                chunks.append(chunk_text.strip())

        # Post-process: ensure code blocks are complete in each chunk
        chunks = self._ensure_complete_code_blocks(chunks)

        return chunks

    def _ensure_complete_code_blocks(self, chunks: List[str]) -> List[str]:
        """
        Ensure each chunk has complete code blocks.

        If a chunk has unbalanced code fences, merge with adjacent chunks.
        """
        result = []
        i = 0

        while i < len(chunks):
            chunk = chunks[i]

            # Count code fences
            fence_count = chunk.count('```') + chunk.count('~~~')

            # If odd number of fences, code block is incomplete
            if fence_count % 2 != 0 and i + 1 < len(chunks):
                # Merge with next chunk
                merged = chunk + "\n\n" + chunks[i + 1]
                chunks[i + 1] = merged
                i += 1
                continue

            result.append(chunk)
            i += 1

        return result

    def chunk_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        """
        Chunk text into semantically meaningful pieces.

        Args:
            text: Text to chunk (markdown format)
            metadata: Metadata to attach to each chunk

        Returns:
            List of Chunk objects
        """
        if not text or not text.strip():
            return []

        metadata = metadata or {}
        text = text.strip()

        # Split preserving code blocks
        chunk_texts = self._split_preserving_code_blocks(text)

        # Convert to Chunk objects
        total_chunks = len(chunk_texts)
        chunks = []

        for i, content in enumerate(chunk_texts):
            chunk = Chunk(
                id=str(uuid4()),
                content=content,
                token_count=self._estimate_tokens(content),
                chunk_index=i,
                total_chunks=total_chunks,
                metadata=metadata.copy(),
            )
            chunks.append(chunk)

        logger.debug(f"Created {len(chunks)} chunks from text ({len(text)} chars)")
        return chunks

    def chunk_document(
        self,
        url: str,
        title: str,
        markdown: str,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        """
        Chunk a document with full metadata.

        Args:
            url: Source URL
            title: Document title
            markdown: Document content in markdown
            extra_metadata: Additional metadata to include

        Returns:
            List of Chunk objects with full metadata
        """
        metadata = {
            "url": url,
            "title": title,
            **(extra_metadata or {}),
        }

        return self.chunk_text(markdown, metadata)


# Alias for backward compatibility
SmartChunker = LlamaIndexChunker


def chunk_documents(documents: List[Dict[str, Any]]) -> List[Chunk]:
    """
    Chunk multiple documents.

    Args:
        documents: List of document dicts with url, title, markdown

    Returns:
        List of all chunks from all documents
    """
    chunker = LlamaIndexChunker()
    all_chunks = []

    for doc in documents:
        chunks = chunker.chunk_document(
            url=doc.get("url", ""),
            title=doc.get("title", ""),
            markdown=doc.get("markdown", ""),
            extra_metadata=doc.get("metadata"),
        )
        all_chunks.extend(chunks)

    logger.info(f"Chunked {len(documents)} documents into {len(all_chunks)} chunks")
    return all_chunks
