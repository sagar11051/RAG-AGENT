"""
Smart chunker for markdown documentation.

Features:
- Respects semantic boundaries (headers, paragraphs)
- Keeps code blocks intact (never splits mid-code)
- Token-based sizing with configurable overlap
- Preserves context through overlap
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

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


class SmartChunker:
    """
    Smart chunker that respects semantic boundaries in markdown.

    Splits content while preserving:
    - Code blocks (never split mid-code)
    - Headers as natural break points
    - Sentence boundaries
    - Configurable overlap for context preservation
    """

    # Regex patterns for markdown elements
    CODE_BLOCK_PATTERN = re.compile(r"```[\s\S]*?```", re.MULTILINE)
    HEADER_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    SENTENCE_END_PATTERN = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")

    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
        min_chunk_size: int = None,
        max_chunk_size: int = None,
        respect_code_blocks: bool = None,
        respect_sentences: bool = None,
    ):
        """
        Initialize smart chunker.

        Args:
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap between chunks in tokens
            min_chunk_size: Minimum chunk size in tokens
            max_chunk_size: Maximum chunk size in tokens
            respect_code_blocks: Keep code blocks intact
            respect_sentences: Split at sentence boundaries
        """
        self.chunk_size = chunk_size or settings.chunker.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunker.chunk_overlap
        self.min_chunk_size = min_chunk_size or settings.chunker.min_chunk_size
        self.max_chunk_size = max_chunk_size or settings.chunker.max_chunk_size
        self.respect_code_blocks = (
            respect_code_blocks
            if respect_code_blocks is not None
            else settings.chunker.respect_code_blocks
        )
        self.respect_sentences = (
            respect_sentences
            if respect_sentences is not None
            else settings.chunker.respect_sentences
        )

        logger.info(
            f"Initialized SmartChunker (size: {self.chunk_size}, overlap: {self.chunk_overlap})"
        )

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (roughly 4 chars per token for English)."""
        return len(text) // 4

    def _extract_code_blocks(self, text: str) -> List[Tuple[int, int, str]]:
        """
        Extract code blocks with their positions.

        Returns:
            List of (start, end, code_block_content) tuples
        """
        blocks = []
        for match in self.CODE_BLOCK_PATTERN.finditer(text):
            blocks.append((match.start(), match.end(), match.group()))
        return blocks

    def _is_in_code_block(self, position: int, code_blocks: List[Tuple[int, int, str]]) -> bool:
        """Check if a position is inside a code block."""
        for start, end, _ in code_blocks:
            if start <= position < end:
                return True
        return False

    def _find_split_point(
        self,
        text: str,
        target_pos: int,
        code_blocks: List[Tuple[int, int, str]],
    ) -> int:
        """
        Find the best split point near the target position.

        Priority:
        1. After code block (if nearby)
        2. At header boundary
        3. At paragraph boundary (double newline)
        4. At sentence boundary
        5. At word boundary
        """
        # Don't split inside code blocks
        if self.respect_code_blocks:
            for start, end, _ in code_blocks:
                if start <= target_pos < end:
                    # If target is in code block, split after it
                    return end

        # Search window around target position
        window_start = max(0, target_pos - 200)
        window_end = min(len(text), target_pos + 200)
        search_text = text[window_start:window_end]

        # Try to find header boundary
        header_matches = list(self.HEADER_PATTERN.finditer(text[window_start:window_end]))
        if header_matches:
            for match in header_matches:
                split_pos = window_start + match.start()
                if split_pos > target_pos - 100 and split_pos <= target_pos + 100:
                    return split_pos

        # Try paragraph boundary (double newline)
        para_pattern = re.compile(r"\n\n+")
        for match in para_pattern.finditer(search_text):
            split_pos = window_start + match.end()
            if split_pos > target_pos - 100 and split_pos <= target_pos + 100:
                if not self._is_in_code_block(split_pos, code_blocks):
                    return split_pos

        # Try sentence boundary
        if self.respect_sentences:
            for match in self.SENTENCE_END_PATTERN.finditer(search_text):
                split_pos = window_start + match.end()
                if split_pos > target_pos - 100 and split_pos <= target_pos + 100:
                    if not self._is_in_code_block(split_pos, code_blocks):
                        return split_pos

        # Fall back to word boundary
        word_pattern = re.compile(r"\s+")
        best_pos = target_pos
        for match in word_pattern.finditer(search_text):
            split_pos = window_start + match.end()
            if abs(split_pos - target_pos) < abs(best_pos - target_pos):
                if not self._is_in_code_block(split_pos, code_blocks):
                    best_pos = split_pos

        return best_pos

    def _create_chunks_from_segments(
        self,
        segments: List[str],
        metadata: Dict[str, Any],
    ) -> List[Chunk]:
        """
        Create chunks from text segments with proper overlap.

        Merges small segments and ensures proper overlap between chunks.
        """
        chunks = []
        current_content = ""
        current_tokens = 0

        for segment in segments:
            segment_tokens = self._estimate_tokens(segment)

            # If adding segment exceeds max size, finalize current chunk
            if current_tokens + segment_tokens > self.max_chunk_size and current_content:
                chunks.append(current_content.strip())
                # Keep overlap from end of current content
                overlap_chars = self.chunk_overlap * 4  # Approximate chars
                current_content = current_content[-overlap_chars:] if len(current_content) > overlap_chars else ""
                current_tokens = self._estimate_tokens(current_content)

            current_content += segment
            current_tokens += segment_tokens

            # If we've reached target size, finalize chunk
            if current_tokens >= self.chunk_size:
                chunks.append(current_content.strip())
                # Keep overlap
                overlap_chars = self.chunk_overlap * 4
                current_content = current_content[-overlap_chars:] if len(current_content) > overlap_chars else ""
                current_tokens = self._estimate_tokens(current_content)

        # Don't forget the last chunk
        if current_content.strip():
            # Only add if it meets minimum size or there are no chunks yet
            if self._estimate_tokens(current_content) >= self.min_chunk_size or not chunks:
                chunks.append(current_content.strip())
            elif chunks:
                # Append to last chunk if too small
                chunks[-1] = chunks[-1] + "\n\n" + current_content.strip()

        # Convert to Chunk objects
        total_chunks = len(chunks)
        result = []
        for i, content in enumerate(chunks):
            chunk = Chunk(
                id=str(uuid4()),
                content=content,
                token_count=self._estimate_tokens(content),
                chunk_index=i,
                total_chunks=total_chunks,
                metadata=metadata.copy(),
            )
            result.append(chunk)

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

        # Extract code blocks to protect them
        code_blocks = self._extract_code_blocks(text)

        # Split into initial segments by headers and paragraphs
        segments = self._split_by_structure(text, code_blocks)

        # Create chunks with proper sizing and overlap
        chunks = self._create_chunks_from_segments(segments, metadata)

        logger.debug(f"Created {len(chunks)} chunks from text ({len(text)} chars)")
        return chunks

    def _split_by_structure(
        self,
        text: str,
        code_blocks: List[Tuple[int, int, str]],
    ) -> List[str]:
        """
        Split text by structural elements (headers, paragraphs, code blocks).

        Returns segments that respect semantic boundaries.
        """
        segments = []
        current_pos = 0

        # Find all structural boundaries
        boundaries = []

        # Add header boundaries
        for match in self.HEADER_PATTERN.finditer(text):
            if not self._is_in_code_block(match.start(), code_blocks):
                boundaries.append(("header", match.start(), match.group()))

        # Add code block boundaries (treat as atomic units)
        for start, end, content in code_blocks:
            boundaries.append(("code_start", start, content))
            boundaries.append(("code_end", end, ""))

        # Add paragraph boundaries
        para_pattern = re.compile(r"\n\n+")
        for match in para_pattern.finditer(text):
            if not self._is_in_code_block(match.start(), code_blocks):
                boundaries.append(("paragraph", match.start(), match.group()))

        # Sort by position
        boundaries.sort(key=lambda x: x[1])

        # Build segments
        current_segment = ""
        i = 0

        while i < len(text):
            # Check if we're at a code block start
            in_code = False
            code_end = i
            for start, end, content in code_blocks:
                if i == start:
                    # Add any accumulated content as segment
                    if current_segment.strip():
                        segments.append(current_segment)
                        current_segment = ""
                    # Add code block as its own segment
                    segments.append(content)
                    i = end
                    in_code = True
                    break

            if in_code:
                continue

            # Check for header
            header_match = self.HEADER_PATTERN.match(text, i)
            if header_match and header_match.start() == i:
                # Add accumulated content
                if current_segment.strip():
                    segments.append(current_segment)
                    current_segment = ""
                # Start new segment with header
                current_segment = header_match.group() + "\n"
                i = header_match.end()
                continue

            # Check for paragraph boundary
            if text[i:i+2] == "\n\n":
                current_segment += "\n\n"
                # Check if current segment is large enough to split
                if self._estimate_tokens(current_segment) >= self.chunk_size // 2:
                    segments.append(current_segment)
                    current_segment = ""
                i += 2
                while i < len(text) and text[i] == "\n":
                    i += 1
                continue

            current_segment += text[i]
            i += 1

        # Add final segment
        if current_segment.strip():
            segments.append(current_segment)

        return segments

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


def chunk_documents(documents: List[Dict[str, Any]]) -> List[Chunk]:
    """
    Chunk multiple documents.

    Args:
        documents: List of document dicts with url, title, markdown

    Returns:
        List of all chunks from all documents
    """
    chunker = SmartChunker()
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
