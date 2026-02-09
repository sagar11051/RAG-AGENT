# Docling Integration Plan

## Overview

This plan outlines the integration of [Docling](https://github.com/docling-project/docling) into the LangGraph Documentation RAG Agent for processing PDF, DOCX, and other document formats. Docling is an IBM open-source library that provides advanced document parsing with layout understanding, table extraction, and structured output.

## Goals

1. Enable ingestion of documents from `data/raw_documents/` folder
2. Support multiple formats: PDF, DOCX, PPTX, XLSX, images, HTML
3. Preserve document structure and metadata during parsing
4. Integrate with existing chunking and embedding pipeline
5. Store processed documents in Supabase with full metadata

## Architecture

### Current Pipeline (Web Crawler)
```
Sitemap → Web Crawler → Document Store (JSON) → SmartChunker → Embeddings → Supabase
```

### New Pipeline (Document Ingestion)
```
data/raw_documents/ → Docling Parser → DoclingDocument → HybridChunker → Embeddings → Supabase
          ↓                                    ↓
    File Watcher (optional)              Metadata Extraction
```

### Unified Pipeline
```
                    ┌─────────────────┐
                    │  Data Sources   │
                    └────────┬────────┘
                             │
           ┌─────────────────┼─────────────────┐
           ▼                 ▼                 ▼
    ┌──────────┐      ┌──────────┐      ┌──────────┐
    │  Web     │      │  Local   │      │  API     │
    │  Crawler │      │  Files   │      │  Upload  │
    └────┬─────┘      └────┬─────┘      └────┬─────┘
         │                 │                 │
         │           ┌─────▼─────┐           │
         │           │  Docling  │           │
         │           │  Parser   │           │
         │           └─────┬─────┘           │
         │                 │                 │
         └────────────────►├◄────────────────┘
                          │
                    ┌─────▼─────┐
                    │ Unified   │
                    │ Ingestion │
                    │ Pipeline  │
                    └─────┬─────┘
                          │
              ┌───────────┼───────────┐
              ▼           ▼           ▼
         ┌────────┐ ┌──────────┐ ┌─────────┐
         │Document│ │  Chunks  │ │Embedding│
         │ Store  │ │ + FTS    │ │ Vectors │
         └────────┘ └──────────┘ └─────────┘
```

## Implementation Phases

---

## Phase 1: Dependencies & Configuration

### 1.1 Install Docling Packages

Add to `pyproject.toml`:
```toml
dependencies = [
    # ... existing dependencies ...
    "docling>=2.15.0",
    "docling-core[chunking]>=2.15.0",  # For HybridChunker with HuggingFace tokenizers
]
```

**Note:** Docling requires Python >= 3.10 (current project uses 3.12, so compatible).

### 1.2 Add Configuration Settings

Create new settings class in `src/config/settings.py`:

```python
class DoclingSettings(BaseSettings):
    """Docling document parsing configuration."""

    model_config = SettingsConfigDict(extra="ignore")

    # Input/Output directories
    raw_documents_dir: str = Field(
        default="./data/raw_documents",
        alias="RAW_DOCUMENTS_DIR",
        description="Directory for raw documents to process"
    )
    processed_dir: str = Field(
        default="./data/processed_documents",
        alias="PROCESSED_DOCUMENTS_DIR",
        description="Directory for processed document cache"
    )

    # Chunking settings (for HybridChunker)
    max_tokens: int = Field(
        default=512,
        description="Maximum tokens per chunk"
    )
    merge_peers: bool = Field(
        default=True,
        description="Merge undersized adjacent chunks"
    )

    # Processing options
    ocr_enabled: bool = Field(
        default=True,
        description="Enable OCR for scanned documents"
    )
    table_structure: bool = Field(
        default=True,
        description="Extract table structure"
    )

    # Supported formats
    supported_extensions: str = Field(
        default=".pdf,.docx,.pptx,.xlsx,.html,.png,.jpg,.jpeg,.tiff",
        description="Comma-separated list of supported file extensions"
    )

    def get_supported_extensions(self) -> List[str]:
        """Return supported extensions as a list."""
        return [ext.strip().lower() for ext in self.supported_extensions.split(",")]
```

Add to main `Settings` class:
```python
docling: DoclingSettings = Field(default_factory=DoclingSettings)
```

### 1.3 Create Directory Structure

```
data/
├── raw_documents/           # Place documents here for processing
│   ├── pdfs/
│   ├── docs/
│   └── other/
├── processed_documents/     # Cached parsed documents (JSON)
└── crawled/                 # Existing web crawler output
```

---

## Phase 2: Document Parser Module

### 2.1 Create Docling Parser

Create `src/parser/__init__.py`:
```python
from src.parser.docling_parser import DoclingParser, ParsedDocument
from src.parser.file_scanner import FileScanner

__all__ = ["DoclingParser", "ParsedDocument", "FileScanner"]
```

Create `src/parser/docling_parser.py`:
```python
"""
Docling-based document parser for PDF, DOCX, and other formats.

Uses IBM's Docling library for advanced document understanding:
- Layout analysis and reading order detection
- Table structure extraction
- Code block detection
- Image/figure classification
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.types.doc import DoclingDocument

from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ParsedDocument:
    """Represents a parsed document with metadata."""

    file_path: str
    file_name: str
    file_type: str
    content_markdown: str
    content_text: str
    content_hash: str
    total_tokens: int

    # Metadata
    title: Optional[str] = None
    author: Optional[str] = None
    created_date: Optional[datetime] = None
    modified_date: Optional[datetime] = None
    page_count: Optional[int] = None

    # Docling-specific
    tables_count: int = 0
    figures_count: int = 0
    code_blocks_count: int = 0

    # Raw docling document (for advanced processing)
    docling_document: Optional[DoclingDocument] = None

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def source_url(self) -> str:
        """Generate a file:// URL for consistency with web documents."""
        return f"file://{self.file_path}"


class DoclingParser:
    """
    Document parser using Docling for advanced document understanding.

    Supports: PDF, DOCX, PPTX, XLSX, HTML, images

    Features:
    - Layout analysis with reading order
    - Table structure extraction
    - Code block detection
    - Markdown and plain text export
    """

    def __init__(
        self,
        ocr_enabled: Optional[bool] = None,
        table_structure: Optional[bool] = None,
    ):
        """
        Initialize Docling parser.

        Args:
            ocr_enabled: Enable OCR for scanned documents
            table_structure: Extract table structure
        """
        self.ocr_enabled = ocr_enabled if ocr_enabled is not None else settings.docling.ocr_enabled
        self.table_structure = table_structure if table_structure is not None else settings.docling.table_structure

        # Configure PDF pipeline options
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = self.ocr_enabled
        pipeline_options.do_table_structure = self.table_structure

        # Initialize converter
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

        logger.info(f"Initialized DoclingParser (OCR: {self.ocr_enabled}, Tables: {self.table_structure})")

    def _compute_hash(self, content: str) -> str:
        """Compute SHA256 hash of content."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (roughly 4 chars per token)."""
        return len(text) // 4

    def _extract_metadata(self, doc: DoclingDocument, file_path: Path) -> Dict[str, Any]:
        """Extract metadata from docling document."""
        metadata = {
            "file_name": file_path.name,
            "file_extension": file_path.suffix.lower(),
            "file_size_bytes": file_path.stat().st_size,
            "parsed_at": datetime.now().isoformat(),
        }

        # Extract document-level metadata if available
        if hasattr(doc, 'metadata') and doc.metadata:
            if hasattr(doc.metadata, 'title'):
                metadata["title"] = doc.metadata.title
            if hasattr(doc.metadata, 'authors'):
                metadata["authors"] = doc.metadata.authors

        return metadata

    def _count_elements(self, doc: DoclingDocument) -> Dict[str, int]:
        """Count document elements (tables, figures, code blocks)."""
        counts = {
            "tables": 0,
            "figures": 0,
            "code_blocks": 0,
        }

        # Count elements from docling document structure
        if hasattr(doc, 'tables'):
            counts["tables"] = len(doc.tables) if doc.tables else 0
        if hasattr(doc, 'pictures'):
            counts["figures"] = len(doc.pictures) if doc.pictures else 0

        # Count code blocks in markdown output
        markdown = doc.export_to_markdown()
        counts["code_blocks"] = markdown.count("```")

        return counts

    def parse_file(self, file_path: Union[str, Path]) -> ParsedDocument:
        """
        Parse a single document file.

        Args:
            file_path: Path to the document file

        Returns:
            ParsedDocument with extracted content and metadata
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        logger.info(f"Parsing document: {file_path.name}")

        # Convert document
        result = self.converter.convert(str(file_path))
        doc = result.document

        # Export to markdown and text
        markdown_content = doc.export_to_markdown()
        text_content = doc.export_to_text() if hasattr(doc, 'export_to_text') else markdown_content

        # Extract metadata
        metadata = self._extract_metadata(doc, file_path)
        element_counts = self._count_elements(doc)

        # Build parsed document
        parsed = ParsedDocument(
            file_path=str(file_path.absolute()),
            file_name=file_path.name,
            file_type=file_path.suffix.lower().lstrip('.'),
            content_markdown=markdown_content,
            content_text=text_content,
            content_hash=self._compute_hash(markdown_content),
            total_tokens=self._estimate_tokens(markdown_content),
            title=metadata.get("title") or file_path.stem,
            tables_count=element_counts["tables"],
            figures_count=element_counts["figures"],
            code_blocks_count=element_counts["code_blocks"],
            docling_document=doc,
            metadata=metadata,
        )

        logger.info(
            f"Parsed {file_path.name}: {parsed.total_tokens} tokens, "
            f"{parsed.tables_count} tables, {parsed.figures_count} figures"
        )

        return parsed

    def parse_directory(
        self,
        directory: Union[str, Path],
        recursive: bool = True,
        extensions: Optional[List[str]] = None,
    ) -> Iterator[ParsedDocument]:
        """
        Parse all documents in a directory.

        Args:
            directory: Directory path
            recursive: Search subdirectories
            extensions: File extensions to process (default from settings)

        Yields:
            ParsedDocument for each successfully parsed file
        """
        directory = Path(directory)
        extensions = extensions or settings.docling.get_supported_extensions()

        # Normalize extensions
        extensions = [ext if ext.startswith('.') else f'.{ext}' for ext in extensions]

        # Find files
        pattern = "**/*" if recursive else "*"
        files = [
            f for f in directory.glob(pattern)
            if f.is_file() and f.suffix.lower() in extensions
        ]

        logger.info(f"Found {len(files)} documents to parse in {directory}")

        for file_path in files:
            try:
                yield self.parse_file(file_path)
            except Exception as e:
                logger.error(f"Failed to parse {file_path}: {e}")
                continue
```

### 2.2 Create File Scanner Utility

Create `src/parser/file_scanner.py`:
```python
"""
File scanner for discovering documents to process.
"""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FileInfo:
    """Information about a discovered file."""
    path: Path
    name: str
    extension: str
    size_bytes: int
    modified_time: datetime
    content_hash: Optional[str] = None


class FileScanner:
    """
    Scans directories for documents and tracks processing state.

    Features:
    - Discovers new/modified files
    - Tracks processed files to avoid re-processing
    - Supports incremental updates
    """

    def __init__(
        self,
        scan_directory: Optional[str] = None,
        state_file: Optional[str] = None,
    ):
        """
        Initialize file scanner.

        Args:
            scan_directory: Directory to scan for documents
            state_file: JSON file to track processed files
        """
        self.scan_directory = Path(scan_directory or settings.docling.raw_documents_dir)
        self.state_file = Path(state_file or settings.docling.processed_dir) / "processing_state.json"

        self.supported_extensions = settings.docling.get_supported_extensions()
        self._processed_files: Dict[str, Dict] = {}

        self._load_state()

    def _load_state(self):
        """Load processing state from file."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    self._processed_files = json.load(f)
                logger.info(f"Loaded state: {len(self._processed_files)} processed files")
            except Exception as e:
                logger.warning(f"Failed to load state file: {e}")
                self._processed_files = {}

    def _save_state(self):
        """Save processing state to file."""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, 'w') as f:
            json.dump(self._processed_files, f, indent=2, default=str)

    def scan(self, recursive: bool = True) -> List[FileInfo]:
        """
        Scan directory for all supported documents.

        Args:
            recursive: Search subdirectories

        Returns:
            List of FileInfo for discovered files
        """
        if not self.scan_directory.exists():
            logger.warning(f"Scan directory does not exist: {self.scan_directory}")
            return []

        # Normalize extensions
        extensions = set(
            ext if ext.startswith('.') else f'.{ext}'
            for ext in self.supported_extensions
        )

        # Find files
        pattern = "**/*" if recursive else "*"
        files = []

        for file_path in self.scan_directory.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in extensions:
                stat = file_path.stat()
                files.append(FileInfo(
                    path=file_path,
                    name=file_path.name,
                    extension=file_path.suffix.lower(),
                    size_bytes=stat.st_size,
                    modified_time=datetime.fromtimestamp(stat.st_mtime),
                ))

        logger.info(f"Scanned {self.scan_directory}: found {len(files)} documents")
        return files

    def get_new_or_modified(self, recursive: bool = True) -> List[FileInfo]:
        """
        Get files that are new or modified since last processing.

        Returns:
            List of FileInfo for files needing processing
        """
        all_files = self.scan(recursive)
        to_process = []

        for file_info in all_files:
            file_key = str(file_info.path)

            if file_key not in self._processed_files:
                # New file
                to_process.append(file_info)
            else:
                # Check if modified
                last_modified = self._processed_files[file_key].get("modified_time")
                if last_modified:
                    last_modified = datetime.fromisoformat(last_modified)
                    if file_info.modified_time > last_modified:
                        to_process.append(file_info)

        logger.info(f"Found {len(to_process)} new/modified files to process")
        return to_process

    def mark_processed(
        self,
        file_path: Path,
        content_hash: str,
        doc_id: Optional[str] = None,
    ):
        """Mark a file as processed."""
        self._processed_files[str(file_path)] = {
            "modified_time": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
            "content_hash": content_hash,
            "doc_id": doc_id,
            "processed_at": datetime.now().isoformat(),
        }
        self._save_state()

    def reset_state(self):
        """Clear all processing state (force re-process all)."""
        self._processed_files = {}
        if self.state_file.exists():
            self.state_file.unlink()
        logger.info("Processing state reset")
```

---

## Phase 3: Docling Chunker Integration

### 3.1 Create Docling Chunker Wrapper

Create `src/chunker/docling_chunker.py`:
```python
"""
Docling HybridChunker wrapper for document chunking.

Uses Docling's native chunking which:
- Operates directly on DoclingDocument structure
- Preserves document hierarchy (headings, sections)
- Handles tables and code blocks intelligently
- Provides rich metadata per chunk
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional
from uuid import uuid4

from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.types.doc import DoclingDocument

from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DoclingChunk:
    """Represents a chunk from Docling's HybridChunker."""

    id: str
    content: str
    token_count: int
    chunk_index: int
    total_chunks: int

    # Docling-specific metadata
    headings: List[str] = field(default_factory=list)
    captions: List[str] = field(default_factory=list)
    page_numbers: List[int] = field(default_factory=list)

    # Content type indicators
    has_code: bool = False
    has_table: bool = False
    has_figure: bool = False

    # Source document reference
    source_file: Optional[str] = None
    source_title: Optional[str] = None

    # Full metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


class DoclingChunkerWrapper:
    """
    Wrapper around Docling's HybridChunker for integration with existing pipeline.

    HybridChunker features:
    - Tokenization-aware chunking
    - Respects document structure
    - Merges undersized chunks
    - Preserves heading/caption context
    """

    def __init__(
        self,
        max_tokens: Optional[int] = None,
        merge_peers: Optional[bool] = None,
        tokenizer_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        """
        Initialize Docling chunker.

        Args:
            max_tokens: Maximum tokens per chunk
            merge_peers: Merge undersized adjacent chunks
            tokenizer_model: HuggingFace tokenizer model name
        """
        self.max_tokens = max_tokens or settings.docling.max_tokens
        self.merge_peers = merge_peers if merge_peers is not None else settings.docling.merge_peers

        # Initialize tokenizer
        try:
            from transformers import AutoTokenizer
            hf_tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
            self.tokenizer = HuggingFaceTokenizer(
                tokenizer=hf_tokenizer,
                max_tokens=self.max_tokens,
            )
        except ImportError:
            logger.warning("transformers not installed, using default tokenizer")
            self.tokenizer = None

        # Initialize chunker
        self.chunker = HybridChunker(
            tokenizer=self.tokenizer,
            max_tokens=self.max_tokens,
            merge_peers=self.merge_peers,
        )

        logger.info(f"Initialized DoclingChunkerWrapper (max_tokens: {self.max_tokens})")

    def _extract_chunk_metadata(self, chunk) -> Dict[str, Any]:
        """Extract metadata from a Docling chunk."""
        metadata = {}

        # Extract headings context
        if hasattr(chunk, 'meta') and chunk.meta:
            if hasattr(chunk.meta, 'headings'):
                metadata["headings"] = chunk.meta.headings or []
            if hasattr(chunk.meta, 'captions'):
                metadata["captions"] = chunk.meta.captions or []
            if hasattr(chunk.meta, 'doc_items'):
                # Extract page numbers and content types
                page_numbers = set()
                has_table = False
                has_figure = False
                has_code = False

                for item in chunk.meta.doc_items or []:
                    if hasattr(item, 'prov') and item.prov:
                        for prov in item.prov:
                            if hasattr(prov, 'page'):
                                page_numbers.add(prov.page)

                    # Check content type
                    item_type = getattr(item, 'label', '') or ''
                    if 'table' in item_type.lower():
                        has_table = True
                    if 'figure' in item_type.lower() or 'picture' in item_type.lower():
                        has_figure = True
                    if 'code' in item_type.lower():
                        has_code = True

                metadata["page_numbers"] = sorted(page_numbers)
                metadata["has_table"] = has_table
                metadata["has_figure"] = has_figure
                metadata["has_code"] = has_code

        return metadata

    def chunk_document(
        self,
        docling_document: DoclingDocument,
        source_file: Optional[str] = None,
        source_title: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[DoclingChunk]:
        """
        Chunk a DoclingDocument using HybridChunker.

        Args:
            docling_document: Parsed DoclingDocument object
            source_file: Source file path
            source_title: Document title
            extra_metadata: Additional metadata to include

        Returns:
            List of DoclingChunk objects
        """
        chunks = []
        chunk_iter = self.chunker.chunk(dl_doc=docling_document)

        for idx, chunk in enumerate(chunk_iter):
            # Get chunk text
            text = chunk.text if hasattr(chunk, 'text') else str(chunk)

            # Extract metadata
            chunk_meta = self._extract_chunk_metadata(chunk)

            # Detect code blocks in text
            has_code = chunk_meta.get("has_code", False) or "```" in text

            # Build full metadata
            full_metadata = {
                "source_file": source_file,
                "source_title": source_title,
                **(extra_metadata or {}),
                **chunk_meta,
            }

            # Create chunk object
            docling_chunk = DoclingChunk(
                id=str(uuid4()),
                content=text,
                token_count=len(text) // 4,  # Rough estimate
                chunk_index=idx,
                total_chunks=0,  # Will update after iteration
                headings=chunk_meta.get("headings", []),
                captions=chunk_meta.get("captions", []),
                page_numbers=chunk_meta.get("page_numbers", []),
                has_code=has_code,
                has_table=chunk_meta.get("has_table", False),
                has_figure=chunk_meta.get("has_figure", False),
                source_file=source_file,
                source_title=source_title,
                metadata=full_metadata,
            )

            chunks.append(docling_chunk)

        # Update total_chunks
        for chunk in chunks:
            chunk.total_chunks = len(chunks)

        logger.debug(f"Created {len(chunks)} chunks from document")
        return chunks

    def chunk_from_markdown(
        self,
        markdown: str,
        source_file: Optional[str] = None,
        source_title: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[DoclingChunk]:
        """
        Chunk markdown text (fallback for non-Docling sources).

        This creates a simple DoclingDocument from markdown and chunks it.
        For best results, use chunk_document with a properly parsed document.
        """
        # For markdown fallback, use the existing SmartChunker
        # and convert to DoclingChunk format
        from src.chunker.smart_chunker import SmartChunker

        smart_chunker = SmartChunker(
            chunk_size=self.max_tokens,
            chunk_overlap=self.max_tokens // 4,
        )

        smart_chunks = smart_chunker.chunk_text(
            markdown,
            metadata={"source_file": source_file, "source_title": source_title},
        )

        docling_chunks = []
        for chunk in smart_chunks:
            docling_chunks.append(DoclingChunk(
                id=chunk.id,
                content=chunk.content,
                token_count=chunk.token_count,
                chunk_index=chunk.chunk_index,
                total_chunks=chunk.total_chunks,
                has_code="```" in chunk.content,
                source_file=source_file,
                source_title=source_title,
                metadata={**chunk.metadata, **(extra_metadata or {})},
            ))

        return docling_chunks
```

### 3.2 Update Chunker Package Exports

Update `src/chunker/__init__.py`:
```python
from src.chunker.smart_chunker import SmartChunker, Chunk, chunk_documents
from src.chunker.content_cleaner import ContentCleaner, ContentCleanerConfig, CleaningStats
from src.chunker.docling_chunker import DoclingChunkerWrapper, DoclingChunk

__all__ = [
    "SmartChunker",
    "Chunk",
    "chunk_documents",
    "ContentCleaner",
    "ContentCleanerConfig",
    "CleaningStats",
    "DoclingChunkerWrapper",
    "DoclingChunk",
]
```

---

## Phase 4: Document Ingestion Pipeline

### 4.1 Create Document Ingestion Pipeline

Create `src/retrieval/document_ingestion.py`:
```python
"""
Document ingestion pipeline for local files.

Handles the full pipeline from raw documents to embedded chunks:
1. Scan for new/modified documents
2. Parse with Docling
3. Chunk with HybridChunker
4. Generate embeddings
5. Store in Supabase
"""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.chunker.docling_chunker import DoclingChunkerWrapper, DoclingChunk
from src.config.settings import settings
from src.database.operations import DatabaseOperations
from src.embeddings.ovh_embeddings import OVHEmbeddings, get_embeddings_client
from src.parser.docling_parser import DoclingParser, ParsedDocument
from src.parser.file_scanner import FileScanner
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DocumentIngestionPipeline:
    """
    Pipeline for ingesting local documents into the vector database.

    Steps:
    1. Scan directory for new/modified documents
    2. Parse documents with Docling
    3. Chunk using DoclingChunkerWrapper (HybridChunker)
    4. Generate embeddings using OVH BGE-M3
    5. Store in Supabase with pgvector
    """

    def __init__(
        self,
        parser: Optional[DoclingParser] = None,
        chunker: Optional[DoclingChunkerWrapper] = None,
        embeddings_client: Optional[OVHEmbeddings] = None,
        db_ops: Optional[DatabaseOperations] = None,
        file_scanner: Optional[FileScanner] = None,
        batch_size: int = 10,
    ):
        """
        Initialize document ingestion pipeline.
        """
        self.parser = parser or DoclingParser()
        self.chunker = chunker or DoclingChunkerWrapper()
        self.embeddings = embeddings_client or get_embeddings_client()
        self.db_ops = db_ops or DatabaseOperations()
        self.file_scanner = file_scanner or FileScanner()
        self.batch_size = batch_size

        logger.info("Initialized DocumentIngestionPipeline")

    async def ingest_document(
        self,
        parsed_doc: ParsedDocument,
        force_update: bool = False,
    ) -> Tuple[str, int]:
        """
        Ingest a single parsed document.

        Args:
            parsed_doc: Parsed document from Docling
            force_update: Force re-ingestion even if unchanged

        Returns:
            Tuple of (document_id, chunks_count)
        """
        logger.info(f"Ingesting document: {parsed_doc.file_name}")

        # Step 1: Insert/update document in database
        doc, is_new = self.db_ops.ingest_document(
            url=parsed_doc.source_url,
            title=parsed_doc.title or parsed_doc.file_name,
            content=parsed_doc.content_markdown,
            total_tokens=parsed_doc.total_tokens,
            metadata={
                "file_path": parsed_doc.file_path,
                "file_type": parsed_doc.file_type,
                "tables_count": parsed_doc.tables_count,
                "figures_count": parsed_doc.figures_count,
                "code_blocks_count": parsed_doc.code_blocks_count,
                **parsed_doc.metadata,
            },
            force_update=force_update,
        )

        if not is_new and not force_update:
            logger.info(f"Document unchanged, skipping: {parsed_doc.file_name}")
            return doc["id"], 0

        doc_id = doc["id"]

        # Step 2: Chunk the document
        if parsed_doc.docling_document:
            # Use native Docling chunking for best results
            chunks = self.chunker.chunk_document(
                docling_document=parsed_doc.docling_document,
                source_file=parsed_doc.file_path,
                source_title=parsed_doc.title,
                extra_metadata={"file_type": parsed_doc.file_type},
            )
        else:
            # Fallback to markdown chunking
            chunks = self.chunker.chunk_from_markdown(
                markdown=parsed_doc.content_markdown,
                source_file=parsed_doc.file_path,
                source_title=parsed_doc.title,
                extra_metadata={"file_type": parsed_doc.file_type},
            )

        if not chunks:
            logger.warning(f"No chunks generated for: {parsed_doc.file_name}")
            return doc_id, 0

        logger.info(f"Generated {len(chunks)} chunks for: {parsed_doc.file_name}")

        # Step 3: Generate embeddings
        chunk_texts = [c.content for c in chunks]
        embeddings = await self.embeddings.aget_text_embeddings(
            texts=chunk_texts,
            batch_size=self.batch_size,
            show_progress=True,
        )

        logger.info(f"Generated {len(embeddings)} embeddings")

        # Step 4: Prepare chunk data
        chunk_data = []
        for chunk, embedding in zip(chunks, embeddings):
            # Build section title from headings
            section_title = None
            if chunk.headings:
                section_title = " > ".join(chunk.headings[-3:])  # Last 3 headings

            chunk_data.append({
                "content": chunk.content,
                "tokens": chunk.token_count,
                "has_code": chunk.has_code,
                "section_title": section_title,
                "metadata": {
                    **chunk.metadata,
                    "page_numbers": chunk.page_numbers,
                    "has_table": chunk.has_table,
                    "has_figure": chunk.has_figure,
                    "headings": chunk.headings,
                },
            })

        # Step 5: Store chunks in database
        self.db_ops.ingest_chunks(
            doc_id=doc_id,
            chunks=chunk_data,
            embeddings=embeddings,
            replace_existing=True,
        )

        # Mark file as processed
        self.file_scanner.mark_processed(
            file_path=Path(parsed_doc.file_path),
            content_hash=parsed_doc.content_hash,
            doc_id=doc_id,
        )

        logger.info(f"Ingested {len(chunks)} chunks for: {parsed_doc.file_name}")
        return doc_id, len(chunks)

    async def ingest_file(
        self,
        file_path: str,
        force_update: bool = False,
    ) -> Tuple[str, int]:
        """
        Ingest a single file.

        Args:
            file_path: Path to file
            force_update: Force re-ingestion

        Returns:
            Tuple of (document_id, chunks_count)
        """
        parsed = self.parser.parse_file(file_path)
        return await self.ingest_document(parsed, force_update)

    async def ingest_directory(
        self,
        directory: Optional[str] = None,
        force_update: bool = False,
        incremental: bool = True,
    ) -> Dict[str, Any]:
        """
        Ingest all documents from a directory.

        Args:
            directory: Directory path (default from settings)
            force_update: Force re-ingestion of all documents
            incremental: Only process new/modified files

        Returns:
            Ingestion statistics
        """
        directory = Path(directory or settings.docling.raw_documents_dir)

        logger.info(f"Starting document ingestion from: {directory}")

        # Get files to process
        if incremental and not force_update:
            files = self.file_scanner.get_new_or_modified()
        else:
            files = self.file_scanner.scan()

        if not files:
            logger.info("No documents to process")
            return {
                "total_files": 0,
                "processed": 0,
                "skipped": 0,
                "errors": [],
                "total_chunks": 0,
            }

        stats = {
            "total_files": len(files),
            "processed": 0,
            "skipped": 0,
            "errors": [],
            "total_chunks": 0,
        }

        for i, file_info in enumerate(files):
            try:
                logger.info(f"Processing {i + 1}/{len(files)}: {file_info.name}")

                doc_id, chunks_count = await self.ingest_file(
                    str(file_info.path),
                    force_update=force_update,
                )

                if chunks_count > 0:
                    stats["processed"] += 1
                    stats["total_chunks"] += chunks_count
                else:
                    stats["skipped"] += 1

            except Exception as e:
                error_msg = f"Error processing {file_info.name}: {e}"
                logger.error(error_msg)
                stats["errors"].append(error_msg)

        logger.info(
            f"Ingestion complete: {stats['processed']} processed, "
            f"{stats['skipped']} skipped, {stats['total_chunks']} chunks"
        )

        return stats


async def run_document_ingestion(
    directory: Optional[str] = None,
    force_update: bool = False,
    incremental: bool = True,
) -> Dict[str, Any]:
    """
    Run the document ingestion pipeline.

    Args:
        directory: Directory to scan (default from settings)
        force_update: Force re-process all documents
        incremental: Only process new/modified files

    Returns:
        Ingestion statistics
    """
    pipeline = DocumentIngestionPipeline()
    return await pipeline.ingest_directory(
        directory=directory,
        force_update=force_update,
        incremental=incremental,
    )
```

---

## Phase 5: CLI Script & Testing

### 5.1 Create Ingestion CLI Script

Create `scripts/ingest_documents.py`:
```python
#!/usr/bin/env python
"""
Document ingestion CLI script.

Usage:
    uv run python scripts/ingest_documents.py              # Incremental (new/modified only)
    uv run python scripts/ingest_documents.py --force      # Re-process all
    uv run python scripts/ingest_documents.py --file doc.pdf  # Single file
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rich.console import Console
from rich.table import Table

from src.config.settings import settings
from src.retrieval.document_ingestion import (
    DocumentIngestionPipeline,
    run_document_ingestion,
)

console = Console()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ingest documents into the RAG vector database"
    )
    parser.add_argument(
        "--directory", "-d",
        type=str,
        default=None,
        help=f"Directory to scan (default: {settings.docling.raw_documents_dir})"
    )
    parser.add_argument(
        "--file", "-f",
        type=str,
        default=None,
        help="Process a single file"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-process all documents"
    )
    parser.add_argument(
        "--no-incremental",
        action="store_true",
        help="Process all files, not just new/modified"
    )
    parser.add_argument(
        "--reset-state",
        action="store_true",
        help="Reset processing state (mark all as unprocessed)"
    )
    return parser.parse_args()


async def main():
    args = parse_args()

    console.print("\n[bold blue]Document Ingestion Pipeline[/bold blue]\n")

    # Handle state reset
    if args.reset_state:
        from src.parser.file_scanner import FileScanner
        scanner = FileScanner()
        scanner.reset_state()
        console.print("[yellow]Processing state reset[/yellow]\n")

    # Single file mode
    if args.file:
        console.print(f"Processing single file: [cyan]{args.file}[/cyan]\n")

        pipeline = DocumentIngestionPipeline()
        try:
            doc_id, chunks = await pipeline.ingest_file(
                args.file,
                force_update=args.force,
            )
            console.print(f"[green]Success![/green] Document ID: {doc_id}, Chunks: {chunks}")
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")
            return 1
        return 0

    # Directory mode
    directory = args.directory or settings.docling.raw_documents_dir
    console.print(f"Scanning directory: [cyan]{directory}[/cyan]")
    console.print(f"Force update: {'Yes' if args.force else 'No'}")
    console.print(f"Incremental: {'No' if args.no_incremental else 'Yes'}\n")

    stats = await run_document_ingestion(
        directory=directory,
        force_update=args.force,
        incremental=not args.no_incremental,
    )

    # Display results
    table = Table(title="Ingestion Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Files Found", str(stats["total_files"]))
    table.add_row("Documents Processed", str(stats["processed"]))
    table.add_row("Documents Skipped", str(stats["skipped"]))
    table.add_row("Total Chunks Created", str(stats["total_chunks"]))
    table.add_row("Errors", str(len(stats["errors"])))

    console.print(table)

    if stats["errors"]:
        console.print("\n[red]Errors:[/red]")
        for error in stats["errors"]:
            console.print(f"  - {error}")

    return 0 if not stats["errors"] else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
```

### 5.2 Create Test Script

Create `scripts/test_phase9_docling.py`:
```python
#!/usr/bin/env python
"""
Phase 9: Test Docling document parsing integration.

Tests:
1. Docling parser initialization
2. Single document parsing
3. Chunking with HybridChunker
4. Full ingestion pipeline
"""

import asyncio
import sys
from pathlib import Path

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rich.console import Console

console = Console()


def print_test(name: str, passed: bool, details: str = ""):
    status = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"
    console.print(f"  {status} {name}")
    if details:
        console.print(f"       {details}")


async def test_docling_parser():
    """Test Docling parser initialization and basic parsing."""
    console.print("\n[bold]Test 1: Docling Parser[/bold]")

    try:
        from src.parser.docling_parser import DoclingParser

        parser = DoclingParser()
        print_test("Parser initialization", True)

        # Check if test document exists
        test_file = Path("data/raw_documents/test.pdf")
        if test_file.exists():
            parsed = parser.parse_file(test_file)
            print_test(
                "Document parsing",
                parsed is not None and len(parsed.content_markdown) > 0,
                f"Tokens: {parsed.total_tokens}, Tables: {parsed.tables_count}"
            )
        else:
            print_test("Document parsing", True, "Skipped - no test file")

        return True
    except Exception as e:
        print_test("Docling parser", False, str(e))
        return False


async def test_docling_chunker():
    """Test Docling chunker wrapper."""
    console.print("\n[bold]Test 2: Docling Chunker[/bold]")

    try:
        from src.chunker.docling_chunker import DoclingChunkerWrapper

        chunker = DoclingChunkerWrapper()
        print_test("Chunker initialization", True)

        # Test markdown chunking fallback
        test_markdown = """
# Introduction
This is a test document for chunking.

## Section 1
Here is some content with a code block:

```python
def hello():
    print("Hello, World!")
```

## Section 2
More content here.
"""
        chunks = chunker.chunk_from_markdown(
            test_markdown,
            source_file="test.md",
            source_title="Test Document",
        )

        print_test(
            "Markdown chunking",
            len(chunks) > 0,
            f"Created {len(chunks)} chunks"
        )

        return True
    except Exception as e:
        print_test("Docling chunker", False, str(e))
        return False


async def test_file_scanner():
    """Test file scanner functionality."""
    console.print("\n[bold]Test 3: File Scanner[/bold]")

    try:
        from src.parser.file_scanner import FileScanner

        scanner = FileScanner()
        print_test("Scanner initialization", True)

        # Scan for files
        files = scanner.scan()
        print_test(
            "Directory scan",
            True,
            f"Found {len(files)} documents"
        )

        return True
    except Exception as e:
        print_test("File scanner", False, str(e))
        return False


async def test_ingestion_pipeline():
    """Test full ingestion pipeline (dry run)."""
    console.print("\n[bold]Test 4: Ingestion Pipeline[/bold]")

    try:
        from src.retrieval.document_ingestion import DocumentIngestionPipeline

        pipeline = DocumentIngestionPipeline()
        print_test("Pipeline initialization", True)

        # Check database connection
        stats = pipeline.db_ops.client.get_stats()
        print_test(
            "Database connection",
            True,
            f"Documents: {stats.get('documents', 0)}, Chunks: {stats.get('chunks', 0)}"
        )

        return True
    except Exception as e:
        print_test("Ingestion pipeline", False, str(e))
        return False


async def main():
    console.print("[bold blue]Phase 9: Docling Integration Tests[/bold blue]")
    console.print("=" * 50)

    results = []

    results.append(await test_docling_parser())
    results.append(await test_docling_chunker())
    results.append(await test_file_scanner())
    results.append(await test_ingestion_pipeline())

    console.print("\n" + "=" * 50)

    passed = sum(results)
    total = len(results)

    if passed == total:
        console.print(f"[bold green]All {total} tests passed![/bold green]")
        return 0
    else:
        console.print(f"[bold red]{total - passed} of {total} tests failed[/bold red]")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
```

---

## Phase 6: Database Schema Updates

### 6.1 Add Document Source Type

Create migration `src/database/schema_docling.sql`:
```sql
-- =============================================================================
-- Docling Integration - Schema Updates
-- =============================================================================

-- Add source_type column to documents table to distinguish web vs file sources
ALTER TABLE documents
ADD COLUMN IF NOT EXISTS source_type TEXT DEFAULT 'web'
CHECK (source_type IN ('web', 'file', 'api'));

-- Add file-specific metadata columns
ALTER TABLE documents
ADD COLUMN IF NOT EXISTS file_path TEXT,
ADD COLUMN IF NOT EXISTS file_type TEXT,
ADD COLUMN IF NOT EXISTS file_size_bytes INTEGER;

-- Add page tracking to chunks
ALTER TABLE chunks
ADD COLUMN IF NOT EXISTS page_numbers INTEGER[] DEFAULT '{}',
ADD COLUMN IF NOT EXISTS has_table BOOLEAN DEFAULT FALSE,
ADD COLUMN IF NOT EXISTS has_figure BOOLEAN DEFAULT FALSE;

-- Index for filtering by source type
CREATE INDEX IF NOT EXISTS idx_documents_source_type ON documents(source_type);

-- Index for file path lookups
CREATE INDEX IF NOT EXISTS idx_documents_file_path ON documents(file_path);

-- Update comments
COMMENT ON COLUMN documents.source_type IS 'Source type: web (crawled), file (local upload), api (external)';
COMMENT ON COLUMN documents.file_path IS 'Original file path for file-based documents';
COMMENT ON COLUMN chunks.page_numbers IS 'Page numbers where this chunk appears (for PDFs)';
```

---

## Phase 7: API Integration (Optional)

### 7.1 Add Upload Endpoint

Add to `src/api/routes/documents.py`:
```python
"""
Document upload and management API endpoints.
"""

import shutil
from pathlib import Path
from typing import List

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from src.config.settings import settings
from src.retrieval.document_ingestion import DocumentIngestionPipeline

router = APIRouter(prefix="/documents", tags=["documents"])


class UploadResponse(BaseModel):
    """Response model for document upload."""
    file_name: str
    document_id: str
    chunks_created: int
    status: str


@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and process a document.

    Supported formats: PDF, DOCX, PPTX, XLSX, HTML, images
    """
    # Validate extension
    supported = settings.docling.get_supported_extensions()
    extension = Path(file.filename).suffix.lower()

    if extension not in supported:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {extension}. Supported: {supported}"
        )

    # Save uploaded file
    upload_dir = Path(settings.docling.raw_documents_dir) / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    file_path = upload_dir / file.filename

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Process document
    pipeline = DocumentIngestionPipeline()

    try:
        doc_id, chunks = await pipeline.ingest_file(str(file_path))

        return UploadResponse(
            file_name=file.filename,
            document_id=doc_id,
            chunks_created=chunks,
            status="success",
        )
    except Exception as e:
        # Clean up on failure
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_document_stats():
    """Get document and chunk statistics."""
    pipeline = DocumentIngestionPipeline()
    return pipeline.db_ops.client.get_stats()
```

---

## Summary Checklist

### Phase 1: Dependencies & Configuration
- [ ] Add `docling` and `docling-core[chunking]` to `pyproject.toml`
- [ ] Create `DoclingSettings` class in `settings.py`
- [ ] Create `data/raw_documents/` directory structure

### Phase 2: Document Parser Module
- [ ] Create `src/parser/__init__.py`
- [ ] Create `src/parser/docling_parser.py`
- [ ] Create `src/parser/file_scanner.py`

### Phase 3: Docling Chunker Integration
- [ ] Create `src/chunker/docling_chunker.py`
- [ ] Update `src/chunker/__init__.py` exports

### Phase 4: Document Ingestion Pipeline
- [ ] Create `src/retrieval/document_ingestion.py`

### Phase 5: CLI Script & Testing
- [ ] Create `scripts/ingest_documents.py`
- [ ] Create `scripts/test_phase9_docling.py`

### Phase 6: Database Schema Updates
- [ ] Create and run `src/database/schema_docling.sql`

### Phase 7: API Integration (Optional)
- [ ] Add upload endpoint to API

---

## Usage After Implementation

```bash
# Install new dependencies
uv sync

# Create directories
mkdir -p data/raw_documents

# Place documents in the folder
cp my_document.pdf data/raw_documents/

# Run ingestion
uv run python scripts/ingest_documents.py

# Or process single file
uv run python scripts/ingest_documents.py --file data/raw_documents/my_document.pdf

# Force re-process all
uv run python scripts/ingest_documents.py --force

# Run tests
uv run python scripts/test_phase9_docling.py
```

---

## Sources

- [Docling GitHub Repository](https://github.com/docling-project/docling)
- [Docling Documentation](https://docling-project.github.io/docling/)
- [Docling Chunking Concepts](https://docling-project.github.io/docling/concepts/chunking/)
- [Hybrid Chunking Example](https://docling-project.github.io/docling/examples/hybrid_chunking/)
- [LangChain Docling Integration](https://docs.langchain.com/oss/python/integrations/document_loaders/docling)
- [Docling PyPI](https://pypi.org/project/docling/)
