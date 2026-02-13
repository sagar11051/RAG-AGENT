"""
Document ingestion pipeline.

Handles the full pipeline from raw documents to embedded chunks in Supabase:
1. Load documents from document store
2. Chunk documents with SmartChunker
3. Generate embeddings with OVH BGE-M3
4. Store in Supabase with pgvector
"""

import asyncio
import re
from typing import Any, Dict, List, Optional, Tuple

from src.chunker import SmartChunker, Chunk, ContentCleaner, ContentCleanerConfig
from src.config.settings import settings
from src.crawler.document_store import DocumentStore
from src.database.operations import DatabaseOperations
from src.database.supabase_client import get_supabase_client
from src.embeddings.ovh_embeddings import OVHEmbeddings, get_embeddings_client
from src.utils.logger import get_logger

logger = get_logger(__name__)


class IngestionPipeline:
    """
    Pipeline for ingesting documents into the vector database.

    Steps:
    1. Load raw documents from file storage
    2. Chunk using SmartChunker (respects semantic boundaries)
    3. Generate embeddings using OVH BGE-M3
    4. Store in Supabase with pgvector indexing
    """

    def __init__(
        self,
        embeddings_client: Optional[OVHEmbeddings] = None,
        db_ops: Optional[DatabaseOperations] = None,
        chunker: Optional[SmartChunker] = None,
        content_cleaner: Optional[ContentCleaner] = None,
        document_store: Optional[DocumentStore] = None,
        batch_size: int = 10,
    ):
        """
        Initialize ingestion pipeline.

        Args:
            embeddings_client: OVH embeddings client
            db_ops: Database operations instance
            chunker: Smart chunker instance
            content_cleaner: Content cleaner for removing boilerplate
            document_store: Document store for loading raw docs
            batch_size: Batch size for embedding generation
        """
        self.embeddings = embeddings_client or get_embeddings_client()
        self.db_ops = db_ops or DatabaseOperations()
        self.chunker = chunker or SmartChunker()
        self.document_store = document_store or DocumentStore()
        self.batch_size = batch_size

        # Initialize content cleaner from settings
        if content_cleaner:
            self.content_cleaner = content_cleaner
        elif settings.content_cleaner.enabled:
            cleaner_config = ContentCleanerConfig(
                simplify_links=settings.content_cleaner.simplify_links,
                preserve_code_blocks=settings.content_cleaner.preserve_code_blocks,
            )
            self.content_cleaner = ContentCleaner(cleaner_config)
        else:
            self.content_cleaner = None

        logger.info(
            f"Initialized IngestionPipeline (content_cleaner={'enabled' if self.content_cleaner else 'disabled'})"
        )

    def _detect_code_in_content(self, content: str) -> bool:
        """Detect if content contains code blocks."""
        return "```" in content

    def _extract_section_title(self, content: str) -> Optional[str]:
        """Extract the first header from content as section title."""
        match = re.search(r"^(#{1,3})\s+(.+)$", content, re.MULTILINE)
        if match:
            return match.group(2).strip()
        return None

    async def ingest_document(
        self,
        url: str,
        title: str,
        markdown: str,
        metadata: Optional[Dict[str, Any]] = None,
        force_update: bool = False,
    ) -> Tuple[str, int]:
        """
        Ingest a single document.

        Args:
            url: Document URL
            title: Document title
            markdown: Document content in markdown
            metadata: Additional metadata
            force_update: Force re-ingestion even if unchanged

        Returns:
            Tuple of (document_id, chunks_count)
        """
        logger.info(f"Ingesting document: {url}")

        # Step 1: Insert/update document in database (store ORIGINAL for backtracking)
        doc, is_new = self.db_ops.ingest_document(
            url=url,
            title=title,
            content=markdown,
            total_tokens=len(markdown) // 4,
            metadata=metadata or {},
            force_update=force_update,
        )

        if not is_new and not force_update:
            logger.info(f"Document unchanged, skipping chunks: {url}")
            return doc["id"], 0

        doc_id = doc["id"]

        # Step 2a: Clean content before chunking (if enabled)
        content_for_chunking = markdown
        if self.content_cleaner:
            content_for_chunking, stats = self.content_cleaner.clean(
                markdown, collect_stats=True
            )
            logger.debug(
                f"Cleaned content: {stats.original_length} -> {stats.cleaned_length} chars "
                f"({stats.reduction_percentage:.1f}% reduction, {stats.links_simplified} links simplified)"
            )

        # Step 2b: Chunk the CLEANED document
        chunks = self.chunker.chunk_document(
            url=url,
            title=title,
            markdown=content_for_chunking,
            extra_metadata=metadata,
        )

        if not chunks:
            logger.warning(f"No chunks generated for: {url}")
            return doc_id, 0

        logger.info(f"Generated {len(chunks)} chunks for: {url}")

        # Step 3: Generate embeddings
        chunk_texts = [c.content for c in chunks]
        embeddings = await self.embeddings.aget_text_embeddings(
            texts=chunk_texts,
            batch_size=self.batch_size,
            show_progress=True,
        )

        logger.info(f"Generated {len(embeddings)} embeddings")

        # Step 4: Prepare chunk data with embeddings
        chunk_data = []
        for chunk, embedding in zip(chunks, embeddings):
            chunk_data.append({
                "content": chunk.content,
                "tokens": chunk.token_count,
                "has_code": self._detect_code_in_content(chunk.content),
                "section_title": self._extract_section_title(chunk.content),
                "metadata": chunk.metadata,
            })

        # Step 5: Store chunks in database
        self.db_ops.ingest_chunks(
            doc_id=doc_id,
            chunks=chunk_data,
            embeddings=embeddings,
            replace_existing=True,
        )

        logger.info(f"Ingested {len(chunks)} chunks for document: {url}")
        return doc_id, len(chunks)

    async def ingest_from_store(
        self,
        force_update: bool = False,
        max_documents: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Ingest all documents from the document store.

        Args:
            force_update: Force re-ingestion of all documents
            max_documents: Maximum number of documents to ingest

        Returns:
            Statistics about the ingestion
        """
        logger.info("Starting ingestion from document store")

        # Load all documents
        documents = self.document_store.load_all()

        if max_documents:
            documents = documents[:max_documents]

        logger.info(f"Found {len(documents)} documents to ingest")

        stats = {
            "total_documents": len(documents),
            "documents_processed": 0,
            "documents_skipped": 0,
            "total_chunks": 0,
            "errors": [],
        }

        for i, doc in enumerate(documents):
            try:
                logger.info(f"Processing document {i + 1}/{len(documents)}: {doc.url}")

                doc_id, chunks_count = await self.ingest_document(
                    url=doc.url,
                    title=doc.title,
                    markdown=doc.markdown,
                    metadata=doc.metadata,
                    force_update=force_update,
                )

                if chunks_count > 0:
                    stats["documents_processed"] += 1
                    stats["total_chunks"] += chunks_count
                else:
                    stats["documents_skipped"] += 1

            except Exception as e:
                error_msg = f"Error ingesting {doc.url}: {e}"
                logger.error(error_msg)
                stats["errors"].append(error_msg)

        logger.info(
            f"Ingestion complete: {stats['documents_processed']} processed, "
            f"{stats['documents_skipped']} skipped, {stats['total_chunks']} chunks"
        )

        return stats

    def get_database_stats(self) -> Dict[str, int]:
        """Get current database statistics."""
        return self.db_ops.client.get_stats()


async def run_ingestion(
    force_update: bool = False,
    max_documents: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run the full ingestion pipeline.

    Args:
        force_update: Force re-ingestion of all documents
        max_documents: Maximum number of documents to ingest

    Returns:
        Ingestion statistics
    """
    pipeline = IngestionPipeline()
    return await pipeline.ingest_from_store(
        force_update=force_update,
        max_documents=max_documents,
    )
