"""
Local document storage for crawled content.

Provides file-based storage for raw documents with:
- JSON format storage
- Directory structure based on URL
- Content hash for deduplication
- Incremental update support
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from src.config.settings import settings
from src.crawler.web_crawler import CrawledDocument
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DocumentStore:
    """
    File-based storage for crawled documents.

    Stores documents as JSON files organized by domain and path.
    """

    def __init__(self, base_dir: Optional[str] = None):
        """
        Initialize document store.

        Args:
            base_dir: Base directory for storage (default from settings)
        """
        self.base_dir = Path(base_dir or settings.data_dir) / "raw_documents"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Document store initialized at: {self.base_dir}")

    def _url_to_path(self, url: str) -> Path:
        """Convert URL to file path."""
        parsed = urlparse(url)
        domain = parsed.netloc.replace(":", "_")
        path = parsed.path.strip("/").replace("/", "_") or "index"

        # Ensure valid filename
        path = "".join(c if c.isalnum() or c in "-_" else "_" for c in path)

        return self.base_dir / domain / f"{path}.json"

    def _document_to_dict(self, doc: CrawledDocument) -> Dict[str, Any]:
        """Convert CrawledDocument to dictionary for storage.

        Note: We only store markdown content (not raw HTML) to keep files clean and small.
        """
        return {
            "url": doc.url,
            "title": doc.title,
            "markdown": doc.markdown,
            "content_hash": doc.content_hash,
            "total_tokens": doc.total_tokens,
            "crawl_date": doc.crawl_date.isoformat(),
            "metadata": doc.metadata,
            "success": doc.success,
            "error": doc.error,
        }

    def _dict_to_document(self, data: Dict[str, Any]) -> CrawledDocument:
        """Convert dictionary to CrawledDocument."""
        return CrawledDocument(
            url=data["url"],
            title=data.get("title", ""),
            content=data.get("content", ""),
            markdown=data.get("markdown", ""),
            content_hash=data.get("content_hash", ""),
            metadata=data.get("metadata", {}),
            crawl_date=datetime.fromisoformat(data["crawl_date"]) if "crawl_date" in data else datetime.now(),
            success=data.get("success", True),
            error=data.get("error"),
        )

    def save(self, document: CrawledDocument) -> Path:
        """
        Save a document to storage.

        Args:
            document: Document to save

        Returns:
            Path where document was saved
        """
        file_path = self._url_to_path(document.url)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        data = self._document_to_dict(document)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.debug(f"Saved document: {document.url} -> {file_path}")
        return file_path

    def save_many(self, documents: List[CrawledDocument]) -> List[Path]:
        """
        Save multiple documents.

        Args:
            documents: List of documents to save

        Returns:
            List of paths where documents were saved
        """
        paths = []
        for doc in documents:
            if doc.success:  # Only save successful crawls
                paths.append(self.save(doc))
        logger.info(f"Saved {len(paths)} documents")
        return paths

    def load(self, url: str) -> Optional[CrawledDocument]:
        """
        Load a document by URL.

        Args:
            url: URL of the document

        Returns:
            CrawledDocument if found, None otherwise
        """
        file_path = self._url_to_path(url)

        if not file_path.exists():
            return None

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return self._dict_to_document(data)
        except Exception as e:
            logger.error(f"Failed to load document {url}: {e}")
            return None

    def load_all(self) -> List[CrawledDocument]:
        """
        Load all documents from storage.

        Returns:
            List of all stored documents
        """
        documents = []

        for json_file in self.base_dir.rglob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                documents.append(self._dict_to_document(data))
            except Exception as e:
                logger.error(f"Failed to load {json_file}: {e}")

        logger.info(f"Loaded {len(documents)} documents from storage")
        return documents

    def exists(self, url: str) -> bool:
        """Check if a document exists in storage."""
        return self._url_to_path(url).exists()

    def get_hash(self, url: str) -> Optional[str]:
        """Get content hash for a stored document."""
        doc = self.load(url)
        return doc.content_hash if doc else None

    def needs_update(self, url: str, new_hash: str) -> bool:
        """
        Check if a document needs to be updated.

        Args:
            url: Document URL
            new_hash: Hash of new content

        Returns:
            True if document doesn't exist or hash is different
        """
        existing_hash = self.get_hash(url)
        return existing_hash is None or existing_hash != new_hash

    def delete(self, url: str) -> bool:
        """Delete a document from storage."""
        file_path = self._url_to_path(url)

        if file_path.exists():
            file_path.unlink()
            logger.debug(f"Deleted document: {url}")
            return True
        return False

    def clear(self) -> int:
        """
        Delete all stored documents.

        Returns:
            Number of documents deleted
        """
        count = 0
        for json_file in self.base_dir.rglob("*.json"):
            json_file.unlink()
            count += 1
        logger.info(f"Cleared {count} documents from storage")
        return count

    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        documents = list(self.base_dir.rglob("*.json"))
        total_size = sum(f.stat().st_size for f in documents)

        return {
            "document_count": len(documents),
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "storage_path": str(self.base_dir),
        }
