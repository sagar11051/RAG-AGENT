"""
Web crawler using Crawl4AI for extracting content from documentation pages.

Features:
- Async crawling with Crawl4AI
- Clean text extraction with markdown output
- Code block preservation
- Metadata extraction (title, URL, timestamp)
- Rate limiting and retry logic
"""

import asyncio
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CrawledDocument:
    """Represents a crawled document with content and metadata."""

    url: str
    title: str
    content: str
    markdown: str
    content_hash: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    crawl_date: datetime = field(default_factory=datetime.now)
    success: bool = True
    error: Optional[str] = None

    @property
    def total_tokens(self) -> int:
        """Estimate token count (rough approximation: 1 token ~ 4 chars)."""
        return len(self.content) // 4


class WebCrawler:
    """
    Web crawler using Crawl4AI for documentation sites.

    Extracts clean text content while preserving code blocks
    and important formatting.
    """

    def __init__(
        self,
        delay: float = None,
        timeout: int = None,
        max_retries: int = None,
        user_agent: str = None,
    ):
        """
        Initialize web crawler.

        Args:
            delay: Delay between requests in seconds
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            user_agent: User agent string
        """
        self.delay = delay or settings.crawler.crawl_delay
        self.timeout = timeout or settings.crawler.timeout
        self.max_retries = max_retries or settings.crawler.retry_attempts
        self.user_agent = user_agent or settings.crawler.user_agent

        logger.info(
            f"Initialized WebCrawler (delay: {self.delay}s, timeout: {self.timeout}s)"
        )

    def _compute_hash(self, content: str) -> str:
        """Compute SHA256 hash of content."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    async def crawl_url(self, url: str) -> CrawledDocument:
        """
        Crawl a single URL and extract content.

        Args:
            url: URL to crawl

        Returns:
            CrawledDocument with extracted content
        """
        logger.debug(f"Crawling: {url}")

        browser_config = BrowserConfig(
            headless=True,
            user_agent=self.user_agent,
            verbose=False,  # Suppress console output (avoids Windows encoding issues)
        )

        crawl_config = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            page_timeout=self.timeout * 1000,  # Convert to ms
        )

        for attempt in range(self.max_retries):
            try:
                async with AsyncWebCrawler(config=browser_config) as crawler:
                    result = await crawler.arun(url=url, config=crawl_config)

                    if result.success:
                        # Extract content
                        title = result.metadata.get("title", "") if result.metadata else ""
                        content = result.cleaned_html or ""
                        markdown = result.markdown or content

                        # Compute content hash
                        content_hash = self._compute_hash(markdown)

                        doc = CrawledDocument(
                            url=url,
                            title=title,
                            content=content,
                            markdown=markdown,
                            content_hash=content_hash,
                            metadata={
                                "status_code": getattr(result, "status_code", 200),
                                "links_count": len(result.links.get("internal", [])) if result.links else 0,
                            },
                        )

                        logger.debug(f"Successfully crawled: {url} ({len(markdown)} chars)")
                        return doc
                    else:
                        error_msg = getattr(result, "error_message", "Unknown error")
                        logger.warning(f"Crawl failed for {url}: {error_msg}")

                        if attempt == self.max_retries - 1:
                            return CrawledDocument(
                                url=url,
                                title="",
                                content="",
                                markdown="",
                                content_hash="",
                                success=False,
                                error=error_msg,
                            )

            except Exception as e:
                logger.error(f"Error crawling {url} (attempt {attempt + 1}): {e}")
                if attempt == self.max_retries - 1:
                    return CrawledDocument(
                        url=url,
                        title="",
                        content="",
                        markdown="",
                        content_hash="",
                        success=False,
                        error=str(e),
                    )

            # Wait before retry
            await asyncio.sleep(2 ** attempt)

        # Should not reach here, but just in case
        return CrawledDocument(
            url=url,
            title="",
            content="",
            markdown="",
            content_hash="",
            success=False,
            error="Max retries exceeded",
        )

    async def crawl_urls(
        self,
        urls: List[str],
        concurrency: int = 3,
        show_progress: bool = True,
    ) -> List[CrawledDocument]:
        """
        Crawl multiple URLs with controlled concurrency.

        Args:
            urls: List of URLs to crawl
            concurrency: Number of concurrent crawls
            show_progress: Whether to log progress

        Returns:
            List of CrawledDocument objects
        """
        documents = []
        total = len(urls)
        semaphore = asyncio.Semaphore(concurrency)

        async def crawl_with_semaphore(url: str, index: int) -> CrawledDocument:
            async with semaphore:
                doc = await self.crawl_url(url)
                if show_progress and (index + 1) % 10 == 0:
                    logger.info(f"Progress: {index + 1}/{total} URLs crawled")
                # Add delay between requests
                await asyncio.sleep(self.delay)
                return doc

        # Create tasks
        tasks = [
            crawl_with_semaphore(url, i)
            for i, url in enumerate(urls)
        ]

        # Execute and gather results
        documents = await asyncio.gather(*tasks)

        # Log summary
        successful = sum(1 for d in documents if d.success)
        logger.info(f"Crawling complete: {successful}/{total} successful")

        return list(documents)


async def crawl_documentation(
    base_url: str = None,
    max_pages: int = None,
    filter_pattern: Optional[str] = None,
) -> List[CrawledDocument]:
    """
    Crawl documentation from a base URL using its sitemap.

    Args:
        base_url: Base URL of the documentation site
        max_pages: Maximum number of pages to crawl
        filter_pattern: Optional URL filter pattern

    Returns:
        List of crawled documents
    """
    from src.crawler.sitemap_parser import SitemapParser, get_sitemap_url

    base_url = base_url or settings.crawler.base_url
    max_pages = max_pages or settings.crawler.max_pages

    # Get sitemap URL
    sitemap_url = get_sitemap_url(base_url)
    logger.info(f"Starting documentation crawl from: {sitemap_url}")

    # Parse sitemap
    parser = SitemapParser()
    urls = await parser.aparse(
        sitemap_url,
        filter_pattern=filter_pattern,
        max_urls=max_pages,
    )

    if not urls:
        logger.warning("No URLs found in sitemap")
        return []

    logger.info(f"Found {len(urls)} URLs to crawl")

    # Crawl URLs
    crawler = WebCrawler()
    documents = await crawler.crawl_urls(urls)

    return documents
