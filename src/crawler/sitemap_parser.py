"""
Sitemap parser for extracting URLs from sitemap.xml files.

Supports:
- Standard sitemap.xml files
- Sitemap index files (sitemaps that reference other sitemaps)
- URL filtering based on patterns
"""

import asyncio
from typing import List, Optional, Set
from urllib.parse import urljoin, urlparse
import xml.etree.ElementTree as ET

import httpx

from src.utils.logger import get_logger

logger = get_logger(__name__)

# XML namespaces used in sitemaps
SITEMAP_NS = {
    "sm": "http://www.sitemaps.org/schemas/sitemap/0.9",
}


class SitemapParser:
    """
    Parser for sitemap.xml files.

    Extracts URLs from sitemaps, handling both regular sitemaps
    and sitemap index files.
    """

    def __init__(
        self,
        timeout: int = 30,
        max_retries: int = 3,
        user_agent: str = "LangGraphRAGBot/1.0",
    ):
        """
        Initialize sitemap parser.

        Args:
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            user_agent: User agent string for requests
        """
        self.timeout = timeout
        self.max_retries = max_retries
        self.user_agent = user_agent

    def _get_client(self) -> httpx.Client:
        """Create HTTP client with configured settings."""
        return httpx.Client(
            timeout=httpx.Timeout(self.timeout),
            headers={"User-Agent": self.user_agent},
            follow_redirects=True,
        )

    async def _get_async_client(self) -> httpx.AsyncClient:
        """Create async HTTP client with configured settings."""
        return httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout),
            headers={"User-Agent": self.user_agent},
            follow_redirects=True,
        )

    def _fetch_sitemap(self, url: str) -> Optional[str]:
        """Fetch sitemap content from URL."""
        with self._get_client() as client:
            for attempt in range(self.max_retries):
                try:
                    response = client.get(url)
                    response.raise_for_status()
                    return response.text
                except Exception as e:
                    logger.warning(f"Attempt {attempt + 1} failed for {url}: {e}")
                    if attempt == self.max_retries - 1:
                        logger.error(f"Failed to fetch sitemap: {url}")
                        return None
        return None

    async def _afetch_sitemap(self, url: str) -> Optional[str]:
        """Fetch sitemap content from URL (async)."""
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout),
            headers={"User-Agent": self.user_agent},
            follow_redirects=True,
        ) as client:
            for attempt in range(self.max_retries):
                try:
                    response = await client.get(url)
                    response.raise_for_status()
                    return response.text
                except Exception as e:
                    logger.warning(f"Attempt {attempt + 1} failed for {url}: {e}")
                    if attempt == self.max_retries - 1:
                        logger.error(f"Failed to fetch sitemap: {url}")
                        return None
                    await asyncio.sleep(2 ** attempt)
        return None

    def _parse_sitemap_xml(self, content: str) -> tuple[List[str], List[str]]:
        """
        Parse sitemap XML content.

        Returns:
            Tuple of (page_urls, sitemap_urls)
        """
        page_urls = []
        sitemap_urls = []

        try:
            root = ET.fromstring(content)

            # Check if this is a sitemap index
            for sitemap in root.findall(".//sm:sitemap", SITEMAP_NS):
                loc = sitemap.find("sm:loc", SITEMAP_NS)
                if loc is not None and loc.text:
                    sitemap_urls.append(loc.text.strip())

            # Also check without namespace (some sitemaps don't use it)
            for sitemap in root.findall(".//sitemap"):
                loc = sitemap.find("loc")
                if loc is not None and loc.text:
                    sitemap_urls.append(loc.text.strip())

            # Get regular URLs
            for url_elem in root.findall(".//sm:url", SITEMAP_NS):
                loc = url_elem.find("sm:loc", SITEMAP_NS)
                if loc is not None and loc.text:
                    page_urls.append(loc.text.strip())

            # Also check without namespace
            for url_elem in root.findall(".//url"):
                loc = url_elem.find("loc")
                if loc is not None and loc.text:
                    url = loc.text.strip()
                    if url not in page_urls:
                        page_urls.append(url)

        except ET.ParseError as e:
            logger.error(f"Failed to parse sitemap XML: {e}")

        return page_urls, sitemap_urls

    def parse(
        self,
        sitemap_url: str,
        filter_pattern: Optional[str] = None,
        max_urls: Optional[int] = None,
    ) -> List[str]:
        """
        Parse sitemap and extract all URLs.

        Args:
            sitemap_url: URL to the sitemap.xml
            filter_pattern: Optional pattern to filter URLs (e.g., "/docs/")
            max_urls: Maximum number of URLs to return

        Returns:
            List of URLs found in the sitemap
        """
        all_urls: Set[str] = set()
        sitemaps_to_process = [sitemap_url]
        processed_sitemaps: Set[str] = set()

        while sitemaps_to_process:
            current_sitemap = sitemaps_to_process.pop(0)

            if current_sitemap in processed_sitemaps:
                continue

            processed_sitemaps.add(current_sitemap)
            logger.info(f"Processing sitemap: {current_sitemap}")

            content = self._fetch_sitemap(current_sitemap)
            if not content:
                continue

            page_urls, sitemap_urls = self._parse_sitemap_xml(content)

            # Add nested sitemaps to process
            for sm_url in sitemap_urls:
                if sm_url not in processed_sitemaps:
                    sitemaps_to_process.append(sm_url)

            # Add page URLs
            for url in page_urls:
                if filter_pattern and filter_pattern not in url:
                    continue
                all_urls.add(url)

                if max_urls and len(all_urls) >= max_urls:
                    break

            if max_urls and len(all_urls) >= max_urls:
                break

        urls = list(all_urls)
        logger.info(f"Found {len(urls)} URLs in sitemap")
        return urls[:max_urls] if max_urls else urls

    async def aparse(
        self,
        sitemap_url: str,
        filter_pattern: Optional[str] = None,
        max_urls: Optional[int] = None,
    ) -> List[str]:
        """
        Parse sitemap and extract all URLs (async).

        Args:
            sitemap_url: URL to the sitemap.xml
            filter_pattern: Optional pattern to filter URLs
            max_urls: Maximum number of URLs to return

        Returns:
            List of URLs found in the sitemap
        """
        all_urls: Set[str] = set()
        sitemaps_to_process = [sitemap_url]
        processed_sitemaps: Set[str] = set()

        while sitemaps_to_process:
            current_sitemap = sitemaps_to_process.pop(0)

            if current_sitemap in processed_sitemaps:
                continue

            processed_sitemaps.add(current_sitemap)
            logger.info(f"Processing sitemap: {current_sitemap}")

            content = await self._afetch_sitemap(current_sitemap)
            if not content:
                continue

            page_urls, sitemap_urls = self._parse_sitemap_xml(content)

            # Add nested sitemaps to process
            for sm_url in sitemap_urls:
                if sm_url not in processed_sitemaps:
                    sitemaps_to_process.append(sm_url)

            # Add page URLs
            for url in page_urls:
                if filter_pattern and filter_pattern not in url:
                    continue
                all_urls.add(url)

                if max_urls and len(all_urls) >= max_urls:
                    break

            if max_urls and len(all_urls) >= max_urls:
                break

        urls = list(all_urls)
        logger.info(f"Found {len(urls)} URLs in sitemap")
        return urls[:max_urls] if max_urls else urls


def get_sitemap_url(base_url: str) -> str:
    """
    Get the sitemap URL for a given base URL.

    Args:
        base_url: Base URL of the website

    Returns:
        URL to sitemap.xml
    """
    parsed = urlparse(base_url)
    return f"{parsed.scheme}://{parsed.netloc}/sitemap.xml"
