"""
Content cleaner for simplifying markdown links.

Converts markdown links [text](url) to plain text while preserving code blocks.
"""

import re
from dataclasses import dataclass
from typing import List, Tuple

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ContentCleanerConfig:
    """Configuration for content cleaning."""

    simplify_links: bool = True
    preserve_code_blocks: bool = True


@dataclass
class CleaningStats:
    """Statistics from a cleaning operation."""

    original_length: int = 0
    cleaned_length: int = 0
    links_simplified: int = 0
    code_blocks_preserved: int = 0

    @property
    def reduction_percentage(self) -> float:
        """Calculate percentage of content removed."""
        if self.original_length == 0:
            return 0.0
        return (1 - self.cleaned_length / self.original_length) * 100


class ContentCleaner:
    """
    Cleans markdown content by simplifying links.

    Converts markdown links [text](url) to just the text,
    while preserving code blocks intact.
    """

    def __init__(self, config: ContentCleanerConfig = None):
        """
        Initialize content cleaner.

        Args:
            config: Cleaning configuration. Uses defaults if not provided.
        """
        self.config = config or ContentCleanerConfig()
        logger.debug(f"ContentCleaner initialized with config: {self.config}")

    def clean(self, markdown: str, collect_stats: bool = False) -> str | Tuple[str, CleaningStats]:
        """
        Clean markdown content by simplifying links.

        Args:
            markdown: Raw markdown content to clean
            collect_stats: Whether to return cleaning statistics

        Returns:
            Cleaned markdown string, or tuple of (cleaned, stats) if collect_stats=True
        """
        if not markdown:
            if collect_stats:
                return markdown, CleaningStats()
            return markdown

        stats = CleaningStats(original_length=len(markdown))
        code_blocks = []

        # Extract and protect code blocks
        if self.config.preserve_code_blocks:
            markdown, code_blocks = self._extract_code_blocks(markdown)
            stats.code_blocks_preserved = len(code_blocks)

        # Simplify links
        if self.config.simplify_links:
            markdown, link_count = self._simplify_links(markdown)
            stats.links_simplified = link_count

        # Restore code blocks
        if self.config.preserve_code_blocks:
            markdown = self._restore_code_blocks(markdown, code_blocks)

        # Light whitespace normalization (just fix excessive blank lines)
        markdown = self._normalize_whitespace(markdown)

        stats.cleaned_length = len(markdown)

        if collect_stats:
            return markdown, stats
        return markdown

    def _extract_code_blocks(self, text: str) -> Tuple[str, List[str]]:
        """
        Extract code blocks and replace with placeholders.

        Args:
            text: Markdown text

        Returns:
            Tuple of (text with placeholders, list of extracted code blocks)
        """
        code_blocks = []
        # Match fenced code blocks (``` or ~~~)
        pattern = r'(```|~~~)[\s\S]*?\1'

        def replace(match):
            code_blocks.append(match.group(0))
            return f'__CODE_BLOCK_{len(code_blocks) - 1}__'

        return re.sub(pattern, replace, text), code_blocks

    def _restore_code_blocks(self, text: str, code_blocks: List[str]) -> str:
        """
        Restore code blocks from placeholders.

        Args:
            text: Text with placeholders
            code_blocks: List of original code blocks

        Returns:
            Text with code blocks restored
        """
        for i, block in enumerate(code_blocks):
            text = text.replace(f'__CODE_BLOCK_{i}__', block)
        return text

    def _simplify_links(self, text: str) -> Tuple[str, int]:
        """
        Convert markdown links to plain text, preserving the link text.

        Converts [text](url) to just text, but keeps image links intact.

        Args:
            text: Markdown text

        Returns:
            Tuple of (text with simplified links, count of links simplified)
        """
        # Match markdown links but not images (which start with !)
        pattern = r'(?<!!)\[([^\]]+)\]\([^)]+\)'

        # Count matches before replacing
        matches = re.findall(pattern, text)
        count = len(matches)

        # Replace links with just the text
        text = re.sub(pattern, r'\1', text)

        return text, count

    def _normalize_whitespace(self, text: str) -> str:
        """
        Light whitespace normalization.

        Only reduces excessive blank lines (3+ to 2).

        Args:
            text: Input text

        Returns:
            Text with normalized whitespace
        """
        # Reduce multiple blank lines to maximum of 2
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text

    def get_stats_for_content(self, markdown: str) -> CleaningStats:
        """
        Get cleaning statistics without modifying the content.

        Args:
            markdown: Markdown content to analyze

        Returns:
            CleaningStats object with statistics
        """
        _, stats = self.clean(markdown, collect_stats=True)
        return stats


def get_content_cleaner(config: ContentCleanerConfig = None) -> ContentCleaner:
    """
    Get a ContentCleaner instance with the given configuration.

    Args:
        config: Optional configuration

    Returns:
        Configured ContentCleaner instance
    """
    return ContentCleaner(config)
