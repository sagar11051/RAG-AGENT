"""
OVH Cloud Embedding client using BGE-M3 model.

Uses OpenAI-compatible API for generating text embeddings via OVH AI Endpoints.
"""

from functools import lru_cache
from typing import List, Optional
import asyncio

import numpy as np
from openai import OpenAI, AsyncOpenAI

from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class OVHEmbeddings:
    """
    OVH Cloud Embedding client using BGE-M3 model via OpenAI-compatible API.

    Features:
    - Single text and batch embedding
    - Synchronous and asynchronous methods
    - Automatic retry with exponential backoff
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        access_token: Optional[str] = None,
        model: str = "bge-m3",
        max_retries: int = 3,
    ):
        """
        Initialize OVH Embeddings client.

        Args:
            base_url: OVH embedding base URL (default from settings)
            access_token: OVH AI Endpoints access token (default from settings)
            model: Embedding model name (default: bge-m3)
            max_retries: Maximum retry attempts for failed requests
        """
        # Use the OpenAI-compatible endpoint
        self.base_url = base_url or "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1"
        self.access_token = access_token or settings.ovh.ai_endpoints_access_token
        self.model = model
        self.dimension = settings.ovh.embedding_dimension
        self.max_retries = max_retries

        # OpenAI clients (lazy initialized)
        self._sync_client: Optional[OpenAI] = None
        self._async_client: Optional[AsyncOpenAI] = None

        logger.info(
            f"Initialized OVH Embeddings client "
            f"(model: {self.model}, dim: {self.dimension})"
        )

    def is_configured(self) -> bool:
        """Check if OVH credentials are configured."""
        return bool(self.base_url and self.access_token)

    @property
    def sync_client(self) -> OpenAI:
        """Get or create synchronous OpenAI client."""
        if self._sync_client is None:
            self._sync_client = OpenAI(
                api_key=self.access_token,
                base_url=self.base_url,
            )
        return self._sync_client

    @property
    def async_client(self) -> AsyncOpenAI:
        """Get or create asynchronous OpenAI client."""
        if self._async_client is None:
            self._async_client = AsyncOpenAI(
                api_key=self.access_token,
                base_url=self.base_url,
            )
        return self._async_client

    def _validate_response(self, embedding: List[float]) -> List[float]:
        """Validate embedding response."""
        if not embedding:
            raise ValueError("Empty embedding received from API")

        if len(embedding) != self.dimension:
            logger.warning(
                f"Embedding dimension mismatch: expected {self.dimension}, "
                f"got {len(embedding)}"
            )

        return embedding

    # =========================================================================
    # Synchronous Methods
    # =========================================================================

    def get_text_embedding(self, text: str) -> List[float]:
        """
        Get embedding for a single text (synchronous).

        Args:
            text: Text to embed

        Returns:
            Embedding vector (1024 dimensions for BGE-M3)

        Raises:
            ValueError: If text is empty or API returns invalid response
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        if not self.is_configured():
            raise ValueError(
                "OVH credentials not configured. "
                "Set OVH_AI_ENDPOINTS_ACCESS_TOKEN in .env"
            )

        for attempt in range(self.max_retries):
            try:
                response = self.sync_client.embeddings.create(
                    model=self.model,
                    input=text,
                )

                embedding = response.data[0].embedding
                return self._validate_response(embedding)

            except Exception as e:
                logger.error(f"Error on attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    raise

                # Exponential backoff
                import time
                time.sleep(2 ** attempt)

        raise RuntimeError("Failed to get embedding after retries")

    def get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for multiple texts (synchronous batch).

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        if not self.is_configured():
            raise ValueError(
                "OVH credentials not configured. "
                "Set OVH_AI_ENDPOINTS_ACCESS_TOKEN in .env"
            )

        for attempt in range(self.max_retries):
            try:
                response = self.sync_client.embeddings.create(
                    model=self.model,
                    input=texts,
                )

                # Sort by index to maintain order
                sorted_data = sorted(response.data, key=lambda x: x.index)
                embeddings = [self._validate_response(d.embedding) for d in sorted_data]
                return embeddings

            except Exception as e:
                logger.error(f"Error on attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    raise

                import time
                time.sleep(2 ** attempt)

        raise RuntimeError("Failed to get embeddings after retries")

    # =========================================================================
    # Asynchronous Methods
    # =========================================================================

    async def aget_text_embedding(self, text: str) -> List[float]:
        """
        Get embedding for a single text (asynchronous).

        Args:
            text: Text to embed

        Returns:
            Embedding vector (1024 dimensions for BGE-M3)
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        if not self.is_configured():
            raise ValueError(
                "OVH credentials not configured. "
                "Set OVH_AI_ENDPOINTS_ACCESS_TOKEN in .env"
            )

        for attempt in range(self.max_retries):
            try:
                response = await self.async_client.embeddings.create(
                    model=self.model,
                    input=text,
                )

                embedding = response.data[0].embedding
                return self._validate_response(embedding)

            except Exception as e:
                logger.error(f"Error on attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    raise

                await asyncio.sleep(2 ** attempt)

        raise RuntimeError("Failed to get embedding after retries")

    async def aget_text_embeddings(
        self,
        texts: List[str],
        batch_size: int = 10,
        show_progress: bool = False,
    ) -> List[List[float]]:
        """
        Get embeddings for multiple texts (asynchronous batch).

        Args:
            texts: List of texts to embed
            batch_size: Number of texts per API call
            show_progress: Whether to show progress

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        if not self.is_configured():
            raise ValueError(
                "OVH credentials not configured. "
                "Set OVH_AI_ENDPOINTS_ACCESS_TOKEN in .env"
            )

        embeddings = []

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            for attempt in range(self.max_retries):
                try:
                    response = await self.async_client.embeddings.create(
                        model=self.model,
                        input=batch,
                    )

                    # Sort by index to maintain order
                    sorted_data = sorted(response.data, key=lambda x: x.index)
                    batch_embeddings = [self._validate_response(d.embedding) for d in sorted_data]
                    embeddings.extend(batch_embeddings)
                    break

                except Exception as e:
                    logger.error(f"Error on attempt {attempt + 1} for batch {i}: {e}")
                    if attempt == self.max_retries - 1:
                        # Add zero vectors for failed batch
                        embeddings.extend([[0.0] * self.dimension] * len(batch))
                    else:
                        await asyncio.sleep(2 ** attempt)

            if show_progress:
                logger.info(f"Embedded {min(i + batch_size, len(texts))}/{len(texts)}")

        return embeddings

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def compute_similarity(
        self,
        embedding1: List[float],
        embedding2: List[float],
    ) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Cosine similarity score (0 to 1)
        """
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)

        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))


@lru_cache()
def get_embeddings_client() -> OVHEmbeddings:
    """Get cached OVH Embeddings client instance."""
    return OVHEmbeddings()
