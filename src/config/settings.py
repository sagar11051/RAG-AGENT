"""
Application settings and configuration management.

Uses pydantic-settings for environment variable loading and validation.
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Find and load .env file from project root
def find_dotenv() -> Optional[Path]:
    """Find .env file by walking up from current directory."""
    current = Path.cwd()

    # Check current directory first
    env_file = current / ".env"
    if env_file.exists():
        return env_file

    # Walk up to find .env
    for parent in current.parents:
        env_file = parent / ".env"
        if env_file.exists():
            return env_file
        # Also check if we find pyproject.toml (project root)
        if (parent / "pyproject.toml").exists():
            env_file = parent / ".env"
            if env_file.exists():
                return env_file
            break

    return None

# Load .env file
_env_file = find_dotenv()
if _env_file:
    load_dotenv(_env_file)


class SupabaseSettings(BaseSettings):
    """Supabase database configuration."""

    model_config = SettingsConfigDict(env_prefix="SUPABASE_", extra="ignore")

    url: str = Field(default="", description="Supabase project URL")
    key: str = Field(default="", description="Supabase anon key")
    service_key: str = Field(default="", description="Supabase service role key")


class OVHSettings(BaseSettings):
    """OVH AI Endpoints configuration for LLM and Embeddings."""

    model_config = SettingsConfigDict(env_prefix="OVH_", extra="ignore")

    # Access token for OVH AI Endpoints
    ai_endpoints_access_token: str = Field(
        default="", description="OVH AI Endpoints access token"
    )

    # Embedding model settings (BGE-M3)
    embedding_endpoint_url: str = Field(
        default="https://bge-m3.endpoints.kepler.ai.cloud.ovh.net/api/text2vec",
        description="OVH embedding endpoint URL",
    )
    embedding_dimension: int = Field(
        default=1024, description="Embedding vector dimension"
    )

    # LLM settings (Mistral-Nemo via OpenAI-compatible API)
    llm_base_url: str = Field(
        default="https://oai.endpoints.kepler.ai.cloud.ovh.net/v1",
        description="OVH LLM base URL (OpenAI-compatible)",
    )
    llm_model: str = Field(
        default="Mistral-Nemo-Instruct-2407", description="LLM model name"
    )


class CrawlerSettings(BaseSettings):
    """Web crawler configuration."""

    model_config = SettingsConfigDict(extra="ignore")

    base_url: str = Field(
        default="https://docs.langchain.com/oss/python/langgraph",
        alias="BASE_URL",
        description="Base URL to crawl",
    )
    max_pages: int = Field(default=1000, alias="MAX_PAGES")
    crawl_delay: float = Field(default=1.0, alias="CRAWL_DELAY")
    timeout: int = Field(default=30, description="Request timeout in seconds")
    retry_attempts: int = Field(default=3, description="Number of retry attempts")
    user_agent: str = Field(
        default="LangGraphRAGBot/1.0", description="User agent string"
    )


class ChunkerSettings(BaseSettings):
    """Smart chunking configuration."""

    model_config = SettingsConfigDict(extra="ignore")

    chunk_size: int = Field(
        default=512, alias="CHUNK_SIZE", description="Target chunk size in tokens"
    )
    chunk_overlap: int = Field(
        default=128, alias="CHUNK_OVERLAP", description="Overlap between chunks"
    )
    min_chunk_size: int = Field(
        default=100, alias="MIN_CHUNK_SIZE", description="Minimum chunk size"
    )
    max_chunk_size: int = Field(
        default=2048, alias="MAX_CHUNK_SIZE", description="Maximum chunk size"
    )
    respect_code_blocks: bool = Field(
        default=True, description="Keep code blocks intact"
    )
    respect_sentences: bool = Field(
        default=True, description="Split at sentence boundaries"
    )


class ContentCleanerSettings(BaseSettings):
    """Content cleaning configuration for simplifying links."""

    model_config = SettingsConfigDict(extra="ignore")

    enabled: bool = Field(
        default=True,
        alias="CONTENT_CLEANER_ENABLED",
        description="Enable content cleaning before chunking",
    )
    simplify_links: bool = Field(
        default=True, description="Convert markdown links to plain text"
    )
    preserve_code_blocks: bool = Field(
        default=True, description="Protect code blocks from cleaning"
    )


class RetrievalSettings(BaseSettings):
    """RAG retrieval configuration."""

    model_config = SettingsConfigDict(extra="ignore")

    top_k: int = Field(
        default=5, alias="TOP_K_CHUNKS", description="Number of chunks to retrieve"
    )
    semantic_weight: float = Field(
        default=0.6, description="Weight for semantic search in hybrid"
    )
    keyword_weight: float = Field(
        default=0.4, description="Weight for keyword search in hybrid"
    )
    rerank_enabled: bool = Field(
        default=False, alias="RERANK_ENABLED", description="Enable re-ranking"
    )
    reranker_model: Optional[str] = Field(
        default=None, alias="RERANKER_MODEL", description="Re-ranker model name"
    )
    reranker_endpoint: Optional[str] = Field(
        default=None, alias="RERANKER_ENDPOINT", description="Re-ranker endpoint URL"
    )


class APISettings(BaseSettings):
    """FastAPI server configuration."""

    model_config = SettingsConfigDict(env_prefix="API_", extra="ignore")

    host: str = Field(default="0.0.0.0", description="API host")
    port: int = Field(default=8000, description="API port")
    workers: int = Field(default=4, description="Number of workers")
    reload: bool = Field(default=True, description="Enable auto-reload")
    key: str = Field(
        default="change_me_in_production", description="API key for authentication"
    )
    allowed_origins: str = Field(
        default="http://localhost:3000,http://localhost:5173,http://localhost:8080",
        alias="ALLOWED_ORIGINS",
        description="Comma-separated list of allowed CORS origins",
    )

    @field_validator("allowed_origins", mode="before")
    @classmethod
    def parse_origins(cls, v):
        """Keep as string, will be split when needed."""
        return v

    def get_origins_list(self) -> List[str]:
        """Return allowed origins as a list."""
        if isinstance(self.allowed_origins, str):
            return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]
        return self.allowed_origins


class RateLimitSettings(BaseSettings):
    """Rate limiting configuration."""

    model_config = SettingsConfigDict(env_prefix="RATE_LIMIT_", extra="ignore")

    enabled: bool = Field(default=True, description="Enable rate limiting")
    per_minute: int = Field(default=100, description="Requests per minute")
    per_session: int = Field(default=20, description="Requests per session per minute")


class AgentSettings(BaseSettings):
    """LangGraph agent configuration."""

    model_config = SettingsConfigDict(extra="ignore")

    session_timeout: int = Field(
        default=3600,
        alias="SESSION_TIMEOUT",
        description="Session timeout in seconds",
    )
    max_history: int = Field(
        default=20,
        alias="MAX_HISTORY_MESSAGES",
        description="Maximum messages in history",
    )
    temperature: float = Field(
        default=0.7, alias="LLM_TEMPERATURE", description="LLM temperature"
    )
    max_tokens: int = Field(
        default=1024, alias="LLM_MAX_TOKENS", description="Max tokens for response"
    )


class LangSmithSettings(BaseSettings):
    """LangSmith tracing configuration."""

    model_config = SettingsConfigDict(extra="ignore")

    api_key: str = Field(
        default="",
        alias="LANGSMITH_API_KEY",
        description="LangSmith API key",
    )
    project: str = Field(
        default="langgraph-rag-agent",
        alias="LANGSMITH_PROJECT",
        description="LangSmith project name",
    )
    tracing_enabled: bool = Field(
        default=False,
        alias="LANGCHAIN_TRACING_V2",
        description="Enable LangSmith tracing",
    )
    endpoint: str = Field(
        default="https://api.smith.langchain.com",
        alias="LANGCHAIN_ENDPOINT",
        description="LangSmith API endpoint",
    )

    @property
    def is_configured(self) -> bool:
        """Check if LangSmith is properly configured."""
        return bool(self.api_key and self.tracing_enabled)


class Settings(BaseSettings):
    """Main application settings aggregating all sub-settings."""

    model_config = SettingsConfigDict(
        extra="ignore",
    )

    # Nested settings - initialized after dotenv is loaded
    supabase: SupabaseSettings = Field(default_factory=SupabaseSettings)
    ovh: OVHSettings = Field(default_factory=OVHSettings)
    crawler: CrawlerSettings = Field(default_factory=CrawlerSettings)
    chunker: ChunkerSettings = Field(default_factory=ChunkerSettings)
    content_cleaner: ContentCleanerSettings = Field(default_factory=ContentCleanerSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    api: APISettings = Field(default_factory=APISettings)
    rate_limit: RateLimitSettings = Field(default_factory=RateLimitSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    langsmith: LangSmithSettings = Field(default_factory=LangSmithSettings)

    # Application settings
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    data_dir: str = Field(default="./data", alias="DATA_DIR")


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Global settings instance
settings = get_settings()
