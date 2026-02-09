# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LangGraph Documentation RAG Agent - A production-ready Retrieval-Augmented Generation system for documentation Q&A. Crawls LangGraph documentation, processes into smart chunks, stores in Supabase with pgvector, and uses LangGraph for conversational retrieval with a hybrid routing architecture.

## Commands

```bash
# Install dependencies (using uv package manager)
uv sync

# Install Playwright browsers (required for Crawl4AI)
uv run playwright install chromium

# Run LangGraph Studio (development server with hot reload)
uv run langgraph dev

# Run phase tests (incremental verification)
uv run python scripts/test_phase1_config.py      # Configuration
uv run python scripts/test_phase2_database.py   # Supabase/pgvector
uv run python scripts/test_phase3_embeddings.py # OVH BGE-M3 embeddings
uv run python scripts/test_phase4_llm.py        # OVH Mistral-Nemo LLM
uv run python scripts/test_phase5_crawler.py    # Web crawler
uv run python scripts/test_phase7_retrieval.py  # RAG Retrieval
uv run python scripts/test_phase7b_retrievers.py # Multi-strategy retrievers
uv run python scripts/test_phase8_agent.py      # LangGraph Agent

# Run pytest
uv run pytest tests/

# Interactive chat agent (CLI)
uv run python scripts/chat_agent.py

# Start FastAPI server
uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

## Architecture

### Data Pipeline Flow
```
Sitemap → Web Crawler → Document Store → Smart Chunker → Embeddings → Supabase (pgvector)
                                              ↓
User Query → Embed → Vector Search → Retrieved Chunks → LLM → Response
```

### LangGraph Agent Architecture

The agent uses a hybrid routing architecture based on query complexity:

```
parallel_init → [complexity_router]
                      │
      ┌───────────────┼───────────────┐
      ▼               ▼               ▼
  [greeting]     [simple_rag]   [complex_rag]
      │               │               │
      ▼               ▼               ▼
 direct_llm      retrieve →      tool_agent
      │          generate        (ReAct)
      └───────────┬───────────────────┘
                  ▼
              [persist]
```

**Query Routes:**
| Route | Trigger | Behavior |
|-------|---------|----------|
| `greeting` | Greetings, thanks, confirmations | Direct LLM response, no retrieval |
| `simple` | Standard documentation queries | Fast node-based retrieval + generation |
| `complex` | Multi-hop, comparison, tutorial queries | ReAct tool agent with query decomposition |

**Agent Files:**
- `src/agent/graph.py`: LangGraph definition with hybrid routing (`get_default_graph`)
- `src/agent/nodes.py`: Node implementations (parallel_init, retrieve_chunks, response_generation)
- `src/agent/react_agent.py`: ReAct tool agent with search_documentation, get_full_document tools
- `src/agent/tools.py`: Query classification (`classify_query_complexity`, `analyze_query_fast`)
- `src/agent/state.py`: AgentState TypedDict with `messages: Annotated[List[AnyMessage], add_messages]`

### Two-Level Storage Strategy
- **Level 1 (documents table)**: Full page content for backtracking retrieval
- **Level 2 (chunks table)**: Smart chunks (512 tokens, 128 overlap) with 1024-dim embeddings + tsvector for FTS

### Multi-Strategy Retrieval
| Strategy | Description | Best For |
|----------|-------------|----------|
| Semantic | Vector similarity (cosine) | Conceptual/semantic queries |
| BM25 | Full-text search (tsvector) | Exact terms, function names |
| Hybrid | RRF fusion (k=60) of both | General queries (default) |

### Key Integrations
| Component | Service | Config Key |
|-----------|---------|------------|
| Vector DB | Supabase + pgvector | `SUPABASE_URL`, `SUPABASE_KEY` |
| Embeddings | OVH BGE-M3 (1024 dims) | `OVH_AI_ENDPOINTS_ACCESS_TOKEN` |
| LLM | OVH Mistral-Nemo-Instruct-2407 | `OVH_LLM_BASE_URL` |
| Web Crawler | Crawl4AI (Playwright) | Settings in `src/config/settings.py` |

### Module Dependencies
```
src/config/settings.py     # All modules depend on this (Pydantic settings with .env loading)
src/utils/logger.py        # Logging with Windows Unicode support
src/database/              # supabase_client.py wraps Supabase, operations.py for high-level CRUD
src/embeddings/            # ovh_embeddings.py uses OpenAI SDK with OVH endpoint
src/llm/                   # ovh_llm.py uses OpenAI SDK with OVH endpoint, get_chat_model() for LangChain
src/crawler/               # web_crawler.py (Crawl4AI), sitemap_parser.py, document_store.py (JSON)
src/retrieval/             # Multi-strategy retrievers with RRF fusion
src/agent/                 # LangGraph agent with hybrid routing
```

## Database Schema

Tables defined in `src/database/schema.sql` + `schema_fts.sql`:
- `documents`: Full pages (url UNIQUE, content_hash, total_tokens)
- `chunks`: Smart chunks with embeddings (VECTOR(1024), IVFFlat index, tsvector for FTS)
- `sessions`: Chat sessions (thread_id UUID)
- `messages`: Conversation history (role: user/assistant/system)

Key functions:
- `match_chunks(query_embedding, match_count, filter_code)` - Semantic search
- `match_chunks_fts(search_query, match_count, filter_code)` - BM25 search
- `match_chunks_hybrid(...)` - Returns both scores for RRF fusion

## Retrieval System

### Retriever Types
```python
from src.retrieval import RAGRetriever, RetrieverType

# Semantic only (vector similarity)
retriever = RAGRetriever(retriever_type=RetrieverType.SEMANTIC)

# BM25 only (exact match)
retriever = RAGRetriever(retriever_type=RetrieverType.BM25)

# Hybrid with RRF (recommended default)
retriever = RAGRetriever(retriever_type=RetrieverType.HYBRID)
```

### Hybrid Search with RRF
The hybrid retriever uses Reciprocal Rank Fusion with k=60:
```
RRF_score = semantic_weight * (1/(60 + semantic_rank)) + bm25_weight * (1/(60 + bm25_rank))
```
Default weights: semantic=0.6, bm25=0.4

### Parent Document Backtracking Strategy

The agent should use backtracking to fetch full parent documents when:

1. **REQUIRED** (always backtrack):
   - Chunk contains "..." (truncated content)
   - Chunk mentions "see full example" or "complete example"

2. **RECOMMENDED** (usually backtrack):
   - Chunk references "see below", "as shown above"
   - Chunk is part of multi-step tutorial
   - Code blocks have unbalanced brackets (incomplete)

3. **SUGGESTED** (consider backtracking):
   - Chunk contains "tutorial", "guide", "walkthrough"
   - Multiple code blocks in chunk

4. **NONE** (self-contained):
   - Chunk is complete and self-contained

```python
# Agent usage pattern
result = retriever.retrieve("how to add memory?")

if result.needs_backtracking:
    for chunk_id in result.chunks_needing_backtracking:
        full_doc = retriever.get_full_document(chunk_id)
        # Use full_doc.content for complete context
```

## Configuration

All settings flow through `src/config/settings.py` using Pydantic:
- Automatically loads `.env` from project root via `python-dotenv`
- Nested settings classes: `SupabaseSettings`, `OVHSettings`, `CrawlerSettings`, `ChunkerSettings`, `RetrievalSettings`
- Access via singleton: `from src.config.settings import settings`

### Retrieval Settings
```python
settings.retrieval.top_k           # Default: 5 chunks
settings.retrieval.semantic_weight # Default: 0.6
settings.retrieval.keyword_weight  # Default: 0.4 (BM25)
```

## LangGraph Studio

The agent is configured for LangGraph Studio in `langgraph.json`:
```json
{
  "graphs": {
    "rag_agent": "./src/agent/graph.py:get_default_graph"
  }
}
```

Run `uv run langgraph dev` to start the development server:
- API: http://127.0.0.1:2024
- Studio UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
- API Docs: http://127.0.0.1:2024/docs

The Studio Chat interface requires the `messages` state key with `add_messages` reducer (configured in `state.py`).

## Windows Development Notes

- Crawl4AI console output can cause Unicode encoding errors; test scripts include encoding fixes
- Logger uses `SafeStreamHandler` to handle encoding issues
- Use `uv run` prefix for all Python commands to ensure correct environment

## FTS Setup

To enable BM25/Hybrid retrieval, run the FTS schema in Supabase SQL Editor:
```sql
-- Execute contents of src/database/schema_fts.sql
```
This adds the `content_tsv` column and full-text search functions.
