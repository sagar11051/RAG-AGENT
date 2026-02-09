# Product Requirements Document: LangGraph Documentation RAG Agent

## 1. Project Overview

### 1.1 Purpose
Build an intelligent RAG (Retrieval-Augmented Generation) agent that crawls documentation websites, processes and stores content intelligently, and provides interactive question-answering capabilities through a LangGraph-based conversational agent.

### 1.2 Target Use Case
- Primary Example: LangChain documentation (https://docs.langchain.com/oss/python/langgraph)
- Extensible to any documentation website with a sitemap

### 1.3 Key Objectives
- Automated documentation crawling and ingestion
- Intelligent chunking that respects code blocks and semantic boundaries
- Multi-strategy retrieval with backtracking capabilities
- Session-based conversational interface supporting multiple users
- Scalable storage with fast similarity search

## 2. System Architecture

### 2.1 High-Level Components
```
┌─────────────────┐
│  Web Crawler    │
│   (Crawl4AI)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Smart Chunker   │
│  (2-Level)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Embedding Model │
│   (Custom)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Supabase DB    │
│  (Vector Store) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  RAG Tool       │
│ (Multi-Strategy)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ LangGraph Agent │
│ (Session-Based) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  FastAPI Server │
│  (REST/SSE)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   UI Client     │
│ (Web/Mobile)    │
└─────────────────┘
```

## 3. Functional Requirements

### 3.1 Web Crawling Module

#### 3.1.1 Crawl4AI Integration
- **Input**: Base URL (e.g., https://docs.langchain.com/oss/python/langgraph)
- **Process**:
  - Fetch sitemap.xml from the domain
  - Extract all URLs from sitemap
  - Crawl each URL using Crawl4AI
  - Extract clean text content, code blocks, and metadata
- **Output**: Raw documents with metadata (URL, title, timestamp)

#### 3.1.2 Storage Requirements
- Store raw documents locally before processing
- Format: JSON or structured format with metadata
- Include: URL, title, content, crawl timestamp, content hash

### 3.2 Smart Chunking Module

#### 3.2.1 Two-Level Chunking Strategy

**Level 1: URL-Level Document**
- Store entire page content as a single document
- Purpose: Enable backtracking retrieval
- Metadata: URL, title, total_tokens, crawl_date

**Level 2: Smart Chunks**
- Chunk size: Configurable (default: 512-1024 tokens)
- Overlap: Configurable (default: 128 tokens)

#### 3.2.2 Code Block Preservation
- Detect code blocks using delimiters:
  - Triple backticks (```)
  - Triple quotes (''' or """)
  - Indented code blocks
- **Rules**:
  - Never split code blocks across chunks
  - If code block > max chunk size, keep it as a single chunk
  - Preserve syntax highlighting language identifiers

#### 3.2.3 Sentence Boundary Respect
- Use sentence tokenization (NLTK, spaCy, or custom)
- Only split at sentence boundaries (., !, ?)
- Avoid splitting mid-sentence
- Handle edge cases: abbreviations (Dr., Mr.), decimals (3.14)

#### 3.2.4 Chunk Metadata
Each chunk must include:
- `chunk_id`: Unique identifier
- `parent_doc_id`: Reference to Level 1 document
- `url`: Source URL
- `chunk_index`: Position in document (0, 1, 2, ...)
- `content`: The chunk text
- `tokens`: Token count
- `has_code`: Boolean flag
- `section_title`: Extracted from headers (if available)
- `previous_chunk_id`: For sequential context
- `next_chunk_id`: For sequential context

### 3.3 Embedding Module

#### 3.3.1 Custom Embedding Model
- Support for custom embedding model specification
- Model will be provided via configuration
- Requirements:
  - Input: Text string
  - Output: Vector embedding (dimension: TBD based on model)

#### 3.3.2 Batch Processing
- Process chunks in batches for efficiency
- Configurable batch size (default: 32)
- Error handling and retry logic

### 3.4 Supabase Database

#### 3.4.1 Schema Design

**Table 1: `documents` (Level 1)**
```sql
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    url TEXT UNIQUE NOT NULL,
    title TEXT,
    content TEXT NOT NULL,
    content_hash TEXT UNIQUE,
    total_tokens INTEGER,
    crawl_date TIMESTAMP DEFAULT NOW(),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_documents_url ON documents(url);
CREATE INDEX idx_documents_content_hash ON documents(content_hash);
```

**Table 2: `chunks` (Level 2)**
```sql
CREATE TABLE chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    parent_doc_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding VECTOR(dimension_tbd), -- dimension based on embedding model
    tokens INTEGER,
    has_code BOOLEAN DEFAULT FALSE,
    section_title TEXT,
    previous_chunk_id UUID REFERENCES chunks(id),
    next_chunk_id UUID REFERENCES chunks(id),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    
    UNIQUE(parent_doc_id, chunk_index)
);

-- Indexes for fast retrieval
CREATE INDEX idx_chunks_parent_doc ON chunks(parent_doc_id);
CREATE INDEX idx_chunks_has_code ON chunks(has_code);

-- Vector similarity search index (ivfflat or hnsw)
CREATE INDEX idx_chunks_embedding ON chunks 
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Alternative: HNSW index for better performance
-- CREATE INDEX idx_chunks_embedding_hnsw ON chunks 
-- USING hnsw (embedding vector_cosine_ops);
```

**Table 3: `sessions` (for multi-user support)**
```sql
CREATE TABLE sessions (
    thread_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    metadata JSONB
);

CREATE INDEX idx_sessions_user_id ON sessions(user_id);
```

**Table 4: `messages` (conversation history)**
```sql
CREATE TABLE messages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    thread_id UUID REFERENCES sessions(thread_id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    metadata JSONB
);

CREATE INDEX idx_messages_thread_id ON messages(thread_id);
CREATE INDEX idx_messages_created_at ON messages(created_at);
```

#### 3.4.2 pgvector Extension
- Enable pgvector extension for vector similarity search
- Support for cosine similarity, L2 distance, inner product

### 3.5 Multi-Strategy RAG Tool

#### 3.5.1 Retrieval Strategies

**Strategy 1: Semantic Search**
- Use vector similarity search on embeddings
- Return top-k most similar chunks (k = 5-10)
- Use cosine similarity metric

**Strategy 2: Keyword Search**
- Full-text search using PostgreSQL's built-in capabilities
- Support for partial matching and ranking
- Use `tsvector` and `tsquery` for efficient search

**Strategy 3: Hybrid Search**
- Combine semantic + keyword results
- Apply weighted scoring (configurable weights)
- Re-rank using custom re-ranker model (1/60 ratio mentioned)

**Strategy 4: Backtracking Retrieval**
- When chunk context is insufficient, retrieve full parent document
- Triggered based on confidence scores or explicit agent decision
- Return Level 1 document content

#### 3.5.2 Strategy Selection
- Agent dynamically selects strategy based on query type
- Heuristics:
  - Code-related queries → Prefer chunks with `has_code=true`
  - Specific API names → Keyword search
  - Conceptual questions → Semantic search
  - Complex multi-part questions → Hybrid search
  - Follow-up questions → Backtracking if needed

#### 3.5.3 Re-ranker Integration
- Post-retrieval re-ranking (1/60 ratio)
- Custom re-ranker model (to be specified)
- Input: query + retrieved chunks
- Output: Re-ordered chunks by relevance

### 3.6 LangGraph Agent

#### 3.6.1 Node Definitions

**Node 1: Input Processing**
- Receives user message
- Loads conversation history from database
- Prepares context for agent

**Node 2: Query Analysis**
- Analyzes user query
- Determines retrieval strategy
- Extracts key entities/topics

**Node 3: RAG Tool Node**
- Executes selected retrieval strategy
- Fetches relevant chunks/documents
- Applies re-ranking if needed

**Node 4: Response Generation**
- Uses custom LLM (to be specified)
- Generates response based on retrieved context
- Incorporates conversation history

**Node 5: Output Processing**
- Formats response for user
- Saves to conversation history
- Updates session metadata

#### 3.6.2 Graph Structure
```python
# Conceptual flow
START → Input Processing → Query Analysis → RAG Tool → Response Generation → Output Processing → END
                                  ↓
                           (Conditional edge based on query type)
                                  ↓
                           Backtracking retrieval (optional)
```

#### 3.6.3 State Management
- Use LangGraph's built-in state management
- State schema:
  - `messages`: List of conversation messages
  - `thread_id`: Session identifier
  - `retrieved_chunks`: Current retrieval results
  - `query_strategy`: Selected retrieval strategy
  - `metadata`: Additional context

#### 3.6.4 Session & Thread Management
- Each user gets unique `thread_id`
- Sessions stored in Supabase `sessions` table
- Messages stored in `messages` table
- Support for:
  - Creating new sessions
  - Loading existing sessions
  - Concurrent sessions for multiple users
  - Session timeout and cleanup

### 3.7 Custom LLM Integration
- Placeholder for custom LLM configuration
- Requirements:
  - Standard interface for text generation
  - Support for chat format (system, user, assistant)
  - Streaming support (optional but recommended)
- Will be specified by user in implementation phase

### 3.8 FastAPI REST API

#### 3.8.1 Purpose
- Expose agent functionality via HTTP endpoints
- Enable UI consumption
- Support web and mobile clients
- Enable external integrations

#### 3.8.2 Core Endpoints

**Chat Endpoints:**

`POST /api/v1/chat/sessions`
- Create new chat session
- Request body: `{ "user_id": "optional_user_id" }`
- Response: `{ "thread_id": "uuid", "created_at": "timestamp" }`

`POST /api/v1/chat/sessions/{thread_id}/messages`
- Send message to agent
- Request body: `{ "message": "user query", "stream": false }`
- Response: `{ "response": "agent response", "metadata": {...} }`

`POST /api/v1/chat/sessions/{thread_id}/messages/stream`
- Send message with streaming response
- Request body: `{ "message": "user query" }`
- Response: Server-Sent Events (SSE) stream

`GET /api/v1/chat/sessions/{thread_id}/messages`
- Get conversation history
- Query params: `?limit=20&offset=0`
- Response: `{ "messages": [...], "total": 50 }`

`GET /api/v1/chat/sessions/{thread_id}`
- Get session details
- Response: `{ "thread_id": "uuid", "user_id": "...", "created_at": "...", "updated_at": "..." }`

`DELETE /api/v1/chat/sessions/{thread_id}`
- Delete session and all messages
- Response: `{ "deleted": true }`

**Search/Retrieval Endpoints (Optional - for debugging/testing):**

`POST /api/v1/search/semantic`
- Direct semantic search
- Request: `{ "query": "search query", "top_k": 5 }`
- Response: `{ "results": [...] }`

`POST /api/v1/search/hybrid`
- Hybrid search with re-ranking
- Request: `{ "query": "search query", "top_k": 5 }`
- Response: `{ "results": [...] }`

**Health & Status:**

`GET /api/v1/health`
- Health check endpoint
- Response: `{ "status": "healthy", "version": "1.0.0", "database": "connected" }`

`GET /api/v1/status`
- System status
- Response: `{ "documents_count": 1000, "chunks_count": 50000, "active_sessions": 5 }`

#### 3.8.3 Request/Response Schemas

**Message Request:**
```json
{
  "message": "How do I create a StateGraph?",
  "stream": false,
  "temperature": 0.7,
  "max_tokens": 1024
}
```

**Message Response:**
```json
{
  "message_id": "uuid",
  "thread_id": "uuid",
  "role": "assistant",
  "content": "To create a StateGraph...",
  "metadata": {
    "retrieval_strategy": "semantic",
    "chunks_retrieved": 5,
    "processing_time_ms": 1234,
    "sources": [
      {
        "url": "https://docs.langchain.com/...",
        "title": "Creating Graphs"
      }
    ]
  },
  "created_at": "2026-01-18T10:30:00Z"
}
```

**Streaming Response (SSE):**
```
event: token
data: {"content": "To ", "delta": "To "}

event: token
data: {"content": "To create ", "delta": "create "}

event: metadata
data: {"retrieval_strategy": "semantic", "chunks_retrieved": 5}

event: done
data: {"message_id": "uuid", "processing_time_ms": 1234}
```

**Conversation History Response:**
```json
{
  "thread_id": "uuid",
  "messages": [
    {
      "message_id": "uuid",
      "role": "user",
      "content": "How do I create a StateGraph?",
      "created_at": "2026-01-18T10:30:00Z"
    },
    {
      "message_id": "uuid",
      "role": "assistant",
      "content": "To create a StateGraph...",
      "created_at": "2026-01-18T10:30:15Z"
    }
  ],
  "total": 2,
  "limit": 20,
  "offset": 0
}
```

#### 3.8.4 Error Handling

**Standard Error Response:**
```json
{
  "error": {
    "code": "INVALID_REQUEST",
    "message": "Message cannot be empty",
    "details": {}
  },
  "timestamp": "2026-01-18T10:30:00Z"
}
```

**Error Codes:**
- `INVALID_REQUEST` (400): Invalid request parameters
- `SESSION_NOT_FOUND` (404): Thread ID doesn't exist
- `SESSION_EXPIRED` (410): Session has expired
- `RATE_LIMIT_EXCEEDED` (429): Too many requests
- `INTERNAL_ERROR` (500): Server error
- `DATABASE_ERROR` (503): Database unavailable

#### 3.8.5 Authentication & Security

**API Key Authentication:**
- Header: `X-API-Key: your_api_key`
- Validate on each request
- Rate limiting per API key

**CORS Configuration:**
- Allow configurable origins
- Support credentials
- Handle preflight requests

**Rate Limiting:**
- Per API key: 100 requests/minute
- Per session: 20 messages/minute
- Configurable limits

#### 3.8.6 WebSocket Support (Optional Enhancement)

`WS /api/v1/chat/sessions/{thread_id}/ws`
- Real-time bidirectional communication
- Message format:
```json
{
  "type": "message",
  "content": "user query"
}
```
- Response format:
```json
{
  "type": "response",
  "content": "agent response",
  "metadata": {}
}
```

## 4. Non-Functional Requirements

### 4.1 Performance
- Crawling: Process 100+ pages in reasonable time (~10-30 min)
- Chunking: Process 1000 chunks in < 5 minutes
- Embedding: Batch processing at ~100 chunks/second (model-dependent)
- Retrieval: Query response time < 2 seconds
- Agent response: Total latency < 5 seconds

### 4.2 Scalability
- Support 10,000+ documents
- Support 100,000+ chunks
- Handle 10+ concurrent user sessions
- Efficient indexing for fast similarity search

### 4.3 Reliability
- Retry logic for crawling failures
- Error handling for embedding failures
- Database connection pooling
- Graceful degradation if retrieval fails

### 4.4 Maintainability
- Modular code structure
- Configuration-driven (YAML/JSON config files)
- Comprehensive logging
- Clear separation of concerns

## 5. Technology Stack

### 5.1 Core Technologies
- **Crawling**: Crawl4AI
- **Chunking**: Custom implementation with NLP libraries (NLTK/spaCy)
- **Embedding**: Custom model (TBD)
- **Vector DB**: Supabase with pgvector
- **Agent Framework**: LangGraph
- **LLM**: Custom (TBD)
- **API Framework**: FastAPI
- **Language**: Python 3.10+

### 5.2 Key Libraries
- `crawl4ai`: Web crawling
- `langraph`: Agent orchestration
- `langchain`: Supporting utilities
- `supabase-py`: Database client
- `pgvector`: Vector operations
- `nltk` or `spacy`: Text processing
- `asyncio`: Async operations
- `pydantic`: Data validation
- `python-dotenv`: Environment management
- `fastapi`: REST API framework
- `uvicorn`: ASGI server
- `sse-starlette`: Server-Sent Events for streaming
- `python-multipart`: File upload support
- `slowapi`: Rate limiting

## 6. Configuration

### 6.1 Environment Variables
```
# Supabase
SUPABASE_URL=
SUPABASE_KEY=

# Custom Embedding Model
EMBEDDING_MODEL_NAME=
EMBEDDING_MODEL_ENDPOINT=
EMBEDDING_DIMENSION=

# Custom LLM
LLM_MODEL_NAME=
LLM_ENDPOINT=
LLM_API_KEY=

# Crawling
BASE_URL=https://docs.langchain.com/oss/python/langgraph
MAX_PAGES=1000

# Chunking
CHUNK_SIZE=512
CHUNK_OVERLAP=128

# Retrieval
TOP_K_CHUNKS=5
RERANK_ENABLED=true
RERANK_MODEL=

# FastAPI
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
API_KEY=your_secret_api_key
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:5173
RATE_LIMIT_PER_MINUTE=100
```

### 6.2 Config File Structure
```yaml
crawler:
  base_url: "https://docs.langchain.com/oss/python/langgraph"
  max_pages: 1000
  timeout: 30
  retry_attempts: 3

chunker:
  chunk_size: 512
  chunk_overlap: 128
  respect_code_blocks: true
  respect_sentences: true
  min_chunk_size: 100

embedding:
  model_name: "custom-embedding-model"
  dimension: 768
  batch_size: 32

database:
  table_documents: "documents"
  table_chunks: "chunks"
  table_sessions: "sessions"
  table_messages: "messages"
  vector_index_type: "ivfflat"  # or "hnsw"

retrieval:
  top_k: 5
  strategies:
    - semantic
    - keyword
    - hybrid
  reranker:
    enabled: true
    model: "custom-reranker"
    ratio: 0.0167  # 1/60

agent:
  session_timeout: 3600  # seconds
  max_history: 20  # messages

api:
  host: "0.0.0.0"
  port: 8000
  workers: 4
  reload: false  # Set to true for development
  cors:
    enabled: true
    origins:
      - "http://localhost:3000"
      - "http://localhost:5173"
    allow_credentials: true
    allow_methods: ["*"]
    allow_headers: ["*"]
  rate_limit:
    enabled: true
    requests_per_minute: 100
    requests_per_session: 20
  authentication:
    enabled: true
    api_key_header: "X-API-Key"
  streaming:
    enabled: true
    chunk_size: 10  # tokens per chunk
```

## 7. Implementation Phases

### Phase 1: Infrastructure Setup
- Set up Supabase database
- Create tables and indexes
- Configure pgvector extension

### Phase 2: Crawling & Storage
- Implement Crawl4AI integration
- Build sitemap parser
- Store raw documents locally and in DB

### Phase 3: Chunking Pipeline
- Implement smart chunking logic
- Add code block detection
- Add sentence boundary detection
- Generate two-level chunks

### Phase 4: Embedding & Indexing
- Integrate custom embedding model
- Batch process chunks
- Store embeddings in Supabase

### Phase 5: RAG Tool Development
- Implement semantic search
- Implement keyword search
- Implement hybrid search with re-ranker
- Add backtracking retrieval

### Phase 6: LangGraph Agent
- Define graph nodes
- Implement state management
- Add session/thread management
- Integrate RAG tool

### Phase 7: FastAPI REST API
- Define Pydantic models for requests/responses
- Implement chat endpoints
- Add streaming support with SSE
- Implement session management endpoints
- Add authentication middleware
- Configure CORS
- Add rate limiting
- Implement error handling

### Phase 8: Custom LLM Integration
- Connect custom LLM
- Implement chat interface
- Add streaming support (if needed)

### Phase 8: Custom LLM Integration
- Connect custom LLM
- Implement chat interface
- Add streaming support (if needed)

### Phase 9: Testing & Optimization
- Unit tests for each component
- Integration tests
- Performance optimization
- Index tuning

## 8. Success Metrics

### 8.1 Crawling
- Successfully crawl 95%+ of sitemap URLs
- Handle rate limiting gracefully
- Detect and skip duplicate content

### 8.2 Chunking
- Zero code blocks split across chunks
- 95%+ sentence boundaries respected
- Appropriate chunk size distribution

### 8.3 Retrieval
- Top-3 accuracy > 80% for test queries
- Average query time < 2 seconds
- Re-ranker improves top-1 accuracy by 10%+

### 8.4 Agent
- Response relevance score > 4/5 (human eval)
- Session management works for 10+ concurrent users
- Zero data leakage between sessions

## 9. Future Enhancements

### 9.1 Potential Improvements
- Multi-language support
- Incremental updates (re-crawl only changed pages)
- Advanced query expansion
- Conversation summarization
- User feedback loop for continuous improvement
- Support for images/diagrams in documentation
- Export conversation history
- Analytics dashboard

### 9.2 Advanced Features
- Multi-modal retrieval (text + images)
- Automatic question generation from docs
- Citation/source tracking in responses
- Collaborative sessions (multiple users in same thread)
- Custom filters (date range, specific sections)

## 10. Risks & Mitigations

### 10.1 Risks
1. **Crawling failures**: Sites may block bots or have anti-scraping measures
   - *Mitigation*: Respect robots.txt, add delays, use user-agent rotation

2. **Embedding quality**: Custom model may not perform well on technical docs
   - *Mitigation*: Test on sample dataset first, have fallback to proven models

3. **Database performance**: Large vector operations may be slow
   - *Mitigation*: Proper indexing, consider approximate nearest neighbor algorithms

4. **Session state management**: Complex state in LangGraph may lead to bugs
   - *Mitigation*: Comprehensive testing, use LangGraph's built-in checkpointing

5. **Cost**: Embedding and LLM inference may be expensive at scale
   - *Mitigation*: Implement caching, batch processing, rate limiting

## 11. Appendix

### 11.1 Example Queries
- "How do I create a StateGraph in LangGraph?"
- "Show me an example of adding nodes to a graph"
- "What's the difference between StateGraph and MessageGraph?"
- "How do I implement human-in-the-loop with LangGraph?"
- "Explain the checkpointing mechanism"

### 11.2 Sample Chunk Metadata
```json
{
  "chunk_id": "uuid-1234",
  "parent_doc_id": "uuid-5678",
  "url": "https://docs.langchain.com/langgraph/concepts/graphs",
  "chunk_index": 2,
  "content": "A StateGraph is the main graph class...",
  "tokens": 156,
  "has_code": true,
  "section_title": "Creating a StateGraph",
  "previous_chunk_id": "uuid-1233",
  "next_chunk_id": "uuid-1235"
}
```

### 11.3 References
- LangGraph Documentation: https://langchain-ai.github.io/langgraph/
- Crawl4AI: https://github.com/unclecode/crawl4ai
- Supabase pgvector: https://supabase.com/docs/guides/database/extensions/pgvector
- LangGraph Tutorials: https://langchain-ai.github.io/langgraph/tutorials/

---

**Document Version**: 1.0  
**Last Updated**: January 18, 2026  
**Author**: Product Requirements  
**Status**: Draft for Implementation