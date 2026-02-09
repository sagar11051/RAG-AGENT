# Claude Implementation Instructions: LangGraph Documentation RAG Agent

## Project Context

You are building a production-ready RAG (Retrieval-Augmented Generation) agent system with the following components:

1. **Web Crawler**: Uses Crawl4AI to crawl documentation websites via sitemap
2. **Smart Chunker**: Two-level chunking strategy respecting code blocks and sentence boundaries
3. **Embedding System**: Custom embedding model integration
4. **Vector Database**: Supabase with pgvector for efficient similarity search
5. **Multi-Strategy RAG Tool**: Semantic, keyword, hybrid, and backtracking retrieval
6. **LangGraph Agent**: Session-based conversational agent with multi-user support
7. **Custom LLM**: User-provided LLM for response generation

**Primary Documentation Target**: https://docs.langchain.com/oss/python/langgraph

Refer to `PRD.md` for complete product requirements.

---

## Project Structure

Create the following directory structure:

```
langgraph-rag-agent/
├── .env                          # Environment variables
├── .env.example                  # Example environment file
├── config.yaml                   # Application configuration
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
├── PRD.md                        # Product requirements document
│
├── src/
│   ├── __init__.py
│   │
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py          # Configuration management
│   │
│   ├── crawler/
│   │   ├── __init__.py
│   │   ├── sitemap_parser.py    # Parse sitemap.xml
│   │   ├── web_crawler.py       # Crawl4AI integration
│   │   └── document_store.py    # Local document storage
│   │
│   ├── chunker/
│   │   ├── __init__.py
│   │   ├── code_detector.py     # Detect code blocks
│   │   ├── sentence_splitter.py # Sentence boundary detection
│   │   └── smart_chunker.py     # Main chunking logic
│   │
│   ├── embeddings/
│   │   ├── __init__.py
│   │   ├── custom_embedder.py   # Custom embedding model wrapper
│   │   └── batch_processor.py   # Batch embedding processing
│   │
│   ├── database/
│   │   ├── __init__.py
│   │   ├── supabase_client.py   # Supabase connection
│   │   ├── schema.sql            # Database schema
│   │   └── operations.py         # CRUD operations
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── semantic_search.py   # Vector similarity search
│   │   ├── keyword_search.py    # Full-text search
│   │   ├── hybrid_search.py     # Combined search with reranker
│   │   ├── backtracking.py      # Parent document retrieval
│   │   └── rag_tool.py          # Unified RAG tool interface
│   │
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── nodes.py             # LangGraph node definitions
│   │   ├── graph.py             # Graph construction
│   │   ├── state.py             # State schema definition
│   │   └── session_manager.py   # Session/thread management
│   │
│   ├── llm/
│   │   ├── __init__.py
│   │   └── custom_llm.py        # Custom LLM wrapper
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py              # FastAPI application
│   │   ├── dependencies.py      # Dependency injection
│   │   ├── middleware.py        # Auth, CORS, rate limiting
│   │   │
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── requests.py      # Request schemas
│   │   │   └── responses.py     # Response schemas
│   │   │
│   │   └── routes/
│   │       ├── __init__.py
│   │       ├── chat.py          # Chat endpoints
│   │       ├── search.py        # Search endpoints (optional)
│   │       └── health.py        # Health check endpoints
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logger.py            # Logging configuration
│       └── helpers.py           # Utility functions
│
├── scripts/
│   ├── setup_database.py        # Initialize Supabase schema
│   ├── crawl_and_ingest.py      # Run full crawl pipeline
│   ├── test_retrieval.py        # Test retrieval strategies
│   └── run_agent.py             # Start interactive agent
│
├── tests/
│   ├── __init__.py
│   ├── test_crawler.py
│   ├── test_chunker.py
│   ├── test_embeddings.py
│   ├── test_retrieval.py
│   └── test_agent.py
│
└── data/
    ├── raw_documents/           # Crawled HTML/text
    ├── processed_chunks/        # Chunked documents
    └── logs/                    # Application logs
```

---

## Implementation Guidelines

### Phase 1: Setup & Infrastructure

#### 1.1 Create `.env.example`
```env
# Supabase Configuration
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_anon_key
SUPABASE_SERVICE_KEY=your_supabase_service_key

# Custom Embedding Model
EMBEDDING_MODEL_NAME=your_embedding_model
EMBEDDING_MODEL_ENDPOINT=http://localhost:8000/embed
EMBEDDING_DIMENSION=768
EMBEDDING_API_KEY=optional_api_key

# Custom LLM
LLM_MODEL_NAME=your_llm_model
LLM_ENDPOINT=http://localhost:8001/generate
LLM_API_KEY=optional_llm_api_key

# Re-ranker Model (optional)
RERANKER_MODEL_NAME=your_reranker_model
RERANKER_ENDPOINT=http://localhost:8002/rerank
RERANKER_API_KEY=optional_reranker_key

# Crawling Configuration
BASE_URL=https://docs.langchain.com/oss/python/langgraph
MAX_PAGES=1000
CRAWL_DELAY=1.0

# FastAPI Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
API_RELOAD=false
API_KEY=your_secret_api_key_change_in_production
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:5173,http://localhost:8080

# Rate Limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_PER_MINUTE=100
RATE_LIMIT_PER_SESSION=20

# Application Settings
LOG_LEVEL=INFO
DATA_DIR=./data
```

#### 1.2 Create `config.yaml`
```yaml
crawler:
  base_url: "${BASE_URL}"
  max_pages: 1000
  timeout: 30
  retry_attempts: 3
  delay: 1.0
  user_agent: "LangGraphRAGBot/1.0"
  respect_robots_txt: true

chunker:
  chunk_size: 512
  chunk_overlap: 128
  min_chunk_size: 100
  max_chunk_size: 2048
  respect_code_blocks: true
  respect_sentences: true
  code_delimiters:
    - "```"
    - "'''"
    - '"""'

embedding:
  model_name: "${EMBEDDING_MODEL_NAME}"
  dimension: 768
  batch_size: 32
  max_retries: 3

database:
  table_documents: "documents"
  table_chunks: "chunks"
  table_sessions: "sessions"
  table_messages: "messages"
  vector_index_type: "ivfflat"
  ivfflat_lists: 100

retrieval:
  top_k: 5
  semantic_weight: 0.6
  keyword_weight: 0.4
  strategies:
    - semantic
    - keyword
    - hybrid
    - backtracking
  reranker:
    enabled: true
    model: "${RERANKER_MODEL_NAME}"
    top_n_before_rerank: 20
    top_n_after_rerank: 5

agent:
  session_timeout: 3600
  max_history: 20
  temperature: 0.7
  max_tokens: 1024
  streaming: false

api:
  host: "0.0.0.0"
  port: 8000
  workers: 4
  reload: false
  cors:
    enabled: true
    origins:
      - "http://localhost:3000"
      - "http://localhost:5173"
      - "http://localhost:8080"
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
    chunk_size: 10

logging:
  level: "${LOG_LEVEL}"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "./data/logs/app.log"
```

#### 1.3 Create `requirements.txt`
```txt
# Core dependencies
python-dotenv==1.0.0
pyyaml==6.0.1
pydantic==2.5.0
pydantic-settings==2.1.0

# Web crawling
crawl4ai==0.3.0
beautifulsoup4==4.12.2
lxml==5.1.0
aiohttp==3.9.1

# NLP and text processing
nltk==3.8.1
spacy==3.7.2
tiktoken==0.5.2

# Database
supabase==2.3.0
pgvector==0.2.4
psycopg2-binary==2.9.9

# LangChain & LangGraph
langchain==0.1.0
langchain-core==0.1.10
langgraph==0.1.0
langchain-community==0.0.13

# Vector operations
numpy==1.26.2
sentence-transformers==2.2.2  # Optional: for testing embeddings

# HTTP clients
httpx==0.26.0
requests==2.31.0

# FastAPI & API
fastapi==0.109.0
uvicorn[standard]==0.27.0
sse-starlette==1.8.2
python-multipart==0.0.6
slowapi==0.1.9

# Utilities
tqdm==4.66.1
rich==13.7.0

# Development
pytest==7.4.3
pytest-asyncio==0.23.2
black==23.12.1
ruff==0.1.9
mypy==1.7.1
```

---

### Phase 2: Database Setup

#### 2.1 Create `src/database/schema.sql`

**Instructions for Claude:**
- Create the complete SQL schema based on PRD section 3.4.1
- Include all tables: documents, chunks, sessions, messages
- Add proper indexes for performance
- Enable pgvector extension
- Add constraints and foreign keys
- Include comments for documentation

**Key Requirements:**
- `documents` table for Level 1 (full page content)
- `chunks` table for Level 2 (smart chunks with embeddings)
- `sessions` table for thread management
- `messages` table for conversation history
- Vector indexes (ivfflat or hnsw) on embeddings
- Indexes on foreign keys and frequently queried columns

#### 2.2 Create `src/database/supabase_client.py`

**Instructions for Claude:**
- Create a Supabase client wrapper class
- Implement connection pooling
- Add retry logic for transient failures
- Support both sync and async operations
- Include methods for:
  - Connecting to Supabase
  - Executing queries
  - Inserting documents and chunks
  - Vector similarity search
  - Full-text search
  - Session management

#### 2.3 Create `scripts/setup_database.py`

**Instructions for Claude:**
- Create a script to initialize the database
- Read schema from `schema.sql`
- Execute SQL statements
- Enable pgvector extension
- Create tables and indexes
- Verify setup with test queries

---

### Phase 3: Web Crawling Module

#### 3.1 Create `src/crawler/sitemap_parser.py`

**Instructions for Claude:**
- Parse sitemap.xml from a given URL
- Extract all URLs from the sitemap
- Handle sitemap index files (sitemaps that reference other sitemaps)
- Filter URLs based on patterns (if needed)
- Return list of URLs to crawl

**Example Usage:**
```python
parser = SitemapParser()
urls = parser.parse("https://docs.langchain.com/sitemap.xml")
# Returns: ['https://docs.langchain.com/page1', 'https://docs.langchain.com/page2', ...]
```

#### 3.2 Create `src/crawler/web_crawler.py`

**Instructions for Claude:**
- Integrate Crawl4AI library
- Implement asynchronous crawling for efficiency
- Extract clean text content from HTML
- Preserve code blocks in raw format
- Extract metadata: title, URL, timestamp
- Implement rate limiting and retry logic
- Handle errors gracefully (404, timeout, etc.)
- Respect robots.txt

**Key Methods:**
- `crawl_url(url: str) -> Document`: Crawl single URL
- `crawl_multiple(urls: List[str]) -> List[Document]`: Batch crawling
- `extract_content(html: str) -> str`: Clean HTML extraction
- `detect_code_blocks(html: str) -> List[str]`: Extract code snippets

#### 3.3 Create `src/crawler/document_store.py`

**Instructions for Claude:**
- Save crawled documents locally in JSON format
- Create directory structure based on URL paths
- Generate content hash to detect duplicates
- Support loading documents from local storage
- Implement incremental updates (detect changed content)

**Storage Format:**
```json
{
  "url": "https://docs.langchain.com/langgraph/concepts",
  "title": "Core Concepts",
  "content": "Full page content...",
  "content_hash": "sha256_hash",
  "metadata": {
    "crawl_date": "2026-01-18T10:30:00",
    "status_code": 200,
    "content_type": "text/html"
  }
}
```

---

### Phase 4: Smart Chunking Module

#### 4.1 Create `src/chunker/code_detector.py`

**Instructions for Claude:**
- Detect code blocks using multiple delimiter patterns:
  - Triple backticks with language identifier: ```python
  - Triple quotes (single or double): ''' or """
  - Indented code blocks (4 spaces or tab)
- Return positions (start, end) of code blocks
- Identify programming language if possible
- Handle nested code blocks

**Example:**
```python
detector = CodeDetector()
code_blocks = detector.find_code_blocks(text)
# Returns: [
#   {"start": 100, "end": 250, "language": "python", "delimiter": "```"},
#   {"start": 300, "end": 400, "language": None, "delimiter": "'''"}
# ]
```

#### 4.2 Create `src/chunker/sentence_splitter.py`

**Instructions for Claude:**
- Use NLTK's sentence tokenizer (or spaCy)
- Handle edge cases:
  - Abbreviations (Dr., Mr., Inc., etc.)
  - Decimal numbers (3.14, 2.5)
  - URLs and email addresses
  - Ellipsis (...)
- Return sentence boundaries (character positions)
- Support multiple languages (focus on English first)

#### 4.3 Create `src/chunker/smart_chunker.py`

**Instructions for Claude:**
This is the core chunking logic. Implement a two-level chunking strategy:

**Level 1: Full Document Chunks**
- Store entire page content as single chunk
- Add metadata: URL, title, token count
- Purpose: Backtracking retrieval

**Level 2: Smart Chunks**
- Target chunk size: 512 tokens (configurable)
- Overlap: 128 tokens (configurable)

**Chunking Rules (Priority Order):**
1. **Never split code blocks**: If a code block is detected, keep it intact
   - If code block > max_chunk_size, make it a single chunk anyway
   - If adding code block to current chunk exceeds limit, start new chunk

2. **Respect sentence boundaries**: Only split at sentence endings
   - Don't split mid-sentence
   - If next sentence would exceed limit, start new chunk

3. **Maintain context**: Add overlap from previous chunk
   - Use last N tokens from previous chunk as context

4. **Track relationships**: Link chunks sequentially
   - Store previous_chunk_id and next_chunk_id

**Metadata per Chunk:**
```python
{
    "chunk_id": "uuid",
    "parent_doc_id": "uuid",
    "url": "source_url",
    "chunk_index": 0,
    "content": "chunk text",
    "tokens": 456,
    "has_code": True/False,
    "section_title": "extracted from headers",
    "previous_chunk_id": "uuid or None",
    "next_chunk_id": "uuid or None"
}
```

**Implementation Approach:**
```python
class SmartChunker:
    def __init__(self, chunk_size=512, overlap=128):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.code_detector = CodeDetector()
        self.sentence_splitter = SentenceSplitter()
    
    def chunk_document(self, document: Document) -> Tuple[Document, List[Chunk]]:
        """
        Returns:
            - Level 1: Full document chunk
            - Level 2: List of smart chunks
        """
        # Detect code blocks
        code_blocks = self.code_detector.find_code_blocks(document.content)
        
        # Find sentence boundaries
        sentences = self.sentence_splitter.split(document.content)
        
        # Create Level 1 chunk (full document)
        level1_chunk = self.create_full_document_chunk(document)
        
        # Create Level 2 chunks
        level2_chunks = self.create_smart_chunks(
            content=document.content,
            code_blocks=code_blocks,
            sentences=sentences,
            parent_id=level1_chunk.id
        )
        
        return level1_chunk, level2_chunks
    
    def create_smart_chunks(self, content, code_blocks, sentences, parent_id):
        chunks = []
        current_position = 0
        current_tokens = 0
        current_content = ""
        
        # Iterate through content, respecting code blocks and sentences
        # ... implementation details ...
        
        return chunks
```

---

### Phase 5: Embedding & Database Integration

#### 5.1 Create `src/embeddings/custom_embedder.py`

**Instructions for Claude:**
- Create wrapper for custom embedding model
- Support HTTP API calls to model endpoint
- Handle authentication (API keys)
- Implement retry logic
- Support batch processing

**Interface:**
```python
class CustomEmbedder:
    def __init__(self, model_name, endpoint, api_key=None):
        self.model_name = model_name
        self.endpoint = endpoint
        self.api_key = api_key
    
    def embed_text(self, text: str) -> List[float]:
        """Embed single text string"""
        pass
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed batch of texts"""
        pass
    
    async def embed_text_async(self, text: str) -> List[float]:
        """Async embedding"""
        pass
```

**Note:** User will provide the actual model implementation later. Create a flexible interface that can accommodate various embedding models.

#### 5.2 Create `src/embeddings/batch_processor.py`

**Instructions for Claude:**
- Process chunks in batches for efficiency
- Configurable batch size (default: 32)
- Progress tracking with tqdm
- Error handling and logging
- Save embeddings back to database

#### 5.3 Create `src/database/operations.py`

**Instructions for Claude:**
- CRUD operations for all tables
- Batch insert for chunks with embeddings
- Vector similarity search queries
- Full-text search queries
- Session and message management

**Key Functions:**
```python
# Documents
def insert_document(supabase_client, document: Document) -> str
def get_document_by_url(supabase_client, url: str) -> Optional[Document]
def get_document_by_id(supabase_client, doc_id: str) -> Optional[Document]

# Chunks
def insert_chunks_batch(supabase_client, chunks: List[Chunk]) -> List[str]
def get_chunks_by_parent_id(supabase_client, parent_id: str) -> List[Chunk]

# Search
def semantic_search(supabase_client, query_embedding: List[float], top_k: int) -> List[Chunk]
def keyword_search(supabase_client, query: str, top_k: int) -> List[Chunk]
def hybrid_search(supabase_client, query: str, query_embedding: List[float], top_k: int) -> List[Chunk]

# Sessions
def create_session(supabase_client, user_id: str) -> str  # Returns thread_id
def get_session(supabase_client, thread_id: str) -> Optional[Session]
def get_messages(supabase_client, thread_id: str, limit: int) -> List[Message]
def save_message(supabase_client, thread_id: str, role: str, content: str) -> str
```

---

### Phase 6: Multi-Strategy RAG Tool

#### 6.1 Create `src/retrieval/semantic_search.py`

**Instructions for Claude:**
- Implement vector similarity search using pgvector
- Use cosine similarity metric
- Return top-k most similar chunks
- Include similarity scores in results

```python
class SemanticSearcher:
    def __init__(self, supabase_client, embedder):
        self.supabase_client = supabase_client
        self.embedder = embedder
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        """
        Returns:
            List of (Chunk, similarity_score) tuples
        """
        # Embed query
        query_embedding = self.embedder.embed_text(query)
        
        # Search database
        results = semantic_search(
            self.supabase_client,
            query_embedding,
            top_k
        )
        
        return results
```

#### 6.2 Create `src/retrieval/keyword_search.py`

**Instructions for Claude:**
- Implement full-text search using PostgreSQL tsvector
- Support partial matching
- Rank results by relevance
- Handle special characters and stopwords

```python
class KeywordSearcher:
    def __init__(self, supabase_client):
        self.supabase_client = supabase_client
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        """
        Returns:
            List of (Chunk, relevance_score) tuples
        """
        results = keyword_search(
            self.supabase_client,
            query,
            top_k
        )
        
        return results
```

#### 6.3 Create `src/retrieval/hybrid_search.py`

**Instructions for Claude:**
- Combine semantic and keyword search results
- Apply weighted scoring (configurable weights)
- Remove duplicates
- Optionally apply re-ranker (1/60 ratio)

```python
class HybridSearcher:
    def __init__(self, supabase_client, embedder, reranker=None):
        self.semantic_searcher = SemanticSearcher(supabase_client, embedder)
        self.keyword_searcher = KeywordSearcher(supabase_client)
        self.reranker = reranker
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        semantic_weight: float = 0.6,
        keyword_weight: float = 0.4
    ) -> List[Tuple[Chunk, float]]:
        """
        Returns:
            List of (Chunk, combined_score) tuples
        """
        # Get results from both methods
        semantic_results = self.semantic_searcher.search(query, top_k=20)
        keyword_results = self.keyword_searcher.search(query, top_k=20)
        
        # Combine and score
        combined = self.combine_results(
            semantic_results,
            keyword_results,
            semantic_weight,
            keyword_weight
        )
        
        # Apply re-ranker if available
        if self.reranker:
            combined = self.reranker.rerank(query, combined, top_k)
        
        return combined[:top_k]
```

#### 6.4 Create `src/retrieval/backtracking.py`

**Instructions for Claude:**
- Retrieve full parent document when needed
- Triggered when chunk context is insufficient
- Include method to get all chunks from same document

```python
class BacktrackingRetriever:
    def __init__(self, supabase_client):
        self.supabase_client = supabase_client
    
    def get_parent_document(self, chunk_id: str) -> Document:
        """Get full parent document from a chunk"""
        chunk = get_chunk_by_id(self.supabase_client, chunk_id)
        document = get_document_by_id(self.supabase_client, chunk.parent_doc_id)
        return document
    
    def get_sibling_chunks(self, chunk_id: str) -> List[Chunk]:
        """Get all chunks from the same parent document"""
        chunk = get_chunk_by_id(self.supabase_client, chunk_id)
        siblings = get_chunks_by_parent_id(self.supabase_client, chunk.parent_doc_id)
        return siblings
```

#### 6.5 Create `src/retrieval/rag_tool.py`

**Instructions for Claude:**
- Unified interface for all retrieval strategies
- Strategy selection based on query analysis
- Tool interface for LangGraph integration

```python
class RAGTool:
    def __init__(self, config, supabase_client, embedder, reranker=None):
        self.config = config
        self.semantic_searcher = SemanticSearcher(supabase_client, embedder)
        self.keyword_searcher = KeywordSearcher(supabase_client)
        self.hybrid_searcher = HybridSearcher(supabase_client, embedder, reranker)
        self.backtracker = BacktrackingRetriever(supabase_client)
    
    def select_strategy(self, query: str) -> str:
        """
        Analyze query and select best retrieval strategy
        
        Heuristics:
        - Contains code-related keywords → semantic with has_code filter
        - Specific API/function names → keyword search
        - Conceptual questions → semantic search
        - Complex multi-part → hybrid search
        """
        # Implementation logic...
        pass
    
    def retrieve(
        self,
        query: str,
        strategy: Optional[str] = None,
        top_k: int = 5
    ) -> List[Chunk]:
        """
        Main retrieval method
        
        Args:
            query: User query
            strategy: 'semantic', 'keyword', 'hybrid', or None (auto-select)
            top_k: Number of chunks to return
        
        Returns:
            List of most relevant chunks
        """
        if strategy is None:
            strategy = self.select_strategy(query)
        
        if strategy == "semantic":
            results = self.semantic_searcher.search(query, top_k)
        elif strategy == "keyword":
            results = self.keyword_searcher.search(query, top_k)
        elif strategy == "hybrid":
            results = self.hybrid_searcher.search(query, top_k)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Extract just the chunks (remove scores for now)
        chunks = [chunk for chunk, score in results]
        
        return chunks
    
    def as_tool(self):
        """Return LangGraph-compatible tool"""
        from langchain.tools import Tool
        
        return Tool(
            name="documentation_search",
            description="Search LangGraph documentation for relevant information",
            func=self.retrieve
        )
```

---

### Phase 7: LangGraph Agent

#### 7.1 Create `src/agent/state.py`

**Instructions for Claude:**
- Define state schema using TypedDict or Pydantic
- Include all necessary fields for agent operation

```python
from typing import TypedDict, List, Optional, Annotated
from operator import add

class AgentState(TypedDict):
    """State schema for the LangGraph agent"""
    
    # Conversation
    messages: Annotated[List[dict], add]  # List of messages (accumulative)
    
    # Current query processing
    current_query: str
    query_strategy: Optional[str]  # Selected retrieval strategy
    
    # Retrieved context
    retrieved_chunks: List[dict]  # Retrieved chunks from RAG
    retrieved_documents: List[dict]  # Full documents (if backtracking)
    
    # Session management
    thread_id: str  # Session identifier
    user_id: Optional[str]
    
    # Agent metadata
    iteration_count: int
    needs_backtracking: bool
    
    # Output
    final_response: Optional[str]
```

#### 7.2 Create `src/agent/nodes.py`

**Instructions for Claude:**
- Implement all node functions for the graph
- Each node takes state and returns updated state

```python
from typing import Dict, Any
from src.agent.state import AgentState

class AgentNodes:
    def __init__(self, rag_tool, llm, session_manager, supabase_client):
        self.rag_tool = rag_tool
        self.llm = llm
        self.session_manager = session_manager
        self.supabase_client = supabase_client
    
    def input_processing(self, state: AgentState) -> AgentState:
        """
        Node 1: Process user input
        - Extract current query from messages
        - Load conversation history
        - Prepare context
        """
        # Get latest user message
        user_messages = [m for m in state["messages"] if m["role"] == "user"]
        current_query = user_messages[-1]["content"] if user_messages else ""
        
        # Load conversation history from database
        history = self.session_manager.get_conversation_history(
            state["thread_id"],
            limit=20
        )
        
        # Update state
        state["current_query"] = current_query
        state["messages"] = history + state["messages"]  # Merge
        
        return state
    
    def query_analysis(self, state: AgentState) -> AgentState:
        """
        Node 2: Analyze query
        - Determine retrieval strategy
        - Extract key entities/topics
        """
        query = state["current_query"]
        
        # Use heuristics or LLM to select strategy
        strategy = self.rag_tool.select_strategy(query)
        
        state["query_strategy"] = strategy
        state["iteration_count"] = state.get("iteration_count", 0) + 1
        
        return state
    
    def rag_retrieval(self, state: AgentState) -> AgentState:
        """
        Node 3: Execute RAG retrieval
        - Use selected strategy
        - Fetch relevant chunks
        """
        query = state["current_query"]
        strategy = state["query_strategy"]
        
        # Retrieve chunks
        chunks = self.rag_tool.retrieve(
            query=query,
            strategy=strategy,
            top_k=5
        )
        
        # Convert to dict for JSON serialization
        state["retrieved_chunks"] = [chunk.dict() for chunk in chunks]
        
        return state
    
    def response_generation(self, state: AgentState) -> AgentState:
        """
        Node 4: Generate response using LLM
        - Use retrieved context
        - Incorporate conversation history
        """
        query = state["current_query"]
        chunks = state["retrieved_chunks"]
        
        # Build context from chunks
        context = self.build_context_from_chunks(chunks)
        
        # Build prompt
        prompt = self.build_prompt(query, context, state["messages"])
        
        # Generate response
        response = self.llm.generate(prompt)
        
        state["final_response"] = response
        
        return state
    
    def output_processing(self, state: AgentState) -> AgentState:
        """
        Node 5: Process and save output
        - Format response
        - Save to conversation history
        - Update session metadata
        """
        response = state["final_response"]
        thread_id = state["thread_id"]
        
        # Save messages to database
        self.session_manager.save_message(
            thread_id=thread_id,
            role="user",
            content=state["current_query"]
        )
        
        self.session_manager.save_message(
            thread_id=thread_id,
            role="assistant",
            content=response
        )
        
        # Add to state messages
        state["messages"].append({"role": "assistant", "content": response})
        
        return state
    
    def backtracking_node(self, state: AgentState) -> AgentState:
        """
        Optional node: Retrieve full parent documents
        - Triggered when chunk context is insufficient
        """
        chunks = state["retrieved_chunks"]
        
        # Get parent documents
        documents = []
        for chunk in chunks[:2]:  # Top 2 chunks
            doc = self.rag_tool.backtracker.get_parent_document(chunk["chunk_id"])
            documents.append(doc.dict())
        
        state["retrieved_documents"] = documents
        state["needs_backtracking"] = False
        
        return state
    
    def build_context_from_chunks(self, chunks: List[dict]) -> str:
        """Helper: Build context string from retrieved chunks"""
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(f"[{i}] Source: {chunk['url']}")
            context_parts.append(f"{chunk['content']}\n")
        
        return "\n".join(context_parts)
    
    def build_prompt(self, query: str, context: str, history: List[dict]) -> str:
        """Helper: Build LLM prompt"""
        # Build conversation history
        history_text = ""
        for msg in history[-10:]:  # Last 10 messages
            role = msg["role"].capitalize()
            history_text += f"{role}: {msg['content']}\n"
        
        prompt = f"""You are a helpful assistant that answers questions about LangGraph documentation.

Conversation History:
{history_text}

Context from documentation:
{context}

User Question: {query}

Please provide a clear and accurate answer based on the documentation context above. If the context doesn't contain enough information, say so.

Answer:"""
        
        return prompt
```

#### 7.3 Create `src/agent/graph.py`

**Instructions for Claude:**
- Construct LangGraph workflow
- Define nodes and edges
- Add conditional routing if needed

```python
from langgraph.graph import StateGraph, END
from src.agent.state import AgentState
from src.agent.nodes import AgentNodes

def create_agent_graph(rag_tool, llm, session_manager, supabase_client):
    """Create and compile the LangGraph agent"""
    
    # Initialize nodes
    nodes = AgentNodes(rag_tool, llm, session_manager, supabase_client)
    
    # Create graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("input_processing", nodes.input_processing)
    workflow.add_node("query_analysis", nodes.query_analysis)
    workflow.add_node("rag_retrieval", nodes.rag_retrieval)
    workflow.add_node("response_generation", nodes.response_generation)
    workflow.add_node("output_processing", nodes.output_processing)
    workflow.add_node("backtracking", nodes.backtracking_node)
    
    # Define edges
    workflow.set_entry_point("input_processing")
    
    workflow.add_edge("input_processing", "query_analysis")
    workflow.add_edge("query_analysis", "rag_retrieval")
    
    # Conditional edge: backtracking if needed
    def should_backtrack(state: AgentState) -> str:
        """Decide if backtracking is needed"""
        # Simple heuristic: if fewer than 3 chunks retrieved
        if len(state["retrieved_chunks"]) < 3:
            return "backtracking"
        return "response_generation"
    
    workflow.add_conditional_edges(
        "rag_retrieval",
        should_backtrack,
        {
            "backtracking": "backtracking",
            "response_generation": "response_generation"
        }
    )
    
    workflow.add_edge("backtracking", "response_generation")
    workflow.add_edge("response_generation", "output_processing")
    workflow.add_edge("output_processing", END)
    
    # Compile graph
    app = workflow.compile()
    
    return app
```

#### 7.4 Create `src/agent/session_manager.py`

**Instructions for Claude:**
- Manage user sessions and threads
- Load/save conversation history
- Handle session timeouts

```python
from typing import List, Optional
from datetime import datetime, timedelta
import uuid

class SessionManager:
    def __init__(self, supabase_client, timeout_seconds=3600):
        self.supabase_client = supabase_client
        self.timeout = timeout_seconds
    
    def create_session(self, user_id: Optional[str] = None) -> str:
        """Create a new session"""
        thread_id = str(uuid.uuid4())
        
        # Insert into database
        self.supabase_client.table("sessions").insert({
            "thread_id": thread_id,
            "user_id": user_id,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }).execute()
        
        return thread_id
    
    def get_session(self, thread_id: str) -> Optional[dict]:
        """Get session details"""
        result = self.supabase_client.table("sessions").select("*").eq(
            "thread_id", thread_id
        ).execute()
        
        if result.data:
            return result.data[0]
        return None
    
    def is_session_active(self, thread_id: str) -> bool:
        """Check if session is still active"""
        session = self.get_session(thread_id)
        if not session:
            return False
        
        updated_at = datetime.fromisoformat(session["updated_at"])
        if datetime.now() - updated_at > timedelta(seconds=self.timeout):
            return False
        
        return True
    
    def update_session(self, thread_id: str):
        """Update session timestamp"""
        self.supabase_client.table("sessions").update({
            "updated_at": datetime.now().isoformat()
        }).eq("thread_id", thread_id).execute()
    
    def get_conversation_history(
        self,
        thread_id: str,
        limit: int = 20
    ) -> List[dict]:
        """Get conversation history for a session"""
        result = self.supabase_client.table("messages").select(
            "role, content, created_at"
        ).eq("thread_id", thread_id).order(
            "created_at", desc=False
        ).limit(limit).execute()
        
        messages = []
        for msg in result.data:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        return messages
    
    def save_message(
        self,
        thread_id: str,
        role: str,
        content: str
    ) -> str:
        """Save a message to the conversation history"""
        message_id = str(uuid.uuid4())
        
        self.supabase_client.table("messages").insert({
            "id": message_id,
            "thread_id": thread_id,
            "role": role,
            "content": content,
            "created_at": datetime.now().isoformat()
        }).execute()
        
        # Update session timestamp
        self.update_session(thread_id)
        
        return message_id
    
    def clear_session(self, thread_id: str):
        """Clear all messages from a session"""
        self.supabase_client.table("messages").delete().eq(
            "thread_id", thread_id
        ).execute()
```

---

### Phase 8: Custom LLM Integration

#### 8.1 Create `src/llm/custom_llm.py`

**Instructions for Claude:**
- Create a flexible wrapper for custom LLM
- Support HTTP API calls
- Handle streaming (optional)
- Compatible with LangChain's LLM interface

```python
from typing import Optional, List, Any
import httpx

class CustomLLM:
    """
    Wrapper for custom LLM
    
    This is a placeholder that should be customized based on the
    actual LLM API being used.
    """
    
    def __init__(
        self,
        model_name: str,
        endpoint: str,
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024
    ):
        self.model_name = model_name
        self.endpoint = endpoint
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate text from prompt
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature (overrides default)
            max_tokens: Max tokens to generate (overrides default)
        
        Returns:
            Generated text
        """
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens
        
        # Prepare request
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": temp,
            "max_tokens": max_tok
        }
        
        # Make API call
        response = httpx.post(
            self.endpoint,
            json=payload,
            headers=headers,
            timeout=30.0
        )
        
        response.raise_for_status()
        
        # Parse response (adjust based on actual API)
        result = response.json()
        generated_text = result.get("text", result.get("response", ""))
        
        return generated_text
    
    def generate_chat(
        self,
        messages: List[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate response from chat messages
        
        Args:
            messages: List of {"role": "user/assistant", "content": "..."}
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
        
        Returns:
            Generated response
        """
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens
        
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temp,
            "max_tokens": max_tok
        }
        
        response = httpx.post(
            self.endpoint,
            json=payload,
            headers=headers,
            timeout=30.0
        )
        
        response.raise_for_status()
        
        result = response.json()
        generated_text = result.get("message", {}).get("content", "")
        
        return generated_text
    
    async def generate_async(self, prompt: str) -> str:
        """Async version of generate"""
        # Implementation similar to generate but with async httpx
        pass
    
    def stream_generate(self, prompt: str):
        """Stream generated tokens (if supported)"""
        # Streaming implementation
        pass
```

**Note to User:**
This is a generic template. You'll need to:
1. Adjust the API endpoint format
2. Modify request/response parsing based on your LLM's API
3. Add any authentication specifics
4. Implement streaming if needed

---

### Phase 9: Orchestration Scripts

#### 9.1 Create `scripts/crawl_and_ingest.py`

**Instructions for Claude:**
- Orchestrate the full ingestion pipeline
- Crawl → Chunk → Embed → Store

```python
#!/usr/bin/env python3
"""
Full ingestion pipeline:
1. Parse sitemap
2. Crawl URLs
3. Save raw documents
4. Chunk documents
5. Generate embeddings
6. Store in database
"""

import asyncio
from pathlib import Path
from src.config.settings import load_config
from src.crawler.sitemap_parser import SitemapParser
from src.crawler.web_crawler import WebCrawler
from src.crawler.document_store import DocumentStore
from src.chunker.smart_chunker import SmartChunker
from src.embeddings.custom_embedder import CustomEmbedder
from src.embeddings.batch_processor import BatchProcessor
from src.database.supabase_client import SupabaseClient
from src.database.operations import insert_document, insert_chunks_batch
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

async def main():
    # Load configuration
    config = load_config()
    
    # Initialize components
    logger.info("Initializing components...")
    sitemap_parser = SitemapParser()
    crawler = WebCrawler(config)
    doc_store = DocumentStore(config["data_dir"])
    chunker = SmartChunker(
        chunk_size=config["chunker"]["chunk_size"],
        overlap=config["chunker"]["chunk_overlap"]
    )
    embedder = CustomEmbedder(
        model_name=config["embedding"]["model_name"],
        endpoint=config["embedding"]["endpoint"]
    )
    supabase_client = SupabaseClient(config)
    
    # Step 1: Parse sitemap
    logger.info("Parsing sitemap...")
    sitemap_url = f"{config['crawler']['base_url']}/sitemap.xml"
    urls = sitemap_parser.parse(sitemap_url)
    logger.info(f"Found {len(urls)} URLs to crawl")
    
    # Step 2: Crawl URLs
    logger.info("Crawling URLs...")
    documents = await crawler.crawl_multiple(urls[:config["crawler"]["max_pages"]])
    logger.info(f"Crawled {len(documents)} documents")
    
    # Step 3: Save raw documents
    logger.info("Saving raw documents...")
    for doc in documents:
        doc_store.save_document(doc)
    
    # Step 4 & 5: Chunk and embed
    logger.info("Chunking and embedding documents...")
    batch_processor = BatchProcessor(embedder, supabase_client, config)
    
    for doc in documents:
        # Check if already processed
        existing = supabase_client.get_document_by_url(doc.url)
        if existing:
            logger.info(f"Skipping already processed: {doc.url}")
            continue
        
        # Insert Level 1 document
        doc_id = insert_document(supabase_client, doc)
        logger.info(f"Inserted document: {doc.url}")
        
        # Chunk document
        level1_chunk, level2_chunks = chunker.chunk_document(doc)
        
        # Set parent IDs
        for chunk in level2_chunks:
            chunk.parent_doc_id = doc_id
        
        # Embed chunks
        embeddings = await batch_processor.process_chunks(level2_chunks)
        
        # Store chunks with embeddings
        insert_chunks_batch(supabase_client, level2_chunks)
        logger.info(f"Inserted {len(level2_chunks)} chunks")
    
    logger.info("Ingestion complete!")

if __name__ == "__main__":
    asyncio.run(main())
```

#### 9.2 Create `scripts/run_agent.py`

**Instructions for Claude:**
- Interactive CLI for testing the agent
- Create sessions
- Send queries
- Display responses

```python
#!/usr/bin/env python3
"""
Run the LangGraph RAG agent interactively
"""

from src.config.settings import load_config
from src.database.supabase_client import SupabaseClient
from src.embeddings.custom_embedder import CustomEmbedder
from src.retrieval.rag_tool import RAGTool
from src.llm.custom_llm import CustomLLM
from src.agent.session_manager import SessionManager
from src.agent.graph import create_agent_graph
from src.agent.state import AgentState
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def main():
    # Load configuration
    config = load_config()
    
    # Initialize components
    logger.info("Initializing agent...")
    supabase_client = SupabaseClient(config)
    embedder = CustomEmbedder(
        model_name=config["embedding"]["model_name"],
        endpoint=config["embedding"]["endpoint"]
    )
    
    rag_tool = RAGTool(config, supabase_client, embedder)
    
    llm = CustomLLM(
        model_name=config["llm"]["model_name"],
        endpoint=config["llm"]["endpoint"],
        temperature=config["agent"]["temperature"],
        max_tokens=config["agent"]["max_tokens"]
    )
    
    session_manager = SessionManager(
        supabase_client,
        timeout_seconds=config["agent"]["session_timeout"]
    )
    
    # Create agent graph
    agent = create_agent_graph(rag_tool, llm, session_manager, supabase_client)
    
    # Create session
    thread_id = session_manager.create_session(user_id="cli_user")
    logger.info(f"Created session: {thread_id}")
    
    print("\n" + "="*60)
    print("LangGraph Documentation Assistant")
    print("="*60)
    print("Type 'quit' to exit, 'new' for new session")
    print("="*60 + "\n")
    
    # Interactive loop
    while True:
        # Get user input
        user_input = input("\nYou: ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() == "quit":
            print("Goodbye!")
            break
        
        if user_input.lower() == "new":
            thread_id = session_manager.create_session(user_id="cli_user")
            print(f"Started new session: {thread_id}")
            continue
        
        # Prepare initial state
        initial_state: AgentState = {
            "messages": [{"role": "user", "content": user_input}],
            "current_query": "",
            "query_strategy": None,
            "retrieved_chunks": [],
            "retrieved_documents": [],
            "thread_id": thread_id,
            "user_id": "cli_user",
            "iteration_count": 0,
            "needs_backtracking": False,
            "final_response": None
        }
        
        # Run agent
        try:
            result = agent.invoke(initial_state)
            
            # Display response
            response = result["final_response"]
            print(f"\nAssistant: {response}")
            
            # Optional: show metadata
            if config.get("show_metadata", False):
                print(f"\n[Strategy: {result['query_strategy']}]")
                print(f"[Retrieved {len(result['retrieved_chunks'])} chunks]")
        
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            print(f"\nError: {e}")

if __name__ == "__main__":
    main()
```

---

### Phase 10: FastAPI REST API

Now we'll create a production-ready REST API that exposes the agent functionality to UI clients.

#### 10.1 Create `src/api/models/requests.py`

**Instructions for Claude:**
- Define Pydantic models for all API requests
- Include validation rules

```python
from pydantic import BaseModel, Field, validator
from typing import Optional, List

class CreateSessionRequest(BaseModel):
    """Request to create a new chat session"""
    user_id: Optional[str] = Field(None, description="Optional user identifier")

class SendMessageRequest(BaseModel):
    """Request to send a message in a chat session"""
    message: str = Field(..., min_length=1, max_length=10000, description="User message")
    stream: bool = Field(False, description="Whether to stream the response")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0, description="LLM temperature")
    max_tokens: Optional[int] = Field(None, ge=1, le=4096, description="Max tokens to generate")
    
    @validator('message')
    def message_not_empty(cls, v):
        if not v.strip():
            raise ValueError('Message cannot be empty or whitespace')
        return v.strip()

class SearchRequest(BaseModel):
    """Request for direct search (debugging endpoint)"""
    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(5, ge=1, le=50)
    strategy: Optional[str] = Field(None, regex="^(semantic|keyword|hybrid)$")
```

#### 10.2 Create `src/api/models/responses.py`

**Instructions for Claude:**
- Define Pydantic models for all API responses
- Include proper typing

```python
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class ErrorResponse(BaseModel):
    """Standard error response"""
    error: Dict[str, Any] = Field(..., description="Error details")
    timestamp: datetime = Field(default_factory=datetime.now)

class SessionResponse(BaseModel):
    """Response for session creation/retrieval"""
    thread_id: str
    user_id: Optional[str]
    created_at: datetime
    updated_at: Optional[datetime] = None

class SourceInfo(BaseModel):
    """Information about a source document"""
    url: str
    title: Optional[str] = None
    chunk_index: Optional[int] = None

class MessageMetadata(BaseModel):
    """Metadata about message processing"""
    retrieval_strategy: Optional[str] = None
    chunks_retrieved: int = 0
    processing_time_ms: float = 0
    sources: List[SourceInfo] = []

class MessageResponse(BaseModel):
    """Response for a chat message"""
    message_id: str
    thread_id: str
    role: str
    content: str
    metadata: Optional[MessageMetadata] = None
    created_at: datetime

class ConversationMessage(BaseModel):
    """Single message in conversation history"""
    message_id: str
    role: str
    content: str
    created_at: datetime

class ConversationHistoryResponse(BaseModel):
    """Response for conversation history"""
    thread_id: str
    messages: List[ConversationMessage]
    total: int
    limit: int
    offset: int

class SearchResult(BaseModel):
    """Single search result"""
    chunk_id: str
    content: str
    url: str
    similarity_score: Optional[float] = None
    has_code: bool = False

class SearchResponse(BaseModel):
    """Response for search endpoint"""
    query: str
    strategy: str
    results: List[SearchResult]
    total: int

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    database: str
    timestamp: datetime = Field(default_factory=datetime.now)

class StatusResponse(BaseModel):
    """System status response"""
    documents_count: int
    chunks_count: int
    active_sessions: int
    uptime_seconds: float
```

#### 10.3 Create `src/api/dependencies.py`

**Instructions for Claude:**
- Dependency injection for FastAPI
- Shared resources across endpoints

```python
from fastapi import Header, HTTPException, status
from typing import Optional
from src.config.settings import load_config
from src.database.supabase_client import SupabaseClient
from src.embeddings.custom_embedder import CustomEmbedder
from src.retrieval.rag_tool import RAGTool
from src.llm.custom_llm import CustomLLM
from src.agent.session_manager import SessionManager
from src.agent.graph import create_agent_graph

# Global instances (initialized at startup)
_config = None
_supabase_client = None
_embedder = None
_rag_tool = None
_llm = None
_session_manager = None
_agent = None

def init_dependencies():
    """Initialize all dependencies at startup"""
    global _config, _supabase_client, _embedder, _rag_tool, _llm, _session_manager, _agent
    
    _config = load_config()
    _supabase_client = SupabaseClient(_config)
    
    _embedder = CustomEmbedder(
        model_name=_config["embedding"]["model_name"],
        endpoint=_config["embedding"]["endpoint"]
    )
    
    _rag_tool = RAGTool(_config, _supabase_client, _embedder)
    
    _llm = CustomLLM(
        model_name=_config["llm"]["model_name"],
        endpoint=_config["llm"]["endpoint"],
        temperature=_config["agent"]["temperature"],
        max_tokens=_config["agent"]["max_tokens"]
    )
    
    _session_manager = SessionManager(
        _supabase_client,
        timeout_seconds=_config["agent"]["session_timeout"]
    )
    
    _agent = create_agent_graph(_rag_tool, _llm, _session_manager, _supabase_client)

def get_config():
    """Dependency: Get configuration"""
    return _config

def get_supabase_client():
    """Dependency: Get Supabase client"""
    return _supabase_client

def get_rag_tool():
    """Dependency: Get RAG tool"""
    return _rag_tool

def get_agent():
    """Dependency: Get LangGraph agent"""
    return _agent

def get_session_manager():
    """Dependency: Get session manager"""
    return _session_manager

async def verify_api_key(
    x_api_key: Optional[str] = Header(None),
    config = get_config
):
    """Dependency: Verify API key authentication"""
    if not config["api"]["authentication"]["enabled"]:
        return True
    
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required"
        )
    
    # In production, check against database or secret manager
    expected_key = config.get("api_key")
    if x_api_key != expected_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key"
        )
    
    return True
```

#### 10.4 Create `src/api/middleware.py`

**Instructions for Claude:**
- CORS middleware
- Rate limiting
- Error handling

```python
from fastapi import Request, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from starlette.middleware.base import BaseHTTPMiddleware
import time

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

def setup_cors(app, config):
    """Setup CORS middleware"""
    if config["api"]["cors"]["enabled"]:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=config["api"]["cors"]["origins"],
            allow_credentials=config["api"]["cors"]["allow_credentials"],
            allow_methods=config["api"]["cors"]["allow_methods"],
            allow_headers=config["api"]["cors"]["allow_headers"],
        )

def setup_rate_limiting(app):
    """Setup rate limiting"""
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

class TimingMiddleware(BaseHTTPMiddleware):
    """Add request timing to responses"""
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = (time.time() - start_time) * 1000  # ms
        response.headers["X-Process-Time"] = str(process_time)
        return response
```

#### 10.5 Create `src/api/routes/health.py`

**Instructions for Claude:**
- Health check endpoints
- System status

```python
from fastapi import APIRouter, Depends
from src.api.models.responses import HealthResponse, StatusResponse
from src.api.dependencies import get_supabase_client, get_config
from datetime import datetime
import time

router = APIRouter(prefix="/api/v1", tags=["health"])

# Track start time for uptime
_start_time = time.time()

@router.get("/health", response_model=HealthResponse)
async def health_check(
    supabase_client=Depends(get_supabase_client),
    config=Depends(get_config)
):
    """Health check endpoint"""
    
    # Test database connection
    try:
        result = supabase_client.table("documents").select("id").limit(1).execute()
        db_status = "connected"
    except Exception:
        db_status = "disconnected"
    
    return HealthResponse(
        status="healthy" if db_status == "connected" else "degraded",
        version="1.0.0",
        database=db_status
    )

@router.get("/status", response_model=StatusResponse)
async def system_status(
    supabase_client=Depends(get_supabase_client)
):
    """System status endpoint"""
    
    # Get counts
    docs_result = supabase_client.table("documents").select("id", count="exact").execute()
    chunks_result = supabase_client.table("chunks").select("id", count="exact").execute()
    sessions_result = supabase_client.table("sessions").select("id", count="exact").execute()
    
    documents_count = docs_result.count if hasattr(docs_result, 'count') else 0
    chunks_count = chunks_result.count if hasattr(chunks_result, 'count') else 0
    active_sessions = sessions_result.count if hasattr(sessions_result, 'count') else 0
    
    uptime = time.time() - _start_time
    
    return StatusResponse(
        documents_count=documents_count,
        chunks_count=chunks_count,
        active_sessions=active_sessions,
        uptime_seconds=uptime
    )
```

#### 10.6 Create `src/api/routes/chat.py`

**Instructions for Claude:**
- Main chat endpoints
- Session management
- Message handling
- Streaming support

```python
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse
from typing import Optional
import json
import time
from datetime import datetime

from src.api.models.requests import CreateSessionRequest, SendMessageRequest
from src.api.models.responses import (
    SessionResponse, MessageResponse, ConversationHistoryResponse,
    MessageMetadata, SourceInfo, ErrorResponse, ConversationMessage
)
from src.api.dependencies import (
    get_agent, get_session_manager, get_supabase_client,
    verify_api_key
)
from src.api.middleware import limiter
from src.agent.state import AgentState
from src.utils.logger import setup_logger

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])
logger = setup_logger(__name__)

@router.post("/sessions", response_model=SessionResponse, dependencies=[Depends(verify_api_key)])
@limiter.limit("10/minute")
async def create_session(
    request: CreateSessionRequest,
    session_manager=Depends(get_session_manager)
):
    """Create a new chat session"""
    try:
        thread_id = session_manager.create_session(user_id=request.user_id)
        session = session_manager.get_session(thread_id)
        
        return SessionResponse(
            thread_id=thread_id,
            user_id=request.user_id,
            created_at=datetime.fromisoformat(session["created_at"]),
            updated_at=datetime.fromisoformat(session.get("updated_at", session["created_at"]))
        )
    except Exception as e:
        logger.error(f"Error creating session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create session"
        )

@router.get("/sessions/{thread_id}", response_model=SessionResponse, dependencies=[Depends(verify_api_key)])
async def get_session(
    thread_id: str,
    session_manager=Depends(get_session_manager)
):
    """Get session details"""
    session = session_manager.get_session(thread_id)
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )
    
    return SessionResponse(
        thread_id=session["thread_id"],
        user_id=session.get("user_id"),
        created_at=datetime.fromisoformat(session["created_at"]),
        updated_at=datetime.fromisoformat(session.get("updated_at", session["created_at"]))
    )

@router.delete("/sessions/{thread_id}", dependencies=[Depends(verify_api_key)])
async def delete_session(
    thread_id: str,
    session_manager=Depends(get_session_manager)
):
    """Delete a session and all its messages"""
    try:
        session_manager.clear_session(thread_id)
        return {"deleted": True, "thread_id": thread_id}
    except Exception as e:
        logger.error(f"Error deleting session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete session"
        )

@router.post("/sessions/{thread_id}/messages", response_model=MessageResponse, dependencies=[Depends(verify_api_key)])
@limiter.limit("20/minute")
async def send_message(
    thread_id: str,
    request: SendMessageRequest,
    agent=Depends(get_agent),
    session_manager=Depends(get_session_manager)
):
    """Send a message to the agent (non-streaming)"""
    
    # Check if session exists
    if not session_manager.is_session_active(thread_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found or expired"
        )
    
    # Prepare initial state
    initial_state: AgentState = {
        "messages": [{"role": "user", "content": request.message}],
        "current_query": "",
        "query_strategy": None,
        "retrieved_chunks": [],
        "retrieved_documents": [],
        "thread_id": thread_id,
        "user_id": session_manager.get_session(thread_id).get("user_id"),
        "iteration_count": 0,
        "needs_backtracking": False,
        "final_response": None
    }
    
    # Run agent
    start_time = time.time()
    try:
        result = agent.invoke(initial_state)
        processing_time = (time.time() - start_time) * 1000
        
        # Extract sources from retrieved chunks
        sources = []
        for chunk in result.get("retrieved_chunks", [])[:5]:  # Top 5 sources
            sources.append(SourceInfo(
                url=chunk.get("url", ""),
                title=chunk.get("section_title"),
                chunk_index=chunk.get("chunk_index")
            ))
        
        # Create metadata
        metadata = MessageMetadata(
            retrieval_strategy=result.get("query_strategy"),
            chunks_retrieved=len(result.get("retrieved_chunks", [])),
            processing_time_ms=processing_time,
            sources=sources
        )
        
        # Get the saved message from database
        # (The agent should have already saved it via session_manager)
        messages = session_manager.get_conversation_history(thread_id, limit=1)
        latest_message = messages[-1] if messages else None
        
        return MessageResponse(
            message_id=str(hash(result["final_response"])),  # Simple hash as ID
            thread_id=thread_id,
            role="assistant",
            content=result["final_response"],
            metadata=metadata,
            created_at=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing message: {str(e)}"
        )

@router.post("/sessions/{thread_id}/messages/stream", dependencies=[Depends(verify_api_key)])
@limiter.limit("20/minute")
async def send_message_stream(
    thread_id: str,
    request: SendMessageRequest,
    agent=Depends(get_agent),
    session_manager=Depends(get_session_manager)
):
    """Send a message with streaming response (SSE)"""
    
    # Check if session exists
    if not session_manager.is_session_active(thread_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found or expired"
        )
    
    async def event_generator():
        """Generate SSE events"""
        try:
            # Prepare initial state
            initial_state: AgentState = {
                "messages": [{"role": "user", "content": request.message}],
                "current_query": "",
                "query_strategy": None,
                "retrieved_chunks": [],
                "retrieved_documents": [],
                "thread_id": thread_id,
                "user_id": session_manager.get_session(thread_id).get("user_id"),
                "iteration_count": 0,
                "needs_backtracking": False,
                "final_response": None
            }
            
            # Note: For true streaming, you'd need to modify the LLM to support streaming
            # For now, we'll simulate by running the agent and yielding the full response
            start_time = time.time()
            result = agent.invoke(initial_state)
            processing_time = (time.time() - start_time) * 1000
            
            # Yield response in chunks (simulated streaming)
            response = result["final_response"]
            chunk_size = 10  # words per chunk
            words = response.split()
            
            for i in range(0, len(words), chunk_size):
                chunk = " ".join(words[i:i+chunk_size])
                yield {
                    "event": "token",
                    "data": json.dumps({"content": chunk + " ", "delta": chunk + " "})
                }
            
            # Yield metadata
            yield {
                "event": "metadata",
                "data": json.dumps({
                    "retrieval_strategy": result.get("query_strategy"),
                    "chunks_retrieved": len(result.get("retrieved_chunks", []))
                })
            }
            
            # Yield done event
            yield {
                "event": "done",
                "data": json.dumps({
                    "message_id": str(hash(response)),
                    "processing_time_ms": processing_time
                })
            }
            
        except Exception as e:
            logger.error(f"Error in streaming: {e}")
            yield {
                "event": "error",
                "data": json.dumps({"error": str(e)})
            }
    
    return EventSourceResponse(event_generator())

@router.get("/sessions/{thread_id}/messages", response_model=ConversationHistoryResponse, dependencies=[Depends(verify_api_key)])
async def get_conversation_history(
    thread_id: str,
    limit: int = 20,
    offset: int = 0,
    session_manager=Depends(get_session_manager),
    supabase_client=Depends(get_supabase_client)
):
    """Get conversation history for a session"""
    
    # Check if session exists
    session = session_manager.get_session(thread_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )
    
    # Get messages with pagination
    result = supabase_client.table("messages").select(
        "id, role, content, created_at"
    ).eq("thread_id", thread_id).order(
        "created_at", desc=False
    ).range(offset, offset + limit - 1).execute()
    
    # Get total count
    count_result = supabase_client.table("messages").select(
        "id", count="exact"
    ).eq("thread_id", thread_id).execute()
    
    total = count_result.count if hasattr(count_result, 'count') else len(result.data)
    
    messages = [
        ConversationMessage(
            message_id=msg["id"],
            role=msg["role"],
            content=msg["content"],
            created_at=datetime.fromisoformat(msg["created_at"])
        )
        for msg in result.data
    ]
    
    return ConversationHistoryResponse(
        thread_id=thread_id,
        messages=messages,
        total=total,
        limit=limit,
        offset=offset
    )
```

#### 10.7 Create `src/api/routes/search.py` (Optional - for debugging)

**Instructions for Claude:**
- Direct search endpoints for testing

```python
from fastapi import APIRouter, Depends, HTTPException, status
from src.api.models.requests import SearchRequest
from src.api.models.responses import SearchResponse, SearchResult
from src.api.dependencies import get_rag_tool, verify_api_key
from src.utils.logger import setup_logger

router = APIRouter(prefix="/api/v1/search", tags=["search"])
logger = setup_logger(__name__)

@router.post("/semantic", response_model=SearchResponse, dependencies=[Depends(verify_api_key)])
async def semantic_search(
    request: SearchRequest,
    rag_tool=Depends(get_rag_tool)
):
    """Direct semantic search (for testing/debugging)"""
    try:
        chunks = rag_tool.retrieve(
            query=request.query,
            strategy="semantic",
            top_k=request.top_k
        )
        
        results = [
            SearchResult(
                chunk_id=chunk.chunk_id,
                content=chunk.content,
                url=chunk.url,
                has_code=chunk.has_code,
                similarity_score=None  # Would need to be returned from search
            )
            for chunk in chunks
        ]
        
        return SearchResponse(
            query=request.query,
            strategy="semantic",
            results=results,
            total=len(results)
        )
    except Exception as e:
        logger.error(f"Error in semantic search: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.post("/hybrid", response_model=SearchResponse, dependencies=[Depends(verify_api_key)])
async def hybrid_search(
    request: SearchRequest,
    rag_tool=Depends(get_rag_tool)
):
    """Direct hybrid search (for testing/debugging)"""
    try:
        chunks = rag_tool.retrieve(
            query=request.query,
            strategy="hybrid",
            top_k=request.top_k
        )
        
        results = [
            SearchResult(
                chunk_id=chunk.chunk_id,
                content=chunk.content,
                url=chunk.url,
                has_code=chunk.has_code
            )
            for chunk in chunks
        ]
        
        return SearchResponse(
            query=request.query,
            strategy="hybrid",
            results=results,
            total=len(results)
        )
    except Exception as e:
        logger.error(f"Error in hybrid search: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
```

#### 10.8 Create `src/api/main.py`

**Instructions for Claude:**
- Main FastAPI application
- Bring everything together

```python
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging

from src.api.dependencies import init_dependencies, get_config
from src.api.middleware import setup_cors, setup_rate_limiting, TimingMiddleware
from src.api.routes import health, chat, search
from src.api.models.responses import ErrorResponse
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    logger.info("Starting up application...")
    init_dependencies()
    logger.info("Dependencies initialized")
    yield
    # Shutdown
    logger.info("Shutting down application...")

# Create FastAPI app
app = FastAPI(
    title="LangGraph RAG Agent API",
    description="REST API for LangGraph documentation assistant",
    version="1.0.0",
    lifespan=lifespan
)

# Setup middleware
config = get_config() if get_config() else {}
if config:
    setup_cors(app, config)
setup_rate_limiting(app)
app.add_middleware(TimingMiddleware)

# Include routers
app.include_router(health.router)
app.include_router(chat.router)
app.include_router(search.router)  # Optional, for debugging

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error={
                "code": "INTERNAL_ERROR",
                "message": "An unexpected error occurred",
                "details": {}
            }
        ).dict()
    )

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "LangGraph RAG Agent API",
        "version": "1.0.0",
        "docs": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    
    config = get_config()
    uvicorn.run(
        "src.api.main:app",
        host=config["api"]["host"],
        port=config["api"]["port"],
        reload=config["api"]["reload"],
        workers=config["api"]["workers"]
    )
```

#### 10.9 Create `scripts/run_api.py`

**Instructions for Claude:**
- Script to run the FastAPI server

```python
#!/usr/bin/env python3
"""
Run the FastAPI server
"""

import uvicorn
from src.config.settings import load_config

def main():
    config = load_config()
    
    uvicorn.run(
        "src.api.main:app",
        host=config["api"]["host"],
        port=config["api"]["port"],
        reload=config["api"]["reload"],
        workers=config["api"]["workers"] if not config["api"]["reload"] else 1,
        log_level=config.get("log_level", "info").lower()
    )

if __name__ == "__main__":
    main()
```

Make it executable:
```bash
chmod +x scripts/run_api.py
```

---

## Testing Strategy

### Unit Tests

Create tests for each component:

1. **test_crawler.py**: Test sitemap parsing, URL extraction, content crawling
2. **test_chunker.py**: Test code block detection, sentence splitting, chunk generation
3. **test_embeddings.py**: Test embedding generation (with mock model)
4. **test_retrieval.py**: Test each search strategy
5. **test_agent.py**: Test node functions and graph execution
6. **test_api.py**: Test all API endpoints

### Integration Tests

Test end-to-end workflows:
- Full ingestion pipeline
- Query → Retrieval → Response
- Multi-turn conversations
- Session management
- API request/response cycles

### API Testing Example

Create `tests/test_api.py`:

```python
import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] in ["healthy", "degraded"]

def test_create_session():
    response = client.post(
        "/api/v1/chat/sessions",
        json={"user_id": "test_user"},
        headers={"X-API-Key": "your_test_api_key"}
    )
    assert response.status_code == 200
    assert "thread_id" in response.json()

def test_send_message():
    # First create a session
    session_response = client.post(
        "/api/v1/chat/sessions",
        json={"user_id": "test_user"},
        headers={"X-API-Key": "your_test_api_key"}
    )
    thread_id = session_response.json()["thread_id"]
    
    # Send a message
    message_response = client.post(
        f"/api/v1/chat/sessions/{thread_id}/messages",
        json={"message": "How do I create a StateGraph?"},
        headers={"X-API-Key": "your_test_api_key"}
    )
    assert message_response.status_code == 200
    assert "content" in message_response.json()
```

---

## Configuration Checklist

Before running, ensure you have:

- [ ] Supabase project created
- [ ] Database schema initialized (`scripts/setup_database.py`)
- [ ] pgvector extension enabled
- [ ] Environment variables configured (`.env`)
- [ ] Custom embedding model endpoint available
- [ ] Custom LLM endpoint available
- [ ] Re-ranker model (optional) configured

---

## Running the System

### 1. Setup Database
```bash
python scripts/setup_database.py
```

### 2. Crawl and Ingest Documentation
```bash
python scripts/crawl_and_ingest.py
```

### 3. Test Retrieval
```bash
python scripts/test_retrieval.py
```

### 4. Run Interactive Agent
```bash
python scripts/run_agent.py
```

### 5. Run FastAPI Server
```bash
# Development mode (with auto-reload)
python scripts/run_api.py

# Or directly with uvicorn
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Production mode (multiple workers)
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### 6. Test API Endpoints

```bash
# Health check
curl http://localhost:8000/api/v1/health

# Create session
curl -X POST http://localhost:8000/api/v1/chat/sessions \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key" \
  -d '{"user_id": "test_user"}'

# Send message
curl -X POST http://localhost:8000/api/v1/chat/sessions/{thread_id}/messages \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key" \
  -d '{"message": "How do I create a StateGraph?"}'
```

### 7. Access API Documentation

Once the server is running:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

---

## Key Implementation Notes

### Code Block Handling
- Always preserve code blocks intact
- Don't split mid-code
- Maintain language identifiers
- If code > max_chunk_size, keep as single chunk

### Sentence Boundaries
- Use NLTK's `sent_tokenize` or spaCy
- Handle abbreviations carefully
- Avoid splitting URLs or code snippets

### Metadata Tracking
- Every chunk must link to parent document (Level 1)
- Store URL for source attribution
- Track sequential relationships (prev/next chunk)

### Vector Indexes
- Use ivfflat for medium datasets (< 1M vectors)
- Use hnsw for larger datasets (better query performance)
- Tune list count for ivfflat (rule of thumb: sqrt(num_rows))

### Session Management
- Each user session = unique thread_id
- Store in `sessions` table
- Messages linked via foreign key
- Implement timeout cleanup

### Multi-User Support
- Thread IDs prevent data leakage between users
- Each agent invocation gets isolated state
- Messages queried by thread_id only

### FastAPI Best Practices
- Use dependency injection for shared resources
- Initialize expensive resources (DB, models) at startup
- Implement proper error handling with standard responses
- Add request/response validation with Pydantic
- Enable CORS for web client access
- Implement rate limiting to prevent abuse
- Use API key authentication for security
- Support streaming for better UX
- Add comprehensive logging and monitoring

### API Security
- Always use HTTPS in production
- Store API keys securely (environment variables, secret manager)
- Implement rate limiting per user/IP
- Validate all input data
- Add request size limits
- Enable CORS only for trusted origins
- Log all API access for auditing

---

## Customization Points

You will need to customize:

1. **Embedding Model** (`src/embeddings/custom_embedder.py`)
   - Update API endpoint format
   - Adjust request/response parsing
   - Set embedding dimension

2. **LLM** (`src/llm/custom_llm.py`)
   - Update API endpoint format
   - Adjust prompt template if needed
   - Configure parameters (temperature, max_tokens)

3. **Re-ranker** (optional in `src/retrieval/hybrid_search.py`)
   - Add re-ranker API integration
   - Implement re-ranking logic

4. **Query Analysis** (`src/retrieval/rag_tool.py`)
   - Refine strategy selection heuristics
   - Add custom query preprocessing

---

## Success Criteria

Your implementation should:

✅ Successfully crawl LangGraph documentation
✅ Generate smart chunks respecting code blocks and sentences
✅ Store two-level chunks in Supabase
✅ Perform fast vector similarity search (< 2s per query)
✅ Support multiple retrieval strategies
✅ Maintain conversation context across turns
✅ Handle multiple concurrent users with isolated sessions
✅ Provide relevant, accurate responses

---

## Troubleshooting

### Common Issues

**Crawling fails:**
- Check robots.txt compliance
- Add delays between requests
- Verify sitemap URL

**Chunking splits code:**
- Review code block detection patterns
- Check delimiter matching
- Adjust chunk size if needed

**Slow retrieval:**
- Verify vector indexes are created
- Consider using hnsw instead of ivfflat
- Tune index parameters

**Session state issues:**
- Check thread_id is properly passed
- Verify foreign key constraints
- Review session timeout logic

**Embeddings fail:**
- Test embedding endpoint separately
- Check batch size (reduce if OOM)
- Verify API authentication

---

## Next Steps After Implementation

1. Evaluate retrieval quality (precision@k, recall@k)
2. Fine-tune chunk sizes based on performance
3. Optimize database indexes
4. Add caching for frequent queries
5. Implement incremental updates (re-crawl changed pages)
6. Add user feedback collection
7. Create evaluation dataset for testing

---

## Additional Resources

### Example UI Integration

Here's how a frontend would integrate with the API:

```javascript
// JavaScript/TypeScript example for frontend integration

class ChatClient {
  constructor(apiUrl, apiKey) {
    this.apiUrl = apiUrl;
    this.apiKey = apiKey;
  }

  async createSession(userId = null) {
    const response = await fetch(`${this.apiUrl}/api/v1/chat/sessions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-API-Key': this.apiKey
      },
      body: JSON.stringify({ user_id: userId })
    });
    
    const data = await response.json();
    return data.thread_id;
  }

  async sendMessage(threadId, message) {
    const response = await fetch(
      `${this.apiUrl}/api/v1/chat/sessions/${threadId}/messages`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-API-Key': this.apiKey
        },
        body: JSON.stringify({ message })
      }
    );
    
    return await response.json();
  }

  async sendMessageStream(threadId, message, onToken, onComplete) {
    const response = await fetch(
      `${this.apiUrl}/api/v1/chat/sessions/${threadId}/messages/stream`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-API-Key': this.apiKey
        },
        body: JSON.stringify({ message })
      }
    );

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value);
      const lines = chunk.split('\n');

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = JSON.parse(line.slice(6));
          
          if (line.startsWith('event: token')) {
            onToken(data.delta);
          } else if (line.startsWith('event: done')) {
            onComplete(data);
          }
        }
      }
    }
  }

  async getHistory(threadId, limit = 20) {
    const response = await fetch(
      `${this.apiUrl}/api/v1/chat/sessions/${threadId}/messages?limit=${limit}`,
      {
        headers: {
          'X-API-Key': this.apiKey
        }
      }
    );
    
    return await response.json();
  }
}

// Usage example
const client = new ChatClient('http://localhost:8000', 'your_api_key');

// Create session and send message
const threadId = await client.createSession('user123');
const response = await client.sendMessage(threadId, 'How do I create a StateGraph?');
console.log(response.content);

// Streaming example
await client.sendMessageStream(
  threadId,
  'Explain state management in LangGraph',
  (token) => console.log(token),  // Called for each token
  (data) => console.log('Done:', data)  // Called when complete
);
```

### Python Client Example

```python
import requests

class ChatClient:
    def __init__(self, api_url: str, api_key: str):
        self.api_url = api_url
        self.api_key = api_key
        self.headers = {
            "Content-Type": "application/json",
            "X-API-Key": api_key
        }
    
    def create_session(self, user_id: str = None) -> str:
        response = requests.post(
            f"{self.api_url}/api/v1/chat/sessions",
            json={"user_id": user_id},
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()["thread_id"]
    
    def send_message(self, thread_id: str, message: str) -> dict:
        response = requests.post(
            f"{self.api_url}/api/v1/chat/sessions/{thread_id}/messages",
            json={"message": message},
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()
    
    def get_history(self, thread_id: str, limit: int = 20) -> dict:
        response = requests.get(
            f"{self.api_url}/api/v1/chat/sessions/{thread_id}/messages",
            params={"limit": limit},
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()

# Usage
client = ChatClient("http://localhost:8000", "your_api_key")
thread_id = client.create_session("user123")
response = client.send_message(thread_id, "How do I create a StateGraph?")
print(response["content"])
```

### Documentation Links
- LangGraph Documentation: https://langchain-ai.github.io/langgraph/
- Crawl4AI: https://github.com/unclecode/crawl4ai
- Supabase pgvector: https://supabase.com/docs/guides/ai
- LangChain Tools: https://python.langchain.com/docs/modules/tools/
- FastAPI: https://fastapi.tiangolo.com/
- Server-Sent Events: https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events

---

**Good luck with your implementation!** 

Remember:
- Start with small-scale tests (crawl 10-20 pages first)
- Test each component independently before integration
- Monitor performance and optimize as you go
- Keep chunks metadata-rich for better retrieval

If you encounter issues, refer back to the PRD for detailed requirements.