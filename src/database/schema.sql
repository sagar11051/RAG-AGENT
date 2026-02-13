-- =============================================================================
-- LangGraph Documentation RAG Agent - Database Schema
-- =============================================================================
-- This schema supports the two-level document storage strategy:
-- Level 1: Full page documents (for backtracking retrieval)
-- Level 2: Smart chunks with embeddings (for semantic search)
-- =============================================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";

-- =============================================================================
-- Table 1: documents (Level 1 - Full Page Content)
-- =============================================================================
-- Stores complete page content for backtracking retrieval
-- Each document represents one crawled page

CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    url TEXT UNIQUE NOT NULL,
    title TEXT,
    content TEXT NOT NULL,
    content_hash TEXT UNIQUE,
    total_tokens INTEGER,
    crawl_date TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for documents table
CREATE INDEX IF NOT EXISTS idx_documents_url ON documents(url);
CREATE INDEX IF NOT EXISTS idx_documents_content_hash ON documents(content_hash);
CREATE INDEX IF NOT EXISTS idx_documents_crawl_date ON documents(crawl_date);

-- =============================================================================
-- Table 2: chunks (Level 2 - Smart Chunks with Embeddings)
-- =============================================================================
-- Stores chunked content with embeddings for semantic search
-- Dimension is 1024 for OVH BGE-M3 model

CREATE TABLE IF NOT EXISTS chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    parent_doc_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding VECTOR(1024),  -- OVH BGE-M3 produces 1024-dimensional vectors
    tokens INTEGER,
    has_code BOOLEAN DEFAULT FALSE,
    section_title TEXT,
    previous_chunk_id UUID REFERENCES chunks(id) ON DELETE SET NULL,
    next_chunk_id UUID REFERENCES chunks(id) ON DELETE SET NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE(parent_doc_id, chunk_index)
);

-- Indexes for chunks table
CREATE INDEX IF NOT EXISTS idx_chunks_parent_doc ON chunks(parent_doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_has_code ON chunks(has_code);
CREATE INDEX IF NOT EXISTS idx_chunks_section_title ON chunks(section_title);

-- Vector similarity search index using IVFFlat
-- Note: This index should be created AFTER inserting data for best performance
-- The 'lists' parameter should be approximately sqrt(num_rows)
CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Alternative: HNSW index (better query performance, slower build)
-- CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw ON chunks
-- USING hnsw (embedding vector_cosine_ops)
-- WITH (m = 16, ef_construction = 64);

-- =============================================================================
-- Table 3: sessions (Chat Sessions/Threads)
-- =============================================================================
-- Stores chat session metadata for multi-user support

CREATE TABLE IF NOT EXISTS sessions (
    thread_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id TEXT,
    title TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ,
    metadata JSONB DEFAULT '{}'
);

-- Indexes for sessions table
CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_created_at ON sessions(created_at);
CREATE INDEX IF NOT EXISTS idx_sessions_expires_at ON sessions(expires_at);

-- =============================================================================
-- Table 4: messages (Conversation History)
-- =============================================================================
-- Stores all messages within sessions

CREATE TABLE IF NOT EXISTS messages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    thread_id UUID REFERENCES sessions(thread_id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

-- Indexes for messages table
CREATE INDEX IF NOT EXISTS idx_messages_thread_id ON messages(thread_id);
CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at);
CREATE INDEX IF NOT EXISTS idx_messages_role ON messages(role);

-- =============================================================================
-- Functions for Vector Search
-- =============================================================================

-- Function: Semantic search using cosine similarity
CREATE OR REPLACE FUNCTION match_chunks(
    query_embedding VECTOR(1024),
    match_count INTEGER DEFAULT 5,
    filter_has_code BOOLEAN DEFAULT NULL
)
RETURNS TABLE (
    id UUID,
    parent_doc_id UUID,
    chunk_index INTEGER,
    content TEXT,
    section_title TEXT,
    has_code BOOLEAN,
    tokens INTEGER,
    similarity FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        c.id,
        c.parent_doc_id,
        c.chunk_index,
        c.content,
        c.section_title,
        c.has_code,
        c.tokens,
        1 - (c.embedding <=> query_embedding) AS similarity
    FROM chunks c
    WHERE
        c.embedding IS NOT NULL
        AND (filter_has_code IS NULL OR c.has_code = filter_has_code)
    ORDER BY c.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- Function: Get chunks with parent document info
CREATE OR REPLACE FUNCTION get_chunks_with_context(
    chunk_ids UUID[]
)
RETURNS TABLE (
    chunk_id UUID,
    chunk_content TEXT,
    chunk_section_title TEXT,
    chunk_has_code BOOLEAN,
    doc_id UUID,
    doc_url TEXT,
    doc_title TEXT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        c.id AS chunk_id,
        c.content AS chunk_content,
        c.section_title AS chunk_section_title,
        c.has_code AS chunk_has_code,
        d.id AS doc_id,
        d.url AS doc_url,
        d.title AS doc_title
    FROM chunks c
    JOIN documents d ON c.parent_doc_id = d.id
    WHERE c.id = ANY(chunk_ids);
END;
$$;

-- =============================================================================
-- Triggers for Updated Timestamps
-- =============================================================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger for documents table
DROP TRIGGER IF EXISTS update_documents_updated_at ON documents;
CREATE TRIGGER update_documents_updated_at
    BEFORE UPDATE ON documents
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Trigger for sessions table
DROP TRIGGER IF EXISTS update_sessions_updated_at ON sessions;
CREATE TRIGGER update_sessions_updated_at
    BEFORE UPDATE ON sessions
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- =============================================================================
-- Row Level Security (RLS) Policies - Optional
-- =============================================================================
-- Uncomment and customize if you need user-based access control

-- ALTER TABLE sessions ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE messages ENABLE ROW LEVEL SECURITY;

-- CREATE POLICY "Users can access their own sessions" ON sessions
--     FOR ALL USING (auth.uid()::text = user_id);

-- CREATE POLICY "Users can access messages in their sessions" ON messages
--     FOR ALL USING (
--         thread_id IN (
--             SELECT thread_id FROM sessions WHERE user_id = auth.uid()::text
--         )
--     );

-- =============================================================================
-- Comments for Documentation
-- =============================================================================

COMMENT ON TABLE documents IS 'Level 1 storage: Full page content for backtracking retrieval';
COMMENT ON TABLE chunks IS 'Level 2 storage: Smart chunks with embeddings for semantic search';
COMMENT ON TABLE sessions IS 'Chat sessions for multi-user support';
COMMENT ON TABLE messages IS 'Conversation history within sessions';

COMMENT ON COLUMN chunks.embedding IS 'Vector embedding from OVH BGE-M3 model (1024 dimensions)';
COMMENT ON COLUMN chunks.has_code IS 'Flag indicating if chunk contains code blocks';
COMMENT ON COLUMN chunks.section_title IS 'Section header the chunk belongs to';
