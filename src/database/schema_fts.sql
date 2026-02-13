-- =============================================================================
-- Full-Text Search Extension for BM25 Retrieval
-- =============================================================================
-- Adds tsvector columns and indexes for PostgreSQL full-text search
-- This enables BM25-style keyword matching alongside vector similarity
-- =============================================================================

-- Add tsvector column to chunks table for full-text search
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS content_tsv TSVECTOR;

-- Create trigger to auto-populate tsvector on insert/update
CREATE OR REPLACE FUNCTION chunks_tsv_trigger() RETURNS trigger AS $$
BEGIN
    NEW.content_tsv := to_tsvector('english', COALESCE(NEW.content, ''));
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Drop existing trigger if it exists and recreate
DROP TRIGGER IF EXISTS chunks_content_tsv_update ON chunks;
CREATE TRIGGER chunks_content_tsv_update
    BEFORE INSERT OR UPDATE OF content ON chunks
    FOR EACH ROW
    EXECUTE FUNCTION chunks_tsv_trigger();

-- Create GIN index for fast full-text search
CREATE INDEX IF NOT EXISTS idx_chunks_content_tsv ON chunks USING gin(content_tsv);

-- Backfill existing rows with tsvector data
UPDATE chunks SET content_tsv = to_tsvector('english', COALESCE(content, ''))
WHERE content_tsv IS NULL;

-- =============================================================================
-- Function: BM25-style full-text search using ts_rank
-- =============================================================================
-- PostgreSQL's ts_rank provides TF-IDF style ranking similar to BM25

-- Drop existing function if return type changed
DROP FUNCTION IF EXISTS match_chunks_fts(TEXT, INTEGER, BOOLEAN);

CREATE OR REPLACE FUNCTION match_chunks_fts(
    search_query TEXT,
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
    rank REAL
)
LANGUAGE plpgsql
AS $$
DECLARE
    tsquery_val TSQUERY;
BEGIN
    -- Convert search query to tsquery with OR between words for broader matching
    -- This handles multi-word queries more gracefully
    tsquery_val := plainto_tsquery('english', search_query);

    RETURN QUERY
    SELECT
        c.id,
        c.parent_doc_id,
        c.chunk_index,
        c.content,
        c.section_title,
        c.has_code,
        c.tokens,
        ts_rank_cd(c.content_tsv, tsquery_val, 32) AS rank  -- 32 = normalization by document length
    FROM chunks c
    WHERE
        c.content_tsv @@ tsquery_val
        AND (filter_has_code IS NULL OR c.has_code = filter_has_code)
    ORDER BY ts_rank_cd(c.content_tsv, tsquery_val, 32) DESC
    LIMIT match_count;
END;
$$;

-- =============================================================================
-- Function: Hybrid search combining vector + full-text
-- =============================================================================
-- Returns both similarity and rank scores for client-side fusion

-- Drop existing function if return type changed
DROP FUNCTION IF EXISTS match_chunks_hybrid(VECTOR(1024), TEXT, INTEGER, BOOLEAN);

CREATE OR REPLACE FUNCTION match_chunks_hybrid(
    query_embedding VECTOR(1024),
    search_query TEXT,
    match_count INTEGER DEFAULT 10,
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
    similarity REAL,
    fts_rank REAL
)
LANGUAGE plpgsql
AS $$
DECLARE
    tsquery_val TSQUERY;
BEGIN
    tsquery_val := plainto_tsquery('english', search_query);

    RETURN QUERY
    SELECT
        c.id,
        c.parent_doc_id,
        c.chunk_index,
        c.content,
        c.section_title,
        c.has_code,
        c.tokens,
        (1 - (c.embedding <=> query_embedding))::REAL AS similarity,
        COALESCE(ts_rank_cd(c.content_tsv, tsquery_val, 32), 0.0::REAL) AS fts_rank
    FROM chunks c
    WHERE
        c.embedding IS NOT NULL
        AND (filter_has_code IS NULL OR c.has_code = filter_has_code)
    ORDER BY c.embedding <=> query_embedding
    LIMIT match_count * 2;  -- Get more results for fusion
END;
$$;

-- =============================================================================
-- Comments
-- =============================================================================

COMMENT ON COLUMN chunks.content_tsv IS 'Full-text search vector for BM25-style retrieval';
COMMENT ON FUNCTION match_chunks_fts IS 'BM25-style full-text search using PostgreSQL ts_rank';
COMMENT ON FUNCTION match_chunks_hybrid IS 'Hybrid search returning both vector similarity and FTS rank';
