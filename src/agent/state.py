"""
Agent state definitions for the LangGraph RAG agent.

Provides TypedDict state schemas with pre-allocated fields
for optimal performance and type safety.
"""

from enum import Enum
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class QueryType(str, Enum):
    """Classification of user query type for retriever selection."""

    EXACT_MATCH = "exact_match"  # Function names, identifiers -> BM25
    CONCEPTUAL = "conceptual"  # What is, explain, why -> Semantic
    CODE_EXAMPLE = "code_example"  # How to, implement -> Hybrid
    TUTORIAL = "tutorial"  # Step-by-step guides -> Hybrid
    GENERAL = "general"  # Default -> Hybrid


class RetrieverChoice(str, Enum):
    """Selected retriever strategy based on query analysis."""

    SEMANTIC = "semantic"
    BM25 = "bm25"
    HYBRID = "hybrid"


class AgentState(TypedDict, total=False):
    """
    State schema for the LangGraph RAG agent.

    Uses TypedDict for type safety and pre-allocated fields
    for optimal performance during graph execution.

    Attributes:
        thread_id: Session thread identifier
        user_id: Optional user identifier
        user_query: Current user query
        query_embedding: Cached embedding vector for the query
        query_type: Classified query type (EXACT_MATCH, CONCEPTUAL, etc.)
        retriever_choice: Selected retriever (SEMANTIC, BM25, HYBRID)
        messages: Conversation history as list of dicts
        chunks: Retrieved chunks from retrieval
        context: Formatted context string for LLM
        relevance_score: Average relevance score of retrieved chunks
        response: Generated response
        cached: Whether response was from cache
        timings: Node execution times for observability
        error: Error message if any
    """

    # Session (pre-loaded)
    thread_id: str
    user_id: Optional[str]

    # Query (computed in parallel)
    user_query: str
    query_embedding: List[float]
    query_type: QueryType
    retriever_choice: RetrieverChoice

    # Memory (async loaded) - uses add_messages reducer for chat interface
    messages: Annotated[List[AnyMessage], add_messages]

    # Retrieval (fast path)
    chunks: List[Dict[str, Any]]
    context: str
    relevance_score: float

    # Response (streamed)
    response: str
    cached: bool

    # Timing (for observability)
    timings: Dict[str, float]

    # Error handling
    error: Optional[str]


def create_initial_state(
    thread_id: str,
    user_query: str,
    user_id: Optional[str] = None,
    query_embedding: Optional[List[float]] = None,
) -> AgentState:
    """
    Create initial agent state with default values.

    Args:
        thread_id: Session thread identifier
        user_query: User's query
        user_id: Optional user identifier
        query_embedding: Pre-computed query embedding (optional)

    Returns:
        Initialized AgentState with defaults
    """
    return AgentState(
        thread_id=thread_id,
        user_id=user_id,
        user_query=user_query,
        query_embedding=query_embedding or [],
        query_type=QueryType.GENERAL,
        retriever_choice=RetrieverChoice.HYBRID,
        messages=[],
        chunks=[],
        context="",
        relevance_score=0.0,
        response="",
        cached=False,
        timings={},
        error=None,
    )


def merge_state(state: AgentState, updates: Dict[str, Any]) -> AgentState:
    """
    Merge updates into state (returns new state dict).

    Args:
        state: Current state
        updates: Updates to merge

    Returns:
        New state with updates merged
    """
    return {**state, **updates}
