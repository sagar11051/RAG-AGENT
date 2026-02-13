"""
Agent module for LangGraph-based conversational RAG agent.

This module provides a production-ready RAG agent with:
- Semantic query caching for instant responses to similar queries
- Embedding caching to avoid redundant API calls
- Parallel initialization (memory + embedding + query analysis)
- Intelligent retriever selection based on query type
- Streaming response generation
- Background persistence (fire-and-forget)

Example Usage:
    from src.agent import RAGAgent

    # Create agent
    agent = RAGAgent()

    # Non-streaming chat
    result = await agent.chat("What is StateGraph?")
    print(result["response"])

    # Streaming chat
    async for token in agent.chat_stream("How to add memory?"):
        print(token, end="")
"""

from src.agent.agent import (
    RAGAgent,
    get_default_agent,
    quick_chat,
    reset_agent,
)
from src.agent.cache import (
    AgentCaches,
    get_caches,
    reset_caches,
)
from src.agent.graph import (
    build_rag_graph,
    compile_rag_graph,
    get_default_graph,
    get_fast_graph,
    reset_graphs,
)
from src.agent.memory import (
    MemoryManager,
    Session,
    get_memory_manager,
    reset_memory_manager,
)
from src.agent.nodes import (
    parallel_init,
    persist_memory,
    response_generation,
    response_generation_stream,
    retrieve_chunks,
)
from src.agent.state import (
    AgentState,
    QueryType,
    RetrieverChoice,
    create_initial_state,
    merge_state,
)
from src.agent.tools import (
    analyze_query_fast,
    build_messages,
    compute_relevance_score,
    format_chunks_fast,
    get_fallback_response,
    get_retriever_for_choice,
    get_system_prompt,
    should_fallback,
)

__all__ = [
    # Main agent
    "RAGAgent",
    "get_default_agent",
    "quick_chat",
    "reset_agent",
    # State
    "AgentState",
    "QueryType",
    "RetrieverChoice",
    "create_initial_state",
    "merge_state",
    # Caches
    "AgentCaches",
    "get_caches",
    "reset_caches",
    # Memory
    "MemoryManager",
    "Session",
    "get_memory_manager",
    "reset_memory_manager",
    # Graph
    "build_rag_graph",
    "compile_rag_graph",
    "get_default_graph",
    "get_fast_graph",
    "reset_graphs",
    # Nodes
    "parallel_init",
    "retrieve_chunks",
    "response_generation",
    "response_generation_stream",
    "persist_memory",
    # Tools
    "analyze_query_fast",
    "build_messages",
    "compute_relevance_score",
    "format_chunks_fast",
    "get_fallback_response",
    "get_retriever_for_choice",
    "get_system_prompt",
    "should_fallback",
]
