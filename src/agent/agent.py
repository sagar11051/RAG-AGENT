"""
Main RAGAgent class for the LangGraph RAG agent.

Provides the high-level interface for interacting with the agent:
- chat(): Non-streaming chat with caching
- chat_stream(): Streaming chat with immediate token delivery
- Session management with automatic persistence
"""

import asyncio
import time
import uuid
from typing import Any, AsyncGenerator, Dict, Optional

from src.agent.cache import get_caches
from src.agent.graph import get_default_graph, get_fast_graph
from src.agent.memory import get_memory_manager
from src.agent.nodes import response_generation_stream
from src.agent.state import AgentState, QueryType, RetrieverChoice, create_initial_state
from src.agent.tools import (
    analyze_query_fast,
    build_messages,
    format_chunks_fast,
    get_fallback_response,
    get_retriever_for_choice,
    should_fallback,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RAGAgent:
    """
    Production-ready RAG agent with latency optimizations.

    Features:
    - Semantic query caching for instant responses to similar queries
    - Embedding caching to avoid redundant API calls
    - Parallel initialization (memory + embedding + query analysis)
    - Intelligent retriever selection based on query type
    - Streaming response generation
    - Background persistence (fire-and-forget)

    Usage:
        agent = RAGAgent()
        response = await agent.chat("What is StateGraph?")

        # Or with streaming
        async for token in agent.chat_stream("How to add memory?"):
            print(token, end="")
    """

    def __init__(
        self,
        thread_id: Optional[str] = None,
        user_id: Optional[str] = None,
        use_persistence: bool = True,
    ):
        """
        Initialize RAG agent.

        Args:
            thread_id: Session thread ID (None to create new)
            user_id: User identifier for session
            use_persistence: Whether to persist conversations
        """
        self._caches = get_caches()
        self._memory = get_memory_manager()

        # Get or create session
        if thread_id:
            self.thread_id = thread_id
        else:
            session = self._memory.get_or_create_session(user_id=user_id)
            self.thread_id = session.thread_id

        self.user_id = user_id
        self._use_persistence = use_persistence

        # Get compiled graph
        if use_persistence:
            self._graph = get_default_graph()
        else:
            self._graph = get_fast_graph()

        logger.info(
            f"Initialized RAGAgent (thread_id={self.thread_id}, "
            f"persistence={use_persistence})"
        )

    @property
    def session(self):
        """Get current session."""
        return self._memory.get_session(self.thread_id)

    async def chat(self, query: str) -> Dict[str, Any]:
        """
        Non-streaming chat with caching and automatic tool selection.

        Args:
            query: User's question

        Returns:
            Dict with:
                - response: Generated response text
                - cached: Whether response was from cache
                - latency_ms: Total latency in milliseconds
                - query_type: Classified query type
                - retriever_used: Retriever strategy used
                - timings: Per-node timing breakdown
        """
        start = time.perf_counter()

        # Get embedding (cached if possible)
        query_embedding = await self._caches.get_or_compute_embedding(query)

        # Check semantic cache first
        cached = self._caches.check_semantic_cache(query_embedding)
        if cached:
            response, similarity = cached
            logger.info(f"Cache hit (similarity={similarity:.3f})")
            return {
                "response": response,
                "cached": True,
                "cache_similarity": similarity,
                "latency_ms": (time.perf_counter() - start) * 1000,
                "query_type": "cached",
                "retriever_used": "cached",
                "timings": {},
            }

        # Create initial state
        initial_state = create_initial_state(
            thread_id=self.thread_id,
            user_query=query,
            user_id=self.user_id,
            query_embedding=query_embedding,
        )

        # Run the graph
        try:
            result = await self._graph.ainvoke(initial_state)
        except Exception as e:
            logger.error(f"Graph execution failed: {e}")
            return {
                "response": "I encountered an error. Please try again.",
                "cached": False,
                "latency_ms": (time.perf_counter() - start) * 1000,
                "error": str(e),
            }

        latency_ms = (time.perf_counter() - start) * 1000

        # Update semantic cache in background
        response = result.get("response", "")
        if response and not result.get("error"):
            asyncio.create_task(
                self._update_cache_async(query, query_embedding, response)
            )

        return {
            "response": response,
            "cached": False,
            "latency_ms": latency_ms,
            "query_type": result.get("query_type", QueryType.GENERAL).value
            if isinstance(result.get("query_type"), QueryType)
            else str(result.get("query_type", "general")),
            "retriever_used": result.get("retriever_choice", RetrieverChoice.HYBRID).value
            if isinstance(result.get("retriever_choice"), RetrieverChoice)
            else str(result.get("retriever_choice", "hybrid")),
            "relevance_score": result.get("relevance_score", 0.0),
            "timings": result.get("timings", {}),
            "chunks_count": len(result.get("chunks", [])),
        }

    async def chat_stream(self, query: str) -> AsyncGenerator[str, None]:
        """
        Streaming chat - tokens sent immediately.

        Args:
            query: User's question

        Yields:
            Response tokens as they are generated
        """
        start = time.perf_counter()

        # Get embedding (cached if possible)
        query_embedding = await self._caches.get_or_compute_embedding(query)

        # Check semantic cache
        cached = self._caches.check_semantic_cache(query_embedding)
        if cached:
            response, _ = cached
            # Stream cached response
            for word in response.split():
                yield word + " "
            return

        # Run parallel_init and retrieve first
        from src.agent.nodes import parallel_init, retrieve_chunks

        initial_state = create_initial_state(
            thread_id=self.thread_id,
            user_query=query,
            user_id=self.user_id,
            query_embedding=query_embedding,
        )

        # Execute init and retrieval
        state = await parallel_init(initial_state)
        state = await retrieve_chunks(state)

        # Stream response
        full_response = ""
        async for event in response_generation_stream(state):
            if event.get("type") == "token":
                token = event.get("content", "")
                full_response += token
                yield token
            elif event.get("type") == "final":
                full_response = event.get("response", full_response)

        # Persist in background
        if self._use_persistence and full_response:
            asyncio.create_task(
                self._persist_async(query, full_response, state)
            )
            asyncio.create_task(
                self._update_cache_async(query, query_embedding, full_response)
            )

    async def _update_cache_async(
        self,
        query: str,
        embedding: list,
        response: str,
    ) -> None:
        """Update semantic cache asynchronously."""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self._caches.update_response_cache(query, embedding, response),
            )
        except Exception as e:
            logger.warning(f"Failed to update cache: {e}")

    async def _persist_async(
        self,
        query: str,
        response: str,
        state: AgentState,
    ) -> None:
        """Persist conversation turn asynchronously."""
        try:
            await self._memory.add_turn_async(
                thread_id=self.thread_id,
                user_message=query,
                assistant_response=response,
                metadata={
                    "query_type": state.get("query_type", QueryType.GENERAL).value
                    if isinstance(state.get("query_type"), QueryType)
                    else str(state.get("query_type", "general")),
                    "retriever": state.get("retriever_choice", RetrieverChoice.HYBRID).value
                    if isinstance(state.get("retriever_choice"), RetrieverChoice)
                    else str(state.get("retriever_choice", "hybrid")),
                    "relevance_score": state.get("relevance_score", 0.0),
                },
            )
        except Exception as e:
            logger.warning(f"Failed to persist turn: {e}")

    def get_history(self, max_messages: int = 20) -> list:
        """
        Get conversation history.

        Args:
            max_messages: Maximum messages to return

        Returns:
            List of message dicts with role and content
        """
        return self._memory.get_history(self.thread_id, max_messages)

    def new_session(self, user_id: Optional[str] = None) -> str:
        """
        Create a new session.

        Args:
            user_id: User identifier

        Returns:
            New thread_id
        """
        session = self._memory.get_or_create_session(user_id=user_id or self.user_id)
        self.thread_id = session.thread_id
        logger.info(f"Created new session: {self.thread_id}")
        return self.thread_id

    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        return self._caches.get_stats()


# Convenience function for quick usage
async def quick_chat(query: str, thread_id: Optional[str] = None) -> str:
    """
    Quick chat function for simple usage.

    Args:
        query: User's question
        thread_id: Optional session ID

    Returns:
        Response text
    """
    agent = RAGAgent(thread_id=thread_id, use_persistence=False)
    result = await agent.chat(query)
    return result["response"]


# Singleton agent for API usage
_default_agent: Optional[RAGAgent] = None


def get_default_agent() -> RAGAgent:
    """Get singleton RAGAgent instance."""
    global _default_agent
    if _default_agent is None:
        _default_agent = RAGAgent()
    return _default_agent


def reset_agent() -> None:
    """Reset singleton agent (for testing)."""
    global _default_agent
    _default_agent = None
