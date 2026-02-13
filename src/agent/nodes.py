"""
LangGraph node implementations for the RAG agent.

Nodes are optimized for low latency:
- parallel_init: Parallel execution of memory, embedding, and query analysis
- retrieve_chunks: Execute retrieval using selected strategy
- response_generation: Stream response to client with background persistence
"""

import asyncio
import time
from typing import Any, AsyncGenerator, Dict, List

from langchain_core.messages import AIMessage, HumanMessage

from src.agent.cache import get_caches
from src.agent.memory import get_memory_manager
from src.agent.state import AgentState, QueryType, RetrieverChoice
from src.agent.tools import (
    analyze_query_fast,
    build_messages,
    classify_query_complexity,
    compute_relevance_score,
    extract_text_from_content,
    format_chunks_fast,
    get_fallback_response,
    get_retriever_for_choice,
    should_fallback,
)
from src.llm.ovh_llm import OVHLLM, get_llm_client
from src.utils.logger import get_logger

logger = get_logger(__name__)


async def parallel_init(state: AgentState) -> AgentState:
    """
    Parallel initialization node.

    Executes in parallel:
    1. Load conversation history (async DB)
    2. Get/compute query embedding (cached)
    3. Analyze query type and select retriever (fast regex, no LLM)

    Latency: max(memory_load, embed, analyze) instead of sum
    For greetings: Skips embedding entirely for faster response.

    Args:
        state: Current agent state

    Returns:
        Updated state with memory, embedding, query_type, and retriever_choice
    """
    start = time.perf_counter()

    thread_id = state.get("thread_id", "")

    # Extract user query from messages (for chat interface) or from user_query field
    messages = state.get("messages", [])
    raw_user_query = state.get("user_query", "")

    # ALWAYS ensure user_query is a proper string
    # First try to use user_query from state if it's a valid string
    if isinstance(raw_user_query, str) and raw_user_query.strip():
        user_query = raw_user_query
    # Otherwise extract from messages
    elif messages:
        # Get the last human message content using the robust helper
        last_msg = messages[-1]
        user_query = extract_text_from_content(last_msg)
    # If user_query was set but not a string, convert it
    elif raw_user_query:
        user_query = extract_text_from_content(raw_user_query)
    else:
        user_query = ""

    # Final safety check - ensure it's always a string
    if not isinstance(user_query, str):
        logger.warning(f"user_query still not string after extraction: {type(user_query)}")
        user_query = str(user_query) if user_query else ""

    logger.debug(f"Extracted user_query: '{user_query[:50]}...' from messages={bool(messages)}")

    query_embedding = state.get("query_embedding")

    # FAST PATH: Check if greeting first (< 1ms regex check)
    # Skip embedding entirely for greetings to reduce latency
    complexity = classify_query_complexity(user_query)
    if complexity == "greeting":
        elapsed = (time.perf_counter() - start) * 1000
        logger.info(f"parallel_init (greeting fast path) completed in {elapsed:.0f}ms")
        return {
            **state,
            "user_query": user_query,
            "query_embedding": [],  # No embedding needed for greetings
            "query_type": QueryType.GENERAL,
            "retriever_choice": RetrieverChoice.HYBRID,
            "timings": {"parallel_init": elapsed},
        }

    caches = get_caches()

    # Define async tasks
    async def get_embedding_task():
        """Get or compute query embedding."""
        if query_embedding:
            return query_embedding
        return await caches.get_or_compute_embedding(user_query)

    async def analyze_query_task():
        """Analyze query (sync but fast, wrapped for gather)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, analyze_query_fast, user_query)

    # Run tasks in parallel (memory comes from chat interface messages)
    try:
        embedding, (query_type, retriever_choice) = await asyncio.gather(
            get_embedding_task(),
            analyze_query_task(),
        )
    except Exception as e:
        logger.error(f"parallel_init failed: {e}")
        # Return with defaults on error
        return {
            **state,
            "user_query": user_query,
            "query_embedding": [],
            "query_type": QueryType.GENERAL,
            "retriever_choice": RetrieverChoice.HYBRID,
            "timings": {"parallel_init": (time.perf_counter() - start) * 1000},
            "error": str(e),
        }

    elapsed = (time.perf_counter() - start) * 1000
    logger.info(
        f"parallel_init completed in {elapsed:.0f}ms "
        f"(type={query_type.value}, retriever={retriever_choice.value})"
    )

    return {
        **state,
        "user_query": user_query,
        "query_embedding": embedding,
        "query_type": query_type,
        "retriever_choice": retriever_choice,
        "timings": {**state.get("timings", {}), "parallel_init": elapsed},
    }


async def retrieve_chunks(state: AgentState) -> AgentState:
    """
    Retrieval node - execute search using selected retriever.

    Uses the retriever tool selected by query analysis:
    - BM25 for exact matches (function names, identifiers)
    - Semantic for conceptual queries (what is, explain)
    - Hybrid for code examples and general queries

    Args:
        state: Current agent state

    Returns:
        Updated state with chunks, context, and relevance_score
    """
    start = time.perf_counter()

    user_query = state.get("user_query", "")
    retriever_choice = state.get("retriever_choice", RetrieverChoice.HYBRID)
    query_embedding = state.get("query_embedding", [])

    try:
        # Get the appropriate retriever
        retriever = get_retriever_for_choice(retriever_choice)

        # Execute retrieval
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: retriever.retrieve(user_query, analyze_backtracking=True),
        )

        # Convert chunks to dict format
        chunks = [
            {
                "id": chunk.id,
                "content": chunk.content,
                "similarity": chunk.similarity,
                "parent_doc_url": chunk.parent_doc_url,
                "parent_doc_title": chunk.parent_doc_title,
                "section_title": chunk.section_title,
                "has_code": chunk.has_code,
            }
            for chunk in result.chunks
        ]

        # Compute relevance and format context
        relevance_score = compute_relevance_score(chunks)
        context = format_chunks_fast(chunks)

        elapsed = (time.perf_counter() - start) * 1000
        logger.info(
            f"retrieve_chunks completed in {elapsed:.0f}ms "
            f"(chunks={len(chunks)}, relevance={relevance_score:.2f})"
        )

        return {
            **state,
            "chunks": chunks,
            "context": context,
            "relevance_score": relevance_score,
            "timings": {**state.get("timings", {}), "retrieve": elapsed},
        }

    except Exception as e:
        logger.error(f"retrieve_chunks failed: {e}")
        elapsed = (time.perf_counter() - start) * 1000
        return {
            **state,
            "chunks": [],
            "context": "",
            "relevance_score": 0.0,
            "timings": {**state.get("timings", {}), "retrieve": elapsed},
            "error": str(e),
        }


async def response_generation(state: AgentState) -> AgentState:
    """
    Response generation node (non-streaming).

    Generates response using LLM with retrieved context.
    For streaming, use response_generation_stream instead.
    If no context (retrieval skipped), generates conversational response.

    Args:
        state: Current agent state

    Returns:
        Updated state with response
    """
    start = time.perf_counter()

    context = state.get("context", "")
    relevance_score = state.get("relevance_score", 0.0)

    # Determine if we have valid context (retrieval was performed and found results)
    has_context = bool(context and context.strip())

    # Only use fallback if retrieval was attempted but found nothing relevant
    if has_context and should_fallback(relevance_score):
        logger.info("Using fallback response due to low relevance")
        fallback = get_fallback_response()
        return {
            **state,
            "response": fallback,
            "messages": [AIMessage(content=fallback)],
            "timings": {
                **state.get("timings", {}),
                "generate": (time.perf_counter() - start) * 1000,
            },
        }

    # Build messages for LLM (with or without context)
    llm_messages = build_messages(
        history=state.get("messages", []),
        context=context,
        user_query=state.get("user_query", ""),
        query_type=state.get("query_type", QueryType.GENERAL),
        include_context=has_context,  # Skip context formatting for conversational queries
    )

    try:
        # Generate response
        llm = get_llm_client()
        response = await llm.achat(llm_messages)

        elapsed = (time.perf_counter() - start) * 1000
        logger.info(f"response_generation completed in {elapsed:.0f}ms")

        return {
            **state,
            "response": response,
            "messages": [AIMessage(content=response)],
            "timings": {**state.get("timings", {}), "generate": elapsed},
        }

    except Exception as e:
        logger.error(f"response_generation failed: {e}")
        elapsed = (time.perf_counter() - start) * 1000
        error_msg = "I encountered an error generating a response. Please try again."
        return {
            **state,
            "response": error_msg,
            "messages": [AIMessage(content=error_msg)],
            "timings": {**state.get("timings", {}), "generate": elapsed},
            "error": str(e),
        }


async def response_generation_stream(
    state: AgentState,
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Streaming response generation node.

    Yields tokens immediately for low TTFT (Time to First Token).
    Final yield includes the complete response for state update.

    Args:
        state: Current agent state

    Yields:
        Dict with either {"type": "token", "content": str} or
        {"type": "final", "response": str, "timings": dict}
    """
    start = time.perf_counter()

    # Check if we should use fallback
    relevance_score = state.get("relevance_score", 0.0)
    if should_fallback(relevance_score):
        logger.info("Using fallback response due to low relevance")
        fallback = get_fallback_response()
        # Stream fallback in chunks for consistency
        for word in fallback.split():
            yield {"type": "token", "content": word + " "}
        yield {
            "type": "final",
            "response": fallback,
            "timings": {
                **state.get("timings", {}),
                "generate": (time.perf_counter() - start) * 1000,
            },
        }
        return

    # Build messages for LLM
    messages = build_messages(
        history=state.get("messages", []),
        context=state.get("context", ""),
        user_query=state.get("user_query", ""),
        query_type=state.get("query_type", QueryType.GENERAL),
    )

    try:
        llm = get_llm_client()
        response_tokens: List[str] = []
        first_token = True

        async for token in llm.achat_stream(messages):
            if first_token:
                ttft = (time.perf_counter() - start) * 1000
                logger.debug(f"TTFT: {ttft:.0f}ms")
                first_token = False

            response_tokens.append(token)
            yield {"type": "token", "content": token}

        full_response = "".join(response_tokens)
        elapsed = (time.perf_counter() - start) * 1000

        logger.info(f"response_generation_stream completed in {elapsed:.0f}ms")

        yield {
            "type": "final",
            "response": full_response,
            "timings": {**state.get("timings", {}), "generate": elapsed},
        }

    except Exception as e:
        logger.error(f"response_generation_stream failed: {e}")
        error_response = "I encountered an error generating a response. Please try again."
        yield {"type": "token", "content": error_response}
        yield {
            "type": "final",
            "response": error_response,
            "timings": {
                **state.get("timings", {}),
                "generate": (time.perf_counter() - start) * 1000,
            },
            "error": str(e),
        }


async def persist_memory(state: AgentState) -> AgentState:
    """
    Background persistence node (fire-and-forget).

    Persists the conversation turn to memory and database.
    This node should not block the response.

    Args:
        state: Current agent state

    Returns:
        Updated state (unchanged except for timings)
    """
    start = time.perf_counter()

    thread_id = state.get("thread_id", "")
    user_query = state.get("user_query", "")
    response = state.get("response", "")
    query_embedding = state.get("query_embedding", [])

    if not thread_id or not response:
        return state

    try:
        # Run persistence tasks in parallel
        memory = get_memory_manager()
        caches = get_caches()

        async def persist_turn():
            await memory.add_turn_async(
                thread_id=thread_id,
                user_message=user_query,
                assistant_response=response,
                metadata={
                    "query_type": state.get("query_type", QueryType.GENERAL).value,
                    "retriever": state.get("retriever_choice", RetrieverChoice.HYBRID).value,
                    "relevance_score": state.get("relevance_score", 0.0),
                },
            )

        async def update_cache():
            # Update semantic cache for similar queries
            if query_embedding and response:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    lambda: caches.update_response_cache(
                        user_query, query_embedding, response
                    ),
                )

        await asyncio.gather(persist_turn(), update_cache())

        elapsed = (time.perf_counter() - start) * 1000
        logger.debug(f"persist_memory completed in {elapsed:.0f}ms")

        return {
            **state,
            "timings": {**state.get("timings", {}), "persist": elapsed},
        }

    except Exception as e:
        logger.warning(f"persist_memory failed: {e}")
        return state
