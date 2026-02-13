"""
LangGraph definition for the RAG agent.

Defines the hybrid agent graph with:
- Parallel initialization (memory + embedding + query analysis)
- Smart complexity routing (greeting / simple / complex)
- Fast retrieval for simple queries
- ReAct tool agent for complex queries
- Streaming response generation
- Background persistence

Graph Flow:
    parallel_init -> [complexity_router]
                           |
           +---------------+---------------+
           |               |               |
           v               v               v
       [greeting]     [simple_rag]   [complex_rag]
           |               |               |
           v               v               v
      direct_llm      retrieve →      tool_agent
           |          generate             |
           |               |               |
           +-------+-------+---------------+
                   |
                   v
               [persist]
                   |
                   v
                 [END]
"""

import time
from typing import Literal, Optional

from langchain_core.messages import AIMessage
from langgraph.graph import END, StateGraph

from src.agent.nodes import (
    parallel_init,
    persist_memory,
    response_generation,
    retrieve_chunks,
)
from src.agent.react_agent import run_react_agent
from src.agent.state import AgentState
from src.agent.tools import classify_query_complexity, extract_text_from_content
from src.llm.ovh_llm import get_llm_client
from src.utils.logger import get_logger

logger = get_logger(__name__)


def route_by_complexity(state: AgentState) -> Literal["greeting", "simple", "complex"]:
    """
    Route based on query complexity.

    Returns:
    - "greeting": For greetings/thanks, skip retrieval entirely
    - "simple": Standard queries, use fast node-based retrieval
    - "complex": Multi-hop/comparison queries, use ReAct tool agent
    """
    user_query = state.get("user_query", "")
    # Ensure user_query is a string for classification and logging
    user_query_str = extract_text_from_content(user_query) if not isinstance(user_query, str) else user_query
    complexity = classify_query_complexity(user_query_str)
    logger.info(f"Query complexity: {complexity} for: {user_query_str[:50]}...")
    return complexity


async def direct_response(state: AgentState) -> AgentState:
    """
    Direct LLM response without retrieval (for greetings).

    Fast path for simple conversational queries that don't
    need documentation context.

    Args:
        state: Current agent state

    Returns:
        Updated state with response
    """
    start = time.perf_counter()

    user_query = state.get("user_query", "")
    # Ensure user_query is a string
    if not isinstance(user_query, str):
        user_query = extract_text_from_content(user_query)
    messages = state.get("messages", [])

    # Build simple conversational prompt
    llm_messages = [
        {
            "role": "system",
            "content": """You are a friendly LangGraph documentation assistant.
For greetings and simple questions, respond naturally and conversationally.
If the user asks about LangGraph topics, let them know you can help with documentation questions.
Keep responses brief and friendly.""",
        }
    ]

    # Add recent history
    if messages:
        for msg in messages[-4:-1]:  # Last few messages, excluding current
            if hasattr(msg, "type") and hasattr(msg, "content"):
                role_map = {"human": "user", "ai": "assistant"}
                role = role_map.get(msg.type, "user")
                # Safely extract content
                content = extract_text_from_content(msg.content)
                llm_messages.append({"role": role, "content": content})
            elif isinstance(msg, dict):
                content = extract_text_from_content(msg.get("content", ""))
                llm_messages.append({
                    "role": msg.get("role", "user"),
                    "content": content,
                })

    llm_messages.append({"role": "user", "content": user_query})

    try:
        llm = get_llm_client()
        response = await llm.achat(llm_messages)

        elapsed = (time.perf_counter() - start) * 1000
        logger.info(f"direct_response completed in {elapsed:.0f}ms")

        return {
            **state,
            "response": response,
            "messages": [AIMessage(content=response)],
            "timings": {**state.get("timings", {}), "direct_generate": elapsed},
        }

    except Exception as e:
        logger.error(f"direct_response failed: {e}")
        elapsed = (time.perf_counter() - start) * 1000
        fallback = "Hello! I'm here to help with LangGraph documentation questions. What would you like to know?"
        return {
            **state,
            "response": fallback,
            "messages": [AIMessage(content=fallback)],
            "timings": {**state.get("timings", {}), "direct_generate": elapsed},
            "error": str(e),
        }


async def complex_rag_response(state: AgentState) -> AgentState:
    """
    ReAct tool agent for complex queries.

    Uses the ReAct agent with retrieval tools for:
    - Multi-hop reasoning
    - Query decomposition
    - Comparison queries

    Args:
        state: Current agent state

    Returns:
        Updated state with response
    """
    start = time.perf_counter()

    user_query = state.get("user_query", "")
    messages = state.get("messages", [])
    thread_id = state.get("thread_id", "")

    try:
        result = await run_react_agent(
            query=user_query,
            history=messages,
            thread_id=thread_id,
        )

        response = result.get("response", "")
        tool_calls = result.get("tool_calls", 0)

        elapsed = (time.perf_counter() - start) * 1000
        logger.info(
            f"complex_rag_response completed in {elapsed:.0f}ms "
            f"(tool_calls={tool_calls})"
        )

        return {
            **state,
            "response": response,
            "messages": [AIMessage(content=response)],
            "timings": {**state.get("timings", {}), "complex_generate": elapsed},
        }

    except Exception as e:
        logger.error(f"complex_rag_response failed: {e}")
        elapsed = (time.perf_counter() - start) * 1000
        error_msg = "I encountered an error processing your complex query. Please try again with a simpler question."
        return {
            **state,
            "response": error_msg,
            "messages": [AIMessage(content=error_msg)],
            "timings": {**state.get("timings", {}), "complex_generate": elapsed},
            "error": str(e),
        }


def build_rag_graph(include_persistence: bool = True) -> StateGraph:
    """
    Build the hybrid RAG agent graph.

    Creates a LangGraph StateGraph with:
    1. parallel_init: Parallel execution of memory, embedding, query analysis
    2. Complexity-based routing:
       - greeting: Direct LLM response (no retrieval)
       - simple: Fast node-based retrieval -> generate
       - complex: ReAct tool agent with multi-step reasoning
    3. persist (optional): Background memory persistence

    Args:
        include_persistence: Whether to include persistence node

    Returns:
        Compiled StateGraph
    """
    # Create graph with AgentState
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("parallel_init", parallel_init)

    # Three response paths
    graph.add_node("direct_response", direct_response)  # Greeting path
    graph.add_node("retrieve", retrieve_chunks)  # Simple RAG path
    graph.add_node("generate", response_generation)  # Simple RAG generation
    graph.add_node("complex_response", complex_rag_response)  # Complex RAG path

    if include_persistence:
        graph.add_node("persist", persist_memory)

    # Set entry point
    graph.set_entry_point("parallel_init")

    # Add complexity-based routing
    graph.add_conditional_edges(
        "parallel_init",
        route_by_complexity,
        {
            "greeting": "direct_response",
            "simple": "retrieve",
            "complex": "complex_response",
        },
    )

    # Simple RAG path: retrieve -> generate
    graph.add_edge("retrieve", "generate")

    # All paths converge to persistence or END
    if include_persistence:
        graph.add_edge("direct_response", "persist")
        graph.add_edge("generate", "persist")
        graph.add_edge("complex_response", "persist")
        graph.add_edge("persist", END)
    else:
        graph.add_edge("direct_response", END)
        graph.add_edge("generate", END)
        graph.add_edge("complex_response", END)

    logger.info(f"Built hybrid RAG graph (persistence={include_persistence})")

    return graph


def compile_rag_graph(
    include_persistence: bool = True,
    checkpointer: Optional[object] = None,
) -> object:
    """
    Build and compile the RAG agent graph.

    Args:
        include_persistence: Whether to include persistence node
        checkpointer: Optional LangGraph checkpointer for state persistence

    Returns:
        Compiled graph ready for execution
    """
    graph = build_rag_graph(include_persistence=include_persistence)

    if checkpointer:
        compiled = graph.compile(checkpointer=checkpointer)
    else:
        compiled = graph.compile()

    logger.info("Compiled hybrid RAG graph")
    return compiled


# For LangGraph Studio - always create fresh graph to pick up code changes
# Caching is disabled to support hot-reload during development
def get_default_graph():
    """Get the default compiled graph with persistence.

    Note: Creates a fresh graph each time to support hot-reload in LangGraph Studio.
    For production, consider adding caching back.
    """
    logger.info("Creating fresh RAG graph (dev mode - no caching)")
    return compile_rag_graph(include_persistence=True)


def get_fast_graph():
    """Get the fast compiled graph without persistence (for benchmarks)."""
    return compile_rag_graph(include_persistence=False)


def reset_graphs():
    """Reset compiled graphs (for testing).

    Note: With dev mode (no caching), this is a no-op but kept for compatibility.
    """
    pass  # No caching in dev mode, nothing to reset
