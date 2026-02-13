"""
ReAct Tool Agent for complex RAG queries.

Uses LangGraph's prebuilt ReAct agent with retrieval tools for:
- Multi-hop reasoning queries
- Comparison queries
- Queries requiring query decomposition/reformulation
"""

import asyncio
from typing import Annotated, Any, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from src.agent.cache import get_caches
from src.agent.state import QueryType, RetrieverChoice
from src.agent.tools import (
    compute_relevance_score,
    format_chunks_fast,
    get_retriever_for_choice,
)
from src.llm.ovh_llm import get_llm_client
from src.retrieval import RAGRetriever, RetrieverType
from src.utils.logger import get_logger

logger = get_logger(__name__)


# Define retrieval tools for the ReAct agent
@tool
def search_documentation(
    query: Annotated[str, "The search query to find relevant documentation"],
    retriever_type: Annotated[
        str, "Type of retriever: 'semantic', 'bm25', or 'hybrid' (default)"
    ] = "hybrid",
) -> str:
    """
    Search the LangGraph documentation for relevant information.

    Use this tool to find documentation chunks related to your query.
    Choose the retriever type based on query:
    - 'semantic': Best for conceptual questions (what is, explain, why)
    - 'bm25': Best for exact terms, function names, identifiers
    - 'hybrid': Best for general queries (recommended default)

    Returns formatted documentation chunks with sources.
    """
    try:
        # Map string to RetrieverChoice
        type_map = {
            "semantic": RetrieverChoice.SEMANTIC,
            "bm25": RetrieverChoice.BM25,
            "hybrid": RetrieverChoice.HYBRID,
        }
        choice = type_map.get(retriever_type.lower(), RetrieverChoice.HYBRID)

        retriever = get_retriever_for_choice(choice)
        result = retriever.retrieve(query, analyze_backtracking=True)

        if not result.chunks:
            return f"No relevant documentation found for: {query}"

        # Format chunks for the agent
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

        context = format_chunks_fast(chunks, max_tokens=3000)
        relevance = compute_relevance_score(chunks)

        return f"Found {len(chunks)} relevant chunks (relevance: {relevance:.2f}):\n\n{context}"

    except Exception as e:
        logger.error(f"search_documentation failed: {e}")
        return f"Error searching documentation: {str(e)}"


@tool
def get_full_document(
    chunk_id: Annotated[str, "The chunk ID to get the full parent document for"],
) -> str:
    """
    Get the full content of a parent document by chunk ID.

    Use this when a chunk references incomplete content or you need
    more context from the full document. Pass the chunk 'id' field
    from a previous search result.

    This performs backtracking retrieval to get complete context.
    """
    try:
        retriever = RAGRetriever(retriever_type=RetrieverType.HYBRID)
        doc = retriever.get_full_document(chunk_id)

        if not doc:
            return f"Document not found for chunk ID: {chunk_id}"

        # Truncate if too long
        content = doc.get("content", "")
        if len(content) > 8000:
            content = content[:8000] + "\n\n[Content truncated for length...]"

        title = doc.get("title", "Unknown")
        url = doc.get("url", "")
        return f"# {title}\n\nURL: {url}\n\n{content}"

    except Exception as e:
        logger.error(f"get_full_document failed: {e}")
        return f"Error retrieving document: {str(e)}"


@tool
def decompose_query(
    original_query: Annotated[str, "The original complex query to decompose"],
    sub_queries: Annotated[
        List[str], "List of simpler sub-queries to answer the original"
    ],
) -> str:
    """
    Record a query decomposition plan for multi-hop reasoning.

    Use this tool when facing a complex query that requires multiple
    search steps. Provide the original query and a list of simpler
    sub-queries that, when answered, will help answer the original.

    This tool just records your plan - you still need to execute
    the searches using search_documentation.
    """
    plan = f"Query Decomposition Plan:\n"
    plan += f"Original: {original_query}\n\n"
    plan += "Sub-queries:\n"
    for i, sq in enumerate(sub_queries, 1):
        plan += f"  {i}. {sq}\n"
    plan += "\nProceed to search for each sub-query."
    return plan


# ReAct Agent system prompt
REACT_SYSTEM_PROMPT = """You are an expert LangGraph documentation assistant with access to search tools.

Your goal is to thoroughly answer the user's question by:
1. Understanding what information is needed
2. Using search tools strategically to find relevant documentation
3. Synthesizing information from multiple sources if needed
4. Providing a comprehensive answer with citations

For complex queries:
- Decompose into simpler sub-queries if needed
- Search for each component separately
- Use different retriever types for different query types:
  - 'semantic' for conceptual questions
  - 'bm25' for exact function/class names
  - 'hybrid' for general queries

Always cite your sources using the URLs provided in search results.
If the documentation doesn't contain the answer, say so honestly."""


def create_tool_agent():
    """
    Create a ReAct agent with retrieval tools.

    Returns a LangGraph compiled graph that can be invoked.
    """
    llm = get_llm_client()

    # Get the underlying ChatOpenAI client for the agent
    chat_model = llm.get_chat_model()

    tools = [search_documentation, get_full_document, decompose_query]

    # Create ReAct agent with our tools
    agent = create_react_agent(
        model=chat_model,
        tools=tools,
        state_modifier=REACT_SYSTEM_PROMPT,
    )

    return agent


# Cached agent instance
_react_agent = None


def get_react_agent():
    """Get or create the cached ReAct agent."""
    global _react_agent
    if _react_agent is None:
        _react_agent = create_tool_agent()
    return _react_agent


async def run_react_agent(
    query: str,
    history: List[Any] = None,
    thread_id: str = None,
) -> Dict[str, Any]:
    """
    Run the ReAct agent for a complex query.

    Args:
        query: User's query
        history: Conversation history
        thread_id: Thread ID for persistence

    Returns:
        Dict with response and metadata
    """
    agent = get_react_agent()

    # Build messages
    messages = []
    if history:
        for msg in history[:-1]:  # Exclude current query
            if hasattr(msg, "type"):
                if msg.type == "human":
                    messages.append(HumanMessage(content=msg.content))
                elif msg.type == "ai":
                    messages.append(AIMessage(content=msg.content))
            elif isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "user":
                    messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    messages.append(AIMessage(content=content))

    messages.append(HumanMessage(content=query))

    try:
        # Run the agent
        config = {"configurable": {"thread_id": thread_id}} if thread_id else {}
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: agent.invoke({"messages": messages}, config),
        )

        # Extract final response
        final_messages = result.get("messages", [])
        if final_messages:
            last_msg = final_messages[-1]
            response = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
        else:
            response = "I couldn't generate a response. Please try again."

        return {
            "response": response,
            "messages": final_messages,
            "tool_calls": len([m for m in final_messages if hasattr(m, "tool_calls") and m.tool_calls]),
        }

    except Exception as e:
        logger.error(f"ReAct agent failed: {e}")
        return {
            "response": f"I encountered an error processing your query. Please try again.",
            "error": str(e),
        }
