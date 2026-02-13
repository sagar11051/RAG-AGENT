"""
Query analysis and retriever tool selection for the RAG agent.

Provides fast regex-based query classification (< 1ms) to select
the optimal retriever strategy without LLM calls.

Query Types:
- EXACT_MATCH: Function names, identifiers -> BM25 retriever
- CONCEPTUAL: What is, explain, why -> Semantic retriever
- CODE_EXAMPLE: How to, implement -> Hybrid retriever
- TUTORIAL: Step-by-step guides -> Hybrid retriever
- GENERAL: Default -> Hybrid retriever
"""

import re
from typing import List, Tuple

from src.agent.state import QueryType, RetrieverChoice
from src.retrieval import RAGRetriever, RetrieverType
from src.utils.logger import get_logger

logger = get_logger(__name__)


def extract_text_from_content(content) -> str:
    """
    Extract text string from any content format.

    Handles:
    - str: returns as-is
    - list: extracts text from content blocks (multimodal messages)
    - dict: extracts 'content', 'text', or converts to string
    - Message objects: extracts .content recursively
    - Other: converts to string

    Args:
        content: Content in any format (str, list, dict, Message)

    Returns:
        Extracted text as a string
    """
    # Already a string
    if isinstance(content, str):
        return content

    # None or empty
    if not content:
        return ""

    # List of content blocks (multimodal messages)
    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, str):
                text_parts.append(block)
            elif isinstance(block, dict):
                # Handle {"type": "text", "text": "..."} format
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif "content" in block:
                    text_parts.append(extract_text_from_content(block["content"]))
                elif "text" in block:
                    text_parts.append(block["text"])
            elif hasattr(block, "text"):
                # Handle TextContent and similar objects with .text attribute
                text_parts.append(str(block.text))
            elif hasattr(block, "content"):
                # Nested message object
                text_parts.append(extract_text_from_content(block.content))
            else:
                # Fallback: convert unknown types to string
                text_parts.append(str(block))
        return " ".join(text_parts)

    # Dict with content
    if isinstance(content, dict):
        if "content" in content:
            return extract_text_from_content(content["content"])
        if "text" in content:
            return str(content["text"])
        return str(content)

    # Message object with .content attribute
    if hasattr(content, "content"):
        return extract_text_from_content(content.content)

    # Fallback: convert to string
    return str(content)


# Patterns for queries that DON'T need retrieval (greetings, thanks, etc.)
NO_RETRIEVAL_PATTERNS = [
    r"^(hi|hello|hey|greetings|good\s*(morning|afternoon|evening))[\s!.,?]*$",
    r"^(thanks|thank you|thx|ty|cheers)[\s!.,?]*$",
    r"^(yes|no|ok|okay|sure|got it|understood|makes sense)[\s!.,?]*$",
    r"^(bye|goodbye|see you|later)[\s!.,?]*$",
    r"^(how are you|what's up|how's it going)[\s!.,?]*$",
]

# Pre-compile no-retrieval patterns
COMPILED_NO_RETRIEVAL = [re.compile(p, re.IGNORECASE) for p in NO_RETRIEVAL_PATTERNS]

# Patterns for COMPLEX queries that need tool-based agent
# These queries benefit from query decomposition, reformulation, and multi-step retrieval
COMPLEX_QUERY_PATTERNS = [
    # Multi-hop / comparison queries
    r"(compare|difference|differences|vs\.?|versus)\s+.+\s+(and|with|to)\s+",
    r"what.+difference.+between",
    r"how.+different.+(from|than)",
    # Multi-step / tutorial requests
    r"(step.?by.?step|tutorial|guide|walkthrough)",
    r"build.+(from scratch|complete|full)",
    r"create.+(complete|full|entire)",
    # Complex conceptual queries
    r"(explain|describe).+in.+detail",
    r"how.+work.+together",
    r"when.+(should|would).+use.+(instead|over|vs)",
    # Architecture / design queries
    r"(architecture|design|pattern).+(for|of|in)",
    r"best.+practice.+for",
    r"(production|scalable|enterprise)",
    # Debugging / troubleshooting
    r"(debug|troubleshoot|fix|solve).+problem",
    r"why.+(not working|failing|error)",
    # Integration queries
    r"(integrate|connect|combine).+with",
    r"use.+with.+and",
]

# Pre-compile complex patterns
COMPILED_COMPLEX_PATTERNS = [re.compile(p, re.IGNORECASE) for p in COMPLEX_QUERY_PATTERNS]


def classify_query_complexity(query) -> str:
    """
    Classify query complexity for routing.

    Returns one of:
    - "greeting": Simple greeting/thanks, no retrieval needed
    - "simple": Standard documentation query, fast retrieval
    - "complex": Multi-hop/comparison/tutorial, needs tool agent

    Args:
        query: User's query (string, list, dict, or Message object)

    Returns:
        Complexity level string
    """
    # Extract text from any format using helper
    query_text = extract_text_from_content(query)

    # Defensive: ensure we always have a string
    if not isinstance(query_text, str):
        logger.warning(f"extract_text_from_content returned non-string: {type(query_text)}")
        query_text = str(query_text) if query_text else ""

    if not query_text:
        return "greeting"

    query_stripped = query_text.strip()

    # Check greeting patterns first
    for pattern in COMPILED_NO_RETRIEVAL:
        if pattern.match(query_stripped):
            logger.debug(f"Query classified as GREETING: {query_stripped[:30]}...")
            return "greeting"

    # Check complex patterns
    for pattern in COMPILED_COMPLEX_PATTERNS:
        if pattern.search(query_stripped):
            logger.debug(f"Query classified as COMPLEX: {query_stripped[:50]}...")
            return "complex"

    # Check query length and structure for additional complexity signals
    words = query_stripped.split()
    if len(words) > 15:  # Long queries often need more context
        logger.debug(f"Query classified as COMPLEX (length): {query_stripped[:50]}...")
        return "complex"

    # Check for multiple question marks (multi-part questions)
    if query_stripped.count("?") > 1:
        logger.debug(f"Query classified as COMPLEX (multi-part): {query_stripped[:50]}...")
        return "complex"

    # Default to simple
    logger.debug(f"Query classified as SIMPLE: {query_stripped[:50]}...")
    return "simple"


def should_retrieve(query, history: list = None) -> bool:
    """
    Determine if retrieval is needed for this query.

    Returns False for:
    - Greetings, thanks, confirmations
    - Very short queries (< 3 words) that are conversational
    - Follow-up questions that can be answered from context

    Args:
        query: User's query (string, list, dict, or Message object)
        history: Conversation history (optional)

    Returns:
        True if retrieval should be performed
    """
    # Extract text from any format using helper
    query_text = extract_text_from_content(query)

    # Defensive: ensure we always have a string
    if not isinstance(query_text, str):
        query_text = str(query_text) if query_text else ""

    if not query_text:
        return False

    query_stripped = query_text.strip()

    # Check no-retrieval patterns (greetings, thanks, etc.)
    for pattern in COMPILED_NO_RETRIEVAL:
        if pattern.match(query_stripped):
            logger.info(f"Skipping retrieval for conversational query: {query_stripped[:30]}")
            return False

    # Very short queries (1-2 words) that don't look like documentation queries
    words = query_stripped.split()
    if len(words) <= 2:
        # Check if it looks like a documentation term
        has_doc_term = any(
            term in query_stripped.lower()
            for term in ["langgraph", "langchain", "state", "node", "edge", "graph", "agent", "tool"]
        )
        if not has_doc_term:
            logger.info(f"Skipping retrieval for short non-technical query: {query_stripped}")
            return False

    return True


# Query patterns for instant classification (< 1ms)
# Patterns are checked in order, first match wins
PATTERNS = {
    "exact_match": [
        r"\b[A-Z][a-z]+[A-Z]\w+\b",  # CamelCase: StateGraph, ToolNode
        r"\b[a-z]+_[a-z_]+\b",  # snake_case: add_node, get_state
        r"`[^`]+`",  # Backticks: `compile()`
        r"error|exception|traceback",  # Error lookups
        r"class\s+\w+",  # Class references
        r"function\s+\w+",  # Function references
        r"\w+\(\)",  # Function calls: method()
    ],
    "conceptual": [
        r"^what (is|are)\b",
        r"^how does\b",
        r"^why\b",
        r"^explain\b",
        r"^describe\b",
        r"^define\b",
        r"difference between",
        r"compared to",
        r"vs\.?\s",
    ],
    "code_example": [
        r"^how (do|to|can|should)\b",
        r"example|sample|code|implement",
        r"show me",
        r"write .*(code|function|class)",
        r"create .*(code|function|class)",
    ],
    "tutorial": [
        r"tutorial|guide|walkthrough|step.?by.?step",
        r"getting started",
        r"learn how",
        r"beginner",
        r"introduction to",
    ],
}

# Pre-compile patterns for performance
# Note: exact_match patterns should NOT use IGNORECASE to preserve case sensitivity
COMPILED_PATTERNS = {
    "exact_match": [re.compile(pattern) for pattern in PATTERNS["exact_match"]],
    "conceptual": [re.compile(pattern, re.IGNORECASE) for pattern in PATTERNS["conceptual"]],
    "code_example": [re.compile(pattern, re.IGNORECASE) for pattern in PATTERNS["code_example"]],
    "tutorial": [re.compile(pattern, re.IGNORECASE) for pattern in PATTERNS["tutorial"]],
}


def analyze_query_fast(query) -> Tuple[QueryType, RetrieverChoice]:
    """
    Fast regex-based query classification (< 1ms).

    Analyzes the query to determine:
    1. Query type (what kind of information is being requested)
    2. Best retriever to use for this query type

    Pattern priority:
    1. Conceptual patterns (what is, explain, why) - checked first
    2. Tutorial patterns (tutorial, guide, step-by-step)
    3. Code example patterns (how to, implement, example)
    4. Exact match patterns (identifiers only when standalone)

    Args:
        query: User's query (string, list, dict, or Message object)

    Returns:
        Tuple of (QueryType, RetrieverChoice)
    """
    # Extract text from any format using helper
    query_text = extract_text_from_content(query)

    # Defensive: ensure we always have a string
    if not isinstance(query_text, str):
        query_text = str(query_text) if query_text else ""

    if not query_text:
        return QueryType.GENERAL, RetrieverChoice.HYBRID

    query_lower = query_text.lower().strip()

    # Check conceptual patterns FIRST -> Semantic (best for concepts)
    # These are question patterns that should take priority
    for pattern in COMPILED_PATTERNS["conceptual"]:
        if pattern.search(query_lower):
            logger.debug(f"Query classified as CONCEPTUAL: {query_text[:50]}...")
            return QueryType.CONCEPTUAL, RetrieverChoice.SEMANTIC

    # Check tutorial patterns -> Hybrid
    for pattern in COMPILED_PATTERNS["tutorial"]:
        if pattern.search(query_lower):
            logger.debug(f"Query classified as TUTORIAL: {query_text[:50]}...")
            return QueryType.TUTORIAL, RetrieverChoice.HYBRID

    # Check code example patterns -> Hybrid
    for pattern in COMPILED_PATTERNS["code_example"]:
        if pattern.search(query_lower):
            logger.debug(f"Query classified as CODE_EXAMPLE: {query_text[:50]}...")
            return QueryType.CODE_EXAMPLE, RetrieverChoice.HYBRID

    # Check exact match patterns -> BM25 (best for standalone identifiers)
    # Only match if query is primarily an identifier lookup
    for pattern in COMPILED_PATTERNS["exact_match"]:
        if pattern.search(query_text):
            logger.debug(f"Query classified as EXACT_MATCH: {query_text[:50]}...")
            return QueryType.EXACT_MATCH, RetrieverChoice.BM25

    # Default -> Hybrid (safest choice)
    logger.debug(f"Query classified as GENERAL: {query_text[:50]}...")
    return QueryType.GENERAL, RetrieverChoice.HYBRID


def get_retriever_for_choice(
    choice: RetrieverChoice,
    top_k: int = 5,
) -> RAGRetriever:
    """
    Get the appropriate RAGRetriever based on the choice.

    Args:
        choice: Selected retriever type
        top_k: Number of chunks to retrieve

    Returns:
        Configured RAGRetriever instance
    """
    type_map = {
        RetrieverChoice.SEMANTIC: RetrieverType.SEMANTIC,
        RetrieverChoice.BM25: RetrieverType.BM25,
        RetrieverChoice.HYBRID: RetrieverType.HYBRID,
    }
    return RAGRetriever(retriever_type=type_map[choice], top_k=top_k)


def compute_relevance_score(chunks: List[dict]) -> float:
    """
    Compute average relevance score from retrieved chunks.

    Args:
        chunks: List of retrieved chunk dictionaries

    Returns:
        Average similarity score (0.0-1.0)
    """
    if not chunks:
        return 0.0

    scores = []
    for chunk in chunks:
        # Check for various score field names
        score = chunk.get("similarity") or chunk.get("score") or chunk.get("rrf_score")
        if score is not None:
            scores.append(float(score))

    if not scores:
        return 0.0

    return sum(scores) / len(scores)


def should_fallback(relevance_score: float, threshold: float = 0.3) -> bool:
    """
    Determine if we should use fallback response due to low relevance.

    Args:
        relevance_score: Average relevance score from retrieval
        threshold: Minimum score to proceed with generation

    Returns:
        True if should use fallback response
    """
    return relevance_score < threshold


def format_chunks_fast(chunks: List[dict], max_tokens: int = 4000) -> str:
    """
    Fast formatting of chunks for LLM context.

    Args:
        chunks: List of retrieved chunk dictionaries
        max_tokens: Approximate max tokens for context

    Returns:
        Formatted context string
    """
    if not chunks:
        return ""

    context_parts = []
    estimated_tokens = 0

    for i, chunk in enumerate(chunks):
        content = chunk.get("content", "")
        source = chunk.get("parent_doc_url") or chunk.get("source_url", "")
        title = chunk.get("section_title") or chunk.get("title", "")

        # Build chunk text
        chunk_text = f"[Source {i + 1}]"
        if title:
            chunk_text += f" {title}"
        if source:
            chunk_text += f"\nURL: {source}"
        chunk_text += f"\n{content}\n"

        # Rough token estimate (4 chars per token)
        chunk_tokens = len(chunk_text) // 4
        if estimated_tokens + chunk_tokens > max_tokens:
            break

        context_parts.append(chunk_text)
        estimated_tokens += chunk_tokens

    return "\n---\n".join(context_parts)


# Pre-built system prompts for different scenarios
SYSTEM_PROMPTS = {
    "default": """You are a helpful LangGraph documentation assistant.
Answer questions based on the provided context from the LangGraph documentation.
If the context doesn't contain relevant information, say so honestly.
Always cite the source when possible using the provided URLs.
Be concise but thorough.""",
    "code": """You are a helpful LangGraph code assistant.
Provide code examples and implementations based on the documentation context.
Explain the code clearly and cite sources.
If the context doesn't have relevant code, mention what you found and suggest alternatives.""",
    "conceptual": """You are a helpful LangGraph documentation assistant.
Explain concepts clearly based on the provided documentation context.
Use simple language and provide examples when helpful.
Cite sources for your explanations.""",
}


def get_system_prompt(query_type: QueryType) -> str:
    """
    Get appropriate system prompt based on query type.

    Args:
        query_type: Classified query type

    Returns:
        System prompt string
    """
    if query_type in (QueryType.CODE_EXAMPLE, QueryType.EXACT_MATCH):
        return SYSTEM_PROMPTS["code"]
    elif query_type == QueryType.CONCEPTUAL:
        return SYSTEM_PROMPTS["conceptual"]
    else:
        return SYSTEM_PROMPTS["default"]


def _extract_message_content(msg) -> Tuple[str, str]:
    """
    Extract role and content from a message (dict or Message object).

    Args:
        msg: Message dict or langchain Message object

    Returns:
        Tuple of (role, content)
    """
    # Handle langchain Message objects
    if hasattr(msg, "type") and hasattr(msg, "content"):
        role = msg.type  # 'human', 'ai', 'system'
        # Map langchain types to standard roles
        role_map = {"human": "user", "ai": "assistant", "system": "system"}
        return role_map.get(role, role), msg.content

    # Handle dict format
    if isinstance(msg, dict):
        return msg.get("role", "user"), msg.get("content", "")

    return "user", str(msg)


def build_messages(
    history: list,
    context: str,
    user_query: str,
    query_type: QueryType,
    include_context: bool = True,
) -> List[dict]:
    """
    Build message list for LLM with history, context, and query.

    Args:
        history: Previous conversation messages (dict or Message objects)
        context: Retrieved context
        user_query: Current user query
        query_type: Classified query type
        include_context: Whether to include retrieved context

    Returns:
        List of messages for LLM
    """
    messages = []

    # System prompt
    system_content = get_system_prompt(query_type)
    if not include_context:
        system_content = """You are a friendly LangGraph documentation assistant.
For greetings and simple questions, respond naturally and conversationally.
If the user asks about LangGraph topics, let them know you can help with documentation questions."""

    messages.append({
        "role": "system",
        "content": system_content,
    })

    # Add conversation history (last N turns, excluding current query)
    max_history = 10
    if history:
        # Skip the last message if it's the current query
        history_to_use = history[:-1] if history else []
        for msg in history_to_use[-max_history:]:
            role, content = _extract_message_content(msg)
            if role in ("user", "assistant"):
                messages.append({"role": role, "content": content})

    # Build user message
    if include_context and context:
        user_content = f"""Context from LangGraph documentation:

{context}

---

Question: {user_query}

Please answer based on the context above. Reference the previous conversation if relevant."""
    else:
        user_content = user_query

    messages.append({
        "role": "user",
        "content": user_content,
    })

    return messages


FALLBACK_RESPONSE = """I couldn't find relevant information in the LangGraph documentation for your query.

Here are some suggestions:
1. Try rephrasing your question
2. Be more specific about what you're looking for
3. Check the official LangGraph documentation at https://langchain-ai.github.io/langgraph/

Would you like to try a different question?"""


def get_fallback_response() -> str:
    """Get fallback response for low-relevance queries."""
    return FALLBACK_RESPONSE
