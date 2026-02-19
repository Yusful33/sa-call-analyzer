"""
Runner registry: maps (framework, use_case) to the appropriate runner function.
Uses lazy imports to avoid loading all frameworks at startup.
Fallback chain: requested framework -> langgraph -> langchain generic.
"""

from typing import Callable

# Use case name constants
RAG = "retrieval-augmented-search"
CHATBOT = "multiturn-chatbot-with-tools"
TEXT_TO_SQL = "text-to-sql-bi-agent"
MULTI_AGENT = "multi-agent-orchestration"
CLASSIFICATION = "classification-routing"
MULTIMODAL = "multimodal-ai"
MCP = "mcp-tool-use"
TRAVEL_AGENT = "travel-agent"

# Framework name constants
LANGGRAPH = "langgraph"
LANGCHAIN = "langchain"
CREWAI = "crewai"
ADK = "adk"


def get_runner(framework: str, use_case: str) -> Callable:
    """Get the runner function for a given framework and use case.

    Fallback chain:
    1. Exact match (framework, use_case)
    2. Same framework, generic use case
    3. LangGraph with same use case
    4. LangGraph generic
    """
    runner = _try_get_runner(framework, use_case)
    if runner:
        return runner

    # Fallback: same framework, generic
    runner = _try_get_runner(framework, "generic")
    if runner:
        return runner

    # Fallback: langgraph with same use case
    if framework != LANGGRAPH:
        runner = _try_get_runner(LANGGRAPH, use_case)
        if runner:
            return runner

    # Final fallback: langgraph generic
    from .langgraph.generic import run_generic
    return run_generic


def _try_get_runner(framework: str, use_case: str) -> Callable | None:
    """Try to import a runner for the given framework and use case. Returns None on failure."""
    try:
        if framework == LANGGRAPH:
            return _get_langgraph_runner(use_case)
        elif framework == LANGCHAIN:
            return _get_langchain_runner(use_case)
        elif framework == CREWAI:
            return _get_crewai_runner(use_case)
        elif framework == ADK:
            return _get_adk_runner(use_case)
    except ImportError:
        return None
    return None


def _get_langgraph_runner(use_case: str) -> Callable | None:
    if use_case == RAG:
        from .langgraph.rag import run_rag
        return run_rag
    elif use_case == CHATBOT:
        from .langgraph.chatbot import run_chatbot
        return run_chatbot
    elif use_case == TEXT_TO_SQL:
        from .langgraph.text_to_sql import run_text_to_sql
        return run_text_to_sql
    elif use_case == MULTI_AGENT:
        from .langgraph.multi_agent import run_multi_agent
        return run_multi_agent
    elif use_case == CLASSIFICATION:
        from .langgraph.classification import run_classification
        return run_classification
    elif use_case == MULTIMODAL:
        from .langgraph.multimodal import run_multimodal
        return run_multimodal
    elif use_case == MCP:
        from .langgraph.mcp import run_mcp
        return run_mcp
    elif use_case == TRAVEL_AGENT:
        from .langgraph.travel_agent import run_travel_agent
        return run_travel_agent
    else:
        from .langgraph.generic import run_generic
        return run_generic


def _get_langchain_runner(use_case: str) -> Callable | None:
    if use_case == RAG:
        from .langchain.rag import run_rag
        return run_rag
    elif use_case == CHATBOT:
        from .langchain.chatbot import run_chatbot
        return run_chatbot
    elif use_case == TEXT_TO_SQL:
        from .langchain.text_to_sql import run_text_to_sql
        return run_text_to_sql
    elif use_case == MULTI_AGENT:
        from .langchain.multi_agent import run_multi_agent
        return run_multi_agent
    elif use_case == CLASSIFICATION:
        from .langchain.classification import run_classification
        return run_classification
    elif use_case == MULTIMODAL:
        from .langchain.multimodal import run_multimodal
        return run_multimodal
    elif use_case == MCP:
        from .langchain.mcp import run_mcp
        return run_mcp
    elif use_case == TRAVEL_AGENT:
        from .langchain.travel_agent import run_travel_agent
        return run_travel_agent
    else:
        from .langchain.generic import run_generic
        return run_generic


def _get_crewai_runner(use_case: str) -> Callable | None:
    if use_case == RAG:
        from .crewai_fw.rag import run_rag
        return run_rag
    elif use_case == CHATBOT:
        from .crewai_fw.chatbot import run_chatbot
        return run_chatbot
    elif use_case == TEXT_TO_SQL:
        from .crewai_fw.text_to_sql import run_text_to_sql
        return run_text_to_sql
    elif use_case == MULTI_AGENT:
        from .crewai_fw.multi_agent import run_multi_agent
        return run_multi_agent
    elif use_case == CLASSIFICATION:
        from .crewai_fw.classification import run_classification
        return run_classification
    elif use_case == MULTIMODAL:
        from .crewai_fw.multimodal import run_multimodal
        return run_multimodal
    elif use_case == MCP:
        from .crewai_fw.mcp import run_mcp
        return run_mcp
    elif use_case == TRAVEL_AGENT:
        from .crewai_fw.travel_agent import run_travel_agent
        return run_travel_agent
    else:
        from .crewai_fw.generic import run_generic
        return run_generic


def _get_adk_runner(use_case: str) -> Callable | None:
    if use_case == RAG:
        from .adk.rag import run_rag
        return run_rag
    elif use_case == CHATBOT:
        from .adk.chatbot import run_chatbot
        return run_chatbot
    elif use_case == TEXT_TO_SQL:
        from .adk.text_to_sql import run_text_to_sql
        return run_text_to_sql
    elif use_case == MULTI_AGENT:
        from .adk.multi_agent import run_multi_agent
        return run_multi_agent
    elif use_case == CLASSIFICATION:
        from .adk.classification import run_classification
        return run_classification
    elif use_case == MULTIMODAL:
        from .adk.multimodal import run_multimodal
        return run_multimodal
    elif use_case == MCP:
        from .adk.mcp import run_mcp
        return run_mcp
    elif use_case == TRAVEL_AGENT:
        from .adk.travel_agent import run_travel_agent
        return run_travel_agent
    else:
        from .adk.generic import run_generic
        return run_generic
