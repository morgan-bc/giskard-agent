"""Web search tools."""

from .parrallel import DEFAULT_PARALLEL_MCP_URL, PARALLEL_MCP_CONFIG, ParallelSearchClient
from .tavily import DEFAULT_TAVILY_BASE_URL, TAVILY_API_KEY_ENV, TavilySearchClient

__all__ = [
    "DEFAULT_PARALLEL_MCP_URL",
    "DEFAULT_TAVILY_BASE_URL",
    "PARALLEL_MCP_CONFIG",
    "TAVILY_API_KEY_ENV",
    "ParallelSearchClient",
    "TavilySearchClient",
]
