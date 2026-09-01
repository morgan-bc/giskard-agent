# Copyright (c) Microsoft. All rights reserved.

"""Parallel Search MCP wrapper.

Wraps the remote MCP server at ``https://search.parallel.ai/mcp`` as
:class:`ParallelSearchClient`, exposing two framework-native tools:

* ``web_search`` — search the web
* ``web_fetch``  — fetch a URL's content

Use :meth:`ParallelSearchClient.get_tools` to obtain the two
:class:`~giskard.core.tools.FunctionTool` instances for an agent.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Literal

from giskard.core.mcp import MCPSpecificApproval, MCPStreamableHTTPTool
from giskard.core.tools import FunctionTool, tool

logger = logging.getLogger(__name__)

DEFAULT_PARALLEL_MCP_URL = "https://search.parallel.ai/mcp"

# MCP config mirror for external wiring (e.g. mcpServers JSON).
PARALLEL_MCP_CONFIG: dict[str, Any] = {
    "mcpServers": {
        "Parallel Search MCP": {
            "url": DEFAULT_PARALLEL_MCP_URL,
        }
    }
}


def _extract_search_results(data: str| dict, max_results: int) -> str:
    """Extract ``results`` from a search payload and cap its length.

    The Parallel MCP ``web_search`` tool returns a JSON payload whose
    ``results`` list holds the ranked hits. Only the first ``max_results``
    entries are kept; the list is re-serialized with ``ensure_ascii=False``.
    Non-JSON or unexpected payloads pass through untouched.
    """
    if isinstance(data, dict):
        results = data.get("results", [])
        return results[:max_results]
    else:
        return data

class ParallelSearchClient:
    """Wrapper around the Parallel Search MCP server.

    Internally owns an :class:`~giskard.core.mcp.MCPStreamableHTTPTool`
    connected to ``https://search.parallel.ai/mcp``. The MCP server's
    native tools are surfaced as two framework tools via :meth:`get_tools`.

    The wrapper delegates lifecycle (``connect`` / ``close`` / async
    context manager) to the inner MCP tool and maps its remote tools to
    stable local names ``web_search`` and ``web_fetch``.

    Args:
        url: MCP endpoint. Defaults to ``https://search.parallel.ai/mcp``.
        name: Logical MCP tool name. Defaults to ``parallel-search``.
        approval_mode: Tool approval policy forwarded to the inner MCP
            tool and to the surfaced ``FunctionTool`` s.
        request_timeout: Request timeout in seconds.
        **mcp_kwargs: Additional kwargs forwarded to
            :class:`MCPStreamableHTTPTool` (e.g. ``header_provider``,
            ``http_client``, ``allowed_tools``).

    Example::

        from giskard.tools.web_search.parrallel import ParallelSearchClient
        from giskard.core.agents import Agent

        client = ParallelSearchClient()
        async with client:
            agent = Agent(client=chat_client, tools=client.get_tools())
            resp = await agent.run("Search for giskard agent framework")
    """

    def __init__(
        self,
        url: str = DEFAULT_PARALLEL_MCP_URL,
        *,
        name: str = "parallel-search",
        description: str | None = None,
        approval_mode: Literal["always_require", "never_require"] | MCPSpecificApproval | None = None,
        request_timeout: int | None = None,
        **mcp_kwargs: Any,
    ) -> None:
        self.url = url
        self.name = name
        self._mcp = MCPStreamableHTTPTool(
            name=name,
            url=url,
            description=description or "Parallel AI web search and fetch (MCP)",
            approval_mode=approval_mode,
            request_timeout=request_timeout,
            # Do not restrict by default — let server advertise; filtering
            # happens in get_tools() mapping. Users may override via mcp_kwargs.
            **mcp_kwargs,
        )
        # Cached FunctionTools so get_tools() is idempotent.
        self._tools: list[FunctionTool] | None = None

    # ------------------------------------------------------------------ lifecycle

    async def connect(self, *, reset: bool = False) -> None:
        """Connect to the underlying MCP server."""
        await self._mcp.connect(reset=reset)

    async def close(self) -> None:
        """Close the underlying MCP connection."""
        await self._mcp.close()

    async def __aenter__(self) -> ParallelSearchClient:
        await self.connect()
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.close()

    @property
    def mcp_tool(self) -> MCPStreamableHTTPTool:
        """Access the inner :class:`MCPStreamableHTTPTool`."""
        return self._mcp

    @property
    def is_connected(self) -> bool:
        return self._mcp.is_connected


    async def _call_remote(self, remote_name: str, **kwargs: Any) -> dict | str:
        """Call a remote MCP tool and return text."""
        # Ensure connected — MCPTool.connect is idempotent.
        if not self._mcp.is_connected:
            await self._mcp.connect()
        result = await self._mcp.call_tool(remote_name, **kwargs)
        if isinstance(result, str):
            return json.loads(result)
        elif len(result) > 0:
            return json.loads(result[0].text)
        else:
            return str(result)

    async def search(
        self,
        objective: str,
        search_queries: list[str],
        max_results: int = 5,
    ) -> str:
        """Search the web.

        Args:
            objective: Natural-language description of what the web search is trying to find.
            search_queries: Concise keyword queries (3-6 words each, 1-3 items). At least one required.
            max_results: Maximum number of results to return (default 5).
        """

        text = await self._call_remote(
            "web_search",
            objective=objective,
            search_queries=search_queries,
        )
        return _extract_search_results(text, max_results)

    async def extract(
        self,
        urls: list[str],
    ) -> str:
        """Fetch URL(s).

        Args:
            urls: List of valid HTTP/HTTPS URLs to extract (max 20).
        """

        text = await self._call_remote(
            "web_fetch",
            urls=urls,
        )
        return text

    def _build_tools(self) -> list[FunctionTool]:
        """Create the two FunctionTools wrapping the MCP calls."""

        # ---- web_search -------------------------------------------------
        @tool(
            name="web_search",
            description="Search the web via Parallel Search MCP. Returns ranked results with snippets.",
        )
        async def web_search(
            objective: str,
            search_queries: list[str],
            max_results: int = 5,
        ) -> str:
            """Search the web.

            Args:
                objective: Natural-language description of what the web search is trying to find.
                search_queries: Concise keyword queries (3-6 words each, 1-3 items). At least one required.
                max_results: Maximum number of results to return (default 5).
            """
            return await self.search(
                objective=objective,
                search_queries=search_queries,
                max_results=max_results,
            )
            
        # ---- web_fetch --------------------------------------------------
        @tool(
            name="web_fetch",
            description="Fetch and extract content from specific URLs via Parallel Search MCP.",
        )
        async def web_fetch(
            urls: list[str],
        ) -> str:
            """Fetch URL(s).

            Args:
                urls: List of valid HTTP/HTTPS URLs to extract (max 20).
            """
            return await self.extract(urls=urls)

        return [web_search, web_fetch]

    def get_tools(self) -> list[FunctionTool]:
        """Return the two tools ``[web_search, web_fetch]``.

        Idempotent — subsequent calls return the same instances.
        The tools are thin wrappers that lazily connect to the MCP server
        on first invocation.
        """
        if self._tools is None:
            self._tools = self._build_tools()
        return self._tools


__all__ = ["DEFAULT_PARALLEL_MCP_URL", "PARALLEL_MCP_CONFIG", "ParallelSearchClient"]
