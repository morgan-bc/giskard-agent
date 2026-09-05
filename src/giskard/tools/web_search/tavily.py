# Copyright (c) Microsoft. All rights reserved.

"""Tavily search client.

Wraps the Tavily REST API (``https://api.tavily.com``) as
:class:`TavilySearchClient`, exposing two framework-native tools:

* ``web_search`` — Tavily ``POST /search``
* ``web_fetch``  — Tavily ``POST /extract``

Use :meth:`TavilySearchClient.get_tools` to obtain the two
:class:`~giskard.core.tools.FunctionTool` instances for an agent.

API reference:
* https://docs.tavily.com/documentation/api-reference/endpoint/search
* https://docs.tavily.com/documentation/api-reference/endpoint/extract
"""

from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING, Any, Literal

from giskard.core.tools import FunctionTool, tool

if TYPE_CHECKING:
    from httpx import AsyncClient, Response

logger = logging.getLogger(__name__)

DEFAULT_TAVILY_BASE_URL = "https://api.tavily.com"
TAVILY_API_KEY_ENV = "TAVILY_API_KEY"

SearchDepth = Literal["basic", "advanced", "fast", "ultra-fast"]
SearchTopic = Literal["general", "news", "finance"]
SearchTimeRange = Literal["day", "week", "month", "year"]
ExtractDepth = Literal["basic", "advanced"]
ExtractFormat = Literal["markdown", "text"]


def _error_text(message: str) -> str:
    """Format an error as JSON text (``ensure_ascii=False`` per project convention)."""
    return json.dumps({"error": message}, ensure_ascii=False)


class TavilySearchClient:
    """Wrapper around the Tavily REST API.

    Unlike :class:`~giskard.tools.web_search.parrallel.ParallelSearchClient`
    (which proxies a remote MCP server), this client talks to the Tavily REST
    endpoints directly and owns its own HTTP lifecycle. The two Tavily
    endpoints are surfaced as framework tools via :meth:`get_tools`:

    * ``web_search`` → ``POST /search``
    * ``web_fetch``  → ``POST /extract``

    The API key is resolved in order: the explicit ``api_key`` argument, then
    the ``TAVILY_API_KEY`` environment variable (checked per request, so it
    may be set after construction). Results are returned as JSON text
    serialized with ``ensure_ascii=False`` so non-ASCII content (e.g. Chinese)
    stays readable.

    Args:
        api_key: Tavily API key. Falls back to the ``TAVILY_API_KEY``
            environment variable when None.
        base_url: Tavily API base URL. Defaults to ``https://api.tavily.com``.
        timeout: Request timeout in seconds.
        http_client: Optional pre-built :class:`httpx.AsyncClient`. When
            supplied, the caller owns its lifecycle (``close()`` will not
            close it).

    Example::

        from giskard.tools.web_search.tavily import TavilySearchClient
        from giskard.core.harness.agents import create_harness_agent

        client = TavilySearchClient()  # reads TAVILY_API_KEY from env
        async with client:
            agent = create_harness_agent(
                chat_client, web_search_client=client, workdir=...
            )
            resp = await agent.run("Search for giskard agent framework")
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str = DEFAULT_TAVILY_BASE_URL,
        timeout: float = 30.0,
        http_client: AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout
        self._http_client = http_client
        self._owns_http_client = http_client is None
        # Cached FunctionTools so get_tools() is idempotent.
        self._tools: list[FunctionTool] | None = None

    # ------------------------------------------------------------------ lifecycle

    async def connect(self) -> None:
        """Create the underlying HTTP client if not already available."""
        if self._http_client is None:
            try:
                import httpx
            except ImportError as ex:
                raise ModuleNotFoundError(
                    "`TavilySearchClient` requires `httpx`. Please install `httpx`."
                ) from ex
            self._http_client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self._timeout,
            )

    async def close(self) -> None:
        """Close the underlying HTTP client (only if self-owned)."""
        if self._http_client is not None and self._owns_http_client:
            await self._http_client.aclose()
            self._http_client = None

    async def __aenter__(self) -> TavilySearchClient:
        await self.connect()
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.close()

    @property
    def is_connected(self) -> bool:
        return self._http_client is not None

    # ------------------------------------------------------------------ request core

    def _resolve_api_key(self) -> str | None:
        return self._api_key or os.environ.get(TAVILY_API_KEY_ENV)

    async def _post(self, path: str, payload: dict[str, Any]) -> str:
        """POST a JSON payload to a Tavily endpoint and return the response text."""
        api_key = self._resolve_api_key()
        if not api_key:
            return _error_text(
                "Missing Tavily API key: pass api_key= to TavilySearchClient "
                f"or set the {TAVILY_API_KEY_ENV} environment variable."
            )
        await self.connect()
        client = self._http_client
        if client is None:  # pragma: no cover - connect() guarantees a client
            return _error_text("Tavily HTTP client is not available.")
        import httpx

        try:
            response = await client.post(
                path,
                json=payload,
                headers={"Authorization": f"Bearer {api_key}"},
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as ex:
            return _error_text(self._describe_http_error(ex.response))
        except httpx.HTTPError as ex:
            return _error_text(f"Tavily request failed: {ex}")
        return json.dumps(response.json(), ensure_ascii=False, default=str)

    @staticmethod
    def _describe_http_error(response: Response) -> str:
        """Extract a human-readable message from a Tavily error response.

        Tavily error bodies look like ``{"detail": {"error": "..."}}`` (or
        ``{"detail": "..."}``). Status-specific hints are appended for the
        documented auth/quota codes (401/429/432/433).
        """
        detail: Any = None
        try:
            body: Any = response.json()
        except ValueError:
            body = None
        if isinstance(body, dict):
            detail = body.get("detail", body.get("error"))
        if isinstance(detail, dict):
            detail = detail.get("error") or detail
        hints = {
            400: " (invalid request)",
            401: " (check your API key)",
            429: " (rate limit exceeded)",
            432: " (plan usage limit exceeded)",
            433: " (pay-as-you-go limit exceeded)",
        }
        message = f"Tavily API error {response.status_code}: {detail or response.text}"
        hint = hints.get(response.status_code)
        return f"{message}{hint}" if hint else message

    # ------------------------------------------------------------------ REST API

    async def search(
        self,
        query: str,
        *,
        search_depth: SearchDepth = "basic",
        topic: SearchTopic = "general",
        max_results: int = 5,
        time_range: SearchTimeRange | None = None,
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
        include_raw_content: bool = False,
        include_answer: bool = False,
    ) -> str:
        """Call ``POST /search`` and return the response as JSON text.

        Args:
            query: The search query to execute.
            search_depth: Latency vs relevance tradeoff.
            topic: Category of the search.
            max_results: Maximum number of results (default 5).
            time_range: Only return results within this range back from today.
            include_domains: Restrict results to these domains.
            exclude_domains: Exclude results from these domains.
            include_raw_content: Include the cleaned page content per result.
            include_answer: Include an LLM-generated answer for the query.
        """
        payload: dict[str, Any] = {
            "query": query,
            "search_depth": search_depth,
            "topic": topic,
            "max_results": max_results,
            "include_raw_content": include_raw_content,
            "include_answer": include_answer,
        }
        if time_range is not None:
            payload["time_range"] = time_range
        if include_domains:
            payload["include_domains"] = include_domains
        if exclude_domains:
            payload["exclude_domains"] = exclude_domains
        return await self._post("/search", payload)

    async def extract(
        self,
        urls: list[str],
        *,
        query: str | None = None,
        extract_depth: ExtractDepth = "basic",
        output_format: ExtractFormat = "markdown",
    ) -> str:
        """Call ``POST /extract`` and return the response as JSON text.

        Args:
            urls: URLs to extract content from (max 20).
            query: Optional intent used to rerank the extracted content chunks.
            extract_depth: ``basic`` or ``advanced`` extraction.
            output_format: Extracted content format, ``markdown`` or ``text``.
        """
        payload: dict[str, Any] = {
            "urls": urls,
            "extract_depth": extract_depth,
            "format": output_format,
        }
        if query:
            payload["query"] = query
        return await self._post("/extract", payload)

    # ------------------------------------------------------------------ tool factories

    def _build_tools(self) -> list[FunctionTool]:
        """Create the two FunctionTools wrapping the REST calls."""

        # ---- web_search -------------------------------------------------
        @tool(
            name="web_search",
            description=(
                "Search the web via Tavily. Returns ranked results with title, url, "
                "content snippet and relevance score."
            ),
        )
        async def web_search(
            query: str,
            search_depth: SearchDepth = "basic",
            topic: SearchTopic = "general",
            max_results: int = 5,
            time_range: SearchTimeRange | None = None,
            include_domains: list[str] | None = None,
            exclude_domains: list[str] | None = None,
            include_raw_content: bool = False,
            include_answer: bool = False,
        ) -> str:
            """Search the web.

            Args:
                query: The search query to execute.
                search_depth: Latency vs relevance tradeoff ("basic" default).
                topic: Category of the search ("general", "news" or "finance").
                max_results: Maximum number of results to return (default 5).
                time_range: Only return results within this range back from today.
                include_domains: Only include results from these domains.
                exclude_domains: Exclude results from these domains.
                include_raw_content: Include the cleaned page content per result.
                include_answer: Include an LLM-generated answer for the query.
            """
            return await self.search(
                query,
                search_depth=search_depth,
                topic=topic,
                max_results=max_results,
                time_range=time_range,
                include_domains=include_domains,
                exclude_domains=exclude_domains,
                include_raw_content=include_raw_content,
                include_answer=include_answer,
            )

        # ---- web_fetch --------------------------------------------------
        @tool(
            name="web_fetch",
            description=(
                "Fetch and extract raw content from specific URLs (max 20) via Tavily Extract."
            ),
        )
        async def web_fetch(
            urls: list[str],
            query: str | None = None,
            extract_depth: ExtractDepth = "basic",
            output_format: ExtractFormat = "markdown",
        ) -> str:
            """Fetch URL(s).

            Args:
                urls: List of valid HTTP/HTTPS URLs to extract (max 20).
                query: Optional intent used to rerank the extracted content chunks.
                extract_depth: "basic" (default) or "advanced" extraction.
                output_format: Extracted content format, "markdown" (default) or "text".
            """
            return await self.extract(
                urls,
                query=query,
                extract_depth=extract_depth,
                output_format=output_format,
            )

        return [web_search, web_fetch]

    def get_tools(self) -> list[FunctionTool]:
        """Return the two tools ``[web_search, web_fetch]``.

        Idempotent — subsequent calls return the same instances. The tools
        lazily create the HTTP client on first invocation.
        """
        if self._tools is None:
            self._tools = self._build_tools()
        return self._tools


__all__ = ["DEFAULT_TAVILY_BASE_URL", "TAVILY_API_KEY_ENV", "TavilySearchClient"]
