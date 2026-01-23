"""Internet search tool using DuckDuckGo.

Provides web search capability with strict policy enforcement and
rate limiting. Requires internet_access to be enabled.
"""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING, Any
from urllib.parse import quote_plus, urlencode

from maxim.tools.base import Tool, ToolResult

if TYPE_CHECKING:
    from maxim.utils.internet_access import InternetAccessPolicy

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Search Result Types
# ─────────────────────────────────────────────────────────────────────────────


def _make_search_result(
    title: str, url: str, snippet: str, source: str = "duckduckgo"
) -> dict[str, str]:
    """Create a standardized search result."""
    return {
        "title": title,
        "url": url,
        "snippet": snippet,
        "source": source,
    }


# ─────────────────────────────────────────────────────────────────────────────
# DuckDuckGo Search Implementation
# ─────────────────────────────────────────────────────────────────────────────


def _search_duckduckgo(
    query: str,
    max_results: int = 5,
    timeout_s: float = 8.0,
) -> list[dict[str, str]]:
    """Search DuckDuckGo using the Instant Answer API.

    Note: The Instant Answer API returns limited results. For more results,
    we fall back to HTML scraping of the lite version.
    """
    try:
        import urllib.request
        import urllib.error
    except ImportError:
        logger.error("urllib not available")
        return []

    results: list[dict[str, str]] = []

    # Try the Instant Answer API first
    api_url = f"https://api.duckduckgo.com/?q={quote_plus(query)}&format=json&no_html=1&skip_disambig=1"

    try:
        request = urllib.request.Request(
            api_url,
            headers={
                "User-Agent": "Maxim/1.0 (Research Assistant; +https://github.com/maxim)"
            },
        )
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            data = json.loads(response.read().decode("utf-8"))

            # Check for abstract (main result)
            if data.get("Abstract"):
                results.append(
                    _make_search_result(
                        title=data.get("Heading", query),
                        url=data.get("AbstractURL", ""),
                        snippet=data.get("Abstract", ""),
                    )
                )

            # Check for related topics
            for topic in data.get("RelatedTopics", [])[:max_results - len(results)]:
                if isinstance(topic, dict):
                    if "Topics" in topic:
                        # Nested topics
                        for subtopic in topic.get("Topics", []):
                            if len(results) >= max_results:
                                break
                            if isinstance(subtopic, dict) and subtopic.get("FirstURL"):
                                results.append(
                                    _make_search_result(
                                        title=subtopic.get("Text", "")[:100],
                                        url=subtopic.get("FirstURL", ""),
                                        snippet=subtopic.get("Text", ""),
                                    )
                                )
                    elif topic.get("FirstURL"):
                        results.append(
                            _make_search_result(
                                title=topic.get("Text", "")[:100],
                                url=topic.get("FirstURL", ""),
                                snippet=topic.get("Text", ""),
                            )
                        )

            # Check for results (rarely populated)
            for result in data.get("Results", [])[:max_results - len(results)]:
                if isinstance(result, dict) and result.get("FirstURL"):
                    results.append(
                        _make_search_result(
                            title=result.get("Text", "")[:100],
                            url=result.get("FirstURL", ""),
                            snippet=result.get("Text", ""),
                        )
                    )

    except urllib.error.URLError as e:
        logger.warning(f"DuckDuckGo API request failed: {e}")
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse DuckDuckGo response: {e}")
    except Exception as e:
        logger.warning(f"DuckDuckGo search error: {e}")

    # If we didn't get enough results from API, try the lite HTML version
    if len(results) < max_results:
        try:
            results.extend(
                _search_duckduckgo_lite(
                    query,
                    max_results=max_results - len(results),
                    timeout_s=timeout_s,
                )
            )
        except Exception as e:
            logger.debug(f"DuckDuckGo lite fallback failed: {e}")

    return results[:max_results]


def _search_duckduckgo_lite(
    query: str,
    max_results: int = 5,
    timeout_s: float = 8.0,
) -> list[dict[str, str]]:
    """Search DuckDuckGo using the lite HTML version.

    This is a fallback for when the Instant Answer API doesn't return results.
    """
    try:
        import urllib.request
        import urllib.error
        import re
    except ImportError:
        return []

    results: list[dict[str, str]] = []

    # Use the lite (text-only) version
    lite_url = f"https://lite.duckduckgo.com/lite/?q={quote_plus(query)}"

    try:
        request = urllib.request.Request(
            lite_url,
            headers={
                "User-Agent": "Maxim/1.0 (Research Assistant)"
            },
        )
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            html = response.read().decode("utf-8", errors="ignore")

            # Parse results from lite HTML
            # Results are in table rows with class "result-link"
            # Pattern: <a rel="nofollow" class="result-link" href="URL">TITLE</a>
            link_pattern = re.compile(
                r'<a[^>]*class="result-link"[^>]*href="([^"]+)"[^>]*>([^<]+)</a>',
                re.IGNORECASE,
            )

            # Also try to find snippets
            # Pattern: <td class="result-snippet">SNIPPET</td>
            snippet_pattern = re.compile(
                r'<td[^>]*class="result-snippet"[^>]*>([^<]+)</td>',
                re.IGNORECASE,
            )

            links = link_pattern.findall(html)
            snippets = snippet_pattern.findall(html)

            for i, (url, title) in enumerate(links[:max_results]):
                snippet = snippets[i] if i < len(snippets) else ""
                results.append(
                    _make_search_result(
                        title=title.strip(),
                        url=url.strip(),
                        snippet=snippet.strip(),
                    )
                )

    except Exception as e:
        logger.debug(f"DuckDuckGo lite search error: {e}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Internet Search Tool
# ─────────────────────────────────────────────────────────────────────────────


class InternetSearchTool(Tool):
    """Tool for searching the internet via DuckDuckGo.

    Requires internet_access to be enabled. Results include citations
    for transparency.
    """

    name = "internet_search"
    description = "Search the internet for information using DuckDuckGo"
    input_schema = {
        "query": str,  # Required: search query
        "max_results": (int, 5),  # Optional: max results (default 5)
    }

    # Tool metadata
    requires_internet_access = True

    def __init__(
        self,
        get_internet_policy: Any | None = None,
        rate_limit_per_minute: int = 10,
    ):
        super().__init__()
        self._get_internet_policy = get_internet_policy
        self._rate_limit = rate_limit_per_minute
        self._request_times: list[float] = []

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the search."""
        query = str(kwargs.get("query", "")).strip()
        max_results = int(kwargs.get("max_results", 5))

        if not query:
            return ToolResult(
                success=False,
                error="No search query provided",
            )

        # Check internet access policy
        if self._get_internet_policy:
            policy = self._get_internet_policy()
            if not policy.enabled:
                return ToolResult(
                    success=False,
                    error="Internet access is disabled",
                    metadata={"policy_blocked": True},
                )
        else:
            # No policy checker - default to blocked
            return ToolResult(
                success=False,
                error="Internet access policy not configured",
                metadata={"policy_blocked": True},
            )

        # Check rate limit
        now = time.time()
        self._request_times = [t for t in self._request_times if now - t < 60]
        if len(self._request_times) >= self._rate_limit:
            return ToolResult(
                success=False,
                error=f"Rate limit exceeded ({self._rate_limit} requests/minute)",
                metadata={"rate_limited": True},
            )

        self._request_times.append(now)

        # Perform the search
        try:
            timeout = 8.0
            if self._get_internet_policy:
                policy = self._get_internet_policy()
                timeout = policy.request_timeout_s

            results = _search_duckduckgo(
                query,
                max_results=min(max_results, 10),  # Cap at 10
                timeout_s=timeout,
            )

            if not results:
                return ToolResult(
                    success=True,
                    output=[],
                    metadata={
                        "query": query,
                        "result_count": 0,
                        "message": "No results found",
                    },
                )

            # Build citations
            citations = [
                {"title": r["title"], "url": r["url"]}
                for r in results
                if r.get("url")
            ]

            return ToolResult(
                success=True,
                output=results,
                metadata={
                    "query": query,
                    "result_count": len(results),
                    "citations": citations,
                    "provider": "duckduckgo",
                },
            )

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return ToolResult(
                success=False,
                error=f"Search failed: {e}",
            )


# ─────────────────────────────────────────────────────────────────────────────
# Internet Access Toggle Tool
# ─────────────────────────────────────────────────────────────────────────────


class InternetAccessTool(Tool):
    """Tool for enabling/disabling internet access.

    This is a privileged tool that should require confirmation.
    """

    name = "internet_access_toggle"
    description = "Enable or disable internet access"
    input_schema = {
        "enabled": bool,  # Required: enable or disable
        "reason": (str, ""),  # Optional: reason for change
    }

    def __init__(
        self,
        set_internet_access_fn: Any | None = None,
    ):
        super().__init__()
        self._set_internet_access = set_internet_access_fn

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the toggle."""
        enabled = bool(kwargs.get("enabled", False))
        reason = str(kwargs.get("reason", "")).strip()

        if not self._set_internet_access:
            # Use default implementation
            from maxim.utils.internet_access import set_internet_access

            state = set_internet_access(enabled, source="tool")
        else:
            state = self._set_internet_access(enabled, source="tool")

        return ToolResult(
            success=True,
            output=f"Internet access {'enabled' if enabled else 'disabled'}",
            metadata={
                "enabled": enabled,
                "reason": reason,
                "source": "tool",
            },
        )
