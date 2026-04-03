"""Internet search tool using DuckDuckGo.

Provides web search capability with strict policy enforcement and
rate limiting. Requires internet_access to be enabled.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from typing import TYPE_CHECKING, Any
from urllib.parse import quote_plus, urlparse

from maxim.tools.base import Tool, ToolResult

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# HTTP Connection Pool (for performance)
# ─────────────────────────────────────────────────────────────────────────────

# Try to use urllib3 for connection pooling (saves 100-300ms per request)
_http_pool_manager = None
_http_pool_lock = threading.Lock()
_URLLIB3_AVAILABLE = False

try:
    import urllib3

    _URLLIB3_AVAILABLE = True
except ImportError:
    urllib3 = None  # type: ignore


def _get_http_pool():
    """Get or create the shared HTTP connection pool manager."""
    global _http_pool_manager
    if not _URLLIB3_AVAILABLE:
        return None
    if _http_pool_manager is None:
        with _http_pool_lock:
            if _http_pool_manager is None:
                _http_pool_manager = urllib3.PoolManager(
                    num_pools=4,
                    maxsize=4,
                    timeout=urllib3.Timeout(connect=5.0, read=10.0),
                    retries=urllib3.Retry(total=1, backoff_factor=0.5),
                )
    return _http_pool_manager


# ─────────────────────────────────────────────────────────────────────────────
# Pre-compiled Regex Patterns (for performance)
# ─────────────────────────────────────────────────────────────────────────────

# DuckDuckGo HTML parsing patterns - compiled once at module load
_LINK_PATTERN = re.compile(
    r'<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>([^<]+)</a>',
    re.IGNORECASE,
)
_FALLBACK_LINK_PATTERN = re.compile(
    r'<a[^>]*rel="nofollow"[^>]*href="(https?://[^"]+)"[^>]*>([^<]+)</a>',
    re.IGNORECASE,
)
_SNIPPET_PATTERN = re.compile(
    r'<a[^>]*class="[^"]*result__snippet[^"]*"[^>]*>([^<]+)</a>',
    re.IGNORECASE,
)
_SNIPPET_FALLBACK_PATTERN = re.compile(
    r'class="result__snippet"[^>]*>([^<]{20,})',
    re.IGNORECASE,
)


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


def _search_with_ddg_package(
    query: str,
    max_results: int = 5,
) -> list[dict[str, str]] | None:
    """Try using the ddgs/duckduckgo_search package for proper web search.

    Returns None if package is not available, otherwise returns results list.
    """
    try:
        from ddgs import DDGS
    except ImportError:
        return None  # Package not installed

    results: list[dict[str, str]] = []
    try:
        ddgs = DDGS()

        # Convert generator to list for better debugging
        search_results = list(ddgs.text(query, max_results=max_results))
        logger.debug(f"ddgs.text() returned {len(search_results)} items")

        # If text search fails, try news search (better for real-time content like sports)
        if not search_results:
            logger.debug("Text search empty, trying news search")
            try:
                search_results = list(ddgs.news(query, max_results=max_results))
                logger.debug(f"ddgs.news() returned {len(search_results)} items")
            except Exception as e:
                logger.debug(f"News search failed: {e}")

        # Log the raw results for debugging
        if search_results:
            logger.debug(f"First result keys: {search_results[0].keys() if isinstance(search_results[0], dict) else type(search_results[0])}")

        # Parse results
        for r in search_results:
            if isinstance(r, dict):
                # Handle both text and news result formats
                title = r.get("title", "")
                url = r.get("href") or r.get("link") or r.get("url") or ""
                snippet = r.get("body") or r.get("snippet") or r.get("description") or r.get("excerpt", "")

                if title and url:
                    results.append(
                        _make_search_result(
                            title=title,
                            url=url,
                            snippet=snippet,
                        )
                    )

        logger.info(f"ddgs package returned {len(results)} results")

        # If still no results, log structure info for debugging (without exposing content)
        if not results and search_results:
            first_item = search_results[0] if search_results else None
            item_type = type(first_item).__name__ if first_item else "None"
            item_keys = list(first_item.keys())[:5] if isinstance(first_item, dict) else "N/A"
            logger.warning(f"Search returned {len(search_results)} items but couldn't parse. Item type: {item_type}, keys: {item_keys}")

        return results
    except Exception as e:
        logger.warning(f"ddgs package failed: {e}", exc_info=True)
        return None


def _search_duckduckgo(
    query: str,
    max_results: int = 5,
    timeout_s: float = 8.0,
) -> list[dict[str, str]]:
    """Search DuckDuckGo for web results.

    Tries multiple methods in order:
    1. duckduckgo_search package (best, if installed)
    2. HTML scraping of DuckDuckGo lite
    3. Instant Answer API (limited, only for knowledge base queries)
    """
    # Method 1: Try the duckduckgo_search package first (best results)
    pkg_results = _search_with_ddg_package(query, max_results)
    if pkg_results is not None:
        return pkg_results

    logger.info("duckduckgo_search package not available, trying HTML scraping")

    # Method 2: Try HTML scraping first (better for web search)
    try:
        lite_results = _search_duckduckgo_lite(query, max_results, timeout_s)
        if lite_results:
            logger.info(f"DuckDuckGo lite scraper returned {len(lite_results)} results")
            return lite_results
    except Exception as e:
        logger.warning(f"DuckDuckGo lite scraper failed: {e}")

    logger.info("Lite scraper failed, trying Instant Answer API (limited)")

    # Method 3: Fall back to Instant Answer API (only works for knowledge queries)
    try:
        import urllib.request
        import urllib.error
    except ImportError:
        logger.error("urllib not available")
        return []

    results: list[dict[str, str]] = []

    # Try the Instant Answer API (note: this only returns knowledge base results, not web search)
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

    if results:
        logger.info(f"Instant Answer API returned {len(results)} results")
    else:
        logger.warning("All search methods failed - no results found")

    return results[:max_results]


def _search_duckduckgo_lite(
    query: str,
    max_results: int = 5,
    timeout_s: float = 8.0,
) -> list[dict[str, str]]:
    """Search DuckDuckGo using the HTML version.

    This scrapes the HTML search results page.
    Uses connection pooling when urllib3 is available for better performance.
    """
    results: list[dict[str, str]] = []

    # Use the HTML search page (not lite, which may be deprecated)
    search_url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"

    headers = {
        # Use a more realistic User-Agent to avoid blocks
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }

    try:
        # Try to use connection pool for better performance
        pool = _get_http_pool()
        if pool is not None:
            response = pool.request("GET", search_url, headers=headers)
            html = response.data.decode("utf-8", errors="ignore")
        else:
            # Fall back to urllib
            import urllib.request
            import urllib.error

            request = urllib.request.Request(search_url, headers=headers)
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                html = response.read().decode("utf-8", errors="ignore")

        logger.debug(f"DuckDuckGo HTML response length: {len(html)}")

        # Parse results from HTML using pre-compiled patterns (module level)
        links = _LINK_PATTERN.findall(html)
        if not links:
            logger.debug("Primary pattern found no results, trying fallback")
            links = _FALLBACK_LINK_PATTERN.findall(html)

        snippets = _SNIPPET_PATTERN.findall(html)
        if not snippets:
            snippets = _SNIPPET_FALLBACK_PATTERN.findall(html)

        logger.debug(f"Found {len(links)} links and {len(snippets)} snippets")

        # Filter out DuckDuckGo internal links
        filtered_links = [
            (url, title) for url, title in links
            if not url.startswith("//duckduckgo.com")
            and "duckduckgo.com" not in url
        ]

        for i, (url, title) in enumerate(filtered_links[:max_results]):
            snippet = snippets[i] if i < len(snippets) else ""
            # Clean up the URL (sometimes it's a redirect URL)
            if "uddg=" in url:
                # Extract actual URL from DuckDuckGo redirect
                import urllib.parse
                parsed = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
                actual_url = parsed.get("uddg", [url])[0]
                url = urllib.parse.unquote(actual_url)

            results.append(
                _make_search_result(
                    title=title.strip(),
                    url=url.strip(),
                    snippet=snippet.strip(),
                )
            )

    except Exception as e:
        logger.warning(f"DuckDuckGo HTML search error: {e}")

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

    # Domains to exclude from search results
    BLOCKED_DOMAINS: set[str] = {
        "grokipedia",
    }

    def __init__(
        self,
        get_internet_policy: Any | None = None,
        rate_limit_per_minute: int = 10,
        blocked_domains: set[str] | None = None,
    ):
        super().__init__()
        self._get_internet_policy = get_internet_policy
        self._rate_limit = rate_limit_per_minute
        self._request_times: list[float] = []
        # Merge instance-level blocked domains with class defaults
        self._blocked_domains = self.BLOCKED_DOMAINS.copy()
        if blocked_domains:
            self._blocked_domains.update(blocked_domains)

    def _is_blocked_domain(self, url: str) -> bool:
        """Check if URL is from a blocked domain.

        Uses proper URL parsing to extract the hostname and check if it
        matches or ends with any blocked domain pattern.
        """
        try:
            parsed = urlparse(url)
            hostname = (parsed.hostname or "").lower()

            if not hostname:
                return False

            for blocked in self._blocked_domains:
                blocked_lower = blocked.lower()
                # Check if hostname matches exactly or is a subdomain
                # e.g., "example.com" blocks "example.com" and "sub.example.com"
                # but not "notexample.com" or "example.com.malicious.net"
                if hostname == blocked_lower:
                    return True
                if hostname.endswith("." + blocked_lower):
                    return True

            return False
        except Exception as e:
            # If URL parsing fails, log and don't block
            logger.debug(f"Failed to parse URL for domain blocking: {url} - {e}")
            return False

    def _filter_results(self, results: list[dict[str, str]]) -> list[dict[str, str]]:
        """Filter out results from blocked domains."""
        filtered = []
        for result in results:
            url = result.get("url", "")
            if not self._is_blocked_domain(url):
                filtered.append(result)
            else:
                logger.debug(f"Filtered blocked domain: {url}")
        return filtered

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

            # Filter out blocked domains
            results = self._filter_results(results)

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

            set_internet_access(enabled, source="tool")
        else:
            self._set_internet_access(enabled, source="tool")

        return ToolResult(
            success=True,
            output=f"Internet access {'enabled' if enabled else 'disabled'}",
            metadata={
                "enabled": enabled,
                "reason": reason,
                "source": "tool",
            },
        )
