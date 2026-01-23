"""HTTP fetch tool with safety checks.

Provides URL fetching capability with SSRF protection, robots.txt
compliance, paywall detection, and content safety checks.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from maxim.tools.base import Tool, ToolResult

if TYPE_CHECKING:
    from maxim.utils.internet_access import InternetAccessPolicy

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Robots.txt Cache
# ─────────────────────────────────────────────────────────────────────────────


class RobotsCache:
    """Simple cache for robots.txt files."""

    def __init__(self, ttl_seconds: int = 3600):
        self._cache: dict[str, tuple[float, set[str]]] = {}
        self._ttl = ttl_seconds

    def get_disallowed(self, domain: str) -> set[str] | None:
        """Get disallowed paths for domain, or None if not cached."""
        if domain in self._cache:
            timestamp, paths = self._cache[domain]
            if time.time() - timestamp < self._ttl:
                return paths
            del self._cache[domain]
        return None

    def set_disallowed(self, domain: str, paths: set[str]) -> None:
        """Cache disallowed paths for domain."""
        self._cache[domain] = (time.time(), paths)

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()


_robots_cache = RobotsCache()


def _fetch_robots_txt(domain: str, timeout_s: float = 5.0) -> set[str]:
    """Fetch and parse robots.txt for a domain."""
    import urllib.request
    import urllib.error

    disallowed: set[str] = set()

    try:
        url = f"https://{domain}/robots.txt"
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "Maxim/1.0 (Research Assistant)"},
        )
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            content = response.read().decode("utf-8", errors="ignore")

            # Parse robots.txt (simple parser for User-agent: * rules)
            current_agent = None
            for line in content.split("\n"):
                line = line.strip().lower()

                if line.startswith("user-agent:"):
                    agent = line.split(":", 1)[1].strip()
                    current_agent = agent

                elif line.startswith("disallow:") and current_agent in ("*", "maxim"):
                    path = line.split(":", 1)[1].strip()
                    if path:
                        disallowed.add(path)

    except Exception as e:
        logger.debug(f"Failed to fetch robots.txt for {domain}: {e}")

    return disallowed


def check_robots_txt(url: str, timeout_s: float = 5.0) -> tuple[bool, str | None]:
    """Check if URL is allowed by robots.txt.

    Returns (allowed, reason).
    """
    try:
        parsed = urlparse(url)
        domain = parsed.hostname or ""
        path = parsed.path or "/"

        # Check cache first
        disallowed = _robots_cache.get_disallowed(domain)
        if disallowed is None:
            disallowed = _fetch_robots_txt(domain, timeout_s)
            _robots_cache.set_disallowed(domain, disallowed)

        # Check if path matches any disallowed pattern
        for pattern in disallowed:
            if path.startswith(pattern):
                return False, f"Path '{path}' disallowed by robots.txt"

        return True, None

    except Exception as e:
        logger.debug(f"robots.txt check error: {e}")
        return True, None  # Allow on error (fail open for now)


# ─────────────────────────────────────────────────────────────────────────────
# Paywall Detection
# ─────────────────────────────────────────────────────────────────────────────


PAYWALL_INDICATORS = [
    # Meta tags
    r'<meta[^>]*name=["\']?robots["\']?[^>]*content=["\']?noindex',
    r'<meta[^>]*name=["\']?paywall["\']?',
    # Common paywall class names
    r'class=["\'][^"\']*paywall[^"\']*["\']',
    r'class=["\'][^"\']*subscribe-wall[^"\']*["\']',
    r'class=["\'][^"\']*premium-content[^"\']*["\']',
    # Common paywall text patterns
    r"subscribe to (continue reading|read more|access)",
    r"(log in|sign in) to (continue|read|access)",
    r"this (content|article) is (for|only available to) (subscribers|members|premium)",
    # JSON-LD paywall indicator
    r'"isAccessibleForFree"\s*:\s*false',
]


def detect_paywall(html: str) -> bool:
    """Detect if content appears to be behind a paywall."""
    html_lower = html.lower()

    for pattern in PAYWALL_INDICATORS:
        if re.search(pattern, html_lower, re.IGNORECASE):
            return True

    return False


# ─────────────────────────────────────────────────────────────────────────────
# Content Extraction
# ─────────────────────────────────────────────────────────────────────────────


def extract_text_content(html: str, max_length: int = 50000) -> str:
    """Extract readable text content from HTML.

    Strips scripts, styles, and HTML tags to get clean text.
    """
    # Remove script and style elements
    html = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<noscript[^>]*>.*?</noscript>", "", html, flags=re.DOTALL | re.IGNORECASE)

    # Remove HTML comments
    html = re.sub(r"<!--.*?-->", "", html, flags=re.DOTALL)

    # Replace common block elements with newlines
    html = re.sub(r"<(br|p|div|h[1-6]|li|tr)[^>]*>", "\n", html, flags=re.IGNORECASE)

    # Remove all remaining HTML tags
    text = re.sub(r"<[^>]+>", " ", html)

    # Decode common HTML entities
    text = text.replace("&nbsp;", " ")
    text = text.replace("&amp;", "&")
    text = text.replace("&lt;", "<")
    text = text.replace("&gt;", ">")
    text = text.replace("&quot;", '"')
    text = text.replace("&#39;", "'")

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\n\s*\n", "\n\n", text)
    text = text.strip()

    return text[:max_length]


def extract_title(html: str) -> str:
    """Extract page title from HTML."""
    match = re.search(r"<title[^>]*>([^<]+)</title>", html, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# HTTP Fetch Tool
# ─────────────────────────────────────────────────────────────────────────────


class HttpFetchTool(Tool):
    """Tool for fetching content from URLs.

    Includes safety checks:
    - SSRF protection (blocks private IPs)
    - robots.txt compliance
    - Paywall detection
    - Content size limits
    - Timeout enforcement
    """

    name = "http_fetch"
    description = "Fetch content from a URL with safety checks"
    input_schema = {
        "url": str,  # Required: URL to fetch
        "extract_text": (bool, True),  # Optional: extract text from HTML
        "max_bytes": (int, None),  # Optional: override max bytes
    }

    # Tool metadata
    requires_internet_access = True

    def __init__(
        self,
        get_internet_policy: Any | None = None,
        content_safety_checker: Any | None = None,
    ):
        super().__init__()
        self._get_internet_policy = get_internet_policy
        self._content_safety_checker = content_safety_checker
        self._request_times: list[float] = []

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the fetch."""
        url = str(kwargs.get("url", "")).strip()
        extract_text = bool(kwargs.get("extract_text", True))
        max_bytes_override = kwargs.get("max_bytes")

        if not url:
            return ToolResult(
                success=False,
                error="No URL provided",
            )

        # Ensure URL has scheme
        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        # Check internet access policy
        if self._get_internet_policy:
            policy = self._get_internet_policy()
            can_access, reason = policy.can_access(url)
            if not can_access:
                return ToolResult(
                    success=False,
                    error=reason or "URL not allowed by policy",
                    metadata={"policy_blocked": True, "url": url},
                )

            max_bytes = max_bytes_override or policy.max_fetch_bytes
            timeout_s = policy.request_timeout_s
            check_robots = policy.require_robots_ok
            block_paywalled = policy.block_paywalled
            check_safety = policy.unsafe_content_checks
        else:
            return ToolResult(
                success=False,
                error="Internet access policy not configured",
                metadata={"policy_blocked": True},
            )

        # Check rate limit
        now = time.time()
        self._request_times = [t for t in self._request_times if now - t < 60]
        if len(self._request_times) >= policy.max_pages_per_minute:
            return ToolResult(
                success=False,
                error=f"Rate limit exceeded ({policy.max_pages_per_minute} pages/minute)",
                metadata={"rate_limited": True},
            )

        # Check robots.txt
        if check_robots:
            robots_allowed, robots_reason = check_robots_txt(url, timeout_s=timeout_s)
            if not robots_allowed:
                return ToolResult(
                    success=False,
                    error=robots_reason or "Blocked by robots.txt",
                    metadata={"robots_blocked": True, "url": url},
                )

        # Perform the fetch
        try:
            import urllib.request
            import urllib.error

            self._request_times.append(now)

            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "Maxim/1.0 (Research Assistant; +https://github.com/maxim)",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.5",
                },
            )

            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                # Check content type
                content_type = response.headers.get("Content-Type", "")

                # Read content with size limit
                content = response.read(max_bytes)
                if len(content) >= max_bytes:
                    logger.warning(f"Content truncated at {max_bytes} bytes")

                # Get final URL (after redirects)
                final_url = response.geturl()

                # Decode content
                encoding = "utf-8"
                if "charset=" in content_type:
                    match = re.search(r"charset=([^\s;]+)", content_type)
                    if match:
                        encoding = match.group(1)

                try:
                    html = content.decode(encoding, errors="ignore")
                except Exception:
                    html = content.decode("utf-8", errors="ignore")

                # Check for paywall
                if block_paywalled and detect_paywall(html):
                    return ToolResult(
                        success=False,
                        error="Content appears to be behind a paywall",
                        metadata={"paywalled": True, "url": final_url},
                    )

                # Check content safety
                if check_safety and self._content_safety_checker:
                    is_safe, safety_reason = self._content_safety_checker(html)
                    if not is_safe:
                        return ToolResult(
                            success=False,
                            error=safety_reason or "Content failed safety check",
                            metadata={"unsafe_content": True, "url": final_url},
                        )

                # Extract content
                title = extract_title(html)
                text = extract_text_content(html) if extract_text else ""

                # Build result
                result = {
                    "url": final_url,
                    "title": title,
                    "content_type": content_type,
                    "content_length": len(content),
                }

                if extract_text:
                    result["text"] = text
                    result["text_length"] = len(text)

                # Build citation
                citation = {
                    "url": final_url,
                    "title": title,
                    "accessed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }

                return ToolResult(
                    success=True,
                    output=result,
                    metadata={
                        "citation": citation,
                        "content_hash": hashlib.sha256(content).hexdigest()[:16],
                    },
                )

        except urllib.error.HTTPError as e:
            return ToolResult(
                success=False,
                error=f"HTTP error {e.code}: {e.reason}",
                metadata={"http_code": e.code, "url": url},
            )

        except urllib.error.URLError as e:
            return ToolResult(
                success=False,
                error=f"URL error: {e.reason}",
                metadata={"url": url},
            )

        except TimeoutError:
            return ToolResult(
                success=False,
                error=f"Request timed out after {timeout_s}s",
                metadata={"timeout": True, "url": url},
            )

        except Exception as e:
            logger.error(f"Fetch failed: {e}")
            return ToolResult(
                success=False,
                error=f"Fetch failed: {e}",
                metadata={"url": url},
            )
