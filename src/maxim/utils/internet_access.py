"""Internet access control and policy management.

Provides persistent internet_access flag and InternetAccessPolicy for
controlling network access independently of autonomy levels.
"""

from __future__ import annotations

import ipaddress
import json
import logging
import re
import socket
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# DNS Resolution Cache (for performance)
# ─────────────────────────────────────────────────────────────────────────────

_dns_cache: dict[str, tuple[float, list[tuple]]] = {}
_dns_cache_lock = threading.Lock()
_DNS_CACHE_TTL = 300.0  # 5 minutes - DNS records rarely change faster
_DNS_CACHE_MAX_SIZE = 1000


def _cached_getaddrinfo(
    hostname: str,
    port: int | None = None,
    ttl: float = _DNS_CACHE_TTL,
) -> list[tuple]:
    """Cached DNS resolution to avoid repeated lookups.

    DNS lookups can take 50-200ms per request. This cache eliminates
    repeated lookups for the same hostname within the TTL window.
    """
    cache_key = f"{hostname}:{port}"
    now = time.time()

    # Check cache first (fast path)
    with _dns_cache_lock:
        if cache_key in _dns_cache:
            cached_time, cached_result = _dns_cache[cache_key]
            if now - cached_time < ttl:
                return cached_result

    # Perform actual DNS lookup (outside lock to avoid blocking)
    result = socket.getaddrinfo(hostname, port, socket.AF_UNSPEC, socket.SOCK_STREAM)

    # Store in cache
    with _dns_cache_lock:
        # Evict oldest entries if cache is full
        if len(_dns_cache) >= _DNS_CACHE_MAX_SIZE:
            sorted_keys = sorted(_dns_cache.keys(), key=lambda k: _dns_cache[k][0])
            for k in sorted_keys[: _DNS_CACHE_MAX_SIZE // 5]:  # Evict oldest 20%
                del _dns_cache[k]
        _dns_cache[cache_key] = (now, result)

    return result


def _redact_hostname(hostname: str) -> str:
    """Redact hostname for logging while preserving TLD for debugging.

    Examples:
        internal.corp.company.com -> ***.company.com
        192.168.1.100 -> 192.168.*.*
        secret-server.local -> ***.local
    """
    if not hostname:
        return hostname

    # Check if it looks like an IP address
    if re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", hostname):
        parts = hostname.split(".")
        return f"{parts[0]}.{parts[1]}.*.*"

    # For hostnames, preserve TLD and last domain part
    parts = hostname.split(".")
    if len(parts) <= 2:
        return "***." + ".".join(parts[-1:]) if parts else "***"
    return "***." + ".".join(parts[-2:])


# ─────────────────────────────────────────────────────────────────────────────
# Persistent Internet Access Flag
# ─────────────────────────────────────────────────────────────────────────────


DEFAULT_INTERNET_ACCESS_PATH = Path("data/util/internet_access.json")


@dataclass
class InternetAccessState:
    """Persistent state for internet access."""

    enabled: bool = True  # Default to enabled for exploration mode
    updated_at: str = ""
    source: str = "default"  # "cli", "voice", "tool", "default"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "enabled": self.enabled,
            "updated_at": self.updated_at,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InternetAccessState:
        """Deserialize from dictionary."""
        return cls(
            enabled=bool(data.get("enabled", True)),  # Default to enabled
            updated_at=str(data.get("updated_at", "")),
            source=str(data.get("source", "default")),
        )


def load_internet_access(path: Path | str | None = None) -> InternetAccessState:
    """Load internet access state from persistent storage."""
    path = Path(path) if path else DEFAULT_INTERNET_ACCESS_PATH

    if not path.exists():
        return InternetAccessState()

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return InternetAccessState.from_dict(data)
    except Exception as e:
        logger.warning(f"Failed to load internet access state: {e}")
        return InternetAccessState()


def save_internet_access(
    state: InternetAccessState, path: Path | str | None = None
) -> bool:
    """Save internet access state to persistent storage."""
    path = Path(path) if path else DEFAULT_INTERNET_ACCESS_PATH

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        state.updated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        with open(path, "w", encoding="utf-8") as f:
            json.dump(state.to_dict(), f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Failed to save internet access state: {e}")
        return False


def set_internet_access(
    enabled: bool, source: str = "tool", path: Path | str | None = None
) -> InternetAccessState:
    """Set internet access state."""
    state = InternetAccessState(enabled=enabled, source=source)
    save_internet_access(state, path)
    logger.info(f"Internet access {'enabled' if enabled else 'disabled'} (source={source})")
    return state


# ─────────────────────────────────────────────────────────────────────────────
# Internet Access Policy
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class InternetAccessPolicy:
    """Network access rules independent of autonomy."""

    enabled: bool = False

    # Domain filtering
    allow_domains: set[str] = field(default_factory=set)
    block_domains: set[str] = field(default_factory=set)

    # Content policies
    require_robots_ok: bool = True
    block_paywalled: bool = True
    allow_paywalled_with_credentials: bool = False
    unsafe_content_checks: bool = True

    # Rate limits and size limits
    max_fetch_bytes: int = 1_000_000  # 1 MB
    max_pages_per_minute: int = 10
    request_timeout_s: float = 8.0

    # Cache settings
    retention_seconds: int = 900  # 15 minutes TTL for raw content

    # Output requirements
    citations_required: bool = True

    # SSRF protection - private IP ranges to block
    block_private_ips: bool = True

    # Performance: pre-lowercased domain sets (computed in __post_init__)
    _allow_domains_lower: frozenset[str] = field(default_factory=frozenset, repr=False, compare=False)
    _block_domains_lower: frozenset[str] = field(default_factory=frozenset, repr=False, compare=False)

    def __post_init__(self):
        """Pre-compute lowercased domain sets for O(1) lookup."""
        # Only update if the lowercase sets are empty but the original sets have values
        # This avoids re-processing when loading from cache
        if not self._allow_domains_lower and self.allow_domains:
            object.__setattr__(self, '_allow_domains_lower', frozenset(d.lower() for d in self.allow_domains))
        if not self._block_domains_lower and self.block_domains:
            object.__setattr__(self, '_block_domains_lower', frozenset(d.lower() for d in self.block_domains))

    def can_access(self, url: str) -> tuple[bool, str | None]:
        """Check if URL can be accessed according to policy.

        Returns (allowed, reason) where reason explains why access was denied.
        """
        if not self.enabled:
            return False, "Internet access is disabled"

        try:
            parsed = urlparse(url)
        except Exception as e:
            return False, f"Invalid URL: {e}"

        # Check scheme
        if parsed.scheme not in ("http", "https"):
            return False, f"Scheme '{parsed.scheme}' not allowed (only http/https)"

        # Check for private IPs (SSRF protection)
        if self.block_private_ips:
            hostname = parsed.hostname or ""
            if self._is_private_ip(hostname):
                return False, "Access to private/internal IPs is blocked"

        # Check domain allow/block lists
        domain = parsed.hostname or ""
        domain_lower = domain.lower()

        # Block list takes precedence (using pre-lowercased set for performance)
        if self._block_domains_lower:
            # Check exact match first (O(1))
            if domain_lower in self._block_domains_lower:
                return False, f"Domain '{domain}' is blocked"
            # Check subdomain matches
            for blocked in self._block_domains_lower:
                if domain_lower.endswith("." + blocked):
                    return False, f"Domain '{domain}' is blocked"

        # Allow list (if non-empty, only allow listed domains)
        if self._allow_domains_lower:
            # Check exact match first (O(1))
            if domain_lower in self._allow_domains_lower:
                return True, None
            # Check subdomain matches
            allowed = False
            for allow in self._allow_domains_lower:
                if domain_lower.endswith("." + allow):
                    allowed = True
                    break
            if not allowed:
                return False, f"Domain '{domain}' not in allow list"

        return True, None

    def _is_private_ip(self, hostname: str) -> bool:
        """Check if hostname resolves to a private IP.

        Resolves hostnames via DNS to prevent DNS rebinding attacks where
        an attacker's hostname initially resolves to a public IP (to pass
        the check) but later resolves to a private IP during the actual fetch.
        """
        # Check obvious cases first
        if hostname in ("localhost", "127.0.0.1", "::1", "0.0.0.0"):
            return True

        # Check if it's a local hostname pattern
        if hostname.endswith(".local") or hostname.endswith(".localhost"):
            return True

        # Try to parse as IP address first
        try:
            ip = ipaddress.ip_address(hostname)
            return self._check_ip_is_private(ip)
        except ValueError:
            # Not an IP address - resolve via DNS and check ALL resolved IPs
            pass

        # Resolve hostname and check all resulting IPs (using cached DNS)
        try:
            # getaddrinfo returns all resolved addresses (IPv4 and IPv6)
            addr_info = _cached_getaddrinfo(hostname, None)
            for family, _, _, _, sockaddr in addr_info:
                ip_str = sockaddr[0]
                try:
                    ip = ipaddress.ip_address(ip_str)
                    if self._check_ip_is_private(ip):
                        logger.warning(
                            f"DNS rebinding protection: {_redact_hostname(hostname)} resolved to private IP"
                        )
                        return True
                except ValueError:
                    continue
        except socket.gaierror as e:
            # DNS resolution failed - block to be safe
            logger.warning(f"DNS resolution failed for {_redact_hostname(hostname)}: {type(e).__name__}")
            return True
        except Exception as e:
            # Any other error - block to be safe
            logger.warning(f"Error checking hostname {_redact_hostname(hostname)}: {type(e).__name__}")
            return True

        return False

    def _check_ip_is_private(self, ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
        """Check if an IP address is private, loopback, link-local, or reserved."""
        return ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved

    def summary(self) -> str:
        """Get a summary of the policy for LLM context."""
        parts = []

        if not self.enabled:
            return "Internet access is DISABLED."

        parts.append("Internet access is ENABLED.")

        if self.require_robots_ok:
            parts.append("Must respect robots.txt.")
        if self.block_paywalled:
            parts.append("Paywalled content blocked.")
        if self.unsafe_content_checks:
            parts.append("Unsafe content checks active.")
        if self.citations_required:
            parts.append("Citations required for web sources.")
        if self.allow_domains:
            parts.append(f"Allowed domains: {', '.join(sorted(self.allow_domains))}")
        if self.block_domains:
            parts.append(f"Blocked domains: {', '.join(sorted(self.block_domains))}")

        return " ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "enabled": self.enabled,
            "allow_domains": list(self.allow_domains),
            "block_domains": list(self.block_domains),
            "require_robots_ok": self.require_robots_ok,
            "block_paywalled": self.block_paywalled,
            "allow_paywalled_with_credentials": self.allow_paywalled_with_credentials,
            "unsafe_content_checks": self.unsafe_content_checks,
            "max_fetch_bytes": self.max_fetch_bytes,
            "max_pages_per_minute": self.max_pages_per_minute,
            "request_timeout_s": self.request_timeout_s,
            "retention_seconds": self.retention_seconds,
            "citations_required": self.citations_required,
            "block_private_ips": self.block_private_ips,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InternetAccessPolicy:
        """Deserialize from dictionary."""
        return cls(
            enabled=bool(data.get("enabled", False)),
            allow_domains=set(data.get("allow_domains", [])),
            block_domains=set(data.get("block_domains", [])),
            require_robots_ok=bool(data.get("require_robots_ok", True)),
            block_paywalled=bool(data.get("block_paywalled", True)),
            allow_paywalled_with_credentials=bool(
                data.get("allow_paywalled_with_credentials", False)
            ),
            unsafe_content_checks=bool(data.get("unsafe_content_checks", True)),
            max_fetch_bytes=int(data.get("max_fetch_bytes", 1_000_000)),
            max_pages_per_minute=int(data.get("max_pages_per_minute", 10)),
            request_timeout_s=float(data.get("request_timeout_s", 8.0)),
            retention_seconds=int(data.get("retention_seconds", 900)),
            citations_required=bool(data.get("citations_required", True)),
            block_private_ips=bool(data.get("block_private_ips", True)),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Policy Loader
# ─────────────────────────────────────────────────────────────────────────────


DEFAULT_POLICY_PATH = Path("data/util/internet_policy.json")

# Policy cache with mtime validation
_cached_policy: InternetAccessPolicy | None = None
_cached_policy_mtime: float = 0.0
_cached_policy_enabled: bool | None = None  # Track access state too
_cached_policy_lock = threading.Lock()


def load_internet_policy(path: Path | str | None = None) -> InternetAccessPolicy:
    """Load internet access policy from file (cached with mtime validation).

    The policy is cached and only reloaded if the file has been modified
    or the internet access enabled state has changed.
    """
    global _cached_policy, _cached_policy_mtime, _cached_policy_enabled
    path = Path(path) if path else DEFAULT_POLICY_PATH

    # First load the access state to get enabled flag
    access_state = load_internet_access()

    # Check cache validity
    with _cached_policy_lock:
        if _cached_policy is not None:
            try:
                current_mtime = path.stat().st_mtime if path.exists() else 0.0
                # Cache hit if file unchanged AND enabled state unchanged
                if (
                    current_mtime == _cached_policy_mtime
                    and access_state.enabled == _cached_policy_enabled
                ):
                    return _cached_policy
            except OSError:
                pass  # File access issue, reload

    # Load fresh
    if not path.exists():
        policy = InternetAccessPolicy(enabled=access_state.enabled)
    else:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            policy = InternetAccessPolicy.from_dict(data)
            # Override enabled from access state
            policy.enabled = access_state.enabled
        except Exception as e:
            logger.warning(f"Failed to load internet policy: {e}")
            policy = InternetAccessPolicy(enabled=access_state.enabled)

    # Update cache
    with _cached_policy_lock:
        _cached_policy = policy
        _cached_policy_enabled = access_state.enabled
        try:
            _cached_policy_mtime = path.stat().st_mtime if path.exists() else 0.0
        except OSError:
            _cached_policy_mtime = 0.0

    return policy


def save_internet_policy(
    policy: InternetAccessPolicy, path: Path | str | None = None
) -> bool:
    """Save internet access policy to file."""
    path = Path(path) if path else DEFAULT_POLICY_PATH

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(policy.to_dict(), f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Failed to save internet policy: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Citations
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Citation:
    """A citation for web-sourced content."""

    url: str
    title: str = ""
    accessed_at: str = ""
    snippet: str = ""

    def to_dict(self) -> dict[str, str]:
        """Serialize to dictionary."""
        return {
            "url": self.url,
            "title": self.title,
            "accessed_at": self.accessed_at,
            "snippet": self.snippet,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Citation:
        """Deserialize from dictionary."""
        return cls(
            url=str(data.get("url", "")),
            title=str(data.get("title", "")),
            accessed_at=str(data.get("accessed_at", "")),
            snippet=str(data.get("snippet", "")),
        )

    def format_short(self) -> str:
        """Format as short citation for CLI/voice."""
        if self.title:
            return f"[{self.title}]({self.url})"
        return self.url

    def format_full(self) -> str:
        """Format as full citation."""
        parts = []
        if self.title:
            parts.append(f"**{self.title}**")
        parts.append(self.url)
        if self.accessed_at:
            parts.append(f"(accessed {self.accessed_at})")
        return " - ".join(parts)
