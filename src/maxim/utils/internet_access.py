"""Internet access control and policy management.

Provides persistent internet_access flag and InternetAccessPolicy for
controlling network access independently of autonomy levels.
"""

from __future__ import annotations

import ipaddress
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Persistent Internet Access Flag
# ─────────────────────────────────────────────────────────────────────────────


DEFAULT_INTERNET_ACCESS_PATH = Path("data/util/internet_access.json")


@dataclass
class InternetAccessState:
    """Persistent state for internet access."""

    enabled: bool = False
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
            enabled=bool(data.get("enabled", False)),
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

        # Block list takes precedence
        if self.block_domains:
            for blocked in self.block_domains:
                if domain_lower == blocked.lower() or domain_lower.endswith(
                    "." + blocked.lower()
                ):
                    return False, f"Domain '{domain}' is blocked"

        # Allow list (if non-empty, only allow listed domains)
        if self.allow_domains:
            allowed = False
            for allow in self.allow_domains:
                if domain_lower == allow.lower() or domain_lower.endswith(
                    "." + allow.lower()
                ):
                    allowed = True
                    break
            if not allowed:
                return False, f"Domain '{domain}' not in allow list"

        return True, None

    def _is_private_ip(self, hostname: str) -> bool:
        """Check if hostname resolves to a private IP."""
        # Check obvious cases first
        if hostname in ("localhost", "127.0.0.1", "::1", "0.0.0.0"):
            return True

        # Check if it's a local hostname pattern
        if hostname.endswith(".local") or hostname.endswith(".localhost"):
            return True

        # Try to parse as IP address
        try:
            ip = ipaddress.ip_address(hostname)
            return ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved
        except ValueError:
            # Not an IP address, could be a hostname
            # We'll let it through and rely on DNS resolution checks at fetch time
            pass

        return False

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


def load_internet_policy(path: Path | str | None = None) -> InternetAccessPolicy:
    """Load internet access policy from file."""
    path = Path(path) if path else DEFAULT_POLICY_PATH

    # First load the access state to get enabled flag
    access_state = load_internet_access()

    if not path.exists():
        return InternetAccessPolicy(enabled=access_state.enabled)

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        policy = InternetAccessPolicy.from_dict(data)
        # Override enabled from access state
        policy.enabled = access_state.enabled
        return policy
    except Exception as e:
        logger.warning(f"Failed to load internet policy: {e}")
        return InternetAccessPolicy(enabled=access_state.enabled)


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
