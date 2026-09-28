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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
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


def _default_internet_access_path() -> Path:
    from maxim.utils.paths import resolve_user_state

    return resolve_user_state("util/internet_access.json")


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
        """Deserialize. ``ValueError`` unless ``enabled`` is a real boolean: a hand-written ``"false"``
        is a truthy string, and reading it as ON would defeat the fail-closed load (#832)."""
        enabled = data.get("enabled") if isinstance(data, dict) else None
        if not isinstance(enabled, bool):
            raise ValueError(f"internet access state needs 'enabled': true or false, got {enabled!r}")
        return cls(
            enabled=enabled,
            updated_at=str(data.get("updated_at", "")),
            source=str(data.get("source", "default")),
        )


def load_internet_access(path: Path | str | None = None) -> InternetAccessState:
    """Load internet access state from persistent storage.

    No file means the default (on). A file that exists but cannot be read means an unknown operator
    or agent choice, so it FAILS CLOSED (off, source ``"unreadable"``), like the policy file (#832);
    before, it fell back to on, so a corrupted "off" turned internet back on.
    """
    path = Path(path) if path else _default_internet_access_path()

    try:
        if not path.exists():
            return InternetAccessState()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        from maxim.utils.format_version import check_format_version

        check_format_version(data, "internet_access", log=logger)
        return InternetAccessState.from_dict(data)
    except Exception as e:  # noqa: BLE001 -- any read/parse failure means the state is unknown
        logger.error("Internet access state %s is unreadable (%s); internet access DISABLED.", path, e)
        return InternetAccessState(enabled=False, source="unreadable")


def save_internet_access(state: InternetAccessState, path: Path | str | None = None) -> bool:
    """Save internet access state to persistent storage."""
    path = Path(path) if path else _default_internet_access_path()

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        state.updated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        from maxim.utils.atomic_io import atomic_write_json
        from maxim.utils.format_version import with_format_version

        atomic_write_json(str(path), with_format_version(state.to_dict()))
        return True
    except Exception as e:
        logger.error(f"Failed to save internet access state: {e}")
        return False


def set_internet_access(enabled: bool, source: str = "tool", path: Path | str | None = None) -> InternetAccessState:
    """Set internet access state."""
    state = InternetAccessState(enabled=enabled, source=source)
    save_internet_access(state, path)
    logger.info(f"Internet access {'enabled' if enabled else 'disabled'} (source={source})")
    return state


# ─────────────────────────────────────────────────────────────────────────────
# Internet Access Policy
# ─────────────────────────────────────────────────────────────────────────────


def normalize_host(name: str) -> str:
    """A host name in the one form every list check compares: stripped, lowercased, no trailing dot,
    and punycode for a non-ASCII name. ``ValueError`` when it is not a valid IDNA name.

    The codec is Python's IDNA 2003: for the few deviation characters (``ß``, ``ς``, ZWJ/ZWNJ) it
    maps differently from the IDNA 2008 form a URL may carry, so an entry such as ``straße.de`` does
    not match ``xn--strae-oqa.de`` (#968). Use the punycode form in the policy for those names.
    """
    host = name.strip().lower().rstrip(".")
    if host and not host.isascii():
        try:
            host = host.encode("idna").decode("ascii")
        except UnicodeError as e:
            raise ValueError(f"not a valid domain name: {name!r} ({e})") from e
    return host


def _domain_set(value: Any, field_name: str) -> frozenset[str]:
    """A domain list as normalized names (``normalize_host``), or ``ValueError``.

    A bare string would iterate as single characters and block nothing, so it is refused, as is a
    non-string member, an empty name, or a name that is not valid IDNA.
    """
    if not isinstance(value, (list, tuple, set, frozenset)):  # a bare str is refused here
        raise ValueError(f"{field_name} must be a list of domain strings, got {type(value).__name__}")
    names = []
    for item in value:
        host = normalize_host(item) if isinstance(item, str) else ""
        if not host:
            raise ValueError(f"{field_name} must contain non-empty domain strings, got {item!r}")
        if "*" in host:
            # A wildcard matches nothing here (a domain already covers its subdomains): refusing it
            # beats silently enforcing nothing.
            raise ValueError(f"{field_name}: wildcards are not supported ({item!r}); list the domain itself")
        names.append(host)
    return frozenset(names)


def host_matches(host: str, domains: frozenset[str]) -> bool:
    """Whether the NORMALIZED ``host`` is one of ``domains`` or a subdomain of one (label boundary)."""
    return host in domains or any(host.endswith("." + d) for d in domains)


def _host_for_check(hostname: str) -> str:
    """``normalize_host`` for a host taken from a URL; an invalid IDNA host is compared as given
    (lowercased), so it can still match nothing on an allow list and be refused."""
    try:
        return normalize_host(hostname or "")
    except ValueError:
        return (hostname or "").strip().lower()


@dataclass(frozen=True)
class InternetAccessPolicy:
    """The OPERATOR's network rules, independent of autonomy (``util/internet_policy.json``).

    Whether internet is ON is not part of it: the runtime toggle owns that, and
    ``EffectiveInternetPolicy`` composes the two at read time (#832). Frozen, so the one cached
    instance every tool reads cannot be changed under them; the domain lists are lowercased
    frozensets and the limits are validated on construction.

    Forward compat (CC3) is LOADER-OWNED, like ``EncodingSignals``: every field has a default, so an
    OLDER file loads on a newer build; ``from_dict`` refuses an unknown key (the loader then fails
    closed) and accepts ``_RETIRED_POLICY_KEYS`` with a warning. So a NEWER file carrying an added
    field turns internet off on an older build, deliberately: a build that cannot read a rule must not
    run without it. No ``extra`` dict: on a security policy it would keep a typo such as
    ``block_domain`` while enforcing nothing. #832 item 2 may later move these rules into a
    ``maxim config`` section, with this file as its migration source.
    """

    # Domain filtering
    allow_domains: frozenset[str] = frozenset()
    block_domains: frozenset[str] = frozenset()

    # Content policies (enforced by http_fetch)
    require_robots_ok: bool = True
    block_paywalled: bool = True
    unsafe_content_checks: bool = True

    # Rate limits and size limits
    max_fetch_bytes: int = 1_000_000  # 1 MB
    max_pages_per_minute: int = 10
    request_timeout_s: float = 8.0

    # SSRF protection: also switches the connect-time check (fetch_url(public_only=...), #824)
    block_private_ips: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "allow_domains", _domain_set(self.allow_domains, "allow_domains"))
        object.__setattr__(self, "block_domains", _domain_set(self.block_domains, "block_domains"))
        for name in ("require_robots_ok", "block_paywalled", "unsafe_content_checks", "block_private_ips"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be true or false, got {getattr(self, name)!r}")
        for name, minimum in (("max_fetch_bytes", 1), ("max_pages_per_minute", 1)):
            value = getattr(self, name)
            if isinstance(value, float) and value.is_integer():
                value = int(value)  # a hand-edited 10.0 means 10
                object.__setattr__(self, name, value)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
        timeout = self.request_timeout_s
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 0 < timeout < float("inf"):
            raise ValueError(f"request_timeout_s must be a positive number of seconds, got {timeout!r}")
        object.__setattr__(self, "request_timeout_s", float(timeout))

    def domain_refusal(self, hostname: str) -> str | None:
        """Why the allow/block lists refuse ``hostname``, or None. Resolves nothing (no DNS), so it
        can filter search results as well as gate a fetch."""
        host = _host_for_check(hostname)
        if self.block_domains and host_matches(host, self.block_domains):  # the block list wins
            return f"Domain '{hostname}' is blocked"
        if self.allow_domains and not host_matches(host, self.allow_domains):
            return f"Domain '{hostname}' not in allow list"
        return None

    def url_refusal(self, url: str) -> str | None:
        """Why this policy refuses ``url`` (scheme, the private-address pre-check, the domain lists)."""
        try:
            parsed = urlparse(url)
        except Exception as e:  # noqa: BLE001 -- urlparse raises several types on malformed input
            return f"Invalid URL: {e}"
        if parsed.scheme not in ("http", "https"):
            return f"Scheme '{parsed.scheme}' not allowed (only http/https)"
        hostname = parsed.hostname or ""
        if self.block_private_ips and self._is_private_ip(hostname):
            return "Access to private/internal IPs is blocked"
        return self.domain_refusal(hostname)

    def _is_private_ip(self, hostname: str) -> bool:
        """Check if hostname resolves to a private IP -- a policy PRE-check.

        This resolves the name separately from the fetch (and through a 5-minute cache), so on its
        own it cannot stop DNS rebinding: a host can answer public here and private to the connect.
        The enforcement is at connect time, on the address actually dialled
        (``maxim.utils.http.fetch_url(public_only=True)``, #824); this check refuses the obvious cases
        early with a clear reason.
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
                        logger.warning(f"DNS rebinding protection: {_redact_hostname(hostname)} resolved to private IP")
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
        """The operator's rules in one line (the on/off state is the effective view's)."""
        parts = []
        if self.require_robots_ok:
            parts.append("Must respect robots.txt.")
        if self.block_paywalled:
            parts.append("Paywalled content blocked.")
        if self.unsafe_content_checks:
            parts.append("Unsafe content checks active.")
        if self.allow_domains:
            parts.append(f"Allowed domains: {', '.join(sorted(self.allow_domains))}")
        if self.block_domains:
            parts.append(f"Blocked domains: {', '.join(sorted(self.block_domains))}")
        return " ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary (no on/off state: the access toggle owns it)."""
        return {
            "allow_domains": sorted(self.allow_domains),
            "block_domains": sorted(self.block_domains),
            "require_robots_ok": self.require_robots_ok,
            "block_paywalled": self.block_paywalled,
            "unsafe_content_checks": self.unsafe_content_checks,
            "max_fetch_bytes": self.max_fetch_bytes,
            "max_pages_per_minute": self.max_pages_per_minute,
            "request_timeout_s": self.request_timeout_s,
            "block_private_ips": self.block_private_ips,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InternetAccessPolicy:
        """Deserialize. Raises ``ValueError`` on an unknown key or an ill-typed value (the loader then
        fails closed); a retired key is accepted with a warning."""
        if not isinstance(data, dict):
            raise ValueError(f"internet policy must be a JSON object, got {type(data).__name__}")
        known = set(cls.__dataclass_fields__)
        unknown = sorted(k for k in data if k not in known and k not in _RETIRED_POLICY_KEYS and k != "_format_version")
        if unknown:
            raise ValueError(f"unknown internet policy key(s) {unknown}; known keys: {sorted(known)}")
        retired = sorted(k for k in data if k in _RETIRED_POLICY_KEYS)
        if retired:
            logger.warning(
                "Internet policy keys %s are retired and ignored (#832): %s",
                retired,
                "; ".join(_RETIRED_POLICY_KEYS[k] for k in retired),
            )
        return cls(**{k: v for k, v in data.items() if k in known})


# Keys an older file may carry that no longer mean anything (#832). Accepted with one warning each load,
# so a file written by an earlier `save_internet_policy` keeps working; any OTHER unknown key fails closed.
_RETIRED_POLICY_KEYS: dict[str, str] = {
    "enabled": "the runtime access toggle owns on/off",
    "allow_paywalled_with_credentials": "no credential mechanism ever read it",
    "retention_seconds": "no raw-content cache ever read it",
    "citations_required": "nothing enforced it; both tools always attach citations",
}


@dataclass(frozen=True)
class EffectiveInternetPolicy:
    """What a tool enforces for ONE request: the operator's policy composed with the live toggle.

    Built on every read by ``load_internet_policy`` and never persisted or sent anywhere, so it is
    outside CC3. #922/#834's in-session grant (its authority, its expiry) belongs here, not on the
    persisted ``InternetAccessPolicy``; since only the loader builds views, a grant arrives through
    ``load_internet_policy`` / ``live_internet_policy_getter``'s parameters.
    """

    policy: InternetAccessPolicy
    enabled: bool
    source: str = "default"

    def disabled_reason(self) -> str:
        """Why access is off, in words the model and the operator can act on."""
        if self.source == "policy-unreadable":
            return "Internet access is disabled: the internet policy file is unreadable or invalid (see the log)"
        if self.source == "unreadable":
            return (
                "Internet access is disabled: the internet access toggle file (util/internet_access.json) is "
                "unreadable or invalid (see the log). Ask the operator to check it."
            )
        return "Internet access is disabled"

    def can_access(self, url: str) -> tuple[bool, str | None]:
        """Whether ``url`` may be fetched now, and why not."""
        if not self.enabled:
            return False, self.disabled_reason()
        reason = self.policy.url_refusal(url)
        return reason is None, reason

    def domain_refusal(self, hostname: str) -> str | None:
        """The operator's allow/block lists for ``hostname``; resolves nothing."""
        return self.policy.domain_refusal(hostname)

    def summary(self) -> str:
        if not self.enabled:
            return "Internet access is DISABLED."
        rules = self.policy.summary()
        return f"Internet access is ENABLED. {rules}".strip()


# ─────────────────────────────────────────────────────────────────────────────
# Policy Loader
# ─────────────────────────────────────────────────────────────────────────────


def _default_policy_path() -> Path:
    from maxim.utils.paths import resolve_user_state

    return resolve_user_state("util/internet_policy.json")


# The operator policy, cached per file and file identity. None + unreadable means fail closed.
_cached_policy: InternetAccessPolicy | None = None
_cached_policy_unreadable: bool = False
_cached_policy_key: tuple[int, int, int] | None = None
_cached_policy_path: Path | None = None  # The cache is only valid for the file it was read from
_cached_policy_lock = threading.Lock()


def _reset_policy_cache() -> None:
    """Forget the cached operator policy (tests; the next read reloads)."""
    global _cached_policy, _cached_policy_unreadable, _cached_policy_key, _cached_policy_path
    with _cached_policy_lock:
        _cached_policy, _cached_policy_unreadable, _cached_policy_key, _cached_policy_path = None, False, None, None


def _file_key(path: Path) -> tuple[int, int, int] | None:
    """The file's identity for the cache: (mtime_ns, inode, size), (0, 0, 0) when it does not exist,
    or None when it cannot be stat'ed (never cached). ``atomic_write_json`` replaces the file, so every
    save changes the inode even within one mtime tick."""
    try:
        st = path.stat()
    except FileNotFoundError:
        return (0, 0, 0)
    except OSError:
        return None
    return (st.st_mtime_ns, st.st_ino, st.st_size)


def _operator_policy(path: Path) -> InternetAccessPolicy | None:
    """The operator policy in ``path`` (the default policy when there is no file), or None when the
    file exists but cannot be read or validated. Cached on the file's identity, taken BEFORE the read,
    so a replacement during the read is picked up on the next call rather than cached as the old
    content."""
    global _cached_policy, _cached_policy_unreadable, _cached_policy_key, _cached_policy_path
    key = _file_key(path)
    with _cached_policy_lock:
        if key is not None and _cached_policy_path == path and _cached_policy_key == key:
            return None if _cached_policy_unreadable else _cached_policy

    policy: InternetAccessPolicy | None
    if key == (0, 0, 0):
        policy = InternetAccessPolicy()
    else:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            from maxim.utils.format_version import check_format_version

            check_format_version(data, "internet_policy", log=logger)
            policy = InternetAccessPolicy.from_dict(data)
        except (OSError, ValueError, TypeError) as e:
            # FAIL CLOSED: the file exists, so it may carry a block list; silently falling back to
            # an unrestricted default would drop that control (#822 review).
            logger.error("Internet policy %s is unreadable (%s); internet access DISABLED until it is fixed.", path, e)
            policy = None

    with _cached_policy_lock:
        _cached_policy = policy
        _cached_policy_unreadable = policy is None
        _cached_policy_path = path
        _cached_policy_key = key
    return policy


def load_internet_policy(
    path: Path | str | None = None, *, access_path: Path | str | None = None
) -> EffectiveInternetPolicy:
    """The policy a tool enforces now: the operator's file composed with the persisted toggle.

    The operator policy is cached on the file's mtime; the toggle is read on every call, so a toggle
    takes effect on the next request. An unreadable policy file disables access.
    """
    path = Path(path) if path else _default_policy_path()
    access_state = load_internet_access(access_path)
    operator = _operator_policy(path)
    if operator is None:
        return EffectiveInternetPolicy(policy=InternetAccessPolicy(), enabled=False, source="policy-unreadable")
    return EffectiveInternetPolicy(policy=operator, enabled=access_state.enabled, source=access_state.source)


def save_internet_policy(policy: InternetAccessPolicy, path: Path | str | None = None) -> bool:
    """Save internet access policy to file."""
    path = Path(path) if path else _default_policy_path()

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        from maxim.utils.atomic_io import atomic_write_json
        from maxim.utils.format_version import with_format_version

        atomic_write_json(str(path), with_format_version(policy.to_dict()))
        return True
    except Exception as e:
        logger.error(f"Failed to save internet policy: {e}")
        return False


def effective_internet_enabled(
    launch_enabled: bool,
    *,
    policy_path: Path | str | None = None,
    access_path: Path | str | None = None,
) -> bool:
    """Whether internet is actually on at this moment: the launch cap AND the live policy (the toggle,
    and a readable policy file). What a runtime records as ``state.data["internet_access"]`` (#832)."""
    return bool(launch_enabled) and load_internet_policy(policy_path, access_path=access_path).enabled


def live_internet_policy_getter(
    launch_enabled: bool,
    *,
    policy_path: Path | str | None = None,
    access_path: Path | str | None = None,
) -> Callable[[], EffectiveInternetPolicy] | None:
    """The ONE internet-policy getter; ``runtime.bootstrap.build_tool_registry`` builds it (#822, #832).

    ``launch_enabled`` is the launch-time cap (``--no-internet``, an exploration policy's
    ``allow_internet``): when False there is no getter and no internet tools are registered.
    Otherwise every call returns the persisted operator policy -- its domain allow/block lists, byte and
    rate limits from ``util/internet_policy.json`` -- composed with the persisted access toggle, so the
    ``internet_access_toggle`` tool takes effect on the next request. Tools call the getter per request.
    """
    if not launch_enabled:
        return None
    initial = load_internet_policy(policy_path, access_path=access_path)
    if not initial.enabled or initial.policy.allow_domains or initial.policy.block_domains:
        # WARNING, not INFO: a persisted "off" (the agent's toggle tool is its only writer) or a
        # domain list silently changes what every later session can reach (#822 review).
        logger.warning(
            "Internet policy in effect: %s (toggle: %s; policy: %s). To turn internet back on, delete the "
            "toggle file; the policy file holds your domain lists and limits.",
            initial.summary(),
            Path(access_path) if access_path else _default_internet_access_path(),
            Path(policy_path) if policy_path else _default_policy_path(),
        )
    else:
        logger.info("Internet policy in effect: %s", initial.summary())

    def get() -> EffectiveInternetPolicy:
        return load_internet_policy(policy_path, access_path=access_path)

    return get


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
