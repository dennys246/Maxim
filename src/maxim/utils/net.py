"""Network utilities (Plan 2 R2d).

Shared network-hygiene helpers. The SSRF check here is imported by
``_OpenAIBackend`` (existing consumer) and Plan 3 ``_MaximPeerBackend``
(future consumer). Keeping it in ``utils`` avoids backend-to-backend
imports.
"""

from __future__ import annotations

import ipaddress
import logging
import socket
from typing import Literal
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


AddressPolicy = Literal["any", "public", "private"]
"""Which addresses a connection may dial (#921): ``"public"`` = globally routable only; ``"private"`` =
non-public only (LAN, loopback); ``"any"`` = unchecked."""

_EMBEDS_IPV4 = tuple(ipaddress.ip_network(n) for n in ("64:ff9b::/96", "::ffff:0:0:0/96", "::/96"))


def is_public_address(address: str) -> bool:
    """Globally routable -- the classifier ``validate_base_url`` and the connect-time checks in
    ``utils/http.py`` share (#824, #921), so the two can never disagree (they did on CGNAT,
    100.64.0.0/10). ``InternetAccessPolicy``'s pre-check keeps its own; the connect check decides. Judged explicitly rather than by ``is_global`` alone, which calls multicast,
    site-local and NAT64-around-10.x addresses global (and varies across Python versions). An IPv6
    answer that embeds an IPv4 address is judged by that IPv4 address. An unparseable address is not
    public (fail closed)."""
    try:
        ip = ipaddress.ip_address(address.split("%", 1)[0])
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address):
        embedded = ip.ipv4_mapped or ip.sixtofour
        if embedded is None and any(ip in net for net in _EMBEDS_IPV4):
            embedded = ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF)
        if embedded is not None:
            return is_public_address(str(embedded))
        if ip.is_site_local:
            return False
    return ip.is_global and not (ip.is_multicast or ip.is_reserved or ip.is_unspecified)


def is_loopback_url(url: str) -> bool:
    """True only when ``url``'s host is ``localhost`` or a literal loopback IP -- fail-closed.

    No DNS resolution: a name that merely resolves to 127.0.0.1 does not count, and a parse error
    is ``False``. Used to decide whether a credential that belongs to THIS machine may be sent.
    """
    try:
        host = urlparse(url).hostname
    except ValueError:  # what urlparse raises on a malformed netloc (e.g. "http://[::1")
        return False
    if not host:
        return False
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def base_url_address_policy(base_url: str, allow_local: bool) -> AddressPolicy:
    """The address policy a validated backend URL is held to at CONNECT time (#921): the same rules
    ``validate_base_url`` states, enforced on the address actually dialled rather than on its own
    earlier lookup (a DNS answer that changes between the two defeated both rules).

    ``"public"`` when local endpoints are not allowed; ``"private"`` for a cleartext ``http://`` URL (it is
    only ever allowed to a private network); ``"any"`` for ``https://`` with local endpoints allowed.
    """
    if not allow_local:
        return "public"
    if urlparse(base_url).scheme == "http":
        return "private"
    return "any"


def validate_base_url(base_url: str, allow_local: bool) -> str | None:
    """Return ``base_url`` if safe to dispatch to, else ``None``.

    Rules:

    - ``https://`` is required for public endpoints.
    - ``http://`` is acceptable only for private-IP LAN servers when
      ``allow_local=True`` (self-hosted llama-cpp-server, Ollama).
    - Host must resolve. If ``allow_local=False``, every resolved IP must
      be public. If ``allow_local=True``, private IPs are acceptable.
    - ``http://`` + public IP is always rejected, even when
      ``allow_local=True`` (explicit cleartext over the internet).

    Plan 3's ``_MaximPeerBackend`` will also use this function. The
    ``_OpenAIBackend`` continues to import it from here.

    Returns ``None`` on any failure (malformed URL, DNS failure, SSRF
    rejection) — callers log ``ssrf_rejected`` and skip the endpoint.
    """
    try:
        parsed = urlparse(base_url)
    except Exception:
        return None
    if parsed.scheme == "https":
        pass
    elif parsed.scheme == "http" and allow_local:
        pass
    else:
        return None
    host = parsed.hostname
    if not host:
        return None
    default_port = 443 if parsed.scheme == "https" else 80
    try:
        addrinfos = socket.getaddrinfo(host, parsed.port or default_port, proto=socket.IPPROTO_TCP)
    except Exception:
        return None
    for info in addrinfos:
        ip = info[4][0]
        if not is_public_address(ip) and not allow_local:
            return None
        # http scheme is only allowed for private IPs, even when allow_local
        if parsed.scheme == "http" and is_public_address(ip):
            return None
    return base_url


__all__ = ["validate_base_url"]
