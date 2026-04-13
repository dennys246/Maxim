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
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


def _is_private_ip(ip: str) -> bool:
    """Return True for private, loopback, link-local, reserved, or multicast IPs.

    Returns True on parse error — fail-closed: if we cannot prove the IP is
    public, treat it as private so the caller rejects it under strict mode.
    """
    try:
        addr = ipaddress.ip_address(ip)
    except Exception:
        return True
    return addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_reserved or addr.is_multicast


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
        if _is_private_ip(ip) and not allow_local:
            return None
        # http scheme is only allowed for private IPs, even when allow_local
        if parsed.scheme == "http" and not _is_private_ip(ip):
            return None
    return base_url


__all__ = ["validate_base_url"]
