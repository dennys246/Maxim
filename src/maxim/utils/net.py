"""Network utilities (Plan 2 R2d).

Shared network-hygiene helpers. The SSRF check here is imported by
``_OpenAIBackend`` (existing consumer) and Plan 3 ``_MaximPeerBackend``
(future consumer). Keeping it in ``utils`` avoids backend-to-backend
imports.
"""

from __future__ import annotations

import ipaddress
import logging
import re
import socket
import sys
from typing import Any, Literal
from urllib.parse import urlparse, urlsplit

logger = logging.getLogger(__name__)


AddressPolicy = Literal["any", "public", "private"]
"""Which addresses a connection may dial (#921): ``"public"`` = globally routable only; ``"private"`` =
non-public only (LAN, loopback); ``"any"`` = unchecked."""

_EMBEDS_IPV4 = tuple(ipaddress.ip_network(n) for n in ("64:ff9b::/96", "::ffff:0:0:0/96", "::/96"))


def is_public_address(address: str) -> bool:
    """Globally routable -- the classifier ``validate_base_url`` and the connect-time checks in
    ``utils/http.py`` share (#824, #921), so the two can never disagree (they did on CGNAT,
    100.64.0.0/10). ``InternetAccessPolicy``'s pre-check uses it too (#966); the connect check decides. Judged explicitly rather than by ``is_global`` alone, which calls multicast,
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


def primary_ipv4_address() -> str | None:
    """This host's primary non-loopback IPv4 address, or None.

    A UDP ``connect`` to a TEST-NET-1 address (RFC 5737) picks the outbound
    interface without sending a packet. Used to check that a server meant to
    listen on loopback is NOT reachable on the LAN address.
    """
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    except OSError:
        return None
    try:
        sock.connect(("192.0.2.1", 9))
        address = str(sock.getsockname()[0])
    except OSError:
        return None
    finally:
        sock.close()
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        return None
    return None if ip.is_loopback or ip.is_unspecified else address


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


# ─── URL text reduction (#997, #1404, #1801) ──────────────────────────────
# Stdlib-only on purpose (#1801 S5): ``utils/logging.py``'s record factory, ``structured_logging``, ``sim_logger``
# and ``tools/base.py`` import these, and ``utils/http.py`` (which re-exports them) imports httpx and
# ``structured_logging``, so importing them from ``http`` would be a cycle and would load httpx into the logging path.
# Scope: URL userinfo, path and query only. A URL without a scheme, a bearer token and a percent-encoded whole URL
# are out of scope (docs/plans/log_url_redaction_1801_approach.md, §Inputs).


def url_origin(url: str | None) -> str:
    """An endpoint as a committed record keeps it: ``scheme://host[:port]`` (#997). Userinfo, path, query and
    fragment (where a tunnel's credentials or token live) are dropped; the port stays, because two local servers
    differ only by it and it is no secret. Anything that is not an http(s) URL with a host, that does not parse
    (an unbalanced IPv6 bracket), or whose port is invalid or whose host holds a ``%`` (a percent-encoded
    ``user:pass@`` urlsplit leaves in the host) or a backslash (urlsplit does not end the host
    at one) gives ``""``. Never raises. ``scripts/o19_verdict.py::
    _endpoint_origin`` is a stdlib copy (that script imports no ``maxim``); a parity test pins the two."""
    try:
        parts = urlsplit(str(url or "").strip())
        port = parts.port
    except ValueError:
        return ""
    host = parts.hostname
    if parts.scheme.lower() not in ("http", "https") or not host or "%" in host or "\\" in host:
        return ""
    host = f"[{host}]" if ":" in host else host
    return f"{parts.scheme.lower()}://{host}" + (f":{port}" if port is not None else "")


# A URL in free text (#1801, approach note v28 §Shape ``redact_urls``). Three ways a URL starts:
# - the double-slash form ``scheme://`` (any scheme, today's): a whole run of scheme characters (the lookbehind keeps
#   the scan linear: a match starts only at the start of a run, and taking the whole run means a URL glued to a digit
#   or ``.+-`` is still matched), then optional userinfo up to its ``@`` (which may hold a quote, so a password's tail
#   is not left behind), then everything up to whitespace, a quote or an angle bracket; reduced by ``url_origin``;
# - the separator forms, which always reduce to ``<url>`` (``url_origin`` cannot parse them): ANY scheme of 2+
#   characters whose separator holds a JSON-escaped slash (``https:\/\/``, JSON-in-JSON); a LISTED scheme
#   (``_URL_PATH_SCHEMES``) with a backslash-holding separator (Windows, ``PureWindowsPath``); a listed scheme with a
#   lone ``/`` (``Path`` collapses ``://`` to ``:/``). Other schemes in the last two forms are E1801-3;
# - on either branch, the CATCH-ALL: a whitespace within 640 non-space characters after the separator, and an ``@``
#   within 640 characters after that whitespace, take the span through the last such ``@`` and the rest of its token
#   (userinfo with whitespace in it). Past the bounds the tail leaks (E1801-1); a URL followed in bounds by prose
#   holding an ``@`` is erased (E1801-2, fail-closed).
# Every body is TEMPERED by ``_URL_START`` (the regex's own start), so one URL's match never crosses the start of
# another (AF1); the residual is E1801-4. The separator is atomic by a lookahead plus a backreference, which keeps
# Python 3.10 (the floor) able to compile it. A word ENDING in a listed scheme (``views:/x``, ``Windows:\n``) counts
# as one: fail-closed over-match.
_URL_PATH_SCHEMES = (
    r"(?i:[A-Za-z0-9+.-]*(?:https?|wss?|ftps?|sftp|postgres(?:ql)?|mysql|rediss?|mongodb(?:\+srv)?|amqps?|ssh|s3))"
)
_URL_START = (
    r"(?<![A-Za-z0-9+.-])(?:[A-Za-z0-9+.-]+://|[A-Za-z0-9+.-]{2,}:[\\/]*\\/|"
    + _URL_PATH_SCHEMES
    + r":[\\/]*\\|"
    + _URL_PATH_SCHEMES
    + r":/(?![/\\]))"
)
_T = r"(?!" + _URL_START + r")"
_URL_IN_TEXT = re.compile(
    r"(?<![A-Za-z0-9+.-])"
    r"(?:(?:[A-Za-z0-9+.-]{2,}:(?=(?P<sep>(?=[\\/]*\\/)[\\/]+))(?P=sep)"  # any scheme: a JSON-escaped slash
    r"|" + _URL_PATH_SCHEMES + r":(?=(?P<wsep>(?=[\\/]*\\)[\\/]+))(?P=wsep)"  # listed schemes: a backslash separator
    r"|" + _URL_PATH_SCHEMES + r":(?P<psep>/)(?![/\\]))"  # the lone-``/`` Path form, listed schemes only
    r"(?:(?P<swsui>(?:" + _T + r"\S){0,640}\s(?:" + _T + r"[\s\S]){0,640}@)(?:" + _T + r"\S)*|(?:" + _T + r"\S)+)"
    r"|[A-Za-z0-9+.-]+://"  # any scheme, the double-slash form (today's)
    r"(?:(?P<wsui>(?:" + _T + r"\S){0,640}\s(?:" + _T + r"[\s\S]){0,640}@)(?:" + _T + r"\S)*"
    r"|(?:[^\s/@]*@)?(?:" + _T + r"[^\s'\"<>])+))"
)
_URL_TRAILING_PUNCT = ".,;:!?)"
_URL_NON_SCHEME_START = "0123456789+.-"  # characters a scheme may hold but not start with


def _reduce_url_match(match: re.Match[str]) -> str:
    raw = match.group(0)
    core = raw.rstrip(_URL_TRAILING_PUNCT)
    lead = core[: len(core) - len(core.lstrip(_URL_NON_SCHEME_START))]  # ``1https://`` keeps ``1``
    if match.group("sep") or match.group("psep") or match.group("wsep") or match.group("wsui") or match.group("swsui"):
        return lead + "<url>" + raw[len(core) :]
    return lead + (url_origin(core[len(lead) :]) or "<url>") + raw[len(core) :]


def redact_urls(text: str) -> str:
    """``text`` with every URL in it reduced by :func:`url_origin` (``"<url>"`` where nothing is left), and the
    sentence punctuation after a URL kept: for free text a record keeps, such as an exception string, which httpx
    formats with the request URL (#997). Idempotent; never raises. The one prefilter, ``":/" in s or ":\\" in s``
    (#1801 AF2), skips the scan on text no branch could match. Scope: URL userinfo, path and query only (no
    scheme-less URL, bearer token or percent-encoded whole URL); the accepted residuals are E1801-1..4 in
    docs/plans/log_url_redaction_1801_approach.md."""
    text = str(text)
    if ":/" not in text and ":\\" not in text:  # the one prefilter (AF2)
        return text
    return _URL_IN_TEXT.sub(_reduce_url_match, text)


def url_scrubbed(value: Any, _depth: int = 0) -> Any:
    """``value`` with every string in it passed through ``redact_urls`` (#1404 item 2; moved from
    ``simulation/report.py`` for #1801): dict values, list and tuple items and string keys, at any depth up to 16
    (deeper structure, a cycle included, is written as its scrubbed ``str``). Numbers, bools and None are kept, and
    text without a URL is unchanged. Any other leaf (a Path, an exception, a custom object) becomes its scrubbed
    ``str`` here, since ``json.dumps(default=str)`` would otherwise stringify it after the scrub. Shape changes: a
    tuple becomes a list, a set or bytes become a str, and a URL-keyed dict collapses keys that share an origin.
    A numpy NUMERIC scalar (bool, int, float or complex kind) is kept too, by exact type and only once numpy is in
    ``sys.modules``, as the record factory decides it: numpy is never imported here."""
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return redact_urls(value)
    if _depth >= 16:
        return redact_urls(str(value))
    if isinstance(value, dict):
        # String keys too: a map keyed by URL would otherwise keep the URL whole. Keys that share an origin
        # collapse into one, the later value winning: a URL-keyed map loses entries (rare in tool args).
        return {(redact_urls(k) if isinstance(k, str) else k): url_scrubbed(v, _depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [url_scrubbed(v, _depth + 1) for v in value]
    if "numpy" in sys.modules:
        from maxim.utils.logging import _numpy_numeric_types  # lazy: ``maxim.utils.logging`` imports this module

        if type(value) in _numpy_numeric_types():
            return value
    return redact_urls(str(value))


__all__ = ["validate_base_url", "url_origin", "redact_urls", "url_scrubbed"]
