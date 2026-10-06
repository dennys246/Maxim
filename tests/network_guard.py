"""The test suite's network guard (roadmap 1.3.1 "Network blocked in tests").

Hermeticity was HOME/HF isolation plus ~48 env scrubs, with nothing that stopped a test reaching the
network: a measured run of the fast suite made 52 outbound attempts from 29 tests -- real DNS queries
for ``api.anthropic.com``, ``*.example.com``, ``fake.invalid``, and a TCP probe to a LAN address. Installed
once for the session by ``tests/conftest.py``:

- a TCP ``connect`` to a non-loopback address raises ``NetworkBlocked`` (an ``OSError``, so code under
  test sees what an unreachable host looks like);
- a DNS lookup of a non-loopback NAME raises ``socket.gaierror`` (what an unknown host looks like);
- loopback, IP-literal lookups and UDP ``connect`` (which sends no packet -- the "find my LAN address"
  idiom) stay allowed, so local test servers work.

``@pytest.mark.allow_network`` lifts THIS guard for one test. It does not lift CI's: the fast suite and the nightly
lanes run inside a loopback-only network namespace with no OS-level exception (``scripts/ci_netns.sh``, #940), so
in CI a marked test still cannot reach the network, and a test that truly needs it belongs in its own lane.

Scope: IN-PROCESS only (CI's namespace covers the process tree). A subprocess a test spawns (git, pip, curl) is not
guarded here, nor are the rarely
used resolver calls ``gethostbyname_ex`` / ``gethostbyaddr`` / ``getnameinfo`` (none is used in
``src/maxim``). ``attempts`` records every blocked call; the terminal summary prints the count.
"""

from __future__ import annotations

import errno
import ipaddress
import socket
from typing import Any

_real_connect = socket.socket.connect
_real_connect_ex = socket.socket.connect_ex
_real_getaddrinfo = socket.getaddrinfo
_real_gethostbyname = socket.gethostbyname

enabled = True
attempts: list[tuple[str, str]] = []


class NetworkBlocked(OSError):
    """A test tried to reach the network."""


def is_loopback(host: Any) -> bool:
    if host in (None, "", "localhost") or str(host).endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(str(host).split("%", 1)[0]).is_loopback
    except ValueError:
        return False


def _is_ip_literal(host: Any) -> bool:
    try:
        ipaddress.ip_address(str(host).split("%", 1)[0])
        return True
    except ValueError:
        return False


def _blocked_connect(sock: socket.socket, address: Any) -> str | None:
    if not enabled or sock.family not in (socket.AF_INET, socket.AF_INET6):
        return None
    if sock.type & socket.SOCK_STREAM != socket.SOCK_STREAM or is_loopback(address[0]):
        return None
    attempts.append(("connect", str(address[0])))
    return f"network blocked in tests: connect to {address[0]!r} (mark the test allow_network if it must)"


def _connect(self: socket.socket, address: Any) -> None:
    reason = _blocked_connect(self, address)
    if reason is not None:
        raise NetworkBlocked(errno.ENETUNREACH, reason)
    return _real_connect(self, address)


def _connect_ex(self: socket.socket, address: Any) -> int:
    if _blocked_connect(self, address) is not None:
        return errno.ENETUNREACH
    return _real_connect_ex(self, address)


def _blocked_name(host: Any) -> bool:
    if not enabled or host is None or is_loopback(host) or _is_ip_literal(host):
        return False
    attempts.append(("dns", str(host)))
    return True


def _getaddrinfo(host: Any, *args: Any, **kwargs: Any) -> Any:
    if _blocked_name(host):
        raise socket.gaierror(socket.EAI_NONAME, f"network blocked in tests: DNS lookup of {host!r}")
    return _real_getaddrinfo(host, *args, **kwargs)


def _gethostbyname(host: str) -> str:
    if _blocked_name(host):
        raise socket.gaierror(socket.EAI_NONAME, f"network blocked in tests: DNS lookup of {host!r}")
    return _real_gethostbyname(host)


def install() -> None:
    socket.socket.connect = _connect  # type: ignore[method-assign]
    socket.socket.connect_ex = _connect_ex  # type: ignore[method-assign]
    socket.getaddrinfo = _getaddrinfo
    socket.gethostbyname = _gethostbyname
