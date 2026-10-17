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

Under pytest-xdist (M22) every worker is its own process with its own ``attempts``, and the controller, which prints
the summary, runs no tests. So each worker hands its per-kind :func:`counts` to the controller in
``pytest_sessionfinish`` (``config.workeroutput``), the controller folds them in with :func:`merge_worker` in
``pytest_testnodedown`` (a worker that went down without reporting is counted and named in the line, never read
as zero), and :func:`summary_line` totals this process's own attempts plus every worker's: the same
line, with the same total, in a serial run and in an xdist run (``tests/conftest.py`` holds the hooks).
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
# Per-kind counts handed over by xdist workers (controller side only; empty in a serial run).
worker_counts: dict[str, int] = {}
# Workers that went down without handing counts over (crashed, killed): their blocked calls are unknown.
workers_silent = 0
WORKEROUTPUT_KEY = "network_guard_counts"


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


def counts() -> dict[str, int]:
    """This process's blocked calls, per kind (``connect`` / ``dns``)."""
    out: dict[str, int] = {}
    for kind, _ in attempts:
        out[kind] = out.get(kind, 0) + 1
    return out


def merge_worker(worker: dict[str, int] | None) -> None:
    """Fold one xdist worker's :func:`counts` into :data:`worker_counts`; None = the worker never reported."""
    global workers_silent
    if worker is None:
        workers_silent += 1
        return
    for kind, n in worker.items():
        worker_counts[kind] = worker_counts.get(kind, 0) + int(n)


def summary_line() -> str | None:
    """The terminal-summary line over this process and every merged worker, or None when nothing was blocked and
    every worker reported. A silent worker is named in the line: its count is unknown, not zero."""
    total = dict(worker_counts)
    for kind, n in counts().items():
        total[kind] = total.get(kind, 0) + n
    n = sum(total.values())
    if not n and not workers_silent:
        return None
    line = f"network guard: blocked {n} outbound call(s)"
    if n:
        line += f" ({', '.join(sorted(k for k, v in total.items() if v))})"
    if workers_silent:
        line += f" ({workers_silent} worker(s) did not report)"
    return line
