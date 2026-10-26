"""The offline latch (#1315 item 10, step 0b; ``docs/plans/atonia_gate_approach_notes.md`` §1A.1 and §6).

Offline is a one-way, process-wide latch. :func:`engage` raises it; nothing in production lowers it. The first
``engage()`` adds one ``sys.addaudithook`` hook, which, while the latch is up, refuses at the Python level:

* creating a socket of any family other than ``AF_INET``, ``AF_INET6`` and ``AF_UNIX`` (fd adoption reports
  ``family=-1`` and is refused too);
* every ``socket.connect`` / ``connect_ex`` (loopback and ``AF_UNIX`` included: there is no allowed endpoint, V8);
* every ``socket.sendto`` / ``socket.sendmsg`` to an ``AF_INET``/``AF_INET6`` destination;
* every forward and reverse name lookup (``getaddrinfo``, ``gethostbyname``, ``gethostbyaddr``, ``getnameinfo``);
* every spawn: ``os.system``, ``subprocess.Popen``, ``os.exec*``, ``os.posix_spawn*`` and ``os.spawn*`` (V8).

Inbound ``bind``/``listen``/``accept`` pass. Each refusal is appended to the refusal ledger (:func:`ledger`) BEFORE
:class:`OfflineRefusal` is raised, so a refusal a caller swallows is still on record.

**The boundary, stated.** The hook sees the Python ``socket``, ``subprocess`` and ``os`` modules only. Native
networking (uvloop/libuv, gRPC, libzmq, libcurl, Rust clients such as ddgs's primp) raises no audit event, and
neither do ``send``/``sendall`` on a connection opened before the latch went up, nor multiprocessing's
``_posixsubprocess.fork_exec``. Those are the call-site refusals', the self-engage's and the OS sandbox's (§6, AD5).

This module is a top-level LEAF and stdlib-only (guard:
``tests/integration/test_offline_latch.py::test_offline_module_is_stdlib_only``), so importing it never runs
``maxim.utils``'s package init; ``maxim/__init__`` imports it only when ``MAXIM_OFFLINE`` is set.
"""

from __future__ import annotations

import os
import socket
import sys
import threading
from typing import Any, Callable, NamedTuple, NoReturn, Sequence

OFFLINE_ENV = "MAXIM_OFFLINE"

# The canonical truthy set for MAXIM_* toggles (§1A.1): one parser for the parent, the child and maxim/__init__.
TRUTHY_VALUES: frozenset[str] = frozenset({"1", "true", "yes", "on"})

# engage() step 1: exactly these five os.environ writes (the observer's env-write allow-list), so libraries read
# them in-process and every child inherits them.
ENGAGE_ENV: tuple[tuple[str, str], ...] = (
    ("HF_HUB_OFFLINE", "1"),
    ("TRANSFORMERS_OFFLINE", "1"),
    ("HF_HUB_DISABLE_TELEMETRY", "1"),
    ("YOLO_OFFLINE", "true"),
    (OFFLINE_ENV, "1"),
)

_ALLOWED_FAMILIES = frozenset({socket.AF_INET, socket.AF_INET6, socket.AF_UNIX})
_INET_FAMILIES = frozenset({socket.AF_INET, socket.AF_INET6})
_LOOKUP_EVENTS = frozenset({"socket.getaddrinfo", "socket.gethostbyname", "socket.gethostbyaddr", "socket.getnameinfo"})
_SPAWN_EVENTS = frozenset({"os.system", "subprocess.Popen", "os.exec", "os.posix_spawn", "os.spawn", "os.startfile"})


class OfflineRefusal(RuntimeError):
    """An action the offline latch refused. A ``RuntimeError`` and never an ``OSError``, so the narrow
    ``except OSError`` sites that return plausible values (``code_tree.git_out``, ``_find_orphans_ps``,
    ``git_root_refusal``) cannot absorb it (§6 "Engaging the latch")."""


class Refusal(NamedTuple):
    """One refusal ledger entry: the audit event (or call site) refused, a short detail, and the engage reason."""

    event: str
    detail: str
    reason: str


_lock = threading.RLock()
_engaged = False
_reason = ""
_hook_added = False
_ledger: list[Refusal] = []
_callbacks: list[Callable[[], Any]] = []
_saved_env: dict[str, str | None] | None = None


def truthy(value: str | None) -> bool:
    """The canonical parser: "1"/"true"/"yes"/"on", case-insensitive and whitespace-tolerant; anything else
    (``None``, empty, "0", "y", ...) is False."""
    if not value:
        return False
    return value.strip().lower() in TRUTHY_VALUES


def is_offline() -> bool:
    """Whether the latch is up. Engages on ANY call that sees ``MAXIM_OFFLINE`` truthy, not only the first read,
    so a variable raised later still counts."""
    if not _engaged and truthy(os.environ.get(OFFLINE_ENV)):
        engage(f"{OFFLINE_ENV} is set")
    return _engaged


def engage(reason: str) -> None:
    """Raise the latch, one way. In order: set the latch, add the audit hook if this process has none yet, then
    (1) write the five ``ENGAGE_ENV`` variables, (2) set the offline flags of libraries ALREADY imported (found
    through ``sys.modules``, never imported here), (3) run the :func:`on_engage` callbacks. A second call is a
    no-op. A callback that raises does not stop the others; the first failure is re-raised after all have run,
    with the latch already up."""
    global _engaged, _reason, _hook_added, _saved_env
    with _lock:
        if _engaged:
            return
        _saved_env = {key: os.environ.get(key) for key, _ in ENGAGE_ENV}
        _engaged = True
        _reason = reason
        if not _hook_added:
            sys.addaudithook(_hook)
            _hook_added = True
        for key, value in ENGAGE_ENV:
            os.environ[key] = value
        _set_imported_library_flags()
        callbacks = list(_callbacks)
    _run_callbacks(callbacks)


def _set_imported_library_flags() -> None:
    hf_constants = sys.modules.get("huggingface_hub.constants")
    if hf_constants is not None:
        hf_constants.HF_HUB_OFFLINE = True  # type: ignore[attr-defined]
    ultra_utils = sys.modules.get("ultralytics.utils")
    if ultra_utils is not None:
        ultra_utils.ONLINE = False  # type: ignore[attr-defined]


def _run_callbacks(callbacks: Sequence[Callable[[], Any]]) -> None:
    first_error: BaseException | None = None
    for fn in callbacks:
        try:
            fn()
        except Exception as exc:  # re-raised below, after every callback has had its turn
            if first_error is None:
                first_error = exc
    if first_error is not None:
        raise first_error


def on_engage(fn: Callable[[], Any]) -> None:
    """Register ``fn`` to run when the latch goes up; run it at once if it is already up. Registering the same
    callable twice keeps one entry. Callbacks stay registered across a test reset and run again on the next
    ``engage()``."""
    with _lock:
        if fn not in _callbacks:
            _callbacks.append(fn)
        run_now = _engaged
    if run_now:
        _run_callbacks([fn])


def engage_callbacks() -> tuple[Callable[[], Any], ...]:
    """The registered :func:`on_engage` callbacks, in registration order."""
    with _lock:
        return tuple(_callbacks)


def ledger() -> tuple[Refusal, ...]:
    """Every refusal so far, oldest first (a fresh tuple per call)."""
    with _lock:
        return tuple(_ledger)


def hook_installed() -> bool:
    """Whether THIS process ever added the audit hook (it cannot be removed; it is inert while disengaged)."""
    return _hook_added


def engaged_reason() -> str:
    """The reason the latch was engaged with ("" while disengaged)."""
    return _reason if _engaged else ""


def refuse(site: str, detail: str = "") -> NoReturn:
    """Book a call-site refusal in the ledger, then raise :class:`OfflineRefusal`. For the call sites that refuse
    before reaching a socket (§6 layer 2), so their refusal is typed, early and on record."""
    _book(site, detail)
    raise OfflineRefusal(f"offline ({_reason or 'engaged'}): {site} refused{': ' + detail if detail else ''}")


def _book(event: str, detail: str) -> None:
    with _lock:
        _ledger.append(Refusal(event, detail, _reason))


def _family(sock: Any) -> Any:
    return getattr(sock, "family", None)


def _refused_detail(event: str, args: tuple[Any, ...]) -> str | None:
    """Why ``event`` is refused while engaged, or None when it passes."""
    if event == "socket.__new__":
        family = args[1] if len(args) > 1 else None
        return None if family in _ALLOWED_FAMILIES else f"socket family {family!r}"
    if event == "socket.connect":
        return f"connect to {args[1]!r}" if len(args) > 1 else "connect"
    if event in ("socket.sendto", "socket.sendmsg"):
        dest = args[1] if len(args) > 1 else None
        if dest is not None and _family(args[0]) in _INET_FAMILIES:
            return f"send to {dest!r}"
        return None
    if event in _LOOKUP_EVENTS:
        return f"name lookup {args[:2]!r}"
    if event in _SPAWN_EVENTS:
        return f"spawn {args[:2]!r}"[:200]
    return None


def _hook(event: str, args: tuple[Any, ...]) -> None:
    if not _engaged:
        return
    if not event.startswith(("socket.", "os.", "subprocess.")):
        return
    detail = _refused_detail(event, args)
    if detail is None:
        return
    _book(event, detail)
    raise OfflineRefusal(f"offline ({_reason}): {event} refused: {detail}")


def _reset_for_tests() -> None:
    # Tests only: a lint (tests/integration/test_offline_latch.py::test_no_caller_of_reset_for_tests_outside_tests)
    # forbids any caller outside tests/. Disengages, restores the five env keys to their pre-engage values and
    # clears the ledger; the on_engage callbacks stay registered and the (inert) hook stays installed.
    global _engaged, _reason, _saved_env
    with _lock:
        saved = _saved_env
        _engaged = False
        _reason = ""
        _saved_env = None
        _ledger.clear()
    if saved is not None:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
