"""The offline latch (#1315 item 10, step 0b; ``docs/plans/atonia_gate_approach_notes.md`` §1A.1 and §6).

Offline is a one-way, process-wide latch. :func:`engage` raises it; nothing in production lowers it. The first
``engage()`` adds one ``sys.addaudithook`` hook, which, while the latch is up, refuses at the Python level:

* creating a socket of any family other than ``AF_INET``, ``AF_INET6`` and ``AF_UNIX``, and fd adoption
  WITHOUT a family (``socket.socket(fileno=fd)`` reports ``family=-1``);
* every ``socket.connect`` / ``connect_ex`` (loopback and ``AF_UNIX`` included: there is no allowed endpoint, V8);
* every ``socket.sendto`` / ``socket.sendmsg`` to an explicit destination, whatever the family (``AF_UNIX``
  datagrams included; a ``None`` destination is a connected socket, whose connect was already refused);
* every forward and reverse name lookup (``getaddrinfo``, ``gethostbyname``, ``gethostbyaddr``, ``getnameinfo``);
* every spawn: ``os.system``, ``subprocess.Popen``, ``os.exec*``, ``os.posix_spawn*`` and ``os.spawn*`` (V8);
* every ``os.fork`` / ``os.forkpty`` (owner decision 2026-10-26). On POSIX ``os.spawn*`` and ``pty.spawn`` are a
  Python fork+exec, so refusing the fork keeps their refusal typed and booked in the CALLER, rather than an exec
  refused in a duplicate child that unwinds the caller's stack. Consequence: multiprocessing's ``"fork"`` start
  method does not work offline, nor does ``"forkserver"`` (its ``AF_UNIX`` connect to the server is refused; the
  Linux default from Python 3.14). Only ``"spawn"`` works: its children start through the unaudited ``fork_exec``
  and self-engage.

Inbound ``bind``/``listen``/``accept`` pass. Each refusal is appended to the refusal ledger (:func:`ledger`) BEFORE
:class:`OfflineRefusal` is raised, so a refusal a caller swallows is still on record.

**The boundary, stated.** The hook sees the Python ``socket``, ``subprocess`` and ``os`` modules only. Native
networking (uvloop/libuv, gRPC, libzmq, libcurl, Rust clients such as ddgs's primp) raises no audit event, and
neither do ``send``/``sendall`` on a connection opened before the latch went up, nor multiprocessing's
``_posixsubprocess.fork_exec``. **Fd adoption WITH an explicit family** (``socket.fromfd(fd, AF_INET, ...)``,
``socket.socket(AF_INET, SOCK_STREAM, 0, fd)``) is the same residual as a pre-latch connection: the
``socket.__new__`` event carries no fd, so an adopted connection opened before the latch can keep sending. The hook
is POSIX-only in what it names: Windows' ``_winapi.CreateProcess`` is not listed. Those are the call-site
refusals', the self-engage's and the OS sandbox's (§6, AD5).

This module is a top-level LEAF (not ``maxim/utils/``, so importing it never runs ``maxim.utils``'s package init)
and stdlib-only (guard:
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
_LOOKUP_EVENTS = frozenset({"socket.getaddrinfo", "socket.gethostbyname", "socket.gethostbyaddr", "socket.getnameinfo"})
_SPAWN_EVENTS = frozenset(
    {"os.system", "subprocess.Popen", "os.exec", "os.posix_spawn", "os.spawn", "os.startfile", "os.fork", "os.forkpty"}
)


class OfflineRefusal(RuntimeError):
    """An action the offline latch refused. A ``RuntimeError`` and never an ``OSError``, so the narrow
    ``except OSError`` sites that return plausible values (``code_tree.git_out``, ``_find_orphans_ps``,
    ``git_root_refusal``) cannot absorb it (§6 "Engaging the latch")."""


class Refusal(NamedTuple):
    """One refusal ledger entry: the audit event (or call site) refused, a short detail, the engage reason, and the
    ident of the thread it was refused in (so a reader can tell its own refusals from another thread's)."""

    event: str
    detail: str
    reason: str
    thread: int = 0


class Degraded(NamedTuple):
    """One degraded record: a fallback a site took because the latch is up (§6 "The fallbacks become loud")."""

    site: str
    reason: str


_lock = threading.RLock()
_engaged = False
_reason = ""
_hook_added = False
_ledger: list[Refusal] = []
_degraded: list[Degraded] = []
_degraded_warned: set[str] = set()
_callbacks: list[Callable[[], Any]] = []
_saved_flags: list[tuple[Any, str, Any]] = []  # (module, attribute, value before engage) for the reset


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
    global _engaged, _reason, _hook_added
    with _lock:
        if _engaged:
            return
        # The five keys are written, never read first: the observer's env read set is pinned (#1315 item 1,
        # test_advisor_gate.py::test_env_read_set), so the pre-engage values a test needs back are the test
        # harness's to keep (tests/conftest.py::_reset_offline_latch snapshots and restores them).
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


# engage() step 2: (module, attribute, offline value). Best effort: a module that ran ``from ultralytics.utils
# import ONLINE`` keeps its own copy; item 10's call-site refusals own the real refusal.
_LIBRARY_FLAGS: tuple[tuple[str, str, bool], ...] = (
    ("huggingface_hub.constants", "HF_HUB_OFFLINE", True),
    ("ultralytics.utils", "ONLINE", False),
)


def _set_imported_library_flags() -> None:
    for module_name, attr, value in _LIBRARY_FLAGS:
        module = sys.modules.get(module_name)
        if module is not None:
            _saved_flags.append((module, attr, getattr(module, attr, None)))
            setattr(module, attr, value)


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


def mark_degraded(site: str, reason: str) -> None:
    """Record that ``site`` fell back because the latch is up (a missing encoder or semantic weight it may not
    fetch), and log ONE WARNING naming "degraded" per site per process, so a degraded run says so (§6)."""
    import logging

    with _lock:
        _degraded.append(Degraded(site, reason))
        first = site not in _degraded_warned
        _degraded_warned.add(site)
    if first:
        logging.getLogger("maxim.offline").warning("offline: %s degraded: %s", site, reason)


def local_files_only_kwargs(loader: Callable[..., Any] | None = None) -> dict[str, bool]:
    """``{"local_files_only": True}`` while the latch is up, else ``{}``: the kwarg the Hugging Face load sites pass
    (sentence-transformers, faster-whisper; §6 layer 2), so a missing model fails locally instead of reaching for the
    hub. Empty online, so an online load is unchanged. Given ``loader``, empty too when it takes no such parameter
    (sentence-transformers before 2.3); engage()'s ``HF_HUB_OFFLINE`` still keeps that load local."""
    if not is_offline():
        return {}
    if loader is not None and not _accepts_kwarg(loader, "local_files_only"):
        return {}
    return {"local_files_only": True}


def _accepts_kwarg(fn: Callable[..., Any], name: str) -> bool:
    import inspect

    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):  # no introspectable signature: pass it, as the library documents
        return True
    return name in params or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


# Hugging Face's own "not in the local cache, and I may not fetch it" errors (huggingface_hub), matched by name along
# the exception chain so this leaf never imports the library.
_HUB_LOCAL_MISS = frozenset({"LocalEntryNotFoundError", "OfflineModeIsEnabled"})


def is_hub_local_miss(exc: BaseException) -> bool:
    """Whether ``exc`` (or anything in its cause/context chain) is a Hugging Face local-cache miss: the one failure a
    load site may report as "not cached locally". A CUDA error, a bad compute type or a corrupt model is not."""
    seen: set[int] = set()
    node: BaseException | None = exc
    while node is not None and id(node) not in seen:
        seen.add(id(node))
        if any(cls.__name__ in _HUB_LOCAL_MISS for cls in type(node).__mro__):
            return True
        node = node.__cause__ or node.__context__
    return False


def load_failure_reason(model: str, exc: BaseException) -> str:
    """The degraded reason for a model load that failed offline: "not cached locally" only for a real cache miss."""
    if is_hub_local_miss(exc):
        return f"{model} is not cached locally"
    return f"{model} failed to load ({type(exc).__name__})"


def degraded() -> tuple[Degraded, ...]:
    """Every degraded record so far, oldest first (a fresh tuple per call)."""
    with _lock:
        return tuple(_degraded)


def hook_installed() -> bool:
    """Whether THIS process ever added the audit hook (it cannot be removed; it is inert while disengaged)."""
    return _hook_added


def refuse(site: str, detail: str = "") -> NoReturn:
    """Book a call-site refusal in the ledger, then raise :class:`OfflineRefusal`. For the call sites that refuse
    before reaching a socket (§6 layer 2), so their refusal is typed, early and on record."""
    _book(site, detail)
    raise OfflineRefusal(f"offline ({_reason or 'engaged'}): {site} refused{': ' + detail if detail else ''}")


def refuse_if_offline(site: str, detail: str = "") -> None:
    """:func:`refuse` while the latch is up, else nothing: the one-line guard each call site puts before its side
    effect (a spawn, a socket, a consent prompt, a record-as-sent), so its refusal is typed, early and booked."""
    if is_offline():
        refuse(site, detail)


def _book(event: str, detail: str) -> None:
    with _lock:
        _ledger.append(Refusal(event, detail, _reason, threading.get_ident()))


def _refused_detail(event: str, args: tuple[Any, ...]) -> str | None:
    """Why ``event`` is refused while engaged, or None when it passes."""
    if event == "socket.__new__":
        family = args[1] if len(args) > 1 else None
        return None if family in _ALLOWED_FAMILIES else f"socket family {family!r}"
    if event == "socket.connect":
        return f"connect to {args[1]!r}" if len(args) > 1 else "connect"
    if event in ("socket.sendto", "socket.sendmsg"):
        dest = args[1] if len(args) > 1 else None
        return None if dest is None else f"send to {dest!r}"
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
    # forbids any caller outside tests/. Disengages, restores the library flags engage() set to their pre-engage
    # values and clears the ledger; the on_engage callbacks stay registered and the (inert) hook stays installed.
    # The five env keys engage() wrote are REMOVED, not restored: engage() never reads them (the observer's pinned
    # env read set), so their pre-engage values are the caller's to keep (tests/conftest.py::_reset_offline_latch
    # puts them back around every test). Removing them stops a MAXIM_OFFLINE left by engage() from re-raising the
    # latch at the next is_offline().
    global _engaged, _reason
    with _lock:
        flags = list(reversed(_saved_flags))
        _saved_flags.clear()
        _engaged = False
        _reason = ""
        _ledger.clear()
        _degraded.clear()
        _degraded_warned.clear()
    for module, attr, value in flags:
        setattr(module, attr, value)
    for key, _value in ENGAGE_ENV:
        os.environ.pop(key, None)
