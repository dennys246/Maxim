from __future__ import annotations

import logging
import operator
import os
import sys
import threading
from collections.abc import Mapping
from decimal import Decimal
from fractions import Fraction
from typing import Any, Callable, Optional

from maxim.utils.net import redact_urls

DEFAULT_DATEFMT = "%H:%M:%S"
DEFAULT_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"


# ─── The redacting record factory (#1801, approach note v28 §Shape, family 1 part a) ─────────────────────────────
# Installed (chained) by ``configure_logging`` only (D1), so an embedding host's records are untouched until it
# opts in. Every stdlib ``LogRecord`` the chain returns has its class swapped to ``_RedactingLogRecord``, whose
# ``getMessage`` reduces URLs lazily, when a handler formats the record: each argument (but an exact pass-through
# number type) is wrapped in ``_ReducedArg``, so a precision cut in the message (``%.50s``) cuts text that is already
# reduced, and the formatted result is reduced again. ``msg`` and ``args`` stay untouched (R3), so tests that read
# ``record.args`` keep their meaning. A record of a real ``LogRecord`` subclass is rendered eagerly instead. Gaps,
# stated: records made before ``configure_logging``; ``handleError``'s stderr dump of a failed render; a record's
# raw ``__dict__`` (``extra=`` keys, ``msg`` as an exception object; K1); ``makeLogRecord``'s dict update, whose
# ``exc_text`` is reduced on read only (H2).

_PASS_THROUGH_TYPES: frozenset[type] = frozenset({type(None), bool, int, float, complex, Decimal, Fraction})
_NUMPY_NUMERIC_KINDS = frozenset("biufc")
_numpy_numeric_cache: frozenset[type] | None = None
_numpy_failure_logged = False


def _numpy_numeric_types() -> frozenset[type]:
    """numpy's own NUMERIC scalar types, by exact type, built lazily WITHOUT importing numpy (X5): an import inside
    ``getMessage`` runs with the handler lock held and could deadlock against a thread part-way through importing
    numpy that logs. Absent or part-way imported (``__spec__._initializing``): empty and NOT cached, since no numpy
    scalar can exist yet (Y5, AB4). A build that raises for any other reason is cached as empty (AA5) and logged once
    in the Stage-1 form, the flags set BEFORE logging, because the swallow log's own record goes through here too.
    Empty only means numpy scalars are proxied, which is safe: the proxy forwards ``__float__``/``__int__``/
    ``__index__``."""
    global _numpy_numeric_cache, _numpy_failure_logged
    if _numpy_numeric_cache is not None:
        return _numpy_numeric_cache
    np = sys.modules.get("numpy")
    if np is None or getattr(getattr(np, "__spec__", None), "_initializing", False):
        return frozenset()
    try:
        built = frozenset(t for t in set(np.sctypeDict.values()) if np.dtype(t).kind in _NUMPY_NUMERIC_KINDS)
    except Exception:
        if getattr(getattr(np, "__spec__", None), "_initializing", False):  # transient: neither cache nor log (AB4)
            return frozenset()
        _numpy_numeric_cache = frozenset()
        if not _numpy_failure_logged:
            _numpy_failure_logged = True
            log_swallowed_exception(site="logging.py:_numpy_numeric_types:build")
        return frozenset()
    _numpy_numeric_cache = built
    return built


def _passes_through(value: Any) -> bool:
    """An argument rendered raw: an EXACT number type (never ``isinstance``, T1: a subclass can carry its own
    ``__str__``), a numpy numeric scalar by exact type, or a 1-character ``str`` (``%c``, Q4)."""
    t = type(value)
    if t in _PASS_THROUGH_TYPES:
        return True
    if t is str:
        return len(value) == 1
    return t in _numpy_numeric_types()


class _ReducedArg:
    """One logging argument whose text renders reduced (``%s``, ``%r``, ``%a``, ``str.format``), and whose number
    protocol forwards to the wrapped value, as CPython would call it: ``type(o).__float__(o)`` (never ``float(o)``,
    which would make ``%f`` accept a str), falling back to ``float(operator.index(o))``; ``__int__`` falling back to
    ``operator.index(o)``. A value with neither raises ``TypeError``, as the unproxied value would (U4)."""

    __slots__ = ("_o",)

    def __init__(self, o: Any) -> None:
        self._o: Any = o

    def __str__(self) -> str:
        return redact_urls(str(self._o))

    def __repr__(self) -> str:
        return redact_urls(repr(self._o))

    def __format__(self, spec: str) -> str:
        if spec and spec[-1] in "bcdoxXneEfFgG%" and hasattr(type(self._o), "__float__"):
            return format(self._o, spec)  # a number rendered as a number holds no URL
        return format(str(self), spec)

    def __float__(self) -> float:
        o = self._o
        f = getattr(type(o), "__float__", None)
        if f is not None:
            return f(o)
        return float(operator.index(o))

    def __int__(self) -> int:
        o = self._o
        i = getattr(type(o), "__int__", None)
        if i is not None:
            return i(o)
        return operator.index(o)

    def __index__(self) -> int:
        return operator.index(self._o)


def _reduced_args(args: Any) -> Any:
    if isinstance(args, Mapping):  # ``LogRecord.__init__`` unwrapped a single Mapping argument to a dict (``%(k).Ns``)
        return {k: (v if _passes_through(v) else _ReducedArg(v)) for k, v in args.items()}
    if isinstance(args, tuple):
        return tuple(a if _passes_through(a) else _ReducedArg(a) for a in args)
    return args if _passes_through(args) else _ReducedArg(args)


class _ReducingField:
    """A data descriptor for ``exc_text``/``stack_info``: ``__set__`` stores the REDUCED value in ``__dict__`` (which
    ``SocketHandler.makePickle``, ``pickle`` and ``copy`` read), and ``__get__`` reduces too, for a ``__dict__``
    written around it (``makeLogRecord``'s dict update, G3)."""

    def __init__(self, name: str) -> None:
        self._name = name

    def __get__(self, obj: Any, objtype: Any = None) -> Any:
        if obj is None:
            return self
        value = obj.__dict__.get(self._name)
        return redact_urls(value) if isinstance(value, str) else value

    def __set__(self, obj: Any, value: Any) -> None:
        obj.__dict__[self._name] = redact_urls(value) if isinstance(value, str) else value


class _RedactingLogRecord(logging.LogRecord):
    """A stock ``LogRecord`` whose rendering reduces URLs (#1801). Set by a ``__class__`` swap in the factory, never
    constructed: a module-level class, so the record pickles (S3), with no closure or bound method in ``__dict__``
    (F1)."""

    exc_text = _ReducingField("exc_text")
    stack_info = _ReducingField("stack_info")

    def getMessage(self) -> str:  # noqa: N802 — the stdlib's name
        msg = str(self.msg)
        if self.args:
            msg = msg % _reduced_args(self.args)
        return redact_urls(msg)


_chained_factory: Callable[..., logging.LogRecord] = logging.LogRecord
_factory_reentry = threading.local()


def _make_redacting(record: logging.LogRecord) -> None:
    if type(record) is logging.LogRecord:
        record.__class__ = _RedactingLogRecord
        # ``LogRecord.__init__`` wrote these into ``__dict__`` before the swap: write them back through the
        # descriptors so ``__dict__`` holds the reduced value (J1).
        record.exc_text = record.__dict__.get("exc_text")
        record.stack_info = record.__dict__.get("stack_info")
        return
    if isinstance(record, _RedactingLogRecord) or not isinstance(record, logging.LogRecord):
        return
    _render_subclass_record(record)


def _render_subclass_record(record: logging.LogRecord) -> None:
    """A real ``LogRecord`` subclass keeps its class (a swap would drop its own ``getMessage``, or raise on
    ``__slots__``, H3): it is rendered eagerly. A bad format must not raise out of the ``logger`` call, where inside an
    ``except`` it would mask the original exception, so a failed render leaves the record untouched and
    ``handleError`` reports it at emit (the stated terminal-only gap). ANY exception counts, not only a bad format:
    an argument whose ``__str__`` raises ``RuntimeError`` must not escape the ``logger`` call either (code review
    SF2). The traceback fields are reduced OUTSIDE the render's ``try``, so a bad format cannot leave them raw (K5)."""
    try:
        rendered = redact_urls(record.getMessage())
    except Exception:
        log_swallowed_exception(site="logging.py:_render_subclass_record:render")
    else:
        record.msg = rendered
        record.args = ()
    if record.exc_info:
        record.exc_text = redact_urls(logging.Formatter().formatException(record.exc_info))
    if record.stack_info:
        record.stack_info = redact_urls(record.stack_info)


def _redacting_record_factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
    """The installed factory: the chained factory's record, made to render reduced. A re-entry (a factory installed
    over this one that calls back into it) builds a stock record, so the chain can never recurse."""
    if getattr(_factory_reentry, "active", False):
        return logging.LogRecord(*args, **kwargs)
    _factory_reentry.active = True
    try:
        record = _chained_factory(*args, **kwargs)
    finally:
        _factory_reentry.active = False
    _make_redacting(record)
    return record


_CHAIN_PROBE_SECRET = "maxim-chain-probe-1801"


def _current_factory_reduces() -> bool:
    """The behavioural chain check (R4): a record built through the CURRENT factory renders a credential URL
    reduced. Identity alone is not trusted: a chaining wrapper over ours hides it (N2)."""
    try:
        record = logging.getLogRecordFactory()(
            "maxim.chain_check", logging.INFO, __file__, 0, "probe %s", (f"https://u:{_CHAIN_PROBE_SECRET}@h/p",), None
        )
        return _CHAIN_PROBE_SECRET not in record.getMessage()
    except Exception:
        log_swallowed_exception(site="logging.py:_current_factory_reduces:probe")
        return False


_factory_installed_once = False
# The check and the install are one step (code review SF1): two installs racing on different threads would both see
# a chain that does not reduce, and the second would chain to the first's factory, dropping the host factory, and log
# a false WARNING. An RLock, because the WARNING below is logged while it is held.
_factory_install_lock = threading.RLock()


def ensure_redacting_record_factory() -> bool:
    """Install the redacting record factory over the current one, chained, unless the current chain already reduces.
    Returns True when it (re-)installed. Called at the end of ``configure_logging`` and on each ``_LogBuffer``
    install; a factory set by someone else after the last call that does not chain is caught here, re-wrapped and
    reported with one WARNING. Never raises."""
    global _chained_factory, _factory_installed_once
    with _factory_install_lock:
        if _current_factory_reduces():
            return False
        current = logging.getLogRecordFactory()
        _chained_factory = logging.LogRecord if current is _redacting_record_factory else current
        logging.setLogRecordFactory(_redacting_record_factory)
        reinstalled = _factory_installed_once
        _factory_installed_once = True
        if reinstalled:
            logging.getLogger("maxim").warning(
                "log record factory %r does not chain to Maxim's URL-redacting factory; re-installed on top (#1801)",
                getattr(current, "__qualname__", current),
            )
        return True


def verbosity_to_level(verbosity: int) -> int:
    verbosity = int(verbosity or 0)
    if verbosity <= 0:
        return logging.WARNING
    if verbosity == 1:
        return logging.INFO
    return logging.DEBUG


def configure_logging(
    verbosity: int = 0,
    *,
    fmt: str = DEFAULT_FORMAT,
    datefmt: str = DEFAULT_DATEFMT,
    force: bool = False,
    log_file: str | None = None,
) -> None:
    """
    Configure root logging once, and always ensure the effective level tracks `verbosity`.

    Safe to call multiple times (e.g., from both CLI entrypoints and library code).

    If ``MAXIM_LOG_FILE`` is set in the environment, a JSONL file handler is
    attached using the shared ``StructuredFormatter`` from
    :mod:`maxim.utils.structured_logging`. This is the Plan 1 R1 dual-format
    contract: stdout stays human-readable, file gets machine-parseable JSONL.
    """
    level = verbosity_to_level(verbosity)

    root = logging.getLogger()
    # Plan 1 R1: when MAXIM_LOG_FILE is set we want DEBUG-level visibility
    # to the JSONL file while keeping stdout at the user-configured level.
    # The root logger filter runs BEFORE handler-level filters, so if we
    # set root=WARNING the JSONL handler never sees INFO/DEBUG records.
    # Solution: root is DEBUG when JSONL is active, and stdout handler
    # applies the verbosity-based filter itself.
    jsonl_active = bool(os.environ.get("MAXIM_LOG_FILE", "").strip())
    root.setLevel(logging.DEBUG if jsonl_active else level)

    # Quiet chatty third-party HTTP loggers. httpx announces EVERY request
    # at INFO ("HTTP Request: GET ..."); with the live DoA feed polling the
    # robot's REST endpoint at ~6.7 Hz that is ~7 console lines per second
    # and a JSONL flood (2026-08-01 live smoke). Maxim's own structured
    # ``http_request`` events in ``utils/http.py`` (DEBUG; INFO under
    # MAXIM_HTTP_TRACE=1) are the sanctioned per-call trace — the raw
    # library line adds nothing over them. WARNING+ (connect errors,
    # protocol violations) still passes.
    for _noisy in ("httpx", "httpcore"):
        logging.getLogger(_noisy).setLevel(logging.WARNING)

    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    def _ensure_file_handler(path: str) -> None:
        if not path:
            return

        abs_path = os.path.abspath(path)
        for handler in root.handlers:
            if (
                isinstance(handler, logging.FileHandler)
                and os.path.abspath(getattr(handler, "baseFilename", "")) == abs_path
            ):
                return

        os.makedirs(os.path.dirname(abs_path) or ".", exist_ok=True)
        file_handler = logging.FileHandler(abs_path, mode="a", encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    def _ensure_jsonl_file_handler(path: str) -> None:
        """Attach a JSONL handler using StructuredFormatter. MAXIM_LOG_FILE.

        Uses ``RotatingFileHandler`` so long-running peer leaders with
        ``MAXIM_HTTP_TRACE=1`` / ``MAXIM_BACKEND_TRACE=1`` don't fill disks.
        Defaults: 100 MB per file, 3 backups (so ~400 MB worst case at the
        same path).  Override via ``MAXIM_LOG_FILE_MAX_BYTES`` and
        ``MAXIM_LOG_FILE_BACKUP_COUNT`` env vars when a longer history is
        wanted (e.g. multi-day leader debugging).
        """
        if not path:
            return
        abs_path = os.path.abspath(path)
        # Dedupe: if a JSONL handler already points at this path, skip.
        for handler in root.handlers:
            if (
                isinstance(handler, logging.FileHandler)
                and os.path.abspath(getattr(handler, "baseFilename", "")) == abs_path
                and getattr(handler, "_maxim_jsonl", False)
            ):
                return
        # Lazy import — avoid circular at module-init time.
        from logging.handlers import RotatingFileHandler

        from maxim.utils.structured_logging import StructuredFormatter

        os.makedirs(os.path.dirname(abs_path) or ".", exist_ok=True)
        try:
            max_bytes = int(os.environ.get("MAXIM_LOG_FILE_MAX_BYTES", "100000000"))
        except ValueError:
            max_bytes = 100_000_000
        try:
            backup_count = int(os.environ.get("MAXIM_LOG_FILE_BACKUP_COUNT", "3"))
        except ValueError:
            backup_count = 3
        # ``MAXIM_LOG_FILE_MAX_BYTES=0`` is the explicit opt-out for rotation
        # — use the plain ``FileHandler`` so log shippers / ``tail -F``
        # consumers that depend on the open-fd-stays-open semantics keep
        # working.  Any positive value engages rotation; clamp to a sane
        # range to avoid accidental 1-byte rollovers.  Negative values are
        # treated as 0 (disabled) rather than silently turning into a tiny
        # rotation, since "negative" reads as "no, do not rotate".
        backup_count = max(0, min(backup_count, 50))
        jsonl_handler: logging.FileHandler
        if max_bytes <= 0:
            jsonl_handler = logging.FileHandler(abs_path, mode="a", encoding="utf-8")
        else:
            max_bytes = max(1_000_000, min(max_bytes, 10_000_000_000))
            jsonl_handler = RotatingFileHandler(
                abs_path,
                mode="a",
                encoding="utf-8",
                maxBytes=max_bytes,
                backupCount=backup_count,
            )
        # JSONL file captures DEBUG+ regardless of stdout verbosity — the
        # file is opt-in, noisy is fine, and low-level events are the
        # whole point.
        jsonl_handler.setLevel(logging.DEBUG)
        jsonl_handler.setFormatter(StructuredFormatter())
        jsonl_handler._maxim_jsonl = True  # type: ignore[attr-defined]
        root.addHandler(jsonl_handler)

    jsonl_path = os.environ.get("MAXIM_LOG_FILE", "").strip() or None

    if force or not root.handlers:
        # Always attach a console handler; add a file handler if requested.
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(level)
        stream_handler.setFormatter(formatter)
        handlers: list[logging.Handler] = [stream_handler]
        logging.basicConfig(level=level, handlers=handlers, force=force)
        # basicConfig just set root to `level`; override back to DEBUG if
        # JSONL is active so the JSONL handler sees everything.
        if jsonl_active:
            root.setLevel(logging.DEBUG)
        if log_file:
            _ensure_file_handler(log_file)
        if jsonl_path:
            _ensure_jsonl_file_handler(jsonl_path)
        ensure_redacting_record_factory()
        return

    # Keep existing handlers but align their levels with the requested
    # verbosity. The JSONL handler is EXEMPT: its contract is "captures
    # DEBUG+ regardless of stdout verbosity" (see _ensure_jsonl_file_handler),
    # and this align loop was silently clobbering it to the console level on
    # any later configure_logging(force=False) call (e.g. selfy.__init__ at
    # verbosity 1) — DEBUG-level structured events then never reached the
    # MAXIM_LOG_FILE they exist for (caught by the Exp 49 arm-C probe: zero
    # sim_doa.read events in a 156 s session with the feed demonstrably up).
    for handler in root.handlers:
        if getattr(handler, "_maxim_jsonl", False):
            continue
        try:
            handler.setLevel(level)
        except Exception:
            continue

    if log_file:
        _ensure_file_handler(log_file)
    if jsonl_path:
        _ensure_jsonl_file_handler(jsonl_path)
    ensure_redacting_record_factory()


def quiet_console_handlers(level: int = logging.WARNING) -> None:
    """Raise the terminal handlers on the root and ``maxim`` loggers to at least ``level``: for a shutdown that hides
    in-flight INFO lines from the terminal. A terminal handler is a ``StreamHandler`` that is not a ``FileHandler``
    (the test ``sim_logger.set_active_display`` uses); the JSONL run log, a plain log file and the leader's remote
    log buffer keep their levels.

    Raising a LOGGER's level instead would cut the JSONL file too, whose contract is DEBUG and up whatever the
    console shows: the sim teardown did that, and every run log lost what the run logged after it, Exp 63b's final
    store digest among it (#1476, #1588)."""
    for name in ("", "maxim"):
        for handler in logging.getLogger(name).handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                handler.setLevel(max(handler.level, level))


def get_logger(name: str = "maxim") -> logging.Logger:
    return logging.getLogger(name)


def log_exception(
    logger: logging.Logger,
    exc: BaseException,
    *,
    verbosity: int = 0,
    message: str = "Unhandled exception",
) -> None:
    if int(verbosity or 0) >= 2:
        logger.exception(message)
    else:
        logger.error("%s: %s", message, exc)


def warn(message: str, *args: object, logger: Optional[logging.Logger] = None) -> None:
    """
    Convenience warning logger that falls back to a simple print when logging
    isn't configured (useful for module-level utilities).
    """
    if logger is None:
        logger = logging.getLogger("maxim")
    if logging.getLogger().handlers:
        logger.warning(message, *args)
    else:
        try:
            formatted = message % args if args else message
        except Exception:
            formatted = message
        print(f"[WARN] {formatted}")


def warn_unless_refusal(message: str, exc: BaseException, *, logger: Optional[logging.Logger] = None) -> None:
    """``warn(message, exc)`` for a failure the caller survives, except a security refusal, which is re-raised.

    For a broad ``except Exception`` around starting the LLM (``build_primary_router``,
    the agentic runtime): degrading to "run without the LLM" is right for an outage and
    wrong for a :class:`maxim.exceptions.LeaderRefusal` (a leader with no key, an exposed
    llama-cpp-server), where carrying on leaves the proxy forwarding to what was refused, or for an
    ``OfflineRefusal`` (a lane the offline latch refused), where carrying on runs without the LLM the operator
    configured: both are :data:`maxim.exceptions.SECURITY_REFUSALS`. Nor for a model config that cannot be applied as
    written (``profiles.yml``, llm.json routing): :data:`maxim.exceptions.CONFIG_REFUSALS`, re-raised too (#1601).
    """
    from maxim.exceptions import CONFIG_REFUSALS, SECURITY_REFUSALS

    if isinstance(exc, (*SECURITY_REFUSALS, *CONFIG_REFUSALS)):
        raise exc
    warn(message, exc, logger=logger)


def user_warn(
    message: str,
    *,
    fix: str = "",
    source: str = "",
    event: str = "user_warning",
    data: Optional[dict[str, object]] = None,
) -> None:
    """Surface a user-actionable warning across every available channel.

    Routes the warning to whichever sinks are active so users see
    actionable information regardless of how they're running:

    * Active ``MaximDisplay`` → calls ``display.warn()`` so the line
      lands in the persistent warnings panel (above the scrolling log).
    * Python ``logging`` → emits via ``logger.warning`` with a structured
      ``extra={"event": ..., "data": ...}`` so ``MAXIM_LOG_FILE`` JSONL
      captures it with full attribution.
    * Plain ``stderr`` → fallback when neither display nor logging is
      configured (early-boot path, scripts, CI without ``--interactive``).

    The function NEVER raises — observability code that breaks
    application flow is worse than silence.

    Args:
        message: One-line headline the user will read.
        fix: Optional remediation hint (typically from
            ``BackendError.fix_hint``).  Rendered after the message.
        source: Originating subsystem (``"backend"``, ``"leader-boot"``,
            ``"foundry"``, ...).  Visible in JSONL ``data.source``.
        event: Structured event tag for log filtering.
        data: Additional structured context preserved in the JSONL.
    """
    full = message if not fix else f"{message}\n  Fix: {fix}"
    debug_log = logging.getLogger(__name__)

    # 1. Display path — only when a MaximDisplay is currently active.
    try:
        from maxim.simulation.sim_logger import get_active_display

        display = get_active_display()
        if display is not None and hasattr(display, "warn"):
            display.warn(full)
    except Exception as exc:
        debug_log.debug("user_warn display sink failed: %s", exc)

    # 2. Structured logging path — always when Python logging is set up.
    # When no handlers are configured, Python's ``lastResort`` handler
    # fires and writes a minimal warning to stderr; that's our fallback
    # for early-boot paths (no separate stderr branch needed — adding
    # one would double-print).
    try:
        log = logging.getLogger("maxim")
        payload: dict[str, object] = {
            "message": message,
            "fix": fix,
            "source": source,
        }
        if data:
            payload.update(data)
        log.warning(full, extra={"event": event, "data": payload})
    except Exception as exc:
        debug_log.debug("user_warn logging sink failed: %s", exc)


def info(message: str, *args: object, logger: Optional[logging.Logger] = None) -> None:
    """
    Convenience info logger that falls back to a simple print when logging
    isn't configured (useful for module-level utilities).
    """
    if logger is None:
        logger = logging.getLogger("maxim")
    if logging.getLogger().handlers:
        logger.info(message, *args)
    else:
        try:
            formatted = message % args if args else message
        except Exception:
            formatted = message
        print(f"[INFO] {formatted}")


# Fail-loud Stage 1 (docs/plans/deferred/measurement_path_fail_loud.md): per-site dedup
# so every distinct swallow site announces itself at WARNING exactly once per
# process, DEBUG after — loud enough for Stage 2's firing-site inventory,
# quiet enough for hot defensive paths.
_swallow_seen: set[str] = set()
_swallow_seen_lock = threading.Lock()


def _reset_swallow_seen_for_test() -> None:
    with _swallow_seen_lock:
        _swallow_seen.clear()


# Listeners on the Stage-1 swallow events (the console's "degraded" view,
# console/health.py). A seam rather than a log handler on purpose: a site's
# SECOND and later fires log at DEBUG, which a WARNING-level logger drops before
# any handler runs, so a handler would see each site once and report a count of
# 1 for a site that fired a thousand times. Listeners see every fire, regardless
# of logger level or MAXIM_LOG_FILE. Called synchronously from inside an
# `except` block on the failing thread: keep them O(1) and never raising. A
# raising listener is contained and WARNED once per listener through the plain
# logger (the `sim_logger._dispatch_to_sinks` precedent) — never reported back
# through log_swallowed_exception, which would re-enter this very path.
SwallowListener = Callable[[str, Optional[str], Optional[str]], None]
_swallow_listeners: list[SwallowListener] = []
_swallow_listeners_lock = threading.Lock()


def register_swallow_listener(fn: SwallowListener) -> None:
    """Register ``fn(site, exc_type, exc_text)`` for every Stage-1 swallow."""
    with _swallow_listeners_lock:
        if fn not in _swallow_listeners:
            _swallow_listeners.append(fn)


def unregister_swallow_listener(fn: SwallowListener) -> None:
    with _swallow_listeners_lock:
        if fn in _swallow_listeners:
            _swallow_listeners.remove(fn)


_swallow_listener_warned: set[SwallowListener] = set()  # the fn itself: an id() can be reused


def _notify_swallowed(site: str, exc_type: str | None, exc_text: str | None) -> None:
    with _swallow_listeners_lock:
        listeners = list(_swallow_listeners)
    for fn in listeners:
        try:
            fn(site, exc_type, exc_text)
        except Exception as e:  # a listener bug must not reach the failing site
            if fn not in _swallow_listener_warned:
                _swallow_listener_warned.add(fn)
                logging.getLogger("maxim").warning(
                    "swallow listener %r raised %s; further failures of this listener are not reported",
                    getattr(fn, "__qualname__", fn),
                    type(e).__name__,
                )


def log_swallowed_exception(
    exc: BaseException | None = None,
    *,
    operation: str | None = None,
    context: dict[str, object] | None = None,
    logger: Optional[logging.Logger] = None,
    level: int | None = None,
    site: str | None = None,
) -> None:
    """Log an exception that is intentionally swallowed. NEVER raises.

    Two forms:

    **Zero-arg form (fail-loud Stage 1)** — call from inside an ``except``
    block; the exception comes from ``sys.exc_info()`` and the site from the
    caller's frame (``file:function:line``). Emits a structured
    ``swallowed_exception`` event at WARNING the first time each site fires in
    this process, DEBUG after (per-site dedup)::

        except Exception:
            log_swallowed_exception()   # was: pass

    **Supplied-site form (Stage 1, #863)** — as the zero-arg form, but the site is given rather
    than read from the frame. For code that contains failures on behalf of OTHER functions (a
    decorator), where the frame is always its own::

        except Exception:
            log_swallowed_exception(site=f"sim_logger.py:{fn.__name__}:{lineno}")

    **Legacy explicit form** — unchanged semantics (DEBUG default, no dedup, and NO structured
    event: the JSONL line is content-free, so the Stage-2 gate cannot see it)::

        except SomeError as e:
            log_swallowed_exception(e, operation="risky_operation", context={"input": x})

    Args:
        exc: The exception being swallowed (default: current ``sys.exc_info``).
        operation: Brief description of what was attempted (default: caller site).
        context: Optional dict of relevant context values.
        site: Stage-1 site key (``file:function:line``) supplied by the caller; takes the
            Stage-1 branch (structured event, WARNING-first per site). See the supplied-site form.
            It WINS over ``operation`` if both are passed. Must be a STATIC code location: each
            distinct value is a permanent dedup key, so a per-call string would warn every time
            and grow the seen-set without bound.
        logger: Logger to use (defaults to "maxim").
        level: Log level. Default: DEBUG for the explicit form; for the
            zero-arg form, WARNING on a site's first fire then DEBUG.
    """
    try:
        if logger is None:
            logger = logging.getLogger("maxim")

        # ``site=`` is the Stage-1 form with the site SUPPLIED instead of read from the caller's
        # frame (#863). A decorator that contains failures for many functions calls this from ONE
        # frame, so the frame-derived site would collapse every function into one dedup key: the
        # first failure anywhere would WARN and every later one, from any function, would drop to
        # DEBUG. With ``site=`` each contained function keeps its own key, and still gets the
        # structured ``swallowed_exception`` event the Stage-2 gate counts. The explicit form
        # (``exc=``/``operation=``) remains a plain DEBUG line with NO event — it is not a report
        # anything downstream can see, and the lint's check 3 does not count it as one.
        stage1_form = site is not None or (exc is None and operation is None)
        if site is not None:
            operation = site
        if exc is None:
            exc = sys.exc_info()[1]
        if operation is None:
            frame = sys._getframe(1)
            operation = f"{os.path.basename(frame.f_code.co_filename)}:{frame.f_code.co_name}:{frame.f_lineno}"

        if level is None:
            if stage1_form:
                with _swallow_seen_lock:
                    first = operation not in _swallow_seen
                    if first:
                        _swallow_seen.add(operation)
                level = logging.WARNING if first else logging.DEBUG
            else:
                level = logging.DEBUG

        ctx_str = ""
        if context:
            ctx_parts = [f"{k}={v!r}" for k, v in context.items()]
            ctx_str = f" [{', '.join(ctx_parts)}]"

        if stage1_form:
            if _swallow_listeners:  # no str(exc) on the failing thread unless someone listens
                _notify_swallowed(
                    operation,
                    type(exc).__name__ if exc is not None else None,
                    str(exc) if exc is not None else None,
                )
            # The `extra` event/data pair is LOAD-BEARING (review fold, PR #487):
            # the MAXIM_LOG_FILE JSONL handler's StructuredFormatter serializes
            # ONLY record.event + record.data and discards the printf message
            # and exc_info — without this, Stage 2's "grep the JSONL for
            # swallowed_exception" finds nothing even when sites fire.
            logger.log(
                level,
                "swallowed_exception site=%s%s",
                operation,
                ctx_str,
                exc_info=exc,
                extra={
                    "event": "swallowed_exception",
                    "data": {
                        "site": operation,
                        "exc_type": type(exc).__name__ if exc is not None else None,
                        "exc": str(exc) if exc is not None else None,
                    },
                },
            )
        else:
            logger.log(level, "Swallowed %s in %s: %s%s", type(exc).__name__, operation, exc, ctx_str)
    except Exception:  # noqa: BLE001 — the logger must never take down the site it guards
        pass


def log_recoverable_error(
    exc: BaseException,
    *,
    operation: str,
    recovery: str,
    context: dict[str, object] | None = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Log an error that was recovered from with a fallback.

    Use when an exception triggers fallback behavior rather than failure.

    Args:
        exc: The exception that occurred.
        operation: What was attempted.
        recovery: What recovery action was taken.
        context: Optional context values.
        logger: Logger to use.

    Example:
        try:
            config = load_config(path)
        except FileNotFoundError as e:
            log_recoverable_error(e, operation="load_config", recovery="using defaults")
            config = default_config
    """
    if logger is None:
        logger = logging.getLogger("maxim")

    ctx_str = ""
    if context:
        ctx_parts = [f"{k}={v!r}" for k, v in context.items()]
        ctx_str = f" [{', '.join(ctx_parts)}]"

    logger.warning(
        "%s failed (%s: %s), %s%s",
        operation,
        type(exc).__name__,
        exc,
        recovery,
        ctx_str,
    )
