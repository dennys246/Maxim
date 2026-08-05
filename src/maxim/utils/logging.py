from __future__ import annotations

import logging
import os
from typing import Optional

DEFAULT_DATEFMT = "%H:%M:%S"
DEFAULT_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"


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


def log_swallowed_exception(
    exc: BaseException,
    *,
    operation: str,
    context: dict[str, object] | None = None,
    logger: Optional[logging.Logger] = None,
    level: int = logging.DEBUG,
) -> None:
    """Log an exception that is intentionally swallowed.

    Use this instead of bare `except: pass` to maintain visibility
    into silently handled errors.

    Args:
        exc: The exception being swallowed.
        operation: Brief description of what was attempted.
        context: Optional dict of relevant context values.
        logger: Logger to use (defaults to "maxim").
        level: Log level (defaults to DEBUG for non-critical swallows).

    Example:
        try:
            result = risky_operation()
        except SomeError as e:
            log_swallowed_exception(e, operation="risky_operation", context={"input": x})
            result = fallback_value
    """
    if logger is None:
        logger = logging.getLogger("maxim")

    ctx_str = ""
    if context:
        ctx_parts = [f"{k}={v!r}" for k, v in context.items()]
        ctx_str = f" [{', '.join(ctx_parts)}]"

    logger.log(level, "Swallowed %s in %s: %s%s", type(exc).__name__, operation, exc, ctx_str)


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
