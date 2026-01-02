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
    """
    level = verbosity_to_level(verbosity)

    root = logging.getLogger()
    root.setLevel(level)

    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    def _ensure_file_handler(path: str) -> None:
        if not path:
            return

        abs_path = os.path.abspath(path)
        for handler in root.handlers:
            if isinstance(handler, logging.FileHandler) and os.path.abspath(
                getattr(handler, "baseFilename", "")
            ) == abs_path:
                return

        os.makedirs(os.path.dirname(abs_path) or ".", exist_ok=True)
        file_handler = logging.FileHandler(abs_path, mode="a", encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    if force or not root.handlers:
        # Always attach a console handler; add a file handler if requested.
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(level)
        stream_handler.setFormatter(formatter)
        handlers: list[logging.Handler] = [stream_handler]
        logging.basicConfig(level=level, handlers=handlers, force=force)
        if log_file:
            _ensure_file_handler(log_file)
        return

    # Keep existing handlers but align their levels with the requested verbosity.
    for handler in root.handlers:
        try:
            handler.setLevel(level)
        except Exception:
            continue

    if log_file:
        _ensure_file_handler(log_file)


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
