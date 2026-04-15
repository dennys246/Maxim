"""Minimal cross-platform advisory file lock.

**Renamed from ``maxim.utils.filelock`` in Plan 4 C2** (2026-04-14) to
avoid a name collision with the third-party ``filelock`` package that
Plan 4 C2 added as a core dep (``filelock>=3.0,<4.0``, imported as
``from filelock import FileLock``). Two locking abstractions with the
same basename was a footgun — this module is now ``process_lock`` to
make the distinction clear. The unification of both locking APIs into
one in-house wrapper is tracked in
``docs/plans/cross_platform_file_lock.md``.

Used by the model-download path (and any future code that needs to
serialize file access across concurrent Maxim invocations) to avoid
two processes racing on the same target — e.g., two ``maxim --sim``
runs both trying to download qwen2.5-14b into ``~/.maxim/models/``.

The lock is **advisory** (not mandatory): it only works when all
cooperating processes use this API. That's the intended scope — we're
not defending against adversarial file writes, just coordinating our
own subprocesses.

Platform support:
- POSIX: ``fcntl.flock`` with ``LOCK_EX | LOCK_NB`` for non-blocking
  acquire; ``LOCK_UN`` on release.
- Windows: ``msvcrt.locking`` with ``LK_NBLCK`` for non-blocking
  acquire. Less battle-tested than fcntl but adequate for the "one
  process at a time" contract.

The lock file is created if missing. It is **not** deleted on release
— leaving it on disk is the correct behavior, because another process
may be waiting to acquire it. Lock file cleanup is the caller's
responsibility (typically: never, since they're tiny and live in
``~/.maxim/util/``).

NFS and network filesystems are **not** officially supported. ``flock``
on NFS is flaky across kernel versions; treat ``MAXIM_DATA_HOME`` on a
network drive as unsupported. This matches the substrate_plan's
checkpoint assumption.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)


class LockContended(RuntimeError):
    """Raised when a non-blocking lock acquire fails because another
    process holds the lock. Callers should treat this as "another
    process is doing the work — wait or skip."
    """


@contextmanager
def file_lock(path: Path | str, *, timeout_s: float = 0.0) -> Iterator[None]:
    """Acquire an exclusive advisory lock on ``path``.

    Args:
        path: Lock file path. Created if missing. Parent directory is
            created if missing.
        timeout_s: If > 0, poll for the lock up to this many seconds
            before raising ``LockContended``. If 0 (default), acquire
            is non-blocking — raises ``LockContended`` immediately on
            contention.

    Yields:
        ``None`` while the lock is held.

    Raises:
        LockContended: If another process holds the lock and the
            timeout expires (or is zero).
        OSError: On filesystem errors (permission denied, disk full,
            etc.). These are propagated so the caller can log them
            and fall back to no-locking behavior if appropriate.
    """
    lock_path = Path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    # Open the lock file — on POSIX, keeping the fd alive is what
    # holds the lock. The file contents are ignored; it exists purely
    # as an inode to lock against.
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)

    try:
        deadline = time.monotonic() + max(0.0, timeout_s)
        poll_interval_s = 0.05
        first_attempt = True
        while True:
            try:
                _acquire(fd)
                break
            except (BlockingIOError, OSError) as e:
                # BlockingIOError is what fcntl.flock raises on POSIX
                # when the lock is held. Windows msvcrt raises OSError
                # with errno=EDEADLK-ish. Treat both as "contended".
                if first_attempt and timeout_s <= 0.0:
                    raise LockContended(f"lock held: {lock_path}") from e
                first_attempt = False
                if time.monotonic() >= deadline:
                    raise LockContended(f"lock held after {timeout_s:.1f}s: {lock_path}") from e
                time.sleep(poll_interval_s)

        # Lock acquired — yield to caller's code.
        try:
            yield
        finally:
            try:
                _release(fd)
            except OSError as e:
                # Release failures are logged but not raised — the fd
                # close below will also drop the lock as a safety net
                # on most platforms.
                logger.debug("file_lock release failed on %s: %s", lock_path, e)
    finally:
        try:
            os.close(fd)
        except OSError:
            pass


def _acquire(fd: int) -> None:
    """Platform-specific exclusive lock acquire (non-blocking)."""
    if sys.platform == "win32":
        import msvcrt

        # LK_NBLCK = non-blocking exclusive lock on the first byte
        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)


def _release(fd: int) -> None:
    """Platform-specific lock release."""
    if sys.platform == "win32":
        import msvcrt

        try:
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        except OSError:
            # Windows raises if the lock region doesn't exist, which
            # can happen if the file was truncated externally.
            pass
    else:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_UN)


__all__ = ["LockContended", "file_lock"]
