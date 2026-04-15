"""Tests for utils.filelock — advisory cross-process file lock.

The lock is used to serialize concurrent Maxim invocations that touch
the same file (model downloads, probe cache writes, etc.). Tests cover:
- Basic acquire + release
- Non-blocking contention raises LockContended immediately
- Blocking mode with timeout
- File is created if missing
- Release on context exit even if the caller raises
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import pytest

from maxim.utils.process_lock import LockContended, file_lock


class TestFileLockBasic:
    def test_acquire_and_release(self, tmp_path):
        """Basic contract: `with file_lock` acquires and releases cleanly."""
        lock_path = tmp_path / "test.lock"
        with file_lock(lock_path):
            assert lock_path.exists()
        # Lock file stays on disk after release — that's intentional
        assert lock_path.exists()

    def test_creates_missing_parent_directory(self, tmp_path):
        """The lock file's parent directory is created if missing."""
        lock_path = tmp_path / "nested" / "deeper" / "test.lock"
        with file_lock(lock_path):
            assert lock_path.exists()

    def test_accepts_string_path(self, tmp_path):
        """file_lock accepts both Path and str arguments."""
        lock_path = str(tmp_path / "test.lock")
        with file_lock(lock_path):
            assert Path(lock_path).exists()

    def test_release_on_exception(self, tmp_path):
        """If the `with` body raises, the lock is still released so
        subsequent acquirers can proceed."""
        lock_path = tmp_path / "test.lock"

        class _CustomError(Exception):
            pass

        with pytest.raises(_CustomError):
            with file_lock(lock_path):
                raise _CustomError("caller exploded")

        # Should be able to re-acquire immediately
        with file_lock(lock_path):
            pass


class TestFileLockContention:
    def test_non_blocking_raises_immediately(self, tmp_path):
        """Default timeout_s=0 means "non-blocking" — if another thread
        holds the lock, raise LockContended right away rather than wait.
        """
        lock_path = tmp_path / "test.lock"
        acquired_outer = threading.Event()
        release_outer = threading.Event()
        error: list[Exception] = []

        def _outer_holder():
            try:
                with file_lock(lock_path):
                    acquired_outer.set()
                    # Hold until the test tells us to release
                    release_outer.wait(timeout=5.0)
            except Exception as e:
                error.append(e)

        holder_thread = threading.Thread(target=_outer_holder)
        holder_thread.start()
        assert acquired_outer.wait(timeout=2.0), "outer holder did not acquire"

        # Now the test thread tries to acquire — should fail fast
        try:
            with pytest.raises(LockContended):
                with file_lock(lock_path, timeout_s=0.0):
                    pytest.fail("acquire should have raised LockContended")
        finally:
            release_outer.set()
            holder_thread.join(timeout=2.0)

        # No errors in the holder
        assert not error

    def test_timeout_waits_then_raises(self, tmp_path):
        """timeout_s > 0 polls until the lock is free or the timeout fires."""
        lock_path = tmp_path / "test.lock"
        acquired_outer = threading.Event()
        release_outer = threading.Event()

        def _outer_holder():
            with file_lock(lock_path):
                acquired_outer.set()
                release_outer.wait(timeout=5.0)

        holder_thread = threading.Thread(target=_outer_holder)
        holder_thread.start()
        assert acquired_outer.wait(timeout=2.0)

        start = time.monotonic()
        try:
            with pytest.raises(LockContended):
                with file_lock(lock_path, timeout_s=0.3):
                    pytest.fail("acquire should have raised after timeout")
            elapsed = time.monotonic() - start
            # Should have waited at least 0.3s before raising
            assert elapsed >= 0.25
            # And not pathologically longer than the timeout
            assert elapsed < 1.0
        finally:
            release_outer.set()
            holder_thread.join(timeout=2.0)

    def test_timeout_succeeds_when_holder_releases(self, tmp_path):
        """If the holder releases before the timeout, the waiter gets
        the lock and the `with` body runs."""
        lock_path = tmp_path / "test.lock"
        acquired_outer = threading.Event()

        def _outer_holder():
            with file_lock(lock_path):
                acquired_outer.set()
                time.sleep(0.15)  # hold briefly then release

        holder_thread = threading.Thread(target=_outer_holder)
        holder_thread.start()
        assert acquired_outer.wait(timeout=2.0)

        # Try to acquire with enough timeout to outlast the holder
        body_ran = False
        with file_lock(lock_path, timeout_s=1.0):
            body_ran = True
        assert body_ran
        holder_thread.join(timeout=2.0)


class TestFileLockResourceHandling:
    def test_file_descriptor_closed_on_exit(self, tmp_path):
        """The internal file descriptor must be closed on exit so we
        don't leak fds across many acquires. Indirect test: acquire
        the same lock many times in a row and verify no resource
        warnings or system errors."""
        lock_path = tmp_path / "test.lock"
        for _ in range(100):
            with file_lock(lock_path):
                pass
        # If we reach here without "too many open files" or similar,
        # the fd lifecycle is clean.
        assert lock_path.exists()

    def test_does_not_delete_lock_file(self, tmp_path):
        """The lock file is advisory infrastructure — we deliberately
        do NOT delete it on release because another process may be
        waiting to acquire it and deleting would race."""
        lock_path = tmp_path / "test.lock"
        with file_lock(lock_path):
            pass
        assert lock_path.exists()
        os.unlink(lock_path)  # Caller can delete explicitly if they want
        assert not lock_path.exists()
