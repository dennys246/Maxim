"""Read-Write Lock for concurrent hippocampus access.

Allows multiple concurrent readers OR one exclusive writer.
Writers have priority to prevent starvation.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Generator


class RWLock:
    """Read-Write Lock for concurrent access.

    Allows multiple concurrent readers OR one exclusive writer.
    Writers have priority to prevent starvation.

    Example:
        rwlock = RWLock()

        # Multiple threads can read simultaneously
        with rwlock.read():
            data = hippocampus.query(...)

        # Only one thread can write at a time
        with rwlock.write():
            hippocampus.capture(...)
    """

    def __init__(self) -> None:
        self._read_ready = threading.Condition(threading.Lock())
        self._readers = 0
        self._writers_waiting = 0
        self._writer_active = False

    def acquire_read(self) -> None:
        """Acquire a read lock (blocks if writer is active or waiting)."""
        with self._read_ready:
            # Wait while a writer is active or writers are waiting (priority)
            while self._writer_active or self._writers_waiting > 0:
                self._read_ready.wait()
            self._readers += 1

    def release_read(self) -> None:
        """Release a read lock."""
        with self._read_ready:
            self._readers -= 1
            if self._readers == 0:
                # Wake up any waiting writers
                self._read_ready.notify_all()

    def acquire_write(self) -> None:
        """Acquire a write lock (exclusive, blocks all readers)."""
        with self._read_ready:
            self._writers_waiting += 1
            # Wait for all readers to finish and no other writer active
            while self._readers > 0 or self._writer_active:
                self._read_ready.wait()
            self._writers_waiting -= 1
            self._writer_active = True

    def release_write(self) -> None:
        """Release a write lock."""
        with self._read_ready:
            self._writer_active = False
            # Wake up all waiting threads (readers and writers)
            self._read_ready.notify_all()

    @contextmanager
    def read(self) -> Generator[None, None, None]:
        """Context manager for read access.

        Example:
            with rwlock.read():
                data = do_read_operation()
        """
        self.acquire_read()
        try:
            yield
        finally:
            self.release_read()

    @contextmanager
    def write(self) -> Generator[None, None, None]:
        """Context manager for write access.

        Example:
            with rwlock.write():
                do_write_operation()
        """
        self.acquire_write()
        try:
            yield
        finally:
            self.release_write()

    def reader_count(self) -> int:
        """Return current number of active readers."""
        with self._read_ready:
            return self._readers

    def is_write_locked(self) -> bool:
        """Check if write lock is currently held."""
        with self._read_ready:
            return self._writer_active


__all__ = ["RWLock"]
