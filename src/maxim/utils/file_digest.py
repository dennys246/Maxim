"""A file's SHA-256, computed in the background, trusted only if the file did not change while it was read.

Used for model files (baseline floors B0): a multi-GB GGUF takes seconds to tens of seconds to hash, so the
digest starts when the file is first known (an owned server's spawn) and is collected later with a bounded
wait. :meth:`BackgroundDigest.result` returns ``None`` when the hash is unfinished, failed, or the file's
(device, inode, size, mtime) differs from when hashing started.
"""

from __future__ import annotations

import hashlib
import os
import threading


def file_stat(path: str) -> tuple[int, int, int, int] | None:
    try:
        st = os.stat(path)
    except OSError:
        return None
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)


class BackgroundDigest:
    """Start hashing ``path`` (resolved) now; collect with :meth:`result`."""

    def __init__(self, path: str) -> None:
        self.path = os.path.realpath(path)
        self.stat = file_stat(self.path)
        self.error: str | None = None
        self._sha256: str | None = None
        self._thread = threading.Thread(target=self._run, name="file-digest", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            h = hashlib.sha256()
            with open(self.path, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            self._sha256 = h.hexdigest()
        except OSError as e:
            self.error = f"{type(e).__name__}: {e}"

    def result(self, timeout: float) -> str | None:
        """The digest, or ``None`` if unfinished after ``timeout`` seconds, failed, or the file changed."""
        self._thread.join(timeout)
        if self._thread.is_alive() or self._sha256 is None or self.stat is None:
            return None
        return self._sha256 if file_stat(self.path) == self.stat else None
