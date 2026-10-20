"""Shared apparatus for the key-file tests (#1311, #1312)."""

from __future__ import annotations

import os
from pathlib import Path


def deny_os_access(monkeypatch, path: Path) -> None:
    """An unreadable parent directory denies ``open()`` and ``stat()`` on the path, which is what the key reader
    uses (one descriptor; never ``Path.is_file``)."""
    real_open, real_stat = os.open, os.stat

    def _open(p, *a, **k):
        if os.fspath(p) == os.fspath(path):
            raise PermissionError(13, "Permission denied", os.fspath(p))
        return real_open(p, *a, **k)

    def _stat(p, *a, **k):
        if not isinstance(p, int) and os.fspath(p) == os.fspath(path):
            raise PermissionError(13, "Permission denied", os.fspath(p))
        return real_stat(p, *a, **k)

    monkeypatch.setattr(os, "open", _open)
    monkeypatch.setattr(os, "stat", _stat)
