"""Atomic file write helpers with fsync + crash-safe cleanup.

All persistence paths in Maxim use a write-to-.tmp-then-os.replace pattern.
This module centralizes that pattern with proper error handling:

- fsync before rename so the data is durable before the replace is visible
- cleanup of the .tmp file if replace fails (no orphaned tmp files)
- directory creation so callers don't need to remember

Example:
    from maxim.utils.atomic_io import atomic_write_json
    atomic_write_json(path, payload)
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def atomic_write_text(path: str, content: str, *, encoding: str = "utf-8") -> None:
    """Atomically write text to ``path``.

    Writes to ``{path}.tmp``, fsyncs, then os.replace() to the final path.
    If the replace fails, the tmp file is cleaned up.
    """
    parent = os.path.dirname(path) or "."
    os.makedirs(parent, exist_ok=True)
    tmp_path = f"{path}.tmp"
    try:
        with open(tmp_path, "w", encoding=encoding) as f:
            f.write(content)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                # fsync not supported on all filesystems (e.g. some network FSes);
                # the write still happened, just without a durability guarantee.
                pass
        os.replace(tmp_path, path)
    except Exception:
        # Don't leave orphan .tmp files behind on failure.
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError as cleanup_err:
            logger.warning("Failed to clean up %s: %s", tmp_path, cleanup_err)
        raise


def atomic_write_json(
    path: str,
    payload: Any,
    *,
    indent: int | None = 2,
    default: Any = str,
) -> None:
    """Atomically write ``payload`` to ``path`` as JSON."""
    atomic_write_text(
        path,
        json.dumps(payload, indent=indent, default=default),
    )
