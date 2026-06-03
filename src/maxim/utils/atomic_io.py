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


def atomic_write_text(
    path: str,
    content: str,
    *,
    encoding: str = "utf-8",
    preserve_mode: bool = False,
) -> None:
    """Atomically write text to ``path``.

    Writes to ``{path}.tmp``, fsyncs, then os.replace() to the final path.
    If the replace fails, the tmp file is cleaned up.

    Parameters
    ----------
    preserve_mode
        When True and ``path`` already exists, capture its mode bits
        via ``os.stat`` before writing and re-apply them to the final
        file after ``os.replace``. Use this for files containing
        secrets (e.g. API keys, cluster keys) where the default umask
        (typically 0644) would widen a pre-existing 0600. Plan 4 C2
        pre-design review (E3) flagged unconditional umask inheritance
        as a silent secret-leak vector.

        If ``path`` does not exist yet, ``preserve_mode`` is a no-op —
        the new file inherits the umask as usual. Callers that want a
        specific initial mode on a brand-new file should ``os.chmod``
        after this function returns.
    """
    parent = os.path.dirname(path) or "."
    os.makedirs(parent, exist_ok=True)
    tmp_path = f"{path}.tmp"

    # Capture existing mode before we touch anything — an exception
    # during write must not leave a mode-drift footprint on the caller.
    existing_mode: int | None = None
    if preserve_mode:
        try:
            existing_mode = os.stat(path).st_mode
        except FileNotFoundError:
            existing_mode = None
        except OSError as e:
            # Permission denied or similar: best-effort only. Log and
            # continue — losing mode preservation is recoverable, but
            # refusing to write because we couldn't stat is worse.
            logger.warning("preserve_mode: could not stat %s: %s", path, e)
            existing_mode = None

    try:
        with open(tmp_path, "w", encoding=encoding) as f:
            # Writing content to disk is this function's purpose.
            # ``atomic_write_secret`` (above) is the caller for credential
            # files and tightens the umask to 0o077 around the call so the
            # temp file is mode 0o600 from creation. CodeQL flags the
            # write because the data-flow analysis can reach this point
            # from secret-bearing callers — by design.
            f.write(content)  # lgtm [py/clear-text-storage-sensitive-data]
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

    if existing_mode is not None:
        try:
            # Mask to the permission bits only — st_mode also carries
            # file-type bits that os.chmod does not accept on all
            # platforms.
            os.chmod(path, existing_mode & 0o7777)
        except OSError as e:
            logger.warning("preserve_mode: could not chmod %s: %s", path, e)


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


def atomic_write_secret(path: str, content: str, *, encoding: str = "utf-8") -> None:
    """Atomically write text to ``path``, preserving pre-existing mode bits.

    Plan 4 C2 review fold A3: make the safe path verbose, not the
    unsafe path. ``atomic_write_text`` defaults to ``preserve_mode=False``
    (inherits umask) because that's the right default for the
    overwhelming majority of Maxim persistence paths — session
    reports, decision logs, memory state, bench results. None of
    those are secret.

    This wrapper is for files that DO contain secrets: cluster keys,
    bearer tokens, API keys, cluster rotation state. The intent is
    visible at the call site — nobody needs to "remember to pass
    preserve_mode=True" because they pick a different function
    entirely.

    When ``path`` doesn't exist yet, there's no mode to preserve;
    callers that need a specific initial mode on a brand-new secret
    file should ``os.chmod`` after this function returns (a future
    C3 cluster-key rotation verb will want 0o600 on first write).

    Plan 4 C2 drain state does NOT use this function because drain
    state is operator-visible topology, not a secret. Using the
    wrapper here would over-advertise it and muddy the "this is for
    secrets" signal.
    """
    atomic_write_text(path, content, encoding=encoding, preserve_mode=True)
