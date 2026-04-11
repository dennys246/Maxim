"""Last-known probe outcomes for remote LLM endpoints (P6).

The lane-bringup path probes every configured ``remote_url`` with
:func:`maxim.runtime.llm_server.probe_llm_server` so dead leaders don't
wedge the runtime into retry storms. Probing has cost (one round-trip
per lane per startup, ~800 ms worst-case), so we cache the outcome on
disk and re-probe only after the entry goes stale.

Cache shape::

    {
        "https://maxim.example.com/v1": {
            "outcome": "ok",
            "detail": "HTTP 200",
            "probed_at": 1712843212.4,
            "latency_ms": 312.5,
        },
        ...
    }

The TTL is short by design (60 s default, 600 s cap) so the cache is
*hint* state, not source of truth. Hot operations like ``maxim peer
restart`` and ``maxim peer llm <model>`` clear the cache via
:func:`clear_cache` and :func:`clear_cache_for_url` so the next startup
re-probes immediately.

Concurrent writers: the cache uses :func:`atomic_write_json` so a
crashing writer never leaves a partial JSON on disk. Two parallel
``maxim`` invocations can clobber each other's most-recent updates, but
the next probe re-fills the missing entry, so no probe data is
permanently lost. We accept that race instead of paying for a file
lock around every cache write.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_CACHE_FILENAME = "last_probe_status.json"


def _cache_path() -> Path:
    """Return the on-disk cache path under ``$MAXIM_DATA_HOME/util``.

    Resolved lazily so tests can point ``MAXIM_DATA_HOME`` at a tmp dir
    between calls without re-importing this module.
    """
    from maxim.utils.paths import data_home

    return data_home() / "util" / _CACHE_FILENAME


def load_cache() -> dict[str, dict[str, Any]]:
    """Read the cache file. Returns ``{}`` on any error or missing file."""
    path = _cache_path()
    try:
        if not path.is_file():
            return {}
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {}
        # Defensive: drop any entries that aren't dicts.
        return {k: v for k, v in data.items() if isinstance(v, dict)}
    except (OSError, json.JSONDecodeError) as e:
        logger.debug("probe_cache load failed (%s) — starting fresh", e)
        return {}


def save_cache(cache: dict[str, dict[str, Any]]) -> None:
    """Atomically write the cache. Best-effort — never raises."""
    from maxim.utils.atomic_io import atomic_write_json

    path = _cache_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(path, cache)
    except OSError as e:
        logger.debug("probe_cache save failed (%s) — continuing", e)


def is_fresh(entry: dict[str, Any], ttl_s: float) -> bool:
    """True iff ``entry`` was recorded within the last ``ttl_s`` seconds."""
    probed_at = entry.get("probed_at")
    if not isinstance(probed_at, (int, float)):
        return False
    return (time.time() - float(probed_at)) < ttl_s


def clear_cache() -> None:
    """Best-effort delete of the entire cache file."""
    path = _cache_path()
    try:
        path.unlink(missing_ok=True)
    except OSError as e:
        logger.debug("probe_cache clear failed (%s)", e)


def clear_cache_for_url(url: str) -> None:
    """Remove a single entry. Used when one URL is known transiently stale."""
    if not url:
        return
    cache = load_cache()
    if url in cache:
        cache.pop(url, None)
        save_cache(cache)


__all__ = [
    "clear_cache",
    "clear_cache_for_url",
    "is_fresh",
    "load_cache",
    "save_cache",
]
