"""Last-known probe outcomes for remote LLM endpoints (P6).

The lane-bringup path probes every configured ``remote_url`` via
:meth:`maxim.models.language.maxim_peer_backend._MaximPeerBackend.health_check`
so dead leaders don't
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


# ─── Plan 2 R2c — per-outcome cache TTL ───────────────────────────────────
# Short TTL for `inference_broken` (model might be mid-load, retry sooner).
# Sourced from models.language.types.INFERENCE_BROKEN_BACKOFF_S — Plan 3's
# router backoff imports the same constant so the two values cannot drift.
def _default_ttl_table() -> dict[str, float]:
    from maxim.models.language.types import INFERENCE_BROKEN_BACKOFF_S

    return {
        "ok": 60.0,
        "auth_rejected": 60.0,
        "http_5xx": 60.0,
        "timeout": 60.0,
        "connection_refused": 60.0,
        "dns_fail": 60.0,
        "tls_error": 60.0,
        "other": 60.0,
        "inference_broken": INFERENCE_BROKEN_BACKOFF_S,
    }


def ttl_for_outcome(outcome: str) -> float:
    """Return the cache freshness window for a given probe outcome."""
    return _default_ttl_table().get(outcome, 60.0)


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
        # Plan 2 R2c: promoted from debug → warning. A corrupt probe cache
        # is operationally significant — operators shouldn't need -v to see it.
        logger.warning("probe_cache load failed (%s) — starting fresh at %s", e, path)
        try:
            from maxim.utils.structured_logging import log_structured

            log_structured(
                logger,
                logging.WARNING,
                event="probe_cache_corrupt",
                data={"path": str(path), "error": f"{type(e).__name__}: {e}"},
            )
        except Exception:
            pass
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


def _canonical_url(url: str) -> str:
    """Canonical form for tolerant probe-cache lookups.

    Plan 4 C3.3 fold (CC1): the cache is keyed by whatever
    ``lane_configs[name].remote_url`` held at write time, with no
    normalization. When a caller hands us a URL resolved from a
    different source (``mesh.yml::nodes`` vs ``peer.yml::url``), the
    shape may not match the cache key byte-for-byte — trailing slash,
    trailing ``/v1``, etc. — and the exact-match lookup silently
    becomes a no-op, leaving a stale entry that can wire a lane off
    on the next startup.

    Strip trailing slashes and a single trailing ``/v1`` so two URLs
    that differ only in those axes compare equal.
    """
    if not url:
        return ""
    base = url.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    return base


def clear_cache_for_url(url: str) -> None:
    """Remove entries matching ``url`` under canonical-form comparison.

    Plan 4 C3.3 fold (CC1): uses :func:`_canonical_url` on both sides
    of the comparison so the clear is tolerant of trailing-slash or
    ``/v1``-suffix drift between the URL we were handed and the URL
    the cache was keyed with. Any entry whose canonical form matches
    the target's canonical form is removed, not just the literal key.
    """
    if not url:
        return
    target = _canonical_url(url)
    cache = load_cache()
    to_remove = [k for k in cache if _canonical_url(k) == target]
    if to_remove:
        for k in to_remove:
            cache.pop(k, None)
        save_cache(cache)


__all__ = [
    "clear_cache",
    "clear_cache_for_url",
    "is_fresh",
    "load_cache",
    "save_cache",
    "ttl_for_outcome",
]
