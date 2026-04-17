"""Drain-state routing constraint for the LLM router (Plan 4 Stage C4).

Bridges the gap between the mesh drain-state layer (``peer/drain_state.py``,
which operates on node names) and the LLM router (``models/language/router.py``,
which operates on provider keys). The constraint is injected into the router
as an optional ``drain_constraint`` callback — when absent, drain has zero
effect on routing.

**Read-only.** This module never writes to drain state. Writing is the CLI's
job (``peer drain``/``resume``) or the future admin API's job (C6).

The URL lookup table is built once at construction from ``mesh.yml`` topology.
Topology changes require a router restart to take effect — consistent with the
router's static provider list.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("maxim.drain_routing")

# ─── URL canonicalization ─────────────────────────────────────────────────
# Reuse the same normalization as probe_cache to keep URL matching consistent
# across the codebase. Lazy import avoids pulling probe_cache at module load.


def _canonical_url(url: str) -> str:
    """Normalize a URL for drain-state matching.

    Delegates to :func:`maxim.runtime.probe_cache._canonical_url` —
    single source of truth for URL normalization across drain routing,
    probe cache clearing, and health checks.
    """
    from maxim.runtime.probe_cache import _canonical_url as _probe_canonical

    return _probe_canonical(url)


# ─── mtime cache ──────────────────────────────────────────────────────────


_DEFAULT_CACHE_TTL_S = 1.0
_MIN_CACHE_TTL_S = 0.0
_MAX_CACHE_TTL_S = 60.0


def _cache_ttl() -> float:
    """Read ``MAXIM_DRAIN_CACHE_TTL_S`` env var, clamped to [0.0, 60.0]."""
    raw = os.environ.get("MAXIM_DRAIN_CACHE_TTL_S", "").strip()
    if not raw:
        return _DEFAULT_CACHE_TTL_S
    try:
        val = float(raw)
    except (ValueError, TypeError):
        return _DEFAULT_CACHE_TTL_S
    return max(_MIN_CACHE_TTL_S, min(_MAX_CACHE_TTL_S, val))


# ─── DrainConstraint ──────────────────────────────────────────────────────


class DrainConstraint:
    """Stateful callback checking whether a provider is drained.

    Built from mesh.yml topology + drain state file. Caches the drain set
    in memory, refreshing on file mtime change.

    Usage::

        constraint = DrainConstraint(url_to_node, provider_urls, drain_path)
        # Inject into router:
        router = LLMRouter(cfg, drain_constraint=constraint.is_drained)

    The ``is_drained`` method is the callback signature the router expects:
    ``Callable[[str], bool]`` where the argument is a provider key.
    """

    def __init__(
        self,
        url_to_node: dict[str, str],
        provider_urls: dict[str, str],
        drain_path: Path,
        *,
        cache_ttl_s: float | None = None,
    ) -> None:
        # canonical_url -> mesh node name
        self._url_to_node = url_to_node
        # provider_key -> canonical_url (only for providers with a URL)
        self._provider_urls = provider_urls
        # provider_key -> mesh node name (precomputed join)
        self._provider_to_node: dict[str, str] = {}
        for pkey, purl in provider_urls.items():
            node = url_to_node.get(purl)
            if node is not None:
                self._provider_to_node[pkey] = node

        self._drain_path = drain_path
        self._cache_ttl_s = cache_ttl_s if cache_ttl_s is not None else _cache_ttl()

        # mtime cache state
        self._cached_mtime: float = 0.0
        self._cached_drained: frozenset[str] = frozenset()
        self._last_check_time: float = 0.0

    def is_drained(self, provider_key: str) -> bool:
        """Return True if *provider_key* maps to a drained mesh node.

        This is the callback injected into the router. Provider keys
        not in the URL table are never drained (cloud providers, local
        backends without a mesh.yml entry).
        """
        node = self._provider_to_node.get(provider_key)
        if node is None:
            return False
        drained = self._read_drained()
        return node in drained

    def drained_providers(self) -> frozenset[str]:
        """Return set of currently drained provider keys (for logging)."""
        drained_nodes = self._read_drained()
        return frozenset(pkey for pkey, node in self._provider_to_node.items() if node in drained_nodes)

    def _read_drained(self) -> frozenset[str]:
        """Return the current drained node set, using the mtime cache."""
        now = time.monotonic()

        # TTL gate: skip stat() if we checked recently enough.
        if now - self._last_check_time < self._cache_ttl_s:
            return self._cached_drained

        self._last_check_time = now

        # stat() the drain file to check for changes.
        try:
            st = self._drain_path.stat()
            mtime = st.st_mtime
        except FileNotFoundError:
            self._cached_mtime = 0.0
            self._cached_drained = frozenset()
            return self._cached_drained
        except OSError:
            # Can't stat — return stale cache rather than crash routing.
            return self._cached_drained

        if mtime == self._cached_mtime:
            return self._cached_drained

        # mtime changed — re-read the file.
        self._cached_mtime = mtime
        self._cached_drained = self._load_drain_file()
        return self._cached_drained

    def _load_drain_file(self) -> frozenset[str]:
        """Parse the drain state file into a set of node names.

        Delegates to :func:`drain_state._load_names` — single source
        of truth for drain file parsing.
        """
        from maxim.peer.drain_state import _load_names

        return frozenset(_load_names(self._drain_path))


# ─── factory ──────────────────────────────────────────────────────────────


def build_drain_constraint(
    mesh_cfg: Any,
    provider_cfgs: dict[str, dict[str, Any]],
    drain_path: Path | None = None,
) -> DrainConstraint | None:
    """Build a :class:`DrainConstraint` from mesh config + provider configs.

    Returns ``None`` if no provider URL matches any mesh node URL —
    drain would have no effect, so the router skips the overhead entirely.

    Parameters
    ----------
    mesh_cfg
        A ``MeshConfig`` instance (from ``peer/mesh_config.py``). Must have
        a ``.nodes`` iterable of objects with ``.url`` and ``.name`` attrs.
    provider_cfgs
        The router's ``self._providers`` dict — ``{key: config_dict}``.
        Peer backends store their URL in ``config_dict["base_url"]``.
    drain_path
        Override for the drain state file path. Defaults to
        ``drain_state.drain_state_path()`` (role-scoped).
    """
    # Build url -> node_name table from mesh.yml nodes.
    url_to_node: dict[str, str] = {}
    for node in mesh_cfg.nodes:
        canonical = _canonical_url(node.url)
        if canonical:
            url_to_node[canonical] = node.name

    if not url_to_node:
        return None

    # Build provider_key -> canonical_url table from provider configs.
    provider_urls: dict[str, str] = {}
    for key, cfg in provider_cfgs.items():
        raw_url = cfg.get("base_url") or cfg.get("url") or ""
        if isinstance(raw_url, str) and raw_url.strip():
            canonical = _canonical_url(raw_url.strip())
            if canonical:
                provider_urls[key] = canonical

    if not provider_urls:
        return None

    # Check if ANY provider URL matches a mesh node.
    has_match = any(purl in url_to_node for purl in provider_urls.values())
    if not has_match:
        return None

    if drain_path is None:
        from maxim.peer.drain_state import drain_state_path

        drain_path = drain_state_path()

    return DrainConstraint(url_to_node, provider_urls, drain_path)
