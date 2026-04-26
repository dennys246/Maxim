"""LLM server state — model persistence, server tracking, health checks.

Extracted from lane_backends.py for single-responsibility decomposition.
Manages the global state for the auto-spawned LLM server process and
provides model persistence + health check utilities.
"""

from __future__ import annotations

import contextlib
import logging
import os
import threading
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)


# ─── Active server tracking (for hot-swap) ────────────────────────────────
_active_spawner: Any | None = None
_active_model: str | None = None
_llm_start_time: float | None = None
_swap_lock = threading.Lock()


_LAZY_MIGRATION_DONE = False


def _model_state_file() -> Path:
    """Return the path to the persisted active model file.

    Plan 2 R2a split the file per role: ``active_llm_model.{role}.txt``.
    Reads ``MAXIM_ROLE`` from env (exported by ``runtime/role.py::detect_and_apply_role``
    at CLI startup). Falls back to ``leader`` when the env var is missing,
    matching the conservative migration default.

    Fix #7 (R2 review): runs the legacy-file migration lazily on first
    access per process, so Python-API users (``import maxim``) who never
    go through ``cli.py::main`` also get their ``active_llm_model.txt``
    migrated instead of orphaned. The migration is idempotent and
    gated by a module-level flag so the cost is one ``is_file()`` check
    after the first call.
    """
    from maxim.utils.paths import resolve_user_state

    global _LAZY_MIGRATION_DONE  # noqa: PLW0603
    role = os.environ.get("MAXIM_ROLE", "").strip().lower() or "leader"
    if role not in ("leader", "peer", "solo"):
        role = "leader"
    if not _LAZY_MIGRATION_DONE:
        _LAZY_MIGRATION_DONE = True
        try:
            from maxim.runtime.role import migrate_persisted_model_file

            migrate_persisted_model_file(role)  # type: ignore[arg-type]
        except Exception as e:
            logger.warning("lazy persisted-model migration failed: %s", e)
    return resolve_user_state(f"util/active_llm_model.{role}.txt")


def stop_active_spawner() -> None:
    """Stop the global auto-spawned server, releasing VRAM.

    Called during Maxim shutdown so the server doesn't linger until
    the atexit handler fires (which may never run if shutdown hangs).
    """
    global _active_spawner, _active_model, _llm_start_time  # noqa: PLW0603
    with _swap_lock:
        spawner = _active_spawner
        _active_spawner = None
        _active_model = None
        _llm_start_time = None
    if spawner is not None:
        try:
            spawner.stop()
        except Exception:
            pass


# Weak references to active LLMRouter instances so swap_llm_server can
# update cached n_ctx without importing the router at module level.
_active_routers: list[weakref.ref] = []
_active_routers_lock = threading.Lock()


def register_router(router: Any) -> None:
    """Register an LLMRouter for n_ctx updates on hot-swap."""
    with _active_routers_lock:
        _active_routers.append(weakref.ref(router))


def _find_active_routers() -> list[Any]:
    """Return all still-alive routers, pruning dead refs."""
    global _active_routers  # noqa: PLW0603
    with _active_routers_lock:
        alive = []
        for ref in _active_routers:
            r = ref()
            if r is not None:
                alive.append(r)
        _active_routers = [weakref.ref(r) for r in alive]
    return alive


def read_persisted_model() -> str | None:
    """Read the last swapped model name from disk (survives restarts)."""
    try:
        text = _model_state_file().read_text().strip()
        return text if text else None
    except Exception:
        return None


def write_persisted_model(profile: str | None) -> None:
    """Persist the active model name so auto-spawn uses it after restart."""
    try:
        sf = _model_state_file()
        sf.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write: tmp file + os.replace to avoid partial reads
        import tempfile

        content = (profile or "").encode("utf-8")
        fd, tmp = tempfile.mkstemp(dir=str(sf.parent), suffix=".tmp")
        closed = False
        try:
            os.write(fd, content)
            os.fsync(fd)
            os.close(fd)
            closed = True
            os.replace(tmp, str(sf))
        except BaseException:
            if not closed:
                os.close(fd)
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise
    except Exception as e:
        logger.warning("Failed to persist model state: %s", e)


# ─── Structured probe (peer_leader_flexibility_plan P6) ──────────────────────


ProbeOutcome = Literal[
    "ok",
    "auth_rejected",
    "dns_fail",
    "connection_refused",
    "tls_error",
    "timeout",
    "http_5xx",
    "other",
    "inference_broken",
]


@dataclass(frozen=True)
class ProbeResult:
    """Outcome of probing a remote OpenAI-compatible LLM endpoint.

    The structured outcome lets callers distinguish between actually
    unreachable servers and auth-gated servers (where the leader is alive
    but rejected our token). Each outcome maps to a different fix-hint
    message.

    ``latency_ms`` is populated only on outcomes where we actually got an
    HTTP response back (``ok``, ``auth_rejected``, ``http_5xx``, ``other``).
    For network-layer failures it's left None.
    """

    url: str
    outcome: ProbeOutcome
    detail: str
    latency_ms: float | None = None

    @property
    def is_reachable(self) -> bool:
        """True iff the leader is alive and listening (auth-gated counts)."""
        return self.outcome in ("ok", "auth_rejected")


def profile_has_local_file(profile_name: str) -> bool:
    """Return True iff the profile's GGUF file actually exists on disk.

    Used to filter the capability-driven tier table down to models the user
    has actually downloaded. Returns False on any error — fail-closed.

    Explicitly rejects ``.partial`` paths: ``download_file`` writes to a
    ``{dest}.partial`` tmp file and only ``os.replace()`` to the final
    path after size verification. A resolved ``model_path`` ending in
    ``.partial`` means either (a) the profile is mis-configured to point
    at a tmp file, or (b) we somehow resolved against a stale partial
    from a crashed download. Either way, returning True would let the
    spawner try to load a truncated GGUF, which fails in cryptic ways.
    """
    if not profile_name:
        return False
    try:
        from maxim.models.language.config import load_llm_config

        cfg = load_llm_config(profile_override=profile_name)
        model_path = getattr(cfg, "model_path", "") or ""
        if not model_path:
            return False
        if model_path.endswith(".partial"):
            return False
        return Path(model_path).is_file()
    except Exception:
        return False
