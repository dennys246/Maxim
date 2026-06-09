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


# ─── Singleton spawn guard (harness-on-leader cascade fix) ───────────────────
#
# Before binding port 8100 for a fresh llama-cpp-server, probe the port. If a
# server is already alive there serving the right model, REUSE it instead of
# spawning a colliding process. This is the structural fix for the
# harness-on-leader cascade documented in CLAUDE.md ("Don't run the benchmark
# harness on the same machine as the leader"): a sub-sim that auto-detects
# leader role on the leader machine must NOT kill + respawn the leader's live
# server (which collides on the port, kills the upstream, 502s the proxy,
# and takes the tunnel down).


# Decision values returned by :func:`check_existing_llm_server`.
REUSE_EXISTING = "reuse"
SPAWN_NEW = "spawn"


def _read_served_model_id(url: str, api_key: str | None) -> str | None:
    """Return the model id reported by ``GET <url>/v1/models``, or None.

    Reuses :meth:`_MaximPeerBackend._try_discover_served_model` (the canonical
    ``/v1/models`` reader) rather than re-implementing the parse — the served
    name is the first ``data[].id`` entry. Best-effort: any failure (network,
    parse, auth) returns None so the caller treats the served model as
    unknown rather than fabricating a mismatch.
    """
    try:
        from maxim.models.language.maxim_peer_backend import _MaximPeerBackend

        backend = _MaximPeerBackend.for_url(url, api_key=api_key)
        backend._try_discover_served_model()
        served = backend.served_model
        return served or None
    except Exception:  # noqa: BLE001 — discovery is best-effort
        return None


def _served_model_matches(
    served: str | None,
    expected_model_path: str | None,
    expected_profile: str | None,
) -> bool | None:
    """Compare a served model id against the configured model.

    Returns:
      - ``True``  — served model matches the configured profile / GGUF path.
      - ``False`` — served model is positively a DIFFERENT model.
      - ``None``  — served model is unknown / unparseable (cannot decide).

    llama-cpp-server reports ``data[0].id`` as the value passed to its
    ``--model`` flag (the GGUF path) when no ``--model_alias`` is set, so a
    basename comparison against the resolved GGUF path is robust to the
    harness's per-trial ``models/`` symlink (same file, different absolute
    path). The profile name is also accepted in case a future spawn sets an
    explicit alias.
    """
    if not served:
        return None
    s = str(served).strip()
    if not s:
        return None
    s_base = os.path.basename(s)
    if expected_model_path:
        ep = str(expected_model_path)
        if s == ep or (s_base and s_base == os.path.basename(ep)):
            return True
    if expected_profile:
        epf = str(expected_profile).strip()
        if epf and (s == epf or s_base == epf):
            return True
    # We have something to compare against and it didn't match → wrong model.
    if expected_model_path or expected_profile:
        return False
    return None


def check_existing_llm_server(
    *,
    url: str,
    api_key: str | None,
    expected_model_path: str | None = None,
    expected_profile: str | None = None,
    logger: Any | None = None,
    first_timeout_s: float = 0.8,
    retry_timeout_s: float = 2.0,
) -> Literal["reuse", "spawn"]:
    """Singleton guard: decide whether to reuse an existing LLM server.

    Probes ``url`` (``GET /v1/models``) via the canonical probe entry point
    (:meth:`_MaximPeerBackend.health_check`) with a short timeout, then:

    - **HTTP 200, right model** → return ``"reuse"`` (caller points the lane
      at ``url`` and skips spawn).
    - **HTTP 401 (auth-gated but alive)** → return ``"reuse"``. Per the
      CLAUDE.md "auth in health probes" lesson, a 401 proves the HTTP
      listener is up; on the singleton port that is the leader's own
      auth-gated server. We cannot read ``/v1/models`` to verify the model,
      but reusing is strictly safer than the kill+respawn that triggers the
      cascade.
    - **Connection-refused / timeout / unreachable** → return ``"spawn"``
      (no existing server; proceed with spawn as today).
    - **HTTP 200, DIFFERENT served model** → raise :class:`RuntimeError`
      (fail loud rather than silently serving the wrong model).

    The probe is delegated to :meth:`_MaximPeerBackend.health_check` so this
    function does not re-implement the probe surface (Plan 3 R2.6 invariant).
    """
    from maxim.models.language.maxim_peer_backend import _MaximPeerBackend

    backend = _MaximPeerBackend.for_url(url, api_key=api_key, model=expected_profile or "default")
    result = backend.health_check(
        first_timeout_s=first_timeout_s,
        retry_timeout_s=retry_timeout_s,
        enable_stage2=False,
    )

    if result.outcome == "auth_rejected":
        if logger is not None:
            logger.info(
                "Singleton check: existing auth-gated llama-cpp-server at %s — "
                "treating as alive and reusing (cannot verify served model behind auth)",
                url,
            )
        return REUSE_EXISTING

    if not result.is_reachable:
        # connection_refused / timeout / dns_fail / http_5xx / other →
        # no server we can reuse. Spawn as today.
        if logger is not None:
            logger.debug(
                "Singleton check: no reusable server at %s (probe=%s: %s) — will spawn",
                url,
                result.outcome,
                result.detail,
            )
        return SPAWN_NEW

    # outcome == "ok": listener answered /v1/models without auth rejection.
    # Verify the served model before reusing.
    served = _read_served_model_id(url, api_key)
    match = _served_model_matches(served, expected_model_path, expected_profile)

    if match is False:
        configured = expected_profile or expected_model_path or "<unknown>"
        # Most common cause when configured and served diverge: the operator
        # ran ``maxim --llm <new>`` to swap the runtime server but did not
        # also update ``config.json::llm.profile``. The two state files are
        # intentionally separate (declarative vs runtime, per
        # config_unification.md C2) but the divergence surfaces here when a
        # sub-sim consults the declarative config. Surface concrete fix
        # commands so the operator does not have to read CLAUDE.md to
        # disambiguate.
        raise RuntimeError(
            f"Singleton check: an LLM server is already running on {url} but it is "
            f"serving model {served!r}, not the configured {configured!r}. Refusing "
            f"to silently use the wrong model.\n"
            f"\n"
            f"Most likely cause: ``maxim --llm <model>`` swaps the running server "
            f"but does NOT update ``~/.config/maxim/config.json::llm.profile`` "
            f"(declarative config, intentionally separate per config_unification.md "
            f"C2). The two state files have drifted.\n"
            f"\n"
            f"Resolutions (pick one):\n"
            f"  (a) Adopt the served model as the configured one:\n"
            f"        maxim --list-models   # find the profile name matching the served GGUF\n"
            f"        maxim config set llm.profile <profile-name>\n"
            f"  (b) Restart the server with the configured model:\n"
            f"        pkill -f llama_cpp.server && maxim --llm {configured}\n"
            f"  (c) Stop the existing server entirely and let this run spawn fresh:\n"
            f"        pkill -f llama_cpp.server\n"
            f"\n"
            f"See the CLAUDE.md lesson 'Config.json vs active_llm_model.txt drift "
            f"after maxim --llm <model>'."
        )

    if logger is not None:
        if match is True:
            logger.info(
                "Singleton check: reusing existing llama-cpp-server at %s (served=%s)",
                url,
                served,
            )
        else:
            # match is None — alive but served model unverifiable. Reuse is
            # still the safe choice (avoids the kill+respawn cascade).
            logger.warning(
                "Singleton check: server at %s is alive but its served model could "
                "not be verified (served=%r) — reusing anyway to avoid kill+respawn",
                url,
                served,
            )
    return REUSE_EXISTING


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
