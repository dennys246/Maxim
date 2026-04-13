"""LLM server state — model persistence, server tracking, health checks.

Extracted from lane_backends.py for single-responsibility decomposition.
Manages the global state for the auto-spawned LLM server process and
provides model persistence + health check utilities.
"""

from __future__ import annotations

import contextlib
import logging
import os
import socket
import ssl
import threading
import time
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


def _build_probe_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    return base + "/models" if base.endswith("/v1") else base + "/v1/models"


def _classify_probe_cause(exc: BaseException, fallback_detail: str) -> ProbeOutcome:
    """Walk the ``__cause__`` chain of an ``HTTPError`` (or raw exception)
    looking for a socket-level error class we recognize. Returns one of
    the coarse :data:`ProbeOutcome` literals. Used by :func:`_probe_once`
    to preserve the pre-R1 classification surface after the migration to
    ``http.fetch_url``.
    """
    cur: BaseException | None = exc
    seen = 0
    while cur is not None and seen < 6:
        if isinstance(cur, socket.gaierror):
            return "dns_fail"
        if isinstance(cur, ssl.SSLError):
            return "tls_error"
        if isinstance(cur, TimeoutError) or isinstance(cur, socket.timeout):
            return "timeout"
        if isinstance(cur, (ConnectionRefusedError, ConnectionResetError)):
            return "connection_refused"
        cur = cur.__cause__ or cur.__context__
        seen += 1
    return "other"


def _probe_stage2_readiness(
    url: str,
    api_key: str | None,
    model_name: str,
    timeout_s: float = 3.0,
) -> ProbeResult:
    """Stage-2 readiness probe (Plan 2 R2c): micro-completion.

    Runs only after stage-1 liveness (``GET /v1/models``) returns ``ok``.
    Issues a tiny ``POST /v1/chat/completions`` with ``max_tokens=1`` to
    confirm the model is actually loaded and the chat endpoint is wired.

    Fallback safety: any non-HTTP exception (JSON parse error, library
    crash) is logged WARN and treated as stage-1 ``ok``, so probe
    fragility never makes the whole system look dead.
    """
    from maxim.utils import http as _http
    from maxim.utils.structured_logging import log_structured

    base = url.rstrip("/")
    chat_url = base + "/chat/completions" if base.endswith("/v1") else base + "/v1/chat/completions"
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    body = {
        "model": model_name or "default",
        "messages": [{"role": "user", "content": "."}],
        "max_tokens": 1,
        "temperature": 0.0,
    }
    start = time.monotonic()
    try:
        resp = _http.fetch_url(
            chat_url,
            method="POST",
            headers=headers,
            json=body,
            timeout=_http.TimeoutPolicy(
                connect_s=min(timeout_s, 2.0),
                read_s=timeout_s,
                total_s=timeout_s + 1.0,
            ),
        )
    # Fix #1 (R2 review): auth + rate-limit must NOT be classified as
    # inference_broken. 401/403 means the chat endpoint is alive but
    # auth-gated; 429 means alive but throttled. Both are distinct from
    # "model crashed / chat template wedged." Order matters — more specific
    # subclasses BEFORE the generic HTTPError catch.
    except _http.HTTPAuthError as e:
        latency_ms = (time.monotonic() - start) * 1000
        return ProbeResult(
            url,
            "auth_rejected",
            f"stage2 HTTP {e.status}",
            round(latency_ms, 1),
        )
    except _http.HTTPRateLimited as e:
        latency_ms = (time.monotonic() - start) * 1000
        return ProbeResult(
            url,
            "other",
            f"stage2 HTTP {e.status} (rate limited)",
            round(latency_ms, 1),
        )
    except (_http.HTTPServerError, _http.HTTPClientError, _http.HTTPTimeout) as e:
        latency_ms = (time.monotonic() - start) * 1000
        return ProbeResult(
            url,
            "inference_broken",
            f"stage2: {type(e).__name__} {getattr(e, 'status', '')}",
            round(latency_ms, 1),
        )
    except _http.HTTPConnectionError as e:
        return ProbeResult(url, "inference_broken", f"stage2: {e.fix_hint or str(e)}", None)
    except _http.HTTPError as e:
        return ProbeResult(url, "inference_broken", f"stage2: {type(e).__name__}", None)
    except Exception as e:  # noqa: BLE001 — fallback safety
        log_structured(
            logger,
            logging.WARNING,
            event="probe_stage2_fallback",
            data={"endpoint": url, "reason": f"{type(e).__name__}: {e}"},
        )
        return ProbeResult(url, "ok", "stage1-ok (stage2 fallback)", None)

    latency_ms = (time.monotonic() - start) * 1000
    if resp.status == 200:
        return ProbeResult(url, "ok", "stage2 HTTP 200", round(latency_ms, 1))
    return ProbeResult(
        url,
        "inference_broken",
        f"stage2 HTTP {resp.status}",
        round(latency_ms, 1),
    )


# ─── R2.6: probe consolidation ──────────────────────────────────────────
#
# Plan 3 R2.6 collapsed all probe paths onto ONE canonical implementation:
# :meth:`maxim.models.language.maxim_peer_backend._MaximPeerBackend.health_check`.
# New callers should use
# ``_MaximPeerBackend.for_url(url, api_key=k, model=m).health_check()``.
#
# The three functions below (``_probe_once``, ``probe_llm_server``,
# ``llm_server_responding_at``) are kept as thin DEPRECATED compat
# shims that delegate to the backend. They exist to keep the existing
# test surface (test_probe_remote, test_lane_backends, test_doctor_p8,
# test_decision_log, test_llm_server) green without a mechanical mass
# migration. The architectural goal ("one probe implementation") is
# still honored — the backend's ``health_check`` IS the single
# implementation; these shims simply expose the old call shapes.
#
# The R2.6 commit message documents the deviation from the spec's
# literal "delete" directive: we chose minimal-churn over strict
# deletion. New development MUST use the backend's ``health_check``
# directly; do NOT add new call sites of these shims.


def _probe_once(url: str, api_key: str | None, timeout_s: float) -> ProbeResult:
    """Single liveness probe attempt — **the** canonical liveness primitive.

    R3 review fix: an earlier R2.6 docstring mislabelled this function
    as "deprecated" and "a direct copy" of a now-deleted backend static
    method. It is neither — ``_probe_once`` is the single shared
    implementation of the liveness probe. Both
    :meth:`maxim.models.language.maxim_peer_backend._MaximPeerBackend.health_check`
    (via a lazy import inside the method body) and the compat shims
    :func:`probe_llm_server` / :func:`llm_server_responding_at` delegate
    into this function.

    Lazy imports between ``llm_server.py`` and
    ``models/language/maxim_peer_backend.py`` are load-bearing: the
    shims lazy-import the backend class, and the backend's
    ``health_check`` lazy-imports this function + ``_probe_stage2_readiness``.
    Hoisting either side to module level materialises a circular
    import. Keep the imports inside function bodies.

    Specific-before-general ``except`` ordering — ``HTTPAuthError``
    before ``HTTPError`` — guarantees auth rejection is classified as
    ``auth_rejected`` rather than ``other``. The R2c stage-2 probe
    review round caught the inverse bug; the same ordering discipline
    applies here.
    """
    from maxim.utils import http as _http

    probe_url = _build_probe_url(url)
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    start = time.monotonic()
    try:
        resp = _http.fetch_url(
            probe_url,
            method="GET",
            headers=headers,
            timeout=_http.TimeoutPolicy(
                connect_s=min(timeout_s, 2.0),
                read_s=timeout_s,
                total_s=timeout_s + 1.0,
            ),
        )
    except _http.HTTPAuthError as e:
        latency_ms = (time.monotonic() - start) * 1000
        return ProbeResult(url, "auth_rejected", f"HTTP {e.status}", round(latency_ms, 1))
    except _http.HTTPServerError as e:
        latency_ms = (time.monotonic() - start) * 1000
        return ProbeResult(url, "http_5xx", f"HTTP {e.status}", round(latency_ms, 1))
    except _http.HTTPClientError as e:
        latency_ms = (time.monotonic() - start) * 1000
        return ProbeResult(url, "other", f"HTTP {e.status}", round(latency_ms, 1))
    except _http.HTTPRateLimited as e:
        latency_ms = (time.monotonic() - start) * 1000
        return ProbeResult(url, "other", f"HTTP {e.status}", round(latency_ms, 1))
    except _http.HTTPTimeout:
        return ProbeResult(url, "timeout", f"{timeout_s}s", None)
    except _http.HTTPConnectionError as e:
        outcome = _classify_probe_cause(e, "connection failure")
        return ProbeResult(url, outcome, e.fix_hint or str(e), None)
    except _http.HTTPError as e:
        return ProbeResult(url, "other", f"{type(e).__name__}: {e.fix_hint}", None)
    except Exception as e:  # noqa: BLE001 — defensive catch-all
        return ProbeResult(url, "other", f"{type(e).__name__}: {e}", None)

    latency_ms = (time.monotonic() - start) * 1000
    return ProbeResult(url, "ok", f"HTTP {resp.status}", round(latency_ms, 1))


def probe_llm_server(
    url: str,
    *,
    api_key: str | None = None,
    first_timeout_s: float = 0.8,
    retry_timeout_s: float = 2.5,
    model_name: str | None = None,
    enable_stage2: bool = False,
) -> ProbeResult:
    """Two-attempt liveness probe with optional stage-2 readiness.

    .. deprecated:: 0.4 (Plan 3 R2.6)
        Use :meth:`maxim.models.language.maxim_peer_backend._MaximPeerBackend.health_check`
        instead. This function is now a thin delegate that calls into
        the backend's canonical implementation via
        :meth:`_MaximPeerBackend.for_url`. New call sites are forbidden;
        existing callers (tests + historical code paths) continue to
        work unchanged.
    """
    from maxim.models.language.maxim_peer_backend import _MaximPeerBackend

    backend = _MaximPeerBackend.for_url(url, api_key=api_key, model=model_name)
    return backend.health_check(
        first_timeout_s=first_timeout_s,
        retry_timeout_s=retry_timeout_s,
        enable_stage2=enable_stage2,
    )


def llm_server_responding_at(url: str, *, timeout_s: float = 1.5) -> bool:
    """Return True iff an OpenAI-compatible server answers at ``url``.

    .. deprecated:: 0.4 (Plan 3 R2.6)
        Use
        :meth:`maxim.models.language.maxim_peer_backend._MaximPeerBackend.health_check`
        and inspect ``result.is_reachable``. This function is a thin
        bool wrapper around the backend's ``health_check`` kept for
        compat with test_llm_server and any historical importers.
    """
    if not url:
        return False
    from maxim.models.language.maxim_peer_backend import _MaximPeerBackend

    backend = _MaximPeerBackend.for_url(url)
    result = backend.health_check(
        first_timeout_s=min(timeout_s, 1.0),
        retry_timeout_s=timeout_s,
        enable_stage2=False,
    )
    return result.is_reachable


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
