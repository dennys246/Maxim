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


def _model_state_file() -> Path:
    """Return the path to the persisted active model file."""
    from maxim.utils.paths import resolve_user_state

    return resolve_user_state("util/active_llm_model.txt")


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


def _probe_once(url: str, api_key: str | None, timeout_s: float) -> ProbeResult:
    """Single probe attempt. Classifies errors into structured outcomes.

    Catches each network/HTTP exception class explicitly so the caller
    can act on the outcome (drop the lane, warn the user, suggest a fix)
    instead of branching on str(exception). New exception types fall into
    ``other`` rather than crashing the probe.
    """
    import urllib.error
    import urllib.request

    probe_url = _build_probe_url(url)
    req = urllib.request.Request(probe_url)
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    # Cloudflare Bot Fight Mode rejects urllib's default ``Python-urllib/*``
    # User-Agent with a 403 + CF error 1010, even with a valid Bearer token.
    # Every other urllib caller in the codebase already sets this header;
    # this one got missed, which made leader probes silently fail for any
    # peer whose leader is behind a Cloudflare tunnel with Bot Fight Mode.
    req.add_header("User-Agent", "maxim-peer/1.0")

    start = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:  # noqa: S310
            latency_ms = (time.monotonic() - start) * 1000
            return ProbeResult(url, "ok", f"HTTP {resp.status}", round(latency_ms, 1))
    except urllib.error.HTTPError as e:
        latency_ms = (time.monotonic() - start) * 1000
        if e.code == 401:
            return ProbeResult(url, "auth_rejected", "401 Unauthorized", round(latency_ms, 1))
        if 500 <= e.code < 600:
            return ProbeResult(url, "http_5xx", f"HTTP {e.code}", round(latency_ms, 1))
        return ProbeResult(url, "other", f"HTTP {e.code}", round(latency_ms, 1))
    except urllib.error.URLError as e:
        # URLError wraps lower-level errors. Unwrap one layer to classify.
        reason = getattr(e, "reason", e)
        if isinstance(reason, socket.gaierror):
            return ProbeResult(url, "dns_fail", str(reason), None)
        if isinstance(reason, ssl.SSLError):
            return ProbeResult(url, "tls_error", str(reason), None)
        if isinstance(reason, TimeoutError) or isinstance(reason, socket.timeout):
            return ProbeResult(url, "timeout", f"{timeout_s}s", None)
        if isinstance(reason, (ConnectionRefusedError, ConnectionResetError)):
            return ProbeResult(url, "connection_refused", str(reason), None)
        return ProbeResult(url, "other", f"URLError: {reason}", None)
    except socket.gaierror as e:
        return ProbeResult(url, "dns_fail", str(e), None)
    except ssl.SSLError as e:
        return ProbeResult(url, "tls_error", str(e), None)
    except TimeoutError:
        return ProbeResult(url, "timeout", f"{timeout_s}s", None)
    except (ConnectionRefusedError, ConnectionResetError) as e:
        return ProbeResult(url, "connection_refused", str(e), None)
    except Exception as e:  # noqa: BLE001 — defensive catch-all
        return ProbeResult(url, "other", f"{type(e).__name__}: {e}", None)


def probe_llm_server(
    url: str,
    *,
    api_key: str | None = None,
    first_timeout_s: float = 0.8,
    retry_timeout_s: float = 2.5,
) -> ProbeResult:
    """Probe ``GET /v1/models`` with optional Bearer auth, two-attempt retry.

    The first probe uses an aggressive timeout so a healthy leader returns
    fast. On any unreachable outcome we retry once with a longer budget,
    which catches slow leaders mid-cold-start without making the happy
    path slow. ``ok`` and ``auth_rejected`` short-circuit — both mean the
    HTTP listener is alive, no point retrying.

    Returns a :class:`ProbeResult` whose ``outcome`` the caller switches
    on. See :func:`_log_probe_failure` in lane_backends for the human-
    readable warning template per outcome.
    """
    result = _probe_once(url, api_key, first_timeout_s)
    if result.is_reachable:
        return result
    return _probe_once(url, api_key, retry_timeout_s)


def llm_server_responding_at(url: str, *, timeout_s: float = 1.5) -> bool:
    """Return True iff an OpenAI-compatible server answers GET /v1/models at `url`.

    Used for auto-discovery (reuse an already-running server) and for
    validating env-supplied remote URLs before wiring a lane to them.

    Treats both 200 and 401 as "server is up": 401 means an HTTP listener
    with auth enabled is answering, which is still a valid signal the port
    is in use by a real llama-cpp-server (we're just not authenticated).
    """
    if not url:
        return False
    base = url.rstrip("/")
    probe = base + "/models" if base.endswith("/v1") else base + "/v1/models"
    try:
        import urllib.error
        import urllib.request

        # Cloudflare Bot Fight Mode rejects urllib's default User-Agent; set
        # a stable identifier so leaders behind Cloudflare tunnels respond.
        req = urllib.request.Request(probe)
        req.add_header("User-Agent", "maxim-peer/1.0")
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:  # noqa: S310
            return resp.status == 200
    except urllib.error.HTTPError as e:
        return e.code == 401  # server reachable, auth-gated
    except Exception:
        return False


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
