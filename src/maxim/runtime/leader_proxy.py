"""LeaderProxy — reverse proxy sitting in front of llama-cpp-server (Phase 7a).

Sits on port 8099, forwards inference requests to llama-cpp-server on :8100.
Tunnel ingress points here instead of directly at the inference server.

Responsibilities:
  - Authoritative Bearer auth BEFORE requests reach llama-cpp-server
  - Per-request logging: request-id, peer IP, model, latency, tokens
  - /v1/debug/status: GPU utilization, VRAM, temperature, uptime
  - /v1/debug/last-requests: ring buffer of last 100 requests
  - Injects X-Maxim-* response headers for peer-side trace enrichment

Design: stdlib http.server for the listener side, maxim.utils.http
(httpx) for upstream forwarding (post Plan 1 R1). No FastAPI/uvicorn dep.
Adds ~1-2ms per request vs direct llama-cpp-server access.

Supersedes debug_status_server.py (which only served /v1/debug/status).
"""

from __future__ import annotations

import collections
import json
import logging
import os
import re
import secrets
import subprocess
import sys
import threading
import time

# ── Input validation ─────────────────────────────────────────────────────────

_BRANCH_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._/-]*$")

# PEP 440 subset: 1.2.3, 1.2.3rc1, 1.2.3.post1, 1.2.3.dev0 etc.
# Rejects shell metacharacters; all pip commands use list args (no shell).
_VERSION_RE = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+([a-zA-Z0-9.]+)?$")

# ── Extras / import mapping (shared by _handle_debug_deps + pip upgrade) ────

# Maps pymaxim extras to the import name used for detection.
# Keys MUST be a subset of _ALLOWED_EXTRAS below.
_EXTRA_IMPORT_MAP: dict[str, str] = {
    "semantic": "sentence_transformers",
    "llm-llama": "llama_cpp",
    "llm-torch": "torch",
    "llm-anthropic": "anthropic",
    "llm-openai": "openai",
    "vision": "cv2",
    "audio": "sounddevice",
    "search": "duckduckgo_search",
    "tts": "piper",
    "yolo": "ultralytics",
}

# Canonical allowlist — authoritative for both /v1/admin/install and pip
# upgrade. Duplicated from _handle_admin_install() so the upgrade path can
# filter without reaching into a method body. Keep these in sync.
_ALLOWED_EXTRAS: frozenset[str] = frozenset(
    {
        "semantic",
        "llm-llama",
        "llm-server",
        "llm-torch",
        "llm-anthropic",
        "llm-openai",
        "vision",
        "audio",
        "reachy",
        "comms",
        "search",
        "temporal",
        "training",
        "tts",
        "yolo",
        "database",
    }
)

# Shared state for async install — written by the install thread, read by
# the /v1/debug/install-status endpoint. Dict (not dataclass) so the
# background thread can mutate in place without import gymnastics.
_install_status: dict[str, Any] = {
    "status": "idle",
    "installed": [],
    "pip_output": "",
    "error": "",
    "started_by": "",
}


def _remote_update_allowed() -> bool:
    """True iff ``MAXIM_ALLOW_REMOTE_UPDATE`` is set to a truthy value."""
    return os.environ.get("MAXIM_ALLOW_REMOTE_UPDATE", "").strip().lower() in (
        "1",
        "true",
        "t",
        "yes",
        "y",
        "on",
    )


def _validate_branch(branch: str) -> str:
    """Validate and return a sanitized branch name.

    Raises ``ValueError`` for names that could cause unexpected git behavior:
    path traversal (``..``), flag injection (leading ``-``), or invalid chars.
    """
    branch = str(branch).strip()
    if not branch:
        raise ValueError("Empty branch name")
    if ".." in branch:
        raise ValueError(f"Branch name contains '..': {branch!r}")
    if branch.startswith("-"):
        raise ValueError(f"Branch name starts with '-': {branch!r}")
    if not _BRANCH_RE.match(branch):
        raise ValueError(f"Invalid branch name: {branch!r}")
    return branch


def _sanitize_git_output(text: str | None, max_len: int = 300) -> str:
    """Remove file system paths and truncate for safe error reporting."""
    if not text:
        return ""
    # Replace absolute paths that could leak system info
    text = re.sub(r"/[\w./-]{5,}", "<path>", text)
    return text[-max_len:] if len(text) > max_len else text


import shutil
import tempfile


def _detect_install_mode() -> str:
    """Return ``'dev'`` if running from a git checkout, ``'pip'`` otherwise."""
    repo_root = Path(__file__).resolve().parents[3]
    if (repo_root / ".git").is_dir():
        return "dev"
    return "pip"


def _try_import(module_name: str) -> bool:
    """Return True if *module_name* is importable."""
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False


def _detect_installed_extras() -> list[str]:
    """Return pymaxim extras that are importable AND in the allowlist."""
    detected = [name for name, mod in _EXTRA_IMPORT_MAP.items() if _try_import(mod)]
    return [e for e in detected if e in _ALLOWED_EXTRAS]


def _get_current_version() -> str:
    """Return the installed pymaxim version, or ``'unknown'``."""
    try:
        from importlib.metadata import version as _pkg_version

        return _pkg_version("pymaxim")
    except Exception:
        return "unknown"


def _get_latest_pypi_version() -> str | None:
    """Query PyPI for the latest pymaxim version. Returns None on any failure.

    Uses ``pip index versions`` (pip >= 21.2) with a 15s timeout.
    Gracefully returns None if pip is too old or PyPI is unreachable.
    """
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "index", "versions", "pymaxim"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode == 0 and result.stdout:
            # Output: "pymaxim (0.3.1)\n  ..."  — version in first parens.
            m = re.search(r"\(([0-9][0-9a-zA-Z.]+)\)", result.stdout)
            if m:
                return m.group(1)
    except Exception:
        pass
    return None


from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from pathlib import Path
from typing import Any

logger = logging.getLogger("maxim.leader_proxy")

DEFAULT_PROXY_PORT = 8099
DEFAULT_UPSTREAM_PORT = 8100
_MAX_RECENT_REQUESTS = 100
_INFERENCE_PROXY_TIMEOUT_S = 300.0
"""Timeout for leader-proxy → llama-cpp-server inference calls.

60s (the former value) is too short for Qwen-14B on long-context prompts —
prefill alone can take 10-20s, leaving <40s for generation. 300s covers
worst-case long-context + long-output on mid-range hardware and matches the
peer backend's default ``timeout_s``.
"""

# ─── TTFT keepalive (llm_timeout_scalability.md Stage 3) ──────────────────
#
# Cloudflare tunnels enforce a ~100s idle timeout on the origin connection.
# When the agent hits a self-hosted leader through a cloudflared tunnel and
# the upstream LLM's time-to-first-token exceeds the idle window (common on
# 30B+ models with long prompts on edge hardware), the tunnel closes
# silently — the proxy doesn't notice until it tries to write the first
# real chunk and gets EPIPE. Sending an SSE comment frame at regular
# intervals during the silent window resets the idle timer without
# affecting the response semantics (clients ignore lines beginning with
# ":" per the SSE spec).
_PROXY_KEEPALIVE_DEFAULT_INTERVAL_S = 30.0
_PROXY_KEEPALIVE_MIN_INTERVAL_S = 5.0
_PROXY_KEEPALIVE_MAX_INTERVAL_S = 90.0

# Pre-formatted HTTP/1.1 chunked frame carrying an SSE keepalive comment.
# Inner payload ``: keepalive\n\n`` is 13 bytes (0xd in hex). Total frame:
# ``d\r\n`` (chunk size) + ``: keepalive\n\n`` (payload) + ``\r\n`` (trailer)
# = 18 bytes. Compile-time constant so the hot path doesn't re-encode it.
_KEEPALIVE_CHUNK_FRAME: bytes = b"d\r\n: keepalive\n\n\r\n"


def _resolve_keepalive_interval_s() -> float:
    """Parse ``MAXIM_PROXY_KEEPALIVE_INTERVAL_S`` with bounds.

    Invalid / unparseable values fall back to the default rather than
    bricking every streaming response with a 0-second interval.
    """
    raw = os.environ.get("MAXIM_PROXY_KEEPALIVE_INTERVAL_S")
    if raw is None or raw.strip() == "":
        return _PROXY_KEEPALIVE_DEFAULT_INTERVAL_S
    try:
        value = float(raw)
    except (ValueError, TypeError):
        return _PROXY_KEEPALIVE_DEFAULT_INTERVAL_S
    if value < _PROXY_KEEPALIVE_MIN_INTERVAL_S:
        return _PROXY_KEEPALIVE_MIN_INTERVAL_S
    if value > _PROXY_KEEPALIVE_MAX_INTERVAL_S:
        return _PROXY_KEEPALIVE_MAX_INTERVAL_S
    return value


def _is_event_stream_response(response_headers: dict[str, str]) -> bool:
    """Check whether the upstream response is a Server-Sent-Events stream.

    Returns True iff Content-Type indicates ``text/event-stream``
    (case-insensitive header lookup; value may include parameters like
    ``charset=utf-8``). Buffered (``application/json``) responses must
    skip keepalive injection — there's no in-band slot for SSE comment
    frames in a single JSON body.
    """
    for key, val in response_headers.items():
        if key.lower() == "content-type":
            return "text/event-stream" in val.lower()
    return False


# ─── Proxy context-overflow admission (llm_timeout_scalability.md F/U) ──
#
# The heavy stress test on 2026-06-03 (Stage 3 validation) surfaced a
# second failure class: prompts larger than the upstream model's context
# window cause llama-cpp-server to silently abort generation, producing a
# stream that limps along with keepalives until the upstream gives up.
# The agent gets no actionable error.
#
# The fix is admission control at the proxy. Before forwarding an
# inference request, we estimate prompt tokens from the request body and
# compare against the configured context window. Oversize requests get a
# clean HTTP 413 with a typed error carrying the numbers the operator
# needs to debug.
#
# Token counting is char-based (no new dep, no extra HTTP call). English
# averages ~4 chars/token; we use 3.5 to slightly overestimate tokens
# (defensive). For Qwen / Llama tokenizers on code or markdown, the ratio
# can drop to ~3.0 — the overhead margin (default 256 tokens) covers the
# slop. Operators on tight budgets can tune both knobs.
_PROXY_CONTEXT_OVERHEAD_DEFAULT = 256
_PROXY_CONTEXT_OVERHEAD_MIN = 0
_PROXY_CONTEXT_OVERHEAD_MAX = 4096
_PROXY_CHAR_TO_TOKEN_RATIO = 3.5
_PROXY_DEFAULT_MAX_TOKENS = 1024

# Endpoints subject to admission. Other ``/v1/*`` paths (e.g. ``/v1/models``,
# ``/v1/embeddings``) pass through unchecked — admission only applies to
# generation calls where context-window violations are a real failure mode.
_PROXY_INFERENCE_PATHS: frozenset[str] = frozenset({"/v1/chat/completions", "/v1/completions"})


def _resolve_context_overhead_tokens() -> int:
    """Parse ``MAXIM_PROXY_CONTEXT_OVERHEAD_TOKENS`` with bounds."""
    raw = os.environ.get("MAXIM_PROXY_CONTEXT_OVERHEAD_TOKENS")
    if raw is None or raw.strip() == "":
        return _PROXY_CONTEXT_OVERHEAD_DEFAULT
    try:
        value = int(raw)
    except (ValueError, TypeError):
        return _PROXY_CONTEXT_OVERHEAD_DEFAULT
    return max(_PROXY_CONTEXT_OVERHEAD_MIN, min(_PROXY_CONTEXT_OVERHEAD_MAX, value))


def _is_admission_enabled() -> bool:
    """Operator opt-out via ``MAXIM_PROXY_CONTEXT_ADMISSION``.

    Default ON when ``MAXIM_LLM_N_CTX`` is resolvable, OFF otherwise (the
    proxy emits a one-time startup WARNING when it can't determine the
    context window, then passes through). Setting the var to ``0`` /
    ``false`` / ``no`` / ``off`` forces OFF even if n_ctx is known.
    """
    raw = os.environ.get("MAXIM_PROXY_CONTEXT_ADMISSION")
    if raw is None:
        return True
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


def _resolve_proxy_context_window() -> int | None:
    """Resolve the upstream model's runtime context window.

    Reads through the standard ``resolve_setting`` precedence chain
    (CLI > env > config.json > default). ``MAXIM_LLM_N_CTX`` is what
    the leader sets when launching llama-cpp-server, so it's the
    authoritative runtime value (vs llama-cpp's ``/v1/models`` which
    reports the *model's* training context, not the runtime ``-c`` arg).

    Returns ``None`` if no value is configured anywhere — the caller
    should treat this as "skip admission, log a warning, pass through".
    """
    try:
        from maxim.runtime.config_loader import load_config, resolve_setting

        cfg = load_config()
        value, source = resolve_setting("llm.n_ctx", config=cfg)
        if value is not None and source != "default":
            return int(value)
    except Exception:
        return None
    return None


def _estimate_inference_input_tokens(body: bytes) -> tuple[int, int]:
    """Estimate prompt tokens + read ``max_tokens`` from a request body.

    Handles both ``/v1/chat/completions`` (messages list with role/content
    objects, content may be a string OR a multipart list) and the legacy
    ``/v1/completions`` shape (single ``prompt`` string or list of strings).

    Returns ``(estimated_prompt_tokens, max_tokens)``. Either may be 0
    if the body is malformed or the field is missing — the caller
    decides whether to (a) pass through on a 0 (let upstream handle the
    malformed body) or (b) apply a default ``max_tokens`` for admission
    arithmetic.

    The +16 chars per message is the per-role JSON+template overhead
    (e.g. ``<|im_start|>user\\n...<|im_end|>``) — chat templates eat
    roughly 8-12 tokens per turn beyond the raw content; 16 chars ≈ 4-5
    tokens at our ratio, defensively rounded up.
    """
    try:
        data = json.loads(body)
    except (ValueError, TypeError):
        return 0, 0
    if not isinstance(data, dict):
        return 0, 0

    char_count = 0
    messages = data.get("messages")
    if isinstance(messages, list):
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if isinstance(content, str):
                char_count += len(content)
            elif isinstance(content, list):
                # Multipart content (vision / mixed media). Sum the
                # text parts only — image / audio parts contribute
                # different token costs that this estimator doesn't
                # try to model. The admission gate is a safety net,
                # not a precise predictor.
                for part in content:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        char_count += len(part["text"])
            char_count += 16  # role + chat-template scaffolding overhead

    # Legacy /v1/completions: "prompt" may be string or list-of-strings.
    prompt = data.get("prompt")
    if isinstance(prompt, str):
        char_count += len(prompt)
    elif isinstance(prompt, list):
        for p in prompt:
            if isinstance(p, str):
                char_count += len(p)

    estimated_tokens = int(char_count / _PROXY_CHAR_TO_TOKEN_RATIO)
    raw_max = data.get("max_tokens")
    if raw_max is None:
        max_tokens = 0
    else:
        try:
            max_tokens = max(0, int(raw_max))
        except (ValueError, TypeError):
            max_tokens = 0
    return estimated_tokens, max_tokens


# Cached resolution of the upstream context window. ``MAXIM_LLM_N_CTX``
# doesn't change at runtime (the leader sets it when launching
# llama-cpp-server), so we resolve once and cache. Tests reset via
# ``_reset_proxy_context_window_cache_for_test``.
#
# State shape: ``(resolved: bool, value: int | None)`` — the explicit
# ``resolved`` flag distinguishes "haven't checked yet" from "checked,
# nothing configured" so we don't re-walk the loader on every request
# in the disabled-by-config case.
_proxy_context_window_state: tuple[bool, int | None] = (False, None)
_proxy_context_window_lock = threading.Lock()


def _get_proxy_context_window() -> int | None:
    """Thread-safe cached resolution of the upstream context window.

    Returns the resolved value (int) or ``None`` if no value is
    configured anywhere. On the first None resolution, logs a WARNING
    that admission is disabled; on the first resolved value, logs an
    INFO with the active window + overhead so operators see the gate
    is on. The double-checked locking pattern is safe here because the
    state tuple is replaced atomically (Python tuple assignment is
    atomic; the lock only prevents redundant loader walks).
    """
    global _proxy_context_window_state
    resolved, value = _proxy_context_window_state
    if resolved:
        return value
    with _proxy_context_window_lock:
        resolved, value = _proxy_context_window_state
        if resolved:
            return value
        value = _resolve_proxy_context_window()
        _proxy_context_window_state = (True, value)
        if value is None:
            logger.warning(
                "Proxy context-overflow admission disabled: MAXIM_LLM_N_CTX "
                "is not configured (env or config.json::llm.n_ctx). "
                "Set MAXIM_LLM_N_CTX explicitly to enable the gate; "
                "oversize prompts may otherwise hang the upstream model. "
                "Set MAXIM_PROXY_CONTEXT_ADMISSION=0 to silence this warning."
            )
        else:
            logger.info(
                "Proxy context-overflow admission enabled (context_window=%d, "
                "overhead=%d tokens, char_to_token_ratio=%.1f)",
                value,
                _resolve_context_overhead_tokens(),
                _PROXY_CHAR_TO_TOKEN_RATIO,
            )
        return value


def _reset_proxy_context_window_cache_for_test() -> None:
    """Test helper: clear the cached context-window resolution.

    Production code never calls this. The autouse env-scrub fixture in
    ``tests/conftest.py`` invokes it so per-test env-var changes to
    ``MAXIM_LLM_N_CTX`` actually take effect.
    """
    global _proxy_context_window_state
    with _proxy_context_window_lock:
        _proxy_context_window_state = (False, None)


def _check_context_admission(
    body: bytes,
    n_ctx: int,
    overhead: int,
) -> dict[str, Any] | None:
    """Decide whether a request would fit in the model's context window.

    Returns ``None`` if admitted (or if the body can't be parsed —
    don't false-reject on malformed input, let upstream return a
    cleaner 400). Returns an error dict suitable for JSON serialisation
    if rejected: the dict carries the numbers the operator needs to
    debug (``estimated_prompt_tokens``, ``max_tokens``,
    ``context_window``, ``safety_overhead_tokens``) plus an
    OpenAI-compatible ``error`` envelope so agent-side error-handlers
    that key on ``error.code`` / ``error.type`` work without
    Maxim-specific code.
    """
    prompt_tokens, max_tokens = _estimate_inference_input_tokens(body)
    if prompt_tokens == 0:
        return None
    effective_max = max_tokens if max_tokens > 0 else _PROXY_DEFAULT_MAX_TOKENS
    total = prompt_tokens + effective_max + overhead
    if total <= n_ctx:
        return None
    return {
        "error": {
            "code": "context_overflow",
            "type": "invalid_request_error",
            "message": (
                f"Estimated prompt tokens ({prompt_tokens}) + max_tokens "
                f"({effective_max}) + safety overhead ({overhead}) = {total} "
                f"exceeds the model's context window ({n_ctx}). Shrink the "
                f"prompt, reduce max_tokens, or run a model with a larger "
                f"context window."
            ),
            "estimated_prompt_tokens": prompt_tokens,
            "max_tokens": effective_max,
            "context_window": n_ctx,
            "safety_overhead_tokens": overhead,
        }
    }


def _run_keepalive_emitter(
    wfile: Any,
    write_lock: threading.Lock,
    stop_event: threading.Event,
    interval_s: float,
) -> None:
    """Emit SSE keepalive frames to ``wfile`` until ``stop_event`` is set.

    Runs in a daemon thread alongside the main proxy streaming loop.
    Acquires ``write_lock`` around each frame so writes serialise with
    the main thread's chunk forwarder — ``wfile`` is backed by a single
    socket and concurrent writes would interleave bytes.

    Exits silently on any write failure: a broken pipe means the client
    has already closed; the main thread will detect the same error when
    it tries to write the first real chunk and handle logging there.

    The wait-then-check ordering is deliberate: we want to emit a
    keepalive AFTER each interval elapses, not before. So
    ``stop_event.wait(timeout=interval_s)`` returns False on timeout
    (still streaming, emit a frame), True if stopped (exit).
    """
    while True:
        if stop_event.wait(timeout=interval_s):
            return
        with write_lock:
            if stop_event.is_set():
                # Race: main thread set the event between our wait()
                # returning False and our lock acquisition. The first
                # real chunk is about to write; skip this keepalive.
                return
            try:
                wfile.write(_KEEPALIVE_CHUNK_FRAME)
                wfile.flush()
            except Exception:
                stop_event.set()
                return


# ─── GPU metrics ──────────────────────────────────────────────────────────


def _query_nvidia_smi() -> dict[str, Any] | None:
    """Query nvidia-smi for GPU metrics. Returns None if unavailable."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if result.returncode != 0:
            return None
        line = result.stdout.strip().split("\n")[0]
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            return None
        return {
            "utilization_pct": float(parts[0]),
            "vram_used_gb": round(float(parts[1]) / 1024, 2),
            "vram_total_gb": round(float(parts[2]) / 1024, 2),
            "temperature_c": float(parts[3]),
        }
    except Exception:
        return None


def _current_llama_server_n_ctx(port: int) -> int | None:
    """Return the n_ctx the running llama-cpp-server is configured for, or None.

    Probes ``/v1/models`` on the upstream llama-cpp-server for
    ``context_length``, falling back to process command-line inspection on
    Linux. Shared by :func:`check_vram_pressure` in ``doctor/checks.py``
    and :func:`_handle_debug_vram` in this module.
    """
    import socket

    try:
        req_bytes = (f"GET /v1/models HTTP/1.0\r\nHost: 127.0.0.1:{port}\r\n\r\n").encode()
        with socket.create_connection(("127.0.0.1", port), timeout=2.0) as s:
            s.sendall(req_bytes)
            raw = b""
            while True:
                chunk = s.recv(4096)
                if not chunk:
                    break
                raw += chunk
        body = raw.split(b"\r\n\r\n", 1)[-1]
        data = json.loads(body)
        for model in data.get("data", []):
            ctx = model.get("context_length") or model.get("n_ctx") or model.get("max_context_length")
            if ctx:
                return int(ctx)
    except Exception:
        pass

    # Process args fallback — best-effort, cross-platform
    try:
        import platform as _platform

        system = _platform.system().lower()
        if system == "linux":
            import glob

            for cmdline_path in glob.glob("/proc/*/cmdline"):
                try:
                    with open(cmdline_path, "rb") as f:
                        args = f.read().split(b"\x00")
                    args_str = [a.decode("utf-8", errors="ignore") for a in args]
                    if any("llama" in a for a in args_str):
                        for i, arg in enumerate(args_str):
                            if arg in ("--ctx-size", "-c", "--n-ctx") and i + 1 < len(args_str):
                                return int(args_str[i + 1])
                            if arg.startswith("--ctx-size="):
                                return int(arg.split("=", 1)[1])
                except Exception:
                    continue
    except Exception:
        pass
    return None


# ─── log ring buffer ─────────────────────────────────────────────────────

_MAX_LOG_LINES = 500


class _LogBuffer(logging.Handler):
    """Logging handler that captures records into a thread-safe ring buffer.

    Installed on the root 'maxim' logger so all subsystem logs are captured.
    The /v1/debug/logs endpoint reads from this buffer.
    """

    def __init__(self, maxlen: int = _MAX_LOG_LINES) -> None:
        super().__init__()
        self._buffer: collections.deque[dict[str, Any]] = collections.deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._seq = 0

    def emit(self, record: logging.LogRecord) -> None:
        try:
            entry = {
                "seq": self._seq,
                "ts": record.created,
                "level": record.levelname,
                "logger": record.name,
                "message": self.format(record),
            }
            with self._lock:
                self._buffer.append(entry)
                self._seq += 1
        except Exception:
            pass  # Never crash the logging system

    def get_since(self, since_ts: float = 0.0, since_seq: int = -1, limit: int = 200) -> list[dict[str, Any]]:
        """Return log entries after the given timestamp or sequence number."""
        with self._lock:
            if since_seq >= 0:
                entries = [e for e in self._buffer if e["seq"] > since_seq]
            else:
                entries = [e for e in self._buffer if e["ts"] > since_ts]
        return entries[-limit:]

    def latest_seq(self) -> int:
        with self._lock:
            return self._seq - 1 if self._seq > 0 else -1


# Singleton — installed once, shared across handler instances via class attr.
_log_buffer: _LogBuffer | None = None


def _ensure_log_buffer() -> _LogBuffer:
    """Install the log buffer handler on the maxim logger (idempotent)."""
    global _log_buffer
    if _log_buffer is not None:
        return _log_buffer
    _log_buffer = _LogBuffer()
    _log_buffer.setFormatter(logging.Formatter("%(message)s"))
    _log_buffer.setLevel(logging.DEBUG)
    logging.getLogger("maxim").addHandler(_log_buffer)
    return _log_buffer


# ─── request ring buffer ──────────────────────────────────────────────────


class _RequestLog:
    """Thread-safe ring buffer of recent request records."""

    def __init__(self, maxlen: int = _MAX_RECENT_REQUESTS) -> None:
        self._buffer: collections.deque[dict[str, Any]] = collections.deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._total: int = 0

    def record(self, entry: dict[str, Any]) -> None:
        with self._lock:
            self._buffer.append(entry)
            self._total += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "total_requests": self._total,
                "recent": list(self._buffer),
            }


# ─── proxy handler ────────────────────────────────────────────────────────


class _ProxyHandler(BaseHTTPRequestHandler):
    """Reverse-proxy handler with auth, logging, and debug endpoints."""

    # HTTP/1.1 is required for Transfer-Encoding: chunked.  Without this,
    # BaseHTTPRequestHandler defaults to HTTP/1.0 which doesn't support
    # chunked responses, and _proxy_request() would have to buffer the entire
    # llama-cpp response before sending — causing Cloudflare 524 timeouts on
    # long-inference requests (~105-125s generation time > Cloudflare's edge
    # timeout). HTTP/1.1 is a superset of HTTP/1.0 for all non-proxy paths.
    protocol_version = "HTTP/1.1"

    # Set by the server factory (class-level attrs)
    api_key: str | None = None
    upstream_url: str = "http://127.0.0.1:8100"
    request_log: _RequestLog | None = None
    lane_metrics: Any = None  # LaneMetrics instance for the large tier
    concurrency_semaphore: threading.Semaphore | None = None  # Phase 7b
    rate_limiter: Any = None  # PeerRateLimiter instance (Phase 7b)
    start_time: float = 0.0

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        # Route to structured logger instead of stderr
        pass

    # ─── admission control (Phase 7b) ────────────────────────────────

    def _check_admission(self) -> bool:
        """Check concurrency cap + per-peer rate limit. Returns True if allowed."""
        # Per-peer rate limit
        if self.rate_limiter is not None:
            peer_key = self.client_address[0]  # bucket by IP; all peers share one cluster key
            if not self.rate_limiter.try_acquire(peer_key):
                self._send_json(
                    429,
                    {
                        "error": "Rate limit exceeded",
                        "retry_after_s": 1,
                    },
                )
                logger.info(
                    "proxy: rate-limited peer=%s",
                    self.client_address[0],
                )
                return False
        # Concurrency cap
        if self.concurrency_semaphore is not None:
            if not self.concurrency_semaphore.acquire(blocking=False):
                queue_depth = self._queue_depth()
                self._send_json(
                    429,
                    {
                        "error": "Too many concurrent requests",
                        "queue_depth": queue_depth,
                    },
                )
                logger.info(
                    "proxy: concurrency-limited peer=%s depth=%d",
                    self.client_address[0],
                    queue_depth,
                )
                return False
        return True

    def _release_concurrency(self) -> None:
        """Release the concurrency semaphore after a proxied request completes."""
        if self.concurrency_semaphore is not None:
            self.concurrency_semaphore.release()

    def _queue_depth(self) -> int:
        """Approximate in-flight request count for X-Maxim-Queue-Depth header."""
        if self.lane_metrics is not None:
            return self.lane_metrics.current_in_flight
        return 0

    # Maximum request body size (1 MB). Prevents memory exhaustion from oversized payloads.
    MAX_BODY_SIZE = 1_048_576

    def _read_body(self, max_size: int | None = None) -> bytes | None:
        """Read request body with size limit. Returns None if no body or over limit."""
        limit = max_size or self.MAX_BODY_SIZE
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length <= 0:
            return None
        if content_length > limit:
            self._send_json(413, {"error": f"Request body too large (max {limit} bytes)"})
            return None
        return self.rfile.read(content_length)

    # ─── auth ─────────────────────────────────────────────────────────

    def _check_auth(self) -> bool:
        """Enforce Bearer auth. Returns True if authorized."""
        if not self.api_key:
            return True
        auth = self.headers.get("Authorization", "")
        if secrets.compare_digest(auth, f"Bearer {self.api_key}"):
            return True
        # Log enough to diagnose mismatches without leaking the full key
        expected_prefix = self.api_key[:6] if self.api_key else "?"
        got_prefix = auth[7:13] if auth.startswith("Bearer ") else repr(auth[:20])
        logger.warning(
            "Auth failed: peer=%s expected=%s... got=%s...",
            self.client_address[0],
            expected_prefix,
            got_prefix,
        )
        self._send_json(401, {"error": "Invalid API key"})
        return False

    # ─── debug endpoints (served directly, not proxied) ───────────────

    def _handle_debug_status(self) -> None:
        gpu = _query_nvidia_smi()
        # Include active LLM model info from lane_backends
        active_model = None
        llm_uptime_s = None
        try:
            import maxim.runtime.llm_server as _srv

            active_model = _srv._active_model
            if _srv._llm_start_time is not None:
                llm_uptime_s = round(time.time() - _srv._llm_start_time, 1)
        except Exception:
            pass

        # Lane metrics (infer lane)
        lane_metrics = None
        try:
            from maxim.models.language.lane_metrics import get_metrics_registry

            m = get_metrics_registry().get("large")
            lane_metrics = {
                "total_requests": m.total_requests,
                "in_flight": m.current_in_flight,
                "avg_latency_ms": round(m.avg_latency_ms, 1) if m.avg_latency_ms else None,
            }
        except Exception:
            pass

        self._send_json(
            200,
            {
                "status": "ok",
                "maxim_uptime_s": round(time.time() - self.start_time, 1),
                "llm_model": active_model,
                "llm_uptime_s": llm_uptime_s,
                "gpu": gpu,
                "infer_lane": lane_metrics,
                "timestamp": time.time(),
            },
        )

    def _handle_debug_heartbeat(self) -> None:
        try:
            from maxim.runtime.heartbeat import get_heartbeat_monitor

            self._send_json(200, get_heartbeat_monitor().snapshot())
        except Exception:
            # Fallback: at least return system metrics
            try:
                from maxim.runtime.system_metrics import collect_all

                self._send_json(200, collect_all())
            except Exception:
                self._send_json(200, {"error": "heartbeat not available"})

    def _handle_debug_metrics(self) -> None:
        try:
            from maxim.models.language.lane_metrics import get_metrics_registry

            self._send_json(200, get_metrics_registry().snapshot())
        except Exception:
            self._send_json(200, {})

    def _handle_debug_version(self) -> None:
        try:
            from maxim import get_version_info

            self._send_json(200, get_version_info())
        except Exception:
            self._send_json(200, {"version": "unknown"})

    def _handle_debug_logs(self) -> None:
        """GET /v1/debug/logs?since_seq=N&limit=200 — return recent log lines."""
        buf = _ensure_log_buffer()

        # Parse query params
        from urllib.parse import parse_qs, urlparse

        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        since_seq = int(params.get("since_seq", ["-1"])[0])
        since_ts = float(params.get("since_ts", ["0"])[0])
        limit = min(int(params.get("limit", ["200"])[0]), 500)

        entries = buf.get_since(since_ts=since_ts, since_seq=since_seq, limit=limit)
        self._send_json(
            200,
            {
                "entries": entries,
                "count": len(entries),
                "latest_seq": buf.latest_seq(),
            },
        )

    def _handle_debug_ping(self) -> None:
        """GET /v1/debug/ping — identify this service as LeaderProxy.

        Unlike /v1/debug/version (which llama-cpp-server could also serve),
        this endpoint is ONLY served by LeaderProxy. Peers use it to confirm
        the tunnel is routing to the proxy (port 8099), not directly to the
        upstream inference server (port 8100).

        Includes ``llm_ready`` field so peers can distinguish "proxy up" from
        "LLM model loaded and ready to serve inference."
        """
        llm_ready = self._probe_upstream_ready()
        self._send_json(
            200,
            {
                "service": "LeaderProxy",
                "proxy_port": self.server.server_address[1],
                "upstream": self.upstream_url,
                "auth_enabled": bool(self.api_key),
                "uptime_s": round(time.time() - self.start_time, 1),
                "llm_ready": llm_ready,
            },
        )

    def _probe_upstream_ready(self) -> bool:
        """Check if the upstream LLM server is responding to /v1/models.

        A fast probe (1.5s timeout) — returns True only if the upstream
        returns HTTP 200 or 401 (auth-gated but alive). Returns False on
        connection refused, timeout, or any error (model still loading).

        Sends the same API key the proxy uses — the upstream llama-cpp-server
        may have been started with --api_key, in which case unauthenticated
        probes get 401 and would never report ready.
        """
        from maxim.utils import http as _http

        probe = f"{self.upstream_url}/v1/models"
        headers: dict[str, str] = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            resp = _http.fetch_url(
                probe,
                method="GET",
                headers=headers,
                timeout=_http.TimeoutPolicy(connect_s=0.8, read_s=1.5, total_s=2.0),
            )
            return resp.status == 200
        except _http.HTTPAuthError:
            # 401 means the server is up but auth failed — still "ready"
            return True
        except Exception:
            return False

    def _handle_debug_last_requests(self) -> None:
        if self.request_log is None:
            self._send_json(200, {"total_requests": 0, "recent": []})
            return
        # Restrict to localhost for security
        peer = self.client_address[0]
        if peer not in ("127.0.0.1", "::1"):
            self._send_json(403, {"error": "localhost only"})
            return
        self._send_json(200, self.request_log.snapshot())

    def _handle_debug_vram(self) -> None:
        """GET /v1/debug/vram — VRAM projection and live nvidia-smi metrics.

        Returns the leader's current VRAM state as JSON. Calls
        :func:`project_vram_usage` from ``runtime/lane_models.py`` (the
        pure-data helper that both ``check_vram_pressure`` and
        ``_check_vram_spillover_risk`` already delegate to) and formats
        the ``VRAMProjection`` dataclass + live nvidia-smi ratio into a
        JSON response.

        503 if nvidia-smi is unavailable (not a GPU node).
        """
        gpu = _query_nvidia_smi()
        if gpu is None:
            self._send_json(
                503,
                {
                    "error": "nvidia-smi unavailable",
                    "fix": "This endpoint requires an NVIDIA GPU with nvidia-smi installed.",
                },
            )
            return

        # Import thresholds from the single source of truth in lane_models.
        from maxim.runtime.lane_models import (
            _SPILLOVER_RATIO,
            _SPILLOVER_WARN_RATIO,
            project_vram_usage,
        )

        vram_used_gb = float(gpu.get("vram_used_gb") or 0.0)
        vram_total_gb = float(gpu.get("vram_total_gb") or 0.0)
        ratio = round(vram_used_gb / vram_total_gb, 4) if vram_total_gb > 0 else 0.0

        live = {
            "vram_used_gb": vram_used_gb,
            "vram_total_gb": vram_total_gb,
            "vram_utilization_pct": gpu.get("utilization_pct"),
            "temperature_c": gpu.get("temperature_c"),
            "ratio": ratio,
            "spillover": ratio > _SPILLOVER_RATIO,
            "warning": ratio > _SPILLOVER_WARN_RATIO,
        }

        # Build projection from active model + running n_ctx.
        projection = None
        try:
            import maxim.runtime.llm_server as _srv

            active_profile = _srv._active_model
            if active_profile:
                # Extract port from upstream_url (always http://127.0.0.1:{port}).
                try:
                    from urllib.parse import urlparse

                    upstream_port = urlparse(self.upstream_url).port or DEFAULT_UPSTREAM_PORT
                except Exception:
                    upstream_port = DEFAULT_UPSTREAM_PORT
                running_n_ctx = _current_llama_server_n_ctx(upstream_port)
                if running_n_ctx and running_n_ctx > 0:
                    from maxim.models.language.config import _BUILTIN_PROFILES

                    profile_meta = _BUILTIN_PROFILES.get(active_profile, {})
                    vram_proj = project_vram_usage(
                        active_profile,
                        profile_meta,
                        running_n_ctx,
                        vram_total_gb,
                    )
                    if vram_proj is not None:
                        projection = {
                            "profile": vram_proj.profile,
                            "n_ctx": vram_proj.n_ctx,
                            "weights_gb": vram_proj.weights_gb,
                            "kv_cache_gb": round(vram_proj.kv_cache_gb, 3),
                            "headroom_gb": round(vram_proj.headroom_gb, 3),
                            "projected_total_gb": round(vram_proj.projected_total_gb, 3),
                            "physical_vram_gb": vram_proj.physical_vram_gb,
                            "spillover_risk": vram_proj.spillover_risk,
                            "recommended_n_ctx": vram_proj.recommended_n_ctx,
                        }
        except Exception:
            logger.debug("vram endpoint: projection unavailable", exc_info=True)

        self._send_json(
            200,
            {
                "live": live,
                "projection": projection,
                "thresholds": {
                    "spillover_ratio": _SPILLOVER_RATIO,
                    "warn_ratio": _SPILLOVER_WARN_RATIO,
                },
                "timestamp": time.time(),
            },
        )

    def _is_debug_path(self, path: str) -> bool:
        stripped = path.rstrip("/").split("?")[0]
        return stripped in (
            "/v1/debug/ping",
            "/v1/debug/status",
            "/v1/debug/heartbeat",
            "/v1/debug/metrics",
            "/v1/debug/version",
            "/v1/debug/logs",
            "/v1/debug/last-requests",
            "/v1/debug/vram",
            "/v1/debug/deps",
            "/v1/debug/install-status",
        )

    def _route_debug(self, path: str) -> None:
        stripped = path.rstrip("/").split("?")[0]
        if stripped == "/v1/debug/ping":
            self._handle_debug_ping()
        elif stripped == "/v1/debug/status":
            self._handle_debug_status()
        elif stripped == "/v1/debug/heartbeat":
            self._handle_debug_heartbeat()
        elif stripped == "/v1/debug/metrics":
            self._handle_debug_metrics()
        elif stripped == "/v1/debug/version":
            self._handle_debug_version()
        elif stripped == "/v1/debug/logs":
            self._handle_debug_logs()
        elif stripped == "/v1/debug/last-requests":
            self._handle_debug_last_requests()
        elif stripped == "/v1/debug/vram":
            self._handle_debug_vram()
        elif stripped == "/v1/debug/deps":
            self._handle_debug_deps()
        elif stripped == "/v1/debug/install-status":
            self._send_json(200, dict(_install_status))
        else:
            self._send_json(404, {"error": "Not found"})

    # ─── reverse proxy ────────────────────────────────────────────────

    def _proxy_request(self, method: str) -> None:
        """Forward the request to the upstream llama-cpp-server."""
        from maxim.utils import http as _http

        request_id = self.headers.get("X-Maxim-Request-Id", "")
        peer_ip = self.client_address[0]
        t0 = time.time()

        # Read request body
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else None

        # llm_timeout_scalability.md follow-up: context-overflow admission.
        # Reject prompts that would exceed the upstream model's context
        # window with a typed 413 BEFORE forwarding. Without this gate,
        # oversize prompts cause llama-cpp-server to silently abort
        # mid-generation, producing a stream that limps along with
        # keepalives until upstream gives up — terrible debugging UX and
        # wasted upstream compute on a guaranteed-to-fail request.
        #
        # The gate is OFF when MAXIM_LLM_N_CTX is unset (graceful default
        # for existing setups) or when MAXIM_PROXY_CONTEXT_ADMISSION=0
        # (explicit opt-out).
        if body is not None and self.path in _PROXY_INFERENCE_PATHS and _is_admission_enabled():
            n_ctx = _get_proxy_context_window()
            if n_ctx is not None:
                overhead = _resolve_context_overhead_tokens()
                rejection = _check_context_admission(body, n_ctx, overhead)
                if rejection is not None:
                    elapsed_ms = (time.time() - t0) * 1000
                    err = rejection["error"]
                    logger.warning(
                        "Rejected oversize request: req=%s peer=%s path=%s prompt_tokens=%d max_tokens=%d ctx=%d",
                        request_id[:8] if request_id else "none",
                        peer_ip,
                        self.path,
                        err["estimated_prompt_tokens"],
                        err["max_tokens"],
                        n_ctx,
                    )
                    body_bytes = json.dumps(rejection).encode()
                    self.send_response(413)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("X-Maxim-Proxy-Ms", f"{elapsed_ms:.0f}")
                    self.send_header("Content-Length", str(len(body_bytes)))
                    self.end_headers()
                    self.wfile.write(body_bytes)
                    self.wfile.flush()
                    return

        # Build upstream request — forward client headers verbatim
        upstream = f"{self.upstream_url}{self.path}"

        # Forward headers (skip hop-by-hop)
        skip_headers = {
            "host",
            "connection",
            "transfer-encoding",
            "proxy-connection",
            "keep-alive",
            "content-length",  # httpx sets this from the body automatically
        }
        forwarded_headers: dict[str, str] = {}
        for key, val in self.headers.items():
            if key.lower() not in skip_headers:
                forwarded_headers[key] = val

        # Forward to upstream — use raw_proxy_forward_streaming to stream
        # chunks as they arrive from llama-cpp-server.  This prevents
        # Cloudflare 524 timeouts: the edge sees the first byte of the
        # response as soon as the model starts generating (~0.5s), rather
        # than waiting for the full 105-125s inference to complete before
        # any data is sent downstream.  protocol_version="HTTP/1.1" (set
        # on the class) enables Transfer-Encoding: chunked on this path.
        #
        # For connection/timeout failures the stream never opens — fall
        # through to the buffered 502 path instead.
        resp_code = 502
        resp_headers: dict[str, str] = {}
        resp_body = b""
        server_ms: float | None = None
        gpu: dict | None = None
        elapsed_ms: float = 0.0

        stream: "_http.StreamingResponse | None" = None
        # Stage 3 keepalive state — declared at the same scope as
        # ``stream`` so the ``finally`` block can stop the emitter on
        # every exit path, including upstream errors that occur before
        # the SSE check.
        keepalive_thread: threading.Thread | None = None
        keepalive_stop = threading.Event()
        write_lock = threading.Lock()
        try:
            stream = _http.raw_proxy_forward_streaming(
                upstream,
                method,
                headers=forwarded_headers,
                body=body,
                timeout=_INFERENCE_PROXY_TIMEOUT_S,
            )
            resp_code = stream.status
            resp_headers = dict(stream.headers)
            server_ms_raw = resp_headers.get("openai-processing-ms") or resp_headers.get("Openai-Processing-Ms")
            if server_ms_raw:
                try:
                    server_ms = float(server_ms_raw)
                except (ValueError, TypeError):
                    pass

            elapsed_ms = (time.time() - t0) * 1000

            # Headers are available immediately (before body) — send them
            # now so Cloudflare and the peer know the response has started.
            self.send_response(resp_code)
            # Forward upstream headers; strip hop-by-hop and content-length
            # (we're using chunked encoding so content-length is invalid).
            skip_resp_headers = {"transfer-encoding", "connection", "keep-alive", "content-length"}
            for key, val in resp_headers.items():
                if key.lower() not in skip_resp_headers:
                    self.send_header(key, val)
            # Inject Maxim trace headers
            self.send_header("X-Maxim-Proxy-Ms", f"{elapsed_ms:.0f}")
            self.send_header("X-Maxim-Queue-Depth", f"{self._queue_depth()}")
            if server_ms is not None:
                self.send_header("X-Maxim-Server-Ms", f"{server_ms:.0f}")
            gpu = _query_nvidia_smi()
            if gpu is not None:
                self.send_header("X-Maxim-GPU-Util", f"{gpu['utilization_pct']:.0f}")
                self.send_header("X-Maxim-GPU-VRAM", f"{gpu['vram_used_gb']:.1f}/{gpu['vram_total_gb']:.0f}")
                if gpu.get("temperature_c") is not None:
                    self.send_header("X-Maxim-GPU-Temp", f"{gpu['temperature_c']:.0f}")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()

            # Stage 3 (llm_timeout_scalability.md): start the TTFT
            # keepalive emitter if upstream responded with an SSE stream.
            # During the silent window between response headers and the
            # first generated token, the emitter writes ``: keepalive``
            # comment frames every ~30s to defeat cloudflared's ~100s
            # tunnel idle timeout. The stop_event is set on the first
            # real chunk (or any error path) so the daemon thread exits
            # cleanly. ``write_lock`` serialises the emitter and the
            # main chunk writer because both target ``self.wfile``.
            if _is_event_stream_response(resp_headers):
                keepalive_thread = threading.Thread(
                    target=_run_keepalive_emitter,
                    args=(
                        self.wfile,
                        write_lock,
                        keepalive_stop,
                        _resolve_keepalive_interval_s(),
                    ),
                    daemon=True,
                    name="proxy-keepalive",
                )
                keepalive_thread.start()

            # Stream body — write each chunk in HTTP/1.1 chunked format so
            # the peer receives tokens progressively.  Accumulate locally
            # for post-send usage logging (body is at most a few KB for
            # non-streaming chat completions).
            body_chunks: list[bytes] = []
            try:
                for chunk in stream.iter_bytes(chunk_size=16384):
                    if chunk:
                        # First non-empty chunk: stop the keepalive
                        # emitter BEFORE acquiring the write lock so the
                        # emitter sees the stop signal on its next
                        # wakeup. Order matters — if we acquired the lock
                        # first, the emitter could already be parked at
                        # ``with write_lock`` and would emit one more
                        # keepalive after our first real chunk.
                        keepalive_stop.set()
                        with write_lock:
                            body_chunks.append(chunk)
                            self.wfile.write(f"{len(chunk):x}\r\n".encode())
                            self.wfile.write(chunk)
                            self.wfile.write(b"\r\n")
                            self.wfile.flush()
            except Exception as chunk_err:
                # Mid-stream failure: log with request context so operators
                # can correlate with the peer-side dispatch_exhausted and
                # distinguish upstream crashes from client disconnects.
                logger.warning(
                    "Upstream stream interrupted (req=%s peer=%s path=%s chunks=%d elapsed_ms=%.0f): %s",
                    request_id[:8] if request_id else "none",
                    peer_ip,
                    self.path,
                    len(body_chunks),
                    (time.time() - t0) * 1000,
                    chunk_err,
                )
            # HTTP/1.1 chunked terminator
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
            resp_body = b"".join(body_chunks)

        except _http.HTTPError as e:
            # Typed HTTP failure from httpx (connection error, timeout, etc.)
            # — stream never opened, return a 502 with our error body.
            err_type = type(e).__name__
            logger.warning("Upstream error: %s: %s", err_type, e.fix_hint)
            resp_code = 502
            resp_body = json.dumps(
                {
                    "error": "Upstream connection failed — LLM server may be restarting",
                    "retry_after_s": 5,
                }
            ).encode()
            resp_headers = {}
        except Exception as e:
            # Anything else — log single line (no traceback for transient
            # connect errors during server respawn).
            err_type = type(e).__name__
            err_msg = str(e)
            if "Connection refused" in err_msg or "Connection reset" in err_msg:
                logger.warning("Upstream unavailable: %s (server may be restarting)", err_type)
            else:
                logger.warning("Upstream connection error: %s: %s", err_type, err_msg)
            resp_code = 502
            resp_body = json.dumps(
                {
                    "error": "Upstream connection failed — LLM server may be restarting",
                    "retry_after_s": 5,
                }
            ).encode()
            resp_headers = {}
        finally:
            # Stop the keepalive emitter on every exit path. Setting the
            # event is sufficient — the daemon thread is a no-op once
            # the flag flips. Join with a short timeout so a stuck
            # emitter (e.g., blocked in a slow ``flush``) can't pin the
            # proxy worker; daemon=True lets it die with the process if
            # the join times out.
            keepalive_stop.set()
            if keepalive_thread is not None:
                keepalive_thread.join(timeout=1.0)
            if stream is not None:
                stream.close()

        # Error paths: stream never opened — send buffered 502 response.
        if stream is None:
            elapsed_ms = (time.time() - t0) * 1000
            gpu = _query_nvidia_smi()
            self.send_response(resp_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("X-Maxim-Proxy-Ms", f"{elapsed_ms:.0f}")
            self.send_header("X-Maxim-Queue-Depth", f"{self._queue_depth()}")
            if gpu is not None:
                self.send_header("X-Maxim-GPU-Util", f"{gpu['utilization_pct']:.0f}")
            self.send_header("Content-Length", str(len(resp_body)))
            self.end_headers()
            self.wfile.write(resp_body)

        # Extract token counts from response body
        input_tokens = 0
        output_tokens = 0
        cached_tokens = 0
        model = ""
        try:
            if resp_code == 200 and resp_body:
                data = json.loads(resp_body)
                usage = data.get("usage", {}) or {}
                input_tokens = usage.get("prompt_tokens", 0)
                output_tokens = usage.get("completion_tokens", 0)
                # OpenAI-compatible servers expose cached portion under
                # usage.prompt_tokens_details.cached_tokens. Surface it
                # under the public-contract name (CC12) so JSONL
                # consumers don't have to peek into nested
                # OpenAI-specific shapes.
                details = usage.get("prompt_tokens_details") or {}
                if isinstance(details, dict):
                    cached_tokens = int(details.get("cached_tokens") or 0)
                model = data.get("model", "")
        except Exception:
            pass

        # Log the request
        log_entry = {
            "request_id": request_id,
            "peer_ip": peer_ip,
            "method": method,
            "path": self.path,
            "model": model,
            "status": resp_code,
            "latency_ms": round(elapsed_ms, 1),
            "server_ms": round(server_ms, 1) if server_ms is not None else None,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            # CC12: standard-name token telemetry on JSONL emissions.
            "cached_tokens": cached_tokens,
            "gpu": gpu,
            "timestamp": time.time(),
        }

        if self.request_log is not None:
            self.request_log.record(log_entry)

        # Phase 8: record into LaneMetrics for doctor / admission control
        if self.lane_metrics is not None and "/chat/completions" in self.path:
            self.lane_metrics.record_call(
                elapsed_ms,
                success=(resp_code == 200),
                kind="remote",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

        # Structured log line
        parts = [
            f"req={request_id[:8]}" if request_id else "req=none",
            f"peer={peer_ip}",
            f"{method} {self.path}",
            f"status={resp_code}",
            f"latency={elapsed_ms:.0f}ms",
        ]
        if server_ms is not None:
            parts.append(f"server={server_ms:.0f}ms")
        if input_tokens or output_tokens:
            parts.append(f"tokens={input_tokens}+{output_tokens}")
        if gpu:
            parts.append(f"gpu={gpu['utilization_pct']:.0f}%")
            parts.append(f"vram={gpu['vram_used_gb']:.1f}/{gpu['vram_total_gb']:.0f}G")
        logger.info("proxy: %s", " ".join(parts))

    # ─── admin endpoints ──────────────────────────────────────────────

    def _handle_admin_update(self) -> None:
        """POST /v1/admin/update — update the leader via git or pip.

        Requires MAXIM_ALLOW_REMOTE_UPDATE=1 on the leader process.
        Accepts JSON body::

            {"mode": "auto", "branch": "main", "dry_run": true,
             "force": false, "version": null}

        Mode resolution:
        - ``"auto"`` — detect install mode (git checkout → dev, else pip)
        - ``"pip"``  — force pip upgrade path
        - ``"dev"``  — force git pull path (409 if no ``.git`` dir)

        dry_run defaults to True (safe-by-default).
        """
        if not _remote_update_allowed():
            self._send_json(
                403,
                {"error": "Remote update disabled. Set MAXIM_ALLOW_REMOTE_UPDATE=1 on the leader."},
            )
            return

        body = self._parse_admin_update_body()
        if body is None:
            return  # error response already sent

        branch = body["branch"]
        dry_run = body["dry_run"]
        force = body["force"]
        mode = body["mode"]

        # ── Mode resolution ──────────────────────────────────────────────
        resolved_mode = mode if mode != "auto" else _detect_install_mode()

        repo_root = str(Path(__file__).resolve().parents[3])

        if resolved_mode == "dev" and not (Path(repo_root) / ".git").is_dir():
            self._send_json(
                409,
                {
                    "error": "No git repository found on leader.",
                    "fix": "Clone the repo to use --dev mode, or omit --dev to update via pip.",
                },
            )
            return

        logger.info(
            "admin/update: peer=%s mode=%s (resolved=%s) branch=%s dry_run=%s force=%s",
            self.client_address[0],
            mode,
            resolved_mode,
            branch,
            dry_run,
            force,
        )

        if resolved_mode == "pip":
            self._handle_pip_update(body)
            return

        # ── Dev (git) path — unchanged from pre-plan behavior ────────────
        # 1. Stash dirty tree if --force; bail with 409 otherwise
        stashed = self._stash_if_dirty(repo_root, force)
        if stashed is None:
            return  # error response already sent

        # 2. Fetch latest from origin
        if not self._fetch_branch(repo_root, branch):
            return  # error response already sent

        # 3. Compute pending commits + handle dry-run / up-to-date paths
        pending = self._list_pending_commits(repo_root, branch)
        if not pending:
            self._send_json(
                200,
                {
                    "status": "up_to_date",
                    "install_mode": "dev",
                    "message": f"Already up to date with origin/{branch}.",
                    "commits": [],
                },
            )
            return
        if dry_run:
            self._send_json(
                200,
                {
                    "status": "preview",
                    "install_mode": "dev",
                    "branch": branch,
                    "pending_commits": pending,
                    "message": f"{len(pending)} commit(s) pending. Send dry_run=false to apply.",
                },
            )
            return

        # 4. Apply git pull (with stash restore on failure)
        if not self._apply_git_pull(repo_root, branch, stashed):
            return  # error response already sent

        if stashed:
            subprocess.run(
                ["git", "stash", "pop"],
                capture_output=True,
                timeout=10,
                cwd=repo_root,
            )
            logger.info("admin/update: restored stashed changes")

        # 5. pip install + rollback on failure
        pip_output = self._run_pip_install(repo_root)
        if pip_output is None:
            return  # error response already sent (with rollback status)

        # 6. Record + reply success
        if self.request_log is not None:
            self.request_log.record(
                {
                    "type": "admin_update",
                    "peer_ip": self.client_address[0],
                    "branch": branch,
                    "commits_applied": pending,
                    "timestamp": time.time(),
                }
            )
        logger.info("admin/update: SUCCESS — %d commits applied from origin/%s", len(pending), branch)
        self._send_json(
            200,
            {
                "status": "updated",
                "install_mode": "dev",
                "branch": branch,
                "commits_applied": pending,
                "pip_output": pip_output,
                "message": f"Applied {len(pending)} commit(s). Restart maxim to load new code.",
            },
        )

    def _parse_admin_update_body(self) -> dict[str, Any] | None:
        """Parse the JSON body for ``/v1/admin/update``.

        Returns the full parsed dict (with validated/defaulted fields) or
        ``None`` if an error response has already been sent.

        Parsed fields::

            branch   (str)  — validated git branch name, default ``"main"``
            dry_run  (bool) — default ``True`` (safe-by-default)
            force    (bool) — default ``False``
            mode     (str)  — ``"auto"`` | ``"pip"`` | ``"dev"``, default ``"auto"``
            version  (str|None) — pinned PyPI version, default ``None``
        """
        raw = self._read_body(max_size=4096)
        body: dict[str, Any] = {}
        if raw:
            try:
                body = json.loads(raw)
            except Exception as e:
                logger.debug("admin/update: ignoring malformed JSON body: %s", e)

        raw_branch = body.get("branch", "main")
        try:
            branch = _validate_branch(raw_branch)
        except ValueError as e:
            self._send_json(400, {"error": f"Invalid branch: {e}"})
            return None

        # Mode validation
        mode = body.get("mode", "auto")
        if mode not in ("auto", "pip", "dev"):
            self._send_json(400, {"error": f"Invalid mode: {mode!r}. Must be auto, pip, or dev."})
            return None

        # Version validation (pip mode only)
        version = body.get("version") or None
        if version is not None:
            version = str(version).strip()
            if not _VERSION_RE.match(version):
                self._send_json(
                    400,
                    {
                        "error": f"Invalid version: {version!r}. Expected format: X.Y.Z or X.Y.Zsuffix.",
                    },
                )
                return None

        return {
            "branch": branch,
            "dry_run": bool(body.get("dry_run", True)),
            "force": bool(body.get("force", False)),
            "mode": mode,
            "version": version,
        }

    def _stash_if_dirty(self, repo_root: str, force: bool) -> bool | None:
        """Stash dirty working tree under ``--force``; bail with 409 otherwise.

        Returns ``True`` if a stash was created, ``False`` if the tree was
        clean, or ``None`` if a 4xx/5xx error response has already been sent.
        """
        try:
            status = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=repo_root,
            )
        except Exception as e:
            logger.exception("git status failed: %s", e)
            self._send_json(500, {"error": "git status check failed"})
            return None

        if not status.stdout.strip():
            return False

        if not force:
            self._send_json(
                409,
                {
                    "error": "Working tree is dirty. Commit or stash changes first.",
                    "hint": "Use: maxim peer update --force (stashes and restores automatically)",
                    "dirty_files": status.stdout.strip().split("\n"),
                },
            )
            return None

        stash_result = subprocess.run(
            ["git", "stash", "--include-untracked"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=repo_root,
        )
        stashed = stash_result.returncode == 0 and bool(stash_result.stdout.strip())
        logger.info("admin/update: stashed dirty tree (stashed=%s)", stashed)
        return stashed

    def _fetch_branch(self, repo_root: str, branch: str) -> bool:
        """``git fetch origin <branch>``. Returns False after sending a 500."""
        try:
            subprocess.run(
                ["git", "fetch", "origin", branch],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=repo_root,
                check=True,
            )
            return True
        except Exception as e:
            logger.exception("git fetch failed: %s", e)
            self._send_json(500, {"error": "git fetch failed"})
            return False

    @staticmethod
    def _list_pending_commits(repo_root: str, branch: str) -> list[str]:
        """Return the list of ``HEAD..origin/<branch>`` commit lines (oneline)."""
        try:
            log_result = subprocess.run(
                ["git", "log", f"HEAD..origin/{branch}", "--oneline"],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=repo_root,
            )
            return log_result.stdout.strip().split("\n") if log_result.stdout.strip() else []
        except Exception as e:
            logger.debug("git log HEAD..origin/%s failed: %s", branch, e)
            return []

    def _apply_git_pull(self, repo_root: str, branch: str, stashed: bool) -> bool:
        """``git pull --rebase``. Restores stash and sends 500 on failure."""
        try:
            subprocess.run(
                ["git", "-c", "pull.rebase=true", "pull", "origin", branch],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=repo_root,
                check=True,
            )
            return True
        except subprocess.CalledProcessError as e:
            if stashed:
                subprocess.run(
                    ["git", "stash", "pop"],
                    capture_output=True,
                    timeout=10,
                    cwd=repo_root,
                )
            self._send_json(
                500,
                {
                    "error": "git pull failed",
                    "stdout": _sanitize_git_output(e.stdout),
                    "stderr": _sanitize_git_output(e.stderr),
                },
            )
            return False

    def _run_pip_install(self, repo_root: str) -> str | None:
        """Run ``pip install -e .`` and roll back on failure.

        Returns the (truncated) pip stdout on success, or ``None`` if an
        error response has already been sent (with rollback status).
        """
        try:
            pip_result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-e", "."],
                capture_output=True,
                text=True,
                timeout=120,
                cwd=repo_root,
            )
        except Exception as e:
            logger.exception("pip install failed: %s", e)
            self._send_json(500, {"error": "pip install failed"})
            return None

        pip_output = pip_result.stdout[-500:] if pip_result.stdout else ""
        if pip_result.returncode == 0:
            return pip_output

        # Rollback: revert git + reinstall
        rollback = subprocess.run(
            ["git", "checkout", "HEAD~1"],
            capture_output=True,
            timeout=10,
            cwd=repo_root,
        )
        rollback_ok = rollback.returncode == 0
        reinstall = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", "."],
            capture_output=True,
            timeout=120,
            cwd=repo_root,
        )
        reinstall_ok = reinstall.returncode == 0
        if not rollback_ok or not reinstall_ok:
            logger.error(
                "admin/update: ROLLBACK INCOMPLETE — git=%s pip=%s",
                "ok" if rollback_ok else "FAILED",
                "ok" if reinstall_ok else "FAILED",
            )
        rollback_status = "complete" if (rollback_ok and reinstall_ok) else "INCOMPLETE"
        self._send_json(
            500,
            {
                "error": f"pip install failed, rollback {rollback_status}",
                "pip_stderr": _sanitize_git_output(pip_result.stderr),
                "rollback_git": "ok" if rollback_ok else "failed",
                "rollback_pip": "ok" if reinstall_ok else "failed",
            },
        )
        return None

    # ── Pip update path (peer_update_pip_mode plan) ────────────────────────

    def _handle_pip_update(self, body: dict[str, Any]) -> None:
        """Orchestrate a PyPI-based update. Called from _handle_admin_update
        when resolved mode is ``"pip"``.
        """
        version = body.get("version")
        dry_run = body.get("dry_run", True)

        old_version = _get_current_version()
        extras = _detect_installed_extras()

        if dry_run:
            latest = _get_latest_pypi_version()
            if latest is not None and latest == old_version:
                self._send_json(
                    200,
                    {
                        "status": "up_to_date",
                        "install_mode": "pip",
                        "current_version": old_version,
                        "message": "Already at latest version.",
                    },
                )
            else:
                display_latest = latest or "?"
                self._send_json(
                    200,
                    {
                        "status": "preview",
                        "install_mode": "pip",
                        "current_version": old_version,
                        "latest_version": latest,
                        "extras_detected": extras,
                        # Synthetic entry so old clients that read pending_commits
                        # display something informative instead of "0 pending".
                        "pending_commits": [f"{old_version} \u2192 {display_latest}"],
                        "message": f"{old_version} \u2192 {display_latest} available. Send dry_run=false to apply.",
                    },
                )
            return

        pip_output = self._run_pip_upgrade(version, extras)
        if pip_output is None:
            return  # error response already sent

        new_version = _get_current_version()
        if new_version == old_version:
            self._send_json(
                200,
                {
                    "status": "up_to_date",
                    "install_mode": "pip",
                    "current_version": old_version,
                    "message": "Already at latest version.",
                },
            )
        else:
            if self.request_log is not None:
                self.request_log.record(
                    {
                        "type": "admin_update",
                        "install_mode": "pip",
                        "peer_ip": self.client_address[0],
                        "from_version": old_version,
                        "to_version": new_version,
                        "timestamp": time.time(),
                    }
                )
            logger.info(
                "admin/update(pip): SUCCESS — %s → %s, extras=%s",
                old_version,
                new_version,
                extras,
            )
            self._send_json(
                200,
                {
                    "status": "updated",
                    "install_mode": "pip",
                    "from_version": old_version,
                    "to_version": new_version,
                    "extras_preserved": extras,
                    "message": f"Updated {old_version} \u2192 {new_version}. Restart maxim to load new code.",
                },
            )

    def _run_pip_upgrade(self, version: str | None, extras: list[str]) -> str | None:
        """Run ``pip install --upgrade pymaxim[...]`` with rollback on failure.

        Returns the (truncated) pip stdout on success, or ``None`` if an
        error response has already been sent.
        """
        old_version = _get_current_version()

        # ── Pre-flight: disk space check ─────────────────────────────────
        has_heavy = bool({"llm-torch", "vision", "yolo"} & set(extras))
        min_gb = 6.0 if has_heavy else 1.0
        try:
            usage = shutil.disk_usage(sys.prefix)
            free_gb = usage.free / (1 << 30)
            if free_gb < min_gb:
                self._send_json(
                    507,
                    {
                        "error": f"Insufficient disk space: {free_gb:.1f} GB free, need ~{min_gb:.0f} GB.",
                        "fix": "Free space or remove unused models with: maxim --delete-model <name>",
                    },
                )
                return None
        except OSError:
            pass  # non-fatal — proceed without space check

        # ── Pre-flight: cache old version for network-independent rollback
        rollback_dir = Path(tempfile.mkdtemp(prefix="maxim-rollback-"))
        rollback_spec = f"pymaxim=={old_version}"
        if extras:
            rollback_spec += f"[{','.join(extras)}]"
        try:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "download",
                    rollback_spec,
                    "-d",
                    str(rollback_dir),
                    "--index-url",
                    "https://pypi.org/simple/",
                ],
                capture_output=True,
                timeout=120,
            )
        except Exception:
            logger.debug("admin/update(pip): rollback pre-cache failed (non-fatal)")

        # ── Upgrade ──────────────────────────────────────────────────────
        spec = "pymaxim"
        if extras:
            spec += f"[{','.join(extras)}]"
        if version:
            spec += f"=={version}"

        cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--index-url",
            "https://pypi.org/simple/",
            spec,
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        except subprocess.TimeoutExpired:
            logger.error("admin/update(pip): pip upgrade timed out (600s)")
            self._send_json(
                500,
                {
                    "error": "pip upgrade timed out (600s). Environment may be inconsistent.",
                    "fix": "Check `pip show pymaxim` on the leader and restart if needed.",
                },
            )
            shutil.rmtree(rollback_dir, ignore_errors=True)
            return None
        except Exception as e:
            logger.exception("admin/update(pip): pip upgrade failed: %s", e)
            self._send_json(500, {"error": f"pip upgrade failed: {e}"})
            shutil.rmtree(rollback_dir, ignore_errors=True)
            return None

        pip_output = result.stdout[-500:] if result.stdout else ""
        if result.returncode == 0:
            shutil.rmtree(rollback_dir, ignore_errors=True)
            return pip_output

        # ── Rollback from local cache (network-independent) ──────────────
        logger.warning("admin/update(pip): upgrade failed (rc=%d), rolling back to %s", result.returncode, old_version)
        try:
            rollback = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--no-index",
                    "--find-links",
                    str(rollback_dir),
                    rollback_spec,
                ],
                capture_output=True,
                timeout=120,
            )
            rollback_ok = rollback.returncode == 0
        except Exception:
            rollback_ok = False

        shutil.rmtree(rollback_dir, ignore_errors=True)
        rollback_status = "complete" if rollback_ok else "INCOMPLETE"
        if not rollback_ok:
            logger.error("admin/update(pip): ROLLBACK INCOMPLETE — old version %s", old_version)
        self._send_json(
            500,
            {
                "error": f"pip upgrade failed, rollback {rollback_status}",
                "stderr": (result.stderr or "")[-500:],
            },
        )
        return None

    def _handle_admin_install(self) -> None:
        """POST /v1/admin/install — install packages on the leader.

        Requires MAXIM_ALLOW_REMOTE_UPDATE=1 on the leader process.
        Accepts JSON body: {"extras": ["semantic", "llm-torch"], "packages": ["some-pkg"]}
        Extras are installed as pymaxim[extra], raw packages via pip install.
        """
        if not _remote_update_allowed():
            self._send_json(
                403,
                {"error": "Remote install disabled. Set MAXIM_ALLOW_REMOTE_UPDATE=1 on the leader."},
            )
            return

        try:
            content_length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(content_length) if content_length else b"{}"
            params = json.loads(raw)
        except Exception as e:
            self._send_json(400, {"error": f"Invalid JSON body: {e}"})
            return

        extras = params.get("extras", [])
        packages = params.get("packages", [])

        if not extras and not packages:
            self._send_json(400, {"error": "No extras or packages specified."})
            return

        # ── Input validation — allowlist-only to prevent command injection ──
        # Uses module-level _ALLOWED_EXTRAS (single source of truth).
        # Strict pattern: alphanumeric, hyphens, dots, underscores only.
        # Rejects shell metacharacters, paths, URLs, version specifiers.
        _SAFE_PKG_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,100}$")

        for extra in extras:
            if extra not in _ALLOWED_EXTRAS:
                self._send_json(
                    400,
                    {"error": f"Unknown extra: {extra!r}. Allowed: {sorted(_ALLOWED_EXTRAS)}"},
                )
                return

        for pkg in packages:
            if not _SAFE_PKG_RE.match(pkg):
                self._send_json(
                    400,
                    {"error": f"Invalid package name: {pkg!r}. Only alphanumeric, hyphens, dots, underscores allowed."},
                )
                return

        # ── Build pip install commands ────────────────────────────────────
        # SECURITY: rebuild values from allowlist lookups — never pass the
        # original user strings to subprocess. This severs the dataflow
        # chain that static analyzers trace from request body to exec.
        repo_root = str(Path(__file__).resolve().parents[3])
        installed: list[str] = []
        pip_cmds: list[list[str]] = []

        # Rebuild extras from the allowlist (not from user input)
        safe_extras = [e for e in _ALLOWED_EXTRAS if e in extras]
        if safe_extras:
            extra_str = ",".join(safe_extras)
            spec = f".[{extra_str}]"
            pip_cmds.append([sys.executable, "-m", "pip", "install", spec])
            installed.extend(f"pymaxim[{e}]" for e in safe_extras)

        # Rebuild package names via regex match groups (severs taint chain)
        safe_packages: list[str] = []
        for pkg in packages:
            m = _SAFE_PKG_RE.match(pkg)
            if m:
                safe_packages.append(m.group(0))
        if safe_packages:
            pip_cmds.append([sys.executable, "-m", "pip", "install", *safe_packages])
            installed.extend(safe_packages)

        # Respond immediately — pip runs in a background thread to avoid
        # Cloudflare tunnel timeouts (524) on long-running installs.
        # Peer polls /v1/debug/install-status for completion.
        def _run_pip_background() -> None:
            for cmd in pip_cmds:
                try:
                    r = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=600,
                        cwd=repo_root,
                    )
                    _install_status["pip_output"] += (r.stdout or "")[-500:]
                    if r.returncode != 0:
                        _install_status["status"] = "error"
                        _install_status["error"] = r.stderr[-500:] if r.stderr else "pip failed"
                        return
                except Exception as exc:
                    _install_status["status"] = "error"
                    _install_status["error"] = str(exc)
                    return
            _install_status["status"] = "ok"

        # Store status in the module-level dict so the debug endpoint can read it
        _install_status.update(
            {
                "status": "running",
                "installed": installed,
                "pip_output": "",
                "error": "",
                "started_by": self.client_address[0],
            }
        )

        logger.info("admin/install: %s from peer %s (async)", installed, self.client_address[0])
        t = threading.Thread(target=_run_pip_background, name="admin.install", daemon=True)
        t.start()

        self._send_json(
            202,
            {
                "status": "started",
                "installed": installed,
                "message": "Install started. Poll /v1/debug/install-status for progress.",
            },
        )

    def _handle_debug_deps(self) -> None:
        """GET /v1/debug/deps — show installed packages relevant to maxim."""
        # Check which optional extras are importable (uses shared map).
        extras: dict[str, bool] = {}
        for extra_name, import_name in _EXTRA_IMPORT_MAP.items():
            extras[extra_name] = _try_import(import_name)

        # Key packages with versions
        key_packages = [
            "numpy",
            "scipy",
            "pyyaml",
            "torch",
            "sentence-transformers",
            "anthropic",
            "openai",
            "open-clip-torch",
            "opencv-python",
            "pymaxim",
        ]
        packages: dict[str, str] = {}
        try:
            from importlib.metadata import version as pkg_version

            for pkg in key_packages:
                try:
                    packages[pkg] = pkg_version(pkg)
                except Exception:
                    pass
        except ImportError:
            pass

        self._send_json(200, {"extras": extras, "packages": packages})

    def _handle_admin_restart(self) -> None:
        """POST /v1/admin/restart — soft-restart the maxim process via os.execv.

        Replaces the current process image with a fresh Python invocation,
        reloading all code. Same PID, clean import cycle. Gated by
        MAXIM_ALLOW_REMOTE_UPDATE (if you trust remote code pulls, you
        trust remote restarts).

        Accepts JSON body: {"delay_s": 1.5} (default 1.5s delay so the
        HTTP response reaches the peer before the process replaces itself).
        """
        if os.environ.get("MAXIM_ALLOW_REMOTE_UPDATE", "").strip().lower() not in (
            "1",
            "true",
            "t",
            "yes",
            "y",
            "on",
        ):
            self._send_json(
                403,
                {
                    "error": "Remote restart disabled. Set MAXIM_ALLOW_REMOTE_UPDATE=1 on the leader.",
                },
            )
            return

        # Parse optional delay from body
        raw = self._read_body(max_size=4096)
        body: dict[str, Any] = {}
        if raw:
            try:
                body = json.loads(raw)
            except Exception:
                pass

        delay_s = max(0.5, min(float(body.get("delay_s", 1.5)), 10.0))

        uptime = round(time.time() - self.start_time, 1)
        logger.info(
            "admin/restart: peer=%s delay=%.1fs uptime=%.1fs",
            self.client_address[0],
            delay_s,
            uptime,
        )

        # Send response BEFORE restarting — the peer needs to receive this
        self._send_json(
            200,
            {
                "status": "restarting",
                "message": f"Restart initiated (delay {delay_s}s). Process will reload.",
                "uptime_s": uptime,
            },
        )

        # Schedule the restart on a background thread so the response
        # is fully flushed before os.execv replaces the process image.
        def _do_restart() -> None:
            time.sleep(delay_s)
            # Kill LLM server + cloudflared before replacing process image.
            # os.execv() does NOT fire atexit handlers, so we must clean up
            # child processes explicitly to prevent ghost servers.
            try:
                from maxim.runtime.lane_backends import stop_active_spawner

                stop_active_spawner()
                logger.info("admin/restart: LLM server stopped")
            except Exception as e:
                logger.warning("admin/restart: failed to stop LLM server: %s", e)
            try:
                from maxim.runtime.local_server_spawner import kill_stale_llm_servers

                kill_stale_llm_servers()
                logger.info("admin/restart: cleaned stale servers")
            except Exception:
                pass
            logger.info("admin/restart: executing os.execv — goodbye")
            # Re-exec with the original command line
            os.execv(sys.executable, [sys.executable] + sys.argv)

        t = threading.Thread(target=_do_restart, name="admin.restart", daemon=True)
        t.start()

    def _handle_admin_llm_swap(self) -> None:
        """POST /v1/admin/llm-swap — hot-swap the LLM model.

        Stops the current llama-cpp-server, resolves the new profile to a
        GGUF path, and starts a fresh server.  Does NOT restart the Maxim
        process — LeaderProxy stays alive throughout.

        Body: {"model": "qwen2.5-14b"}
        """
        if os.environ.get("MAXIM_ALLOW_REMOTE_UPDATE", "").strip().lower() not in (
            "1",
            "true",
            "t",
            "yes",
            "y",
            "on",
        ):
            self._send_json(
                403,
                {"error": "Remote LLM swap disabled. Set MAXIM_ALLOW_REMOTE_UPDATE=1 on the leader."},
            )
            return

        raw = self._read_body(max_size=4096)  # Admin body should be tiny
        body: dict[str, Any] = {}
        if raw:
            try:
                body = json.loads(raw)
            except Exception:
                self._send_json(400, {"error": "Invalid JSON body"})
                return

        model = str(body.get("model", "")).strip()
        if not model:
            self._send_json(400, {"error": "Missing 'model' field in request body"})
            return

        logger.info("admin/llm-swap: peer=%s model=%s", self.client_address[0], model)

        try:
            from maxim.runtime.lane_backends import swap_llm_server

            result = swap_llm_server(model, logger=logger)
            self._send_json(200, result)
        except ValueError as e:
            self._send_json(400, {"error": str(e)})
        except FileNotFoundError as e:
            msg = str(e)
            parts = msg.split("|", 1)
            # Sanitize — don't leak file paths in error responses
            safe_error = re.sub(r"/[\w./-]{5,}", "<path>", parts[0])
            resp: dict[str, Any] = {"error": safe_error}
            if len(parts) > 1:
                resp["hint"] = re.sub(r"/[\w./-]{5,}", "<path>", parts[1])
            self._send_json(404, resp)
        except RuntimeError as e:
            if "already in progress" in str(e):
                self._send_json(409, {"error": "LLM swap already in progress"})
            else:
                logger.error("LLM swap failed: %s", e)
                self._send_json(500, {"error": "LLM swap failed. Check server logs."})

    # ─── HTTP method dispatchers ──────────────────────────────────────

    def _is_localhost(self) -> bool:
        """Check if the request originates from localhost."""
        addr = self.client_address[0] if self.client_address else ""
        return addr in ("127.0.0.1", "::1", "localhost")

    def do_GET(self) -> None:  # noqa: N802
        # Debug endpoints: require auth OR restrict to localhost.
        # Prevents information disclosure (GPU state, model info, logs) to
        # unauthenticated remote callers via tunnel.
        if self._is_debug_path(self.path):
            if self._is_localhost() or self._check_auth():
                self._route_debug(self.path)
            return
        if not self._check_auth():
            return
        self._proxy_request("GET")

    def do_POST(self) -> None:  # noqa: N802
        if not self._check_auth():
            return
        stripped = self.path.rstrip("/").split("?")[0]
        if stripped == "/v1/admin/update":
            self._handle_admin_update()
            return
        if stripped == "/v1/admin/restart":
            self._handle_admin_restart()
            return
        if stripped == "/v1/admin/llm-swap":
            self._handle_admin_llm_swap()
            return
        if stripped == "/v1/admin/install":
            self._handle_admin_install()
            return
        if not self._check_admission():
            return
        try:
            self._proxy_request("POST")
        finally:
            self._release_concurrency()

    def do_HEAD(self) -> None:  # noqa: N802
        if not self._check_auth():
            return
        self._proxy_request("HEAD")

    def do_OPTIONS(self) -> None:  # noqa: N802
        """Handle CORS preflight."""
        self.send_response(204)
        cors_origin = os.environ.get("MAXIM_CORS_ORIGIN", "")
        if cors_origin:
            self.send_header("Access-Control-Allow-Origin", cors_origin)
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type, X-Maxim-Request-Id")
            self.send_header("Access-Control-Max-Age", "86400")
        self.end_headers()

    # ─── helpers ──────────────────────────────────────────────────────

    def _send_json(self, code: int, body: dict) -> None:
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        # Security headers
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Cache-Control", "no-store")
        cors_origin = os.environ.get("MAXIM_CORS_ORIGIN", "")
        if cors_origin:
            self.send_header("Access-Control-Allow-Origin", cors_origin)
        self.end_headers()
        self.wfile.write(data)


# ─── server lifecycle ─────────────────────────────────────────────────────


_leader_proxy_server: HTTPServer | None = None


def start_leader_proxy(
    *,
    proxy_port: int | None = None,
    upstream_port: int | None = None,
    api_key: str | None = None,
    bind_host: str = "0.0.0.0",
) -> HTTPServer | None:
    """Start the LeaderProxy in a daemon thread.

    Returns the HTTPServer instance (for shutdown), or None on failure.
    Idempotent — returns the existing server if already started.
    """
    global _leader_proxy_server
    if proxy_port is None:
        proxy_port = int(
            os.environ.get(
                "MAXIM_LEADER_PROXY_PORT",
                str(DEFAULT_PROXY_PORT),
            )
        )
    # Idempotent guard for the default port (production). Tests use custom
    # ports and should always get fresh servers.
    if _leader_proxy_server is not None and proxy_port == DEFAULT_PROXY_PORT:
        return _leader_proxy_server
    if upstream_port is None:
        upstream_port = int(
            os.environ.get(
                "MAXIM_AUTO_SPAWN_PORT",
                str(DEFAULT_UPSTREAM_PORT),
            )
        )

    upstream_url = f"http://127.0.0.1:{upstream_port}"
    request_log = _RequestLog()

    # Install log buffer handler so /v1/debug/logs can serve log lines
    _ensure_log_buffer()

    # Phase 8: get the shared large tier metrics for proxy traffic recording
    infer_metrics = None
    try:
        from maxim.models.language.lane_metrics import get_metrics_registry

        infer_metrics = get_metrics_registry().get("large")
    except Exception:
        pass

    # Phase 7b: concurrency cap + per-peer rate limiter
    try:
        max_concurrent = int(os.environ.get("MAXIM_PROXY_MAX_CONCURRENT", "4"))
    except (ValueError, TypeError):
        logger.warning("Invalid MAXIM_PROXY_MAX_CONCURRENT value, using default 4")
        max_concurrent = 4
    semaphore = threading.Semaphore(max_concurrent) if max_concurrent > 0 else None

    peer_rate_limiter = None
    try:
        from maxim.runtime.rate_limiter import PeerRateLimiter

        peer_rate_limiter = PeerRateLimiter()
        if peer_rate_limiter.enabled:
            logger.info(
                "Per-peer rate limit: %s RPM",
                os.environ.get("MAXIM_PROXY_RATE_LIMIT_RPM", "0"),
            )
    except Exception as e:
        # Rate limiter init failed — proxy will run without per-peer rate
        # limiting even if MAXIM_PROXY_RATE_LIMIT_RPM is set. This is a
        # silent security gap if the operator configured a rate limit
        # expecting protection. Log at WARNING so it's immediately visible.
        logger.warning(
            "Failed to initialize per-peer rate limiter: %s — rate limiting disabled (check MAXIM_PROXY_RATE_LIMIT_RPM config)",
            e,
        )

    handler = type(
        "ProxyHandler",
        (_ProxyHandler,),
        {
            "api_key": api_key,
            "upstream_url": upstream_url,
            "request_log": request_log,
            "lane_metrics": infer_metrics,
            "concurrency_semaphore": semaphore,
            "rate_limiter": peer_rate_limiter,
            "start_time": time.time(),
        },
    )

    # ThreadingHTTPServer handles concurrent connections from cloudflared's
    # HTTP/2 multiplexing. Plain HTTPServer is single-threaded and blocks
    # POST requests while a keep-alive GET connection is open.
    class _ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
        daemon_threads = True

    try:
        server = _ThreadingHTTPServer((bind_host, proxy_port), handler)
    except OSError as e:
        logger.warning("LeaderProxy port %d in use: %s", proxy_port, e)
        return None

    thread = threading.Thread(
        target=server.serve_forever,
        daemon=True,
        name="leader-proxy",
    )
    thread.start()

    logger.info(
        "LeaderProxy listening on %s:%d → upstream %s (auth=%s, debug endpoints enabled)",
        bind_host,
        proxy_port,
        upstream_url,
        "on" if api_key else "off",
    )
    _leader_proxy_server = server
    return server


__all__ = ["start_leader_proxy", "DEFAULT_PROXY_PORT", "DEFAULT_UPSTREAM_PORT"]
