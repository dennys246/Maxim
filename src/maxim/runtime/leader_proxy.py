"""LeaderProxy — reverse proxy sitting in front of llama-cpp-server (Phase 7a).

Sits on port 8099, forwards inference requests to llama-cpp-server on :8100.
Tunnel ingress points here instead of directly at the inference server.

Responsibilities:
  - Authoritative Bearer auth BEFORE requests reach llama-cpp-server
  - Per-request logging: request-id, peer IP, model, latency, tokens
  - /v1/debug/status: GPU utilization, VRAM, temperature, uptime
  - /v1/debug/last-requests: ring buffer of last 100 requests
  - Injects X-Maxim-* response headers for peer-side trace enrichment

Design: stdlib-only (http.server + urllib). No FastAPI/uvicorn dependency.
Adds ~1-2ms per request vs direct llama-cpp-server access.

Supersedes debug_status_server.py (which only served /v1/debug/status).
"""

from __future__ import annotations

import collections
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

# ── Input validation ─────────────────────────────────────────────────────────

_BRANCH_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._/-]*$")


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


from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from pathlib import Path
from typing import Any

logger = logging.getLogger("maxim.leader_proxy")

DEFAULT_PROXY_PORT = 8099
DEFAULT_UPSTREAM_PORT = 8100
_MAX_RECENT_REQUESTS = 100


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

    # Set by the server factory (class-level attrs)
    api_key: str | None = None
    upstream_url: str = "http://127.0.0.1:8100"
    request_log: _RequestLog | None = None
    lane_metrics: Any = None  # LaneMetrics instance for the infer lane
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
            peer_key = self.headers.get("Authorization", self.client_address[0])
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
        if auth == f"Bearer {self.api_key}":
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
            from maxim.runtime.lane_backends import _active_model, _llm_start_time

            active_model = _active_model
            if _llm_start_time is not None:
                llm_uptime_s = round(time.time() - _llm_start_time, 1)
        except Exception:
            pass

        # Lane metrics (infer lane)
        lane_metrics = None
        try:
            from maxim.models.language.lane_metrics import get_metrics_registry

            m = get_metrics_registry().get("infer")
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
        """
        self._send_json(
            200,
            {
                "service": "LeaderProxy",
                "proxy_port": self.server.server_address[1],
                "upstream": self.upstream_url,
                "auth_enabled": bool(self.api_key),
                "uptime_s": round(time.time() - self.start_time, 1),
            },
        )

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
        else:
            self._send_json(404, {"error": "Not found"})

    # ─── reverse proxy ────────────────────────────────────────────────

    def _proxy_request(self, method: str) -> None:
        """Forward the request to the upstream llama-cpp-server."""
        request_id = self.headers.get("X-Maxim-Request-Id", "")
        peer_ip = self.client_address[0]
        t0 = time.time()

        # Read request body
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else None

        # Build upstream request
        upstream = f"{self.upstream_url}{self.path}"
        req = urllib.request.Request(upstream, data=body, method=method)

        # Forward headers (skip hop-by-hop)
        skip_headers = {"host", "connection", "transfer-encoding", "proxy-connection", "keep-alive"}
        for key, val in self.headers.items():
            if key.lower() not in skip_headers:
                req.add_header(key, val)

        # Forward to upstream
        resp_code = 502
        resp_headers: dict[str, str] = {}
        resp_body = b""
        server_ms: float | None = None

        try:
            with urllib.request.urlopen(req, timeout=300) as resp:  # noqa: S310
                resp_code = resp.status
                resp_headers = dict(resp.headers)
                resp_body = resp.read()
                server_ms_raw = resp.headers.get("openai-processing-ms")
                if server_ms_raw:
                    try:
                        server_ms = float(server_ms_raw)
                    except (ValueError, TypeError):
                        pass
        except urllib.error.HTTPError as e:
            resp_code = e.code
            resp_body = e.read()
            resp_headers = dict(e.headers)
        except Exception as e:
            logger.exception("Upstream connection error: %s", e)
            resp_code = 502
            resp_body = json.dumps({"error": "Upstream connection failed"}).encode()

        elapsed_ms = (time.time() - t0) * 1000

        # Send response to client
        self.send_response(resp_code)
        # Forward upstream headers
        skip_resp_headers = {"transfer-encoding", "connection", "keep-alive"}
        for key, val in resp_headers.items():
            if key.lower() not in skip_resp_headers:
                self.send_header(key, val)
        # Inject Maxim headers for peer-side trace enrichment
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
        self.send_header("Content-Length", str(len(resp_body)))
        self.end_headers()
        self.wfile.write(resp_body)

        # Extract token counts from response body
        input_tokens = 0
        output_tokens = 0
        model = ""
        try:
            if resp_code == 200 and resp_body:
                data = json.loads(resp_body)
                usage = data.get("usage", {})
                input_tokens = usage.get("prompt_tokens", 0)
                output_tokens = usage.get("completion_tokens", 0)
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
        """POST /v1/admin/update — git pull + pip install on the leader.

        Requires MAXIM_ALLOW_REMOTE_UPDATE=1 on the leader process.
        Accepts JSON body: {"branch": "main", "dry_run": true/false}
        dry_run=true (default) previews pending commits without applying.
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
                    "error": "Remote update disabled. Set MAXIM_ALLOW_REMOTE_UPDATE=1 on the leader.",
                },
            )
            return

        # Parse request body (small limit — admin requests are tiny)
        raw = self._read_body(max_size=4096)
        body: dict[str, Any] = {}
        if raw:
            try:
                body = json.loads(raw)
            except Exception:
                pass

        raw_branch = body.get("branch", "main")
        try:
            branch = _validate_branch(raw_branch)
        except ValueError as e:
            self._send_json(400, {"error": f"Invalid branch: {e}"})
            return
        dry_run = body.get("dry_run", True)
        force = body.get("force", False)

        # Find repo root (leader_proxy.py is in src/maxim/runtime/)
        repo_root = str(Path(__file__).resolve().parents[3])

        logger.info(
            "admin/update: peer=%s branch=%s dry_run=%s force=%s repo=%s",
            self.client_address[0],
            branch,
            dry_run,
            force,
            repo_root,
        )

        # Check for dirty working tree
        stashed = False
        try:
            status = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=repo_root,
            )
            if status.stdout.strip():
                if force:
                    # Stash changes so pull can proceed
                    stash_result = subprocess.run(
                        ["git", "stash", "--include-untracked"],
                        capture_output=True,
                        text=True,
                        timeout=10,
                        cwd=repo_root,
                    )
                    stashed = stash_result.returncode == 0 and bool(stash_result.stdout.strip())
                    logger.info("admin/update: stashed dirty tree (stashed=%s)", stashed)
                else:
                    self._send_json(
                        409,
                        {
                            "error": "Working tree is dirty. Commit or stash changes first.",
                            "hint": "Use: maxim peer update --force (stashes and restores automatically)",
                            "dirty_files": status.stdout.strip().split("\n"),
                        },
                    )
                    return
        except Exception as e:
            logger.exception("git status failed: %s", e)
            self._send_json(500, {"error": "git status check failed"})
            return

        # Fetch latest
        try:
            subprocess.run(
                ["git", "fetch", "origin", branch],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=repo_root,
                check=True,
            )
        except Exception as e:
            logger.exception("git fetch failed: %s", e)
            self._send_json(500, {"error": "git fetch failed"})
            return

        # Preview: what commits are pending?
        try:
            log_result = subprocess.run(
                ["git", "log", f"HEAD..origin/{branch}", "--oneline"],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=repo_root,
            )
            pending = log_result.stdout.strip().split("\n") if log_result.stdout.strip() else []
        except Exception:
            pending = []

        if not pending:
            self._send_json(
                200,
                {
                    "status": "up_to_date",
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
                    "branch": branch,
                    "pending_commits": pending,
                    "message": f"{len(pending)} commit(s) pending. Send dry_run=false to apply.",
                },
            )
            return

        # Apply: git pull
        try:
            subprocess.run(
                ["git", "-c", "pull.rebase=true", "pull", "origin", branch],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=repo_root,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            # Restore stashed changes before reporting failure
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
            return

        # Restore stashed changes after successful pull
        if stashed:
            subprocess.run(
                ["git", "stash", "pop"],
                capture_output=True,
                timeout=10,
                cwd=repo_root,
            )
            logger.info("admin/update: restored stashed changes")

        # pip install -e .
        pip_output = ""
        try:
            pip_result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-e", "."],
                capture_output=True,
                text=True,
                timeout=120,
                cwd=repo_root,
            )
            pip_output = pip_result.stdout[-500:] if pip_result.stdout else ""
            if pip_result.returncode != 0:
                # Rollback: revert git and reinstall
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
                return
        except Exception as e:
            logger.exception("pip install failed: %s", e)
            self._send_json(500, {"error": "pip install failed"})
            return

        # Log to request ring buffer
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

        logger.info(
            "admin/update: SUCCESS — %d commits applied from origin/%s",
            len(pending),
            branch,
        )

        self._send_json(
            200,
            {
                "status": "updated",
                "branch": branch,
                "commits_applied": pending,
                "pip_output": pip_output,
                "message": f"Applied {len(pending)} commit(s). Restart maxim to load new code.",
            },
        )

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

    # Phase 8: get the shared infer lane metrics for proxy traffic recording
    infer_metrics = None
    try:
        from maxim.models.language.lane_metrics import get_metrics_registry

        infer_metrics = get_metrics_registry().get("infer")
    except Exception:
        pass

    # Phase 7b: concurrency cap + per-peer rate limiter
    max_concurrent = int(os.environ.get("MAXIM_PROXY_MAX_CONCURRENT", "4"))
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
    except Exception:
        pass

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
