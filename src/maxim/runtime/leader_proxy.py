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
import subprocess
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
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
                self._send_json(429, {
                    "error": "Rate limit exceeded",
                    "retry_after_s": 1,
                })
                logger.info(
                    "proxy: rate-limited peer=%s", self.client_address[0],
                )
                return False
        # Concurrency cap
        if self.concurrency_semaphore is not None:
            if not self.concurrency_semaphore.acquire(blocking=False):
                queue_depth = self._queue_depth()
                self._send_json(429, {
                    "error": "Too many concurrent requests",
                    "queue_depth": queue_depth,
                })
                logger.info(
                    "proxy: concurrency-limited peer=%s depth=%d",
                    self.client_address[0], queue_depth,
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

    # ─── auth ─────────────────────────────────────────────────────────

    def _check_auth(self) -> bool:
        """Enforce Bearer auth. Returns True if authorized."""
        if not self.api_key:
            return True
        auth = self.headers.get("Authorization", "")
        if auth == f"Bearer {self.api_key}":
            return True
        self._send_json(401, {"error": "Invalid API key"})
        return False

    # ─── debug endpoints (served directly, not proxied) ───────────────

    def _handle_debug_status(self) -> None:
        gpu = _query_nvidia_smi()
        self._send_json(200, {
            "status": "ok",
            "uptime_s": round(time.time() - self.start_time, 1),
            "gpu": gpu,
            "timestamp": time.time(),
        })

    def _handle_debug_metrics(self) -> None:
        try:
            from maxim.models.language.lane_metrics import get_metrics_registry
            self._send_json(200, get_metrics_registry().snapshot())
        except Exception:
            self._send_json(200, {})

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
            "/v1/debug/status", "/v1/debug/last-requests", "/v1/debug/metrics",
        )

    def _route_debug(self, path: str) -> None:
        stripped = path.rstrip("/").split("?")[0]
        if stripped == "/v1/debug/status":
            self._handle_debug_status()
        elif stripped == "/v1/debug/metrics":
            self._handle_debug_metrics()
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
        skip_headers = {"host", "connection", "transfer-encoding",
                        "proxy-connection", "keep-alive"}
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
            resp_code = 502
            resp_body = json.dumps({"error": f"Upstream error: {e}"}).encode()

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
            self.send_header("X-Maxim-GPU-VRAM",
                             f"{gpu['vram_used_gb']:.1f}/{gpu['vram_total_gb']:.0f}")
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

    # ─── HTTP method dispatchers ──────────────────────────────────────

    def do_GET(self) -> None:  # noqa: N802
        if not self._check_auth():
            return
        if self._is_debug_path(self.path):
            self._route_debug(self.path)
        else:
            self._proxy_request("GET")

    def do_POST(self) -> None:  # noqa: N802
        if not self._check_auth():
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
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers",
                         "Authorization, Content-Type, X-Maxim-Request-Id")
        self.send_header("Access-Control-Max-Age", "86400")
        self.end_headers()

    # ─── helpers ──────────────────────────────────────────────────────

    def _send_json(self, code: int, body: dict) -> None:
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)


# ─── server lifecycle ─────────────────────────────────────────────────────

def start_leader_proxy(
    *,
    proxy_port: int | None = None,
    upstream_port: int | None = None,
    api_key: str | None = None,
    bind_host: str = "0.0.0.0",
) -> HTTPServer | None:
    """Start the LeaderProxy in a daemon thread.

    Returns the HTTPServer instance (for shutdown), or None on failure.
    """
    if proxy_port is None:
        proxy_port = int(os.environ.get(
            "MAXIM_LEADER_PROXY_PORT", str(DEFAULT_PROXY_PORT),
        ))
    if upstream_port is None:
        upstream_port = int(os.environ.get(
            "MAXIM_AUTO_SPAWN_PORT", str(DEFAULT_UPSTREAM_PORT),
        ))

    upstream_url = f"http://127.0.0.1:{upstream_port}"
    request_log = _RequestLog()

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

    handler = type("ProxyHandler", (_ProxyHandler,), {
        "api_key": api_key,
        "upstream_url": upstream_url,
        "request_log": request_log,
        "lane_metrics": infer_metrics,
        "concurrency_semaphore": semaphore,
        "rate_limiter": peer_rate_limiter,
        "start_time": time.time(),
    })

    try:
        server = HTTPServer((bind_host, proxy_port), handler)
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
        "LeaderProxy listening on %s:%d → upstream %s "
        "(auth=%s, debug endpoints enabled)",
        bind_host, proxy_port, upstream_url,
        "on" if api_key else "off",
    )
    return server


__all__ = ["start_leader_proxy", "DEFAULT_PROXY_PORT", "DEFAULT_UPSTREAM_PORT"]
