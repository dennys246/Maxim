"""Lightweight HTTP server exposing /v1/debug/status for peer diagnostics.

Runs on a separate port (default 8101) alongside llama-cpp-server (8100).
Serves a single endpoint that returns GPU utilization, VRAM, temperature,
and server uptime. Auth-gated with the same API key as the inference server.

Started automatically by the local_server_spawner when in leader mode.
Peers poll this after each inference call when MAXIM_LANE_TRACE=1 to get
server-side metrics without needing SSH access to the leader.

Design: stdlib-only (http.server), no dependencies. Intentionally minimal
so it adds no import cost when not used.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

DEFAULT_DEBUG_PORT = 8101
_server_start_time: float = 0.0


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


def _build_status() -> dict[str, Any]:
    """Build the full status response."""
    gpu = _query_nvidia_smi()
    return {
        "status": "ok",
        "uptime_s": round(time.time() - _server_start_time, 1) if _server_start_time else 0,
        "gpu": gpu,
        "timestamp": time.time(),
    }


class _DebugHandler(BaseHTTPRequestHandler):
    """Handle GET /v1/debug/status with optional Bearer auth."""

    # Set by the server factory
    api_key: str | None = None

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        # Silence default stderr logging
        pass

    def _check_auth(self) -> bool:
        if not self.api_key:
            return True
        auth = self.headers.get("Authorization", "")
        if auth == f"Bearer {self.api_key}":
            return True
        self._send_json(401, {"error": "Invalid API key"})
        return False

    def _send_json(self, code: int, body: dict) -> None:
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") == "/v1/debug/status":
            if not self._check_auth():
                return
            self._send_json(200, _build_status())
        else:
            self._send_json(404, {"error": "Not found"})


def start_debug_server(
    *,
    port: int | None = None,
    api_key: str | None = None,
    bind_host: str = "0.0.0.0",
) -> HTTPServer | None:
    """Start the debug status server in a daemon thread.

    Returns the HTTPServer instance (for shutdown), or None on failure.
    """
    global _server_start_time

    if port is None:
        port = int(os.environ.get("MAXIM_DEBUG_STATUS_PORT", str(DEFAULT_DEBUG_PORT)))

    # Create handler class with the api_key baked in
    handler = type("Handler", (_DebugHandler,), {"api_key": api_key})

    try:
        server = HTTPServer((bind_host, port), handler)
    except OSError as e:
        # Port already in use — another debug server running
        import logging

        logging.getLogger("maxim").debug(
            "Debug status server port %d in use: %s",
            port,
            e,
        )
        return None

    _server_start_time = time.time()
    thread = threading.Thread(target=server.serve_forever, daemon=True, name="debug-status")
    thread.start()

    import logging

    logging.getLogger("maxim").info(
        "Debug status server listening on %s:%d/v1/debug/status",
        bind_host,
        port,
    )
    return server


__all__ = ["start_debug_server", "DEFAULT_DEBUG_PORT"]
