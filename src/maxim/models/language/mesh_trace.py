"""Stage A observability — request-id propagation + structured peer-side logging.

Debug-plan foundation (see docs/plans/peer_leader_debug_plan.md). Gives peer
inference calls a correlation ID that can be traced across machines, and
emits structured log lines for every outbound request when MAXIM_LANE_TRACE
or MAXIM_PEER_LOG_REQUESTS is set.

The dedicated `maxim.mesh.trace` logger is also used by Phase 7a's LeaderProxy
when that ships — same channel, end-to-end correlation.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any

TRACE_LOGGER_NAME = "maxim.mesh.trace"
REQUEST_ID_HEADER = "X-Maxim-Request-Id"

# Header used by the openai-python client's extra_headers; cached since env
# can't change mid-process.
_trace_logger = logging.getLogger(TRACE_LOGGER_NAME)


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in (
        "1",
        "true",
        "t",
        "yes",
        "y",
        "on",
    )


def lane_trace_enabled() -> bool:
    """MAXIM_LANE_TRACE=1 — log every lane LLM dispatch."""
    return _env_flag("MAXIM_LANE_TRACE")


def peer_log_enabled() -> bool:
    """MAXIM_PEER_LOG_REQUESTS=1 — structured JSON log per outbound peer call."""
    return _env_flag("MAXIM_PEER_LOG_REQUESTS")


def any_trace_enabled() -> bool:
    return lane_trace_enabled() or peer_log_enabled()


def generate_request_id() -> str:
    """UUID4 hex (32 chars, no dashes) for correlation across machines."""
    return uuid.uuid4().hex


@dataclass
class TraceRecord:
    request_id: str
    provider: str
    base_url: str
    model: str
    status: str  # "ok" | "error" | "timeout"
    http_status: int | None
    latency_ms: float
    input_tokens: int = 0
    output_tokens: int = 0
    error: str | None = None
    # Server-side timing (from openai-processing-ms header)
    server_processing_ms: float | None = None
    # Leader GPU debug info (from /v1/debug/status poll)
    gpu_util_pct: float | None = None
    gpu_vram_used_gb: float | None = None
    gpu_vram_total_gb: float | None = None
    gpu_temp_c: float | None = None

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "request_id": self.request_id,
            "provider": self.provider,
            "base_url": self.base_url,
            "model": self.model,
            "status": self.status,
            "http_status": self.http_status,
            "latency_ms": round(self.latency_ms, 1),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "error": self.error,
        }
        if self.server_processing_ms is not None:
            d["server_processing_ms"] = round(self.server_processing_ms, 1)
        if self.gpu_util_pct is not None:
            d["gpu_util_pct"] = round(self.gpu_util_pct, 1)
        if self.gpu_vram_used_gb is not None:
            d["gpu_vram_used_gb"] = round(self.gpu_vram_used_gb, 2)
            d["gpu_vram_total_gb"] = round(self.gpu_vram_total_gb or 0, 2)
        if self.gpu_temp_c is not None:
            d["gpu_temp_c"] = round(self.gpu_temp_c, 0)
        return d

    def format_compact(self) -> str:
        """Compact one-line INFO format for MAXIM_LANE_TRACE=1."""
        parts = [
            f"req={self.request_id[:8]}",
            f"provider={self.provider}",
            f"model={self.model}",
            f"status={self.status}",
            f"latency={self.latency_ms:.0f}ms",
        ]
        if self.server_processing_ms is not None:
            parts.append(f"server={self.server_processing_ms:.0f}ms")
        if self.http_status is not None:
            parts.append(f"http={self.http_status}")
        if self.input_tokens or self.output_tokens:
            parts.append(f"tokens={self.input_tokens}+{self.output_tokens}")
        if self.gpu_util_pct is not None:
            vram = ""
            if self.gpu_vram_used_gb is not None and self.gpu_vram_total_gb:
                vram = f" vram={self.gpu_vram_used_gb:.1f}/{self.gpu_vram_total_gb:.0f}G"
            temp = f" {self.gpu_temp_c:.0f}C" if self.gpu_temp_c is not None else ""
            parts.append(f"gpu={self.gpu_util_pct:.0f}%{vram}{temp}")
        if self.error:
            parts.append(f"err={self.error[:50]}")
        return "peer_infer " + " ".join(parts)


def emit_trace(record: TraceRecord) -> None:
    """Emit a trace record to the mesh.trace logger. No-op if no flag enabled."""
    if not any_trace_enabled():
        return
    if lane_trace_enabled():
        _trace_logger.info(record.format_compact())
    if peer_log_enabled():
        _trace_logger.info(json.dumps(record.to_dict()))


def fetch_leader_debug_status(
    base_url: str,
    api_key: str | None = None,
    *,
    timeout_s: float = 1.5,
) -> dict | None:
    """Poll /v1/debug/status on the leader for GPU metrics.

    Called after each inference request when MAXIM_LANE_TRACE=1 to capture
    server-side GPU utilization, VRAM, and temperature. Returns None silently
    if the endpoint is unavailable (leader hasn't been upgraded, or debug
    endpoint not implemented yet).
    """
    import urllib.error
    import urllib.request

    url = base_url.rstrip("/")
    if url.endswith("/v1"):
        url = url[:-3]
    url += "/v1/debug/status"

    req = urllib.request.Request(url, method="GET")
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    req.add_header("User-Agent", "maxim-peer/1.0")

    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:  # noqa: S310
            return json.loads(resp.read())
    except Exception:
        return None


def enrich_trace_from_headers(record: TraceRecord, headers: Any) -> None:
    """Extract GPU metrics from X-Maxim-* response headers (LeaderProxy).

    When the leader runs the LeaderProxy (Phase 7a), it injects GPU
    metrics directly into inference response headers. This is faster
    and more reliable than polling /v1/debug/status separately.
    """
    if not lane_trace_enabled() or not hasattr(headers, "get"):
        return
    gpu_util = headers.get("X-Maxim-GPU-Util")
    if gpu_util is not None:
        try:
            record.gpu_util_pct = float(gpu_util)
        except (ValueError, TypeError):
            pass
    gpu_vram = headers.get("X-Maxim-GPU-VRAM")
    if gpu_vram and "/" in gpu_vram:
        try:
            used, total = gpu_vram.split("/")
            record.gpu_vram_used_gb = float(used)
            record.gpu_vram_total_gb = float(total.rstrip("G"))
        except (ValueError, TypeError):
            pass
    gpu_temp = headers.get("X-Maxim-GPU-Temp")
    if gpu_temp is not None:
        try:
            record.gpu_temp_c = float(gpu_temp)
        except (ValueError, TypeError):
            pass


def enrich_trace_with_leader_status(
    record: TraceRecord,
    base_url: str,
    api_key: str | None = None,
    response_headers: Any = None,
) -> None:
    """Enrich a trace record with leader-side GPU metrics.

    Tries two sources in order:
    1. X-Maxim-* response headers (if LeaderProxy is running, zero-cost)
    2. Poll /v1/debug/status (fallback for pre-7a leaders, adds ~2ms)
    """
    if not lane_trace_enabled():
        return
    # Try headers first (LeaderProxy injects these)
    if response_headers is not None:
        enrich_trace_from_headers(record, response_headers)
        if record.gpu_util_pct is not None:
            return  # Got metrics from headers, skip poll
    # Fallback: poll /v1/debug/status
    status = fetch_leader_debug_status(base_url, api_key)
    if status is None:
        return
    gpu = status.get("gpu")
    if isinstance(gpu, dict):
        record.gpu_util_pct = gpu.get("utilization_pct")
        record.gpu_vram_used_gb = gpu.get("vram_used_gb")
        record.gpu_vram_total_gb = gpu.get("vram_total_gb")
        record.gpu_temp_c = gpu.get("temperature_c")


def print_startup_warning_if_enabled() -> None:
    """Print a loud startup banner when any debug flag is on.

    These flags create log volume + potential privacy exposure (request IDs,
    URLs in logs). Always visible at startup so users don't leave them on
    accidentally.
    """
    if not any_trace_enabled():
        return

    flags = []
    if lane_trace_enabled():
        flags.append("MAXIM_LANE_TRACE")
    if peer_log_enabled():
        flags.append("MAXIM_PEER_LOG_REQUESTS")
    bar = " " + "!" * 62
    _trace_logger.warning("%s", bar)
    _trace_logger.warning("  DEBUG FLAGS ACTIVE: %s", ", ".join(flags))
    _trace_logger.warning("  Peer inference calls will be logged with request IDs + URLs.")
    _trace_logger.warning("  Unset these env vars to silence.")
    _trace_logger.warning("%s", bar)


def start_call() -> tuple[str, float]:
    """Generate a request id + start timestamp for a new outbound call."""
    return generate_request_id(), time.time()


__all__ = [
    "TRACE_LOGGER_NAME",
    "REQUEST_ID_HEADER",
    "TraceRecord",
    "generate_request_id",
    "lane_trace_enabled",
    "peer_log_enabled",
    "any_trace_enabled",
    "emit_trace",
    "enrich_trace_from_headers",
    "enrich_trace_with_leader_status",
    "fetch_leader_debug_status",
    "print_startup_warning_if_enabled",
    "start_call",
]
