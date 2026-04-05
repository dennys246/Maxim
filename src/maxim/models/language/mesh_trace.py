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

TRACE_LOGGER_NAME = "maxim.mesh.trace"
REQUEST_ID_HEADER = "X-Maxim-Request-Id"

# Header used by the openai-python client's extra_headers; cached since env
# can't change mid-process.
_trace_logger = logging.getLogger(TRACE_LOGGER_NAME)


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in (
        "1", "true", "t", "yes", "y", "on",
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

    def to_dict(self) -> dict:
        return {
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

    def format_compact(self) -> str:
        """Compact one-line INFO format for MAXIM_LANE_TRACE=1."""
        parts = [
            f"req={self.request_id[:8]}",
            f"provider={self.provider}",
            f"model={self.model}",
            f"status={self.status}",
            f"latency={self.latency_ms:.0f}ms",
        ]
        if self.http_status is not None:
            parts.append(f"http={self.http_status}")
        if self.input_tokens or self.output_tokens:
            parts.append(f"tokens={self.input_tokens}+{self.output_tokens}")
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


def print_startup_warning_if_enabled() -> None:
    """Print a loud startup banner when any debug flag is on.

    These flags create log volume + potential privacy exposure (request IDs,
    URLs in logs). Always visible at startup so users don't leave them on
    accidentally.
    """
    if not any_trace_enabled():
        return
    import sys
    flags = []
    if lane_trace_enabled():
        flags.append("MAXIM_LANE_TRACE")
    if peer_log_enabled():
        flags.append("MAXIM_PEER_LOG_REQUESTS")
    bar = " " + "!" * 62
    print(bar, file=sys.stderr)
    print("  DEBUG FLAGS ACTIVE:", ", ".join(flags), file=sys.stderr)
    print("  Peer inference calls will be logged with request IDs + URLs.", file=sys.stderr)
    print("  Unset these env vars to silence.", file=sys.stderr)
    print(bar, file=sys.stderr)


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
    "print_startup_warning_if_enabled",
    "start_call",
]
