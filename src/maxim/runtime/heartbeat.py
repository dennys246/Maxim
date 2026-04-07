"""HeartbeatMonitor — periodic system + LLM health sampling with stall detection.

Runs a daemon thread that samples system metrics (GPU, CPU, RAM, disk, WiFi)
and LLM lane metrics every N seconds. Prints a compact heartbeat line to the
logger and detects stalls (no LLM calls for M seconds).

Enabled by:
  - MAXIM_HEARTBEAT=1 (explicit opt-in)
  - MAXIM_LANE_TRACE=1 (piggybacks on trace mode)
  - Always on in leader mode

Heartbeat line format:
  [heartbeat] gpu=87% vram=6.2/16G 72C | cpu=23% ram=8.1/16G | infer: 5 calls/10s p50=280ms | loop: idle=2.1s

Stall detection:
  If no LLM calls have been recorded for MAXIM_HEARTBEAT_STALL_S (default 30s),
  emits a WARNING with the last known state. This would have caught the
  'respond success=False → 60s dead zone' bug instantly.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any

logger = logging.getLogger("maxim.heartbeat")

DEFAULT_INTERVAL_S = 10
DEFAULT_STALL_THRESHOLD_S = 30


class HeartbeatMonitor:
    """Periodic health sampler with stall detection."""

    def __init__(
        self,
        *,
        interval_s: float | None = None,
        stall_threshold_s: float | None = None,
    ) -> None:
        if interval_s is None:
            interval_s = float(
                os.environ.get(
                    "MAXIM_HEARTBEAT_INTERVAL_S",
                    str(DEFAULT_INTERVAL_S),
                )
            )
        if stall_threshold_s is None:
            stall_threshold_s = float(
                os.environ.get(
                    "MAXIM_HEARTBEAT_STALL_S",
                    str(DEFAULT_STALL_THRESHOLD_S),
                )
            )
        self._interval = interval_s
        self._stall_threshold = stall_threshold_s
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        # External state hooks (set by caller)
        self._loop_state_fn: Any = None  # () -> dict with last_event_time, state
        self._last_stall_warn: float = 0.0

    def set_loop_state_hook(self, fn: Any) -> None:
        """Register a callable that returns agent loop state dict.

        Expected keys: last_event_time (float), state (str).
        Called every heartbeat interval.
        """
        self._loop_state_fn = fn

    def start(self) -> None:
        """Start the heartbeat thread."""
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="heartbeat-monitor",
        )
        self._thread.start()
        logger.info(
            "Heartbeat monitor started (interval=%ds, stall_threshold=%ds)",
            int(self._interval),
            int(self._stall_threshold),
        )

    def stop(self) -> None:
        """Stop the heartbeat thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval + 1)
            self._thread = None

    def snapshot(self) -> dict[str, Any]:
        """On-demand snapshot (used by /v1/debug/heartbeat endpoint)."""
        return self._collect()

    def _run(self) -> None:
        """Main heartbeat loop."""
        while not self._stop_event.is_set():
            try:
                data = self._collect()
                self._emit(data)
                self._check_stall(data)
            except Exception as e:
                logger.debug("Heartbeat error: %s", e)
            self._stop_event.wait(self._interval)

    def _collect(self) -> dict[str, Any]:
        """Collect system + LLM metrics."""
        from maxim.runtime.system_metrics import collect_all

        data: dict[str, Any] = collect_all()

        # LLM lane metrics
        try:
            from maxim.models.language.lane_metrics import get_metrics_registry

            data["lanes"] = get_metrics_registry().snapshot()
        except Exception:
            data["lanes"] = {}

        # Agent loop state (if hook registered)
        if self._loop_state_fn is not None:
            try:
                data["loop"] = self._loop_state_fn()
            except Exception:
                data["loop"] = None
        else:
            data["loop"] = None

        return data

    def _emit(self, data: dict[str, Any]) -> None:
        """Format and log the heartbeat line."""
        parts: list[str] = []

        # GPU
        gpu = data.get("gpu")
        if isinstance(gpu, dict) and gpu.get("utilization_pct") is not None:
            vram = f" vram={gpu['vram_used_gb']:.1f}/{gpu['vram_total_gb']:.0f}G"
            temp = f" {gpu['temperature_c']:.0f}C" if gpu.get("temperature_c") else ""
            power = f" {gpu['power_draw_w']:.0f}W" if gpu.get("power_draw_w") else ""
            parts.append(f"gpu={gpu['utilization_pct']:.0f}%{vram}{temp}{power}")

        # CPU
        cpu = data.get("cpu")
        if isinstance(cpu, dict):
            parts.append(f"cpu={cpu.get('usage_pct', 0):.0f}%")

        # Memory
        mem = data.get("memory")
        if isinstance(mem, dict):
            parts.append(f"ram={mem['used_gb']:.1f}/{mem['total_gb']:.0f}G")

        # Disk
        disk = data.get("disk")
        if isinstance(disk, dict):
            parts.append(f"disk={disk['free_gb']:.0f}G free")

        # WiFi
        wifi = data.get("wifi")
        if isinstance(wifi, dict) and wifi.get("rssi_dbm"):
            parts.append(f"wifi={wifi['rssi_dbm']:.0f}dBm")

        # Lane metrics (infer lane if present)
        lanes = data.get("lanes", {})
        infer = lanes.get("infer")
        if isinstance(infer, dict) and infer.get("jobs_completed", 0) > 0:
            parts.append(
                f"infer: {infer['jobs_completed']}calls "
                f"p50={infer['p50_latency_ms']:.0f}ms "
                f"fail={infer['failure_rate']:.0%}"
            )

        # Agent loop state
        loop = data.get("loop")
        if isinstance(loop, dict):
            idle = loop.get("idle_s", 0)
            state = loop.get("state", "unknown")
            parts.append(f"loop: idle={idle:.1f}s state={state}")

        if parts:
            logger.debug("[heartbeat] %s", " | ".join(parts))

    def _check_stall(self, data: dict[str, Any]) -> None:
        """Warn if no LLM calls have happened within the stall threshold."""
        lanes = data.get("lanes", {})
        infer = lanes.get("infer")
        if not isinstance(infer, dict):
            return

        total_calls = infer.get("jobs_completed", 0) + infer.get("jobs_failed", 0)
        if total_calls == 0:
            return  # no calls yet, not a stall

        # Check loop idle time
        loop = data.get("loop")
        if isinstance(loop, dict):
            idle_s = loop.get("idle_s", 0)
            if idle_s > self._stall_threshold:
                now = time.time()
                # Don't spam — warn at most once per stall threshold
                if now - self._last_stall_warn > self._stall_threshold:
                    self._last_stall_warn = now
                    logger.warning(
                        "[heartbeat] STALL DETECTED — agent loop idle for %.0fs "
                        "(state=%s, threshold=%ds). Last infer call: p50=%s ms, "
                        "failure_rate=%s",
                        idle_s,
                        loop.get("state", "?"),
                        int(self._stall_threshold),
                        infer.get("p50_latency_ms", "?"),
                        infer.get("failure_rate", "?"),
                    )


# ─── module-level singleton ───────────────────────────────────────────────

_global_monitor: HeartbeatMonitor | None = None
_monitor_lock = threading.Lock()


def get_heartbeat_monitor() -> HeartbeatMonitor:
    """Get or create the global HeartbeatMonitor."""
    global _global_monitor
    with _monitor_lock:
        if _global_monitor is None:
            _global_monitor = HeartbeatMonitor()
        return _global_monitor


def should_enable_heartbeat() -> bool:
    """Check if heartbeat should auto-start based on env flags."""
    if os.environ.get("MAXIM_HEARTBEAT", "").strip().lower() in (
        "1",
        "true",
        "t",
        "yes",
        "y",
        "on",
    ):
        return True
    if os.environ.get("MAXIM_LANE_TRACE", "").strip().lower() in (
        "1",
        "true",
        "t",
        "yes",
        "y",
        "on",
    ):
        return True
    return False


__all__ = [
    "HeartbeatMonitor",
    "get_heartbeat_monitor",
    "should_enable_heartbeat",
]
