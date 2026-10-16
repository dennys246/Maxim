"""HeartbeatMonitor — periodic system + LLM health sampling with stall detection.

Runs a daemon thread that samples system metrics (GPU, CPU, RAM, disk, WiFi)
and LLM lane metrics every N seconds. Prints a compact heartbeat line to the
logger and detects stalls (no LLM calls for M seconds).

Enabled by:
  - MAXIM_HEARTBEAT=1 (explicit opt-in)
  - MAXIM_LANE_TRACE=1 (piggybacks on trace mode)
  - Always on in leader mode

Heartbeat line format:
  [heartbeat] gpu=87% vram=6.2/16G 72C | cpu=23% ram=8.1/16G | large: 5 calls/10s p50=280ms | loop: planning planning_turn=2.1s runs=1

Stall detection (#1268 dormant producer 1; owner decisions 2026-10-14):
  A stall is a STUCK PLANNING TURN. Each agent-loop run registers its LLM worker
  through the loop-state hook (``runtime/loop_state.py::loop_liveness_scope``);
  a stall is a planning attempt waiting on the provider (queued or running) for
  longer than ``stall_threshold.compute_stall_threshold`` (floor
  ``MAXIM_HEARTBEAT_STALL_S``) while ``stall_threshold.stall_suppression``
  finds no live planning-lane call. Waiting for a human, running a tool and quiet
  perception are never stalls; a loop that stops submitting at all is not caught
  (a known limit). It emits a WARNING and a ``heartbeat_stall`` notification.
  Without a registered run there is no stall detection (``has_loop_state_hook``).
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any

logger = logging.getLogger("maxim.heartbeat")

DEFAULT_INTERVAL_S = 10


class HeartbeatMonitor:
    """Periodic health sampler with stall detection."""

    def __init__(
        self,
        *,
        interval_s: float | None = None,
    ) -> None:
        if interval_s is None:
            interval_s = float(
                os.environ.get(
                    "MAXIM_HEARTBEAT_INTERVAL_S",
                    str(DEFAULT_INTERVAL_S),
                )
            )
        self._interval = interval_s
        # Each sampler thread owns its stop Event (F2, as console/health.py's writer does): stop() can time out on a
        # slow sample, and a later start() must not revive the thread it left behind by clearing a shared one.
        self._thread: threading.Thread | None = None
        self._thread_stop: threading.Event | None = None
        self._life_lock = threading.Lock()

        # External state hooks (set by caller)
        self._loop_state_fn: Any = None  # () -> {"state", "outstanding_s", "loops"} or None (loop_state.py)
        self._last_stall_warn: float = 0.0
        # Structured listeners (the console's health strip, console/health.py).
        # The records already go to the "maxim.heartbeat" logger, but at DEBUG,
        # so a log handler would only see them when MAXIM_LOG_FILE forces the
        # root to DEBUG. Listeners get every snapshot and stall regardless.
        self._listeners: list[Any] = []
        self._listeners_lock = threading.Lock()
        self._listener_warned: set[Any] = set()

    @property
    def is_running(self) -> bool:
        """Whether a sampler is current: False from the moment stop() takes it (a slow sample may still finish)."""
        return self._thread is not None

    @property
    def has_loop_state_hook(self) -> bool:
        """Stall detection needs this hook; without it `heartbeat_stall` cannot fire."""
        return self._loop_state_fn is not None

    def register_listener(self, fn: Any) -> None:
        """Register ``fn(kind, data)``; kind is "heartbeat" or "heartbeat_stall"."""
        with self._listeners_lock:
            if fn not in self._listeners:
                self._listeners.append(fn)

    def unregister_listener(self, fn: Any) -> None:
        with self._listeners_lock:
            if fn in self._listeners:
                self._listeners.remove(fn)

    def _notify(self, kind: str, data: dict[str, Any]) -> None:
        with self._listeners_lock:
            listeners = list(self._listeners)
        for fn in listeners:
            try:
                fn(kind, data)
            except Exception as e:  # a listener bug must not stop the sampler — but must not hide
                if fn not in self._listener_warned:
                    self._listener_warned.add(fn)
                    logger.warning(
                        "heartbeat listener %r raised %s; further failures of this listener are not reported",
                        getattr(fn, "__qualname__", fn),
                        type(e).__name__,
                    )

    def set_loop_state_hook(self, fn: Any) -> None:
        """Register (or, with ``None``, clear) the callable returning the agent loops' liveness.

        It returns ``{"state": "planning" | "idle", "outstanding_s": float, "loops": int}`` (the longest-outstanding
        planning turn over every live run), or ``None`` when no run is registered; ``runtime/loop_state.py`` owns the
        only producer. Called every interval.
        """
        self._loop_state_fn = fn

    def start(self) -> None:
        """Start the heartbeat thread."""
        with self._life_lock:
            if self._thread is not None:
                return
            stop_event = threading.Event()
            thread = threading.Thread(target=self._run, args=(stop_event,), daemon=True, name="heartbeat-monitor")
            thread.start()
            self._thread, self._thread_stop = thread, stop_event
        logger.info(
            "Heartbeat monitor started (interval=%ds, stall_threshold=%ds)",
            int(self._interval),
            int(_stall_threshold_s()),
        )

    def stop(self) -> None:
        """Stop the heartbeat thread: bounded, and acting only on the thread it took (a slow sample may outlive the
        join; that thread exits after it, whatever a later start() does)."""
        with self._life_lock:
            thread, stop_event = self._thread, self._thread_stop
            self._thread, self._thread_stop = None, None
        if thread is not None and stop_event is not None:
            stop_event.set()
            thread.join(timeout=self._interval + 1)

    def snapshot(self) -> dict[str, Any]:
        """On-demand snapshot (used by /v1/debug/heartbeat endpoint)."""
        return self._collect()

    def _run(self, stop_event: threading.Event) -> None:
        """Main heartbeat loop, until ``stop_event`` (this thread's own) is set."""
        while not stop_event.is_set():
            try:
                data = self._collect()
                self._emit(data)
                self._check_stall(data)
            except Exception as e:
                logger.warning("Heartbeat error: %s", e)
            stop_event.wait(self._interval)

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
        """Notify listeners, then format and log the heartbeat line."""
        self._notify("heartbeat", data)
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

        # Lane metrics (large tier if present)
        lanes = data.get("lanes", {})
        large = lanes.get("large")
        if isinstance(large, dict) and large.get("jobs_completed", 0) > 0:
            parts.append(
                f"large: {large['jobs_completed']}calls "
                f"p50={large['p50_latency_ms']:.0f}ms "
                f"fail={large['failure_rate']:.0%}"
            )

        # Agent loop state
        loop = data.get("loop")
        if isinstance(loop, dict):
            outstanding = loop.get("outstanding_s", 0.0)
            parts.append(
                f"loop: {loop.get('state', 'unknown')} planning_turn={outstanding:.1f}s runs={loop.get('loops')}"
            )

        if parts:
            # Pair the human message with structured ``data`` so
            # MAXIM_LOG_FILE JSONL captures GPU / VRAM / RAM / loop-idle
            # fields as proper keys instead of message-string text that
            # StructuredFormatter would strip down to ``{"e":"log"}``.
            logger.debug(
                "[heartbeat] %s",
                " | ".join(parts),
                extra={"event": "heartbeat", "data": data},
            )

    def _check_stall(self, data: dict[str, Any]) -> None:
        """Warn on a STUCK PLANNING TURN: an attempt outstanding past the canonical threshold while
        ``stall_suppression`` finds no live planning-lane call (the llm-routing stall-threshold invariant: one
        threshold derivation, one suppression decision). ``turn_in_progress`` is False: a narrator waiting on the
        agent's turn does so inside a tool call, outside any planning attempt of its own."""
        from maxim.runtime.stall_threshold import stall_suppression

        loop = data.get("loop")
        if not isinstance(loop, dict):
            return
        outstanding_s = loop.get("outstanding_s")  # 0.0 when no turn is outstanding
        threshold = _stall_threshold_s()
        if not isinstance(outstanding_s, (int, float)) or outstanding_s <= threshold:
            return
        suppress, wedged_s = stall_suppression(turn_in_progress=False)
        if suppress:
            return  # the provider call is in flight and alive: slow, not stalled
        now = time.time()
        if now - self._last_stall_warn <= threshold:
            return  # at most one warning per threshold
        self._last_stall_warn = now
        large = (data.get("lanes") or {}).get("large")
        large = large if isinstance(large, dict) else {}
        stall = {
            "outstanding_s": outstanding_s,
            "loops": loop.get("loops"),
            "threshold_s": threshold,
            "wedged_s": wedged_s,
            "p50_latency_ms": large.get("p50_latency_ms"),
            "failure_rate": large.get("failure_rate"),
            "jobs_completed": large.get("jobs_completed"),
            "jobs_failed": large.get("jobs_failed"),
        }
        self._notify("heartbeat_stall", stall)
        logger.warning(
            "[heartbeat] STALL DETECTED — a planning turn has been outstanding for %.0fs (threshold=%ds) with %s. "
            "Last large-tier call: p50=%s ms, failure_rate=%s",
            outstanding_s,
            int(threshold),
            "no call in flight" if wedged_s is None else f"its call wedged for {wedged_s:.0f}s",
            large.get("p50_latency_ms", "?"),
            large.get("failure_rate", "?"),
            extra={"event": "heartbeat_stall", "data": stall},
        )


def _stall_threshold_s() -> float:
    """The heartbeat's stall threshold, from the canonical derivation: its own floor (``MAXIM_HEARTBEAT_STALL_S``,
    clamped to [5, 3600]) and no lane-timeout ceiling, since a live call is judged by ``stall_suppression`` against
    the time IT was allowed (stall_detector_timeout_awareness.md Stage 2, Executor review I3). Tier: innate prior
    (a hard-coded 30 s default floor), not learned."""
    from maxim.agents.llm_types import PLANNING_LANE
    from maxim.runtime.stall_threshold import compute_stall_threshold

    return compute_stall_threshold(lane_tier=PLANNING_LANE, floor_env_var="MAXIM_HEARTBEAT_STALL_S")


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
