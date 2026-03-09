"""PlanDashboard — writes human-readable ACTIVE_PLAN.md to workspace.

Subscribes to plan bus events and rewrites the dashboard file on each
state change. Uses a background thread with event coalescing to avoid
blocking PlanManager's RLock during synchronous bus dispatch.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any

from maxim.agents.bus import (
    AgentBus,
    PhaseCompleted,
    PhaseStarted,
    PlanCompleted,
    PlanCreated,
    PlanReplanRequested,
    PlanRestored,
)

logger = logging.getLogger(__name__)


class PlanDashboard:
    """Maintains .maxim_workspace/plans/ACTIVE_PLAN.md from bus events."""

    FILENAME = "ACTIVE_PLAN.md"

    def __init__(self, workspace_path: str, bus: AgentBus) -> None:
        self._workspace_path = workspace_path
        self._bus = bus

        # Current plan state (updated by bus events)
        self._plan_id: str | None = None
        self._objective: str = ""
        self._status: str = ""
        self._started_at: float = 0.0
        self._phases: list[dict[str, Any]] = []
        self._current_phase_index: int = -1
        self._replan_count: int = 0

        # Background writer with coalescing
        self._write_pending = threading.Event()
        self._clear_pending = False
        self._lock = threading.Lock()
        self._writer_thread = threading.Thread(
            target=self._writer_loop, daemon=True, name="plan-dashboard"
        )
        self._writer_thread.start()

        # Subscribe to plan events
        self._bus.subscribe(PlanCreated, self._on_plan_created)
        self._bus.subscribe(PhaseStarted, self._on_phase_started)
        self._bus.subscribe(PhaseCompleted, self._on_phase_completed)
        self._bus.subscribe(PlanCompleted, self._on_plan_completed)
        self._bus.subscribe(PlanReplanRequested, self._on_replan)
        self._bus.subscribe(PlanRestored, self._on_plan_restored)

    @property
    def _dashboard_path(self) -> str:
        return os.path.join(self._workspace_path, "plans", self.FILENAME)

    # ── Bus event handlers ────────────────────────────────────────────────

    def _on_plan_created(self, event: PlanCreated) -> None:
        with self._lock:
            self._plan_id = event.plan_id
            self._objective = event.objective
            self._status = "ACTIVE"
            self._started_at = event.timestamp
            self._phases = [
                {"description": f"Phase {i + 1}", "status": "pending", "duration": None}
                for i in range(event.phase_count)
            ]
            self._current_phase_index = -1
            self._replan_count = 0
        self._request_write()

    def _on_phase_started(self, event: PhaseStarted) -> None:
        with self._lock:
            if event.plan_id != self._plan_id:
                return
            self._current_phase_index = event.phase_index
            if event.phase_index < len(self._phases):
                self._phases[event.phase_index]["description"] = event.description
                self._phases[event.phase_index]["status"] = "active"
                self._phases[event.phase_index]["started_at"] = event.timestamp
        self._request_write()

    def _on_phase_completed(self, event: PhaseCompleted) -> None:
        with self._lock:
            if event.plan_id != self._plan_id:
                return
            for phase in self._phases:
                if phase.get("status") == "active":
                    phase["status"] = "completed" if event.success else "failed"
                    started = phase.get("started_at")
                    if started:
                        phase["duration"] = event.timestamp - started
                    break
        self._request_write()

    def _on_plan_completed(self, event: PlanCompleted) -> None:
        with self._lock:
            if event.plan_id != self._plan_id:
                return
            self._status = "COMPLETED" if event.success else "FAILED"
            self._plan_id = None
            self._clear_pending = True
        self._request_write()

    def _on_replan(self, event: PlanReplanRequested) -> None:
        with self._lock:
            if event.plan_id != self._plan_id:
                return
            self._status = "REPLANNING"
            self._replan_count += 1
        self._request_write()

    def _on_plan_restored(self, event: PlanRestored) -> None:
        with self._lock:
            self._plan_id = event.plan_id
            self._objective = event.objective
            self._status = event.status
            self._current_phase_index = event.current_phase_index
            # Phases will be filled in by subsequent PhaseStarted events;
            # for now create placeholder entries
            if not self._phases:
                self._phases = [
                    {
                        "description": f"Phase {i + 1}",
                        "status": "completed" if i < event.current_phase_index else "pending",
                        "duration": None,
                    }
                    for i in range(event.current_phase_index + 2)  # at least current + 1
                ]
        self._request_write()

    # ── Background writer ─────────────────────────────────────────────────

    def _request_write(self) -> None:
        self._write_pending.set()

    def _writer_loop(self) -> None:
        """Background thread: waits for write events, coalesces rapid updates."""
        while True:
            self._write_pending.wait()
            # Brief coalescing delay — batch rapid events
            time.sleep(0.1)
            self._write_pending.clear()

            with self._lock:
                should_clear = self._clear_pending
                self._clear_pending = False
                content = self._render() if not should_clear else ""

            try:
                plans_dir = os.path.join(self._workspace_path, "plans")
                os.makedirs(plans_dir, exist_ok=True)
                path = self._dashboard_path

                if should_clear:
                    # Clear dashboard on plan completion
                    if os.path.exists(path):
                        os.remove(path)
                else:
                    tmp = path + ".tmp"
                    with open(tmp, "w") as f:
                        f.write(content)
                    os.replace(tmp, path)
            except OSError as e:
                logger.warning("Dashboard write failed: %s", e)

    def _render(self) -> str:
        """Render the dashboard markdown. Called under lock."""
        lines = [f"# Active Plan: {self._objective}"]

        started_str = ""
        if self._started_at:
            t = time.localtime(self._started_at)
            started_str = f" | Started: {t.tm_hour:02d}:{t.tm_min:02d}"

        total = len(self._phases)
        current = self._current_phase_index + 1
        replan_note = f" | Replan #{self._replan_count}" if self._replan_count else ""
        lines.append(
            f"Status: {self._status} | Phase {current}/{total}{started_str}{replan_note}"
        )
        lines.append("")
        lines.append("## Phases")

        for i, phase in enumerate(self._phases):
            status = phase["status"]
            desc = phase["description"]
            duration = phase.get("duration")

            if status == "completed":
                dur_str = f" ({duration:.0f}s)" if duration else ""
                lines.append(f"{i + 1}. [x] {desc}{dur_str}")
            elif status == "failed":
                dur_str = f" ({duration:.0f}s)" if duration else ""
                lines.append(f"{i + 1}. [FAILED] {desc}{dur_str}")
            elif status == "active":
                lines.append(f"{i + 1}. [ ] {desc} ← CURRENT")
            else:
                lines.append(f"{i + 1}. [ ] {desc}")

        lines.append("")
        return "\n".join(lines)


__all__ = ["PlanDashboard"]
