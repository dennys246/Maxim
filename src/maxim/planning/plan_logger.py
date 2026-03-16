"""PlanLogger — appends one-line entries to .maxim_workspace/plans/history.md.

Subscribes to plan bus events and logs each transition as a timestamped
single-line entry. Uses an async write queue (background thread) to avoid
blocking PlanManager's RLock during synchronous bus dispatch.

Rolls at 500 lines, archiving old entries.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time

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


class PlanLogger:
    """Append-only plan history log in .maxim_workspace/plans/history.md."""

    FILENAME = "history.md"
    ARCHIVE_DIR = "history_archive"
    MAX_LINES = 500

    def __init__(self, workspace_path: str, bus: AgentBus) -> None:
        self._workspace_path = workspace_path
        self._bus = bus
        self._queue: queue.Queue[str] = queue.Queue()

        # Track phase counts per plan for summary lines
        self._plan_phases: dict[str, int] = {}

        # Background writer thread
        self._writer_thread = threading.Thread(
            target=self._writer_loop, daemon=True, name="plan-logger"
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
    def _history_path(self) -> str:
        return os.path.join(self._workspace_path, "plans", self.FILENAME)

    @property
    def _archive_dir(self) -> str:
        return os.path.join(self._workspace_path, "plans", self.ARCHIVE_DIR)

    # ── Bus event handlers ────────────────────────────────────────────────

    def _on_plan_created(self, event: PlanCreated) -> None:
        self._plan_phases[event.plan_id] = event.phase_count
        self._enqueue(
            event.timestamp,
            f'CREATED {event.plan_id[:8]}: "{event.objective}" '
            f"({event.phase_count} phases)",
        )

    def _on_phase_started(self, event: PhaseStarted) -> None:
        total = self._plan_phases.get(event.plan_id, "?")
        self._enqueue(
            event.timestamp,
            f"PHASE {event.phase_index + 1}/{total} STARTED "
            f'{event.plan_id[:8]}: "{event.description}"',
        )

    def _on_phase_completed(self, event: PhaseCompleted) -> None:
        self._plan_phases.get(event.plan_id, "?")
        status = "success" if event.success else f"failed: {event.error or 'unknown'}"
        summary = event.result_summary or ""
        suffix = f" → {summary}" if summary else ""
        self._enqueue(
            event.timestamp,
            f"PHASE {event.plan_id[:8]} COMPLETED: {status}{suffix}",
        )

    def _on_plan_completed(self, event: PlanCompleted) -> None:
        status = "SUCCESS" if event.success else "FAILED"
        self._enqueue(
            event.timestamp,
            f"COMPLETED {event.plan_id[:8]}: {status} "
            f"({event.phases_completed}/{event.phases_total} phases)",
        )
        self._plan_phases.pop(event.plan_id, None)

    def _on_replan(self, event: PlanReplanRequested) -> None:
        self._enqueue(
            event.timestamp,
            f"REPLAN {event.plan_id[:8]}: {event.reason}",
        )

    def _on_plan_restored(self, event: PlanRestored) -> None:
        self._enqueue(
            event.timestamp,
            f'RESTORED {event.plan_id[:8]}: "{event.objective}" '
            f"(phase {event.current_phase_index + 1}, status={event.status})",
        )

    # ── Write queue ───────────────────────────────────────────────────────

    def _enqueue(self, timestamp: float, message: str) -> None:
        t = time.localtime(timestamp)
        line = (
            f"[{t.tm_year}-{t.tm_mon:02d}-{t.tm_mday:02d} "
            f"{t.tm_hour:02d}:{t.tm_min:02d}] {message}\n"
        )
        self._queue.put(line)

    def _writer_loop(self) -> None:
        """Background thread: drains queue and appends to history file."""
        while True:
            # Block until at least one entry
            line = self._queue.get()
            # Drain any additional queued entries
            lines = [line]
            while not self._queue.empty():
                try:
                    lines.append(self._queue.get_nowait())
                except queue.Empty:
                    break

            try:
                plans_dir = os.path.join(self._workspace_path, "plans")
                os.makedirs(plans_dir, exist_ok=True)
                path = self._history_path

                with open(path, "a") as f:
                    f.writelines(lines)

                # Check line count for rolling
                self._maybe_roll(path)
            except OSError as e:
                logger.warning("Plan log write failed: %s", e)

    def _maybe_roll(self, path: str) -> None:
        """Roll the history file if it exceeds MAX_LINES."""
        try:
            with open(path, "r") as f:
                all_lines = f.readlines()

            if len(all_lines) <= self.MAX_LINES:
                return

            # Archive old lines
            os.makedirs(self._archive_dir, exist_ok=True)
            t = time.localtime()
            archive_name = (
                f"history_{t.tm_year}{t.tm_mon:02d}{t.tm_mday:02d}_"
                f"{t.tm_hour:02d}{t.tm_min:02d}{t.tm_sec:02d}.md"
            )
            archive_path = os.path.join(self._archive_dir, archive_name)

            # Move the old content to archive
            with open(archive_path, "w") as f:
                f.writelines(all_lines[:-self.MAX_LINES])

            # Keep only the last MAX_LINES
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                f.writelines(all_lines[-self.MAX_LINES:])
            os.replace(tmp, path)

            logger.debug(
                "Rolled plan history: archived %d lines to %s",
                len(all_lines) - self.MAX_LINES,
                archive_name,
            )
        except OSError as e:
            logger.warning("Plan log roll failed: %s", e)


__all__ = ["PlanLogger"]
