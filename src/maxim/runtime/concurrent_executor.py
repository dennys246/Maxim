"""Concurrent tool execution — ThreadPoolExecutor for parallel batches.

Wraps the existing Executor to support parallel tool batches. Individual
tool execution is unchanged — concurrency only applies when the LLM
explicitly proposes parallel_actions.
"""

from __future__ import annotations

import logging
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from enum import Enum
from typing import Any

from maxim.tools.base import ToolOutput

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Resource Conflict Classification
# ─────────────────────────────────────────────────────────────────────────────


class ConflictType(Enum):
    """Tool conflict profile for parallel execution safety."""

    NONE = "none"  # Pure read-only, always parallelizable
    RESOURCE = "resource"  # Needs exclusive access to a named resource
    GLOBAL = "global"  # Must run alone (e.g., mode change)


TOOL_CONFLICTS: dict[str, ConflictType] = {
    # Always safe to parallelize
    "glob": ConflictType.NONE,
    "read_file": ConflictType.NONE,
    "focus_interests": ConflictType.NONE,
    "respond": ConflictType.NONE,
    "speak": ConflictType.NONE,
    "novelty_track": ConflictType.NONE,
    # Safe if operating on different resources
    "write_file": ConflictType.RESOURCE,
    "execute_file": ConflictType.RESOURCE,
    "move": ConflictType.RESOURCE,
    "track_target": ConflictType.RESOURCE,
    "send_message": ConflictType.RESOURCE,
    "call_user": ConflictType.RESOURCE,
    # Must run alone
    "maxim_command": ConflictType.GLOBAL,
}


def _get_resource_key(action: dict[str, Any]) -> str:
    """Extract the resource key from an action for conflict detection."""
    tool = action.get("tool_name", "")
    raw_params = action.get("params")
    params = raw_params if isinstance(raw_params, dict) else {}

    if tool in ("write_file", "read_file", "execute_file"):
        return f"file:{params.get('path', params.get('file_path', ''))}"
    if tool == "move":
        return "robot:movement"
    if tool == "track_target":
        return "robot:movement"
    if tool == "send_message":
        return f"comms:{params.get('channel', '')}:{params.get('recipient', '')}"
    if tool == "call_user":
        return f"comms:voice:{params.get('recipient', '')}"
    return f"{tool}:default"


def partition_for_concurrency(actions: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Split actions into batches that can safely run in parallel.

    Returns a list of batches — each batch runs concurrently,
    batches run sequentially.
    """
    batches: list[list[dict[str, Any]]] = []
    current_batch: list[dict[str, Any]] = []
    claimed_resources: set[str] = set()

    for action in actions:
        tool = action.get("tool_name", "")
        conflict = TOOL_CONFLICTS.get(tool, ConflictType.RESOURCE)

        if conflict == ConflictType.GLOBAL:
            if current_batch:
                batches.append(current_batch)
            batches.append([action])
            current_batch = []
            claimed_resources = set()

        elif conflict == ConflictType.RESOURCE:
            resource = _get_resource_key(action)
            if resource in claimed_resources:
                batches.append(current_batch)
                current_batch = [action]
                claimed_resources = {resource}
            else:
                current_batch.append(action)
                claimed_resources.add(resource)

        else:  # NONE — always safe
            current_batch.append(action)

    if current_batch:
        batches.append(current_batch)
    return batches


# ─────────────────────────────────────────────────────────────────────────────
# Concurrent Executor
# ─────────────────────────────────────────────────────────────────────────────


class ConcurrentExecutor:
    """Wraps the existing Executor to support parallel tool batches.

    Individual tool execution is unchanged — concurrency only applies
    when the LLM explicitly proposes parallel_actions.

    Note: Executor.execute() returns ToolOutput (raw tool result).
    The agent loop converts these to bus ToolResult objects with
    tool_call_id/tool_name/params before publishing on the bus.
    """

    def __init__(
        self,
        executor: Any,  # runtime.executor.Executor
        max_workers: int = 4,
        per_tool_timeout: float = 30.0,
    ) -> None:
        self.executor = executor
        self.max_workers = max_workers
        self.per_tool_timeout = per_tool_timeout
        self._active_futures: dict[int, Future] = {}

    def execute(self, action: dict[str, Any]) -> ToolOutput:
        """Single tool — delegates directly, no threading overhead."""
        return self.executor.execute(action)

    def execute_batch(self, actions: list[dict[str, Any]]) -> list[ToolOutput]:
        """Parallel tool batch — runs independent tools concurrently.

        Uses per-tool timeout to prevent one slow tool from starving others.
        """
        if len(actions) <= 1:
            return [self.executor.execute(a) for a in actions]

        results: list[ToolOutput | None] = [None] * len(actions)

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            future_to_idx: dict[Future, int] = {
                pool.submit(self.executor.execute, action): i for i, action in enumerate(actions)
            }
            self._active_futures = {i: f for f, i in future_to_idx.items()}

            try:
                for future in as_completed(future_to_idx, timeout=self.per_tool_timeout):
                    idx = future_to_idx[future]
                    try:
                        results[idx] = future.result(timeout=0)
                    except Exception as e:
                        results[idx] = ToolOutput(
                            success=False,
                            error=f"Concurrent execution failed: {e}",
                        )
            except TimeoutError:
                logger.warning("Some tools timed out in concurrent batch")

            self._active_futures.clear()

        # Fill any timed-out futures
        for i, r in enumerate(results):
            if r is None:
                results[i] = ToolOutput(
                    success=False,
                    error=f"Tool execution timed out after {self.per_tool_timeout}s",
                )

        return results  # type: ignore[return-value]

    def cancel_all(self) -> None:
        """Called by GoalAgent._on_preemption to cancel in-flight tools."""
        for idx, future in self._active_futures.items():
            future.cancel()
        self._active_futures.clear()
