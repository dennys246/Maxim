"""HippocampusTracer — real-time color-coded tracing of Hippocampus operations.

Subscribes to Hippocampus callbacks and wraps recall methods to produce
a live trace of memory capture, association formation, recall, and
consolidation events. Designed for watching experiments in real time.

Activated by --debug-hippo CLI flag or MAXIM_HIPPO_TRACE=1 env var.
Output goes to stderr so it doesn't interfere with sim output.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from maxim.memory.hippocampus import Hippocampus
    from maxim.memory.types import EpisodicMemory

logger = logging.getLogger(__name__)

# ANSI colors
_GREEN = "\033[32m"
_BLUE = "\033[34m"
_CYAN = "\033[36m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_MAGENTA = "\033[35m"
_DIM = "\033[90m"
_BOLD = "\033[1m"
_RESET = "\033[0m"


def _ts() -> str:
    """Compact timestamp for trace lines."""
    return time.strftime("%H:%M:%S")


def _short_id(memory_id: str) -> str:
    """Truncate memory ID for readability."""
    return memory_id[:10] if memory_id else "?"


def _excerpt(text: str | None, max_len: int = 60) -> str:
    """Truncate text for trace output."""
    if not text:
        return ""
    text = text.replace("\n", " ").strip()
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


class HippocampusTracer:
    """Real-time tracer for Hippocampus operations.

    Hooks into capture, recall, association, consolidation, and eviction
    events. All output goes to stderr with color coding.
    """

    def __init__(self, hippocampus: Hippocampus) -> None:
        self._hippo = hippocampus
        self._lock = threading.Lock()
        self._capture_count = 0
        self._recall_count = 0
        self._association_count = 0
        self._start_time = time.time()

        # Register callbacks
        hippocampus.register_capture_callback(self._on_capture)
        hippocampus.register_compression_callback(self._on_compressed)
        hippocampus._on_memory_deleted.append(self._on_deleted)

        # Wrap recall methods to trace them
        self._wrap_recall(hippocampus)
        self._wrap_recall_associated(hippocampus)

        self._print(f"{_BOLD}[HIPPO TRACER]{_RESET} Active — tracing capture, recall, associations, consolidation")

    def _print(self, msg: str) -> None:
        """Thread-safe print to stderr."""
        with self._lock:
            print(msg, file=sys.stderr, flush=True)

    def _elapsed(self) -> str:
        """Seconds since tracer started."""
        return f"{time.time() - self._start_time:.1f}s"

    # ── Capture ──────────────────────────────────────────────────────────

    def _on_capture(self, memory_id: str, memory: EpisodicMemory) -> None:
        self._capture_count += 1

        # Extract key fields
        goal = _excerpt(memory.context.active_goal, 40) if memory.context else ""
        tool = memory.action.tool_name if memory.action else ""
        success = memory.outcome.success if memory.outcome else None
        salience = memory.perception.salience if memory.perception else 0
        novelty = memory.perception.novelty if memory.perception else 0
        cli = _excerpt(memory.perception.cli_input, 50) if memory.perception else ""

        # Check for associations formed (graph edges from this memory)
        edge_count = 0
        try:
            edges = self._hippo._graph.get_outgoing(memory_id)
            edge_count = len(edges) if edges else 0
        except Exception:
            pass  # graph may not be initialized yet — display best-effort

        success_str = f"{'✓' if success else '✗'}" if success is not None else "?"

        self._print(
            f"{_GREEN}{_ts()} [CAPTURE ]{_RESET} "
            f"{_DIM}#{self._capture_count}{_RESET} "
            f"{_short_id(memory_id)} "
            f"sal={salience:.2f} nov={novelty:.2f} "
            f"{success_str} "
            f"{_CYAN}{tool or 'obs'}{_RESET}"
            + (f" edges={edge_count}" if edge_count > 0 else "")
            + (f" {_DIM}goal={goal}{_RESET}" if goal else "")
            + (f" {_DIM}cli={cli}{_RESET}" if cli else "")
        )

    # ── Associations ─────────────────────────────────────────────────────

    def _trace_associations(self, memory_id: str, edges: list[tuple[str, float]]) -> None:
        """Called after _form_associations to trace new edges."""
        for target_id, weight in edges:
            self._association_count += 1
            self._print(
                f"{_MAGENTA}{_ts()} [ASSOC   ]{_RESET} "
                f"{_short_id(memory_id)} ↔ {_short_id(target_id)} "
                f"weight={weight:.3f}"
            )

    # ── Recall ───────────────────────────────────────────────────────────

    def _wrap_recall(self, hippo: Hippocampus) -> None:
        """Wrap recall() to trace queries and results."""
        original = hippo.recall

        def traced_recall(**kwargs: Any) -> list:
            results = original(**kwargs)
            self._recall_count += 1

            # Build query description
            query_parts = []
            for k, v in kwargs.items():
                if v is not None and k != "limit":
                    query_parts.append(f"{k}={_excerpt(str(v), 20)}")
            query_str = ", ".join(query_parts) or "all"

            self._print(
                f"{_BLUE}{_ts()} [RECALL  ]{_RESET} "
                f"#{self._recall_count} "
                f"query=({query_str}) "
                f"→ {len(results)} result(s)"
            )

            for mem in results[:3]:
                mid = _short_id(mem.id)
                tool = getattr(mem, "action", None)
                tool_name = tool.tool_name if tool else "?"
                self._print(f"  {_DIM}  └ {mid} {tool_name}{_RESET}")

            return results

        hippo.recall = traced_recall  # type: ignore[assignment]

    def _wrap_recall_associated(self, hippo: Hippocampus) -> None:
        """Wrap recall_associated() to trace spreading activation."""
        original = hippo.recall_associated

        def traced_recall_associated(seed_ids: list[str], limit: int = 10, **kwargs: Any) -> list[tuple[Any, float]]:
            results = original(seed_ids, limit=limit, **kwargs)

            seed_str = ", ".join(_short_id(s) for s in seed_ids)
            self._print(f"{_CYAN}{_ts()} [SPREAD  ]{_RESET} seeds=[{seed_str}] → {len(results)} activated")

            for mem, score in results[:5]:
                mid = _short_id(mem.id)
                tool = getattr(mem, "action", None)
                tool_name = tool.tool_name if tool else "?"
                self._print(f"  {_DIM}  └ {mid} score={score:.4f} {tool_name}{_RESET}")

            return results

        hippo.recall_associated = traced_recall_associated  # type: ignore[assignment]

    # ── Consolidation ────────────────────────────────────────────────────

    def _on_compressed(self, memory_id: str) -> None:
        self._print(f"{_YELLOW}{_ts()} [COMPRESS]{_RESET} {_short_id(memory_id)} → CompressedMemory")

    def _on_deleted(self, memory_id: str) -> None:
        self._print(f"{_RED}{_ts()} [EVICT   ]{_RESET} {_short_id(memory_id)} removed")

    # ── Summary ──────────────────────────────────────────────────────────

    def summary(self) -> str:
        """Return a compact summary of traced operations."""
        mem_count = len(self._hippo)
        elapsed = time.time() - self._start_time
        return (
            f"Hippo trace: {self._capture_count} captures, "
            f"{self._recall_count} recalls, "
            f"{self._association_count} associations, "
            f"{mem_count} memories alive, "
            f"{elapsed:.1f}s elapsed"
        )

    def print_summary(self) -> None:
        self._print(f"\n{_BOLD}[HIPPO SUMMARY]{_RESET} {self.summary()}")

        # Graph stats
        try:
            total_edges = sum(len(edges) for edges in self._hippo._graph._outgoing.values())
            self._print(f"  {_DIM}Graph: {len(self._hippo._graph._outgoing)} nodes, {total_edges} edges{_RESET}")
        except Exception:
            pass
