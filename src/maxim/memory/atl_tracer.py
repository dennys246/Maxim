"""ATLTracer — real-time color-coded tracing of ATL semantic memory.

Subscribes to ATL callbacks and wraps recall/store methods to trace
concept creation, recall, deletion, and cross-layer promotion.

Activated by --debug atl or --debug all, or MAXIM_ATL_TRACE=1 env var.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from maxim.memory.atl import ATL
    from maxim.memory.semantic_types import SemanticMemory

logger = logging.getLogger(__name__)

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
    return time.strftime("%H:%M:%S")


def _short(s: str, n: int = 40) -> str:
    if len(s) > n:
        return s[: n - 3] + "..."
    return s


class ATLTracer:
    """Real-time tracer for ATL semantic memory operations."""

    def __init__(self, atl: ATL) -> None:
        self._atl = atl
        self._lock = threading.Lock()
        self._store_count = 0
        self._recall_count = 0
        self._delete_count = 0

        atl.register_capture_callback(self._on_concept_stored)
        atl.register_deletion_callback(self._on_concept_deleted)
        self._wrap_recall(atl)
        self._wrap_store(atl)

        self._print(f"{_BOLD}[ATL TRACER]{_RESET} Active — tracing concept store, recall, deletion")

    def _print(self, msg: str) -> None:
        with self._lock:
            print(msg, file=sys.stderr, flush=True)

    def _on_concept_stored(self, concept_id: str, concept: SemanticMemory) -> None:
        self._store_count += 1
        name = getattr(concept, "name", "") or getattr(concept, "concept_name", "") or concept_id
        category = getattr(concept, "category", "")
        self._print(
            f"{_GREEN}{_ts()} [ATL STORE ]{_RESET} "
            f"#{self._store_count} {_short(str(name))}" + (f" {_DIM}cat={category}{_RESET}" if category else "")
        )

    def _on_concept_deleted(self, concept_id: str) -> None:
        self._delete_count += 1
        self._print(f"{_RED}{_ts()} [ATL DELETE]{_RESET} #{self._delete_count} {concept_id[:12]}")

    def _wrap_store(self, atl: ATL) -> None:
        original = atl.store

        def traced(record: Any, **kwargs: Any) -> str:
            result = original(record, **kwargs)
            return result

        atl.store = traced  # type: ignore[assignment]

    def _wrap_recall(self, atl: ATL) -> None:
        original = atl.recall

        def traced(**kwargs: Any) -> list:
            results = original(**kwargs)
            self._recall_count += 1

            query_parts = []
            for k, v in kwargs.items():
                if v is not None and k != "limit":
                    query_parts.append(f"{k}={_short(str(v), 15)}")
            query_str = ", ".join(query_parts) or "all"

            self._print(
                f"{_BLUE}{_ts()} [ATL RECALL]{_RESET} "
                f"#{self._recall_count} "
                f"query=({query_str}) "
                f"→ {len(results)} result(s)"
            )

            for concept in results[:3]:
                cid = getattr(concept, "id", "?")[:12]
                name = getattr(concept, "name", "") or getattr(concept, "concept_name", "") or cid
                self._print(f"  {_DIM}  └ {_short(str(name))}{_RESET}")

            return results

        atl.recall = traced  # type: ignore[assignment]

    def summary(self) -> str:
        concept_count = len(self._atl) if hasattr(self._atl, "__len__") else "?"
        return (
            f"ATL trace: {self._store_count} stored, "
            f"{self._recall_count} recalls, "
            f"{self._delete_count} deleted, "
            f"{concept_count} concepts alive"
        )

    def print_summary(self) -> None:
        self._print(f"\n{_BOLD}[ATL SUMMARY]{_RESET} {self.summary()}")
