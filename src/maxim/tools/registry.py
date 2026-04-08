from __future__ import annotations

import threading
from typing import Iterable

from .base import Tool


def _token_overlap(a: str, b: str) -> float:
    """Score similarity between two strings using token + substring overlap."""
    a_lower = a.lower().replace("_", " ").replace("-", " ")
    b_lower = b.lower().replace("_", " ").replace("-", " ")

    a_tokens = set(a_lower.split())
    b_tokens = set(b_lower.split())

    # Exact substring match scores highest
    if a_lower in b_lower or b_lower in a_lower:
        return 0.9

    # Token overlap (Jaccard-ish)
    if not a_tokens or not b_tokens:
        return 0.0
    intersection = a_tokens & b_tokens
    union = a_tokens | b_tokens
    return len(intersection) / len(union)


class ToolRegistry:
    """Thread-safe tool registry.

    Uses RLock so methods that call other methods (e.g., register calling
    list for collision detection) don't deadlock.
    """

    def __init__(self, tools: Iterable[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        self._lock = threading.RLock()
        if tools:
            for tool in tools:
                self.register(tool)

    def register(self, tool: Tool) -> None:
        with self._lock:
            self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        with self._lock:
            if name not in self._tools:
                raise KeyError(f"Tool not registered: {name}")
            return self._tools[name]

    def deregister(self, name: str) -> bool:
        """Remove a tool by name. Returns True if found and removed."""
        with self._lock:
            return self._tools.pop(name, None) is not None

    def list(self) -> list[str]:
        with self._lock:
            return list(self._tools.keys())

    def find_similar(self, name: str, limit: int = 2) -> list[str]:
        """Find registered tools with names similar to *name*.

        Uses token overlap between the query and both tool names and
        descriptions to surface the most relevant suggestions.
        """
        with self._lock:
            tools_snapshot = list(self._tools.items())

        scored: list[tuple[float, str]] = []
        for tool_name, tool in tools_snapshot:
            # Score against tool name
            name_score = _token_overlap(name, tool_name)
            # Score against description tokens
            desc = getattr(tool, "description", "") or ""
            desc_score = _token_overlap(name, desc) * 0.5  # weight lower
            score = max(name_score, desc_score)
            if score > 0.05:
                scored.append((score, tool_name))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [t for _, t in scored[:limit]]
