from __future__ import annotations

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
    def __init__(self, tools: Iterable[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        if tools:
            for tool in tools:
                self.register(tool)

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"Tool not registered: {name}")
        return self._tools[name]

    def deregister(self, name: str) -> bool:
        """Remove a tool by name. Returns True if found and removed."""
        return self._tools.pop(name, None) is not None

    def list(self) -> list[str]:
        return list(self._tools.keys())

    def find_similar(self, name: str, limit: int = 2) -> list[str]:
        """Find registered tools with names similar to *name*.

        Uses token overlap between the query and both tool names and
        descriptions to surface the most relevant suggestions.
        """
        scored: list[tuple[float, str]] = []
        for tool_name, tool in self._tools.items():
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
