from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Iterable

from .base import Tool

logger = logging.getLogger(__name__)

# Default cap on simultaneously active *scene* tools to prevent prompt overflow.
# Core tools (registered via register()) are excluded from this count.
DEFAULT_ACTIVE_TOOL_CAP = 20


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


@dataclass
class _SceneMeta:
    """Per-tool scene metadata (internal)."""

    scene_id: str
    registered_at: float = field(default_factory=time.monotonic)
    active: bool = True


class ToolRegistry:
    """Thread-safe tool registry with scene-scoped activation.

    Uses RLock so methods that call other methods (e.g., register calling
    list for collision detection) don't deadlock.

    Scene-scoped tools can be activated/deactivated by scene transition.
    Deactivated tools remain in the registry for re-activation — they are
    never deleted mid-session.  An active tool cap (applied to *scene*
    tools only — core tools are exempt) prevents prompt overflow in long
    campaigns; when registering a new scene would exceed the cap, the
    oldest scene is auto-deactivated first.
    """

    def __init__(
        self,
        tools: Iterable[Tool] | None = None,
        *,
        active_tool_cap: int = DEFAULT_ACTIVE_TOOL_CAP,
    ) -> None:
        self._tools: dict[str, Tool] = {}
        self._scene_meta: dict[str, _SceneMeta] = {}
        self._active_tool_cap = active_tool_cap
        self._lock = threading.RLock()
        if tools:
            for tool in tools:
                self.register(tool)

    # ── Core registration ───────────────────────────────────────────────

    def register(self, tool: Tool, *, kind: str | None = None) -> None:
        """Register a tool. Optionally override its declared ``kind``.

        ``kind`` is a backstop for callers that wrap an existing Tool
        instance whose own ``kind`` is wrong for this registration site
        (e.g., wiring an SEM-derived tool as a core tool in a test
        harness). When ``kind`` is ``None`` (default), the tool's
        declared ``kind`` class attribute is preserved untouched.

        Tools registered via plain ``register()`` have no scene scope
        and are always active (the historical "core tool" semantics).
        Scene-scoped lifecycle is handled separately via
        ``register_scene_tools``.

        The override uses ``object.__setattr__`` so frozen-dataclass
        third-party ``Tool`` subclasses (rare today but possible per
        the plan's Open Question #3) survive the assignment. Plain
        ``tool.kind = kind`` would raise ``FrozenInstanceError`` on a
        ``@dataclass(frozen=True)`` subclass.
        """
        with self._lock:
            if kind is not None:
                # Override the instance's kind — useful when a registration
                # site has more authoritative provenance than the tool's
                # class default. Use object.__setattr__ for frozen-
                # subclass safety.
                object.__setattr__(tool, "kind", kind)
            self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        with self._lock:
            if name not in self._tools:
                raise KeyError(f"Tool not registered: {name}")
            return self._tools[name]

    def deregister(self, name: str) -> bool:
        """Remove a tool by name. Returns True if found and removed."""
        with self._lock:
            self._scene_meta.pop(name, None)
            return self._tools.pop(name, None) is not None

    # ── Listing ─────────────────────────────────────────────────────────

    def list(self) -> list[str]:
        """Return names of all **active** tools.

        Tools deactivated by scene transition are excluded.  Core tools
        (registered via ``register()``) are always active.
        """
        with self._lock:
            return [name for name in self._tools if name not in self._scene_meta or self._scene_meta[name].active]

    def list_all(self) -> list[str]:
        """Return names of ALL tools, including deactivated scene tools."""
        with self._lock:
            return list(self._tools.keys())

    # ── Scene-scoped registration ───────────────────────────────────────

    def register_scene_tools(
        self,
        tools: list[Tool],
        scene_id: str,
    ) -> list[str]:
        """Register tools under a scene scope.

        All tools are marked active.  If the active *scene* tool count
        would exceed ``active_tool_cap``, the oldest active scene is
        auto-deactivated first.  Core tools are excluded from the cap.

        Returns list of auto-deactivated scene IDs (empty if none).
        """
        with self._lock:
            return self._register_scene_locked(tools, scene_id)

    def deactivate_scene(self, scene_id: str) -> int:
        """Deactivate all tools belonging to *scene_id*.

        Returns the number of tools deactivated.
        """
        with self._lock:
            return self._deactivate_scene_locked(scene_id)

    def deactivate_tool(self, name: str) -> bool:
        """Deactivate a single tool (set active=False).

        Per-tool deactivation for LRU eviction.  Scenes remain the
        unit of *bulk* activation; this method enables individual
        deactivation for tools that haven't been used recently.

        Returns True if the tool was found and deactivated.
        """
        with self._lock:
            meta = self._scene_meta.get(name)
            if meta is not None and meta.active:
                meta.active = False
                return True
            return False

    def activate_scene(self, scene_id: str) -> list[str]:
        """Re-activate all tools belonging to *scene_id*.

        Enforces the active tool cap — if re-activation would exceed the
        cap, the oldest *other* active scene is auto-deactivated first.

        Returns list of auto-deactivated scene IDs (empty if none).
        """
        with self._lock:
            # Count how many tools this scene would add.
            scene_tool_count = sum(
                1 for meta in self._scene_meta.values() if meta.scene_id == scene_id and not meta.active
            )
            if scene_tool_count == 0:
                return []

            deactivated: list[str] = []
            active_scene_count = self._count_active_scene_tools()
            would_be = active_scene_count + scene_tool_count

            while would_be > self._active_tool_cap and self._active_tool_cap > 0:
                oldest = self._find_oldest_active_scene(exclude=scene_id)
                if oldest is None:
                    break
                n = self._deactivate_scene_locked(oldest)
                would_be -= n
                deactivated.append(oldest)
                logger.info(
                    "Deactivated scene %r tools (%d tools) during re-activation to stay within active tool budget (%d)",
                    oldest,
                    n,
                    self._active_tool_cap,
                )

            # Now activate.
            for meta in self._scene_meta.values():
                if meta.scene_id == scene_id and not meta.active:
                    meta.active = True

            return deactivated

    def get_active_tools(self) -> list[Tool]:
        """Return Tool objects for all active tools."""
        with self._lock:
            return [
                self._tools[name]
                for name in self._tools
                if name not in self._scene_meta or self._scene_meta[name].active
            ]

    def get_scene_tools(self, scene_id: str) -> list[str]:
        """Return tool names belonging to a specific scene."""
        with self._lock:
            return [name for name, meta in self._scene_meta.items() if meta.scene_id == scene_id]

    def get_active_scenes(self) -> list[str]:
        """Return scene IDs that have at least one active tool."""
        with self._lock:
            scenes: set[str] = set()
            for meta in self._scene_meta.values():
                if meta.active:
                    scenes.add(meta.scene_id)
            return sorted(scenes)

    def get_tool_scene(self, tool_name: str) -> str | None:
        """Return the scene_id for a tool, or None if it's a core tool."""
        with self._lock:
            meta = self._scene_meta.get(tool_name)
            return meta.scene_id if meta is not None else None

    def is_tool_active(self, tool_name: str) -> bool:
        """Check whether a tool is currently active."""
        with self._lock:
            if tool_name not in self._tools:
                return False
            meta = self._scene_meta.get(tool_name)
            return meta is None or meta.active

    # ── Metadata-aware enumeration (W1 sense_tool_registry MVP) ─────────

    def get_auto_fire_tools(self) -> list[Tool]:
        """Return all registered tools with ``auto_fire=True``.

        Used by the agent loop's auto-sense dispatch to drive tools
        whose ``execute()`` runs implicitly each tick (the result is
        injected into the next prompt as passive perception rather than
        logged as a chosen action). The dispatcher's bypass of
        ``actions.jsonl`` is gated on this metadata, replacing the
        pre-W1 hardcoded ``sense_presence`` lookup. See
        [docs/plans/sense_tool_registry.md].
        """
        with self._lock:
            return [t for t in self._tools.values() if getattr(t, "auto_fire", False)]

    def get_tools_by_kind(self, kind: str) -> list[Tool]:
        """Return all registered tools whose ``kind`` matches.

        Includes both active and deactivated tools — the caller decides
        whether to filter further by ``is_tool_active``. Used today by
        the prompt builder to find SEM-derived tools that are knowable
        but inactive (grayscale visibility).
        """
        with self._lock:
            return [t for t in self._tools.values() if getattr(t, "kind", "core-universal") == kind]

    # ── Search ──────────────────────────────────────────────────────────

    def find_similar(self, name: str, limit: int = 2) -> list[str]:
        """Find registered tools with names similar to *name*.

        Searches ALL tools (including deactivated) so the agent can get
        suggestions about tools it may have used in prior scenes.
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

    # ── Private helpers ─────────────────────────────────────────────────

    def _register_scene_locked(self, tools: list[Tool], scene_id: str) -> list[str]:
        """Register scene tools with cap enforcement. Caller MUST hold self._lock."""
        deactivated_scenes: list[str] = []

        active_scene_count = self._count_active_scene_tools()
        would_be = active_scene_count + len(tools)

        # Auto-deactivate oldest scenes until we're under cap.
        while would_be > self._active_tool_cap and self._active_tool_cap > 0:
            oldest_scene = self._find_oldest_active_scene(exclude=scene_id)
            if oldest_scene is None:
                break  # No more scenes to deactivate
            n_deactivated = self._deactivate_scene_locked(oldest_scene)
            would_be -= n_deactivated
            deactivated_scenes.append(oldest_scene)
            logger.info(
                "Deactivated scene %r tools (%d tools) to stay within active tool budget (%d)",
                oldest_scene,
                n_deactivated,
                self._active_tool_cap,
            )

        if would_be > self._active_tool_cap and self._active_tool_cap > 0:
            logger.warning(
                "Active scene tool count (%d) exceeds cap (%d) — no more scenes to evict",
                would_be,
                self._active_tool_cap,
            )

        # Register the new tools.
        now = time.monotonic()
        for tool in tools:
            self._tools[tool.name] = tool
            self._scene_meta[tool.name] = _SceneMeta(
                scene_id=scene_id,
                registered_at=now,
                active=True,
            )

        return deactivated_scenes

    def _count_active_scene_tools(self) -> int:
        """Count active tools that belong to a scene (excludes core tools).

        Caller MUST hold self._lock.
        """
        return sum(1 for meta in self._scene_meta.values() if meta.active)

    def _deactivate_scene_locked(self, scene_id: str) -> int:
        """Deactivate tools for a scene. Caller MUST hold self._lock."""
        count = 0
        for meta in self._scene_meta.values():
            if meta.scene_id == scene_id and meta.active:
                meta.active = False
                count += 1
        return count

    def _find_oldest_active_scene(self, *, exclude: str | None = None) -> str | None:
        """Find the active scene with the earliest registration timestamp.

        Caller MUST hold self._lock.
        """
        oldest_time = float("inf")
        oldest_scene: str | None = None
        for meta in self._scene_meta.values():
            if not meta.active:
                continue
            if meta.scene_id == exclude:
                continue
            if meta.registered_at < oldest_time:
                oldest_time = meta.registered_at
                oldest_scene = meta.scene_id
        return oldest_scene
