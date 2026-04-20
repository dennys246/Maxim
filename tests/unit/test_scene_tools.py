"""Tests for ToolRegistry scene-scoped activation (I3).

Covers: scene registration, deactivation, re-activation, auto-deactivation
on cap exceeded, list() vs list_all() behaviour, thread safety,
executor active-tool gating, and the prompt builder flow.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from maxim.tools.base import Tool
from maxim.tools.registry import DEFAULT_ACTIVE_TOOL_CAP, ToolRegistry


# ── Test fixtures ───────────────────────────────────────────────────────


class _StubTool(Tool):
    """Minimal concrete Tool for testing."""

    def __init__(self, name: str, description: str = "") -> None:
        self.name = name
        self.description = description
        super().__init__()

    def execute(self, **kwargs):
        return f"{self.name} executed"


def _make_tools(prefix: str, count: int) -> list[Tool]:
    return [_StubTool(f"{prefix}_{i}") for i in range(count)]


# ── Basic scene registration ───────────────────────────────────────────


class TestSceneRegistration:
    def test_register_scene_tools_marks_active(self):
        reg = ToolRegistry()
        tools = _make_tools("sword", 3)
        reg.register_scene_tools(tools, scene_id="scene_1")

        active = reg.list()
        assert all(f"sword_{i}" in active for i in range(3))

    def test_scene_tools_appear_in_list(self):
        reg = ToolRegistry()
        # Core tool
        reg.register(_StubTool("respond"))
        # Scene tools
        tools = _make_tools("gate", 2)
        reg.register_scene_tools(tools, scene_id="dungeon_entrance")

        active = reg.list()
        assert "respond" in active
        assert "gate_0" in active
        assert "gate_1" in active
        assert len(active) == 3

    def test_list_all_includes_everything(self):
        reg = ToolRegistry()
        reg.register(_StubTool("respond"))
        tools = _make_tools("gate", 2)
        reg.register_scene_tools(tools, scene_id="s1")
        reg.deactivate_scene("s1")

        assert len(reg.list()) == 1  # only "respond"
        assert len(reg.list_all()) == 3  # all three

    def test_get_scene_tools(self):
        reg = ToolRegistry()
        tools = _make_tools("axe", 3)
        reg.register_scene_tools(tools, scene_id="forge")

        scene_tools = reg.get_scene_tools("forge")
        assert sorted(scene_tools) == ["axe_0", "axe_1", "axe_2"]

    def test_get_tool_scene(self):
        reg = ToolRegistry()
        reg.register(_StubTool("respond"))
        tools = _make_tools("gate", 1)
        reg.register_scene_tools(tools, scene_id="s1")

        assert reg.get_tool_scene("gate_0") == "s1"
        assert reg.get_tool_scene("respond") is None
        assert reg.get_tool_scene("nonexistent") is None


# ── Deactivation / re-activation ───────────────────────────────────────


class TestSceneDeactivation:
    def test_deactivate_removes_from_active_list(self):
        reg = ToolRegistry()
        reg.register(_StubTool("respond"))
        tools = _make_tools("gate", 3)
        reg.register_scene_tools(tools, scene_id="s1")

        reg.deactivate_scene("s1")

        active = reg.list()
        assert "respond" in active
        assert "gate_0" not in active
        assert len(active) == 1

    def test_deactivate_returns_count(self):
        reg = ToolRegistry()
        tools = _make_tools("torch", 5)
        reg.register_scene_tools(tools, scene_id="cave")

        count = reg.deactivate_scene("cave")
        assert count == 5

    def test_deactivate_nonexistent_scene_returns_zero(self):
        reg = ToolRegistry()
        assert reg.deactivate_scene("nowhere") == 0

    def test_deactivated_tools_still_in_registry(self):
        """Invariant: tools are deactivated, never deleted mid-session."""
        reg = ToolRegistry()
        tools = _make_tools("sword", 2)
        reg.register_scene_tools(tools, scene_id="arena")
        reg.deactivate_scene("arena")

        # Still retrievable by name
        tool = reg.get("sword_0")
        assert tool.name == "sword_0"

        # Still in list_all
        assert "sword_0" in reg.list_all()

    def test_reactivate_scene(self):
        reg = ToolRegistry()
        tools = _make_tools("potion", 3)
        reg.register_scene_tools(tools, scene_id="shop")

        reg.deactivate_scene("shop")
        assert len(reg.list()) == 0

        deactivated = reg.activate_scene("shop")
        assert deactivated == []  # no eviction needed
        assert len(reg.list()) == 3

    def test_reactivate_idempotent(self):
        reg = ToolRegistry()
        tools = _make_tools("bow", 2)
        reg.register_scene_tools(tools, scene_id="forest")

        # Already active — returns empty list
        deactivated = reg.activate_scene("forest")
        assert deactivated == []

    def test_reactivate_enforces_cap(self):
        """Re-activating a scene auto-deactivates the oldest scene if cap exceeded."""
        reg = ToolRegistry(active_tool_cap=5)

        reg.register_scene_tools(_make_tools("s1", 3), scene_id="scene_1")
        time.sleep(0.01)
        reg.register_scene_tools(_make_tools("s2", 3), scene_id="scene_2")
        # scene_1 was auto-deactivated (3 + 3 > 5)

        # Deactivate scene_2, then reactivate scene_1
        reg.deactivate_scene("scene_2")
        reg.activate_scene("scene_1")
        assert "s1_0" in reg.list()

        # Now try reactivating scene_2 — should evict scene_1
        deactivated = reg.activate_scene("scene_2")
        assert "scene_1" in deactivated
        assert "s2_0" in reg.list()
        assert "s1_0" not in reg.list()

    def test_is_tool_active(self):
        reg = ToolRegistry()
        reg.register(_StubTool("respond"))
        tools = _make_tools("gate", 1)
        reg.register_scene_tools(tools, scene_id="s1")

        assert reg.is_tool_active("respond") is True
        assert reg.is_tool_active("gate_0") is True

        reg.deactivate_scene("s1")
        assert reg.is_tool_active("gate_0") is False
        assert reg.is_tool_active("respond") is True

        # Non-existent
        assert reg.is_tool_active("nope") is False

    def test_get_active_scenes(self):
        reg = ToolRegistry()
        reg.register_scene_tools(_make_tools("a", 2), scene_id="s1")
        reg.register_scene_tools(_make_tools("b", 2), scene_id="s2")

        assert reg.get_active_scenes() == ["s1", "s2"]

        reg.deactivate_scene("s1")
        assert reg.get_active_scenes() == ["s2"]

    def test_deregister_cleans_scene_meta(self):
        reg = ToolRegistry()
        tools = _make_tools("key", 1)
        reg.register_scene_tools(tools, scene_id="dungeon")

        reg.deregister("key_0")
        assert "key_0" not in reg.list()
        assert "key_0" not in reg.list_all()
        assert reg.get_tool_scene("key_0") is None


# ── Active tool cap ────────────────────────────────────────────────────


class TestActiveToolCap:
    def test_auto_deactivates_oldest_scene(self):
        reg = ToolRegistry(active_tool_cap=10)

        # Scene 1: 5 tools (registered first = oldest)
        reg.register_scene_tools(_make_tools("s1", 5), scene_id="scene_1")
        time.sleep(0.01)

        # Scene 2: 5 tools (fills cap)
        reg.register_scene_tools(_make_tools("s2", 5), scene_id="scene_2")

        assert len(reg.list()) == 10

        # Scene 3: 5 more tools — should auto-deactivate scene_1
        deactivated = reg.register_scene_tools(_make_tools("s3", 5), scene_id="scene_3")

        assert "scene_1" in deactivated
        active = reg.list()
        assert len(active) <= 10
        assert "s1_0" not in active
        assert "s3_0" in active

    def test_auto_deactivates_multiple_scenes_if_needed(self):
        reg = ToolRegistry(active_tool_cap=12)

        # Fill with 3 small scenes (4 tools each = 12 total, at cap)
        reg.register_scene_tools(_make_tools("s1", 4), scene_id="scene_1")
        time.sleep(0.01)
        reg.register_scene_tools(_make_tools("s2", 4), scene_id="scene_2")
        time.sleep(0.01)
        reg.register_scene_tools(_make_tools("s3", 4), scene_id="scene_3")

        assert len(reg.list()) == 12

        # Register 5 more — needs to evict scene_1 (4 tools) and scene_2 (4 tools)
        # to fit: 12 - 4 - 4 + 5 = 9 <= 12
        deactivated = reg.register_scene_tools(_make_tools("s4", 5), scene_id="scene_4")

        assert "scene_1" in deactivated
        assert "scene_2" in deactivated
        active = reg.list()
        assert "s1_0" not in active
        assert "s2_0" not in active
        assert "s3_0" in active
        assert "s4_0" in active

    def test_core_tools_excluded_from_cap(self):
        """Core tools don't count toward active_tool_cap — cap is scene-only."""
        reg = ToolRegistry(active_tool_cap=5)

        # Register 3 core tools — these don't count toward cap
        for name in ["respond", "think", "say"]:
            reg.register(_StubTool(name))

        # Scene with 5 tools — all fit because cap counts scene tools only
        reg.register_scene_tools(_make_tools("sword", 5), scene_id="s1")

        active = reg.list()
        assert "respond" in active
        assert "think" in active
        assert "say" in active
        assert len(active) == 8  # 3 core + 5 scene

    def test_cap_does_not_deactivate_incoming_scene(self):
        """The scene being registered should never be auto-deactivated."""
        reg = ToolRegistry(active_tool_cap=5)
        reg.register_scene_tools(_make_tools("s1", 5), scene_id="scene_1")
        time.sleep(0.01)

        # Register another scene that exceeds cap — scene_1 should go, not the new one
        deactivated = reg.register_scene_tools(_make_tools("s2", 5), scene_id="scene_2")

        assert "scene_1" in deactivated
        assert "scene_2" not in deactivated
        assert all(f"s2_{i}" in reg.list() for i in range(5))

    def test_default_cap_is_20(self):
        reg = ToolRegistry()
        assert reg._active_tool_cap == DEFAULT_ACTIVE_TOOL_CAP == 20

    def test_custom_cap(self):
        reg = ToolRegistry(active_tool_cap=8)
        assert reg._active_tool_cap == 8

    def test_returns_deactivated_scene_ids(self):
        reg = ToolRegistry(active_tool_cap=3)
        reg.register_scene_tools(_make_tools("s1", 3), scene_id="old")
        time.sleep(0.01)

        deactivated = reg.register_scene_tools(_make_tools("s2", 3), scene_id="new")
        assert deactivated == ["old"]


# ── get_active_tools (Tool objects) ────────────────────────────────────


class TestGetActiveTools:
    def test_returns_tool_objects(self):
        reg = ToolRegistry()
        reg.register(_StubTool("respond"))
        reg.register_scene_tools(_make_tools("axe", 2), scene_id="s1")

        active = reg.get_active_tools()
        assert all(isinstance(t, Tool) for t in active)
        assert len(active) == 3

    def test_excludes_deactivated(self):
        reg = ToolRegistry()
        reg.register_scene_tools(_make_tools("axe", 2), scene_id="s1")
        reg.deactivate_scene("s1")

        assert len(reg.get_active_tools()) == 0


# ── find_similar searches all tools ────────────────────────────────────


class TestFindSimilarWithScenes:
    def test_finds_deactivated_tools(self):
        """find_similar should search ALL tools so the agent can get suggestions
        about tools from prior scenes."""
        reg = ToolRegistry()
        reg.register_scene_tools([_StubTool("sword_slash", "Slash with a sword")], scene_id="s1")
        reg.deactivate_scene("s1")

        results = reg.find_similar("sword", limit=5)
        assert "sword_slash" in results


# ── Thread safety ──────────────────────────────────────────────────────


class TestThreadSafety:
    def test_concurrent_register_and_deactivate(self):
        """Concurrent scene registration + deactivation must not corrupt state."""
        reg = ToolRegistry(active_tool_cap=50)
        errors: list[Exception] = []
        barrier = threading.Barrier(8)

        def register_scene(scene_idx: int):
            try:
                barrier.wait(timeout=5)
                tools = _make_tools(f"scene{scene_idx}", 5)
                reg.register_scene_tools(tools, scene_id=f"scene_{scene_idx}")
            except Exception as e:
                errors.append(e)

        def deactivate_scene(scene_idx: int):
            try:
                barrier.wait(timeout=5)
                time.sleep(0.005)  # let registrations happen first
                reg.deactivate_scene(f"scene_{scene_idx}")
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=8) as pool:
            futs = []
            for i in range(4):
                futs.append(pool.submit(register_scene, i))
            for i in range(4):
                futs.append(pool.submit(deactivate_scene, i))
            for f in as_completed(futs):
                f.result()

        assert not errors, f"Thread safety errors: {errors}"
        # Registry should be in a consistent state
        all_tools = reg.list_all()
        active_tools = reg.list()
        assert len(all_tools) >= len(active_tools)

    def test_concurrent_list_during_mutation(self):
        """list() must not raise during concurrent mutations."""
        reg = ToolRegistry(active_tool_cap=100)
        stop = threading.Event()
        errors: list[Exception] = []

        def mutate():
            try:
                for i in range(50):
                    tools = _make_tools(f"m{i}", 2)
                    reg.register_scene_tools(tools, scene_id=f"ms{i}")
                    if i % 3 == 0:
                        reg.deactivate_scene(f"ms{i}")
            except Exception as e:
                errors.append(e)
            finally:
                stop.set()

        def read():
            try:
                while not stop.is_set():
                    reg.list()
                    reg.list_all()
                    reg.get_active_scenes()
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=4) as pool:
            futs = [pool.submit(mutate)]
            for _ in range(3):
                futs.append(pool.submit(read))
            for f in as_completed(futs):
                f.result()

        assert not errors


# ── Executor active-tool gate ──────────────────────────────────────────


class TestExecutorActiveGate:
    """Verify that Executor.execute() rejects deactivated scene tools."""

    def test_deactivated_tool_rejected(self):
        from maxim.runtime.executor import Executor

        reg = ToolRegistry()
        reg.register_scene_tools([_StubTool("sword_slash")], scene_id="arena")
        reg.deactivate_scene("arena")

        executor = Executor(tool_registry=reg)
        result = executor.execute({"tool_name": "sword_slash", "params": {}})

        assert not result.success
        assert "not active" in result.error
        assert "arena" in result.error

    def test_active_tool_executes(self):
        from maxim.runtime.executor import Executor

        reg = ToolRegistry()
        reg.register_scene_tools([_StubTool("sword_slash")], scene_id="arena")

        executor = Executor(tool_registry=reg)
        result = executor.execute({"tool_name": "sword_slash", "params": {}})

        assert result.success

    def test_core_tool_always_executes(self):
        from maxim.runtime.executor import Executor

        reg = ToolRegistry()
        reg.register(_StubTool("respond"))

        executor = Executor(tool_registry=reg)
        result = executor.execute({"tool_name": "respond", "params": {}})

        assert result.success


# ── Integration: prompt builder flow ───────────────────────────────────


class TestPromptBuilderIntegration:
    """Verify that the tools reaching the prompt respect scene activation.

    The flow is: ToolRegistry.list() → LoopController.get_all_tools() →
    ModeInfo.get_available_tools() → LLMRequest.available_tools →
    build_tools_section(). We test the first link — registry.list()
    filtering — which drives everything downstream.
    """

    def test_25_tools_across_5_scenes_cap_at_20(self):
        """Register 25 tools across 5 scenes (5 each).
        With default cap=20, the oldest scene should auto-deactivate.
        """
        reg = ToolRegistry()  # default cap = 20

        for i in range(5):
            tools = _make_tools(f"scene{i}", 5)
            reg.register_scene_tools(tools, scene_id=f"scene_{i}")
            time.sleep(0.01)  # ensure ordering

        active = reg.list()
        # 25 > 20 cap, so at least the oldest scene (scene_0) should be deactivated
        assert len(active) <= 20
        # Oldest scene's tools should not be active
        assert "scene0_0" not in active
        # Newest scene's tools should be active
        assert "scene4_0" in active

    def test_deactivated_tools_excluded_from_available(self):
        """Simulates the LoopController.get_all_tools() call path."""
        reg = ToolRegistry()
        reg.register(_StubTool("respond"))
        reg.register(_StubTool("think"))
        reg.register_scene_tools(_make_tools("sword", 3), scene_id="arena")
        reg.register_scene_tools(_make_tools("potion", 2), scene_id="shop")

        # Simulate LoopController.get_all_tools():
        available = set(reg.list())
        assert "respond" in available
        assert "sword_0" in available
        assert "potion_0" in available

        # Scene transition: deactivate arena
        reg.deactivate_scene("arena")

        available = set(reg.list())
        assert "respond" in available
        assert "sword_0" not in available
        assert "potion_0" in available

    def test_full_lifecycle(self):
        """Enter scene → use tools → transition → return → tools reappear."""
        reg = ToolRegistry()
        reg.register(_StubTool("respond"))

        # Enter dungeon
        reg.register_scene_tools(_make_tools("gate", 3), scene_id="dungeon")
        assert "gate_0" in reg.list()

        # Move to shop (dungeon tools deactivate)
        reg.deactivate_scene("dungeon")
        reg.register_scene_tools(_make_tools("buy", 2), scene_id="shop")
        assert "gate_0" not in reg.list()
        assert "buy_0" in reg.list()

        # Return to dungeon (re-activate)
        reg.deactivate_scene("shop")
        reg.activate_scene("dungeon")
        assert "gate_0" in reg.list()
        assert "buy_0" not in reg.list()

        # Core tool always present
        assert "respond" in reg.list()
