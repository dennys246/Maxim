"""Tests for declarative auto_fire dispatch (W1 sense_tool_registry MVP).

The pre-W1 auto-sense path hardcoded a ``_tool_reg.get("sense_presence")``
lookup at agent_loop.py. Post-W1 it iterates
``_tool_reg.get_auto_fire_tools()`` and dispatches each one via
``tool.execute()``, preserving the load-bearing invariant that auto-fired
output never flows through the Executor (so it never lands in
``actions.jsonl``).

These tests pin the routing contract at the registry level — the
agent-loop integration sits inside a 4,000-line module that is awkward
to test in isolation; the contract that matters is "auto-fire tools
are findable via the registry, and a tool's auto_fire flag determines
whether it gets dispatched implicitly."
"""

from __future__ import annotations

from typing import Any

from maxim.tools.base import Tool, ToolOutput
from maxim.tools.discovery import SensePresenceTool
from maxim.tools.registry import ToolRegistry


class _MinimalEntityMap:
    def list_entities(self):
        return []

    def list_self_entities(self):
        return []

    def list_scene_entities(self):
        return []

    def is_self(self, _entity):
        return False


class _RecordingAutoFireTool(Tool):
    """Auto-fire tool that records every call so the test can assert dispatch."""

    name = "recorder"
    description = "Records auto-fire dispatches"
    input_schema: dict[str, Any] = {}
    auto_fire = True
    kind = "auto-discovery"

    def __init__(self) -> None:
        self.dispatched_calls: list[dict[str, Any]] = []
        super().__init__()

    def execute(self, **kwargs: Any) -> Any:
        self.dispatched_calls.append(dict(kwargs))
        return ToolOutput(success=True, output="recorded")


class _RecordingLLMCallableTool(Tool):
    """Non-auto-fire tool — must NOT be dispatched by the auto-fire loop."""

    name = "llm_only"
    description = "LLM chooses when to call this"
    input_schema: dict[str, Any] = {}
    # auto_fire defaults to False — no override

    def __init__(self) -> None:
        self.dispatched_calls: list[dict[str, Any]] = []
        super().__init__()

    def execute(self, **kwargs: Any) -> Any:
        self.dispatched_calls.append(dict(kwargs))
        return ToolOutput(success=True, output="llm-called")


def test_get_auto_fire_tools_only_returns_auto_fire():
    """The registry's auto_fire enumerator filters by metadata, not name."""
    reg = ToolRegistry()
    rec = _RecordingAutoFireTool()
    llm = _RecordingLLMCallableTool()
    reg.register(rec)
    reg.register(llm)

    auto = reg.get_auto_fire_tools()
    names = {t.name for t in auto}
    assert names == {"recorder"}
    # Both tools are in the registry; the difference is the metadata flag.
    assert set(reg.list_all()) == {"recorder", "llm_only"}


def test_auto_fire_dispatch_simulation_records_only_auto_fire():
    """Simulate the agent-loop auto-fire dispatch and verify routing.

    Mirrors the post-W1 loop body at runtime/agent_loop.py: iterate
    ``get_auto_fire_tools()`` and call ``.execute()`` on each. The LLM
    -callable tool must NOT receive an implicit dispatch.
    """
    reg = ToolRegistry()
    rec = _RecordingAutoFireTool()
    llm = _RecordingLLMCallableTool()
    reg.register(rec)
    reg.register(llm)

    for tool in reg.get_auto_fire_tools():
        tool.execute()

    assert len(rec.dispatched_calls) == 1
    assert llm.dispatched_calls == []


def test_sense_presence_is_auto_fire_in_registry():
    """SensePresenceTool registered into a real registry is enumerated as auto_fire.

    Load-bearing end-to-end check: the canonical auto-discovery tool
    must remain reachable via the metadata-aware path. Regression guard
    against accidentally removing the ``auto_fire`` declaration.
    """
    reg = ToolRegistry()
    sense = SensePresenceTool(entity_map=_MinimalEntityMap())
    reg.register(sense)
    names = {t.name for t in reg.get_auto_fire_tools()}
    assert "sense_presence" in names


def test_legacy_by_name_lookup_still_works():
    """``registry.get("sense_presence")`` still returns the tool.

    The W1 refactor adds a metadata-aware enumeration path; it does
    NOT remove the by-name lookup. Test harnesses, the body-state
    interoception block, and any other call site that knows the tool
    name directly continue to work.
    """
    reg = ToolRegistry()
    sense = SensePresenceTool(entity_map=_MinimalEntityMap())
    reg.register(sense)
    assert reg.get("sense_presence") is sense
