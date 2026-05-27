"""Tests for Tool metadata fields and ToolRegistry helpers (W1 sense_tool_registry MVP).

Pins the metadata contract:

- ``Tool.auto_fire`` defaults to False, ``Tool.kind`` defaults to
  ``"core-universal"`` so existing third-party Tool subclasses are
  unaffected.
- ``SensePresenceTool`` declares ``auto_fire=True, kind="auto-discovery"``
  (the declarative replacement for the agent-loop hardcoded
  ``sense_presence`` lookup).
- SEM-derived factories declare ``kind="sem-modulator-derived"`` so the
  prompt builder can grayscale them when inactive but substrate-biased.
- ``ToolRegistry.register(kind=...)`` overrides the instance's kind at
  registration time.
- ``ToolRegistry.get_auto_fire_tools()`` and ``.get_tools_by_kind()``
  enumerate by the new metadata.
"""

from __future__ import annotations

from typing import Any

from maxim.tools.base import Tool, ToolOutput
from maxim.tools.discovery import SensePresenceTool, SenseToolsTool, UniversalSenseTool
from maxim.tools.registry import ToolRegistry


class _DefaultsTool(Tool):
    """Subclass that declares no metadata — exercises the Tool defaults."""

    name = "defaults"
    description = "Smoke-test tool"
    input_schema: dict[str, Any] = {}

    def execute(self, **kwargs: Any) -> Any:
        return ToolOutput(success=True, output="ok")


def test_tool_default_auto_fire_is_false():
    """A Tool subclass that declares no metadata stays non-auto-fire.

    Load-bearing: existing third-party tools must not start firing
    implicitly after the W1 metadata addition.
    """
    tool = _DefaultsTool()
    assert tool.auto_fire is False


def test_tool_default_kind_is_core_universal():
    """A Tool subclass that declares no metadata is treated as a core tool.

    Load-bearing: existing third-party tools keep the historical
    register()-as-core semantics and stay exempt from scene-scope +
    LRU + grayscale handling.
    """
    tool = _DefaultsTool()
    assert tool.kind == "core-universal"


def test_sense_presence_tool_declares_auto_fire():
    """SensePresenceTool is the canonical auto-fire tool today.

    Pins the declarative metadata that replaces the pre-W1 hardcoded
    ``sense_presence`` lookup in the agent loop's auto-sense dispatch.
    """

    class _StubEntityMap:
        def list_entities(self):
            return []

        def list_self_entities(self):
            return []

        def list_scene_entities(self):
            return []

        def is_self(self, _entity):
            return False

    tool = SensePresenceTool(entity_map=_StubEntityMap())
    assert tool.auto_fire is True
    assert tool.kind == "auto-discovery"


def test_universal_sense_tool_keeps_default_kind():
    """UniversalSenseTool is LLM-callable, not auto-fire — stays core-universal.

    Sanity check that we did not over-tag the sense family.
    """

    class _StubEntityMap:
        def list_names(self):
            return []

        def resolve(self, _name):
            return None

    tool = UniversalSenseTool(entity_map=_StubEntityMap())
    assert tool.auto_fire is False
    assert tool.kind == "core-universal"


def test_sense_tools_tool_keeps_default_kind():
    """SenseToolsTool is LLM-callable, not auto-fire — stays core-universal."""

    class _StubEntityMap:
        def list_self_entities(self):
            return []

        def list_scene_entities(self):
            return []

    tool = SenseToolsTool(entity_map=_StubEntityMap(), tool_registry=ToolRegistry())
    assert tool.auto_fire is False
    assert tool.kind == "core-universal"


def test_sem_derived_tools_declare_sem_kind():
    """SEM-derived factory tools declare kind=sem-modulator-derived.

    These are the only tools eligible for grayscale visibility in the
    prompt builder (per the W1 MVP scope). Built via the same YAML-parser
    factory used in production rather than hand-rolling Sensor/Modulator
    subclasses so the test stays robust to internal refactors.
    """
    from maxim.embodiment.spec import _parse_entity
    from maxim.embodiment.tool_bridge import (
        EntitySenseTool,
        ModulatorAffordanceTool,
        SensorReadTool,
    )

    entity = _parse_entity(
        {
            "name": "probe",
            "entity_type": "test",
            "sensors": {
                "temp": {"unit": "celsius", "range": [0, 100], "initial": 20.0},
            },
            "modulators": {
                "actuator": {
                    "affordances": {
                        "do_thing": {
                            "params": {},
                            "description": "smoke test affordance",
                        },
                    },
                },
            },
        }
    )

    sensor = entity.sensors["temp"]
    mod = entity.modulators["actuator"]
    schema = mod.affordances["do_thing"]

    sr = SensorReadTool(entity, sensor, "read_probe_temp")
    es = EntitySenseTool(entity, "sense_probe")
    ma = ModulatorAffordanceTool(entity, mod, "do_thing", schema, "probe_do_thing")

    assert sr.kind == "sem-modulator-derived"
    assert es.kind == "sem-modulator-derived"
    assert ma.kind == "sem-modulator-derived"
    # auto_fire stays default False — SEM-derived tools are LLM-callable.
    assert sr.auto_fire is False
    assert es.auto_fire is False
    assert ma.auto_fire is False


def test_registry_register_preserves_declared_kind():
    """ToolRegistry.register() without kind= leaves the tool's kind alone."""
    reg = ToolRegistry()
    tool = _DefaultsTool()
    reg.register(tool)
    assert reg.get("defaults").kind == "core-universal"


def test_registry_register_kind_override():
    """ToolRegistry.register(kind=...) overrides the instance's kind.

    Backstop for registration sites that have more authoritative
    provenance than the tool class default.
    """
    reg = ToolRegistry()
    tool = _DefaultsTool()
    reg.register(tool, kind="scene-scoped")
    assert reg.get("defaults").kind == "scene-scoped"


def test_registry_get_auto_fire_tools():
    """get_auto_fire_tools() returns only tools declaring auto_fire=True."""

    class _AutoFire(Tool):
        name = "auto"
        description = "fires every tick"
        input_schema: dict[str, Any] = {}
        auto_fire = True
        kind = "auto-discovery"

        def execute(self, **kwargs: Any) -> Any:
            return ToolOutput(success=True, output="auto")

    reg = ToolRegistry()
    reg.register(_DefaultsTool())
    reg.register(_AutoFire())

    auto = reg.get_auto_fire_tools()
    assert [t.name for t in auto] == ["auto"]


def test_registry_get_tools_by_kind_includes_deactivated():
    """get_tools_by_kind enumerates regardless of active state.

    The grayscale render path needs to see SEM-derived tools even when
    deactivated (that's the whole point — making inactive-but-knowable
    tools visible to the LLM).
    """
    from maxim.embodiment.spec import _parse_entity
    from maxim.embodiment.tool_bridge import EntitySenseTool

    entity = _parse_entity(
        {
            "name": "probe",
            "entity_type": "test",
            "sensors": {
                "temp": {"unit": "celsius", "range": [0, 100], "initial": 20.0},
            },
        }
    )
    sem_tool = EntitySenseTool(entity, "sense_probe")

    reg = ToolRegistry()
    reg.register(_DefaultsTool())  # core-universal
    reg.register_scene_tools([sem_tool], scene_id="scene_a")
    reg.deactivate_scene("scene_a")

    # Even after deactivation, get_tools_by_kind still finds the SEM tool.
    sem_tools = reg.get_tools_by_kind("sem-modulator-derived")
    assert [t.name for t in sem_tools] == ["sense_probe"]
    assert not reg.is_tool_active("sense_probe")
