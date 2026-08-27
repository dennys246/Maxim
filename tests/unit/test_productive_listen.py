"""Guards for Phase 1 of the productive orienting affordance.

`listen` becomes goal-advancing: the audio-orient channel world-sets the body's
`azimuth` root sensor (agent_loop §1.16 via world_set_azimuth), and `listen`
reads it back in its output so the agent can act on the direction. Capability-
driven: any body that DECLARES an `azimuth` root sensor gets the behavior, no
per-body code. Pins the round-trip for both shipped bodies + the fail-soft gate.
See docs/plans/deferred/productive_orienting_affordance.md.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from maxim.embodiment.audio_localization import world_set_azimuth

_BODIES = Path("src/maxim/_data/components/bodies")


class _Root:
    def __init__(self, sensors: dict, vital: dict) -> None:
        self.sensors = sensors
        self.vital_metrics = vital


class _Emb:
    def __init__(self, root) -> None:
        self.root = root


# ── world_set_azimuth: capability gate + clamp ──────────────────────────────


def test_world_set_azimuth_writes_when_body_has_sensor():
    emb = _Emb(_Root(sensors={"azimuth": object()}, vital={"azimuth": 0.0}))
    assert world_set_azimuth(emb, -0.6) is True
    assert emb.root.vital_metrics["azimuth"] == -0.6


def test_world_set_azimuth_clamps_to_unit_range():
    emb = _Emb(_Root(sensors={"azimuth": object()}, vital={"azimuth": 0.0}))
    world_set_azimuth(emb, 5.0)
    assert emb.root.vital_metrics["azimuth"] == 1.0
    world_set_azimuth(emb, -5.0)
    assert emb.root.vital_metrics["azimuth"] == -1.0


def test_world_set_azimuth_fail_soft_without_sensor():
    """A body that doesn't declare an azimuth sensor is simply unaffected."""
    emb = _Emb(_Root(sensors={"stamina": object()}, vital={"stamina": 1.0}))
    assert world_set_azimuth(emb, -0.6) is False
    assert "azimuth" not in emb.root.vital_metrics


def test_world_set_azimuth_none_embodiment_is_false():
    assert world_set_azimuth(None, -0.6) is False
    assert world_set_azimuth(_Emb(None), -0.6) is False


# ── end-to-end: world-set azimuth → listen reads it back ────────────────────


def _load_entity(body: str):
    from maxim.embodiment.spec import _parse_entity

    doc = yaml.safe_load((_BODIES / f"{body}.yaml").read_text())
    return _parse_entity(doc["entity"])


@pytest.mark.parametrize("body", ["base_humanoid", "reachy_mini"])
def test_orient_affordances_stay_active_for_unrelated_goal(body):
    """The bug that blocked orienting: the goal-relevance top-k deactivated the
    body's own listen/turn affordances (they scored 0 against a goal that didn't
    mention sound), so the agent's `listen`/`turn` failed 'not active'. The
    `always_active: true` marker exempts them — a body can always attend to a
    sound / turn its head, regardless of the static goal."""
    from maxim.embodiment.body import Embodiment
    from maxim.embodiment.entity_map import EntityMap
    from maxim.embodiment.tool_bridge import generate_tools_for_entity
    from maxim.tools.discovery import select_goal_relevant_tools
    from maxim.tools.registry import ToolRegistry

    entity = _load_entity(body)
    reg = ToolRegistry()
    generate_tools_for_entity(entity, reg, embodiment=Embodiment(entity), entity_map=EntityMap())

    # a goal with zero keyword overlap with the orient affordances
    kept = set(select_goal_relevant_tools("wait calmly and report anything new", EntityMap(), reg))
    for aff in ("listen", "turn_left", "turn_right"):
        name = f"{body}_{aff}"
        assert name in kept, f"{name} must stay active (always_active) — it was deactivated pre-fix"


@pytest.mark.parametrize("body", ["base_humanoid", "reachy_mini"])
def test_listen_returns_world_set_azimuth(body):
    from maxim.embodiment.body import Embodiment
    from maxim.embodiment.tool_bridge import generate_tools_for_entity
    from maxim.tools.registry import ToolRegistry

    entity = _load_entity(body)
    assert "azimuth" in entity.sensors, f"{body} must declare an azimuth root sensor"
    emb = Embodiment(entity)

    # §1.16 world-sets the azimuth from a delivered DoA percept.
    assert world_set_azimuth(emb, -0.6) is True

    tools = generate_tools_for_entity(entity, ToolRegistry(), embodiment=emb)
    listen = next((t for t in tools if t.name.endswith("_listen")), None)
    assert listen is not None, f"{body} must expose a listen affordance"

    out = listen.execute()
    assert out.success
    entity_state = (out.output or {}).get("entity_state", {})
    assert entity_state.get("azimuth") == -0.6, "listen must surface the world-set azimuth"
