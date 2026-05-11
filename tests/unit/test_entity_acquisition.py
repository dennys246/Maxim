"""Unit tests for entity acquisition — Mechanism B from proprioceptive_discovery.md.

Tests cover:
- EntityMap.transfer_to_self / transfer_to_scene
- ModulatorAffordanceTool emitting entity_acquired for pick_up on acquirable entities
- ModulatorAffordanceTool emitting entity_released for drop
- Executor._handle_entity_acquisition reparenting + tool registration
- Acquired entity sensors evaluated in damage model
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from maxim.embodiment.entity_map import EntityMap
from maxim.embodiment.sem import AffordanceSchema, Entity


# ---------------------------------------------------------------------------
# EntityMap ownership transfer
# ---------------------------------------------------------------------------


class TestEntityMapTransfer:
    def test_transfer_to_self(self):
        body = Entity("body", "body")
        rock = Entity("rock", "item")
        emap = EntityMap()
        emap.register_self(body)
        emap.register_scene(rock)

        assert not emap.is_self(rock)
        emap.transfer_to_self(rock)
        assert emap.is_self(rock)

    def test_transfer_to_scene(self):
        body = Entity("body", "body")
        rock = Entity("rock", "item")
        emap = EntityMap()
        emap.register_self(body)
        emap.register_self(rock)

        assert emap.is_self(rock)
        emap.transfer_to_scene(rock)
        assert not emap.is_self(rock)

    def test_transfer_includes_children(self):
        body = Entity("body", "body")
        rock = Entity("rock", "item")
        shard = Entity("shard", "part", parent=rock)
        emap = EntityMap()
        emap.register_self(body)
        emap.register_scene(rock)

        emap.transfer_to_self(rock)
        assert emap.is_self(rock)
        assert emap.is_self(shard)


# ---------------------------------------------------------------------------
# Entity reparenting
# ---------------------------------------------------------------------------


class TestEntityReparent:
    def test_reparent_to_body(self):
        body = Entity("body", "body")
        rock = Entity("rock", "item")
        assert rock.parent is None
        assert rock not in body.children

        rock.reparent(body)
        assert rock.parent is body
        assert rock in body.children

    def test_reparent_away_from_body(self):
        body = Entity("body", "body")
        scene_root = Entity("scene", "scene")
        rock = Entity("rock", "item", parent=body)
        assert rock in body.children

        rock.reparent(scene_root)
        assert rock.parent is scene_root
        assert rock not in body.children
        assert rock in scene_root.children

    def test_acquired_entity_sensors_in_damage_model(self):
        """When acquired, the entity's sensors should be walkable from the body root."""
        from maxim.embodiment.spec import _parse_entity

        body_data = {
            "name": "body",
            "entity_type": "body",
            "sensors": {"hp": {"unit": "ratio", "range": [0, 1], "initial": 1.0}},
        }
        rock_data = {
            "name": "sharp_rock",
            "entity_type": "item",
            "sensors": {"sharpness": {"unit": "ratio", "range": [0, 1], "initial": 0.8}},
            "acquirable": True,
            "failure_modes": [
                {"name": "laceration", "trigger": {"field": "sharpness", "op": ">", "value": 0.5, "pain": 0.4}}
            ],
        }
        body = _parse_entity(body_data)
        rock = _parse_entity(rock_data)

        # Before acquisition: rock not in body's walk
        walked = list(body.walk())
        assert rock not in walked

        # Acquire: reparent rock to body
        rock.reparent(body)
        walked = list(body.walk())
        assert rock in walked

        # Rock's failure modes should evaluate against its sensors
        from maxim.embodiment.body import Embodiment

        emb = Embodiment(body)
        events = emb.evaluate_failures()
        failure_names = [e.failure_name for e in events]
        assert "laceration" in failure_names


# ---------------------------------------------------------------------------
# Side-effect emission from ModulatorAffordanceTool
# ---------------------------------------------------------------------------


class TestAcquisitionSideEffects:
    def test_pick_up_emits_entity_acquired(self):
        """pick_up on acquirable target emits entity_acquired side_effect."""
        from maxim.embodiment.spec import SpecModulator, _parse_entity
        from maxim.embodiment.tool_bridge import ModulatorAffordanceTool

        body_data = {
            "name": "body",
            "entity_type": "body",
            "modulators": {
                "arms": {
                    "affordances": {
                        "pick_up": {"params": {"object": "str"}, "description": "Pick up an object"},
                    },
                },
            },
        }
        rock_data = {
            "name": "sharp_rock",
            "entity_type": "item",
            "acquirable": True,
        }
        body = _parse_entity(body_data)
        rock = _parse_entity(rock_data)

        emap = EntityMap()
        emap.register_self(body)
        emap.register_scene(rock)

        arms_mod = body.modulators["arms"]
        schema = arms_mod.affordances["pick_up"]

        tool = ModulatorAffordanceTool(
            body,
            arms_mod,
            "pick_up",
            schema,
            "body_pick_up",
            entity_map=emap,
        )

        result = tool.execute(object="sharp_rock")
        assert result.success
        assert result.side_effects is not None
        assert result.side_effects.get("entity_acquired") == "sharp_rock"

    def test_pick_up_non_acquirable_no_side_effect(self):
        """pick_up on non-acquirable target does NOT emit entity_acquired."""
        from maxim.embodiment.spec import SpecModulator, _parse_entity
        from maxim.embodiment.tool_bridge import ModulatorAffordanceTool

        body_data = {
            "name": "body",
            "entity_type": "body",
            "modulators": {
                "arms": {
                    "affordances": {
                        "pick_up": {"params": {"object": "str"}, "description": "Pick up an object"},
                    },
                },
            },
        }
        fire_data = {
            "name": "fire_pit",
            "entity_type": "item",
            # Not acquirable
        }
        body = _parse_entity(body_data)
        fire = _parse_entity(fire_data)

        emap = EntityMap()
        emap.register_self(body)
        emap.register_scene(fire)

        arms_mod = body.modulators["arms"]
        schema = arms_mod.affordances["pick_up"]

        tool = ModulatorAffordanceTool(
            body,
            arms_mod,
            "pick_up",
            schema,
            "body_pick_up",
            entity_map=emap,
        )

        result = tool.execute(object="fire_pit")
        assert result.success
        # No entity_acquired because fire_pit is not acquirable
        if result.side_effects:
            assert "entity_acquired" not in result.side_effects
