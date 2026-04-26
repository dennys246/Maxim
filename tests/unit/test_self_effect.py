"""Unit tests for self-effect on affordances.

Self-effects allow scene entity affordances to write back to the agent's
body sensors. Example: eating food reduces hunger.
"""

from __future__ import annotations

import pytest

from maxim.embodiment.sem import AffordanceSchema


class TestSelfEffectSchema:
    def test_self_effect_parsed_from_yaml(self):
        from maxim.embodiment.spec import _parse_entity

        data = {
            "name": "food",
            "entity_type": "item",
            "modulators": {
                "nutrition": {
                    "affordances": {
                        "eat": {
                            "params": {},
                            "description": "Eat the food",
                            "self_effect": {"hunger": -0.4},
                        },
                    },
                },
            },
        }
        entity = _parse_entity(data)
        schema = entity.modulators["nutrition"].affordances["eat"]
        assert schema.self_effect == {"hunger": -0.4}

    def test_no_self_effect_default(self):
        schema = AffordanceSchema(params={}, description="test")
        assert schema.self_effect == {}


class TestSelfEffectExecution:
    def test_eat_reduces_hunger(self):
        from maxim.embodiment.body import Embodiment
        from maxim.embodiment.spec import _parse_entity
        from maxim.embodiment.tool_bridge import ModulatorAffordanceTool

        # Agent body with hunger sensor
        body_data = {
            "name": "agent",
            "entity_type": "body",
            "sensors": {"hunger": {"unit": "ratio", "range": [0, 1], "initial": 0.8}},
        }
        # Food entity with self_effect
        food_data = {
            "name": "food",
            "entity_type": "item",
            "modulators": {
                "nutrition": {
                    "affordances": {
                        "eat": {
                            "params": {},
                            "description": "Eat",
                            "self_effect": {"hunger": -0.4},
                        },
                    },
                },
            },
        }
        body = _parse_entity(body_data)
        food = _parse_entity(food_data)
        emb = Embodiment(body)

        # Execute eat on food entity, with embodiment pointing to agent body
        nutrition_mod = food.modulators["nutrition"]
        schema = nutrition_mod.affordances["eat"]
        tool = ModulatorAffordanceTool(
            food, nutrition_mod, "eat", schema, "food_eat",
            embodiment=emb,
        )
        result = tool.execute()
        assert result.success

        # hunger should drop from 0.8 to 0.4
        assert body.vital_metrics["hunger"] == pytest.approx(0.4, abs=0.01)

    def test_self_effect_clamps_at_zero(self):
        from maxim.embodiment.body import Embodiment
        from maxim.embodiment.spec import _parse_entity
        from maxim.embodiment.tool_bridge import ModulatorAffordanceTool

        body_data = {
            "name": "agent",
            "entity_type": "body",
            "sensors": {"hunger": {"unit": "ratio", "range": [0, 1], "initial": 0.2}},
        }
        food_data = {
            "name": "food",
            "entity_type": "item",
            "modulators": {
                "nutrition": {
                    "affordances": {
                        "eat": {
                            "params": {},
                            "description": "Eat",
                            "self_effect": {"hunger": -0.4},
                        },
                    },
                },
            },
        }
        body = _parse_entity(body_data)
        food = _parse_entity(food_data)
        emb = Embodiment(body)

        nutrition_mod = food.modulators["nutrition"]
        schema = nutrition_mod.affordances["eat"]
        tool = ModulatorAffordanceTool(food, nutrition_mod, "eat", schema, "food_eat", embodiment=emb)
        tool.execute()

        # 0.2 - 0.4 = -0.2 → clamped to 0.0
        assert body.vital_metrics["hunger"] == 0.0

    def test_self_effect_missing_sensor_logs_warning(self, caplog):
        from maxim.embodiment.body import Embodiment
        from maxim.embodiment.spec import _parse_entity
        from maxim.embodiment.tool_bridge import ModulatorAffordanceTool

        body_data = {
            "name": "agent",
            "entity_type": "body",
            "sensors": {"hp": {"unit": "ratio", "range": [0, 1], "initial": 1.0}},
        }
        food_data = {
            "name": "food",
            "entity_type": "item",
            "modulators": {
                "nutrition": {
                    "affordances": {
                        "eat": {
                            "params": {},
                            "description": "Eat",
                            "self_effect": {"hunger": -0.4},  # hunger doesn't exist on body
                        },
                    },
                },
            },
        }
        body = _parse_entity(body_data)
        food = _parse_entity(food_data)
        emb = Embodiment(body)

        nutrition_mod = food.modulators["nutrition"]
        schema = nutrition_mod.affordances["eat"]
        tool = ModulatorAffordanceTool(food, nutrition_mod, "eat", schema, "food_eat", embodiment=emb)

        import logging
        with caplog.at_level(logging.WARNING):
            tool.execute()

        # Should log warning about missing sensor, not crash
        assert any("self_effect target" in r.message for r in caplog.records)
