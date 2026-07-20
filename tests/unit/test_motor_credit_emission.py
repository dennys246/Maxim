"""GAP 1b: ``ModulatorAffordanceTool.execute`` emits ``drive_potential_diff``.

The motor-credit signal the substrate-primary orient policy needs: the drive
RELIEF an action's own ``self_effect`` produced. Turning toward the sound
(|azimuth| down) must emit a POSITIVE value; turning away a NEGATIVE one; an
affordance that touches no drive sensor emits nothing (consumer falls back to
the ±1 tool-success signal). See docs/user/tool_side_effects.md +
reference_orient_motor_credit_gap.
"""

from __future__ import annotations

from maxim.embodiment.body import Embodiment
from maxim.embodiment.spec import _parse_entity
from maxim.embodiment.tool_bridge import ModulatorAffordanceTool


def _orienting_body(initial_azimuth: float):
    """A body with a centeredness (homeostatic) drive on azimuth + an orient
    modulator whose turns move azimuth toward/away from center. Mirrors
    base_humanoid.yaml."""
    data = {
        "name": "agent",
        "entity_type": "body",
        "sensors": {
            "azimuth": {
                "unit": "normalized",
                "range": [-1, 1],
                "initial": initial_azimuth,
                "drive": {
                    "drift_mode": "homeostatic",
                    "set_point": 0.0,
                    "drift_rate": 0.0,
                    "comfort_band": 0.1,
                    "pain_scale": 0.3,
                },
            },
        },
        "modulators": {
            "orient": {
                "abstract": True,
                "affordances": {
                    # +azimuth moves a left (negative) sound toward center.
                    "turn_left": {"params": {}, "description": "turn left", "self_effect": {"azimuth": 0.3}},
                    "turn_right": {"params": {}, "description": "turn right", "self_effect": {"azimuth": -0.3}},
                },
            },
        },
    }
    body = _parse_entity(data)
    return body, Embodiment(body)


def _run(body, emb, affordance: str):
    mod = body.modulators["orient"]
    tool = ModulatorAffordanceTool(body, mod, affordance, mod.affordances[affordance], affordance, embodiment=emb)
    return tool.execute()


def test_turn_toward_sound_emits_positive_relief():
    # Sound on the left (azimuth -0.7); turn_left moves azimuth -0.7 -> -0.4.
    body, emb = _orienting_body(-0.7)
    result = _run(body, emb, "turn_left")
    assert result.success
    assert result.side_effects is not None
    diff = result.side_effects["drive_potential_diff"]
    # pain(-0.7)=0.18, pain(-0.4)=0.09 -> relief +0.09
    assert diff > 0
    assert abs(diff - 0.09) < 1e-6


def test_turn_away_from_sound_emits_negative_reward():
    # From -0.7, turn_right pushes to -1.0 (more off-center -> worse).
    body, emb = _orienting_body(-0.7)
    result = _run(body, emb, "turn_right")
    assert result.success
    diff = result.side_effects["drive_potential_diff"]
    # pain(-0.7)=0.18, pain(-1.0)=0.27 -> -0.09
    assert diff < 0
    assert abs(diff - (-0.09)) < 1e-6


def test_no_drive_sensor_emits_no_potential_diff():
    """An eat affordance on a body whose hunger has NO drive spec must not emit
    drive_potential_diff — the consumer falls back to the ±1 tool-success."""
    body_data = {
        "name": "agent",
        "entity_type": "body",
        "sensors": {"hunger": {"unit": "ratio", "range": [0, 1], "initial": 0.8}},
    }
    food_data = {
        "name": "food",
        "entity_type": "item",
        "modulators": {
            "nutrition": {
                "abstract": True,
                "affordances": {"eat": {"params": {}, "description": "Eat", "self_effect": {"hunger": -0.4}}},
            },
        },
    }
    body = _parse_entity(body_data)
    food = _parse_entity(food_data)
    emb = Embodiment(body)
    mod = food.modulators["nutrition"]
    tool = ModulatorAffordanceTool(food, mod, "eat", mod.affordances["eat"], "food_eat", embodiment=emb)
    result = tool.execute()
    assert result.success
    assert result.side_effects is None or "drive_potential_diff" not in result.side_effects


def test_feeding_relief_on_entropic_hunger_drive_is_positive():
    """A body whose hunger IS a drive (entropic): eating past the deprivation
    threshold back to comfort emits positive relief."""
    body_data = {
        "name": "agent",
        "entity_type": "body",
        "sensors": {
            "hunger": {
                "unit": "ratio",
                "range": [0, 1],
                "initial": 0.8,  # deprived (threshold 0.6)
                "drive": {
                    "drift_mode": "entropic",
                    "drift_direction": "up",
                    "drift_rate": 0.01,
                    "deprivation_threshold": 0.6,
                    "deprivation_pain": 0.5,
                    "satisfaction_threshold": 0.2,
                },
            },
        },
        "modulators": {
            "belly": {
                "abstract": True,
                "affordances": {"eat": {"params": {}, "description": "Eat", "self_effect": {"hunger": -0.4}}},
            },
        },
    }
    body = _parse_entity(body_data)
    emb = Embodiment(body)
    mod = body.modulators["belly"]
    tool = ModulatorAffordanceTool(body, mod, "eat", mod.affordances["eat"], "eat", embodiment=emb)
    result = tool.execute()
    # hunger 0.8 -> 0.4: pain 0.5 -> 0.0 -> relief +0.5
    diff = result.side_effects["drive_potential_diff"]
    assert abs(diff - 0.5) < 1e-6
