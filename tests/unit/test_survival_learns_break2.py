"""Break 2 of the 1.3 survival loop: measured-relief credit for interoceptive drives.

R2 break 2: `eat`'s `self_effect {food:+4}` targets a live-owned sensor, so the modeled
credit is stripped (`drive_credit_withheld`) and no learned bias could form — the want
stayed innate (break 1), never LEARNED. This pins the fix: `ModulatorAffordanceTool.execute`
now measures the affordance's DECLARED live-owned drives before/after the world action and
credits the real relief (`drive_comfort_progress`) on the INTEROCEPTIVE channel — while an
affordance that declares no drive effect is never blamed for ambient drain.
"""

from __future__ import annotations


def _world_body(food_initial: float):
    """A player body whose `food` is an entropic, live-owned world drive."""
    from maxim.embodiment.body import Embodiment
    from maxim.embodiment.spec import _parse_entity

    body = _parse_entity(
        {
            "name": "player",
            "entity_type": "body",
            "sensors": {
                "food": {
                    "unit": "points",
                    "range": [0, 40],
                    "initial": food_initial,
                    "modality": "world",  # world-owned interoceptive drive (as minecraft_player)
                    "drive": {
                        "drift_mode": "entropic",
                        "drift_direction": "down",
                        "drift_rate": 0.0,
                        "deprivation_threshold": 6.0,
                        "deprivation_pain": 0.5,
                        "satisfaction_threshold": 16.0,
                    },
                }
            },
        }
    )
    emb = Embodiment(body)
    emb.live_world_set_sensors = {"food"}  # the bridge owns food (world writes truth)
    return body, emb


class _WorldModulator:
    """Stand-in for MinecraftWorldBackend: its execute writes POST-action world truth."""

    name = "world"

    def __init__(self, body, food_after: float | None):
        self._body = body
        self._food_after = food_after

    def check_affordance_requires(self, _name):
        return (True, "")

    def execute(self, affordance, params):
        from maxim.embodiment.sem import ModulatorResult

        if self._food_after is not None:  # world sync raised food (ate)
            self._body.vital_metrics["food"] = self._food_after
        return ModulatorResult(
            modulator_name="world",
            entity_name="player",
            affordance=affordance,
            params=params,
            success=True,
        )


def _eat_tool(body, emb, food_after):
    from maxim.embodiment.sem import AffordanceSchema
    from maxim.embodiment.tool_bridge import ModulatorAffordanceTool

    schema = AffordanceSchema(description="Eat", self_effect={"food": 4.0})
    mod = _WorldModulator(body, food_after)
    return ModulatorAffordanceTool(body, mod, "eat", schema, "minecraft_player_eat", embodiment=emb)


def test_eat_relief_is_measured_and_credited_interoceptive():
    body, emb = _world_body(food_initial=2.0)  # starving
    out = _eat_tool(body, emb, food_after=6.0).execute()  # world: food 2 -> 6
    side = out.side_effects or {}
    assert side.get("drive_potential_diff") is not None
    assert side["drive_potential_diff"] > 0, side  # real relief, learnable
    assert side.get("drive_relief_channel") == "interoceptive"
    assert "drive_credit_withheld" not in side  # measured relief REPLACES the withheld marker


def test_eat_with_no_relief_withholds_credit_not_a_phantom_plus_one():
    body, emb = _world_body(food_initial=2.0)
    out = _eat_tool(body, emb, food_after=2.0).execute()  # world: food unchanged
    side = out.side_effects or {}
    # Honest "no change": withhold the floor, never book a phantom +1.
    assert side.get("drive_potential_diff") is None or side["drive_potential_diff"] == 0
    assert side.get("drive_relief_channel") != "interoceptive"


def test_affordance_declaring_no_drive_is_not_blamed_for_ambient_drain():
    # move_to declares no food self_effect; even though the world drains food, it gets
    # NO interoceptive drive credit (only DECLARED drives are measured).
    from maxim.embodiment.sem import AffordanceSchema
    from maxim.embodiment.tool_bridge import ModulatorAffordanceTool

    body, emb = _world_body(food_initial=10.0)
    schema = AffordanceSchema(description="Move", self_effect={})
    mod = _WorldModulator(body, food_after=9.5)  # ambient drain during the move
    tool = ModulatorAffordanceTool(body, mod, "move_to", schema, "minecraft_player_move_to", embodiment=emb)
    side = tool.execute().side_effects or {}
    assert side.get("drive_relief_channel") != "interoceptive"  # not blamed for the drain


def test_exteroceptive_live_drive_is_not_locally_measured_when_backend_abstains():
    """The two-lens DO-NOT-SHIP regression: a live-owned EXTEROCEPTIVE drive (azimuth,
    modality 'audio') in self_effect, whose motor backend ABSTAINS this cycle (timeout —
    no measured_drive_transitions) while the async DoA thread moved vital_metrics toward
    center, must NOT fabricate interoceptive credit. Modality-gating (not backend-reporting)
    is what keeps azimuth off the local path.
    """
    from maxim.embodiment.body import Embodiment
    from maxim.embodiment.sem import AffordanceSchema, ModulatorResult
    from maxim.embodiment.spec import _parse_entity
    from maxim.embodiment.tool_bridge import ModulatorAffordanceTool

    body = _parse_entity(
        {
            "name": "robot",
            "entity_type": "body",
            "sensors": {
                "azimuth": {
                    "unit": "rad",
                    "range": [-1, 1],
                    "initial": -0.5,
                    "modality": "audio",  # EXTEROCEPTIVE — backend-measured, never local
                    "drive": {
                        "drift_mode": "homeostatic",
                        "set_point": 0.0,
                        "drift_rate": 0.0,
                        "comfort_band": 0.0,
                    },
                }
            },
        }
    )
    emb = Embodiment(body)
    emb.live_world_set_sensors = {"azimuth"}

    class _AbstainingBackend:  # the Reachy motor backend on a post-turn DoA timeout
        name = "motor"

        def check_affordance_requires(self, _n):
            return (True, "")

        def execute(self, affordance, params):
            body.vital_metrics["azimuth"] = -0.2  # async DoA thread moved it toward center
            return ModulatorResult(  # success, but NO measured_drive_transitions (abstained)
                modulator_name="motor",
                entity_name="robot",
                affordance=affordance,
                params=params,
                success=True,
            )

    schema = AffordanceSchema(description="Turn", self_effect={"azimuth": -0.3})
    tool = ModulatorAffordanceTool(body, _AbstainingBackend(), "turn", schema, "robot_turn", embodiment=emb)
    side = tool.execute().side_effects or {}
    # No fabricated sign, and never routed interoceptive (old behavior: withheld).
    assert side.get("drive_relief_channel") != "interoceptive"
    assert side.get("drive_potential_diff") is None
