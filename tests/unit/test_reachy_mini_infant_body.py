"""Exp 54 nursery bodies: ``bodies/reachy_mini_infant`` (+ ``_satiated``) are the
Reachy Mini's OWN body component with the innate azimuth drive removed and the
infant's hunger/thirst added — so the want a nursery teaches is keyed on the
robot's own tool names and read out through the production motor factory with
no δ map anywhere (exp54_nurture_reachy_body_preregistration.md, "The body").

Pins every structural fact the experiment rides on: the entity name is the
production body's (``reachy_mini`` — the tool prefix / learned-bias key prefix),
the four ``reachy_mini_turn_*`` tools register, azimuth carries NO drive, hunger
and thirst carry infant_humanoid's entropic specs, ``make_reachy_orient_factory``
returns a backend with exactly the YAML's four deltas, the ``turn_left,turn_right``
whitelist substring-matches the ``_big`` pair (the 4-tool repertoire, declared S6),
and the satiated variant never drifts and mints no relief credit.
"""

from __future__ import annotations

import pytest

from maxim.embodiment.body import Embodiment
from maxim.embodiment.component_registry import ComponentRegistry
from maxim.embodiment.tool_bridge import generate_tools_for_entity
from maxim.hardware.reachy.motor_backend import ReachyOrientMotorBackend, make_reachy_orient_factory
from maxim.simulation.cradle_mother import MotherScaffold, reactive_mother_tick
from maxim.tools.registry import ToolRegistry

ORIENT_TOOLS = (
    "reachy_mini_turn_left",
    "reachy_mini_turn_right",
    "reachy_mini_turn_left_big",
    "reachy_mini_turn_right_big",
)
# The production factory reads self_effect["head_yaw"] — the robot's own step sizes.
EXPECTED_DELTAS = {"turn_left": 0.3, "turn_right": -0.3, "turn_left_big": 0.9, "turn_right_big": -0.9}


def _infant():
    return ComponentRegistry().instantiate("bodies/reachy_mini_infant")


def _satiated():
    return ComponentRegistry().instantiate("bodies/reachy_mini_infant_satiated")


def _tools(body) -> list[str]:
    registry = ToolRegistry()
    generate_tools_for_entity(body, registry, embodiment=Embodiment(body))
    return list(registry.list())


class _FakeRobot:
    def get_current_pose(self):
        return {"body_yaw": 0.0, "yaw": 0.0}


@pytest.mark.parametrize("factory", [_infant, _satiated])
def test_entity_name_is_the_production_bodys(factory):
    # Load-bearing: learned bias keys are tool:<entity.name>_<affordance>. A
    # nursery on this body must write tool:reachy_mini_turn_left, not
    # tool:reachy_mini_infant_turn_left — or the user's robot never sees them.
    assert factory().name == "reachy_mini"


@pytest.mark.parametrize("factory", [_infant, _satiated])
def test_four_reachy_mini_turn_tools_register(factory):
    names = _tools(factory())
    for want in ORIENT_TOOLS:
        assert want in names, f"{want} missing from {names}"


def test_azimuth_sensor_kept_but_drive_removed():
    body = _infant()
    assert "azimuth" in body.vital_metrics
    assert "azimuth" not in body.drive_specs
    # The parent reachy_mini body DOES carry the drive — this body's edit is the removal.
    assert "azimuth" in ComponentRegistry().instantiate("bodies/reachy_mini").drive_specs


def test_hunger_and_thirst_carry_the_infant_specs():
    body = _infant()
    hunger, thirst = body.drive_specs["hunger"], body.drive_specs["thirst"]
    assert body.vital_metrics["hunger"] == 0.0 and body.vital_metrics["thirst"] == 0.0
    assert hunger.drift_direction == "up" and hunger.drift_rate == pytest.approx(0.006)
    assert hunger.deprivation_threshold == pytest.approx(0.7) and hunger.deprivation_pain == pytest.approx(0.3)
    assert thirst.drift_direction == "up" and thirst.drift_rate == pytest.approx(0.008)
    assert thirst.deprivation_threshold == pytest.approx(0.6) and thirst.deprivation_pain == pytest.approx(0.25)
    # Same specs as the infant body Exp 52 learned on (the declared S6 differences are
    # the step sizes and the repertoire, NOT the need).
    infant = ComponentRegistry().instantiate("bodies/infant_operant")
    for name in ("hunger", "thirst"):
        for field in ("drift_rate", "deprivation_threshold", "deprivation_pain", "drift_direction"):
            assert getattr(body.drive_specs[name], field) == getattr(infant.drive_specs[name], field)


def test_hunger_drifts_up_on_the_taught_body():
    body = _infant()
    Embodiment(body).tick_vital_drift(dt=10.0)
    assert body.vital_metrics["hunger"] > 0.0
    assert body.vital_metrics["thirst"] > 0.0


def test_production_factory_reads_the_robots_own_deltas():
    body = _infant()
    factory = make_reachy_orient_factory(_FakeRobot())
    bound = {}
    for ent in body.walk():
        for mname, mod in ent.modulators.items():
            backend = factory(ent, mname, mod)
            if backend is not None:
                bound[mname] = backend
    assert set(bound) == {"orient"}
    assert isinstance(bound["orient"], ReachyOrientMotorBackend)
    assert bound["orient"]._deltas == pytest.approx(EXPECTED_DELTAS)
    assert set(bound["orient"]._deltas) == set(EXPECTED_DELTAS)  # exactly four — no δ map anywhere


def test_whitelist_substring_matches_the_big_pair():
    # benchmark_cradle_mother's MAXIM_SUBSTRATE_TOOL_WHITELIST=turn_left,turn_right is
    # applied as a substring filter (agent_loop.propose_via_substrate) — on this body
    # that is the full 4-tool repertoire, declared in the pre-registration (S6).
    terms = ["turn_left", "turn_right"]
    names = _tools(_infant())
    kept = [t for t in names if any(term in t for term in terms)]
    assert sorted(kept) == sorted(ORIENT_TOOLS)


def test_satiated_never_drifts_and_starts_fed():
    body = _satiated()
    assert body.vital_metrics["hunger"] == 0.0 and body.vital_metrics["thirst"] == 0.0
    assert body.drive_specs["hunger"].drift_rate == 0.0
    assert body.drive_specs["thirst"].drift_rate == 0.0
    # The rest of the spec is inherited, not blanked, by the partial override.
    assert body.drive_specs["hunger"].deprivation_threshold == pytest.approx(0.7)
    Embodiment(body).tick_vital_drift(dt=600.0)
    assert body.vital_metrics["hunger"] == 0.0 and body.vital_metrics["thirst"] == 0.0
    # Same edit as the taught body otherwise: no orient drive, same four tools.
    assert "azimuth" not in body.drive_specs


def test_mother_feed_on_satiated_body_mints_no_credit():
    class _Nac:
        def __init__(self):
            self.credits = []

        def credit_operant_reward(self, agent_id, reward):
            self.credits.append(reward)
            return ("c", "t")

    body = _satiated()
    body.vital_metrics["azimuth"] = -0.3
    nac = _Nac()
    out = reactive_mother_tick(
        Embodiment(root=body),
        scaffold=MotherScaffold(feed_amount=0.5, stimulus_azimuths=(0.6,)),
        nac=nac,
        agent_id="infant",
        prev_stimulus=-0.7,
    )
    assert out["fed"] is True and out["credited"] is False and nac.credits == []


def test_mother_feed_on_hungry_body_mints_credit():
    class _Nac:
        def __init__(self):
            self.credits = []

        def credit_operant_reward(self, agent_id, reward):
            self.credits.append(reward)
            return ("c", "t")

    body = _infant()
    body.vital_metrics["hunger"] = 0.4  # hungry: the feed relieves something
    body.vital_metrics["azimuth"] = -0.3
    nac = _Nac()
    out = reactive_mother_tick(
        Embodiment(root=body),
        scaffold=MotherScaffold(feed_amount=0.5, stimulus_azimuths=(0.6,)),
        nac=nac,
        agent_id="infant",
        prev_stimulus=-0.7,
    )
    assert out["fed"] is True and out["credited"] is True and nac.credits == [1.0]
