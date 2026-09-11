"""Break 1 of the 1.3 survival loop: the drive prior points at corrective affordances.

R2 (docs/experiments/r2_drive_premise_check.md) found the intrinsic path dead: `food`/`health`
had no corrective affinity, `_read_drive_states` emitted RAW sensor values (largest when
SATIATED), so the drive prior scored a passive `read_*` tool MORE strongly when healthy —
"behaviour moves backwards". These pin the fix: a deficit now emits a positive corrective
NEED (hunger / threat) that the affinity table lands on `eat` / the defensive repertoire, and
the need VANISHES when satiated (polarity correct).
"""

from __future__ import annotations

from types import SimpleNamespace

from maxim.embodiment.sem import EntropicDriveSpec, HomeostaticDriveSpec
from maxim.runtime.agent_loop import (
    _corrective_intensity,
    _corrective_need_for,
    _read_drive_states,
)


def _executor(health: float, food: float):
    # minecraft_player specs: health homeostatic set_point 20 / comfort 6; food entropic
    # drift-down, deprivation 6 / satisfaction 16 (raw 0-40 scale).
    ent = SimpleNamespace(
        drive_specs={
            "health": HomeostaticDriveSpec(set_point=20.0, drift_rate=0.0, comfort_band=6.0),
            "food": EntropicDriveSpec(
                drift_direction="down",
                drift_rate=0.0,
                deprivation_threshold=6.0,
                deprivation_pain=0.5,
                satisfaction_threshold=16.0,
            ),
        },
        vital_metrics={"health": health, "food": food},
        modulators={},
    )
    ent.walk = lambda: [ent]
    return SimpleNamespace(embodiment=SimpleNamespace(root=ent))


def test_need_name_mapping():
    assert _corrective_need_for("food") == "hunger"
    assert _corrective_need_for("health") == "threat"
    assert _corrective_need_for("core_temperature") == "cold"
    assert _corrective_need_for("offset_x") is None


def test_homeostatic_intensity_preserves_legacy_cold_formula():
    spec = HomeostaticDriveSpec(set_point=20.0, drift_rate=0.0, comfort_band=6.0)
    assert _corrective_intensity(spec, 5.0) == 1.0  # min(1, |5-20|) — deep deficit saturates
    assert _corrective_intensity(spec, 20.0) is None  # at set_point
    assert _corrective_intensity(spec, 16.0) is None  # within the comfort band (dev -4 < 6)


def test_entropic_intensity_is_graded():
    spec = EntropicDriveSpec(
        drift_direction="down",
        drift_rate=0.0,
        deprivation_threshold=6.0,
        deprivation_pain=0.5,
        satisfaction_threshold=16.0,
    )
    assert _corrective_intensity(spec, 2.0) == 1.0  # below deprivation -> max
    assert _corrective_intensity(spec, 11.0) == 0.5  # (16-11)/(16-6)
    assert _corrective_intensity(spec, 20.0) is None  # above satisfaction -> no need


def test_starving_and_hurt_emits_corrective_needs():
    drives = _read_drive_states(_executor(health=5.0, food=2.0))
    # raw values still present...
    assert drives["health"] == 5.0 and drives["food"] == 2.0
    # ...plus the derived corrective needs, positive (the R2 fix).
    assert drives.get("threat", 0.0) > 0.0
    assert drives.get("hunger", 0.0) > 0.0


def test_satiated_and_healthy_emits_no_corrective_need():
    drives = _read_drive_states(_executor(health=20.0, food=20.0))
    # Polarity fixed: no corrective need when there is no deficit (R2's "moves
    # backwards" — the need was strongest when satiated — cannot recur).
    assert "threat" not in drives
    assert "hunger" not in drives


# End-to-end through the real consumer (recommend_action), mirroring the R2 probe:
# this is the committed form of the PREMISE-NULL -> PREMISE-HELD flip.
_ROSTER = [
    "read_minecraft_player_health",
    "read_minecraft_player_food",
    "minecraft_player_eat",
    "minecraft_player_attack_nearest",
    "minecraft_player_place_block",
    "minecraft_player_move_forward",
]
_CORRECTIVE = {"minecraft_player_eat", "minecraft_player_attack_nearest"}


def _fresh_nac():
    from maxim.decisions.nac import NAc, NACConfig

    return NAc(config=NACConfig())  # no learned bias / causal / reward — prior only


def test_deficit_selects_a_corrective_affordance_not_a_passive_read():
    rec = _fresh_nac().recommend_action(
        agent_id="t",
        available_tools=_ROSTER,
        current_drives=_read_drive_states(_executor(health=5.0, food=2.0)),
        current_clusters=None,
    )
    assert rec is not None
    assert rec["tool_name"] in _CORRECTIVE, rec.get("reasoning")
    assert not rec["tool_name"].startswith("read_")  # never relieved by reading a sensor


def test_satiated_does_not_select_a_corrective_read_or_action():
    rec = _fresh_nac().recommend_action(
        agent_id="t",
        available_tools=_ROSTER,
        current_drives=_read_drive_states(_executor(health=20.0, food=20.0)),
        current_clusters=None,
    )
    # Behaviour moves WITH need: satiated selects nothing corrective (and never a
    # passive read). The R2 pathology (a read tool winning, more strongly when healthy)
    # cannot recur.
    if rec is not None:
        assert rec["tool_name"] not in _CORRECTIVE
        assert not rec["tool_name"].startswith("read_")
