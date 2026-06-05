"""Cradle YAML calibration regression guard (cradle_activation_fixes.md P2).

Pins the calibrated drive + affordance values introduced in PR E so that
future YAML edits surface explicitly in PR diffs rather than silently
moving the substrate-transfer signal.

What this test fixates:

  - ``infant_humanoid.yaml::core_temperature.drive.comfort_band`` = 0.25
    and ``pain_scale`` = 1.5 (calibration so one ``cradle_cool_air::feel``
    breaches comfort by 0.10 → drive-pain 0.15/tick — large enough to
    overcome the LLM's adult fire-is-dangerous prior on Arm A).
  - ``cradle_fire_pit.yaml`` declares both ``warm_self`` (positive
    proximity affordance, no thermal_contact failure) and ``touch``
    (aversive contact affordance, breaches arms.thermal comfort band).

These values are load-bearing for Exp 37 — moving them changes the
substrate-transfer signal the experiment measures. Bumping them is fine,
it just needs an explicit PR with a calibration justification + a pre-reg
amendment if the metric direction changes.
"""

from __future__ import annotations

from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DATA_ROOT = _REPO_ROOT / "src" / "maxim" / "_data" / "components"


def _load_yaml(rel_path: str) -> dict:
    path = _DATA_ROOT / rel_path
    with path.open() as fh:
        return yaml.safe_load(fh)


def test_infant_humanoid_core_temperature_drive_pinned_for_exp37():
    """One feel(cool_air) (self_effect -0.2 on core_temperature) takes the
    body from initial -0.15 to -0.35; with comfort_band=0.25 this breaches
    comfort by 0.10 and produces pain_scale × 0.10 = 0.15 per tick.
    Adjusting either value changes the drive-pain headwind against the
    LLM's adult prior — re-check the PR E calibration math before bumping.
    """
    body = _load_yaml("bodies/infant_humanoid.yaml")
    sensors = body["entity"]["sensors"]
    core_temp = sensors["core_temperature"]
    assert core_temp["initial"] == -0.15
    drive = core_temp["drive"]
    assert drive["drift_mode"] == "homeostatic"
    assert drive["set_point"] == 0.0
    assert drive["comfort_band"] == 0.25, (
        "PR E (cradle_activation_fixes.md P2) tightened comfort_band from "
        "0.4 → 0.25; bumping it back loosens drive-pain motivation and may "
        "re-introduce Arm A's perfect-avoidance failure mode."
    )
    assert drive["pain_scale"] == 1.5, (
        "PR E (cradle_activation_fixes.md P2) raised pain_scale from "
        "0.5 → 1.5; lowering it weakens drive-pain magnitude."
    )


def test_cradle_fire_pit_warm_self_proximity_affordance_present():
    """warm_self is the positive substrate edge ('fire = warm') that
    pairs with the negative touch edge ('touch = pain'). Removing it
    collapses the Exp 37 corroborating descriptive metric to 0 and
    eliminates the discrimination signal the substrate-transfer claim
    predicts.
    """
    fire_pit = _load_yaml("items/cradle_fire_pit.yaml")
    flame = fire_pit["entity"]["modulators"]["flame"]
    affordances = flame["affordances"]

    assert "warm_self" in affordances, (
        "PR E added warm_self as the positive proximity affordance; "
        "removing it eliminates the substrate-transfer positive edge."
    )
    warm_self = affordances["warm_self"]
    self_effect = warm_self["self_effect"]

    # Writes warming on core_temperature (drives toward set_point).
    assert self_effect["core_temperature"] == 0.2

    # Writes thermal on arms WITHOUT breaching the arms' comfort_band (0.5
    # per base_humanoid + infant_humanoid). +0.2 stays well inside.
    assert self_effect["arms.thermal"] == 0.2


def test_cradle_fire_pit_touch_still_aversive():
    """The pre-existing touch affordance MUST stay aversive — arms.thermal
    self_effect must exceed the infant_humanoid arms.thermal comfort_band
    (0.5) so thermal_contact fires and forms the negative substrate edge.
    """
    fire_pit = _load_yaml("items/cradle_fire_pit.yaml")
    affordances = fire_pit["entity"]["modulators"]["flame"]["affordances"]
    assert "touch" in affordances
    touch_effect = affordances["touch"]["self_effect"]
    assert touch_effect["arms.thermal"] > 0.5, (
        "touch must breach the arms.thermal comfort_band (0.5) to trigger "
        "thermal_contact failure — without this the negative substrate "
        "edge does not form and Exp 37's failure_class_action_count "
        "primary metric becomes structurally 0."
    )
