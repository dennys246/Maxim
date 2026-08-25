"""Exp 52 control body: ``bodies/infant_operant_satiated`` is infant_operant with a
hunger/thirst that never rises — the arm that separates "learns to want" from
"learns to be fed" (exp52_nurture_preregistration.md). Pins the deep-merge shape
the experiment depends on: no orient drive (inherited from infant_operant),
hunger + thirst start at 0 with drift 0, and the mother's feed on this body
relieves nothing under relief-sourced credit."""

from __future__ import annotations

from maxim.embodiment.body import Embodiment
from maxim.embodiment.component_registry import ComponentRegistry
from maxim.simulation.cradle_mother import MotherScaffold, reactive_mother_tick


def _satiated():
    return ComponentRegistry().instantiate("bodies/infant_operant_satiated")


def test_body_resolves_and_keeps_the_operant_shape():
    body = _satiated()
    assert body.name == "infant_operant_satiated"
    # infant_operant's defining edit is inherited: azimuth sensor present, NO drive.
    assert "azimuth" in body.vital_metrics
    assert "azimuth" not in body.drive_specs
    # orient affordances still there (the task is unchanged).
    assert "orient" in body.modulators


def test_hunger_and_thirst_are_zero_and_static():
    body = _satiated()
    assert body.vital_metrics["hunger"] == 0.0
    assert body.vital_metrics["thirst"] == 0.0
    assert body.drive_specs["hunger"].drift_rate == 0.0
    assert body.drive_specs["thirst"].drift_rate == 0.0
    # The rest of the hunger spec is inherited, not blanked, by the partial override.
    assert body.drive_specs["hunger"].deprivation_threshold == 0.7


def test_hungry_sibling_differs_only_in_need():
    hungry = ComponentRegistry().instantiate("bodies/infant_operant")
    assert hungry.drive_specs["hunger"].drift_rate > 0.0
    assert "azimuth" not in hungry.drive_specs


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
