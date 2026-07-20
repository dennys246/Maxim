"""Unit tests for the reactive mother scaffold (cradle orient sim).

Pins the fade-scaffold mechanics on a fixture infant body: full guide centers the
head and unlocks feeding; partial guide moves it partway (may not orient); no
guide feeds only if the infant is ALREADY oriented (the Act-3 autonomous case);
motherese injects through the caller's channel; fail-soft on unembodied bodies.
"""

from __future__ import annotations

from maxim.embodiment.body import Embodiment
from maxim.embodiment.spec import _parse_entity
from maxim.simulation.cradle_mother import MotherScaffold, reactive_mother_tick


def _infant(azimuth: float, hunger: float = 0.9):
    """Fixture infant: azimuth (centeredness drive) + entropic hunger + thirst,
    mirroring infant_humanoid."""
    d = {
        "name": "infant",
        "entity_type": "body",
        "sensors": {
            "azimuth": {
                "unit": "normalized",
                "range": [-1, 1],
                "initial": azimuth,
                "drive": {
                    "drift_mode": "homeostatic",
                    "set_point": 0.0,
                    "drift_rate": 0.0,
                    "comfort_band": 0.1,
                    "pain_scale": 0.3,
                },
            },
            "hunger": {
                "unit": "ratio",
                "range": [0, 1],
                "initial": hunger,
                "drive": {
                    "drift_mode": "entropic",
                    "drift_direction": "up",
                    "drift_rate": 0.0,
                    "deprivation_threshold": 0.7,
                    "deprivation_pain": 0.3,
                    "satisfaction_threshold": 0.2,
                },
            },
            "thirst": {"unit": "ratio", "range": [0, 1], "initial": 0.8},
        },
    }
    body = _parse_entity(d)
    return body, Embodiment(root=body)


def test_full_guide_centers_head_and_feeds():
    body, emb = _infant(azimuth=-0.7, hunger=0.9)
    out = reactive_mother_tick(emb, scaffold=MotherScaffold(guide_strength=1.0, feed_amount=0.5))
    assert out["guided"] is True
    # fully centered (|az| ~ 0 <= threshold) -> oriented -> fed
    assert abs(body.vital_metrics["azimuth"]) <= 0.1
    assert out["fed"] is True
    assert body.vital_metrics["hunger"] < 0.9  # relieved


def test_partial_guide_moves_halfway_and_may_not_orient():
    body, emb = _infant(azimuth=-0.8)
    out = reactive_mother_tick(emb, scaffold=MotherScaffold(guide_strength=0.5, feed_amount=0.5))
    assert out["guided"] is True
    # -0.8 * (1-0.5) = -0.4, still off-center -> NOT oriented -> not fed
    assert abs(body.vital_metrics["azimuth"] - (-0.4)) < 1e-6
    assert out["fed"] is False


def test_no_guide_feeds_only_if_already_oriented():
    # Autonomous act (Act 3): the mother does not guide. The infant is fed iff its
    # OWN prior turns left it oriented.
    body_o, emb_o = _infant(azimuth=0.05)  # already facing mother
    out_o = reactive_mother_tick(emb_o, scaffold=MotherScaffold(guide_strength=0.0, feed_amount=0.5))
    assert out_o["guided"] is False and out_o["fed"] is True

    body_x, emb_x = _infant(azimuth=0.6)  # still off-center
    out_x = reactive_mother_tick(emb_x, scaffold=MotherScaffold(guide_strength=0.0, feed_amount=0.5))
    assert out_x["guided"] is False and out_x["fed"] is False
    assert body_x.vital_metrics["azimuth"] == 0.6  # untouched


def test_mother_can_center_beyond_infant_reach():
    # A caregiver physically turns the head; not bound by the infant's own motor
    # reach clamp (reflex_oriented_azimuth). Full guide centers from a hard offset.
    body, emb = _infant(azimuth=-0.95)
    reactive_mother_tick(emb, scaffold=MotherScaffold(guide_strength=1.0, feed_amount=0.0))
    assert abs(body.vital_metrics["azimuth"]) <= 0.1


def test_motherese_injects_and_rotates():
    _, emb = _infant(azimuth=0.0)
    said: list[str] = []
    sc = MotherScaffold(guide_strength=0.0, feed_amount=0.0, speech=("aaa", "bbb"))
    o0 = reactive_mother_tick(emb, scaffold=sc, turn_idx=0, inject=said.append)
    o1 = reactive_mother_tick(emb, scaffold=sc, turn_idx=1, inject=said.append)
    o2 = reactive_mother_tick(emb, scaffold=sc, turn_idx=2, inject=said.append)
    assert said == ["aaa", "bbb", "aaa"]  # rotates deterministically
    assert o0["spoke"] == "aaa" and o1["spoke"] == "bbb" and o2["spoke"] == "aaa"


def test_no_inject_no_speech_recorded():
    _, emb = _infant(azimuth=0.0)
    out = reactive_mother_tick(emb, scaffold=MotherScaffold(speech=("hi",)), inject=None)
    assert out["spoke"] is None


def test_fail_soft_on_unembodied():
    assert reactive_mother_tick(None, scaffold=MotherScaffold())["guided"] is False

    class _NoRoot:
        root = None

    assert reactive_mother_tick(_NoRoot(), scaffold=MotherScaffold())["fed"] is False


def test_body_without_azimuth_is_noop_for_guidance():
    d = {
        "name": "plain",
        "entity_type": "body",
        "sensors": {"hunger": {"unit": "ratio", "range": [0, 1], "initial": 0.9}},
    }
    body = _parse_entity(d)
    emb = Embodiment(root=body)
    out = reactive_mother_tick(emb, scaffold=MotherScaffold(guide_strength=1.0, feed_amount=0.5))
    assert out["guided"] is False  # no azimuth sensor
    # az_after is None -> feeding contingency (oriented) is not met -> not fed
    assert out["fed"] is False
