"""Unit tests for the reactive mother scaffold (cradle orient sim).

Pins the LOAD-BEARING per-turn order — feed-if-oriented (on the PRIOR azimuth) →
place stimulus → guide (fading) → speak — plus the fade mechanics on a fixture
infant body. The order matters: feeding must reward the azimuth the infant's own
prior turn left, BEFORE the new stimulus overwrites it, or the infant never
learns (each stimulus would erase its orient before it could be rewarded).
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


# ── feed rewards the PRIOR orient (the load-bearing order) ───────────────────


def test_feed_rewards_prior_orient():
    body_o, emb_o = _infant(azimuth=0.05)  # faced the mother last turn
    out_o = reactive_mother_tick(emb_o, scaffold=MotherScaffold(feed_amount=0.5))
    assert out_o["fed"] is True and body_o.vital_metrics["hunger"] < 0.9

    body_x, emb_x = _infant(azimuth=0.6)  # not oriented last turn
    out_x = reactive_mother_tick(emb_x, scaffold=MotherScaffold(feed_amount=0.5))
    assert out_x["fed"] is False and body_x.vital_metrics["hunger"] == 0.9


def test_feed_reads_prior_azimuth_before_stimulus_overwrites_it():
    # The infant oriented last turn (az 0.05). This turn the mother calls from -0.7.
    # It MUST still be fed (rewarding the prior orient) even though the stimulus
    # then moves the head off-center. Feeding after the stimulus would wrongly
    # deny the reward.
    body, emb = _infant(azimuth=0.05)
    out = reactive_mother_tick(
        emb, scaffold=MotherScaffold(feed_amount=0.5, guide_strength=0.0, stimulus_azimuths=(-0.7,))
    )
    assert out["fed"] is True  # rewarded the prior orient
    assert out["az_prior"] == 0.05
    assert abs(body.vital_metrics["azimuth"] - (-0.7)) < 1e-9  # stimulus then placed it off-center
    assert out["az_guided"] is not None and abs(out["az_guided"] - (-0.7)) < 1e-9


# ── stimulus + fading guide ──────────────────────────────────────────────────


def test_stimulus_places_azimuth_when_no_guide():
    body, emb = _infant(azimuth=0.0)
    out = reactive_mother_tick(
        emb, scaffold=MotherScaffold(guide_strength=0.0, feed_amount=0.0, stimulus_azimuths=(-0.7, 0.4))
    )
    # turn 0 -> first stimulus -0.7, no guide -> stays there
    assert abs(body.vital_metrics["azimuth"] - (-0.7)) < 1e-9
    assert out["guided"] is False


def test_full_guide_centers_after_stimulus():
    body, emb = _infant(azimuth=0.0)
    out = reactive_mother_tick(
        emb, scaffold=MotherScaffold(guide_strength=1.0, feed_amount=0.0, stimulus_azimuths=(-0.8,))
    )
    # stimulus -0.8 then full guide -> centered
    assert out["guided"] is True
    assert abs(body.vital_metrics["azimuth"]) <= 0.1


def test_partial_guide_moves_halfway():
    body, emb = _infant(azimuth=0.0)
    reactive_mother_tick(emb, scaffold=MotherScaffold(guide_strength=0.5, feed_amount=0.0, stimulus_azimuths=(-0.8,)))
    # -0.8 * (1-0.5) = -0.4 (still off-center; the infant must complete the turn)
    assert abs(body.vital_metrics["azimuth"] - (-0.4)) < 1e-9


def test_stimulus_rotates_by_turn():
    body, emb = _infant(azimuth=0.0)
    sc = MotherScaffold(guide_strength=0.0, feed_amount=0.0, stimulus_azimuths=(-0.7, 0.4, -0.9))
    reactive_mother_tick(emb, scaffold=sc, turn_idx=1)
    assert abs(body.vital_metrics["azimuth"] - 0.4) < 1e-9  # turn 1 -> second stimulus


def test_mother_can_center_beyond_infant_reach():
    body, emb = _infant(azimuth=0.0)
    reactive_mother_tick(emb, scaffold=MotherScaffold(guide_strength=1.0, feed_amount=0.0, stimulus_azimuths=(-0.95,)))
    assert abs(body.vital_metrics["azimuth"]) <= 0.1  # caregiver turns the head fully


# ── speak + fail-soft ────────────────────────────────────────────────────────


def test_motherese_injects_and_rotates():
    _, emb = _infant(azimuth=0.0)
    said: list[str] = []
    sc = MotherScaffold(guide_strength=0.0, feed_amount=0.0, speech=("aaa", "bbb"))
    r0 = reactive_mother_tick(emb, scaffold=sc, turn_idx=0, inject=said.append)
    r1 = reactive_mother_tick(emb, scaffold=sc, turn_idx=1, inject=said.append)
    r2 = reactive_mother_tick(emb, scaffold=sc, turn_idx=2, inject=said.append)
    assert said == ["aaa", "bbb", "aaa"]
    assert (r0["spoke"], r1["spoke"], r2["spoke"]) == ("aaa", "bbb", "aaa")


def test_no_inject_no_speech():
    _, emb = _infant(azimuth=0.0)
    out = reactive_mother_tick(emb, scaffold=MotherScaffold(speech=("hi",)), inject=None)
    assert out["spoke"] is None


def test_fail_soft_on_unembodied():
    assert reactive_mother_tick(None, scaffold=MotherScaffold())["fed"] is False

    class _NoRoot:
        root = None

    assert reactive_mother_tick(_NoRoot(), scaffold=MotherScaffold())["guided"] is False


def test_body_without_azimuth_is_noop_for_orient():
    d = {
        "name": "plain",
        "entity_type": "body",
        "sensors": {"hunger": {"unit": "ratio", "range": [0, 1], "initial": 0.9}},
    }
    emb = Embodiment(root=_parse_entity(d))
    out = reactive_mother_tick(
        emb, scaffold=MotherScaffold(guide_strength=1.0, feed_amount=0.5, stimulus_azimuths=(-0.7,))
    )
    assert out["guided"] is False  # no azimuth sensor to place/guide
    # az_prior is None -> feed contingency (oriented) not met -> not fed
    assert out["fed"] is False
