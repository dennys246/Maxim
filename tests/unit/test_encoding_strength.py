"""The encoding tag and the storage-strength stamp (memory-strength Phase 2c-2).

Write-only by design: nothing reads ``storage_strength`` until the Phase 2c-3 strategy, so these
tests pin the VALUES and the persistence, and one arm pins that retention did not start reading
them (the byte-identical-default promise the plan makes).
"""

from __future__ import annotations

import math

import pytest

from maxim.memory.encoding import (
    K_DEFAULT,
    NOVELTY_CONFIDENCE_N0,
    S_BASE_DEFAULT,
    SALIENCE_BASELINE,
    EncodingSignals,
    encoding_tag,
    initial_storage_strength,
)
from maxim.memory.hippocampus import Hippocampus, HippocampusConfig


def _signals(**kw) -> EncodingSignals:
    base = dict(
        site="loop",
        salience=None,
        novelty=None,
        surprise=None,
        pain=None,
        drive_pressure=None,
        drive_relief=None,
    )
    base.update(kw)
    return EncodingSignals(**base)


# ── the tag ──────────────────────────────────────────────────────────────────


def test_unmeasured_capture_tags_zero():
    assert encoding_tag(EncodingSignals.unmeasured("loop"), novelty_reference_size=1000) == 0.0


def test_default_salience_is_not_importance():
    """The 0.5 default that every capture carries must not put a floor under the tag."""
    assert encoding_tag(_signals(salience=SALIENCE_BASELINE), novelty_reference_size=1000) == 0.0


def test_salience_deviation_is_positive_part_only():
    assert encoding_tag(_signals(salience=1.0), novelty_reference_size=1000) == pytest.approx(1.0)
    assert encoding_tag(_signals(salience=0.75), novelty_reference_size=1000) == pytest.approx(0.5)
    # Below baseline is the ABSENCE of evidence, not evidence against.
    assert encoding_tag(_signals(salience=0.0), novelty_reference_size=1000) == 0.0


def test_surprise_and_pain_are_their_own_deviation():
    assert encoding_tag(_signals(surprise=0.4), novelty_reference_size=1000) == pytest.approx(0.4)
    assert encoding_tag(_signals(pain=0.7), novelty_reference_size=1000) == pytest.approx(0.7)


def test_noisy_or_saturates_rather_than_summing():
    """Coincident signals add, but a crowd of weak ones cannot manufacture importance."""
    tag = encoding_tag(_signals(surprise=0.5, pain=0.5), novelty_reference_size=1000)
    assert tag == pytest.approx(0.75)  # 1 - 0.5*0.5, not 1.0
    many = encoding_tag(
        _signals(salience=0.6, novelty=1.0, surprise=0.2, pain=0.2),
        novelty_reference_size=10_000,
    )
    assert many < 1.0


def test_novelty_is_weighted_by_how_much_the_store_knows():
    """An empty store calls everything novel; its first traces must not all encode at maximum."""
    empty = encoding_tag(_signals(novelty=1.0), novelty_reference_size=0)
    assert empty == 0.0
    at_n0 = encoding_tag(_signals(novelty=1.0), novelty_reference_size=int(NOVELTY_CONFIDENCE_N0))
    assert at_n0 == pytest.approx(0.5)
    experienced = encoding_tag(_signals(novelty=1.0), novelty_reference_size=100_000)
    assert experienced > 0.99
    assert empty < at_n0 < experienced


def test_drive_pressure_counts_only_for_drives_the_action_relieved():
    """Relevance gating: a starving stretch tags what touched hunger, not everything."""
    relieved = _signals(drive_pressure=(("hunger", 0.8),), drive_relief=(("hunger", 0.5),))
    unrelated = _signals(drive_pressure=(("hunger", 0.8),), drive_relief=(("curiosity", 0.5),))
    assert encoding_tag(relieved, novelty_reference_size=1000) == pytest.approx(1 - 0.5 * 0.2)
    # curiosity's own relief still counts; hunger's PRESSURE does not.
    assert encoding_tag(unrelated, novelty_reference_size=1000) == pytest.approx(0.5)


def test_drive_pressure_fails_closed_when_nothing_measured_relief():
    """The R4 gap: on minecraft_player every action but `eat` declares no self_effect."""
    no_relief = _signals(drive_pressure=(("air", 1.0),), drive_relief=())
    assert encoding_tag(no_relief, novelty_reference_size=1000) == 0.0
    unmeasured_relief = _signals(drive_pressure=(("air", 1.0),), drive_relief=None)
    assert encoding_tag(unmeasured_relief, novelty_reference_size=1000) == 0.0


def test_tag_is_always_a_probability():
    for signals in (
        _signals(salience=1.0, novelty=1.0, surprise=1.0, pain=1.0),
        _signals(drive_relief=(("a", 1.0), ("b", 1.0)), drive_pressure=(("a", 1.0), ("b", 1.0))),
    ):
        assert 0.0 <= encoding_tag(signals, novelty_reference_size=10_000) <= 1.0


# ── S0 ───────────────────────────────────────────────────────────────────────


def test_initial_strength_follows_the_plan_equation():
    assert initial_storage_strength(0.0, s_base=10.0, k=1.0) == pytest.approx(10.0)
    assert initial_storage_strength(1.0, s_base=10.0, k=1.0) == pytest.approx(20.0)
    assert initial_storage_strength(0.5, s_base=4.0, k=2.0) == pytest.approx(8.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tag": 1.5, "s_base": 10.0, "k": 1.0},
        {"tag": -0.1, "s_base": 10.0, "k": 1.0},
        {"tag": 0.5, "s_base": 0.0, "k": 1.0},
        {"tag": 0.5, "s_base": 10.0, "k": -1.0},
    ],
)
def test_initial_strength_rejects_out_of_range_inputs(kwargs):
    with pytest.raises(ValueError):
        initial_storage_strength(**kwargs)


def test_strength_is_positive_so_retrievability_can_never_divide_by_zero():
    """R = exp(-dt/S) in 2c-3; an S of 0 would be a ZeroDivisionError at the first recall."""
    assert initial_storage_strength(0.0, s_base=S_BASE_DEFAULT, k=K_DEFAULT) > 0.0


# ── the stamp, through the real capture door ─────────────────────────────────


def _capture(hippo: Hippocampus, signals: EncodingSignals) -> str:
    return hippo.capture(encoding=signals)


def test_capture_stamps_tag_and_strength():
    hippo = Hippocampus(HippocampusConfig())
    mid = _capture(hippo, _signals(pain=1.0))
    memory = hippo.get(mid)
    assert memory.encoding_tag == pytest.approx(1.0)
    assert memory.storage_strength == pytest.approx(S_BASE_DEFAULT * (1 + K_DEFAULT))


def test_a_boring_capture_stamps_the_floor_not_zero():
    hippo = Hippocampus(HippocampusConfig())
    memory = hippo.get(_capture(hippo, EncodingSignals.unmeasured("loop")))
    assert memory.encoding_tag == 0.0
    assert memory.storage_strength == pytest.approx(S_BASE_DEFAULT)


def test_the_knobs_reach_the_stamp():
    hippo = Hippocampus(HippocampusConfig(strength_s_base=2.0, strength_k=3.0))
    memory = hippo.get(_capture(hippo, _signals(surprise=1.0)))
    assert memory.storage_strength == pytest.approx(8.0)


def test_the_stamp_survives_a_save_load_round_trip(tmp_path):
    hippo = Hippocampus(HippocampusConfig())
    mid = _capture(hippo, _signals(salience=0.9, pain=0.3))
    stamped = hippo.get(mid)
    path = tmp_path / "hippocampus.json"
    hippo.save(str(path))

    restored = Hippocampus(HippocampusConfig())
    restored.load(str(path))
    loaded = restored.get(mid)
    assert loaded.encoding_tag == pytest.approx(stamped.encoding_tag)
    assert loaded.storage_strength == pytest.approx(stamped.storage_strength)


def test_a_file_written_before_phase_2c_loads_as_never_stamped(tmp_path):
    """Absent is not zero: the strategy must encode such a trace, not treat it as worthless."""
    hippo = Hippocampus(HippocampusConfig())
    mid = _capture(hippo, _signals(pain=1.0))
    path = tmp_path / "hippocampus.json"
    hippo.save(str(path))

    import json

    raw = json.loads(path.read_text())
    stripped = 0
    for record in raw["memories"].values() if isinstance(raw.get("memories"), dict) else raw.get("memories", []):
        if isinstance(record, dict):
            record.pop("storage_strength", None)
            record.pop("encoding_tag", None)
            stripped += 1
    assert stripped, "the fixture stripped nothing -- the persisted shape moved"
    path.write_text(json.dumps(raw))

    restored = Hippocampus(HippocampusConfig())
    restored.load(str(path))
    loaded = restored.get(mid)
    assert loaded.storage_strength is None
    assert loaded.encoding_tag is None


def test_novelty_weighting_makes_the_same_signals_encode_differently_in_a_fuller_store():
    """Why the tag is stamped and never recomputed."""
    hippo = Hippocampus(HippocampusConfig())
    first = hippo.get(_capture(hippo, _signals(novelty=1.0)))
    for _ in range(200):
        _capture(hippo, EncodingSignals.unmeasured("loop"))
    later = hippo.get(_capture(hippo, _signals(novelty=1.0)))
    assert later.encoding_tag > first.encoding_tag
    assert first.encoding_tag == pytest.approx(0.0)  # an empty store's judgement is worth nothing


# ── the promise that nothing reads it yet ────────────────────────────────────


def test_retention_scoring_still_ignores_strength():
    """Byte-identical default retention: 2c-2 records, 2c-3 is what reads."""
    import inspect

    from maxim.memory import strategies

    source = inspect.getsource(strategies)
    assert "storage_strength" not in source
    assert "encoding_tag" not in source


@pytest.mark.parametrize("bad", [0.0, -1.0, math.inf, math.nan])
def test_an_unusable_knob_is_rejected_where_it_was_set(bad):
    """Not inside capture(): the async worker's broad handler would swallow it and LOSE the memory,
    reporting nothing about the real mistake."""
    with pytest.raises(ValueError):
        HippocampusConfig(strength_s_base=bad)


def test_stamping_cannot_fail_for_any_valid_config():
    """With the config validated at construction, every reachable capture stamps."""
    hippo = Hippocampus(HippocampusConfig(strength_s_base=1e-6, strength_k=0.0))
    mid = _capture(hippo, _signals(pain=1.0))
    assert hippo.get(mid).storage_strength == pytest.approx(1e-6)


def test_compression_carries_the_stamp_rather_than_resetting_it():
    """A compressed episode keeps how well-learned it was; re-encoding from scratch would make
    every compression look like a fresh, weakly-encoded memory."""
    from maxim.memory.types import CompressedMemory

    hippo = Hippocampus(HippocampusConfig())
    memory = hippo.get(_capture(hippo, _signals(pain=1.0)))
    compressed = CompressedMemory.from_episodic(memory, edge_count=0)
    assert compressed.storage_strength == pytest.approx(memory.storage_strength)
    assert compressed.encoding_tag == pytest.approx(memory.encoding_tag)
    assert CompressedMemory.from_dict(compressed.to_dict()).storage_strength == pytest.approx(memory.storage_strength)


def test_pressure_counts_when_an_aversive_action_relieved_nothing():
    """Drowning: air pressure 1.0 with air relief 0.0 recorded. The executor writes 0.0 for a drive
    the action moved away from comfort, so relevance is the KEY's presence, not a positive value."""
    drowning = _signals(drive_pressure=(("air", 1.0),), drive_relief=(("air", 0.0),))
    assert encoding_tag(drowning, novelty_reference_size=1000) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "field_name,bad",
    [
        ("storage_strength", -5.0),
        ("storage_strength", 0.0),
        ("storage_strength", float("nan")),
        ("encoding_tag", 2.0),
        ("encoding_tag", -0.5),
        ("encoding_tag", float("inf")),
    ],
)
def test_nonsense_on_disk_loads_as_never_stamped(field_name, bad):
    """A negative S would make R = exp(-dt/S) exceed 1: a trace more retrievable as it ages."""
    from maxim.memory.types import EpisodicMemory

    hippo = Hippocampus(HippocampusConfig())
    memory = hippo.get(_capture(hippo, _signals(pain=1.0)))
    raw = memory.to_dict()
    raw[field_name] = bad
    assert getattr(EpisodicMemory.from_dict(raw), field_name) is None


def test_the_tag_does_not_scale_with_how_many_drives_a_body_has():
    """Per-drive deviations made relief 0.3 tag 0.657 on a 3-drive body and 0.942 on an 8-drive one,
    so tags were not comparable across bodies -- which is what cross-body transfer rests on."""
    tags = []
    for n in (1, 3, 8):
        drives = tuple((f"d{i}", 0.3) for i in range(n))
        tags.append(encoding_tag(_signals(drive_relief=drives), novelty_reference_size=1000))
    assert tags == [pytest.approx(0.3)] * 3


def test_relief_is_weighted_by_how_badly_the_body_wanted_each_drive():
    """Relieving a drive the body was desperate for beats topping up a satisfied one."""
    desperate = _signals(
        drive_relief=(("air", 1.0), ("curiosity", 0.0)),
        drive_pressure=(("air", 1.0), ("curiosity", 0.0)),
    )
    # air carries all the weight: the relief channel reads 1.0, not the plain mean 0.5.
    assert encoding_tag(desperate, novelty_reference_size=1000) == pytest.approx(1.0)

    satisfied = _signals(
        drive_relief=(("air", 0.0), ("curiosity", 1.0)),
        drive_pressure=(("air", 1.0), ("curiosity", 0.0)),
    )
    # the relief that happened was of a drive nothing wanted; pressure's max still counts once.
    assert encoding_tag(satisfied, novelty_reference_size=1000) == pytest.approx(1.0 - 1.0 * 0.0)


def test_with_no_pressure_measured_the_weights_carry_no_information():
    """Falls back to the plain mean rather than inventing a ranking."""
    signals = _signals(drive_relief=(("a", 0.2), ("b", 0.8)))
    assert encoding_tag(signals, novelty_reference_size=1000) == pytest.approx(0.5)


def test_pressure_is_counted_once_per_action_not_once_per_drive():
    one = _signals(drive_relief=(("a", 0.0),), drive_pressure=(("a", 0.6),))
    many = _signals(
        drive_relief=(("a", 0.0), ("b", 0.0), ("c", 0.0)),
        drive_pressure=(("a", 0.6), ("b", 0.6), ("c", 0.6)),
    )
    assert encoding_tag(one, novelty_reference_size=1000) == pytest.approx(0.6)
    assert encoding_tag(many, novelty_reference_size=1000) == pytest.approx(0.6)


def test_the_reference_set_novelty_was_judged_against_is_recorded():
    """So a trace can say WHICH denominator it used when 2b-iii moves it to the producer."""
    hippo = Hippocampus(HippocampusConfig())
    first = hippo.get(_capture(hippo, _signals(novelty=1.0)))
    assert first.novelty_reference_size == 0
    for _ in range(5):
        _capture(hippo, EncodingSignals.unmeasured("loop"))
    later = hippo.get(_capture(hippo, _signals(novelty=1.0)))
    assert later.novelty_reference_size == 6


def test_the_recorded_reference_size_survives_persistence(tmp_path):
    hippo = Hippocampus(HippocampusConfig())
    for _ in range(3):
        _capture(hippo, EncodingSignals.unmeasured("loop"))
    mid = _capture(hippo, _signals(novelty=1.0))
    path = tmp_path / "h.json"
    hippo.save(str(path))
    restored = Hippocampus(HippocampusConfig())
    restored.load(str(path))
    assert restored.get(mid).novelty_reference_size == 3
