"""Memory-strength 2d-2: retroactive tagging (docs/plans/memory_2d2_retroactive_tagging.md).

At consolidation, a trace whose encoding_tag is STRICTLY above the threshold reaches back over the
related traces encoded before it within the cutoff and raises their ``retro_tag`` to
``tag_e * exp(-dt/tau) * rel`` (max, never sum). Relatedness = the fraction of the event's world/audio
clusters the earlier trace shares; no situation, no tag. Every trace here goes through the real
``capture()`` with real encoding signals, and every tag through the real ``sleep()``.
"""

from __future__ import annotations

import dataclasses
import math

import pytest

from maxim.memory.encoding import EncodingSignals

S = 1_000_000  # one second of experience, in µs
T0 = 100.0  # every test time is offset from here: experience time is non-negative (the 2d-1 door)
WATER = {"world": "c-water", "interoception": "c-drowning"}
SHORE = {"world": "c-shore", "interoception": "c-calm"}


def _hippo(tmp_path, **cfg):
    from maxim.memory.hippocampus import Hippocampus, HippocampusConfig

    config = HippocampusConfig(
        persistence_path=str(tmp_path / "hippo.json"), enable_sleep_consolidation=False, auto_save_after_sleep=False
    )
    return Hippocampus(dataclasses.replace(config, **cfg))


def _cap(hippo, t_s: float, situation, pain: float = 0.0):
    signals = dataclasses.replace(EncodingSignals.unmeasured("loop"), pain=pain)
    return hippo.get(hippo.capture(encoding=signals, situation=situation, experience_us=int((T0 + t_s) * S)))


def test_a_strong_event_tags_a_related_trace_encoded_before_it(tmp_path):
    hippo = _hippo(tmp_path)
    lead_in = _cap(hippo, 0.0, WATER)
    event = _cap(hippo, 2.0, WATER, pain=1.0)
    hippo.sleep()
    assert lead_in.retro_tag == pytest.approx(math.exp(-2.0 / 10.0))  # tag 1.0 * exp(-dt/tau) * rel 1.0
    assert event.retro_tag is None  # never tags itself


@pytest.mark.parametrize(
    ("first_at", "first_situation", "why"),
    [
        (0.0, SHORE, "unrelated: a different world cluster"),
        (-31.0, WATER, "outside the 30 s cutoff"),
        (3.0, WATER, "encoded AFTER the event (forward window 0)"),
        (0.0, None, "no situation, no relatedness"),
        (0.0, {"world": "c-other", "interoception": "c-drowning"}, "shares only interoception"),
    ],
)
def test_what_is_not_tagged(tmp_path, first_at, first_situation, why):
    hippo = _hippo(tmp_path)
    other = _cap(hippo, first_at, first_situation)
    _cap(hippo, 2.0, WATER, pain=1.0)
    hippo.sleep()
    assert other.retro_tag is None, why


def test_the_threshold_is_strict(tmp_path):
    """A first outcome's surprise is exactly 0.5: an event AT the threshold tags nothing."""
    at, above = _hippo(tmp_path / "a"), _hippo(tmp_path / "b")
    t_at, t_above = _cap(at, 0.0, WATER), _cap(above, 0.0, WATER)
    _cap(at, 1.0, WATER, pain=0.5)
    _cap(above, 1.0, WATER, pain=0.51)
    at.sleep()
    above.sleep()
    assert t_at.retro_tag is None and t_above.retro_tag is not None


def test_tags_saturate_at_the_strongest_event_never_sum(tmp_path):
    hippo = _hippo(tmp_path)
    trace = _cap(hippo, 0.0, WATER)
    _cap(hippo, 1.0, WATER, pain=0.6)
    _cap(hippo, 2.0, WATER, pain=1.0)
    hippo.sleep()
    strongest = max(0.6 * math.exp(-0.1), 1.0 * math.exp(-0.2))
    assert trace.retro_tag == pytest.approx(strongest)


def test_resolution_is_idempotent(tmp_path):
    hippo = _hippo(tmp_path)
    trace = _cap(hippo, 0.0, WATER)
    _cap(hippo, 1.0, WATER, pain=1.0)
    hippo.sleep()
    first = trace.retro_tag
    with hippo._rwlock.write():
        assert hippo._resolve_retro_tags_locked() == 0  # re-resolving raises nothing...
    hippo.sleep()
    assert trace.retro_tag == first  # ...and leaves the same value (max, never sum)


def test_a_capture_that_lands_after_a_sleep_is_still_resolved(tmp_path):
    """An async capture reserves its seq when QUEUED and is stored later -- possibly after a sleep that
    saw a later seq. Both directions must still resolve at the next sleep: the late strong event tags
    its lead-in, and a late lead-in is tagged by an event already resolved."""
    hippo = _hippo(tmp_path)
    late_event_seq, late_lead_seq = hippo.next_capture_seq(), hippo.next_capture_seq()
    lead_in = _cap(hippo, 0.0, WATER)
    resolved_event = _cap(hippo, 10.0, SHORE, pain=1.0)
    hippo.sleep()  # sees the highest seq before either late capture lands
    shore_lead = hippo.get(
        hippo.capture(
            encoding=EncodingSignals.unmeasured("loop"),
            situation=SHORE,
            experience_us=int((T0 + 9.0) * S),
            capture_seq=late_lead_seq,
        )
    )
    signals = dataclasses.replace(EncodingSignals.unmeasured("loop"), pain=1.0)
    hippo.capture(encoding=signals, situation=WATER, experience_us=int((T0 + 1.0) * S), capture_seq=late_event_seq)
    assert lead_in.retro_tag is None and shore_lead.retro_tag is None
    hippo.sleep()
    assert lead_in.retro_tag == pytest.approx(math.exp(-0.1))
    assert shore_lead.retro_tag == pytest.approx(resolved_event.encoding_tag * math.exp(-0.1))


def test_a_strong_event_without_a_situation_tags_nothing(tmp_path):
    """The sync pain/reflection traces of an action carry no situation, so they never reach back."""
    hippo = _hippo(tmp_path)
    lead_in = _cap(hippo, 0.0, WATER)
    _cap(hippo, 1.0, None, pain=1.0)
    hippo.sleep()
    assert lead_in.retro_tag is None


def test_the_window_constants_come_from_the_config(tmp_path):
    hippo = _hippo(tmp_path, retro_tau_us=5 * S, retro_cutoff_us=3 * S)
    near, far = _cap(hippo, 0.0, WATER), _cap(hippo, -2.0, WATER)
    _cap(hippo, 2.0, WATER, pain=1.0)
    hippo.sleep()
    assert near.retro_tag == pytest.approx(math.exp(-2.0 / 5.0))
    assert far.retro_tag is None  # 4 s > the 3 s cutoff


def test_retro_tag_survives_a_save_and_compression(tmp_path):
    from maxim.memory.types import CompressedMemory, EpisodicMemory

    hippo = _hippo(tmp_path)
    trace = _cap(hippo, 0.0, WATER)
    _cap(hippo, 1.0, WATER, pain=1.0)
    hippo.sleep()
    hippo.save()
    reloaded = _hippo(tmp_path)
    reloaded.load()
    back = reloaded.get(trace.id)
    assert back.retro_tag == pytest.approx(trace.retro_tag)
    assert CompressedMemory.from_episodic(back).retro_tag == pytest.approx(trace.retro_tag)
    data = EpisodicMemory.from_dict(back.to_dict()).to_dict()
    data["retro_tag"] = 2.0  # out of range loads as never tagged
    assert EpisodicMemory.from_dict(data).retro_tag is None


def test_only_the_strength_model_reads_it(tmp_path):
    """Under ``strength`` a retro-tagged trace inside the reach is protected; the default strategy's
    score does not move -- the default path stays byte-identical."""
    from maxim.memory.strategies import AccessBasedStrategy, StrengthStrategy

    hippo = _hippo(tmp_path)
    tagged = _cap(hippo, 0.0, WATER)
    plain = _cap(hippo, 0.0, SHORE)
    _cap(hippo, 1.0, WATER, pain=1.0)
    # 15 s past the captures (stamped at T0 + t): R alone is e^-1.5 ~ 0.22, the retro floor
    # 0.5 * e^-0.1 * e^-0.15 ~ 0.39 -- so only the tagged trace is held up by its floor.
    hippo.experience_clock.advance(int((T0 + 15.0) * S) - hippo.experience_clock.now_us())
    access = AccessBasedStrategy()
    before = access.score_for_retention(tagged, 0.0)
    hippo.sleep()
    assert tagged.retro_tag is not None and plain.retro_tag is None
    assert access.score_for_retention(tagged, 0.0) == before
    strength = StrengthStrategy(hippo.experience_clock, s_base=hippo.config.strength_s_base)
    held, fading = strength.score_for_retention(tagged, 0.0), strength.score_for_retention(plain, 0.0)
    assert fading == pytest.approx(math.exp(-1.5)) and held == pytest.approx(0.5 * math.exp(-0.1 - 0.15))


@pytest.mark.parametrize("bad", [0, -1, 1.5, True])
def test_the_window_config_refuses_a_non_positive_int(tmp_path, bad):
    with pytest.raises(ValueError, match="retro_tau_us"):
        _hippo(tmp_path, retro_tau_us=bad)


def test_the_clustering_sleep_resolves_too(tmp_path):
    """``sleep_with_clustering`` (the console's consolidation, when an SCN is connected) removes
    traces on its own path; it must resolve tags first, exactly as ``sleep()`` does."""
    from maxim.time.scn import SCN

    hippo = _hippo(tmp_path)
    hippo.connect_scn(SCN())
    lead_in = _cap(hippo, 0.0, WATER)
    _cap(hippo, 1.0, WATER, pain=1.0)
    hippo.sleep_with_clustering()
    assert lead_in.retro_tag == pytest.approx(math.exp(-0.1))


def test_under_strength_a_real_sleep_keeps_the_tagged_trace_whole_and_drops_its_twin(tmp_path):
    """The removal path itself, not just the score: 15 s on, the untagged twin (R ~ 0.22) falls under
    the 0.3 retention threshold and is removed; the retro-tagged trace (floor ~ 0.39) is kept, and --
    held by its floor rather than fading -- is NOT compressed (``should_compress`` reads the same floor)."""
    from maxim.memory.strategies import StrengthStrategy
    from maxim.memory.types import CompressedMemory

    hippo = _hippo(tmp_path, memory_strategy="strength", enable_sleep_consolidation=True)
    tagged, plain = _cap(hippo, 0.0, WATER), _cap(hippo, 0.0, SHORE)
    _cap(hippo, 1.0, WATER, pain=1.0)
    hippo.experience_clock.advance(int((T0 + 15.0) * S) - hippo.experience_clock.now_us())
    hippo.sleep()
    kept = hippo.get(tagged.id)
    assert kept is not None and not isinstance(kept, CompressedMemory)
    assert hippo.get(plain.id) is None
    strength = StrengthStrategy(hippo.experience_clock, s_base=hippo.config.strength_s_base)
    assert strength.should_compress(kept, 0.0) is False


@pytest.mark.parametrize("bad", [True, float("nan"), -0.1, 2.0, "0.5"])
def test_a_malformed_retro_tag_loads_as_never_tagged(tmp_path, bad):
    from maxim.memory.types import EpisodicMemory

    hippo = _hippo(tmp_path)
    data = _cap(hippo, 0.0, WATER).to_dict()
    data["retro_tag"] = bad
    assert EpisodicMemory.from_dict(data).retro_tag is None


def test_only_the_loop_capture_path_stamps_a_situation():
    """Guards the build note "same action, by construction": one action's loop, pain and reflection
    traces never tag each other ONLY because the loop capture is the sole production site passing a
    situation. A new site that does must revisit the resolver (e.g. add an action key) and this list."""
    import ast
    import pathlib

    import maxim

    root = pathlib.Path(maxim.__file__).parent
    capture_calls = {"capture", "capture_from_loop", "capture_from_loop_async", "_capture_episodic"}
    allowed = {
        "memory/hippocampus.py",  # the capture doors threading the argument through
        "runtime/bio_integration.py",  # the loop capture: _capture_episodic -> capture_from_loop_async
        "runtime/agent_loop.py",  # hands the loop proposal's clusters to _capture_episodic
    }
    found = set()
    for path in root.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
            if name not in capture_calls:
                continue
            for kw in node.keywords:
                if kw.arg == "situation" and not (isinstance(kw.value, ast.Constant) and kw.value.value is None):
                    found.add(path.relative_to(root).as_posix())
    assert found <= allowed, f"a new capture site stamps a situation: {sorted(found - allowed)}"
    assert {"runtime/bio_integration.py", "runtime/agent_loop.py"} <= found  # the scan still sees the loop


def test_a_trace_without_a_capture_seq_takes_no_part(tmp_path):
    """Only a malformed load leaves ``encoded_at_us`` without ``capture_seq``; such a trace has no
    total order against a same-moment neighbour (it could tag it at dt = 0), so it is skipped."""
    hippo = _hippo(tmp_path)
    unordered = _cap(hippo, 0.0, WATER)
    unordered.capture_seq = None
    _cap(hippo, 1.0, WATER, pain=1.0)
    hippo.sleep()
    assert unordered.retro_tag is None
