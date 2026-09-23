"""The strength model, where storage strength is finally READ (memory-strength Phase 2c-3).

``R = exp(-dt/S)`` on the EXPERIENCE clock, the credited-gap retrieval update through
``strategy.on_activation``, and the promise that none of it happens unless
``memory.strategy=strength`` is set.

Every arm here is written so that deleting the mechanism it names makes it FAIL -- several were
rewritten after exactly that check (see the anti-vacuity arms at the bottom, which pin the
properties that a passing-but-blind test would have missed).
"""

from __future__ import annotations

import json
import math

import pytest

from maxim.memory.encoding import EncodingSignals
from maxim.memory.experience_clock import ExperienceClock, ExperienceClockStalled
from maxim.memory.hippocampus import Hippocampus, HippocampusConfig
from maxim.memory.strategies import (
    CREDITED_GAP_US,
    PROTECTION_FLOOR_WEIGHT,
    RETRIEVAL_SOURCE_WEIGHTS,
    TAG_FADE_MULTIPLIER,
    AccessBasedStrategy,
    CompositeStrategy,
    ImportanceBasedStrategy,
    StrengthStrategy,
    TemporalAwareStrategy,
)
from maxim.memory.types import CompressedMemory, EpisodicMemory

SECOND = 1_000_000  # one second of experience, in the clock's own microseconds


def _signals(**kw) -> EncodingSignals:
    base = dict(
        site="loop", salience=None, novelty=None, surprise=None, pain=None, drive_pressure=None, drive_relief=None
    )
    base.update(kw)
    return EncodingSignals(**base)


def _trace(*, strength: float | None = 10 * SECOND, tag: float | None = 0.0, anchor: int | None = 0) -> EpisodicMemory:
    return EpisodicMemory(
        id="m1", timestamp=1.0, storage_strength=strength, encoding_tag=tag, retrievability_anchor_us=anchor
    )


def _strategy(clock: ExperienceClock, **kw) -> StrengthStrategy:
    kw.setdefault("s_base", 10 * SECOND)
    return StrengthStrategy(clock, **kw)


# ── R = exp(-dt/S), on the experience clock ──────────────────────────────────


def test_retrievability_is_the_plan_equation_to_the_digit():
    clock = ExperienceClock()
    strategy = _strategy(clock)
    trace = _trace(strength=10 * SECOND)
    clock.advance(10 * SECOND)
    assert strategy.score_for_retention(trace, 0.0) == pytest.approx(math.exp(-1.0))
    clock.advance(10 * SECOND)
    assert strategy.score_for_retention(trace, 0.0) == pytest.approx(math.exp(-2.0))


def test_s_is_carried_in_the_clocks_own_unit_so_no_conversion_exists():
    """A seconds-vs-microseconds seam here forgets everything in 10 microseconds and every test
    still passes. One second of experience against S = one second must read exactly 1/e."""
    clock = ExperienceClock()
    trace = _trace(strength=float(SECOND))
    clock.advance(SECOND)
    assert _strategy(clock).score_for_retention(trace, 0.0) == pytest.approx(1 / math.e)


def test_wall_clock_time_is_ignored_entirely():
    """The ``now`` the sleep path passes is ``time.time()``. Forgetting must track EXPERIENCE --
    a robot switched off for a month does not wake with its memory wiped."""
    clock = ExperienceClock()
    strategy = _strategy(clock)
    trace = _trace(strength=10 * SECOND)
    for wall_now in (0.0, 1e9, -1e9, 1e12):
        assert strategy.score_for_retention(trace, wall_now) == 1.0
    clock.advance(10 * SECOND)
    assert strategy.score_for_retention(trace, 0.0) == pytest.approx(math.exp(-1.0))


def test_a_restarted_clock_never_makes_a_trace_more_retrievable():
    """A corrupt clock record restarts at 0 while anchors stay large: dt would go NEGATIVE and
    R = exp(-dt/S) would exceed 1 -- a trace that gets easier to recall as it ages.

    Asserted on the RAW retrievability, not on the score: ``score_for_retention``'s own
    ``min(1.0, ...)`` hides an R of 148 completely, and this test passed with the clamp deleted
    until it was checked by deleting it. What an unclamped R actually breaks is downstream --
    ``1 - R`` goes negative and a retrieval SHRINKS the trace it was supposed to strengthen.
    """
    clock = ExperienceClock()
    strategy = _strategy(clock)
    trace = _trace(strength=10 * SECOND, anchor=500 * SECOND)
    assert strategy.retrievability(trace) == 1.0
    assert strategy.score_for_retention(trace, 0.0) == 1.0

    # The gap check refuses a negative elapsed time before the update runs, so it is the gap check
    # -- not a clamp inside it -- that stops a backwards clock from crediting anything. Asserted as
    # the refusal, because the earlier spelling ("S did not go down") held by equality whatever the
    # update said, and so could not see what it claimed to (review round, Executor #9).
    assert strategy.on_activation(trace, strategy.activation_now(), "tool") is False
    assert trace.storage_strength == 10 * SECOND


def test_the_strategy_refuses_to_exist_without_a_clock():
    """Without one it would score every trace at R = 1 forever: not "no forgetting" but "silently
    broken", and indistinguishable from the outside."""
    with pytest.raises(ValueError, match="ExperienceClock"):
        StrengthStrategy(None)  # type: ignore[arg-type]


def test_a_record_that_carries_no_strength_is_a_loud_error_not_an_immortal_one():
    """An ATL concept reaching this model would otherwise score R = 1 forever (plan decision 5:
    Phase 2's model is the hippocampal one)."""
    from maxim.memory.semantic_types import SemanticMemory

    with pytest.raises(TypeError, match="storage strength"):
        _strategy(ExperienceClock()).score_for_retention(SemanticMemory(id="c", timestamp=1.0), 0.0)


def test_an_unstamped_trace_is_read_as_freshly_encoded_not_as_worthless():
    """Absent is not zero: a file written before Phase 2c must not be deleted for lacking a field.

    Scored after the clock has MOVED, so the fallback value is actually visible -- with dt at 0
    every S in the world gives R = 1 and this test could not see the difference between
    ``s_base`` and a fallback of one microsecond (found by substituting one).
    """
    clock = ExperienceClock()
    strategy = _strategy(clock, s_base=10 * SECOND)
    unstamped = _trace(strength=None, anchor=0)
    assert strategy.score_for_retention(unstamped, 0.0) == 1.0
    clock.advance(10 * SECOND)
    assert strategy.score_for_retention(unstamped, 0.0) == pytest.approx(math.exp(-1.0))
    assert strategy.score_for_retention(_trace(strength=None, anchor=None), 0.0) == 1.0


# ── the credited-gap retrieval update ────────────────────────────────────────


def test_a_credited_retrieval_raises_s_and_resets_r():
    clock = ExperienceClock()
    strategy = _strategy(clock)
    trace = _trace(strength=10 * SECOND)
    clock.advance(10 * SECOND)
    assert strategy.score_for_retention(trace, 0.0) < 0.4

    assert strategy.on_activation(trace, strategy.activation_now(), "tool") is True
    assert trace.storage_strength > 10 * SECOND
    assert trace.retrievability_anchor_us == 10 * SECOND
    assert strategy.score_for_retention(trace, 0.0) == 1.0


def test_an_activation_inside_the_credited_gap_changes_nothing():
    clock = ExperienceClock()
    strategy = _strategy(clock)
    trace = _trace(strength=10 * SECOND)
    clock.advance(CREDITED_GAP_US - 1)
    before = (trace.storage_strength, trace.retrievability_anchor_us)
    assert strategy.on_activation(trace, strategy.activation_now(), "tool") is False
    assert (trace.storage_strength, trace.retrievability_anchor_us) == before


def test_massing_cannot_buy_what_spacing_earns():
    """The plan's correction: a trace re-rendered every tick used to out-earn a genuinely spaced
    recall over the same span, because ``1 - e^-x <= x`` makes many tiny gains beat one large one."""

    def run(gaps: list[int]) -> float:
        clock = ExperienceClock()
        strategy = _strategy(clock)
        trace = _trace(strength=10 * SECOND)
        for gap in gaps:
            clock.advance(gap)
            strategy.on_activation(trace, strategy.activation_now(), "tool")
        return trace.storage_strength

    every_tick = run([SECOND // 10] * 100)  # 100 renders, 0.1 s apart -- 10 s of deliberation
    at_the_gap = run([CREDITED_GAP_US] * 5)  # what those 10 s are WORTH once massing is discounted
    genuinely_spaced = run([20 * SECOND] * 5)  # the same five retrievals, spread out

    assert every_tick == pytest.approx(at_the_gap), "massed exposure bought more than its gap allows"
    assert genuinely_spaced > every_tick * 1.5, "the spacing effect is not the right way round"


def test_the_gain_is_largest_when_the_trace_had_faded():
    def gain_after(dt: int) -> float:
        clock = ExperienceClock()
        strategy = _strategy(clock)
        trace = _trace(strength=10 * SECOND)
        clock.advance(dt)
        strategy.on_activation(trace, strategy.activation_now(), "tool")
        return trace.storage_strength

    assert gain_after(40 * SECOND) > gain_after(10 * SECOND) > gain_after(3 * SECOND)


def test_effortful_recall_is_worth_more_than_re_exposure():
    """The testing effect: a memory the LLM asked for beats one enrichment rendered at it."""

    def gain(source: str) -> float:
        clock = ExperienceClock()
        strategy = _strategy(clock)
        trace = _trace(strength=10 * SECOND)
        clock.advance(20 * SECOND)
        strategy.on_activation(trace, strategy.activation_now(), source)
        return trace.storage_strength

    assert gain("tool") > gain("replan") > gain("enrichment")
    assert gain("enrichment") == pytest.approx(gain("prediction"))


def test_every_activation_source_has_a_declared_weight():
    """The weights map duplicates the closed source vocabulary. A source added there and forgotten
    here would credit nothing and say nothing, so the equality is pinned rather than commented."""
    from maxim.memory.types import ACTIVATION_SOURCES

    assert set(RETRIEVAL_SOURCE_WEIGHTS) == set(ACTIVATION_SOURCES)


def test_a_source_with_no_declared_weight_credits_nothing():
    clock = ExperienceClock()
    strategy = _strategy(clock, source_weights={"tool": 1.0})
    trace = _trace(strength=10 * SECOND)
    clock.advance(20 * SECOND)
    assert strategy.on_activation(trace, strategy.activation_now(), "enrichment") is False
    assert trace.storage_strength == 10 * SECOND


def test_repeated_spaced_retrievals_saturate_rather_than_compound_without_bound():
    """Isolates the ``(S/s_base)^-w`` factor from the ``1 - R`` one.

    The gaps below grow WITH S, so every retrieval finds the trace equally faded (``1 - R`` pinned
    at ~1) and the only thing left that can shrink the gain is saturation. With a fixed gap the
    ``1 - R`` term shrinks on its own and this test passed with the saturation factor deleted --
    found by deleting it.
    """
    clock = ExperienceClock()
    strategy = _strategy(clock)
    trace = _trace(strength=10 * SECOND)
    gains = []
    for _ in range(6):
        before = trace.storage_strength
        clock.advance(int(before * 50))  # 50 time-constants: R is 0 to every digit that matters
        assert strategy.retrievability(trace) == pytest.approx(0.0, abs=1e-12)
        strategy.on_activation(trace, strategy.activation_now(), "tool")
        gains.append(trace.storage_strength / before)
    assert gains == sorted(gains, reverse=True)
    assert gains[-1] < gains[0] * 0.9, "the gain never saturated -- S compounds without bound"


# ── protection is a floor, never immortality ─────────────────────────────────


def test_a_strongly_tagged_trace_outlives_a_boring_one():
    clock = ExperienceClock()
    strategy = _strategy(clock)
    drowning = _trace(strength=10 * SECOND, tag=1.0)
    boring = _trace(strength=10 * SECOND, tag=0.0)
    clock.advance(30 * SECOND)
    assert strategy.score_for_retention(drowning, 0.0) > strategy.score_for_retention(boring, 0.0)
    assert strategy.score_for_retention(drowning, 0.0) == pytest.approx(
        PROTECTION_FLOOR_WEIGHT * math.exp(-30 / (10 * TAG_FADE_MULTIPLIER))
    )


def test_the_protection_floor_itself_fades_so_nothing_is_immortal():
    """``access_count >= 10``'s 0.8 floor never lifted; that is what this plan exists to remove."""
    clock = ExperienceClock()
    strategy = _strategy(clock)
    maximal = _trace(strength=10 * SECOND, tag=1.0)
    scores = []
    for _ in range(6):
        clock.advance(100 * SECOND)
        scores.append(strategy.score_for_retention(maximal, 0.0))
    assert scores == sorted(scores, reverse=True)
    assert scores[-1] < HippocampusConfig().retention_threshold, "a maximally tagged trace never became forgettable"


def test_the_stamped_tag_is_history_and_the_fade_is_only_a_view():
    clock = ExperienceClock()
    strategy = _strategy(clock)
    trace = _trace(strength=10 * SECOND, tag=0.8)
    clock.advance(500 * SECOND)
    strategy.score_for_retention(trace, 0.0)
    assert trace.encoding_tag == 0.8, "scoring rewrote what the trace was encoded with"


def test_a_trace_held_up_by_its_tag_keeps_its_detail():
    """Gist is what survives ordinary fading; detail is most of what a tagged trace MEANT."""
    clock = ExperienceClock()
    strategy = _strategy(clock)
    clock.advance(30 * SECOND)
    assert strategy.should_compress(_trace(strength=10 * SECOND, tag=1.0), 0.0) is False
    assert strategy.should_compress(_trace(strength=10 * SECOND, tag=0.0), 0.0) is True
    already = CompressedMemory(id="c", timestamp=1.0, storage_strength=float(SECOND), retrievability_anchor_us=0)
    assert strategy.should_compress(already, 0.0) is False


# ── the capability, and the wrappers that must not drop it ───────────────────


def test_only_the_strength_model_declares_it_needs_the_clock():
    assert StrengthStrategy(ExperienceClock()).requires_experience_clock is True
    assert AccessBasedStrategy().requires_experience_clock is False
    assert ImportanceBasedStrategy().requires_experience_clock is False


@pytest.mark.parametrize("wrap", ["composite", "temporal"])
def test_a_wrapper_carries_the_capability_and_the_hook_through(wrap, scn):
    clock = ExperienceClock()
    inner = _strategy(clock)
    wrapped = (
        CompositeStrategy([(inner, 0.5), (AccessBasedStrategy(), 0.5)])
        if wrap == "composite"
        else TemporalAwareStrategy(scn, base_strategy=inner)
    )
    assert wrapped.requires_experience_clock is True
    assert wrapped.activation_now() == 0.0
    clock.advance(20 * SECOND)
    trace = _trace(strength=10 * SECOND)
    assert wrapped.on_activation(trace, wrapped.activation_now(), "tool") is True
    assert trace.storage_strength > 10 * SECOND


def test_a_blend_of_timeless_models_needs_no_clock():
    blend = CompositeStrategy([(AccessBasedStrategy(), 0.6), (ImportanceBasedStrategy(), 0.4)])
    assert blend.requires_experience_clock is False
    assert blend.activation_now() == 0.0


# ── through the real store ───────────────────────────────────────────────────


def _hippo(**kw) -> Hippocampus:
    kw.setdefault("persistence_path", None)
    kw.setdefault("memory_strategy", "strength")
    # NOT ``10 * SECOND``: that is byte-identical to ``S_BASE_DEFAULT``, so every arm below passed
    # with the ``s_base=`` argument dropped from the strategy construction entirely (review round,
    # Executor #4 -- a vacuous fixture rather than a vacuous test).
    kw.setdefault("strength_s_base", 7 * SECOND)
    return Hippocampus(HippocampusConfig(**kw))


def test_capture_stamps_the_anchor_from_the_stores_own_clock():
    hippo = _hippo()
    hippo.experience_clock.advance(7 * SECOND)
    memory = hippo.get(hippo.capture(encoding=EncodingSignals.unmeasured("loop")))
    assert memory.retrievability_anchor_us == 7 * SECOND


def test_the_store_hands_the_strategy_its_own_live_clock(tmp_path):
    """Built by reference, so a strategy held across a load reads the RESTORED time -- ``load``
    restores the clock in place exactly so references survive."""
    hippo = _hippo(persistence_path=str(tmp_path / "h.json"))
    strategy = hippo.activation_strategy()
    assert strategy.clock is hippo.experience_clock
    hippo.experience_clock.advance(3 * SECOND)
    hippo.save()
    reloaded = _hippo(persistence_path=str(tmp_path / "h.json"))
    held = reloaded.activation_strategy()
    reloaded.load()
    assert held.clock.now_us() == 3 * SECOND


def test_activating_through_the_store_credits_the_trace():
    hippo = _hippo()
    mid = hippo.capture(encoding=EncodingSignals.unmeasured("loop"))
    before = hippo.get(mid).storage_strength
    hippo.experience_clock.advance(20 * SECOND)
    assert hippo.activate([mid], source="tool") == 1
    memory = hippo.get(mid)
    assert memory.storage_strength > before
    assert memory.retrievability_anchor_us == 20 * SECOND
    assert memory.activation_count == 1  # Phase 1's counter still counts, credited or not


def test_an_uncredited_activation_is_still_counted():
    hippo = _hippo()
    mid = hippo.capture(encoding=EncodingSignals.unmeasured("loop"))
    before = hippo.get(mid).storage_strength
    hippo.activate([mid], source="tool")
    hippo.activate([mid], source="tool")
    memory = hippo.get(mid)
    assert memory.activation_count == 2
    assert memory.storage_strength == before


def test_sleep_forgets_the_faded_and_keeps_what_mattered():
    hippo = _hippo(auto_save_after_sleep=False)
    faded = hippo.capture(encoding=EncodingSignals.unmeasured("loop"))
    drowning = hippo.capture(encoding=_signals(pain=1.0))
    hippo.experience_clock.advance(60 * SECOND)
    hippo.sleep()
    assert hippo.get(faded) is None, "a fully faded trace survived"
    assert hippo.get(drowning) is not None, "a maximally tagged trace was forgotten"


def test_the_anchor_survives_a_save_load_round_trip(tmp_path):
    hippo = _hippo(persistence_path=str(tmp_path / "h.json"))
    hippo.experience_clock.advance(5 * SECOND)
    mid = hippo.capture(encoding=EncodingSignals.unmeasured("loop"))
    hippo.save()
    restored = _hippo(persistence_path=str(tmp_path / "h.json"))
    restored.load()
    assert restored.get(mid).retrievability_anchor_us == 5 * SECOND


def test_compression_carries_the_anchor_rather_than_resetting_it():
    memory = EpisodicMemory(id="m", timestamp=1.0, storage_strength=float(SECOND), retrievability_anchor_us=9 * SECOND)
    compressed = CompressedMemory.from_episodic(memory, edge_count=0)
    assert compressed.retrievability_anchor_us == 9 * SECOND
    assert CompressedMemory.from_dict(compressed.to_dict()).retrievability_anchor_us == 9 * SECOND


def test_a_trace_from_before_the_anchor_starts_living_at_load_rather_than_forever(tmp_path):
    """No anchor means no ``dt``, which means R = 1 forever -- the immortality this plan removes."""
    hippo = _hippo(persistence_path=str(tmp_path / "h.json"))
    mid = hippo.capture(encoding=EncodingSignals.unmeasured("loop"))
    hippo.experience_clock.advance(40 * SECOND)
    hippo.save()

    raw = json.loads((tmp_path / "h.json").read_text())
    stripped = 0
    for record in raw["memories"]:
        stripped += record.pop("retrievability_anchor_us", None) is not None
    assert stripped == 1, "the fixture stripped nothing -- the persisted shape moved"
    (tmp_path / "h.json").write_text(json.dumps(raw))

    restored = _hippo(persistence_path=str(tmp_path / "h.json"))
    restored.load()
    memory = restored.get(mid)
    assert memory.retrievability_anchor_us == 40 * SECOND
    restored.experience_clock.advance(400 * SECOND)
    assert restored.activation_strategy().score_for_retention(memory, 0.0) < 0.01


@pytest.mark.parametrize("bad", [-1, 1.5, "0", True])
def test_a_nonsense_anchor_on_disk_loads_as_never_anchored(bad):
    memory = EpisodicMemory(id="m", timestamp=1.0, retrievability_anchor_us=5)
    raw = memory.to_dict()
    raw["retrievability_anchor_us"] = bad
    assert EpisodicMemory.from_dict(raw).retrievability_anchor_us is None


# ── the stalled-clock assert ─────────────────────────────────────────────────


def _hub(strategy: str):
    from maxim.decisions.nac import NAc
    from maxim.integration.memory_hub import MemoryHub
    from maxim.similarity.ec import EntorhinalCortex
    from maxim.time.scn import SCN

    return MemoryHub(
        hippocampus=_hippo(memory_strategy=strategy),
        scn=SCN(),
        nac=NAc(),
        ec=EntorhinalCortex(),
        _allow_raw=True,
    )


@pytest.mark.parametrize("closer", ["on_session_end", "on_session_end_lightweight"])
def test_a_stalled_clock_is_loud_on_both_session_end_paths(closer):
    hub = _hub("strength")
    hub.on_session_start()
    hub.hippocampus.capture(encoding=EncodingSignals.unmeasured("loop"))
    with pytest.raises(ExperienceClockStalled) as caught:
        getattr(hub, closer)()
    assert caught.value.captures == 1
    assert caught.value.results is not None, "the diagnostic cost the caller its session results"


@pytest.mark.parametrize("closer", ["on_session_end", "on_session_end_lightweight"])
def test_activations_alone_are_enough_to_make_a_stalled_clock_loud(closer):
    hub = _hub("strength")
    mid = hub.hippocampus.capture(encoding=EncodingSignals.unmeasured("loop"))
    hub.on_session_start()  # the capture happened BEFORE this session
    hub.hippocampus.activate([mid], source="tool")
    with pytest.raises(ExperienceClockStalled) as caught:
        getattr(hub, closer)()
    assert (caught.value.captures, caught.value.activations) == (0, 1)


def test_an_advanced_clock_is_silent():
    hub = _hub("strength")
    hub.on_session_start()
    hub.hippocampus.capture(encoding=EncodingSignals.unmeasured("loop"))
    hub.hippocampus.experience_clock.advance(1)
    assert hub.on_session_end_lightweight()["lightweight"] is True


def test_an_honestly_empty_session_is_silent():
    """Nothing happened, so nothing should have aged. Raising here would make every idle shutdown
    a failure and the assert something nobody reads."""
    hub = _hub("strength")
    hub.on_session_start()
    assert hub.on_session_end_lightweight()["lightweight"] is True


@pytest.mark.parametrize("strategy", ["access_based", "importance_based", "composite"])
def test_the_assert_never_fires_for_a_model_that_does_not_run_on_experience(strategy):
    hub = _hub(strategy)
    hub.on_session_start()
    hub.hippocampus.capture(encoding=EncodingSignals.unmeasured("loop"))
    assert hub.on_session_end_lightweight()["lightweight"] is True


def test_the_configured_s_base_reaches_the_SCORER_not_only_the_stamp():
    """Two different things read ``memory.s_base``: the capture stamp and the model.

    Only the stamp was covered, and the fixture's value happened to equal ``S_BASE_DEFAULT``, so
    the strategy could have been built with no ``s_base`` at all (review round, Executor #4). The
    consequence if it regresses: traces stamped at the operator's base, scored against 10 s.
    """
    hippo = _hippo(strength_s_base=3 * SECOND)
    assert hippo.activation_strategy().s_base == 3 * SECOND
    memory = hippo.get(hippo.capture(encoding=EncodingSignals.unmeasured("loop")))
    assert memory.storage_strength == pytest.approx(3 * SECOND)  # tag 0, so S0 == s_base


def test_a_resumed_session_is_still_watched():
    """The hole this closes: ``load`` RESTORES the clock, so a snapshot-and-compare on ``now_us``
    sees the restored jump and concludes the clock is healthy -- on exactly the resumed harnesses
    the assert exists for (review round, Executor #2, reproduced before the fix).

    Every other stalled-clock arm starts the clock at 0, which is why none of them could see it.
    """
    # A previous run's file, written by a DIFFERENT store -- the shape a resume actually has.
    previous = _hippo()
    previous.experience_clock.advance(4 * SECOND)
    state = previous.dump()

    hub = _hub("strength")  # fresh process: nothing has driven this clock
    assert hub.hippocampus.experience_clock.advanced_us() == 0
    hub.on_session_start()
    hub.hippocampus.load_state(state)  # --resume-sim: the clock comes back reading 4 s
    assert hub.hippocampus.experience_clock.now_us() == 4 * SECOND
    hub.hippocampus.capture(encoding=EncodingSignals.unmeasured("loop"))
    with pytest.raises(ExperienceClockStalled):
        hub.on_session_end_lightweight()


def test_the_snapshot_is_taken_from_the_session_not_assumed_to_be_zero():
    """A clock already carrying experience when the session opens: still stalled if nothing drives
    it DURING the session. Pins the snapshot itself, which a clock starting at 0 cannot."""
    hub = _hub("strength")
    hub.hippocampus.experience_clock.advance(9 * SECOND)
    hub.on_session_start()
    hub.hippocampus.capture(encoding=EncodingSignals.unmeasured("loop"))
    with pytest.raises(ExperienceClockStalled):
        hub.on_session_end_lightweight()


def test_a_load_is_not_experience_but_a_real_advance_still_is():
    clock = ExperienceClock()
    clock.advance(5)
    restored = ExperienceClock(9_000)
    clock.restore(restored)
    assert clock.now_us() == 9_000
    assert clock.advanced_us() == 5, "a load counted as experience"
    clock.advance(3)
    assert clock.advanced_us() == 8


def test_a_raised_s_base_cannot_buy_a_trace_more_than_one_retrieval_is_worth():
    """An operator raising ``memory.s_base`` leaves already-stamped traces with ``S < s_base``, so
    the saturating factor exceeds 1 and a single retrieval multiplies ``S`` by far more than the
    documented ``1 + a*w`` bound -- ~11x at a 100-fold raise (review round, Executor #7)."""
    clock = ExperienceClock()
    retuned = _strategy(clock, s_base=1000 * SECOND)  # the operator's new, much slower base
    legacy = _trace(strength=10 * SECOND)  # stamped under the old one
    clock.advance(500 * SECOND)
    retuned.on_activation(legacy, retuned.activation_now(), "tool")

    bound = 10 * SECOND * (1.0 + retuned.gain * 1.0)  # a * w_src at their maximum
    assert legacy.storage_strength <= bound, "one retrieval bought more than the documented bound"


def test_compression_reads_the_strength_pair_through_the_LOCKED_reader():
    """``from_episodic`` runs under the store's write lock while ``activate`` deliberately holds no
    store lock, so a credited retrieval can be mid-write on the very record being compressed. Four
    separate unlocked attribute reads could freeze a NEW ``S`` beside an OLD anchor into the
    compressed record that REPLACES the episode -- persisted, unlike a torn save, which the next
    save corrects (review round, Executor #3).

    Structural, deliberately: the window is two adjacent ``STORE_ATTR`` instructions inside
    ``update_strength_atomically``, which no scheduler can be asked to land in reliably. What is
    checkable is that there is exactly ONE reader of the pair and that compression uses it.
    """
    import ast
    import inspect

    from maxim.memory.types import CompressedMemory, _strength_fields

    source = inspect.getsource(CompressedMemory.from_episodic)
    tree = ast.parse(inspect.cleandoc(source))
    starred = {
        node.value.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.keyword) and node.arg is None and isinstance(node.value, ast.Call)
        for _ in [0]
        if isinstance(node.value.func, ast.Name)
    }
    assert "_strength_fields" in starred, "compression does not read the pair through the locked reader"
    assert "_touch_lock" in inspect.getsource(_strength_fields), "the one reader stopped taking the lock"


def test_the_stall_is_logged_before_it_is_raised(caplog):
    """Every production session-end caller catches broad ``Exception``, and two log it at DEBUG --
    so an exception alone is the silent failure wearing a type (review round, cross-confirmed)."""
    import logging

    hub = _hub("strength")
    hub.on_session_start()
    hub.hippocampus.capture(encoding=EncodingSignals.unmeasured("loop"))
    with caplog.at_level(logging.ERROR), pytest.raises(ExperienceClockStalled):
        hub.on_session_end_lightweight()
    assert any("stalled" in r.getMessage() and r.levelno >= logging.ERROR for r in caplog.records)


@pytest.mark.parametrize("module", ["maxim.runtime.bio_integration", "maxim.simulation.interactive"])
def test_the_two_sim_paths_name_the_exception_before_their_broad_handler(module):
    """A typed handler, ahead of the ``except Exception`` that would bury it -- and it uses
    ``exc.results``, so the diagnostic does not also cost the run its session telemetry."""
    import ast
    import importlib
    import inspect

    tree = ast.parse(inspect.getsource(importlib.import_module(module)))
    named = [
        h
        for node in ast.walk(tree)
        if isinstance(node, ast.Try)
        for h in node.handlers
        if isinstance(h.type, ast.Name) and h.type.id == "ExperienceClockStalled"
    ]
    assert named, f"{module} does not name ExperienceClockStalled"
    for handler in named:
        body = ast.dump(ast.Module(body=handler.body, type_ignores=[]))
        assert "results" in body, "the handler ignores exc.results"


def test_the_gate_reads_a_capability_not_a_strategy_name():
    """A third-party model running on experience time must get the same assert, and comparing
    against the name ``"strength"`` would be a second source of truth it could never satisfy."""
    import inspect

    from maxim.integration import memory_hub

    source = inspect.getsource(memory_hub.MemoryHub._assert_experience_advanced)
    assert "requires_experience_clock" in source
    assert '"strength"' not in source and "'strength'" not in source

    hub = _hub("access_based")
    hub.on_session_start()
    hub.hippocampus.capture(encoding=EncodingSignals.unmeasured("loop"))

    class _OwnModel(AccessBasedStrategy):
        requires_experience_clock = True

    hub.hippocampus._build_base_strategy = lambda: _OwnModel()  # a strategy the codebase never names
    with pytest.raises(ExperienceClockStalled):
        hub.on_session_end_lightweight()
