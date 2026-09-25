"""Memory-strength 2S-d: the situation cue (docs/plans/memory_2s_d_situation_cue.md).

On a situation CHANGE, the tick's ``{modality: cluster}`` ids cue the ATL concepts with those ids;
their linked memories (2S-b's refs, formed here by the real ConceptExtractor link) are scored on
their OWN recorded situation, qualify only through a shared world/audio cluster, and the best tier is
returned. Recall only: nothing is activated until 2S-e consumes it.
"""

from __future__ import annotations

import ast
import dataclasses
import logging
import pathlib
from unittest.mock import MagicMock

import pytest

from maxim.memory.encoding import EncodingSignals

S = 1_000_000


@pytest.fixture
def world(tmp_path):
    from maxim.memory.atl import ATL
    from maxim.memory.concept_extractor import ConceptExtractor
    from maxim.memory.hippocampus import Hippocampus, HippocampusConfig
    from maxim.memory.pattern_completer import PatternCompleter

    hippo = Hippocampus(HippocampusConfig(persistence_path=str(tmp_path / "h.json"), memory_strategy="strength"))
    atl = ATL()
    extractor = ConceptExtractor(atl=atl, cross_layer=MagicMock())
    ids = {}
    for name in ("water", "shore", "cave", "calm", "drowning"):
        ids[name], _ = atl.find_or_create(name=f"sensors:{name}", category="sensor", definition=name)

    def capture(t_s, pain=None, **situation):
        sit = {m: ids[c] for m, c in situation.items()} or None
        signals = dataclasses.replace(EncodingSignals.unmeasured("loop"), pain=pain)
        mid = hippo.capture(encoding=signals, situation=sit, experience_us=int((100 + t_s) * S))
        extractor._process_capture(mid, hippo.recall_by_ids([mid])[0])  # the real 2S-b link
        return mid

    def cue(**situation):
        return {m: ids[c] for m, c in situation.items()}

    completer = PatternCompleter(atl=atl, layers={"hippocampus": hippo})
    yield completer, capture, cue, hippo, atl, ids
    extractor.shutdown()


def test_a_situation_change_recalls_its_memories_newest_first_and_the_same_situation_does_not(world):
    completer, capture, cue, *_ = world
    old, new = capture(0, world="water"), capture(5, world="water")
    capture(3, world="shore")
    assert completer.cue_situation("a", cue(world="water")) == (new, old)
    assert completer.cue_situation("a", cue(world="water")) == ()  # unchanged: no re-cue
    assert completer.cue_situation("b", cue(world="water")) == (new, old)  # per agent


def test_the_place_outranks_the_sound_and_a_shared_sound_only_orders_inside_the_place_tier(world):
    completer, capture, cue, *_ = world
    both = capture(0, world="water", audio="drowning")  # older, but also shares the sound
    same_place = capture(1, world="water", audio="shore")
    same_sound = capture(2, world="cave", audio="drowning")
    assert completer.recall_situation(cue(world="water", audio="drowning")) == (both, same_place)
    assert completer.recall_situation(cue(world="calm", audio="shore")) == (same_place,)  # no place: sound tier
    assert same_sound not in completer.recall_situation(cue(world="water", audio="drowning"))


def test_a_drowning_with_a_different_sound_is_not_dropped_below_safe_swims(world):
    """The re-review's case: today's tick hears the splash that safe swims share; the drowning was
    recorded with another sound. It is the same PLACE, so it stays in the tier -- and, as the most
    salient, comes first."""
    completer, capture, cue, *_ = world
    drowning = capture(0, pain=1.0, world="water", audio="cave")
    safe = [capture(10 + i, world="water", audio="shore") for i in range(3)]
    recalled = completer.recall_situation(cue(world="water", audio="shore"))
    assert recalled[0] == drowning and set(safe) <= set(recalled)


def test_a_retro_tagged_lead_in_ranks_as_salient_like_the_strength_floor_reads_it(world):
    completer, capture, cue, hippo, *_ = world
    lead_in = capture(0, world="water")
    later = capture(5, world="water")
    hippo.recall_by_ids([lead_in])[0].retro_tag = 0.9  # 2d-2 raised it at sleep
    assert completer.recall_situation(cue(world="water")) == (lead_in, later)


def test_interoception_never_ranks_so_the_drowning_is_recalled_at_the_dive_start(world):
    """The review's scenario: at second 0 of a dive the cue carries FULL air. Past safe swims share it,
    past drownings do not -- yet the drownings are the memories to recall. Interoception must not
    split the tier, and within it the strongly encoded memory comes first."""
    completer, capture, cue, *_ = world
    drowning = capture(0, pain=1.0, world="water", interoception="drowning")
    safe = capture(5, world="water", interoception="calm")
    assert completer.recall_situation(cue(world="water", interoception="calm")) == (drowning, safe)


def test_a_long_run_of_uneventful_visits_cannot_crowd_out_the_one_that_hurt(world):
    completer, capture, cue, *_ = world
    drowning = capture(0, pain=1.0, world="water")
    for i in range(completer.MAX_EPISODES + 5):
        capture(10 + i, world="water")
    recalled = completer.recall_situation(cue(world="water"))
    assert len(recalled) == completer.MAX_EPISODES and recalled[0] == drowning


def test_interoception_alone_never_qualifies_a_memory(world):
    """In a new place the only shared cluster is the broad interoception one: recall nothing."""
    completer, capture, cue, *_ = world
    capture(0, world="shore", interoception="calm")
    assert completer.cue_situation("a", cue(world="cave", interoception="calm")) == ()


def test_the_refs_only_nominate_a_memory_is_scored_on_its_own_situation(world):
    completer, capture, cue, hippo, atl, ids = world
    stray = capture(0)  # no situation, so 2S-b links nothing...
    atl.recall_by_ids([ids["water"]])[0].add_ref("hippocampus", stray)  # ...a stray ref nominates it anyway
    assert completer.cue_situation("a", cue(world="water")) == ()


def test_the_cue_is_recall_only_no_activation_and_no_strength_credit(world):
    completer, capture, cue, hippo, *_ = world
    mid = capture(0, world="water")
    hippo.experience_clock.advance(200 * S)  # past the credit gap: a credited activation WOULD move S
    record = hippo.recall_by_ids([mid])[0]
    before = (record.activation_count, record.storage_strength, record.retrievability_anchor_us)
    work = hippo.session_work()
    assert completer.cue_situation("a", cue(world="water")) == (mid,)
    assert (record.activation_count, record.storage_strength, record.retrievability_anchor_us) == before
    assert hippo.session_work() == work


def test_the_cue_does_not_touch_the_concepts_or_the_memories(world):
    completer, capture, cue, hippo, atl, ids = world
    mid = capture(0, world="water")
    concept, record = atl.recall_by_ids([ids["water"]])[0], hippo.recall_by_ids([mid])[0]
    before = (concept.access_count, record.access_count)
    completer.cue_situation("a", cue(world="water"))
    assert (concept.access_count, record.access_count) == before


def test_a_new_session_makes_the_next_cue_an_entry_and_stats_count_what_was_found(world):
    completer, capture, cue, *_ = world
    mid = capture(0, world="water")
    assert completer.cue_situation("a", cue(world="water")) == (mid,)
    completer.reset_situations()
    assert completer.cue_situation("a", cue(world="water")) == (mid,)
    completer.cue_situation("a", cue(world="cave"))  # a change that finds nothing
    assert completer.situation_cue_stats() == {"cues": 3, "changes": 3, "with_matches": 2, "matched": 2}


# ── the hub accessor and the session reset ────────────────────────────────────


def test_the_hub_refuses_to_hand_over_a_cue_it_does_not_have_and_resets_at_session_start(tmp_path):
    from maxim.runtime.bio_stack import build_bio_stack

    hub = build_bio_stack(agent_id="a", persistence_dir=str(tmp_path / "bio")).memory_hub
    completer = hub._pattern_completer
    assert hub.situation_cue == completer.cue_situation
    completer._last_situation["a"] = {"world": "x"}
    hub.on_session_end()
    hub.on_session_start()
    assert completer._last_situation == {}
    hub._pattern_completer = None
    with pytest.raises(RuntimeError, match="NO_SITUATION_CUE"):
        hub.situation_cue


# ── propose_via_substrate: the required, typed seam ──────────────────────────


def _propose(**kw):
    from tests.unit.test_modality_seam import _ClusterRecordingNac, _multi_drive_body, _StubExecutor

    from maxim.runtime.agent_loop import propose_via_substrate
    from maxim.similarity.ec import EntorhinalCortex
    from maxim.similarity.encoder import SensorEncoder

    emb = _multi_drive_body()
    nac = _ClusterRecordingNac()
    executor = _StubExecutor(["infant_turn_left"], embodiment=emb)
    enc = SensorEncoder(ec=EntorhinalCortex(), atl=None)
    propose_via_substrate(nac=nac, agent_id="infant", executor=executor, sensor_encoder=enc, **kw)
    return nac


def test_the_cue_is_required_and_none_is_not_an_opt_out():
    with pytest.raises(TypeError):
        _propose()
    with pytest.raises(TypeError, match="NO_SITUATION_CUE"):
        _propose(situation_cue=None)


def test_the_tick_hands_its_clusters_to_the_cue_and_a_failing_cue_does_not_cost_the_tick(caplog):
    calls = []
    nac = _propose(situation_cue=lambda agent, clusters: calls.append((agent, clusters)))
    assert calls and calls[0][0] == "infant" and calls[0][1] == dict(nac.seen_clusters)

    def broken(agent, clusters):
        raise ValueError("boom")

    with caplog.at_level(logging.DEBUG):
        _propose(situation_cue=broken)  # never raised...
    assert any("boom" in (r.getMessage() + str(r.exc_info)) for r in caplog.records)  # ...but reported


def test_every_production_call_passes_the_cue():
    """The harnesses call ``propose_via_substrate`` directly and are not run by the suite, so the
    runtime TypeError alone would surface only on the rig: pin every call site statically."""
    repo = pathlib.Path(__file__).resolve().parents[2]  # THIS checkout, never an installed shadow
    missing, seen = [], 0
    for path in [*(repo / "src").rglob("*.py"), *(repo / "scripts").rglob("*.py")]:
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)  # f(...) and mod.f(...)
            if name == "propose_via_substrate":
                seen += 1
                if "situation_cue" not in {kw.arg for kw in node.keywords}:
                    missing.append(f"{path.relative_to(repo)}:{node.lineno}")
    assert seen >= 7, seen  # the loop, water_trial x2, exp58_run x2, offline gates, exp53 readout
    assert not missing, missing


def test_the_real_hub_composes_capture_link_and_cue(tmp_path):
    """Not a hand-composed chain: the hub's own capture -> ConceptExtractor link -> situation cue."""
    import time

    from maxim.runtime.bio_stack import build_bio_stack

    hub = build_bio_stack(agent_id="a", persistence_dir=str(tmp_path / "bio")).memory_hub
    water, _ = hub.atl.find_or_create(name="sensors:water", category="sensor", definition="water")
    mid = hub.hippocampus.capture(encoding=EncodingSignals.unmeasured("loop"), situation={"world": water})
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and mid not in hub.atl.recall_by_ids([water])[0].memory_refs.get(
        "hippocampus", {}
    ):
        time.sleep(0.02)  # the extractor links on its own worker thread
    assert hub.situation_cue("a", {"world": water}) == (mid,)


def test_a_failed_recall_does_not_mark_the_situation_as_cued(world, monkeypatch):
    completer, capture, cue, *_ = world
    mid = capture(0, world="water")

    def boom(_cue):
        raise RuntimeError("dictionary changed size during iteration")

    monkeypatch.setattr(completer, "recall_situation", boom)
    with pytest.raises(RuntimeError):
        completer.cue_situation("a", cue(world="water"))
    monkeypatch.undo()
    assert completer.cue_situation("a", cue(world="water")) == (mid,)  # retried, not silenced


def test_the_stateless_recall_does_not_move_change_detection(world):
    """2S-e completes a cue to a NEIGHBOURING situation and recalls with it; that must not overwrite
    the agent's last situation."""
    completer, capture, cue, *_ = world
    mid, shore = capture(0, world="water"), capture(1, world="shore")
    assert completer.cue_situation("a", cue(world="shore")) == (shore,)
    assert completer.recall_situation(cue(world="water")) == (mid,)
    # still "shore": unchanged, so no re-cue (a corrupted last situation would recall `shore` again)
    assert completer.cue_situation("a", cue(world="shore")) == ()
    assert completer.situation_cue_stats()["cues"] == 2


def test_the_session_end_reports_what_the_cue_found(tmp_path):
    from maxim.runtime.bio_stack import build_bio_stack

    hub = build_bio_stack(agent_id="a", persistence_dir=str(tmp_path / "bio")).memory_hub
    hub.on_session_start()
    hub.situation_cue("a", {"world": "nowhere"})
    report = hub.on_session_end()
    assert report["situation_cue_cues"] == 1 and report["situation_cue_changes"] == 1
    assert report["situation_cue_with_matches"] == 0
    hub.on_session_start()  # the sim path closes with the LIGHTWEIGHT end: it reports too
    hub.situation_cue("a", {"world": "elsewhere"})
    assert hub.on_session_end_lightweight()["situation_cue_cues"] == 2  # cumulative for the hub's life


def test_the_loop_resolves_its_cue_once_and_degrades_loudly(tmp_path, caplog):
    from maxim.runtime.agent_loop import NO_SITUATION_CUE, _resolve_situation_cue
    from maxim.runtime.bio_stack import build_bio_stack

    assert _resolve_situation_cue(None) is NO_SITUATION_CUE  # no memory at all: the explicit opt-out
    hub = build_bio_stack(agent_id="a", persistence_dir=str(tmp_path / "bio")).memory_hub
    assert _resolve_situation_cue(hub) == hub.situation_cue
    hub._pattern_completer = None  # a hub whose ATL failed: the loop runs on, but SAYS so
    with caplog.at_level(logging.WARNING):
        assert _resolve_situation_cue(hub) is NO_SITUATION_CUE
    assert any("no situation cue this run" in r.getMessage() for r in caplog.records)


def test_the_loop_sensor_encoder_is_built_only_with_an_ec(tmp_path):
    from types import SimpleNamespace

    from maxim.runtime.agent_loop import _build_loop_sensor_encoder
    from maxim.runtime.bio_stack import build_bio_stack
    from maxim.similarity.encoder import SensorEncoder

    assert _build_loop_sensor_encoder(None, nac=None) is None
    assert _build_loop_sensor_encoder(SimpleNamespace(ec=None), nac=None) is None
    hub = build_bio_stack(agent_id="a", persistence_dir=str(tmp_path / "bio")).memory_hub
    assert isinstance(_build_loop_sensor_encoder(hub, nac=None), SensorEncoder)


def test_a_sensor_encoder_that_fails_to_build_is_reported_not_hidden(monkeypatch, caplog):
    from types import SimpleNamespace

    import maxim.similarity.encoder as encoder_module
    from maxim.runtime.agent_loop import _build_loop_sensor_encoder

    def broken(**_kw):
        raise ValueError("encoder boom")

    monkeypatch.setattr(encoder_module, "SensorEncoder", broken)
    with caplog.at_level(logging.DEBUG):
        assert _build_loop_sensor_encoder(SimpleNamespace(ec=object(), atl=None), nac=None) is None
    assert any("encoder boom" in (r.getMessage() + str(r.exc_info)) for r in caplog.records)
