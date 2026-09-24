"""What a memory was encoded WITH (memory-strength plan, Phase 2b-i).

Every capture must say, per signal, what it measured or that it measured nothing, and which site
captured it; the record rides on the trace through persistence and compression. Nothing reads it
yet (the Phase 2c strength strategy is its first consumer), so none of this changes what is kept or
forgotten.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from unittest.mock import MagicMock

import pytest

from maxim.memory.encoding import ENCODING_SITES, EncodingContractError, EncodingSignals
from maxim.memory.hippocampus import Hippocampus, HippocampusConfig
from maxim.memory.types import CompressedMemory, EpisodicMemory, Perception

_SIGNALS = ("salience", "novelty", "surprise", "pain")


def _hippo() -> Hippocampus:
    return Hippocampus(HippocampusConfig(persistence_path=None))


def _only(site: str = "api", **measured: float) -> EncodingSignals:
    per_drive = {name: measured.get(name) for name in ("drive_pressure", "drive_relief")}
    return EncodingSignals(site=site, **{name: measured.get(name) for name in _SIGNALS}, **per_drive)


# ── the type ─────────────────────────────────────────────────────────────────


def test_the_type_declares_exactly_the_decided_fields():
    assert {f.name for f in dataclasses.fields(EncodingSignals)} == {
        "site",
        *_SIGNALS,
        "drive_pressure",
        "drive_relief",
        "extra",
    }


def test_every_field_is_required():
    with pytest.raises(TypeError):
        EncodingSignals(site="api", salience=None)  # type: ignore[call-arg]


def test_the_site_is_a_closed_vocabulary():
    assert {"loop", "memory_agent", "pain_bus", "reflexion", "engram", "observation", "api"} == ENCODING_SITES
    with pytest.raises(ValueError, match="site"):
        EncodingSignals.unmeasured("lop")


@pytest.mark.parametrize("bad", [-0.1, 1.1, float("nan"), float("inf")])
def test_values_outside_the_unit_interval_are_refused(bad):
    with pytest.raises(ValueError):
        _only(pain=bad)


def test_a_bool_is_not_a_measurement():
    with pytest.raises(TypeError):
        _only(pain=True)


def test_the_record_is_a_historical_fact():
    signals = _only(pain=0.5)
    with pytest.raises(dataclasses.FrozenInstanceError):
        signals.pain = 0.9  # type: ignore[misc]


def test_unmeasured_is_explicit_and_measured_names_what_was():
    assert EncodingSignals.unmeasured("loop").measured() == ()
    assert _only(surprise=0.4, pain=0.9).measured() == ("surprise", "pain")


def test_round_trip_keeps_unknown_keys_and_refuses_bad_extras():
    data = {**_only(site="loop", pain=0.5).to_dict(), "future_signal": 0.25}
    restored = EncodingSignals.from_dict(data)
    assert (restored.site, restored.pain) == ("loop", 0.5) and restored.to_dict() == data
    assert EncodingSignals.from_dict({"site": "api"}).measured() == ()  # a missing signal is "not measured"
    with pytest.raises(ValueError, match="site"):
        EncodingSignals.from_dict({"pain": 0.5})  # a missing SITE is malformed, never guessed
    with pytest.raises(ValueError, match="collide"):
        EncodingSignals(
            site="api",
            salience=None,
            novelty=None,
            surprise=None,
            pain=None,
            drive_pressure=None,
            drive_relief=None,
            extra={"pain": 1},
        )
    with pytest.raises(ValueError, match="JSON"):
        EncodingSignals(
            site="api",
            salience=None,
            novelty=None,
            surprise=None,
            pain=None,
            drive_pressure=None,
            drive_relief=None,
            extra={"x": object()},
        )


# ── the Hippocampus requires it at every door ───────────────────────────────


def test_capture_requires_the_encoding(complete_memory_args):
    args = {k: v for k, v in complete_memory_args.items() if k != "encoding"}
    with pytest.raises(TypeError, match="encoding"):
        _hippo().capture(**args)
    with pytest.raises(EncodingContractError, match="encoding"):
        _hippo().capture(**args, encoding={"pain": 0.5})  # a dict is not the contract


def test_capture_from_loop_requires_the_encoding():
    with pytest.raises(TypeError, match="encoding"):
        _hippo().capture_from_loop({}, None, {}, {}, {}, None, situation=None)


def test_the_async_path_fails_on_the_callers_thread_not_in_the_worker():
    hippo = _hippo()
    with pytest.raises(TypeError, match="encoding"):
        hippo.capture_from_loop_async(
            observation={}, state=None, intent={}, decision={}, action={}, result=None, situation=None
        )
    with pytest.raises(EncodingContractError):
        hippo.capture_from_loop_async(
            observation={}, state=None, intent={}, decision={}, action={}, result=None, situation=None, encoding=None
        )
    assert hippo._capture_queue.qsize() == 0  # nothing was queued


def test_the_async_path_carries_the_encoding_to_the_trace():
    hippo = _hippo()
    hippo.capture_from_loop_async(
        observation={},
        state=None,
        intent={},
        decision={},
        action={},
        result=None,
        situation=None,
        encoding=_only("loop", surprise=0.3),
    )
    hippo._process_capture(hippo._capture_queue.get_nowait())
    [trace] = list(hippo)
    assert (trace.encoding.site, trace.encoding.surprise) == ("loop", 0.3)


def test_store_observation_declares_its_constants_unmeasured():
    hippo = _hippo()
    mid = hippo.store_observation("a quiet corridor")
    assert hippo.recall_by_ids([mid])[0].encoding == EncodingSignals.unmeasured("observation")


# ── persistence and compression ──────────────────────────────────────────────


def test_the_encoding_survives_save_and_load(tmp_path, complete_memory_args):
    hippo = _hippo()
    mid = hippo.capture(**{**complete_memory_args, "encoding": _only("pain_bus", pain=0.7, surprise=0.2)})
    path = str(tmp_path / "hippocampus.json")
    hippo.save(path)
    restored = _hippo()
    restored.load(path)
    assert restored.recall_by_ids([mid])[0].encoding == _only("pain_bus", pain=0.7, surprise=0.2)


def test_a_trace_from_before_encoding_loads_as_not_recorded():
    legacy = EpisodicMemory(id="old", timestamp=1.0).to_dict()
    del legacy["encoding"]
    assert EpisodicMemory.from_dict(legacy).encoding is None  # distinct from unmeasured(site)


def test_one_malformed_record_never_fails_the_whole_load(caplog, complete_memory_args):
    hippo = _hippo()
    good = hippo.capture(**complete_memory_args)
    bad = hippo.capture(**complete_memory_args)
    state = hippo.dump()
    for m in state["memories"]:
        if m["id"] == bad:
            m["encoding"] = {"site": "api", "pain": 7.0}  # out of range
    restored = _hippo()
    with caplog.at_level(logging.WARNING):
        restored.load_state(state)
    assert restored.recall_by_ids([good])[0].encoding is not None
    assert restored.recall_by_ids([bad])[0].encoding is None
    assert any("malformed encoding" in r.getMessage() for r in caplog.records)


def test_compression_carries_the_encoding():
    full = EpisodicMemory(id="m", timestamp=1.0, encoding=_only(novelty=0.6))
    compressed = CompressedMemory.from_dict(CompressedMemory.from_episodic(full).to_dict())
    assert compressed.encoding == _only(novelty=0.6)


# ── each live site records what it MEASURED, and nothing it did not ─────────


def test_the_loop_capture_records_the_outcomes_surprise():
    from maxim.runtime.bio_integration import capture_episodic_memory
    from maxim.tools.base import ToolOutput

    hippo = MagicMock()
    common = dict(situation=None, executor=None, state=None, intent={}, action={"tool_name": "grab"}, run_id="r")
    capture_episodic_memory(
        hippocampus=hippo, observation={"salience": 0.5}, result=ToolOutput(success=False, rpe=0.4), **common
    )
    assert hippo.capture_from_loop_async.call_args.kwargs["encoding"] == _only("loop", surprise=0.4)
    capture_episodic_memory(hippocampus=hippo, observation={}, result="not a ToolOutput", **common)
    assert hippo.capture_from_loop_async.call_args.kwargs["encoding"] == EncodingSignals.unmeasured("loop")


def test_only_a_capture_contract_break_escapes_the_loop_capture():
    from maxim.runtime.bio_integration import capture_episodic_memory

    common = dict(
        situation=None, executor=None, observation={}, state=None, intent={}, action={}, result=None, run_id=""
    )
    hippo = MagicMock()
    hippo.capture_from_loop_async.side_effect = EncodingContractError("encoding missing")
    with pytest.raises(EncodingContractError):
        capture_episodic_memory(hippocampus=hippo, **common)
    hippo.capture_from_loop_async.side_effect = TypeError("dict() on a non-mapping")  # an ordinary failure
    capture_episodic_memory(hippocampus=hippo, **common)  # logged, as before


def _pain(pain_type, intensity=0.8):
    from maxim.proprioception.pain import PainSignal

    return PainSignal(pain_type=pain_type, intensity=intensity, timestamp=0.0)


def test_every_pain_type_is_classified_and_only_nociception_is_pain():
    from maxim.proprioception.pain import PainType
    from maxim.proprioception.pain_bus import NOCICEPTIVE_PAIN_TYPES, _pain_encoding

    for pain_type in PainType:
        encoding = _pain_encoding(_pain(pain_type))
        assert encoding.site == "pain_bus"
        if pain_type in NOCICEPTIVE_PAIN_TYPES:
            assert encoding.measured() == ("pain",) and encoding.pain == 0.8
        else:
            assert encoding.measured() == (), pain_type
    assert PainType.TOOL_FAILURE not in NOCICEPTIVE_PAIN_TYPES  # a frustration count, not injury
    assert PainType.EXTERNAL_SIGNAL in NOCICEPTIVE_PAIN_TYPES  # world / game damage


def test_anticipated_pain_is_a_labelled_prediction_never_measured_pain():
    from maxim.proprioception.pain import PainType
    from maxim.proprioception.pain_bus import _pain_encoding

    encoding = _pain_encoding(_pain(PainType.ANTICIPATED, 0.6))
    assert encoding.pain is None and encoding.extra == {"anticipated_pain": 0.6}


def test_the_pain_subscriber_records_through_the_mapping():
    from maxim.proprioception.pain import PainType
    from maxim.proprioception.pain_bus import create_pain_memory_subscriber

    hippo = _hippo()
    subscriber = create_pain_memory_subscriber(hippo)
    subscriber(_pain(PainType.SUSTAINED_STRAIN))
    subscriber(_pain(PainType.TOOL_FAILURE))
    by_type = {t.perception.observations["pain_type"]: t.encoding for t in hippo}
    assert by_type["sustained_strain"] == _only("pain_bus", pain=0.8)
    assert by_type["tool_failure"] == EncodingSignals.unmeasured("pain_bus")
    assert all(t.perception.salience == pytest.approx(1.0) for t in hippo)  # default path unchanged (2b-ii)


def test_drive_pain_is_a_drive_not_injury_except_health():
    from maxim.proprioception.pain import PainSignal, PainType
    from maxim.proprioception.pain_bus import _pain_encoding

    def drive(name: str) -> EncodingSignals:
        return _pain_encoding(
            PainSignal(pain_type=PainType.EXTERNAL_SIGNAL, intensity=0.7, timestamp=0.0, context={"source": name})
        )

    air_hunger = drive("drive:oxygen")  # the Exp 60 air hunger: a drive, before any tissue damage
    assert air_hunger.pain is None and air_hunger.extra == {"drive_pain": 0.7, "drive": "oxygen"}
    assert drive("drive:food").pain is None
    assert drive("drive:health").pain == 0.7  # health loss IS tissue damage
    assert drive("embodiment").pain == 0.7  # a body failure mode: injury


def test_an_extra_survives_the_hippocampus_round_trip(tmp_path, complete_memory_args):
    extra = {"drive_pain": 0.7, "drive": "oxygen"}
    signals = EncodingSignals(
        site="pain_bus",
        salience=None,
        novelty=None,
        surprise=None,
        pain=None,
        drive_pressure=None,
        drive_relief=None,
        extra=extra,
    )
    hippo = _hippo()
    mid = hippo.capture(**{**complete_memory_args, "encoding": signals})
    path = str(tmp_path / "h.json")
    hippo.save(path)
    restored = _hippo()
    restored.load(path)
    assert restored.recall_by_ids([mid])[0].encoding.extra == extra  # == ignores extra, so check it directly


def test_a_reflection_records_the_surprise_of_the_failure_it_reflects_on():
    from maxim.bridges.tool_pain_bridge import ToolPainBridge

    hippo = _hippo()
    bridge = ToolPainBridge(nac=MagicMock(), pain_detector=MagicMock(), hippocampus=hippo)
    bridge._store_reflection("the grip slipped", {"tool_name": "grab"}, surprise=0.6)
    [trace] = list(hippo)
    assert trace.encoding == _only("reflexion", surprise=0.6)


def test_memory_agent_counts_only_detection_salience_as_measured():
    from maxim.agents.bus import Percept
    from maxim.agents.memory_agent import _percept_encoding

    seen = Percept(timestamp=time.time(), source="vision", detections=[{"class_id": 1}], salience=0.4, novelty=0.7)
    assert _percept_encoding(seen) == _only("memory_agent", salience=0.4, novelty=0.7)
    typed = Percept(timestamp=time.time(), source="cli", detections=[{"class_id": 1}], cli_input="hi", salience=0.9)
    assert _percept_encoding(typed) == EncodingSignals.unmeasured("memory_agent")  # 0.9 is a constant
    idle = Percept(timestamp=time.time(), source="idle")
    assert _percept_encoding(idle) == EncodingSignals.unmeasured("memory_agent")
    assert _percept_encoding(None) == EncodingSignals.unmeasured("memory_agent")


def test_the_dormant_engram_records_its_real_signals():
    from maxim.embodiment.cerebellum import Cerebellum

    hippo = _hippo()
    program = MagicMock(confidence=0.25, steps=[1], goal_signature="lift")
    program.name = "lift"
    mid = Cerebellum().form_engram(program, {}, "NEGATIVE", 1, hippo, pain_intensity=0.8, rpe_magnitude=0.5)
    assert mid is not None
    assert hippo.recall_by_ids([mid])[0].encoding == _only("engram", novelty=0.75, surprise=0.5, pain=0.8)


def test_perception_itself_is_untouched_by_recording(complete_memory_args):
    hippo = _hippo()
    mid = hippo.capture(**{**complete_memory_args, "perception": Perception(salience=0.95, novelty=0.1)})
    assert hippo.recall_by_ids([mid])[0].long_term  # immediate promotion still reads perception, as before


# ── the surprise a capture records never exceeds 1, at the PRODUCER ─────────


def _link_dict(pv: float | None) -> dict:
    from maxim.decisions.causal_link import CausalLink, TemporalDelta, Valence

    link = CausalLink(
        id="l",
        event_type="tool",
        event_signature="grab",
        event_context={},
        outcome_type="o",
        outcome_signature="o",
        outcome_valence=Valence.POSITIVE,
        temporal_delta=TemporalDelta(),
    )
    return {**link.to_dict(), "predicted_value": pv}


def test_a_loaded_link_is_bounded_so_its_surprise_stays_in_range(caplog):
    from maxim.decisions.causal_link import CausalLink, Valence

    with caplog.at_level(logging.WARNING):
        loaded = CausalLink.from_dict(_link_dict(-0.8))  # what hivemind used to let in
    assert loaded.predicted_value == 0.0
    loaded.update_prediction_rw(Valence.POSITIVE)
    assert loaded.last_rpe <= 1.0  # 1.8 before the bound
    assert any("bounded" in r.getMessage() for r in caplog.records)


def test_a_null_predicted_value_takes_the_neutral_prior_out_loud(caplog):
    from maxim.decisions.causal_link import CausalLink

    with caplog.at_level(logging.WARNING):
        assert CausalLink.from_dict(_link_dict(None)).predicted_value == 0.5
    assert any("null" in r.getMessage() for r in caplog.records)


def test_a_hive_merge_never_emits_a_value_outside_the_range():
    from maxim.hivemind.merge import _merge_link_pair

    merged = _merge_link_pair(_link_dict(-0.8), _link_dict(-0.2), left_source="a", right_source="b")
    assert 0.0 <= merged["predicted_value"] <= 1.0
