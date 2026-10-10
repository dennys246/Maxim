"""Grounding GL2a: the tool-path body-consequence record (``docs/plans/autonomic_layer.md`` §3.1, §5.2).

One ``InteroceptiveOutcome`` per tool invocation that reached ``tool.run`` on an agent-bound body, built
by ``embodiment/sem.py::interoceptive_outcome`` and stamped by ``Executor._stamp_invocation`` onto
``ToolOutput.interoceptive_outcome``; the loop capture writes it into
``EncodingSignals.extra["interoception"]``. Record-only: nothing reads it to act.

The gates drive the REAL loop and capture (``tests/unit/_cradle_loop_driver.py``) and read the record
back from the Hippocampus, never by calling ``executor.execute`` (wiring S4). Expected values are
computed BY HAND from the YAML deltas and drive specs, never by calling the ``sem.py`` helpers under
test (confounding NIT-1):

- ``infant_humanoid``: ``core_temperature`` homeostatic, set point 0, comfort band 0.25, pain scale
  1.5, range [-1, 1] (span 1, headroom 0.75); ``arms.thermal`` homeostatic, set point 0, band 0.5, pain
  scale 0.4, range [-1, 1] (span 1, headroom 0.5); ``hunger`` entropic up, deprivation 0.7,
  satisfaction 0.3 (span 0.4). The latch clears inside ``band * (1 - 0.2)``.
- ``cradle_cool_air.feel``: core -0.2, arms.thermal -0.15. ``cradle_fire_pit.warm_self``: core +0.2,
  arms.thermal +0.2; ``touch``: arms.thermal +0.6, core +0.15. ``cradle_food.eat``: hunger -0.4.
- ``infant_humanoid_chilled``: ``cold`` entropic up, 0.08/s, deprivation 0.5, satisfaction 0.3 (span
  0.2), range [0, 1]. ``warmth_beta_safe.warm_self``: cold -0.3, arms.thermal +0.05.

The driver ticks the step clock by 100 us, so drift between scripted actions is ~1e-7: ``TOL``.
"""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path
from typing import Any

import pytest

from tests.unit._cradle_loop_driver import AGENT_ID, Step, run_cradle

REPO_ROOT = Path(__file__).resolve().parents[2]
TOL = 1e-5
CRADLE = ("items/cradle_cool_air", "items/cradle_fire_pit")
CRADLE_SCRIPT = ("cool_air_feel", "cool_air_feel", "fire_pit_warm_self", "fire_pit_warm_self", "fire_pit_touch")
pytestmark = pytest.mark.timeout(120)


def _records(run: Any) -> list[dict[str, Any] | None]:
    """The record each LOOP capture persisted, in capture order (None where it carried none)."""
    return [t.encoding.extra.get("interoception") for t in run.traces if t.encoding.site == "loop"]


def _block(rec: dict[str, Any], name: str) -> dict[str, float]:
    return {k: v for k, v in rec[name]}


def _close(actual: dict[str, float], expected: dict[str, float]) -> bool:
    return set(actual) == set(expected) and all(abs(actual[k] - expected[k]) <= TOL for k in expected)


@pytest.fixture(scope="module")
def cradle_run(tmp_path_factory: pytest.TempPathFactory) -> Any:
    mp = pytest.MonkeyPatch()
    try:
        yield run_cradle(
            mp,
            tmp_path_factory.mktemp("cradle"),
            body_ref="bodies/infant_humanoid",
            entity_refs=CRADLE,
            script=tuple(Step(t) for t in CRADLE_SCRIPT),
        )
    finally:
        mp.undo()


# ── the scripted cradle sequence, by hand ────────────────────────────────────
#
#   value before -> after      core_temperature            arms.thermal
#   feel        -0.15 -> -0.35   0.00 -> -0.15
#   feel        -0.35 -> -0.55  -0.15 -> -0.30
#   warm_self   -0.55 -> -0.35  -0.30 -> -0.10
#   warm_self   -0.35 -> -0.15  -0.10 -> +0.10   core back inside 0.25 * 0.8: the latch set by the
#   touch       -0.15 ->  0.00  +0.10 -> +0.70   first feel clears here (satiation)
#
# drive_delta = (|before| - |after|) / span(=1); pressure = max(0, (|v| - band) / headroom);
# deviation_after = after / span; relief/harm = max drop/rise in pressure; urgency = max pressure after;
# drive_pain = the pain level after (core: (|v| - 0.25) * 1.5; thermal: (|v| - 0.5) * 0.4).

CRADLE_TABLE = [
    {
        "drive_delta": {"core_temperature": -0.20, "arms.thermal": -0.15},
        "pressure_before": {"core_temperature": 0.0, "arms.thermal": 0.0},
        "pressure_after": {"core_temperature": 0.10 / 0.75, "arms.thermal": 0.0},
        "deviation_after": {"core_temperature": -0.35, "arms.thermal": -0.15},
        "relief": 0.0,
        "harm": 0.10 / 0.75,
        "urgency": 0.10 / 0.75,
        "drive_pain": 0.10 * 1.5,
        "satiated": [],
    },
    {
        "drive_delta": {"core_temperature": -0.20, "arms.thermal": -0.15},
        "pressure_before": {"core_temperature": 0.10 / 0.75, "arms.thermal": 0.0},
        "pressure_after": {"core_temperature": 0.30 / 0.75, "arms.thermal": 0.0},
        "deviation_after": {"core_temperature": -0.55, "arms.thermal": -0.30},
        "relief": 0.0,
        "harm": 0.20 / 0.75,
        "urgency": 0.30 / 0.75,
        "drive_pain": 0.30 * 1.5,
        "satiated": [],
    },
    {
        "drive_delta": {"core_temperature": 0.20, "arms.thermal": 0.20},
        "pressure_before": {"core_temperature": 0.30 / 0.75, "arms.thermal": 0.0},
        "pressure_after": {"core_temperature": 0.10 / 0.75, "arms.thermal": 0.0},
        "deviation_after": {"core_temperature": -0.35, "arms.thermal": -0.10},
        "relief": 0.20 / 0.75,
        "harm": 0.0,
        "urgency": 0.10 / 0.75,
        "drive_pain": 0.10 * 1.5,
        "satiated": [],
    },
    {
        "drive_delta": {"core_temperature": 0.20, "arms.thermal": 0.0},
        "pressure_before": {"core_temperature": 0.10 / 0.75, "arms.thermal": 0.0},
        "pressure_after": {"core_temperature": 0.0, "arms.thermal": 0.0},
        "deviation_after": {"core_temperature": -0.15, "arms.thermal": 0.10},
        "relief": 0.10 / 0.75,
        "harm": 0.0,
        "urgency": 0.0,
        "drive_pain": 0.0,
        "satiated": ["core_temperature"],
    },
    {
        "drive_delta": {"core_temperature": 0.15, "arms.thermal": -0.60},
        "pressure_before": {"core_temperature": 0.0, "arms.thermal": 0.0},
        "pressure_after": {"core_temperature": 0.0, "arms.thermal": 0.20 / 0.5},
        "deviation_after": {"core_temperature": 0.0, "arms.thermal": 0.70},
        "relief": 0.0,
        "harm": 0.20 / 0.5,
        "urgency": 0.20 / 0.5,
        "drive_pain": 0.20 * 0.4,
        "satiated": [],
    },
]


def test_every_loop_capture_persists_its_record(cradle_run: Any) -> None:
    records = _records(cradle_run)
    assert len(records) == len(CRADLE_SCRIPT)
    assert all(r is not None for r in records)
    causes = {
        "cool_air_feel": ("cool_air", "feel"),
        "fire_pit_warm_self": ("fire_pit", "warm_self"),
        "fire_pit_touch": ("fire_pit", "touch"),
    }
    for rec, tool in zip(records, CRADLE_SCRIPT):
        assert rec["provenance"] == "experienced"
        assert rec["agent_id"] == AGENT_ID
        assert rec["body_path"] == rec["sufferer"] == cradle_run.embodiment.root.full_path
        entity, affordance = causes[tool]
        assert rec["cause"]["tool"] == tool
        assert rec["cause"]["entity"] == entity
        assert rec["cause"]["affordance"] == affordance
        assert "pid" not in rec  # G17: the identity lands at the post-fence resume stage


@pytest.mark.parametrize("index", range(len(CRADLE_TABLE)))
def test_the_cradle_records_match_the_hand_table(cradle_run: Any, index: int) -> None:
    rec = _records(cradle_run)[index]
    want = CRADLE_TABLE[index]
    for name in ("drive_delta", "pressure_before", "pressure_after", "deviation_after"):
        assert _close(_block(rec, name), want[name]), (name, rec[name], want[name])
    for name in ("relief", "harm", "urgency", "drive_pain"):
        assert abs(rec[name] - want[name]) <= TOL, (name, rec[name], want[name])
    assert rec["nociception"] == 0.0  # a thermal breach is a DRIVE pain, not nociception, until GL2b(ii)
    assert list(rec["satiated"]) == want["satiated"]
    assert _block(rec, "caused") == {d: True for d in want["drive_delta"]}  # every tool-path entry is caused


def test_gate_d_homeostatic_crossing_is_one_satiation_on_the_second_warm(cradle_run: Any) -> None:
    """Red gate (d), the homeostatic half on the tool path: exactly one satiation event."""
    satiations = [(i, list(r["satiated"])) for i, r in enumerate(_records(cradle_run)) if r["satiated"]]
    assert satiations == [(3, ["core_temperature"])]


def test_relief_pin_positive_drive_delta_equals_drive_relief(cradle_run: Any) -> None:
    """The record and the trio are computed by different code from different reads (§3.1): on every
    drive present in ``drive_relief``, the positive part of the record's ``drive_delta`` equals it."""
    for out in cradle_run.outputs:
        rec = out.interoceptive_outcome
        relief = dict(out.drive_relief or ())
        delta = dict(rec.drive_delta)
        for drive, value in relief.items():
            assert abs(max(0.0, delta[drive]) - value) <= TOL, (drive, delta[drive], value)


@pytest.mark.xfail(
    strict=True,
    reason="#1161: the credit reads in tool_bridge are blind to modulator drives, so drive_relief "
    "omits arms.thermal while the record carries it; flips when #1161 resolves the read",
)
def test_drive_relief_carries_arms_thermal_like_the_record(cradle_run: Any) -> None:
    warm = cradle_run.outputs[2]
    assert "arms.thermal" in dict(warm.interoceptive_outcome.drive_delta)
    assert "arms.thermal" in dict(warm.drive_relief or ())


# ── the 20 s step-clock case: the record nets the APPLIED (clamped) drift ─────────
#
# infant_humanoid_chilled, warmth_beta_safe. ``observe`` evaluates the body first (cold 0.6, the
# first evaluation applies no drift). Then 20 s pass between the loop tick and ``warm_self``:
#   cold 0.6 -> 0.3 after the delta -> 0.3 + 20 * 0.08 = 1.9, clamped to 1.0 by the drift.
#   applied drift +0.7 (not the declared +1.6); observed change +0.4; net change -0.3.
#   arms.thermal 0.0 -> 0.05 -> drifts toward 0 by 20 * 0.0008 = 0.016: applied -0.016, net +0.05.
# cold: pressure before (0.6 - 0.3) / 0.2 = 1.5 -> 1.0; after (net 0.3) -> 0.0; drive_delta
# (0.6 - 0.3) / 0.2 = 1.5 -> 1.0; relief 1.0.


@pytest.fixture(scope="module")
def chilled_run(tmp_path_factory: pytest.TempPathFactory) -> Any:
    mp = pytest.MonkeyPatch()
    try:
        yield run_cradle(
            mp,
            tmp_path_factory.mktemp("chilled"),
            body_ref="bodies/infant_humanoid_chilled",
            entity_refs=("items/warmth_beta_safe",),
            script=(Step("warmth_beta_safe_observe"), Step("warmth_beta_safe_warm_self", advance_s=20.0)),
        )
    finally:
        mp.undo()


def test_twenty_seconds_of_drift_are_netted_at_their_applied_clamped_value(chilled_run: Any) -> None:
    rec = _records(chilled_run)[1]
    assert abs(rec["extra"]["drift_dt_s"] - 20.0) <= 1e-3
    assert abs(rec["extra"]["drift"]["cold"] - 0.7) <= TOL
    assert abs(rec["extra"]["observed_change"]["cold"] - 0.4) <= TOL
    assert abs(rec["extra"]["drift"]["arms.thermal"] - (-0.016)) <= TOL
    assert abs(rec["extra"]["observed_change"]["arms.thermal"] - (0.05 - 0.016)) <= TOL
    assert abs(_block(rec, "drive_delta")["cold"] - 1.0) <= TOL
    assert abs(_block(rec, "pressure_before")["cold"] - 1.0) <= TOL
    assert abs(_block(rec, "pressure_after")["cold"] - 0.0) <= TOL
    assert abs(_block(rec, "drive_delta")["arms.thermal"] - (-0.05)) <= TOL
    assert abs(rec["relief"] - 1.0) <= TOL


def test_the_relief_pin_holds_across_twenty_seconds_of_drift(chilled_run: Any) -> None:
    """The netting's deletion probe re-reds THIS: un-netted, cold's change is +0.4 (away from comfort)."""
    out = chilled_run.outputs[1]
    relief = dict(out.drive_relief or ())
    assert abs(relief["cold"] - 1.0) <= TOL
    assert abs(max(0.0, dict(out.interoceptive_outcome.drive_delta)["cold"]) - relief["cold"]) <= TOL


# ── red gate (d), the entropic half: the cradle feed ──────────────────────────
#
# hunger set to 0.75. ``shelter`` (core +0.05) evaluates the body: hunger >= 0.7 is deprived -> latch.
# eat: 0.75 -> 0.35 (> 0.3, not cleared; not deprived). eat: 0.35 -> -0.05, clamped to 0.0 -> cleared.


def test_gate_d_entropic_feed_cycle_is_one_satiation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = run_cradle(
        monkeypatch,
        tmp_path,
        body_ref="bodies/infant_humanoid",
        entity_refs=("items/cradle_cool_air", "items/cradle_food"),
        script=(Step("cool_air_shelter"), Step("food_source_eat"), Step("food_source_eat")),
        initial={"hunger": 0.75},
    )
    records = _records(run)
    assert [list(r["satiated"]) for r in records] == [[], [], ["hunger"]]
    assert _block(records[0], "drive_delta") == {"core_temperature": pytest.approx(0.05, abs=TOL)}


@pytest.mark.xfail(
    strict=True, reason="GL2a is tool-path only (G9): drift-only crossings wait for the out-of-band producer"
)
def test_gate_d_a_crossing_by_drift_alone_is_one_satiation() -> None:
    from maxim.embodiment.body import Embodiment

    assert hasattr(Embodiment, "drain_outcomes")


# ── when a record exists (wiring S3) ────────────────────────────────────────────


def _executor(tmp_path: Path, agent_id: str, *extra_tools: Any) -> Any:
    """An executor on the canonical builders (``build_bio_stack`` + ``build_executor``), fire pit in scene."""
    from maxim.embodiment.component_registry import ComponentRegistry
    from maxim.runtime.bio_stack import build_bio_stack
    from maxim.runtime.bootstrap import build_executor
    from maxim.tools.registry import ToolRegistry

    components = ComponentRegistry()
    registry = ToolRegistry()
    for tool in extra_tools:
        registry.register(tool)
    bio = build_bio_stack(agent_id=agent_id or "gl2a_unbound", persistence_dir=str(tmp_path / (agent_id or "unbound")))
    executor = build_executor(
        tool_registry=registry,
        permissions=None,
        agent_id=agent_id,
        pain_bus=bio.pain_bus,
        nac=bio.nac,
        hippocampus=bio.hippocampus,
        entity_ref="bodies/infant_humanoid",
        component_registry=components,
    )
    executor.generate_entity_tools(components.instantiate("items/cradle_fire_pit"))
    return executor


def _probe_tools() -> tuple[Any, Any]:
    from maxim.tools.base import Tool, ToolOutput

    class _Plain(Tool):
        name = "gl2a_plain"
        description = "stub"
        input_schema: dict = {}

        def execute(self, **kwargs: Any) -> Any:
            return ToolOutput(success=True, output="ok")

    class _Raises(Tool):
        """Raises out of ``run`` itself: ``Tool.run`` turns an ``execute`` raise into a failed output,
        which is a tool that RAN (and records). This drives the executor's raised path."""

        name = "gl2a_raises"
        description = "stub"
        input_schema: dict = {}

        def execute(self, **kwargs: Any) -> Any:
            raise RuntimeError("probe")

        def run(self, **kwargs: Any) -> Any:
            raise RuntimeError("probe")

    return _Plain(), _Raises()


def test_record_iff_an_agent_bound_body_and_the_tool_ran(tmp_path: Path) -> None:
    plain, raises = _probe_tools()
    bound = _executor(tmp_path, "gl2a_bound", plain, raises)
    warm = bound.execute({"tool_name": "fire_pit_warm_self", "params": {}})
    assert warm.interoceptive_outcome is not None
    other = bound.execute({"tool_name": "gl2a_plain", "params": {}})
    assert other.interoceptive_outcome is not None  # the tool ran: a record with an empty drive block
    assert other.interoceptive_outcome.drive_delta == ()
    assert bound.execute({"tool_name": "gl2a_raises", "params": {}}).interoceptive_outcome is None
    assert bound.execute({"tool_name": "gl2a_not_registered", "params": {}}).interoceptive_outcome is None
    unbound = _executor(tmp_path, "")  # create.embodiment(), foundry and probe bodies: agent_id == ""
    assert unbound.execute({"tool_name": "fire_pit_warm_self", "params": {}}).interoceptive_outcome is None


def test_str_of_a_tool_output_is_byte_identical_with_and_without_the_record(tmp_path: Path) -> None:
    """#1189: ``ToolOutput`` reaches the Hippocampus's searchable text through ``str``."""
    bound = _executor(tmp_path, "gl2a_bound")
    out = bound.execute({"tool_name": "fire_pit_touch", "params": {}})
    assert out.interoceptive_outcome is not None
    bare = dataclasses.replace(out, interoceptive_outcome=None)
    assert str(out) == str(bare) and repr(out) == repr(bare)
    # Tokens only the record would print (``drive_relief=`` already prints "relief": #1189's root).
    for token in ("interoceptive", "nociception", "urgency", "satiated", "caused_or_felt"):
        assert token not in str(out)


def test_the_unreadable_sensor_pop_records_no_satiation(tmp_path: Path) -> None:
    """Satiation is the two ``elif cleared:`` sites only, never the silent pop of an unreadable sensor."""
    bound = _executor(tmp_path, "gl2a_bound")
    root = bound.embodiment.root
    root.vital_metrics["core_temperature"] = -0.6
    bound.embodiment.evaluate_failures()  # latches core_temperature
    assert "core_temperature" in root.drive_breach_severity
    with bound.embodiment.outcome_window() as window:
        # Unreadable: no sensor to read and no vital metric to fall back on (the ``current is None`` pop).
        root.sensors.pop("core_temperature")
        del root.vital_metrics["core_temperature"]
        bound.embodiment.evaluate_failures()
    assert "core_temperature" not in root.drive_breach_severity
    assert window.cleared == {}


# ── the factory's structure (§3.1.4) ──────────────────────────────────────────


def test_the_factory_requires_cause_and_provenance() -> None:
    import inspect

    from maxim.embodiment import sem

    params = inspect.signature(sem.interoceptive_outcome).parameters
    for name in ("cause", "provenance"):
        assert params[name].kind is inspect.Parameter.KEYWORD_ONLY
        assert params[name].default is inspect.Parameter.empty


def test_the_record_refuses_missing_or_unknown_provenance_and_has_no_pid() -> None:
    from maxim.embodiment.sem import InteroceptiveOutcome

    with pytest.raises(ValueError):
        InteroceptiveOutcome()
    with pytest.raises(ValueError):
        InteroceptiveOutcome(provenance="observed")
    assert "pid" not in {f.name for f in dataclasses.fields(InteroceptiveOutcome)}
    with pytest.raises(ValueError):
        InteroceptiveOutcome(provenance="experienced", extra={"relief": 1.0})  # collides with a field


def test_the_record_round_trips_through_its_persisted_form(cradle_run: Any) -> None:
    from maxim.embodiment.sem import InteroceptiveOutcome

    rec = cradle_run.outputs[4].interoceptive_outcome
    assert InteroceptiveOutcome.from_dict(rec.to_dict()) == rec
    assert _records(cradle_run)[4] == rec.to_dict()


def test_no_src_code_constructs_the_record_except_its_factory() -> None:
    """Structural guard: ``InteroceptiveOutcome(`` only inside ``sem.py``'s factory and ``from_dict``."""
    offenders = []
    for path in sorted((REPO_ROOT / "src" / "maxim").rglob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                allowed = path.name == "sem.py" and node.name in ("interoceptive_outcome", "from_dict")
                for call in ast.walk(node):
                    if (
                        isinstance(call, ast.Call)
                        and getattr(call.func, "id", getattr(call.func, "attr", None)) == "InteroceptiveOutcome"
                        and not allowed
                    ):
                        offenders.append(f"{path.relative_to(REPO_ROOT)}::{node.name}")
    assert offenders == []
    from maxim.embodiment import sem  # the factory exists, so the guard is not vacuous

    assert "InteroceptiveOutcome(" in Path(sem.__file__).read_text()


# ── T1-16 structural discharge: the persisted key moves no retrieval (autonomic_layer.md §5.2) ──
#
# GL2a grows every loop capture's saved trace by ``extra["interoception"]``, which fires T1-16's "the
# memory record shape" trigger. The discharge's three legs: the ranking readers read no ``extra`` key;
# ranking is byte-identical over a store with and without the record; and the substring path
# (``Hippocampus.search_by_content``, Path 3 of ``_query_hippocampus``) is too, on queries that would
# match a record token -- which holds only because ``ToolOutput.interoceptive_outcome`` is
# ``repr=False`` (wiring D3, #1189).

_RECORD_QUERIES = ("relief", "oxygen", "thermal", "nociception", "urgency", "interoceptive", "caused_or_felt")


def test_t1_16_the_ranking_readers_read_no_extra_key() -> None:
    import inspect

    from maxim.integration.bio_enrichment import BioEnrichmentPipeline
    from maxim.memory.hippocampus_retrieval import _rank_by_relevance

    readers = [_rank_by_relevance]
    for name in ("_query_hippocampus", "_query_hippocampus_traced"):
        readers.append(getattr(BioEnrichmentPipeline, name))
    for reader in readers:
        source = inspect.getsource(reader)
        for token in ("extra", "interoception", ".encoding"):
            assert token not in source, (reader.__qualname__, token)


def _strip_record(trace: Any) -> None:
    """The same trace as if GL2a had never run: no persisted key, no record on the ToolOutput."""
    trace.encoding = dataclasses.replace(trace.encoding, extra={})
    result = getattr(trace.outcome, "result", None)
    if getattr(result, "interoceptive_outcome", None) is not None:
        trace.outcome.result = dataclasses.replace(result, interoceptive_outcome=None)


def test_t1_16_ranking_and_substring_search_are_identical_without_the_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from maxim.memory.hippocampus_retrieval import _rank_by_relevance

    run = run_cradle(
        monkeypatch,
        tmp_path,
        body_ref="bodies/infant_humanoid",
        entity_refs=CRADLE,
        script=tuple(Step(t) for t in CRADLE_SCRIPT),
    )
    hippo = run.hippocampus
    assert sum(1 for t in run.traces if "interoception" in t.encoding.extra) == len(CRADLE_SCRIPT)
    queries = _RECORD_QUERIES + ("fire_pit_touch", "cool_air_feel", "cradle script")

    def views() -> tuple[list[list[str]], list[list[str]]]:
        memories = list(hippo._memories.values())
        ranked = [[id(m) for m in _rank_by_relevance(list(memories), q, 50)] for q in queries]
        found = [sorted(id(m) for m in hippo.search_by_content(q, limit=50)) for q in queries]
        return ranked, found

    with_record = views()
    for trace in run.traces:
        _strip_record(trace)
    assert all("interoception" not in t.encoding.extra for t in run.traces)
    assert views() == with_record


def test_a_record_on_an_otherwise_unchanged_output_is_not_dropped(tmp_path: Path) -> None:
    """``_stamp_invocation`` returns its input unchanged when every stamp already matches; the record is
    in that comparison, or a record on an output with no other stamp (no bridge, no readable drive)
    would be silently dropped (§5.2)."""
    from maxim.embodiment.sem import CauseRef
    from maxim.runtime.executor import _OutcomeStart
    from maxim.tools.base import ToolOutput

    bound = _executor(tmp_path, "gl2a_bound")
    bound._tool_pain_bridge = None  # no surprise and no pain to stamp
    bare = ToolOutput(success=True)
    start = _OutcomeStart(cause=CauseRef(tool="gl2a_plain"), specs={}, before={})
    stamped = bound._stamp_invocation(bare, "inv-1", None, outcome=start)
    assert (stamped.rpe, stamped.drive_pressure_before, stamped.drive_relief, stamped.pain) == (None, None, None, None)
    assert stamped.interoceptive_outcome is not None
    assert stamped.interoceptive_outcome.cause == CauseRef(tool="gl2a_plain")


def test_a_window_sees_only_its_own_threads_evaluations(tmp_path: Path) -> None:
    """The orchestrator's narrator tools evaluate the AUT's body on ANOTHER thread (§3.1.2); their drift
    and clears never land in the AUT executor's window."""
    import threading
    import time as _time

    bound = _executor(tmp_path, "gl2a_bound")
    emb = bound.embodiment
    root = emb.root
    root.vital_metrics["core_temperature"] = -0.6
    emb.evaluate_failures()  # baseline poll + latch
    root.vital_metrics["core_temperature"] = 0.0  # back in band: the next evaluation clears the latch
    with emb.outcome_window() as window:
        _time.sleep(0.01)  # some drift to apply
        other = threading.Thread(target=emb.evaluate_failures)
        other.start()
        other.join()
    assert "core_temperature" not in root.drive_breach_severity  # the other thread did clear it
    assert window.drift == {} and window.drift_dt_s == 0.0 and window.cleared == {}


def test_nested_windows_close_by_identity(tmp_path: Path) -> None:
    """Two empty windows compare equal by value; closing the inner one must not close the outer."""
    import time as _time

    emb = _executor(tmp_path, "gl2a_bound").embodiment
    emb.evaluate_failures()  # baseline poll
    with emb.outcome_window() as outer:
        with emb.outcome_window() as inner:
            pass
        _time.sleep(0.01)
        emb.evaluate_failures()
    assert inner.drift == {} and outer.drift  # only the still-open window recorded the later drift
    assert emb._open_windows() == []


def test_a_self_affordance_names_no_causing_entity(tmp_path: Path) -> None:
    """``CauseRef.entity`` is never the sufferer: the body's own affordance names the act, not an entity."""
    bound = _executor(tmp_path, "gl2a_bound")
    out = bound.execute({"tool_name": "infant_humanoid_turn_left", "params": {}})
    cause = out.interoceptive_outcome.cause
    assert (cause.entity, cause.affordance, cause.tool) == ("", "turn_left", "infant_humanoid_turn_left")


def test_only_an_affordance_names_a_causing_entity(tmp_path: Path) -> None:
    """Sensing a scene item does nothing to it: a sensor read or sense names no causing entity."""
    bound = _executor(tmp_path, "gl2a_bound")
    for tool in ("sense_fire_pit", "read_fire_pit_heat_output"):
        cause = bound.execute({"tool_name": tool, "params": {}}).interoceptive_outcome.cause
        assert (cause.entity, cause.affordance, cause.tool) == ("", "", tool)
    touch = bound.execute({"tool_name": "fire_pit_touch", "params": {}}).interoceptive_outcome.cause
    assert (touch.entity, touch.affordance) == ("fire_pit", "touch")
