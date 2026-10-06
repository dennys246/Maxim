"""``runtime/loop_setup.py`` pieces the public-entry characterization cannot reach (1.3.2 slice 1).

``tests/unit/test_loop_setup_characterization.py`` pins the setup through ``run_agentic_loop``. The
thought-novelty gate is consumed only on the LLM-primary deliberation path (outside phase 1), so its
semantics, moved verbatim from the loop's old closure, are pinned here directly.
"""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from maxim.runtime.loop_setup import LoopRun, _novelty_gate


def test_a_novel_thought_is_shown_and_a_repeat_is_not():
    is_novel = _novelty_gate()
    assert is_novel("the river is cold today") is True
    assert is_novel("The river is COLD today") is False  # same words, case-folded
    assert is_novel("climb the hill to find food") is True


def test_an_empty_thought_is_never_shown():
    assert _novelty_gate()("   ") is False


def test_overlap_at_the_threshold_is_suppressed():
    # Jaccard 3/5 = 0.6 >= 1 - 0.40: suppressed; 2/6 = 0.33 < 0.6: shown.
    is_novel = _novelty_gate()
    assert is_novel("a b c d") is True
    assert is_novel("a b c e") is False
    assert is_novel("a b x y") is True


def test_a_suppressed_thought_is_not_remembered():
    is_novel = _novelty_gate()
    assert is_novel("a b c d") is True
    assert is_novel("a b c e") is False  # not added: "e" alone stays new against the tracker
    assert is_novel("e f g h") is True


def test_the_tracker_keeps_the_last_eight():
    is_novel = _novelty_gate()
    assert is_novel("first thought here") is True
    for i in range(8):
        assert is_novel(f"filler{i} words{i} only{i}") is True
    assert is_novel("first thought here") is True  # evicted from the 8-deep tracker


def test_each_gate_has_its_own_tracker():
    a, b = _novelty_gate(), _novelty_gate()
    assert a("same words") is True
    assert b("same words") is True


def test_loop_run_is_frozen():
    fields = {f.name: None for f in dataclasses.fields(LoopRun)}
    run = LoopRun(**fields)  # type: ignore[arg-type]
    with pytest.raises(dataclasses.FrozenInstanceError):
        run.ctrl = object()  # type: ignore[misc]


# ── build_loop_run's own outputs, and the loop's unpack of them (slice 1 review, executor lens) ──


def _build(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, **overrides: Any) -> LoopRun:
    from maxim.runtime.bootstrap import build_executor
    from maxim.runtime.loop_setup import build_loop_run
    from maxim.runtime.state import RuntimeState
    from maxim.tools.registry import ToolRegistry

    monkeypatch.chdir(tmp_path)  # the first persist is CWD-relative
    kwargs: dict[str, Any] = dict(
        agent=SimpleNamespace(name="unit"),
        environment=None,
        state=RuntimeState(),
        memory=None,
        decision_engine=None,
        executor=build_executor(ToolRegistry(), pain_bus=None, permissions=None),
        autonomy_controller=None,
        llm_worker=None,
        default_network=None,
        hippocampus=None,
        memory_hub=None,
        evaluators=None,
        max_steps=1,
        run_id="unit",
        stop_event=None,
        on_step=None,
        on_event=None,
        idle_sleep_s=0.0,
        persist_every_n_steps=10,
        target_hz=30.0,
        context_pool_config=None,
        use_tool_prompting=True,
        protocol_registry=None,
        percept_source=None,
        action_sink=None,
        pain_bus=None,
        aut_mode="substrate-primary",
        planning_liveness=False,
        sim_adapter=None,
    )
    kwargs.update(overrides)
    return build_loop_run(**kwargs)


def test_each_run_gets_its_own_novelty_gate(monkeypatch, tmp_path):
    a, b = _build(monkeypatch, tmp_path), _build(monkeypatch, tmp_path)
    assert a.is_novel_thought is not b.is_novel_thought
    assert a.is_novel_thought("same words") is True
    assert b.is_novel_thought("same words") is True  # not a tracker shared across runs


def test_the_overrides_come_from_their_own_positions(monkeypatch, tmp_path):
    from maxim.runtime import agent_loop as AL

    monkeypatch.setattr(AL, "resolve_llm_loop_overrides", lambda: (111, 2))
    run = _build(monkeypatch, tmp_path)
    assert run.max_response_tokens_override == 111
    assert run.max_cycles_override == 2


def test_the_result_cache_is_the_global_one(monkeypatch, tmp_path):
    from maxim.runtime.prefetch import get_result_cache

    assert _build(monkeypatch, tmp_path).result_cache is get_result_cache()


# The loop's unpack: every local the body reads maps to its own LoopRun field. A source pin (AST of
# run_agentic_loop): the override and result-cache consumers sit only on the LLM-primary submit and
# deliberation paths, which no cheap drive reaches (phase 1 stays off that path). FUNCTION-SPECIFIC:
# the slice that moves the unpack updates it.
_UNPACK = {
    "executor": "executor",
    "sim": "sim",
    "ctrl": "ctrl",
    "autonomy_controller": "autonomy_controller",
    "run_id": "run_id",
    "agent_name": "agent_name",
    "state_path": "state_path",
    "context_pool": "context_pool",
    "prefetcher": "prefetcher",
    "result_cache": "result_cache",
    "_is_novel_thought": "is_novel_thought",
    "_max_response_tokens_override": "max_response_tokens_override",
    "_max_cycles_override": "max_cycles_override",
    "dn_enabled": "dn_enabled",
    "memory_hub_enabled": "memory_hub_enabled",
    "_loop_nac": "nac",
    "_loop_xclock": "xclock",
    "_loop_agent_id": "agent_id",
    "_drive_relief_only": "drive_relief_only",
    "_rec_outcome": "rec_outcome",
    "_loop_sensor_encoder": "sensor_encoder",
    "_loop_situation_cue": "situation_cue",
    "_planning_liveness_on": "planning_liveness_on",
}


def _unpack_map() -> dict[str, list[str]]:
    import maxim.runtime.agent_loop as agent_loop

    tree = ast.parse(Path(agent_loop.__file__).read_text())
    [fn] = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_agentic_loop"]
    seen: dict[str, list[str]] = {}
    for stmt in fn.body:
        if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
            continue
        target, value = stmt.targets[0], stmt.value
        names = target.elts if isinstance(target, ast.Tuple) else [target]
        values = value.elts if isinstance(value, ast.Tuple) else [value]
        if len(names) != len(values):
            continue
        for n, v in zip(names, values):
            if isinstance(n, ast.Name) and isinstance(v, ast.Attribute) and getattr(v.value, "id", None) == "run":
                seen.setdefault(n.id, []).append(v.attr)
    return seen


def test_the_loop_unpacks_every_field_into_its_own_local():
    assert _unpack_map() == {local: [field] for local, field in _UNPACK.items()}
    assert set(_UNPACK.values()) == {f.name for f in dataclasses.fields(LoopRun)}


@pytest.mark.parametrize("bad", [{"target_hz": 0.0}, {"max_steps": "x"}])
def test_bad_timing_args_are_refused_before_any_thread_starts(monkeypatch, tmp_path, bad):
    """The loop's three timing lines (``target_period``, ``max_steps_i``, ``step_iter``) sat inside the
    setup block and now run after ``build_loop_run``, i.e. after the Default Network and bio-session
    starts (wire-integrity review, slice 1). The behaviour is unchanged only because
    ``LoopController.__init__``, built early in setup, refuses these values first: this pins that, so a
    bad timing argument never leaves a started Default Network or capture worker without its stop."""
    from tests.unit.test_loop_setup_characterization import _DN, _Hub, _run

    events: list[str] = []
    with pytest.raises((ValueError, TypeError, ZeroDivisionError)):
        _run(monkeypatch, tmp_path, events=events, default_network=_DN(events), memory_hub=_Hub(events), **bad)
    assert "dn.start" not in events, events
    assert not any(e.startswith("hub.on_session_start") for e in events), events
