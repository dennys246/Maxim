"""Characterization of ``run_agentic_loop``'s §6 submit, first half (1.3.6 loop phase 2, slice 7a).

Pins, through the PUBLIC entry ``agent_loop.run_agentic_loop`` on the LLM-primary path, what §6 does between the
submit cadence check and the request assembly: it builds the tick's context (``memory.build_context()``, or a minimal
one for a pending follow-up), adds the Wire-A cluster-bias annotations and their sources and the W1 grayscale-tool
annotations (one kill switch, ``MAXIM_DISABLE_CLUSTER_BIAS_ANNOTATION``), and decides whether anything is worth a
submission: a fresh CLI percept, a new CLI input in the context (voice needs a wake word, Selfy's commands are skipped),
spoken "maxim", a keyword or hard-override percept, or a pending action follow-up (folded into the inputs and cleared);
nothing meaningful, or sleep without a wake keyword, drops the context. Written BEFORE slice 7a moves the block out of
``_run_agentic_loop_body`` and kept green unchanged by that move (``docs/plans/roadmap_1_3_x.md`` §1.3.6;
coverage-first rule). The request assembly and deliberation (slice 7b) are pinned only as far as their output, the
``submit_context`` call.

Observation is location-independent: the worker (a fake ``llm_worker``) records every ``submit_context`` call; the
memory is a real ``InMemoryMemory`` whose ``build_context`` an arm may amend (it is the loop's injected memory); the hub
is the minimal one the #1200 tests use, carrying a real ``NAc`` subclass that serves scripted tool biases; the
controller is captured by a class-level ``LoopController.__init__`` spy; logs through caplog. Nothing patches
``agent_loop`` or the module 7a moves to; the grayscale builder is spied at its own module, where §6 imports it per call.

Covered elsewhere, through the same public entry: the §1.16 B1 minimal context for an escalating audio-only pass and the
auto-sense text (``test_loop_perception_characterization.py``). Not reachable: the sim-mode "LLM skip: ... unprocessed
cli_inputs" trace (every input the gate sees is marked processed before it), pinned as not firing. Equivalent mutant:
grayscale's ``= None`` in its error handler (the field is set only after a build succeeds).

Pinned AS-IS, each a defect with its own issue (never fixed inside an extraction; #1721, #1722): a Selfy command typed as a
fresh percept is logged skipped yet submitted, and recorded processed twice; a follow-up arriving with a fresh input is
folded in and cleared, but the submission keeps only the input, so the tool result never reaches the model; with no
built context a fresh input is marked processed and never sent.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest

from tests.policy_helpers import TEST_POLICY

LOGGER = "maxim.runtime.agent_loop"
PROBE = "probe_submit"


class _Worker:
    """Records each ``submit_context``. Never proposes (§2 sees nothing, so every pass may submit again), unless
    ``answer`` is set: then the first poll after a submit returns one ready-to-act proposal (the PFC cycle-1 wait)."""

    def __init__(self, answer: bool = False) -> None:
        self.submits: list[dict[str, Any]] = []
        self.answer, self._owed = answer, False

    def latest_attempt_state(self) -> Any:
        from maxim.agents.llm_types import LLMAttemptState

        return LLMAttemptState.NONE

    def get_latest_proposal(self) -> Any:
        if not self._owed:
            return None
        from maxim.agents.llm_types import LLMProposal

        self._owed = False
        return LLMProposal(
            request_id="p-7a", action={}, reasoning="", strategy_used="llm", confidence=0.9, mode_goal_achieved=True
        )

    def submit_context(self, **kw: Any) -> bool:
        ctx = kw.get("context")
        self.submits.append(
            {
                "cli_inputs": list(getattr(ctx, "cli_inputs", None) or []),
                "mode": getattr(ctx, "mode", None),
                "autonomy_level": getattr(ctx, "autonomy_level", None),
                "cluster_bias_annotations": getattr(ctx, "cluster_bias_annotations", "<unset>"),
                "cluster_bias_sources": getattr(ctx, "cluster_bias_sources", "<unset>"),
                "grayscale_tool_annotations": getattr(ctx, "grayscale_tool_annotations", "<unset>"),
                "bio_enrichment_context": getattr(ctx, "bio_enrichment_context", "<unset>"),
                "triggering_input": kw.get("triggering_input"),
                "kw": kw,
            }
        )
        self._owed = self.answer
        return True

    def requeue_request(self, *_a: Any, **_k: Any) -> bool:
        return True

    def requeue_last_request(self, **_k: Any) -> bool:
        return True

    def record_outcome(self, *_a: Any, **_k: Any) -> None:
        return None


class _Agent:
    name = "submit-probe"

    def propose_intent(self, state: Any, memory: Any) -> None:
        return None


def _make_nac(biases: Any, sources: Any) -> Any:
    """A real NAc whose two Wire-A reads are scripted (an Exception instance is raised); ``.calls`` records them."""
    from maxim.decisions.nac import NAc

    calls: list[tuple] = []

    class _ScriptedNAc(NAc):
        def get_agent_tool_biases(self, *, agent_id: str, top_n: int = 5) -> Any:  # type: ignore[override]
            calls.append(("biases", agent_id, top_n))
            if isinstance(biases, Exception):
                raise biases
            return list(biases)

        def get_cluster_reward_sources(self, *, agent_id: str) -> Any:  # type: ignore[override]
            calls.append(("sources", agent_id))
            if isinstance(sources, Exception):
                raise sources
            return dict(sources)

    nac = _ScriptedNAc()
    nac.calls = calls  # type: ignore[attr-defined]
    return nac


class _Hub:
    """The few MemoryHub members the loop reads (the #1200 tests' stand-in), carrying the scripted NAc."""

    agent_id = "submit_hub"
    ec = None
    hippocampus = None

    def __init__(self, nac: Any) -> None:
        from maxim.runtime.substrate_proposal import NO_SITUATION_CUE

        self.nac = nac
        self.situation_cue = NO_SITUATION_CUE

    def on_session_start(self, *, owner: Any, live: bool = False, deadline_s: float | None = None) -> Any:
        self._ticket = SimpleNamespace(generation=1, owner=owner, start_stats={})
        return self._ticket

    def on_session_end(self, ticket: Any, *, deadline: Any = None, consolidation_bound: Any = None) -> dict:
        return {}

    def on_session_end_lightweight(self, ticket: Any, *, deadline: Any = None) -> dict:
        return {}

    def release_session_claim(self, ticket: Any) -> None:
        return None

    def record_plan_outcome(self, **kw: Any) -> None:
        return None


def _run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    steps: int = 3,
    inputs: dict[int, str] | None = None,
    amend: Callable[[int, Any], Any] | None = None,
    hooks: dict[int, Callable[[Any], None]] | None = None,
    state_data: dict[str, Any] | None = None,
    nac: Any = None,
    sim_mode: bool = False,
    pipeline: Any = None,
    planning_liveness: bool = False,
) -> SimpleNamespace:
    """Run the real loop for ``steps`` passes. ``inputs[k]`` is pass k's fresh CLI percept; ``amend(k, ctx)`` may edit
    (or replace, by returning) the context memory built on pass k; ``hooks[k](obs)`` runs at the start of pass k."""
    import maxim.simulation.sim_logger as sim_logger
    from maxim.agents.autonomy import AutonomyController, AutonomyLevel
    from maxim.environment.filesystem_env import FileSystemEnv
    from maxim.memory.base import InMemoryMemory
    from maxim.runtime import agent_loop as AL
    from maxim.runtime.bootstrap import build_decision_engine, build_executor
    from maxim.runtime.loop_controller import LoopController
    from maxim.runtime.sim_adapter import NullSimulationAdapter
    from maxim.runtime.state import RuntimeState
    from maxim.tools.base import Tool, ToolOutput
    from maxim.tools.registry import ToolRegistry

    tmp_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sim_logger, "_interactive_mode", sim_logger.InteractiveMode.OFF)
    sim_logs: list[tuple[str, str]] = []
    real_sim_log = sim_logger.sim_log

    def _sim_log(category: str, msg: str, data: Any = None) -> None:
        sim_logs.append((category, msg))
        real_sim_log(category, msg, data)

    monkeypatch.setattr(sim_logger, "sim_log", _sim_log)
    obs = SimpleNamespace(
        ctrl=None, state=None, worker=_Worker(answer=pipeline is not None), sim_logs=sim_logs, passes=-1
    )

    real_init = LoopController.__init__

    def _init(self: Any, *a: Any, **k: Any) -> None:
        real_init(self, *a, **k)
        self.llm_submit_interval = 0.0  # every pass may submit; the passes take microseconds
        obs.ctrl = self

    monkeypatch.setattr(LoopController, "__init__", _init)

    class _Memory(InMemoryMemory):
        def build_context(self) -> Any:  # the loop's injected memory: an arm may amend what it builds
            ctx = super().build_context()
            if amend is not None:
                out = amend(obs.passes, ctx)
                return ctx if out is _KEEP else out
            return ctx

    def _on_pass() -> dict:
        obs.passes += 1
        if hooks and obs.passes in hooks:
            hooks[obs.passes](obs)
        text = (inputs or {}).get(obs.passes)
        return {"cli_input": text} if text else {}

    class _Env(FileSystemEnv):
        def observe(self) -> dict[str, Any]:
            return _on_pass()

    class _Adapter(NullSimulationAdapter):
        def has_carried_percept(self) -> bool:
            return True  # wake every pass

    class _Probe(Tool):
        name = PROBE
        description = "stub"
        input_schema: dict = {}

        def execute(self, **kwargs: Any) -> Any:
            return ToolOutput(success=True, output="ok")

    registry = ToolRegistry()
    registry.register(_Probe())
    state = RuntimeState()
    state.data["mode"] = "active"
    state.data.update(state_data or {})
    obs.state = state
    (tmp_path / "ws").mkdir(exist_ok=True)
    kwargs: dict[str, Any] = {}
    if sim_mode:

        class _Source:
            def next_percept(self) -> Any:
                _on_pass_out = _on_pass()
                text = _on_pass_out.get("cli_input")
                if not text:
                    return None
                return SimpleNamespace(
                    source="cli",
                    content=text,
                    cli_input=text,
                    transcript_chunk=None,
                    hard_override=None,
                    raw_transcript_text=None,
                    metadata={},
                    sensory=None,
                    timestamp=time.time(),
                )

            def is_exhausted(self) -> bool:
                return False

        kwargs["percept_source"] = _Source()
        kwargs["response_policy"] = None  # no bridge, so no ResponsePolicy: no gating prompt is answered (#1185)
    else:
        kwargs["sim_adapter"] = _Adapter()
    if nac is not None:
        kwargs["memory_hub"] = _Hub(nac)
    if pipeline is not None:
        kwargs["bio_enrichment_pipeline"] = pipeline
    if planning_liveness:
        kwargs["planning_liveness"] = True
    AL.run_agentic_loop(
        _Agent(),
        _Env(str(tmp_path / "ws")),
        state,
        _Memory(),
        build_decision_engine(),
        build_executor(registry, pain_bus=None, policy=TEST_POLICY),
        llm_worker=obs.worker,
        autonomy_controller=AutonomyController(initial_level=AutonomyLevel.AUTONOMOUS),
        max_steps=steps,
        target_hz=1e6,
        idle_sleep_s=0.0,
        stop_event=threading.Event(),
        **kwargs,
    )
    return obs


_KEEP = object()  # ``amend`` returns this to keep the (edited) context


def _followup(**kw: Any) -> Any:
    from maxim.runtime.loop_types import ActionFollowup

    base = dict(tool=PROBE, result="the result", original_query="what was asked", followup_type="process")
    base.update(kw)
    return ActionFollowup(mode="active", timestamp=time.time(), **base)


def _set_followup(**kw: Any) -> Callable[[Any], None]:
    def hook(obs: Any) -> None:
        obs.ctrl.pending_action_followup = _followup(**kw)

    return hook


def _ctx_add(**fields: Any) -> Callable[[int, Any], Any]:
    """Amend pass 0's built context with ``fields`` (later passes build the plain context)."""

    def amend(k: int, ctx: Any) -> Any:
        if k == 0:
            for name, value in fields.items():
                setattr(ctx, name, value)
        return _KEEP

    return amend


def _logs(caplog: Any, *prefixes: str) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name == LOGGER and r.getMessage().startswith(prefixes)]


def _brief(obs: Any) -> list[tuple[list[str], str]]:
    return [(s["cli_inputs"], s["triggering_input"]) for s in obs.worker.submits]


@pytest.fixture(autouse=True)
def _loop_logs(caplog: Any) -> None:
    caplog.set_level(logging.DEBUG, logger="maxim")


# ── what is worth a submission ───────────────────────────────────────────────


def test_a_fresh_cli_input_is_submitted_once_then_deduped(monkeypatch, tmp_path):
    """The percept submits; on later passes memory still lists it, but it is processed, so nothing resubmits."""
    obs = _run(monkeypatch, tmp_path, inputs={0: "hello there"}, steps=3)
    assert _brief(obs) == [(["hello there"], "hello there")]
    assert list(obs.ctrl.processed_cli_inputs) == ["hello there"]


def test_the_same_words_as_a_new_percept_are_a_new_turn(monkeypatch, tmp_path):
    obs = _run(monkeypatch, tmp_path, inputs={0: "hello there", 2: "hello there"}, steps=4)
    assert _brief(obs) == [(["hello there"], "hello there")] * 2
    assert list(obs.ctrl.processed_cli_inputs) == ["hello there"] * 2


def test_a_fresh_skip_command_is_logged_skipped_yet_submitted(monkeypatch, tmp_path, caplog):
    """Pinned as-is (#1722): the fresh-percept branch marks the input meaningful before the context loop's skip check, so a
    Selfy command typed as a percept is logged "Skipping LLM", submitted anyway and recorded processed twice."""
    obs = _run(monkeypatch, tmp_path, inputs={0: "maxim pause"}, steps=2)
    assert _brief(obs) == [(["maxim pause"], "maxim pause")]
    assert list(obs.ctrl.processed_cli_inputs) == ["maxim pause", "maxim pause"]
    assert _logs(caplog, "Skipping LLM for command") == ["Skipping LLM for command: maxim pause"]


def test_a_new_cli_input_in_the_context_is_submitted(monkeypatch, tmp_path):
    obs = _run(monkeypatch, tmp_path, amend=_ctx_add(cli_inputs=["from context"]), steps=2)
    assert _brief(obs) == [(["from context"], "from context")]
    assert list(obs.ctrl.processed_cli_inputs) == ["from context"]


def test_voice_input_needs_a_wake_word_and_without_one_is_marked_processed(monkeypatch, tmp_path):
    voice = {"pending_user_input_source": "voice"}
    obs = _run(monkeypatch, tmp_path, amend=_ctx_add(cli_inputs=["just talking"]), state_data=voice, steps=2)
    assert obs.worker.submits == [] and list(obs.ctrl.processed_cli_inputs) == ["just talking"]
    obs = _run(monkeypatch, tmp_path / "w", amend=_ctx_add(cli_inputs=["hey maxim look"]), state_data=voice, steps=2)
    assert _brief(obs) == [(["hey maxim look"], "hey maxim look")]


def test_a_skip_command_in_the_context_is_not_submitted(monkeypatch, tmp_path, caplog):
    obs = _run(monkeypatch, tmp_path, amend=_ctx_add(cli_inputs=["maxim pause"]), steps=2)
    assert obs.worker.submits == [] and list(obs.ctrl.processed_cli_inputs) == ["maxim pause"]
    assert _logs(caplog, "Skipping LLM for command") == ["Skipping LLM for command: maxim pause"]


def test_spoken_maxim_submits_and_other_speech_does_not(monkeypatch, tmp_path):
    obs = _run(monkeypatch, tmp_path, amend=_ctx_add(detected_speech=["hi maxim"]), steps=2)
    assert _brief(obs) == [([], "")] and list(obs.ctrl.processed_cli_inputs) == []
    obs = _run(monkeypatch, tmp_path / "o", amend=_ctx_add(detected_speech=["hi there"]), steps=2)
    assert obs.worker.submits == []


@pytest.mark.parametrize("flag", [{"has_maxim_keyword": True}, {"hard_override": "request_sleep"}])
def test_a_keyword_or_hard_override_percept_submits(monkeypatch, tmp_path, flag):
    from maxim.agents.bus import Percept

    obs = _run(monkeypatch, tmp_path, amend=_ctx_add(current_percept=Percept(timestamp=0.0, source="cli", **flag)))
    assert _brief(obs) == [([], "")]


def test_nothing_meaningful_submits_nothing_in_either_mode(monkeypatch, tmp_path):
    """Pinned: in sim mode the "LLM skip: ... unprocessed cli_inputs" trace does not fire either; every input the gate
    can see is already marked processed by then."""
    assert _run(monkeypatch, tmp_path, steps=2).worker.submits == []
    obs = _run(monkeypatch, tmp_path / "sim", steps=2, sim_mode=True)
    assert obs.worker.submits == [] and not [m for c, m in obs.sim_logs if m.startswith("LLM skip")]


# ── a pending action follow-up ───────────────────────────────────────────────


def test_a_follow_up_becomes_the_input_and_is_cleared(monkeypatch, tmp_path, caplog):
    obs = _run(monkeypatch, tmp_path, hooks={0: _set_followup()}, steps=2)
    [(inputs, trigger)] = _brief(obs)
    assert len(inputs) == 1 and inputs[0].startswith(
        "[ACTION_FOLLOWUP type=process tool=probe_submit mode=active query='what was asked']: <<TOOL_OUTPUT"
    )
    assert "\nthe result\n" in inputs[0] and trigger == "what was asked"
    assert obs.ctrl.pending_action_followup is None and list(obs.ctrl.processed_cli_inputs) == []
    assert _logs(caplog, "Injected action followup") == [
        "Injected action followup into context: type=process, tool=probe_submit, result_len=10"
    ]
    assert _logs(caplog, "Submitted followup") == ["Submitted followup to LLM for processing"]


def test_a_follow_up_with_a_fresh_input_submits_the_input_and_drops_the_result(monkeypatch, tmp_path):
    """Pinned as-is (#1721): the follow-up is folded in and cleared, but the submission keeps only the fresh input."""
    obs = _run(monkeypatch, tmp_path, hooks={0: _set_followup()}, inputs={0: "new words"}, steps=2)
    assert _brief(obs) == [(["new words"], "new words")]
    assert obs.ctrl.pending_action_followup is None


def test_a_follow_up_with_no_built_context_gets_a_minimal_one(monkeypatch, tmp_path, caplog):
    obs = _run(monkeypatch, tmp_path, hooks={0: _set_followup()}, amend=lambda k, ctx: None, steps=2)
    [sub] = obs.worker.submits
    assert sub["mode"] == "active" and sub["autonomy_level"] == "supervised"  # operational mode; the state default
    assert sub["triggering_input"] == "what was asked" and sub["cli_inputs"][0].startswith("[ACTION_FOLLOWUP")
    assert _logs(caplog, "Created minimal context") == ["Created minimal context for pending action followup"]


def test_with_planning_liveness_a_held_input_rides_the_follow_up(monkeypatch, tmp_path):
    """#1048 through the seam: the held input reaches ``submit_context(deferred_inputs=...)`` beside the follow-up,
    marked processed; without planning liveness nothing is held, and the input preempts the follow-up (#1721's shape)."""
    held = {"hooks": {0: _set_followup()}, "amend": _ctx_add(cli_inputs=["held words"]), "steps": 2}
    obs = _run(monkeypatch, tmp_path, planning_liveness=True, **held)
    [sub] = obs.worker.submits
    assert sub["kw"]["deferred_inputs"] == ["held words"] and sub["triggering_input"] == "what was asked"
    assert len(sub["cli_inputs"]) == 1 and sub["cli_inputs"][0].startswith("[ACTION_FOLLOWUP")
    assert list(obs.ctrl.processed_cli_inputs) == ["held words"]
    obs = _run(monkeypatch, tmp_path / "off", **held)
    [sub] = obs.worker.submits
    assert sub["kw"]["deferred_inputs"] == [] and sub["cli_inputs"] == ["held words"]


def test_with_no_built_context_a_fresh_input_is_marked_processed_unsent(monkeypatch, tmp_path):
    """Pinned as-is (#1722): nothing to submit into, yet the input is recorded processed."""
    obs = _run(monkeypatch, tmp_path, amend=lambda k, ctx: None, inputs={0: "lost"}, steps=2)
    assert obs.worker.submits == [] and list(obs.ctrl.processed_cli_inputs) == ["lost"]


# ── sleep ────────────────────────────────────────────────────────────────────


def test_asleep_only_a_wake_keyword_reaches_the_llm(monkeypatch, tmp_path, caplog):
    asleep = {"processing_state": "sleep"}
    obs = _run(monkeypatch, tmp_path, inputs={0: "what time is it"}, state_data=asleep, steps=2)
    assert obs.worker.submits == [] and list(obs.ctrl.processed_cli_inputs) == ["what time is it"]
    assert _logs(caplog, "Sleep state") == ["Sleep state: skipping LLM (no wake keyword)"]
    obs = _run(monkeypatch, tmp_path / "w", inputs={0: "hey maxim are you up"}, state_data=asleep, steps=2)
    assert _brief(obs) == [(["hey maxim are you up"], "hey maxim are you up")]
    assert [s["kw"]["is_sleeping"] for s in obs.worker.submits] == [True]  # the request says it was woken
    obs = _run(monkeypatch, tmp_path / "awake", inputs={0: "hey maxim are you up"}, steps=2)
    assert [s["kw"]["is_sleeping"] for s in obs.worker.submits] == [False]


# ── Wire-A and W1 annotations ────────────────────────────────────────────────

BIASES = [("tool:probe_submit", 0.4), ("tool:other", -0.2)]


def test_wire_a_annotates_the_agent_wide_biases_and_their_sources(monkeypatch, tmp_path):
    nac = _make_nac(BIASES, {"tool:probe_submit": "operant"})
    obs = _run(monkeypatch, tmp_path, inputs={0: "hi"}, nac=nac, steps=2)
    [sub] = obs.worker.submits
    assert sub["cluster_bias_annotations"] == BIASES
    assert sub["cluster_bias_sources"] == {"tool:probe_submit": "operant"}
    assert nac.calls[:2] == [("biases", "submit_hub", 5), ("sources", "submit_hub")]  # per pass, the hub's agent


def test_the_kill_switch_turns_off_both_annotations_and_reads_nothing(monkeypatch, tmp_path):
    monkeypatch.setenv("MAXIM_DISABLE_CLUSTER_BIAS_ANNOTATION", "1")
    nac = _make_nac(BIASES, {})
    spy = _grayscale_spy(monkeypatch, ["SENTINEL"])
    obs = _run(monkeypatch, tmp_path, inputs={0: "hi"}, nac=nac, steps=2)
    [sub] = obs.worker.submits
    assert sub["cluster_bias_annotations"] is None and sub["grayscale_tool_annotations"] is None
    assert nac.calls == [] and spy == []


def test_an_invalid_agent_id_skips_wire_a_loudly_and_still_submits(monkeypatch, tmp_path, caplog):
    obs = _run(monkeypatch, tmp_path, inputs={0: "hi"}, nac=_make_nac(ValueError("empty agent_id"), {}), steps=2)
    [sub] = obs.worker.submits
    assert sub["cluster_bias_annotations"] is None and sub["cluster_bias_sources"] is None
    assert [r.levelno for r in caplog.records if r.getMessage().startswith("Wire-A annotation skipped")] == [
        logging.WARNING
    ] * 2


def test_a_sources_value_error_also_clears_the_biases_already_read(monkeypatch, tmp_path):
    """The handler resets BOTH fields: with the bias read done and the sources read raising, the biases are dropped."""
    obs = _run(monkeypatch, tmp_path, inputs={0: "hi"}, nac=_make_nac(BIASES, ValueError("empty agent_id")), steps=2)
    [sub] = obs.worker.submits
    assert sub["cluster_bias_annotations"] is None and sub["cluster_bias_sources"] is None


def _grayscale_spy(monkeypatch: pytest.MonkeyPatch, result: Any) -> list[tuple]:
    import maxim.prompts.grayscale_tools_annotation as grayscale

    calls: list[tuple] = []

    def spy(biases: Any, registry: Any, top_n: int = 5) -> Any:
        calls.append((list(biases), registry, top_n))
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(grayscale, "build_grayscale_annotations", spy)
    return calls


def test_grayscale_is_built_from_wire_a_biases_and_the_executor_registry(monkeypatch, tmp_path):
    spy = _grayscale_spy(monkeypatch, ["SENTINEL"])
    obs = _run(monkeypatch, tmp_path, inputs={0: "hi"}, nac=_make_nac(BIASES, {}), steps=2)
    [sub] = obs.worker.submits
    assert sub["grayscale_tool_annotations"] == ["SENTINEL"]
    from maxim.tools.registry import ToolRegistry

    biases, registry, top_n = spy[0]
    assert biases == BIASES and isinstance(registry, ToolRegistry) and PROBE in registry.list_all() and top_n == 5


def test_grayscale_needs_biases(monkeypatch, tmp_path):
    spy = _grayscale_spy(monkeypatch, ["SENTINEL"])
    obs = _run(monkeypatch, tmp_path, inputs={0: "hi"}, nac=_make_nac([], {}), steps=2)
    assert spy == [] and obs.worker.submits[0]["grayscale_tool_annotations"] is None


@pytest.mark.parametrize("error", [ValueError, AttributeError])
def test_a_grayscale_value_error_is_logged_and_the_annotation_dropped(monkeypatch, tmp_path, caplog, error):
    """Both caught types (the handler is ``except (AttributeError, ValueError)``). The handler's
    ``grayscale_tool_annotations = None`` is unobservable (the field is only ever set after a build succeeds), so
    deleting it changes nothing: an equivalent mutant, recorded here rather than pinned."""
    _grayscale_spy(monkeypatch, error("bad shape"))
    obs = _run(monkeypatch, tmp_path, inputs={0: "hi"}, nac=_make_nac(BIASES, {}), steps=2)
    assert obs.worker.submits[0]["grayscale_tool_annotations"] is None
    assert "Grayscale tools annotation skipped: bad shape" in _logs(caplog, "Grayscale")


def test_a_follow_up_joins_the_inputs_memory_already_lists(monkeypatch, tmp_path):
    """Pass 0 submits a fresh input; on pass 1 memory still lists it (processed) and the follow-up is appended after it,
    and the submission keeps the follow-up, not the stale input."""
    obs = _run(monkeypatch, tmp_path, inputs={0: "first"}, hooks={1: _set_followup()}, steps=3)
    assert [s["cli_inputs"][0][:16] for s in obs.worker.submits] == ["first", "[ACTION_FOLLOWUP"]
    assert obs.worker.submits[1]["triggering_input"] == "what was asked"


def test_asleep_spoken_maxim_with_a_wake_keyword_reaches_the_llm(monkeypatch, tmp_path):
    asleep = {"processing_state": "sleep"}
    obs = _run(monkeypatch, tmp_path, amend=_ctx_add(detected_speech=["hey maxim"]), state_data=asleep, steps=2)
    assert _brief(obs) == [([], "")]


class _Pipeline:
    """The two members §1.2 reads off a BioEnrichmentPipeline; its formatted text is what §6 injects."""

    def enrich(self, text: str, **kw: Any) -> Any:
        return SimpleNamespace(memories=["m"], predictions=[], concepts=[], affordances=[], recent_context=None)

    def format_thought_response(self, result: Any) -> str:
        return "ENRICHED: you remember m"


def test_the_percepts_enrichment_text_is_injected_into_the_context(monkeypatch, tmp_path):
    obs = _run(monkeypatch, tmp_path, inputs={0: "hi"}, pipeline=_Pipeline(), steps=2)
    assert obs.worker.submits[0]["bio_enrichment_context"] == "ENRICHED: you remember m"
    obs = _run(monkeypatch, tmp_path / "none", inputs={0: "hi"}, steps=2)
    assert obs.worker.submits[0]["bio_enrichment_context"] == ""
