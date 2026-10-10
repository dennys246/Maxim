"""Characterization of ``run_agentic_loop``'s PERCEPTION sections, §1-§1.16 (1.3.2 decomposition, slice 5).

Pins, through the PUBLIC entry ``agent_loop.run_agentic_loop`` on the LLM-primary path (the default
``aut_mode``), what one pass does between the pre-tick gate and §1.2 bio-enrichment, and where its three
outputs land:

- §1 order: ``next_observation`` -> ``state.update(observation)`` -> §1.1 ``process_percept`` -> §1.15 the
  auto-fire executes -> §1.16 the azimuth world-set, in that order within one pass.
- §1.1 IMAGINATION: which text the trigger sees (a dict observation: ``transcript`` > ``raw_transcript_text``
  > ``cli_input``; an attribute observation: ``transcript`` > ``cli_input``), the scene it gets from
  ``state.data``, the SEM_TRACE line when there is no text, a raising trigger, and its results' ``ref``s
  reaching §1.2's ``EnrichmentContext.resolved_entities`` (``None`` refs dropped).
- §1.15 AUTO-SENSE: only on a new percept; every auto-fire tool in registry order with no arguments; failed,
  empty and raising tools; the legacy registry fallback; interoception per self-entity from the first tool
  that exposes ``_entity_map``; the debug line when none does; the ``"\\n"`` join; and the exact string arriving
  as ``context.auto_sense_context`` at the LLM submission (captured from the worker).
- §1.16 AUDIO ORIENTATION: the gate (a carried percept on a non-sim adapter fires; substrate-primary or no
  percept skips), an azimuth-less percept, the world-set of the REAL ``bodies/reachy_mini_infant`` body's
  ``azimuth`` sensor (skipped when the sensor is live-owned), the reflex tier (sim runs only: its exact line,
  the oriented ``_last_audio_orient_az`` and the world-set to it), the deliberative tier (the change gate, the
  clamp, escalation only above the body's threshold), the append after auto-sense, the B1 minimal context
  that carries an escalating audio-only pass to the LLM (not while sleeping), and a raise inside the section.
  §1.16 commands no motion: it only world-sets a sensor value (pinned below).
- The percept-text divergence (#1202, pinned as it is): an attribute observation with ``cli_input`` and no
  ``transcript`` feeds §1.1 and §1.2 but does NOT auto-sense.

Written BEFORE slice 5 moved the block out of ``agent_loop.py`` and kept green unchanged by that move
(``docs/plans/roadmap_1_3_x.md`` §"The decomposition", coverage-first rule). Observation is
location-independent: the driver is slice 2's (``test_loop_gates_characterization._run``), passes are seen
through the loop's own ``loop_iteration`` event, the observation through a scripted adapter, the update
through ``RuntimeState.update``, the trigger, tools, pipeline and worker are fakes passed in, the sim lines
through ``sim_logger.sim_log``, the world-set through ``audio_localization.world_set_azimuth`` (both read at
call time by the section's lazy imports) and the swallowed-exception reports through the ``maxim`` logger.
Nothing patches ``agent_loop`` or the module the sections live in.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from maxim.runtime.sim_adapter import NullSimulationAdapter
from maxim.tools.base import Tool, ToolOutput
from maxim.tools.registry import ToolRegistry
from tests.unit.test_loop_gates_characterization import _run as _gate_run

pytestmark = pytest.mark.timeout(60)

LOGGER = "maxim.runtime.agent_loop"


# ── harness ──────────────────────────────────────────────────────────────────


class _Scripted(NullSimulationAdapter):
    """A non-sim adapter whose k-th full pass observes ``obs[k]`` (``{}`` past the end) and carries
    ``percepts[k]`` on the modality-preserving side-channel. It always reports a carried percept, so the
    idle gate runs every pass in full."""

    def __init__(self, ev: list[tuple], obs: list[Any] | None = None, percepts: dict[int, Any] | None = None):
        super().__init__()
        self.ev, self.obs, self.percepts = ev, list(obs or []), dict(percepts or {})
        self.k = 0

    def has_carried_percept(self) -> bool:
        return True

    def next_observation(self, environment: Any, default_network: Any | None = None) -> Any:
        k, self.k = self.k, self.k + 1
        if k in self.percepts:
            self.carry_percept(self.percepts[k])
        super().next_observation(environment, default_network)  # surfaces (or clears) current_percept
        self.ev.append(("observe", k))
        return self.obs[k] if k < len(self.obs) else {}


class _Percepts:
    """A percept source (it makes the run a SIM run, ``is_sim_mode``) that hands out ``items`` in order."""

    def __init__(self, ev: list[tuple], items: list[Any]):
        self.ev, self.items = ev, list(items)

    def next_percept(self) -> Any:
        self.ev.append(("observe", None))
        return self.items.pop(0) if self.items else None

    def has_pending(self) -> bool:
        return bool(self.items)

    def is_exhausted(self) -> bool:
        return False


class _Trigger:
    """``imagination_trigger``: records each ``process_percept`` call; returns ``results`` or raises."""

    def __init__(self, ev: list[tuple], results: list[Any] | None = None, raises: bool = False):
        self.ev, self.results, self.raises = ev, list(results or []), raises
        self.calls: list[tuple] = []

    def process_percept(self, text: str, *, scene_context: Any = None, scene_id: Any = None) -> list:
        self.ev.append(("imagine", text))
        self.calls.append((text, scene_context, scene_id))
        if self.raises:
            raise RuntimeError("designer down")
        return list(self.results)


class _Pipeline:
    """``bio_enrichment_pipeline``: records what §1.2 hands ``enrich``; enriches nothing."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def enrich(self, text: str, *, context: Any = None, bypass_gate: bool = False) -> None:
        self.calls.append((text, context))
        return None


class _Worker:
    """``llm_worker``: captures every submitted ``context``; never proposes."""

    def __init__(self) -> None:
        self.contexts: list[Any] = []

    def get_latest_proposal(self) -> None:
        return None

    def submit_context(self, *_a: Any, context: Any = None, **_k: Any) -> bool:
        self.contexts.append(context)
        return True

    def latest_attempt_state(self) -> Any:  # read only under planning liveness, which these runs leave off
        raise AssertionError("planning liveness is off in these runs")


class _AutoFire(Tool):
    """An auto-fire tool: records its calls; returns ``out`` (or fails, or raises)."""

    description = "auto-fire probe"
    input_schema: dict[str, Any] = {}
    auto_fire = True
    kind = "auto-discovery"

    def __init__(self, ev: list[tuple], name: str, out: str = "", success: bool = True, raises: bool = False):
        self.name = name
        self.ev, self.out, self.success, self.raises = ev, out, success, raises
        self.kwargs: list[dict] = []
        super().__init__()

    def execute(self, **kwargs: Any) -> ToolOutput:
        self.ev.append(("auto_fire", self.name))
        self.kwargs.append(dict(kwargs))
        if self.raises:
            raise RuntimeError(f"{self.name} broke")
        return ToolOutput(success=self.success, output=self.out)


class _EntityMap:
    def __init__(self, *names: str) -> None:
        self.names = names

    def list_self_entities(self) -> list[Any]:
        return [SimpleNamespace(name=n) for n in self.names]


class _Sense(Tool):
    """The (LLM-callable, not auto-fire) ``sense`` tool: answers per entity name."""

    name = "sense"
    description = "sense probe"
    input_schema: dict[str, Any] = {}
    auto_fire = False
    kind = "core-universal"

    def __init__(self, answers: dict[str, tuple[bool, str]]) -> None:
        self.answers = answers
        self.asked: list[str] = []
        super().__init__()

    def execute(self, **kwargs: Any) -> ToolOutput:
        name = kwargs["entity_name"]
        self.asked.append(name)
        ok, out = self.answers.get(name, (True, f"{name} ok"))
        return ToolOutput(success=ok, output=out)


def _executor(*tools: Tool, registry: Any = None) -> Any:
    from maxim.runtime.bootstrap import build_executor

    reg = ToolRegistry() if registry is None else registry
    for t in tools:
        reg.register(t)
    return build_executor(reg, pain_bus=None, permissions=None)


def _infant_body() -> Any:
    """The REAL ``bodies/reachy_mini_infant`` (an ``azimuth`` sensor, range [-1, 1], initial 0.0, the default
    orienting profile), as the executor's embodiment."""
    from maxim.embodiment.body import Embodiment
    from maxim.embodiment.component_registry import ComponentRegistry

    return Embodiment(ComponentRegistry().instantiate("bodies/reachy_mini_infant"))


def _audio(az: float, salience: float = 0.3, novelty: float = 0.3) -> Any:
    from maxim.agents.percept_factory import make_audio_percept

    return make_audio_percept(az, salience=salience, novelty=novelty)


class _Obs(SimpleNamespace):
    sim_logs: list[tuple]
    world_sets: list[float]


def _run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    obs: list[Any] | None = None,
    percepts: dict[int, Any] | None = None,
    sim_percepts: list[Any] | None = None,
    steps: int = 1,
    executor: Any = None,
    body: Any = None,
    worker: Any = None,
    no_memory_context: bool = False,
    state_data: dict[str, Any] | None = None,
    sim_log_raises: tuple[str, ...] = (),
    **kwargs: Any,
) -> Any:
    """Run the real loop (slice 2's driver) for ``steps`` passes on a scripted adapter, or on a sim run's
    percept source with ``sim_percepts``. ``worker`` submits on every pass that has something to submit;
    ``no_memory_context`` makes ``memory.build_context()`` return None (the B1 arm); ``sim_log`` raises for
    the categories in ``sim_log_raises``."""
    import maxim.embodiment.audio_localization as audio_localization
    import maxim.simulation.sim_logger as sim_logger
    from maxim.runtime.state import RuntimeState

    ev: list[tuple] = []
    sim_logs: list[tuple] = []
    world_sets: list[float] = []

    real_update = RuntimeState.update

    def _update(self: Any, observation: Any) -> None:
        ev.append(("update",))
        real_update(self, observation)

    monkeypatch.setattr(RuntimeState, "update", _update)

    def _sim_log(category: str, msg: str, data: Any = None, **kw: Any) -> None:
        sim_logs.append((category, msg, data, kw))
        if category in sim_log_raises:
            raise RuntimeError("sim log down")

    monkeypatch.setattr(sim_logger, "sim_log", _sim_log)
    real_world_set = audio_localization.world_set_azimuth

    def _world_set(embodiment: Any, azimuth: float, **kw: Any) -> bool:
        ev.append(("world_set", azimuth))
        world_sets.append(azimuth)
        return real_world_set(embodiment, azimuth, **kw)

    monkeypatch.setattr(audio_localization, "world_set_azimuth", _world_set)
    if no_memory_context:
        from maxim.memory.base import InMemoryMemory

        monkeypatch.setattr(InMemoryMemory, "build_context", lambda self: None)
    if state_data:
        real_init = RuntimeState.__init__

        def _init(self: Any, *a: Any, **k: Any) -> None:
            real_init(self, *a, **k)
            self.data.update(state_data)

        monkeypatch.setattr(RuntimeState, "__init__", _init)

    if executor is None:
        executor = _executor()
    if body is not None:
        executor.embodiment = body
    if worker is not None:
        kwargs["llm_worker"] = worker
    if sim_percepts is not None:
        kwargs["percept_source"] = _Percepts(ev, sim_percepts)
        adapter = None
    else:
        adapter = _Scripted(ev, obs, percepts)
    run = _gate_run(
        monkeypatch,
        tmp_path,
        steps=steps,
        adapter=adapter,
        ev=ev,
        executor=executor,
        submit_interval=0.0 if worker is not None else 1e12,
        **kwargs,
    )
    if run.error is not None:
        raise run.error
    return _Obs(ev=ev, sim_logs=sim_logs, world_sets=world_sets, state=run.state, ctrl=run.ctrl, executor=executor)


def _kinds(ev: list[tuple]) -> list[str]:
    return [e[0] for e in ev]


def _swallowed(caplog: Any, operation: str) -> list[str]:
    return [r.getMessage() for r in caplog.records if f" in {operation}:" in r.getMessage()]


# ── §1 order ─────────────────────────────────────────────────────────────────


def test_the_section_order_within_a_pass_is_observe_update_imagine_auto_fire_world_set(monkeypatch, tmp_path):
    shared: list[tuple] = []

    class _T(_Trigger):
        def process_percept(self, text: str, **kw: Any) -> list:
            shared.append(("imagine",))
            return []

    class _A(_AutoFire):
        def execute(self, **kwargs: Any) -> ToolOutput:
            shared.append(("auto_fire",))
            return ToolOutput(success=True, output="seen")

    import maxim.embodiment.audio_localization as audio_localization
    from maxim.runtime.state import RuntimeState as _RS

    real_update = _RS.update
    real_ws = audio_localization.world_set_azimuth

    def _upd(self: Any, observation: Any) -> None:
        shared.append(("update",))
        real_update(self, observation)

    def _ws(emb: Any, az: float, **kw: Any) -> bool:
        shared.append(("world_set",))
        return real_ws(emb, az, **kw)

    class _S(_Scripted):
        def next_observation(self, environment: Any, default_network: Any | None = None) -> Any:
            out = super().next_observation(environment, default_network)
            shared.append(("observe",))
            return out

    import maxim.simulation.sim_logger as sim_logger

    monkeypatch.setattr(sim_logger, "sim_log", lambda *a, **k: None)
    monkeypatch.setattr(_RS, "update", _upd)
    monkeypatch.setattr(audio_localization, "world_set_azimuth", _ws)
    executor = _executor(_A([], "scan"))
    executor.embodiment = _infant_body()
    run = _gate_run(
        monkeypatch,
        tmp_path,
        steps=1,
        adapter=_S([], [{"transcript": "a door creaks"}], {0: _audio(0.6)}),
        executor=executor,
        imagination_trigger=_T([]),
    )
    assert run.error is None
    assert [e[0] for e in shared] == ["observe", "update", "imagine", "auto_fire", "world_set"]


# ── §1.1 imagination ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("observation", "text"),
    [
        ({"transcript": "T", "raw_transcript_text": "R", "cli_input": "C"}, "T"),
        ({"transcript": "", "raw_transcript_text": "R", "cli_input": "C"}, "R"),
        ({"transcript": None, "raw_transcript_text": "", "cli_input": "C"}, "C"),
        (SimpleNamespace(transcript="T", cli_input="C"), "T"),
        (SimpleNamespace(transcript="", cli_input="C"), "C"),
    ],
    ids=["dict-transcript", "dict-raw", "dict-cli", "attr-transcript", "attr-cli"],
)
def test_imagination_sees_the_first_percept_text_by_priority_and_the_states_scene(
    monkeypatch, tmp_path, observation, text
):
    trig = _Trigger([])
    _run(
        monkeypatch,
        tmp_path,
        obs=[observation],
        imagination_trigger=trig,
        state_data={"current_scene_id": "scene-7", "scene_context": {"room": "nursery"}},
    )
    assert trig.calls == [(text, {"room": "nursery"}, "scene-7")]


def test_an_attribute_observation_never_reads_raw_transcript_text(monkeypatch, tmp_path):
    trig = _Trigger([])
    o = _run(
        monkeypatch, tmp_path, obs=[SimpleNamespace(transcript="", raw_transcript_text="R")], imagination_trigger=trig
    )
    assert trig.calls == []
    [(_, msg, _, kw)] = [s for s in o.sim_logs if s[0] == "SEM_TRACE"]
    assert msg == "Imagination skipped: no percept_text (obs keys: SimpleNamespace)"
    assert kw == {"_force_debug": True}


def test_no_percept_text_logs_a_sem_trace_line_with_the_observation_keys_and_never_imagines(monkeypatch, tmp_path):
    trig = _Trigger([])
    o = _run(monkeypatch, tmp_path, obs=[{"source": "world", "transcript": ""}], imagination_trigger=trig)
    assert trig.calls == []
    [(_, msg, data, kw)] = [s for s in o.sim_logs if s[0] == "SEM_TRACE"]
    assert msg == "Imagination skipped: no percept_text (obs keys: ['source', 'transcript'])"
    assert data is None and kw == {"_force_debug": True}


def test_without_a_trigger_imagination_is_skipped_silently(monkeypatch, tmp_path):
    o = _run(monkeypatch, tmp_path, obs=[{}])
    assert [s for s in o.sim_logs if s[0] == "SEM_TRACE"] == []


def test_a_raising_trigger_is_reported_and_the_loop_goes_on(monkeypatch, tmp_path, caplog):
    trig = _Trigger([], raises=True)
    pipe = _Pipeline()
    with caplog.at_level(logging.DEBUG, logger="maxim"):
        o = _run(
            monkeypatch,
            tmp_path,
            obs=[{"transcript": "x"}, {"transcript": "y"}],
            steps=2,
            imagination_trigger=trig,
            bio_enrichment_pipeline=pipe,
        )
    assert [c[0] for c in trig.calls] == ["x", "y"]
    assert _swallowed(caplog, "imagination_trigger") == [
        "Swallowed RuntimeError in imagination_trigger: designer down [step=0]",
        "Swallowed RuntimeError in imagination_trigger: designer down [step=1]",
    ]
    # The failed pass hands §1.2 no resolved entities.
    assert [ctx.resolved_entities for _, ctx in pipe.calls] == [(), ()]
    assert _kinds(o.ev).count("observe") == 2


def test_imagination_refs_reach_bio_enrichment_with_none_refs_dropped(monkeypatch, tmp_path):
    results = [SimpleNamespace(ref="sem:lamp"), SimpleNamespace(ref=None), SimpleNamespace(ref="sem:door")]
    trig = _Trigger([], results=results)
    pipe = _Pipeline()
    _run(
        monkeypatch,
        tmp_path,
        obs=[{"transcript": "a lamp and a door"}],
        imagination_trigger=trig,
        bio_enrichment_pipeline=pipe,
    )
    [(text, ctx)] = pipe.calls
    assert text == "a lamp and a door"
    assert ctx.resolved_entities == ("sem:lamp", "sem:door")


def test_no_imagination_results_reach_bio_enrichment_as_no_entities(monkeypatch, tmp_path):
    pipe = _Pipeline()
    _run(
        monkeypatch,
        tmp_path,
        obs=[{"transcript": "hello"}],
        imagination_trigger=_Trigger([]),
        bio_enrichment_pipeline=pipe,
    )
    (tmp_path / "b").mkdir()
    _run(monkeypatch, tmp_path / "b", obs=[{"transcript": "hello"}], bio_enrichment_pipeline=pipe)
    assert [ctx.resolved_entities for _, ctx in pipe.calls] == [(), ()]


# ── §1.15 auto-sense ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("observation", "fires"),
    [
        ({}, False),
        ({"transcript": "", "raw_transcript_text": None, "cli_input": ""}, False),
        ({"transcript": "t"}, True),
        ({"raw_transcript_text": "r"}, True),
        ({"cli_input": "c"}, True),
        (SimpleNamespace(transcript="t"), True),
        # #1202, pinned as it is: an attribute observation with cli_input and no transcript does NOT
        # auto-sense, though §1.1 and §1.2 accept its cli_input.
        (SimpleNamespace(transcript="", cli_input="c"), False),
        (SimpleNamespace(cli_input="c"), False),
    ],
    ids=["empty", "all-blank", "transcript", "raw", "cli", "attr-transcript", "attr-cli-1202", "attr-no-transcript"],
)
def test_auto_sense_fires_only_on_a_new_percept(monkeypatch, tmp_path, observation, fires):
    tool = _AutoFire([], "scan", out="seen")
    _run(monkeypatch, tmp_path, obs=[observation], executor=_executor(tool))
    assert tool.kwargs == ([{}] if fires else [])


def test_the_1202_divergence_one_cli_only_percept_feeds_imagination_and_enrichment_but_not_auto_sense(
    monkeypatch, tmp_path
):
    trig, pipe, tool = _Trigger([]), _Pipeline(), _AutoFire([], "scan", out="seen")
    _run(
        monkeypatch,
        tmp_path,
        obs=[SimpleNamespace(transcript="", cli_input="look around")],
        executor=_executor(tool),
        imagination_trigger=trig,
        bio_enrichment_pipeline=pipe,
    )
    assert [c[0] for c in trig.calls] == ["look around"]
    assert [t for t, _ in pipe.calls] == ["look around"]
    assert tool.kwargs == []


def _auto_sense_submitted(monkeypatch: Any, tmp_path: Path, *tools: Tool, registry: Any = None, **kw: Any) -> Any:
    worker = _Worker()
    o = _run(
        monkeypatch,
        tmp_path,
        obs=kw.pop("obs", [{"cli_input": "what is around me"}]),
        executor=_executor(*tools, registry=registry),
        worker=worker,
        **kw,
    )
    assert worker.contexts, "the pass submitted nothing"
    return worker.contexts[0], o


def test_every_auto_fire_tool_runs_in_registry_order_with_no_arguments_and_reaches_the_submission(
    monkeypatch, tmp_path
):
    ev: list[tuple] = []
    a, b = _AutoFire(ev, "scan_a", out="[SCENE] a lamp"), _AutoFire(ev, "scan_b", out="[YOU] standing")
    ctx, o = _auto_sense_submitted(monkeypatch, tmp_path, a, b)
    assert ev == [("auto_fire", "scan_a"), ("auto_fire", "scan_b")]
    assert a.kwargs == [{}] and b.kwargs == [{}]
    assert ctx.auto_sense_context == "[SCENE] a lamp\n[YOU] standing"
    [(_, msg, _, _)] = [s for s in o.sim_logs if s[0] == "PERCEPTION"]
    assert msg == "auto-sense: 2 entities, body state updated"


def test_failed_empty_and_raising_tools_are_skipped_and_the_rest_still_run(monkeypatch, tmp_path, caplog):
    ev: list[tuple] = []
    tools = (
        _AutoFire(ev, "fails", out="nope", success=False),
        _AutoFire(ev, "empty", out=""),
        _AutoFire(ev, "raises", raises=True),
        _AutoFire(ev, "works", out="a quiet room"),
    )
    with caplog.at_level(logging.DEBUG, logger="maxim"):
        ctx, _ = _auto_sense_submitted(monkeypatch, tmp_path, *tools)
    assert [e[1] for e in ev] == ["fails", "empty", "raises", "works"]
    assert ctx.auto_sense_context == "a quiet room"
    assert _swallowed(caplog, "auto_fire:raises") == [
        "Swallowed RuntimeError in auto_fire:raises: raises broke [step=0]"
    ]


def test_no_auto_sense_output_leaves_the_submission_without_auto_sense_context(monkeypatch, tmp_path):
    ctx, o = _auto_sense_submitted(monkeypatch, tmp_path, _AutoFire([], "empty", out=""))
    assert ctx.auto_sense_context == ""
    assert [s for s in o.sim_logs if s[0] == "PERCEPTION"] == []


def test_interoception_senses_each_self_entity_of_the_first_tool_exposing_an_entity_map(monkeypatch, tmp_path):
    first = _AutoFire([], "scan", out="[SCENE] a crib")
    first._entity_map = _EntityMap("baby", "twin")
    second = _AutoFire([], "scan2", out="")
    second._entity_map = _EntityMap("other")
    sense = _Sense({"twin": (True, "tired")})
    ctx, _ = _auto_sense_submitted(monkeypatch, tmp_path, first, second, sense)
    assert sense.asked == ["baby", "twin"]
    assert ctx.auto_sense_context == "[SCENE] a crib\nBody state (baby): baby ok\nBody state (twin): tired"


def test_a_failed_or_empty_sense_result_is_left_out(monkeypatch, tmp_path):
    tool = _AutoFire([], "scan", out="here")
    tool._entity_map = _EntityMap("a", "b", "c")
    sense = _Sense({"a": (False, "err"), "b": (True, "")})
    ctx, _ = _auto_sense_submitted(monkeypatch, tmp_path, tool, sense)
    assert ctx.auto_sense_context == "here\nBody state (c): c ok"


def test_a_raising_entity_map_is_reported_and_keeps_the_exteroception(monkeypatch, tmp_path, caplog):
    class _Broken:
        def list_self_entities(self) -> list:
            raise RuntimeError("map gone")

    tool = _AutoFire([], "scan", out="here")
    tool._entity_map = _Broken()
    with caplog.at_level(logging.DEBUG, logger="maxim"):
        ctx, _ = _auto_sense_submitted(monkeypatch, tmp_path, tool, _Sense({}))
    assert ctx.auto_sense_context == "here"
    assert _swallowed(caplog, "auto_sense_self") == ["Swallowed RuntimeError in auto_sense_self: map gone [step=0]"]


def test_interoception_without_an_entity_map_logs_a_debug_line(monkeypatch, tmp_path, caplog):
    sense = _Sense({})
    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        ctx, _ = _auto_sense_submitted(monkeypatch, tmp_path, _AutoFire([], "scan", out="here"), sense)
    assert sense.asked == []
    assert ctx.auto_sense_context == "here"
    assert [
        r.getMessage() for r in caplog.records if r.name == LOGGER and "interoception skipped" in r.getMessage()
    ] == [
        "auto-sense interoception skipped: no auto-fire tool exposes _entity_map "
        "(sense tool present but cannot be dispatched per-entity)"
    ]


def test_no_auto_fire_tools_means_no_interoception_and_no_debug_line(monkeypatch, tmp_path, caplog):
    sense = _Sense({})
    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        ctx, _ = _auto_sense_submitted(monkeypatch, tmp_path, sense)
    assert sense.asked == [] and ctx.auto_sense_context == ""
    assert not [r for r in caplog.records if "interoception skipped" in r.getMessage()]


class _LegacyRegistry(ToolRegistry):
    """An older registry without ``get_auto_fire_tools``."""

    @property
    def get_auto_fire_tools(self) -> Any:  # type: ignore[override]
        raise AttributeError("get_auto_fire_tools")


def test_a_legacy_registry_falls_back_to_sense_presence_by_name(monkeypatch, tmp_path):
    presence = _AutoFire([], "sense_presence", out="[SCENE] legacy scan")
    presence._entity_map = _EntityMap("me")
    ctx, _ = _auto_sense_submitted(monkeypatch, tmp_path, presence, _Sense({}), registry=_LegacyRegistry())
    assert presence.kwargs == [{}]
    assert ctx.auto_sense_context == "[SCENE] legacy scan\nBody state (me): me ok"


def test_a_legacy_registry_without_sense_presence_auto_senses_nothing(monkeypatch, tmp_path):
    other = _AutoFire([], "scan", out="never")
    ctx, _ = _auto_sense_submitted(monkeypatch, tmp_path, other, registry=_LegacyRegistry())
    assert other.kwargs == [] and ctx.auto_sense_context == ""


def test_a_registry_that_breaks_is_reported_as_auto_sense(monkeypatch, tmp_path, caplog):
    class _Broken(ToolRegistry):
        def get_auto_fire_tools(self) -> Any:  # type: ignore[override]
            raise RuntimeError("registry broke")

    with caplog.at_level(logging.DEBUG, logger="maxim"):
        ctx, _ = _auto_sense_submitted(monkeypatch, tmp_path, registry=_Broken())
    assert ctx.auto_sense_context == ""
    assert _swallowed(caplog, "auto_sense") == ["Swallowed RuntimeError in auto_sense: registry broke [step=0]"]


# ── §1.16 audio orientation ──────────────────────────────────────────────────


def _azimuth(body: Any) -> float:
    return body.root.vital_metrics["azimuth"]


def test_a_carried_audio_percept_world_sets_the_real_bodys_azimuth_sensor(monkeypatch, tmp_path):
    body = _infant_body()
    assert _azimuth(body) == 0.0
    o = _run(monkeypatch, tmp_path, obs=[{}], percepts={0: _audio(-0.4)}, body=body)
    assert o.world_sets == [-0.4]
    assert _azimuth(body) == -0.4


def test_the_world_set_is_clamped_to_the_sensors_declared_range(monkeypatch, tmp_path):
    body = _infant_body()
    o = _run(monkeypatch, tmp_path, obs=[{}], percepts={0: _audio(1.7)}, body=body)
    assert o.world_sets == [1.7] and _azimuth(body) == 1.0


def test_a_live_owned_azimuth_sensor_is_not_world_set(monkeypatch, tmp_path, caplog):
    body = _infant_body()
    body.live_world_set_sensors.add("azimuth")
    with caplog.at_level(logging.WARNING):
        o = _run(monkeypatch, tmp_path, obs=[{}], percepts={0: _audio(-0.4)}, body=body)
    assert o.world_sets == [] and _azimuth(body) == 0.0
    assert not [r for r in caplog.records if "refused anonymous write" in r.getMessage()]


def test_section_1_16_commands_no_motion(monkeypatch, tmp_path):
    """No tool dispatch; the only body write is the azimuth sensor."""
    body = _infant_body()
    before = dict(body.root.vital_metrics)
    executed: list[Any] = []
    executor = _executor()
    real_execute = type(executor).execute

    def _spy(self: Any, *a: Any, **k: Any) -> Any:
        executed.append((a, k))
        return real_execute(self, *a, **k)

    monkeypatch.setattr(type(executor), "execute", _spy)
    _run(
        monkeypatch,
        tmp_path,
        obs=[{}],
        sim_percepts=[_audio(0.7, salience=0.95, novelty=0.95)],
        executor=executor,
        body=body,
    )
    assert executed == []
    after = dict(body.root.vital_metrics)
    assert {k for k in after if after[k] != before[k]} <= {"azimuth"}


def test_no_carried_percept_skips_section_1_16(monkeypatch, tmp_path):
    body = _infant_body()
    o = _run(monkeypatch, tmp_path, obs=[{}], body=body)
    assert o.world_sets == [] and [s for s in o.sim_logs if s[1].startswith("audio-orient")] == []


def test_substrate_primary_skips_section_1_16(monkeypatch, tmp_path):
    body = _infant_body()
    o = _run(monkeypatch, tmp_path, obs=[{}], percepts={0: _audio(-0.4)}, body=body, aut_mode="substrate-primary")
    assert o.world_sets == [] and _azimuth(body) == 0.0
    assert "_last_audio_orient_az" not in o.state.data


def test_a_percept_without_an_azimuth_does_nothing(monkeypatch, tmp_path):
    body = _infant_body()
    no_az = SimpleNamespace(metadata={"source": "mic"}, salience=0.99, novelty=0.99)
    o = _run(monkeypatch, tmp_path, obs=[{}], percepts={0: no_az}, body=body)
    assert o.world_sets == [] and "_last_audio_orient_az" not in o.state.data
    assert [s for s in o.sim_logs if s[0] in ("PERCEPTION", "REACTION")] == []


def test_without_an_embodiment_the_deliberative_line_still_folds(monkeypatch, tmp_path):
    worker = _Worker()
    o = _run(monkeypatch, tmp_path, obs=[{"cli_input": "hi"}], percepts={0: _audio(-0.6)}, worker=worker)
    assert o.world_sets == []
    assert worker.contexts[0].auto_sense_context == "You hear a sound well to your left (azimuth -0.60)."


def test_the_reflex_tier_orients_on_a_sim_run(monkeypatch, tmp_path):
    body = _infant_body()
    worker = _Worker()
    o = _run(
        monkeypatch,
        tmp_path,
        sim_percepts=[_audio(0.7, salience=0.95, novelty=0.95)],
        body=body,
        worker=worker,
        no_memory_context=True,
    )
    line = "A loud, sudden sound made you orient toward it (it was at azimuth +0.70)."
    # Any audio percept world-sets the sensor; the reflex then world-sets the oriented value (the default
    # profile turns to face the sound: 0.0).
    assert o.world_sets == [0.7, 0.0] and _azimuth(body) == 0.0
    assert o.state.data["_last_audio_orient_az"] == 0.0
    [(_, msg, data, _)] = [s for s in o.sim_logs if s[0] == "REACTION"]
    assert msg == "orienting reflex: turned toward a loud, sudden sound (was +0.70, now +0.00)"
    assert data["reflex"] is True and data["escalated"] is True
    # The reflex escalates: an audio-only pass with no memory context is minted a minimal one (B1).
    [ctx] = worker.contexts
    assert ctx.auto_sense_context == line
    assert ctx.cli_inputs == []


def test_the_reflex_tier_never_fires_on_a_non_sim_run(monkeypatch, tmp_path):
    body = _infant_body()
    o = _run(monkeypatch, tmp_path, obs=[{}], percepts={0: _audio(0.7, salience=0.95, novelty=0.95)}, body=body)
    assert o.world_sets == [0.7]  # the percept's own world-set only: no modeled turn on a live-shaped run
    assert [s for s in o.sim_logs if s[0] == "REACTION"] == []
    assert o.state.data["_last_audio_orient_az"] == 0.7  # the deliberative tier instead


def test_the_reflex_tier_needs_a_body(monkeypatch, tmp_path):
    o = _run(monkeypatch, tmp_path, sim_percepts=[_audio(0.7, salience=0.95, novelty=0.95)])
    assert [s for s in o.sim_logs if s[0] == "REACTION"] == []
    assert o.state.data["_last_audio_orient_az"] == 0.7


def test_the_deliberative_tier_folds_logs_and_advances_the_change_gate(monkeypatch, tmp_path):
    body = _infant_body()
    o = _run(monkeypatch, tmp_path, obs=[{}], percepts={0: _audio(-0.6, salience=0.3)}, body=body)
    assert o.state.data["_last_audio_orient_az"] == -0.6
    [(_, msg, data, _)] = [s for s in o.sim_logs if s[0] == "PERCEPTION"]
    assert msg == "audio-orient: You hear a sound well to your left (azimuth -0.60)."
    assert data["reflex"] is False and data["escalated"] is False
    assert data["salience"] == 0.3


def test_the_change_gate_suppresses_an_unchanged_direction_on_the_next_pass(monkeypatch, tmp_path):
    worker = _Worker()
    o = _run(
        monkeypatch,
        tmp_path,
        obs=[{"cli_input": "one"}, {"cli_input": "two"}, {"cli_input": "three"}],
        percepts={0: _audio(-0.6), 1: _audio(-0.55), 2: _audio(-0.2)},
        steps=3,
        worker=worker,
    )
    assert [c.auto_sense_context for c in worker.contexts] == [
        "You hear a sound well to your left (azimuth -0.60).",
        "",  # moved 0.05 < 0.15: suppressed
        "You hear a sound slightly to your left (azimuth -0.20).",
    ]
    assert o.state.data["_last_audio_orient_az"] == -0.2


def test_the_change_gate_stores_the_clamped_azimuth(monkeypatch, tmp_path):
    worker = _Worker()
    o = _run(
        monkeypatch,
        tmp_path,
        obs=[{"cli_input": "one"}, {"cli_input": "two"}],
        percepts={0: _audio(1.7), 1: _audio(0.9)},
        steps=2,
        worker=worker,
    )
    # Stored clamped (1.0), so 0.9 is within the gate's 0.15 and is suppressed; stored raw (1.7) it would not be.
    assert o.state.data["_last_audio_orient_az"] == 1.0
    assert [c.auto_sense_context for c in worker.contexts] == [
        "You hear a sound well to your right (azimuth +1.00).",
        "",
    ]


def test_the_audio_line_is_appended_after_auto_sense(monkeypatch, tmp_path):
    ctx, _ = _auto_sense_submitted(
        monkeypatch, tmp_path, _AutoFire([], "scan", out="[SCENE] a hall"), percepts={0: _audio(0.05)}
    )
    assert ctx.auto_sense_context == "[SCENE] a hall\nYou hear a sound directly ahead of you (centered, azimuth 0.00)."


def test_the_reflex_line_is_appended_after_auto_sense(monkeypatch, tmp_path):
    worker = _Worker()
    _run(
        monkeypatch,
        tmp_path,
        sim_percepts=[_audio(-0.8, salience=0.95, novelty=0.95)],
        executor=_executor(_AutoFire([], "scan", out="[SCENE] a hall")),
        body=_infant_body(),
        worker=worker,
    )
    # A sound percept carries no text to the observation, so auto-sense does not fire on it: the reflex line
    # stands alone.
    [ctx] = worker.contexts
    assert ctx.auto_sense_context == "A loud, sudden sound made you orient toward it (it was at azimuth -0.80)."


def test_the_reflex_line_is_appended_after_auto_sense_when_the_sound_carries_text(monkeypatch, tmp_path):
    worker = _Worker()
    bang = _audio(-0.8, salience=0.95, novelty=0.95)
    bang.transcript_chunk = "a bang"  # a sim percept's transcript reaches the observation; its sound still orients
    _run(
        monkeypatch,
        tmp_path,
        sim_percepts=[bang],
        executor=_executor(_AutoFire([], "scan", out="[SCENE] a hall")),
        body=_infant_body(),
        worker=worker,
    )
    [ctx] = worker.contexts
    assert ctx.auto_sense_context == (
        "[SCENE] a hall\nA loud, sudden sound made you orient toward it (it was at azimuth -0.80)."
    )


@pytest.mark.parametrize(("salience", "submits"), [(0.6, True), (0.5, False)])
def test_b1_an_escalating_audio_only_pass_is_submitted_with_a_minimal_context(monkeypatch, tmp_path, salience, submits):
    worker = _Worker()
    _run(
        monkeypatch,
        tmp_path,
        obs=[{}],
        percepts={0: _audio(0.6, salience=salience)},
        worker=worker,
        no_memory_context=True,
    )
    if submits:
        [ctx] = worker.contexts
        assert ctx.auto_sense_context == "You hear a sound well to your right (azimuth +0.60)."
        assert ctx.mode == "active"  # the loop's minimal context carries the operational mode
        assert ctx.cli_inputs == []
    else:
        assert worker.contexts == []


def test_b1_an_escalating_audio_only_pass_with_a_memory_context_is_submitted(monkeypatch, tmp_path):
    worker = _Worker()
    _run(monkeypatch, tmp_path, obs=[{}], percepts={0: _audio(0.6, salience=0.6)}, worker=worker)
    [ctx] = worker.contexts
    assert ctx.auto_sense_context == "You hear a sound well to your right (azimuth +0.60)."
    assert ctx.mode == "observe"  # memory's own context, not the minimal one


def test_b1_an_unchanged_direction_does_not_escalate_again(monkeypatch, tmp_path):
    worker = _Worker()
    _run(
        monkeypatch,
        tmp_path,
        obs=[{}, {}],
        percepts={0: _audio(0.6, salience=0.6), 1: _audio(0.6, salience=0.6)},
        steps=2,
        worker=worker,
        no_memory_context=True,
    )
    assert len(worker.contexts) == 1


def test_b1_a_sleeping_pass_is_not_submitted(monkeypatch, tmp_path):
    worker = _Worker()
    o = _run(
        monkeypatch,
        tmp_path,
        obs=[{}],
        percepts={0: _audio(0.6, salience=0.6)},
        worker=worker,
        no_memory_context=True,
        state_data={"processing_state": "sleep"},
    )
    assert worker.contexts == []
    assert o.state.data["_last_audio_orient_az"] == 0.6  # perceived, just not submitted


def test_a_raise_inside_section_1_16_is_reported_and_the_loop_goes_on(monkeypatch, tmp_path, caplog):
    import maxim.embodiment.audio_localization as audio_localization

    def _boom(_emb: Any) -> Any:
        raise RuntimeError("profile broke")

    monkeypatch.setattr(audio_localization, "resolve_orienting_profile", _boom)
    with caplog.at_level(logging.DEBUG, logger="maxim"):
        o = _run(monkeypatch, tmp_path, obs=[{}, {}], percepts={0: _audio(0.6), 1: _audio(0.1)}, steps=2)
    assert _swallowed(caplog, "audio_orientation") == [
        "Swallowed RuntimeError in audio_orientation: profile broke [step=0]",
        "Swallowed RuntimeError in audio_orientation: profile broke [step=1]",
    ]
    assert _kinds(o.ev).count("observe") == 2


# ── the sim-log guards and a None tool ───────────────────────────────────────


def test_a_none_in_the_auto_fire_roster_is_skipped(monkeypatch, tmp_path):
    tool = _AutoFire([], "scan", out="after the gap")

    class _Gappy(ToolRegistry):
        def get_auto_fire_tools(self) -> Any:  # type: ignore[override]
            return [None, tool]

    ctx, _ = _auto_sense_submitted(monkeypatch, tmp_path, registry=_Gappy())
    assert tool.kwargs == [{}] and ctx.auto_sense_context == "after the gap"


def _stage1_reports(caplog: Any, exc: str) -> int:
    return sum(
        1
        for r in caplog.records
        if getattr(r, "event", None) == "swallowed_exception" and (getattr(r, "data", None) or {}).get("exc") == exc
    )


def test_a_raising_sim_log_is_reported_and_perception_still_reaches_the_submission(monkeypatch, tmp_path, caplog):
    worker = _Worker()
    with caplog.at_level(logging.DEBUG, logger="maxim"):
        _run(
            monkeypatch,
            tmp_path,
            obs=[{}, {"cli_input": "x"}],
            percepts={1: _audio(-0.6)},
            steps=2,
            executor=_executor(_AutoFire([], "scan", out="[SCENE] a hall")),
            worker=worker,
            imagination_trigger=_Trigger([]),
            sim_log_raises=("SEM_TRACE", "PERCEPTION", "REACTION"),
        )
    # §1.1's SEM_TRACE (pass 0), §1.15's and §1.16's PERCEPTION lines (pass 1): three Stage-1 reports.
    assert _stage1_reports(caplog, "sim log down") == 3
    assert (
        worker.contexts[-1].auto_sense_context == "[SCENE] a hall\nYou hear a sound well to your left (azimuth -0.60)."
    )


def test_a_raising_sim_log_in_the_reflex_tier_is_reported_and_the_reflex_still_escalates(monkeypatch, tmp_path, caplog):
    worker = _Worker()
    with caplog.at_level(logging.DEBUG, logger="maxim"):
        o = _run(
            monkeypatch,
            tmp_path,
            sim_percepts=[_audio(0.7, salience=0.95, novelty=0.95)],
            body=_infant_body(),
            worker=worker,
            no_memory_context=True,
            sim_log_raises=("REACTION",),
        )
    assert _stage1_reports(caplog, "sim log down") == 1
    assert o.state.data["_last_audio_orient_az"] == 0.0
    [ctx] = worker.contexts
    assert ctx.auto_sense_context == "A loud, sudden sound made you orient toward it (it was at azimuth +0.70)."
