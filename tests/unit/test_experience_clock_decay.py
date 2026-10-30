"""Experience-clock decay (``docs/plans/experience_clock_decay.md``, #1361): the strict red gates, landed in S0.

Each gate below is ``xfail(strict=True, raises=AssertionError)`` and fails TODAY for the reason the plan
states; the slice named in its ``reason`` flips it IN PLACE (never removed or re-pointed, CLAUDE.md "a red
gate that does not flip is data"). ``raises=AssertionError`` keeps a gate honest in both directions: a
harness fault (a ``RuntimeError``, an ``AttributeError`` from an API the slice has not built yet) shows as
a FAILURE, not as the expected red. Flipped so far: G1 (S1, 2026-10-18).

The loop gates drive the REAL ``run_agentic_loop`` (``_drive``): the decay a slice moves lives in the
loop's composition (§8.5 today, ``_loop_live_tick`` after S3a), so a hand-composed sequence would test the
wrong thing. Observation is location-independent, as in ``test_loop_gates_characterization.py``: a pass is
seen through the loop's own ``loop_iteration`` event, credit is written into the real NAc at fixed passes,
and the assertion reads the NAc's maps at the end. Nothing patches ``agent_loop``, ``loop_gates`` or NAc's
decay methods.

Every loop gate runs on a hub WITHOUT a Hippocampus (``_Hub``), so it also requires S3a's D1 (the driver
computes dt with no Hippocampus): today the driver is inert there, which is exactly the silent zero-dt
path the plan's Risk 5 names.
"""

from __future__ import annotations

import math
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest

from maxim.decisions.nac import NAc, NACConfig

# One learning subject per run: decay pacing is a property of the clock, not of agent scoping (the per-agent
# keying is exercised by tests/multi_agent_fixtures.py users, which S3a's Impact list already names).
pytestmark = pytest.mark.single_agent_only

AGENT = "xclock_aut"
NODE = "node:xclock"
CLUSTER = "cluster:xclock"
TOOL = "tool:xclock"
GOAL = "goal:xclock"
ENTITY = "entity:xclock"
IDLE = 0.0007  # a distinctive idle_sleep_s; the harness never really sleeps on the loop thread


# ── the NAc's decaying state, as one comparable value ────────────────────────


def _maps(nac: NAc) -> dict[str, dict[Any, float]]:
    """Every map the plan's ``advance_experience`` decays (§(b)), plus fear (which must NOT decay)."""
    return {
        "eligibility": dict(nac._eligibility),
        "reward_bias": dict(nac._reward_bias),
        "goal_reward_bias": dict(nac._goal_reward_bias),
        "cluster_reward_bias": dict(nac._cluster_reward_bias),
        "percept_valences": dict(nac._percept_valences),
        "cluster_fear": dict(nac._cluster_fear),
    }


def _persisted(nac: NAc) -> dict[str, dict[Any, float]]:
    """The persisted maps (eligibility traces are session-scoped and never saved)."""
    maps = _maps(nac)
    del maps["eligibility"]
    return maps


def _credit(nac: NAc, scale: float = 1.0) -> None:
    """One identical credit write on every decaying surface, through NAc's public writers."""
    nac.update_eligibility(AGENT, NODE, 0.8 * scale)
    nac.credit_node(AGENT, NODE, 1.0 * scale)
    nac.credit_goal(GOAL, 1.0 * scale)
    nac.update_cluster_reward(AGENT, CLUSTER, TOOL, 0.6 * scale)
    nac.record_percept_valence(ENTITY, "burn", -0.5 * scale, agent_id=AGENT)


def _nonvacuous(maps: dict[str, dict[Any, float]], *names: str) -> None:
    """A gate must compare something: an arm whose named maps are empty would make two empty arms compare
    equal and XPASS for the wrong reason (e.g. a units bug that prunes everything). A harness fault, not red."""
    for name in names:
        if not maps[name]:
            raise HarnessFault(f"{name} is empty in the reference arm; the comparison would be vacuous")


def _assert_same_maps(a: dict[str, dict[Any, float]], b: dict[str, dict[Any, float]], *, rel: float = 1e-12) -> None:
    for name in a:
        assert set(a[name]) == set(b[name]), f"{name}: key sets differ {sorted(a[name])} vs {sorted(b[name])}"
        for key in a[name]:
            assert math.isclose(a[name][key], b[name][key], rel_tol=rel, abs_tol=0.0), (
                f"{name}[{key}]: {a[name][key]!r} vs {b[name][key]!r}"
            )


# ── the loop harness ─────────────────────────────────────────────────────────


class HarnessFault(RuntimeError):
    """The harness itself went wrong (not the gate's assertion): reported as a FAILURE, never as xfail."""


class _Hub:
    """The MemoryHub members the loop reads, carrying a real NAc and NO Hippocampus (module docstring)."""

    agent_id = AGENT
    ec = None
    hippocampus = None

    def __init__(self, nac: NAc) -> None:
        from maxim.runtime.substrate_proposal import NO_SITUATION_CUE

        self.nac = nac
        self.situation_cue = NO_SITUATION_CUE
        self._ticket: SimpleNamespace | None = None
        self._generation = 0

    # #1438 PR 0: the session API is owner tickets; the fake asserts a session was opened before it is ended.
    def on_session_start(self, *, owner: Any, live: bool = False, deadline_s: float | None = None) -> SimpleNamespace:
        del live, deadline_s
        if self._ticket is None or self._ticket.owner is not owner:
            self._generation += 1
            self._ticket = SimpleNamespace(start_stats={}, generation=self._generation, owner=owner)
        return self._ticket

    def _end(self, ticket: Any) -> dict[str, Any]:
        assert self._generation > 0, "the session was ended without ever being opened"
        if ticket is not self._ticket:
            return {}
        self._ticket = None
        return {}

    def on_session_end(self, ticket: Any, *, deadline: Any = None) -> dict[str, Any]:
        return self._end(ticket)

    def on_session_end_lightweight(self, ticket: Any, *, deadline: Any = None) -> dict[str, Any]:
        return self._end(ticket)

    def release_session_claim(self, ticket: Any) -> None:
        del ticket

    def record_plan_outcome(self, **kw: Any) -> None:
        return None


class _Agent:
    name = "xclock-probe"

    def propose_intent(self, state: Any, memory: Any) -> None:
        return None


class FakeClock:
    """A world clock the test owns: ``t`` seconds since an origin, read as monotonic and (optionally) wall."""

    def __init__(self) -> None:
        self.t = 0.0

    def monotonic(self) -> float:
        return 1000.0 + self.t

    def time(self) -> float:
        return 1_800_000_000.0 + self.t


class AwakeConversation:
    """A turn-based world held awake on every poll.

    The real ``ConversationalSource`` (its turn declaration is what the driver reads) with ``has_pending``
    always true: it stands in for the orchestrator AUT's ``_submitted_recently`` hold, which keeps a text-sim
    loop polling ~2 Hz between turns (plan §"What a tick is today"). A new turn is ``mark_turn()``, the
    bridge's own turn marker for a turn that injects no text.
    """

    def __new__(cls) -> Any:
        from maxim.simulation.conversational_source import ConversationalSource

        class _Awake(ConversationalSource):
            def has_pending(self) -> bool:  # the 120 s submit hold, per poll
                return True

        return _Awake()


class AwakeRealtime:
    """A real-time world (no turn declaration) that wakes every pass, like a 10-30 Hz sensor loop."""

    name = "xclock-sensor"

    def next_percept(self) -> None:
        return None

    def is_exhausted(self) -> bool:
        return False

    def has_pending(self) -> bool:
        return True


def _drive(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    nac: NAc,
    passes: int,
    percept_source: Any,
    hooks: dict[int, Callable[[SimpleNamespace], None]] | None = None,
    eventful: frozenset[int] = frozenset(),
    clock: FakeClock | None = None,
    patch_wall: bool = False,
    body: Any = None,
    aut_mode: str = "llm-primary",
    hub: Any = None,
) -> SimpleNamespace:
    """Run the real loop for ``passes`` passes. ``hooks[k](run)`` runs at pass k's ``loop_iteration`` event
    (before the pre-tick gate). ``eventful``: passes whose confirmation handler consumes the input and
    ``continue``s (the plan's "eventful pass": it reaches §1, so a fixture step advances, but not §8.5).
    ``clock``: the experience driver reads it as its monotonic source (the realtime gates); with
    ``patch_wall`` the global ``time.time``/``time.monotonic`` read it too (the gates where wall time is the
    confound). Returns the run's namespace (``ctrl``, ``state``, ``passes_seen``).

    Every patch is undone when the run returns (a fresh ``MonkeyPatch.context()``): a gate drives two arms
    in one test, and a second arm's wrappers must wrap the REAL methods, not the first arm's."""
    del monkeypatch  # each run patches in its own context, below
    with pytest.MonkeyPatch.context() as mp:
        return _drive_in(
            mp,
            tmp_path,
            nac=nac,
            passes=passes,
            percept_source=percept_source,
            hooks=hooks,
            eventful=eventful,
            clock=clock,
            patch_wall=patch_wall,
            body=body,
            aut_mode=aut_mode,
            hub=hub,
        )


def _drive_in(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    nac: NAc,
    passes: int,
    percept_source: Any,
    hooks: dict[int, Callable[[SimpleNamespace], None]] | None,
    eventful: frozenset[int],
    clock: FakeClock | None,
    patch_wall: bool,
    body: Any,
    aut_mode: str,
    hub: Any,
) -> SimpleNamespace:
    import maxim.simulation.sim_logger as sim_logger
    import maxim.utils.structured_logging as sl
    from maxim.agents.autonomy import AutonomyController, AutonomyLevel
    from maxim.environment.filesystem_env import FileSystemEnv
    from maxim.runtime import agent_loop as AL
    from maxim.runtime.bootstrap import build_decision_engine, build_executor, build_memory
    from maxim.runtime.experience_time import ExperienceClockDriver
    from maxim.runtime.loop_controller import LoopController
    from maxim.runtime.state import RuntimeState
    from maxim.tools.registry import ToolRegistry

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sim_logger, "_interactive_mode", sim_logger.InteractiveMode.OFF)
    hooks = dict(hooks or {})
    run = SimpleNamespace(ctrl=None, state=None, passes_seen=[], step=-1, controller=None)

    real_init = LoopController.__init__

    def _init(self: Any, *a: Any, **k: Any) -> None:
        real_init(self, *a, **k)
        self.llm_submit_interval = 1e12  # no worker; never submit on a cadence
        run.ctrl = self

    monkeypatch.setattr(LoopController, "__init__", _init)

    real_confirm = LoopController.handle_confirmation

    def _confirm(self: Any, cli_text: Any, *a: Any, **k: Any) -> bool:
        if run.step in eventful:
            return True  # the confirmation consumed this pass's input: the loop `continue`s
        return real_confirm(self, cli_text, *a, **k)

    monkeypatch.setattr(LoopController, "handle_confirmation", _confirm)

    if clock is not None:
        real_driver_init = ExperienceClockDriver.__init__

        def _driver_init(self: Any, *a: Any, **k: Any) -> None:
            k.setdefault("monotonic", clock.monotonic)
            real_driver_init(self, *a, **k)

        monkeypatch.setattr(ExperienceClockDriver, "__init__", _driver_init)

    real_buffer = sl.get_abstraction_buffer()

    class _Buffer:
        def append(self, record: Any) -> None:
            if record.source == "agent_loop" and record.event == "loop_iteration":
                run.step = record.data["step"]
                run.passes_seen.append(run.step)
                if run.step in eventful:
                    run.state.data["pending_cli_input"] = "yes"
                hook = hooks.get(run.step)
                if hook is not None:
                    hook(run)
            real_buffer.append(record)

        def __getattr__(self, name: str) -> Any:
            return getattr(real_buffer, name)

    monkeypatch.setattr(sl, "get_abstraction_buffer", lambda: _Buffer())

    loop_thread = threading.get_ident()
    real_sleep = time.sleep

    def _sleep(dt: float) -> None:
        if threading.get_ident() == loop_thread:
            return  # the loop's own pacing never sleeps here; the world clock is the test's
        real_sleep(min(dt, 0.001))

    monkeypatch.setattr(time, "sleep", _sleep)
    if patch_wall:
        if clock is None:
            raise HarnessFault("patch_wall needs a clock")
        from tests.unit._loop_harness import preimport_loop_graph

        preimport_loop_graph()  # nothing may bind the fake clock at import (checked after the run)
    before = set(sys.modules)  # snapshot BEFORE the fake clock goes in
    if patch_wall:
        monkeypatch.setattr(time, "time", clock.time)
        monkeypatch.setattr(time, "monotonic", clock.monotonic)

    executor = build_executor(ToolRegistry(), pain_bus=None, permissions=None)
    if body is not None:
        executor.embodiment = body
    state = RuntimeState()
    state.data["mode"] = "active"
    run.state = state
    controller = AutonomyController(initial_level=AutonomyLevel.AUTONOMOUS)
    run.controller = controller
    workspace = tmp_path / "ws"
    workspace.mkdir(exist_ok=True)
    AL.run_agentic_loop(
        _Agent(),
        FileSystemEnv(str(workspace)),
        state,
        build_memory(),
        build_decision_engine(),
        executor,
        autonomy_controller=controller,
        memory_hub=hub if hub is not None else _Hub(nac),
        percept_source=percept_source,
        aut_mode=aut_mode,
        stop_event=threading.Event(),
        max_steps=passes,
        target_hz=1e6,
        idle_sleep_s=IDLE,
    )
    if patch_wall:
        late = sorted(m for m in set(sys.modules) - before if m.startswith("maxim"))
        if late:  # a module imported under the fake clock binds it forever (tests/unit/_loop_harness.py)
            raise HarnessFault(f"maxim modules first imported inside the fake-clock window: {late}")
    if run.passes_seen != list(range(passes)):
        raise HarnessFault(f"the loop ran passes {run.passes_seen}, expected 0..{passes - 1}")
    return run


def _fixture_source(tmp_path: Path) -> Any:
    """A real step-based ``ScenarioSource`` (``timing: step_based``, every fixture's mode): one percept far
    beyond the run, so the world is never exhausted and no percept arrives; every pass wakes (the source has
    no ``has_pending``) and its §1 ``next_observation`` advances the step (``SimulationAdapter``)."""
    from maxim.simulation.scenario_source import ScenarioSource

    path = tmp_path / "xclock_fixture.yaml"
    path.write_text(
        "name: xclock_fixture\ntiming: step_based\npercepts:\n"
        "  - at: 100000\n    source: cli\n    cli_input: never reached\nexpectations: []\n"
    )
    return ScenarioSource(path)


def _gate(slice_: str, why: str) -> pytest.MarkDecorator:
    return pytest.mark.xfail(strict=True, raises=AssertionError, reason=f"flips at {slice_}: {why}")


# ── G1 (S1): no NAc forgetting while switched off ────────────────────────────


# FLIPPED at S1 (2026-10-18): the S0 xfail is removed in place. Plan §(f): under offline_wall_decay=True every
# map that decays today, _cluster_fear included, decays as before; the default (False) decays nothing offline.
@pytest.mark.single_agent_only
def test_g1_a_week_switched_off_forgets_nothing_unless_opted_in(tmp_path, monkeypatch):
    import maxim.decisions.nac as nac_module

    path = str(tmp_path / "nac.json")
    saved = NAc(NACConfig(persistence_path=path))
    _credit(saved)
    saved.record_cluster_fear(AGENT, CLUSTER, "drive:health", 0.8)
    assert saved._cluster_fear, "the fear write must land, or the fear half of G1 is vacuous"
    before = _persisted(saved)
    saved.save()

    week = 7 * 86400.0
    real_time = nac_module.time.time
    before_modules = set(sys.modules)  # snapshot BEFORE the patched clock goes in
    monkeypatch.setattr(nac_module.time, "time", lambda: real_time() + week)

    loaded = NAc(NACConfig(persistence_path=path))
    loaded.load()  # the agent-restart default
    late = sorted(m for m in set(sys.modules) - before_modules if m.startswith("maxim"))
    if late:  # a module imported under the patched clock binds it (tests/unit/_loop_harness.py)
        raise HarnessFault(f"maxim modules first imported inside the patched-clock window: {late}")
    assert _persisted(loaded) == before, "a week switched off changed the NAc's learned maps"

    # The opt-in arm (Dormant, NACConfig.offline_wall_decay): the old behaviour, kept reachable.
    opted = NAc(NACConfig(persistence_path=path, offline_wall_decay=True))
    opted.load()
    assert opted._reward_bias[(AGENT, NODE)] < before["reward_bias"][(AGENT, NODE)]
    assert opted._cluster_fear != before["cluster_fear"]


# ── G2a (S2 + S3a): fixture decay is step-indexed, whatever passes were eventful ──


@_gate(
    "S3a (made measurable by S2's ScenarioSource step declaration)",
    "§8.5 decay is skipped on eventful passes that still advance the fixture step, so the same step K "
    "carries a different number of decays",
)
def test_g2a_fixture_decay_is_step_indexed_not_pass_indexed(monkeypatch, tmp_path):
    def arm(eventful: frozenset[int]) -> tuple[dict, int]:
        nac = NAc(NACConfig())
        source = _fixture_source(tmp_path)
        hooks = {0: lambda r: _credit(nac), 6: lambda r: _credit(nac, 0.5)}
        _drive(monkeypatch, tmp_path, nac=nac, passes=12, percept_source=source, hooks=hooks, eventful=eventful)
        return _maps(nac), source._current_step

    calm, calm_step = arm(frozenset())
    busy, busy_step = arm(frozenset({2, 3, 4, 8, 9}))  # confirmations consume these passes' input
    # Precondition, not the gate: the eventful passes DID advance the world (§1 ran), so both arms end at
    # the same step K. This is what makes today's mismatch a decay/step misalignment (plan §(a′)).
    if calm_step != busy_step or calm_step != 12:
        raise HarnessFault(f"fixture steps differ: {calm_step} vs {busy_step}")
    _nonvacuous(calm, "eligibility", "reward_bias", "cluster_reward_bias", "percept_valences")
    _assert_same_maps(calm, busy)


# ── G2 (S3a): text-sim decay is independent of LLM latency ───────────────────


@_gate("S3a", "decay runs once per §8.5 pass, so a slow model (more polls per turn) forgets faster")
def test_g2_turn_world_decay_is_latency_independent(monkeypatch, tmp_path):
    """Identical credit is injected directly (no ``distribute_reward``), so the SCN anchor fallback is not in
    the path and the gate tests decay alone (the plan's review fix). Poll at 20 Hz on a fake wall clock:
    latency 0.05 s = 1 pass per turn, 1.0 s = 20 passes per turn; six turns each."""

    def arm(passes_per_turn: int) -> dict:
        nac = NAc(NACConfig())
        source = AwakeConversation()
        clock = FakeClock()

        def tick(k: int) -> Callable[[SimpleNamespace], None]:
            def hook(_r: SimpleNamespace) -> None:
                clock.t = k * 0.05
                if k % passes_per_turn == 0:
                    source.mark_turn()
                    turn = k // passes_per_turn
                    if turn in (0, 3):
                        _credit(nac)

            return hook

        passes = 6 * passes_per_turn
        _drive(
            monkeypatch,
            tmp_path,
            nac=nac,
            passes=passes,
            percept_source=source,
            hooks={k: tick(k) for k in range(passes)},
            clock=clock,
            patch_wall=True,
        )
        if source.experience_turns() != 6:
            raise HarnessFault(f"{source.experience_turns()} turns")
        return _maps(nac)

    fast = arm(1)
    _nonvacuous(fast, "reward_bias", "goal_reward_bias", "cluster_reward_bias", "percept_valences")
    _assert_same_maps(fast, arm(20))


# ── G3 (S3a): real-time decay is independent of the poll rate ────────────────


@_gate("S3a", "decay runs once per §8.5 pass, so 20 Hz polling forgets 10x faster than 2 Hz over the same 60 s")
def test_g3_realtime_decay_is_poll_rate_independent(monkeypatch, tmp_path):
    """Same 60 s of world time (a fake monotonic the driver reads), polled at 2 Hz and at 20 Hz. Same key
    set (no value in the fixture sits within 1e-9 of a prune floor) and values within 1e-12 relative."""

    def arm(hz: int) -> dict:
        nac = NAc(NACConfig())
        clock = FakeClock()
        period = 1.0 / hz

        def tick(k: int) -> Callable[[SimpleNamespace], None]:
            def hook(_r: SimpleNamespace) -> None:
                clock.t = k * period
                if k == 0:
                    _credit(nac)

            return hook

        passes = 60 * hz + 1  # pass 0 starts the count; the last pass lands on t = 60 s
        _drive(
            monkeypatch,
            tmp_path,
            nac=nac,
            passes=passes,
            percept_source=AwakeRealtime(),
            hooks={k: tick(k) for k in range(passes)},
            clock=clock,
        )
        return _maps(nac)

    slow_poll = arm(2)
    # reward_bias, not eligibility: after a correct S3a the trace is legitimately pruned (tau 4.75 s over 60 s)
    _nonvacuous(slow_poll, "reward_bias", "goal_reward_bias", "cluster_reward_bias", "percept_valences")
    _assert_same_maps(slow_poll, arm(20))


# ── G2b (S3b): eligibility, anchors and pending events age on ONE clock ──────


@_gate(
    "S3b",
    "temporal anchors and pending events age on the 300 s WALL window while the trace decays per pass, so a "
    "slow model turns trace credit into anchor credit or none, and drops the causal link",
)
@pytest.mark.single_agent_only
def test_g2b_trace_and_anchor_credit_split_is_poll_rate_independent(monkeypatch, tmp_path):
    """Ten turns of the same credit sequence: an event + trace + anchor at turn 0, the outcome and the reward
    after turn 9. Fast arm: 1 pass per turn at 0.5 s wall (5 s of wall); slow arm: 40 passes per turn at
    1.0 s wall (400 s, past the 300 s window). Same experience (10 turns) in both. Anchor phase SIMILARITY is
    pinned to 1.0: it stays on host time by design (plan §(c)); only age and expiry move."""
    from maxim.decisions.causal_link import Valence
    from maxim.time.temporal_signature import TemporalSignature

    monkeypatch.setattr(TemporalSignature, "similarity", lambda self, other: 1.0)

    def arm(passes_per_turn: int, wall_per_pass: float) -> tuple:
        nac = NAc(NACConfig())
        source = AwakeConversation()
        clock = FakeClock()
        out: dict[str, Any] = {}
        last = 10 * passes_per_turn

        def tick(k: int) -> Callable[[SimpleNamespace], None]:
            def hook(_r: SimpleNamespace) -> None:
                clock.t = k * wall_per_pass
                if k < last and k % passes_per_turn == 0:
                    source.mark_turn()
                if k == 0:
                    nac.record_event("tool", TOOL, context={})
                    nac.update_eligibility(AGENT, NODE, 0.8, temporal_sig=TemporalSignature.now())
                if k == last:
                    traced = {key for key in nac._eligibility if key[0] == AGENT}
                    nac.record_outcome("tool", TOOL, Valence.POSITIVE)
                    credited = nac.distribute_reward(AGENT, 1.0)
                    split = {"trace": 0.0, "anchor": 0.0}
                    for node, credit in credited:
                        split["trace" if (AGENT, node) in traced else "anchor"] += credit
                    out["split"] = split

            return hook

        _drive(
            monkeypatch,
            tmp_path,
            nac=nac,
            passes=last + 1,
            percept_source=source,
            hooks={k: tick(k) for k in range(last + 1)},
            clock=clock,
            patch_wall=True,
        )
        links = sorted((sig, link.outcome_signature) for sig, ls in nac._links.items() for link in ls)
        return links, out["split"], dict(nac._reward_bias)

    fast_links, fast_split, fast_bias = arm(1, 0.5)
    slow_links, slow_split, slow_bias = arm(40, 1.0)
    if not fast_links or not fast_bias or sum(fast_split.values()) <= 0.0:
        raise HarnessFault(f"the fast arm learned nothing ({fast_links}, {fast_split}); the comparison is vacuous")
    assert fast_links == slow_links, f"link sets differ: {fast_links} vs {slow_links}"
    for path in ("trace", "anchor"):
        assert math.isclose(fast_split[path], slow_split[path], rel_tol=1e-12, abs_tol=1e-15), (
            f"credit split differs: {fast_split} vs {slow_split}"
        )
    _assert_same_maps({"reward_bias": fast_bias}, {"reward_bias": slow_bias})


# ── G4 (S4): body drift excludes pause and latency ───────────────────────────


def _infant_body() -> Any:
    from maxim.embodiment.body import Embodiment
    from maxim.embodiment.component_registry import ComponentRegistry

    return Embodiment(ComponentRegistry().instantiate("bodies/infant_humanoid"))


def _drives(body: Any) -> dict[str, float]:
    return {
        f"{ent.full_path}:{name}": float(value)
        for ent in body.root.walk()
        for name, value in ent.vital_metrics.items()
        if name in ent.drive_specs
    }


@_gate("S4", "Embodiment.evaluate_failures drifts by time.time() - _last_poll, which an operator pause does not reset")
def test_g4_a_thirty_second_pause_adds_zero_drift(monkeypatch, tmp_path):
    """llm-primary (the loop's live tick drifts the body) on a fake wall + monotonic clock. The paused arm
    halts the autonomy controller at pass 10 and resumes it 30 s later at pass 13; the control arm halts and
    resumes in the same pass (zero paused time, the same autonomy transitions). Both live the same world
    time outside the pause, so equal drift is exactly "the pause added none"."""

    def world(k: int) -> float:
        # One second per live pass; the world does not move across passes 11-12 (the paused arm's
        # pause passes), so both arms live the same 21 s of world time.
        return float(k) if k <= 10 else (10.0 if k <= 12 else k - 3.0)

    def arm(pause_s: float) -> dict[str, float]:
        body = _infant_body()
        clock = FakeClock()
        hooks: dict[int, Callable[[SimpleNamespace], None]] = {}
        for k in range(25):

            def hook(r: SimpleNamespace, k: int = k) -> None:
                clock.t = world(k) + (pause_s if k >= 13 else 0.0)
                if k == 10:
                    r.controller.emergency_halt("xclock pause")
                    if pause_s == 0.0:
                        r.controller.resume()
                if k == 13 and pause_s:
                    r.controller.resume()

            hooks[k] = hook
        _drive(
            monkeypatch,
            tmp_path,
            nac=NAc(NACConfig()),
            passes=25,
            percept_source=AwakeRealtime(),
            hooks=hooks,
            clock=clock,
            patch_wall=True,
            body=body,
        )
        return _drives(body)

    control, paused = arm(0.0), arm(30.0)
    if control == _drives(_infant_body()):
        raise HarnessFault("the control arm never drifted; the gate would be vacuous")
    for key in control:
        assert math.isclose(control[key], paused[key], rel_tol=0.0, abs_tol=1e-9), (
            f"{key}: {control[key]} vs {paused[key]} after a 30 s pause"
        )


@_gate("S4", "text-sim body drift is wall-clock, so it scales with LLM latency instead of world turns")
def test_g4_text_sim_drift_is_latency_independent(monkeypatch, tmp_path):
    def arm(wall_per_turn: float) -> dict[str, float]:
        body = _infant_body()
        clock = FakeClock()
        source = AwakeConversation()

        def tick(k: int) -> Callable[[SimpleNamespace], None]:
            def hook(_r: SimpleNamespace) -> None:
                clock.t = k * wall_per_turn
                source.mark_turn()

            return hook

        _drive(
            monkeypatch,
            tmp_path,
            nac=NAc(NACConfig()),
            passes=6,
            percept_source=source,
            hooks={k: tick(k) for k in range(6)},
            clock=clock,
            patch_wall=True,
            body=body,
        )
        return _drives(body)

    fast, slow = arm(0.05), arm(1.0)
    if fast == _drives(_infant_body()):
        raise HarnessFault("the fast arm never drifted; the gate would be vacuous")
    for key in fast:
        assert math.isclose(fast[key], slow[key], rel_tol=0.0, abs_tol=1e-9), f"{key}: {fast[key]} vs {slow[key]}"


# ── G5 (S5): link confidence decays by experience, not by session count ──────


@_gate("S5", "MemoryHub session end applies a flat decay_all(0.95), so ten short sessions erode links ~0.63x vs one")
def test_g5_ten_six_turn_sessions_equal_one_sixty_turn_session(monkeypatch, tmp_path):
    from maxim.decisions.causal_link import Valence
    from maxim.runtime.bio_stack import build_bio_stack

    def arm(name: str, sessions: int, turns: int) -> float:
        bio = build_bio_stack(agent_id=AGENT, persistence_dir=str(tmp_path / name))
        nac = bio.nac
        hub = bio.memory_hub
        for s in range(sessions):
            source = AwakeConversation()

            def tick(k: int, s: int = s) -> Callable[[SimpleNamespace], None]:
                def hook(_r: SimpleNamespace) -> None:
                    source.mark_turn()
                    if s == 0 and k == 0:
                        nac.record_event("tool", TOOL, context={})
                        nac.record_outcome("tool", TOOL, Valence.POSITIVE)

                return hook

            _drive(
                monkeypatch,
                tmp_path,
                nac=nac,
                passes=turns,
                percept_source=source,
                hooks={k: tick(k) for k in range(turns)},
                hub=hub,
            )
            # #1438 PR 0: the loop's bio_session ends its own session (with its ticket) on exit; the old explicit
            # ticketless end after it was a no-op. Pin that the loop really closed it, as a harness fault.
            if hub._ownership.state != "CLOSED":
                raise HarnessFault(f"session {s}: the loop left the memory session {hub._ownership.state}")
        [link] = [link for ls in nac._links.values() for link in ls]
        return link.confidence

    many, one = arm("ten", 10, 6), arm("one", 1, 60)
    assert math.isclose(many, one, rel_tol=1e-12), f"10 x 6 turns -> {many}; 1 x 60 turns -> {one}"


# ── S0 measurements (pass, not xfail): what the slices are converting FROM ───


def test_s0_which_builders_give_a_nac_without_a_hippocampus(tmp_path):
    """Plan §(a) D1: the runtimes that would silently lose NAc decay and body drift if S3a only drove them
    through the Hippocampus clock. Builders that reach ``run_agentic_loop`` in production all go through
    ``build_bio_stack`` (always a Hippocampus). The only NAc-without-Hippocampus is the bare ``create_agent``
    path (``maxim.create.agent``/``create_npc_agent``/``maxim.load.agent`` with ``remembers=False,
    learns=True``), and it gets NO MemoryHub at all, so it cannot reach the loop's NAc either (the loop takes
    its NAc only from the hub, ``_loop_bio_handles``). If that changes, D1's list changes: this test says so.

    The ``bare.memory_hub is None`` pin is a DEFECT (#1366: a learns-only agent's hub build raises and is logged as
    corruption). Update this assertion, and plan §(a)'s D1 list, when #1366 lands."""
    from maxim.runtime.agent_factory import AgentConfig, AgentFactory
    from maxim.runtime.bio_stack import build_bio_stack

    bio = build_bio_stack(agent_id="xclock_bio", persistence_dir=str(tmp_path / "bio"))
    assert bio.nac is not None and bio.memory_hub.hippocampus is not None

    factory = AgentFactory(base_data_dir=tmp_path / "agents")
    bare = factory.create_agent(AgentConfig(agent_id="xclock_bare", remembers=False, learns=True))
    try:
        assert bare.nac is not None and bare.hippocampus is None
        # Measured 2026-10-17: building the hub without a Hippocampus raises inside
        # ``_create_memory_hub`` (AttributeError on ``connect_scn``), which is logged as "Corrupt memory_hub
        # state" and leaves the hub None -- a pre-existing defect, recorded in the plan, not fixed in S0.
        assert bare.memory_hub is None
    finally:
        bare.shutdown()

    full = factory.create_full_agent(
        AgentConfig(agent_id="xclock_full", remembers=False, learns=True, with_bio_stack=True)
    )
    try:  # with a bio-stack, remembers=False does NOT drop the Hippocampus: the bio-stack's replaces it
        assert full.nac is not None and full.memory_hub.hippocampus is not None
    finally:
        full.shutdown()


def _count_decays(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Count §8.5 runs today through a class-level spy on one of its six calls (behaviour unchanged)."""
    calls: list[int] = []
    real = NAc.decay_reward_biases

    def _spy(self: NAc) -> int:
        calls.append(1)
        return real(self)

    monkeypatch.setattr(NAc, "decay_reward_biases", _spy)
    return calls


def test_s0_todays_decay_quantum_is_the_non_eventful_full_pass(monkeypatch, tmp_path):
    """The pacing S3a converts FROM, measured on the real loop (characterization; S3a rewrites this test
    when it deletes §8.5). One NAc decay step per full pass that is not eventful; none on an idle pass;
    none on an eventful pass, which still advances a fixture step."""
    calls = _count_decays(monkeypatch)
    source = _fixture_source(tmp_path)
    _drive(monkeypatch, tmp_path, nac=NAc(NACConfig()), passes=12, percept_source=source, eventful=frozenset({2, 3}))
    assert (source._current_step, len(calls)) == (12, 10)

    calls.clear()
    _drive(monkeypatch, tmp_path, nac=NAc(NACConfig()), passes=40, percept_source=AwakeRealtime())
    assert len(calls) == 40  # an awake real-time loop: one decay per poll, so decay/s == poll rate

    class _Quiet(AwakeRealtime):
        def has_pending(self) -> bool:
            return False

    calls.clear()
    _drive(monkeypatch, tmp_path, nac=NAc(NACConfig()), passes=40, percept_source=_Quiet())
    assert len(calls) == 1  # only the first pass is full; idle passes never decay (plan Q2 reverses this)


def _tau_eff(tau_passes: float) -> float:
    return -1.0 / math.log(1.0 - 1.0 / tau_passes)


def test_s0_tau_conversion_inputs():
    """Plan §"Tau conversion": tau_us = tau_eff * m, m = experience_us_per_step = round(1e6 / target_hz) at
    Exp 30's 2 Hz fixture regime (Q1, Q8). Pins today's per-pass taus and the converted world-seconds S3a
    starts from (the plan's worked numbers: 50 -> 49.5 passes -> ~25 s; x0.9 -> 9.49 passes)."""
    cfg = NACConfig()
    m_s = round(1e6 / 2.0) / 1e6
    assert m_s == 0.5
    taus = {
        "reward_bias": cfg.reward_bias_decay_tau,
        "goal_reward_bias": cfg.reward_bias_decay_tau,
        "percept_valences": cfg.percept_valence_decay_tau,
        "cluster_reward_bias": cfg.cluster_reward_bias_decay_tau,
        "exploration_visits": cfg.substrate_explore_decay_tau,
    }
    assert taus == {
        "reward_bias": 50.0,
        "goal_reward_bias": 50.0,
        "percept_valences": 200.0,
        "cluster_reward_bias": 300.0,
        "exploration_visits": 50.0,
    }
    world_s = {name: round(_tau_eff(t) * m_s, 2) for name, t in taus.items()}
    assert world_s == {
        "reward_bias": 24.75,
        "goal_reward_bias": 24.75,
        "percept_valences": 99.75,
        "cluster_reward_bias": 149.75,
        "exploration_visits": 24.75,
    }
    eligibility_tau_eff = -1.0 / math.log(0.9)  # decay_eligibility's default factor
    assert round(eligibility_tau_eff, 2) == 9.49 and round(eligibility_tau_eff * m_s, 2) == 4.75
