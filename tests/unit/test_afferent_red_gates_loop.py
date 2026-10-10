"""GL3.B0 strict red gates (d), (e), (f): where a fast body signal waits today, through the REAL loop.

WHAT GL3.B0 IS. The first stage of the thalamic-relay plan (``docs/plans/thalamic_relay.md`` §6
"GL3.B0"): tests only, inside the decomposition fence. It pins the timing defects §3.4 found by
code-read as strict red gates (``xfail(strict=True)``), so each defect's fix is visible the day it
lands (the gate XPASSes, and strict turns that into a failure until the decorator is removed), and so
none of them is vacuous (each fails today, for the reason it names). This file holds the three gates
that need the loop itself; the census and gates (a)/(b)/(c)/(g) live elsewhere. The latency
characterization (not a gate) is ``test_afferent_latency_characterization.py``, which reuses this
module's driver.

THE GATES, AND WHAT FLIPS EACH
  (d) ``test_tracks_d1_stale_substrate_proposal`` -- #1176 (L1). Substrate-primary sets
      ``ctrl.pending_proposal`` at §6b of pass N and §4 executes it at pass N+1, after the sleep, with
      no re-evaluation. Health drops in that sleep; the pre-damage proposal is executed anyway. Flips
      with the L1 fix or with GL3.B4, whichever #1176 decides. The stale execution is matched by
      object identity AND by (pass N+1, the stale tool), so copying the action dict cannot XPASS it.
  (e) ``test_tracks_d2_turn_gate_skips_pain`` -- #1177 (L2). A denying ``substrate_action_gate``
      skips ``propose_via_substrate`` whole, and with it ``evaluate_failures``: a DRIFT-driven breach
      is not published on the pass a free-running loop publishes it (it is never published while the
      gate denies). Owner decision TR2 (2026-10-09): the budget may delay only ACTION, never
      nociception. Flips with the L2 fix.
  (f) ``test_tracks_d5_reflex_behind_thought_gate`` -- #1178 (L5). The percept reflexes run inside
      ``BioEnrichmentPipeline.enrich``, which runs only if the ThoughtGate passed; a refractory or
      energy-exhausted gate drops a matching percept's reflex. Flips with the L5 fix.

HOW THEY RUN. ``drive_loop`` below is the ``_loop_harness.run_arm`` recipe (which this file may not
edit, so the recipe is re-composed here from the harness's exported pieces): the REAL
``run_agentic_loop``, substrate-primary, on the canonical builders, single-threaded on the harness's
GLOBAL step clock (``_StepClock``), the loop's module graph pre-imported and a first import inside the
window refused (``ImportedInsideWindow``), EC node ids pinned to a counter. Two rigs:
  * ``minecraft`` -- ``build_minecraft_aut`` + ``_loop_kwargs`` against ``_HurtWorld``, a subclass of
    the harness's ``_ScriptedWorld`` whose health can drop at a chosen step-clock time (gate (d), and
    the characterization's two arms).
  * ``body`` -- ``build_bio_stack`` + ``build_executor(entity_ref=...)`` on a drifting cradle body
    (``bodies/infant_humanoid``; the ``_cradle_loop_driver`` composition, with the REAL substrate
    proposer), optionally with the orchestrator AUT's ``ConversationalSource`` and its percept-reflex
    wiring (gates (e) and (f)).
A "pass" is one iteration of the loop (its ``step_num``; idle passes count), observed at
``pre_tick_gate``'s entry; every pain publish, substrate tick and executor call is stamped with the
pass it happened in.

PRECONDITIONS ARE NOT THE GATE. Each gate first checks that its scenario happened as designed (the
proposal was made before the damage, the gate actually rejected, the control run published, ...).
Those checks raise ``ScenarioBroken`` (not an ``AssertionError``), and each gate is
``xfail(strict=True, raises=AssertionError)``, so a broken scenario FAILS the run instead of hiding
as an expected failure. Only the final assertion -- the defect -- is the expected failure.
"""

from __future__ import annotations

import importlib
import os
import sys
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pytest

if str(Path(__file__).resolve().parents[2]) not in sys.path:  # run as a script by the characterization
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.unit._loop_harness import (  # noqa: E402
    ImportedInsideWindow,
    _ScriptedWorld,
    _StepClock,
    assert_this_checkout,
    preimport_loop_graph,
)

MC_AGENT = "afferent_mc"
BODY_AGENT = "afferent_body"
BODY_REF = "bodies/infant_humanoid"
HURT_HEALTH = 8.0  # < 14 hp: inside minecraft_player's health pain band (set_point 20, comfort_band 6)
PROTECTIVE_TOOLS = frozenset({"minecraft_player_flee", "minecraft_player_escape_water"})

# The modules the body rig and the reflex wiring import lazily, beyond ``preimport_loop_graph``'s graph.
_PREIMPORT_EXTRA: tuple[str, ...] = (
    "maxim.agents.llm_types",
    "maxim.memory.encoding",
    "maxim.memory.hippocampus",
    "maxim.runtime.bio_integration",
    "maxim.runtime.tool_dispatch",
    "maxim.runtime.loop_setup",
    "maxim.runtime.loop_gates",
    "maxim.runtime.substrate_proposal",
    "maxim.proprioception.pain",
    "maxim.embodiment.reflex",
    "maxim.embodiment.sem",
    "maxim.simulation.tools",
    "maxim.simulation.conversational_source",
    "maxim.energy.llm_tracker",
    "maxim.agents.working_memory",
    "maxim.runtime.approval",
)


class ScenarioBroken(Exception):
    """The scenario did not happen as designed: NOT the defect a gate pins (module docstring)."""


def require(cond: Any, msg: str) -> None:
    if not cond:
        raise ScenarioBroken(msg)


# ── the worlds ────────────────────────────────────────────────────────────


class _HurtWorld(_ScriptedWorld):
    """``_ScriptedWorld`` whose health reads ``HURT_HEALTH`` from step-clock time ``hurt_after``
    (strictly after it), so a hurt armed at time t is first seen by the sensor sync of the next
    clock advance, i.e. the loop's sleep: "health drops between passes"."""

    def __init__(self, clock: _StepClock, *, submerged: bool, hurt_after: float | None = None) -> None:
        super().__init__(clock, submerged=submerged)
        self.hurt_after = hurt_after

    def latest_state(self) -> dict[str, float]:
        state = super().latest_state()
        if self.hurt_after is not None and self._clock.t > self.hurt_after:
            state["health"] = HURT_HEALTH
        return state


# ── the trace ─────────────────────────────────────────────────────────────


@dataclass
class Trace:
    passes: list[dict[str, Any]] = field(default_factory=list)  # {pass, t}: every loop iteration
    ticks: list[dict[str, Any]] = field(default_factory=list)  # substrate ticks: {tick, pass, t, gated, tool}
    proposals: list[Any] = field(default_factory=list)  # the LLMProposal per tick (None when idle/gated)
    calls: list[dict[str, Any]] = field(default_factory=list)  # executor calls: {pass, t, tool, proposal_tick, success}
    pains: list[dict[str, Any]] = field(default_factory=list)  # PainBus.publish: {pass, t, failure_mode, source}
    gate: list[dict[str, Any]] = field(default_factory=list)  # ThoughtGate decisions: {pass, passed, reason}
    reflex_evals: list[dict[str, Any]] = field(default_factory=list)  # ReflexRegistry.evaluate: {pass, fired}
    enrich_calls: list[int] = field(default_factory=list)  # the pass of each BioEnrichmentPipeline.enrich
    world: Any = None
    end_t: float = 0.0

    def pass_at(self, t: float) -> int:
        """The first pass whose start is at or after step-clock time ``t``."""
        for p in self.passes:
            if p["t"] >= t:
                return int(p["pass"])
        raise ScenarioBroken(f"no pass starts at or after t={t}")


@dataclass
class Rig:
    executor: Any
    bio: Any
    kwargs: dict[str, Any]
    world: Any = None
    percept_source: Any = None


# ── rigs (built INSIDE the step-clock window, as run_arm builds its AUT) ──


def minecraft_rig(
    clock: _StepClock,
    workdir: Path,
    *,
    submerged: bool,
    fear_writes: int = 0,
    seed_reward: bool = False,
    seed_tool: str = "minecraft_player_mine_block",
    hurt_after: float | None = None,
) -> Callable[..., Rig]:
    """The ``_loop_harness`` Minecraft AUT (``run_arm``'s setup) against a ``_HurtWorld``."""

    def build(*, max_steps: int, stop_event: threading.Event, target_hz: float, telemetry: Any) -> Rig:
        from maxim.runtime import loop_setup as LS
        from maxim.runtime import substrate_proposal as SP
        from maxim.simulation.minecraft_harness import _loop_kwargs, build_minecraft_aut

        world = _HurtWorld(clock, submerged=submerged, hurt_after=hurt_after)
        aut = build_minecraft_aut(
            agent_id=MC_AGENT, bridge_port=0, persistence_dir=str(workdir / MC_AGENT), client=world
        )
        clock.on_advance = aut.backend.sync_world_sensors
        aut.backend.sync_world_sensors()
        if fear_writes or seed_reward:  # run_arm's seeded NAc, keyed on the first situation the loop encodes
            enc = LS._build_loop_sensor_encoder(aut.bio.memory_hub, aut.bio.nac)
            world_cluster = SP._encode_current_clusters(enc, MC_AGENT, aut.executor)["world"]
            for _ in range(fear_writes):
                aut.bio.nac.record_cluster_fear(MC_AGENT, world_cluster, "drive:oxygen", 1.0)
            if seed_reward:
                aut.bio.nac.update_cluster_reward(MC_AGENT, world_cluster, f"tool:{seed_tool}", reward=2.0)
        kwargs = _loop_kwargs(
            aut, max_steps=max_steps, stop_event=stop_event, target_hz=target_hz, substrate_telemetry=telemetry
        )
        return Rig(executor=aut.executor, bio=aut.bio, kwargs=kwargs, world=world, percept_source=aut.percept_source)

    return build


def body_rig(
    workdir: Path,
    *,
    initial: dict[str, float] | None = None,
    conversational: bool = False,
    wire_reflexes: bool = False,
) -> Callable[..., Rig]:
    """A drifting cradle body (``_cradle_loop_driver``'s composition) under the REAL substrate proposer.

    ``conversational`` gives the loop the orchestrator AUT's percept source (``ConversationalSource``);
    ``wire_reflexes`` wires the percept reflexes onto the bio stack's ``BioEnrichmentPipeline`` exactly
    as ``simulation/orchestrator.py`` does for the AUT: the archetype's reflex specs in a
    ``ReflexRegistry`` with the live integrity closure, and the real ``DamageComponentTool`` /
    ``SetEntitySensorTool`` on the AUT's body as the dispatch targets."""

    def build(*, max_steps: int, stop_event: threading.Event, target_hz: float, telemetry: Any) -> Rig:
        from maxim.agents.autonomy import AutonomyController, AutonomyLevel
        from maxim.embodiment.component_registry import ComponentRegistry
        from maxim.embodiment.reflex import ReflexRegistry, load_archetype_reflexes
        from maxim.embodiment.sem import _resolve_sensor_slot
        from maxim.runtime.bio_stack import build_bio_stack
        from maxim.runtime.bootstrap import build_executor
        from maxim.simulation.conversational_source import ConversationalSource
        from maxim.simulation.tools import DamageComponentTool, SetEntitySensorTool
        from maxim.tools.registry import ToolRegistry

        components = ComponentRegistry()
        bio = build_bio_stack(agent_id=BODY_AGENT, persistence_dir=str(workdir / BODY_AGENT))
        executor = build_executor(
            tool_registry=ToolRegistry(),
            permissions=None,
            agent_id=BODY_AGENT,
            pain_bus=bio.pain_bus,
            nac=bio.nac,
            hippocampus=bio.hippocampus,
            scn=bio.scn,
            cerebellum=bio.cerebellum,
            distributor=bio.distributor,
            entity_ref=BODY_REF,
            component_registry=components,
        )
        emb = executor.embodiment
        require(emb is not None and emb.root is not None, "the body did not attach")
        for name, value in (initial or {}).items():
            slot = _resolve_sensor_slot(emb.root, name)
            require(slot is not None, f"the body has no sensor {name!r}")
            slot[0][slot[1]] = float(value)
        source = ConversationalSource() if conversational else None
        if wire_reflexes:
            pipeline = bio.bio_enrichment_pipeline
            require(pipeline is not None, "the bio stack built no BioEnrichmentPipeline")
            archetype = components.get(BODY_REF).get("component", {}).get("archetype")
            specs = load_archetype_reflexes(archetype)
            require(specs, f"no reflexes for archetype {archetype!r}")

            def _integrity(name: str) -> float:
                comp = emb.root.get_component(name) if emb.root is not None else None
                return comp.compute_integrity() if comp is not None and hasattr(comp, "compute_integrity") else 1.0

            pipeline._reflex_registry = ReflexRegistry(specs, get_component_integrity=_integrity)
            pipeline._reflex_damage_tool = DamageComponentTool(embodiment=emb, entity_map=None)
            pipeline._reflex_sensor_tool = SetEntitySensorTool(embodiment=emb, entity_map=None)
            pipeline._entity_root = emb.root
        kwargs: dict[str, Any] = {
            "aut_mode": "substrate-primary",
            "autonomy_controller": AutonomyController(initial_level=AutonomyLevel.AUTONOMOUS),
            "pain_bus": bio.pain_bus,
            "memory_hub": bio.memory_hub,
            "hippocampus": bio.hippocampus,
            "max_steps": max_steps,
            "stop_event": stop_event,
            "target_hz": target_hz,
            "consolidation": "full",
            "substrate_telemetry": telemetry,
        }
        if source is not None:
            kwargs["percept_source"] = source
        return Rig(executor=executor, bio=bio, kwargs=kwargs, percept_source=source)

    return build


# ── the driven run ────────────────────────────────────────────────────────


def drive_loop(
    rig_builder: Callable[..., Rig],
    workdir: Path,
    *,
    clock: _StepClock,
    max_steps: int,
    target_hz: float = 4.0,
    substrate_action_gate: Callable[[], bool] | None = None,
    thought_gate: Any = "none",
    bio_enrichment: bool = False,
    on_pass: Callable[[int, Rig], None] | None = None,
    on_tick: Callable[[dict[str, Any], Any, Rig], None] | None = None,
) -> Trace:
    """Drive the REAL ``run_agentic_loop`` once (module docstring). ``thought_gate``: ``"none"`` passes
    none, ``"bio"`` passes the bio stack's own ThoughtGate, a callable ``f(bio)`` builds one.
    ``on_pass(pass, rig)`` runs at each pass's gate entry; ``on_tick(tick, proposal, rig)`` at each
    substrate tick (inside §6b, after the proposal is made)."""
    assert_this_checkout()
    preimport_loop_graph()
    for name in _PREIMPORT_EXTRA:
        importlib.import_module(name)
    from maxim.agents.maxim_agent import MaximAgent
    from maxim.environment.filesystem_env import FileSystemEnv
    from maxim.runtime import agent_loop as AL
    from maxim.runtime.bootstrap import build_decision_engine, build_memory
    from maxim.runtime.state import RuntimeState
    from maxim.similarity import ec as ec_mod

    trace = Trace()
    cur = {"pass": -1}
    counter = iter(range(1, 1_000_000))
    rig_holder: list[Rig] = []

    class _Telemetry:
        def snapshot(self, *, step: int, nac: Any, ec: Any, executor: Any, proposal: Any, gated: bool = False):
            action = (getattr(proposal, "action", None) or {}) if proposal is not None else {}
            tick = {
                "tick": len(trace.ticks),
                "pass": cur["pass"],
                "t": clock.t,
                "gated": gated,
                "tool": action.get("tool_name"),
            }
            trace.ticks.append(tick)
            trace.proposals.append(proposal)
            if on_tick is not None:
                on_tick(tick, proposal, rig_holder[0])

    orig_gate = AL.pre_tick_gate

    def _gate(**kw: Any) -> Any:
        cur["pass"] = int(kw["step_num"])
        trace.passes.append({"pass": cur["pass"], "t": clock.t})
        if on_pass is not None:
            on_pass(cur["pass"], rig_holder[0])
        return orig_gate(**kw)

    patches: list[tuple[Any, str, Any]] = [
        (ec_mod, "uuid4", lambda: uuid.UUID(int=next(counter))),
        (AL, "pre_tick_gate", _gate),
    ]
    saved = [(obj, name, getattr(obj, name)) for obj, name, _ in patches]
    old_cwd = os.getcwd()
    modules_before = set(sys.modules)
    for obj, name, replacement in patches:
        setattr(obj, name, replacement)
    clock.install()
    rig: Rig | None = None
    try:
        os.chdir(workdir)
        stop = threading.Event()
        rig = rig_builder(max_steps=max_steps, stop_event=stop, target_hz=target_hz, telemetry=_Telemetry())
        rig_holder.append(rig)
        trace.world = rig.world
        bio, executor = rig.bio, rig.executor
        require(getattr(executor.embodiment, "_pain_bus", None) is bio.pain_bus, "the body publishes elsewhere")

        orig_publish = bio.pain_bus.publish

        def _publish(signal: Any) -> Any:
            ctx = getattr(signal, "context", None) or {}
            trace.pains.append(
                {
                    "pass": cur["pass"],
                    "t": clock.t,
                    "failure_mode": str(ctx.get("failure_mode", "")),
                    "source": str(ctx.get("source", "")),
                    "intensity": float(signal.intensity),
                }
            )
            return orig_publish(signal)

        bio.pain_bus.publish = _publish

        orig_execute = executor.execute

        def _execute(action: Any) -> Any:
            owner = next((i for i, p in enumerate(trace.proposals) if p is not None and p.action is action), None)
            call = {"pass": cur["pass"], "t": clock.t, "tool": (action or {}).get("tool_name"), "proposal_tick": owner}
            trace.calls.append(call)
            result = orig_execute(action)
            call["success"] = bool(getattr(result, "success", False))
            return result

        executor.execute = _execute

        gate_obj = None
        if thought_gate == "bio":
            gate_obj = bio.thought_gate
        elif callable(thought_gate):
            gate_obj = thought_gate(bio)
        if gate_obj is not None:
            orig_should = gate_obj.should_think

            def _should(**kw: Any) -> Any:
                decision = orig_should(**kw)
                trace.gate.append({"pass": cur["pass"], "passed": bool(decision.passed), "reason": decision.reason})
                return decision

            gate_obj.should_think = _should
        pipeline = bio.bio_enrichment_pipeline if bio_enrichment else None
        if pipeline is not None:
            orig_enrich = pipeline.enrich

            def _enrich(*a: Any, **k: Any) -> Any:
                trace.enrich_calls.append(cur["pass"])
                return orig_enrich(*a, **k)

            pipeline.enrich = _enrich
            registry = getattr(pipeline, "_reflex_registry", None)
            if registry is not None:
                orig_eval = registry.evaluate

                def _evaluate(text: str, **k: Any) -> Any:
                    firings = orig_eval(text, **k)
                    trace.reflex_evals.append(
                        {"pass": cur["pass"], "fired": [(f.reflex_name, f.outcome) for f in firings]}
                    )
                    return firings

                registry.evaluate = _evaluate

        workspace = workdir / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        agent = MaximAgent()
        agent.wire_memory_hub(bio.memory_hub)
        state = RuntimeState()
        state.data["mode"] = "active"
        state.data["active_goal"] = "survive in the world"
        extra: dict[str, Any] = {}
        if substrate_action_gate is not None:
            extra["substrate_action_gate"] = substrate_action_gate
        if gate_obj is not None:
            extra["thought_gate"] = gate_obj
        if pipeline is not None:
            extra["bio_enrichment_pipeline"] = pipeline
        AL.run_agentic_loop(
            agent,
            FileSystemEnv(str(workspace)),
            state,
            build_memory(),
            build_decision_engine(),
            executor,
            **rig.kwargs,
            **extra,
        )
        new_modules = sorted(m for m in set(sys.modules) - modules_before if m == "maxim" or m.startswith("maxim."))
        if new_modules:
            raise ImportedInsideWindow(
                f"first imported inside the step-clock window; add to _PREIMPORT_EXTRA: {new_modules}"
            )
        trace.end_t = clock.t
        return trace
    finally:
        clock.uninstall()
        os.chdir(old_cwd)
        for obj, name, orig in saved:
            setattr(obj, name, orig)
        if rig is not None:
            try:
                rig.bio.on_session_end()
            except Exception as exc:  # teardown only; the trace is already taken
                print(f"bio-stack session end raised: {exc!r}", file=sys.stderr)


def _mkdir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


# ── (d) L1: the stale substrate proposal ──────────────────────────────────


# The shore arm seeds ``mine_block``, which substrate-primary proposes with no coordinates, so the
# executor refuses it ("Missing required input: x") before it reaches the world. Gate (d) seeds ``eat``
# instead (param-free), so the stale action is one the body really performs.
D1_SEED_TOOL = "minecraft_player_eat"


def run_d1(tmp_path: Path) -> tuple[Trace, int]:
    clock = _StepClock()
    armed: list[int] = []

    def on_tick(tick: dict[str, Any], proposal: Any, rig: Rig) -> None:
        if proposal is not None and not armed:
            armed.append(tick["tick"])
            rig.world.hurt_after = clock.t

    trace = drive_loop(
        minecraft_rig(clock, tmp_path, submerged=False, seed_reward=True, seed_tool=D1_SEED_TOOL),
        tmp_path,
        clock=clock,
        max_steps=8,
        on_tick=on_tick,
    )
    require(armed, "no substrate proposal was ever made: the hurt was never armed")
    return trace, armed[0]


def _stale_executions(trace: Trace, k: int) -> list[int]:
    """The executor calls that ran tick ``k``'s proposal: zero (fixed) or exactly one (today).

    Two independent matches, so a fix or refactor that COPIES the action dict before
    ``executor.execute`` (breaking object identity while the stale action still runs) cannot XPASS
    the gate: (1) identity -- the executed dict IS the proposal's (``proposal_tick == k``); (2)
    signature -- the call ran on the pass after the proposing pass with the stale tool, and is not
    identity-bound to a DIFFERENT (re-evaluated) proposal. A disagreement between them, or more than
    one candidate, is a broken scenario (``ScenarioBroken``), never a pass."""
    tick = trace.ticks[k]
    identity = [i for i, c in enumerate(trace.calls) if c["proposal_tick"] == k]
    signature = [
        i
        for i, c in enumerate(trace.calls)
        if c["pass"] == tick["pass"] + 1 and c["tool"] == tick["tool"] and c["proposal_tick"] in (None, k)
    ]
    candidates = sorted(set(identity) | set(signature))
    require(
        len(candidates) <= 1,
        f"ambiguous stale match for tick {k} ({tick['tool']} at pass {tick['pass']}): identity {identity}, "
        f"signature {signature}, calls {trace.calls}",
    )
    require(
        not identity or identity == signature,
        f"tick {k}'s proposal ran, but not on pass {tick['pass'] + 1} as {tick['tool']!r}: "
        f"{[trace.calls[i] for i in identity]}",
    )
    return candidates


@pytest.mark.timeout(240)
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="#1176 L1: flips with the L1 fix or GL3.B4, whichever #1176 decides",
)
def test_tracks_d1_stale_substrate_proposal(tmp_path: Path) -> None:
    """Health drops between passes; the proposal decided on the pre-damage world is NOT executed."""
    trace, k = run_d1(_mkdir(tmp_path / "d1"))
    tick = trace.ticks[k]
    stale = trace.proposals[k]
    require(stale is not None and tick["tool"] not in PROTECTIVE_TOOLS, f"the pre-damage pick was {tick['tool']!r}")
    breach_pass = trace.pass_at(tick["t"] + 1e-9)  # the first pass that starts after the hurt is armed
    require(breach_pass == tick["pass"] + 1, f"the damage was not between two adjacent passes: {trace.passes[:6]}")
    health_pain = [p for p in trace.pains if p["failure_mode"] == "drive:health"]
    require(health_pain, "the health breach never reached the PainBus: the damage did not happen")
    executed = [trace.calls[i] for i in _stale_executions(trace, k)]
    assert not executed, (
        f"the proposal decided at pass {tick['pass']} on the PRE-damage world ({tick['tool']}) was executed at "
        f"pass {executed[0]['pass']} (success={executed[0].get('success')}; world actions {trace.world.actions}), "
        f"after health dropped to {HURT_HEALTH} in the sleep between them (first drive:health pain at pass "
        f"{health_pain[0]['pass']})"
    )


# ── (e) L2: the turn budget gates nociception ─────────────────────────────

# thirst (entropic up, 0.008/s, pain at the 0.6 deprivation threshold) starts just below it, so the
# drift carries it into the band at t ~ 0.94 s, between two substrate ticks (no boundary tie).
THIRST_START = 0.5925
_THIRST = "thirst"


def run_d2(tmp_path: Path, *, deny: bool) -> Trace:
    clock = _StepClock()
    return drive_loop(
        body_rig(tmp_path, initial={"thirst": THIRST_START}),
        tmp_path,
        clock=clock,
        max_steps=16,
        substrate_action_gate=(lambda: False) if deny else None,
    )


def _first_thirst_pain(trace: Trace) -> dict[str, Any] | None:
    return next((p for p in trace.pains if _THIRST in p["failure_mode"]), None)


@pytest.mark.timeout(240)
def test_d2_control_a_free_loop_publishes_the_drift_breach(tmp_path: Path) -> None:
    """Non-vacuity of (e): with no budget, the drift breach publishes on a substrate tick, and the run
    is action-free (no proposal is executed), so the denying run differs ONLY in the gate."""
    trace = run_d2(_mkdir(tmp_path / "free"), deny=False)
    first = _first_thirst_pain(trace)
    assert first is not None, [p["failure_mode"] for p in trace.pains]
    assert first["pass"] in {t["pass"] for t in trace.ticks}
    assert not trace.calls, trace.calls


@pytest.mark.timeout(240)
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="#1177 L2 (TR2: the budget delays only action): flips with the L2 fix",
)
def test_tracks_d2_turn_gate_skips_pain(tmp_path: Path) -> None:
    """With a DENYING ``substrate_action_gate``, the drift-driven breach is still published on the
    breach pass (the pass a free-running loop publishes it on)."""
    free = run_d2(_mkdir(tmp_path / "free"), deny=False)
    breach = _first_thirst_pain(free)
    require(breach is not None and not free.calls, "the control run did not publish the breach action-free")
    denied = run_d2(_mkdir(tmp_path / "denied"), deny=True)
    require(denied.ticks and all(t["gated"] for t in denied.ticks), "the gate did not deny every substrate tick")
    require(
        [t["pass"] for t in denied.ticks] == [t["pass"] for t in free.ticks],
        "the substrate cadence differs between the runs: the gate changed more than the gate",
    )
    got = [p["pass"] for p in denied.pains if _THIRST in p["failure_mode"]]
    assert breach["pass"] in got, (
        f"the thirst breach published at pass {breach['pass']} on the free loop; under the denying turn gate "
        f"it was published at passes {got} (the denied tick skips propose_via_substrate and evaluate_failures)"
    )


# ── (f) L5: the reflex behind the ThoughtGate ─────────────────────────────

REFLEX_PASS = 4  # the pass the matching percept is delivered on
REFLEX_TEXT = "a flame leaps from the hearth and fire licks across your skin"  # humanoid ``fire_burn``


def _exhausted_gate(bio: Any) -> Any:
    """The bio stack's ThoughtGate configuration with an LLM energy tracker over its token budget."""
    from maxim.energy.llm_tracker import LLMEnergyTracker
    from maxim.runtime.thought_gate import ThoughtGate

    tracker = LLMEnergyTracker()
    tracker.record(input_tokens=100_000, output_tokens=0)
    return ThoughtGate(scorer=getattr(bio.thought_gate, "_scorer", None), energy_tracker=tracker)


def run_d5(tmp_path: Path, *, gate: str) -> Trace:
    """``gate``: ``"open"`` (no ThoughtGate: the enrichment always runs -- the control),
    ``"refractory"`` (the bio stack's gate, its refractory reset on the pass before the percept, as the
    loop's own ``reset_refractory`` does after a deliberation), or ``"energy"`` (exhausted)."""
    clock = _StepClock()

    def on_pass(n: int, rig: Rig) -> None:
        if n == REFLEX_PASS - 1 and gate == "refractory":
            rig.bio.thought_gate.reset_refractory(n)
        if n == REFLEX_PASS:
            rig.percept_source.inject_cli(REFLEX_TEXT)

    return drive_loop(
        body_rig(tmp_path, conversational=True, wire_reflexes=True),
        tmp_path,
        clock=clock,
        max_steps=REFLEX_PASS + 4,
        thought_gate={"open": "none", "refractory": "bio", "energy": _exhausted_gate}[gate],
        bio_enrichment=True,
        on_pass=on_pass,
    )


def _reflex_pain(trace: Trace) -> list[dict[str, Any]]:
    return [p for p in trace.pains if p["source"] == "damage_component" and p["failure_mode"].startswith("reflex_")]


@pytest.mark.timeout(240)
def test_d5_control_an_open_gate_fires_the_reflex(tmp_path: Path) -> None:
    """Non-vacuity of (f): with no ThoughtGate the same percept, on the same pass, fires the reflex,
    which acts on the body (the real ``DamageComponentTool`` publishes its pain)."""
    trace = run_d5(_mkdir(tmp_path / "open"), gate="open")
    assert REFLEX_PASS in trace.enrich_calls, trace.enrich_calls
    fired = [e for e in trace.reflex_evals if e["pass"] == REFLEX_PASS]
    assert fired and ("fire_burn", "acted") in fired[0]["fired"], trace.reflex_evals
    assert [p["pass"] for p in _reflex_pain(trace)] == [REFLEX_PASS], trace.pains


@pytest.mark.timeout(240)
@pytest.mark.parametrize("gate", ["refractory", "energy"])
@pytest.mark.xfail(strict=True, raises=AssertionError, reason="#1178 L5: flips with the L5 fix")
def test_tracks_d5_reflex_behind_thought_gate(gate: str, tmp_path: Path) -> None:
    """With a refractory or energy-exhausted ThoughtGate, a matching percept's reflex still fires."""
    trace = run_d5(_mkdir(tmp_path / gate), gate=gate)
    decisions = [d for d in trace.gate if d["pass"] == REFLEX_PASS]
    want = {"refractory": "refractory", "energy": "energy exhausted"}[gate]
    require(
        decisions and not decisions[0]["passed"] and decisions[0]["reason"].startswith(want),
        f"the gate did not reject the percept's pass for {want!r}: {decisions}",
    )
    fired = [e for e in trace.reflex_evals if e["pass"] == REFLEX_PASS and ("fire_burn", "acted") in e["fired"]]
    assert fired and _reflex_pain(trace), (
        f"the fire percept at pass {REFLEX_PASS} matched the humanoid fire_burn reflex, but the ThoughtGate "
        f"rejected it ({decisions[0]['reason']}) so enrich() -- and the reflex inside it -- never ran "
        f"(enrich passes {trace.enrich_calls}, reflex evaluations {trace.reflex_evals})"
    )
