"""Lockstep driver for the REAL ``run_agentic_loop`` (1.3.2 decomposition, slice 0).

Used by ``test_agent_loop_selection_golden.py`` (the loop-level selection gate) and meant to be
reused by the decomposition's later characterization arms (slice 4's PLANNING arm, slice 5's
percept-bearing arm; ``docs/plans/roadmap_1_3_x.md`` §"The decomposition").

WHAT IT DRIVES. The real loop in ``aut_mode="substrate-primary"`` with the Minecraft harness's own
kwargs (``minecraft_harness._loop_kwargs``: AUTONOMOUS, full consolidation) on the canonical
builders (``build_minecraft_aut``), the loop building its OWN sensor encoder. Single-threaded: the
bridge is ``_ScriptedWorld``, an in-process client passed through ``build_minecraft_aut(client=)``
whose state is a pure function of the step clock -- no socket, no reader thread, no sync pump.

WHY NOT ``scripts/survival_world/scripted_water.py::StepClock``/``LockstepTime`` (#951). Those pace a
socket bridge and patch ONE harness module's ``time`` (``WaterTrial``'s), with the loop still on a
thread on wall time. The golden needs the opposite: the LOOP itself on the step clock, which means
patching the GLOBAL ``time`` module (so code a slice moves into ``runtime/loop_*.py`` reads it too),
and an in-process world behind the ``client=`` seam, so nothing races the loop.

THE STEP CLOCK. ``_StepClock.install()`` replaces ``time.time``/``monotonic``/``perf_counter``/
``sleep`` and the ``_ns`` variants on the ``time`` module for the run. Only the loop's thread
advances it: a ``sleep`` there advances by exactly the requested amount and re-syncs the world
sensors (the pump's job live); any other thread really sleeps, briefly, and reads the step clock
without advancing it.

IMPORT-TIME BINDINGS (executor-lens review, slice 0). A module imported INSIDE the window would bind
``default_factory=time.time`` / ``monotonic=time.monotonic`` defaults (about 30 sites in ``maxim``,
e.g. ``LLMProposal.timestamp``, ``EpisodicMemory.created_at``, ``ExperienceClockDriver``) to that
run's clock object forever -- a dead clock for every later run and test in the process. So
``preimport_loop_graph()`` imports the loop's whole module graph BEFORE the window, and ``run_arm``
FAILS if any module is first imported inside it (``ImportedInsideWindow``: add it to
``_PREIMPORT``). The stated limit: those import-time-bound defaults read the REAL clock during a
run; only call-time reads (``time.time()`` in a function body, wherever the function lives) are on
the step clock. The trace records none of the import-time-bound values, and the determinism tests
(two in-process runs, two hash seeds, subprocesses) are what show it does not depend on them.
"""

from __future__ import annotations

import concurrent.futures  # noqa: F401 -- stdlib modules that bind time.monotonic at import
import importlib
import json
import os
import queue  # noqa: F401
import subprocess  # noqa: F401
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_ID = "golden_aut"
FLOAT_DECIMALS = 9
EPOCH = 1_800_000_000.0  # the step clock's wall origin (any fixed value; the trace records offsets)
ARMS: dict[str, dict[str, Any]] = {
    "shore": {"submerged": False, "fear_writes": 0, "seed_reward": True, "max_steps": 160},
    "fear_water": {"submerged": True, "fear_writes": 2, "seed_reward": False, "max_steps": 60},
}
ESCAPE = "minecraft_player_escape_water"
SEEDED_TOOL = "minecraft_player_mine_block"  # the shore arm's learned bias (any non-drive affordance)

# The loop's module graph, imported BEFORE the step-clock window (module docstring). The first group
# are the run's entry points; the rest are what those import lazily during a run. A run that first
# imports anything else fails with ImportedInsideWindow naming it -- add it here.
_PREIMPORT: tuple[str, ...] = (
    "maxim.runtime.agent_loop",
    "maxim.simulation.minecraft_harness",
    "maxim.agents.maxim_agent",
    "maxim.environment.filesystem_env",
    "maxim.runtime.bootstrap",
    "maxim.runtime.state",
    "maxim.runtime.bio_stack",
    "maxim.decisions.nac",
    "maxim.similarity.ec",
    "maxim.similarity.encoder",
    "maxim.embodiment.backends.minecraft",
    "maxim.simulation.minecraft",
    "maxim.embodiment.component_registry",
    "maxim.tools.registry",
    # lazily imported during a run (measured 2026-10-05)
    "maxim.agents.percept_factory",
    "maxim.agents.permissions",
    "maxim.decisions.temporal_credit",
    "maxim.decisions.valence_signal",
    "maxim.embodiment.audio_localization",
    "maxim.embodiment.body",
    "maxim.embodiment.cerebellum",
    "maxim.embodiment.motor",
    "maxim.embodiment.spec",
    "maxim.embodiment.tool_bridge",
    "maxim.integration.bio_enrichment",
    "maxim.memory.concept_context",
    "maxim.memory.concept_extractor",
    "maxim.memory.concept_grounder",
    "maxim.memory.context_index",
    "maxim.memory.pattern_completer",
    "maxim.modes",
    "maxim.modes.definitions",
    "maxim.prompts.cluster_bias_annotation",
    "maxim.proprioception.pain_bus",
    "maxim.reactions.bus",
    "maxim.reactions.compat",
    "maxim.runtime.agent_factory",
    "maxim.runtime.dn_controller",
    "maxim.runtime.experience_time",
    "maxim.runtime.fetch_cache",
    "maxim.runtime.file_patterns",
    "maxim.runtime.loop_controller",
    "maxim.runtime.loop_types",
    "maxim.runtime.prefetch",
    "maxim.runtime.sim_adapter",
    "maxim.runtime.thought_gate",
    "maxim.runtime.worker_pool",
    "maxim.simulation.sim_logger",
    "maxim.time.temporal_event",
    "maxim.tools.introspection",
    "maxim.utils.agent_output",
    "maxim.utils.paths",
    "maxim.utils.singleton",
    # imported only under pytest's isolated HOME (the conftest environment)
    "maxim.peer",
    "maxim.peer.cli",
    "maxim.peer.config",
    "maxim.peer.install_core",
)


class ImportedInsideWindow(AssertionError):
    """A module was first imported while the step clock was installed (module docstring)."""


class WrongCheckout(AssertionError):
    """The imported ``maxim`` is not this checkout's (an installed package shadowing a worktree)."""


def assert_this_checkout() -> None:
    """A worktree run without PYTHONPATH would test main's code and still pass: refuse it."""
    import maxim

    here = Path(maxim.__file__).resolve()
    if not here.is_relative_to(REPO_ROOT):
        raise WrongCheckout(
            f"maxim imported from {here}, not this checkout ({REPO_ROOT}); set PYTHONPATH=<checkout>/src"
        )


def preimport_loop_graph() -> None:
    for name in _PREIMPORT:
        importlib.import_module(name)


# ── the step clock ────────────────────────────────────────────────────────


class _StepClock:
    """Simulated time for the GLOBAL ``time`` module; only the owner thread advances it."""

    _NAMES = ("time", "monotonic", "perf_counter", "sleep", "time_ns", "monotonic_ns", "perf_counter_ns")

    def __init__(self) -> None:
        self.t = 0.0
        self.owner = threading.get_ident()
        self.on_advance: Any = None
        self._real: dict[str, Any] = {}

    def time(self) -> float:
        return EPOCH + self.t

    def monotonic(self) -> float:
        return 1000.0 + self.t

    perf_counter = monotonic

    def time_ns(self) -> int:
        return int(round(self.time() * 1e9))

    def monotonic_ns(self) -> int:
        return int(round(self.monotonic() * 1e9))

    perf_counter_ns = monotonic_ns

    def sleep(self, dt: float) -> None:
        if threading.get_ident() != self.owner:
            # a background worker: never advances the clock (it would make the trace scheduling-
            # dependent); sleep for real, briefly, so it cannot spin
            self._real["sleep"](min(max(float(dt), 0.0), 0.02))
            return
        self.t = round(self.t + max(float(dt), 0.0), 9)
        if self.on_advance is not None:
            self.on_advance()

    def install(self) -> None:
        for name in self._NAMES:
            self._real[name] = getattr(time, name)
            setattr(time, name, getattr(self, name))

    def uninstall(self) -> None:
        for name, fn in self._real.items():
            setattr(time, name, fn)


# ── the scripted world (an in-process MinecraftClient) ────────────────────


class _ScriptedWorld:
    """The ``MinecraftClient`` surface the backend + percept source read, as a pure function of the
    step clock. No socket, no reader thread, no events (the live condition)."""

    SHORE_Y, WATER_Y = 64.0, 60.0
    SURFACE_DELAY_S = 0.5  # escape_water surfaces the bot this long after the call (step clock)
    FOOD_DRAIN_PER_S = 1.0  # fed at t=0; hunger (food < 16) from t=4 s
    EAT_FOOD = 8.0

    def __init__(self, clock: _StepClock, *, submerged: bool) -> None:
        self._clock = clock
        self.submerged = submerged
        self._since = clock.t if submerged else None
        self._surface_at: float | None = None
        self._food0, self._food_t0 = 20.0, 0.0
        self.actions: list[str] = []

    def _advance(self) -> None:
        if self._surface_at is not None and self._clock.t >= self._surface_at:
            self.submerged, self._since, self._surface_at = False, None, None

    def latest_state(self) -> dict[str, float]:
        self._advance()
        t, wet = self._clock.t, self.submerged
        oxygen = max(0.0, 20.0 - 1.0 * (t - self._since)) if wet and self._since is not None else 20.0
        food = max(0.0, self._food0 - self.FOOD_DRAIN_PER_S * (t - self._food_t0))
        return {
            "health": 20.0,
            "food": round(food, 6),
            "saturation": 10.0,
            "oxygen": round(oxygen, 6),
            "light_level": 9.0 if wet else 14.0,
            "y_altitude": self.WATER_Y if wet else self.SHORE_Y,
            "nearest_hostile_dist": 64.0,
            "hostile_count": 0.0,
            "nearest_player_dist": 64.0,
            "distance_from_spawn": 0.0,
            "speed": 0.0,
            "on_ground": 0.0 if wet else 1.0,
            "is_raining": 0.0,
            "is_in_water": 1.0 if wet else 0.0,
            "xp_level": 0.0,
            "look_pitch": 0.0,
            "time_of_day": 0.25,
        }

    def state_age_s(self) -> float:
        return 0.0

    def has_events(self) -> bool:
        return False

    def pop_event(self) -> None:
        return None

    def call_action(self, name: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        self._advance()
        self.actions.append(name)
        if name == "escape_water":
            if not self.submerged:
                return {"ok": True, "detail": "already in air"}
            if self._surface_at is None:
                self._surface_at = self._clock.t + self.SURFACE_DELAY_S
            return {"ok": True, "detail": "surfaced"}
        if name == "flee" and self.submerged:
            return {"ok": False, "detail": "flee: submerged — the pathfinder is dead in water"}
        if name == "eat":
            food_now = self.latest_state()["food"]
            self._food0, self._food_t0 = min(20.0, food_now + self.EAT_FOOD), self._clock.t
        return {"ok": True, "detail": f"did {name}"}

    def close(self) -> None:
        pass


# ── canonical form ────────────────────────────────────────────────────────


class _Labels:
    """EC node id -> ``c<k>`` by first appearance in the trace."""

    def __init__(self) -> None:
        self._map: dict[str, str] = {}

    def __call__(self, node_id: Any) -> Any:
        if not isinstance(node_id, str):
            return node_id
        if node_id not in self._map:
            self._map[node_id] = f"c{len(self._map)}"
        return self._map[node_id]

    def clusters(self, clusters: Any) -> Any:
        if not clusters:
            return None
        return {str(k): self(v) for k, v in sorted(dict(clusters).items())}


def _canon(value: Any) -> Any:
    if isinstance(value, bool) or value is None or isinstance(value, (int, str)):
        return value
    if isinstance(value, float):
        r = round(value, FLOAT_DECIMALS)
        return 0.0 if r == 0 else r  # -0.0 and 0.0 are the same decision
    if isinstance(value, dict):
        return {str(k): _canon(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_canon(v) for v in value]
    return repr(value)


def canonical_json(trace: dict[str, Any]) -> str:
    return json.dumps(_canon(trace), sort_keys=True, indent=1)


# ── the driven run ────────────────────────────────────────────────────────

_RECOMMEND_FIELDS = (
    "best_tool",
    "best_score",
    "min_confidence",
    "passed_gate",
    "score_components",
    "runner_up_score",
    "n_candidates",
    "visit_count",
    "explore_decisive",
    "learned_margin",
    "cluster_reward_bias_consulted",
    "consulted_bias_by_modality",
)


def _wrap(obj: Any, name: str, record: Any) -> None:
    """Instance-level spy: ``record(*args, **kwargs)`` then the original. Instance attributes, never
    module functions, so the spy holds wherever a slice moves the CALLER."""
    orig = getattr(obj, name)

    def spy(*args: Any, **kwargs: Any) -> Any:
        record(*args, **kwargs)
        return orig(*args, **kwargs)

    setattr(obj, name, spy)


def run_arm(arm: str, workdir: Path) -> dict[str, Any]:
    """Drive the real loop for one arm; return the raw trace (canonicalise with ``canonical_json``)."""
    assert_this_checkout()
    preimport_loop_graph()
    from maxim.agents.maxim_agent import MaximAgent
    from maxim.decisions import nac as nac_mod
    from maxim.environment.filesystem_env import FileSystemEnv
    from maxim.runtime import agent_loop as AL
    from maxim.runtime.bootstrap import build_decision_engine, build_memory
    from maxim.runtime.state import RuntimeState
    from maxim.similarity import ec as ec_mod
    from maxim.simulation.minecraft_harness import _loop_kwargs, build_minecraft_aut

    spec = ARMS[arm]
    clock = _StepClock()
    labels = _Labels()
    counter = iter(range(1, 1_000_000))
    patches: list[tuple[Any, str, Any]] = [
        (ec_mod, "uuid4", lambda: uuid.UUID(int=next(counter))),  # deterministic EC node ids
    ]
    ticks: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []
    pending_recs: list[dict[str, Any]] = []
    lifecycle: dict[str, Any] = {
        "session_start": 0,
        "capture_worker_started": 0,
        "captures": [],
        "persists": 0,
        "session_end": [],
        "cue_calls": [],
    }
    orig_emit = nac_mod._emit_recommend_action_event

    def _emit(**kw: Any) -> None:
        if kw.get("agent_id") == AGENT_ID:
            rec = {k: kw.get(k) for k in _RECOMMEND_FIELDS}
            rec["current_clusters"] = labels.clusters(kw.get("current_clusters"))
            rec["current_cluster_id"] = labels(kw.get("current_cluster_id"))
            pending_recs.append(rec)
        orig_emit(**kw)

    patches.append((nac_mod, "_emit_recommend_action_event", _emit))

    class _Telemetry:
        def snapshot(self, *, step: int, nac: Any, ec: Any, executor: Any, proposal: Any, gated: bool = False):
            action = (getattr(proposal, "action", None) or {}) if proposal is not None else {}
            ticks.append(
                {
                    "tick": len(ticks),
                    "step": step,
                    "t": clock.t,
                    "gated": gated,
                    "tool": action.get("tool_name"),
                    "params": dict(action.get("params") or {}),
                    "confidence": getattr(proposal, "confidence", None),
                    "clusters": labels.clusters(getattr(proposal, "clusters", None)),
                    "recommend": list(pending_recs),
                }
            )
            pending_recs.clear()

    old_cwd = os.getcwd()
    saved = [(obj, name, getattr(obj, name)) for obj, name, _ in patches]
    for obj, name, replacement in patches:
        setattr(obj, name, replacement)
    modules_before = set(sys.modules)
    clock.install()
    aut = None
    try:
        os.chdir(workdir)  # the loop's CWD-relative data/agents/<name>/runtime/ state files land here
        world = _ScriptedWorld(clock, submerged=spec["submerged"])
        home = workdir / AGENT_ID
        aut = build_minecraft_aut(agent_id=AGENT_ID, bridge_port=0, persistence_dir=str(home), client=world)
        clock.on_advance = aut.backend.sync_world_sensors
        aut.backend.sync_world_sensors()
        hub, hippo = aut.bio.memory_hub, aut.bio.hippocampus

        seed_clusters: dict[str, str] = {}
        if spec["fear_writes"] or spec["seed_reward"]:
            # the seeded NAc: keyed on the situation the loop itself will encode first
            enc = AL._build_loop_sensor_encoder(hub, aut.bio.nac)
            seed_clusters = AL._encode_current_clusters(enc, AGENT_ID, aut.executor)
            world_cluster = seed_clusters["world"]
            for _ in range(spec["fear_writes"]):
                aut.bio.nac.record_cluster_fear(AGENT_ID, world_cluster, "drive:oxygen", 1.0)
            if spec["seed_reward"]:
                aut.bio.nac.update_cluster_reward(AGENT_ID, world_cluster, f"tool:{SEEDED_TOOL}", reward=2.0)

        # Lifecycle + 2S-d cue (architecture/executor review, slice 0): the session the loop opens and
        # closes, what it captures and where, how often it persists, and that the substrate tick hands
        # the hub's situation cue the tick's clusters. Recorded at the loop's call, never the async
        # result: ``capture_from_loop`` runs on the hippocampus-capture worker, so when it runs relative
        # to the ticks is scheduling -- recording it made the trace flaky under load.
        def _bump(key: str) -> Any:
            def rec(*_a: Any, **_k: Any) -> None:
                lifecycle[key] += 1

            return rec

        def _capture(*_a: Any, **kw: Any) -> None:
            action = kw.get("action") or {}
            lifecycle["captures"].append(
                {
                    "after_tick": len(ticks) - 1,
                    "tool": (action.get("tool") or action.get("tool_name"))
                    if isinstance(action, dict)
                    else repr(action),
                    "situation": labels.clusters(kw.get("situation")),
                }
            )

        def _cue(agent_id: str, clusters: Any = None, *_a: Any, **_k: Any) -> None:
            lifecycle["cue_calls"].append({"after_tick": len(ticks) - 1, "clusters": labels.clusters(clusters)})

        _wrap(hub, "on_session_start", _bump("session_start"))
        _wrap(hippo, "start_capture_worker", _bump("capture_worker_started"))
        _wrap(hippo, "capture_from_loop_async", _capture)
        _wrap(hub, "on_session_end", lambda *a, **k: lifecycle["session_end"].append("full"))
        _wrap(hub, "on_session_end_lightweight", lambda *a, **k: lifecycle["session_end"].append("lightweight"))
        _wrap(hub._pattern_completer, "cue_situation", _cue)

        orig_execute = aut.executor.execute

        def _execute(action: Any) -> Any:
            result = orig_execute(action)
            calls.append(
                {
                    "order": len(calls),
                    "after_tick": len(ticks) - 1,
                    "t": clock.t,
                    "tool": (action or {}).get("tool_name"),
                    "params": dict((action or {}).get("params") or {}),
                    "success": bool(getattr(result, "success", None)),
                }
            )
            return result

        aut.executor.execute = _execute
        workspace = home / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        agent = MaximAgent()
        agent.wire_memory_hub(hub)
        state = RuntimeState()
        state.data["mode"] = "active"
        state.data["active_goal"] = "survive in the world"
        _wrap(state, "save_json", _bump("persists"))
        AL.run_agentic_loop(
            agent,
            FileSystemEnv(str(workspace)),
            state,
            build_memory(),
            build_decision_engine(),
            aut.executor,
            **_loop_kwargs(
                aut,
                max_steps=spec["max_steps"],
                stop_event=threading.Event(),
                target_hz=4.0,
                substrate_telemetry=_Telemetry(),
            ),
        )
        # maxim.* only: those are the modules whose import-time defaults would bind the step clock. A lazy
        # stdlib import (e.g. importlib.readers) varies by platform and Python version and would make the
        # guard red on one runner only; the stdlib modules that DO bind time at import (concurrent.futures,
        # queue, subprocess) are imported at the top of this module, before any window.
        new_modules = sorted(m for m in set(sys.modules) - modules_before if m == "maxim" or m.startswith("maxim."))
        if new_modules:
            raise ImportedInsideWindow(f"first imported inside the step-clock window; add to _PREIMPORT: {new_modules}")
        return {
            "arm": arm,
            "max_steps": spec["max_steps"],
            "end_t": clock.t,
            "seed_clusters": labels.clusters(seed_clusters),
            "ticks": ticks,
            "executor_calls": calls,
            "lifecycle": json.loads(json.dumps(lifecycle)),
            "world_actions": list(world.actions),
            "end_submerged": world.submerged,
        }
    finally:
        clock.uninstall()
        os.chdir(old_cwd)
        for obj, name, orig in saved:
            setattr(obj, name, orig)
        if aut is not None:
            try:
                aut.bio.on_session_end()
            except Exception as exc:  # teardown only; the trace is already taken
                print(f"bio-stack session end raised: {exc!r}", file=sys.stderr)
