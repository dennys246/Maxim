"""The agent loop's per-run setup (1.3.2 decomposition, slice 1).

``build_loop_run`` is the setup block of ``agent_loop.run_agentic_loop``, moved verbatim: everything
the loop builds and starts before its first tick, in the same order. It returns a frozen ``LoopRun``
holding the run's handles; the loop unpacks them into the local names its body has always read.
One textual difference: the loop's three timing lines (``target_period``, ``max_steps_i``,
``step_iter``), which sat inside the block, now run in ``run_agentic_loop`` after it, i.e. after the
Default Network and bio-session starts. The behaviour is the same: ``LoopController.__init__`` (built
early in the block) already refuses a zero ``target_hz`` or a non-numeric ``max_steps`` before any
thread starts, pinned by ``tests/unit/test_loop_setup.py::test_bad_timing_args_are_refused_before_any_thread_starts``.

Owner decisions 2026-10-05 (the decomposition's layout): flat ``runtime/loop_<concern>.py`` modules;
a FROZEN ``LoopRun`` carries the per-run handles, and ``ctrl`` (the ``LoopController``) stays the
only mutable carrier of loop state. A ``LoopRun`` field is fixed for the run; anything the loop body
rebinds (step counters, the consecutive-tool cap, the planning-liveness exhaustion flag, the ctrl
container aliases) stays a local of ``run_agentic_loop``.

**Imports.** Helpers with a home outside ``agent_loop`` are imported from it directly
(``bio_integration.start_bio_session``, ``loop_state._persist_state_json``,
``tool_dispatch.safe_agent_name``; each is the same object ``agent_loop`` re-binds). The five helpers
only the setup calls moved here with it, bodies verbatim (rule (a) of the roadmap's import-direction
paragraph): ``_prepare_executor``, ``_planning_liveness_enabled_via_env``, ``_loop_bio_handles``,
``_build_loop_sensor_encoder``, ``_resolve_situation_cue`` (which imports the ``NO_SITUATION_CUE``
sentinel from ``substrate_proposal``, where ``propose_via_substrate`` also uses it; slice 3). Two names are PATCH SEAMS
that existing tests replace on ``agent_loop``, so they are read through the ``agent_loop`` module at
call time: ``agent_loop._record_outcome`` and ``agent_loop.resolve_llm_loop_overrides``. ``agent_loop``
is imported inside the functions, because ``agent_loop`` imports this module. The setup LOGS on the
``maxim.runtime.agent_loop`` logger (``logger`` below is that same object), so its records keep the
agent loop's logger name. ``run_agentic_loop`` binds ``build_loop_run`` at import, so a test that
wants to replace it patches ``agent_loop.build_loop_run``, not ``loop_setup.build_loop_run``.

Characterization: ``tests/unit/test_loop_setup_characterization.py`` (written before the move, kept
green unchanged by it).
"""

from __future__ import annotations

import functools
import logging
import os
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from maxim.runtime.bio_integration import start_bio_session
from maxim.runtime.loop_state import _persist_state_json
from maxim.runtime.tool_dispatch import book_refusal, execute_and_learn, safe_agent_name
from maxim.utils.logging import log_swallowed_exception

# The setup logs as the agent loop always has: the SAME logger object as ``agent_loop.logger`` (logging
# returns one logger per name), so records keep the ``maxim.runtime.agent_loop`` name.
logger = logging.getLogger("maxim.runtime.agent_loop")

if TYPE_CHECKING:
    from maxim.agents.autonomy import AutonomyController
    from maxim.agents.context_pool import ContextPool, ContextPoolConfig
    from maxim.agents.llm_worker import LLMWorker
    from maxim.evaluation.base import Evaluator
    from maxim.runtime.loop_controller import LoopController
    from maxim.runtime.sim_adapter import NullSimulationAdapter, SimulationAdapter
    from maxim.runtime.tool_dispatch import ExecutionOutcome


@dataclass(frozen=True)
class LoopRun:
    """One run's handles, built once by ``build_loop_run`` before the first tick.

    Runtime-ephemeral: never persisted and never sent anywhere, so the CC3 forward-compat rule for
    frozen dataclasses does not apply. Frozen so a handle cannot be swapped mid-run; the objects it
    holds (``ctrl``, the context pool, the encoder) are themselves mutable, and ``ctrl`` is where the
    loop's mutable state lives.

    ``dn_enabled`` duplicates ``ctrl.dn_enabled`` (a duplication that predates the move: the loop's
    local and the controller's field were both set by the setup). Carried as-is so the move stays
    pure; a later slice should read one source of truth.
    """

    executor: Any  # the caller's executor, instrumented when an action sink is given
    sim: SimulationAdapter | NullSimulationAdapter
    run_id: str
    agent_name: str
    state_path: str
    autonomy_controller: AutonomyController
    context_pool: ContextPool
    prefetcher: Any
    result_cache: Any
    ctrl: LoopController
    is_novel_thought: Callable[..., bool]
    max_response_tokens_override: int | None
    max_cycles_override: int | None
    dn_enabled: bool  # the Default Network started (False when absent or its start failed)
    nac: Any
    xclock: Any  # the ExperienceClockDriver
    agent_id: str
    drive_relief_only: bool
    rec_outcome: Callable[..., Any]
    sensor_encoder: Any
    situation_cue: Any
    memory_hub_enabled: bool
    planning_liveness_on: bool
    # ``tool_dispatch.execute_and_learn`` with every per-run argument bound (#1133): callers pass only
    # ``action``, ``confidence``, ``proposal``, ``observation`` and ``human_involved``.
    execute_and_learn: Callable[..., ExecutionOutcome]
    # ``tool_dispatch.book_refusal`` likewise (#1133): callers pass ``source``, ``tool_name``, ``error``, ``reasoning``.
    book_refusal: Callable[..., None]


# ── helpers moved from agent_loop.py (slice 1 review, rule (a): only the setup calls them) ──


def _prepare_executor(executor: Any, action_sink: Any, state: Any) -> Any:
    """The loop's executor: wrapped with instrumentation when an action sink is given, and reading the
    loop's LIVE mode at every dispatch (#826) -- the same ``state.data["mode"]`` the prompt roster reads
    each tick, so a tool the mode refuses is refused when it runs, not merely left unadvertised.
    Every wrapper delegates ``set_mode_source`` to the inner Executor."""
    if action_sink is not None:
        from maxim.simulation.instrumented_executor import InstrumentedExecutor  # noqa: PLC0415

        executor = InstrumentedExecutor(executor, action_sink)
    if executor is not None:
        executor.set_mode_source(lambda: state.data.get("mode", "observe"))
    return executor


def _planning_liveness_enabled_via_env() -> bool:
    """Operator opt-OUT for the D13 planning-liveness abort.

    The abort terminates a campaign, and it lives inside the measurement
    instrument — apparatus standard S5/S6 say such a control must be
    experiment-visible and disableable (pre-merge review, architecture lens
    S5; mirrors ``MAXIM_SIM_HARD_ABORT`` for the D12 abort). Default ON;
    set to 0/false/no/off to fall back to pre-fix behavior (a dropped
    planning turn idles, which is the bug — use only to reproduce it).
    It also gates the narrator-reliability pieces that ride on liveness:
    the reason-specific retry corrections and the one-planning-request-
    in-flight hold with its deferred-input fold.
    """
    # Deliberately the MAXIM_SIM_HARD_ABORT idiom, not the canonical
    # ``annotation_disabled_via_env``: that parser is for MAXIM_DISABLE_*
    # style vars where a TRUTHY value means "disable". This is an
    # enable-with-opt-out control, so it mirrors its sibling abort toggle
    # exactly — same file family, same falsy-set, same default-ON meaning.
    return os.environ.get("MAXIM_SIM_PLANNING_LIVENESS", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _loop_bio_handles(memory_hub: Any, hippocampus: Any, sim: Any, autonomy_controller: Any) -> tuple[Any, Any]:
    """The loop's bio handles, bound once before the loop: ``(nac, experience_clock_driver)``.

    The clock is the Hippocampus's the loop CAPTURES into (the ``hippocampus`` argument), falling
    back to the hub's, so a caller that passes a Hippocampus without a hub still gets a clock that
    moves. The driver reads the world's kind from the percept source (real-time vs turn-based; see
    ``runtime/experience_time.py``) and subtracts the autonomy controller's paused time. With no
    Hippocampus there is no clock and the driver is inert.
    """
    from maxim.runtime.experience_time import ExperienceClockDriver

    nac = getattr(memory_hub, "nac", None) if memory_hub is not None else None
    hub_hippocampus = getattr(memory_hub, "hippocampus", None) if memory_hub is not None else None
    owner = hippocampus if hippocampus is not None else hub_hippocampus
    if hippocampus is not None and hub_hippocampus is not None and hub_hippocampus is not hippocampus:
        logger.warning("agent loop: hippocampus argument is not the hub's; the experience clock follows the argument")
    clock = getattr(owner, "experience_clock", None)
    return nac, ExperienceClockDriver(
        clock,
        percept_source=getattr(sim, "percept_source", None),
        paused_seconds=getattr(autonomy_controller, "paused_seconds_total", None),
    )


def _build_loop_sensor_encoder(memory_hub: Any, nac: Any) -> Any | None:
    """The loop's Phase 0 sensor encoder, built once per loop when EC is reachable through the hub.

    Without it, substrate-primary bypasses the LinguisticEncoder text path and EC node_count stays
    at zero forever (which is what blocked the Phase 0 smoke run from being a measurement). See
    docs/plans/grounded_language_acquisition.md Phase 0 + the SensorEncoder docstring in
    similarity/encoder.py. Built in ALL modes (Phase 1, substrate_learns_from_experience.md), not
    just substrate-primary: llm-primary / real-hardware actions also encode the current
    interoception cluster at outcome time (section 4) so their real drive-relief outcomes reinforce
    the cluster-reward substrate. Harmless when unused (an unembodied chat agent never calls
    encode); cheap to construct.
    """
    if memory_hub is None:
        return None
    ec = getattr(memory_hub, "ec", None)
    if ec is None:
        return None
    try:
        from maxim.similarity.encoder import SensorEncoder

        return SensorEncoder(ec=ec, atl=getattr(memory_hub, "atl", None), nac=nac)
    except Exception:
        # Stage-1 (measurement path): without an encoder the substrate records no situation at all,
        # so a failed build is reported, not left at DEBUG as it was when this lived inline.
        log_swallowed_exception()
        return None


def _resolve_situation_cue(memory_hub: Any) -> Any:
    """The loop's memory 2S-d situation cue, resolved once per loop.

    No hub = no episodic memory: the explicit opt-out, ``NO_SITUATION_CUE``. A hub WITHOUT a cue
    (its ATL failed to build) is a degraded memory: said loudly here, once, and the loop runs on
    with the opt-out (fail-soft, like the rest of the loop) -- where the survival harnesses, which
    read ``MemoryHub.situation_cue`` directly, stop instead.
    """
    from maxim.runtime.substrate_proposal import NO_SITUATION_CUE

    if memory_hub is None:
        return NO_SITUATION_CUE
    try:
        return memory_hub.situation_cue
    except RuntimeError as e:
        logger.warning("memory 2S-d: no situation cue this run (%s)", e)
        return NO_SITUATION_CUE


def _build_sim_adapter(
    *, percept_source: Any, action_sink: Any, pain_bus: Any, sim_adapter: Any, executor: Any
) -> SimulationAdapter | NullSimulationAdapter:
    # Create simulation adapter (Phase 4: isolate sim concerns)
    from maxim.runtime.sim_adapter import SimulationAdapter, NullSimulationAdapter

    sim: SimulationAdapter | NullSimulationAdapter
    if percept_source is not None:
        sim = SimulationAdapter(percept_source, action_sink, pain_bus)
        # Wire tool registry for deregistered-tool filtering in should_skip_fallback_proposal
        if executor is not None and hasattr(executor, "registry"):
            sim._tool_registry = executor.registry
    else:
        # Stage 3 (live_audio_orient_wiring.md): a caller-held adapter lets a
        # live producer carry_percept() into the side-channel; is_sim_mode
        # stays False either way. A sim-mode adapter smuggled through this
        # kwarg would flip the 12 is_sim_mode consumer sites without a
        # percept_source — fail loud instead (pre-merge review fold).
        if sim_adapter is not None and getattr(sim_adapter, "is_sim_mode", True) is not False:
            raise ValueError(
                "sim_adapter= must be a non-sim adapter (is_sim_mode False); "
                "sim mode is entered via percept_source=, never this kwarg"
            )
        sim = sim_adapter if sim_adapter is not None else NullSimulationAdapter()
    return sim


def _context_pool_config(context_pool_config: dict[str, Any] | None) -> ContextPoolConfig:
    from maxim.agents.context_pool import ContextPoolConfig

    pool_config = ContextPoolConfig()
    if context_pool_config:
        pool_config = ContextPoolConfig(
            max_tokens=context_pool_config.get("max_tokens", 2000),
            summary_target_tokens=context_pool_config.get("summary_target_tokens", 500),
            max_entries=context_pool_config.get("max_entries", 50),
            keep_recent=context_pool_config.get("keep_recent", 5),
            include_agent_states=context_pool_config.get("include_agent_states", True),
            include_outcomes=context_pool_config.get("include_outcomes", True),
            include_abstractions=context_pool_config.get("include_abstractions", True),
            persistence_path=context_pool_config.get("persistence_path"),
        )
    return pool_config


def _novelty_gate() -> Callable[..., bool]:
    """The loop's thought-novelty check, with its own tracker (one per run)."""
    # Thought novelty tracker: deque of recent thought word-sets for
    # cross-turn novelty gating.  Thoughts with >= 75% word overlap with
    # any recent entry are suppressed from the display (they're redundant).
    _recent_thought_words: deque[set[str]] = deque(maxlen=8)

    def _is_novel_thought(text: str, min_novelty: float = 0.40) -> bool:
        """Check if a thought is sufficiently novel vs recent thoughts.

        Returns True if the thought should be shown (novel enough).
        Side effect: appends the thought's words to the tracker if novel.
        """
        words = set(text.lower().split())
        if not words:
            return False
        for recent in _recent_thought_words:
            union = len(words | recent)
            if union and len(words & recent) / union >= (1.0 - min_novelty):
                return False  # Too similar to a recent thought
        _recent_thought_words.append(words)
        return True

    return _is_novel_thought


def _planning_liveness_gate(*, planning_liveness: bool, aut_mode: str, llm_worker: Any) -> bool:
    # ONE gate for every planning-liveness failure site, computed once so no
    # site can drift
    # (pre-merge review, architecture lens S7: the substrate-primary exclusion
    # was previously only incidental — an aut_llm_worker IS constructed in
    # substrate-primary runs, so `if llm_worker:` does run there).
    # Substrate-primary proposals never flow through get_latest_proposal.
    _planning_liveness_on = (
        bool(planning_liveness)
        and aut_mode != "substrate-primary"
        and llm_worker is not None
        and _planning_liveness_enabled_via_env()
    )
    if planning_liveness and not _planning_liveness_on:
        logger.info(
            "planning liveness requested but inactive (aut_mode=%s, llm_worker=%s, env_opt_out=%s)",
            aut_mode,
            "yes" if llm_worker is not None else "no",
            "yes" if not _planning_liveness_enabled_via_env() else "no",
        )
    return _planning_liveness_on


def _bind_execute_and_learn(
    ctrl: LoopController,
    *,
    sim: Any,
    agent_name: str,
    agent_id: str,
    result_cache: Any,
    rec_outcome: Callable[..., Any],
    nac: Any,
    memory_hub_enabled: bool,
) -> Callable[..., ExecutionOutcome]:
    """The run's ``tool_dispatch.execute_and_learn`` with every per-run argument bound, like ``rec_outcome``
    (#1133). The loop's own handles are read off ``ctrl``, which holds the same objects the loop does for
    the whole run (the setup built it from them). ``memory_hub`` is ``None`` when the hub's session did
    not start: that is how the function knows to skip the plan outcome."""
    return functools.partial(
        execute_and_learn,
        agent=ctrl.agent,
        agent_name=agent_name,
        agent_id=agent_id,
        executor=ctrl.executor,
        sim=sim,
        state=ctrl.state,
        environment=ctrl.environment,
        memory=ctrl.memory,
        hippocampus=ctrl.hippocampus,
        memory_hub=ctrl.memory_hub if memory_hub_enabled else None,
        result_cache=result_cache,
        autonomy_controller=ctrl.autonomy_controller,
        rec_outcome=rec_outcome,
        recent_outcomes=ctrl.recent_outcomes,
        max_recent=ctrl.max_recent_outcomes,
        llm_worker=ctrl.llm_worker,
        context_pool=ctrl.context_pool,
        nac=nac,
        run_id=ctrl.run_id,
    )


def _bind_book_refusal(
    ctrl: LoopController, *, agent_id: str, rec_outcome: Callable[..., Any], nac: Any
) -> Callable[..., None]:
    """The run's ``tool_dispatch.book_refusal`` with every per-run argument bound (#1133): the run's
    recorder, the hub ``agent_id`` and NAc, and the controller's outcome list, worker, pool and state."""
    return functools.partial(
        book_refusal,
        rec_outcome=rec_outcome,
        agent_id=agent_id,
        recent_outcomes=ctrl.recent_outcomes,
        max_recent=ctrl.max_recent_outcomes,
        llm_worker=ctrl.llm_worker,
        context_pool=ctrl.context_pool,
        nac=nac,
        state=ctrl.state,
    )


def build_loop_run(
    *,
    agent: Any,
    environment: Any,
    state: Any,
    memory: Any,
    decision_engine: Any,
    executor: Any,
    autonomy_controller: AutonomyController | None,
    llm_worker: LLMWorker | None,
    default_network: Any | None,
    hippocampus: Any | None,
    memory_hub: Any | None,
    evaluators: list[Evaluator] | None,
    max_steps: int,
    run_id: str | None,
    stop_event: Any | None,
    on_step: Any | None,
    on_event: Any | None,
    idle_sleep_s: float,
    persist_every_n_steps: int,
    target_hz: float,
    context_pool_config: dict[str, Any] | None,
    use_tool_prompting: bool,
    protocol_registry: Any | None,
    percept_source: Any | None,
    action_sink: Any | None,
    pain_bus: Any | None,
    aut_mode: str,
    planning_liveness: bool,
    sim_adapter: Any | None,
) -> LoopRun:
    """Build and start everything ``run_agentic_loop`` needs before its first tick (module docstring).

    Side effects, in order: the first state persist, the global prefetcher, the LLM loop overrides
    (a malformed config raises here), the Default Network start, the bio session start, and the
    planning-liveness "inactive" log. A sim-mode ``sim_adapter`` raises ``ValueError`` before any.
    """
    from maxim.agents.autonomy import AutonomyController
    from maxim.agents.context_pool import ContextPool
    from maxim.runtime import agent_loop as _al
    from maxim.runtime.loop_controller import LoopController
    from maxim.runtime.prefetch import init_prefetcher, get_result_cache

    if evaluators is None:
        evaluators = []

    executor = _prepare_executor(executor, action_sink, state)

    sim = _build_sim_adapter(
        percept_source=percept_source,
        action_sink=action_sink,
        pain_bus=pain_bus,
        sim_adapter=sim_adapter,
        executor=executor,
    )

    if not run_id:
        run_id = time.strftime("%Y-%m-%d_%H%M%S")
    agent_name = safe_agent_name(agent)
    state_path = os.path.join("data", "agents", agent_name, "runtime", f"state_{run_id}.json")
    _persist_state_json(state, state_path, meta={"run_id": run_id, "agent_name": agent_name})

    # Initialize autonomy controller if not provided
    if autonomy_controller is None:
        autonomy_controller = AutonomyController()

    # Initialize context pool for accumulated observations
    context_pool = ContextPool(config=_context_pool_config(context_pool_config))

    # Initialize speculative pre-fetcher for efficient context gathering
    prefetcher = init_prefetcher(executor=executor, base_path=os.getcwd())
    result_cache = get_result_cache()

    # ── LoopController holds all transient state (Phase 1+2) ─────────────
    ctrl = LoopController(
        agent=agent,
        environment=environment,
        state=state,
        memory=memory,
        decision_engine=decision_engine,
        executor=executor,
        autonomy_controller=autonomy_controller,
        llm_worker=llm_worker,
        default_network=default_network,
        hippocampus=hippocampus,
        memory_hub=memory_hub,
        evaluators=evaluators,
        max_steps=max_steps,
        run_id=run_id,
        stop_event=stop_event,
        on_step=on_step,
        on_event=on_event,
        idle_sleep_s=idle_sleep_s,
        persist_every_n_steps=persist_every_n_steps,
        target_hz=target_hz,
        use_tool_prompting=use_tool_prompting,
        protocol_registry=protocol_registry,
        percept_source=percept_source,
        action_sink=action_sink,
        pain_bus=pain_bus,
    )
    ctrl.context_pool = context_pool
    ctrl.prefetcher = prefetcher

    is_novel_thought = _novelty_gate()

    # Operator overrides for the per-call response reserve and the PFC
    # deliberation cap (``llm.max_response_tokens`` / ``llm.deliberation_
    # max_cycles``, P21 of the sandbox plan). Resolved ONCE per loop — the
    # precedence chain logs on every call and the value cannot change
    # mid-session anyway.
    _max_response_tokens_override, _max_cycles_override = _al.resolve_llm_loop_overrides()

    # Default Network lifecycle — managed by controller
    dn_enabled = ctrl.dn_enabled
    if dn_enabled:
        if not ctrl.dn_ctrl.start():
            dn_enabled = False
            ctrl.dn_enabled = False

    # Extract NAc reference for causal learning (passed to _record_outcome)
    _loop_nac, _loop_xclock = _loop_bio_handles(memory_hub, hippocampus, sim, autonomy_controller)

    # P4 multi-agent attribution: per-agent stash key.  Producer
    # (MemoryHub.on_percept_received) writes substrate nodes keyed by
    # the hub's owning agent_id; the consumer here must use the same
    # key or the stash leaks (consumer never finds the producer's
    # write).  Prefer memory_hub.agent_id (canonical per-agent
    # identifier from AgentFactory.create_agent) and fall back to
    # the loop's filesystem-safe agent_name for raw-loop callers
    # that don't construct a MemoryHub.
    _loop_agent_id: str = (getattr(memory_hub, "agent_id", None) if memory_hub is not None else None) or agent_name

    # Phase 1 (substrate_learns_from_experience.md): outside substrate-primary the
    # LLM issues a broad always-succeed action stream, so the tool-success floor in
    # record_outcome would flood the interoception cluster with "this tool ran".
    # Credit the cluster surface from the body's real drive signal ONLY.
    _drive_relief_only: bool = aut_mode != "substrate-primary"
    # Bind the flag once so every outcome site inherits it (no per-call threading).
    _rec_outcome = functools.partial(_al._record_outcome, drive_relief_only=_drive_relief_only)

    _loop_sensor_encoder = _build_loop_sensor_encoder(memory_hub, _loop_nac)
    _loop_situation_cue = _resolve_situation_cue(memory_hub)

    # Initialize bio-system session (MemoryHub + hippocampus capture worker)
    memory_hub_enabled = start_bio_session(memory_hub=memory_hub, hippocampus=hippocampus)

    _planning_liveness_on = _planning_liveness_gate(
        planning_liveness=planning_liveness, aut_mode=aut_mode, llm_worker=llm_worker
    )

    _execute_and_learn = _bind_execute_and_learn(
        ctrl,
        sim=sim,
        agent_name=agent_name,
        agent_id=_loop_agent_id,
        result_cache=result_cache,
        rec_outcome=_rec_outcome,
        nac=_loop_nac,
        memory_hub_enabled=memory_hub_enabled,
    )
    _book_refusal = _bind_book_refusal(ctrl, agent_id=_loop_agent_id, rec_outcome=_rec_outcome, nac=_loop_nac)

    return LoopRun(
        executor=executor,
        sim=sim,
        run_id=run_id,
        agent_name=agent_name,
        state_path=state_path,
        autonomy_controller=autonomy_controller,
        context_pool=context_pool,
        prefetcher=prefetcher,
        result_cache=result_cache,
        ctrl=ctrl,
        is_novel_thought=is_novel_thought,
        max_response_tokens_override=_max_response_tokens_override,
        max_cycles_override=_max_cycles_override,
        dn_enabled=dn_enabled,
        nac=_loop_nac,
        xclock=_loop_xclock,
        agent_id=_loop_agent_id,
        drive_relief_only=_drive_relief_only,
        rec_outcome=_rec_outcome,
        sensor_encoder=_loop_sensor_encoder,
        situation_cue=_loop_situation_cue,
        memory_hub_enabled=memory_hub_enabled,
        planning_liveness_on=_planning_liveness_on,
        execute_and_learn=_execute_and_learn,
        book_refusal=_book_refusal,
    )
