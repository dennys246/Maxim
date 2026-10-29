"""Bio-system integration helpers for the agent loop.

Extracted from agent_loop.py for single-responsibility decomposition.
Handles hippocampus episodic capture, MemoryHub session lifecycle,
and RPE-based salience boosting.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Literal

from maxim.integration.session_ownership import SessionTicket
from maxim.memory.experience_clock import ExperienceClockStalled
from maxim.utils.cancellation import current_stop
from maxim.utils.deadline import Deadline
from maxim.utils.logging import log_swallowed_exception
from maxim.utils.structured_logging import log_agentic

logger = logging.getLogger(__name__)


def capture_episodic_memory(
    *,
    hippocampus: Any,
    executor: Any,
    observation: Any,
    state: Any,
    intent: dict[str, Any],
    action: dict[str, Any],
    result: Any,
    run_id: str,
    situation: "dict[str, str] | None",
    novelty: float | None,
) -> None:
    """Capture an episodic memory to hippocampus with RPE salience boost.

    ``situation`` (REQUIRED, memory-strength Phase 2S-b): the loop's substrate clusters for this
    action, ``{modality: EC cluster id}``, or ``None`` when the path computed none. Recorded on the
    trace; ConceptExtractor links the clusters' ATL concepts (same ids) to it.

    ``novelty`` (REQUIRED, Phase 2S-c): how unfamiliar that situation was, in ``[0, 1]``
    (``situation_novelty`` of the proposal's EC match margins), or ``None`` when no
    margin was recorded.

    This is the standard per-action hippocampus capture pattern used
    in both the agent fallback path (Section 3) and the LLM execution
    path (Section 4) of the agent loop.

    Args:
        hippocampus: Hippocampus instance (must not be None).
        executor: Tool executor (unused for surprise since #847; kept for the call shape).
        observation: Current observation dict.
        state: Agent state object.
        intent: Intent dict (e.g. {"goal": "...", "source": "..."}).
        action: Action dict with "tool_name" and "params".
        result: Tool execution result.
        run_id: Current run identifier.
    """
    # Boost salience for a surprising outcome -- THIS action's (#847): the executor stamps the
    # invocation's |RPE| onto its ToolOutput. It used to read the executor's last-RPE slot, which
    # nothing reset, so an action with no surprise of its own inherited an earlier tool's.
    from maxim.memory.encoding import EncodingContractError, EncodingSignals
    from maxim.tools.base import ToolOutput

    rpe = result.rpe if isinstance(result, ToolOutput) else None
    # What this trace is encoded WITH (memory-strength Phase 2b), built before any fold or queue so
    # a bad value fails here, loudly. MEASURED on this path: the outcome's surprise, the action's
    # pain (the executor's stamp, Phase 2S-c), and the situation's novelty (from the EC margins the
    # proposal carried, Phase 2S-c). Salience stays unmeasured ON PURPOSE: the tag already scores
    # pain and drive pressure on their own, so a salience built from them would count them twice.
    # Drive pressure and relief ride on the executor's stamp, per drive.
    # ``rpe`` is in [0, 1] by construction (the Rescorla-Wagner value is bounded at every producer),
    # so a value outside it is a broken invariant and fails here, loudly.
    # Per-drive pressure (read BEFORE the action) and relief ride on the same stamp (Phase 2b-ii).
    # The body-consequence record (grounding GL2a) rides in ``extra``, which ``encoding_tag`` never
    # reads: it is persisted with the trace and changes nothing the trace encodes with.
    outcome = result.interoceptive_outcome if isinstance(result, ToolOutput) else None
    encoding = EncodingSignals(
        site="loop",
        salience=None,
        novelty=novelty,
        surprise=rpe,
        pain=result.pain if isinstance(result, ToolOutput) else None,
        drive_pressure=result.drive_pressure_before if isinstance(result, ToolOutput) else None,
        drive_relief=result.drive_relief if isinstance(result, ToolOutput) else None,
        extra={"interoception": outcome.to_dict()} if outcome is not None else {},
    )
    if rpe is not None and rpe > 0.0 and isinstance(observation, dict):
        current_salience = observation.get("salience", 0.5)
        observation["salience"] = min(1.0, current_salience + rpe * 0.5)
    try:
        hippocampus.capture_from_loop_async(
            observation=observation if isinstance(observation, dict) else {},
            state=state,
            intent=intent,
            decision={"action": action, "confidence": action.get("confidence", 1.0)},
            action={
                "tool": action.get("tool_name", action.get("tool", "")),
                "params": action.get("params", {}),
            },
            result=result,
            run_id=run_id or "",
            encoding=encoding,
            situation=situation,
        )
    except EncodingContractError:
        raise  # a capture-contract break is never a runtime hiccup
    except Exception as e:
        logger.debug("Hippocampus capture failed: %s", e)


def situation_novelty(margins: dict[str, float] | None) -> float | None:
    """How unfamiliar a situation was, in ``[0, 1]``: ``1 - best similarity`` of its clusters' encodes.

    Per modality ``1 - best_similarity`` (clamped), and the situation's novelty is its MOST novel
    modality. Dynamic range, stated: sensor completions sit at or above the 0.85 threshold, so a
    familiar situation reads ~0.0-0.15 and a separation reads higher -- a coarse, near-binary signal,
    not a graded one. ``-1.0`` ("nothing comparable": a fresh EC or a new modality) is NOT scored as
    maximal novelty: the tag weights novelty by a reference-set size that is still the Hippocampus's
    trace count, which would give an empty comparison full confidence -- so it is left unmeasured
    until the producer supplies its own reference set. ``None`` when no margin remains.
    """
    measured = [min(1.0, max(0.0, 1.0 - m)) for m in (margins or {}).values() if m >= 0.0]
    return max(measured) if measured else None


def capture_loop_action(
    hippocampus: Any,
    executor: Any,
    observation: Any,
    state: Any,
    intent: dict[str, Any],
    action: dict[str, Any],
    confidence: Any,
    result: Any,
    run_id: str | None,
    agent_id: str,
    *,
    proposal: Any,
) -> None:
    """Capture one executed action to the Hippocampus and close its episode step.

    Called by ``tool_dispatch.execute_and_learn`` (the autonomous and the SUPERVISED-confirmed paths) and by
    the loop's agent-fallback path. ``proposal`` (REQUIRED; ``None`` on the agent-fallback path, which has
    none; on the confirmed path ``PendingConfirmation.source``) carries WHERE the
    action was chosen and how unfamiliar that was:

    - the situation (memory-strength Phase 2S-b) is its ``clusters`` -- ``propose_via_substrate``'s
      in substrate-primary (encoded when the proposal was made, the tick before execution), or the
      outcome-time encode made before execution in llm-primary. It is the same key ``_rec_outcome``
      credits. Re-encoding at capture would read the POST-action state (the wrong key) and write the EC.
    - the novelty (Phase 2S-c) is ``situation_novelty`` of its ``cluster_margins``, recorded with
      those clusters at encode time.
    """
    if hippocampus is None:
        return
    capture_episodic_memory(
        hippocampus=hippocampus,
        executor=executor,
        observation=observation,
        state=state,
        intent=intent,
        action={
            "tool_name": action.get("tool_name"),
            "params": action.get("params", {}),
            "confidence": confidence,
        },
        result=result,
        run_id=run_id or "",
        situation=getattr(proposal, "clusters", None),
        novelty=situation_novelty(getattr(proposal, "cluster_margins", None)),
    )
    observe_episode(
        hippocampus=hippocampus,
        agent_id=agent_id,
        channel="text",
        activated_nodes=(),
        after_tool_execution=True,
        salience_spike=consume_pain_intensity(agent_id=agent_id),
    )


def record_plan_outcome(
    *,
    memory_hub: Any,
    goal: str,
    tool_name: str,
    success: bool,
) -> None:
    """Record a plan outcome in MemoryHub for causal learning.

    Args:
        memory_hub: MemoryHub instance (must not be None).
        goal: The goal/reasoning text (truncated to 200 chars).
        tool_name: Tool that was executed.
        success: Whether the tool succeeded.
    """
    try:
        memory_hub.record_plan_outcome(
            goal=goal[:200],
            tool_sequence=[tool_name],
            success=success,
        )
    except Exception as e:
        logger.debug("Failed to record plan outcome: %s", e)


ABNORMAL_EXIT_CAP_S = (
    30.0  # a session's abnormal-exit persist starts only while this (or the stop request's bound) lasts
)


@dataclass(eq=False)
class BioSession:
    """One agent loop's claim on its memory session (#1438 PR 0; docs/plans/bounded_stop.md §4).

    Built by :func:`bio_session` BEFORE ``build_loop_run``, so the Default Network, prefetcher and context pool start
    inside a claimed session. ``build_loop_run`` records the run's facts that ``__exit__`` needs: whether it is a sim
    (the consolidation flavor is RESOLVED at exit, since ``sim`` is built in setup) and how to stop the Default
    Network it started. In-process only.
    """

    memory_hub: Any | None
    hippocampus: Any | None
    ticket: SessionTicket | None  # None: no memory hub configured (the one soft skip)
    capture: Any | None = None  # the Hippocampus CaptureWorker this session started
    consolidation: Literal["full", "lightweight"] | None = None  # the caller's override, if any
    is_sim_mode: bool = False  # recorded by build_loop_run
    stop_default_network: Callable[[], None] | None = None  # recorded by build_loop_run when the DN started

    @property
    def memory_hub_enabled(self) -> bool:
        return self.ticket is not None

    def resolved_consolidation(self) -> Literal["full", "lightweight"]:
        """The caller's override first (``minecraft_harness`` and a persistent HANDLE pass "full"), then the sim."""
        if self.consolidation is not None:
            return self.consolidation
        return "lightweight" if self.is_sim_mode else "full"


@contextmanager
def bio_session(
    *,
    memory_hub: Any | None,
    hippocampus: Any | None,
    owner: Any,
    consolidation: Literal["full", "lightweight"] | None = None,
) -> Iterator[BioSession]:
    """The ONE seam that opens and ends an agent loop's memory session (#1438 PR 0).

    ``__enter__`` claims the hub's session for ``owner`` as a LIVE loop (``SessionHeld`` / ``SessionStillEnding``
    propagate as a typed abort of that loop) and starts this session's capture worker. ``__exit__``, on every path:
    stops the Default Network and the capture worker first, then ends the session as owner. On a clean exit the end
    uses the resolved flavor; on an abnormal one (any BaseException, KeyboardInterrupt included) it persists with the
    lightweight flavor, and only while the stop request's deadline (else a 30 s cap) has time left. The claim is
    released in a ``finally`` after those steps return. ``__exit__`` never replaces an in-flight exception: its own
    errors are logged and chained.
    """
    ticket = memory_hub.on_session_start(owner=owner, live=True) if memory_hub is not None else None
    # The claim is held from here: everything after it runs inside the try, so no raise can leak a live claim.
    session = BioSession(memory_hub=memory_hub, hippocampus=hippocampus, ticket=ticket, consolidation=consolidation)
    try:
        if ticket is not None:
            log_agentic("memory_hub", "session_start", ticket.start_stats, level="INFO")
        if hippocampus is not None:
            try:
                session.capture = hippocampus.start_capture_worker()
            except Exception:
                log_swallowed_exception(site="bio_integration.py:bio_session:capture_worker")
        yield session
    except BaseException:
        stop = current_stop()
        bound = stop.deadline if stop is not None else Deadline.after(ABNORMAL_EXIT_CAP_S)
        _close_session(session, consolidation="lightweight", bound=bound)
        raise
    else:
        _close_session(session, consolidation=session.resolved_consolidation(), bound=None)


def _close_session(
    session: BioSession, *, consolidation: Literal["full", "lightweight"], bound: Deadline | None
) -> None:
    """Stop the session's workers, then end it as owner. Never raises: an exit must not replace an in-flight error.

    The Default Network and this session's capture worker stop on EVERY path; only the persist is gated on the bound.
    A bound that expired before the persist could start abandons the generation (closed, nothing persisted), so the
    session never stays OPEN with nobody able to end it."""
    try:
        if session.stop_default_network is not None:
            try:
                session.stop_default_network()
            except Exception:
                log_swallowed_exception(site="bio_integration.py:_close_session:stop_default_network")
        if bound is not None and bound.expired():
            if session.capture is not None:
                session.capture.stop.set()  # no flush: the bound is spent
            if session.memory_hub is not None and session.ticket is not None:
                session.memory_hub.abandon_session(session.ticket)
            logger.warning("Session end skipped: the stop bound expired before the persist could start")
            return
        end_bio_session(
            memory_hub=session.memory_hub,
            ticket=session.ticket,
            hippocampus=session.hippocampus,
            capture=session.capture,
            consolidation=consolidation,
        )
    finally:
        if session.memory_hub is not None and session.ticket is not None:
            session.memory_hub.release_session_claim(session.ticket)


def end_bio_session(
    *,
    memory_hub: Any | None,
    ticket: SessionTicket | None,
    hippocampus: Any | None,
    capture: Any | None = None,
    consolidation: Literal["full", "lightweight"] = "full",
) -> None:
    """Drain and stop this session's capture worker, save the hippocampus, and end the MemoryHub session as owner.

    The consolidation flavor is an **explicit choice** (HANDLE seam, part b): ``"full"`` runs the blocking
    sleep/replay + bridge consolidation (``on_session_end``); ``"lightweight"`` persists NAc decay + embeddings +
    subsystem state but skips the expensive replay (``on_session_end_lightweight``). **Default is ``"full"``**: wrongly
    lightweight loses consolidation silently; wrongly full is a slightly slow shutdown that is loud and harmless.

    ``ticket`` is the session's (``bio_session``); ``None`` means no hub session to end. ``capture`` is the worker this
    session started; only it is stopped, never another loop's on the same Hippocampus.
    """
    use_lightweight = consolidation == "lightweight"
    # Drain async capture queue and save hippocampus
    if hippocampus is not None:
        try:
            hippocampus.flush(timeout=5.0)
            hippocampus.stop_capture_worker(capture)
        except Exception as e:
            logger.warning("Failed to flush/stop the hippocampus capture worker: %s", e)
        try:
            if hasattr(hippocampus, "config") and hippocampus.config.persistence_path:
                hippocampus.save()
                log_agentic("hippocampus", "saved", {"memories": len(hippocampus)}, level="INFO")
        except FileExistsError as e:  # StoreOverwriteRefused (#939): never at DEBUG, the memories are unsaved
            logger.error("Hippocampus not saved at session end: %s", e)
        except Exception as e:
            logger.warning("Failed to save hippocampus: %s", e)

    # End the MemoryHub session as its owner. Sim mode skips hippocampus.sleep() (expensive replay) but still
    # persists NAc decay, semantic embeddings, and subsystem state so learning is not lost.
    if memory_hub is not None and ticket is not None:
        try:
            if use_lightweight:
                session_stats = memory_hub.on_session_end_lightweight(ticket)
            else:
                session_stats = memory_hub.on_session_end(ticket)
            log_agentic(
                "memory_hub",
                "session_end",
                session_stats,
                level="INFO",
            )
        except ExperienceClockStalled as e:
            # Named BEFORE the broad handler (memory-strength 2c-3 review round, cross-confirmed):
            # this is the one session-end failure that means the run's memory model silently did
            # nothing, and the generic branch below would hide it. The
            # session's own stats ride on the exception precisely so the diagnostic does not also
            # cost us the telemetry line.
            logger.error("MemoryHub session ended on a stalled experience clock: %s", e)
            log_agentic("memory_hub", "session_end", e.results or {}, level="INFO")
        except Exception as e:
            logger.warning("Failed to end MemoryHub session: %s", e)


# ── Episode observation (P3a binding in production) ──────────────────
#
# Per-agent stash dicts. Replaces the previous module-level globals
# (``_episode_tick``, ``_latest_pain_intensity``, ``_latest_substrate_nodes``)
# which trampled under concurrent agents — every agent shared one
# tick counter, one pain intensity slot, and one substrate-nodes slot,
# so substrate nodes encoded for agent A could land on agent B's next
# episode event, and every agent's tick number was a global ordering
# of all events instead of per-agent.
#
# Concurrency: dict ``__getitem__``/``__setitem__``/``pop`` on a single
# key are GIL-atomic in CPython — read-modify-write on a numeric value
# (``_latest_pain_intensity``'s "max" merge) is the only RMW path and
# happens while a single agent is producing pain signals on its own
# pain bus, so contention on a single agent_id is negligible. Multi-
# agent isolation is by key; concurrency on the dict structure itself
# is GIL-protected and sized at most O(num_agents).

_episode_ticks: dict[str, int] = {}
_latest_pain_intensity: dict[str, float] = {}
_latest_substrate_nodes: dict[str, tuple[str, ...]] = {}


def _check_agent_id(agent_id: str) -> None:
    """Reject empty / non-string agent_id at the entry point.

    The whole P4 fix is structural enforcement: forgetting an
    ``agent_id`` is a ``TypeError`` (missing kwarg) instead of a
    silent cross-agent attribution bug.  An empty string slips past
    the missing-kwarg check, routes every "agent" through one shared
    ``""`` key, and re-introduces the bug class.  Reject it loudly.
    Pre-merge architecture review caught this as the same band-aid
    pattern P4 was supposed to eliminate.
    """
    if not isinstance(agent_id, str) or not agent_id:
        raise ValueError(
            f"agent_id must be a non-empty string, got {agent_id!r}. "
            "Bio-integration stash entries are keyed by agent — empty / "
            "missing values would silently merge attribution across agents."
        )


# Lock guarding the per-agent pain intensity max-merge.  The merge is a
# read-modify-write (``current = get(); if intensity > current: set``)
# and is NOT atomic under the GIL in CPython 3.11+ (specialised
# bytecode breaks the "single bytecode" assumption).  Today there are
# no production callers of ``record_pain_intensity`` — the bus path
# bypasses this stash — so the lock is a future-proofing net for when
# a producer wires up.  Pre-merge architecture review flagged the
# original docstring's "GIL makes RMW safe" claim as incorrect.
import threading as _threading

_pain_intensity_lock = _threading.Lock()


def record_substrate_nodes(node_ids: tuple[str, ...], *, agent_id: str) -> None:
    """Stash substrate node IDs from the latest percept encoding.

    **Dormant as a BINDING source since 2026-08-29 (bugs ledger D6).** The
    llm-primary percept path stashes a 1-tuple, and ``apply_hebbian_on_close``
    returns early below two nodes, so episode binding never grows from percepts in
    production. The stash itself is live (episodes carry the node ids); it is the
    multi-node case that is dormant. Resurrection trigger (re-pointed 2026-10-08,
    as in ``memory/episode.py::apply_hebbian_on_close``; the fabric is deferred on a
    second robot body): the 1.4 grounding line's binding stage
    (``docs/plans/grounding.md`` GL4), or the fabric's orient-windowed binding if it
    revives first. Do not build new behaviour on percept-driven binding until then
    (roadmap item 13).

    Called by MemoryHub.on_percept_received after LinguisticEncoder
    produces substrate_node_id(s). Consumed by the next observe_episode
    call, which passes them as activated_nodes to the CaptureEvent.

    This bridges the encoding path (memory_hub → encoder) to the episode
    observation path (agent_loop → bio_integration → hippocampus).

    ``agent_id`` is required and must be non-empty — multiple agents
    running in parallel (AgentPool, sim AUT + orchestrator pair,
    agent-backed entities) must NOT trample each other's stash.
    """
    _check_agent_id(agent_id)
    _latest_substrate_nodes[agent_id] = node_ids


def consume_substrate_nodes(*, agent_id: str) -> tuple[str, ...]:
    """Consume and reset the stashed substrate node IDs for ``agent_id``."""
    _check_agent_id(agent_id)
    return _latest_substrate_nodes.pop(agent_id, ())


def observe_episode(
    *,
    hippocampus: Any,
    agent_id: str,
    channel: str = "text",
    sender_id: str | None = None,
    activated_nodes: tuple[str, ...] = (),
    salience_spike: float | None = None,
    after_tool_execution: bool = False,
) -> None:
    """Feed an event into the episode boundary detector.

    Called from the agent loop alongside capture_episodic_memory.
    If the caller passes empty activated_nodes, we consume any
    stashed substrate nodes from the latest percept encoding for
    this ``agent_id``.
    """
    _check_agent_id(agent_id)
    tick = _episode_ticks.get(agent_id, 0) + 1
    _episode_ticks[agent_id] = tick

    # Merge caller-provided nodes with stashed substrate nodes (per agent)
    if not activated_nodes:
        activated_nodes = consume_substrate_nodes(agent_id=agent_id)

    try:
        from maxim.memory.episode import CaptureEvent

        # TODO(v1-p2-producer): wire ``embedding=`` from the substrate-path
        # encoder so ``semantic_shift_rule`` has input. Today the rule is
        # opt-in (not in the default detector) and dormant infrastructure;
        # adding this CaptureEvent producer + installing the rule on the
        # default detector is the follow-up that activates topical-drift
        # episode boundaries. The encoder layer
        # (``MemoryHub.on_percept_received`` → ``LinguisticEncoder.encode``)
        # already produces a per-percept embedding; pass it through a
        # per-agent stash analogous to ``_latest_substrate_nodes`` and
        # consume it here.
        hippocampus.observe_episode_event(
            CaptureEvent(
                tick=tick,
                channel=channel,
                sender_id=sender_id,
                activated_nodes=activated_nodes,
                after_tool_execution=after_tool_execution,
                salience_spike=salience_spike,
            )
        )
    except Exception as e:
        logger.debug("Episode observation failed: %s", e)


def record_pain_intensity(intensity: float, *, agent_id: str) -> None:
    """Dormant since 2026-05-26: the bus path bypasses this stash — no
    production producer wires up. The per-agent lock + validator + max-merge
    semantics remain for a future producer that wants to feed
    ``salience_spike`` from a non-bus source. Awaits a new experiment that
    earns the alternative path back in.

    Record a pain intensity for the next episode event's salience_spike.

    Per-agent: signals from agent A's pain bus must not land on
    agent B's next episode.  Read-modify-write is serialised by an
    internal lock so concurrent producers don't drop a higher
    intensity.
    """
    _check_agent_id(agent_id)
    with _pain_intensity_lock:
        current = _latest_pain_intensity.get(agent_id, 0.0)
        if intensity > current:
            _latest_pain_intensity[agent_id] = intensity


def consume_pain_intensity(*, agent_id: str) -> float | None:
    """Consume the recorded pain intensity for ``agent_id`` (reset to 0)."""
    _check_agent_id(agent_id)
    with _pain_intensity_lock:
        val = _latest_pain_intensity.pop(agent_id, 0.0)
    return val if val > 0.0 else None


def reset_agent_stash(agent_id: str) -> None:
    """Drop all per-agent stash entries. Intended for test isolation
    and end-of-session cleanup."""
    _episode_ticks.pop(agent_id, None)
    _latest_pain_intensity.pop(agent_id, None)
    _latest_substrate_nodes.pop(agent_id, None)
