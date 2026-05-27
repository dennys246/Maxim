"""Bio-system integration helpers for the agent loop.

Extracted from agent_loop.py for single-responsibility decomposition.
Handles hippocampus episodic capture, MemoryHub session lifecycle,
and RPE-based salience boosting.
"""

from __future__ import annotations

import logging
from typing import Any

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
) -> None:
    """Capture an episodic memory to hippocampus with RPE salience boost.

    This is the standard per-action hippocampus capture pattern used
    in both the agent fallback path (Section 3) and the LLM execution
    path (Section 4) of the agent loop.

    Args:
        hippocampus: Hippocampus instance (must not be None).
        executor: Tool executor (checked for get_last_rpe).
        observation: Current observation dict.
        state: Agent state object.
        intent: Intent dict (e.g. {"goal": "...", "source": "..."}).
        action: Action dict with "tool_name" and "params".
        result: Tool execution result.
        run_id: Current run identifier.
    """
    # Boost salience for surprising outcomes (high RPE)
    if hasattr(executor, "get_last_rpe"):
        rpe = executor.get_last_rpe()
        if rpe > 0.0 and isinstance(observation, dict):
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
        )
    except Exception as e:
        logger.debug("Hippocampus capture failed: %s", e)


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


def start_bio_session(
    *,
    memory_hub: Any | None,
    hippocampus: Any | None,
) -> bool:
    """Initialize MemoryHub session and start hippocampus capture worker.

    Returns True if memory_hub was successfully started, False otherwise.
    """
    memory_hub_enabled = memory_hub is not None
    if memory_hub_enabled:
        try:
            startup_stats = memory_hub.on_session_start()
            log_agentic(
                "memory_hub",
                "session_start",
                startup_stats,
                level="INFO",
            )
        except Exception as e:
            logger.warning("Failed to start MemoryHub session: %s", e)
            memory_hub_enabled = False

    if hippocampus is not None:
        try:
            hippocampus.start_capture_worker()
        except Exception as e:
            logger.debug("Failed to start hippocampus capture worker: %s", e)

    return memory_hub_enabled


def end_bio_session(
    *,
    memory_hub: Any | None,
    memory_hub_enabled: bool,
    hippocampus: Any | None,
    is_sim_mode: bool,
) -> None:
    """Drain hippocampus capture queue and end MemoryHub session.

    Args:
        memory_hub: MemoryHub instance (may be None).
        memory_hub_enabled: Whether the session was started successfully.
        hippocampus: Hippocampus instance (may be None).
        is_sim_mode: If True, skip MemoryHub session_end (avoids blocking consolidation).
    """
    # Drain async capture queue and save hippocampus
    if hippocampus is not None:
        try:
            hippocampus.flush(timeout=5.0)
            hippocampus.stop_capture_worker()
        except Exception as e:
            logger.debug("Failed to flush hippocampus: %s", e)
        try:
            if hasattr(hippocampus, "config") and hippocampus.config.persistence_path:
                hippocampus.save()
                log_agentic("hippocampus", "saved", {"memories": len(hippocampus)}, level="INFO")
        except Exception as e:
            logger.debug("Failed to save hippocampus: %s", e)

    # End MemoryHub session — full consolidation for non-sim, lightweight for sim.
    # Sim mode skips hippocampus.sleep() (expensive replay) but still persists
    # NAc decay, semantic embeddings, and subsystem state so learning is not lost.
    if memory_hub_enabled and memory_hub is not None:
        try:
            if is_sim_mode:
                session_stats = memory_hub.on_session_end_lightweight()
            else:
                session_stats = memory_hub.on_session_end()
            log_agentic(
                "memory_hub",
                "session_end",
                session_stats,
                level="INFO",
            )
        except Exception as e:
            logger.debug("Failed to end MemoryHub session: %s", e)


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
