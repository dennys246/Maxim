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

    # End MemoryHub session (runs sleep consolidation and bridge cleanup)
    # Skip session_end in simulation mode — it runs consolidation which
    # can block for a long time and we'll start a new turn immediately
    if memory_hub_enabled and memory_hub is not None and not is_sim_mode:
        try:
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

_episode_tick: int = 0
_latest_pain_intensity: float = 0.0


def observe_episode(
    *,
    hippocampus: Any,
    channel: str = "text",
    sender_id: str | None = None,
    activated_nodes: tuple[str, ...] = (),
    salience_spike: float | None = None,
) -> None:
    """Feed an event into the episode boundary detector.

    Called from the agent loop alongside capture_episodic_memory.
    """
    global _episode_tick
    _episode_tick += 1

    try:
        from maxim.memory.episode import CaptureEvent

        hippocampus.observe_episode_event(
            CaptureEvent(
                tick=_episode_tick,
                channel=channel,
                sender_id=sender_id,
                activated_nodes=activated_nodes,
                salience_spike=salience_spike,
            )
        )
    except Exception as e:
        logger.debug("Episode observation failed: %s", e)


def record_pain_intensity(intensity: float) -> None:
    """Record a pain intensity for the next episode event's salience_spike."""
    global _latest_pain_intensity
    if intensity > _latest_pain_intensity:
        _latest_pain_intensity = intensity


def consume_pain_intensity() -> float | None:
    """Consume the recorded pain intensity (reset to 0)."""
    global _latest_pain_intensity
    val = _latest_pain_intensity
    _latest_pain_intensity = 0.0
    return val if val > 0.0 else None
