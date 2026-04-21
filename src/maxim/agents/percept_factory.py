"""Named Percept factories — canonical constructors for common percept types.

Phase 4 of reaction_abstraction_plan. Every Percept produced in the
codebase should flow through one of these factories so context, modality,
and agent_id are populated uniformly. F0.6 (Percept factory consolidation)
is absorbed into this module.

The factories are functions, not classes. They return Percept instances
with the right PerceptContext and modality set. Callers that need a
PerceptProducer-shaped object (next_percept() → Percept | None) wrap
the factory in a polling adapter — the factory itself is stateless.
"""

from __future__ import annotations

import time
from typing import Any

from maxim.agents.bus import Percept
from maxim.agents.modality import SensoryModality, SensoryTag
from maxim.agents.percept_context import PerceptContext


def _log_percept(source: str, content: str | None, modality: str) -> None:
    """Best-effort sim_percept call. No-op outside simulation context."""
    try:
        from maxim.simulation.sim_logger import sim_percept

        sim_percept(source, (content or "")[:200], modality=modality)
    except Exception:
        pass


def make_text_percept(
    text: str,
    *,
    source: str = "text",
    agent_id: str | None = None,
    channel: str | None = None,
    sender: str | None = None,
    salience: float = 0.5,
    novelty: float = 0.3,
    metadata: dict[str, Any] | None = None,
    sensory: SensoryTag | None = None,
) -> Percept:
    """Create a text-modality Percept with proper context.

    This is the canonical factory for string-based stimuli: CLI input,
    NPC scene text, narrative, conversation archives. Both MaximAgent
    (single-agent) and AgentPool (multi-agent) use this to produce
    typed Percepts from string input.
    """
    ts = time.time()
    ctx = PerceptContext(
        channel=channel,  # type: ignore[arg-type]
        sender=sender,
        timestamp=ts,
        agent_id=agent_id,
    )
    if sensory is None:
        sensory = SensoryTag(modality=SensoryModality.NARRATIVE)
    _log_percept(source, text, "text")
    return Percept(
        timestamp=ts,
        source=source,
        content=text,
        salience=salience,
        novelty=novelty,
        context=ctx,
        modality="text",
        metadata=metadata,
        sensory=sensory,
    )


def make_scene_percept(
    scene_text: str,
    *,
    agent_id: str | None = None,
    salience: float = 0.6,
    metadata: dict[str, Any] | None = None,
    sensory: SensoryTag | None = None,
) -> Percept:
    """Create a narrative/scene Percept for DM campaign encounters."""
    ts = time.time()
    ctx = PerceptContext(
        channel="narrative",
        timestamp=ts,
        agent_id=agent_id,
    )
    if sensory is None:
        sensory = SensoryTag(modality=SensoryModality.NARRATIVE)
    _log_percept("narrative", scene_text, "text")
    return Percept(
        timestamp=ts,
        source="narrative",
        content=scene_text,
        salience=salience,
        novelty=0.5,
        context=ctx,
        modality="text",
        metadata=metadata,
        sensory=sensory,
    )


def make_intero_percept(
    content: str,
    *,
    source: str = "embodiment",
    agent_id: str | None = None,
    salience: float = 0.3,
    novelty: float = 0.2,
    metadata: dict[str, Any] | None = None,
    sensory: SensoryTag | None = None,
) -> Percept:
    """Create an interoceptive Percept (body state, vital metrics)."""
    ts = time.time()
    ctx = PerceptContext(
        channel="internal",
        timestamp=ts,
        agent_id=agent_id,
    )
    if sensory is None:
        sensory = SensoryTag(modality=SensoryModality.INTEROCEPTION)
    _log_percept(source, content, "intero")
    return Percept(
        timestamp=ts,
        source=source,
        content=content,
        salience=salience,
        novelty=novelty,
        context=ctx,
        modality="intero",
        metadata=metadata,
        sensory=sensory,
    )
