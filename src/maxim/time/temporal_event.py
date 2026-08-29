"""TemporalEvent — temporally-anchored event envelope for credit attribution.

Signal sources (tool outcomes, pain, reactions, percepts, deliberation events)
emit ``TemporalEvent`` instead of making ad-hoc ``TemporalSignature.now()`` +
``SCN.register()`` + ``NAc.update_eligibility()`` calls separately.

Consumed by ``TemporalCreditDistributor`` (decisions/temporal_credit.py) for
centralized temporal credit attribution and SCN registration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from maxim.time.temporal_signature import TemporalSignature


@dataclass(frozen=True, slots=True)
class TemporalEvent:
    """A temporally-anchored event for credit attribution.

    Emitted by signal sources.  Consumed by TemporalCreditDistributor
    for NAc temporal credit and SCN temporal indexing.
    """

    event_id: str
    """Unique event identifier (typically uuid4().hex)."""

    event_type: str
    """Signal source category: "tool", "pain", "reaction", "percept",
    "affordance", "deliberation".

    ``"drive"`` is emitted by ``embodiment/body.py::_emit_drive_temporal_event``
    and is PROVISIONAL: that producer is dormant (bugs ledger D9 — nothing in
    production passes it a distributor), so the category is recorded here for
    honesty rather than blessed. Whether drive transitions get their own category
    is decided by the 1.3 revival (`docs/plans/deferred/scn_event_producer_gap.md`),
    not by the emitter."""

    event_signature: str
    """Hashable identifier for NAc lookup (tool name,
    "deliberation:goal:<tag>", etc.)."""

    agent_id: str
    """Agent scope for per-agent isolation."""

    temporal_sig: TemporalSignature
    """SCN phase at emission time."""

    activation: float = 1.0
    """Signal strength (1.0 for direct, <1.0 for indirect)."""

    context: dict[str, Any] = field(default_factory=dict)
    """Event-specific metadata (e.g., {"goal": goal_tag, "cycle": N})."""
