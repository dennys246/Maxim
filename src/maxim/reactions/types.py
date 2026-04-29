"""Core Reaction types — typed evaluative signals that drive learning.

Reactions are the middle layer between Percepts (sensory input) and
bio-systems (memory, learning, decision). A Percept says "I observed X";
a Reaction says "X caused an internal state change that should drive
learning." Every reaction producer writes to this typed surface; every
consumer (NAc, fear circuit, goal interruption) reads from it.

Isolation hygiene — BAKE THIS RULE INTO REVIEW DISCIPLINE
----------------------------------------------------------

A Reaction produced by Agent A must be safe to deliver to Agent B
without changing B's learning trajectory beyond what a real evaluative
signal on that kind would do. Concretely, ``ReactionContext`` MUST NOT
carry fields that encode:

- **Cross-agent intent.** No ``mother_lesson``, no
  ``narrator_desired_avoidance``. If Mother wants Baby to learn to
  avoid fire, Mother produces a percept of fire; Baby's own pain
  system generates the avoidance Reaction. The Reaction is a function
  of Baby's internal state + recent percepts, not Mother's goals.
- **Private state of another agent.** No ``sender_reward_history``,
  no ``peer_pain_threshold``. Agents see each other as black boxes
  across both the Percept and Reaction surfaces.
- **Scenario/test oracles.** No ``expected_reaction_kind``, no
  ``correct_avoidance_target``. Scenario tagging for post-hoc analysis
  belongs in ``Percept.metadata`` (the escape hatch), not in
  ``ReactionContext``.
- **Learned-policy hints.** No ``suggested_response``, no
  ``optimal_action_from_nac``. The receiving agent's NAc computes its
  own policy from its own causal links.

Adding a new field to ``ReactionContext`` is a deliberate act that needs
reviewer sign-off on the isolation question: *could a malicious (or
careless) producer use this field to leak information into a downstream
learner beyond what a real evaluative signal would carry?* If the answer
is "yes" or "maybe," the field does not belong here.

This rule is the Reaction-side complement to PerceptContext's rule.
Together they define the **information barrier** for the deferred
``mother_npc_stimulus_plan``: Mother communicates with Baby through
Percept content only; Baby generates Reactions from its own bio-stack
only. No back-channel through either typed surface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from maxim.decisions.causal_link import Valence
from maxim.time.circadian import CircadianContext

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

ReactionKind = Literal["pain", "fear", "hunger", "surprise", "fatigue", "satiation"]


# ---------------------------------------------------------------------------
# TraceSnapshot
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TraceSnapshot:
    """A typed reference to a percept that was active when a reaction fired.

    Populated from the ``PerceptTraceBuffer`` (F0.2) at emission time.
    Before F0.2 lands, construct manually with ``activation_strength=1.0``
    and ``decay_factor=1.0`` (full-strength, no decay information).

    SHAPE-FROZEN at 1.0 (CC3). No ``extra`` escape hatch by design —
    ``TraceSnapshot`` is reachable from every Reaction via
    :class:`ReactionContext` ``bindings: dict[str, TraceSnapshot]``, so a
    free-form dict here would re-open the isolation back-channel that
    ``ReactionContext`` explicitly closes (cross-agent intent, scenario
    oracles, learned-policy hints). The pre-merge architecture review
    flagged the original ``extra`` draft as exactly this leak path.
    Adding any new field is a deliberate act requiring isolation review
    per the module docstring; adding a *required* field post-1.0 is a
    major-version-bump change.
    """

    percept_id: str
    activation_strength: float = 1.0
    content_hash: str | None = None
    decay_factor: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "percept_id": self.percept_id,
            "activation_strength": self.activation_strength,
            "content_hash": self.content_hash,
            "decay_factor": self.decay_factor,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TraceSnapshot":
        return cls(
            percept_id=data["percept_id"],
            activation_strength=data.get("activation_strength", 1.0),
            content_hash=data.get("content_hash"),
            decay_factor=data.get("decay_factor", 1.0),
        )


# ---------------------------------------------------------------------------
# ReactionContext
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReactionContext:
    """Typed context carried on every Reaction.

    Parallel to :class:`~maxim.agents.percept_context.PerceptContext`.
    Carries binding information that lets consumers attribute a reaction
    to specific percepts and actions.

    All fields are optional so producers can adopt the schema
    incrementally. See the module docstring for the isolation-hygiene
    rule governing what fields may be added here.

    SHAPE-FROZEN at 1.0 (CC3). No ``extra`` escape hatch by design — a
    free-form dict here would re-open the back-channel the
    isolation-hygiene rule forbids. Adding any new field is a deliberate
    act requiring isolation review; adding a *required* field post-1.0
    is a major-version-bump change.
    """

    agent_id: str | None = None
    timestamp: float | None = None
    scn_tag: CircadianContext | None = None
    bindings: dict[str, TraceSnapshot] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "timestamp": self.timestamp,
            "scn_tag": self.scn_tag.to_dict() if self.scn_tag is not None else None,
            "bindings": {k: v.to_dict() for k, v in self.bindings.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReactionContext":
        scn_raw = data.get("scn_tag")
        bindings_raw = data.get("bindings", {})
        return cls(
            agent_id=data.get("agent_id"),
            timestamp=data.get("timestamp"),
            scn_tag=CircadianContext.from_dict(scn_raw) if scn_raw else None,
            bindings={k: TraceSnapshot.from_dict(v) for k, v in bindings_raw.items()},
        )


# ---------------------------------------------------------------------------
# Reaction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Reaction:
    """A typed evaluative signal that drives learning.

    Frozen so producers cannot mutate in-flight. ``kind`` is a
    :data:`ReactionKind` literal for extensibility — new reaction kinds
    don't require code changes, just documentation.

    SHAPE-FROZEN at 1.0 (CC3). Every field is load-bearing for the
    typed-evaluative-signal contract; an ``extra`` dict would dilute
    the type by inviting producers to stash unstructured side data. New
    fields appended at the end with sensible defaults are non-breaking
    (additive) but still require isolation review per the module
    docstring. Adding a *required* field post-1.0 is a
    major-version-bump change.
    """

    kind: ReactionKind
    intensity: float  # 0.0-1.0
    valence: Valence
    timestamp: float
    context: ReactionContext
    source: str  # e.g. "cerebellum:motor_failure", "pain_detector:velocity"

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "intensity": self.intensity,
            "valence": self.valence.value,
            "timestamp": self.timestamp,
            "context": self.context.to_dict(),
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Reaction":
        return cls(
            kind=data["kind"],
            intensity=data["intensity"],
            valence=Valence(data["valence"]),
            timestamp=data["timestamp"],
            context=ReactionContext.from_dict(data["context"]),
            source=data["source"],
        )
