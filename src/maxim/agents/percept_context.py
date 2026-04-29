"""PerceptContext — typed framing carried on every Percept.

F0.4 replaces the ad-hoc ``Percept.metadata`` dict (for messaging-style
framing data) with a typed :class:`PerceptContext` struct. The dict is
retained on :class:`~maxim.agents.bus.Percept` as an escape hatch for
legacy passthrough (pain signals, YAML scenario extras), but new
messaging framing — channel, sender, thread, subject, latency, circadian
tag, agent_id — flows through this typed surface so every consumer can
make well-typed decisions instead of metadata dict lookups with default
fallbacks.

Isolation hygiene — BAKE THIS RULE INTO REVIEW DISCIPLINE
----------------------------------------------------------

A percept produced by Agent A must be safe to deliver verbatim to
Agent B without changing Agent B's learning trajectory beyond what a
real sensor reading on that channel would do. Concretely, this means
``PerceptContext`` MUST NOT carry fields that encode:

- **Cross-agent intent.** No ``mother_intent``, no ``narrator_goal``,
  no ``lesson_to_teach``. A Mother NPC that wants Baby to learn a
  concept has to earn it via the percept's *content*, not by stashing
  a hint in the context.
- **Private state of another agent.** No ``sender_internal_state``,
  no ``peer_reward_hint``, no ``upstream_confidence``. Agents see each
  other as black boxes across the percept channel.
- **Scenario/test oracles.** No ``scenario_answer``, no ``expected_flag``,
  no ``test_mode``. Scenario tagging that only matters for post-hoc
  analysis belongs in ``Percept.metadata`` (the escape hatch), not in
  ``PerceptContext``.
- **Goal hints.** No ``active_goal_summary``, no ``suggested_response``.
  The receiving agent computes goal relevance from its own state.

Adding a new field to ``PerceptContext`` is a deliberate act that needs
reviewer sign-off on the isolation question: *could a malicious (or
careless) producer use this field to leak information into a
downstream learner beyond what a real sensor would carry?* If the
answer is "yes" or "maybe," the field does not belong here.

The deferred ``mother_npc_stimulus_plan`` makes this rule load-bearing
for the multi-agent training surface at revive time. F0.5 adds
``agent_id`` as a scoping key so per-agent state stays isolated; this
module's rule is the complementary *content* constraint.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from maxim.time.circadian import CircadianContext

# Allowed communication channels. Keep this list small and curated —
# a new channel means a new producer, which means a new review surface.
Channel = Literal[
    "sms",
    "email",
    "slack",
    "narrative",  # Narrator/DM text in simulations
    "speech",  # Spoken audio (TTS or STT-transcribed)
    "self",  # Internal monologue / self-speech
    "internal",  # System-internal signals (sensor telemetry, heartbeat)
    "mesh",  # Inter-agent message over the mesh channel
]

# Rough expected latency class of the channel. Consumers can use this
# to decide how aggressively to interrupt the current agent loop.
LatencyClass = Literal["realtime", "seconds", "minutes", "hours", "days"]

# Modality is declared on Percept itself; the literal type lives here so
# producers import a single place.
Modality = Literal["text", "vision", "audio", "intero"]


@dataclass(frozen=True)
class PerceptContext:
    """Typed framing carried on every Percept.

    All fields are optional so existing construction sites can adopt
    the schema incrementally. Sites that do not have messaging semantics
    (e.g. vision detection, proprioceptive pain) can leave the whole
    context ``None`` on the Percept and rely on the escape-hatch
    metadata dict for non-messaging attributes.

    See the module docstring for the isolation-hygiene rule governing
    what fields may be added here.

    SHAPE-FROZEN at 1.0 (CC3). The percept-level escape hatch is
    ``Percept.metadata`` (free-form dict on the carrying ``Percept``
    instance, not on this struct) — DO NOT add an ``extra`` dict here,
    that would re-open the cross-agent leakage path the isolation-
    hygiene rule above forbids. New fields appended with defaults are
    non-breaking but require isolation review; adding a *required*
    field post-1.0 is a major-version-bump change.
    """

    channel: Channel | None = None
    sender: str | None = None  # contact ID, NPC ID, "self", "narrator"
    thread_id: str | None = None
    subject: str | None = None
    timestamp: float | None = None  # monotonic intra-session, wall cross-session
    latency_class: LatencyClass | None = None
    scn_tag: CircadianContext | None = None
    # agent_id is populated by F0.5 (agent_id threading). Until then,
    # producers leave it None and downstream consumers must tolerate it.
    agent_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "sender": self.sender,
            "thread_id": self.thread_id,
            "subject": self.subject,
            "timestamp": self.timestamp,
            "latency_class": self.latency_class,
            "scn_tag": self.scn_tag.to_dict() if self.scn_tag is not None else None,
            "agent_id": self.agent_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PerceptContext":
        scn_raw = data.get("scn_tag")
        return cls(
            channel=data.get("channel"),
            sender=data.get("sender"),
            thread_id=data.get("thread_id"),
            subject=data.get("subject"),
            timestamp=data.get("timestamp"),
            latency_class=data.get("latency_class"),
            scn_tag=CircadianContext.from_dict(scn_raw) if scn_raw else None,
            agent_id=data.get("agent_id"),
        )
