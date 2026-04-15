"""P3a Stage 1 — Episode, EpisodeStore, and episode boundary detection.

An ``Episode`` is an immutable multi-event time window during which
substrate nodes co-activated. On episode close, Hippocampus applies
Hebbian updates to its own binding graph (``Hippocampus._binding_graph``,
a ``DependencyGraph`` instance — **NOT** ``ATL.graph``); presenting one
node from a prior episode as a cue later retrieves the others via
``Hippocampus.retrieve_on_cue``.

Orthogonal to other "episode" concepts in the codebase:

- ``memory.types.EpisodicMemory`` — a single loop cycle (perception →
  decide → act → outcome). Fine-grained, one-per-tick.
- Simulation "episodes" in ``simulation/`` — campaign runs in a
  different domain.

These are distinct concepts; P3a's ``Episode`` is a new type.

Binding graph ownership — architectural rationale
==================================================

The P3a Round 1 pre-merge review (Architecture lens) flagged that
putting Hebbian edges on ``ATL.graph.ASSOCIATES`` would couple the
binding mechanism to ATL's concept eviction + compression lifecycle —
compressing an ATL concept would silently destroy any Hebbian edge
touching it. Deferring that question to Stage 3 was rejected as the
band-aid class CLAUDE.md forbids.

Resolution: Hebbian edges live on a separate ``DependencyGraph``
instance owned by ``Hippocampus`` (``self._binding_graph``). This:

- Still reuses the existing ``DependencyGraph`` utility class + the
  ``EdgeType.ASSOCIATES`` edge type — the split-proposal audit's
  "no new infrastructure" intent is preserved.
- Decouples the binding layer from ATL lifecycle entirely. ATL
  compression no longer affects binding weights because binding edges
  are keyed on stable substrate node IDs (per P1+P2 encoder
  invariants).
- Establishes a clean architectural split: ``ATL.graph`` is the
  concept topology (REQUIRES / ENABLES / CAUSES / ...); the binding
  graph is the co-activation history. Two distinct layers.

Lock acquire order
==================

Any code that mutates both ``EpisodeStore`` and the binding graph MUST
acquire locks in this order: ``EpisodeStore._lock`` (RLock) then
``DependencyGraph._lock`` (regular Lock). Never the reverse.
Regression-guarded by ``TestP3aMechanism::test_boundary_close_no_deadlock``.
"""

from __future__ import annotations

import itertools
import threading
from collections.abc import Callable
import dataclasses
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class Episode:
    """Immutable record of a closed multi-event time window.

    Episodes are created by ``EpisodeBoundaryDetector`` when a boundary
    rule fires, and added to the ``EpisodeStore`` at that point. Once
    created they are not mutated; any new co-activation opens a fresh
    pending episode.
    """

    id: str
    start_tick: int
    end_tick: int
    channel: str
    sender_ids: tuple[str, ...]
    thread_id: str | None
    activated_nodes: tuple[str, ...]
    # (tick, reward_delta) per reward event during this episode
    reward_events: tuple[tuple[int, float], ...]
    scn_tag: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "start_tick": self.start_tick,
            "end_tick": self.end_tick,
            "channel": self.channel,
            "sender_ids": list(self.sender_ids),
            "thread_id": self.thread_id,
            "activated_nodes": list(self.activated_nodes),
            "reward_events": [list(re) for re in self.reward_events],
            "scn_tag": self.scn_tag,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Episode:
        return cls(
            id=data["id"],
            start_tick=int(data["start_tick"]),
            end_tick=int(data["end_tick"]),
            channel=data["channel"],
            sender_ids=tuple(data.get("sender_ids", [])),
            thread_id=data.get("thread_id"),
            activated_nodes=tuple(data.get("activated_nodes", [])),
            reward_events=tuple((int(t), float(r)) for t, r in data.get("reward_events", [])),
            scn_tag=data.get("scn_tag"),
        )


# ─────────────────────────────────────────────────────────────────────────
# Pending episode state + boundary rules
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class CaptureEvent:
    """Input event fed into the episode boundary detector.

    This is a lightweight wrapper that holds only the fields the
    detector rules care about. Real captures in P3a Stage 2+ will come
    from ``Hippocampus.capture_from_loop`` or the runtime agent loop.
    """

    tick: int
    channel: str
    sender_id: str | None = None
    thread_id: str | None = None
    scn_tag: str | None = None
    # Substrate node IDs activated by this event; used as the seed for
    # the pending episode's activated_nodes set.
    activated_nodes: tuple[str, ...] = ()


@dataclass
class PendingEpisodeState:
    """Mutable state for the episode currently being built."""

    id: str
    start_tick: int
    last_tick: int
    channel: str
    sender_ids: set[str] = field(default_factory=set)
    thread_id: str | None = None
    activated_nodes: list[str] = field(default_factory=list)
    reward_events: list[tuple[int, float]] = field(default_factory=list)
    scn_tag: str | None = None

    def finalize(self) -> Episode:
        return Episode(
            id=self.id,
            start_tick=self.start_tick,
            end_tick=self.last_tick,
            channel=self.channel,
            sender_ids=tuple(sorted(self.sender_ids)),
            thread_id=self.thread_id,
            # Preserve activation order while keeping the collection unique:
            activated_nodes=tuple(dict.fromkeys(self.activated_nodes)),
            reward_events=tuple(self.reward_events),
            scn_tag=self.scn_tag,
        )


BoundaryRule = Callable[[PendingEpisodeState, CaptureEvent], bool]
"""A rule returns True when the incoming event should CLOSE the pending
episode and start a new one. Rules are consulted in order via ``any()``,
so the first rule that fires wins. Stage 1 ships three default rules
(tick gap, channel change, scn_tag change). P3b will append additional
per-channel rules without touching Stage 1 code."""


def tick_gap_rule(max_gap: int) -> BoundaryRule:
    """Close the pending episode when the next event is more than
    ``max_gap`` ticks after the previous capture."""

    def _rule(pending: PendingEpisodeState, event: CaptureEvent) -> bool:
        return event.tick > pending.last_tick + max_gap

    return _rule


def channel_change_rule() -> BoundaryRule:
    """Close the pending episode when the incoming event is on a
    different channel than the one the episode opened on."""

    def _rule(pending: PendingEpisodeState, event: CaptureEvent) -> bool:
        return event.channel != pending.channel

    return _rule


def scn_tag_change_rule() -> BoundaryRule:
    """Close the pending episode when the incoming event's ``scn_tag``
    differs from the pending episode's.

    A ``None`` scn_tag on either side is treated as "no change" — the
    rule only fires when both sides are non-None and different. This
    matches the intent of "a scene change closes an episode" while
    avoiding spurious closes when SCN tagging is incomplete.
    """

    def _rule(pending: PendingEpisodeState, event: CaptureEvent) -> bool:
        if event.scn_tag is None or pending.scn_tag is None:
            return False
        return event.scn_tag != pending.scn_tag

    return _rule


# ─────────────────────────────────────────────────────────────────────────
# P3b Stage 1 — channel-aware boundary rules
# ─────────────────────────────────────────────────────────────────────────


def channel_specific_rule(channel: str, inner: BoundaryRule) -> BoundaryRule:
    """Wrap a boundary rule so it only fires when the INCOMING EVENT is
    on the given channel.

    Round 1 Exec important #2: the wrapper gates on ``event.channel``,
    NOT ``pending.channel``. Gating on ``pending.channel`` only happens
    to work when the default ``channel_change_rule`` is installed
    (which guarantees the pending episode's channel matches every
    incoming event's channel before any other rule evaluates).
    Removing ``channel_change_rule`` (reasonable for cross-channel
    threading) would silently disable every wrapped rule under the
    pending-channel gating. Gating on ``event.channel`` stays correct
    under any default-rule configuration.
    """

    def _rule(pending: PendingEpisodeState, event: CaptureEvent) -> bool:
        if event.channel != channel:
            return False
        return inner(pending, event)

    return _rule


def channel_gap_rule(channel: str, max_gap_ticks: int) -> BoundaryRule:
    """Channel-gated tick gap rule.

    Closes the pending episode when (a) the incoming event's channel
    matches ``channel`` AND (b) the next event is more than
    ``max_gap_ticks`` ticks after the previous capture. Used to model
    channels with different natural gap thresholds (e.g., SMS
    conversations have longer gaps than narrative scenes).

    Round 2 Arch important #2 fold: replaces the earlier
    ``sms_gap_rule(max_gap_ticks=500)`` factory which hard-coded
    ``"sms"`` three layers deep. The generalized factory takes the
    channel name as a parameter so P5/P6 can use it for "email",
    "slack", "voice", etc., without spawning a per-channel
    factory zoo.

    The choice of ``max_gap_ticks`` is calibration-pending — Stage 2
    sweep will refine. Note that ``ticks`` are loop-iteration counts,
    not seconds; at the canonical 2-30Hz loop rate a ``max_gap_ticks
    = 500`` corresponds to roughly 16-250 seconds of wall clock.
    """
    return channel_specific_rule(channel, tick_gap_rule(max_gap_ticks))


def sender_change_rule() -> BoundaryRule:
    """Boundary rule that closes the pending episode when a new sender
    appears (sender_id not previously seen in ``pending.sender_ids``).

    **Channel-agnostic.** Wrap with ``channel_specific_rule(channel,
    sender_change_rule())`` if you want to gate it to a specific
    channel (e.g., SMS contact-switch detection). Round 2 Arch
    important #2 fold: replaces the earlier
    ``sms_sender_change_rule()`` factory which hard-coded the SMS
    channel. The channel-agnostic shape lets P5/P6 reuse the
    sender-change logic on any channel without a per-channel factory.

    **Cold-start guard (Round 1 Exec critical #1):** returns
    ``False`` when ``pending.sender_ids`` is empty. Without this
    guard, a pending episode opened by an event with
    ``sender_id=None`` (so ``sender_ids`` stays empty) would close
    the moment any later event arrived with a non-None sender —
    because ``"bob" not in empty_set`` is trivially True. The guard
    preserves the intended semantic ("a NEW sender joins an existing
    conversation") while silently no-op-ing on cold start.
    """

    def _rule(pending: PendingEpisodeState, event: CaptureEvent) -> bool:
        if event.sender_id is None or not pending.sender_ids:
            return False
        return event.sender_id not in pending.sender_ids

    return _rule


class EpisodeBoundaryDetector:
    """Runs a list of boundary rules against each incoming event.

    Designed as a rule-list shape (per P3a Round 1 Arch important #2)
    so P3b channel integration can append new rules without touching
    Stage 1 code. Rules are commutative — any rule firing closes the
    pending episode, so insertion order within the list is irrelevant.

    **Per-rule O(1) invariant (P5 perf concern, Round 2 Arch
    important #3):** rules are evaluated on every event via
    ``any(rule(...) for rule in self._rules)``, so a Slack event
    pays the SMS-rule cost. At Stage 1's 3-5 rules this is
    irrelevant; at P5's anticipated channel count (~8-10) it's
    still cheap. P8 sleep replay pumps O(episodes) events through
    the detector in a batch — that's where quadratic behavior
    would bite. **Rule authors:** keep boundary rules to O(1) work
    per event. P5 should consider a per-channel rule index if the
    rule list crosses ~8-10 entries.
    """

    def __init__(self, rules: list[BoundaryRule]) -> None:
        self._rules = list(rules)

    def should_close(self, pending: PendingEpisodeState, event: CaptureEvent) -> bool:
        return any(rule(pending, event) for rule in self._rules)

    def add_rule(self, rule: BoundaryRule) -> None:
        """Append a rule. Used by P3b channel integration."""
        self._rules.append(rule)


# ─────────────────────────────────────────────────────────────────────────
# EpisodeStore
# ─────────────────────────────────────────────────────────────────────────


class EpisodeStore:
    """Owns ``Episode`` records and the node→episode inverted index.

    Held as ``Hippocampus._episode_store``; kept as a standalone class
    rather than inlined on Hippocampus so P3b channel rules and P5
    bounded-storage eviction can extend this class without touching the
    Hippocampus class itself (Round 1 Arch important #1).
    """

    def __init__(self) -> None:
        self._episodes: dict[str, Episode] = {}
        # node_id -> set of episode ids containing that node
        self._by_node: dict[str, set[str]] = {}
        self._lock = threading.RLock()

    def add(self, episode: Episode) -> None:
        """Add a closed episode to the store + update the inverted index."""
        with self._lock:
            if episode.id in self._episodes:
                raise ValueError(f"duplicate episode id: {episode.id!r}")
            self._episodes[episode.id] = episode
            for node_id in episode.activated_nodes:
                self._by_node.setdefault(node_id, set()).add(episode.id)

    def get(self, episode_id: str) -> Episode | None:
        with self._lock:
            return self._episodes.get(episode_id)

    def episodes_containing(self, node_id: str) -> list[Episode]:
        """Return all episodes whose activated_nodes contain ``node_id``."""
        with self._lock:
            ids = list(self._by_node.get(node_id, ()))
            return [self._episodes[eid] for eid in ids if eid in self._episodes]

    def all_episodes(self) -> list[Episode]:
        with self._lock:
            return list(self._episodes.values())

    def __len__(self) -> int:
        with self._lock:
            return len(self._episodes)

    def clear(self) -> None:
        with self._lock:
            self._episodes.clear()
            self._by_node.clear()

    # ─────────────────────────────────────────────────────────────────
    # Persistence — Hippocampus.dump() / load_state() delegates here
    # ─────────────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {"episodes": [ep.to_dict() for ep in self._episodes.values()]}

    def load_from_dict(self, data: dict[str, Any]) -> None:
        """Mutate in place — replace all episodes from a state dict.

        Per-episode duplicate-id check: if the loaded state contains
        two episodes with the same id (e.g., a corrupt file), raise
        ``ValueError`` rather than silently overwriting. Round 2 Exec
        important #4 — the pre-fold version did
        ``self._episodes[ep.id] = ep`` which collapsed duplicates and
        left stale node→episode references in ``_by_node``.
        """
        episodes_list = data.get("episodes", [])
        with self._lock:
            self._episodes.clear()
            self._by_node.clear()
            seen_ids: set[str] = set()
            for ep_data in episodes_list:
                ep = Episode.from_dict(ep_data)
                if ep.id in seen_ids:
                    raise ValueError(f"duplicate episode id in loaded state: {ep.id!r}")
                seen_ids.add(ep.id)
                self._episodes[ep.id] = ep
                for node_id in ep.activated_nodes:
                    self._by_node.setdefault(node_id, set()).add(ep.id)

    # ─────────────────────────────────────────────────────────────────
    # P3b Stage 1 — channel-aware retrieval filter
    # ─────────────────────────────────────────────────────────────────

    def episode_membership_filter(
        self,
        *,
        membership_mode: Literal["any", "exclusive"] = "any",
        **criteria: Any,
    ) -> Callable[[str], bool]:
        """Build a ``node_filter`` callable for ``Hippocampus.retrieve_on_cue``.

        Returns a predicate ``(node_id) -> bool`` that retains nodes
        appearing in episodes matching the given ``criteria``. Designed
        as the channel/sender filter for P3b channel integration AND
        the modality filter seam P4 cross-modal will extend.

        **Snapshot semantics — load-bearing for thread safety
        (Round 2 self-found critical):** the filter is built by
        walking ``self._episodes`` ONCE under ``self._lock`` and
        producing a frozen set of allowed node ids. The returned
        callable is a closure over that frozen set — it acquires NO
        locks per call. This is required because
        ``DependencyGraph.spreading_activation`` holds the binding
        graph's lock for the duration of the BFS traversal and
        invokes ``node_filter`` while it's held. A lazy filter that
        re-queried ``self.episodes_containing`` on each call would
        acquire ``EpisodeStore._lock`` while the binding graph lock
        is held → AB-BA deadlock against
        ``Hippocampus._close_pending_episode_locked`` (which
        acquires store lock first via ``store.add(episode)`` then
        binding graph lock via ``apply_hebbian_on_close``). The
        snapshot pattern eliminates the inversion by collapsing all
        store-side work into a single up-front lock acquisition.

        Snapshot side effect: episodes added AFTER filter
        construction are NOT reflected. This is the right retrieval
        semantics — a multi-hop walk should see a consistent
        point-in-time view of the binding state, not mutate
        mid-traversal.

        **Filter axes are introspected from `Episode` dataclass fields**
        (Round 1 Arch important #4) — pass any combination as keyword
        arguments:

        - Scalar fields (``channel``, ``thread_id``, ``scn_tag``):
          matched via equality (``episode.<field> == value``).
        - Collection fields (``sender_ids``, which is a tuple):
          matched via membership (``value in episode.<field>``).

        Examples::

            # SMS channel only
            filt = store.episode_membership_filter(channel="sms")
            results = h.retrieve_on_cue(cue, node_filter=filt)

            # SMS + alice as sender
            filt = store.episode_membership_filter(channel="sms", sender_ids="alice")

            # P4 will add: vision modality only
            filt = store.episode_membership_filter(modality="vision")

        **Membership modes** (Round 1 Arch critical #1):

        - ``"any"`` (default, Stage 1): node retained if it appears
          in ≥1 matching episode.
        - ``"exclusive"`` (parameter shape committed for P4): node
          retained ONLY if every episode containing it matches all
          criteria. Stage 1 implements ``"exclusive"`` per the spec
          but P4 cross-modal will be the primary consumer (it needs
          this mode to identify cross-modal bridge nodes).

        **Empty inputs**: a node with zero containing episodes is
        NOT in the snapshot (no episodes to match against).
        """
        if membership_mode not in ("any", "exclusive"):
            raise ValueError(f"membership_mode must be 'any' or 'exclusive', got {membership_mode!r}")

        # Round 2 critical fold (cross-confirmed by Exec + Arch lenses):
        # validate criteria keys against `Episode` dataclass fields at
        # filter build time. The pre-fold version used
        # `getattr(episode, key, None)` which silently returned None for
        # unknown fields, conflating typos and missing fields with
        # legitimate-but-None scalar values. The two consequences:
        #   1. A typo (e.g. `channnel="sms"`) returns an empty filter
        #      with no warning. P4 cross-modal mug test would pass at
        #      0% collapse and nobody would know why.
        #   2. A legitimate null filter (e.g. `thread_id=None` meaning
        #      "episodes with no thread") is indistinguishable from the
        #      typo case and always returns False.
        # Fix: enumerate Episode fields once at build time and raise on
        # unknown keys; let the equality / membership check handle None
        # values explicitly.
        valid_fields = {f.name for f in dataclasses.fields(Episode)}
        unknown = [k for k in criteria if k not in valid_fields]
        if unknown:
            raise ValueError(
                f"unknown Episode field(s) in criteria: {sorted(unknown)}. Valid fields: {sorted(valid_fields)}"
            )

        # Disallow non-string collection criteria. Round 2 Exec
        # important #3: passing `sender_ids=("alice", "bob")` for "any
        # of these senders" silently returns nothing because
        # `tuple in tuple` checks element membership, not subset. Raise
        # explicitly; subset / set-intersection semantics are deferred
        # to a future plan if a real call site ever needs them.
        for key, value in criteria.items():
            if isinstance(value, (tuple, list, set, frozenset)):
                raise TypeError(
                    f"criterion {key!r} is a collection {type(value).__name__}; "
                    f"pass a single scalar (str/int) instead. Subset / "
                    f"set-intersection semantics are not supported in Stage 1."
                )

        def _matches(episode: Episode) -> bool:
            for key, value in criteria.items():
                field_value = getattr(episode, key)
                # Collection fields: membership match. Strings are NOT
                # treated as collections — they're scalar identifiers.
                if isinstance(field_value, (tuple, list, set, frozenset)):
                    if value not in field_value:
                        return False
                elif field_value != value:
                    return False
            return True

        # Pre-compute the allowed node set under the store lock, ONCE.
        # The returned closure is lock-free — see the deadlock-safety
        # docstring above.
        with self._lock:
            if membership_mode == "any":
                allowed_nodes: set[str] = set()
                for ep in self._episodes.values():
                    if _matches(ep):
                        allowed_nodes.update(ep.activated_nodes)
            else:  # exclusive
                allowed_nodes = set()
                for node_id, ep_ids in self._by_node.items():
                    episodes = [self._episodes[eid] for eid in ep_ids if eid in self._episodes]
                    if episodes and all(_matches(ep) for ep in episodes):
                        allowed_nodes.add(node_id)
            allowed: frozenset[str] = frozenset(allowed_nodes)

        return lambda node_id: node_id in allowed


# ─────────────────────────────────────────────────────────────────────────
# Hebbian on close
# ─────────────────────────────────────────────────────────────────────────


def apply_hebbian_on_close(
    binding_graph: Any,  # DependencyGraph — typed Any to avoid a hard import cycle
    episode: Episode,
    *,
    hebbian_init: float,
    hebbian_delta: float,
    hebbian_max: float,
) -> None:
    """Apply Hebbian updates to ``binding_graph`` for a closed episode.

    For every UNORDERED pair of activated nodes in the episode:

    - If the edge does not exist, create both directions at
      ``hebbian_init`` via ``add_bidirectional``.
    - Otherwise, increment both directions by ``hebbian_delta``,
      clamped at ``hebbian_max``.

    Pair enumeration uses ``itertools.combinations`` (unordered). Per
    the Round 1 Exec-lens critical finding #1, iterating ordered pairs
    visits each unordered pair twice under ``add_bidirectional``, which
    would double-apply the delta. The unordered form fixes that.

    ``DependencyGraph.add_edge`` has no dedupe (Round 1 Exec critical
    #2), so the ``find_edge`` check before ``add_bidirectional`` is
    load-bearing — without it, repeated episode closes would stack
    parallel edges. Regression-guarded by
    ``TestP3aMechanism::test_repeated_closes_no_edge_duplication``.

    Explicit ``add_node`` calls before edge creation (Round 2 Exec
    important #3): ``DependencyGraph.add_edge`` appends to
    ``_outgoing`` / ``_incoming`` but never touches ``_nodes``, so
    ``to_dict`` would emit an empty nodes list and Stage 2 binding-
    graph persistence would silently lose node identity. ``add_node``
    is idempotent via its internal presence check, so repeat calls
    are safe.
    """
    from maxim.agents.bus import EdgeType  # local import avoids startup cost

    nodes = episode.activated_nodes
    if len(nodes) < 2:
        return

    # Ensure every activated node is a first-class graph node BEFORE any
    # edge reference. Round 2 fix for the add_edge-does-not-touch-_nodes
    # invariant gap.
    for node_id in nodes:
        binding_graph.add_node(node_id, node_id)

    for a, b in itertools.combinations(nodes, 2):
        existing = binding_graph.find_edge(a, b, EdgeType.ASSOCIATES)
        if existing is None:
            binding_graph.add_bidirectional(a, b, EdgeType.ASSOCIATES, weight=hebbian_init)
        else:
            new_weight = min(hebbian_max, existing.weight + hebbian_delta)
            binding_graph.update_edge(a, b, EdgeType.ASSOCIATES, weight=new_weight)
            binding_graph.update_edge(b, a, EdgeType.ASSOCIATES, weight=new_weight)


__all__ = [
    "BoundaryRule",
    "CaptureEvent",
    "Episode",
    "EpisodeBoundaryDetector",
    "EpisodeStore",
    "PendingEpisodeState",
    "apply_hebbian_on_close",
    "channel_change_rule",
    "channel_gap_rule",
    "channel_specific_rule",
    "scn_tag_change_rule",
    "sender_change_rule",
    "tick_gap_rule",
]
