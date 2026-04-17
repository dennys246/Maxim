"""ReactionBus — typed publish/subscribe for evaluative signals.

Generalized from PainBus. Subscribers register for specific reaction
kinds ("pain", "fear", etc.) or for all kinds. Refractory periods are
configurable per-kind.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict, deque
from typing import Any, Callable

from maxim.reactions.types import Reaction, ReactionKind

logger = logging.getLogger(__name__)


class ReactionBus:
    """Typed publish/subscribe for Reaction signals.

    Subscribers register for a specific ``kind`` or for all kinds via
    ``subscribe_all``. Refractory periods prevent spam from rapid
    repeated signals of the same (kind, source) pair.

    Example::

        bus = ReactionBus()
        bus.subscribe("pain", lambda r: print(f"Pain: {r.intensity}"))
        bus.publish(Reaction(kind="pain", intensity=0.8, ...))
    """

    DEFAULT_REFRACTORY_S: float = 0.5

    def __init__(
        self,
        history_size: int = 200,
        refractory_overrides: dict[str, float] | None = None,
    ) -> None:
        self._per_kind: dict[str, list[Callable[[Reaction], None]]] = defaultdict(list)
        self._all_subscribers: list[Callable[[Reaction], None]] = []
        self._lock = threading.Lock()
        self._history: deque[Reaction] = deque(maxlen=history_size)
        self._total_published: int = 0
        self._last_published: dict[str, float] = {}
        self._refractory: dict[str, float] = dict(refractory_overrides or {})

    def subscribe(self, kind: ReactionKind, callback: Callable[[Reaction], None]) -> None:
        with self._lock:
            self._per_kind[kind].append(callback)

    def subscribe_all(self, callback: Callable[[Reaction], None]) -> None:
        with self._lock:
            self._all_subscribers.append(callback)

    def unsubscribe(self, kind: ReactionKind, callback: Callable[[Reaction], None]) -> None:
        with self._lock:
            subs = self._per_kind.get(kind, [])
            if callback in subs:
                subs.remove(callback)

    def publish(self, reaction: Reaction) -> None:
        refractory_s = self._refractory.get(reaction.kind, self.DEFAULT_REFRACTORY_S)
        refractory_key = f"{reaction.kind}:{reaction.source}"
        now = time.monotonic()

        with self._lock:
            last = self._last_published.get(refractory_key, 0.0)
            if now - last < refractory_s:
                return
            self._last_published[refractory_key] = now
            self._history.append(reaction)
            self._total_published += 1
            per_kind = list(self._per_kind.get(reaction.kind, []))
            all_subs = list(self._all_subscribers)

        for callback in per_kind + all_subs:
            try:
                callback(reaction)
            except Exception as e:
                logger.warning("ReactionBus subscriber error (%s): %s", reaction.kind, e)

    def history(self, kind: ReactionKind | None = None) -> list[Reaction]:
        with self._lock:
            if kind is None:
                return list(self._history)
            return [r for r in self._history if r.kind == kind]

    def get_stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "total_published": self._total_published,
                "subscriber_count": sum(len(v) for v in self._per_kind.values()) + len(self._all_subscribers),
                "history_size": len(self._history),
            }

    def __len__(self) -> int:
        with self._lock:
            return len(self._history)


# ─────────────────────────────────────────────────────────────────────────────
# Canonical ReactionBus construction site (Wave 1, biosystem_unification)
# ─────────────────────────────────────────────────────────────────────────────


def build_reaction_bus(
    *,
    per_kind_subscribers: dict[ReactionKind, tuple[Callable[[Reaction], None], ...]] | None = None,
    all_subscribers: tuple[Callable[[Reaction], None], ...] = (),
    history_size: int = 200,
    refractory_overrides: dict[str, float] | None = None,
) -> ReactionBus:
    """Construct a ReactionBus with explicit subscriber decisions.

    This is the canonical ReactionBus construction site for production
    code. It exists to establish the construction door that Wave 3
    (``docs/plans/bio_stack_unification.md``) requires: ``build_bio_stack``
    will construct ``reaction_bus = build_reaction_bus(...)`` BEFORE
    ``pain_bus = build_pain_bus(..., reaction_bus=reaction_bus)`` because
    PainBus depends on ReactionBus at construction time.

    Today the builder has ZERO production callers —
    ``PainBus.__init__`` constructs ``ReactionBus()`` directly.
    Wave 3's ``build_bio_stack`` will be the first production caller
    when it constructs a standalone ``ReactionBus`` and passes it to
    ``build_pain_bus(..., reaction_bus=rb)``. The thin wrapper is
    intentional — the interface is the deliverable, not the call
    count. Establishing it now when the surface is clean avoids a
    refactor during Wave 3 when the surface is complex.

    The audited silent gap (``docs/plans/reaction_bus_unification.md``
    Gap A): ``cerebellum_modulator_factory`` never passes
    ``reaction_bus=`` to ``CerebellumModulator``, so every SEM modulator
    failure signal is silently dropped. That gap is fixed separately in
    the factory wiring commit, not here — the builder's job is to
    construct the bus, not to inject it into consumers.

    Unlike ``build_pain_bus``, this builder does NOT have required
    keyword-only learning-subject parameters (no equivalent to
    ``hippocampus``/``nac``). ReactionBus is a generic typed pub/sub;
    its subscribers are domain-specific and vary by caller. The
    structural enforcement here is at the **construction door** level
    (callers go through ``build_reaction_bus``, not raw
    ``ReactionBus()``), not at the **parameter-must-not-be-forgotten**
    level.

    Args:
        per_kind_subscribers: Optional dict mapping ``ReactionKind``
            (e.g., ``"pain"``, ``"fear"``) to tuples of callbacks.
            Each callback is registered via ``bus.subscribe(kind, cb)``.
        all_subscribers: Optional tuple of callbacks registered via
            ``bus.subscribe_all(cb)``. These fire for every published
            Reaction regardless of kind.
        history_size: Ring buffer size for ``bus.history()``. Forwarded
            to ``ReactionBus.__init__``.
        refractory_overrides: Per-kind refractory windows (seconds).
            Forwarded to ``ReactionBus.__init__``.

    Returns:
        A fully-constructed ``ReactionBus`` with subscribers already
        registered.

    Examples:
        Production path (inside ``build_pain_bus`` or ``build_bio_stack``)::

            reaction_bus = build_reaction_bus(
                per_kind_subscribers={"pain": (bridge_callback,)},
                all_subscribers=(sim_log_callback,),
                refractory_overrides={"pain": 0.5},
            )

        Minimal construction (test / standalone)::

            reaction_bus = build_reaction_bus()

    See also:
        - ``docs/plans/reaction_bus_unification.md`` for the audit + design
        - ``docs/plans/bio_stack_unification.md`` for the Wave 3 ordering
        - ``proprioception/pain_bus.py::build_pain_bus`` for the sibling
    """
    bus = ReactionBus(
        history_size=history_size,
        refractory_overrides=refractory_overrides,
    )
    if per_kind_subscribers:
        for kind, callbacks in per_kind_subscribers.items():
            for cb in callbacks:
                bus.subscribe(kind, cb)
    for cb in all_subscribers:
        bus.subscribe_all(cb)
    return bus
