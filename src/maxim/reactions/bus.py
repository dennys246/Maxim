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
