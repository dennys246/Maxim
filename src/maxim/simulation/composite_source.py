"""CompositePerceptSource — multiplex N child PerceptSources behind one seam.

The agent loop and orchestrator take exactly ONE ``percept_source``
(``agent_loop.py`` / ``orchestrator.py``). To feed a second channel (e.g. an
``AzimuthDoASource`` alongside the interactive ``ConversationalSource``) without
touching that seam, wrap the children in a ``CompositePerceptSource`` — it *is*
a ``PerceptSource``, so the single-source loop is unchanged; the multiplexer
hides inside the existing seam.

This is the thalamic relay body in embryo (modality-preserving multiplex), the
first slice of ``docs/plans/archive/thalamus_relay_design_pass.md``. Deliberately a
**dumb multiplexer**, NOT a routing-policy engine:

- ``next_percept()`` — deterministic ordered scan, **first non-None child wins**.
  NOT priority selection, back-pressure, or per-child rate limiting — that
  routing policy is exactly what earns the (deferred) thalamic coordinator, and
  baking the first channel's assumptions in here is the N=1 abstraction mistake
  this codebase has scars from.
- ``is_exhausted()`` — driven by the **exhaustible (scripted) children**;
  ``ambient`` children (perpetual-live sensors whose ``is_exhausted()`` is
  forever False) do NOT veto termination, or a live sensor would block a
  scenario-complete sim from ever self-terminating.
- **Injection / lifecycle** (``inject_cli`` / ``inject_pain`` / ``finish`` — the
  injection contract a multiplexer must honor *if* it becomes the bridge's
  ``percept_source``) route to the **single** child that implements each. A
  SECOND implementer of any of them raises at construction: addressed routing is
  the coordinator's job, not this dumb multiplexer's. (In the current audio
  wiring the bridge injects into its own ``ConversationalSource`` child directly
  and the composite is handed only to the loop, so this apparatus is
  tested-but-latent — cheap contract-completeness for the next wiring.)
- **Fan-out** (``advance_step`` / ``has_pending``) broadcasts to every child.

``[engineering]`` — this is un-flatten + multiplex mechanics, not a validated
"thalamus." It graduates only when the audio-orient experiment earns it.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Injection / lifecycle methods the orchestrator + bridge call on their
# ``percept_source`` (grounded in bridge.py:{142,223,241} + orchestrator.py:1977).
# Each routes to the ONE child implementing it. Adding a new such method here is
# the deliberate way to extend the contract; a SECOND implementer of any of them
# is the trigger to introduce the coordinator's addressed routing.
_INJECTION_METHODS = ("inject_cli", "inject_pain", "finish")


class CompositeRoutingError(RuntimeError):
    """Raised when an injection/lifecycle call cannot be routed unambiguously.

    Two shapes: >1 child implements the method (addressed routing needed — earns
    the coordinator), or 0 children implement it (the composite was asked to
    inject/finish but holds no source that can).
    """


class CompositePerceptSource:
    """Multiplex several child ``PerceptSource``s as one ``PerceptSource``.

    Args:
        children: ordered child sources; scanned first-to-last by
            ``next_percept``. Must be non-empty.
        ambient: the subset of ``children`` that are perpetual-live (e.g. a
            hardware sensor whose ``is_exhausted()`` never becomes True). They
            do NOT gate termination. Compared by identity. Default: none, so
            every child gates termination (correct for scripted-only composites).
        name: human-readable source identifier.
    """

    def __init__(
        self,
        children: list[Any],
        *,
        ambient: list[Any] | None = None,
        name: str = "composite",
    ) -> None:
        children = list(children)
        if not children:
            raise ValueError("CompositePerceptSource requires at least one child source")
        self._children = children
        self._name = name
        # Identity set — sources are not guaranteed hashable-by-value, and we
        # want "is this the same object" semantics, which default object hashing
        # (by id) gives us.
        self._ambient = set(ambient or [])
        for a in self._ambient:
            if not any(a is c for c in self._children):
                raise ValueError("ambient source is not one of the children")

        # Fail fast on ambiguous injection/lifecycle routing (single-implementer
        # assumption). Resolved once at construction; call sites just look up.
        self._injectors: dict[str, Any] = {}
        for method in _INJECTION_METHODS:
            impls = [c for c in self._children if callable(getattr(c, method, None))]
            if len(impls) > 1:
                raise CompositeRoutingError(
                    f"{len(impls)} children implement {method!r}; a composite is a "
                    "dumb multiplexer and cannot disambiguate — addressed routing "
                    "is the thalamic coordinator's job (see thalamus_relay_design_pass.md)"
                )
            if impls:
                self._injectors[method] = impls[0]

    # ------------------------------------------------------------------ core

    @property
    def name(self) -> str:
        return self._name

    @property
    def capabilities(self) -> set[str]:
        caps: set[str] = set()
        for c in self._children:
            caps |= set(getattr(c, "capabilities", set()))
        return caps

    def next_percept(self) -> Any | None:
        """First non-None percept in child order; None if all children idle."""
        for c in self._children:
            percept = c.next_percept()
            if percept is not None:
                return percept
        return None

    def is_exhausted(self) -> bool:
        """True iff every *gating* (non-ambient) child is exhausted.

        A composite of only ambient (perpetual-live) children never exhausts —
        it is a pure sensor stream with no scripted end.
        """
        gating = [c for c in self._children if c not in self._ambient]
        if not gating:
            return False
        return all(c.is_exhausted() for c in gating)

    # -------------------------------------------------------------- fan-out

    def advance_step(self) -> None:
        """Broadcast the per-tick step advance to every child that supports it."""
        for c in self._children:
            method = getattr(c, "advance_step", None)
            if callable(method):
                method()

    def has_pending(self) -> bool:
        """True if any child (definitely or possibly) has a percept queued.

        A child without ``has_pending`` is treated as possibly-pending (``True``),
        matching the agent loop's own ``getattr(src, "has_pending", lambda: True)``
        default — a live sensor may always produce a reading.
        """
        return any(getattr(c, "has_pending", lambda: True)() for c in self._children)

    # ------------------------------------------------- injection / lifecycle

    def inject_cli(self, text: str, salience: float = 0.8, novelty: float = 0.7) -> None:
        self._route("inject_cli", text, salience=salience, novelty=novelty)

    def inject_pain(self, *args: Any, **kwargs: Any) -> None:
        self._route("inject_pain", *args, **kwargs)

    def finish(self) -> None:
        self._route("finish")

    def _route(self, method: str, *args: Any, **kwargs: Any) -> Any:
        target = self._injectors.get(method)
        if target is None:
            raise CompositeRoutingError(f"no child implements {method!r}; composite {self._name!r} cannot route it")
        return getattr(target, method)(*args, **kwargs)
