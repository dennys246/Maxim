"""PainBus — rich-context PainSignal delivery layered over ReactionBus.

PainBus coexists with ReactionBus (the strict typed-surface layer).
ReactionBus enforces the typed isolation rules documented in
``reactions/types.py`` — its ``ReactionContext`` carries only
TraceSnapshot bindings and scn/agent metadata. That's deliberate: it
prevents cross-agent learning hints from leaking through the Reaction
surface.

Bio-pipeline-internal signals (embodiment failures, tool errors,
sandbox violations) need to carry **cause-description metadata** —
``source``, ``entity``, ``failure_mode``, ``sensor_readings`` —
that feeds NAc causal learning. This metadata is not a learning hint
in the isolation-rule sense; it's the description of *what happened*.
``PainSignal.context`` (a free-form dict) is the carrier, and PainBus
is the bus that delivers it without the round-trip loss the old
Reaction-adapter path imposed.

Delivery model:

    body.py / sandbox.py / tool failure    →  PainBus.publish(signal)
                                              ├→ direct PainSignal subscribers
                                              │   (full signal.context)
                                              └→ reaction_bus.publish(reaction)
                                                  └→ ReactionBus subscribers
                                                      (typed Reaction, lossy)

Reactions published *directly* on ``self.reaction_bus`` (e.g., code
that constructs its own Reaction) still reach PainBus subscribers via
an internal bridge that falls back to the lossy
``_reaction_to_pain_signal`` converter.

Refractory filtering: PainBus applies a **per-(entity, failure_mode)
refractory** on the PainSignal path (default 0.5s), finer-grained than
ReactionBus's ``(kind, source)`` gate. This prevents two distinct
embodiment failures (e.g., sword shattering + arm straining in the same
tick) from collapsing into one dispatch because
``pain_signal_to_reaction`` assigns them the same synthetic source
string. The converted Reaction is ALSO delivered to ``reaction_bus``
which applies its own gate for typed subscribers — the two gates serve
different audiences and are allowed to disagree.

Pre-Stage-2 this module used a ``contextvars.ContextVar`` signal-stash
to smuggle the full signal into a nested reaction_bus subscriber. That
pattern had two hazards: (a) re-entrancy when a subscriber triggered
another publish, silently falling back to the lossy path on the outer
dispatch, and (b) invisible coupling with ``@resilient`` retry and
async contexts. The current design calls direct subscribers from
``PainBus.publish`` itself, gated by PainBus's own refractory state.
No ContextVar, no nested subscriber, no re-entrancy hazard.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
import warnings
from typing import TYPE_CHECKING, Any, Callable

from maxim.proprioception.pain import PainSignal, PainType
from maxim.reactions.bus import ReactionBus
from maxim.reactions.compat import pain_signal_to_reaction

if TYPE_CHECKING:
    from maxim.decisions.nac import NAc
    from maxim.memory.hippocampus import Hippocampus
    from maxim.reactions.types import Reaction

logger = logging.getLogger(__name__)


def _sim_log_reaction(reaction: "Reaction") -> None:
    """Best-effort sim_reaction call for the simulation display."""
    try:
        from maxim.simulation.sim_logger import sim_reaction

        sim_reaction(reaction.kind, reaction.intensity, reaction.source)
    except Exception:
        pass


class PainBus:
    """PainSignal-carrying bus with rich-context delivery.

    Subscribers registered via :meth:`subscribe` receive ``PainSignal``
    with the full publisher-supplied ``context`` dict. Refractory is
    keyed on ``(entity, failure_mode)`` so distinct failures do not
    shadow each other during rapid cascades.

    ``self.reaction_bus`` is exposed for code that wants typed
    ``Reaction`` semantics directly; publishes on that bus still reach
    PainSignal subscribers, but only through the lossy
    ``_reaction_to_pain_signal`` fallback.
    """

    #: Default refractory window (seconds) for PainSignal dispatch to
    #: direct subscribers, keyed on ``(entity, failure_mode)``.
    DEFAULT_PAIN_REFRACTORY_S: float = 0.5

    def __init__(
        self,
        history_size: int = 200,
        pain_refractory_s: float | None = None,
        *,
        _allow_raw: bool = False,
    ) -> None:
        if not _allow_raw:
            msg = (
                "Raw PainBus() construction is deprecated; use "
                "maxim.proprioception.pain_bus.build_pain_bus(hippocampus=..., nac=...) "
                "instead. The builder enforces required learning-subject wiring "
                "(see docs/plans/pain_bus_unification.md) — three CLI entry points "
                "previously forgot to subscribe NAc, silently disabling causal "
                "learning for out-of-band SEM pain. Tests that need a bare bus "
                "may pass _allow_raw=True. This becomes a hard error in 1.0. "
                "(C6 deprecation)"
            )
            print(f"DeprecationWarning: {msg}", file=sys.stderr)
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
        self.reaction_bus = ReactionBus(
            history_size=history_size,
            refractory_overrides={"pain": 0.5},
            _allow_raw=True,
        )
        self._pain_signal_subs: list[Callable[[PainSignal], None]] = []
        self._pain_refractory_s = pain_refractory_s if pain_refractory_s is not None else self.DEFAULT_PAIN_REFRACTORY_S
        # Per-(entity, failure_mode) last-fired timestamp (monotonic).
        self._last_pain_fired: dict[tuple[str, str], float] = {}
        self._lock = threading.Lock()
        # sim_logger fires for every Reaction regardless of origin.
        self.reaction_bus.subscribe_all(_sim_log_reaction)
        # Bridge: Reactions published directly on reaction_bus (e.g.,
        # sandbox) ALSO reach our direct subscribers via the lossy
        # reconstruction path. Publishes that go through PainBus.publish
        # short-circuit this by setting ``_suppress_bridge`` for the
        # duration of the nested reaction_bus.publish call.
        self._suppress_bridge = False
        self.reaction_bus.subscribe("pain", self._bridge_reaction_to_pain_subs)

    def subscribe(self, callback: Callable[[PainSignal], None]) -> None:
        """Register a direct PainSignal subscriber.

        Callbacks are stored on ``self._pain_signal_subs`` and invoked
        either from :meth:`publish` (rich-context path) or from the
        internal :meth:`_bridge_reaction_to_pain_subs` (lossy fallback
        for Reactions published directly on ``reaction_bus``).
        """
        with self._lock:
            self._pain_signal_subs.append(callback)

    def unsubscribe(self, callback: Callable[[PainSignal], None]) -> None:
        with self._lock:
            try:
                self._pain_signal_subs.remove(callback)
            except ValueError:
                pass

    def publish(self, signal: PainSignal) -> None:
        """Publish a PainSignal with full context.

        Applies PainBus's own ``(entity, failure_mode)`` refractory gate
        on the PainSignal path, then dispatches to direct subscribers
        with the full ``signal.context``, then forwards the converted
        Reaction to ``reaction_bus`` for typed subscribers (with
        ``_suppress_bridge`` set so the bridge doesn't re-fire direct
        subscribers for this signal).
        """
        ctx = signal.context or {}
        entity = ctx.get("entity") or ctx.get("entity_path") or "unknown"
        failure = ctx.get("failure_mode") or signal.pain_type.name
        key = (str(entity), str(failure))
        now = time.monotonic()

        with self._lock:
            last = self._last_pain_fired.get(key, 0.0)
            if now - last < self._pain_refractory_s:
                # Refractory — drop silently for direct subscribers.
                # The Reaction ALSO gets dropped by reaction_bus's own
                # (kind, source) gate, which is coarser but aligned.
                return
            self._last_pain_fired[key] = now
            subs = list(self._pain_signal_subs)

        # Dispatch to direct subscribers with full context.
        for cb in subs:
            try:
                cb(signal)
            except Exception as e:
                logger.warning(
                    "PainBus subscriber error (%s): %s",
                    getattr(cb, "__name__", repr(cb)),
                    e,
                )

        # Forward to reaction_bus for typed subscribers. Suppress the
        # internal bridge during this call so direct subscribers are
        # not re-fired with a lossy reconstruction.
        self._suppress_bridge = True
        try:
            reaction = pain_signal_to_reaction(signal)
            self.reaction_bus.publish(reaction)
        finally:
            self._suppress_bridge = False

    def _bridge_reaction_to_pain_subs(self, reaction: "Reaction") -> None:
        """Deliver reaction_bus-origin pain Reactions to direct subscribers.

        Runs as a ``reaction_bus.subscribe("pain", ...)`` callback. When
        :meth:`publish` forwards a reaction to ``reaction_bus`` it sets
        ``_suppress_bridge`` so this handler no-ops and direct subscribers
        are not dispatched twice. For reactions that arrive on
        ``reaction_bus`` from outside PainBus (e.g., sandbox publishing
        a Reaction directly), the bridge reconstructs a lossy PainSignal
        via ``_reaction_to_pain_signal`` and fans it out to direct
        subscribers.
        """
        if self._suppress_bridge:
            return
        signal = _reaction_to_pain_signal(reaction)
        with self._lock:
            subs = list(self._pain_signal_subs)
        for cb in subs:
            try:
                cb(signal)
            except Exception as e:
                logger.warning(
                    "PainBus subscriber error (%s): %s",
                    getattr(cb, "__name__", repr(cb)),
                    e,
                )

    @property
    def recent(self) -> list[PainSignal]:
        # Lossy reconstruction from reaction_bus history. No current
        # callers in-tree — kept for API compatibility.
        # TODO: back this with a dedicated PainSignal ring once callers
        # materialize. Until then, direct subscribers see rich context
        # while `.recent` sees the lossy view.
        return [_reaction_to_pain_signal(r) for r in self.reaction_bus.history("pain")]

    def recent_by_type(self, pain_type: PainType) -> list[PainSignal]:
        return [s for s in self.recent if s.pain_type == pain_type]

    def get_stats(self) -> dict[str, int]:
        stats = self.reaction_bus.get_stats()
        # ReactionBus's subscriber_count does NOT know about our direct
        # PainSignal subscriber list. Sum them explicitly so doctor /
        # telemetry see the full count.
        with self._lock:
            direct_count = len(self._pain_signal_subs)
        return {
            "total_published": stats["total_published"],
            "subscriber_count": stats["subscriber_count"] + direct_count,
            "direct_pain_subscribers": direct_count,
            "history_size": stats["history_size"],
        }


def _reaction_to_pain_signal(reaction: "Reaction") -> PainSignal:
    """Convert a Reaction(kind="pain") back to PainSignal for legacy callers."""
    source = reaction.source or ""
    pain_type_str = source.split(":")[-1] if ":" in source else "external_signal"
    try:
        pain_type = PainType(pain_type_str)
    except ValueError:
        pain_type = PainType.EXTERNAL_SIGNAL

    entity_binding = reaction.context.bindings.get("entity_path")
    context: dict[str, Any] = {}
    if entity_binding:
        context["entity_path"] = entity_binding.percept_id

    return PainSignal(
        pain_type=pain_type,
        intensity=reaction.intensity,
        timestamp=reaction.timestamp,
        context=context,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Step 6: PainBus → Hippocampus episodic memory subscriber
# ─────────────────────────────────────────────────────────────────────────────


def create_pain_memory_subscriber(
    hippocampus: Hippocampus,
    intensity_threshold: float = 0.4,
) -> Callable[[PainSignal], None]:
    """Create a PainBus subscriber that captures pain as episodic memory.

    When pain intensity exceeds the threshold, an episodic memory is
    created in the hippocampus with the pain type, intensity, and context.

    Args:
        hippocampus: Hippocampus instance for memory capture.
        intensity_threshold: Minimum pain intensity to trigger memory
            formation. Default 0.4 captures moderate-to-severe pain.
    """
    from maxim.memory.types import Decision, Outcome, Perception

    def _on_pain(signal: PainSignal) -> None:
        if signal.intensity < intensity_threshold:
            return

        hippocampus.capture(
            perception=Perception(
                observations={
                    "pain_type": signal.pain_type.value,
                    "intensity": signal.intensity,
                    **signal.context,
                },
                salience=min(signal.intensity + 0.2, 1.0),
                novelty=0.6,
            ),
            decision=Decision(
                intent={"goal": "pain_response"},
                reasoning=(f"Pain detected: {signal.pain_type.value} (intensity={signal.intensity:.2f})"),
            ),
            outcome=Outcome(
                success=False,
                result={
                    "pain_type": signal.pain_type.value,
                    "intensity": signal.intensity,
                    "context": signal.context,
                },
            ),
        )

        # Simulation verbosity
        try:
            from maxim.simulation.sim_logger import sim_memory

            sim_memory(
                f"Pain memory captured: {signal.pain_type.value} (intensity={signal.intensity:.2f})",
            )
        except Exception:
            pass

    return _on_pain


def create_pain_nac_subscriber(
    nac: Any,
    intensity_threshold: float = 0.3,
) -> Callable[[PainSignal], None]:
    """Create a PainBus subscriber that attributes pain to recent actions.

    Pre-Stage-2 this function recorded tautological ``pain → pain``
    causal links via ``nac.observe``, which could never satisfy a
    ``nac.predict("action", ...)`` query. The rewrite calls
    ``nac.record_outcome_full`` with NO ``attributed_event_id`` and
    NO ``attributed_event_signature`` so NAc walks its
    ``_pending_events`` buffer within the temporal window and links
    the pain outcome to any pending action event whose context is
    similar enough to the pain context (per
    ``NACConfig.context_similarity_threshold``). For the link to
    form, the agent must have called
    ``nac.record_event("action", signature, context={"source":...,
    "entity":...})`` before taking the action that caused the pain.
    This is the full Percept→Reaction→Learning loop.

    Context handling: the subscriber passes the **full**
    ``signal.context`` dict through to ``record_outcome_full``.
    ``NAc._context_similarity`` is directional — ``len(event_context)``
    is the denominator, so extra cause-description keys on the outcome
    side do NOT dilute the match. This lets the outcome side carry rich
    metadata (``source``, ``entity``, ``failure_mode``, ``composes``,
    ``sensor_readings``, ...) without hurting attribution. A pre-
    Stage-2 Substrate P2 draft tried a "slim attribution context"
    workaround here — do not re-introduce it; the root-cause fix lives
    in ``NAc._context_similarity`` and any subscriber-side shim re-
    creates the inconsistency with ``ToolPainBridge._on_embodiment_pain``
    (which passes a 7-key dict on the same bus).

    Exceptions are logged (not silently swallowed) so a broken wiring
    surfaces in the log rather than silently degrading learning.

    Args:
        nac: NAc instance.
        intensity_threshold: Minimum intensity to trigger learning.
    """
    from maxim.decisions.causal_link import Valence

    def _on_pain(signal: PainSignal) -> None:
        if signal.intensity < intensity_threshold:
            return

        # Suppress NAc causal learning during interactive mode — human-
        # directed actions would corrupt the causal model. See
        # plans/README.md "Interactive NAc attribution".
        try:
            from maxim.simulation.sim_logger import get_interactive_mode, InteractiveMode

            if get_interactive_mode() == InteractiveMode.ON:
                return
        except Exception:
            pass

        context = dict(signal.context or {})
        source = context.get("source", "unknown")
        entity = context.get("entity") or context.get("entity_path") or "unknown"
        failure = context.get("failure_mode") or signal.pain_type.name
        outcome_signature = f"pain:{source}:{entity}:{failure}"

        try:
            nac.record_outcome_full(
                outcome_type="pain",
                outcome_signature=outcome_signature,
                outcome_valence=Valence.NEGATIVE,
                context=context,
            )
        except Exception:
            logger.exception("pain→NAc outcome recording failed")

    return _on_pain


# ─────────────────────────────────────────────────────────────────────────────
# Canonical PainBus construction site (Wave 1, biosystem_unification)
# ─────────────────────────────────────────────────────────────────────────────


def build_pain_bus(
    *,
    hippocampus: "Hippocampus | None",
    nac: "NAc | None",
    additional_subscribers: tuple[Callable[[PainSignal], None], ...] = (),
    history_size: int = 200,
    pain_refractory_s: float | None = None,
    bus: PainBus | None = None,
) -> PainBus:
    """Construct a PainBus with explicit learning-subscriber decisions.

    This is the canonical PainBus construction site for production
    code. ``hippocampus`` and ``nac`` are REQUIRED keyword-only
    arguments — forgetting either is a ``TypeError``, not a silent
    no-op. The required-kwarg shape mirrors
    ``runtime/bootstrap.py::build_executor`` and applies the same L1
    structural-enforcement rule from
    ``docs/architecture/structural_enforcement.md``: silent-failure
    bug classes get pushed down into the constructor so the next call
    site cannot reproduce them.

    The motivating bug (audited in
    ``docs/plans/pain_bus_unification.md``): three CLI entry points
    constructed a ``PainBus()`` and subscribed
    ``create_pain_memory_subscriber`` but never
    ``create_pain_nac_subscriber``, even though ``_cli_nac`` was in
    scope. Out-of-band SEM pain (autonomous body-sensor decay,
    sandbox triggers, ``body.py::_publish_pain`` while no tool is in
    flight) reached hippocampus on those paths but never reached NAc.
    The bug was invisible to existing tests because tool-invoked pain
    still reached NAc via the direct-attribution path through
    ``ToolPainBridge.record_tool_embodiment_failure``. Three identical
    instances across three entry points = the canonical L1
    silent-failure-mode shape.

    Args:
        hippocampus: ``Hippocampus`` instance to subscribe via
            ``create_pain_memory_subscriber``. Pass ``None`` to
            explicitly opt out (sandbox / test / headless agents that
            don't capture pain memories). Required keyword-only.
        nac: ``NAc`` instance to subscribe via
            ``create_pain_nac_subscriber``. Pass ``None`` to explicitly
            opt out. Required keyword-only. Note: tool-invoked SEM pain
            attributes to NAc directly via
            ``ToolPainBridge.record_tool_embodiment_failure`` regardless
            of this subscription — this kwarg controls the bus-fallback
            path used for **out-of-band** pain (autonomous SEM ticks,
            sandbox events, ambient body-sensor decay).
        additional_subscribers: Optional extra ``PainSignal``
            callbacks registered after the standard learners. Use
            for adapter-shaped subscribers (telemetry, fear-gate, etc.)
            that need the rich ``signal.context``. Keep additional
            subscribers cheap — they run synchronously inside
            ``PainBus.publish``.
        history_size: Forwarded to the underlying ``ReactionBus``.
            Ignored when ``bus`` is provided.
        pain_refractory_s: Forwarded to ``PainBus.__init__``. Default
            is ``PainBus.DEFAULT_PAIN_REFRACTORY_S``.  Ignored when
            ``bus`` is provided.
        bus: Pre-constructed PainBus.  When provided, subscribers are
            registered on this bus instead of constructing a new one.
            Use for the sim AUT pattern where the sandbox needs a
            PainBus before the rest of the bio-pipeline.  Added in
            Wave 3 (``bio_stack_unification.md``) so that
            ``build_bio_stack`` can route through this door even when
            the bus exists before the bio-systems.

    Returns:
        A fully-constructed ``PainBus`` with the standard learners
        already subscribed.

    Examples:
        Production CLI agent path (the post-migration shape)::

            pain_bus = build_pain_bus(
                hippocampus=cli_hippocampus,
                nac=cli_nac,
            )
            executor = build_executor(
                registry,
                pain_bus=pain_bus,
                nac=cli_nac,
                hippocampus=cli_hippocampus,
                ...,
            )

        Explicit no-learners opt-out (test / sandbox / headless)::

            pain_bus = build_pain_bus(hippocampus=None, nac=None)

    See also:
        - ``docs/plans/pain_bus_unification.md`` for the audit + design
        - ``docs/architecture/structural_enforcement.md`` for the rule
        - ``runtime/bootstrap.py::build_executor`` for the precedent
    """
    if bus is None:
        bus = PainBus(
            history_size=history_size,
            pain_refractory_s=pain_refractory_s,
            _allow_raw=True,
        )
    if hippocampus is not None:
        bus.subscribe(create_pain_memory_subscriber(hippocampus))
    if nac is not None:
        bus.subscribe(create_pain_nac_subscriber(nac))
    for sub in additional_subscribers:
        bus.subscribe(sub)
    return bus
