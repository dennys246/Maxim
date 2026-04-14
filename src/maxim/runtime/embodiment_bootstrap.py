"""Shared embodiment + ToolPainBridge bootstrap — Stage 2 of sem_execution_hook.

Three production paths construct an ``Executor`` + optional SEM
embodiment:

1. ``cli.py`` — the ``maxim --llm X`` non-sim agent entry point.
2. ``simulation/orchestrator.py`` — the AUT sim runner.
3. ``embodied_runtime/agentic_runtime.py`` — the Reachy robot runtime.

Before this helper, each path built its own ``ToolPainBridge`` (or
didn't — the ``cli.py`` path was missing one entirely, which meant the
Stage 1 embodiment-failure attribution shipped in PR #107 was a silent
no-op on every regular ``maxim --llm X`` run). This file exposes one
function the three sites call so the wiring stays in sync, the CLI gap
is closed, and adding a SEM body to a regular CLI session is a single
kwarg flip.

Contract:

- ``ToolPainBridge`` is constructed and assigned to
  ``executor._tool_pain_bridge`` **unconditionally**. This is the
  load-bearing behavior that closes the CLI ``ToolPainBridge`` gap; do
  NOT gate it on ``entity_ref``.
- ``pain_bus`` is OPTIONAL when ``entity_ref`` is ``None`` (legacy
  Reachy path uses ``pain_detector`` instead of a PainBus) but
  REQUIRED when ``entity_ref`` is provided, because
  ``Embodiment._publish_pain`` emits ``PainSignal`` through the bus.
- ``pain_detector`` is an OPTIONAL legacy subscription path for
  ``ToolPainBridge`` — pre-PainBus callers (Reachy runtime) wire
  tool-error signals through ``PainDetector.add_pain_callback``. New
  call sites should prefer ``pain_bus=``; leave ``pain_detector`` as
  ``None``.
- When ``entity_ref`` is provided, the helper loads the component
  from ``component_registry``, wraps it in ``Embodiment(pain_bus=...)``,
  and calls ``generate_tools_for_entity`` so affordance tools
  (``rusty_sword_slash`` etc.) land in ``executor.registry``.
- ``entity_ref`` without ``component_registry`` is a caller bug and
  raises ``ValueError``. A typo in a ref propagates
  ``ComponentNotFoundError`` from the registry (which includes a list
  of available refs) — both failures are loud by design.

See ``docs/plans/sem_execution_hook.md`` Stage 2 for the full shape +
why we chose to fold the CLI bridge gap into this stage instead of
splitting it into its own PR.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from maxim.embodiment.body import Embodiment
    from maxim.embodiment.component_registry import ComponentRegistry
    from maxim.bridges.tool_pain_bridge import ToolPainBridge

logger = logging.getLogger(__name__)


def bootstrap_embodiment_and_pain_bridge(
    *,
    nac: Any,
    hippocampus: Any | None,
    scn: Any | None,
    executor: Any,
    pain_bus: Any | None = None,
    pain_detector: Any | None = None,
    entity_ref: str | None = None,
    component_registry: "ComponentRegistry | None" = None,
    tool_index: Any | None = None,
    cerebellum: Any | None = None,
) -> tuple["Embodiment | None", "ToolPainBridge"]:
    """Wire a ``ToolPainBridge`` onto ``executor`` and optionally load a SEM body.

    Args:
        nac: NAc instance the bridge will record outcomes against.
            Required — there is no meaningful bridge without NAc.
        hippocampus: Optional ``Hippocampus`` for reflexion memory
            storage. Forwarded to ``ToolPainBridge``.
        scn: Optional ``SCN`` for temporal signature registration.
            Forwarded to ``ToolPainBridge``.
        executor: The ``Executor`` to receive ``_tool_pain_bridge``.
            The helper assigns the bridge directly — consistent with
            the existing wiring in ``simulation/orchestrator.py`` and
            ``embodied_runtime/agentic_runtime.py``.
        pain_bus: Optional ``PainBus`` for the bridge to subscribe
            against. REQUIRED when ``entity_ref`` is set (because
            ``Embodiment._publish_pain`` emits through the bus).
            Every caller that needs embodiment owns construction.
        pain_detector: Optional legacy subscription path for
            ``ToolPainBridge`` — forwarded to the bridge constructor
            for call sites (Reachy runtime) that pre-date PainBus.
            New call sites should leave this ``None`` and use
            ``pain_bus`` instead.
        entity_ref: Optional SEM component reference (e.g.,
            ``"weapons/rusty_sword"``). When provided, the helper
            instantiates the entity, wraps it in ``Embodiment``, and
            generates affordance tools into ``executor.registry``.
        component_registry: Required IFF ``entity_ref`` is provided.
            Caller bug to pass one without the other.
        tool_index: Optional ``LearnedToolIndex`` forwarded to
            ``ToolPainBridge`` for keyword-weight updates.
        cerebellum: Optional ``Cerebellum`` forwarded to
            ``generate_tools_for_entity`` for forward-model training.

    Returns:
        ``(embodiment, bridge)`` — ``embodiment`` is ``None`` when
        ``entity_ref`` was not provided; ``bridge`` is always a live
        ``ToolPainBridge`` attached to the executor.

    Raises:
        ValueError: If ``entity_ref`` is set but ``component_registry``
            is ``None``, or if ``entity_ref`` is set but ``pain_bus``
            is ``None``, or if BOTH ``pain_bus`` and ``pain_detector``
            are passed (ambiguous subscription path).
        ComponentNotFoundError: If ``entity_ref`` does not resolve in
            the registry. The exception includes a sorted list of
            available refs so the user can spot a typo.

    Wrapping-order invariant: ``executor`` MUST be the unwrapped inner
    ``Executor``. Stage 1's embodiment-failure attribution reads
    ``self._tool_pain_bridge`` from the Executor that runs
    ``tool.run()`` — wrapping layers (``FearGatedExecutor``,
    ``PainInterceptorExecutor``) MUST be applied AFTER this call.
    Calling the helper on a wrapper would assign the bridge to the
    wrapper, leaving the inner executor's ``_tool_pain_bridge=None``
    and silently restoring the Stage 1 no-op bug. The helper validates
    this by rejecting any object that doesn't expose a ``registry``
    attribute AND lacks ``_tool_pain_bridge`` as a writable instance
    field pattern — but the contract is your responsibility at the
    call site first.
    """
    # ── Fail-fast precondition checks (run BEFORE any mutation) ────
    # Contract: if the caller has made a mistake, the executor must
    # remain untouched so the caller's exception handler (if any) can
    # safely continue without a half-wired bridge. Cross-confirmed
    # finding from the Stage 2 pre-merge review: a post-construction
    # ValueError was leaving executor._tool_pain_bridge already
    # assigned to a dead-subscription bridge.
    if pain_bus is not None and pain_detector is not None:
        raise ValueError(
            "bootstrap_embodiment_and_pain_bridge: pain_bus AND pain_detector "
            "were both provided. ToolPainBridge.__init__ uses pain_bus if set "
            "and silently ignores pain_detector — pick one."
        )
    if entity_ref is not None and component_registry is None:
        raise ValueError(
            "bootstrap_embodiment_and_pain_bridge: entity_ref=%r was provided "
            "but component_registry is None. Pass a ComponentRegistry instance "
            "or leave entity_ref as None." % entity_ref
        )
    if entity_ref is not None and pain_bus is None:
        raise ValueError(
            "bootstrap_embodiment_and_pain_bridge: entity_ref=%r was provided "
            "but pain_bus is None. Embodiment._publish_pain emits through the "
            "bus; pass a PainBus instance." % entity_ref
        )

    # Local imports keep the helper cheap to import when unused —
    # `maxim.bridges.tool_pain_bridge` in turn imports NAc which
    # imports substrate components we don't want to pay for on a
    # non-agent CLI entry point like `maxim doctor`.
    from maxim.bridges.tool_pain_bridge import ToolPainBridge

    bridge = ToolPainBridge(
        nac=nac,
        pain_detector=pain_detector,
        scn=scn,
        hippocampus=hippocampus,
        tool_index=tool_index,
        pain_bus=pain_bus,
    )
    executor._tool_pain_bridge = bridge
    logger.info(
        "ToolPainBridge wired to executor (entity_ref=%s)",
        entity_ref or "<none>",
    )

    if entity_ref is None:
        return None, bridge

    from maxim.embodiment.body import Embodiment
    from maxim.embodiment.tool_bridge import generate_tools_for_entity

    # Resolver errors (unknown ref, malformed spec) propagate to the
    # caller with their original exception type + message. The CLI
    # catches these at a higher level and produces a user-visible
    # "component not found" error.
    entity = component_registry.instantiate(entity_ref)
    embodiment = Embodiment(entity, pain_bus=pain_bus)

    generated = generate_tools_for_entity(
        entity,
        executor.registry,
        embodiment=embodiment,
        cerebellum=cerebellum,
    )
    logger.info(
        "Embodiment loaded from %r: %d affordance tools registered",
        entity_ref,
        len(generated),
    )

    return embodiment, bridge


__all__ = ["bootstrap_embodiment_and_pain_bridge"]
