"""Tests for ``runtime/bootstrap.py::build_executor``.

The canonical agent-construction site after the executor bootstrap
unification plan landed. This file replaces ``test_embodiment_bootstrap.py``
which exercised the previous helper-discipline shape.

Three identical bug instances (sem_execution_hook Stages 1, 2, 2c)
showed that "remember to call the helper after build_executor" is not a
strong enough invariant. ``build_executor`` now requires an explicit
``pain_bus=`` decision — forgetting becomes a ``TypeError`` instead of a
silent no-op.

See ``docs/plans/executor_bootstrap_unification.md``.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Required-keyword contract — the structural enforcement
# ---------------------------------------------------------------------------


class TestBuildExecutorRequiredKeyword:
    """``pain_bus`` is a required keyword-only arg with no default. The
    structural intent: forgetting the bridge decision is impossible.
    ``pain_bus=None`` is a legal explicit opt-out; missing the kwarg
    entirely is a TypeError."""

    def test_missing_pain_bus_kwarg_raises_type_error(self):
        from maxim.runtime.bootstrap import build_executor
        from maxim.tools.registry import ToolRegistry

        # Tight match: must be the missing-keyword-only TypeError, not
        # some other TypeError that incidentally contains "pain_bus".
        with pytest.raises(TypeError, match=r"missing.*keyword-only argument.*pain_bus"):
            build_executor(ToolRegistry())  # type: ignore[call-arg]

    def test_pain_bus_none_is_explicit_opt_out_no_bridge(self):
        """Sandbox executors / headless tests pass ``pain_bus=None``
        explicitly. The executor exists but has no bridge; tool
        outcomes are not learned."""
        from maxim.runtime.bootstrap import build_executor
        from maxim.tools.registry import ToolRegistry

        executor = build_executor(ToolRegistry(), pain_bus=None)

        assert executor is not None
        assert executor._tool_pain_bridge is None

    def test_positional_pain_bus_rejected_keyword_only(self):
        """``pain_bus`` is keyword-only — passing it positionally
        should fail at the signature level."""
        from maxim.proprioception.pain_bus import PainBus
        from maxim.runtime.bootstrap import build_executor
        from maxim.tools.registry import ToolRegistry

        with pytest.raises(TypeError):
            build_executor(ToolRegistry(), PainBus(_allow_raw=True))  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Bridge wiring — fires when the caller opts in
# ---------------------------------------------------------------------------


class TestBuildExecutorBridgeWiring:
    """When ``pain_bus`` (or ``pain_detector``) is provided, the bridge
    is constructed and attached to the inner Executor. This is the
    invariant that closed the CLI gap in sem_execution_hook Stage 2 —
    now structurally enforced."""

    def test_bridge_wired_with_nac_only_no_subscription_source(self):
        """C2 cross-confirmed regression guard: the bridge's PRIMARY
        value is direct attribution via record_tool_embodiment_failure
        (Stage 1). Subscription via pain_bus/pain_detector is the
        SECONDARY out-of-band path. A caller with NAc but no
        subscription source (e.g., the sim orchestrator's AUT, where
        NAc is subscribed to the bus directly via
        create_pain_nac_subscriber) MUST get a bridge for direct
        attribution. Pre-fold the bridge was gated on
        pain_bus|pain_detector — direct-attribution-only callers had
        to pass a no-op PainDetector to trick the constructor."""
        from maxim.bridges.tool_pain_bridge import ToolPainBridge
        from maxim.runtime.bootstrap import build_executor
        from maxim.tools.registry import ToolRegistry

        nac = MagicMock()

        executor = build_executor(
            ToolRegistry(),
            pain_bus=None,
            pain_detector=None,
            nac=nac,
        )

        assert isinstance(executor._tool_pain_bridge, ToolPainBridge), (
            "build_executor with nac=NAc, pain_bus=None, pain_detector=None "
            "must construct a bridge for direct attribution. The bridge is "
            "the primary path for record_tool_embodiment_failure (Stage 1)."
        )

    def test_no_bridge_when_nac_is_none(self):
        """Inverse of the above: explicit opt-out via nac=None
        produces no bridge regardless of any subscription source."""
        from maxim.runtime.bootstrap import build_executor
        from maxim.tools.registry import ToolRegistry

        executor = build_executor(
            ToolRegistry(),
            pain_bus=None,
            nac=None,
        )

        assert executor._tool_pain_bridge is None

    def test_bridge_wired_when_pain_bus_provided(self):
        from maxim.bridges.tool_pain_bridge import ToolPainBridge
        from maxim.proprioception.pain_bus import PainBus
        from maxim.runtime.bootstrap import build_executor
        from maxim.tools.registry import ToolRegistry

        nac = MagicMock()
        pain_bus = PainBus(_allow_raw=True)

        executor = build_executor(
            ToolRegistry(),
            pain_bus=pain_bus,
            nac=nac,
        )

        assert isinstance(executor._tool_pain_bridge, ToolPainBridge)

    def test_bridge_wired_when_pain_detector_provided(self):
        """Legacy Reachy path uses ``pain_detector`` instead of a
        PainBus. Same bridge construction, different subscription."""
        from maxim.bridges.tool_pain_bridge import ToolPainBridge
        from maxim.runtime.bootstrap import build_executor
        from maxim.tools.registry import ToolRegistry

        nac = MagicMock()
        pain_detector = MagicMock()

        executor = build_executor(
            ToolRegistry(),
            pain_bus=None,
            pain_detector=pain_detector,
            nac=nac,
        )

        assert isinstance(executor._tool_pain_bridge, ToolPainBridge)

    def test_no_bridge_without_nac_raises(self):
        """A bridge with ``nac=None`` has no meaningful behavior. Fail
        fast so the caller passes NAc or opts out explicitly."""
        from maxim.proprioception.pain_bus import PainBus
        from maxim.runtime.bootstrap import build_executor
        from maxim.tools.registry import ToolRegistry

        with pytest.raises(ValueError, match="nac"):
            build_executor(
                ToolRegistry(),
                pain_bus=PainBus(_allow_raw=True),
                nac=None,
            )


# ---------------------------------------------------------------------------
# Fail-fast preconditions — checked BEFORE any construction
# ---------------------------------------------------------------------------


class TestBuildExecutorFailFastPreconditions:
    """Precondition checks must run BEFORE any object construction so
    a ValueError leaves no half-built state. Cross-confirmed Stage 2
    review finding: pre-fold the helper built the bridge BEFORE
    checking, leaving a dead-subscription bridge stuck on the
    executor when callers recovered from the exception."""

    def test_pain_bus_and_pain_detector_together_raises(self):
        """Passing both is ambiguous — ToolPainBridge uses pain_bus if
        set and silently drops pain_detector. Fail fast."""
        from maxim.proprioception.pain_bus import PainBus
        from maxim.runtime.bootstrap import build_executor
        from maxim.tools.registry import ToolRegistry

        nac = MagicMock()
        pain_bus = PainBus(_allow_raw=True)
        pain_detector = MagicMock()

        with pytest.raises(ValueError, match="pain_bus AND pain_detector"):
            build_executor(
                ToolRegistry(),
                pain_bus=pain_bus,
                pain_detector=pain_detector,
                nac=nac,
            )

    def test_entity_ref_without_pain_bus_raises(self):
        """Embodiment._publish_pain emits through the bus — without a
        bus, embodiment failures fire into the void."""
        from maxim.embodiment.component_registry import ComponentRegistry
        from maxim.runtime.bootstrap import build_executor
        from maxim.tools.registry import ToolRegistry

        nac = MagicMock()

        with pytest.raises(ValueError, match="pain_bus"):
            build_executor(
                ToolRegistry(),
                pain_bus=None,
                nac=nac,
                entity_ref="weapons/rusty_sword",
                component_registry=ComponentRegistry(),
            )

    def test_entity_ref_without_component_registry_raises(self):
        from maxim.proprioception.pain_bus import PainBus
        from maxim.runtime.bootstrap import build_executor
        from maxim.tools.registry import ToolRegistry

        nac = MagicMock()
        pain_bus = PainBus(_allow_raw=True)

        with pytest.raises(ValueError, match="component_registry"):
            build_executor(
                ToolRegistry(),
                pain_bus=pain_bus,
                nac=nac,
                entity_ref="weapons/rusty_sword",
                component_registry=None,
            )


# ---------------------------------------------------------------------------
# Embodiment loading — activates when entity_ref is provided
# ---------------------------------------------------------------------------


class TestBuildExecutorEmbodiment:
    """When ``entity_ref`` is provided, the function loads the
    component, wraps it in ``Embodiment``, and registers affordance
    tools. Uses bundled ``weapons/rusty_sword`` — no mocks in the SEM
    chain."""

    def test_rusty_sword_ref_registers_affordance_tools(self):
        from maxim.embodiment.body import Embodiment
        from maxim.embodiment.component_registry import ComponentRegistry
        from maxim.proprioception.pain_bus import PainBus
        from maxim.runtime.bootstrap import build_executor
        from maxim.tools.registry import ToolRegistry

        nac = MagicMock()
        pain_bus = PainBus(_allow_raw=True)
        registry = ToolRegistry()

        executor = build_executor(
            registry,
            pain_bus=pain_bus,
            nac=nac,
            entity_ref="weapons/rusty_sword",
            component_registry=ComponentRegistry(),
        )

        # The embodiment is stashed on the executor for caller access.
        embodiment = executor.embodiment
        assert isinstance(embodiment, Embodiment)

        tool_names = set(executor.registry.list())
        for expected in (
            "rusty_sword_slash",
            "rusty_sword_parry",
            "rusty_sword_throw",
            "rusty_sword_sharpen",
            "rusty_sword_repair",
        ):
            assert expected in tool_names, f"expected {expected!r} in executor.registry, got {sorted(tool_names)}"

    def test_embodiment_holds_pain_bus_reference(self):
        """The Embodiment must hold the pain_bus reference so
        evaluate_failures can publish PainSignals — the cascade Stage 1
        fixed depends on this link."""
        from maxim.embodiment.component_registry import ComponentRegistry
        from maxim.proprioception.pain_bus import PainBus
        from maxim.runtime.bootstrap import build_executor
        from maxim.tools.registry import ToolRegistry

        nac = MagicMock()
        pain_bus = PainBus(_allow_raw=True)

        executor = build_executor(
            ToolRegistry(),
            pain_bus=pain_bus,
            nac=nac,
            entity_ref="weapons/rusty_sword",
            component_registry=ComponentRegistry(),
        )

        embodiment = executor.embodiment
        assert embodiment._pain_bus is pain_bus

    def test_missing_entity_ref_raises_with_actionable_hint(self):
        """A typo in entity_ref must fail with a hint listing real
        alternatives — never silently no-op."""
        from maxim.embodiment.component_registry import ComponentRegistry
        from maxim.exceptions import ComponentNotFoundError
        from maxim.proprioception.pain_bus import PainBus
        from maxim.runtime.bootstrap import build_executor
        from maxim.tools.registry import ToolRegistry

        nac = MagicMock()
        pain_bus = PainBus(_allow_raw=True)

        with pytest.raises(ComponentNotFoundError) as exc_info:
            build_executor(
                ToolRegistry(),
                pain_bus=pain_bus,
                nac=nac,
                entity_ref="weapons/nonexistent_sword",
                component_registry=ComponentRegistry(),
            )
        msg = str(exc_info.value)
        assert "weapons/nonexistent_sword" in msg
        assert "weapons/rusty_sword" in msg


# ---------------------------------------------------------------------------
# End-to-end cascade through the new constructor
# ---------------------------------------------------------------------------


class TestBuildExecutorEndToEndCascade:
    """The integration story: after build_executor, invoking a
    generated affordance tool through executor.execute drives the full
    Stage 1 pain cascade. NAc records a NEGATIVE link on failure. Any
    failure here means the cascade is broken at one of: bridge
    construction, executor wiring, embodiment loading, side_effects
    routing, or NAc record_tool_embodiment_failure."""

    def test_rusty_sword_slash_to_shatter_records_negative(self):
        from maxim.decisions.causal_link import Valence
        from maxim.decisions.nac import NAc
        from maxim.embodiment.component_registry import ComponentRegistry
        from maxim.proprioception.pain_bus import PainBus
        from maxim.runtime.bootstrap import build_executor
        from maxim.tools.registry import ToolRegistry

        nac = NAc()
        pain_bus = PainBus(_allow_raw=True)

        executor = build_executor(
            ToolRegistry(),
            pain_bus=pain_bus,
            nac=nac,
            entity_ref="weapons/rusty_sword",
            component_registry=ComponentRegistry(),
        )

        # Drive durability to zero so shatter fires on the next
        # evaluate_failures() call. Deterministic at this durability —
        # verified 2026-04-14, regression-guarded here.
        embodiment = executor.embodiment
        sword = embodiment.root
        sword.vital_metrics["durability"] = 0.0

        result = executor.execute(
            {
                "tool_name": "rusty_sword_slash",
                "params": {"target": "dummy", "force": 0.9},
            }
        )

        assert result.success is True
        assert result.side_effects is not None
        assert result.side_effects.get("embodiment_failures"), (
            "rusty_sword at durability=0.0 did not produce "
            "embodiment_failures; check embodiment.evaluate_failures() "
            "wiring in ModulatorAffordanceTool.execute"
        )

        prediction = nac.predict(
            event_type="tool",
            event_signature="tool:rusty_sword_slash",
        )
        assert prediction is not None, (
            "NAc has no prediction for tool:rusty_sword_slash after a "
            "deterministic shatter. The cascade is broken — either the "
            "bridge was not attached by build_executor, "
            "record_tool_embodiment_failure was not called, or NAc did "
            "not form a causal link."
        )
        assert prediction.predicted_valence == Valence.NEGATIVE
