"""Tests for runtime/embodiment_bootstrap.py — the shared Stage 2 helper.

This helper closes two pre-existing production gaps:

1. `maxim --llm X` in `cli.py` never constructed a `ToolPainBridge`.
   Stage 1's embodiment-failure attribution is a silent no-op on that
   path without the bridge. Stage 2 wires it unconditionally.

2. `runtime/agent_factory.py::AgentFactory.create_agent` stores a raw
   entity on `AgentInstance.entity` but never wraps it in an
   `Embodiment` or generates SEM affordance tools. `maxim --llm X
   --embodiment weapons/rusty_sword` had no way to load a SEM body
   before this helper.

The helper is the single bootstrap site for three production paths
(sim orchestrator / Reachy runtime / CLI) — each invokes the same
function so there's one implementation, one invariant, one place to
break.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Bridge wiring — fires unconditionally, embodiment optional
# ---------------------------------------------------------------------------


class TestBootstrapWiresBridgeUnconditionally:
    """The whole point of this helper: `_tool_pain_bridge` MUST be wired
    on the executor whether or not `entity_ref` was passed. Pre-fix the
    CLI path had no bridge at all — tool failures silently skipped NAc
    learning."""

    def _make_executor(self):
        from maxim.runtime.executor import Executor
        from maxim.tools.registry import ToolRegistry

        return Executor(tool_registry=ToolRegistry())

    def test_bridge_wired_when_entity_ref_is_none(self):
        """Core regression guard for the pre-existing CLI gap."""
        from maxim.bridges.tool_pain_bridge import ToolPainBridge
        from maxim.proprioception.pain_bus import PainBus
        from maxim.runtime.embodiment_bootstrap import (
            bootstrap_embodiment_and_pain_bridge,
        )

        executor = self._make_executor()
        nac = MagicMock()
        pain_bus = PainBus()

        embodiment, bridge = bootstrap_embodiment_and_pain_bridge(
            nac=nac,
            hippocampus=None,
            scn=None,
            pain_bus=pain_bus,
            executor=executor,
            entity_ref=None,
        )

        assert embodiment is None
        assert isinstance(bridge, ToolPainBridge)
        assert executor._tool_pain_bridge is bridge

    def test_bridge_wired_even_when_component_registry_is_none(self):
        """A CLI user with no `--embodiment` flag doesn't need a registry.
        The helper must still produce a live bridge."""
        from maxim.proprioception.pain_bus import PainBus
        from maxim.runtime.embodiment_bootstrap import (
            bootstrap_embodiment_and_pain_bridge,
        )

        executor = self._make_executor()
        nac = MagicMock()
        pain_bus = PainBus()

        embodiment, bridge = bootstrap_embodiment_and_pain_bridge(
            nac=nac,
            hippocampus=None,
            scn=None,
            pain_bus=pain_bus,
            executor=executor,
            entity_ref=None,
            component_registry=None,
        )

        assert embodiment is None
        assert executor._tool_pain_bridge is not None


# ---------------------------------------------------------------------------
# Embodiment wiring — activates when entity_ref is provided
# ---------------------------------------------------------------------------


class TestBootstrapWiresEmbodimentOnRef:
    """When `entity_ref` is provided, the helper loads the component,
    wraps in Embodiment, and generates affordance tools into the
    executor's registry. Uses the bundled `weapons/rusty_sword` for the
    test — no mocks in the SEM chain."""

    def _make_executor(self):
        from maxim.runtime.executor import Executor
        from maxim.tools.registry import ToolRegistry

        return Executor(tool_registry=ToolRegistry())

    def test_rusty_sword_ref_registers_affordance_tools(self):
        from maxim.embodiment.body import Embodiment
        from maxim.embodiment.component_registry import ComponentRegistry
        from maxim.proprioception.pain_bus import PainBus
        from maxim.runtime.embodiment_bootstrap import (
            bootstrap_embodiment_and_pain_bridge,
        )

        executor = self._make_executor()
        nac = MagicMock()
        pain_bus = PainBus()
        registry = ComponentRegistry()

        embodiment, bridge = bootstrap_embodiment_and_pain_bridge(
            nac=nac,
            hippocampus=None,
            scn=None,
            pain_bus=pain_bus,
            executor=executor,
            entity_ref="weapons/rusty_sword",
            component_registry=registry,
        )

        assert isinstance(embodiment, Embodiment)
        # The rusty_sword spec has 5 affordances: slash, parry, throw,
        # sharpen, repair. Verify all five land in the tool registry.
        tool_names = set(executor.registry.list())
        for expected in (
            "rusty_sword_slash",
            "rusty_sword_parry",
            "rusty_sword_throw",
            "rusty_sword_sharpen",
            "rusty_sword_repair",
        ):
            assert expected in tool_names, (
                f"expected {expected!r} in executor.registry after bootstrap, got {sorted(tool_names)}"
            )

    def test_embodiment_constructed_with_pain_bus(self):
        """The Embodiment must hold the pain_bus reference so
        `evaluate_failures` can publish PainSignals — the pain cascade
        Stage 1 fixed depends on this link."""
        from maxim.embodiment.component_registry import ComponentRegistry
        from maxim.proprioception.pain_bus import PainBus
        from maxim.runtime.embodiment_bootstrap import (
            bootstrap_embodiment_and_pain_bridge,
        )

        executor = self._make_executor()
        nac = MagicMock()
        pain_bus = PainBus()

        embodiment, _ = bootstrap_embodiment_and_pain_bridge(
            nac=nac,
            hippocampus=None,
            scn=None,
            pain_bus=pain_bus,
            executor=executor,
            entity_ref="weapons/rusty_sword",
            component_registry=ComponentRegistry(),
        )

        assert embodiment is not None
        assert embodiment._pain_bus is pain_bus

    def test_missing_entity_ref_raises_with_actionable_hint(self):
        """A typo in `--embodiment` should fail fast with a hint that
        lists available refs — NOT silently no-op."""
        from maxim.embodiment.component_registry import ComponentRegistry
        from maxim.exceptions import ComponentNotFoundError
        from maxim.proprioception.pain_bus import PainBus
        from maxim.runtime.embodiment_bootstrap import (
            bootstrap_embodiment_and_pain_bridge,
        )

        executor = self._make_executor()
        nac = MagicMock()
        pain_bus = PainBus()
        registry = ComponentRegistry()

        with pytest.raises(ComponentNotFoundError) as exc_info:
            bootstrap_embodiment_and_pain_bridge(
                nac=nac,
                hippocampus=None,
                scn=None,
                pain_bus=pain_bus,
                executor=executor,
                entity_ref="weapons/nonexistent_sword",
                component_registry=registry,
            )
        # The error must include the ref the user typed AND at least
        # one real available alternative — this is the "actionable
        # hint" contract. Without it a typo produces a cryptic failure.
        msg = str(exc_info.value)
        assert "weapons/nonexistent_sword" in msg
        assert "weapons/rusty_sword" in msg

    def test_entity_ref_without_pain_bus_raises(self):
        """Embodiment needs a PainBus to publish failures. Passing
        `entity_ref` without `pain_bus` is a caller bug — fail fast
        AND leave the executor untouched (fail-fast contract)."""
        from maxim.embodiment.component_registry import ComponentRegistry
        from maxim.runtime.embodiment_bootstrap import (
            bootstrap_embodiment_and_pain_bridge,
        )

        executor = self._make_executor()
        nac = MagicMock()

        with pytest.raises(ValueError, match="pain_bus"):
            bootstrap_embodiment_and_pain_bridge(
                nac=nac,
                hippocampus=None,
                scn=None,
                executor=executor,
                pain_bus=None,
                entity_ref="weapons/rusty_sword",
                component_registry=ComponentRegistry(),
            )
        # CRITICAL: the executor must be untouched. Pre-fold, the
        # helper built the bridge + assigned it to
        # `executor._tool_pain_bridge` BEFORE the ValueError check,
        # leaving a dead-subscription bridge attached when the caller
        # recovered from the exception. Pre-merge review caught this.
        assert executor._tool_pain_bridge is None

    def test_pain_bus_and_pain_detector_together_raises(self):
        """Passing both pain_bus AND pain_detector is a caller bug —
        ToolPainBridge uses pain_bus if set and silently drops
        pain_detector. Fail fast so the caller fixes the call site."""
        from maxim.proprioception.pain_bus import PainBus
        from maxim.runtime.embodiment_bootstrap import (
            bootstrap_embodiment_and_pain_bridge,
        )

        executor = self._make_executor()
        nac = MagicMock()
        pain_bus = PainBus()
        pain_detector = MagicMock()

        with pytest.raises(ValueError, match="pain_bus AND pain_detector"):
            bootstrap_embodiment_and_pain_bridge(
                nac=nac,
                hippocampus=None,
                scn=None,
                executor=executor,
                pain_bus=pain_bus,
                pain_detector=pain_detector,
            )
        # Executor untouched — same fail-fast contract as above.
        assert executor._tool_pain_bridge is None

    def test_entity_ref_without_registry_raises(self):
        """Providing `entity_ref` without a `component_registry` is a
        programming error — the caller wanted embodiment but forgot the
        resolver. Fail fast."""
        from maxim.proprioception.pain_bus import PainBus
        from maxim.runtime.embodiment_bootstrap import (
            bootstrap_embodiment_and_pain_bridge,
        )

        executor = self._make_executor()
        nac = MagicMock()
        pain_bus = PainBus()

        with pytest.raises(ValueError, match="component_registry"):
            bootstrap_embodiment_and_pain_bridge(
                nac=nac,
                hippocampus=None,
                scn=None,
                pain_bus=pain_bus,
                executor=executor,
                entity_ref="weapons/rusty_sword",
                component_registry=None,
            )


# ---------------------------------------------------------------------------
# End-to-end cascade: helper → executor.execute → NAc learns
# ---------------------------------------------------------------------------


class TestBootstrapEndToEndCascade:
    """The integration story: after bootstrap, invoking a generated
    affordance tool through `executor.execute` drives the full Stage 1
    pain cascade. NAc records a NEGATIVE link on failure."""

    def test_rusty_sword_slash_to_shatter_records_negative(self):
        """Drive durability to ~0 so `slash` triggers `shatter`. After
        `executor.execute` returns, NAc should have a NEGATIVE causal
        link on `tool:rusty_sword_slash`."""
        from maxim.decisions.causal_link import Valence
        from maxim.decisions.nac import NAc
        from maxim.embodiment.component_registry import ComponentRegistry
        from maxim.proprioception.pain_bus import PainBus
        from maxim.runtime.embodiment_bootstrap import (
            bootstrap_embodiment_and_pain_bridge,
        )
        from maxim.runtime.executor import Executor
        from maxim.tools.registry import ToolRegistry

        executor = Executor(tool_registry=ToolRegistry())
        nac = NAc()
        pain_bus = PainBus()

        embodiment, _ = bootstrap_embodiment_and_pain_bridge(
            nac=nac,
            hippocampus=None,
            scn=None,
            pain_bus=pain_bus,
            executor=executor,
            entity_ref="weapons/rusty_sword",
            component_registry=ComponentRegistry(),
        )

        # Drive durability to zero so `shatter` (threshold < 0.1)
        # fires on the next evaluate_failures() call. The cascade is
        # deterministic at this durability — verified 2026-04-14.
        assert embodiment is not None
        sword = embodiment.root
        sword.vital_metrics["durability"] = 0.0

        result = executor.execute(
            {
                "tool_name": "rusty_sword_slash",
                "params": {"target": "dummy", "force": 0.9},
            }
        )

        # The tool itself succeeded (it ran); the failure is reported
        # via side_effects, consistent with Stage 1's contract.
        assert result.success is True
        # Stage 1 invariant: side_effects["embodiment_failures"] carries
        # the shatter failure. If this assertion flips, it's a Stage 1
        # regression, not a Stage 2 issue — but we want the guard on
        # this path regardless.
        assert result.side_effects is not None
        assert result.side_effects.get("embodiment_failures"), (
            "rusty_sword at durability=0.0 did not produce embodiment_failures; "
            "check embodiment.evaluate_failures() wiring in ModulatorAffordanceTool.execute"
        )

        # NAc must now have a NEGATIVE link for tool:rusty_sword_slash.
        # No skip path — if this fails, the cascade is broken and we
        # want to see the failure loudly, not silently.
        prediction = nac.predict(
            event_type="tool",
            event_signature="tool:rusty_sword_slash",
        )
        assert prediction is not None, (
            "NAc has no prediction for tool:rusty_sword_slash after a "
            "deterministic shatter. The bootstrap helper failed to wire the "
            "cascade end-to-end — either ToolPainBridge was not attached, "
            "record_tool_embodiment_failure was not called, or NAc did not "
            "form a causal link."
        )
        assert prediction.predicted_valence == Valence.NEGATIVE
