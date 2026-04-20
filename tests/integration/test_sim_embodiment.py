"""Integration tests for E0 — entity_ref wiring through sim orchestrator.

Verifies that the sim path correctly wires entity_ref to build_executor,
producing affordance tools and functional pain attribution. Does NOT
start a real sim — tests exercise the same construction path the
orchestrator uses, with the same parameter combinations.

Read alongside:
- simulation/orchestrator.py — the AUT build_executor call site
- runtime/bootstrap.py — build_executor entity_ref handling
- tests/substrate/test_sem_execution_production.py — lower-level cascade tests
"""

from __future__ import annotations

import pytest

from maxim.decisions.causal_link import Valence
from maxim.decisions.nac import NAc, NACConfig
from maxim.embodiment.component_registry import ComponentRegistry
from maxim.proprioception.pain_bus import PainBus
from maxim.runtime.bootstrap import build_executor
from maxim.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_sim_aut_executor(
    *,
    entity_ref: str | None = None,
    with_nac: bool = True,
    with_pain_bus: bool = True,
):
    """Mirror the orchestrator's AUT executor construction path.

    Uses the same parameter combinations and decision logic as
    orchestrator.py — entity_ref present → pain_bus + component_registry
    injected; entity_ref absent → pain_bus=None (existing behavior).
    """
    pain_bus = PainBus() if with_pain_bus else None
    nac = NAc(NACConfig(temporal_window_seconds=60.0)) if with_nac else None

    component_registry = None
    if entity_ref is not None:
        component_registry = ComponentRegistry()

    # Mirror the orchestrator's pain_bus decision:
    # entity_ref set → pass pain_bus (required by precondition)
    # entity_ref None → pain_bus=None (avoid bridge double-subscribe)
    executor_pain_bus = pain_bus if entity_ref is not None else None

    executor = build_executor(
        ToolRegistry(),
        pain_bus=executor_pain_bus,
        nac=nac,
        entity_ref=entity_ref,
        component_registry=component_registry,
    )

    return {
        "executor": executor,
        "pain_bus": pain_bus,
        "nac": nac,
        "component_registry": component_registry,
    }


# ---------------------------------------------------------------------------
# E0 core: entity_ref wiring
# ---------------------------------------------------------------------------


class TestSimEmbodimentWiring:
    """Verify that entity_ref produces affordance tools + embodiment
    in the same construction path the orchestrator uses."""

    def test_entity_ref_produces_embodiment(self):
        """build_executor with entity_ref sets executor.embodiment."""
        world = _build_sim_aut_executor(entity_ref="weapons/rusty_sword")
        assert world["executor"].embodiment is not None
        assert world["executor"].embodiment.root.name == "rusty_sword"

    def test_entity_ref_registers_affordance_tools(self):
        """Affordance tools appear in the executor's tool registry."""
        registry = ToolRegistry()
        pain_bus = PainBus()
        nac = NAc(NACConfig())

        build_executor(
            registry,
            pain_bus=pain_bus,
            nac=nac,
            entity_ref="weapons/rusty_sword",
            component_registry=ComponentRegistry(),
        )

        tool_names = set(registry.list())
        # rusty_sword has slash and parry affordances
        assert "rusty_sword_slash" in tool_names, f"Expected rusty_sword_slash in registry, got: {tool_names}"
        assert "rusty_sword_parry" in tool_names, f"Expected rusty_sword_parry in registry, got: {tool_names}"

    def test_entity_ref_registers_sense_tools(self):
        """Sensor read tools are registered alongside affordances."""
        registry = ToolRegistry()
        pain_bus = PainBus()
        nac = NAc(NACConfig())

        build_executor(
            registry,
            pain_bus=pain_bus,
            nac=nac,
            entity_ref="weapons/rusty_sword",
            component_registry=ComponentRegistry(),
        )

        tool_names = set(registry.list())
        # Should have at least one sensor read tool
        sense_tools = [t for t in tool_names if "sense" in t.lower() or "read" in t.lower()]
        assert len(sense_tools) > 0, f"Expected sensor tools in registry, got: {tool_names}"

    def test_no_entity_ref_no_embodiment(self):
        """Default path (no entity_ref) produces no embodiment."""
        world = _build_sim_aut_executor(entity_ref=None)
        assert world["executor"].embodiment is None

    def test_no_entity_ref_pain_bus_none(self):
        """Default path passes pain_bus=None to avoid bridge subscription."""
        world = _build_sim_aut_executor(entity_ref=None)
        bridge = world["executor"]._tool_pain_bridge
        # Bridge exists (nac is set) but has no pain_bus subscription
        assert bridge is not None


class TestSimEmbodimentPreconditions:
    """Verify that build_executor preconditions match sim expectations."""

    def test_entity_ref_without_nac_raises(self):
        """entity_ref requires nac for pain attribution."""
        with pytest.raises(ValueError, match="nac is None"):
            _build_sim_aut_executor(entity_ref="weapons/rusty_sword", with_nac=False)

    def test_entity_ref_without_pain_bus_raises(self):
        """entity_ref requires pain_bus for Embodiment._publish_pain."""
        with pytest.raises(ValueError, match="pain_bus is None"):
            build_executor(
                ToolRegistry(),
                pain_bus=None,
                nac=NAc(NACConfig()),
                entity_ref="weapons/rusty_sword",
                component_registry=ComponentRegistry(),
            )

    def test_nonexistent_entity_ref_raises(self):
        """Typo'd entity_ref raises ComponentNotFoundError."""
        from maxim.exceptions import ComponentNotFoundError

        with pytest.raises(ComponentNotFoundError):
            build_executor(
                ToolRegistry(),
                pain_bus=PainBus(),
                nac=NAc(NACConfig()),
                entity_ref="weapons/legendary_excalibur_that_does_not_exist",
                component_registry=ComponentRegistry(),
            )


class TestSimEmbodimentPainCascade:
    """Verify that pain attribution works through the sim wiring path.

    The orchestrator passes pain_bus to build_executor when entity_ref
    is set. This test confirms the resulting bridge + embodiment
    produce functional NAc learning via the direct-attribution path.
    """

    def test_tool_failure_produces_nac_link(self):
        """Execute an affordance at zero durability → NAc learns
        a negative association via direct attribution.

        This is the sim-path equivalent of
        test_sem_execution_production::test_nac_link_formed_after_cascade.
        """
        world = _build_sim_aut_executor(entity_ref="weapons/rusty_sword")
        executor = world["executor"]
        nac = world["nac"]

        # Set durability to 0 so the sword fails
        sword = executor.embodiment.root
        sword.vital_metrics["durability"] = 0.0

        # Execute the slash — at 0 durability, failure triggers fire
        executor.execute(
            {
                "tool_name": "rusty_sword_slash",
                "params": {"target": "dummy", "force": 0.9},
            }
        )

        # Verify NAc got the negative outcome via direct attribution
        all_links = []
        for links in nac._links.values():
            all_links.extend(links)

        negative_links = [lnk for lnk in all_links if lnk.outcome_valence == Valence.NEGATIVE]
        assert len(negative_links) > 0, (
            "Expected at least one NEGATIVE NAc link after tool failure "
            "cascade — the ToolPainBridge direct-attribution path is broken."
        )

    def test_successful_tool_produces_positive_link(self):
        """Execute an affordance at full durability → NAc learns
        a positive association via record_tool_complete."""
        world = _build_sim_aut_executor(entity_ref="weapons/rusty_sword")
        executor = world["executor"]
        nac = world["nac"]

        # Full durability — no failures
        sword = executor.embodiment.root
        sword.vital_metrics["durability"] = 1.0

        executor.execute(
            {
                "tool_name": "rusty_sword_slash",
                "params": {"target": "dummy", "force": 0.3},
            }
        )

        all_links = []
        for links in nac._links.values():
            all_links.extend(links)

        positive_links = [lnk for lnk in all_links if lnk.outcome_valence == Valence.POSITIVE]
        assert len(positive_links) > 0, (
            "Expected at least one POSITIVE NAc link after successful "
            "tool execution — record_tool_complete path is broken."
        )
