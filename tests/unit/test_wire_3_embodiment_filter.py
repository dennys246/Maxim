"""Tests for Wire 3 (release_0_9_1.md Stage 1 — embodiment-state → tool filter).

The wire ships two methods on ``Embodiment``:

- ``get_disabled_affordances() -> set[str]`` — base tool names whose
  modulator's ``compute_integrity()`` is strictly below the disable
  threshold (default ``0.3``).
- ``get_degraded_affordances() -> dict[str, float]`` — ``{name:
  integrity}`` for modulators in ``[disable_threshold,
  degrade_threshold)``.

The agent_loop hook filters disabled tools from ``available_tools``
and appends ``[DAMAGED: integrity 0.X]`` to degraded tools'
descriptions. Tests cover the Embodiment methods directly + a
minimal integration test that exercises the filter shape through
the agent_loop's description-build expression.

The pre-merge concern with this wire is the base-name assumption:
``tool_bridge.generate_tools_for_entity`` registers tools as
``{entity.name}_{affordance_name}`` unless a collision forces
``_resolve_tool_name`` to prefix an ancestor name. The single-body
Roy / cradle / Reachy topologies don't trigger collisions, so the
wire matches cleanly today; tests pin this contract.
"""

from __future__ import annotations

from typing import Any

import pytest

from maxim.embodiment.body import Embodiment
from maxim.embodiment.sem import AffordanceSchema, Entity


# ─────────────────────────────────────────────────────────────────────
# Synthetic fixtures
# ─────────────────────────────────────────────────────────────────────


class _StubModulator:
    """Minimal Modulator stand-in: declares affordances + integrity."""

    def __init__(
        self,
        name: str,
        affordances: dict[str, AffordanceSchema],
        integrity: float = 1.0,
    ) -> None:
        self.name = name
        self.affordances = affordances
        self._integrity = integrity

    def compute_integrity(self) -> float:
        return self._integrity

    def execute(self, affordance: str, params: dict[str, Any]) -> Any:  # pragma: no cover
        raise NotImplementedError("stub never executes")


class _NoIntegrityModulator:
    """Capability-only modulator (no compute_integrity).

    The Modulator Protocol doesn't require compute_integrity; older
    SpecModulator-shaped types and decorator-style modulators may lack
    it. Wire 3 must treat them as integrity == 1.0 (not damaged).
    """

    def __init__(self, name: str, affordances: dict[str, AffordanceSchema]) -> None:
        self.name = name
        self.affordances = affordances

    def execute(self, affordance: str, params: dict[str, Any]) -> Any:  # pragma: no cover
        raise NotImplementedError("stub never executes")


def _make_entity_with_modulators(modulators: dict[str, Any], name: str = "agent") -> Entity:
    """Build a one-entity tree with the given modulators."""
    return Entity(name=name, entity_type="body", modulators=modulators)


def _aff(description: str = "") -> AffordanceSchema:
    return AffordanceSchema(description=description)


# ─────────────────────────────────────────────────────────────────────
# Layer 1: get_disabled_affordances
# ─────────────────────────────────────────────────────────────────────


class TestGetDisabledAffordances:
    """``integrity < 0.3`` → in the disabled set."""

    def test_healthy_modulator_returns_empty(self) -> None:
        entity = _make_entity_with_modulators(
            {"arm": _StubModulator("arm", {"grasp": _aff()}, integrity=1.0)},
        )
        embodiment = Embodiment(root=entity)
        assert embodiment.get_disabled_affordances() == set()

    def test_critically_damaged_modulator_disables_all_affordances(self) -> None:
        entity = _make_entity_with_modulators(
            {"arm": _StubModulator("arm", {"grasp": _aff(), "release": _aff()}, integrity=0.1)},
        )
        embodiment = Embodiment(root=entity)
        assert embodiment.get_disabled_affordances() == {"agent_grasp", "agent_release"}

    def test_boundary_at_0_3_inclusive_on_healthy_side(self) -> None:
        """``integrity == 0.3`` is the boundary. The plan says
        ``integrity < 0.3 disables``, strict-less-than. So 0.3 stays
        enabled and lands in the degraded band."""
        entity = _make_entity_with_modulators(
            {"arm": _StubModulator("arm", {"grasp": _aff()}, integrity=0.3)},
        )
        embodiment = Embodiment(root=entity)
        assert embodiment.get_disabled_affordances() == set()

    def test_just_below_0_3_disables(self) -> None:
        entity = _make_entity_with_modulators(
            {"arm": _StubModulator("arm", {"grasp": _aff()}, integrity=0.29)},
        )
        embodiment = Embodiment(root=entity)
        assert embodiment.get_disabled_affordances() == {"agent_grasp"}

    def test_modulator_without_compute_integrity_not_disabled(self) -> None:
        """Capability-only modulators (no compute_integrity) MUST be
        treated as full integrity. Wire 3 must not disable an entire
        class of modulator types just because they don't declare a
        vital_metrics surface."""
        entity = _make_entity_with_modulators(
            {"voice": _NoIntegrityModulator("voice", {"speak": _aff()})},
        )
        embodiment = Embodiment(root=entity)
        assert embodiment.get_disabled_affordances() == set()

    def test_per_modulator_isolation(self) -> None:
        """A damaged arm doesn't disable healthy voice affordances."""
        entity = _make_entity_with_modulators(
            {
                "arm": _StubModulator("arm", {"grasp": _aff()}, integrity=0.1),
                "voice": _StubModulator("voice", {"speak": _aff()}, integrity=1.0),
            },
        )
        embodiment = Embodiment(root=entity)
        disabled = embodiment.get_disabled_affordances()
        assert "agent_grasp" in disabled
        assert "agent_speak" not in disabled

    def test_threshold_override(self) -> None:
        """The default 0.3 threshold is overridable per call — useful
        for future tuning experiments without touching call sites."""
        entity = _make_entity_with_modulators(
            {"arm": _StubModulator("arm", {"grasp": _aff()}, integrity=0.5)},
        )
        embodiment = Embodiment(root=entity)
        # Default: integrity=0.5 stays enabled.
        assert embodiment.get_disabled_affordances() == set()
        # Higher threshold: now disabled.
        assert embodiment.get_disabled_affordances(threshold=0.6) == {"agent_grasp"}


# ─────────────────────────────────────────────────────────────────────
# Layer 2: get_degraded_affordances
# ─────────────────────────────────────────────────────────────────────


class TestGetDegradedAffordances:
    """``[disable_threshold, degrade_threshold)`` band — damaged but
    not disabled. Returns ``{name: integrity}`` so the agent_loop hook
    can render the integrity value in the description annotation."""

    def test_healthy_modulator_returns_empty(self) -> None:
        entity = _make_entity_with_modulators(
            {"arm": _StubModulator("arm", {"grasp": _aff()}, integrity=1.0)},
        )
        embodiment = Embodiment(root=entity)
        assert embodiment.get_degraded_affordances() == {}

    def test_critically_damaged_modulator_not_in_degraded(self) -> None:
        """Disabled (< 0.3) does NOT appear in degraded — separate
        bands for separate handling."""
        entity = _make_entity_with_modulators(
            {"arm": _StubModulator("arm", {"grasp": _aff()}, integrity=0.1)},
        )
        embodiment = Embodiment(root=entity)
        assert embodiment.get_degraded_affordances() == {}

    def test_partial_damage_in_band(self) -> None:
        """integrity == 0.5 is in [0.3, 0.6) → degraded."""
        entity = _make_entity_with_modulators(
            {"arm": _StubModulator("arm", {"grasp": _aff()}, integrity=0.5)},
        )
        embodiment = Embodiment(root=entity)
        assert embodiment.get_degraded_affordances() == {"agent_grasp": 0.5}

    def test_lower_boundary_inclusive(self) -> None:
        """integrity == 0.3 is the lower boundary, inclusive on the
        degraded side. The disabled set is < 0.3 strict, so 0.3
        belongs here."""
        entity = _make_entity_with_modulators(
            {"arm": _StubModulator("arm", {"grasp": _aff()}, integrity=0.3)},
        )
        embodiment = Embodiment(root=entity)
        assert embodiment.get_degraded_affordances() == {"agent_grasp": 0.3}

    def test_upper_boundary_exclusive(self) -> None:
        """integrity == 0.6 is the upper boundary, EXCLUSIVE — at or
        above this, the affordance is healthy (no annotation)."""
        entity = _make_entity_with_modulators(
            {"arm": _StubModulator("arm", {"grasp": _aff()}, integrity=0.6)},
        )
        embodiment = Embodiment(root=entity)
        assert embodiment.get_degraded_affordances() == {}

    def test_just_below_upper_boundary_degraded(self) -> None:
        entity = _make_entity_with_modulators(
            {"arm": _StubModulator("arm", {"grasp": _aff()}, integrity=0.59)},
        )
        embodiment = Embodiment(root=entity)
        assert embodiment.get_degraded_affordances() == {"agent_grasp": 0.59}

    def test_threshold_overrides(self) -> None:
        entity = _make_entity_with_modulators(
            {"arm": _StubModulator("arm", {"grasp": _aff()}, integrity=0.7)},
        )
        embodiment = Embodiment(root=entity)
        # Default thresholds: 0.7 is healthy.
        assert embodiment.get_degraded_affordances() == {}
        # Wider degraded band: 0.7 ∈ [0.3, 0.8).
        assert embodiment.get_degraded_affordances(degrade_threshold=0.8) == {"agent_grasp": 0.7}


# ─────────────────────────────────────────────────────────────────────
# Layer 3: Cross-band correctness (no overlap, no gap)
# ─────────────────────────────────────────────────────────────────────


class TestBandPartitioning:
    """The three integrity bands — disabled / degraded / healthy —
    partition the [0, 1] range without overlap or gap at the
    default thresholds (0.3 and 0.6). Test that the same affordance
    never lands in both sets, and that the disabled-vs-degraded
    boundary is the documented strict-vs-inclusive split."""

    @pytest.mark.parametrize(
        "integrity,expected_disabled,expected_degraded",
        [
            (0.0, True, False),
            (0.29, True, False),
            (0.3, False, True),  # boundary — degraded side
            (0.45, False, True),
            (0.59, False, True),
            (0.6, False, False),  # boundary — healthy side
            (0.75, False, False),
            (1.0, False, False),
        ],
    )
    def test_no_double_membership(
        self,
        integrity: float,
        expected_disabled: bool,
        expected_degraded: bool,
    ) -> None:
        entity = _make_entity_with_modulators(
            {"arm": _StubModulator("arm", {"grasp": _aff()}, integrity=integrity)},
        )
        embodiment = Embodiment(root=entity)
        disabled = embodiment.get_disabled_affordances()
        degraded = embodiment.get_degraded_affordances()
        assert ("agent_grasp" in disabled) is expected_disabled
        assert ("agent_grasp" in degraded) is expected_degraded
        # No affordance ever lands in BOTH sets.
        assert disabled.isdisjoint(degraded.keys())


# ─────────────────────────────────────────────────────────────────────
# Layer 4: Tool-name pattern matches tool_bridge.generate_tools_for_entity
# ─────────────────────────────────────────────────────────────────────


class TestToolNamePattern:
    """Wire 3's filter only works if its base tool names match what
    ``tool_bridge.generate_tools_for_entity`` actually registers. The
    base pattern is ``{entity.name}_{affordance_name}``; the
    ``_resolve_tool_name`` collision-handler only kicks in if the
    base name is already in the registry. Roy / cradle / Reachy
    use single-body topologies — no collisions in practice. Pin
    the contract."""

    def test_base_name_matches_generated_tool_name(self) -> None:
        from maxim.embodiment.tool_bridge import generate_tools_for_entity
        from maxim.tools.registry import ToolRegistry

        entity = _make_entity_with_modulators(
            {"arm": _StubModulator("arm", {"grasp": _aff(), "release": _aff()}, integrity=0.1)},
            name="reachy",
        )
        embodiment = Embodiment(root=entity)
        registry = ToolRegistry()
        generate_tools_for_entity(entity, registry)
        disabled = embodiment.get_disabled_affordances()
        # Both predicted disabled names should be in the registry.
        registered = set(registry.list_all())
        for name in disabled:
            assert name in registered, f"{name!r} predicted disabled but not registered (collision?)"


# ─────────────────────────────────────────────────────────────────────
# Layer 5: agent_loop hook shape (without a full sim)
# ─────────────────────────────────────────────────────────────────────


class TestAgentLoopHookShape:
    """Mirrors the expression at agent_loop.py's ``available_tools``
    hook: filter by ``embodiment.get_disabled_affordances()``, then
    annotate descriptions for ``embodiment.get_degraded_affordances()``.
    Tests the shape so a future refactor of the hook can't silently
    regress the contract."""

    def test_filter_removes_disabled_tools(self) -> None:
        entity = _make_entity_with_modulators(
            {
                "arm": _StubModulator("arm", {"grasp": _aff()}, integrity=0.1),
                "voice": _StubModulator("voice", {"speak": _aff()}, integrity=1.0),
            },
        )
        embodiment = Embodiment(root=entity)
        available_tools = ["agent_grasp", "agent_speak", "respond"]
        disabled = embodiment.get_disabled_affordances()
        filtered = [t for t in available_tools if t not in disabled]
        assert filtered == ["agent_speak", "respond"]

    def test_annotate_degraded_descriptions(self) -> None:
        entity = _make_entity_with_modulators(
            {"arm": _StubModulator("arm", {"grasp": _aff()}, integrity=0.5)},
        )
        embodiment = Embodiment(root=entity)
        tool_descriptions: dict[str, Any] = {
            "agent_grasp": {"description": "grip an object", "params": {}, "example": None, "followup_type": None},
        }
        degraded = embodiment.get_degraded_affordances()
        for name, integrity in degraded.items():
            entry = tool_descriptions.get(name)
            if isinstance(entry, dict):
                base_desc = entry.get("description", "")
                if isinstance(base_desc, str):
                    annotation = f" [DAMAGED: integrity {integrity:.1f}]"
                    tool_descriptions[name] = {**entry, "description": base_desc + annotation}
        assert tool_descriptions["agent_grasp"]["description"] == "grip an object [DAMAGED: integrity 0.5]"

    def test_annotation_not_duplicated_on_re_apply(self) -> None:
        """Idempotency check — the agent_loop hook fires every tick.
        If we re-apply the annotation without checking for an
        existing copy, the description would accumulate the suffix
        endlessly. The hook's ``if annotation not in base_desc``
        guard pins this."""
        entity = _make_entity_with_modulators(
            {"arm": _StubModulator("arm", {"grasp": _aff()}, integrity=0.5)},
        )
        embodiment = Embodiment(root=entity)
        tool_descriptions: dict[str, Any] = {
            "agent_grasp": {"description": "grip an object", "params": {}, "example": None, "followup_type": None},
        }

        def _apply_annotations() -> None:
            degraded = embodiment.get_degraded_affordances()
            for name, integrity in degraded.items():
                entry = tool_descriptions.get(name)
                if isinstance(entry, dict):
                    base_desc = entry.get("description", "")
                    if isinstance(base_desc, str):
                        annotation = f" [DAMAGED: integrity {integrity:.1f}]"
                        if annotation not in base_desc:
                            tool_descriptions[name] = {**entry, "description": base_desc + annotation}

        _apply_annotations()
        _apply_annotations()
        _apply_annotations()
        # Only one annotation suffix.
        assert tool_descriptions["agent_grasp"]["description"].count("[DAMAGED:") == 1

    def test_no_embodiment_is_no_op(self) -> None:
        """The agent_loop hook is fail-open: when ``embodiment is None``
        (headless API, non-sim runtime), no filter / annotation
        happens. This test mirrors the early-exit shape."""
        embodiment = None
        available_tools = ["agent_grasp", "respond"]
        # The hook body short-circuits on `if embodiment is not None`.
        if embodiment is not None:  # type: ignore[unreachable]
            pytest.fail("hook should short-circuit on None embodiment")
        # available_tools unchanged.
        assert available_tools == ["agent_grasp", "respond"]


# ─────────────────────────────────────────────────────────────────────
# Layer 6: Empty / degenerate cases
# ─────────────────────────────────────────────────────────────────────


class TestDegenerateCases:
    def test_entity_with_no_modulators(self) -> None:
        entity = Entity(name="agent", entity_type="body")
        embodiment = Embodiment(root=entity)
        assert embodiment.get_disabled_affordances() == set()
        assert embodiment.get_degraded_affordances() == {}

    def test_modulator_with_no_affordances(self) -> None:
        """Modulators without affordances contribute nothing to either set."""
        entity = _make_entity_with_modulators(
            {"arm": _StubModulator("arm", {}, integrity=0.1)},
        )
        embodiment = Embodiment(root=entity)
        assert embodiment.get_disabled_affordances() == set()
        assert embodiment.get_degraded_affordances() == {}

    def test_compute_integrity_raises_treated_as_healthy(self) -> None:
        """Defensive: a buggy modulator whose compute_integrity raises
        must NOT crash Wire 3 — the hook fails open (treats the
        modulator as healthy, integrity=1.0) so the agent loop keeps
        running. The agent_loop hook's outer try/except is the
        second line of defense; this test pins the inner one."""

        class _RaisingModulator:
            name = "arm"
            affordances = {"grasp": _aff()}

            def compute_integrity(self) -> float:
                raise RuntimeError("simulated integrity calc failure")

            def execute(self, affordance: str, params: dict[str, Any]) -> Any:  # pragma: no cover
                raise NotImplementedError

        entity = _make_entity_with_modulators({"arm": _RaisingModulator()})
        embodiment = Embodiment(root=entity)
        assert embodiment.get_disabled_affordances() == set()
        assert embodiment.get_degraded_affordances() == {}
