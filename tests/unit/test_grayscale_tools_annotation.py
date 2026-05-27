"""Tests for the grayscale tools section (W1 sense_tool_registry MVP).

Three layers:

1. ``compose_grayscale_tools_section`` — pure renderer on synthetic input.
2. ``build_grayscale_annotations`` — producer filter on NAc-shaped
   ``tool:<name>`` input + ToolRegistry-like interface. Pinned because
   the pre-merge bio-fidelity review caught a silent prefix mismatch
   that an in-line producer would have hidden in production.
3. ``PromptBuilder._add_grayscale_tools_section`` — section wiring
   against a populated ``StructuredContext``.

The Layer-2 tests are the load-bearing regression guard for the C1 bug
(NAc stores ``tool:sense_food_source``, registry stores
``sense_food_source``; without prefix-strip the filter would always
exclude every entry).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from maxim.prompts.grayscale_tools_annotation import (
    build_grayscale_annotations,
    compose_grayscale_tools_section,
)


# ─────────────────────────────────────────────────────────────────────
# Layer 1: composer
# ─────────────────────────────────────────────────────────────────────


class TestComposeGrayscaleSection:
    """Pure renderer on synthetic (tool, bias, description) input.

    Substrate-voice consistency: the intensity phrase comes from
    Wire-A's ``bias_to_band``, so the two prompt surfaces use the same
    band language ("strongly rewarding", "mildly aversive", etc.).
    """

    def test_renders_canonical_block(self) -> None:
        annotations = [
            ("sense_food_source", 0.9, "Scan the area for food sources nearby"),
            ("forest_gather", 0.05, "Gather forage in the forest"),
        ]
        text = compose_grayscale_tools_section(annotations)
        assert "Tools known but not in current location" in text
        assert "sense_food_source" in text
        # Strong positive band (>= 0.5) maps to "strongly rewarding".
        assert "strongly rewarding from prior experience" in text
        # Neutral-band entry stays in the block, marked "neutral / mixed".
        assert "forest_gather" in text
        assert "neutral / mixed" in text
        # Always-present visibility annotation.
        assert "not in current location" in text

    def test_strong_aversive_band_phrase(self) -> None:
        annotations = [("dragon_breathe_fire", -0.6, "Exhale fire on a target")]
        text = compose_grayscale_tools_section(annotations)
        assert "dragon_breathe_fire" in text
        assert "strongly aversive from prior experience" in text

    def test_mild_rewarding_band_phrase(self) -> None:
        # 0.1 <= bias < 0.5 → "mildly rewarding" per Wire-A's bands.
        annotations = [("sword_swing", 0.2, "Swing the sword in an arc")]
        text = compose_grayscale_tools_section(annotations)
        assert "sword_swing" in text
        assert "mildly rewarding from prior experience" in text

    def test_neutral_band_renders_without_experience_suffix(self) -> None:
        # The "neutral / mixed" band has no "from prior experience"
        # suffix — matches Wire-A's renderer (neutral signals are
        # informational rather than directive).
        annotations = [("dormant_tool", 0.05, "Some neutral-band tool")]
        text = compose_grayscale_tools_section(annotations)
        assert "dormant_tool" in text
        assert "neutral / mixed" in text
        assert "from prior experience" not in text

    def test_empty_returns_empty_string(self) -> None:
        assert compose_grayscale_tools_section([]) == ""
        assert compose_grayscale_tools_section(None) == ""

    def test_description_trims_to_keep_section_bounded(self) -> None:
        very_long = "x" * 200
        annotations = [("long_tool", 0.6, very_long)]
        text = compose_grayscale_tools_section(annotations)
        assert "..." in text
        assert very_long not in text

    def test_handles_missing_description(self) -> None:
        annotations = [("orphan_tool", 0.8, "")]
        text = compose_grayscale_tools_section(annotations)
        assert "orphan_tool" in text
        assert "not in current location" in text
        assert "strongly rewarding from prior experience" in text


# ─────────────────────────────────────────────────────────────────────
# Layer 2: producer filter (CRITICAL — pre-merge review caught the
# prefix mismatch bug here)
# ─────────────────────────────────────────────────────────────────────


class _MinimalTool:
    """Stub matching the metadata surface the producer reads."""

    def __init__(self, name: str, description: str, kind: str = "core-universal") -> None:
        self.name = name
        self.description = description
        self.kind = kind
        self.auto_fire = False
        self.input_schema: dict[str, Any] = {}


class TestBuildGrayscaleAnnotations:
    """The producer filter under NAc-shaped ``tool:<name>`` input."""

    def _registry_with(self, core: list[_MinimalTool], inactive_sem: list[_MinimalTool]) -> Any:
        """Build a real ToolRegistry holding core + inactive SEM tools."""
        from maxim.tools.registry import ToolRegistry

        reg = ToolRegistry()
        for t in core:
            reg.register(t)
        if inactive_sem:
            reg.register_scene_tools(inactive_sem, scene_id="inactive_scene")
            reg.deactivate_scene("inactive_scene")
        return reg

    def test_strips_tool_prefix_before_set_lookup(self) -> None:
        """The C1 regression guard: NAc returns ``tool:sense_food_source``;
        the registry stores ``sense_food_source``. Without prefix-strip
        the filter would always exclude every entry and the grayscale
        list would be silently empty in production."""
        sem = _MinimalTool("sense_food_source", "Scan for food", kind="sem-modulator-derived")
        reg = self._registry_with([], [sem])

        # NAc-shaped input: every signature carries the ``tool:`` prefix.
        biases = [("tool:sense_food_source", 0.85)]
        out = build_grayscale_annotations(biases, reg)

        assert len(out) == 1
        # The output uses the BARE name (matches what the LLM sees in
        # the tools block) — prefix is stripped, not preserved.
        assert out[0][0] == "sense_food_source"
        assert out[0][1] == 0.85
        assert out[0][2] == "Scan for food"

    def test_strips_tool_use_action_prefix(self) -> None:
        """``tool:use:dodge`` strips ONLY the outer ``tool:`` prefix —
        keeps the inner ``use:dodge`` action key intact (matches Wire-A
        composer convention)."""
        sem = _MinimalTool("use:dodge", "Dodge incoming damage", kind="sem-modulator-derived")
        reg = self._registry_with([], [sem])

        biases = [("tool:use:dodge", 0.6)]
        out = build_grayscale_annotations(biases, reg)
        assert len(out) == 1
        assert out[0][0] == "use:dodge"

    def test_excludes_active_tools(self) -> None:
        """A SEM tool with bias that IS in the active roster does NOT
        grayscale (it renders normally in the Available Tools block)."""
        sem = _MinimalTool("active_sword_swing", "Swing the active sword", kind="sem-modulator-derived")
        from maxim.tools.registry import ToolRegistry

        reg = ToolRegistry()
        # Register as active scene tool (not deactivated).
        reg.register_scene_tools([sem], scene_id="active_scene")

        biases = [("tool:active_sword_swing", 0.7)]
        out = build_grayscale_annotations(biases, reg)
        assert out == []

    def test_excludes_core_tools(self) -> None:
        """Core / non-SEM tools don't grayscale even when biased and
        absent — they're either always active (registered via plain
        register()) or out-of-scope for the MVP."""
        core = _MinimalTool("respond", "Reply to user", kind="core-universal")
        reg = self._registry_with([core], [])

        # Active list contains "respond" (it's a core tool). Even if it
        # weren't active, the kind filter would still exclude it.
        biases = [("tool:respond", 0.8)]
        out = build_grayscale_annotations(biases, reg)
        assert out == []

    def test_excludes_unregistered_tools(self) -> None:
        """A bias for a tool not in the registry at all yields no entry —
        defensive against stale NAc data for tools that have been
        deregistered between sessions."""
        reg = self._registry_with([], [])  # empty registry

        biases = [("tool:ghost_tool", 0.9)]
        out = build_grayscale_annotations(biases, reg)
        assert out == []

    def test_cap_at_top_n(self) -> None:
        """``top_n`` caps the result to bound prompt budget."""
        sem_tools = [_MinimalTool(f"sem_tool_{i}", f"description {i}", kind="sem-modulator-derived") for i in range(10)]
        reg = self._registry_with([], sem_tools)

        biases = [(f"tool:sem_tool_{i}", 0.5 + 0.01 * i) for i in range(10)]
        out = build_grayscale_annotations(biases, reg, top_n=3)
        assert len(out) == 3

    def test_empty_biases_returns_empty(self) -> None:
        sem = _MinimalTool("dormant", "Sleep", kind="sem-modulator-derived")
        reg = self._registry_with([], [sem])
        assert build_grayscale_annotations([], reg) == []
        assert build_grayscale_annotations(None, reg) == []

    def test_no_registry_returns_empty(self) -> None:
        biases = [("tool:something", 0.5)]
        assert build_grayscale_annotations(biases, None) == []

    def test_registry_without_metadata_helpers_returns_empty(self) -> None:
        """Out-of-tree ToolRegistry subclass without get_tools_by_kind
        is handled gracefully — grayscale is purely additive, never
        load-bearing for correctness."""

        class _LegacyRegistry:
            def list(self) -> list[str]:
                return []

            def get(self, name: str) -> Any:
                raise KeyError(name)

            # NOTE: deliberately missing get_tools_by_kind

        biases = [("tool:something", 0.5)]
        out = build_grayscale_annotations(biases, _LegacyRegistry())  # type: ignore[arg-type]
        assert out == []

    def test_mixed_input_only_inactive_sem_passes(self) -> None:
        """End-to-end shape: only inactive + SEM + biased tools pass."""
        from maxim.tools.registry import ToolRegistry

        core = _MinimalTool("respond", "Reply", kind="core-universal")
        sem_active = _MinimalTool("active_swing", "swing", kind="sem-modulator-derived")
        sem_inactive = _MinimalTool("inactive_food", "Eat food", kind="sem-modulator-derived")
        sem_no_bias = _MinimalTool("no_bias_tool", "No bias", kind="sem-modulator-derived")

        reg = ToolRegistry()
        reg.register(core)
        reg.register_scene_tools([sem_active], scene_id="active")
        reg.register_scene_tools([sem_inactive, sem_no_bias], scene_id="absent")
        reg.deactivate_scene("absent")

        biases = [
            ("tool:respond", 0.8),
            ("tool:active_swing", 0.5),
            ("tool:inactive_food", 0.7),
            # Note: no entry for no_bias_tool (NAc only returns tools
            # that have accumulated bias, so the producer never sees
            # a (tool, 0.0) entry).
        ]
        out = build_grayscale_annotations(biases, reg)
        assert len(out) == 1
        assert out[0][0] == "inactive_food"
        assert out[0][1] == 0.7
        assert out[0][2] == "Eat food"


# ─────────────────────────────────────────────────────────────────────
# Layer 3: PromptBuilder section helper
# ─────────────────────────────────────────────────────────────────────


class TestPromptBuilderGrayscaleHelper:
    """Helper reads StructuredContext.grayscale_tool_annotations and adds
    an IMPORTANT-priority section. None / empty skips silently."""

    def _make_request(self, annotations):
        request = MagicMock()
        request.context = MagicMock()
        request.context.grayscale_tool_annotations = annotations
        return request

    def test_helper_skips_when_none(self) -> None:
        from maxim.agents.prompt_builder import PromptBuilder

        budgeter = MagicMock()
        request = self._make_request(None)
        PromptBuilder._add_grayscale_tools_section(budgeter, request)
        budgeter.add.assert_not_called()

    def test_helper_skips_when_empty_list(self) -> None:
        from maxim.agents.prompt_builder import PromptBuilder

        budgeter = MagicMock()
        request = self._make_request([])
        PromptBuilder._add_grayscale_tools_section(budgeter, request)
        budgeter.add.assert_not_called()

    def test_helper_adds_section_when_annotations_present(self) -> None:
        from maxim.agents.prompt_builder import PromptBuilder, SectionPriority

        budgeter = MagicMock()
        request = self._make_request([("sense_food_source", 0.9, "Scan for food")])
        PromptBuilder._add_grayscale_tools_section(budgeter, request)
        budgeter.add.assert_called_once()
        args, _ = budgeter.add.call_args
        name, text, priority = args[:3]
        assert name == "grayscale_tools"
        assert "sense_food_source" in text
        assert "Tools known but not in current location" in text
        assert priority == SectionPriority.IMPORTANT
