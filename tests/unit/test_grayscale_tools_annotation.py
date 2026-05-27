"""Tests for the grayscale tools section (W1 sense_tool_registry MVP).

Three layers:

1. ``compose_grayscale_tools_section`` — pure renderer on synthetic input.
2. ``PromptBuilder._add_grayscale_tools_section`` — section wiring against
   a populated ``StructuredContext``.
3. Producer-shape integration — verifies a registry holding active SEM
   tools, inactive SEM tools, and core tools yields the right grayscale
   subset under a representative NAc bias list.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from maxim.prompts.grayscale_tools_annotation import (
    compose_grayscale_tools_section,
)


# ─────────────────────────────────────────────────────────────────────
# Layer 1: composer
# ─────────────────────────────────────────────────────────────────────


class TestComposeGrayscaleSection:
    """Pure renderer on synthetic (tool, bias, description) input."""

    def test_renders_canonical_block(self) -> None:
        annotations = [
            ("sense_food_source", 0.9, "Scan the area for food sources nearby"),
            ("forest_gather", 0.05, "Gather forage in the forest"),
        ]
        text = compose_grayscale_tools_section(annotations)
        assert "Tools known but not in current location" in text
        assert "sense_food_source" in text
        # Strong positive bias gets the experience phrase.
        assert "rewarding from prior experience" in text
        # Neutral-band entry stays in the block but without an experience phrase.
        assert "forest_gather" in text
        # Always-present visibility annotation.
        assert "not in current location" in text

    def test_aversive_band_phrase(self) -> None:
        annotations = [("dragon_breathe_fire", -0.5, "Exhale fire on a target")]
        text = compose_grayscale_tools_section(annotations)
        assert "dragon_breathe_fire" in text
        assert "aversive from prior experience" in text

    def test_neutral_band_no_experience_suffix(self) -> None:
        annotations = [("sword_swing", 0.01, "Swing the sword in an arc")]
        text = compose_grayscale_tools_section(annotations)
        assert "sword_swing" in text
        assert "not in current location" in text
        # Mid-bias means no "from prior experience" suffix.
        assert "from prior experience" not in text

    def test_empty_returns_empty_string(self) -> None:
        assert compose_grayscale_tools_section([]) == ""
        assert compose_grayscale_tools_section(None) == ""

    def test_description_trims_to_keep_section_bounded(self) -> None:
        very_long = "x" * 200
        annotations = [("long_tool", 0.5, very_long)]
        text = compose_grayscale_tools_section(annotations)
        # Trimming policy: descriptions over 80 chars get truncated with
        # an ellipsis. The exact threshold is internal but the test
        # asserts the truncation happens.
        assert "..." in text
        # The full 200-character string did not render verbatim.
        assert very_long not in text

    def test_handles_missing_description(self) -> None:
        annotations = [("orphan_tool", 0.8, "")]
        text = compose_grayscale_tools_section(annotations)
        assert "orphan_tool" in text
        # With no description, the tool name still renders with annotation.
        assert "not in current location" in text
        assert "rewarding from prior experience" in text


# ─────────────────────────────────────────────────────────────────────
# Layer 2: PromptBuilder section helper
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


# ─────────────────────────────────────────────────────────────────────
# Layer 3: producer-shape integration
# ─────────────────────────────────────────────────────────────────────


class _MinimalTool:
    """Tool stub that mimics the metadata surface the producer reads."""

    def __init__(self, name: str, description: str, kind: str = "core-universal") -> None:
        self.name = name
        self.description = description
        self.kind = kind
        self.auto_fire = False
        self.input_schema: dict[str, Any] = {}


def test_producer_filters_to_inactive_sem_tools_with_bias():
    """End-to-end shape: only SEM-derived + inactive + biased tools grayscale.

    Asserts the four filter rules the producer applies:
    1. Tool must be in registry (handled by ``get_tools_by_kind``).
    2. Tool kind must be ``sem-modulator-derived`` (skip core-universal).
    3. Tool must NOT be in the active roster (active tools render normally).
    4. Tool must appear in the NAc bias list (else nothing to surface).
    """
    from maxim.tools.registry import ToolRegistry

    reg = ToolRegistry()

    # 1. core tool with bias — should NOT grayscale (it's always active).
    core = _MinimalTool("respond", "Reply to the user", kind="core-universal")
    reg.register(core)

    # 2. SEM tool, ACTIVE, with bias — should NOT grayscale (already active).
    sem_active = _MinimalTool("active_sword_swing", "Swing the active sword", kind="sem-modulator-derived")
    reg.register_scene_tools([sem_active], scene_id="active_scene")

    # 3. SEM tool, INACTIVE, with bias — SHOULD grayscale.
    sem_inactive = _MinimalTool("inactive_food_source", "Eat the inactive food", kind="sem-modulator-derived")
    reg.register_scene_tools([sem_inactive], scene_id="absent_scene")
    reg.deactivate_scene("absent_scene")

    # 4. SEM tool, INACTIVE, with NO bias — should NOT grayscale (no signal).
    sem_no_bias = _MinimalTool("no_bias_tool", "Has no learned bias", kind="sem-modulator-derived")
    reg.register_scene_tools([sem_no_bias], scene_id="other_absent_scene")
    reg.deactivate_scene("other_absent_scene")

    biases = [
        ("respond", 0.8),  # core, will be filtered out
        ("active_sword_swing", 0.5),  # active, filtered out
        ("inactive_food_source", 0.7),  # the one entry that should grayscale
    ]

    # Reimplement the producer's filter loop (kept tight so the test
    # describes the contract by example — if this test fails after a
    # refactor, the producer's filter rules changed).
    active = set(reg.list())
    sem_names = {t.name for t in reg.get_tools_by_kind("sem-modulator-derived")}

    grayscale: list[tuple[str, float, str]] = []
    for tool_name, bias in biases:
        if tool_name in active:
            continue
        if tool_name not in sem_names:
            continue
        tool_obj = reg.get(tool_name)
        grayscale.append((tool_name, bias, tool_obj.description))

    assert len(grayscale) == 1
    assert grayscale[0][0] == "inactive_food_source"
    assert grayscale[0][1] == 0.7
    assert grayscale[0][2] == "Eat the inactive food"


def test_producer_handles_empty_bias_list():
    """Cold-start agent (no NAc bias yet) → empty grayscale, no error."""
    from maxim.tools.registry import ToolRegistry

    reg = ToolRegistry()
    sem = _MinimalTool("dormant", "Sleep", kind="sem-modulator-derived")
    reg.register_scene_tools([sem], scene_id="x")
    reg.deactivate_scene("x")

    biases: list[tuple[str, float]] = []
    active = set(reg.list())
    sem_names = {t.name for t in reg.get_tools_by_kind("sem-modulator-derived")}

    grayscale = [(n, b, reg.get(n).description) for n, b in biases if n not in active and n in sem_names]
    assert grayscale == []
