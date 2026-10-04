"""#873 (the no-silent-fallback half): damage aimed at a part the body lacks fails; it never "succeeds".

``DamageComponentTool`` used to subtract the damage from the root ``vital_metrics["health"]`` when the named
part was missing, publish pain, and report ``success=True``. On a derived-health body the same call's
``evaluate_failures()`` re-derived health from the parts, so the damage vanished; on a flat-``hp`` body it
landed on an orphan ``health`` key. Either way the call reported a response that did not happen (#870's
shape). The four design points (sum vs weighted mean, partless bodies, archetype reflex sets) stay with
``docs/plans/deferred/reflex_layering.md``.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from maxim.embodiment.component_registry import ComponentRegistry
from maxim.simulation.tools import DamageComponentTool
from maxim.tools.base import ToolErrorKind


_RED_873 = pytest.mark.xfail(strict=True, reason="#873: damage_component falls back to root health and reports success")


class _Bus:
    def __init__(self) -> None:
        self.published: list[Any] = []

    def publish(self, signal: Any) -> None:
        self.published.append(signal)


class _Embodiment:
    """The two things the tool reads: the body root and the pain bus."""

    def __init__(self, root: Any) -> None:
        self.root = root
        self._pain_bus = _Bus()
        self.agent_id = "aut"

    def evaluate_failures(self) -> list[Any]:
        return self.root.evaluate_failures() if hasattr(self.root, "evaluate_failures") else []


def _tool(body_ref: str) -> tuple[DamageComponentTool, _Embodiment]:
    emb = _Embodiment(ComponentRegistry().instantiate(body_ref))
    return DamageComponentTool(embodiment=emb, entity_map=None), emb


@_RED_873
@pytest.mark.parametrize("body_ref", ["bodies/base_humanoid", "creatures/wolf"])
def test_a_missing_part_fails_and_changes_nothing(body_ref):
    tool, emb = _tool(body_ref)
    before = copy.deepcopy(emb.root.vital_metrics)
    out = tool.execute(component="torso_missing", amount=0.3, source="test")
    assert out.success is False
    assert out.error_kind is ToolErrorKind.INVALID_INPUT
    assert "torso_missing" in (out.error or "")
    assert emb.root.vital_metrics == before  # no orphan `health`, no decrement
    assert emb._pain_bus.published == []  # no pain for damage that did not land


@_RED_873
def test_the_error_names_the_parts_that_can_take_damage():
    """A retryable error for the LLM: it says what it COULD have aimed at."""
    tool, emb = _tool("bodies/base_humanoid")
    out = tool.execute(component="wing", amount=0.3)
    assert out.success is False
    damageable = [n for n, m in emb.root.modulators.items() if hasattr(m, "apply_damage") and m.vital_metrics]
    assert damageable
    for name in damageable:
        assert name in (out.error or "")


def test_a_part_that_exists_still_takes_the_damage():
    tool, emb = _tool("bodies/base_humanoid")
    torso = emb.root.get_component("torso")
    assert torso is not None and torso.vital_metrics
    out = tool.execute(component="torso", amount=0.3, source="test")
    assert out.success is True
    assert out.output["integrity"] < 1.0
    assert len(emb._pain_bus.published) == 1
