"""Substrate-primary drive-need derivation — regression guards.

Covers the homeostatic-deficit → corrective-need derivation in
``runtime/agent_loop._read_drive_states`` and its downstream effect on
``NAc.recommend_action``'s drive-affinity heuristic, per the drive-tuning
work for Exp 41 (docs/plans/substrate_exploration_policy.md /
docs/experiments/41_substrate_primary_exploration.md).

The gap this closes: the drive-affinity heuristic only fires on positive
[0,1] need intensities, so a homeostatic *deficit* (cold = a temperature
drive below its set_point) was invisible to substrate-primary action
selection and warmth-seeking never became salient.
"""

from __future__ import annotations

from maxim.decisions.nac import NAc, NACConfig
from maxim.embodiment.component_registry import ComponentRegistry
from maxim.runtime.agent_loop import _read_drive_states


class _Emb:
    def __init__(self, root):
        self.root = root


class _Exec:
    def __init__(self, root):
        self.embodiment = _Emb(root)


def test_cold_body_instantiates_cold_and_inherits_parent_drives():
    """The Exp 41 cold body starts past the comfort band and keeps the shared
    infant's drives (deep-merge via ``extends``)."""
    e = ComponentRegistry().instantiate("bodies/infant_humanoid_cold")
    assert e.name == "infant_humanoid_cold"  # distinct name (avoids tool-name collision with base)
    assert e.vital_metrics.get("core_temperature") == -0.7
    assert "hunger" in e.drive_specs and "arms.thermal" in e.drive_specs


def test_cold_need_derived_from_homeostatic_deficit():
    """A core_temperature well below set_point yields a positive ``cold`` need."""
    e = ComponentRegistry().instantiate("bodies/infant_humanoid_cold")
    drives = _read_drive_states(_Exec(e))
    assert drives.get("cold", 0.0) > 0.5
    # raw value is still exposed; the derived need is additive, not a rename
    assert drives.get("core_temperature") == -0.7


def test_no_cold_need_within_comfort_band():
    """A thermal drive within its comfort band emits no corrective need.

    The shared (non-cold) infant starts at -0.15, inside comfort_band 0.25.
    """
    e = ComponentRegistry().instantiate("bodies/infant_humanoid")
    drives = _read_drive_states(_Exec(e))
    assert "cold" not in drives


def test_cold_need_boosts_warm_affordance_via_affinity():
    """With a cold need, the drive-affinity heuristic surfaces a warm-named
    affordance over an unrelated tool — exploration OFF, pure affinity."""
    nac = NAc(NACConfig())  # exploration off: isolates the affinity path
    r = nac.recommend_action(
        agent_id="a",
        available_tools=["hearth_warm_self", "unrelated_widget"],
        current_drives={"cold": 0.7},
        min_confidence=0.3,
    )
    assert r is not None and r["tool_name"] == "hearth_warm_self"
