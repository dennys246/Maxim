"""Tests for the two-layer agent permissions system (C4)."""

from __future__ import annotations

import pytest

from maxim.agents.permissions import (
    AgentPermissions,
    PerceivedAuthority,
    PerceivedAuthorityTracker,
    SEMAccessRule,
    tool_permissions_from_settings,
)
from maxim.runtime.executor import Executor
from maxim.tools.base import Tool, ToolOutput
from maxim.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _OkTool(Tool):
    name = "ok_tool"
    description = "Always returns success"
    input_schema: dict = {}

    def execute(self, **kwargs: object) -> ToolOutput:
        return ToolOutput(success=True, output="ran")


class _BashTool(Tool):
    name = "bash"
    description = "shell"
    input_schema: dict = {}

    def execute(self, **kwargs: object) -> ToolOutput:  # pragma: no cover - shouldn't run
        return ToolOutput(success=True, output="bash ran")


class _AffordanceTool(Tool):
    """Stand-in for a generated SEM affordance tool: per-entity name, shared kind."""

    name = "rusty_sword_slash"
    description = "slash with the rusty sword"
    input_schema: dict = {}
    kind = "sem-modulator-derived"

    def execute(self, **kwargs: object) -> ToolOutput:
        return ToolOutput(success=True, output="slashed")


def _registry_with(*tools: Tool) -> ToolRegistry:
    reg = ToolRegistry()
    for t in tools:
        reg.register(t)
    return reg


# ---------------------------------------------------------------------------
# Enforced layer
# ---------------------------------------------------------------------------


class TestAgentPermissions:
    def test_default_allows_everything(self):
        perms = AgentPermissions()
        ok, reason = perms.can_invoke_tool("anything")
        assert ok and reason is None

    def test_tool_deny_blocks(self):
        perms = AgentPermissions(tool_deny=frozenset({"bash"}))
        ok, reason = perms.can_invoke_tool("bash")
        assert ok is False
        assert "bash" in (reason or "")
        ok2, _ = perms.can_invoke_tool("read_file")
        assert ok2 is True

    def test_tool_allow_acts_as_allowlist(self):
        perms = AgentPermissions(tool_allow=frozenset({"read_file", "say"}))
        assert perms.can_invoke_tool("read_file")[0] is True
        assert perms.can_invoke_tool("write_file")[0] is False

    def test_sem_access_rule_specific_entity(self):
        perms = AgentPermissions(
            clearance=1,
            sem_access_rules=(SEMAccessRule(entity="vault", deny=("delete",), min_clearance=2),),
        )
        assert perms.can_access_sem("desk", "open") == (True, None)
        ok, reason = perms.can_access_sem("vault", "delete")
        assert ok is False and "denied" in (reason or "").lower()
        ok2, reason2 = perms.can_access_sem("vault", "inspect")
        assert ok2 is False and "clearance" in (reason2 or "").lower()

    def test_sem_access_wildcard_rule(self):
        perms = AgentPermissions(
            sem_access_rules=(SEMAccessRule(entity="*", deny=("self_destruct",)),),
        )
        ok, _ = perms.can_access_sem("anything", "self_destruct")
        assert ok is False

    def test_from_yaml_full_shape(self):
        data = {
            "clearance": 3,
            "tool_deny": ["bash", "write_file"],
            "sem_access": [
                {"entity": "vault", "deny": ["delete"], "min_clearance": 5},
                {"entity": "*", "deny": ["self_destruct"]},
            ],
        }
        perms = AgentPermissions.from_yaml(data)
        assert perms.clearance == 3
        assert "bash" in perms.tool_deny
        assert perms.tool_allow is None
        assert len(perms.sem_access_rules) == 2
        assert perms.sem_access_rules[0].min_clearance == 5

    def test_from_yaml_empty_returns_default(self):
        assert AgentPermissions.from_yaml(None) == AgentPermissions()
        assert AgentPermissions.from_yaml({}) == AgentPermissions()


# ---------------------------------------------------------------------------
# Executor enforcement
# ---------------------------------------------------------------------------


class TestExecutorPermissionGate:
    def test_no_permissions_runs_tool(self):
        exec_ = Executor(_registry_with(_OkTool()))
        result = exec_.execute({"tool_name": "ok_tool", "params": {}})
        assert result.success is True

    def test_deny_blocks_at_dispatch(self):
        perms = AgentPermissions(tool_deny=frozenset({"ok_tool"}))
        exec_ = Executor(_registry_with(_OkTool()), permissions=perms)
        result = exec_.execute({"tool_name": "ok_tool", "params": {}})
        assert result.success is False
        assert "denied" in (result.error or "").lower()

    def test_allow_list_blocks_unlisted(self):
        perms = AgentPermissions(tool_allow=frozenset({"say"}))
        exec_ = Executor(_registry_with(_OkTool()), permissions=perms)
        result = exec_.execute({"tool_name": "ok_tool", "params": {}})
        assert result.success is False
        assert "allow-list" in (result.error or "").lower()

    def test_alias_redirect_still_gated(self):
        """If an LLM uses an alias for a denied tool, the deny still fires
        after alias resolution lands on the canonical name."""
        from maxim.runtime.executor import TOOL_ALIASES

        TOOL_ALIASES["shell"] = "bash"
        try:
            perms = AgentPermissions(tool_deny=frozenset({"bash"}))
            exec_ = Executor(_registry_with(_BashTool()), permissions=perms)
            result = exec_.execute({"tool_name": "shell", "params": {}})
            assert result.success is False
            assert "bash" in (result.error or "")
        finally:
            TOOL_ALIASES.pop("shell", None)


# ---------------------------------------------------------------------------
# Perceived authority tracker
# ---------------------------------------------------------------------------


class TestPerceivedAuthorityTracker:
    def test_default_score_is_05(self):
        tracker = PerceivedAuthorityTracker()
        belief = tracker.get("captain")
        assert belief.score == pytest.approx(0.5)
        assert belief.observations == 0

    def test_positive_outcome_increases_score(self):
        tracker = PerceivedAuthorityTracker(alpha=0.5)
        belief = tracker.observe("captain", outcome_valence=1.0)
        # alpha=0.5, target=(1+1)/2=1.0 → 0.5*0.5 + 0.5*1.0 = 0.75
        assert belief.score == pytest.approx(0.75)
        assert belief.observations == 1

    def test_negative_outcome_decreases_score(self):
        tracker = PerceivedAuthorityTracker(alpha=0.5)
        belief = tracker.observe("traitor", outcome_valence=-1.0)
        # target=0.0 → 0.5*0.5 + 0.5*0.0 = 0.25
        assert belief.score == pytest.approx(0.25)

    def test_repeated_observations_converge(self):
        tracker = PerceivedAuthorityTracker(alpha=0.4)
        for _ in range(50):
            tracker.observe("mentor", outcome_valence=1.0)
        belief = tracker.get("mentor")
        assert belief.score > 0.95
        assert belief.observations == 50

    def test_clamps_out_of_range_valence(self):
        tracker = PerceivedAuthorityTracker(alpha=0.5)
        b1 = tracker.observe("a", outcome_valence=5.0)  # clamped to +1
        b2 = tracker.observe("b", outcome_valence=-5.0)  # clamped to -1
        assert b1.score == pytest.approx(0.75)
        assert b2.score == pytest.approx(0.25)

    def test_snapshot_returns_independent_copies(self):
        tracker = PerceivedAuthorityTracker()
        tracker.observe("x", 1.0)
        snap = tracker.snapshot()
        assert isinstance(snap["x"], PerceivedAuthority)
        snap["x"].score = 0.0
        assert tracker.get("x").score != 0.0  # original untouched

    def test_invalid_alpha_rejected(self):
        with pytest.raises(ValueError):
            PerceivedAuthorityTracker(alpha=0.0)
        with pytest.raises(ValueError):
            PerceivedAuthorityTracker(alpha=1.5)


# ---------------------------------------------------------------------------
# Campaign YAML round-trip
# ---------------------------------------------------------------------------


class TestCampaignPermissionsLoading:
    def test_campaign_def_carries_permissions_dict(self, tmp_path):
        from maxim.simulation.dm_schema import load_campaign

        yaml_text = """
campaign:
  name: test
  goal: smoke
  seed: 1
acts:
  - name: act1
    encounters: [scene1]
encounters:
  scene1:
    scene: "you stand in a room"
    choices: [look]
    branches:
      look: __END__
permissions:
  spymaster:
    clearance: 2
    tool_deny: [bash]
"""
        path = tmp_path / "camp.yaml"
        path.write_text(yaml_text)
        camp = load_campaign(path)
        assert "spymaster" in camp.permissions
        perms = AgentPermissions.from_yaml(camp.permissions["spymaster"])
        assert perms.clearance == 2
        assert "bash" in perms.tool_deny


# ---------------------------------------------------------------------------
# ``kind:<kind>`` selectors (1.1.3 hard tool allowlist)
# ---------------------------------------------------------------------------


class TestKindSelector:
    """Generated SEM affordance tools have per-entity names, so a list that
    wants "every affordance" needs a selector on ``Tool.kind``."""

    def test_allow_selector_matches_tool_of_that_kind(self):
        perms = AgentPermissions(tool_allow=frozenset({"kind:sem-modulator-derived"}))
        allowed, reason = perms.can_invoke_tool("rusty_sword_slash", kind="sem-modulator-derived")
        assert allowed is True and reason is None

    def test_allow_selector_does_not_match_other_kind(self):
        perms = AgentPermissions(tool_allow=frozenset({"kind:sem-modulator-derived"}))
        allowed, reason = perms.can_invoke_tool("ok_tool", kind="core-universal")
        assert allowed is False
        assert "allow-list" in (reason or "")

    def test_selector_never_matches_unknown_kind(self):
        # kind=None (unregistered name / caller without the tool object) is
        # a name-only check — a selector can't admit what it can't see.
        perms = AgentPermissions(tool_allow=frozenset({"kind:sem-modulator-derived"}))
        assert perms.can_invoke_tool("rusty_sword_slash")[0] is False

    def test_deny_selector_beats_exact_name_allow(self):
        perms = AgentPermissions(
            tool_allow=frozenset({"rusty_sword_slash"}),
            tool_deny=frozenset({"kind:sem-modulator-derived"}),
        )
        allowed, reason = perms.can_invoke_tool("rusty_sword_slash", kind="sem-modulator-derived")
        assert allowed is False
        assert "denied" in (reason or "")

    def test_exact_name_matching_unchanged_by_kind_hint(self):
        perms = AgentPermissions(tool_allow=frozenset({"ok_tool"}))
        assert perms.can_invoke_tool("ok_tool", kind="core-universal")[0] is True
        assert perms.can_invoke_tool("bash", kind="core-universal")[0] is False

    def test_executor_resolves_kind_from_registry(self):
        """The registry lookup lives in the Executor, so the selector works
        end-to-end on a real dispatch: the affordance runs, the core tool
        is refused, and permissions.py never imported the registry."""
        import maxim.agents.permissions as permissions_module

        assert "registry" not in permissions_module.__dict__
        perms = AgentPermissions(tool_allow=frozenset({"kind:sem-modulator-derived"}))
        exec_ = Executor(_registry_with(_OkTool(), _AffordanceTool()), permissions=perms)
        assert exec_.execute({"tool_name": "rusty_sword_slash", "params": {}}).success is True
        refused = exec_.execute({"tool_name": "ok_tool", "params": {}})
        assert refused.success is False
        assert "allow-list" in (refused.error or "")

    def test_allow_list_is_judged_on_the_canonical_name_after_alias(self):
        """`recall` is an executor alias for `memory_recall`; an allow-list
        naming the canonical tool must admit the alias (review finding:
        the pre-alias check used to refuse it)."""

        class _RecallTool(_OkTool):
            name = "memory_recall"

        perms = AgentPermissions(tool_allow=frozenset({"memory_recall"}))
        exec_ = Executor(_registry_with(_RecallTool()), permissions=perms)
        assert exec_.execute({"tool_name": "recall", "params": {}}).success is True
        assert exec_.execute({"tool_name": "memory_recall", "params": {}}).success is True
        assert exec_.execute({"tool_name": "ok_tool", "params": {}}).success is False

    def test_deny_on_the_alias_source_still_applies(self):
        class _RecallTool(_OkTool):
            name = "memory_recall"

        perms = AgentPermissions(tool_deny=frozenset({"recall"}))
        exec_ = Executor(_registry_with(_RecallTool()), permissions=perms)
        assert exec_.execute({"tool_name": "recall", "params": {}}).success is False
        assert exec_.execute({"tool_name": "memory_recall", "params": {}}).success is True

    def test_executor_deny_selector_blocks_dispatch(self):
        perms = AgentPermissions(tool_deny=frozenset({"kind:sem-modulator-derived"}))
        exec_ = Executor(_registry_with(_OkTool(), _AffordanceTool()), permissions=perms)
        assert exec_.execute({"tool_name": "ok_tool", "params": {}}).success is True
        assert exec_.execute({"tool_name": "rusty_sword_slash", "params": {}}).success is False


# ---------------------------------------------------------------------------
# Settings → permissions (the console's construction helper)
# ---------------------------------------------------------------------------


class TestToolPermissionsFromSettings:
    def test_unconfigured_is_none_not_an_empty_gate(self):
        # None keeps Executor._permissions None — the pre-1.1.3 console.
        assert tool_permissions_from_settings(None, ()) is None
        assert tool_permissions_from_settings(None, None) is None

    def test_allow_only(self):
        perms = tool_permissions_from_settings(("respond", "speak"), ())
        assert perms == AgentPermissions(tool_allow=frozenset({"respond", "speak"}))

    def test_deny_only_keeps_allow_open(self):
        perms = tool_permissions_from_settings(None, ["bash"])
        assert perms is not None
        assert perms.tool_allow is None
        assert perms.tool_deny == frozenset({"bash"})

    def test_explicit_empty_allow_is_a_real_no_tools_gate(self):
        perms = tool_permissions_from_settings((), ())
        assert perms is not None
        assert perms.can_invoke_tool("respond")[0] is False
