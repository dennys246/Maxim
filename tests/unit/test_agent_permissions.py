"""Tests for the two-layer agent permissions system (C4)."""

from __future__ import annotations

import pytest

from maxim.agents.permissions import (
    AgentPermissions,
    PerceivedAuthority,
    PerceivedAuthorityTracker,
    SEMAccessRule,
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
