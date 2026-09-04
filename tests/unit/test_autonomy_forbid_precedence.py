"""P4d: a hard forbid beats ``AutonomyController.ALWAYS_ALLOWED_TOOLS``.

``can_execute_action`` used to consult the always-allowed shortcut BEFORE
``SafetyConstraints.check_constraints``, so any tool that was both
always-allowed and forbidden (``search_code`` / ``git_diff`` from the
console's Talk path, whose forbidden set is derived from the registry)
executed anyway. Constraints are now evaluated first; the shortcut only
skips the approval / supervision machinery.
"""

from __future__ import annotations

import pytest

from maxim.agents.autonomy import (
    AutonomyController,
    AutonomyLevel,
    SafetyConstraints,
    SupervisionPolicy,
)

ALWAYS = sorted(AutonomyController.ALWAYS_ALLOWED_TOOLS)


def _forbidding(tool: str, level: AutonomyLevel = AutonomyLevel.AUTONOMOUS) -> AutonomyController:
    return AutonomyController(
        initial_level=level,
        safety_constraints=SafetyConstraints(forbidden_tools=frozenset({tool})),
    )


class TestForbidBeatsAlwaysAllowed:
    @pytest.mark.parametrize("tool", ALWAYS)
    def test_every_always_allowed_tool_is_refused_when_forbidden(self, tool):
        allowed, reason = _forbidding(tool).can_execute_action({"tool_name": tool})
        assert allowed is False, f"{tool}: always-allowed must not out-rank SafetyConstraints.forbidden_tools"
        assert reason and "forbidden" in reason

    @pytest.mark.parametrize("level", list(AutonomyLevel))
    def test_forbid_wins_at_every_level(self, level):
        allowed, _ = _forbidding("git_diff", level).can_execute_action({"tool_name": "git_diff"})
        assert allowed is False

    def test_talk_shaped_derived_forbid_covers_search_code_and_git_diff(self):
        # Mirrors console/handle.py: everything outside the conversational
        # set is forbidden, derived from the registry rather than listed.
        from maxim.console.handle import TALK_CONVERSATIONAL_TOOLS

        conversational = set(TALK_CONVERSATIONAL_TOOLS)
        registry = conversational | {"search_code", "git_diff", "bash"}
        ctrl = AutonomyController(
            initial_level=AutonomyLevel.AUTONOMOUS,
            supervision_policy=SupervisionPolicy(allowed_tools=set(conversational)),
            safety_constraints=SafetyConstraints(
                forbidden_tools=frozenset((registry - conversational) | set(SafetyConstraints().forbidden_tools))
            ),
        )
        for tool in ("search_code", "git_diff"):
            assert tool in AutonomyController.ALWAYS_ALLOWED_TOOLS
            assert ctrl.can_execute_action({"tool_name": tool})[0] is False, tool
        for tool in ("respond", "read_file", "glob"):
            assert ctrl.can_execute_action({"tool_name": tool})[0] is True, tool


class TestAlwaysAllowedStillBypassesLevelChecks:
    """The shortcut keeps its purpose: skip approval, not safety."""

    def test_always_allowed_runs_in_planning_mode(self):
        ctrl = AutonomyController(initial_level=AutonomyLevel.PLANNING)
        assert ctrl.can_execute_action({"tool_name": "read_file"}) == (True, None)
        # A non-shortcut tool still needs approval in PLANNING.
        assert ctrl.can_execute_action({"tool_name": "write_file"})[0] is False

    def test_always_allowed_ignores_supervision_allow_list(self):
        ctrl = AutonomyController(
            initial_level=AutonomyLevel.SUPERVISED,
            supervision_policy=SupervisionPolicy(allowed_tools={"respond"}),
        )
        assert ctrl.can_execute_action({"tool_name": "glob"}) == (True, None)

    def test_paused_still_refuses_always_allowed(self):
        ctrl = AutonomyController(initial_level=AutonomyLevel.AUTONOMOUS)
        ctrl.emergency_halt("test")
        allowed, reason = ctrl.can_execute_action({"tool_name": "respond"})
        assert allowed is False
        assert "paused" in (reason or "")
