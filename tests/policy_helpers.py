"""Test-side ``ToolPolicy`` shorthands (#1315 item 3).

Every ``Executor``, ``build_executor`` and executor-building ``AgentConfig`` takes a required ``policy=``. Tests that
are not ABOUT the policy pass :data:`TEST_POLICY`; tests that gate on names build one from an ``AgentPermissions``
with :func:`policy_for`, which goes through the public factories only (``ToolPolicy`` has no public constructor).
"""

from __future__ import annotations

from maxim.agents.permissions import AgentPermissions
from maxim.runtime.tool_policy import ToolPolicy

TEST_POLICY = ToolPolicy.unrestricted(reason="test: not about the tool policy")


def policy_for(permissions: AgentPermissions, *, reason: str = "test: name gate under test") -> ToolPolicy:
    """The policy whose name gate is *permissions*' allow and deny lists (ceiling ``ALL``). Clearance and SEM rules
    are not carried: no factory takes them, and no test of the name gate needs them."""
    if permissions.clearance or permissions.sem_access_rules:
        raise ValueError("policy_for carries the allow and deny lists only")
    policy = ToolPolicy.deny(permissions.tool_deny, reason=reason)
    return policy if permissions.tool_allow is None else policy.narrowed(only=permissions.tool_allow)
