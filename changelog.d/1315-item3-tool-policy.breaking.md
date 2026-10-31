- **Every executor now takes a required `ToolPolicy`; `permissions=` is removed** (#1315 item 3, the atonia gate).
  `Executor(registry, *, policy=...)`, `build_executor(..., policy=...)` and `AgentConfig(..., policy=...)` require a
  `maxim.runtime.tool_policy.ToolPolicy`. Passing `None` or omitting it raises `TypeError`, and so does the old
  `permissions=` keyword, the `AgentConfig.permissions` field and a positional argument after `Executor`'s registry.
  Migrate:
  - `build_executor(reg, pain_bus=..., permissions=None)` →
    `build_executor(reg, pain_bus=..., policy=ToolPolicy.unrestricted(reason="..."))`;
  - an `AgentPermissions` allow or deny list → `ToolPolicy.only(names, ceiling=Effect.ALL, reason=...)` or
    `ToolPolicy.deny(names, reason=...)`;
  - `AgentConfig(...)` for an agent with no executor (`create_agent`) → `policy=None`;
  - `agents.permissions.tool_permissions_from_settings` → `runtime.tool_policy.policy_from_tools_settings`.

  None of these is in `docs/user/stable_api.md`; the stable verbs (`maxim.run`, `create.*`, `load.*`) build their
  own policy and are unchanged.
