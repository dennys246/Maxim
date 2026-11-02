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
  own policy, so their signatures are unchanged (`maxim.run`'s file-tool reach changes; see the next entry).
- **`maxim.run()`'s path-taking file tools reach only the run's workspace, `<home>/.maxim_workspace`** (#1342, first
  half; the prefetch half stays open). `home_dir` defaults to `~/.maxim`. Before, `read_file`, `write_file`,
  `edit_file`, `glob`, `search_code` and the git tools reached the process working directory. They no longer reach
  it, nor the home's own stores, config or other agents' homes. `bash` and `execute_file` now start in the workspace,
  but a shell command, or the file's code, can still reach any absolute path; `maxim.run`'s passive mode refuses both. Migrate: put the files a run should see in `<home_dir>/.maxim_workspace` (created on first
  run), or pass `home_dir=` to choose the home.
