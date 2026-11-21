- **A goal's tool call is no longer proposed on every loop pass** (#1414, second half): the agent-fallback path
  never told the goal agent what happened to its pending call, so a call autonomy refused was refused again on every
  pass, and a call that ran was proposed again. The loop now closes the call: a refused one through the new
  `AgenticGoalAgent.refuse_tool_call` (and `MaximAgent.refuse_tool_call`), which ends the goal unless the sub-goal
  is `SKIP` (`ProposedGoal.handle_sub_goal_refusal`), and one that ran or raised by publishing a `ToolResult`.
  - The bus's `ToolResult` subscribers (`MemoryAgent`, `StatisticianAgent`, the API's `tool_call` callbacks) now
    receive the agent-fallback path's calls; nothing published a `ToolResult` before. A refusal is not published
    as one. The main dispatch path still publishes none (#1147).
