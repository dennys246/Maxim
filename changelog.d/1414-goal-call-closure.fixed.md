- **A goal's tool call is no longer proposed on every loop pass** (#1414, second half): the agent-fallback path
  never told the goal agent what happened to its pending call, so a call autonomy refused was refused again on every
  pass, and a call that ran was proposed again. The loop now closes the call directly: one that ran or raised through
  the new `AgenticGoalAgent.close_tool_call` (and `MaximAgent.close_tool_call`), a refused one through
  `refuse_tool_call`, which ends the goal unless the sub-goal is `SKIP` (`ProposedGoal.handle_sub_goal_refusal`).
  - Goals now end, so `GoalCompleted` reaches memory, the statistician and ExecAgent for these goals, and the
    active plan advances (replanning a failed phase). A refused goal ends with the new `GoalCompleted.refused` set (so does one that went on past a
    refused `SKIP` sub-goal, even as a success): ExecAgent's contemplation stats and NAc link skip it, and
    everything else sees the goal's outcome.
  - **RETRY sub-goals retry.** A launch counted twice toward `max_retries`, so a RETRY sub-goal never retried; it
    now gets `max_retries` launches.
  - The statistician records every goal outcome in one `goal:success` series. It kept one series and one
    detector per goal id, which never recur, so they never reached a detector's minimum and only grew, in
    memory and in its saved state. Goals now complete, so that growth would have been live.
  - Still no `ToolResult` is published by any dispatcher (#1727).
