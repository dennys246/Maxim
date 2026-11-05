- **The sim narrator is no longer primed to call the agent's tools** (#1592): the stall detector's diversity
  checkpoint now gives the agent under test's action and tool counts and points the narrator at its own
  `observe_actions`, instead of listing the agent's tool names, and the idle and repeated-stall nudges name only
  `send_message`. A narrator retry after it proposed a tool it lacks now says that a tool named in a checkpoint or
  a tool result is the agent under test's, and lists the narrator's own tools. The Exp 63b smoke stopped with
  `planning_failed` on this.
