- **The stall detector's texts for the sim narrator no longer name the agent's tools** (#1592): the diversity
  checkpoint gives the agent under test's action and tool counts and points the narrator at its own
  `observe_actions`, instead of listing the agent's tool names, and the stall nudges name only `send_message`. A
  narrator retry after it proposed a tool it lacks no longer repeats that tool's name: it says that a tool not in the
  narrator's list, such as one in the agent's actions, is the agent's, and lists the narrator's own tools. The Exp 63b
  smoke stopped with `planning_failed` on this; whether the narrator now recovers on a live model is for the next
  smoke to show. The narrator still reads the agent's tool names in an `observe_actions` result (#1605).
