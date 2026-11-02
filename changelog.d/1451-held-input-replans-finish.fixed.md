- **An input the narrator receives while it is planning its finish is now read before the run ends** (#1451). The
  narrator plans one request at a time, so an operator message, a checkpoint or a nudge that arrives meanwhile waits
  for its next submit. When the answer it waited behind was `finish_simulation`, the run ended without the narrator
  reading it.
  - Such a finish is now set aside once. The narrator is told why, plans again with every held input in view, and
    that answer runs. If inputs arrive behind that second finish too, it runs, and one WARNING names the inputs it
    was not shown ("unread at finish"). A finish the tool refuses (below the run's turn floor) leaves the rule armed
    for the next one.
  - A plan request now shows the inputs it carries under `=== New Instructions ===`, as a follow-up already did.
  - A tool declares that it ends the run with the new `Tool.ends_run` attribute (`FinishSimulationTool` sets it).
  - A run that has stopped no longer submits a planning call. Until now the tick that set the stop could still send
    one, after the narrator's finish or an operator `/cancel`, and its answer never ran.
