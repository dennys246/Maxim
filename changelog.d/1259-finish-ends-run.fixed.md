- **A narrator's `finish_simulation` now ends the simulation** (#1259): the tool sets the run's stop signal, as a
  campaign runner's finish does, so teardown starts at once. Before, the run waited out the 180 s percept-source
  grace while the stall detector nudged the finished narrator, each nudge a fresh narrator LLM call (a probe saw six).
  `FinishSimulationTool` takes the run's `stop_event` as a required keyword.
