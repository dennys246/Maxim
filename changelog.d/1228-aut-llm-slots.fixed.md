- **A dual-LLM run's AUT runs all of its own cognition on its own model** (#1228): with `aut_model` set, the AUT's
  FearAgent safety review and its imagination designer ran on the narrator's router, and its ExecAgent goal
  proposals built a private router on the narrator's profile. All three now hold the AUT's router. `MaximAgent`
  takes `shared_router=` and hands it to its ExecAgent, so the sim narrator, a spawned sub-simulation and the
  console's talk loop also stop building a second, private router.
