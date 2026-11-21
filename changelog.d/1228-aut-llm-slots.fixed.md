- **A two-model run (`--aut-model`) runs the AUT's own cognition on the AUT's model** (#1228): the AUT's FearAgent
  safety review and its imagination designer ran on the narrator's router, so "the AUT ran on model X" was partly
  false. Both now hold the AUT's router, as do a spawned sub-simulation's agent (its ExecAgent, FearAgent and LLM
  worker) and the AUT's ExecAgent slot (a dormant path in the sim, which never built a router there). The scene
  manifest and the campaign runners generate the world and stay on the narrator's model.
- **`--aut-model` whose router cannot be built refuses the run** (#1228): a build that returns no router or raises
  now raises `maxim.models.language.types.AutRouterUnavailable` (the CLI exits 2) instead of falling back to the
  shared router, so a two-model run never silently becomes a one-model run.
- **The sim report's cost counts the AUT's model** (#1228): `cost_usd`, `total_input_tokens` and
  `total_output_tokens` are the whole run's (the narrator's router plus, in a two-model run, the AUT's; before, a
  two-model run counted the narrator's only), split into the new `narrator_cost_usd` / `narrator_input_tokens` /
  `narrator_output_tokens` and `aut_cost_usd` / `aut_input_tokens` / `aut_output_tokens`. A one-model run's totals
  are unchanged and its `aut_*` fields stay 0. Additive keys: `report.json`'s `_format_version` stays `1.0`.
