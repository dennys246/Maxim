- **llm.json `tiers` now apply, and a routing config that cannot be applied as written refuses the run** (#1601).
  Both `tiers` and `functions` were read from an attribute the config object never had, so neither ever changed
  routing. If your `llm.json` carries either section, check it with `maxim doctor`:
  - A `tiers` entry now sets that lane's `model_profile`, `n_gpu_layers` or `kv_quant_mode` (`f16`, `q8_0`, `q4_0`).
    A lane whose model it swaps gets a context window sized for the new model.
  - A `tiers` `model_profile` that differs from the session's model (`--llm`, `MAXIM_LLM_PROFILE`, config.json
    `llm.profile` or the persisted `--llm` choice) refuses: every lane serves the session model, so the override
    would be reported but never run. Remove one of the two, or make them the same. The same holds for a dual-LLM
    run's `--aut-model` and for each model a benchmark runs (a benchmarked model it refuses is recorded as refused,
    not failed).
  - `device` and `max_workers` refuse: no lane reads them yet, so they would change only what the banner says.
  - Any `functions` entry refuses: nothing asks the function router for a route yet (#1833).
  - An `llm.json` that does not parse, or a malformed entry (an unknown tier name or field, a value of the wrong type),
    refuses with `LLMRoutingConfigInvalid` naming it. Before, a broken file was read as empty and a bad entry skipped.
  The refusal comes at run start, before any download or server spawn, and exits 2: no run mode (server, robot,
  research, the menu's campaigns, a dual-LLM run) turns it into a run without the LLM or a different error;
  `maxim doctor` runs and shows it as "llm.json routing". Offline, a lane's own
  `model_profile` in `tiers` counts as pinned (V28).
