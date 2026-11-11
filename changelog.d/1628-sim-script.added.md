- **`--sim-script <file>`: a scripted narrator** (#1628): `maxim --sim "<goal>" --sim-script lines.yaml
  --sim-max-turns N --interactive false` sends the file's lines (`percepts: - cli_input: <line>`) to the agent one per
  turn, through the same bridge path as the narrator's, then ends `completed`; it combines with `--resume-sim`. It runs
  only on a plain goal-string sim: before any sim starts it refuses every other `--sim` shape (a YAML file, the
  `agent`/`research`/`benchmark` aliases, the REPL, a blank goal), any option set away from its default outside a
  short whitelist (so `--dm`, `--research`, `--continuous`, `--campaign`, `--benchmark`, `--goal`, `--foundry` and the
  rest), a goal that selects a narrative arc, interactive mode, a missing or zero `--sim-max-turns` or one that differs from the line count, and
  an empty, malformed, blank or whitespace-only script. `cli.resolve_sim_script(args)` runs the same check on
  normalized args without output or exit, for a harness's preflight. `simulation/experiment.py::load_campaign_turns` gains an opt-in
  `strict=True` for it; its default behaviour is unchanged.
