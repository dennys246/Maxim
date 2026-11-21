- **Loop state files now live under `<data home>/runtime_state/`** (#1264): `<data home>/runtime_state/<agent>/`
  (`MAXIM_DATA_HOME`, else `~/.maxim`), not the working directory's `data/agents/<agent>/runtime/`.
  A sim (`--sim`, the orchestrator's AUT and narrator, the `spawn_sub_simulation` tool), the interactive sim and the
  Minecraft harness keep theirs in their own sandbox through the new `run_agentic_loop(state_home=...)`.
  `--clear-memory` clears the folder in the data home, and in a home named with `--home-dir`. Old files under
  `data/` are not migrated.
