- **`maxim.run(home_dir=...)` keeps its loop state under that home** (#1264, D15): `<home_dir>/runtime_state/<agent>/`
  rather than the data home's; `maxim --clear-memory runtime_state --home-dir <home_dir>` clears it.
