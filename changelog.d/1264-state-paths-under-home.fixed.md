- **Each agent loop run keeps its own runtime-state file** (#1264): a sim's agent under test and its narrator (both
  `MaximAgent`, started in the same second) wrote one CWD-relative `state_<run_id>.json`, and the later write
  replaced the earlier. Each run now gets `state_<run_id>_<token>.json` with a random token. A run whose file
  already exists fails at start with `StatePathCollision` instead of overwriting it; any other failure to create
  the file is logged and the run goes on. Run ids are unchanged; characters a filename cannot hold become `_` in
  the filename only.
