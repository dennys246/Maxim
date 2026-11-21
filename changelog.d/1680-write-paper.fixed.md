- **`--write-paper` writes a paper again** (#1680): the benchmark's paper step passed `nickname=` to
  `ExperimentLog`, which takes `agent_nickname=`, so it raised `TypeError` on every run. A broad `except` turned that
  into "Paper generation failed (LLM unavailable)". `write_paper` now returns None only when no LLM is usable (no
  router, or no section came back), and any other failure raises.
