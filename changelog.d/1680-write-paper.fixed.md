- **`--write-paper` writes a paper again, and says why when it cannot** (#1680): the benchmark's paper step passed
  `nickname=` to `ExperimentLog`, which takes `agent_nickname=`, so it raised `TypeError` on every run. A broad
  `except` turned that into "Paper generation failed (LLM unavailable)". `BenchmarkRunner.write_paper` now returns
  `(path, None)` or `(None, cause)`: the LLM is disabled in llm.json, the model is not usable (the
  `LocalModelUnavailable` cause), or no section came back from the LLM. The CLI prints that cause. Any other failure
  raises. The router is built before the experiment log is written, so a run with no usable LLM leaves no file.
- The research Writer no longer saves, announces or hands the reviewer an empty `paper.md` when no section came back.
  A partial draft is saved and its missing sections are logged. A section's LLM call now drops to "no text" only on a
  `BackendError`; any other exception propagates.
- A sim's `record_experiment` / `query_experiments` tools are no longer dropped behind a debug log when building
  them fails: the failure raises.
