- **`--write-paper` writes a paper again, and says why when it cannot** (#1680): the benchmark's paper step passed
  `nickname=` to `ExperimentLog`, which takes `agent_nickname=`, so it raised `TypeError` on every run. A broad
  `except` turned that into "Paper generation failed (LLM unavailable)". `BenchmarkRunner.write_paper` now returns
  `(path, None)` or `(None, cause)`: the LLM is disabled in llm.json, the model is not usable (the
  `LocalModelUnavailable` cause), or no section came back from the LLM. The CLI prints that cause. Any other failure
  raises, a malformed `llm.json` included (a traceback after the benchmark report is saved). The router is built
  before the experiment log is written, so a run with no usable LLM leaves no file.
- `models/download.py::consented_router` checks `llm_config.enabled` itself and returns `(None, LLM_DISABLED)`, for
  every caller; `ExecAgent`, `build_run_router` and the benchmark no longer check it each. Foundry and
  `--auto-curate` on a disabled config now take their template fallback with a printed "LLM is disabled" line, instead
  of building a router that answers nothing.
- The research Writer no longer saves, announces or hands the reviewer an empty `paper.md` when no section came back.
  A partial draft is saved and its missing sections are logged. With no paper, the research protocol skips the review
  and revisions and reports the verdict `no_paper` with its cause (in `research_result.json`'s `review_issues` and the
  summary), and the CLI exits 1, where an empty draft used to be reviewed and accepted.
- The Writer's and the Reviewer's LLM calls drop to "no text" or "no feedback" only on a `BackendError`; a
  `LeaderRefusal` or any other exception propagates. A verdict confidence that is not a number is logged and read as
  0.5.
- A sim's `record_experiment` / `query_experiments` tools are no longer dropped behind a debug log when building
  them fails: the failure raises.
