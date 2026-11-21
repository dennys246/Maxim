- **`maxim.run`, `maxim.imagine` and `maxim.campaign` no longer put their model in `MAXIM_LLM_PROFILE`** (#1586).
  Each set the variable for its whole run, so any other thread in the process that resolved a profile saw this
  run's model. `imagine` and `campaign` also set it before loading their scenario or campaign, so a file that failed
  to load left the caller's model as the process's profile. The model is now handed to the run's router build
  (`start_simulation_mode(llm_profile=)` for the sims, carried into a `/new` run), and only `MAXIM_LLM_ENABLED`'s
  default is set, after the inputs load, and restored when the call returns or raises.
- **`maxim.research(model=)` runs on the model it is given** (#1586). It reported that model as
  `orchestrator_model` while its Writer, Reviewer and narrator ran on the session's profile. Like the other API verbs,
  a call with no `model` now runs on the last chosen model, or `mistral-7b` when none was chosen (not the session's
  `llm.profile`), and an explicit `model` is remembered as the next default.
