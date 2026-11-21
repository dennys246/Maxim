- **A dual-LLM sim whose AUT router build fails no longer leaves the process on the AUT's model** (#1538). The
  build changed `MAXIM_LLM_PROFILE` to `--aut-model` and set it back only when it succeeded. After a failure,
  every later router build and `load_llm_config()` in the process, the `/new` re-entry's among them, served the AUT's
  model. `build_primary_router` now takes `profile=` and keeps it for the lanes it builds later, so the AUT build and
  `maxim.create.router(model=)` pass the model in and never change the process's profile, even while the build runs.
