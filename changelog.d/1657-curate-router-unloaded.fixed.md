- **`--auto-curate` no longer holds its model in memory for the whole sim** (#1657): when curation finds a coverage
  gap, the router it builds is waited for before use and unloaded once curation ends, so a local model is not
  loaded twice. New `LLMRouter.unload()` releases every backend a router built, and never waits on a model
  still loading (that load releases itself when it ends). Two shutdown paths that called a release method the
  router never had now really free the model: `maxim --foundry`'s on a load timeout, and the robot runtime's
  shutdown (`LaneBackendManager.unload_all`), which now unloads its in-process models and closes their HTTP
  clients.
