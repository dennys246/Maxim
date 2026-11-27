- **llm.json `tiers` and `functions` now apply** (#1601). Both sections were read from an attribute the config object
  never had, so neither ever changed routing. If your `llm.json` carries either section, it now takes effect: a
  `tiers` entry sets that lane's `model_profile`, `device`, `max_workers` or `n_gpu_layers`, and a `functions` entry
  routes that function to its `tier` and `fallback` tiers. A malformed entry now refuses the router build with
  `InvalidConfigError` naming it, instead of being ignored: an unknown tier name or field, a non-object entry, a
  function without a `tier`, or a value of the wrong type. Offline, a lane's own `model_profile` in `tiers` now counts
  as pinned (V28).
