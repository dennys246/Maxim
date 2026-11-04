- **`llm.offline`: an offline mode enforced in the process** (#1315 item 10a, config format 1.5): with
  `llm.offline: true`, every `maxim` command, the router build and every `maxim.api` verb that builds a run engage
  the offline latch before any subprocess or socket, and the router runs in-process lanes only (`llama_cpp`,
  `transformers`). A cloud, peer or loopback-server lane refuses by name with `OfflineRefusal` at router build, at
  backend construction, at every call and at warmup; auto-spawn of a local server is skipped and the lane loads its
  model in-process. Set it with `maxim config set llm.offline true` (there is no env alias). A `config.json` that
  cannot be parsed now refuses startup, because whether it means offline cannot be known. `maxim doctor` shows each
  check that offline refuses as a "refused (offline)" row. `maxim config downgrade` warns that a build without the
  key runs online.
