- **`llm.offline`: an offline mode enforced in the process** (#1315 item 10a, config format 1.5): with
  `llm.offline: true`, every `maxim` command and every `maxim.api` verb that builds a run engages the offline latch
  before any subprocess or socket, and the router runs in-process lanes only (`llama_cpp`, `transformers`). A
  cloud, peer or loopback-server lane refuses with `OfflineRefusal` at router build and at every call, and
  auto-spawn of a local server is skipped. `maxim config downgrade` warns that a build without the key runs online.
