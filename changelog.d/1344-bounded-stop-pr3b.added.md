- **`maxim llm reap` cleans up a leftover LLM server, at a terminal only** (#1344): every llama-cpp-server maxim
  spawns now leaves a spawn record (pid, start time, model, spawner) in `~/.maxim/run/llm_server/` (the passwd home;
  `MAXIM_LLM_SPAWN_RECORD_DIR` overrides). `maxim llm reap [--port N]` reports the recorded server on the port and
  stops it once you type its pid. It never touches a server without a matching record, and outside a terminal it only
  reports (exit 2). The wrong-model and key-rejected refusals name the command; `maxim doctor` lists stale records and
  recorded servers whose spawner is gone. Nothing is ever reaped automatically.
