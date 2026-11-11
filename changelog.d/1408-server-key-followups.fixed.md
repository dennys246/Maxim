- **Doctor's VRAM check and the leader's `/v1/debug/vram` now know the running server's context window** (#1408):
  their n_ctx probe sent no key, so a keyed llama-cpp-server answered 401, and its command-line fallback never read
  `--n_ctx`, the flag Maxim spawns with. There is now one probe, used by both and by `maxim doctor`'s context-window
  check. It sends the key the server accepts and reads `--n_ctx`, `--ctx-size`, `-c` and vLLM's `max_model_len`.
- **A filesystem that cannot lock is reported as that, at once** (#1408): on NFS without a lock daemon, creating the
  local server key waited 30 s and then blamed "another process" for holding the lock, and a model download blamed
  another download. Both now say the filesystem cannot lock.
- **`maxim peer llm <model>` explains a refused hot swap** (#1408): when the leader's LLM server was not started by
  that leader process, the swap now answers 409 with the reason and the fix (restart with the new model) instead of
  a generic 500.
