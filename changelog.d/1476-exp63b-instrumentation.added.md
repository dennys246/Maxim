- **Exp 63b's recall instrumentation** (#1476): `enrichment_trace` gains `cue_text` and `recall_failed`.
  - `cue_text` is the whole text the cue path queried. That is the input on a turn's first enrichment, the model's own
    reasoning on later deliberation cycles, a thought, or a percept. With `MAXIM_LOG_FILE` set, each is now logged in
    full, where it used to be cut at 120 characters.
  - `recall_failed` is true when the recall itself failed (a failed graph path is not counted).
- **A store digest on save and load.** When INFO logging is on (as it is under `MAXIM_LOG_FILE`), Hippocampus `save()`
  and `load()` each log one `hippocampus_store_digest` line. It holds a SHA-256 over what recall matches each record on
  (its id, content keys and copy key), plus a 16-hex hash per record, so the line grows with the store. The file is
  named by its basename.
  - Cost: computing it takes about 0.25 s per 5,000 records under the store's read lock. Without INFO, nothing is
    computed.
  - No recall or storage behaviour changes.
