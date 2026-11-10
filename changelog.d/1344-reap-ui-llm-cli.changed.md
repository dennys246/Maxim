- **`maxim llm reap`'s operator side now lives in `runtime/llm_cli.py`** (#1344 PR 3b follow-up): the report, the
  terminal gate and the typed-pid prompt moved there. `runtime/llm_spawn_record.py` keeps the records, the identity
  and listener checks and the kill. The command behaves the same.
