- **Two writers of one file no longer share a temp file** (#1663): `atomic_write_text`, `atomic_write_json`,
  `atomic_write_secret` and `atomic_write_bytes` now write through a temp file unique to each write, created
  exclusively beside the target (`.<name>.<12 hex>.tmp`), instead of the fixed `<name>.tmp`. Before, two concurrent
  writers of one path could interleave their bytes in the shared temp, and the second `os.replace` failed with
  `FileNotFoundError` once the first had taken the file. File permissions are unchanged (the umask default, or
  `initial_mode`), and the LLM spawn-record sweep removes a dead writer's temp under either name.
