- **A stopped worker lane no longer leaves jobs waiting forever, or crashes its dispatcher**: `Lane.stop()` shut the
  executor down after a bounded join. A dispatcher that outlived the join, or a dependency-gate thread, could hand a
  job to the closed executor (`RuntimeError: cannot schedule new futures after shutdown` inside the thread, seen on
  CI), and jobs cancelled by the shutdown, jobs still queued at stop, and jobs submitted after it were never run or
  marked. Every job a stopped lane did not run now ends failed, which wakes anything waiting on it; `submit()` and
  `start()` on a stopped lane raise.
  The refusal is a typed `LaneStopped` (a `RuntimeError`), and `LLMWorker` treats it like a full queue: a request or
  retry that races its stop is rejected, not raised.
