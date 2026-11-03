- **A Reachy run that fails while starting now fences motion at once** (#1499 PR 3). If `live()` raised after opening
  its stop event but before its main loop (a leader refusal from the agentic runtime's start, a worker thread that
  would not start, an unwritable audio chunk directory), its teardown was skipped: the motor queue stayed open, the
  controller's stop latch never set, and the stop event stayed registered until the caller's shutdown, or for good
  when `live()` was called directly. The teardown now covers the whole setup and skips a worker thread that never
  started. A shutdown that fails behind such an error is logged, so the original error, a refusal's exit code
  included, still reaches the caller.
