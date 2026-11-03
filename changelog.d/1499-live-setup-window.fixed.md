- **A Reachy run that fails while starting no longer leaves the robot free to move** (#1499 PR 3). If `live()` raised
  after opening its stop event but before its main loop (a transcriber or worker thread that would not start), the
  teardown was skipped: the motor queue stayed open, the controller's stop latch never set, and the agentic runtime
  kept issuing moves. The teardown now covers the whole setup, and it no longer trips over a worker thread that never
  started.
