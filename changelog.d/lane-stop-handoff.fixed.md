- **A worker lane no longer crashes its dispatcher when a job arrives as it stops**: `Lane.stop()` shut the
  executor down after a bounded join, so a dispatcher that outlived the join, or a dependency-gate thread, could hand
  a job to the closed executor (`RuntimeError: cannot schedule new futures after shutdown` inside the thread, seen on
  CI), and that job stayed pending forever. Such a job is now marked failed, which wakes anything waiting on it.
