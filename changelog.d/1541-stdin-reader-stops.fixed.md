- **A simulation's stdin reader now stops when the run ends** (#1541). The reader used to block in `input()` and
  exit only on `/cancel`, end of input or an interrupt. So a run that ended any other way (the narrator's finish, a
  campaign finish, `max_turns`, the hard abort) left it running, and in a process that ran several simulations it
  took the next line typed for a later run.
  - The line reader now polls a selectable stdin (a terminal, a pipe or a file, on POSIX) between checks of the run's
    stop, in `simulation/stdin_lines.py`.
  - Teardown joins it, with a bound, before the post-sim prompt reads stdin.
  - A stdin that cannot be polled (no file descriptor, or Windows) still blocks. Such a reader drops the line it
    reads after the stop, and teardown logs a warning that it did not stop.
