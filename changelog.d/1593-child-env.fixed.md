- Child processes no longer inherit a `PYTHONPATH` that a third-party `.pth` hook rewrote (#1593). On macOS, the
  GStreamer wheels that `reachy_mini` pulls in set `PYTHONPATH` to this venv's site-packages at every interpreter
  start. So a sub-sim, a spawned server, or a leader re-exec'd by `maxim peer restart` could import a stray
  site-packages copy of `maxim` instead of the checkout. A child running another interpreter, such as another
  repo's venv, saw this venv's packages as its own.
  - The spawn sites build their environment with the new `maxim.utils.child_env.child_env()`, which drops the
    `PYTHONPATH` entries inside the running interpreter's own site-packages. That covers the sub-sim harnesses,
    the local LLM server, pip calls, the matplotlib preflight, the tmpdir sandbox, host tools (`host_tool_env`)
    and `box_quiet`. Both re-execs now use `os.execve(..., child_env())`.
  - `maxim doctor` fails when `maxim` runs from a checkout while another copy sits in site-packages, and says how
    to remove it.
  - A new CI lint (`scripts/lint_child_env.py`) ratchets the spawns not built on `child_env`.
