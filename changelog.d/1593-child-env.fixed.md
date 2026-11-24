- Child processes no longer inherit a `PYTHONPATH` that a third-party `.pth` hook rewrote (#1593). On macOS, the
  GStreamer wheels that `reachy_mini` pulls in set `PYTHONPATH` to this venv's site-packages at every interpreter
  start. So a sub-sim, a spawned server, or a leader re-exec'd by `maxim peer restart` could import a stray
  site-packages copy of `maxim` instead of the checkout. A child running another interpreter, such as another
  repo's venv, saw this venv's packages as its own.
  - The spawn sites build their environment with the new `maxim.utils.child_env.child_env()`, which drops the
    `PYTHONPATH` entries inside the running interpreter's own site-packages. That covers the sub-sim harnesses,
    the local LLM server, pip calls, the matplotlib preflight, the tmpdir sandbox, host tools (`host_tool_env`)
    and `box_quiet`. Both re-execs now use `os.execve(..., child_env())`.
  - `maxim doctor` has a new "maxim install" check. It fails when `maxim` is imported from a site-packages copy
    that no installed package claims, which is what a path-style editable install loses to. It also fails when
    `maxim` runs from a checkout while another copy sits in site-packages, or when a child process started like
    Maxim's own imports a different tree. Each failure says how to fix it.
  - A new CI lint (`scripts/lint_child_env.py`) ratchets the spawns not built on `child_env`. The harness
    provenance lint now also refuses a sub-sim environment built from `os.environ` outside `child_env`, so
    `assert_repo_interpreter`'s probe and the real sub-sims are checked to agree.
