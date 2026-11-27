- **A broken `robots.yaml` now stops a robot run instead of silently running headless** (#1806 PR 0): a YAML
  parse error, a non-mapping root or entry, or `config: null` raises `RobotsConfigError`; a missing file is still
  fine. The `./robots.yaml` fallback in the current directory is removed, and a relative `MAXIM_DATA_HOME` is
  refused when resolving `robots.yaml`.
- **`--robot` is a real flag and the robot flags always start the robot runtime** (#1806 PR 0): `--robot` is now an
  explicit alias of `--robot-name` (it only worked as an argparse abbreviation before), so `maxim --robot-name
  reachy_mini` and `maxim --robot=reachy_mini` no longer fall back to the menu. `MAXIM_ROBOT_NAME` is read when the
  robot is resolved, so setting it alone no longer makes a bare `maxim` enter the robot runtime. `--rob`/`--robo`
  are now ambiguous abbreviations. A run-mode switch re-execs with `--robot-name` only when one was typed, and an
  empty `MAXIM_ROBOT_NAME` now counts as unset (it used to become the name `""`).
- **A robot run with a broken robot configuration exits 2 with the fix** (#1806 PR 0): a `RobotsConfigError` or an
  invalid `MAXIM_ROBOT_TIMEOUT` is reported as a configuration error with its fix hint, not as a crash.
- **One Reachy daemon probe** (#1806 PR 0): `maxim doctor`, `maxim-diagnostics` and the controller's connect
  pre-check share `hardware/reachy/probe.py::probe_daemon`. Every read of the robot daemon is now dialled directly at
  a private address, never through an environment proxy: the probe's status read, the controller's fresh pose
  readback before each move, and the REST sound-direction (DoA) reader. A robot reachable only through a proxy, or
  at a public address, now fails these reads.
- **`maxim doctor`'s robot check changed** (#1806 PR 0):
  - it reports info, without dialling, for an entry that is not a `reachy_mini`, or that uses a `tunnel:` or
    `connection_mode: localhost_only` (those are checked at connect);
  - `ok` now also requires a readable daemon version, so an unparseable version is a `warn`;
  - a `host:` name that does not resolve fails with "does not resolve" instead of a TCP error;
  - a broken `robots.yaml` gets its own warning with the fix, and every message names the resolved
    `<data home>/robots.yaml` path rather than `~/.maxim/robots.yaml`;
  - a daemon status with no `state` now grades `warn` instead of `ok`;
  - an IPv6 `host:` (or a name with only an IPv6 address) is reported as not probed, IPv4 only, instead of failing;
    the robot run still connects to it as given.
- **`hardware.load_robots_config` and `RobotsConfig.from_yaml` raise `RobotsConfigError` for a broken file**
  (#1806 PR 0), for Python callers too, where they used to return an empty configuration. A missing file still
  returns an empty one.
- **`create_sandbox_script` returns the script's resolved path** (on macOS `/private/var/...` rather than
  `/var/...`) and refuses names with spaces or non-ASCII characters.
- **`edit_file` now asks the tool-result cache to invalidate the edited path, as `write_file` does** (#1806 PR 0).
  Reads stay fresh either way because the cache checks file modification times; this makes the two write tools
  consistent.
- **A non-number `MAXIM_ROBOT_TIMEOUT` raises a `ConfigurationError`** naming the variable instead of a bare
  `ValueError`.
