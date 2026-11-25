# PR 0 code review: executor lens

Reviewer: executor lens, PR 0 code review, 2026-11-25, 6b4ad2bb

**Verdict: SHIP after fixes** (E1 is the only finding that changes behaviour. It needs at least a typed message and a changelog line. Everything else is a nit or pre-existing.)

Scope: `git diff origin/main...6b4ad2bb` (base 0dd412e4), against approach §3.1, §4.1, §4.2 and §5, and the first bullet of §8. All probes ran in a scratch `git archive` copy with `HOME` set to a scratch directory and `PYTHONPATH` pointing at the copy's `src`. No robot, no LAN host, no sim. After the mutation probes, the copy was re-diffed clean against a fresh archive.

## Findings

| # | file::symbol | finding | severity | fail-open/closed | evidence |
|---|---|---|---|---|---|
| E1 | `hardware/reachy/probe.py::resolve_ipv4`, used by `controller.py::ReachyMiniController._precheck_daemon` and `doctor/checks.py::_check_one_robot` | An IPv6 literal `host:`, or a name with only an AAAA record, now fails the pre-check as "does not resolve", so `connect()` returns False. On main, `_port_open` handed the host verbatim to `socket.create_connection` (which handles IPv6) and the connect went ahead. The changelog does not mention this, and the message names the wrong cause. | important | closed | `probe_daemon("::1")` gives `failure=resolve`, error `::1 does not resolve: [Errno 8] nodename nor servname provided` |
| E2 | `probe.py::probe_daemon` (one shared `deadline_s`) | The deadline now covers the whole probe, so a slow name lookup starves the stages after it. If the lookup runs past the deadline (4 s in doctor, `min(timeout,5)+3` = 8 s in the controller), the result is `failure="tcp"` with "probe deadline spent before TCP". Doctor then shows **fail: daemon port unreachable** and advises rebooting the robot, which is the wrong cause. The controller refuses a connect that main allowed, because main gave the lookup and TCP 5 s each. A starved status read also leaves `media_released=None`, so #1704 will not re-acquire. | nit | closed | a stubbed 4.1 s lookup with a 4 s deadline gives `tcp probe deadline spent before TCP` |
| E3 | `cli.py::_run_stopped_on` | The `InvalidConfigError` branch also matches its subclasses `InvalidAgentIdError` (`decisions/nac.py`) and `AgentConfigContradiction`. That holds in **every** mode inside the run loop's `try`: agentic, sim and embodied. Those errors now exit 2 tagged `event="robot_config_invalid"`, and their traceback drops to DEBUG. Exit 2 (the operator must act) is defensible. The event name and the docstring ("a robot configuration") are not. | nit | n/a (exit-code meaning) | `exceptions.py:652,665` subclass `InvalidConfigError`; the `try` at `cli.py` wraps the agentic, sim and selfy branches |
| E4 | `probe.py::refusal`, `doctor/checks.py::_grade_daemon` | A status response with no `state` key used to grade `ok`. It now grades `warn` ("daemon state is None, not 'running'"). The changelog lists only "an unparseable version is a warn". | nit | closed | the `outcome.state != "running"` branch with `state=None` |
| E5 | `tools/base.py::_matches_by_string`, `::_matches_by_identity` (case-fold) | On case-sensitive filesystems (Linux ext4), the fold also refuses distinct files such as `~/.maxim/ROBOTS.yaml` and `~/.MAXIM/robots.yaml`. This over-refuses and does not under-refuse, so it is acceptable, but it could be stated. | nit | closed | probe: `~/.maxim/ROBOTS.yaml -> refused` |
| E6 | `utils/sandbox_executor.py::SandboxExecutor._contained_script_path` | `write_script` now returns the **realpath**, for example `/private/var/...` instead of `/var/...` on macOS. It also refuses names that contain spaces or non-ASCII characters. This changes the output of `create_sandbox_script`. Execution still works, because the later checks use realpath. | nit | closed | code read; `sandbox.py:465` returns `path` to the model |
| E7 | `cli.py::_requested_run_action` | `--robot-name ""` counts as a typed run, but `resolve_daemon_name("")` then falls through to the env var or the default. Harmless, but "typed" and "used" disagree. | nit | n/a | routing probe: `['--robot-name', '']` gives `run`, while `robot_name=''` resolves to the default |
| E8 | `probe.py::resolve_ipv4` | `socket.setdefaulttimeout` does not bound `gethostbyname`, and it is a process-global value that races other threads. This was carried over unchanged from the three copies the PR merged. | pre_existing | n/a | code read |

## Verified correct (no finding)

- **Tests:** all 587 tests in the touched and new modules pass in the scratch copy: `test_1806_pr0_*`, `test_cli_action_gate`, `test_cli_exit_code_1468`, `test_doctor`, `test_mdns_discovery`, `test_media_release_1704`, `test_media_stream_gating`, `test_motion_finite_1496`, `test_motion_path_1499`, `test_multi_robot`, `test_operational_mode_launch` and `test_reachy_connection_options`. `lint_http_surface`, `lint_orphan_modules` and `ruff` are clean.
- **Deletion probes:** each mechanism was removed in turn, and the suite went red every time:

  | Mechanism removed | Result |
  |---|---|
  | name rule | 2 fail |
  | identity walk | 3 fail |
  | string compare | 1 fail |
  | `proxy="never"` in the probe | 2 fail, including `test_doctor::test_the_status_read_is_direct_private_and_bounded` |
  | `proxy="never"` in `_get_state_full` | 1 fail |
  | `proxy="never"` in the DoA reader | 1 fail |
  | #1704 `media_released_before` forced to None, at either end | 10 and 11 fail |
  | `_run_stopped_on` branch | 2 fail |
  | `_resolved_robot_config = entry` dropped | 5 fail |
  | cwd fallback restored | 1 fail |
  | relative `MAXIM_DATA_HOME` check | 1 fail |
  | script containment | 1 fail |
  | `autonomy.py` `edit_file` branch | 1 fail |

  The tests the PR adds are not vacuous.
- **Controller connect and #1704:**
  - The pre-check keeps main's three cases: `localhost_only` goes to 127.0.0.1; an explicit host goes to the SDK verbatim; with no host, mDNS resolves and the resolved IP goes to the SDK.
  - A failed status read stays non-fatal.
  - `media_released` gets to `_warn_if_media_already_released` and `_connect_released_media` typed, with None meaning unknown, and the "unknown" branch still warns.
  - Any exception that escapes the probe is caught by `connect()`'s broad handler, which keeps the #1704 re-acquire and the tunnel stop. The probe catches `HTTPError` and `ValueError`; `fetch_url` maps `httpx.HTTPError` to `HTTPError`, and a JSON decode error is a `ValueError`.
- **`proxy="never"`:** it reaches `_checked_client(strict=True)`, which builds the client with `mounts=None, trust_env=False`, so no env or system proxy can carry the URL. The private policy accepts loopback (the tunnel) and CGNAT 100.64/10 (`is_public_address` treats it as not public).
- **Doctor never-fail:** `RobotsConfigError` and any other load error each become one warn row, and each robot row is wrapped by `_check_one_robot_never_raises`. One residual: `data_home()`'s `sys.exit` on `PermissionError` is not caught, which is pre-existing; on main it escaped `find_config_file`'s `except Exception` too.
- **CLI quick-start routing:**

  | Invocation | Route |
  |---|---|
  | bare `maxim` | menu |
  | `MAXIM_ROBOT_NAME` set, nothing typed | menu |
  | `--robot x` | run |
  | `--robot=x` | run |
  | `--robot-name reachy_mini` | run |
  | `--mode` | run |
  | `--mode=` | run |
  | `--timeout` alone | menu |
  | `--audio` alone | menu |

  The re-exec replays only a typed name. Every `src/` reader of `args.robot_name` handles None: the CLI gate, the startup log, `reexec_with_mode` and `Maxim(robot_name=...)`.
- **One resolver:** `_resolve_robot_connection` uses `resolve_robot_entry`, the same exact-id, unambiguous-primary rule `agentic_runtime` used before. All four readers (body, DoA, motor binding, `aut_mode`) now see the connect's entry. A robots.yaml load error now propagates and exits 2 through `_run_stopped_on`. Simulation, which has no `src/` caller, resolves no entry, as stated.
- **Write sites:** `write_file`, both writes in `edit_file` (both behind `_is_path_allowed`), `write_sandbox_file`, `write_to_shared_outputs` and `write_script` all call both refusals on the same string.
- **Performance:** `control_plane_write_refusal` costs about 0.9 ms per call on APFS: about 200 calls on a 4-deep workspace path, including the realpath calls and every ancestor `stat`. That is negligible next to the write it guards. Legitimate paths stay writable, including `~/.config/maximal/x` and `<ws>/robots.yaml`.

## Fixes

- **E1 (fix before merge).**
  1. In `resolve_ipv4`, check whether `name` parses as `ipaddress.IPv6Address`. If it does, raise a typed `OSError` subclass, or return `failure="resolve"`, with the text "IPv6 addresses are not supported (IPv4 only, #1806 §4.2): set host: to an IPv4 address".
  2. Make doctor's and the controller's resolve-failure messages pass that text through, instead of the generic "does not resolve".
  3. Add a changelog line under "One Reachy daemon probe": "an IPv6 or AAAA-only `host:` is now refused; use an IPv4 address."

  The alternative is to keep a verbatim dial for an IPv6 literal on the legacy path in PR 0. That contradicts §4.2, so I do not recommend it.
- **E2.** Either give the lookup its own sub-budget and keep a floor for TCP and status, for example `max(remaining, min(1.0, deadline_s/4))`. Or record a lookup that overran the budget as `failure="resolve"` with "name lookup took X s (deadline Y s)", so doctor does not blame the daemon port.
- **E3.** Rename the event to `run_config_invalid` and make the docstring say any `InvalidConfigError` in a run. Alternatively, restrict the branch to `RobotsConfigError` plus a dedicated `RobotTimeoutInvalid(InvalidConfigError)`.
- **E4–E7.** Changelog or docstring sentences only. E6 could return the un-resolved `os.path.join` path for display and keep the realpath for the checks and the write.
