# #1806 device registry: hardware-safety & runtime-lifecycle review

Reviewer: safety & lifecycle lens, read-only, 2026-11-25. Material: `docs/plans/device_registry_1806_approach.md` DRAFT v1, read against the worktree at origin/main `b7380ee74614306404297fca9a4ff6e04345bd6f` and the installed reachy-mini 1.8.3 (`.venv/lib/python3.12/site-packages/reachy_mini/`). No robot, LAN host, sim or runtime was contacted or run. Claims I could not verify in code are marked **(unverified)**.

**Verdict: DO-NOT-BUILD as written for two parts: D9's in-process `maxim.embody()` launch and the `launch.operational_mode` default (with `--save-defaults` writing it). Everything else can be built once the SHOULD-FIX items are folded into the note. Take D9 as plan-only. Factor the CLI robot launch into an internal function the CLI alone calls, leave the `maxim.embody` name for `embodied_device.md`'s `MaximHandle.embody`, and write the identity pin as a fail-closed step that runs before `ReachyMini(`.**

| # | Finding | Severity | Failure mode |
|---|---|---|---|
| 1 | In-process `maxim.embody()` (the D9 recommendation) would `os.execve` the embedder's process on an agent-requested mode switch. It also mutates process-global state and adds a second public body-attach path that collides with `MaximHandle.embody` | DO-NOT-BUILD | fail-open |
| 2 | Persisting `launch.operational_mode` (`--save-defaults`) turns a launch-time grant into stored config, which any file-writing tool can edit. Building `embody` as argv sugar would also launder it past `_explicit_flag` | DO-NOT-BUILD | fail-open |
| 3 | Pin placement and failure semantics in `ReachyMiniController.connect` are unspecified. As the code is shaped today a refusal turns into `return False`, which becomes a headless run. A pin placed after `ReachyMini(` mutates the wrong daemon (media release, auto-yaw) and can leak the media release | SHOULD-FIX | fail-open |
| 4 | `expected_hardware_id` as a "config key" is silently dropped by `RobotRegistry`'s constructor-signature filter unless it is a real constructor parameter. A `type:` or plugin override drops it too | SHOULD-FIX | fail-open |
| 5 | The SDK's `hardware_id` can be `null` (Lite daemon on macOS, sim daemon, audio chip not enumerated). Then `None == None` pins nothing, and `add` would store a null pin | SHOULD-FIX | fail-open |
| 6 | Verify, then dial: for a hostname `host:` the status GET and the SDK each resolve the name separately, and every robot defaults to `reachy-mini.local` | SHOULD-FIX | fail-open |
| 7 | `reexec_with_mode` rebuilds argv with `--robot-name` only. `--device`/`--robot` and the embody preflight are lost on a mode switch, and once D3 separates entry id from `robot_name` the new process resolves the wrong entry or none (unpinned) | SHOULD-FIX | fail-open |
| 8 | `selfy.Maxim.__init__` wraps the robots.yaml load in `except Exception → log.debug`, so §5's new typed yaml error would still degrade to an unpinned mDNS connect. Same root cause as §5, so it is folded there | SHOULD-FIX | fail-open |
| 9 | The registry's reuse branch hands back an already-registered controller without checking the requested pin | SHOULD-FIX | fail-open |
| 10 | "A Python caller cannot skip it" overclaims. `maxim.connect("reachy_mini", config=…)` and `maxim.run(robot=…)` never read robots.yaml, so they connect unpinned even when the user registered a pinned device | SHOULD-FIX | fail-open |
| 11 | Stop handle: `cancellation.request_stop` is process-owner-only (bounded_stop decision 2026-10-30), so a library handle cannot use it. With D9 plan-only no handle is owed in v1, and the later design rides `MaximHandle.stop(deadline=)` | SHOULD-FIX | n/a |
| 12 | Verified: `register_tool` and `on` do NOT reach the CLI robot runtime. Only `api.run` and the sim orchestrator call the injectors. The note's "embody() must honour both" belongs to the `MaximHandle.embody` acceptance criteria, not to v1 | SHOULD-FIX | n/a |
| 13 | Discovery under the offline latch: on macOS `find_robots` spawns `dns-sd` children, whose network I/O escapes an in-process socket latch, and `_filter_alive` TCP-dials every advertised address | SHOULD-FIX | fail-open |
| 14 | `check` and preflight are REST-only as specified (verified), but the note never says what they do for `tunnel: true` entries. They must not borrow `controller.connect()` to get a tunnel | NIT | fail-closed |
| 15 | A typed identity refusal from the robot loop exits 1 (logged as a crash): `_run_stopped_on` re-raises only `BackendError`/`LeaderRefusal`/`SpeechOutputUnavailable` | NIT | fail-closed |

---

## 1. In-process `maxim.embody()` is not feasible in v1 (DO-NOT-BUILD, fail-open)

§3b says `embody` "factors the CLI's robot-runtime launch (the `_main_impl` → `Maxim(...)` path) into one library function" that both the CLI and Python call. The code that would move does not tolerate running inside an embedder:

- **The mode switch replaces the process.** `cli.py::_main_impl` runs the robot path as a `while True` loop. After each run, `cli.py::_after_run_mode_request` reads `maxim.requested_mode`, which the agent's own `mode_switch` tool sets (`runtime/bootstrap.py`, the `ModeSwitchTool` wiring that does `setattr(maxim, "requested_mode", mode)`). On an allowed switch it calls `cli_utils.py::reexec_with_mode`, which ends in `os.execve(sys.executable, [... "-m", "maxim.cli", ...], child_env())`. In an embedder (Pulse, a notebook, a script) that replaces the embedder's process image with a `maxim.cli` robot run, and the agent can trigger it. The in-process fallback ("continue in-process") runs only when `execve` raises.
- **Process-global mutation.** Importing `embodied_runtime/selfy.py` and constructing `Maxim` (`_setup_hardware_env`) runs `mp.set_start_method("spawn", force=True)`, which overrides the embedder's start method, and sets `os.environ.setdefault("PYOPENGL_PLATFORM", "egl")`. `agentic_runtime.py` (CPU fallback) sets `os.environ.setdefault("MAXIM_LLM_PROFILE"/"MAXIM_LLM_N_GPU_LAYERS")`, and `media_loop.py` sets `os.environ["MAXIM_TRANSCRIPTION_WORKER_CPU_ONLY"]`. Logging is reconfigured three times: `configure_logging(..., force=True)` on each loop pass in `_main_impl`, `MediaLoopMixin.live`, and `Maxim.__init__` when verbose. `Maxim.__init__` also calls `atexit.register(self.shutdown)` once per instance, so the registrations accumulate. The `_main_impl` preamble does role detection (`_detect_role_or_warn` → `detect_and_apply_role`, which exports `MAXIM_ROLE`) and `_early_leader_boot` (which can start the leader proxy).
- **stdin and display.** `Maxim(interactive=True)` is the default. `embodied_runtime/input_handlers.py` puts `sys.stdin` into cbreak mode and turns echo off (`termios.tcsetattr`), and `prepare_display()` drives OpenCV GUI calls on the main thread when verbose. Neither belongs in a daemon embedder.
- **Stop.** `MediaLoopMixin._open_live_stop_event` registers the run's event with the process-wide stop (`utils/cancellation.py::register_stop_event`). That stop is raised only by `request_stop`, which is process-owner only (see #11), or by a `KeyboardInterrupt`, which arrives only on the main thread. A `live()` run on a worker thread has no stop.
- **Singleton and concurrency.** `selfy.py` registers controller types on the `RobotRegistry` process singleton at import and connects with `set_primary=True`. That clobbers any primary the embedder already set. Nothing shares `api.py::_RUN_LOCK`, so a concurrent `maxim.run(robot="reachy_mini")` and an in-process embody could open two SDK connections to one daemon, against `embodied_device.md`'s one-connection-per-device rule.
- **It collides with the device plan.** `embodied_device.md` names `MaximHandle.embody(sdk)` as "the one place a body attaches", plus `ReachyMiniController.adopt`. It also schedules "`selfy.Maxim` re-based on `MaximHandle`" for 1.4. A public `maxim.embody(name)` built on `selfy.Maxim` would be exactly the second launch path the note's own §3a forbids, and it would claim the `embody` name with different semantics (a device name, not an SDK handle) before the real seam lands.

**Fix (recommended v1 scope, D9 = plan-only):**
1. Factor the robot run loop (`_main_impl`'s robot loop from `build_home` through `_after_run_mode_request`) into an internal function in an existing CLI-side module. `cli.py` is the only caller, and the extraction carries the touch-rule shrink. It keeps `execve`, signals and stdin, because it only ever runs in a process the CLI owns.
2. Expose `maxim.devices.plan_launch(name, **overrides) -> LaunchPlan` (the resolve + REST preflight + plan block, pure and side-effect free) as Experimental. Do **not** export `maxim.embody` in v1. `stable_api.md` keeps "full capture/vision/DoA remain on the CLI runtime".
3. Write down the conditions under which a later in-process launch may land, as part of `MaximHandle.embody`'s design: no `execve` (a mode switch becomes an in-process restart or a typed refusal), no `set_start_method(force=True)`, no stdin or termios, no root-logging reconfiguration, a run-owned stop (#11), and a share of `_RUN_LOCK`.

## 2. `launch.operational_mode` persists a capability grant (DO-NOT-BUILD, fail-open)

#829/#924, as implemented in `cli.py::_runtime_mode_switch_allowed` and `cli.py::_validate_operational_grant`, say that only the operator's launch flag raises capability; the singularity warning reads "granted at launch by --operational-mode … you launched it here". D8 lets `robots.yaml` carry `launch: {operational_mode}`, so a bare `maxim embody reachy` would launch singularity (code execution) with no flag typed for that launch.

robots.yaml is a plain file under `~/.maxim`. The CI caller allow-list governs *Maxim's* writer, not the filesystem. Singularity's file tools have full filesystem access (`runtime/bootstrap.py::build_tool_registry`, "Singularity mode: full filesystem access"), and active mode is scoped to the cwd, so it reaches `~/.maxim` whenever the operator launches from `~` **(unverified: depends on the cwd)**. A run could therefore write its own next launch's grant, which is capability escalation across a restart.

Second route: if `embody` is built as argv sugar (rewriting `maxim embody reachy` to `maxim --device reachy --mode … --operational-mode …` before parsing), the synthesized flags pass `_explicit_flag(raw_argv, "--mode")` and the grant validation as if the operator had typed them.

**Fix:** `launch:` may carry `mode`, `llm`, `interactive` (and audio). The loader refuses `operational_mode` there with a typed error and the fix hint "pass --operational-mode at launch". `--save-defaults` never writes it. `embody` passes its resolved values as parsed arguments, never synthesized raw argv, and the grant check reads only the real command line. The plan block prints where the capability comes from: "active (implied by run mode exploration)" or "singularity (granted by --operational-mode)". A stored `mode` keeps its implied capability (exploration and agentic imply active, `modes/definitions.py::_LEGACY_NAME_MAP`), which is what `maxim --mode X` gives today, so it is acceptable.

## 3. Pin placement and fail-closed semantics in `connect` (SHOULD-FIX, fail-open)

The order inside `hardware/reachy/controller.py::ReachyMiniController.connect` is: SDK era check → optional `_start_ssh_tunnel` → `probe_host` (127.0.0.1 under a tunnel, explicit `host`, or `_resolve_mdns`) → `_port_open` → `_daemon_status(probe_host)` ("best-effort … never fatal", returns None on any failure) → `_warn_if_media_already_released` → `ReachyMini(...)` → `_media_released_by_connect = _connect_released_media(status)`. Everything sits inside `try: … except Exception: … return False`.

The note says only that `connect` "takes an expected_hardware_id config key and refuses on a mismatch". The three ways that goes wrong:

- **Placed after `ReachyMini(`** (for example by reading the id through the SDK client): the constructor has already set `automatic_body_yaw=False`, which is daemon-wide (embodiment.md §1), and under `no_media` it has called `release_media()`, also daemon-wide (#1704). Both land on the *wrong* robot. If the check sits between `ReachyMini(` and the `_media_released_by_connect` assignment, the failure branch's `if self._mini is not None and self._media_released_by_connect` is False, so the wrong robot's camera stays released, the #1704 bug class.
- **Implemented as a lenient `if status and status.get("hardware_id") != expected`**: a failed or slow status read (`_daemon_status` → None) skips the check and connects unpinned.
- **Raising inside the `try`**: the broad `except Exception` logs it and `return False`s. `RobotRegistry.connect_robot_with_status` then returns None, and `selfy.Maxim.__init__` logs "No robot connected — running in headless mode" and **runs anyway**. The refusal becomes a bodiless run that exits 0.

**Fix (spell it out in §4):** when an `expected_hardware_id` is set, the pin runs after the tunnel is up and the status is read, and **before `ReachyMini(`**, so nothing has touched the daemon yet. No media has been released, so there is nothing to re-acquire. The status read uses `deadline_s`. A None status, a missing or null `hardware_id`, or a mismatch raises `DeviceIdentityMismatch`/`DeviceUnreachable`. The `except` re-raises that type, or the check sits outside the `try`. Before raising, the controller calls `_stop_ssh_tunnel()` and sets state ERROR. The registry already re-raises after `controller.disconnect()`. Red-gate tests through `tests/unit/_reachy_fakes.py`: a mismatch never constructs `ReachyMini`, never releases media, and propagates as the typed error, not as a headless run. Verify by a deletion probe.

## 4. The pin key is silently dropped by the registry (SHOULD-FIX, fail-open)

`hardware/registry.py::RobotRegistry.connect_robot_with_status` filters `config` to the controller constructor's parameters and logs the rest at **DEBUG** ("wiring-layer keys, read elsewhere"). That is exactly how §5's dead `timeout:` key dies. If `expected_hardware_id` stays a free-form `config:` key and is not a `ReachyMiniController.__init__` parameter, the pin is a silent no-op. A robots.yaml `type:` override (`selfy.Maxim.__init__` honours `_entry.robot_type`) or a plugin controller drops it the same way.

**Fix:** make `expected_hardware_id` a keyword parameter of `ReachyMiniController.__init__`, and add a test that a robots.yaml-resolved config reaches the controller with it. The device-kind layer refuses a *pinned* entry whose controller class does not declare pin support (no `expected_hardware_id` parameter, no `probe()` hook), with a typed error rather than a DEBUG line. This is CLAUDE.md "push silent-no-op invariants into types".

## 5. `hardware_id` can be null (SHOULD-FIX, fail-open)

`reachy_mini/utils/hardware_id.py::get_hardware_id` reads the Pollen audio chip's serial from `/sys/bus/usb/devices` and returns None when it is absent. That happens on any non-Linux daemon (a Lite daemon running on the user's Mac), on a sim or mockup daemon **(unverified: assumed to share the code path)**, and whenever the chip is not enumerated. The daemon serves that None as-is (`daemon/app/routers/daemon.py::get_robot_hardware_id`, `daemon.py` `hardware_id=get_hardware_id()`). §2 says it is "stable across reboots and reflashes" without that qualifier.

**Fix:** `add` refuses to pin a null id and records the entry as unpinned with the reason ("daemon reports no hardware_id"). `show` and `check` print that. A pinned entry refuses a daemon reporting null (covered by #3). `maxim embody` prints "UNPINNED" in the plan block for such entries. Whether an unpinned embody should need a confirm is an owner question; I recommend yes.

## 6. Verify, then dial: two resolutions (SHOULD-FIX, fail-open)

When `host:` is a hostname, `connect` reads `_daemon_status(probe_host=<hostname>)`, which resolves inside `utils/http.fetch_url`, and then `ReachyMini(host=<hostname>)` resolves it again inside the SDK. `_resolved_host`, which `get_doa_reader` uses for its REST DoA reader, also holds the hostname. Every robot defaults to `reachy-mini.local` (§6's own row), so the two lookups can land on different robots. The identity is then verified on one robot while motor commands go to the other. The mDNS path (`_resolve_mdns` → IP) does not have this problem, and neither does the tunnel (both dial 127.0.0.1).

§7's "never pre-resolve and then dial separately (#824/#921)" is an http.py SSRF rule. It does not cover the SDK's WebSocket dial, which bypasses http.py.

**Fix:** the controller resolves once to an IP literal, verifies at that literal (`address_policy="private"`), passes the **same literal** to `ReachyMini(host=…)` and `_resolved_host`, and the address cache stores the literal. Add that reconciliation to §7.

## 7. The mode-switch re-exec drops `--device` (SHOULD-FIX, fail-open)

`cli_utils.py::reexec_with_mode` builds a fresh argv: `-m maxim.cli --robot-name <args.robot_name> --home-dir … --mode … [--operational-mode …]`. It has no subcommand, no `--device`, no `--robot`, and no `--tts`. A running `maxim embody reachy` that switches mode therefore comes back as `maxim --robot-name <daemon name> --mode Y`. The preflight and plan block are skipped. `selfy.Maxim.__init__` resolves the entry by `effective_robot_id = self._robot_id or self.name`, and the CLI passes only `robot_name`.

Once D3 separates the entry id from the daemon's `robot_name`, `hardware/config.py::resolve_robot_entry` misses on the exact id and either takes the primary (possibly a *different* entry) or nothing, which means no `host`, no pin, and an mDNS connect to `<robot_name>.local`. The re-exec'd process is pinned only by luck.

**Fix:** `reexec_with_mode` carries `--device <resolved entry id>` (plus `--robot-name` only when the operator overrode it), so the new process re-runs resolution, and the pin in `connect` (#3) re-verifies. An argv round-trip test pins it. The CLI passes `robot_id=<entry id>` to `Maxim(...)` (the note says it never does; with the D3 split it has to). Optionally the re-exec re-runs the REST preflight too. It costs about 1 s against the existing 1.5 s `MAXIM_MODE_SWITCH_DELAY_S`.

## 8. selfy swallows the robots.yaml load (SHOULD-FIX, fail-open)

§5 fixes `RobotsConfig.from_yaml`, but `selfy.Maxim.__init__` wraps `load_robots_config` / `resolve_connection_config` / `resolve_robot_entry` in `except Exception as e: self.log.debug("robots.yaml connection config not loaded: %s", e)` ("best-effort; never block startup"). The new typed error would be caught right there, and the connect falls back to `{"robot_name", "media_backend"}`, which is unpinned and connects by mDNS name.

`embodied_runtime/agentic_runtime.py::_resolve_body_wiring` has the same shape and falls back to bodiless. This is another site of the §5 pattern, so under "Filing is not fixing" it is folded into PR 0. A missing file stays benign. A malformed or newer-format file raises. The body-wiring site moves to `resolve_robot_entry` (already listed in §5), and its swallow is narrowed.

## 9. The registry reuse branch skips the pin (SHOULD-FIX, fail-open)

In `RobotRegistry.connect_robot_with_status`, if `robot_id` is already registered and live, the method returns the existing controller ("Robot already connected") without calling `connect()`. In-process, `maxim.connect("reachy_mini")` (unpinned) followed by `maxim.connect(device="reachy")` under a colliding registration key, or `maxim.run(robot="reachy_mini")` after either, returns a controller whose identity was never checked against the request. A device connect therefore uses a registration key derived from the pinned identity (`reachy_mini:<hardware_id>`, §2), and the reuse branch compares the requested `expected_hardware_id` with the registered controller's verified one: mismatch or unverified → typed refusal.

## 10. Type-valued paths bypass the pin (SHOULD-FIX, fail-open)

`api.py::connect(robot_type, config=…)` and `api.py::_connect_robot_for_run` (used by `run(robot=)`) never read robots.yaml. With no `host` they dial the mDNS name derived from the default `robot_name`. A user who registered and pinned a device still gets unpinned connects from `maxim.run(robot="reachy_mini")`. The pin lives in the controller but is enforced only when the caller supplies it.

**Fix (pick one, owner-visible):** (a) state in §3b and `stable_api.md` that the type-valued paths are unpinned by design and print one WARNING when an unpinned connect lands on a daemon whose `hardware_id` is pinned under a registered entry, since the controller reads the status anyway. (b) Make the controller look the daemon's id up in the device registry on every connect and refuse an address that a registered entry pins to a different id. I recommend (a) plus the WARNING. (b) couples `hardware/` to the device state on every connect.

## 11. The stop handle (SHOULD-FIX, n/a)

`utils/cancellation.py::request_stop`'s docstring says "Only a process owner makes it (the console lifespan, its drain, the serve signal path; owner decision 2026-10-30)". It is sticky: `register_stop_event` sets events born after a request. A library stop handle therefore cannot call it without stopping everything in the embedder. With D9 plan-only (#1), v1 owes no handle. For the later seam, follow `embodied_device.md`'s bounded-stop row: the run owns a `threading.Event`, as `api.py::run` does (`stop_event` → `_tear_down_run`), and the live loop's `_live_stop_event` and agentic stop event key off it. The handle is `handle.stop(deadline=)`. Teardown fences motion through the existing live teardown (`stop_motion` latch, MotorQueue close), then `goto_sleep` with the `api.py::_sleep_run_robot` retry, then disconnect, which re-acquires media (#1704). No signal handler. Remove the line "Ctrl+C in a script" from §3b: it works only when the run is on the main thread.

## 12. `register_tool` / `on` and the CLI runtime: verified, they do not reach it (SHOULD-FIX, n/a)

`api.py::_inject_registered_tools` is called only from `api.py::run` and `simulation/orchestrator.py`. `api.py::_bridge_event_subscriptions` is called only from `api.py::_build_run_agent` and `simulation/orchestrator.py`. The robot runtime builds its own registry (`agentic_runtime.py`, the `build_tool_registry(maxim=_maxim_for_tools, …)` call) and takes `agent._bus` without bridging either.

In a CLI process no user code can call `register_tool`/`on` anyway, so for `maxim embody` (CLI) the point is moot. It matters only for an in-process embody, which #1 defers. **Fix:** drop "embody() must honour both, exactly as run() does" from v1 and record it as a `MaximHandle.embody` acceptance criterion. The injection then lands at the registry and bus build in `agentic_runtime`, under the same operational-mode gating as the built-in tools **(unverified: whether injected tools pass the mode gate today)**.

## 13. Discovery and the offline latch (SHOULD-FIX, fail-open)

`reachy_mini/utils/discovery.py::find_robots` on darwin calls `_find_robots_dnssd`, which runs `subprocess.Popen(["dns-sd", "-B", …])` plus resolves. That I/O runs in a child process, outside any in-process `sendto`/`getaddrinfo` refusal the atonia V8 latch installs. `_filter_alive` then `socket.create_connection`s to every advertised address, which are attacker-chosen under §6. **Fix:** `maxim.devices.discover` checks the offline/advisory latch itself and raises `DiscoveryUnavailable` *before* calling `find_robots`, rather than relying on the latch to trip inside it. Add a test that `find_robots` is not called under the latch. Also note that importing `reachy_mini.utils.discovery` imports the package `__init__` (`ReachyMini`, `ReachyMiniApp`), which is heavy but does no actuation (verified: no construction at import).

## 14. `check` / preflight is REST-only, but tunnels are unspecified (NIT, fail-closed)

Verified: `doctor/checks.py::_check_one_robot`, `utils/reachy_diagnostics.py` and the controller's pre-connect `_daemon_status` are TCP and REST only. Nothing constructs `ReachyMini(`, so there is no media release or wake. §4's layers keep that property. Two gaps:

- A `tunnel: true` / `localhost_only` entry cannot be probed without the SSH tunnel. Specify that `check` either reports "behind a tunnel: not probed" or opens the tunnel through a helper factored out of `_start_ssh_tunnel`/`_stop_ssh_tunnel` (in a `finally`, as a registered child). It must never call `controller.connect()` to borrow the tunnel.
- §3a's media warning should reuse `ReachyMiniController._warn_if_media_already_released`'s text and fix, so there is one message and not two.

## 15. Exit code for a typed refusal from the loop (NIT, fail-closed)

The new connect error raised inside `Maxim(...)` reaches `cli.py::_run_stopped_on`. That function re-raises only `BackendError`, `LeaderRefusal` and `SpeechOutputUnavailable` to `main` (exit 2 with the fix hint). Everything else is logged as "Maxim stopped" and exits 1 like a crash. Add the device error family (or `MaximConnectionError`/`ConfigurationError`) to the re-raise set, so that `maxim embody`'s identity refusal exits 2 with its hint, matching the preflight's refusal.
