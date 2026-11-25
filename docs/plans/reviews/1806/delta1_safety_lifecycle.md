# #1806 device registry: hardware-safety & runtime-lifecycle review, delta round 1

Reviewer: safety & lifecycle lens, delta round 1, read-only, 2026-11-25, v2 @ f0369379 (code at origin/main b7380ee7 in the plan worktree; reachy-mini 1.8.3 from `.venv`). Fresh reader; round 1 was a different reviewer. No robot, LAN host, sim or runtime was contacted. Scope: the dispositions of round-1 S1–S15, the specific checks the round asked for, and defects introduced by v2's new mechanisms.

**Verdict: SHOULD-FIX. No DO-NOT-BUILD. PR 0 can be built as v2 specifies. Fold N3 and N2 into §4.1/§4.3 before PR 1, N7 before PR 2, and N1, N5 and N6 before PR 2b. Five of the new findings fail open, so the stop rule is not met (see the end).**

## Disposition of round-1 findings

| S# | Disposition | Note |
|---|---|---|
| S1 | PARTIAL (#1808) | #1808 lists the execve, the process-global setup, the termios/stdin, the stop and `_RUN_LOCK`. Missing: `selfy.Maxim.__init__`'s `connect_robot(..., set_primary=True)` overwriting an embedder's registry primary; OpenCV `prepare_display()` on the main thread; and, in the **shape**, the "share `_RUN_LOCK`, one SDK connection per device" condition (the inventory only says the lock is not shared) |
| S2 | PARTIAL | The `operational_mode` key is refused and the grant is read from what was typed. But `launch.mode` is not limited to run modes, so an operational name stored as a mode is enforced as that capability (N1). The plan block does not print where the capability comes from |
| S3 | PARTIAL | Step 1 sits before `ReachyMini(`. The typed raise replaces `return False`. The cleanup order (re-acquire, drop the handle, stop the tunnel) matches `controller.py::ReachyMiniController.connect`'s `except` branch and `disconnect`'s "before the tunnel stops" rule. Not yet specified: (a) step 3 must sit after the `_media_released_by_connect = self._connect_released_media(status)` line and before `_assert_auto_yaw_off`, `_clear_latch_after_connect`, the stream wrappers and `CONNECTED`; (b) the broad `except Exception` has to clean up and then re-raise the typed family; (c) the red-gate test and deletion probe through `tests/unit/_reachy_fakes.py`. Step-3 failures also bring a new problem (N2) |
| S4 | FOLDED, with a new defect | The registry refuses classes that do not declare `supports_identity_pin`. A class attribute is inherited, though, so a subclass that overrides `connect` keeps the claim without enforcing it (N3). The note also does not say how the pin reaches the controller, since it bypasses the constructor filter |
| S5 | PARTIAL | A null id never matches, `add` records a null id as UNPINNED, and the plan block shows the label. Round 1's owner question (does an unpinned `embody` need a confirm?) was dropped without a decision. Either list it or record the decision |
| S6 | FOLDED | The host is resolved once to an IPv4 literal. The same literal is dialled with `connection_mode="network"`, and the cache stores that literal |
| S7 | PARTIAL | `reexec_with_mode` carries `--device`. On the no-name path (`default:`/primary), `args.device` is None, so the child resolves `default:` again. A `maxim connect other` run in between moves the child to a *different* verified robot. Always carry `--device <resolved entry id>`, pass `robot_id=` to `Maxim(...)`, and add an argv round-trip test |
| S8 | FOLDED | Both swallows are named. When `_resolve_body_wiring` is narrowed, its `resolve_body_ref` fallback (declared but unresolvable body → bodiless) must stay, and `RobotsConfigError` must propagate |
| S9 | PARTIAL | The reuse compare refuses only on a "mismatch". It must also refuse when the live controller is **unverified** (connected earlier by the unpinned `maxim.connect`). The refusal must be a typed raise; the branch's existing `return None` would become a headless run. The compare must run before the branch's `self.set_primary(robot_id)` |
| S10 | FOLDED | `maxim.connect`/`run(robot=)` are documented and logged as unpinned |
| S11 | FOLDED (#1808) | The run-owned stop, `MaximHandle.stop(deadline=)` and the no-signal-handler rule are in the shape. The teardown order (`stop_motion` latch → MotorQueue close → `goto_sleep` retry → disconnect, which re-acquires media) is left to `bounded_stop.md`, which is acceptable |
| S12 | PARTIAL (#1808) | Only the inventory carries it ("whether injected tools pass the operational-mode gate is unverified"). It should be a shape and acceptance item: injected tools are dispatched under `Executor._mode_denial`, and a test proves it |
| S13 | FOLDED | The latch is checked before `find_robots`, and the census and lint pins are named |
| S14 | PARTIAL | "Verified at connect" defers the tunnel case to `connect`, but `connect`'s pinned sequence applies only the identity and sim refusals. A tunnel entry never gets the era/minor-version or daemon-state refusals (N5). Reusing the media-warning text is not mentioned |
| S15 | FOLDED | `_run_stopped_on` re-raises the device family as a typed exit. I checked against `cli_signals.py::exit_code`: a non-sim mode keeps a real failure code (2) even under a signal, so D13's exit does not interact badly with the bounded-stop codes |

## New findings (introduced by v2 or exposed by its mechanisms)

| # | Finding | Class | fail-open/closed | Covered class? |
|---|---|---|---|---|
| N1 | `launch.mode` is not limited to the CLI's run-mode `choices`. A stored `mode: singularity` (or `passive`) would reach `Executor.effective_operational_mode` through the mode source and be enforced as that capability, which is the persisted-grant escalation refusing `operational_mode` was meant to close | SHOULD-FIX | fail-open | yes (S2) |
| N2 | A step-3 failure (REST and WS reached different daemons) feeds #1704's ownership rule the *other* daemon's REST pre-state, so the wrong daemon's media is either left released or stolen back from its real owner. `automatic_body_yaw=False` stays set on the wrong daemon, and its media manager is not closed by `_drop_handle` | SHOULD-FIX | fail-open (wrong robot's resources; not motion) | yes (S3) |
| N3 | `supports_identity_pin = True` is a class attribute, so it is inherited. A plugin or subclass that overrides `connect()` keeps the claim and enforces nothing, which is a silent no-op again | SHOULD-FIX | fail-open | yes (S4) |
| N4 | `connect()` now raises, but `RobotController.reconnect` and `ConnectionMixin._soft_reconnect` have no handler. A transient `DeviceUnreachable` aborts the retry loop, the exception escapes into `_note_connection_failure`'s callers, `_degrade_capabilities` is skipped, and nothing says what happens mid-run when a reconnect finds a different identity | SHOULD-FIX | fail-closed for motion (the handle is dropped) | **no** (mid-run lifecycle) |
| N5 | Tunnel entries: "verified at connect" leaves out the preflight's era/minor and `state != running` refusals, because `connect` applies only the identity and sim checks. The plan block cannot show a verified address or id, and `--dry-run` proves nothing | SHOULD-FIX | fail-open (a mismatched daemon reaches motion) | yes (S14) |
| N6 | `_validate_operational_grant` requires a literal `--mode` in raw argv. `maxim embody reachy --operational-mode singularity` is therefore refused with exit 2, and the obvious "fix" (synthesising `--mode`) would launder the grant check | NIT | fail-closed | yes (S2) |
| N7 | Nothing says where `add` and `repin` read the new pin from. If they read it from the mDNS `unit_id`/TXT, an advertisement gets pinned. Sim flags at add/repin time are not mentioned | SHOULD-FIX | fail-open if built that way | yes (A4/S5) |
| N8 | D13 is keyed on "named device", but the pinned controller raises however the entry was selected. "Unnamed legacy keeps headless" holds only for unpinned entries, and whether `--robot`/`--robot-name` count as "named" is not stated | NIT | fail-closed | no (spec clarity) |
| N9 | `DaemonStatus.hardware_id` defaults to None on a daemon whose WS status predates the field. Every pinned connect is then refused as an "identity mismatch", and the message is misleading | NIT | fail-closed | yes (S5) |
| N10 | `_wait_local_port` accepts any listener on 127.0.0.1:8000 and never checks `_tunnel_proc.poll()`, so "the tunnel this controller started" is not established in code. The pin is what catches a foreign listener | NIT | fail-closed | yes (S14) |
| N11 | Strict `launch:` validation of unknown keys raises a whole-file `RobotsConfigError`, which makes *every* robot run exit non-zero, not only `embody`. It bricks runs on version skew (a newer build adding a `launch:` key) | NIT | fail-closed | no |

---

## N1. `launch.mode` can carry an operational name (SHOULD-FIX, fail-open)

v2 §3a refuses the `operational_mode` key in `launch:` and calls `launch:` "validated", but it never says what `mode` is validated against. `cli_parser.py` restricts `--mode` with argparse `choices=["live", "train", "reflection", "sleep", "agentic", "exploration"]`. A stored value that `embody` passes "as parsed arguments" never goes through argparse.

The run-mode name is itself a capability input. `runtime/executor.py::Executor.effective_operational_mode` returns the mode source's raw name when there is no launch grant. `modes/definitions.py::get_mode` resolves `"singularity"`, `"passive"` and `"active"` directly (the `OPERATIONAL_MODES` lookup comes before `_LEGACY_NAME_MAP`). `cli_utils.py::reexec_with_mode` also treats an operational name in the mode slot as a grant (`if mode in OPERATIONAL_MODES: operational = mode`). So `launch: {mode: singularity}`, which any file-writing tool can write (S2's threat), becomes code-execution capability with nothing typed at launch.

**Fix:** the `launch:` loader validates `mode` against the same `choices` tuple argparse uses. Import one constant and do not copy the list. Any name that `get_mode` resolves to a definition raising capability beyond the mode's implied `active` is refused, and `llm`/`interactive` are typed. Add a test: `launch: {mode: singularity}` raises `RobotsConfigError`, and a parametrised test over `OPERATIONAL_MODES` keys.

## N2. A step-3 failure mis-attributes media ownership and leaves state on the wrong daemon (SHOULD-FIX, fail-open)

Checked against the SDK: `reachy_mini.py::ReachyMini.__init__` sends commands before v2's step 3 can run.
- `self.set_automatic_body_yaw(automatic_body_yaw)` sends `SetAutomaticBodyYawCmd`, which is daemon-wide.
- `_configure_mediamanager` calls `self.client.get_status()`, and under `no_media` calls `release_media()`, also daemon-wide. Under the default backend on `network` mode it builds a WebRTC media manager against that daemon.

v2 §4.3 step 1 is right that a *REST* mismatch refuses before any of this. Step 3 exists for the case where the REST probe and the WS session reached **different** daemons, and in that case:

- `controller.py::ReachyMiniController._connect_released_media(status)` decides ownership from `status.get("media_released")`, the **REST** pre-state. That pre-state belongs to the *pinned* daemon, not the one the session released. If the pinned daemon reported `false`, the controller claims ownership and re-acquires the wrong daemon's media, even when another client released it first (it steals the release back). If the pinned daemon reported `true`, the controller disclaims ownership and the wrong daemon's camera stays released. That is the #1704 leak.
- `automatic_body_yaw=False` stays set on the wrong daemon. It has no getter, so it cannot be restored. Document it as a residual.
- `_drop_handle` deliberately does not close `media_manager` ("the media lifecycle owner closes it"). On a refused connect no lifecycle owner ever received it, so the WebRTC/GStreamer pipeline to the wrong daemon lives until garbage collection.

**Fix (§4.3 step 4):** on a step-3 failure, ownership comes from the SDK handle alone. `self._mini.media_released is True` means this handle sent the release, because `_configure_mediamanager` releases unconditionally under `no_media` unless the daemon reports `no_media`. Re-acquire on that basis, and log a WARNING naming the session's host. Never consult the REST pre-state from the other daemon. Close the handle's `media_manager` in the refusal path, not in `_drop_handle` (keep #1435's ordering for the normal path). Record the auto-yaw write as a residual. Red-gate test: a fake whose REST status and WS `get_status()` report different ids, and that asserts the re-acquire decision and the `media_manager.close()`.

## N3. `supports_identity_pin` is inherited (SHOULD-FIX, fail-open)

v2 §4.1: "A pinned entry on a controller class that does not declare `supports_identity_pin = True` is refused." A class attribute passes down to every subclass. A plugin that subclasses `ReachyMiniController` and overrides `connect()`, a mock, or a future refactor that moves the pin sequence keeps `True` and enforces nothing. That is the silent-no-op class CLAUDE.md says to push into types. v2 also does not say how the registry hands the pin over, given that it bypasses `RobotRegistry.connect_robot_with_status`'s constructor-signature filter.

**Fix:** the registry enforces a **post-condition** and does not trust a flag. After `controller.connect()` returns True for a pinned request, the registry reads a typed accessor, `controller.verified_identity() -> DevicePin` (the abstract default returns `DevicePin.UNPINNED`). The registry then requires it to equal the requested pin, or it disconnects and raises `DeviceIdentityMismatch`. The pin goes in through one typed call before `connect` (for example `controller.require_identity(pin)`, abstract on `RobotController` and raising `NotImplementedError` by default). The reuse branch (S9) uses the same accessor. This keeps the refusal of classes without pin support, and an inherited flag can no longer silence it.

## N4. A raising `connect()` reaches the reconnect path unhandled (SHOULD-FIX, fail-closed for motion; uncovered class)

`hardware/controller.py::RobotController.reconnect` loops `self.disconnect(); if self.connect(timeout=timeout): return True` with no `try`. `embodied_runtime/connection.py::ConnectionMixin._soft_reconnect` calls it inside a `try/finally` that only releases the lock, and it is reached from `_note_connection_failure` in the capture loops. Once `connect()` raises for pinned entries:
- A transient `DeviceUnreachable` (a Wi-Fi blip) ends the retry loop on attempt 1 instead of retrying, and it escapes into whichever capture loop noted the failure.
- `_degrade_capabilities()` is skipped, so the runtime keeps advertising the robot's capabilities while it has no handle.
- If the DHCP lease moved the cached IP to another robot mid-run (the motivating case), nothing says whether the run stops. Each 20 s cooldown retries; when the divergence is the step-3 kind, every retry repeats N2's writes on the wrong daemon.

Motion stays fail-closed, because the failed connect drops the handle. But the run's lifecycle is unspecified, and this is a new failure path that v2 creates.

**Fix (§4.3):** spell out the reconnect semantics. `reconnect` catches `DeviceUnreachable` and keeps today's retry and `False` behaviour, so the soft path degrades capabilities. `DeviceIdentityMismatch` on a reconnect latches motion (`stop_motion`), degrades capabilities, logs an ERROR with the claimed and expected ids, and stops the run through the run's own stop event. It never retries onto the wrong body. A `KeyboardInterrupt` during step 3's `get_status(wait=True)` (a new wait of up to 5 s) bypasses the controller's `except Exception`. The registry's `except BaseException` → `disconnect()` covers it only if `maxim.devices.connect` goes through the registry, so state that it does. Test: a reconnect that sees a different id ends in the latched, degraded, stopped state, not a capture-loop exception.

## N5. Tunnel entries skip half the preflight (SHOULD-FIX, fail-open)

§3a step 2 refuses on an era/minor mismatch, a daemon not `running`, sim or mockup flags, and identity. For a `tunnel:` entry it reports "verified at connect" instead. `connect`'s pinned sequence (§4.3 steps 1 and 3) checks only identity and the sim flags, and the controller's own era gate checks only the installed SDK client (`_installed_sdk_version() < (1, 5)`). So a tunnel entry with a minor-mismatched or not-`running` daemon reaches motion. The SDK only warns (`_warn_if_daemon_version_mismatch`). The plan block also promises a "verified IPv4 address, `hardware_id`" that it cannot show for a tunnel entry.

**Fix:** make the refusal set one predicate over `probe_daemon`'s `ProbeOutcome` (§5). Both the preflight and the controller's step-1 pre-filter apply it for pinned entries, so the tunnel case gets the same refusals through its own tunnel. For tunnel entries the plan block prints "behind a tunnel: verified at connect", and `--dry-run` says that it did not verify.

## N6. The grant check needs an explicit `--mode` (NIT, fail-closed)

`cli.py::_validate_operational_grant` refuses a grant unless `_explicit_mode_flag(raw_argv)` is true. `embody` defaults the run mode to `exploration` or `launch.mode`, so `maxim embody reachy --operational-mode singularity` exits 2 with "needs an explicit --mode". The rule exists so a grant cannot fall to the menu or server mode, which `embody` never does. **Fix:** the grant check accepts the `embody` subcommand as an explicit robot-run request, and it still reads `--operational-mode` only from raw argv. State outright that `embody` never synthesises `--mode`.

## N7. Where `add`/`repin` read the pin from is unspecified (SHOULD-FIX, fail-open if built that way)

`maxim.devices.add("reachy", found[0])` takes a `DiscoveredDevice`, whose `unit_id` is the unauthenticated advertisement (§4.2's own threat). `repin` shows "old→new" but does not say where "new" comes from. **Fix (§3/§4):** the pin written by `add` and `repin` is always read from REST `/api/daemon/status` at the IPv4 literal the entry will dial, with proxies disabled. The same predicate as N5 applies: sim and mockup flags off, `hardware_id` not null, daemon `running`. The advertised `unit_id` is display-only. If it differs from the REST id, `add` shows both and refuses without a confirm. `repin` refuses when the new id matches a different registered entry's pin. Neither verb opens `ReachyMini(` (that already holds by construction).

## N8. D13's trigger should be the pin status, not the selection path (NIT, fail-closed)

The pinned controller raises however the entry was chosen (`--device`, `--robot-name`, `default:`, primary). So for pinned entries "an unnamed legacy launch keeps today's headless fallback" is not true. **Fix:** a three-row table in §3a. Pinned entry: any refusal raises and exits 2, never headless. Unpinned entry selected by `--device`/`embody`: a connect failure exits 2. Unpinned entry reached by `--robot`/`--robot-name`/no name: today's headless fallback. Exit codes are unaffected by `cli_signals.py::exit_code` (non-sim keeps 2).

## N9–N11 (NITs, fail-closed)

- **N9:** `io/protocol.py::DaemonStatus.hardware_id` is `Optional[str] = None`. A daemon whose WS status predates the field fails step 3 on every connect. Use a distinct error or fix hint ("daemon does not report hardware_id on the session; upgrade to ≥ X") so the message is not "identity mismatch".
- **N10:** `controller.py::ReachyMiniController._start_ssh_tunnel` → `_wait_local_port` returns True for any 127.0.0.1:8000 listener, even after `ssh` has exited under `ExitOnForwardFailure`. Add `self._tunnel_proc.poll() is None` to the wait, and have §4.3 say the pin, not the tunnel, is what covers a foreign listener.
- **N11:** refusing `operational_mode` (correct) and rejecting unknown `launch:` keys (too broad) both raise a whole-file `RobotsConfigError` on load, which stops *every* robot run. Keep `operational_mode` (and N1's mode values) as whole-file refusals. An unknown `launch:` key should be a WARNING scoped to the entry, so a newer build's key does not brick an older build's runs.

## Stop-rule outcome

**Not met.** N1, N2, N3, N5 and N7 fail open. All of them are in classes round 1 already covered (S2 persisted grant, S3 wrong-daemon mutation, S4 silent pin drop, S14 tunnel preflight, A4 advertisement trust), and each is a spec-level fold, not a redesign. N4 fails closed but is in a class round 1 did not cover (mid-run reconnect lifecycle). N8 and N11 fail closed but are also outside the covered classes. Another delta round is owed on the folds to §3a, §4.1 and §4.3. The round can be scoped to those sections.
