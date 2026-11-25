# Device registry (#1806): approach note, DRAFT v3

Status: DRAFT v3, 2026-11-25. Mirrors issue #1806.

**Review history.**
- **Round 1, on DRAFT v1.** Four lenses: [adversarial](reviews/1806/adversarial.md), [safety & lifecycle](reviews/1806/safety_lifecycle.md), [architecture](reviews/1806/architecture.md) and [wire](reviews/1806/wire.md). All four returned DO-NOT-BUILD on one root cause: the identity pin rode a free-form config key and a separate REST probe.
- **Delta round 1, on DRAFT v2 at f0369379.** Fresh readers: [adversarial](reviews/1806/delta1_adversarial.md), [safety](reviews/1806/delta1_safety_lifecycle.md), [architecture](reviews/1806/delta1_architecture.md) and [wire](reviews/1806/delta1_wire.md). **None returned DO-NOT-BUILD.** Every lens found v2 had closed round 1's DO-NOT-BUILDs. Almost every new finding fell in an already-covered class (converging), but several fail open, so the stop rule was not met.
- **v3 folds every delta-round-1 finding** (§11). A **delta round 2 is owed**, scoped to the fail-open folds listed in §11.

Owner decisions: D1–D13 were taken on 2026-11-25. The re-asks raised in the delta round were taken the same day: D2″ rides `primary:`, and D12′ leaves `run_device` in `console/`. D14 was taken the same day: deny agent file-tool writes to the whole control-plane set in every mode, in PR 0 (§9).

**What:** give Maxim one way to find, name, remember and reliably reconnect to the devices it senses and acts through. `maxim connect reachy` and `maxim embody reachy` replace hand-editing `~/.maxim/robots.yaml`, chasing a DHCP-moved IP, and knowing about the `.local` and `media_backend: no_media` quirks. Robots come first. The model leaves room for other kinds (a camera, a mic, a device attached to another node) without building that abstraction early. **Every verb is a library function first**, and the CLI is a thin shell over it (§3b). Making the runtime itself a library is its own issue, #1808 (§3c).

This is a **gate**: discovery and identity pinning form a security and safety boundary. The design pass, the delta rounds and the owner decisions all come before code.

## 1. The model: nodes, devices, clients

| | **Node** | **Device** | **Client** |
|---|---|---|---|
| What | A process running Maxim (leader, peer, solo) | An endpoint Maxim senses and acts through, opened by a driver | Something that dials *in* to a node |
| Examples | big-mac-mini, the RTX leader, a Pi on the robot | the Reachy daemon, the simulated robot; later camera, mic, speaker | Console browser, Pulse, the planned phone |
| Store today | `~/.config/maxim/mesh.yml` (SHAPE-FROZEN), `peer.yml`/`config.json` lanes | `<data_home>/robots.yaml` (hand-edited, no writer) | `console_token`, spoken-code pairing |
| Trust | Mesh key (bearer) | Per driver; Reachy has **no auth** (the LAN is the boundary) | Console token / pairing; per-device tokens planned |
| Direction | Dial out | Dial out | Dial in |

- **A device is attached to exactly one node, the one that opens it.** This is load-bearing: the single clamped motion dispatch point, media-release ownership (#1704) and the stop latch are all per controller and per process. In v1, every device is attached locally. `attached_via: <node>` comes later, with `embodied_device.md`.
- **One name namespace, separate stores.** No store merges. Names follow one validator:
  - allowed characters are `[a-z0-9][a-z0-9_-]{0,62}`, casefolded, with no leading dash;
  - a name that a mesh node also claims is refused, checked at read time as well as write time;
  - a solo box, or a malformed `mesh.yml`, gets a warning instead.

  CLI `--device` resolves through `maxim.devices`, so the read-time check is on the CLI path (architecture R15, N10).
- **Clients are out of scope.** Pairing paths are unchanged.
- **This is not node discovery.** Leader and peer discovery and pairing stay deferred to `embodied_device.md` §2. This is also not POG-2c.

**Only one kind exists today, so v1 builds no kind abstraction.** Entries carry a reserved `kind: robot`, and the router has a single branch. A `DEVICE_KINDS` table, or an entry-point group, arrives with the second real kind (architecture R7).

## 2. Front gate: what this rides on

| Need | Existing infrastructure | Verdict |
|---|---|---|
| Name → robot config | `hardware/config.py`: `RobotsConfig`, `resolve_robot_entry`, `resolve_connection_config` | **Ride.** robots.yaml stays the robot store. Entry-level fields become typed (§4.1) |
| Default device | robots.yaml `primary:` | **Ride it, as the only default marker.** `maxim connect` / `devices.use` moves `primary: true` to the chosen entry. There is no `default:` key: two markers would split old readers from new ones, and the legacy `--robot` path from `connect` (D2″; architecture N4, wire N3) |
| Find a Reachy | SDK `reachy_mini.utils.discovery.find_robots()` | **Ride**, for `discover` and `relocate` only. It is never a resolution step (§4.2) |
| Identity | `hardware_id` (lowercase `sha256(...).hexdigest()[:16]`) in REST `/api/daemon/status` **and** in the WS session's `DaemonStatus`, which the daemon pushes at 1 Hz from the same `_status` object that REST serves (verified, delta adversarial) | **Ride.** The check that counts is on the session (§4.3) |
| Reachability probe | `doctor::_check_one_robot`, `utils/reachy_diagnostics.py`, `scripts/check_reachy_connection.py`, controller pre-checks | **Consolidate** into `hardware/reachy/probe.py::probe_daemon` → a typed `ProbeOutcome`, plus one shared `refusal(outcome, entry)` used by both the preflight and the controller's step 1 (§4.3; safety N5). `scripts/check_reachy_connection.py` stays standalone, because it is documented as working without an install; it gets a deprecation pointer (wire N9) |
| Mutable last-good address | none | **New:** `<data_home>/util/device_state.json`. It is keyed by identity, holds pinned entries only, and is exempt from layer-2 role scoping (an address is not role-dependent). It does not reuse `probe_cache`, which is URL-keyed TTL health data |
| Writing robots.yaml | none | **New:** `write_robots_config` is co-located in `hardware/config.py`, next to its reader (the `mesh_config.py` pattern; §7) |

## 3. Surfaces

### 3.1 CLI

```
maxim discover                      # read-only: LAN candidates (sanitised hints) + registered entries + mesh node names
maxim device add <name> [<host>]    # pick a discovered candidate or give a host/IP; the pin is read from REST at the dialled literal (§4.2)
maxim device list | show <name>
maxim device rename <old> <new> | forget <name>
maxim device relocate <name>        # operator-confirmed address update from discovery (§4.2)
maxim device repin <name> [<host>]  # operator-confirmed identity change, e.g. after an audio-board swap (§4.2)
maxim device check [<name>]         # layered diagnosis, never actuates
maxim connect <name>                # resolve + verify + refresh the cache + make it primary
maxim --device <name> ...           # strict: an exact entry name, or a refusal          [experimental]
maxim embody <name> [--dry-run] [--unpinned]   # preflight + launch embodied (§3a)    [experimental]
```

- **Legacy flags.**
  - `--robot` is an explicit alias of `--robot-name`, with today's behaviour (D3′). Defining it explicitly makes `--rob`/`--robo` ambiguous abbreviations, which is documented (wire N7).
  - `--robot-name`'s argparse default becomes `None`. `MAXIM_ROBOT_NAME` is read at resolution time, not by argparse, so a bare `maxim` with that variable set no longer enters the robot runtime (wire F4).
- **The quick-start gate reads parsed, explicit flags.** It checks `args.device`, an explicit `--robot`/`--robot-name`, `--mode` and the `embody` subcommand. It never reads a resolved primary or environment variables. The gate's test calls the real `cli.py` gate.
- **Selection precedence** applies only once a robot run is requested:
  1. `--device <name>`
  2. `MAXIM_DEVICE`
  3. legacy `--robot`/`--robot-name`, which keep today's matching and set the daemon name
  4. a single entry, or the `primary:` entry

  Steps 1 and 2 are **explicit names**: they must match an entry exactly, or the run is refused. Only step 4 falls back to `primary:`. `--device` together with a conflicting `--robot-name` is refused (wire N5). The plan block and the startup log name which source chose the device (adversarial N12).
- **`robot_id` and `robot_name` are separated end to end.** The CLI passes `Maxim(robot_id=<entry id>, robot_name=<daemon name>)` (architecture R2).
- **Re-exec.** `reexec_with_mode` re-execs with `--device <resolved entry id>` alone, never `--robot-name`. The child re-resolves that exact entry and re-verifies it, so a `maxim connect` run in the meantime cannot switch the robot (safety S7, wire N5). The test is "a run-mode switch keeps the device".
- **Help text and docs.** `maxim connect` (devices) is distinct from `maxim peer connect`, and the help says so.

### 3a. `maxim embody <name>`

1. **Resolve and verify** (§4). An unknown name is refused, and `primary` does not change.
2. **Preflight.** These are REST/TCP-only checks through `probe_daemon` + `refusal`: no `ReachyMini(` and no tunnel.
   - **Refuse** on: an identity mismatch, an unreachable daemon, an era or minor-version mismatch, a daemon whose state is not `running`, or a sim/mockup daemon for a hardware-pinned entry.
   - **Warn** on `no_media`, or on `media_released` when the run mode needs perception.
   - For a `tunnel:` entry, the same `refusal` runs inside the controller at connect (§4.3 step 1). The plan block says "checked at connect (tunnel)" rather than "verified" (safety N5).
3. **Plan block.** It shows: the device and the source that selected it, the verified IPv4, `hardware_id` (or **UNPINNED**), the body ref, the run mode, the operational mode and its source (typed flag, or the run-mode default), whether the head will move, the LLM and whether it is a cloud model, and the media backend. `--dry-run` stops here.
4. **Launch** through the existing robot path, with the **verified address and pin carried forward as typed arguments**; nothing is re-resolved.

**Unpinned entries need `--unpinned`.** That applies to `embody` and to `--device`. Without it, an entry with no `hardware_id:` is refused with an explanation (it may be a non-Linux daemon with a null id, or a legacy hand-written entry). Legacy `--robot`/`--robot-name` launches are unchanged (adversarial N3, safety S5).

**Exit on refusal, defined by pin status (D13′).** A refused or failed connect on a **pinned** entry exits non-zero, however the entry was selected. An unpinned entry selected with `--device`/`embody --unpinned` also exits non-zero. Only a legacy launch with no device flag keeps today's headless fallback (safety N8, wire N10, adversarial N7). `test_cli_exit_code_1468.py` gains rows for this.

**Defaults.**
- The run mode is `exploration`.
- `launch: {mode, llm, interactive}` is an entry-level sibling of `config:`.
- **`launch.mode` must be one of the run-mode choices** (`exploration`, `live`, `sleep`, `train`, `reflection`, `agentic`). Operational names such as `passive`, `active` and `singularity` are refused for the entry (safety N1).
- **`launch:` never holds `operational_mode`**, and the loader refuses that key.
- Other unknown keys inside `launch:` only warn, scoped to the entry (safety N11).
- **The capability grant is the typed flag.** `embody` parses `--operational-mode` itself and hands a typed grant to the launch. It does not fake a `--mode` argv to satisfy `_validate_operational_grant`, because that would launder the check (safety N6).

**Safety.** `embody` is the only verb here that can move motors. It adds no motion code, and it never targets an unregistered or unverified device.

### 3b. The Python API: every verb is a library function first

The CLI is a thin shell. Each verb is a function in the **`maxim.devices`** package (`src/maxim/devices/`, a directory so that `LAYER_RULES` actually checks it; architecture N1). It is registered in `__init__.py::_API_SUBMODULES`, not `_API_VERBS`, and the README verb count does not change (architecture N6).

```python
import maxim

found = maxim.devices.discover(timeout=5.0)       # tuple[DiscoveredDevice, ...]: sanitised hints, never trusted
entry = maxim.devices.add("reachy", host="10.6.0.63")   # pin read from REST at that literal; a null id → refused unless unpinned=True
maxim.devices.entries(); maxim.devices.get("reachy")
maxim.devices.rename("reachy", "rosie"); maxim.devices.forget("rosie")
maxim.devices.relocate("rosie", confirm=lambda cand: ...)   # operator callback required; there is no silent path
maxim.devices.repin("rosie", host="10.6.0.71", confirm=lambda old, new: ...)
report = maxim.devices.check("rosie")             # DiagnosticReport of CheckResult (the existing maxim.diagnose() types)
dev = maxim.devices.resolve("rosie")              # ResolvedDevice: verified IPv4 + identity; changes nothing
maxim.devices.use("rosie")                        # what `maxim connect` calls: resolve + make primary
ctl = maxim.devices.open("rosie")                 # a live RobotController, connected through the pin (§4.3)
plan = maxim.devices.plan_launch("rosie", mode="exploration")   # LaunchPlan
```

- **Names.** The live opener is `devices.open`, not `devices.connect`. CLI `maxim connect` means "make primary", which is `devices.use`. That avoids four different things called `connect` (architecture N7).
- **The stable APIs do not change.** `maxim.connect(robot_type, config=…)` and `maxim.run(robot=)` stay **unpinned**, and this is documented and logged.
- **Who may write (D10).**
  - Write verbs live in `maxim/devices/_write.py`. Python write verbs are operator writers, which needs an amendment to C2 in `docs/agents/persistence-config.md` §1 and to its writer table (architecture N2).
  - **Enforcement is a check, not a name ban** (wire N4, architecture N2):
    - (a) An AST lint allow-lists the callers of `hardware/config.py::write_robots_config`: `maxim/devices/_write.py` only, next to `MESH_WRITER_MATCHES` in `test.yml`.
    - (b) An AST lint refuses any `devices` write-verb attribute call (`.add/.rename/.forget/.relocate/.repin/.use`) on a `maxim.devices` reference in `src/maxim/` outside `cli.py` and `maxim/devices/`. Today that covers `runtime/`, `tools/`, `agents/` and `embodied_runtime/`; the ban does not name `skills/`, which does not exist.
    - (c) `sandbox_executor.BLOCKED_PYTHON_IMPORTS` gains `maxim.devices`.
  - **Scope of the claim.** An agent's in-process tools cannot call the write verbs. An agent whose BashTool may run `maxim device add` is operator-equivalent, and that is governed by the operational-mode gate (passive refuses `bash`). D14 covers the files themselves.
- **Python confirmations.** `relocate` and `repin` require a `confirm=` callback. On the CLI, they need a TTY, or `--yes` (adversarial N10).
- **v1 has no in-process launch** (D9). It arrives with #1808 as `maxim.devices.embody(...)`.
- **`console/device.py::run_device` stays in `console/`** (D12′, architecture N12). It is a node/host concern, and folding it in would pull console/FastAPI imports into `maxim.devices`. At most there is a lazy re-export.
- **Types.** `DiscoveredDevice`, `DeviceEntry`, `ResolvedDevice`, `DevicePin` and `LaunchPlan` are frozen dataclasses. They live in `hardware/devices_types.py`, so runtime and `embodied_runtime` can import them without importing the write verbs.
  - `DeviceEntry` is built from a `RobotConfig` by one copying constructor, with `config` as a `MappingProxyType`. It is not a "view" over it (architecture N11).
  - All five types take CC3 path (a).
  - Errors are typed and subclass the stable `MaximConnectionError` / `ConfigurationError`: `DeviceNotFound`, `DeviceIdentityMismatch`, `DeviceIdentityDivergence`, `DeviceUnpinned`, `DeviceUnreachable`, `DiscoveryUnavailable`, `RobotsConfigError`. `_run_stopped_on` re-raises them as typed exits.
- **Stability.** The namespace is ⚠️ Experimental in `stable_api.md`. The new module goes into `scripts/run_mypy_public.sh`.

### 3c. Library-first runtime seam (#1808; #1806's `embody` is its first consumer)

The owner's goal is that people can use **the core library or the CLI, without the library depending on the CLI**. The robot runtime is not callable from a library today.

#1808 holds the inventory and the shape. The delta round added these items, which go into #1808:
- `Maxim.__init__`'s `set_primary=True` overwrites an embedder's registry primary.
- OpenCV `prepare_display()` runs on the main thread.
- `data_home()` calls `sys.exit` on the library path (architecture N8).
- The shape shares `_RUN_LOCK` and enforces one SDK connection per device (safety S1).
- Injected tools pass `Executor._mode_denial` (acceptance test; safety S12).
- `adopt(sdk)` must take a pin, not only `launch(plan)` (wire F11).
- Re-point `test_operational_mode_launch.py::test_the_robot_runtime_carries_the_grant` (wire F10).

## 4. Connection, resolution and identity

### 4.1 The pin is a typed field, enforced by the registry

- **Entry-level fields.** robots.yaml entries gain `hardware_id:`, `hostname:` and `launch:`. `RobotConfig.from_dict` parses them into typed fields.
  - The pin key is named `hardware_id:`, not `pin:`, so it is never confused with the SDK's `get_pin()` BLE pairing code (adversarial N13).
  - Its value must be 16 lowercase hex characters.
  - An absent key means `DevicePin.UNPINNED`, a sentinel and never `None`. An empty or malformed value is a `RobotsConfigError`.
- **Strict entries** (wire F1b). An **unknown entry-level key** (for example a typo `hardware_idd:`), or `hardware_id`/`hostname` placed under `config:`, is a `RobotsConfigError`. A typo can therefore never yield a silently unpinned entry.
- **How the pin reaches the controller.** The registry passes it as an explicit keyword, `expected_identity: DevicePin`, outside the constructor-signature filter. A controller class whose constructor has no such parameter is **refused** for a pinned entry.
- **Post-connect check** (safety N3, adversarial N11). After **every** connect, the registry requires `controller.verified_identity() == pin`, a typed accessor that the controller sets only after §4.3 step 3. This replaces trusting an inheritable `supports_identity_pin` flag.
- **The reuse branch** ("already connected") refuses, with a typed error and before `set_primary`, when it finds a mismatch **or a controller that was never verified** (safety S9).
- **An unknown `type:` on a pinned entry** raises `RobotsConfigError` rather than returning `None` before the pin check (wire N10).
- **Load errors propagate** (wire N1, F18; adversarial N6). `from_yaml` / `from_dict` raise `RobotsConfigError` on:
  - a parse error;
  - a top-level list;
  - a non-dict entry, or `config: null`;
  - `find_config_file`'s "data_home() unresolvable" branch;
  - a **relative `MAXIM_DATA_HOME`**.

  The relative-path check covers robots.yaml resolution and the writer only, not `utils/paths.py::data_home()`, which benchmarks call with relative work directories (wire N2). A missing file is fine and means there are no entries. `selfy` stops catching these errors at DEBUG. `agentic_runtime._resolve_body_wiring` stops reloading robots.yaml: it **receives the resolved entry** and keeps its body-ref fallback (wire N3, safety S8).
- **`_format_version`.**
  - A root `_format_version: "1.0"` is written by the writer.
  - An explicit refuse-newer check modelled on `config_writer._refuse_if_newer` runs in the loader and the writer. `check_format_version` itself never refuses.
  - A present non-string value is an error: an unquoted `2.0` is a float, and it must not be read as legacy.
  - An absent key is accepted as hand-written legacy, without the "re-save" nag (architecture N3, wire N8).
  - `test_persistence_compat` gains a robots.yaml row.

### 4.2 Resolution never trusts an advertisement

`host:` is the operator-pinned address: an IPv4 literal, or a name. `hostname:` is the name used to re-resolve when `host:` is absent (adversarial N9).

**Candidates for a pinned entry,** in order:
1. `host:`, or `hostname:`, **resolved once** to IPv4;
2. the cached last-good address for that identity.

That is all. Discovery is never a resolution step. `MAXIM_REACHY_HOST` is read only by the `maxim-diagnostics` tool and never by the registry (adversarial A15). On failure, the error names every candidate and why it failed.

**`relocate`** browses the LAN and shows each candidate's claimed identity. It writes nothing until the operator confirms, and it re-verifies the chosen candidate through REST at that literal before writing.

**`add` and `repin` read the identity only from REST `/api/daemon/status` at the literal being dialled** (safety N7). Sim and mockup flags must be off, and the id must be non-null. The mDNS TXT `unit_id` is never used.

**`repin`** (adversarial N1):
- It shows the old and new id, model, version and `wlan_ip`, and when the old id was last seen.
- It refuses a new id that another entry already pins.
- It needs a TTY confirmation, or `--yes`.
- The mismatch error lists "check your network or DHCP" **first**, and mentions `repin` only as the audio-board-swap case.

**An unpinned entry** resolves only from `host:` or `<robot_name>.local`. It never caches. It needs `--unpinned` for `embody`/`--device` (§3a), and it is labelled everywhere.

**Candidate hygiene** (adversarial #7, N9):
- Canonical IPv4 through `ipaddress.IPv4Address`; octal, hex and short forms are refused. IPv6 is out of v1.
- Refused: loopback (unless the entry is explicitly `localhost_only` or `tunnel:`), `0.0.0.0`, link-local, multicast and broadcast.
- CGNAT/Tailscale is allowed and labelled.
- The port is fixed at 8000.
- Trailing dots are normalised. At most 4 candidates are tried.

**Proxies** (adversarial N4). "Disable proxies" needs a mechanism, so PR 0 adds a strict mode to `utils/http.py`: `fetch_url(..., address_policy="private", proxy="never")`. In that mode a private endpoint is dialled directly and never through a proxy. Today the proxy wins per #921, and the private endpoint is `strict=False`. Both the WebSocket (`websockets`) and the SDK's `requests` POST follow environment proxies. A **pinned connect is therefore refused** when `HTTP(S)_PROXY`/`ALL_PROXY` is set and the target is not covered by `NO_PROXY`. The fix hint names `NO_PROXY=<ip>`.

### 4.3 The check that counts is on the session that moves the robot

`ReachyMiniController.connect` runs these steps for a pinned entry:

1. **Pre-filter, before any `ReachyMini(`.** `probe_daemon` at the chosen IPv4 literal, then `refusal(outcome, entry)` (identity, sim flags, era/minor, daemon state). It also requires the status **`wlan_ip` to equal the dialled IP**: `ReachyMini.__init__` opens WebRTC to `wlan_ip` and `camera.open()` before any later step can run (adversarial N5). A null `hardware_id` never matches.
2. **Dial the same literal with `connection_mode="network"` forced.** A tunnel entry dials `127.0.0.1` through this controller's own tunnel. `_wait_local_port` also checks `_tunnel_proc.poll()`, so a foreign listener on :8000 fails there, and is caught by the pin if it does not (safety N10).
3. **Re-verify on the session, immediately after `ReachyMini(` returns.**
   - It runs after the `_media_released_by_connect` assignment, and before `_assert_auto_yaw_off`, `_clear_latch_after_connect` and the `CONNECTED` state (safety S3).
   - It reads `self._mini.client.get_status(wait=True)` with a bounded timeout (a timeout refuses).
   - The status must report the pinned `hardware_id` and sim flags off.
   - A session that reports no `hardware_id` refuses with a distinct reason, "identity not reported by session" (safety N9), not as a mismatch.
   - On success, the controller sets `verified_identity`.
4. **Failure handling.** The broad `except` cleans up, then **re-raises**; it never returns `False` into "headless mode".
   - **Before `ReachyMini(`** (step 1 or 2 refusal): nothing was touched, so it raises `DeviceIdentityMismatch` or `DeviceUnreachable`.
   - **After `ReachyMini(`** (step 3 fails after step 1 passed): this means **REST and the session reached different daemons**, which is an active anomaly. It raises `DeviceIdentityDivergence` (safety N2). The media ownership that was recorded is the other daemon's REST pre-state, so it is not trusted:
     - **do not** re-acquire or release media on the session's daemon;
     - close the media manager, drop the handle, stop the tunnel;
     - log at ERROR exactly what that unknown daemon received: `SetAutomaticBodyYaw(False)`, and for `no_media` the `/api/media/release` POST.
5. **Media residual.** `ReachyMini.__init__` sends `SetAutomaticBodyYaw(False)`, plus the media release (`no_media`) or WebRTC + `camera.open()` (`default`), before step 3. **None of these moves a joint.** For motion, step 3 holds. For these side effects, step 1 (with the `wlan_ip` and proxy checks) is the only protection. This residual is stated in `safety.md`.

**Red gates.** `xfail(strict=True)` tests through the real controller against `tests/unit/_reachy_fakes.py`:
- a mismatch on REST;
- a mismatch on the session only (divergence);
- a session that omits `hardware_id`;
- a plugin subclass whose `connect()` override skips the pin (registry refuses);
- a reuse of an unverified controller;
- a mode-switch re-exec that keeps the device.

Each test is verified red with a deletion probe.

**Scope of "every path."** The pin covers every path through `RobotRegistry` and `ReachyMiniController`. It does **not** cover these residuals:
- scripts that build `ReachyMini(` directly;
- `make_reachy_doa_reader(None)`;
- the stable unpinned `maxim.connect`/`run(robot=)`;
- old builds that share `~/.maxim` and ignore `hardware_id:` (documented downgrade);
- `console`'s `adopt(sdk)` until #1808.

**`hardware_id` limits.**
- It is identity, not authentication.
- It is null off Linux.
- It is cached at daemon start.
- It depends on enumeration order when two Pollen audio devices are attached.
- A legitimate board swap uses `repin`.

### 4.4 `check` layers

All checks are REST/TCP-only, and each layer has its own fix hint:
1. multicast permission (macOS)
2. name resolution
3. TCP :8000
4. daemon status
5. identity + sim flags + `wlan_ip`
6. era/minor
7. daemon state
8. `media_released` (`POST /api/media/acquire`)
9. `media_backend`
10. proxy environment versus `NO_PROXY`

On macOS, an empty `find_robots` result is reported as "none found, or Local Network permission blocked".

### 4.5 Mid-run reconnect (new in v3; safety N4)

`RobotController.reconnect` and `ConnectionMixin._soft_reconnect` handle the new typed errors. Today nothing catches them, so they would escape into the capture loops and skip `_degrade_capabilities`.
- **`DeviceUnreachable`** keeps the existing retry and backoff and calls `_degrade_capabilities`.
- **`DeviceIdentityMismatch` / `DeviceIdentityDivergence` on a reconnect** means a different robot now answers at the address. The controller:
  - **sets the stop latch** (no further motion);
  - degrades capabilities;
  - ends the run with the typed exit.

  It **never retries an identity failure**. A test covers the "robot swapped mid-run" case.

## 5. Fixes folded in (same root cause or exposed by it; not filed)

- **The `--robot` abbreviation and the gate** (§3.1): an explicit `--robot`, a `--robot-name` default of `None`, and a gate on parsed explicit flags.
- **`from_yaml` / `from_dict` silent skips, the `find_config_file` fallback, and the selfy/agentic_runtime DEBUG swallows** (§4.1).
- **The duplicate matching rule** in `agentic_runtime`, and its separate robots.yaml reload (§4.1).
- **Three copies of the Reachy probe** (§2): one `probe_daemon` + `refusal`. Each consumer keeps its own policy: doctor warns; the controller refuses for pinned entries; the #1704 media read uses the same outcome. The tests' monkeypatch seams move with the probe and stay non-vacuous. The `lint_http_surface` raw-dial pins update exactly. `maxim-diagnostics` becomes a wrapper, and the standalone script gets a pointer.
- **Dead code:** the `timeout:` key, `Maxim._connect_kwargs`, and `setup_robots_from_config`/`connect_configured_robots`, which get a Dormant marker or are routed through the pin.
- **Environment variables.**
  - `MAXIM_ROBOT_NAME`, `MAXIM_REACHY_HOST`, `MAXIM_ROBOT_TIMEOUT` and `MAXIM_DEVICE` get brief rows and conftest scrubs.
  - The unguarded `float()` on `MAXIM_ROBOT_TIMEOUT` is fixed.
  - The exp49 `_trial_env` (and the documented harness `child_env` recipe) pops `MAXIM_DEVICE` and `MAXIM_ROBOT_NAME`, so a harness child never becomes a named-device run (wire N6, adversarial N12).
- **`utils/http.py` strict private mode** (`proxy="never"`; §4.2).
- **Agent writes to the trust-bearing files** (adversarial N2). In singularity mode, `filesystem_policy.check_write_permission` allows any write under cwd. Maxim started from `~` therefore lets the agent rewrite `<data_home>/robots.yaml` (repoint or delete the pin) and `util/device_state.json`. PR 0 denies writes to those two files regardless of `cwd_access`. **D14 (taken): the deny covers the whole control-plane set, `~/.config/maxim/` (mesh.yml, api_key, console_token, config.json) as well as the two files above, in every mode.** `safety.md` and the singularity-mode docs state the new limit.

## 6. Threats: every input and how it could lie

| Input | How it lies | Handling |
|---|---|---|
| mDNS advertisement (name, TXT, `address`, `unit_id`, SRV port) | Unauthenticated (RFC 6762 §21) | Hints only (`discover`/`relocate`, with operator confirmation). The pin is never read from TXT, and the SRV port is never dialled |
| TXT and status strings | Attacker-controlled: ESC/OSC, Rich markup, bidi/format characters (Unicode `Cf`), `dns-sd` whitespace key injection | One sanitiser in `hardware/`: strip C0/C1 and `Cf`, escape Rich markup, bound the length, and check TXT key shape `[a-z0-9_]+`. Used for every display and log (adversarial A12). **Never reaches a prompt:** the discovery types live in `hardware/devices_types.py`, and a new mechanization-backlog row adds a lint that prompt builders do not import them |
| REST status vs the WS session | Different daemons (auto mode, re-resolution, tunnel listener, proxy) | Step-1 `wlan_ip` and proxy checks, and the session-bound re-check with divergence handling (§4.3) |
| Cached IP | Reassigned | Re-verified every time; pinned entries only |
| Hostname | Not unique; AAAA stalls | Resolved once to canonical IPv4, then verified |
| `hardware_id` | Public; null off Linux; the real id from a sim on the robot | Not authentication; null never matches; sim flags checked; only ever read from REST at the dialled literal |
| `<data_home>/robots.yaml` | Malformed, newer, typo, a key under the wrong parent; **written by an agent's file tools** | Strict typed parse, refuse-newer, a deny in `check_write_permission` (§5) |
| `./robots.yaml` (cwd) and a relative `MAXIM_DATA_HOME` | A planted file; isolation pointed at the real robot | The cwd fallback is removed (D11). A relative value raises; it never falls back to `~/.maxim` |
| `--device` / `MAXIM_DEVICE` / legacy flags / `primary:` | Stale, a typo, inherited by a child process | Explicit names match exactly or refuse; a conflicting pair is refused; harness children pop the variables; the plan block names the source |
| `launch:` defaults | A persisted capability grant | `mode` limited to run modes; `operational_mode` refused; the grant is the typed flag |
| `repin` / `relocate` | Saving an impostor's identity after a DHCP swap | REST-only identity, uniqueness check, TTY confirmation, network-first hint |
| Offline/advisory latch (V8) | `dns-sd` children escape an in-process latch | The latch is checked before `find_robots`; census and lint pins |
| Proxy environment | Routes the WS and SDK POST differently from the probe | Strict `proxy="never"` probe; pinned connect refused unless `NO_PROXY` covers the target |

The impact is **safety**, not only security. The strict choice is the default everywhere.

## 7. Persistence and invariants

- **One writer.** `hardware/config.py::write_robots_config` always targets `<data_home>/robots.yaml`. It runs under `FileLock`, writes through `atomic_write_text`, writes root `_format_version: "1.0"`, refuses to write over a newer file, and takes a one-time `robots.yaml.bak` before the first rewrite of a hand-written file.
  - **It edits the raw mapping under the lock** rather than doing a load → mutate → dump through the lossy dataclasses, and a round-trip test pins this (architecture N5).
  - `rename`/`forget` move `primary:` within the same write. `device_state.json` is keyed by identity, so `rename` need not touch it, and `forget` removes the identity's row after the robots.yaml write succeeds. The two files are **not** one atomic transaction; the order is chosen so a crash leaves a harmless stale cache row (architecture N9).
  - The callers are enforced by the §3b lints.
- **Layer table.** robots.yaml goes into the persistence-config §1 table as declarative operator intent living in the data home. C2 is amended there for the Python operator verbs (D10).
- **Layers.**
  - `hardware/` holds: types, pin, resolver, probe, sanitiser, writer.
  - `maxim/devices/` (a package with a `LAYER_RULES` entry) holds: verbs, the name validator (it reads mesh.yml names; the `LAYER_RULES` entry permits `peer.mesh_config` reads), and `plan_launch`.
  - `cli.py` dispatches, and its subcommands go in `_NON_SIM_SUBCOMMANDS`.
  - The touch rule applies to `_main_impl` and `Maxim.__init__`.
- **Hardware safety.** `check`, `discover`, `relocate`, `repin` and the preflight open no `ReachyMini(`. The lint rules 3 and 4 stay in force.
- **Optional deps:** `require_optional_dependency("reachy_mini")`, with no direct zeroconf in v1.
- **M76:** this does not trip it.

## 8. Phasing (a three-lens round per PR; estimates are `src/` lines excluding tests)

- **PR 0, groundwork (≈320):**
  - the §5 fixes;
  - typed, loud yaml errors and their propagation, including `find_config_file` and the relative `MAXIM_DATA_HOME` check;
  - one resolver, and the resolved entry handed to `agentic_runtime`;
  - `probe_daemon` + `refusal`;
  - explicit `--robot`, `--robot-name` default `None`, `--device`, the conflict refusal and the explicit-flag gate;
  - removing the cwd fallback;
  - `fetch_url(proxy="never")`;
  - the `check_write_permission` deny for robots.yaml + `device_state.json` and `~/.config/maxim/` (D14);
  - env scrubs and the harness env pops.
- **PR 1, the pin (≈400):**
  - typed `hardware_id:`/`hostname:`/`launch:` fields with strict unknown-key refusal;
  - `expected_identity=` and the post-connect `verified_identity()` check in the registry, and the reuse refusal;
  - the controller's §4.3 sequence, with divergence handling and typed raises;
  - the §4.5 reconnect handling;
  - `device_state.json`;
  - the sanitiser;
  - the `maxim.devices` read verbs (`entries/get/check/resolve/open`) and `maxim device list|show|check`.
- **PR 2, the writer (≈320):** the raw-mapping `write_robots_config`, the `add`, `rename`, `forget` and `repin` verbs, `use`/`maxim connect` (which moves `primary:`); `relocate` lands with PR 3, the lints, and the C2 amendment.
- **PR 2b, `embody` (≈180):** `plan_launch`, preflight, the plan block, carrying the verified address, pin and typed grant into the existing launch, the `--device` re-exec, the D13′ exits, `--unpinned`, and `launch:` + `--save-defaults` (which writes `mode/llm/interactive` only).
- **PR 3, discovery (≈220):** `discover` + `relocate` over `find_robots`, the latch checked first, the census and lint updates, and the macOS hint.
- **Separate, #1808:** the library-first seam, then `maxim.devices.embody()` in process, the local-media kind, and `attached_via: <node>`.

Docs updated in the same PRs: `robot-setup.md`, `getting_started.md`, `cli-reference.md`, `configuration.md`, `safety.md` (LAN trust, the §4.3 media residual, proxies), `stable_api.md`, persistence-config §1, and the embodiment brief's env table.

## 9. Owner decisions

**Taken 2026-11-25:**
- **D1:** `maxim device …` plus top-level `discover`/`connect`.
- **D2:** `connect` means verify + refresh + make default.
- **D3:** a strict `--device`.
- **D4:** per-kind stores.
- **D5:** a format version, and a `.bak` before the first rewrite.
- **D6:** refuse on mismatch.
- **D7:** SDK `find_robots` only.
- **D8:** `maxim embody`, and `launch:` never holds `operational_mode`.
- **D9:** v1 Python is plan-only; the in-process launch comes after #1808.
- **D2′:** the default lives in robots.yaml, not config.json. **D2″ (delta re-ask, taken):** ride the existing `primary:` as the single marker, with no `default:` key.
- **D3′:** `--robot` stays an explicit alias of `--robot-name`.
- **D10:** Python write verbs are operator writers, enforced by lint.
- **D11:** remove the cwd fallback.
- **D12:** one `maxim.devices` namespace. **D12′ (delta re-ask, taken):** `run_device` stays in `console/`, with at most a lazy re-export.
- **D13:** non-zero exit on a refused named-device connect. **D13′ (clarification folded in v3, same intent):** defined by pin status; `--unpinned` is required to run an unpinned entry through `embody`/`--device` (§3a).

**Taken 2026-11-25 (owner took the recommendation):**
- ✅ **D14. Deny agent file-tool writes to all of Maxim's own control-plane files?** In singularity mode, `check_write_permission` allows writes anywhere under cwd. Started from `~`, the agent can rewrite `~/.config/maxim/` (mesh.yml, api_key, console_token, config.json) and `<data_home>/robots.yaml` without approval.
  - *Recommendation:* deny writes to the whole control-plane set (`~/.config/maxim/`, `<data_home>/robots.yaml`, `<data_home>/util/device_state.json`) in every mode, in PR 0. That is about 30 `src/` lines plus tests. Singularity keeps full access to everything else. An agent should never be able to rewrite its own trust anchors.
  - *Alternative:* deny only this design's two files now, and decide the config directory separately.
  - This is a security fix that changes singularity's documented reach, which is why it is an owner decision.

## 10. Disposition of round-1 findings (DRAFT v1)

Abbreviations: A = adversarial, S = safety & lifecycle, R = architecture, W = wire.

| Section | Round-1 findings folded there |
|---|---|
| §4 | A1, A3, A4, A9, A10, A11 · S3–S6, S9 · R3, R4, R13 · W1, W3, W14, W18 |
| §3a/§3.1 | A2, A6 · S2, S7, S14, S15 · W2, W4, W5, W9 |
| §3b | S10 · R6, R9, R16 · W7, W11 |
| #1808 | S1, S11, S12 · R1, R2 · W10 |
| §5–§7 | A5, A7, A8, A12–A16 · S8, S13 · R5, R7, R8, R10–R12, R14, R15, R17 · W6, W8, W12, W13, W15–W17, W19 |

## 11. Disposition of delta-round-1 findings (DRAFT v2), and the round-2 scope

| Lens | Finding | Folded in |
|---|---|---|
| Adversarial | N1 | §4.2 `repin` |
| | N2 | §5 / D14 |
| | N3 | §3a `--unpinned` |
| | N4 | §4.2 proxies |
| | N5 | §4.3 step 1 `wlan_ip` |
| | N6 | §4.1 relative `MAXIM_DATA_HOME` |
| | N7 | §3.1 explicit names, D13′ |
| | N8 | §4.3 step 3 |
| | N9 | §4.2 canonical IPv4, `hostname:` |
| | N10 | §3b confirmations |
| | N11 | §4.1 `verified_identity` |
| | N12 | §3.1 source in plan, §5 env pops |
| | N13 | §4.1 `hardware_id:` key |
| | A12 remainder | §6 sanitiser |
| Safety | N1 | §3a `launch.mode` |
| | N2 | §4.3 divergence |
| | N3 | §4.1 |
| | N4 | §4.5 |
| | N5 | §2 / §3a / §4.3 shared `refusal` |
| | N6 | §3a typed grant |
| | N7 | §4.2 REST-only identity |
| | N8 | D13′ |
| | N9 | §4.3 step 3 |
| | N10 | §4.3 step 2 |
| | N11 | §3a |
| | S1 / S12 remainders | §3c → #1808 |
| | S3, S5, S7, S9 remainders | §4.3, §3a, §3.1, §4.1 |
| Architecture | N1 | §3b package |
| | N2 | §3b enforcement |
| | N3 | §4.1 refuse-newer |
| | N4 | D2″ `primary:` |
| | N5 | §7 raw-mapping writer |
| | N6 | §3b `_API_SUBMODULES` |
| | N7 | §3b `devices.open`, §3.1 id/name split |
| | N8 | §3c / §4.1 scope |
| | N9 | §7 |
| | N10 | §1 |
| | N11 | §3b types |
| | N12 | D12′ |
| Wire | N1 | §4.1 |
| | N2 | §4.1 scope |
| | N3 | D2″ + §4.1 resolved entry |
| | N4 | §3b lints |
| | N5 | §3.1 precedence + re-exec |
| | N6 | §5 env pops |
| | N7 | §3.1 argparse |
| | N8 | §4.1 |
| | N9 | §2 |
| | N10 | D13′ + §4.1 unknown `type:` |
| | F1b | §4.1 strict entries |
| | F11 | §3c / §4.3 residual |
| | F10 | #1808 |

**Delta round 2 scope (the fail-open folds; stop rule as before):**

| Lens | Findings to re-check |
|---|---|
| Adversarial | N1–N6 |
| Safety | N1–N5, N7 |
| Architecture | N1–N5, N7, N12 |
| Wire | N1, N3–N5, N7, F1b, F11 |

Fail-closed NITs that were folded without a round: adversarial N7–N13, safety N6 and N8–N11, architecture N6 and N8–N11, wire N2, N6 and N8–N10.
