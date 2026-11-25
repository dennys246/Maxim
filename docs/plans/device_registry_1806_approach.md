# Device registry (#1806): approach note, DRAFT v2

Status: DRAFT v2, 2026-11-25. Mirrors issue #1806.

DRAFT v1 got a four-lens adversarial design pass: [adversarial](reviews/1806/adversarial.md), [safety & lifecycle](reviews/1806/safety_lifecycle.md), [architecture](reviews/1806/architecture.md) and [wire](reviews/1806/wire.md), all read against b7380ee7. Every lens returned DO-NOT-BUILD, and they converged on one root cause: **the identity pin rode a free-form config key and a separate REST probe, so it could go missing silently or be checked on a different connection from the one that carries motor commands.** v2 makes the pin a typed, fail-closed part of the connection itself (§4). Every finding's disposition is in §10. A delta round on v2 is owed before PR 0.

Owner decisions: D1–D9 taken 2026-11-25, and v2's D2′, D3′ and D10–D13 taken 2026-11-25 (the owner took every recommendation; §9).

**What:** give Maxim one way to find, name, remember and reliably reconnect to the devices it senses and acts through. `maxim connect reachy` and `maxim embody reachy` replace hand-editing `~/.maxim/robots.yaml`, chasing a DHCP-moved IP, and knowing `.local` and `media_backend: no_media` quirks. Robots come first. The model leaves room for other kinds (a camera, a mic, a device attached to another node) without building that abstraction early. **Every verb is a library function first**, and the CLI is a thin shell over it (§3b). The deeper "the runtime is a library, the CLI is a shell" work is its own issue, #1808: the **library-first runtime seam** (§3c).

This is a **gate** (discovery and identity pinning form a security and safety boundary). The design pass, the delta round and the owner decisions all come before code.

## 1. The model: nodes, devices, clients

| | **Node** | **Device** | **Client** |
|---|---|---|---|
| What | A process running Maxim (leader, peer, solo) | An endpoint Maxim senses and acts through, opened by a driver | Something that dials *in* to a node |
| Examples | big-mac-mini, the RTX leader, a Pi on the robot | the Reachy daemon, the simulated robot; later camera, mic, speaker | Console browser, Pulse, the planned phone |
| Store today | `~/.config/maxim/mesh.yml` (SHAPE-FROZEN), `peer.yml`/`config.json` lanes | `<data_home>/robots.yaml` (hand-edited, no writer) | `console_token`, spoken-code pairing |
| Trust | Mesh key (bearer) | Per driver; Reachy has **no auth** (the LAN is the boundary) | Console token / pairing; per-device tokens planned |
| Direction | Dial out | Dial out | Dial in |

- **A device is attached to exactly one node, the one that opens it.** This is load-bearing: the single clamped motion dispatch point, media release ownership (#1704) and the stop latch are all per controller and per process. In v1 every device is attached locally. `attached_via: <node>` comes later, with `embodied_device.md`.
- **One name namespace, separate stores.** No store merges (mesh.yml is frozen; `detect_role` infers role from which files exist; config.json sections bump its format version). The one name validator refuses a name that both a mesh node and a device claim. It runs at read time as well as write time, so a later `peer add-node` cannot shadow a device. It only warns on a solo box or a malformed mesh.yml.
- **Clients are out of scope.** Their pairing paths (`console_tunnel_hardening.md` A9/A9.1, `phone_device.md`) are unchanged.
- **Not node discovery.** LAN discovery and pairing of leaders and peers stays deferred to `embodied_device.md` §2's security pass, and this is not the mesh mDNS design (POG-2c).

**Only one kind exists today, so v1 builds no kind abstraction.** Entries carry a reserved `kind: robot` (the default), and the router has one branch. A `DEVICE_KINDS` table, or a `maxim.device_kinds` entry-point group, arrives with the second real kind, under `config_extensibility.md`'s rules (architecture #7).

## 2. Front gate: what this rides on

| Need | Existing infrastructure | Verdict |
|---|---|---|
| Name → robot config | `hardware/config.py`: `RobotsConfig`, `resolve_robot_entry`, `resolve_connection_config` | **Ride.** robots.yaml stays the robot store. Entry-level fields become typed (§4.1) |
| Find a Reachy | SDK `reachy_mini.utils.discovery.find_robots()` (`_reachy-mini._tcp`, macOS `dns-sd` path) | **Ride** for `discover` and `relocate` only. It is never part of automatic resolution (§4.2) |
| Identity | `hardware_id` (16 hex, from the audio-chip serial) in both REST `/api/daemon/status` and the WS session's `DaemonStatus`, which `WSServer.publish_status` pushes at 1 Hz and `WSClient.get_status()` stores | **Ride.** The check that counts is on the WS session (§4.3) |
| Reachability probe | `doctor::_check_one_robot`, `utils/reachy_diagnostics.py`, `scripts/check_reachy_connection.py`, the controller pre-checks | **Consolidate** into one probe that returns a typed outcome. Each consumer keeps its own policy (§5) |
| Default device | robots.yaml `primary:` | **Ride.** A top-level `default:` key replaces config.json (D2′) |
| Mutable last-good address | none (robots.yaml is declarative; persistence-config §1) | **New:** `<data_home>/util/device_state.json`, keyed by **identity** and written only for pinned entries. Not `probe_cache` because that cache is URL-keyed with a TTL and records health, not identity-bound addresses. It is exempt from layer-2 role scoping because a device's address does not depend on the node's role (architecture #14) |
| Writing robots.yaml | none | **New:** one writer in the `mesh_config.write_mesh_config` shape (§7) |

## 3. Surfaces

### 3.1 CLI

```
maxim discover                      # read-only: LAN candidates (hints) + registered entries + mesh node names
maxim device add <name> [<host>]    # pick a discovered candidate, or give a host/IP (mandatory path when multicast is blocked)
maxim device list | show <name>
maxim device rename <old> <new> | forget <name>
maxim device relocate <name>        # operator-confirmed address update from discovery (§4.2)
maxim device repin <name>           # operator-confirmed identity change (audio-board swap), old→new shown
maxim device check [<name>]         # layered diagnosis, never actuates
maxim connect <name>                # resolve + verify + refresh the cache + set as default
maxim --device <name> ...           # strict: an exact entry name or a refusal   [experimental]
maxim embody <name> [--dry-run]     # preflight + launch embodied (§3a)        [experimental]
```

- **`--robot` is unchanged.** It becomes an explicit alias of `--robot-name`, with today's exact behaviour, so the documented `maxim --robot reachy_mini` keeps working. `--device` is the new strict flag (D3′). Defining `--robot` explicitly removes the reliance on argparse abbreviation. `allow_abbrev` stays as it is.
- **The quick-start gate reads explicit flags, not resolved defaults.** A bare `maxim` still opens the menu after `connect` has set a default; it never launches the robot by itself (wire F4). The gate's test calls the real `cli.py` gate, not a copy of it.
- **Selection precedence** applies only once a robot run is requested (`embody`, `--device`, `--mode`): `--device` > `MAXIM_DEVICE` (a new variable, documented and scrubbed in conftest) > robots.yaml `default:` > a single entry or the `primary:` entry. An **explicit** name matches an entry exactly or is refused. Only the no-name path may use the primary fallback (adversarial #6, wire F5).
- `maxim connect` is distinct from `maxim peer connect`. The help text and docs say which is which.

### 3a. `maxim embody <name>`

`connect` verifies and remembers a device, and `embody` starts Maxim in it.

1. **Resolve and verify** (§4). An unknown name is refused, and the default is not changed.
2. **Preflight.** These are the REST/TCP-only `check` layers: no `ReachyMini(` and no tunnel. A `tunnel:` entry's preflight reports "verified at connect" rather than borrowing `controller.connect()`.
   - **Refuse** on an identity mismatch, an unreachable daemon, an era or minor version mismatch, a daemon not `running`, or a sim/mockup daemon for a hardware-pinned entry.
   - **Warn** on `no_media`, or on `media_released` when the mode needs perception, and print the fix. The preflight never mutates the daemon.
3. **Plan block.** Device, verified IPv4 address, `hardware_id`, body ref, run mode, operational mode (whether the head will move), LLM, media backend. `--dry-run` stops here.
4. **Launch** through the existing robot path, with the **verified address and pin carried forward as typed arguments**. The launch does not re-resolve them (adversarial #11).
   - A refused connect on a named device **exits non-zero**. It does not fall back to running headless (D13).
   - `reexec_with_mode` carries `--device` (and `--robot-name` as today), so a mode-switch child re-resolves and re-verifies the same entry (wire F2, safety #7). The library-first seam (§3c) later removes the re-exec entirely.

**Defaults.** The run mode is `exploration`. `--mode`, `--llm`, `--audio` and `--interactive` pass through. An entry-level `launch: {mode, llm, interactive}` (a sibling of `config:`, validated, so it is never passed to a controller constructor) stores per-device defaults, and `--save-defaults` writes them. **`launch:` never holds `operational_mode`** (owner, 2026-11-25). Capability is granted only by the operator at launch (#829/#924), so the loader refuses the key. The launch also still runs `_validate_operational_grant` against what was actually typed (safety #2, adversarial #2, wire F9).

**Safety.** `embody` is the only verb here that can move motors. `discover`, `add`, `relocate`, `repin`, `check` and `connect` cannot actuate by construction. `embody` adds no motion code, and it never targets an unregistered or unverified device.

### 3b. The Python API: every verb is a library function first

The CLI is a thin shell. Each verb is a function in **`maxim.devices`** (a lazy namespace like `maxim.create`), and `cli.py` only parses and prints. A path that exists only in the CLI is a defect.

```python
import maxim

found = maxim.devices.discover(timeout=5.0)       # tuple[DiscoveredDevice, ...]: sanitised hints, never trusted
entry = maxim.devices.add("reachy", found[0])     # or add("reachy", host="10.6.0.63"); pins hardware_id (refuses a null id → explicit UNPINNED)
maxim.devices.entries(); maxim.devices.get("reachy")
maxim.devices.rename("reachy", "rosie"); maxim.devices.forget("rosie")
maxim.devices.relocate("rosie", candidate)        # operator-confirmed address update
report = maxim.devices.check("rosie")             # DiagnosticReport of CheckResult (the existing maxim.diagnose() types)
dev = maxim.devices.resolve("rosie")              # ResolvedDevice: verified IPv4 + identity; changes nothing
maxim.devices.use("rosie")                        # what `maxim connect` calls: resolve + set robots.yaml default
ctl = maxim.devices.connect("rosie")              # RobotController, connected through the pin (§4.3)
plan = maxim.devices.plan_launch("rosie", mode="exploration")   # LaunchPlan: what will run, what will move
```

- **The stable `maxim.connect(robot_type, config=…)` and `maxim.run(robot=)` are not changed.** They stay **unpinned paths**, and the docs and their log line say so (architecture #6, safety #10). Pinned connection is `maxim.devices.connect(name)`.
- **Python write verbs are operator writers** (`add`, `rename`, `forget`, `relocate`, `repin`, `use`, and `save_defaults=`). They need a DECISIONS amendment to C2, which today permits CLI verbs only. A lint bans importing them from `tools/`, `agents/`, `skills/` and the runtime, so the agent can never call them (D10; architecture #9, wire F7).
- **v1 has no in-process launch from Python** (D9). `plan_launch` returns the plan, and the launch itself arrives with the library-first seam (§3c) as `maxim.devices.embody(...)`.
- **Types.** `DiscoveredDevice`, `DeviceEntry` (a view over `RobotConfig`, not a copy of it), `ResolvedDevice`, `DevicePin` and `LaunchPlan` are frozen dataclasses. Only those that persist or cross a wire claim a CC3 path. Errors are typed: `DeviceNotFound`, `DeviceIdentityMismatch`, `DeviceUnpinned`, `DeviceUnreachable`, `DiscoveryUnavailable`, `RobotsConfigError`, under the stable `MaximConnectionError` / `ConfigurationError`. `_run_stopped_on` re-raises them as typed exits, not as crashes (safety #15).
- **Stability.** The whole namespace enters `stable_api.md` as ⚠️ Experimental. `scripts/run_mypy_public.sh` gains the new module, and `_API_VERBS` and the README verb count update.
- **Namespace.** `maxim.devices` is the single device namespace. `embodied_device.md`'s planned `maxim.device.run_device` folds into it under §3c rather than becoming a second namespace (D12, wire F11).

### 3c. Library-first runtime seam (#1808; #1806's `embody` is its first consumer)

The owner's goal is that people can use **the core library or the CLI, without the library depending on the CLI**. The design pass found the robot runtime cannot be called from a library today. Inventory (safety #1/#12, architecture #1, wire F10):

- `cli.py::_after_run_mode_request` → `cli_utils.py::reexec_with_mode` **`os.execve`s the process** on a mode switch.
- `configure_logging(force=True)` runs on each loop iteration. `multiprocessing.set_start_method(force=True)`, environment `setdefault`s, `atexit` hooks, termios raw mode on stdin, and role/leader boot are all process-global.
- `_end_robot_run` calls `stop_active_spawner` (global). `_RUN_LOCK` is not shared with `api.run`. `request_stop` is owner-only and sticky process-wide.
- `maxim.register_tool` and `maxim.on` reach only `api.run` and the sim orchestrator (`_inject_registered_tools` / `_bridge_event_subscriptions` have no caller in `embodied_runtime/`). **Custom Python tools and event hooks never reach a robot run today.**
- `selfy.py::Maxim.__init__` is about 312 lines, so the touch rule applies.

The seam: `embodied_runtime/launch.py::launch(plan, *, stop)`.
- A mode switch becomes an in-process state change, with no re-exec.
- Registered tools and event subscriptions are wired in the seam.
- Stopping goes through a run-owned event plus `MaximHandle.stop(deadline=)` (`bounded_stop.md`).
- The library installs no signal handler.
- The CLI keeps argv, signals, the terminal, the display, logging setup, exit codes and re-exec.

**Guard:** a lint bans `sys.exit`, `os.exec*`, `signal.signal`, `termios`, `configure_logging(force=True)` and `set_start_method(force=True)` outside the CLI layer, with the existing sites as a ratchet baseline. This coordinates with `embodied_device.md` (`MaximHandle.embody`/`adopt`, `run_device`), #1591, #1741 and #1439. That issue owns this work. #1806 does not build it.

## 4. Connection, resolution and identity (the core, rewritten in v2)

### 4.1 The pin is a typed field, not a config key

- **robots.yaml entries gain entry-level `pin:` (16 lowercase hex, format-checked), `hostname:` and `launch:`**, parsed by `RobotConfig.from_dict` into typed fields. Today `from_dict` drops entry-level keys silently (probe-confirmed, wire F1b), so this is a real parser change. An absent `pin:` means `DevicePin.UNPINNED`, a sentinel and never `None`. An empty `pin:` value, or a malformed one, is a `RobotsConfigError` (adversarial #3).
- **`RobotRegistry.connect_robot` enforces the pin** for every controller class. The registry does not pass the pin through the constructor-signature filter (which drops unknown keys at DEBUG). A pinned entry on a controller class that does not declare `supports_identity_pin = True` is **refused**, not connected unpinned. This covers plugins, `type:` overrides and typos (architecture #4, safety #4, wire F1c).
- **The registry's reuse branch** ("already connected") compares the requested pin with the live controller's verified identity and refuses on a mismatch (safety #9, wire F3).
- **robots.yaml load errors propagate.** `RobotsConfig.from_yaml` raises `RobotsConfigError` on parse errors, a top-level list, non-dict entries or `config: null` (wire F18). A missing file is still fine, and simply means no entries. `selfy.Maxim.__init__` and `agentic_runtime._resolve_body_wiring` stop catching it at DEBUG. A broken file on a robot run exits non-zero (architecture #3, safety #8, wire F1a). The one resolver, `resolve_robot_entry`, replaces `agentic_runtime`'s inline copy.

### 4.2 Resolution never trusts an advertisement

Candidates for a **pinned** entry, in order:
1. the entry's `host:` (an IPv4 literal, or a hostname **resolved once** to IPv4);
2. the cached last-good address for that identity.

That is all. **Discovery is not a resolution step** (adversarial #4). A LAN host can advertise any `unit_id` (the `hardware_id` is public), so a "first match wins and gets cached" step is an auto-connect from an unauthenticated advertisement. When both candidates fail, the error names each candidate and its failure, and suggests `maxim device relocate <name>`. That verb browses, shows the candidates with their claimed identity, and writes the new address only after the operator confirms one.

**An unpinned entry** (a hand-written legacy file, or a daemon whose `hardware_id` is null) is resolved from `host:` or `robot_name.local` only. It **never** caches and never uses the state file. It connects as **labelled unpinned**: `show`, `check`, the plan block and a WARNING line all say so (adversarial #10).

**Candidate hygiene** (adversarial #7):
- IPv4 only in v1. The SDK builds `f"ws://{host}:{port}"`, which an IPv6 literal breaks.
- Refuse loopback (except an explicit `connection_mode: localhost_only` or `tunnel:` entry), `0.0.0.0`, link-local `169.254/16`, multicast, and broadcast.
- CGNAT/Tailscale is allowed (non-public per #921) and labelled.
- The port is fixed at 8000. A different advertised SRV port is shown in `discover` but never dialled.
- Trailing-dot FQDNs are normalised, and at most 4 candidates are tried.
- Probes use `utils/http.fetch_url(address_policy="private", deadline_s=…)` **with proxies disabled**.

### 4.3 The check that counts is on the session that moves the robot

The REST probe is a fast pre-filter. It is not the identity check: the REST probe and the WebSocket can reach different daemons (`connection_mode: auto` tries localhost first; a hostname re-resolves; the SRV port can differ from 8000; a tunnel's local :8000 is any listener; `websockets` honours `HTTPS_PROXY`/`socks` for `ws://` where httpx does not). So `ReachyMiniController.connect` for a pinned entry does the following (adversarial #1, safety #3/#5/#6, wire F14):

1. **Pre-filter, before any `ReachyMini(`.** Fetch REST status at the chosen IPv4 literal and require `hardware_id == pin`, `simulation_enabled is not True` and `mockup_sim_enabled is not True` (a sim on the robot's own CM4 reports the real id; adversarial #9). A null `hardware_id` never matches. On a mismatch, refuse before the SDK touches the daemon, so there is no auto-yaw write and no `no_media` release on the wrong daemon.
2. **Dial the same literal with `connection_mode="network"` forced.** For a tunnel, dial `127.0.0.1` through the tunnel this controller started.
3. **Re-verify on the session.** Before the first motion, wake or torque command, read `self._mini.client.get_status()`, the WS session's own `DaemonStatus` pushed at 1 Hz. It must report the same `hardware_id` and the sim flags off. This binds the identity to the connection that carries the motor commands, which closes the proxy, tunnel and auto-mode gaps in one check.
4. **On any failure, raise `DeviceIdentityMismatch`** (or `DeviceUnreachable`) through `connect()`. It does not `return False` into "headless mode". The controller first undoes what it did: re-acquire media if this connect released it (#1704), drop the handle, stop its tunnel. The selfy and registry paths propagate it.
5. **Residual:** the WebRTC signalling host comes from the status `wlan_ip`. It is compared with the verified IP, and a difference is a WARNING. Media identity is out of scope in v1 (adversarial #14).

**Scope of "every path":** the pin covers every path through `RobotRegistry` and `ReachyMiniController`. **Not covered**, and listed as residual: scripts that build `ReachyMini(` directly, `make_reachy_doa_reader(None)`, the stable unpinned `maxim.connect`/`run(robot=)`, and old builds that share `~/.maxim` and ignore `pin:` (a downgrade connects unpinned; this is documented) (wire F15/F16, safety #10).

**`hardware_id` limits** (documented):
- It is identity, not authentication: it stops mix-ups, not a deliberate impostor, because the daemon has no auth (an upstream residual).
- It is null on non-Linux daemons (a Lite on a Mac, the desktop app, sim), so those entries are unpinned.
- It is cached at daemon start.
- With two Pollen audio devices it depends on which one is found first.
- A legitimate audio-board swap uses `repin`.

### 4.4 `check` layers

Each layer has its own fix hint, and all of them are REST/TCP-only:
1. multicast permission (macOS)
2. name resolution
3. TCP :8000
4. daemon status
5. identity and sim flags
6. client/daemon era and minor version
7. daemon state
8. `media_released` (fix: `POST /api/media/acquire`)
9. `media_backend` (`no_media` versus a perception run mode)

`find_robots` swallows `EHOSTUNREACH`, so on macOS an empty result says "no robots found, or Local Network permission is blocked" and names the setting. `discover` is never reported as definitively empty (adversarial #13).

## 5. Fixes folded in (same root cause or exposed by it; not filed)

- **The `--robot` abbreviation and the quick-start gate.** `--robot-name reachy_mini` and `--robot=reachy_mini` currently land in the menu. Fix: an explicit `--robot`, and a gate on parsed explicit flags tested through the real `cli.py` (§3.1).
- **`from_yaml` / `from_dict` silent skips, and the selfy/agentic_runtime DEBUG swallows** (§4.1).
- **The duplicate matching rule** in `agentic_runtime` (§4.1).
- **Three copies of the Reachy probe.** Consolidate them into `hardware/reachy/probe.py::probe_daemon(ip, *, deadline_s) -> ProbeOutcome`, used by doctor, `check`, the preflight and the controller pre-filter. Each consumer maps the outcome to its own policy: doctor warns; the controller treats a failed status read as non-fatal only for unpinned entries; the #1704 media-ownership read takes it from the same outcome. `maxim-diagnostics` and `scripts/check_reachy_connection.py` become thin wrappers. The probe picks `address_policy="private"` (M18). The tests' monkeypatch seams move with it and stay non-vacuous, and the `lint_http_surface` raw-dial pins are updated exactly (wire F12).
- **The dead `timeout:` key** in `data/util/robots.yaml`, and the dead `Maxim._connect_kwargs` (wire F19).
- **Env vars.** `MAXIM_ROBOT_NAME`, `MAXIM_REACHY_HOST`, `MAXIM_ROBOT_TIMEOUT` and the new `MAXIM_DEVICE` get brief-table rows and conftest scrubs. The unguarded `float()` on `MAXIM_ROBOT_TIMEOUT` is fixed (adversarial #15).
- **`setup_robots_from_config` / `connect_configured_robots`** have no src caller. They are a third launch path that bypasses the pin, so they get marked Dormant, or routed through the registry pin (wire F19).

## 6. Threats: every input and how it could lie

| Input | How it lies | Handling |
|---|---|---|
| mDNS advertisement (name, TXT, `address`, `unit_id`, SRV port) | Unauthenticated (RFC 6762 §21) | Hints only, used by `discover`/`relocate` with operator confirmation, never by resolution (§4.2). The port is never dialled |
| TXT and status strings | Attacker-controlled. Raw ESC/OSC bytes, Rich markup, `dns-sd` whitespace key injection | One sanitiser in `hardware/` strips C0/C1 controls, escapes Rich markup and bounds the length. Used for every display and log. Never reaches a prompt (adversarial #12) |
| REST status vs the WS session | Different daemons (auto mode, re-resolution, tunnel listener, proxy divergence) | The session-bound re-check (§4.3) |
| Cached IP | Reassigned | Identity re-verified every time. Pinned entries only |
| Hostname | Not unique (`reachy-mini.local` on every robot); AAAA stalls | Resolved once to IPv4, then verified |
| `hardware_id` | Public; null off Linux; real id from a sim on the robot | Not authentication; null never matches; sim flags checked |
| `<data_home>/robots.yaml` | Malformed, newer, or a hand-edited typo | Typed error, `_format_version` refuse-newer, strict typed fields |
| `./robots.yaml` (cwd fallback) and `MAXIM_DATA_HOME` (a relative value resolves from cwd) | A repo checkout or a singularity agent with cwd write access plants a whole entry | **Remove the cwd fallback** (D11). The data-home branch requires an absolute `MAXIM_DATA_HOME` (adversarial #5/#15, wire F8) |
| `--device` / `MAXIM_DEVICE` / `default:` | Stale, a typo | An explicit name matches exactly or is refused. No primary fallback for an explicit name |
| `launch:` defaults | A persisted capability grant | `operational_mode` is refused by the loader. `_validate_operational_grant` runs on typed flags |
| Offline/advisory latch (V8) | macOS `find_robots` spawns `dns-sd` children whose network I/O escapes an in-process latch; `_filter_alive` dials | `discover`/`relocate` check the latch **before** calling `find_robots`. They are added to `test_offline_latch.py::CENSUS_SITES`, and `lint_http_surface` `EGRESS_SDK_IMPORTERS["reachy_mini"]` gets the discovery module (safety #13, wire F13) |
| Proxy env | `HTTPS_PROXY` routes the WS but not the probe | Probe proxies disabled. Identity is bound on the session anyway (§4.3) |

The impact is **safety**, not only security. A spoofed endpoint receives motor commands over cleartext WS and feeds fabricated perception into the loop. The strict choice is the default everywhere.

## 7. Persistence and invariants

- **One writer:** `hardware/robots_yaml_writer.py::write_robots_config`.
  - It always targets `<data_home>/robots.yaml`, with `FileLock`, `atomic_write_text`, a root `_format_version: "1.0"` (via `utils/format_version`, with refuse-newer on load), and a one-time `robots.yaml.bak` before the first rewrite of a hand-written file (D5).
  - A CI caller grep allow-lists the `maxim.devices` operator verbs, in the `write_mesh_config` + `SETUP_VERB_MATCHES` shape. The import-ban lint covers runtime, tools, agents and skills (D10).
  - `rename` and `forget` update `default:` and `device_state.json` in the same locked transaction.
- **robots.yaml is placed in the persistence-config §1 layer table** as declarative operator intent that lives in the data home, which is why `MAXIM_DATA_HOME` isolates it (Exp 49).
- **The default lives in robots.yaml `default:`, not config.json** (D2′). That avoids a `CONFIG_FORMAT_VERSION` bump, a `CONFIG_WRITER_MATCHES` change, the #974 refusal by a 1.3.4 build that shares the config directory, an isolation escape, and a layer violation (architecture #5, wire F6).
- **Layers** (architecture #8):
  - `hardware/` holds the entry types, the pin, the resolver, the probe, the sanitiser and the writer, and imports nothing upward.
  - `maxim/devices.py` is the API layer (a new `LAYER_RULES` entry like `api.py`). It holds the name validator that reads mesh.yml names, the verbs and `plan_launch`.
  - `cli.py` dispatches to it. `device`, `discover`, `connect` and `embody` go in `_NON_SIM_SUBCOMMANDS` (#1344).
  - `_main_impl` and `Maxim.__init__` are over 200 lines, so every edit there moves code out (touch rule).
- **Hardware safety.** `check`, `discover`, `relocate`, `repin` and the preflight open no `ReachyMini(`. Any new `ReachyMini(` passes `automatic_body_yaw=False` and undoes `no_media` in the same file (lint rules 3/4).
- **Optional deps.** Discovery goes through `require_optional_dependency("reachy_mini")`, and no direct zeroconf in v1 (D7).
- **M76.** The non-Reachy-controller tripwire does not trip, and phones never register.

## 8. Phasing (three-lens round per PR; estimates are `src/` lines excluding tests)

- **PR 0, groundwork (≈250):**
  - the §5 fixes
  - typed loud yaml errors and their propagation through selfy and agentic_runtime
  - one resolver
  - `probe_daemon` consolidation
  - explicit `--robot`, `--device` and the explicit-flag gate
  - removing the cwd fallback (D11)
  - env scrubs
- **PR 1, the pin (≈350):** typed `pin:`/`hostname:`/`launch:` fields, the registry's pin enforcement and reuse check, the controller's §4.3 sequence with typed raises and cleanup, `device_state.json`, the sanitiser, `maxim.devices` read verbs (`entries/get/check/resolve/connect`), and `maxim device list|show|check`.
- **PR 2, the writer (≈300):** `write_robots_config`, `add/rename/forget/repin/use`, `maxim connect`, `default:`, the CI allow-list and import-ban lint, and the DECISIONS C2 amendment.
- **PR 2b, `embody` (≈150):** `plan_launch`, the preflight, the plan block, carrying the verified address and pin into the existing launch, the `reexec_with_mode` `--device` carry, non-zero exit on refusal, and `launch:` + `--save-defaults`.
- **PR 3, discovery (≈200):** `discover` + `relocate` over `find_robots`, the latch check first, census and lint updates, and the macOS permission hint.
- **Separate issue #1808:** the library-first runtime seam (§3c). Then `maxim.devices.embody()` in-process, the local-media kind (the first test of the kind abstraction), and `attached_via: <node>`.

Docs that change with these PRs: `robot-setup.md`, `getting_started.md`, `cli-reference.md`, `configuration.md`, a LAN-trust section in `safety.md`, `stable_api.md` (experimental rows; `maxim.connect`/`run(robot=)` labelled unpinned), persistence-config §1, and the embodiment brief's env table.

## 9. Owner decisions

**Taken 2026-11-25:**
- D1 `maxim device …` plus top-level `discover`/`connect`.
- D2 `connect` = verify, refresh and set the default.
- D3 a new strict `--device`.
- D4 per-kind stores.
- D5 a format version plus a one-time `.bak`.
- D6 refuse on mismatch.
- D7 SDK `find_robots` only.
- D8 `maxim embody` (and `launch:` defaults never hold `operational_mode`).
- D9 v1 Python is plan-only. The in-process `embody` comes after the §3c seam.

**Taken 2026-11-25 on DRAFT v2 (the owner took every recommendation):**
- ✅ **D2′.** Store the default in robots.yaml `default:` instead of config.json. *Rec: yes.* The meaning of D2 is unchanged. Two lenses independently found the config.json route needs a schema bump, a #974 refusal by older builds, an isolation escape and a layer violation.
- ✅ **D3′.** `--robot` stays an explicit alias of `--robot-name` (today's behaviour), and only `--device` is strict. *Rec: yes.* Making `--robot` strict breaks the documented `maxim --robot reachy_mini` wherever the entry id is `primary` (the shipped template).
- ✅ **D10.** The Python write verbs are operator writers (a DECISIONS C2 amendment plus a lint banning their import from runtime, tools, agents and skills). *Rec: yes, for library parity.* The alternative is a read-only Python namespace.
- ✅ **D11.** Remove the `./robots.yaml` cwd fallback. *Rec: remove (strict).* A repo checkout or an agent with cwd write access could plant a full entry. The alternative is to keep it with a loud WARNING.
- ✅ **D12.** One `maxim.devices` namespace. `embodied_device.md`'s planned `maxim.device.run_device` folds into it under §3c. *Rec: yes.*
- ✅ **D13.** A refused or failed connect on a **named** device exits non-zero instead of running headless. An unnamed legacy launch keeps today's headless fallback. *Rec: yes (strict).*

## 10. Disposition of the design-pass findings

Abbreviations: A = adversarial, S = safety & lifecycle, R = architecture, W = wire. Every DO-NOT-BUILD and SHOULD-FIX finding is folded into the sections above, or moved to the §3c issue. Nothing is dropped.

- **Folded into §4 (the pin):** A1, A3, A4, A9, A10, A11 · S3, S4, S5, S6, S9 · R3, R4, R13 · W1, W3, W14, W18.
- **Folded into §3a/§3.1:** A2, A6 · S2, S7, S14, S15 · W2, W4, W5, W9.
- **Folded into §3b:** S10 · R6, R9, R16 · W7, W11.
- **Moved to the §3c issue:** S1, S11, S12 · R1, R2 · W10.
- **Folded into §5, §6 and §7:** A5, A7, A8, A12, A13, A14, A15, A16 · S8, S13 · R5, R7, R8, R10, R11, R12, R14, R15, R17 (no change) · W6, W8, W12, W13, W15, W16, W17, W19.
