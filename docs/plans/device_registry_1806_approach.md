# Device registry (#1806): approach note, DRAFT v8

Status: DRAFT v8, 2026-11-25. Mirrors issue #1806.

**Review history.**
- **Round 1, on DRAFT v1.** Four lenses: [adversarial](reviews/1806/adversarial.md), [safety & lifecycle](reviews/1806/safety_lifecycle.md), [architecture](reviews/1806/architecture.md) and [wire](reviews/1806/wire.md). All four returned DO-NOT-BUILD on one root cause: the identity pin rode a free-form config key and a separate REST probe.
- **Delta round 1, on DRAFT v2 at f0369379.** Fresh readers: [adversarial](reviews/1806/delta1_adversarial.md), [safety](reviews/1806/delta1_safety_lifecycle.md), [architecture](reviews/1806/delta1_architecture.md) and [wire](reviews/1806/delta1_wire.md). **None returned DO-NOT-BUILD.** Every lens found v2 had closed round 1's DO-NOT-BUILDs. Almost every new finding fell in an already-covered class (converging), but several fail open, so the stop rule was not met.
- **Delta round 2, on DRAFT v3 at 8540cf19:** [adversarial](reviews/1806/delta2_adversarial.md), [safety](reviews/1806/delta2_safety_lifecycle.md), [architecture](reviews/1806/delta2_architecture.md), [wire](reviews/1806/delta2_wire.md). No lens returned DO-NOT-BUILD, and **every finding fell in an already-covered class** (converging). The remaining fail-open items are guard mechanics: lints that cannot match, proxy resolution, the write-deny seam, unpinned-selection paths, the re-exec, and media-ownership edge cases.
- **v4 folds every delta-round-2 finding** (§12).
- **Delta round 3, on DRAFT v4** (staged on 0630487a): [adversarial](reviews/1806/delta3_adversarial.md), [safety](reviews/1806/delta3_safety_lifecycle.md), [architecture](reviews/1806/delta3_architecture.md), [wire](reviews/1806/delta3_wire.md). No lens returned DO-NOT-BUILD, and every finding was in a covered class. The remaining fail-open items are confined to §4.3's media mechanics, the lazy-attribute lint, and selection/phasing.
- **v5 folds them** (§13), including an owner-approved simplification. The **pinned-registry rule** replaces several path-by-path rules, and `--device`, its re-exec, `--unpinned` and the pin refusal ship together in PR 1.
- **Delta round 4, on DRAFT v5 at e1fc3308:** [adversarial](reviews/1806/delta4_adversarial.md), [safety](reviews/1806/delta4_safety_lifecycle.md), [architecture](reviews/1806/delta4_architecture.md), [wire](reviews/1806/delta4_wire.md). No lens returned DO-NOT-BUILD; all findings were in covered classes, and narrower again. **v6 folds them** (§14). A **delta check (round 5)** is owed, scoped to §14's fail-open folds only.
- **Round 5 check, on v6 at b0cbd8f7:** [adversarial](reviews/1806/delta5_adversarial.md), [safety](reviews/1806/delta5_safety_lifecycle.md), [architecture](reviews/1806/delta5_architecture.md), [wire](reviews/1806/delta5_wire.md). **The architecture lens met the stop rule** (§3b(d) is closed). The other three left 2–3 narrow fail-open items each, all in covered classes. **v7 folds them** (§15). A **round 6 check** is owed for the adversarial, safety and wire lenses only.
- **Round 6 check, on v7 at 352ea398:** [adversarial](reviews/1806/delta6_adversarial.md), [safety](reviews/1806/delta6_safety_lifecycle.md), [wire](reviews/1806/delta6_wire.md). **The adversarial lens met the stop rule**, so with architecture two of four lenses are closed. Safety (H1, H2) and wire (W6-1) each left narrow fail-open items in covered classes. **v8 folds them** (§16). A **round 7 check** is owed for safety and wire only.
- **Round 7 check, on v8:** [safety](reviews/1806/delta7_safety_lifecycle.md) **met the stop rule**, so architecture, adversarial and safety are closed. [Wire](reviews/1806/delta7_wire.md) left one narrow item: doctor's private copy of the address default (W7-1). v8 folds it (§16 addendum). A **wire-only check of W7-1** was owed.
- **Final wire check:** [delta8_wire.md](reviews/1806/delta8_wire.md), **stop rule met. All four lenses are closed, and the design gate is CLOSED** (2026-11-25). The owner's media-backend scope decision is folded into §4.2 and gets a one-lens confirmation check. Then PR 0.
- **Spun out as #1811 (security):** the per-mode filesystem policy (`check_write_permission`) has no production caller. v4's PR 0 adds the control-plane deny and the `edit_file` approval fix on the same seam. Wiring the whole policy needs its own design pass.

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
maxim ... --unpinned                # explicit grant: an unpinned entry on a new surface (any registry), or a non-pinned resolution in a pinned registry (§3.1)
maxim embody <name> [--dry-run] [--unpinned]   # preflight + launch embodied (§3a)    [experimental]
```

- **Legacy flags.**
  - `--robot` is an explicit alias of `--robot-name`, with today's behaviour (D3′). Defining it explicitly makes `--rob`/`--robo` ambiguous abbreviations, which is documented (wire N7).
  - `--robot-name`'s argparse default becomes `None`. `MAXIM_ROBOT_NAME` is read at resolution time, not by argparse, so a bare `maxim` with that variable set no longer enters the robot runtime (wire F4).
- **The quick-start gate reads parsed, explicit flags.** It checks `args.device`, an explicit `--robot`/`--robot-name`, `--mode` and the `embody` subcommand. It never reads a resolved primary or environment variables. The gate's test calls the real `cli.py` gate.
- **Selection precedence** (v4; wire W2-2) applies only once a robot run is requested. **Typed flags always beat environment variables.**
  1. `--device <name>`
  2. legacy typed `--robot` / `--robot-name` (see "Legacy matching" below)
  3. `MAXIM_DEVICE`
  4. legacy `MAXIM_ROBOT_NAME`
  5. no name (see "Legacy matching" below: the default `reachy_mini` id, else the unambiguous primary)

  Steps 1 and 3 are **explicit names**: they match an entry exactly, or the run is refused. `--device` given together with **any** typed `--robot`/`--robot-name` is refused, whatever the values; that is the whole definition of "conflict". An environment variable outranked by a typed flag is ignored, and the startup log says so. The plan block and the startup log name the source that chose the device (adversarial N12).
- **The pinned-registry rule** (v5, owner-approved simplification; replaces v4's path-by-path rules; wire W3-2, W3-3, adversarial N3, W2-7). It is one rule, enforced in the one resolver:
  - **In a registry that holds any pinned entry**, a resolution that does not land on a pinned entry is **refused unless `--unpinned` is typed**. That covers no entry, an unpinned entry, a legacy name that matches nothing (today that dials `<name>.local`), and no name with no `primary:`. The refusal exits non-zero (2). Its hint reads: "`<what>` is not a pinned entry, and this registry pins robots. Run it anyway with `--unpinned`, or pin it with `maxim device add` / `maxim device repin`."
  - **In a registry with no pinned entry**, including no robots.yaml at all, every legacy flow is unchanged, including the headless fallback. Users meet the rule only after they opt into pinning, and pinning arrives in PR 1.
  - **Legacy matching is today's, unchanged** (v6; wire W4-2). Steps 2, 4 and 5 use exactly today's `resolve_robot_entry`: an exact entry id, else the unambiguous primary. A bare run still exact-matches an entry keyed `reachy_mini`, the default name, before the primary. v6 adds no `robot_name` matching. The only legacy change is the pinned-registry refusal above, and the changelog fragment lists it.
  - **The new surfaces are always strict** (v6; wire W3-3). `--device`, **`MAXIM_DEVICE`**, `embody` and `maxim.devices.open_controller` need `--unpinned` (or `unpinned=True`) for an unpinned entry **in any registry** (v7; wire delta-5 W5-2). The legacy surfaces (`--robot`, `--robot-name`, `MAXIM_ROBOT_NAME`, no name) follow the pinned-registry rule only.
- **Where a run's daemon name comes from** (v6; wire W3-5, W4-1). This is **today's chain, unchanged**:
  1. the entry's `config.robot_name`;
  2. else the typed `--robot-name`;
  3. else `MAXIM_ROBOT_NAME`;
  4. else `reachy_mini`.

  It is **never the entry id**. v5's "else the entry id" would have made an entry keyed `primary` dial `primary.local`. For a **pinned** entry the daemon name never chooses the address, because pinned candidates come only from `host:`/`hostname:` (§4.2).

  **The chain's steps 2–4 (typed name, `MAXIM_ROBOT_NAME`, the `reachy_mini` default) apply only when a legacy source selected the run** (v7 W5-1; v8 wire delta-6 W6-1). **Any other selection** of an unpinned entry with no `host:` and no `config.robot_name` is **refused**. That covers the new surfaces **and every `maxim.devices` verb**: `resolve`, `use`/`maxim connect`, `check`, `plan_launch`, and `repin` without a host (wire delta-7 N7-1). The refusal hint says `maxim device repin <name> <host>`. Without this, `maxim connect bob` would verify `reachy_mini.local` and move `primary:` to `bob` ("`<name>` has no address: add `host:` or `config.robot_name`, or `maxim device add` it"). It is not mDNS-dialled under an unrelated `MAXIM_ROBOT_NAME`, which could reach a different robot while the log names `<name>`. The writer always emits `config.robot_name`, set to the **daemon-reported `robot_name` it verified at `add`** (wire delta-6 N6-3), so registered entries never hit this. The test rows run with `MAXIM_ROBOT_NAME=alice` and cover: `--device bob --unpinned`, `MAXIM_DEVICE=bob` (including through a re-exec), `devices.resolve("bob")` and `maxim connect bob` (N6-1).

  **Doctor too** (v8; wire delta-7 W7-1): `doctor/checks.py::check_robot_reachable` and the no-name `maxim device check` take each entry's address from the **one resolver**, as a non-legacy walk. `_check_one_robot`'s private `robot_name or "reachy_mini"` + `.local` default is deleted. An addressless unpinned entry reports `warn: no address` **for that entry alone** and is never reported as "reachable" (N7-2). A test row patches `gethostbyname` to succeed and asserts the report says "no address". **A known, intended disagreement** (wire delta-8 N8-1): consider a working legacy single-robot setup whose entry has no `host:` and no `config.robot_name`, in a registry with no pin. Doctor now shows `warn: no address`, while a legacy run still connects through the legacy chain (`MAXIM_ROBOT_NAME` or `reachy_mini` + `.local`). So doctor under-reports, never over-reports. The warn's fix hint names `maxim device add <name> <host>`, or setting `config.robot_name`. The changelog fragment states this. **Tests re-pointed** (N8-2): `tests/unit/test_doctor.py::test_resolution_failure_macos_fix_mentions_local_network` and `::test_resolution_failure_linux_fix_omits_local_network` use an entry with no `robot_name`, so they move to an entry **with** `config.robot_name`. That keeps the mDNS-failure path and its macOS Local Network hint covered, rather than weakening the guard.
- **`robot_id` and `robot_name` are separated end to end.** The CLI passes `Maxim(robot_id=<entry id>, robot_name=<daemon name>)` (architecture R2).
- **Re-exec repeats the parent's choice** (v4; wire W2-3, W2-4; safety S7):
  - **It replays how the parent selected its robot** (v6; wire W4-1, W4-3):
    - a `--device` parent: the child gets `--device <id>`;
    - a parent that used a typed legacy flag: the child gets that same typed flag and value, so a legacy fallback re-resolves identically;
    - an env-only or no-name parent: no robot flag, and the child inherits the same environment through `child_env()` and resolves the same way.
  - The child gets `--unpinned` **whenever the parent typed it**, whether or not an entry resolved. "Granted" means typed.
  - Never an environment-derived argv value, and never `--robot-name None`.
  - The child re-resolves and re-verifies, so a `maxim connect` run in the meantime cannot switch the robot.
  - **Ordering** (v5; wire W3-1, W3-4, W4-4). PR 0 ships only the legacy fixes: explicit `--robot`, `--robot-name` default `None`, the gate rewrite, and a re-exec that omits `None`, all in one commit. **`--device`, the `MAXIM_DEVICE` reader, its re-exec rule, `--unpinned` and the pinned-registry refusal ship together in PR 1**, so no build has `--device` without its re-exec, or a refusal without its escape. Test rows cover: a `--device` parent keeps the device, `--unpinned` is carried, a typed name with no entry, no entry without a typed name, env-only, and the pinned-registry refusal.
- **Help text and docs.** `maxim connect` (devices) is distinct from `maxim peer connect`, and the help says so.

### 3a. `maxim embody <name>`

1. **Resolve and verify** (§4). An unknown name is refused, and `primary` does not change.
2. **Preflight.** These are REST/TCP-only checks through `probe_daemon` + `refusal`: no `ReachyMini(` and no tunnel.
   - **Refuse** on: an identity mismatch, an unreachable daemon, an era or minor-version mismatch, a daemon whose state is not `running`, or a sim/mockup daemon for a hardware-pinned entry.
   - **Warn** on `no_media`, or on `media_released` when the run mode needs perception.
   - For a `tunnel:` entry, the same `refusal` runs inside the controller at connect (§4.3 step 1). The plan block says "checked at connect (tunnel)" rather than "verified" (safety N5).
3. **Plan block.** It shows: the device and the source that selected it, the verified IPv4, `hardware_id` (or **UNPINNED**), the body ref, the run mode, the operational mode and its source (typed flag, or the run-mode default), whether the head will move, the LLM and whether it is a cloud model, and the media backend. `--dry-run` stops here.
4. **Launch** through the existing robot path, with the **verified address and pin carried forward as typed arguments**; nothing is re-resolved.

**Unpinned runs follow §3.1's pinned-registry rule** for `embody` as for every other path. `embody` additionally always requires `--unpinned` for an unpinned entry, even in an unpinned registry, because `embody` is the new strict verb (adversarial N3, safety S5, wire W3-3).

**Exit on refusal, defined by pin status (D13′).** These exit non-zero:
- a refused or failed connect on a **pinned** entry, however it was selected;
- a pinned-registry refusal (§3.1);
- a failed connect on an entry chosen by a new surface (`--device`, `MAXIM_DEVICE`, `embody`, `open_controller`), pinned or not.

Only a legacy launch in a registry with **no** pinned entry keeps today's headless fallback (safety N8, wire N10, adversarial N7). `test_cli_exit_code_1468.py` gains rows for each case.

**Perception without media** (safety delta-3 E5, strict default, owner-confirmed; enforcement v6, safety delta-4 F2).
- **Where it is enforced.** `media_required` is a keyword-only **per-call** argument, `connect(*, media_required=False)`. It is never instance state, so `reconnect()`/`_soft_reconnect` never inherit `True` (safety delta-5 N2). When it is `True` and §4.3 step 1 would force `no_media`, step 1 **raises `DeviceMediaUnavailable`** before `ReachyMini(` instead of forcing it. That exits non-zero under D13′.
- **Postcondition** (v7; safety delta-5 G1). The SDK also selects NO_MEDIA **without being forced**, for a configured `media_backend: no_media` or a daemon that reports `no_media=True`. So when `media_required` is set and `self._mini.media.camera is None` after `ReachyMini(` returns (before `CONNECTED`), the controller raises `DeviceMediaUnavailable` through the normal cleanup. **A configured `no_media` is not a grant**; only the per-run `--no-media` is (strict default). Red gates cover both cases.
- **Liveness** (v8; safety delta-6 H1). `camera is not None` is only the fast first test, because a camera can be constructed but dead:
  - LOCAL's `open()` waits up to 2 s for a first frame inside a bare `try/except` and returns either way, and `MediaManager` "continues without camera" on init errors;
  - WEBRTC with a producer whose video never flows only sets the pipeline to PLAYING.

  So with `media_required` the controller also requires `media.get_frame()` to return non-`None` within a bounded deadline (5 s, polled) before `CONNECTED`, else it raises `DeviceMediaUnavailable`. **Any exception** from `get_frame()` during the poll (e.g. `ValueError` from a caps mismatch in `reshape`) also counts as `DeviceMediaUnavailable` (safety delta-7 N7-2). Red gates cover a LOCAL camera that constructs but never yields, and a WEBRTC producer with no frames. The LOCAL fake must model that `open()` already pulls and discards the first frame, so a camera that yields only during `open()` fails liveness (N7-1).
  - **Residual, stated:** liveness is checked **at connect only**. A reconnect passes `media_required=False` and degrades (§4.5), so the strict-perception guarantee is a connect-time guarantee.
- **Placement** (N3). The postcondition and the liveness check run **after** `_media_released_by_connect = _connect_released_media(status)`, so the cleanup re-acquires a release that a configured `no_media` made. `connect`'s broad `except` re-raises the typed error (§4.3 step 4) and never returns `False`.
- **Who sets it.** `embody` in a perception run mode (`exploration`, `live`) sets it. `--no-media` clears it. `open_controller` defaults to `False` and warns.
- **Preflight** runs the same decision function, but only as an early advisory. It cannot see tunnel entries or a later `wlan_ip` change, so it is never the enforcement point.
- **Red gate:** a tunnel entry with `wlan_ip: None` in `exploration`, without `--no-media`, refuses before `ReachyMini(`.

**Defaults.**
- The run mode is `exploration`.
- `launch: {mode, llm, interactive}` is an entry-level sibling of `config:`.
- **`launch.mode` must be an embodied run mode**: `exploration`, `live`, `sleep`, `train` or `reflection`. These come from **one shared constant** that argparse's `choices` also uses. `agentic` is refused for `embody`, because it runs the CLI agent loop, which never opens the device, so the plan block would be false (safety delta-2 D7). Operational names such as `passive`, `active` and `singularity` are refused for the entry (safety N1).
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
ctl = maxim.devices.open_controller("rosie")      # a live RobotController through RobotRegistry + the pin (§4.3); has daemon side effects
plan = maxim.devices.plan_launch("rosie", mode="exploration")   # LaunchPlan
```

- **Names.** The live opener is `devices.open_controller`. It is not `connect`: CLI `maxim connect` means "make primary", which is `devices.use`. It is not `open`, which would shadow the builtin (architecture N7, delta-2 F6). It is a **session verb**, not a read verb: it opens a daemon session, sends `SetAutomaticBodyYaw(False)` and may release media. It goes through `RobotRegistry`, which also covers a `KeyboardInterrupt` during the step-3 wait (safety delta-2 D5).
- **The stable APIs do not change.** `maxim.connect(robot_type, config=…)` and `maxim.run(robot=)` stay **unpinned**, and this is documented and logged.
- **Who may write (D10).**
  - Write verbs live in `maxim/devices/_write.py`. Python write verbs are operator writers, which needs an amendment to C2 in `docs/agents/persistence-config.md` §1 and to its writer table (architecture N2).
  - **Enforcement rides the architecture audit** (v4; architecture delta-2 F1–F3, wire W2-1):
    - (a) **Writer callers.** An AST caller allow-list for `hardware/config.py::write_robots_config` permits `maxim/devices/_write.py` and the writer's round-trip test, next to `MESH_WRITER_MATCHES` in `test.yml`.
    - (b) **Import ban through `LAYER_RULES`.** `"devices"` joins `must_not_import` for `runtime`, `tools`, `agents`, `planning`, `memory`, `bridges` and `hardware`, plus a **new `embodied_runtime` entry** (which still allows `hardware`). The audit already catches every static import spelling (`from maxim.devices import add`, `import maxim.devices._write`, aliases).
    - (c) **The `devices` entry gets its own non-empty forbidden list**: `["runtime", "tools", "agents", "planning", "console", "cli", "cli_parser", "cli_utils"]`. That **mechanizes "the library does not depend on the CLI"** for this package. `embodied_runtime` is left off the list because #1808 needs it. Both rules are proven by deletion with a `tmp_path` tree, as `test_architecture_audit.py` does.
    - (d) **Lazy-attribute path, receiver-agnostic** (v6; architecture delta-4 D4-1). This replaces v5's alias tracking. It rides `utils/audit.py::audit_architecture`, which already walks exactly the `LAYER_RULES` layers and has the baseline and `tmp_path` probe machinery, so no new `scripts/lint_*` is written. In every layer that bans `"devices"`, it flags:
      - **any** `ast.Attribute(attr="devices", ctx=Load)`, **on any receiver**;
      - `from maxim import *`;
      - a literal `maxim.devices*` argument to `importlib.import_module` / `__import__` (detection copied from `lint_http_surface.py::imported_modules`).

      There are **zero hits** across the 8 banned layers today. A legitimate unrelated `.devices` attribute in those layers would need a rename or a reviewed baseline entry. With no alias table there is no scope tracking to get wrong. The proof is the delta-4 review's 15-spelling probe table, run as a deletion probe.

      **Residuals, stated** (D4-2): `getattr(maxim, "devices")`, `vars(maxim)["devices"]`, `maxim.__dict__["devices"]` and `sys.modules["maxim.devices"]`. A star import is caught above, and by ruff F403.

      **Implementation notes** (architecture delta-5; stop rule met, fail-closed):
      - **D5-1.** Also flag any `str` constant equal to `"devices"`/`".devices"`, or starting with `"maxim.devices"`/`".devices"`, in a banned layer. That has 0 hits today, and it catches every literal spelling, including `getattr`/`vars`/`sys.modules` with literals and a relative or aliased `import_module`. What remains is the class of **computed** names, which is stated as a residual.
      - **D5-2.** Compare the star import's resolved module (`_resolve_from_module(...) == "maxim"`), so `from .. import *` is caught.
      - **D5-3.** A finding carries a non-empty `names`, a valid `scope` from `SCOPES`, and a distinct rule marker. The deletion probe asserts the residual rows return 0.
    - `test_architecture_audit.py::TestLayerRulesCoverage::EXPECTED_LAYERS` gains `devices` and `embodied_runtime`. `ARCHITECTURE.md`'s `hardware/` row adds `devices` to its bans, and it gains an `embodied_runtime` line (D3-2).
    - (c) is a **direct-import** rule. An import-closure test, shaped like `tests/integration/test_advisor_gate.py::test_import_closure`, proves that `maxim.devices` pulls in no `cli*` or `console` module. It **runs in a child process** (pytest's `sys.modules` is already polluted), imports every submodule via `pkgutil.walk_packages` (including `_write`), and asserts by prefix that no `maxim.cli`, `maxim.cli_parser`, `maxim.cli_utils` or `maxim.console*` module loads (D4-3). That keeps the CLI ban true transitively, including after #1808 adds `devices → embodied_runtime` (D3-3).
    - **Dropped:** v3's `BLOCKED_PYTHON_IMPORTS` lever. It matches only top-level names, and lazy loads skip the hook; the sandbox is documented as not a security boundary.
  - **Scope of the claim.** No module in the agent's runtime layers can statically import the write verbs. An agent whose BashTool may run `maxim device add` is operator-equivalent, and the operational-mode gate governs that (passive refuses `bash`). D14 covers the files against the agent's **file tools**. `bash` in singularity mode can still write them, and that residual is stated in `safety.md` (adversarial delta-2 NF3).
- **Python confirmations.** `relocate` and `repin` require a `confirm=` callback, which receives the **same context object the CLI prints** (a `RepinContext` / `RelocateContext`: old and new id, model, version, `wlan_ip`, last seen). An optional `expected_new=` refuses any other id. On the CLI they need a TTY, or `--yes` (adversarial N10, delta-2 N1).
- **v1 has no in-process launch** (D9). It arrives with #1808 as `maxim.devices.embody(...)`.
- **`console/device.py::run_device` stays in `console/`** (D12′, architecture N12). It is a node/host concern, and folding it in would pull console/FastAPI imports into `maxim.devices`. At most there is a lazy re-export.
- **Types.** `DiscoveredDevice`, `DeviceEntry`, `ResolvedDevice`, `DevicePin` and `LaunchPlan` are frozen dataclasses. They live in `hardware/devices_types.py`, so runtime and `embodied_runtime` can import them without importing the write verbs.
  - `DeviceEntry` is built from a `RobotConfig` by one copying constructor, with `config` as a `MappingProxyType`. It is not a "view" over it (architecture N11).
  - All five types take CC3 path (a).
  - Errors are typed and subclass the stable `MaximConnectionError` / `ConfigurationError`: `DeviceNotFound`, `DeviceIdentityMismatch`, `DeviceIdentityDivergence`, `DeviceUnpinned`, `DeviceUnreachable`, `DiscoveryUnavailable`, `RobotsConfigError`. `_run_stopped_on` re-raises them as typed exits.
- **Stability.** The namespace is ⚠️ Experimental in `stable_api.md`. **Every file** in `maxim/devices/` goes into `scripts/run_mypy_public.sh`. `ARCHITECTURE.md` gains a `devices/` row (architecture delta-2 F8).

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
  - Its value must be 16 lowercase hex characters, or the literal `unpinned`, which the writer emits (adversarial delta-4 R4-8). Deleting `_format_version` together with the pin lines downgrades the file to legacy and switches the pinned-registry rule off. That requires robots.yaml write access, which is already enough to forge a pin, and D14 denies it to the agent's file tools. It is a stated residual.
  - An absent key means `DevicePin.UNPINNED`, a sentinel and never `None`, **but only in a hand-written legacy file** (no `_format_version`). In a **writer-managed** file (one that has `_format_version`), `hardware_id:` is **required** on every entry, and the writer emits `hardware_id: unpinned` explicitly. A deleted pin line is then a `RobotsConfigError`, not a silent switch-off of §3.1's pinned-registry rule (adversarial delta-3 R3-9). An empty or malformed value is always a `RobotsConfigError`.
- **Strict entries** (wire F1b, W2-5). The allowed entry-level keys are exactly `{type, primary, config, kind, hardware_id, hostname, launch}`. Anything else (a typo `hardware_idd:`), or `hardware_id`/`hostname` placed under `config:`, is a `RobotsConfigError`, so a typo can never yield a silently unpinned entry. A strict-keys red gate is verified by a deletion probe.
- **Typed scalars** (architecture delta-2 F4, F7):
  - `primary:` must be a YAML bool, checked as `type(v) is bool`, so `primary: 1` and a quoted `"false"` are refused. PyYAML (YAML 1.1) already parses unquoted `yes/no/on/off` as bools, and those are accepted. More than one `primary: true` is a `RobotsConfigError`; it is not "first wins" (D3-4).
  - A non-string `hardware_id:` (an unquoted all-digit id loads as an int, and as octal with a leading 0) raises with a "quote it" hint and is never coerced.
- **How the pin reaches the controller.** The registry passes it as an explicit keyword, `expected_identity: DevicePin`, outside the constructor-signature filter. A controller class whose constructor has no such parameter is **refused** for a pinned entry.
- **Post-connect check** (safety N3, adversarial N11). After **every** connect, the registry requires `controller.verified_identity() == pin`. This is a typed accessor that defaults to `UNPINNED`, is **reset on connect entry and in `_drop_handle`/`disconnect`**, and is set only after §4.3 step 3 (safety delta-2 D6). This replaces trusting an inheritable `supports_identity_pin` flag.
- **The reuse branch** ("already connected") refuses, with a typed error and before `set_primary`, when it finds a mismatch **or a controller that was never verified** (safety S9).
- **An unknown `type:` on a pinned entry** raises `RobotsConfigError` rather than returning `None` before the pin check (wire N10).
- **Load errors propagate** (wire N1, F18; adversarial N6). `from_yaml` / `from_dict` raise `RobotsConfigError` on:
  - a parse error;
  - a top-level list;
  - a non-dict entry, or `config: null`;
  - `find_config_file`'s "data_home() unresolvable" branch;
  - a **relative `MAXIM_DATA_HOME`**.

  The relative-path check covers robots.yaml resolution and the writer only, not `utils/paths.py::data_home()`, which benchmarks call with relative work directories (wire N2). A missing file is fine and means there are no entries. `selfy` stops catching these errors at DEBUG. `agentic_runtime._resolve_body_wiring` stops reloading robots.yaml: it **receives the resolved entry** as `selfy.Maxim._resolved_robot_config`, the one attribute its four readers (the DoA gate, motor binding ×2, `aut_mode`) see. It keeps its body-ref fallback (wire N3, safety S8). The `test_multi_robot::TestResolveBodyWiring` tests and the 1496 tests that patch the now-dead `load_robots_config` are re-pointed to the handoff, so they cannot pass without testing anything. Residual: `selfy` resolves an entry only when `simulation=False` (no `src/` caller passes `True`) (wire W2-6).
- **`_format_version`.**
  - A root `_format_version: "1.0"` is written by the writer.
  - An explicit refuse-newer check modelled on `config_writer._refuse_if_newer` runs in the loader and the writer. `check_format_version` itself never refuses.
  - A present non-string value is an error: an unquoted `2.0` is a float, and it must not be read as legacy.
  - An absent key is accepted as hand-written legacy, without the "re-save" nag: the loader skips `check_format_version` when the key is absent, because it always warns (architecture N3, delta-2 N3 NIT, wire N8).
  - `test_persistence_compat` gains a robots.yaml row.

### 4.2 Resolution never trusts an advertisement

`host:` is the operator-pinned address: an IPv4 literal, or a name. `hostname:` is the name used to re-resolve when `host:` is absent (adversarial N9).

**Candidates for a pinned entry,** in order:
1. `host:`, or `hostname:`, **resolved once** to IPv4;
2. the cached last-good address for that identity.

That is all. Discovery is never a resolution step. `MAXIM_REACHY_HOST` is read only by the `maxim-diagnostics` tool and never by the registry (adversarial A15). On failure, the error names every candidate and why it failed.

**`relocate`** browses the LAN and shows each candidate's claimed identity. It writes nothing until the operator confirms, and it re-verifies the chosen candidate through REST at that literal before writing.

**`add` and `repin` read the identity only through `probe_daemon(proxy="never")` + `refusal` at the literal being dialled** (safety N7). Sim and mockup flags must be off, and the id must be non-null. The mDNS TXT `unit_id` is never used. **`add` refuses an id that another entry already pins**, the same rule as `repin` (adversarial delta-2 N1).

**`repin`** (adversarial N1):
- It shows the old and new id, model, version and `wlan_ip`, and when the old id was last seen.
- It refuses a new id that another entry already pins.
- It needs a TTY confirmation, or `--yes`.
- `repin <name> <host>` also rewrites `host:` to the confirmed literal, and says so.
- The mismatch error lists "check your network or DHCP" **first**, and mentions `repin` only as the audio-board-swap case.

**An unpinned entry** resolves only from `host:` or `<daemon name>.local` (the §3.1 chain). It never caches. It follows §3.1: the new surfaces always need `--unpinned`, and the legacy surfaces need it only in a pinned registry. It is labelled everywhere.

**Candidate hygiene** (adversarial #7, N9):
- Canonical IPv4 through `ipaddress.IPv4Address`; octal, hex and short forms are refused. IPv6 is out of v1.
- Refused: loopback (unless the entry is explicitly `localhost_only` or `tunnel:`), `0.0.0.0`, link-local, multicast and broadcast.
- CGNAT/Tailscale is allowed and labelled.
- The port is fixed at 8000.
- Trailing dots are normalised. At most 4 candidates are tried.

**Proxies** (adversarial N4, delta-2 NF2/NF4/NF5).
- **Our own probe.** PR 0 adds a strict mode to `utils/http.py`: `fetch_url(..., address_policy="private", proxy="never")`, a `strict=True` row in `_ensure_external_endpoint`. In that mode a private endpoint is dialled directly, never through a proxy. Today the proxy wins per #921. The new endpoint is registered in `_EXTERNAL_POLICY_ENDPOINTS` / `_DEADLINE_ENDPOINTS`, so it keeps the #1715 deadline.
- **The SDK's two clients.** The WebSocket (`websockets`) and the SDK's `requests` POST resolve proxies **differently**. The probe-verified disagreements: `no_proxy` CIDR, `ws_proxy`, `ip:port` forms, and macOS system proxies. So the check does not parse variable names. **It asks each library's own resolver:**
  - `websockets.uri.get_proxy(ws://<ip>:8000/ws/sdk)`
  - `requests.utils.get_environ_proxies(http://<ip>:8000/…)`, which includes `urllib.request.getproxies()` for macOS system settings.

  A pinned connect is refused unless **both** say direct. Implementation notes (R3-2):
  - `get_proxy` takes `parse_uri(url)`, not a string.
  - "Direct" for requests means `select_proxy(url, get_environ_proxies(url)) is None`, because the dict can be non-empty for an unrelated scheme.
  - Any exception from a resolver **refuses**.
- **Media-backend allowlist** (v6; adversarial delta-4 R4-1). The SDK's `local` backend, and the deprecated aliases that resolve to LOCAL (`gstreamer*`, `sounddevice_*`), read the **Maxim host's** camera IPC socket and host audio whatever the `connection_mode`. On a networked pinned connect, perception would come from the wrong body, and it opens no URL for the proxy check to see.
  - For a pinned entry that has **not passed the onboard proof** (v8; safety delta-6 H2, which replaces v6's "not `localhost_only`" label test), `media_backend` must be `default`, `auto`, `webrtc` or `no_media`. The proof is static (a local `get_hardware_id()` compared with the pin), so it runs at load (`RobotsConfigError`), for the `selfy` `media_backend=` parameter, and in the controller. A non-string value (YAML `no` parses as a bool) is a `RobotsConfigError`. A red gate sets an explicit `local` and one deprecated alias on an unproven manual-tunnel entry and checks they are refused.
  - LOCAL is allowed only for an entry that passes the **onboard proof** (§4.3 step 1), where the host's camera **is** the robot's.
  - **Unpinned networked entries too** (owner, 2026-11-25). Any entry, pinned or not, whose effective `connection_mode` is `network` has `media_backend` `local` (or a LOCAL alias) **refused** at load and in the controller. That includes a manual tunnel that fails the proof. On a networked robot a host-local camera is always the wrong body. `localhost_only` entries that cannot be proven (a Lite on a Mac has a null `hardware_id`) keep today's behaviour, labelled UNPINNED. This changes legacy behaviour for unpinned networked entries with `local`, and the changelog fragment lists it. A red gate covers an unpinned networked entry with `media_backend: local`, which must be refused.
  - **Matching mirrors the SDK exactly** (adversarial delta-5 R5-1). The check classifies `value.lower()`, exactly the SDK's normalisation, with no strip, NFKC or casefold. The controller passes the matched **canonical literal** to `ReachyMini(`, so Maxim's classification and the SDK's can never desync (e.g. `" no_media"`).
- **The media clients too** (adversarial delta-3 R3-1). The chosen media backend opens more clients: the SDK's `media/webrtc_utils.py::get_producer_list` connects to `ws://<signalling_host>:8443` with `proxy=True`, and the GStreamer signaller's proxy behaviour is unverified. The check therefore asks the same resolvers about **every URL that backend will open**. With network mode forced, `default`/`auto` always resolves to WEBRTC. WEBRTC opens three clients: the Python `websockets` producer lookup at `:8443` (checkable), the Rust GStreamer `webrtcsrc` signaller on the same URI (**not** checkable), and `requests` to `:8000` (covered). So the v6 rule (adversarial delta-4 R4-2) does not ask per-URL resolvers about the signaller. **Media is allowed only when no proxy is configured anywhere.** That means no `*_proxy`/`all_proxy` env var in either case, and an empty `urllib.request.getproxies()`. Otherwise `no_media` is forced for this connect, or `DeviceMediaUnavailable` is raised when `media_required` (§3a). Motion is unaffected either way. The per-URL check still guards the `:8000` control connection.
- **Fix hints.** The hint names `NO_PROXY=<ip>`, or `NO_PROXY=127.0.0.1` for a tunnel entry, since the tunnel's `ws://127.0.0.1` is proxied too.

### 4.3 The check that counts is on the session that moves the robot

`ReachyMiniController.connect` runs these steps for a pinned entry:

1. **Pre-filter, before any `ReachyMini(`.** `probe_daemon` at the chosen IPv4 literal, then `refusal(outcome, entry)` (identity, sim flags, era/minor, daemon state). A null `hardware_id` never matches.

   **The media target is checked too**, because `ReachyMini.__init__` opens WebRTC to the status `wlan_ip` and calls `camera.open()` before any later step can run (adversarial N5). The v4 rule replaces v3's strict equality, which wrongly refused wired, tunnel, Tailscale and Lite setups (adversarial delta-2 NF1, safety delta-2 D3):
   - `wlan_ip` equals the dialled IP: accepted.
   - `wlan_ip` differs, **or the entry is a tunnel**: accepted only if a REST probe at `wlan_ip:8000` (same hygiene, `proxy="never"`) returns the **same pin** with sim flags off.
   - That probe fails: `no_media` is forced **for this connect only**.
   - **`wlan_ip` is `None` forces `no_media` for every networked entry kind** (verified: `_configure_mediamanager` uses `signalling_host = daemon_status.wlan_ip or "localhost"` on port 8443, so signalling would go to the **Maxim host**, never the robot). For a tunnel that is whatever listens on `localhost:8443`, such as a local sim daemon, which would make the camera stream the wrong body (safety delta-3 E4). A red gate covers this. **Onboard carve-out, gated on proof** (v6 F4; v7 safety delta-5 G2). A `localhost_only` entry reports `wlan_ip: None` both when Maxim runs on the robot itself **and** through a manual SSH tunnel, which the controller's own docstring suggests as `tunnel: false, connection_mode: localhost_only`. The label proves nothing, so the carve-out requires a **positive same-host proof**: the SDK's own `reachy_mini.utils.hardware_id.get_hardware_id()`, run **locally** on the Maxim host, must equal the pin. On the robot it matches. Through a tunnel it returns `None` or a different id. Any exception from the call (`PermissionError` from `iterdir`, a non-ASCII serial) means **not proven** (N4). The entry is then treated as a **manual tunnel**: network mode to `127.0.0.1`, no LOCAL, and E4 forces `no_media`. Only a proven-onboard entry skips E4 and may use LOCAL. `get_doa_reader` and the `_fresh_readback` tunnel deadline also key on the **proof outcome**, not the `localhost_only` label (N5). Unpinned entries cannot be proven and get no carve-out. PR 1 changes the controller's manual-tunnel guidance to the `tunnel:` entry kind. Red gates cover: proven onboard (LOCAL allowed), a manual tunnel with a passing pin (LOCAL refused, `no_media` forced), and a stale `/tmp/reachymini_camera_socket` on the Maxim host.
   - **A forced `no_media` is per-connect** (E2). It is a local value passed to `ReachyMini(`, and it never writes `self._media_backend`. If this controller **carries** release ownership from an earlier connect and this connect is not `no_media`, it first re-acquires over REST (`POST /api/media/acquire` through `fetch_url(proxy="never")`). If that fails, it keeps `no_media` for this connect. **Ordering** (v6; safety delta-4 F1):
     - (i) the re-acquire runs only **after** step 1's `refusal` passes, at the verified literal (through the tunnel for a tunnel entry), and never at connect entry;
     - (ii) a 2xx **clears** `_media_released_by_connect`;
     - (iii) the divergence snapshot is taken **after** this step, immediately before `ReachyMini(`;
     - (iv) E2 runs **after every `no_media`-forcing decision** (step 1's rules and §4.2's proxy rule), and a 2xx also sets the recorded pre-connect `media_released` to `False`, so a later SDK release is classed as ours, not "another client's" (safety delta-5 N1).

     A red gate runs: E2 re-acquire succeeds, then divergence, then a third-party release, then a reconnect, and checks that `acquire_media` is never called. That way a carried release is never silently dropped and the camera never left dead. A red gate runs connect A (forced), then connect B (default), and checks the ownership.
   - **Wording** (E5). A forced `no_media` releases the camera **daemon-wide**, for every client of that robot, and a SIGKILL leaks it. The release is ours only when the pre-connect status said `media_released: false`, which is the #1704 rule. The WARNING names the release and the recovery (`POST /api/media/acquire`). §3a covers the perception-mode refusal.
2. **Dial the same literal with `connection_mode="network"` forced.** A tunnel entry dials `127.0.0.1` through this controller's own tunnel. **A `localhost_only` entry** (v7, G2) uses `connection_mode="localhost_only"` only when it passes the onboard proof. Otherwise it is a manual tunnel and dials `127.0.0.1` in network mode. `_wait_local_port` also checks `_tunnel_proc.poll()`, so a foreign listener on :8000 fails there, and is caught by the pin if it does not (safety N10).
3. **Re-verify on the session, immediately after `ReachyMini(` returns.**
   - It runs after the `_media_released_by_connect` assignment, and before `_assert_auto_yaw_off`, `_clear_latch_after_connect` and the `CONNECTED` state (safety S3).
   - It reads `self._mini.client.get_status(wait=True)` with a bounded timeout (a timeout refuses).
   - The status must report the pinned `hardware_id` and sim flags off.
   - A session that reports no `hardware_id` refuses with a distinct reason, "identity not reported by session" (safety N9), not as a mismatch.
   - On success, the controller sets `verified_identity`.
   - **A step-3 timeout, or a session that reports no identity**, is most likely the pinned daemon itself (a slow link or an old daemon). So the controller **re-probes REST at the literal** (safety delta-2 D2):
     - Same pin **and**, for "identity not reported", the session's own status (`version`, `wlan_ip`, `state`) equal to step 1's REST values: our own robot. This takes the normal cleanup, which honours this connect's own media ownership. It raises `DeviceUnreachable` (timeout) or `DeviceIdentityUnreported`. If the session fields differ, it takes the divergence path; a split at one IP (an IP conflict or ARP flap) is otherwise a stated residual (safety delta-3 E3).
     - Different pin or unreachable: it takes the divergence path below.
4. **Failure handling.** The broad `except` cleans up, then **re-raises**; it never returns `False` into "headless mode". **Every non-identity failure is wrapped as `DeviceUnreachable(...) from e`**, so a raw SDK `ConnectionError`/`TimeoutError` (a Wi-Fi blip) reaches §4.5 typed (safety delta-2 D4).
   - **Before `ReachyMini(`** (step 1 or 2 refusal): nothing was touched, so it raises `DeviceIdentityMismatch` or `DeviceUnreachable`.
   - **After `ReachyMini(`** (step 3 fails after step 1 passed): this means **REST and the session reached different daemons**, which is an active anomaly. It raises `DeviceIdentityDivergence` (safety N2). The media ownership that was recorded is the other daemon's REST pre-state, so it is not trusted:
     - **do not** re-acquire or release media on the session's daemon. Reacquiring could steal a deliberate release, while a leftover release can be undone with one command, so leaving it is the safer error;
     - `_media_released_by_connect` is **snapshotted before `ReachyMini(`**. The divergence path **skips the generic `except` re-acquire branch**, and runs in this order: close the media manager, `_drop_handle`, **then** restore the snapshot, then stop the tunnel. The pinned daemon's carried-over ownership is neither lost nor re-acquired on the foreign daemon. A red gate checks that the fake's `acquire_media` is never called on divergence (safety delta-2 D1, delta-3 E1);
     - log at ERROR exactly what that unknown daemon received (`SetAutomaticBodyYaw(False)`, and for `no_media` the `/api/media/release` POST). The log names the other daemon's host and the recovery, `POST http://<host>:8000/api/media/acquire` (safety delta-2 D8).
5. **Media residual.** `ReachyMini.__init__` sends `SetAutomaticBodyYaw(False)`, plus the media release (`no_media`) or WebRTC + `camera.open()` (`default`), before step 3. **None of these moves a joint.** For motion, step 3 holds. For these side effects, step 1 (with the `wlan_ip` and proxy checks) is the only protection. **Also residual** (E6): the SDK reads the media target from the session's status, so a `wlan_ip` that changes between steps 1 and 2 is unverified, and the media path is protected only by v6's "no proxy configured anywhere" rule (§4.2), not by a per-URL check. All of this is stated in `safety.md`.

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
- **`DeviceUnreachable`** keeps the existing retry and backoff and calls `_degrade_capabilities`. **`DeviceMediaUnavailable`** cannot arise on a reconnect, which passes `media_required=False`. If it ever does, it **degrades**, as below (safety delta-5 N2).
- **`DeviceIdentityMismatch` / `DeviceIdentityDivergence` on a reconnect** means a different robot now answers at the address. The split of responsibility follows the existing code (safety delta-2 D5):
  - **The controller** calls `stop_motion()`, the **sticky** sleep latch that only a successful wake clears. This is not the disconnect latch, which the next connect clears. It takes only `_abort_lock`, which `reconnect` does not hold. Then it re-raises.
  - **`ConnectionMixin._soft_reconnect`** owns `_degrade_capabilities`, sets `_live_stop_event` and produces the typed exit.

  Nothing **ever retries an identity failure**. A test covers the "robot swapped mid-run" case.
- **A reconnect that would newly lose media** (safety delta-4 F3) **degrades; it does not stop.** Reconnects pass `media_required=False`, so a Wi-Fi blip or DHCP renew never ends the run. `_restore_capabilities` already re-derives `has_vision`. The controller logs a WARNING naming the reason and emits a run-report event marking the blind interval, so a perception run's results are never read as sighted. Ending the run at the next safe point instead would be an owner policy option.

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
- **Agent writes to the trust-bearing files** (adversarial N2, delta-2 NF3; D14).
  - **What is enforced today.** `filesystem_policy.check_write_permission` **has no production caller** (#1811): no caller passes `mode_config=` or calls `set_mode_config`, and `EditFileTool` never consults it. So today an agent's `write_file`/`edit_file` can rewrite `<data_home>/robots.yaml` (repointing `host:`/`hardware_id:`) and `~/.config/maxim/*` whenever cwd contains them.
  - **The fix rides the #957 seam.** `tools/base.py::control_plane_write_refusal(path)` sits next to `git_dir_write_refusal` and is called at the **same write sites**. Delta-3 verified these are every production agent file-write tool: `WriteFileTool`, `EditFileTool` (both of its writes), `WriteSandboxFileTool` and `WriteToSharedOutputsTool`. It applies in every mode.
    - **The denied set** (adversarial delta-3 R3-3, R3-4, R3-8):
      - the **resolved config directory**, through the same resolver `config_loader` / `peer_config_path` use, which honours `$XDG_CONFIG_HOME/maxim`; never a literal `~/.config/maxim`;
      - `robots.yaml`, `robots.yaml.bak` and `util/device_state.json` under **both** the process's `data_home()` **and** the default `~/.maxim`. Harness children set `MAXIM_DATA_HOME=<tmp>`, and the operator's real files must stay protected when a child's cwd contains them.
      - **The roots are a union** (v6; adversarial delta-4 R4-4). Each root (config dir and data home) is the union of three things: the resolver's output for the current env; the `Path.home()` default (`.config/maxim`, `.maxim`); and the account's real home, `pwd.getpwuid(os.getuid()).pw_dir`, joined the same way. This closes the `HOME`, `XDG_CONFIG_HOME` and `MAXIM_DATA_HOME=<tmp>` children.
      - **An env-independent name rule as well** (v7 R5-2; wording v8 adversarial delta-6 N6-3). The deny also refuses any realpath whose case-folded path components **end with** `.maxim/robots.yaml`, `.maxim/robots.yaml.bak` or `.maxim/util/device_state.json`, or **contain** the component pair `.config`/`maxim`, at any depth. This is a component-wise match, so `~/.config/maximal/x` and `ws/robots.yaml` stay writable. One red gate passes **only** through this rule: it patches `HOME`, `MAXIM_DATA_HOME` and the `pwd` home to a tmp dir, writes `<operator home>/.maxim/robots.yaml`, and expects a refusal. A deletion probe of the name rule therefore fails the suite.
      - **Residual, stated** (v8; adversarial delta-6 N6-1, N6-2): **only** an operator's own **non-default** `MAXIM_DATA_HOME`/`XDG_CONFIG_HOME` that a child re-points. That is launcher configuration, not reachable by the agent. A different-uid child still refuses the default files, because the name rule matches realpaths. (v7's `child_env` "mitigation" sentence had no mechanism behind it and is removed.) The HFS+-only Unicode-ignorable spelling gap is shared with #957 and stated.
    - **Matching** (R3-5, v6 R4-5) reuses **#957's ancestor walk**. It is not a leaf-only `_same_entry`, because `samefile` matches only an existing file, and that misses files inside the config dir, a not-yet-existing robots.yaml (planting a registry on a fresh machine), APFS firmlinks and NFD spellings. The walk visits **every** existing ancestor, as #957's loop does, and compares each by stat identity with each denied root. Each denied root that **does not exist yet** (a fresh `~/.config/maxim`) is anchored at its own deepest existing ancestor plus its tail. The v5 **realpath + NFC + case-fold string compare is kept** as a second, independent test (adversarial delta-5 R5-3). Red gates cover: overwriting an existing robots.yaml, a new `util/device_state.json`, a new file in an existing config subdirectory, and an absent config dir. `control_plane_write_refusal` takes the **same `real_path` string** that `git_dir_write_refusal` gets, so the check and the write agree. `~` is not expanded by either (R4-7).
    - **Hardlinks** (R4-6). A hardlink to an existing denied file is caught by inode. Creating a new one into the config dir needs code execution, which is the stated residual.
    - **Tests** go through `write_file` and `edit_file`, plus an XDG case, a `MAXIM_DATA_HOME` child case and an NFD case. The guard is verified by a deletion probe.
  - **Same pattern, folded** (R3-6). The dormant `SandboxExecutor.write_script` (reached via `create_sandbox_script`) writes an unsanitised `name` with no containment, and with no #957 check. PR 0 sanitises the name, contains the path within the sandbox, and adds both refusals.
  - **Same root cause, folded:** the `agents/autonomy.py` approval branch matches only `write_file`, so `edit_file` writes under cwd without approval. It now covers `edit_file` too.
  - **Residual, stated in `safety.md`** (R3-7): the code-execution tools can still write these files. Those are `bash`, `execute_file`, and `run_tests` when opted in with `MAXIM_ALLOW_*`. Passive mode refuses them. Wiring the whole mode policy is #1811.

## 6. Threats: every input and how it could lie

| Input | How it lies | Handling |
|---|---|---|
| mDNS advertisement (name, TXT, `address`, `unit_id`, SRV port) | Unauthenticated (RFC 6762 §21) | Hints only (`discover`/`relocate`, with operator confirmation). The pin is never read from TXT, and the SRV port is never dialled |
| TXT and status strings | Attacker-controlled: ESC/OSC, Rich markup, bidi/format characters (Unicode `Cf`), `dns-sd` whitespace key injection | One sanitiser in `hardware/`: strip C0/C1 and `Cf`, escape Rich markup, bound the length, and check TXT key shape `[a-z0-9_]+`. Used for every display and log (adversarial A12). **Never reaches a prompt:** the discovery types live in `hardware/devices_types.py`, and a new mechanization-backlog row adds a lint that prompt builders do not import them |
| REST status vs the WS session | Different daemons (auto mode, re-resolution, tunnel listener, proxy) | Step-1 `wlan_ip` and proxy checks, and the session-bound re-check with divergence handling (§4.3) |
| Cached IP | Reassigned | Re-verified every time; pinned entries only |
| Hostname | Not unique; AAAA stalls | Resolved once to canonical IPv4, then verified |
| `hardware_id` | Public; null off Linux; the real id from a sim on the robot | Not authentication; null never matches; sim flags checked; only ever read from REST at the dialled literal |
| `<data_home>/robots.yaml` | Malformed, newer, typo, a key under the wrong parent; **written by an agent's file tools** | Strict typed parse, refuse-newer, and `control_plane_write_refusal` on the #957 seam (§5; `check_write_permission` has no production caller, #1811) |
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
  - `rename` carries `primary:` with the entry. `forget` on the primary entry **clears the marker and never reassigns it**, and it says so. A later no-name run then follows §3.1's "no primary" rule (architecture delta-2 F5, wire W2-7). `device_state.json` is keyed by identity, so `rename` need not touch it, and `forget` removes the identity's row after the robots.yaml write succeeds. The two files are **not** one atomic transaction; the order is chosen so a crash leaves a harmless stale cache row (architecture N9).
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
  - explicit `--robot`, `--robot-name` default `None` (same commit as the gate rewrite and the re-exec omitting `None`), the explicit-flag gate, and the shared run-mode constant. **`--device` is not in PR 0** (v5);
  - removing the cwd fallback;
  - `fetch_url(proxy="never")`;
  - `control_plane_write_refusal` at the #957 write sites (D14), and the `edit_file` approval fix;
  - env scrubs and the harness env pops.
- **PR 1, the pin (≈400):**
  - typed `hardware_id:`/`hostname:`/`launch:` fields with strict unknown-key refusal;
  - `expected_identity=` and the post-connect `verified_identity()` check in the registry, and the reuse refusal;
  - **`--device`, its re-exec rule, `--unpinned`, the conflict refusal and the pinned-registry refusal** (v5: one PR, so no build has `--device` without its re-exec, or a refusal without its escape);
  - the controller's §4.3 sequence, with divergence handling, the per-connect media rule and typed raises;
  - the §4.5 reconnect handling;
  - `device_state.json`;
  - the sanitiser;
  - the `maxim.devices` read verbs (`entries/get/check/resolve`), the session verb `open_controller`, the `LAYER_RULES` entries, and `maxim device list|show|check`.
- **PR 2, the writer (≈320):** the raw-mapping `write_robots_config`, the `add`, `rename`, `forget` and `repin` verbs, `use`/`maxim connect` (which moves `primary:`); `relocate` lands with PR 3, the lints, and the C2 amendment.
- **PR 2b, `embody` (≈180):** `plan_launch`, preflight, the plan block, carrying the verified address, pin and typed grant into the existing launch, `embody`'s D13′ exits, the perception-without-media refusal and `--no-media`, and `launch:` + `--save-defaults` (which writes `mode/llm/interactive` only).
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
- **D13:** non-zero exit on a refused named-device connect. **D13′ (clarification, same intent; reworded v6):** defined by pin status and the §3.1 rules. A refused or failed connect on a pinned entry, a pinned-registry refusal, or a failed connect on an entry chosen by a new surface (`--device`, `MAXIM_DEVICE`, `embody`, `open_controller`) exits non-zero. The new surfaces always need `--unpinned` for an unpinned entry. Only a legacy launch in a registry with no pinned entry keeps the headless fallback (§3a).

**Taken 2026-11-25 (owner took the recommendation):**
- ✅ **D14. Deny agent file-tool writes to all of Maxim's own control-plane files?** In singularity mode, `check_write_permission` allows writes anywhere under cwd. Started from `~`, the agent can rewrite `~/.config/maxim/` (mesh.yml, api_key, console_token, config.json) and `<data_home>/robots.yaml` without approval.
  - *Taken:* deny **file-tool** writes to the whole control-plane set (`~/.config/maxim/`, `<data_home>/robots.yaml`, `<data_home>/util/device_state.json`) in every mode, in PR 0. v4 implements this through the #957 seam, because `check_write_permission` is unwired (#1811). Singularity's `bash` remains a stated residual. That is about 30 `src/` lines plus tests. Singularity keeps full access to everything else. An agent should never be able to rewrite its own trust anchors.
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

## 12. Disposition of delta-round-2 findings (DRAFT v3), and the round-3 scope

| Lens | Finding | Folded in |
|---|---|---|
| Adversarial | N1 rest | §3b context objects, §4.2 `add` uniqueness + `repin` host |
| | N2 / NF3 | §5 #957 seam, #1811 |
| | N3 | §3.1 unpinned on every path |
| | N4 / NF2 | §4.2 library resolvers |
| | NF1 | §4.3 step-1 media rule |
| | NF4 | §4.2 endpoint registration |
| | NF5 | §4.2 tunnel hint |
| Safety | D1 | §4.3 snapshot/restore |
| | D2 | §4.3 step-3 re-probe |
| | D3 | §4.3 step-1 tunnel/media rule |
| | D4 | §4.3 wrap |
| | D5 | §4.5 split, §3b registry path |
| | D6 | §4.1 reset |
| | D7 | §3a `agentic` refused, shared constant |
| | D8 | §4.3 log |
| Architecture | F1 | lever dropped |
| | F2 / F3 | §3b `LAYER_RULES` |
| | F4 / F7 | §4.1 typed scalars |
| | F5 | §7 |
| | F6 | §3b `open_controller` |
| | F8 | §3b docs |
| | N3 NIT | §4.1 |
| | N5 NIT | §3b(a) |
| Wire | W2-1 | lever dropped |
| | W2-2 | §3.1 precedence |
| | W2-3 / W2-4 | §3.1 re-exec + PR 0 ordering |
| | W2-5 | §4.1 allowed keys |
| | W2-6 | §4.1 handoff attribute + tests |
| | W2-7 | §3.1 no-primary rule, §7 `forget` |

**Delta round 3 scope** (the fail-open folds only; same stop rule):

| Lens | Re-check |
|---|---|
| Adversarial | NF2 (resolver-based proxy check), NF3 (#957-seam deny), N3 (unpinned on every path) |
| Safety | D1–D3 (snapshot/restore, step-3 re-probe, step-1 media rule) |
| Architecture | F2–F4 (`LAYER_RULES` enforcement, `primary:` typing) |
| Wire | W2-2–W2-4 and W2-7 (precedence, re-exec, no-primary) |

## 13. Disposition of delta-round-3 findings (DRAFT v4), and the round-4 scope

| Lens | Finding | Folded in |
|---|---|---|
| Adversarial | R3-1 | §4.2 every media URL |
| | R3-2 | §4.2 notes |
| | R3-3 / R3-4 / R3-5 | §5 denied set + matching |
| | R3-6 | §5 `write_script` |
| | R3-7 | §5 residual |
| | R3-8 | §5 `.bak` |
| | R3-9 | §4.1 writer-managed `hardware_id:` |
| | R3-10 / R3-11 | §3.1 pinned-registry rule (v5) |
| Safety | E1 | §4.3 step 4 order |
| | E2 | §4.3 per-connect media |
| | E3 | §4.3 step 3 session fields |
| | E4 | §4.3 `wlan_ip` None → `no_media` |
| | E5 | §4.3 wording, §3a perception refusal + `--no-media` |
| | E6 | §4.3 residual |
| Architecture | D3-1 | §3b(d) reference lint |
| | D3-2 | §3b docs/tests |
| | D3-3 | §3b(c) closure test |
| | D3-4 | §4.1 `type(v) is bool` |
| Wire | W3-1 / W3-4 | §3.1 ordering: one PR |
| | W3-2 / W3-3 | §3.1 pinned-registry rule |
| | W3-5 | §3.1 daemon name |
| | W3-6 | §3.1 / §3a |

**Owner decisions in v5:**
- **Taken:** the simplification, i.e. the pinned-registry rule, and shipping `--device`, its re-exec, `--unpinned` and the refusal together in PR 1.
- **Taken:** `embody` refuses a perception mode when it would have to force `no_media`, with `--no-media` as the explicit override (owner, 2026-11-25).

**Delta round 4 scope** (the fail-open folds only; same stop rule):

| Lens | Re-check |
|---|---|
| Adversarial | R3-1 (every media URL), R3-3/4/5 (denied set and matching), R3-9 (writer-managed pins) |
| Safety | E1, E2, E4 (§4.3 steps 1 and 4) |
| Architecture | D3-1 (reference lint) |
| Wire | W3-1/2/3/5 under the pinned-registry rule and the PR 1 ordering |

## 14. Disposition of delta-round-4 findings (DRAFT v5), and the round-5 check scope

| Lens | Finding | Folded in |
|---|---|---|
| Adversarial | R4-1 | §4.2 media-backend allowlist |
| | R4-2 | §4.2 "no proxy anywhere" media rule |
| | R4-3 | §4.3 step 5, §6 row |
| | R4-4 | §5 union of roots |
| | R4-5 | §5 #957 ancestor walk |
| | R4-6 | §5 hardlinks |
| | R4-7 | §5 same `real_path` |
| | R4-8 | §4.1 `unpinned` literal + downgrade residual |
| Safety | F1 | §4.3 E2 ordering (i)–(iii) |
| | F2 | §3a typed `media_required` + `DeviceMediaUnavailable` |
| | F3 | §4.5 degrade, don't stop |
| | F4 | §4.3 `localhost_only` carve-out |
| Architecture | D4-1 | §3b(d) receiver-agnostic rule in `audit_architecture` |
| | D4-2 | §3b(d) residuals |
| | D4-3 | §3b(c) child-process closure test |
| | D4-4 | moot under D4-1 |
| Wire | W4-1 | §3.1 daemon name = today's chain; re-exec replays the selection |
| | W4-2 | §3.1 legacy matching unchanged |
| | W4-3 | §3.1 `--unpinned` always carried |
| | W4-4 | §3.1 ordering |
| | W3-3 leftover | §3.1 new surfaces always strict, §4.2, §9 D13′ |

**Delta round 5 check scope** (fail-open folds only; same stop rule):

| Lens | Re-check |
|---|---|
| Adversarial | R4-1 (allowlist), R4-4 (root union), R4-5 (ancestor walk) |
| Safety | F1 (E2 ordering), F2 (`media_required`) |
| Architecture | D4-1 (receiver-agnostic rule; re-run the 15-spelling probe table) |
| Wire | W4-1 (daemon-name chain + replayed re-exec), the W3-3 leftover (§4.2 / D13′ wording) |

## 15. Disposition of round-5 check findings (DRAFT v6), and the round-6 check scope

| Lens | Finding | Folded in |
|---|---|---|
| Architecture | stop rule **MET** | D5-1, D5-2, D5-3 folded as §3b(d) implementation notes |
| Adversarial | R5-1 | §4.2 SDK-exact matching + canonical literal |
| | R5-2 | §5 env-independent name rule + stated residual |
| | R5-3 | §5 every-ancestor walk + absent-root anchoring + string compare kept |
| Safety | G1 | §3a postcondition; a configured `no_media` is not a grant |
| | G2 | §4.3 onboard proof (local `get_hardware_id()` == pin), manual tunnel = network mode, no LOCAL |
| | N1 | §4.3 E2 (iv) |
| | N2 | §3a per-call `media_required`, §4.5 |
| Wire | W5-1 | §3.1 new-surface daemon name: refuse an addressless entry |
| | W5-2 | `MAXIM_DEVICE` in the strict list, §3a, D13′ |
| | W5-3 | §3.1 precedence points at Legacy matching |

**Round 6 check scope** (fail-open folds only; same stop rule; architecture is done):

| Lens | Re-check |
|---|---|
| Adversarial | R5-1, R5-2, R5-3 |
| Safety | G1, G2 |
| Wire | W5-1, W5-2 |

## 16. Disposition of round-6 check findings (DRAFT v7), and the round-7 check scope

| Lens | Finding | Folded in |
|---|---|---|
| Adversarial | stop rule **MET** | N6-1 (narrowed residual), N6-2 (removed the unbacked `child_env` sentence), N6-3 (component-wise name rule + a red gate only it passes); out-of-scope notes: HFS+ ignorables stated, allowlist wording (fixed with H2), non-string `media_backend` refused |
| Safety | H1 | §3a first-frame liveness |
| | H2 | §4.2 allowlist keyed on the onboard proof + explicit-`local` red gate |
| | N3 | §3a placement |
| | N4 | §4.3 proof exceptions = not proven |
| | N5 | §4.3 DoA / readback key on the proof |
| Wire | W6-1 | §3.1 every non-legacy selection refuses an addressless entry |
| | N6-1 | test rows |
| | N6-2 | command table comment |
| | N6-3 | writer's `robot_name` value |

**Round 7 check scope** (fail-open folds only; same stop rule; architecture and adversarial are done):

| Lens | Re-check |
|---|---|
| Safety | H1 (first-frame liveness), H2 (proof-keyed allowlist) |
| Wire | W6-1 (every non-legacy selection) |

**§16 addendum: round 7.** Safety met the stop rule; its N7-1 and N7-2 are folded into §3a as liveness wording. Wire W7-1 (doctor's private address default) and its N7-1/N7-2 are folded into §3.1. Owner decision (2026-11-25): the media-backend LOCAL refusal **also covers unpinned networked** entries (§4.2); `localhost_only` is unchanged. **Final wire check** ([delta8_wire.md](reviews/1806/delta8_wire.md)): **W7-1 closed, stop rule met.** N8-1 to N8-3 are folded as text and test-list changes. **All four lenses have met the stop rule. The design gate is closed** (2026-11-25), subject to the one open owner question above.
