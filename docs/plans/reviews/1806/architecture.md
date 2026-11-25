# #1806 device registry: architecture, persistence and API-shape review

Reviewer: architecture/persistence/API lens, read-only, 2026-11-25. Material: `docs/plans/device_registry_1806_approach.md` DRAFT v1, read against the worktree at origin/main `b7380ee7`. No code was run, and no robot or LAN host was contacted. Every claim below cites the code it was checked against. Claims marked *(unverified)* were not checked.

**Verdict:** **DO-NOT-BUILD for PR 2b as scoped** (an in-process `maxim.embody()`, D9's recommended option). **SHOULD-FIX for the rest** before PR 1. PR 0 (groundwork) can start once F3 and F4 are folded into its scope. The design mostly rides existing infrastructure, but in three places its new surface goes beyond what one kind needs: the `DEVICE_KINDS` table, a second default mechanism held in config.json, and an experimental keyword on a stable verb. Two fail-open paths let the identity pin go missing without anyone noticing: F3 and F4.

| # | Finding | Class | Fail mode |
|---|---|---|---|
| 1 | In-process `maxim.embody()` inherits process-global CLI effects: re-exec on mode switch, `configure_logging(force=True)`, termios, the global spawner stop. Registered tools and `on()` events do not reach the robot runtime | DO-NOT-BUILD (PR 2b as scoped) | fail-open |
| 2 | §3a "Later" keeps a Python `maxim.embody` out of this line, but §3b puts it in. §3a also describes the CLI path as `Maxim(robot_id=…)`, which the CLI does not pass | SHOULD-FIX | n/a |
| 3 | `selfy.py::Maxim.__init__` swallows any robots.yaml failure at DEBUG and connects with defaults. That defeats §5's "loud yaml error" and drops the pin | SHOULD-FIX (blocks PR 1) | fail-open |
| 4 | `RobotRegistry.connect_robot` drops constructor-unknown config keys at DEBUG. A pinned entry on a plugin controller (or an older build) connects unpinned | SHOULD-FIX (blocks PR 1) | fail-open |
| 5 | The default device in config.json bumps `CONFIG_FORMAT_VERSION` 1.4→1.5, adds a CC3 section, a fixture row, an IM2 writer allow-list entry and cross-version write refusal. robots.yaml `primary:` already is a default | SHOULD-FIX | n/a |
| 6 | `maxim.connect(device=)` cannot be called without changing the stable signature (`robot_type` is required positional). It mixes stability tiers in one verb and opens a `config=` override of the pin | SHOULD-FIX | fail-open |
| 7 | `DEVICE_KINDS`, with one real kind, is premature. Keep the kind as a reserved field and a single-branch internal router | SHOULD-FIX | n/a |
| 8 | Layer placement is under-specified. Selection precedence (config.json) and node-name reservation (peer/) cannot live in `hardware/`. The embody seam cannot either | SHOULD-FIX | n/a |
| 9 | Python write verbs (`add/rename/forget/use/--save-defaults`) are a new class of writer to the declarative layer. C2 permits CLI verbs only, so they need a decision record and a guard | SHOULD-FIX | fail-open |
| 10 | robots.yaml search includes `./robots.yaml` (cwd). A first `device add` creates `<data_home>/robots.yaml`, which silently shadows the cwd file | SHOULD-FIX | fail-open |
| 11 | robots.yaml versioning: use `_format_version` through `check_format_version`, not a bespoke `format_version`. Older builds ignore the version key and the pin, so a downgrade connects unpinned. The writer needs a FileLock around its read-modify-write | SHOULD-FIX | fail-open |
| 12 | `launch:` under `config:` mixes launch defaults into the controller-constructor dict. Make it a sibling of `config:` | SHOULD-FIX | n/a |
| 13 | Verify-then-dial: the resolver must hand the SDK the verified address literal, not a hostname that is re-resolved | SHOULD-FIX (security lens owns it) | fail-open |
| 14 | `device_state.json` is not role-scoped, against the layer-2 rule. The exemption needs stating, and the note should say why it cannot ride `probe_cache` | NIT | n/a |
| 15 | Name namespace: refusing only at device-write time misses later `add-node` collisions. Names need one validator (charset, casefold). A solo box has no mesh.yml | SHOULD-FIX | n/a |
| 16 | Public-type hygiene: `devices.list` shadows the builtin; `DeviceEntry` duplicates `RobotConfig`; `DeviceCheckReport` should reuse `DiagnosticReport`/`CheckResult`; CC3 is overclaimed; the mypy list, `_API_VERBS` and README count need updating; the env var is unnamed; `peer connect` vs `connect` naming | NIT | n/a |
| 17 | Role detection: no new file enters `detect_role`'s inputs. Confirmed safe | NIT (no change) | n/a |

---

## F1. In-process `maxim.embody()` is not "sugar over the existing launch path" (DO-NOT-BUILD for PR 2b as scoped)

§3b plans to factor "the `_main_impl` → `Maxim(...)` path" into one library function in about 200 lines. In the code, that path is the `while True:` loop in `cli.py::_main_impl`, and it depends on process-global CLI behaviour that a library must not inherit:

- **Re-exec on a mode switch.** `cli.py::_after_run_mode_request` calls `_reexec_with_mode(args, mode=requested)`, an exec of the process. A library caller's process (a notebook, or Pulse) would be replaced.
- **Global logging.** Every iteration runs `configure_logging(args.verbosity, log_file=log_path, force=True)`. `embodied_runtime/selfy.py::Maxim.__init__` and `media_loop.py` call `configure_logging` as well. An embedder's logging would be overwritten.
- **Terminal state.** `embodied_runtime/input_handlers.py` puts the terminal into termios raw/no-echo mode when interactive.
- **Global teardown.** `cli.py::_end_robot_run` calls `lane_backends.stop_active_spawner` and swallows a failing `shutdown()` (`except Exception: pass`).
- **Signals.** `cli.main` installs `cli_signals` around everything. The library must not, and the note agrees. Nothing then replaces the stop path. The note leaves the stop handle "to be set in the design pass".
- **Extension points do not reach the robot runtime.** This was verified, not assumed. `api.py::_inject_registered_tools` has two callers, `api.py::run` and `simulation/orchestrator.py`. `api.py::_bridge_event_subscriptions` likewise has two callers, `api._build_run_agent` and `orchestrator.py`. Neither has a caller in `embodied_runtime/`. Making `embody()` "honour both" means building a tool registry and an event bus inside the embodied runtime. `embodied_runtime` cannot import the `maxim.api` facade, so the seam has to take them as parameters. That is a design pass of its own, not a fold.

**Fix.** Take **D9 = plan-only** for v1. `maxim.devices.plan_embody(name, …) -> LaunchPlan` resolves, runs the preflight and returns the plan; the launch stays CLI-only. Then factor the launch into `embodied_runtime/launch.py::run_embodied(plan, *, stop_event, tools, event_bus)` as a separate, reviewed PR. It must make re-exec, logging and termios caller-owned options, and it must use the bounded-stop seam. Once that exists, `maxim.embody` becomes a thin call. It lines up with `embodied_device.md`'s `MaximHandle.embody()` rather than racing it. The `_main_impl` touch rule still applies to the CLI-side extraction.

## F2. The note contradicts itself on the Python `embody` and on `robot_id` (SHOULD-FIX)

- §3a "Later" says a Python `maxim.embody("<name>")` mirror "is a stable-API decision and stays out of this line". §3b then specifies `maxim.embody(...)`, and PR 2b builds it. Pick one; F1 recommends the plan-only form under `maxim.devices`.
- §3a says the CLI path is "`Maxim(robot_id=…, robot_name=…)` → `RobotRegistry.connect_robot`". The CLI actually calls `Maxim(robot_name=args.robot_name, …)` with no `robot_id`; see the `from maxim.embodied_runtime.selfy import Maxim` block in `cli.py::_main_impl`. §3 says the same thing ("The CLI never passes `robot_id=`") as a statement about today. State that PR 0 *starts* passing `robot_id=<entry id>` and keeps `robot_name=<daemon name>`. That is the entry-id/daemon-name split in §3.

## F3. The selfy connect path swallows the robots.yaml load, so the pin disappears (SHOULD-FIX, blocks PR 1, fail-open)

In `embodied_runtime/selfy.py::Maxim.__init__`, the robots.yaml block (`load_robots_config` → `resolve_connection_config` → `resolve_robot_entry`) is wrapped in `except Exception as e: self.log.debug("robots.yaml connection config not loaded: %s", e)` and then connects with `{"robot_name": self.name, "media_backend": …}`. §5 wants `RobotsConfig.from_yaml` to raise a typed error. That error would be caught here at DEBUG, and the runtime would then dial the default mDNS name **without `expected_hardware_id`**. The pin "enforced inside the seam" depends on the config reaching the seam, and this path drops it.

**Fix (PR 0).** Narrow the handler. "No robots.yaml" is a legitimate empty config. A parse error, a newer version or a bad entry must raise. Add a regression test showing that a malformed robots.yaml stops `Maxim(...)` before `connect_robot` is reached. This is another site of §5's swallow pattern, so under "Filing is not fixing" it belongs in PR 0.

## F4. `connect_robot` drops unknown config keys, so a pin on any controller without the parameter is silently ignored (SHOULD-FIX, blocks PR 1, fail-open)

`hardware/registry.py::RobotRegistry.connect_robot` filters the config to the constructor's parameters and logs what it dropped at DEBUG ("config keys not taken by … constructor"). That is the same mechanism that kills the dead `timeout:` key in §5. §4 puts the pin in `ReachyMiniController.connect` as an `expected_hardware_id` config key. As a result:

- a plugin controller (the note gives plugins `discover()`/`probe()` hooks) without that parameter connects a pinned entry unpinned;
- an **older build** (PyPI 1.3.4 or earlier) reading a newer robots.yaml drops the key the same way (see F11).

"No caller can skip it" is true only for Reachy, on a build that has the parameter.

**Fix.** Enforce the pin one layer up, in the registry. If the resolved config carries `expected_hardware_id` and the controller class does not declare pin support (a class attribute such as `supports_identity_pin = True`, or the optional `probe()` classmethod returning the identity), `connect_robot` refuses with `DeviceIdentityMismatch`/`ConfigurationError` instead of dropping the key. Better still: verification runs in the registry through the kind's `probe()` *before* construction, so the controller only receives an already-verified address. Add a test that a pinned entry on a stub controller without pin support refuses. Add `expected_hardware_id` to the "never dropped silently" set.

## F5. The default device in config.json is a second default mechanism and a schema bump (SHOULD-FIX)

`runtime/config_loader.py` is at `CONFIG_FORMAT_VERSION = "1.4"`. The `MaximConfig` docstring says "Adding ANY field or section, in any section, bumps CONFIG_FORMAT_VERSION's minor"; the guard is `tests/unit/test_config_format_version_856.py`. All sections are path-(b) SHAPE-FROZEN, so a `devices.default` field means:

- 1.5, a new frozen section type with its CC3 choice, and a row in `tests/fixtures/config_resolution_by_version.json` (append-only, #856);
- a new caller in the `CONFIG_WRITER_MATCHES` allow-list (`.github/workflows/test.yml`, IM2);
- the #974 consequence: on a box where a PyPI 1.3.4 leader shares `~/.config/maxim` (the rig case in DECISIONS 2026-09-28 and 2026-10-01), that build tolerates reading a 1.5 file but **refuses `maxim config set`** until a `downgrade` is run. The first `maxim connect` would therefore break `config set` for the older build.

The selection precedence also ends up with two defaults, the config default and then the robots.yaml primary. robots.yaml already has a default: `primary: true` (`RobotsConfig.get_primary`, used by `resolve_robot_entry`). With one kind, `connect` setting the default is the same as writing `primary: true` on that entry, and clearing it on the others, through the one robots.yaml writer. That needs no config.json change.

**Fix.** In v1, `connect`/`use` set `primary:` in robots.yaml, and the precedence is `--device` > env > robots.yaml primary. A cross-kind default in config.json waits for the second kind. Then it goes in with the 1.5 bump, the fixture row and the allow-list entry named in the plan, and with the DECISIONS 2026-10-01 meaning: the field is present only when the operator ran the verb.

## F6. `maxim.connect(device=…)` needs a stable signature change and mixes stability tiers (SHOULD-FIX, fail-open)

`api.py::connect(robot_type: str, *, name=None, config=None, timeout=30.0, set_primary=True)`. `robot_type` is required positional, so `maxim.connect(device="reachy")` raises `TypeError` today. Supporting it means changing the stable row in `docs/user/stable_api.md` ("`maxim.connect(robot_type, ...)`") to `robot_type: str | None = None`. Stability rule 1 covers *adding* a keyword-only argument, not relaxing a required positional, so this is a stable-contract edit, not "additive". It also places an ⚠️ Experimental keyword on a ✅ stable verb, and stable_api.md has no precedent for that. There is a second problem: `config=` is merged on the device path, so a caller passing `config={"expected_hardware_id": None}` or a `host` override could unpin or redirect the device. Note also that `api.connect` today never reads robots.yaml at all. It builds the config from the arguments only.

**Fix.** Ship `maxim.devices.connect(name, *, timeout=…) -> RobotController` inside the experimental namespace, with no `config=` (or with identity and address keys refused). The stable `maxim.connect` stays untouched until the namespace graduates. Graduation is then one decision record, not a silent widening.

## F7. `DEVICE_KINDS` with one real kind is premature (SHOULD-FIX)

The note itself says the abstraction is "unproven" until a second kind lands (§8, Later). `lane_backends.BACKEND_CLASSES` earns its table with several entries. `simulated` is a *robot type* inside the robot kind, not a second kind. Under the front gate ("if it can ride on existing, choose that path"), v1 rides `RobotsConfig` + `RobotRegistry` + the `maxim.robots` entry points with no new table.

**Fix.** Put `kind: Literal["robot"] = "robot"` on the public entry type, and make the name router an internal function with one branch, `_route(name) -> robot entry`, raising `DeviceNotFound` otherwise. No `DEVICE_KINDS` symbol until kind #2 (local media, or `attached_via`) arrives, with its own front-gate justification. The CLI noun `maxim device` (D1) still works with no table behind it.

## F8. Layer placement: name each piece's home (SHOULD-FIX)

`utils/audit.py::LAYER_RULES["hardware"]` forbids `embodied_runtime`, `tools`, `runtime`, `agents` and `planning`. §7 puts "the registry core" in `hardware/` without separating the pieces that cannot live there:

| Piece | Home | Why |
|---|---|---|
| robots.yaml reader (exists), the one writer, candidate resolver, identity verify, the one Reachy REST probe, `device_state.json` | `hardware/` (`hardware/config.py`, `hardware/reachy/probe.py`, `hardware/device_state.py`) | uses only `utils/http`, `utils/atomic_io`, `filelock` |
| doctor's robot check | `doctor/checks.py` imports the hardware probe and adapts it to `CheckResult` | downward import |
| selection precedence (`--device` > env > default) and, if F5 is not taken, the config.json default | `maxim/devices.py` facade (or `runtime/`) | `runtime.config_loader` is forbidden from `hardware/` |
| mesh node-name reservation | `maxim/devices.py` facade | `hardware/` → `peer/` is not in LAYER_RULES but is an upward edge; keep `hardware/` low |
| embody seam | `embodied_runtime/launch.py` (F1) | constructs `selfy.Maxim`; forbidden from `hardware/` |
| typed errors | `maxim/exceptions.py` | the existing hierarchy |
| CLI | `cli.py` dispatch + a `cli_devices.py` handler module, added to `_NON_SIM_SUBCOMMANDS` | the touch rule on `_main_impl` |

Note also that `maxim/devices.py` is a new top-level module. §7's "no new top-level package" is still true, but the first segment `devices` has no `LAYER_RULES` entry. Add one (`"devices": {"must_not_import": ["tools", "agents", "planning"]}`) so that no agent tool can reach the write verbs (see F9).

## F9. Python write verbs are a new writer class for the declarative layer (SHOULD-FIX, fail-open)

`docs/agents/persistence-config.md` §1 says layer-1 files are written "ONLY by operator-explicit one-shot CLI verbs … never by runtime code, never by admin APIs". `maxim.devices.add/rename/forget/use` and `embody(--save-defaults)` called from Python make library calls into robots.yaml writers, and with F5 not taken, into config.json writers as well. A tool or skill written against the namespace could then rewrite operator intent, and a model-driven path could repoint a device's `host:`.

**Fix.** Add a DECISIONS entry that amends C2. Sanctioned writers become "operator-explicit verbs: the CLI, or the documented experimental Python namespace, never runtime or agent code". Enforce it with the writer's CI caller allow-list, limited to `hardware/config.py`'s writer and `maxim/devices.py`. Also add a LAYER_RULES or grep guard so that `tools/`, `agents/`, `skills/` and `embodied_runtime/` never import `maxim.devices`. The alternative, if the owner prefers it: the Python namespace is read-only (`entries/get/check/resolve/plan_embody`) and the writes stay CLI-only.

## F10. The cwd robots.yaml is silently shadowed by the first write (SHOULD-FIX, fail-open)

`hardware/config.py::find_config_file` searches `<data_home>/robots.yaml`, then `./robots.yaml` (`DEFAULT_CONFIG_PATHS`), and the first one that exists wins. An operator whose only file is `./robots.yaml` runs `maxim device add`, and the writer creates `<data_home>/robots.yaml`. From then on the cwd file is never read, so its robots disappear with no message. A cwd lookup is also an input that can lie: a cloned checkout's `robots.yaml` can point the runtime at a host of its choosing.

**Fix.** The writer writes only `<data_home>/robots.yaml`. It refuses when the resolved source is a cwd file, and names a `--import` or move step. Consider deprecating the cwd search path, with a warning when it is used *(owner decision)*. Add the cwd input to §6's threat table.

## F11. robots.yaml format and writer mechanics (SHOULD-FIX, fail-open on downgrade)

- **The key.** D5 says "a top-level `format_version`". The repo's version helper is `utils/format_version.py::check_format_version(data, "<file_type>", log=…)`, which works on any dict, including parsed YAML, and keys on `_format_version`. Using `_format_version: "1.0"` with `check_format_version(data, "robots_yaml")` reuses the one refuse-newer and legacy-absent path. A `tests/integration/test_persistence_compat.py` row should come with it. `mesh.yml` uses `protocol_version` only because its dialect is SHAPE-FROZEN on the wire; robots.yaml has no such constraint.
- **Downgrade fails open.** `RobotsConfig.from_dict` ignores unknown top-level keys, and `connect_robot` drops unknown config keys (F4). A pre-#1806 build therefore loads a versioned, pinned file without complaint and **connects unpinned**. That cannot be fixed retroactively. Document it in `safety.md` and in the changelog fragment, and make `maxim device check` print "pinned (enforced by ≥1.3.x)".
- **Writer primitives.** Use a `filelock.FileLock` around the full read-modify-write. `add`, `connect` (F5's `primary:` flip) and `--save-defaults` can race from two shells, and `write_mesh_config` takes no lock because its verbs are rare. Write with `atomic_write_text` (robots.yaml holds no secret today, so the reason for `atomic_write_secret` does not apply). Use `yaml.safe_dump(sort_keys=False)`. The `.bak` is written once, on the first rewrite of an unversioned or commented file, and is never overwritten. The shipped template `data/util/robots.yaml` is 45 of 54 lines comments, so the first rewrite will strip it, as D5 accepted.
- **Location.** Keeping robots.yaml in `data_home` (not `~/.config/maxim`) is right. `MAXIM_DATA_HOME` isolation is what keeps harness trials off real hardware (the `find_config_file` comment). Record it in the §1 layer table as "declarative, in the data home, by design", as the note's docs list intends.

## F12. `launch:` belongs beside `config:`, not inside it (SHOULD-FIX)

`config:` is the controller-constructor dict. The registry splats it (filtered) into `ReachyMiniController(**config)`, and wiring keys (`body`, `audio_localization`) are already special-cased there. A controller declaring `**kwargs` receives `launch={…}`. Make `launch:` an entry-level sibling (`robots.<id>.launch`) and add a `launch` field on the entry type. `RobotConfig.from_dict` ignores unknown entry keys, so an older build degrades cleanly. Validate `launch.mode` and `operational_mode` against `modes.definitions.OPERATIONAL_MODES` at load, and make `launch.llm` lower-capability only (#829/#924).

## F13. Verify-then-dial must hand the SDK the verified address (SHOULD-FIX, fail-open; detail belongs to the security lens)

§4 verifies `GET /api/daemon/status` at a candidate, and then `ReachyMiniController.connect` dials `ws://<host>:8000/ws/sdk`. If the candidate was a hostname (resolver step 3, or the `reachy-mini.local` default), the SDK resolves it again and may land on a different host. §7 cites the rule ("never pre-resolve and then dial separately", #824/#921), but the composition still has two dials. The resolver's output (`ResolvedDevice.address`) must be an IP literal, passed as `host=` with `connection_mode="network"`. The controller pre-check repeats the identity read on the same literal. *(The SDK's own reconnect behaviour was not checked.)*

## F14. `device_state.json`: the role-scoping exemption and the front-gate reason (NIT)

Layer 2 says "filenames are role-scoped via `MAXIM_ROLE`" (`util/{name}.{role}.json`). A device's last-good address belongs to the machine, not the role, so an exemption is reasonable, as `hive_release_sequence.json` has. The note should state the exemption. It should also say why the file cannot ride `runtime/probe_cache.py`: that module is in `runtime/` (forbidden from `hardware/`), is URL-keyed with TTL outcome semantics, and deliberately accepts lost updates. The same argument is a point *for* a light lock: every cached address is re-verified, so a lost update only costs one extra candidate. Keep the FileLock to follow the rule. Shape: `{"_format_version": "1.0", "devices": {"reachy_mini:<hardware_id>": {"address": …, "verified_at": …}}}`, keyed by the identity rather than the alias, so a `rename` does not orphan the entry.

## F15. Names: one validator, a read-time check, and the solo box (SHOULD-FIX)

- Refusing a collision inside the device writer does not stop `maxim peer add-node <name>` from creating one later. `peer/mesh_setup.py` knows nothing about devices. The router must also refuse at **read** time ("`reachy` is both a device and a mesh node; rename one").
- On a solo box with no `mesh.yml` there is nothing to reserve. On a peer, `mesh.yml` may be stale. Since v1 never routes `connect` to a node, a malformed `mesh.yml` should produce a warning, not block `device` verbs.
- Add one `validate_device_name()`: casefolded comparison, a charset such as `[a-z0-9][a-z0-9_-]{0,62}` (no leading `-`, which argparse would read as a flag; safe as a YAML key), and the entry id separate from the daemon `robot_name`.

## F16. Public-type and surface hygiene (NIT)

- `maxim.devices.list` shadows the builtin inside `devices.py`, so any `list(...)` call in that module breaks. Use `entries()`, after `load.sessions()`.
- `DeviceEntry` and the mutable `hardware/config.py::RobotConfig` would be two types for one YAML entry. Either make `DeviceEntry` the frozen public *view* built from `RobotConfig` in one function, or extend `RobotConfig`. Do not maintain both by hand.
- `DeviceCheckReport` "same shape as `maxim.diagnose()`": reuse `api.py::DiagnosticReport` and `CheckResult` directly. The front gate applies.
- CC3: only `DeviceEntry` persists, and nothing here crosses a wire. Give all five types path (a) (defaults + `extra`), since they are experimental. The note's "persist or cross a wire" is accurate only for `DeviceEntry`.
- Exceptions (they exist: `maxim/exceptions.py`): `DeviceNotFound(ConfigurationError)`, `DeviceIdentityMismatch(MaximConnectionError)`, `DeviceUnreachable(MaximConnectionError)` and `DiscoveryUnavailable(MaximConnectionError)`, plus a robots.yaml newer-version or malformed error as `InvalidConfigError`. Any keyword-only `__init__` fields need the `_rebuild_with_state` `__reduce__` pattern. Export them from `maxim.exceptions` only, not the top level; no stable verb raises them.
- Surface plumbing: add `"devices": "maxim.devices"` to the lazy namespaces in `maxim/__init__.py` and `__all__`, and `src/maxim/devices.py` to `scripts/run_mypy_public.sh`'s `FILES`, next to `create.py`/`load.py`. A top-level `maxim.embody` would change `_API_VERBS` and README's "N verb-based functions" (`tests/unit/test_api_surface.py`). F1 avoids that.
- The env var in the precedence is unnamed. Name it (`MAXIM_DEVICE`?), add it to the embodiment brief's table, and add a conftest autouse scrub in the same commit. §5's three existing robot env vars need the same.
- `maxim connect <name>` (device) sits next to `maxim peer connect <url>` (`peer/cli.py::_cmd_connect`, which writes peer config). That is acceptable with D1, but `cli-reference.md` should name the difference explicitly.

## F17. Role detection: confirmed unaffected (NIT, no change)

`runtime/role.py::detect_role` ranks env, then `config.json::role`, then `mesh.yml` exists, then the cloudflared config, then `peer.yml`. robots.yaml, `util/device_state.json` and a config.json `devices.*` section (if F5 is not taken) are not among its inputs, and writing a config.json without `role` does not set one. No new file changes the detected role. The one cross-file effect is F5's #974 write refusal on a shared config directory, which is version, not role.
