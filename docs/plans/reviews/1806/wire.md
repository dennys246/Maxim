# #1806 device registry: wire-integrity review

Reviewer: wire-integrity lens (docs/CODE_REVIEW.md), read-only, 2026-11-25. Material: `docs/plans/device_registry_1806_approach.md` DRAFT v1, read against the worktree at origin/main `b7380ee7`. Probes ran under a scratch HOME with the worktree `src` on PYTHONPATH. I contacted no robot or LAN host and ran no sim.

**Verdict: DO-NOT-BUILD until v2 folds F1 and F2.** Both are note-level fixes and neither changes a decision. The note's central claim is "the pin is enforced inside the seam ... no caller can skip it". As designed, that does not hold. The pin rides a free-form config key, and three existing consumers drop that key silently. A mode switch also re-execs into a device that was not the one selected. Everything else is SHOULD-FIX or NIT.

## Summary

| # | Finding | Rating | Polarity |
|---|---|---|---|
| F1 | The pin travels as a free-form `config:` key, and three consumers drop it silently. The selfy loader swallows a malformed-yaml error at DEBUG. `RobotConfig.from_dict` drops entry-level keys. The registry signature filter drops the key for plugins, and 1.3.x readers on a shared `~/.maxim` drop it too. In every case the connect goes ahead unpinned | DO-NOT-BUILD | fail-open |
| F2 | The mode-switch re-exec (`cli_utils.py::reexec_with_mode`) rebuilds argv with `--robot-name` only. The child resolves by precedence and can drive a different, default robot | DO-NOT-BUILD | fail-open |
| F3 | The registry's reuse branch returns a live controller without comparing the pin. Positional `maxim.connect("reachy_mini")` and `maxim.run(robot=)` never read robots.yaml, so the "a Python caller cannot skip it" claim covers only the new keyword | SHOULD-FIX | fail-open |
| F4 | The fix that makes the quick-start gate "read the parsed value" would make a bare `maxim` launch the robot once `connect` has set a default. `test_cli_action_gate.py` tests a re-implemented copy of the gate, not `cli.py` | SHOULD-FIX | fail-open |
| F5 | "An unknown name is refused, never defaulted" collides with `resolve_robot_entry`'s load-bearing primary fallback. Under D3, `maxim --robot reachy_mini` would be refused against the shipped template, whose entry id is `primary`. The note's claim that the default entry id equals the default `robot_name` is false | SHOULD-FIX | fail-closed |
| F6 | The default device stored in config.json escapes `MAXIM_DATA_HOME` isolation (Exp 49), needs a schema section plus a `CONFIG_FORMAT_VERSION` bump, trips the `CONFIG_WRITER_MATCHES` CI grep, and cannot be read from `hardware/` (LAYER_RULES) | SHOULD-FIX | fail-closed |
| F7 | The writer allow-list greps one function, but `maxim.devices.add/rename/forget/use` and `embody(save_defaults=)` are un-grepped wrappers. This is the `SETUP_VERB_MATCHES` lesson | SHOULD-FIX | fail-open |
| F8 | Persistence symmetry: the loader reads the first of `[data_home/robots.yaml, ./robots.yaml]`, but the writer target is unstated. A cwd robots.yaml is a trust input that §6 does not list. `rename` and `forget` must also move the default and the state entries | SHOULD-FIX | fail-open |
| F9 | `launch.operational_mode` read from a file raises capability without an operator flag (against #829), and `_validate_operational_grant` requires `--mode` in raw argv | SHOULD-FIX | fail-open |
| F10 | The `_main_impl` robot path is not library-safe: `configure_logging(force=True)`, `os.execve` mode switches, killing the LLM spawner, a while-loop. `register_tool` and `maxim.on` do not reach the robot runtime today (verified). The run lock is not shared. A source-pin test breaks. `Maxim.__init__` falls under the touch rule | SHOULD-FIX | fail-open (silent drop) |
| F11 | `embody` collides with `embodied_device.md`'s `MaximHandle.embody(sdk)` / `adopt(sdk)`, which adopts a handle the embedder opened, with no pin. `maxim.devices` also sits beside the planned `maxim.device.run_device` | SHOULD-FIX | fail-open |
| F12 | Probe consolidation changes outcome semantics (doctor warns, the controller treats a failed status read as non-fatal, the `#1704` media-ownership input) and can make the tests' autouse monkeypatch seams vacuous. `lint_http_surface` raw-dial pins change exactly | SHOULD-FIX | fail-open (tests) |
| F13 | Discovery trips `lint_http_surface` rule C (`EGRESS_SDK_IMPORTERS["reachy_mini"]`, an exact set). zeroconf is unlisted, and the V8 latch census (`test_offline_latch.py::CENSUS_SITES`) does not cover discovery | SHOULD-FIX | fail-open |
| F14 | With `connection_mode: auto`, the SDK tries localhost first, so the identity is verified at one address while the session may land on another. A hostname candidate must hand the SDK the verified IP, never the name | SHOULD-FIX | fail-open |
| F15 | Scripts that build `ReachyMini(` directly, and `make_reachy_doa_reader(None)`, dial unpinned. The "every path" claim should be scoped | NIT | fail-open (residual) |
| F16 | The new key should ride `_format_version` + `utils/format_version.py` rather than a new `format_version`. Old readers ignore it (and the pin) | NIT | fail-open (residual) |
| F17 | The env layer of the selection precedence is unnamed. `MAXIM_ROBOT_NAME` is already documented in a user doc | NIT | n/a |
| F18 | The `from_yaml` fix must also cover `from_dict`'s silent skips. The swallow-lint counts and the doctor test stay valid | NIT | fail-open |
| F19 | Dead or neighbour surfaces: `Maxim._connect_kwargs` has no reader, `setup_robots_from_config` is a third launch path, and `maxim connect` sits beside `maxim peer connect` | NIT | n/a |

## Producer → consumer map

### (a) robots.yaml

| Contract | Producers (before → after) | Consumers (before → after) | What each gets, before → after |
|---|---|---|---|
| File location | none in `src/` (hand-edited); `scripts/exp49/run_trials.py` (`write_text`, isolated by `MAXIM_DATA_HOME`); `tests/unit/test_exp49_harness_seams.py` → adds the new single writer | `hardware/config.py::find_config_file` (data_home, then **cwd**) → `load_robots_config` → `selfy.py::Maxim.__init__`, `agentic_runtime.py::_resolve_body_wiring`, `doctor/checks.py::check_robot_reachable`, `hardware/config.py::setup_robots_from_config` (exported, no `src/` caller) | The writer's target is unstated, see F8 |
| Top-level keys | `robots:` → plus `format_version` (and a `default:` if F6's fix is taken) | `RobotsConfig.from_dict` reads only `robots` | Probe: `format_version: 99` is **ignored** today, so 1.3.x readers never refuse a newer file (F16) |
| Entry keys | `type`, `primary`, `config` → note adds pin/hostname/`launch` "under config:" | `RobotConfig.from_dict` reads only those three | Probe: entry-level `expected_hardware_id:` and `launch:` are **silently dropped** (F1). They must sit under `config:`, or `from_dict` must refuse unknown keys |
| `config:` dict | operator → plus `expected_hardware_id`, `hostname`, `launch` | `resolve_connection_config` merges it over the selfy defaults → `registry.py::connect_robot_with_status` signature filter (drops unknown keys at DEBUG unless `**kwargs`); `agentic_runtime` reads `body`, `audio_localization`, `motor_binding`, `aut_mode`, `audio_*` via `_resolved_robot_config`; `doctor::_check_one_robot` reads `host`, `robot_name` | Reachy: the pin is honoured only if it is a ctor parameter. Plugins: dropped, or received as an unknown kwarg (F1). `launch` is dropped for Reachy and splatted into `**kwargs` plugins (NIT) |
| Dead `timeout:` | `data/util/robots.yaml` | none: `ReachyMiniController.__init__` has no `timeout` | Confirmed dead |
| Template | `data/util/robots.yaml` (entry id `primary`) | users copy it | Breaks the note's D3 premise (F5) |

### (b) `robot_name` vs `robot_id`

| Site | Today | After the note |
|---|---|---|
| `cli_parser.py::_build_parser` `--robot-name` (default `$MAXIM_ROBOT_NAME`) | feeds `Maxim(robot_name=)` | kept as the daemon-name override; `--device`/`--robot` added (`_build_parser` is pinned at 593 lines, so the touch rule applies) |
| `selfy.py::Maxim.__init__` (`self.name`, `effective_robot_id = self._robot_id or self.name`) | entry id = daemon name | CLI passes `robot_id=`. `Maxim.__init__` is pinned at 312 lines, so the touch rule applies (F10) |
| `agentic_runtime.py::_resolve_body_wiring` (`_robot_id or name` + inline primary fallback) | second match rule | collapses to `resolve_robot_entry` (note §5) ✓ |
| `cli.py::_main_impl` gate `"--robot" in raw_argv` | prefix hack | F4 |
| `cli_utils.py::reexec_with_mode` argv `--robot-name` | carries the name only | **drops `--device`** (F2) |
| `api.py::connect` `robot_id = name or robot_type`; `api.py::_connect_robot_for_run` `robot_id=robot_type` | registry keys | unchanged; reuse bypass (F3) |
| `tests/unit/test_cli_action_gate.py::_compute_has_action` | re-implements the gate | pins a copy (F4) |
| Docs: `ARCHITECTURE.md` ("`maxim --robot reachy_mini`"), `docs/embodiment/reachy_mini/getting_started.md` (same), `docs/user/robot-setup.md` (`--robot-name my_reachy`), `docs/user/cli-reference.md` ("Zenoh discovery", stale), `docs/user/configuration.md` (`MAXIM_ROBOT_NAME`), `src/configs/templates/env.example`, `src/tests/basic_move.sh` | | `ARCHITECTURE.md` is missing from the note's doc list |

### (c) Reachy probe

| Copy | Behaviour that consumers and tests pin |
|---|---|
| `doctor/checks.py::_check_one_robot` | `host` verbatim, else `gethostbyname(<robot_name>.local)` with 2 s; TCP :8000; status unreadable → **warn** ("pre-1.5"); era matrix → fail or warn; state ≠ running → warn "1013"; every non-ok result carries `retry_id="robot"`. Pinned by `tests/unit/test_doctor.py` (gate + robot classes), which patches `socket.gethostbyname`, `socket.create_connection` and `maxim.utils.http.fetch_url` **by module attribute** and asserts message strings ("Local Network", "ssh pollen@10.6.0.63", "pre-1.5", "1013", "reachy_mini=="). Ignores `type`, `tunnel` and `connection_mode` |
| `hardware/reachy/controller.py::_resolve_mdns` / `_port_open` / `_daemon_status` | `localhost_only`/tunnel → 127.0.0.1; status best-effort, **never fatal**; the status dict feeds `_warn_if_media_already_released` and `_connect_released_media` (#1704 ownership). Seams monkeypatched by `test_reachy_connection_options.py`, `test_mdns_discovery.py`, `test_media_stream_gating.py` (autouse `_daemon_status → None` in the first three), and `test_media_release_1704.py` (status-driven ownership) |
| `utils/reachy_diagnostics.py` (`maxim-diagnostics` in `pyproject.toml [project.scripts]`) | ping + TCP + status + `/docs`; `$MAXIM_REACHY_HOST` or `10.42.0.1`; raw dial pinned in `scripts/lint_http_surface.py::RAW_DIAL_PINS` |
| `scripts/check_reachy_connection.py` | same; cited by `docs/reference.md`, `docs/user/robot-setup.md`, `docs/embodiment/reachy_mini/engineering.md`; ledger row `docs/bugs/README.md` D33 |

### (d) Python API

| Symbol | Callers / claims | Note's plan |
|---|---|---|
| `api.py::connect(robot_type, *, name, config, ...)` | `src/maxim/__init__.py` and `api.py` docstrings, `tests/unit/test_api_core.py`, `docs/user/stable_api.md` (✅ stable) | additive `device=` ✓. The positional path stays unpinned and never reads robots.yaml (F3) |
| `api.py::run(robot=)` → `_connect_robot_for_run` | `README.md`, `tools/reachy.py`, `tests/unit/test_move_tool_gaze.py`, `stable_api.md` | untouched, unpinned (F3) |
| Pulse's documented imports (`embodied_device.md` §2): `VoiceProvisioner`, `make_pairing_announcer`, `make_device_speak_sink`, `build_app` | not touched by #1806 | ✓ no change. `MaximHandle.embody(sdk)` naming collides (F11) |

### (e) `from_yaml` swallow → typed raise

| Caller | Handler around it | Gets after the raise |
|---|---|---|
| `load_robots_config` ← `selfy.py::Maxim.__init__` | `except Exception: self.log.debug(...)` | **swallowed at DEBUG**: connects with the inline defaults, unpinned (F1) |
| ← `agentic_runtime.py::_resolve_body_wiring` | `except Exception: debug; return None, None` | swallowed; runs bodiless |
| ← `doctor::check_robot_reachable` | `except Exception` → warn | ✓ intended; `test_gate_malformed_config_is_warn_never_fail` stays valid |
| ← `setup_robots_from_config` | none | raises ✓ |
| A missing file | `find_config_file` → None → `RobotsConfig()`; explicit path → warning + empty | must stay empty (conftest HOME isolation means tests normally see no file) ✓ |

### (f) `_main_impl` robot path (to be factored)

Side effects the library function inherits: `build_home(args.home_dir)`; a per-run `configure_logging(..., force=True)` writing `home_dir/logs/reachy_log_<run_id>.log`; `_check_gpu_status`; `Maxim(...)` with `operational_mode` from the grant validated by `_validate_operational_grant(args, raw_argv)`; `maxim.live()` / `maxim.sleep()`; `_run_stopped_on` (re-raises typed errors to `main()` for exit 2); `_end_robot_run` (`maxim.shutdown()` + **`stop_active_spawner()`**); `_after_run_mode_request` → `reexec_with_mode` → **`os.execve`**; the surrounding `while True` mode loop; `cli_signals` around everything. Not wired today: `api._inject_registered_tools` (callers: `api.run`, `simulation/orchestrator.py` only), `api._bridge_event_subscriptions` (same two), and `api._RUN_LOCK`.

### (g) Env vars

| Var | Readers |
|---|---|
| `MAXIM_ROBOT_NAME` | `cli_parser.py` (`--robot-name` default), `selfy.py::Maxim.__init__`, `src/tests/basic_move.sh`; documented in `docs/user/configuration.md`, `env.example`, archived HTML guides |
| `MAXIM_REACHY_HOST` | `utils/reachy_diagnostics.py::main`, `scripts/check_reachy_connection.py`, about 12 `scripts/orient_backbone/*` scripts (`live_1_smoke.py`, `doa_sweep.py`, `live_3_learn.py`, `yaw_verify.py`, ...) |
| `MAXIM_ROBOT_TIMEOUT` | `selfy.py::Maxim.__init__`; set by `scripts/exp49/run_trials.py::_trial_env` |

None is reachable from `build_primary_router`, so the autouse-scrub lesson does not strictly bind. Adding them to the embodiment brief's table is still right.

### (h) Lints and CI

| Gate | Hit by |
|---|---|
| `scripts/lint_motion_safety.py` rules 3/4 (`ReachyMini(` must pass `automatic_body_yaw=False`; no_media undo) | v1 adds no `ReachyMini(` ✓ (discovery is browse-only) |
| `scripts/lint_http_surface.py` `RAW_DIAL_PINS` (exact both ways: doctor 2, controller 1, reachy_diagnostics 1) | probe consolidation (F12) |
| `lint_http_surface.py` `EGRESS_SDK_IMPORTERS["reachy_mini"]` (exact set) | the discovery module (F13) |
| `.github/workflows/test.yml` `MESH_WRITER_MATCHES` pattern | to be mirrored for robots.yaml; it scans `src/maxim/` and `tests/` only (F7) |
| `test.yml` `CONFIG_WRITER_MATCHES` + `SETUP_VERB_MATCHES` | the default-device write (F6, F7) |
| `utils/audit.py::LAYER_RULES` (`hardware/` must not import `runtime/`) | config.json default read (F6) |
| `function_length_baseline.json`: `_main_impl` 1118, `_build_parser` 593, `Maxim.__init__` 312, `_start_agentic_runtime` 774 | §3b, D3 flags, F1/F10 handler edits |
| `scripts/lint_no_silent_swallows.py` | removing `from_yaml`'s handler lowers its count (allowed). The selfy/agentic handlers must re-raise, not add a swallow |
| `scripts/lint_orphan_modules.py` | a new `maxim/devices.py` must be imported (lazy `__init__` namespace, like `create`) ✓ |
| `tests/integration/test_offline_latch.py::CENSUS_SITES` | discovery (F13) |
| `tests/unit/test_operational_mode_launch.py::test_the_robot_runtime_carries_the_grant` (source pin on `cli.py`) | §3b extraction (F10) |

## Findings

### F1 (DO-NOT-BUILD, fail-open): the pin is carried by a channel three consumers drop silently

§4 puts the pin in "an `expected_hardware_id` config key" and says no caller can skip it. Three consumers lose it without a sound.
1. **`selfy.py::Maxim.__init__`** wraps `load_robots_config()` + `resolve_connection_config` in `except Exception as e: self.log.debug("robots.yaml connection config not loaded: %s", e)`. Today a malformed file yields an empty config. After §5's typed raise, the same handler catches it at DEBUG, and `connect_robot` dials `{"robot_name": self.name, ...}` with no pin, through mDNS `reachy-mini.local`, the very name §6 says is not unique. `agentic_runtime.py::_resolve_body_wiring` behaves the same way (body only). So the "loud" fix is not loud on the runtime path, and the pinned robot is replaced by whatever answers.
2. **`hardware/config.py::RobotConfig.from_dict`** reads only `type`/`primary`/`config`. A probe showed entry-level `expected_hardware_id:` and `launch:` silently dropped. The note says "under its `config:`" for `launch` but only "config key" for the pin. An operator, or a future writer, who places it one level up gets an unpinned connect.
3. **`hardware/registry.py::connect_robot_with_status`'s signature filter** drops ctor-unknown keys at DEBUG. A plugin controller (the note claims "including plugins") that does not declare the parameter silently ignores a pinned entry. Every 1.3.x reader does the same. `~/.maxim/` is shared across worktrees and an older pip install (see the rig stray-1.3.2 memory), so a file the new writer produced gets read unpinned by old code.

**Fix (v2 text):**
- The resolver returns a `ResolvedDevice`.
- `ReachyMiniController.__init__` takes `expected_hardware_id` as a declared parameter, so the filter passes it.
- `selfy`/`agentic_runtime` catch only non-config errors and **re-raise** the typed `RobotsConfigError` (use the `utils/logging.py::warn_unless_refusal` pattern, or make it a `SECURITY_REFUSALS` member).
- `RobotConfig.from_dict` refuses unknown entry-level keys.
- The resolver refuses a pinned entry whose controller class does not declare pin support, rather than letting the filter drop the key.
- State the old-reader residual (F16) in `safety.md`.
- Red gate: write a malformed robots.yaml plus a pinned entry, build `Maxim(...)` with a fake controller, and assert that it raises rather than connecting.

### F2 (DO-NOT-BUILD, fail-open): a mode switch re-execs into a different device

`cli.py::_after_run_mode_request` → `cli_utils.py::reexec_with_mode` builds `argv = [..., "--robot-name", args.robot_name, "--mode", ...]` and calls `os.execve`. `--device` is not carried, nor `embody` or its resolved launch arguments. The child then resolves through the note's precedence: env, then the `connect` default, then the primary. After `maxim --device rosie` (default `reachy`), a model-requested switch to `sleep`/`exploration` relaunches on **reachy**: verified, but not the robot the operator chose. The child's motors are a different robot's. §3a claims embody "behaves exactly as on `maxim --device`", which today means it inherits this defect.

**Fix:** carry `--device <entry_id>`, plus `embody`'s resolved `--mode/--llm/--audio/--interactive`, in the re-exec argv. Add a test beside `test_operational_mode_launch.py::test_a_run_mode_switch_keeps_the_operators_grant`: "a run-mode switch keeps the device".

### F3 (SHOULD-FIX, fail-open): registry reuse, and the stable API paths that stay unpinned

- `registry.py::connect_robot_with_status`: when `robot_id in self._robots` and the controller is live, it returns `created=False` with no comparison of the requested config. Suppose `maxim.connect("reachy_mini", name="reachy")` runs first (unpinned) and `maxim.connect(device="reachy")` follows. The second call gets the unpinned controller back unverified, because `RobotRegistry` is a process singleton.
- `api.py::connect` (positional) and `api.py::_connect_robot_for_run` (`run(robot=)`) never read robots.yaml, so they dial the default `robot_name` through mDNS with no pin, even when the operator pinned an entry.

**Fix:**
- On reuse, compare `expected_hardware_id` (and the verified host), and refuse a mismatch or an unpinned → pinned upgrade.
- For the positional path, either consult the unambiguous robots.yaml entry for its pin (a stable-API behaviour change: an owner decision) or state the residual in `stable_api.md` and `safety.md`.
- Narrow the note's claim to "`device=` cannot skip it".

### F4 (SHOULD-FIX, fail-open): the gate fix can make a bare `maxim` move motors

§5 says the gate should "read the parsed value". If that value is the selection after precedence (env, then the `connect` default), every bare `maxim` after `maxim connect reachy` enters the robot runtime instead of `_bare_maxim_menu()`. Today `MAXIM_ROBOT_NAME=x` already does this, because `robot_name != "reachy_mini"` counts as an action. Separately, `tests/unit/test_cli_action_gate.py::_compute_has_action` re-implements the gate. Changing `cli.py` leaves that test green whatever it does.

**Fix:**
- Extract `cli.py::_has_robot_action(args, raw_argv)` (this pays the touch rule).
- Count only explicit argv flags (`--device`, `--robot`, `--robot-name`), never env or the stored default.
- Point `test_cli_action_gate.py` at the extracted function, with cases for `--robot-name x`, `--robot=x`, `--device x`, and "default set, no flag → menu".

### F5 (SHOULD-FIX, fail-closed): "unknown names are refused" against the load-bearing primary fallback

- `resolve_robot_entry` falls back to the unambiguous primary when no entry has the id. `agentic_runtime.py` calls that fallback "load-bearing for the default single-robot config", because the runtime name `reachy_mini` differs from the yaml key `primary` in `data/util/robots.yaml`, and Exp 49's key is `exp49_sim`. The fallback is pinned by `test_exp49_harness_seams.py::test_resolve_robot_entry_exact_match_then_primary` and by `test_robots_yaml_connect_config.py`.
- D3 makes `--robot` entry-id-valued and says "the default entry id equals today's default `robot_name`". That is false for the shipped template, so `maxim --robot reachy_mini` (documented in `ARCHITECTURE.md` and `getting_started.md`) would be refused.
- §6's "unknown name refused, never defaulted" does not say whether it applies to the implicit (no-selection) path. If it does, every default config breaks.

**Fix:** state two rules.
- **Explicit selection** (`--device`/`--robot`, env, `connect` default) is an exact id or a refusal. One bridge: if the value matches no id but equals the single entry's `config.robot_name`, accept it with a one-line hint naming the entry, or refuse with that fix.
- **No selection** keeps the unambiguous-primary fallback.

Add `ARCHITECTURE.md` to the docs list.

### F6 (SHOULD-FIX, fail-closed): where the default device lives

"Stored in `config.json` through `maxim config`" touches four contracts the note does not list.
1. `config_loader.py` rejects unknown keys in a same-version file, so a new key needs a schema section, a `CONFIG_FORMAT_VERSION` minor bump and a downgrade entry (#974).
2. The writer call from `maxim connect` or `maxim.devices.use` is a new `set_field`/`mutate_config` caller that `test.yml`'s `CONFIG_WRITER_MATCHES` fails on.
3. `utils/audit.py::LAYER_RULES` forbids `hardware/` from importing `runtime/`, so §7's "registry core lives in `hardware/`" cannot read the default.
4. config.json lives under XDG, not `MAXIM_DATA_HOME`. `scripts/exp49/run_trials.py` isolates robots.yaml through `MAXIM_DATA_HOME` and runs `maxim --mode live` with no robot flag, so on any box where the operator ran `connect`, a trial resolves a name that is absent from its isolated file and is refused.

**Fix (recommended):** store the default in robots.yaml itself (top-level `default: <entry>`, written only by the one writer). It rides the same store, the same isolation and the same layer. If config.json is kept, list all four items and say the default is ignored when `MAXIM_DATA_HOME` is set.

### F7 (SHOULD-FIX, fail-open): the writer allow-list misses its wrappers

The `write_mesh_config` grep (`test.yml` `MESH_WRITER_MATCHES`) catches callers of one function. §3b makes `maxim.devices.add/rename/forget/use` and `embody(..., save_defaults=)` public library functions that wrap it, so any in-tree module (or an agent tool) calling them writes robots.yaml unseen. That is the gap `SETUP_VERB_MATCHES` closed after the 2026-07-26 post-merge round (Arch #3).

**Fix:**
- A second CI grep allow-listing callers of the wrapper verbs: `cli.py`'s dispatcher plus their tests.
- A rule that no `tools/` module imports `maxim.devices` writers.
- A note that the grep sees function names, not the raw writes in `scripts/exp49/run_trials.py` or test fixtures. Those are isolated by `MAXIM_DATA_HOME`, so this is acceptable as a stated residual.

### F8 (SHOULD-FIX, fail-open): persistence symmetry and the cwd input

- `find_config_file` reads `data_home/robots.yaml`, else **`./robots.yaml`**. If the writer always targets `data_home`, the first `device add` in a directory that holds a cwd file creates a data-home file, which shadows every cwd entry from then on. The writer must write the file the loader resolved, or refuse when that file is the cwd one.
- §6 does not list the cwd file as an input. A cloned repository's `robots.yaml` can supply `host:`, `type:` and, after D8, `launch:` defaults (F9). Treat the cwd file as unpinned and untrusted, and say so in `show`.
- On the clear and move paths: `rename` must move the stored default, and `forget` must clear it and the `device_state.json` row. Unpinned legacy entries have no `reachy_mini:<hardware_id>` key, so §2 needs to say what key their cached address uses, or that they get no cache.
- Prove each with a round trip through the production writer and resolver.

### F9 (SHOULD-FIX, fail-open): file-stored launch defaults raise capability

The #829/#924 invariant at `cli.py::_runtime_mode_switch_allowed` is "Only the operator's launch flag raises capability". D8 lets `launch.operational_mode` (singularity included) come from robots.yaml, which may be the cwd file (F8) or a flag saved months ago. `_validate_operational_grant` also requires `_explicit_mode_flag(raw_argv)`, so a grant from launch defaults is either refused with exit 2 or has to bypass the validator.

**Fix:** launch defaults may set `mode`, `llm` and `interactive`, but `operational_mode` above `passive` must be typed on each invocation, or at least printed in the §3a plan block and refused without a TTY confirmation. Route every grant through `_validate_operational_grant` with an explicit grant source, and print the singularity warning unchanged.

### F10 (SHOULD-FIX): factoring the robot launch into a library function

- The path has CLI-process side effects a library must not perform: `configure_logging(..., force=True)` (clobbers an embedder's logging); `_after_run_mode_request` → `os.execve` (**replaces the embedder's process**); `_end_robot_run` → `stop_active_spawner()` (kills a shared LLM server); and the `while True` mode loop. Split the seam: a library `launch_embodied(resolved, plan, *, stop=...)` with no logging reconfiguration, no exec and no spawner kill, which returns a typed `requested_mode`. The CLI keeps the logging, the loop and the re-exec.
- **Verified:** `register_tool` and `maxim.on` do not reach the robot runtime today. `api._inject_registered_tools` and `api._bridge_event_subscriptions` are called only from `api.run` and `simulation/orchestrator.py`. The fold has two sites, the agentic runtime's tool registry and its bus, and needs a test that a registered tool is offered and a `tool_call` handler fires under `embody`.
- `api._RUN_LOCK` must also guard `embody()`, otherwise `run()` and `embody()` can hold two SDK sessions at once.
- `tests/unit/test_operational_mode_launch.py::test_the_robot_runtime_carries_the_grant` asserts `'operational_mode=getattr(args, "operational_mode", None)' in inspect.getsource(cli)`. Re-point it to the new seam behaviourally; do not delete it.
- `Maxim.__init__` is pinned at 312 lines. F1's handler change and the `robot_id`/`robot_name` split land inside it, so the touch rule applies there as well as to `_main_impl`.

### F11 (SHOULD-FIX, fail-open): `embody` already names a different seam

`embodied_device.md` §2 plans `ReachyMiniController.adopt(sdk)` + `MaximHandle.embody(sdk)`: the embedder (Pulse) opens the SDK and Maxim adopts it, with no connect of its own. §3a says the CLI verb will "route through it", but no pin is checked on adopt, so that later path skips identity entirely. The meanings also diverge: `embody(sdk)` attaches, while `maxim.embody("reachy")` resolves and runs a blocking loop.

**Fix:**
- `adopt` verifies the pin against the adopted handle's host, or takes a `ResolvedDevice`. Note this in the overlap section.
- Choose distinct names, or one signature.
- `maxim.devices` (plural, this note) and `maxim.device.run_device` (singular, `embodied_device.md`) should not both exist.

### F12 (SHOULD-FIX, fail-open in tests): consolidating the probes

- **Outcome semantics differ, and each is pinned.** The doctor treats an unreadable status as `warn` (`test_doctor.py::test_status_unreadable_is_warn_pre_15_daemon_hint`). The controller treats it as non-fatal. With a pin it becomes a refusal, but only for pinned entries. The one probe must return distinct outcomes (probe 4) and let each consumer map them.
- **The status dict is an input to `#1704` media ownership** (`_connect_released_media`, `_warn_if_media_already_released`). A typed report replacing the dict must keep `media_released` tri-state (`True`/`False`/absent).
- **Test seams.** Four test files monkeypatch `ReachyMiniController._daemon_status`/`_port_open`/`_resolve_mdns`, three of them in autouse fixtures. If connect stops calling these methods but the names remain, the fixtures go vacuous and the tests make real HTTP and TCP calls. Either keep them as the probe's delegating seams, or delete them so `monkeypatch.setattr` (raising) fails loudly.
- **Doctor patch targets.** The doctor tests patch `socket.create_connection` and `maxim.utils.http.fetch_url` by module attribute, so a `from maxim.utils.http import fetch_url` in the new module makes every patch miss.
- **Raw-dial pins.** `lint_http_surface.py::RAW_DIAL_PINS` is exact both ways, so every moved `create_connection` re-pins.
- **Surfaces to settle.** Name the fate of the `maxim-diagnostics` console script (`pyproject.toml`) and of `scripts/check_reachy_connection.py` (D33, and three docs cite it).

### F13 (SHOULD-FIX, fail-open): discovery against the egress register and the latch

- `lint_http_surface.py::EGRESS_SDK_IMPORTERS["reachy_mini"]` is an exact set: audio_localization, controller and motion_path. The `find_robots` importer must be added.
- zeroconf multicast is not in rule C's list, which the docstring states as an evasion it cannot see.
- §6 relies on the V8 offline latch, but `tests/integration/test_offline_latch.py::CENSUS_SITES` holds only the `utils.http` and download sites. Add the discovery entry point to the census, with a red gate that discovery under the latch raises `DiscoveryUnavailable` and sends nothing.

### F14 (SHOULD-FIX, fail-open): the verified address must be the dialled address

- `connection_mode: auto` makes the SDK try localhost first and then the host. The controller verifies at `probe_host` (the host), but the session can land on 127.0.0.1: a different daemon, or a tunnel. Refuse `auto` for pinned entries, or verify at both.
- For §4's hostname candidate (step 3), the resolver must pass the **verified IP literal** to `ReachyMini(host=)`, as `_resolve_mdns` does today. Otherwise two `reachy-mini.local` robots can swap between the check and the dial.
- Under `tunnel: true` the check reads 127.0.0.1. That is allowed under `address_policy="private"`, and correct.
- (Unverified) with `HTTP_PROXY` set, `utils/http`'s "the proxy wins for operator paths" routes the status read through the proxy, while the SDK's WebSocket may dial directly.

### F15 (NIT, residual): unpinned SDK construction sites

`scripts/orient_backbone/live_common.py`, `live_1_smoke.py`, `scripts/reachy_streamer.py` and `embodiment/audio_localization.py::make_reachy_doa_reader(None)` build `ReachyMini(` themselves, so they never see the pin. Scope §4's claim to controller paths, and list these as the residual that `embodied_device.md`'s one-SDK-connection grep will retire.

### F16 (NIT, residual): the version key

The repo's persisted-file convention is `_format_version` with `utils/format_version.py::check_format_version` (the refuse-newer and one-shot-warning machinery). Ride that with the same key name rather than adding `format_version`. State that every reader up to 1.3.4 ignores both the key and an in-`config:` pin. A new-writer file read by old code connects unpinned, which is the F1 point 3 residual.

### F17 (NIT): environment variables

- §3's "env" precedence layer names no variable. If it is `MAXIM_ROBOT_NAME`, the two meanings recombine. Name a new one (for example `MAXIM_DEVICE`) and keep `MAXIM_ROBOT_NAME` as the daemon name only.
- §5's "appear in no brief table" is true of `docs/agents/`, but `docs/user/configuration.md` already documents `MAXIM_ROBOT_NAME`, with "zenoh namespace" text that is stale, as is `cli-reference.md`'s "Zenoh discovery".

### F18 (NIT): the loud-yaml fix has more silent sites

`RobotsConfig.from_dict` silently skips non-dict entries (`if isinstance(robot_config, dict)`). The probe showed that `config: null` and a top-level list both become an empty config through the same handler. The typed error should cover all three. Removing `from_yaml`'s broad handler only lowers its swallow-lint count, and `test_gate_malformed_config_is_warn_never_fail` stays valid.

### F19 (NIT): neighbours to note

- `Maxim._connect_kwargs` (`selfy.py`) has no reader: dead.
- `hardware/config.py::setup_robots_from_config`/`connect_configured_robots` are exported, with no `src/` caller. They are a third launch path that connects every entry. They carry the pin only if F1 makes it a ctor parameter.
- Top-level `maxim connect` (device) sits beside `maxim peer connect` (leader config). Use one sentence in the CLI help to disambiguate.
- `_NON_SIM_SUBCOMMANDS` membership for `device`/`discover`/`connect`/`embody` is right (#1344). Dispatch them from a helper like `_dispatch_tool_cli`, not inline in `_main_impl`.
