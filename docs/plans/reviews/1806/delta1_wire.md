# #1806 device registry: wire-integrity review, delta round 1

Reviewer: wire-integrity lens (docs/CODE_REVIEW.md), delta round 1, read-only, 2026-11-25, v2 @ f0369379 (code at origin/main b7380ee7, incl. `scripts/`, `tests/`, `.github/`). A fresh reader; round 1 ([wire.md](wire.md)) was by a different reviewer. One argparse probe ran in a scratch interpreter. I contacted no robot or LAN host and ran no sim.

**Verdict: SHOULD-FIX, no DO-NOT-BUILD.** v2 folds both round-1 DO-NOT-BUILDs (F1, F2). Fourteen findings are FOLDED and five are PARTIAL. The new contracts have ten findings. Five are fail-open (N1, N3, N4, N5, plus the F4 remainder in N7), and each is a fix to the note's text inside a class already covered. None changes an owner decision. The stop rule is **not** met (see the end), so a delta round 2 scoped to N1–N5 is owed before PR 0.

## 1. Disposition of round-1 findings

| F# | Status | Note |
|---|---|---|
| F1 | PARTIAL | (a) the loud load and the end of the selfy/agentic swallow are folded (§4.1). (c) the registry refuses a class that lacks `supports_identity_pin` (§4.1). (d) the old-reader residual is stated (§4.3). **Open:** (b) unknown entry-level keys are not refused. A typo `pinn:`, or a `pin:` placed under `config:` (where v1 told operators to put it), still parses as an absent pin and gives an UNPINNED entry, with only a WARNING. F1's red gate (a malformed or pinned file plus a fake controller must raise) is not named |
| F2 | FOLDED | `reexec_with_mode` carries `--device` (§3a). The test is not named. Add "a run-mode switch keeps the device" beside `test_operational_mode_launch.py::test_a_run_mode_switch_keeps_the_operators_grant`. The interaction with the `--robot-name` the re-exec always emits is new (N5) |
| F3 | FOLDED | The reuse branch compares the pin (§4.1). The positional `maxim.connect` / `run(robot=)` are stated as unpinned (§3b) |
| F4 | PARTIAL | The gate reads explicit flags and its test calls the real gate (§3.1). **Open:** `cli_parser.py::_build_parser` sets `--robot-name`'s default from `$MAXIM_ROBOT_NAME`, so the parsed value cannot tell an explicit flag from the env. A bare `maxim` with `MAXIM_ROBOT_NAME` set still enters the robot runtime (N7) |
| F5 | FOLDED | Exact-or-refused for an explicit name, primary fallback only for the no-name path, `--robot` unchanged (D3′). Because `--robot` keeps working, the `ARCHITECTURE.md` doc edit is no longer needed |
| F6 | FOLDED | `default:` lives in robots.yaml (D2′). Its consumer is unstated (N3) |
| F7 | PARTIAL | An allow-list and an import-ban are named (§3b, §7). A name grep and an import ban cannot see the actual call shape (N4) |
| F8 | FOLDED | The cwd fallback is removed, the writer targets the data home, `rename`/`forget` run in one transaction, and unpinned entries never cache. The fallback branch that D11 leaves behind is new (N1) |
| F9 | FOLDED | The loader refuses `launch.operational_mode`, and `_validate_operational_grant` runs on the typed flags (§3a) |
| F10 | MOVED (#1808) | The inventory is carried over. #1808 does not name the source-pin test `test_operational_mode_launch.py::test_the_robot_runtime_carries_the_grant` (re-point it, do not delete it). Add one line there |
| F11 | PARTIAL | One namespace (D12). **Open:** `embodied_device.md`'s `adopt(sdk)` (where the embedder opens the SDK) still has no pin check. #1808's "the pin is inherited" covers `launch(plan)`, not `adopt`. State that `adopt` takes a `ResolvedDevice` or re-verifies on the session (§4.3 step 3) |
| F12 | FOLDED | Outcome mapping, non-vacuous seams, exact raw-dial pins, and the fate of both surfaces are named. The standalone contract of `check_reachy_connection.py` is new (N9) |
| F13 | FOLDED | Census plus the `EGRESS_SDK_IMPORTERS` entry. zeroconf is excluded by D7 |
| F14 | FOLDED | `connection_mode="network"` is forced, the IPv4 literal is dialled, and the identity is re-verified on the session (§4.3) |
| F15 | FOLDED | Listed as a residual (§4.3) |
| F16 | FOLDED | Rides `_format_version`. The refuse-newer mechanics are new (N8) |
| F17 | FOLDED | `MAXIM_DEVICE` is a new variable, and `configuration.md` is on the docs list |
| F18 | FOLDED | Non-dict entries, `config: null` and a top-level list all raise (§4.1) |
| F19 | FOLDED | `_connect_kwargs` and the dead `timeout:` are removed, the `setup_robots_from_config` path is dormant or routed, and the `peer connect` disambiguation is in |

## 2. Producer → consumer maps for v2's new contracts

### (a) D11: the cwd fallback goes and `MAXIM_DATA_HOME` must be absolute

| Site (outside the change) | Relies on today | After v2 |
|---|---|---|
| `hardware/config.py::DEFAULT_CONFIG_PATHS` (`["robots.yaml"]`) | the cwd fallback | removed. No other `src/`, `scripts/` or `tests/` reader of the constant (grep) |
| `hardware/config.py::find_config_file`, `except Exception` branch ("data_home() unresolvable — search limited to cwd") | cwd-only search | **becomes an empty search, so None, so an empty config, so an unpinned mDNS connect (N1)** |
| `load_robots_config` callers: `selfy.py::Maxim.__init__`, `agentic_runtime.py::_resolve_body_wiring`, `doctor/checks.py::check_robot_reachable`, `hardware/__init__` export | either path | data home only. Doctor maps any raise to warn with fix text "fix ~/.maxim/robots.yaml (YAML syntax / schema)". That text is wrong for a relative data home (N2) |
| Tests: `test_exp49_harness_seams.py::test_find_config_file_honors_data_home` (absolute `tmp_path`); `test_doctor.py`, `test_multi_robot.py`, `test_motion_finite_1496.py` patch `load_robots_config` | none use cwd | unaffected. No test `chdir`s next to a robots.yaml (grep) |
| `MAXIM_DATA_HOME` setters: every test uses `tmp_path` (absolute); `exp49/run_trials.py` uses `out_dir.resolve()`; `exp44/campaign.py` uses `.resolve()`; `o19_rerun.py` and `run_v1_phases.sh` use `mkdtemp` | | absolute ✓ |
| `scripts/benchmark_exp41_exploration.py` (`workdir = Path("data/sim_sandbox/exp41_runs")` → `env["MAXIM_DATA_HOME"] = str(data_home)`); `benchmark_exp42_preference.py`, `benchmark_cradle_mother.py`, `benchmark_cross_session.py` (`--workdir` as the user typed it, not resolved) | **relative** values | breaks only if D11 lands in `utils/paths.py::data_home()` rather than in robots.yaml resolution (N2) |
| Docs: `docs/plans/archive/doctor_robot_reachable.md`, `overdrive_refactor_candidates.md` mention the cwd fallback | | archive/plan text only. No user doc teaches `./robots.yaml` |

### (b) New entry-level fields and top-level keys

| Key | Producers | Consumers (new readers / old readers) |
|---|---|---|
| `pin:`, `hostname:`, `launch:` (entry level) | the writer (PR 2), the operator | `RobotConfig.from_dict` (new typed fields). Old 1.3.x `from_dict` drops them, so an old build connects unpinned (stated). `hostname:` vs `config.host`: §4.2 names only `host:` as candidate 1, so `hostname:` has no stated consumer (NIT, folded into N3's fix text) |
| top-level `default:` | `use` / `connect` (writer) | **unstated** (N3). Old readers ignore it and pick `primary:` |
| top-level `_format_version` | the writer | the new loader's refuse-newer (no helper exists, N8). Old readers ignore it |
| `scripts/exp49/exp49_common.py::robots_yaml_text` → `run_trials.py::run_spawned_trial` (`write_text`) | harness | **Loads under v2:** the root is a dict, the entry `exp49_sim` is a dict, `config:` is a non-null dict, and the entry keys are only `type`/`primary`/`config`. With no `pin:` it is an unpinned `simulated` entry (SimulatedController needs no `supports_identity_pin`). With no `_format_version` it gets the legacy warning once per trial (N8). `--mode live` with no name means the no-name path, and `primary: true` resolves ✓. Only an inherited `MAXIM_DEVICE` breaks it (N6) |
| `data/util/robots.yaml` template (id `primary`, `config.timeout`) | repo | dead `timeout:` removed (§5). Loads ✓ |

### (c) `--robot` alias, `--device`, `MAXIM_DEVICE`

| Site | Effect |
|---|---|
| `cli_parser.py::_build_parser` options starting `--de`/`--dev`/`--rob`: `--deep-embodiment`, `--delete-model`, `--debug`, `--robot-name` | **Probe (py3.12):** `--dev z` and `--devi=z` parse to `--device`, with no collision (`--de` is already ambiguous today). An explicit `--robot` alias makes `--rob`/`--robo` **ambiguous errors** ("could match --robot-name, --robot"); today they abbreviate `--robot-name`. `--robot` and `--robot=y` parse ✓ (N7) |
| `maxim peer update --dev`, `mesh_cli --dev` | dispatched before `parser.parse_args` (`cli.py::_main_impl` peer branch), so unaffected ✓ |
| `cli.py::_main_impl` gate (`"--robot" in raw_argv`), `tests/unit/test_cli_action_gate.py::_compute_has_action` (a copy) | the gate must read `args.device is not None`, not raw tokens (abbreviations, `=`). F4 remainder (N7) |
| `cli_utils.py::reexec_with_mode` | always emits `--robot-name <args.robot_name>`. Adding `--device` gives a child with both (N5). `os.execve(..., child_env())` passes `MAXIM_DEVICE` through |
| `utils/child_env.py::child_env` → every harness spawn (`exp49/run_trials.py::_trial_env`, the exp41/42/44, cradle and cross-session benchmarks, `o19_rerun`) | passes `MAXIM_DEVICE` and `MAXIM_ROBOT_NAME` through unchanged (N6) |
| `tests/conftest.py` | no scrub exists today for `MAXIM_ROBOT_NAME`/`MAXIM_REACHY_HOST`/`MAXIM_ROBOT_TIMEOUT`. v2 adds them ✓ |

### (d) D13: a named device's connect failure exits non-zero

| Path | Today | After v2 |
|---|---|---|
| `ReachyMiniController.connect` → `registry.py::connect_robot_with_status` | `False` → `None`; an exception is re-raised after `disconnect()` | a typed raise propagates ✓ |
| registry: unknown `type:`, or a type mismatch on reuse | `return None` before any pin logic | a pinned entry with a typo'd `type:` gives None, so headless on the unnamed path. Safe, but not "refused" (N10) |
| `selfy.py::Maxim.__init__` (`if self._robot is None: "running in headless mode"`) | headless | needs a typed "named" input to choose between exit and headless. Only one constructor site exists (`cli.py::_main_impl`) |
| `cli.py::_run_stopped_on` | re-raises `BackendError`/`LeaderRefusal`/`SpeechOutputUnavailable`, giving exit 2 | must add the device errors (§3b says so) ✓ |
| Tests that expect the headless fallback: `test_connection_state.py::test_headless_gets_2hz`, `test_capabilities.py` (defaults only), `test_api_core.py::test_run_rejects_robot_in_headless_mode` (an API flag) | none drive a named failed connect | unaffected. `test_cli_exit_code_1468.py` needs a new row |

### (e) `probe_daemon` consolidation

| Surface | Contract | After v2 |
|---|---|---|
| `pyproject.toml [project.scripts] maxim-diagnostics = maxim.utils.reachy_diagnostics:main` | an installed entry point. It uses `maxim_http.fetch_url` (so it passes the latch) plus one raw `socket.socket` (`lint_http_surface.py::RAW_DIAL_PINS` entry) | a wrapper works. The raw-dial pin moves with it ✓ |
| `scripts/check_reachy_connection.py` | **its docstring says "standalone ... doesn't require the maxim package to be installed"**. It uses raw `urllib.request.urlopen` (in `scripts/`, outside the CI urlopen grep). Cited by `docs/reference.md` | a "thin wrapper" breaks the no-install use (N9) |

### (f) Writer allow-list and import ban

| Gate | Where | Sees |
|---|---|---|
| `MESH_WRITER_MATCHES`, `CONFIG_WRITER_MATCHES`, `SETUP_VERB_MATCHES` | `.github/workflows/test.yml`, step "CI grep invariants ...", scanning `src/maxim/` and `tests/` | function-name calls and from-imports. The mirror for `write_robots_config(` fits here ✓ |
| Verb wrappers `maxim.devices.add/use/forget/rename/relocate/repin` | the lazy namespace via `maxim/__init__.py::__getattr__` | generic names, so a name grep cannot work. `import maxim; maxim.devices.add(...)` contains no `maxim.devices` import (N4) |
| `utils/sandbox_executor.py::BLOCKED_PYTHON_IMPORTS` | sandboxed model code | does not list `maxim` (N4 residual) |
| "skills/" in §3b's ban list | | no `src/maxim/skills/` exists (N4) |

## 3. New findings

| # | Finding | Class | Fail-open/closed | Covered class? |
|---|---|---|---|---|
| N1 | Removing the cwd entry turns `find_config_file`'s "data_home() unresolvable" fallback into an empty search: no file, so an empty config, so an unpinned mDNS connect, with a WARNING only. A relative `MAXIM_DATA_HOME` has no stated outcome | silent config loss | fail-open | yes (F1a, loud load) |
| N2 | D11's absolute rule must be scoped to robots.yaml resolution and the writer, not to `utils/paths.py::data_home()`. `benchmark_exp41_exploration.py` sets a relative data home, and three benchmarks pass `--workdir` unresolved. Doctor's fix text is wrong for this case | scope / harness breakage | fail-closed | yes (F8) |
| N3 | No consumer is stated for `default:`. The CLI never passes `robot_id=`, so `resolve_robot_entry` falls back to `primary:`, not `default:`. `_resolve_body_wiring` re-loads robots.yaml on its own, so the body and `motor_binding` can come from a different entry or read. Old readers follow `primary:`, which can be a different robot from the one `connect` chose | resolution split | fail-open | yes (one resolver; adversarial #11 carry-forward) |
| N4 | The writer lint cannot see the call shape: verb names are generic, the call is attribute access through a lazy namespace, `skills/` does not exist, and the sandbox does not block `import maxim` | writer bypass | fail-open | yes (F7) |
| N5 | `--robot`/`--robot-name` is missing from the selection precedence. The re-exec emits `--robot-name` beside `--device`. `maxim --robot X` with `default:` set is undefined | selection precedence | fail-open (wrong, though verified, robot) | yes (F2/F5) |
| N6 | `child_env()` passes `MAXIM_DEVICE` into harness children. exp49 runs `--mode live` against an isolated file holding only `exp49_sim`, so the explicit name is refused and the trial fails | env leak into harness | fail-closed | yes (env-scrub lesson) |
| N7 | The `--robot` alias makes `--rob`/`--robo` ambiguous errors, and `--dev` now abbreviates `--device`. The gate must read the parsed `args.device`. The F4 remainder: the env default of `--robot-name` still makes a bare `maxim` a robot run | argparse / gate | fail-open (F4 part) / fail-closed (ambiguity) | yes (F4) |
| N8 | `utils/format_version.py` has no refuse-newer helper (`check_format_version` only returns and warns). Every hand-written robots.yaml (the template, every exp49 trial) logs "pre-1.0 ... Re-save with this build" | persistence mechanics | fail-closed | yes (F16) |
| N9 | `scripts/check_reachy_connection.py` is documented as no-install and standalone, so a wrapper breaks that contract | surface contract | fail-closed | yes (F12) |
| N10 | D13 vs §4.3 step 4: a typed raise from `connect()` propagates on **every** path, so a pinned entry reached by the no-name path also exits non-zero, against "unnamed keeps headless". The set of "named" sources is undefined. An unknown `type:` returns None before the pin check | lifecycle wording | fail-closed | yes (D13 / safety #15) |

## 4. Evidence and fixes

**N1.** `hardware/config.py::find_config_file`: `paths = [str(data_home() / "robots.yaml"), *DEFAULT_CONFIG_PATHS]`, and in the `except Exception` branch `paths = list(DEFAULT_CONFIG_PATHS)`. Once `DEFAULT_CONFIG_PATHS` is emptied, that branch returns None, and `load_robots_config` returns `RobotsConfig()`, which §4.1 calls "a missing file is fine". In `selfy.py::Maxim.__init__` the connect then dials `{"robot_name": self.name}` over `reachy-mini.local`, the very non-unique name §6 lists. **Fix:** an unresolvable or relative data home raises `RobotsConfigError` on the robot path. "No file means no entries" applies only to a resolved absolute home. `write_robots_config` applies the same check. Red gate: `MAXIM_DATA_HOME=rel/path` plus a pinned entry must refuse.

**N2.** `utils/paths.py::data_home` accepts any value (`Path(raw)`). There are 49 `data_home()` call sites in `src/maxim/`. `scripts/benchmark_exp41_exploration.py` (`workdir = Path("data/sim_sandbox/exp41_runs")`) sets a relative `MAXIM_DATA_HOME`. **Fix:** state that D11's rule lives in `find_config_file` and the writer only. Doctor's robots.yaml warn names the real cause (a relative data home) and the resolved path, not `~/.maxim`.

**N3.**
- `cli.py::_main_impl` builds `Maxim(robot_name=args.robot_name, ...)` with no `robot_id=`.
- `selfy.py::Maxim.__init__` resolves `effective_robot_id = self._robot_id or self.name` through `resolve_robot_entry`, which knows only the exact id and `primary:`.
- `agentic_runtime.py::_resolve_body_wiring` calls `load_robots_config()` a second time and matches inline.

**Fix:**
- The no-name step of `resolve_robot_entry` reads `default:` before `primary:`, so the CLI, selfy, doctor and `maxim.devices.resolve` all share one rule.
- Selfy hands the entry it resolved to the agentic runtime (`_resolved_robot_config`), so there is no second load.
- `use` also moves `primary: true` onto the default entry, so a 1.3.x reader picks the same robot. Front gate: ask whether `default:` should simply *be* `primary:`, since the writer owns the file anyway.
- State the role of `hostname:` relative to `config.host` in §4.2.

**N4.** In `test.yml`, `MESH_WRITER_MATCHES` greps `write_mesh_config\(`. The robots.yaml writer can take the same form for `write_robots_config(`. The verbs cannot. **Fix:**
- An AST lint in the `scripts/lint_http_surface.py` style flags, under the banned trees (`tools/`, `agents/`, `runtime/`, `embodied_runtime/`, `embodiment/`, `reactions/`, `default_network/`), any `maxim.devices.<writer>` attribute chain, `from maxim.devices import <writer>`, and `from maxim import devices`. A deletion probe is required.
- Drop `skills/`.
- Either add `maxim` to `sandbox_executor.BLOCKED_PYTHON_IMPORTS` or state it as a residual next to the "not a security boundary" note.

**N5.** `cli_utils.py::reexec_with_mode` always emits `"--robot-name", str(getattr(args, "robot_name", "reachy_mini"))`. **Fix:** state the order as `--device` > an explicit `--robot`/`--robot-name` (today's rule: exact id, else the unambiguous primary; it never reads `default:`) > `MAXIM_DEVICE` > `default:` > single entry or `primary:`. `--device` together with an explicit `--robot` that names a different entry is a refusal. The re-exec emits `--device <id>` and drops `--robot-name` when the run was device-selected. Test: "a run-mode switch keeps the device".

**N6.** `utils/child_env.py::child_env` changes only `PYTHONPATH`. **Fix:** `exp49/run_trials.py::_trial_env` pops `MAXIM_DEVICE` (and `MAXIM_ROBOT_NAME`). The other spawning harnesses either do the same or are sim-only (no robot path), and the note says which.

**N7.** Probe output: `--rob y` gives "error: ambiguous option: --rob could match --robot-name, --robot". **Fix:** accept the change and list it in the CHANGELOG fragment. Give `--robot-name` `default=None` and apply `$MAXIM_ROBOT_NAME` after the gate, so env never counts as an action. The gate is `args.device is not None or args.robot_name is not None` (parsed, which covers abbreviations and `=`).

**N8.** `utils/format_version.py::check_format_version` returns `LEGACY_VERSION` and warns "Re-save with this build" once per type. It never refuses. **Fix:** name the compare that refuses newer (major > 1, a typed `RobotsConfigError`). For a robots.yaml with no version, log at DEBUG, or give a hint that does not tell a hand-editor to "re-save".

**N9.** `scripts/check_reachy_connection.py` docstring. **Fix:** deprecate it with a pointer to `maxim device check` / `maxim-diagnostics`, which is what `docs/plans/archive/doctor_robot_reachable.md` already chose, and update `docs/reference.md`. Or keep it standalone and list it as a residual. Do not make it import `maxim`.

**N10.** `registry.py::connect_robot_with_status` (`except BaseException: controller.disconnect(); raise`, and `return None` on an unknown `robot_type`). **Fix:** define "named" as `--device`, `MAXIM_DEVICE`, an explicit `--robot`, and say whether `default:` counts. Carry it into `Maxim` as a typed parameter. Decide whether the no-name path to a pinned entry raises (recommended: yes, strict, matching §4.3) and reword D13 to match. Refuse an unknown `type:` on a pinned entry.

## 5. Stop-rule outcome

**Not met.** N1, N3, N4, N5 and the F4 remainder in N7 are fail-open. All of them fall in already-covered classes (loud load, one resolver, writer allow-list, selection precedence, explicit-flag gate), and each is a fix to the note's text. N2, N6, N8, N9 and N10 are fail-closed and covered, and can go to a follow-up under the stop rule. Because fail-open findings remain, a delta round 2 is owed, scoped to the v3 text for N1, N3, N4, N5 and N7 (plus F1b's unknown-key refusal and F11's `adopt` pin). If v3 folds those as written above, I expect that round to meet the stop rule.
