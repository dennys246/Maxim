# #1806 PR 0 code review: wire integrity

Reviewer: wire-integrity lens, PR 0 code review, 2026-11-25, 6b4ad2bb

**Verdict: SHIP after fixes** (W-1 turns the fast suite red; W-2 is a ≤5-line other site of the PR's own pattern).

Method: `git archive 6b4ad2bb` into a scratch tree, scratch `HOME`, `PYTHONPATH=<copy>/src`. Greps over `src/ scripts/ tests/ docs/ bench/`.
Ran 97 neighbouring unit files (reachy/robot/doa/readback/doctor/http/filesystem/sandbox/write/autonomy/cli/selfy/mdns/media/motion/exp49/r3/operational/tool),
all `test_1806_pr0_*`, `test_multi_robot`, `test_motion_finite_1496`, `test_architecture_audit`, `test_lint_http_surface`,
`test_robots_yaml_connect_config`, `test_exp49_harness_seams`, `test_r3_run`, `test_doa_feed`, `test_audio_localization`, `test_http_deadline_1715`,
plus `lint_http_surface`, `lint_orphan_modules`, `lint_no_silent_swallows`, `lint_function_length` (all clean).
Failures: 2 in `test_audio_localization.py` (W-1, caused by this diff). 3 in `test_m1b_writer_stamps_5a3.py`, which come from the scratch copy having no `.git` (`code_tree_sha256: 'unknown'`) and are unrelated.

## Producer / consumer maps

### (a) `--robot-name` default `None` + `--robot` alias

| Site (outside hunks unless noted) | Role | After PR 0 |
|---|---|---|
| `cli.py::_requested_run_action` / `_quick_start_route` (hunk) | consumer: `is not None` | correct |
| `cli.py::_robot_name_for_log` (hunk) | consumer via `resolve_daemon_name` | correct |
| `cli.py::_main_impl` → `Maxim(robot_name=args.robot_name)` (L3025) | consumer | correct: `None` goes to `resolve_daemon_name`, which gives the same env-then-default chain that argparse used to give at parse time |
| `cli_utils.py::reexec_with_mode` (hunk) | producer of child argv | correct: replays only a typed name, and `child_env()` passes `MAXIM_ROBOT_NAME` through unchanged |
| `selfy.Maxim.__init__(robot_name="reachy_mini")` direct callers (`runtime/capture.py` type-only; tests) | consumer | unchanged: the truthy default never read the env before either |
| `api.py::_connect_robot_for_run`, `api.connect` | parallel connect path | does not read `robot_name`/`robots.yaml` at all (pre-existing, W-4) |
| `scripts/reachy_streamer.py` | own parser (`--robot-name`) | independent, unaffected |
| `scripts/exp49/run_trials.py` | argv builder (`--mode live`, no robot flag) + `env.pop("MAXIM_ROBOT_NAME")` | correct: an explicit `--mode` still routes to a run |
| other `_build_parser` users (`hivemind/*`) | different parsers | no `--rob*` options, unaffected |
| docs: `cli-reference.md`, `configuration.md`, `embodiment.md`, `ARCHITECTURE.md:18`, `robot-setup.md:43`, `getting_started.md:99` | examples | all still valid; `--robot` is now an exact alias |

### (b) Removed `ReachyMiniController._resolve_mdns/_port_open/_daemon_status/_installed_sdk_version`, doctor `_robot_sdk_version/_parse_major_minor`

| Site | After |
|---|---|
| src/ scripts/ bench/ | zero references |
| tests (`test_mdns_discovery`, `test_reachy_connection_options`, `test_media_release_1704`, `test_media_stream_gating`, `test_1806_pr0_probe_daemon`) | moved to `reachy_probe.read_daemon_status` and `socket.*` seams; passing |
| `docs/plans/motion_path_model.md:115` | still cites `ReachyMiniController._daemon_status` (W-6) |
| `docs/plans/archive/doctor_robot_reachable.md:119`, `docs/plans/reviews/atonia_gate/item_10_adversarial.md` | archive/review records; leave |

### (c) `RobotsConfig.from_yaml/from_dict` raise, no cwd fallback, `DEFAULT_CONFIG_PATHS` removed

| Caller | After |
|---|---|
| `selfy.Maxim._resolve_robot_connection` | propagates, then `cli._run_stopped_on` exits 2 (tested by `test_cli_exit_code_1468`) |
| `doctor/checks.py::check_robot_reachable` | catches `RobotsConfigError`, gives a warn row; never fails doctor |
| `agentic_runtime._resolve_body_wiring` | no longer loads; reads the handed-over entry. The match rule (`resolve_robot_entry`) and the id (`_robot_id or name`) are identical to the removed inline copy |
| `maxim.hardware` public exports `load_robots_config`, `setup_robots_from_config`, `RobotsConfig` | now raise where they returned empty (W-7) |
| `DEFAULT_CONFIG_PATHS` | zero readers anywhere |
| `scripts/exp49/run_trials.py` | writes `<abs sandbox>/home/robots.yaml`, absolute `MAXIM_DATA_HOME`: correct |
| `scripts/survival_world/r3_run.py` | mentions `robots.yaml` only as a hashed tracked path; no load |
| relative `MAXIM_DATA_HOME` users (benchmarks, `run_v1_phases.sh`) | none launches an embodied mode, so the new refusal is reached only by robot runs and doctor |

### (d) `fetch_url(proxy=)` + `_external_policy_private_strict`

| Site | After |
|---|---|
| `probe.read_daemon_status`, `controller._get_state_full`, `audio_localization.make_reachy_rest_doa_reader` | `private` + `never` |
| `utils/reachy_diagnostics.py` Test 3 `/docs` | still default (`any`, env proxy) (W-2) |
| every other `fetch_url` caller (peer backends, doctor LLM probes, leader_proxy, tools/http_fetch, internet_search, mesh_trace, admin_core, local_server_spawner) | default `proxy="env"`; unchanged |
| `lint_http_surface.RAW_DIAL_PINS` | doctor 2→1, controller pin moved to `probe.py`; lint clean |
| `.github/workflows/test.yml` urlopen grep (L868) | src clean; `scripts/check_reachy_connection.py` + `scripts/orient_backbone/*` use urllib but are outside the grep (W-5) |
| `test_http_deadline_1715.py:173` hand list of endpoints | omits the strict one; the new test iterates `_DEADLINE_ENDPOINTS`, so it is covered |
| test fakes of `fetch_url` with a narrow signature | `test_audio_localization.py` breaks (W-1) |

### (e) `control_plane_write_refusal`

| Write path | Covered |
|---|---|
| `WriteFileTool`, `EditFileTool`, `WriteSandboxFileTool`, `WriteToSharedOutputsTool`, `SandboxExecutor.write_script` (and therefore `CreateSandboxScriptTool`) | yes |
| `SupervisionPolicy` `edit_file` approval (`autonomy.py`) | yes |
| other `WRITE_LOCAL` tools: `git_tools` (fixed argv), `respond`, `research_tools.record_experiment`, `tools_dm.emit_campaign` (harness-chosen path), `request_directory_change` (boundary only; writes still pass the refusal) | none takes a model-chosen path to the control plane |
| advisor `_session.py` | fd-anchored report writes; not a model path |
| bash / execute_file / run_tests | stated residual (`safety.md`) |
| config resolvers: `peer/config.py::peer_config_path`, `runtime/config_loader.py:635`, `runtime/role.py:100` | all `XDG_CONFIG_HOME` or `~/.config`, so they agree with the deny root |
| tests writing under tmp `.config/maxim` through a file tool | none found; the full neighbour set passes |

### (f) `selfy.Maxim._resolved_robot_config`, removed `_connect_kwargs`, `resolve_daemon_name`

| Reader | After |
|---|---|
| `agentic_runtime`: body wiring L91, DoA gate L174, motor binding L637/L655, aut_mode L1079 | all `getattr(..., None)`; the only host is `selfy.Maxim`, which sets it before any of them run |
| `_connect_kwargs` | zero readers |
| `resolve_daemon_name` | `cli._robot_name_for_log` + `selfy.Maxim.__init__` only |
| tests `test_doa_feed`, `test_motion_finite_1496`, `test_multi_robot` | set the attribute on stubs; passing |

### (g) `RUN_MODES` / `EMBODIED_RUN_MODES`

| List | Relation |
|---|---|
| `cli_parser --mode choices` | derived from `RUN_MODES` |
| `cli._RUNTIME_SWITCHABLE_MODES` (`sleep/live/agentic/passive/active`) | a different vocabulary (switch targets); not derived |
| `selfy.py:930` no-park list (`reflection/exploration/live/train/agentic/passive/active`) | `RUN_MODES - {sleep} + OPERATIONAL`; hand-written (W-8) |
| `EMBODIED_RUN_MODES` | zero non-test callers (W-8) |
| `docs/user/modes-guide.md` | updated |

### (h) env scrub

`tests/conftest.py::_scrub_robot_selection_env` scrubs all three variables. Every test that needs one sets it with `monkeypatch.setenv` after the autouse scrub has run: `test_operational_mode_launch`, `test_cli_action_gate`, `test_1806_pr0_robots_config`, `test_1806_pr0_probe_daemon`. `src/tests/basic_move.sh` reads `MAXIM_ROBOT_NAME`, but it is a manual script.

## Findings

| # | file::symbol | finding | severity | evidence |
|---|---|---|---|---|
| W-1 | `tests/unit/test_audio_localization.py::TestRestDoAReader::test_default_path_routes_through_maxim_http_not_raw_urllib`, `::TestBuildReachyAudioOrientingSource::test_network_source_pulls_through_the_rest_reader`, `::TestRestDoAReader::test_default_path_swallows_fetch_errors_to_none` | The fakes are `lambda url, timeout=None`, and the diff now calls `fetch_url(url, timeout=t, address_policy="private", proxy="never")`. The fake raises `TypeError`, the reader's `except Exception` turns it into `None`, and 2 tests FAIL. The third passes vacuously: it raises `TypeError` before its `RuntimeError` is ever reached. A signature mismatch at this seam silently disables DoA, and only these tests watch it. | important | scratch run: `2 failed ... test_audio_localization.py:293 assert percept is not None` |
| W-2 | `utils/reachy_diagnostics.py::diagnose_reachy` (Test 3, `/docs`) | Another site of the PR's own pattern. The robot daemon `/docs` read still goes through the env proxy with policy `any`, so a proxy can answer "HTTP 200" for a robot it never reached. The changelog's "Every read of the robot daemon is now dialled directly" is false for this site. | important | `reachy_diagnostics.py:163 fetch_url(f"http://{host}:8000/docs", timeout=3)` |
| W-3 | `hardware/reachy/probe.py::resolve_ipv4` used by `controller._precheck_daemon` | The connect pre-check is now IPv4-only (`gethostbyname`). A `host:` that is an IPv6 literal, or a name with only AAAA records, now fails as `resolve` and the connect is skipped. Before, `_port_open` used `create_connection`, which dials v6. The doctor half is in the changelog; the connect-path narrowing is not. | nit | `probe.py::resolve_ipv4`; removed `_port_open` |
| W-4 | `api.py::_connect_robot_for_run`, `api.py::connect` | A second live connect path that never reads `robots.yaml` and never calls `resolve_daemon_name`/`_resolve_robot_connection`. The "one entry the connect uses" claim covers only `selfy.Maxim`. This is PR 1's scope, because PR 1 routes every connect through the registry pin. | pre_existing | `api.py:303-316`, `api.py:1080-1100` |
| W-5 | `scripts/orient_backbone/{live_common,live_1_smoke,delivered_shift_block,body_yaw_sweep,ear_map}.py`, `scripts/check_reachy_connection.py` | Duplicated daemon-status probes (probe 9) on raw urllib, which honours env proxies. `check_reachy_connection.py` is now marked deprecated; the orient scripts are not, and they are not pointed at `probe_daemon`. | pre_existing | `grep -rn api/daemon/status scripts/` |
| W-6 | `docs/plans/motion_path_model.md:115` | Cites the removed `ReachyMiniController._daemon_status` as "the existing pattern". | nit | grep |
| W-7 | `hardware/__init__.py` exports `load_robots_config`, `setup_robots_from_config`, `RobotsConfig.from_yaml` | Behaviour change on public `maxim.hardware` API: a broken file now raises `RobotsConfigError` where it returned an empty config. The changelog states this only for robot runs. | nit | `hardware/__init__.py:60-88` |
| W-8 | `modes/definitions.py::EMBODIED_RUN_MODES`; `selfy.py:930`; `cli.py::_RUNTIME_SWITCHABLE_MODES` | `EMBODIED_RUN_MODES` has zero non-test callers. It is capability, not a fix, so say in the PR body that PR 1 consumes it. The selfy no-park list is a third hand-written mode list (`RUN_MODES - {"sleep"}` plus `OPERATIONAL_MODES`) that can drift. | nit | grep |

## Fixes

1. **W-1:** make the three fakes `lambda url, timeout=None, **kw: ...`. In the routing test, also assert `kw == {"address_policy": "private", "proxy": "never"}`, so the test pins the new contract instead of merely surviving it. In the swallow test, assert the `RuntimeError` path was actually taken (for example, the fake records that it was called).
2. **W-2:** `fetch_url(f"http://{host}:8000/docs", timeout=3, address_policy="private", proxy="never")`, with a test beside `test_1806_pr0_daemon_reads_direct.py` (which already pins the other three sites).
3. **W-3:** one changelog line ("the connect pre-check resolves `host:` to IPv4; an IPv6-only host is refused"), or accept a v6 literal in `resolve_ipv4`.
4. **W-6 / W-7 / W-8:** fix the doc citation; add one changelog line for the library API; note in the PR body that PR 1 is the `EMBODIED_RUN_MODES` caller.
5. **W-4 / W-5:** no PR 0 change. Name them in PR 1's scope (W-4) and as a scripts follow-up (W-5), with `Deferred because:` per the filing rule.
