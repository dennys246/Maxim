# #1806 PR 0 delta round 1: wire integrity

Reviewer: wire-integrity lens, PR 0 delta round 1, 2026-11-25

**Verdict: SHIP after one fix** (D1-W1: doctor and the controller now disagree about the new `"unsupported"` outcome. For a no-`host:` entry, doctor says "the robot run still connects" to an address the run actually skips. Everything else is nits.)

Delta: `git diff 6b4ad2bb cb7c7b4b` plus the merge resolutions in `3457f3ef` (`cli.py::_run_stopped_on`, `doctor/checks.py::_check_one_robot`).

Method:
- `git archive 3457f3ef` into a scratch tree, with a scratch `HOME` and `PYTHONPATH=<copy>/src`.
- Greps over `src/ scripts/ tests/ docs/`.
- Ran 30 neighbour files: all `test_1806_pr0_*`, plus audio_localization, doctor, peer_config, tunnel_keys, tunnel, leader_fail_closed, leader_key_stat_1311, ensure_key_1402, config_loader, role_detection, role_unification, mesh_config, hivemind_signing, internet_policy ×3, internet_toggle_1345, clear_memory_1337, tool_dispatch, mdns_discovery, reachy_connection_options, media_release_1704, media_stream_gating, cli_exit_code_1468, console_launcher_seams, pkg_aarch64_guard, git_dir_writes_957 and architecture_audit. Result: **1275 passed, 1 skipped**.
- Lints: `lint_http_surface`, `lint_no_silent_swallows`, `lint_orphan_modules` and `lint_function_length` are all clean.
- Two probe scripts, scratch only: `p1.py` (doctor vs controller on IPv6 outcomes) and `p2.py` (a real `ResultCache`).

## Producer / consumer maps

### (a) The six config-path resolvers now delegate to `utils/paths.py::maxim_config_dir`

| Site | Kind | After the delta |
|---|---|---|
| `peer_config_path`, `config_loader.config_path`, `role._mesh_yml_path`, `profiles_config_path`, `tunnel.keys.key_file_path`, `mesh_config_path` | producers | all `maxim_config_dir() / <name>`; agreement pinned by `test_1806_pr0_config_dir_agreement.py` |
| `tools/base.py::_control_plane_roots` | consumer (deny root) | `paths.maxim_config_dir()`, so it is the same resolver |
| Module-level `platform` patchers: `test_leader_fail_closed.py::test_key_file_path_on_windows_uses_appdata` (`keys.platform.system`), `test_tunnel.py` (`patch("platform.system")`) | tests | both patch the **shared** `platform` module object, so `maxim_config_dir` sees the patch too. Passing |
| Tests that patch `maxim.runtime.config_loader.platform` / `profile_loader.platform` / `role.platform` | tests | none exist. Those modules no longer import `platform`, so a future one would raise `AttributeError` loudly |
| `monkeypatch.setattr(peer_config, "peer_config_path", …)` (`test_peer_config`, `test_doctor` ×2, `test_leader_proxy` via `peer_cli`) | tests | none of them relies on `mesh_config_path` following `peer_config_path` (the coupling that was removed). Passing |
| `key_file_path` patchers (`test_doctor` ×4, `test_tunnel_keys`, `test_ensure_key_1402` ×2, `test_leader_key_stat_1311`) | tests | patch the function itself; unaffected |
| `config_path` patchers (`test_console_launcher_seams`, `test_console_server`, `test_pkg_aarch64_guard`, `test_config_downgrade_974`) | tests | patch the function or its importer's binding; unaffected |
| `role._peer_yml_exists`, `doctor/checks.py:1109`, `peer/mesh_setup.py:184`, `peer/cli.py` | src callers of `peer_config_path` | unchanged values in production |
| `tests/conftest.py` sets `XDG_CONFIG_HOME` at import | isolation | still honoured by the one resolver |

### (b) Deny list extended to `util/internet_access.json`, `util/internet_policy.json` and `util/hive_release_sequence.json`

| Writer / reader | Path source | Through a file tool? |
|---|---|---|
| `utils/internet_access.py` (toggle at L101, policy at L512) | `resolve_user_state("util/…")` = `data_home()/util/…` | no: its own persistence |
| `hivemind/signing.py` sequence path (L367-370) | `data_home()/"util"/"hive_release_sequence.json"` | no |
| `tools/internet_search.py` remedy text | names the files only | n/a |
| Tests writing these files (`test_internet_policy_*`, `test_internet_toggle_session_1345`, `test_clear_memory_1337`, `test_hivemind_signing`) | direct writes or the module API | none goes through `write_file`/`edit_file`/sandbox tools. Passing |

All three files sit in `data_home()/util/`, which is the root the deny unions, so the deny covers them.

### (c) `DaemonStatusUnreadable` is now a `MaximConnectionError`

| Site | After the delta |
|---|---|
| Raiser: `probe.read_daemon_status` | its only src caller is `probe_daemon`, which catches it and records `failure="status"` |
| `except MaximConnectionError` in src (`api.py:318/358/1103`) | wrap SDK connect paths and never call `read_daemon_status`, so they cannot catch it unexpectedly |
| Construction | `MaximError.__init__(message)` accepts the positional message; `__str__` returns it unchanged, so `str(e)` in `ProbeOutcome.error` is unchanged |
| Tests raising it as `side_effect` (connection_options ×5, media_release_1704, mdns_discovery, media_stream_gating, probe_daemon) | all go through `probe_daemon`'s catch. Passing |

### (d) `probe_daemon` deadline per stage, and the new `failure="unsupported"`

| Consumer | `"unsupported"` handling | Deadline passed |
|---|---|---|
| `controller._precheck_daemon` | explicit `host:` → connect as given. mDNS (no host) → skip, logged "could not resolve via mDNS" | `min(timeout, 5)` per stage (was a `+3 s` total) |
| `doctor._check_one_robot` | **always** `info` "…the robot run still connects to it as given" (D1-W1) | 2.0 per stage |
| `probe.refusal` | `"not probed: …"`; doctor only reaches it after a status read, so this branch is currently unreached | n/a |
| `utils/reachy_diagnostics.py` | does not call `probe_daemon` | n/a |
| `ProbeOutcome.reachable` / `status_read` | `"unsupported"` gives False for both, which is correct | n/a |
| Tests | `test_1806_pr0_probe_daemon.py::test_an_ipv6_host_is_unsupported_not_unresolvable` pins the probe only; no test pins doctor's or the controller's handling | n/a |

Probe `p1.py` (stubbed `gethostbyname` fails, AAAA present):

| Entry | doctor | controller |
|---|---|---|
| no `host:`, `reachy-mini.local` AAAA-only | `info` "…still connects to it as given" | `(None, None)`: **skips the connection** |
| `host: fe80::1` | `info` | `('fe80::1', None)`: connects |
| `host: v6only.lan` (AAAA-only) | `info` | `('v6only.lan', None)`: connects |

### (e) `_run_stopped_on` event `robot_config_invalid` → `run_config_invalid`

| Site | After the delta |
|---|---|
| `cli.py:1020` (merge resolution keeps both main's `SECURITY_REFUSALS` re-raise and PR 0's branch, in the right order) | producer |
| `tests/unit/test_cli_exit_code_1468.py:97,111` | asserts the new name |
| Any log consumer, script or doc using `robot_config_invalid` | none (grep is empty outside `pr0_executor.md`, a review record) |
| `type(error) is InvalidConfigError`: bare raisers in src | `controller.py:287` and `selfy.py:103/105` only, both robot run config. The subclasses (`InvalidAgentIdError`, `AgentConfigContradiction`) keep exit 1 |

### (f) `tool_dispatch._invalidate_cache_after_write`

| Site | After the delta |
|---|---|
| Caller | `execute_and_learn` only (L1576); a verbatim move plus `edit_file` |
| `ResultCache.invalidate(path=)` | matches `path` as a **substring of the cached result text**, not the key's params (D1-W4) |
| Cache producer | `prefetch._execute_cached` stores `read_file` etc. `get()` drops an entry whose file mtime changed |
| Parallel write-tool list | `fear_gate.file_write_tools` has 4 tools; the invalidation set has 2 |

### Prior-round findings

| # | Status |
|---|---|
| W-1 | **closed**: the three fakes take `**kw`, the routing test pins `{"address_policy": "private", "proxy": "never"}`, and the swallow test asserts the `RuntimeError` path ran; passing |
| W-2 | **closed**: `reachy_diagnostics.py` `/docs` passes `address_policy="private", proxy="never"` |
| W-6 | closed (`motion_path_model.md` updated) |
| W-7 | closed (changelog line for `load_robots_config` / `from_yaml`) |
| W-3 | superseded by the `"unsupported"` outcome, which gives rise to D1-W1 |

## Findings

| # | file::symbol | finding | severity | class | evidence |
|---|---|---|---|---|---|
| D1-W1 | `doctor/checks.py::_check_one_robot` (`"unsupported"` branch); `changelog.d/1806-pr0-robot-config-groundwork.fixed.md` | Doctor grades every `"unsupported"` outcome `info` with "the robot run still connects to it as given". The controller does that only for an explicit `host:`. For a no-host entry whose `<name>.local` has only an AAAA record, the run **skips** the SDK connection, yet doctor shows a no-fix, no-`retry_id` info row. Before the delta that case was `fail` "does not resolve" with the Local Network / `set host:` fix. The changelog repeats the false claim. Fix (≈3 lines): `if outcome.failure == "unsupported" and host:` keeps the info row; otherwise fall through to the `resolve` fail branch, noting "only an IPv6 address". Add one test that pins doctor and `_precheck_daemon` agreeing for host and no-host. | important | diagnostic over-reports health; **new class** (consumers disagree on a new enum value), not fail-closed at the doctor surface | probe `p1.py` table above |
| D1-W2 | `hardware/reachy/probe.py::_has_ipv6_address` | A second, uninterruptible `getaddrinfo(AF_INET6)` with no timeout runs after **every** failed IPv4 lookup. On a dead mDNS name this can double the worst-case resolve time for doctor and the connect pre-check, and the per-stage docstring does not mention it. | nit | fail-closed (it only delays a refusal) | `probe.py:141-146` |
| D1-W3 | `tools/base.py::control_plane_write_refusal` docstring | The bullet omits `util/hive_release_sequence.json`, and the name-rule sentence says "one of those five data-home files". `_CONTROL_PLANE_DATA_FILES` has six, so the docstring disagrees with the code it documents. | nit | doc only | `tools/base.py` docstring vs tuple |
| D1-W4 | `runtime/tool_dispatch.py::_invalidate_cache_after_write` → `fetch_cache.ResultCache.invalidate` | `invalidate(path=)` matches the path inside the cached **result** text. A `read_file` entry whose output is the file's content is not dropped (probe `p2.py`: the entry remains), so neither the `write_file` invalidation nor the new `edit_file` one removes it. Freshness actually comes from `get()`'s mtime check. The changelog's "a read after a successful edit could return the pre-edit content" overstates the fix. The guard test uses a fake cache, so it pins the call, not the effect. The set (2 tools) is also a second hand-written write list beside `fear_gate.file_write_tools` (4 tools). | nit | pre_existing (mtime guards the read); a pre-existing class | probe `p2.py` |

## Stop-rule outcome

**STOP RULE NOT MET.** D1-W1 is not fail-closed at the doctor surface: doctor reports a connect that the run does not make, and it belongs to a new class (two consumers of `ProbeOutcome.failure` grading the new `"unsupported"` value differently). D1-W2..W4 are fail-closed or doc-only and would not on their own require another round. After the D1-W1 fold (one `src/` hunk plus its test), a delta-scoped round covering only that hunk is owed.
