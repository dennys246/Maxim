# #1806 PR 0 code review: architecture lens

Reviewer: architecture lens, PR 0 code review, 2026-11-25, 6b4ad2bb

**Verdict: SHIP after fixes** (two important findings, A1 and A2; the rest are nits or pre-existing)

Scope reviewed: `git diff origin/main...6b4ad2bb` (52 files, +2948/−723), checked against the approach note §5 and §8 (the PR 0 bullet), `ARCHITECTURE.md`, `utils/audit.py::LAYER_RULES`, and the persistence-config, runtime-tools, embodiment §1 and llm-routing briefs.

Mechanical checks, run in a scratch two-commit git copy (base = merge-base, head = 6b4ad2bb) with a scratch HOME, so the diff-scoped rules ran too:

- `--audit-architecture`: no new, stale or unreviewed findings (33 accepted-debt findings, unchanged).
- Clean: `lint_no_silent_swallows`, `lint_function_length` (touch rule included: `_main_impl` −18, `_build_parser` −34, `Maxim.__init__` −69), `lint_atomic_io_ratchet`, `lint_orphan_modules`, `lint_http_surface` (pins updated exactly), `lint_claude_md_invariants` (8 docs; the briefs are covered).
- `ruff check` and `ruff format --check` clean on the 40 touched .py files. `run_mypy_public.sh` passes (it lists none of the new files).
- 379 tests pass: the PR's own tests, `test_doctor`, `test_multi_robot` and `test_cli_action_gate`.

## Findings

| # | file::symbol | finding | severity | evidence |
|---|---|---|---|---|
| A1 | `tools/base.py::_control_plane_roots` | The deny gate's config-dir root comes from `peer/config.py::peer_config_path`. Its comment says this is "the resolver config_loader and peer_config_path share", which is false. There are six hand-copied XDG/APPDATA resolvers: `peer_config_path`, `runtime/config_loader.py` (config.json), `runtime/role.py::_mesh_yml_path`, `models/language/profile_loader.py` (profiles.yml), `tunnel/keys.py::key_file_path` (api_key, console_token) and `peer/mesh_config.py::mesh_config_path` (the only one that delegates). A security gate now depends on all of them agreeing. One drift, such as a future `MAXIM_CONFIG_DIR` in one copy, would silently open the deny for that file. It also adds a `tools → peer` edge just to get a path. | important | `grep -rn '"maxim" / ' src/maxim`: 6 copies. The design (§5) says "through the same resolver"; the code uses one of six. |
| A2 | `tools/base.py::_CONTROL_PLANE_DATA_FILES` | The denied data-home set is a hand list that leaves out the other operator-only trust files of the same class: `util/internet_access.json` (the operator's internet on/off, "operator-only" #1345) and `util/internet_policy.json` (the operator's frozen network rules, #832). An agent whose file tools reach `~/.maxim` can rewrite them, under the same precondition D14 closes for `robots.yaml`. This is another site of the PR's pattern, and no design round considered these files. | important | `utils/internet_access.py` resolves both through `resolve_user_state("util/internet_access.json")` and `resolve_user_state("util/internet_policy.json")`; the approach note and `reviews/1806/*` never mention them. |
| A3 | `utils/sandbox_executor.py::SandboxExecutor._contained_script_path` | This is the first `utils → tools` import in `src/maxim/utils/` (function-local). `LAYER_RULES` has no `utils` row, so the audit cannot see it. The edge is tolerable because `tools/sandbox.py` is the only caller of `SandboxExecutor`, but it inverts ARCHITECTURE.md's "utils is the bottom" convention. | nit | `git grep "from maxim.tools" origin/main -- src/maxim/utils/` → 0 hits; 6b4ad2bb → 1. |
| A4 | `tools/base.py::_control_plane_roots` | It reads `maxim.utils.paths._data_home_cache`, a private module global of another module. A rename fails loudly (AttributeError on every file write), so this is not silent, but it is private coupling inside a gate. | nit | `paths._data_home_cache is not None` |
| A5 | `modes/definitions.py::EMBODIED_RUN_MODES` | It has no `src/` caller (only `test_1806_pr0_cli_flags.py`). The CLI's real dispatch is still `if mode == "agentic": … else selfy` (`cli.py::_main_impl`), so the "body subset" is a second, unconnected copy of a rule the CLI encodes inline. This is a capability, not a shared constant. | nit | `git grep EMBODIED_RUN_MODES -- src scripts` → the definition only |
| A6 | `hardware/reachy/probe.py::ProbeOutcome` | It uses CC3 path (a) correctly (all fields defaulted, `extra` with `hash=False, compare=False`, collision check), but it is not added to the path-(a) roster in persistence-config.md §3, and `extra` values are not checked as JSON-serializable, which the roster rule requires. | nit | persistence-config.md "Reference: frozen-dataclass path-(a)/(b) registry" |
| A7 | `hardware/reachy/probe.py::DaemonStatusUnreadable` | It is a public exception (`__all__`, raised by the public `read_daemon_status`) that derives from bare `Exception`, outside `maxim.exceptions`' stable hierarchy. Today only `probe_daemon` catches it, so it is contained. | nit | `class DaemonStatusUnreadable(Exception)` |
| A8 | `hardware/reachy/probe.py::resolve_ipv4` | The process-global `socket.setdefaulttimeout` mutation (moved from the controller and doctor) now also runs on the controller's connect for an explicit non-literal `host:`, which was not resolved by Maxim before. It is not thread-safe against other sockets created in that window. | pre_existing | old `controller._resolve_mdns` and `doctor._check_one_robot` both did this |
| A9 | `hardware/reachy/controller.py::ReachyMiniController._precheck_daemon` | (a) For an explicit `host:` name, the probe checks the resolved IPv4 but the SDK dials the name verbatim (it re-resolves). The probe's verdict is therefore about a different dial than the session's. This is the approach note's PR 1 item (carry the verified address), but PR 0's docstring should not imply otherwise, and PR 1 must own it. (b) An IPv6-literal `host:` now fails the pre-check as "does not resolve" (`gethostbyname`), where `create_connection` used to accept it. The changelog does not list this behaviour change. | nit | `return (outcome.address if mdns else target), …` |
| A10 | approach note §5 / `scripts/exp49/run_trials.py::_trial_env` | §5 promises that "the documented harness `child_env` recipe" pops `MAXIM_ROBOT_NAME`/`MAXIM_DEVICE`. No such recipe exists (`docs/agents/simulation-experiments.md` names neither variable; `utils/child_env.py` strips only PYTHONPATH). Deferring `MAXIM_DEVICE` to PR 1 is correct (v5 ordering). | nit | `grep -n "MAXIM_ROBOT_NAME\|child_env" docs/agents/simulation-experiments.md` → none |
| A11 | `docs/agents/runtime-tools.md` (D14 stub) | It says "at the same four sites as `git_dir_write_refusal`". There are five now (`write_script` gained both refusals), and the stub mentions that site separately. Its Residual line leaves out two residuals that `safety.md` states: HFS+ ignorable characters and a new hardlink. The stub ends with a valid Regression guard, and the lint passes. | nit | stub text vs `control_plane_write_refusal` docstring and `safety.md` |
| A12 | `hardware/config.py::resolve_daemon_name` | It labels every truthy argument `"--robot-name"`, including `selfy.Maxim`'s library default `"reachy_mini"`. Only the CLI log consumes the label (correct there), but the function's contract claims to name the source. | nit | `if typed: return typed, "--robot-name"` |
| A13 | `runtime/tool_dispatch.py` (cache invalidation after a write) | The same name-list shape the autonomy fix addressed: the result cache is invalidated only for `write_file`, so a successful `edit_file` leaves stale cached reads. The root cause differs (a stale read, not approval), so this is outside PR 0's root cause. It is a one-line known fix. | pre_existing | `if tool_name == "write_file" and success:` |

## What is right (no action)

- **Placement of the probe.** `hardware/reachy/probe.py` is the lowest common owner of its three consumers. The controller (in `hardware/`) cannot import `doctor/`, so extending doctor was not an option, and the front-gate question is answered. It imports only `utils.http` and stdlib, and LAYER_RULES `hardware` is respected.
- **The strict HTTP endpoint rides existing infra.** `_external_policy_private_strict` reuses `HTTPEndpoint.strict_address_check` and `_checked_client(strict=True)`: no proxy mounts, `trust_env=False`, address checked on the dialled address by `_AddressCheckingBackend`, and registered in `_DEADLINE_ENDPOINTS` (#1715). An unknown `proxy=` value or a `proxy="never"`/non-private combination raises `ValueError`, never a silent fallback. This is consistent with the #824/#921 "check the dialled address" rule, and the llm-routing stub matches the code.
- **Typed errors.** `RobotsConfigError(ConfigurationError)` carries a `fix_hint` and is in `__all__`. Callers catch it narrowly (doctor, `_run_stopped_on` → exit 2). A broad doctor fallback remains, but it reports a row and is not a swallow.
- **One resolver.** `selfy.Maxim._resolve_robot_connection` resolves the entry once. `AgenticRuntimeMixin` reads it with no reload and no second match rule, and the DEBUG swallows are gone. `Maxim` is the only host of the mixin.
- **Stable API: no `stable_api.md` row is needed.** `RobotsConfigError` is a leaf under the stable `ConfigurationError` category (catch-the-category rule). `fetch_url` and `maxim.hardware.load_robots_config` are not on the stable page (rule 6: internal), and the raise-type change is in the changelog fragment. `--robot`/`MAXIM_ROBOT_NAME` are documented in `configuration.md` and `cli-reference.md`.
- **Scope matches §8's PR 0 bullet.** Every item is present: the §5 fixes, loud yaml plus the relative `MAXIM_DATA_HOME` check, one resolver, `probe_daemon` + `refusal`, `--robot`/`None`/gate/re-exec in one commit, `RUN_MODES`, the cwd fallback removed, `proxy="never"`, D14 at the #957 sites, the `edit_file` approval, scrubs, the exp49 pop, the Dormant markers, `_connect_kwargs` and `timeout:` removed. There is no `--device`, `MAXIM_DEVICE` reader, pin or identity check (PR 1). The only PR-1-shaped additions are the stored-but-unused identity fields on `ProbeOutcome`, which are bounded and documented, and `EMBODIED_RUN_MODES` (A5).

## Fixes

**A1 (fold; ≤50 `src/` lines, no decision needed).**
1. Add `utils/paths.py::maxim_config_dir() -> Path` (the APPDATA/XDG/`~/.config` body, verbatim).
2. Route all six resolvers through it as verbatim moves: `peer_config_path`, `config_loader`'s config.json path, `role._mesh_yml_path`, `profile_loader`'s profiles.yml path, `tunnel/keys.key_file_path`, and `mesh_config_path` (already delegating).
3. Have `_control_plane_roots` call `paths.maxim_config_dir()`. That drops the `tools → peer` edge and makes the comment true.
4. Add one test asserting that each of the six file paths has `maxim_config_dir()` as its parent under a patched `XDG_CONFIG_HOME` and under patched Windows APPDATA. That pins agreement structurally instead of by attention.

**A2 (owner decision, asked now).** Does "the whole control-plane set" (D14) include the operator-only internet files?
- Strict default (recommended): add `("util", "internet_access.json")` and `("util", "internet_policy.json")` to `_CONTROL_PLANE_DATA_FILES`. That is 2 lines, plus the name-rule suffixes (derived automatically) and two rows in the existing deny test. Update `safety.md`, `modes-guide.md`, the runtime-tools stub and the changelog fragment to list them.
- Alternative: keep the set as is, and file it with `Deferred because: owner-decision`, labelled `security`.

Either way, survey `<data_home>/util/` once for any other file whose docstring says operator-only, so the list is complete by audit rather than by memory.

**A3.** Accept for PR 0, and state it in the `_contained_script_path` docstring ("the one utils→tools edge; `SandboxExecutor`'s only caller is `tools/sandbox.py`"). Alternatively, move both refusal predicates into a `utils/` module with a `tools/base.py` re-export. That is a verbatim move, and it makes sense if a future non-tool writer needs them.

**A4.** Add `utils/paths.py::data_home_if_resolved() -> Path | None` (returns the cache without the mkdir) and call it.

**A5.** Either derive `EMBODIED_RUN_MODES = frozenset(RUN_MODES) - {"agentic"}` and make the CLI dispatch read `mode in EMBODIED_RUN_MODES`, or drop it from PR 0 and add it with its first caller (PR 2b's `launch.mode` validation).

**A6.** Add `ProbeOutcome` to the path-(a) roster in persistence-config.md §3. Either add a JSON-serializability check to `__post_init__`, or note in the roster that it is runtime-only today and the check arrives with the first persisted use.

**A7.** Either derive `DaemonStatusUnreadable` from `maxim.exceptions.MaximConnectionError`, or drop `read_daemon_status`/`DaemonStatusUnreadable` from `__all__`, since they are test seams.

**A8.** No PR 0 action. Record it as a known residual in the probe docstring. A later fix is to resolve with `getaddrinfo(AF_INET)` in a daemon thread joined with a timeout, so no global mutation is needed.

**A9.** (a) Add one sentence to `_precheck_daemon`'s docstring: "an explicit `host:` is handed to the SDK verbatim and re-resolved there; carrying the probed address is #1806 PR 1." Add it to PR 1's checklist. (b) Add a changelog bullet: "an IPv6-literal `host:` is no longer accepted by the connect pre-check (IPv4 only)". Alternatively, accept an `IPv6Address` literal as is in `resolve_ipv4`, if the owner wants IPv6 kept.

**A10.** Either add one line to `docs/agents/simulation-experiments.md` (a harness child that should not be a named robot run pops `MAXIM_ROBOT_NAME`, and `MAXIM_DEVICE` from PR 1) or strike "documented recipe" from §5. Put the `MAXIM_DEVICE` pop, scrub and brief row on PR 1's checklist.

**A11.** In the stub, change "four sites" to "every `git_dir_write_refusal` site (the four tools plus `SandboxExecutor.write_script`)" and append the HFS+ and new-hardlink residuals to its Residual line.

**A12.** Return `"argument"` (or take a `source=` from the CLI) when the caller is not the CLI. Alternatively, document that the label assumes a CLI caller.

**A13.** Fold if convenient (`tool_name in ("write_file", "edit_file")` at the invalidation site, and one test). Otherwise file it labelled `small-fix`.
