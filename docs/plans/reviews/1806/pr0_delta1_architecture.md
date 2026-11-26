# #1806 PR 0 delta round 1: architecture lens

Reviewer: architecture lens, PR 0 delta round 1, 2026-11-25

**Verdict: SHIP after fixes** (two should-fix findings, D1 and D2. Every A-finding that was meant to fold in PR 0 landed as intended.)

Scope: `git diff 6b4ad2bb cb7c7b4b` (folds c9931a8d and cb7c7b4b) plus the merge resolutions in 3457f3ef (`cli.py`, `doctor/checks.py`, `function_length_baseline.json`). Checked against the prior round (`pr0_architecture.md`, A1–A13), `ARCHITECTURE.md`, the persistence-config and runtime-tools briefs, and the approach note §8.

Mechanical checks ran in a scratch two-commit git copy (base = 77880606, the main tip that was merged; head = 3457f3ef), with a scratch HOME and PYTHONPATH set to the copy's `src`:

- `--audit-architecture`: no new, stale or unreviewed findings (33 accepted-debt findings, unchanged).
- These lints are clean: `lint_no_silent_swallows`, `lint_function_length` (touch rule included; `execute_and_learn` 311 → 304, `_main_impl` pin 1098), `lint_atomic_io_ratchet`, `lint_orphan_modules`, `lint_http_surface` and `lint_claude_md_invariants`.
- `ruff check` and `ruff format --check` are clean on the 50 changed .py files.
- 691 tests pass: every `test_1806_pr0_*` file, plus `test_doctor`, `test_config_loader`, `test_mesh_config`, `test_peer_config`, `test_role_detection`, `test_role_unification` and `test_tunnel_keys`.

## Fold verification

| Prior | Folded? | Evidence |
|---|---|---|
| A1 one config-dir resolver | yes | `utils/paths.py::maxim_config_dir` is the right layer: the persistence brief names `utils/paths.py` as the canonical path module, and it imports only `maxim.exceptions` and the stdlib, so there is no cycle. All six resolvers delegate (`peer_config_path`, `config_path`, `role._mesh_yml_path`, `profiles_config_path`, `key_file_path`, `mesh_config_path`), and the downstream consumers (hivemind keys and registry, `peer/cli`, `config_loader`'s api_key) go through them. The `tools → peer` edge is gone. The agreement test is real: it covers every branch (XDG unset, empty and set; APPDATA unset, empty and set; Windows ignoring XDG) and also pins the resolver's own semantics. Its limit is D5. |
| A2 internet files | yes, plus `hive_release_sequence.json` | The survey of `util/` was partial (D2). |
| A3 utils → tools edge | yes | Docstring added to `_contained_script_path`. |
| A4 private cache read | yes | `paths.data_home_if_resolved()` is used, and it has a no-side-effect test. |
| A6 CC3 roster | yes | `ProbeOutcome` is on the path-(a) roster with "never persisted today". That defers the JSON-serializability check to the first persisted use, which is acceptable. |
| A7 `DaemonStatusUnreadable` | yes, no behaviour change | `read_daemon_status` has one `src/`/`scripts/` caller, `probe_daemon`, which catches the exception at once. No `except MaximConnectionError` site can see it. `MaximError(message)` takes the same constructor call. |
| A11 stub | yes | "five" sites, the full residual list, and the Regression guard line is intact (the lint passes). |
| A13 cache invalidation | yes | The helper is extracted, so the edit shrank the pinned function. The touch-rule lint is clean. |
| A5, A9a, A10, A12 | moved to PR 1 | They are on the approach note's §8 PR 1 checklist. |
| A8 residual | not folded | The probe docstring still does not record the `setdefaulttimeout` global mutation as a residual (nit, unchanged). |
| A9b IPv6 | superseded by E1 | See D1 and D4. |
| Merge 3457f3ef | correct | `_run_stopped_on` re-raises `SECURITY_REFUSALS` before the run-config branch. Doctor checks the offline guard before any probe. |

## Findings

| # | file::symbol | finding | severity | evidence / fix |
|---|---|---|---|---|
| D1 | `doctor/checks.py::_check_one_robot` vs `hardware/reachy/controller.py::ReachyMiniController._precheck_daemon` (E1 fold) | **The three probe consumers read `failure="unsupported"` three different ways.** (1) For an entry with no `host:` whose `<name>.local` has only an IPv6 address, doctor returns `info`: "the robot run still connects to it as given". (2) The controller, on the `mdns` branch, treats `unsupported` like `resolve` and **skips the SDK connection**. (3) `refusal()` returns "not probed: …", which is a refusal. So doctor reports a robot that the run will not connect to as non-failing. A verdict policy that `refusal()` was meant to centralize is now decided separately at each site. | should-fix | The doctor `unsupported` branch has no `host` check, while the controller has `if outcome.failure == "unsupported" and not mdns`. **Fix:** decide it once. Either give the doctor row the controller's condition (an `info` row only when `host` is set, otherwise the `resolve` fail row), or add one probe helper (for example `connects_unprobed(outcome, explicit_host)`) that both call. Add a doctor test row for an AAAA-only `.local` name with no `host:`. |
| D2 | `tools/base.py::_CONTROL_PLANE_DATA_FILES` | **The `util/` survey stopped short.** `util/cost_state.json` is the same class as the newly added `hive_release_sequence.json`: a persisted counter whose reset defeats an operator limit. `LLMRouter._budget_status` blocks cloud calls on the cost tracker's totals, and those totals are reloaded from this file at startup, so an agent with file-tool reach into `~/.maxim` can zero its spend history and reset the budget cap across a restart. The in-process tracker overwrites the file on its next save (about every 10 s or every 5 calls), which narrows the window but does not close it. I checked `learned_bounds.json` / `workspace_bounds.json` and they are not this class: `goto_target`'s hard clamp does not read them. | should-fix (owner decision) | `models/language/cost_tracker.py` (`resolve_user_state("util/cost_state.json")`), `router.py::_budget_status`. **Strict default (recommended):** add `("util", "cost_state.json")`, two test rows, and the doc lists (see D3). **Alternative:** file it with `Deferred because: owner-decision`, labelled `security`. |
| D3 | `tools/base.py::control_plane_write_refusal` docstring, `docs/user/safety.md`, test-module docstring | **The lists drifted after the hive file was added.** The refusal docstring lists five files without `hive_release_sequence.json` and says "one of those five data-home files", but there are six. `safety.md` says "one of those five data-home files" and its safety-stack item 4 leaves out the hive counter. The test module's docstring leaves it out. The changelog has "… and `util/internet_policy.json` and `util/hive_release_sequence.json`" (a doubled "and"). `modes-guide.md` and the runtime-tools stub are correct. The code is unaffected, because the name-rule suffixes are derived from the tuple. | nit | Say "each of the data-home files" without a count, add the hive file to the three places, and fix the changelog grammar. |
| D4 | `hardware/reachy/probe.py::_has_ipv6_address` | **A second, unbounded lookup runs on every IPv4 resolve miss.** `getaddrinfo(AF_INET6)` takes no timeout, and `setdefaulttimeout` does not bound name lookups either. On the commonest failure (robot off, `.local` miss), the connect pre-check and doctor now pay two mDNS timeouts. On the controller's `mdns` branch the result is thrown away, because `unsupported` is treated as `resolve` there. The docstring's "`deadline_s` bounds EACH network stage" overstates this for the resolve stage. | nit | Probe for AAAA only when the caller will act on it (an explicit host; this folds into D1's single policy), or state the residual in the docstring. |
| D5 | `tests/unit/test_1806_pr0_config_dir_agreement.py` | **The agreement test cannot catch A1's own example of drift.** It proves the six resolvers agree for the env inputs it sets. A copy that later adds a new override (A1's example: `MAXIM_CONFIG_DIR` in one copy) would pass, because the test never sets that override. | nit | Add a source scan: no `XDG_CONFIG_HOME` or `APPDATA` read in `src/maxim/` outside `utils/paths.py::maxim_config_dir` and the git-config site in `tools/base.py`. Today there is exactly that one other site, so the scan is a two-entry allow-list. |

## Stop-rule outcome

**STOP RULE NOT MET.** D1 is fail-open: doctor gives a non-failing row for a robot the run will skip. D2 is fail-open: it is another site of the D14 pattern and needs an owner decision. Neither is fail-closed. The D1 and D2 folds are `src/` changes, so they need a delta-scoped round 2. D3–D5 are nits and can fold in the same batch.
