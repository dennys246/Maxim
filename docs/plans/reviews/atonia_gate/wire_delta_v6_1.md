# Atonia gate approach notes v6.1: Wire-only delta round

**Lens:** wire integrity (docs/CODE_REVIEW.md, "The wire-integrity lens", "Delta rounds"). **Scope:** only the folds
of v6-delta findings #1 (§1A.1 "Step 0a", the profile accessors) and #6 (§1A.2 "The booking table" / "What counts as
refused"), as staged on `plan/atonia-approach-notes` over `7aa09146`. Read-only census of `src/`, `scripts/`, `tests/`
in this worktree. Stop rule applied.

## Fold #1: profile load moves from import to `_ensure_user_profiles()`

Today `models/language/config.py::_apply_user_profiles_at_import` merges `profiles.yml` into three module dicts when
`config.py` is imported. `profile_loader.apply_user_profiles` stages everything before it mutates (verified), so a
retry after a raise is safe. Fresh-process check: `import maxim.models.download` and `from maxim.models import
LLM_MODELS` do NOT import `config.py` today.

### Producer → consumer table

| Consumer (outside the plan) | Reads | Before (import-time merge) | After, as folded | Covered by the fold? |
|---|---|---|---|---|
| `cli.py::_handle_list_models` | `_BUILTIN_PROFILES`, `_PROFILE_ALIASES` | sees user profiles | migrates to `iter_profiles()` | yes (named) |
| `cli_utils.py` L124, L178, L477 | `_BUILTIN_PROFILES` (+ `normalize_llm_profile`) | sees | migrates | yes (named) |
| `api.py` L379, `list_models` L455 | `_BUILTIN_PROFILES` | sees | migrates | yes (named) |
| `doctor/checks.py` L663, L1582, L3011 | `_BUILTIN_PROFILES` | sees | migrates | yes (named) |
| `normalize_llm_profile`, `list_llm_profiles`, `build_model_path`, `load_llm_config` | in-module | sees | call ensure | yes (named) |
| `runtime/lane_backends.py`: L937 (`_has_cloud_lane`), L1252, L1617, L1741, L1790, L1834 (`_preflight`), L1966, L2604, L2834, L2891, L2942 | `_BUILTIN_PROFILES`, `_normalize_profile`, `LLM_MODELS` | sees | **not named**; the CI grep catches the `_BUILTIN_PROFILES` imports | grep only |
| `runtime/lane_models.py` L379, L463 | `_BUILTIN_PROFILES` | sees | not named; grep catches | grep only |
| `runtime/leader_proxy.py` L1237 (VRAM projection) | `_BUILTIN_PROFILES` | sees | not named; grep catches | grep only |
| `agents/llm_agent.py` L164, L172 | `_BUILTIN_PROFILES`, `_normalize_profile` (after `load_llm_config()` at L158) | sees | not named; grep catches | grep only |
| `models/download.py::delete_llm` L738 | `_BUILTIN_PROFILES`, `_normalize_profile`, `list_llm_profiles` | sees | not named; grep catches | grep only |
| `models/download.py`: `ensure_available` L566, `download_llm` L680, `delete_llm` L719, `list_models`/`check_models`/`print_status` (L950–1020) | `LLM_MODELS` | sees iff `config.py` was imported | sees iff an accessor ran first | **no** (grep does not cover `LLM_MODELS`) |
| `lane_backends.py::_preflight…` L1858 | `LLM_MODELS` (after a `_BUILTIN_PROFILES.get`) | sees | sees only if the L1856 read migrates to an ensuring accessor | ordering only |
| `models/__init__.py` re-export `LLM_MODELS` | public | sees iff config imported | sees iff an accessor ran | no |
| `scripts/benchmark_cross_session.py` L422 | `_BUILTIN_PROFILES` after `normalize_llm_profile` | sees | sees (ordering); its docstring's "merged at import" claim goes stale | grep, if scoped to `scripts/` |
| `profile_loader.py`, `model_cli.py` | write side / read `profiles.yml` directly | n/a | unchanged | n/a |
| Tests: `test_profile_metadata`, `test_estimate_max_ctx`, `test_lane_backends`, `test_doctor`, `test_cloud_backends`, `test_phase1_fixes`, `test_bootstrap`; `patch.dict` in `test_lane_profile_rewalk` L67, `test_doctor_p8_checks` L110; `monkeypatch.setitem` in `test_exp37_harness_smoke` L1384 | builtin metadata | sees builtins (conftest pins `XDG_CONFIG_HOME`, so no `profiles.yml`) | unchanged | must be excluded from the grep |

Every current `src/` caller of the `LLM_MODELS` readers happens to run an accessor first: `api.download_model` /
`delete_model` and `cli._handle_delete_model` call `normalize_llm_profile`; `ensure_available` runs from
`cli._ensure_agentic_model` and `llama_backend` after `load_llm_config`, and from `lane_backends._preflight` after a
profile read. `python -m maxim.models.download --llm <user profile>` already misses user profiles today (config is
never imported), so that is pre-existing.

### Findings

**W1. SHOULD-FIX, fail-open, incomplete fold: the private leaf `_normalize_profile` is outside both the accessor set
and the grep.** It is the function `normalize_llm_profile` wraps, and it is imported directly by
`lane_backends.py` L937, `agents/llm_agent.py` L164/L172 and `models/download.py` L739. If the ensure call sits in
the public wrapper (as listed), a migrated reader that keeps `_normalize_profile(x)` and then calls `get_profile(key)`
resolves aliases against the UNMERGED alias table: `get_profile` ensures too late. Concretely, `delete_llm` decides
`known_profile` on the un-normalized alias and refuses to delete a user model by its alias; `_has_cloud_lane` misses a
user cloud alias and reports cost tracking inert. Fold: put `_ensure_user_profiles()` in `_normalize_profile` itself
(the leaf every path goes through), and add `_normalize_profile` to the CI grep.

**W2. SHOULD-FIX, fail-open (latent): `LLM_MODELS` readers have no ensure and no grep.** `download.py`'s
`ensure_available`, `download_llm`, `delete_llm`, `list_models`, `check_models`, `print_status`, the
`lane_backends` L1858 read and the `maxim.models.LLM_MODELS` re-export see user profiles only when an accessor ran
first. That holds for every caller today (traced above) by ordering, not by structure; the fold says "the ten direct
readers of ... `LLM_MODELS` migrate", but none of the ten is an `LLM_MODELS` reader. Fold: `download.py` gets a
module accessor (`_llm_models()` that calls `config._ensure_user_profiles()` via a function-local import, the same
direction-safe pattern config already uses) used by every reader, and the CI grep forbids `LLM_MODELS` reads outside
`download.py` (and `config.py`'s merge).

**W3. SHOULD-FIX, fail-open: the guard's failure semantics are unstated.** Several first-access sites sit inside a
broad `except Exception` (`lane_backends._has_cloud_lane` returns `True`, the `_preflight` block, doctor checks,
`benchmark_cross_session` returns `None`). If the once-per-process flag is set before (or regardless of) a successful
merge, a malformed `profiles.yml` raises once into a swallow and every later access silently runs without user
profiles. Today the import raise recurs on every import attempt. Fold: set the flag only after `apply_user_profiles`
returns; a failure re-raises on every access (or caches and re-raises the `ConfigurationError`). Extend the red gate:
"a malformed `profiles.yml` raises on the first AND a second access".

**W4. SHOULD-FIX, fail-closed, incomplete fold: "ten readers" undercounts.** The census above finds 9 more `src/`
modules' worth of sites (`lane_backends` ×11, `lane_models` ×2, `leader_proxy`, `llm_agent` ×2, `download.py`) plus
one script. The CI grep will force them, so nothing ships silently, but the build will find out late. Fold: name them,
and state the grep's scope: `src/` + `scripts/`, excluding `tests/` (15 test files read builtin metadata, and three
patch the dicts).

**W5. NIT: `get_profile(name)` must state whether it normalizes.** Readers today mix `.get(raw)` (doctor, lane_models,
leader_proxy) and `.get(normalize(raw))` (api, cli_utils, lane_backends L1617). Implicit normalization would change
the doctor and lane_models lookups. State "no normalization; callers normalize".

**W6. NIT: `patch.dict` on `_BUILTIN_PROFILES`.** If the first ensure happens inside a `patch.dict` block, the merge
is rolled back on exit while the flag stays set. Harmless today because conftest isolates `XDG_CONFIG_HOME`; the red
gate already uses a fresh process. One sentence in the plan is enough.

## Fold #6: which refusals count as `refused`, and the #1185 precedent

### Producer → consumer table

| Producer | What it emits | Consumer | Today | After, as folded | Closed? |
|---|---|---|---|---|---|
| `executor.execute` `_permission_denial` returns (L356, L383) | `ToolOutput(success=False)`, no kind | `execute_and_learn` → `rec_outcome` (L1268) → NAc, goal, cluster | booked negative | `POLICY_PERMISSIONS` → `refused` → skipped | yes |
| `_run_started` unregistered (L475) and inactive-scene (L445) | `ToolOutput`, no kind, **plus `_report_failure` → `pain_detector.record_tool_error`** | pain | **booked** (`EXTERNAL_FAILURE` fallback) | table says "not booked"; nothing removes the call | **no** (W8) |
| `_run_started` `except` (tool raised) → `_fail_ran` | `_book_failure` (ToolPainBridge NAc negative), `_report_failure` (pain), `nac_failure_booked` stamp | bridge NAc, pain | booked | `OfflineRefusal` "mapped to `OFFLINE`" in the same except; the bridge and pain have already booked | **no** (W9) |
| Tool bodies (`git_tools`, `code_tools`, `filesystem`) | `PERMISSION_DENIED` | `record_outcome`, bridge | booked negative | unchanged (outside the set) | yes: the executor never emits `PERMISSION_DENIED` today, so deriving from `result.error_kind ∈ GATE_REFUSAL_KINDS` does not conflate |
| `execute_and_learn` L1299 `_record_plan_outcome` → `bio_integration.record_plan_outcome` → `MemoryHub.record_plan_outcome` → `planning_bridge.record_plan_outcome` → `nac.observe(NEGATIVE)` | plan link | NAc | booked negative | **not touched**: the call is outside `record_outcome` | **no** (W7) |
| `execute_parallel_actions` L686 | per-action dicts (no `error_kind` key) | `record_outcome` L742 | booked | "threaded"; the dict needs a key | partly (W11) |
| `execute_parallel_actions` L672 autonomy `can_execute_action` rejection | dict, no `ToolOutput` | `record_outcome` with `nac` | booked negative | unchanged | pre-existing, out of scope (W11) |
| `book_machine_refusal` (L893; caller `loop_planning` L201) | `rec_outcome(nac=None, cluster None, goal None, bridge_booked=None, result_summary=None)` | recent, carryover, context_pool | those three only | must now pass `refused=` | fail-closed (W10) |
| `book_refusal` L874, `book_hard_rejection` L989, `execute_and_learn` except L1409, `agent_loop.py` L1962 agent-fallback except | `rec_outcome(...)` | | | must pass `refused=` (not named) | fail-closed (W10) |
| 5 scripts + ~39 test call sites | `record_outcome(...)` | | | scripts named; tests not | fail-closed (W10) |

**Carryover match.** With `refused` set, `record_outcome` still appends to `recent_outcomes`, calls
`llm_worker.record_outcome(success=False, result_summary=…[:200])` and `context_pool.add_outcome`, and skips the whole
`if nac is not None:` block (observe, `sim_nac`, goal credit, cluster and operant credit). That is the same set of
sinks `book_machine_refusal` reaches. The text differs (`result_str` carries the denial; `book_machine_refusal` passes
`None`), which is richer, not a divergence. Matches in sinks: yes.

### Findings

**W7. SHOULD-FIX, fail-open, incomplete fold: `PlanHistoryBridge` is not reached by `refused`.** The table puts it
in the `record_outcome` row, but the plan-outcome booking is a separate call in `execute_and_learn` (L1299). It ends
in `planning_bridge.record_plan_outcome` → `nac.observe(…, NEGATIVE)` under the tool's signature. A gate refusal
still books a negative NAc link on that path, which is exactly the aversion the fold removes elsewhere. The
`book_machine_refusal` precedent never reaches it (`loop_planning` calls only the booker). Fold: `execute_and_learn`
skips `_record_plan_outcome` when `refused is not None`, and the red gate asserts no plan link for a refused tool.

**W8. SHOULD-FIX, fail-open: the Pain row's "Today: not booked (returns before)" is false for two gate refusals.**
Only the `_permission_denial` returns precede pain. `_run_started`'s unregistered branch (L475) and inactive-scene
branch (L445) call `_report_failure` → `pain_detector.record_tool_error` before returning. Once those refusals carry
`NOT_REGISTERED`, `pain.py`'s mapping (`.get(error_kind, PainType.TOOL_FAILURE)`) books `TOOL_FAILURE` pain. A build
that follows the table changes nothing there. Fold: name both `_report_failure` calls as removed (or explicitly kept,
with the table corrected), and give the inactive-scene refusal a kind (`NOT_REGISTERED` or `MODE`). Otherwise it
carries `None` and keeps booking NAc negative through `record_outcome`.

**W9. SHOULD-FIX, fail-open: `OfflineRefusal` inside a tool books the bridge and pain before `OFFLINE` exists.** In
`_run_started`, the `except` goes to `_fail_ran`, which calls `_book_failure` (a ToolPainBridge NAc negative by
event id, unless `suppress_nac`), then `_report_failure` (pain), and stamps `nac_failure_booked`. `refused=OFFLINE` in
`record_outcome` only skips `record_outcome`'s own NAc, which `bridge_booked` already skipped. So "an offline session
does not learn aversion to actuation or to lazy-download tools" fails through the bridge. Fold: branch
`isinstance(e, OfflineRefusal)` BEFORE `_fail_ran`: no `_book_failure`, no `_report_failure`, return
`ToolOutput(success=False, error_kind=OFFLINE)` with the invocation retired. Also state that a tool which catches
`OfflineRefusal` itself and returns a plain failure is a census violation (it would book with `refused=None`). The
swallow-census "re-raises or records degraded" wording allows that today.

**W10. NIT, fail-closed: the required-keyword call-site list is incomplete.** Besides `execute_and_learn` and
`execute_parallel_actions`: `book_refusal` (L874, pass `None`: a human "no" still books NAc), `book_machine_refusal`
(L915), `book_hard_rejection` (L989), `execute_and_learn`'s except site (L1409), `agent_loop.py` L1962, and about 39
test call sites. They all go through `Callable[..., Any]` / `functools.partial`, so mypy cannot see a miss. It is a
runtime `TypeError`, which the suite catches. Name them.

**W11. NIT: the parallel path's carrier.** `execute_parallel_actions` builds dicts, so `refused` needs a dict key
(`"_refused": result.error_kind if … in GATE_REFUSAL_KINDS`), set on the executed branch only. Separately, the
batch's autonomy `can_execute_action` rejection (L672) is a machine refusal that books NAc negative today. That is
inconsistent with #1185 and outside `GATE_REFUSAL_KINDS`: pre-existing and out of scope, worth one owner line.

**W12. NIT: pin who may produce a gate kind.** Nothing stops a tool body returning `MODE` or `NOT_REGISTERED`, which
would exempt it from NAc. A grep/AST test that gate kinds are constructed only in `runtime/executor.py` (plus the
`OFFLINE` mapping) makes "never a tool body" structural.

## Verdict

No BLOCKER. Fail-open SHOULD-FIX remain: **W1, W2, W3** (fold #1) and **W7, W8, W9** (fold #6). Fail-closed:
W4 (SHOULD-FIX), W10. NITs: W5, W6, W11, W12. All six fail-open findings are incomplete folds with a mechanical
remedy (one ensure in the leaf, one `LLM_MODELS` accessor, flag-on-success, one skipped plan-outcome call, the two
`_report_failure` calls named, an `OfflineRefusal` branch before `_fail_ran`).

**Another round is owed:** a Wire-only delta round on the folds of W1–W3 and W7–W9, with the same stop rule.
