# Atonia gate approach notes, v6 delta round (three lenses, one pass)

Scope: `git diff d6389cd4 7aa09146 -- docs/plans/` only: §1A.1–1A.11, the rewritten §7, the V9/V10 rows, the
patched body sentences, and the advisor_mode.md "Superseded" markers. Read-only. Probes ran from a scratch dir with
`python -I` (a static import-closure walker that treats the four V9 package inits as empty; AST scans of the
package inits; JSON reads of `function_length_baseline.json`).

What checked out (no finding):
- **Step 0a is buildable.** With the four inits treated as lazy, `maxim.runtime.executor`'s static module-level
  closure is 8 `maxim` modules (`maxim`, `runtime`, `runtime.executor`, `tools`, `tools.base`, `tools.registry`,
  `utils`, `utils.logging`), down from about 830. `tools.base` and `agents.permissions` are 3 each. No name exported
  by the four inits collides with a submodule name. No test patches a package-level attribute
  (`patch("maxim.runtime.X")` or `setattr("maxim.tools.X")`): zero hits. No module imported by those inits has a
  module-level call with side effects (registrations, `addLevelName`, `atexit`). No `from maxim.<pkg> import *`.
  `isinstance` identity cannot change, because a lazy `__getattr__` returns the same class object.
- **The `closed_local` closure** (via `models/language/__init__`, which step 0a leaves eager) is stdlib plus about 20
  `maxim` modules. None reads env, HOME or the filesystem at import, once `_apply_user_profiles_at_import` is gone.
- **Rule T table:** all ten rows are pinned, with the counts the baseline holds (`build_executor` 250,
  `create_full_agent` 204, `api.py::run` 276, `_main_impl` 1604, `start_simulation_mode` 2889,
  `_start_agentic_runtime` 924, `execute_and_learn` 392, `record_outcome` 399, `_maybe_auto_spawn_server` 271,
  `build_primary_router` 245, `_try_provider` 212, `build_tool_registry` 269). Every `permissions=` executor site in
  `src/` falls in a listed function or in an unpinned one. The one gap is finding 2.
- **`record_outcome(refused=)` thread:** the plan's three `src/` sites are the complete direct and recorder call set:
  `tool_dispatch.py::execute_parallel_actions` (the direct call), plus `execute_and_learn`'s two `rec_outcome(`
  calls (the `loop_setup` partial forwards kwargs). Finding 6 is about what the parameter means, not reach.
- **Config 1.5:** `CONFIG_FORMAT_VERSION` is "1.3" on this branch and on origin/main, and a future minor is
  tolerated with a WARNING (`config_loader.py`, the `is_future_minor` path). §1A.6's downgrade claim is accurate.
  `LLMConfigSection` is SHAPE-FROZEN, so the bump is required.
- **`tools/base.py::current_policy()` with a Protocol handle** respects `LAYER_RULES` (`runtime → tools` is
  allowed, and `tools` imports nothing from `runtime`). **`atomic_write_json_at` / `append_line_fsync` in
  `utils/atomic_io.py`** are consistent with persistence-config.md §4: the ratchet counts renames outside
  `atomic_io.py`, and the module already names a lenient fsync policy (`_fsync_path`).

## Findings

### 1. [Executor + Wire] SHOULD-FIX, fail-open, incomplete fold (V9): moving the profile load to `load_llm_config` hides user profiles from ten modules that read the dicts directly

Evidence: `models/language/config.py::_apply_user_profiles_at_import`. It merges `profiles.yml` into
`_BUILTIN_PROFILES`, `_PROFILE_ALIASES` and `models/download.py::LLM_MODELS` at import. Ten modules outside
`config.py`/`download.py` read those dicts directly, with no `load_llm_config` call first:
- `cli.py` (`--list-models`, L62–72);
- `cli_utils.py` (L124, L178, L477);
- `api.py` (L379, L455–461, `list_models`);
- `doctor/checks.py` (L663, L1582, L3011);
- `normalize_llm_profile`, `list_llm_profiles` and `build_model_path`, inside `config.py` itself.

There are 23 `LLM_MODELS` references outside `download.py`. After §1A.1's "moves to first use (`load_llm_config`)",
a user-defined profile or alias silently disappears from `--list-models`, doctor's VRAM projection, `api.list_models`
and alias normalisation whenever the process has not yet called `load_llm_config`. The docstring also promises that
a malformed `profiles.yml` "raises at startup". That moves to an arbitrary later point.

Fold: one idempotent `_ensure_user_profiles()` (guarded once per process), called by every accessor
(`load_llm_config`, `normalize_llm_profile`, `list_llm_profiles`, `build_model_path`, and a new `get_profile(name)` /
`iter_profiles()`). Migrate the ten direct readers to the accessors. Add a CI grep that forbids importing
`_BUILTIN_PROFILES` / `_PROFILE_ALIASES` outside `config.py`. Add a red gate: a `profiles.yml` profile appears in
`maxim --list-models` and `api.list_models()` in a fresh process. `closed_local` never calls `_ensure_user_profiles`,
which keeps V9's env read-set property.

### 2. [Executor] SHOULD-FIX, fail-closed, incomplete fold (Executor S8): step 0a touches a pinned function

Evidence: §1A.1 says "Rule T applies to nothing here: `__init__` modules hold no pinned function". But step 0a also
edits `models/language/config.py::load_llm_config`, which the baseline pins at 269 lines. `Executor.execute`'s
`sim_logger` move is not pinned. Fold: add a step-0a row to the §7 table: `load_llm_config`, where the profile-ensure
call goes in through a helper (finding 1's accessor), with an extraction that matches the touched lines.
`lint_function_length.py::touch_rule` would catch this loudly.

### 3. [Wire + Architecture] SHOULD-FIX, fail-closed, incomplete fold (Wire B1): `code_tree.py` is stdlib-only by contract, so `is_offline()` cannot be imported there

Evidence: the `utils/code_tree.py` module docstring says "Stdlib only, by contract … nothing here may import
`maxim`" (Regression guard: `tests/unit/test_clean_flag_998.py`). `scripts/_provenance.py::_load_code_tree` loads it
by path, so that the harness never imports a different checkout's `maxim` (the harness-provenance lesson). §1A.7's
`if is_offline(): return OFFLINE_UNKNOWN` needs `maxim._offline`, which breaks that contract. A lazy import would
also bind whichever `maxim` the interpreter finds.

Second, the entry-point list is incomplete. `simulation/report.py::_code_stamp` calls `code_tree.head_commit` (L300),
not `head_hash`. `code_paths`, `tree_difference` and `on_disk` also reach `git_out`.

Fold: put the branch in `git_out` alone, as `if _latched(): return None` with
`_latched = lambda: getattr(sys.modules.get("maxim._offline"), "is_offline", lambda: False)()`. That is stdlib,
imports nothing, and is false under the by-path loader. Every caller already maps `None` to "unknown", or to dirty
for `tree_dirty`, which is fail-closed. `_code_stamp` then labels the field `"unknown: offline"` when
`maxim._offline.is_offline()`. `scripts/_evidence_records.py::unknown` refuses both forms (L207).

### 4. [Architecture] SHOULD-FIX, fail-closed, new: `maxim/_offline.py` is "stdlib-only", yet `engage()` calls into `maxim.utils`

Evidence: §1A.1 calls `_offline` "stdlib-only" and also says "`engage()` also calls
`utils/internet_access.py::lower_internet_for_session`". That import runs `maxim.utils` (lazy after 0a, but still a
package import), and `session_internet_lowered` imports `maxim.prompts.cluster_bias_annotation`. That module will in
turn import `maxim._offline` (the §1A.1 delegation), so the leaf has a back-edge. `lower_internet_for_session` also
logs "lowered for this session **by the agent**", which is the wrong attribution under offline. The stdlib-only
claim has no named guard either.

Fold: invert the edge. Make `internet_access.py::session_internet_lowered()` return `True` when
`maxim._offline.is_offline()`, a read-side implication with no write and no misattributed log, or have `engage()`
set `MAXIM_INTERNET_LOWERED` itself from a constant. Add an AST guard,
`tests/unit/test_offline_latch.py::test_offline_module_is_stdlib_only`, on the `test_clean_flag_998` pattern.

### 5. [Architecture] SHOULD-FIX, fail-open but in a covered class (the subprocess closure test): the `advisor` `LAYER_RULES` entry cannot express `runtime.agent_loop`

Evidence: `utils/audit.py::target_layer` maps an import to its first segment only, and `must_not_import` holds layer
names. A `"runtime.agent_loop"` entry never matches, and `advisor` cannot forbid `runtime` wholesale, because it
needs `runtime.executor` and `runtime.tool_policy`. §1A.3 presents the entry as a working static fence.

Fold: either drop `runtime.agent_loop` from the entry and say the subprocess closure test
(`test_advisor_gate.py::test_import_closure`) is the enforcement for it, or extend the audit with a
`must_not_import_modules` list matched by prefix, with a unit test. The other names (`agents`, `memory`,
`default_network`, `comms`, `hivemind`, `peer`, `mesh`, `tunnel`) are real top-level packages and do work.

### 6. [Wire + Architecture] SHOULD-FIX, fail-open (silent loss of learning signal), incomplete fold (Wire S1/S2, Architecture S2): the booking table ignores the existing machine-refusal route and does not say which kinds count as `refused`

Evidence:
- `runtime/tool_dispatch.py::book_machine_refusal` is the shipped route for a refusal no person made (#1185, owner
  re-decision 2026-10-08). It books `nac=None`, no cluster and no goal, but **does** book `recent_outcomes`,
  `llm_worker.record_outcome` (the reasoning carryover) and `context_pool`. §1A.2's table marks
  `llm_worker.record_outcome` as "not booked" for gate refusals. That contradicts the precedent with no stated
  reason, and drops the carryover that tells the model the tool was refused.
- `refused: ToolErrorKind | None` is not bounded. `ToolErrorKind.PERMISSION_DENIED` already exists (`tools/base.py`
  L48), and tools that ran return it for their own path refusals. If the build derives `refused` from
  `result.error_kind`, a tool-level refusal stops booking its NAc negative, with nothing to show it happened.
- Five scripts call `record_outcome` directly: `scripts/orient_substrate/9_tool_competition_repro.py`,
  `2_full_path_probe.py`, `scripts/survival_world/dark_danger_probe.py`, `r2_learned_bias.py` and
  `scripts/exp56/common.py`. The plan does not say whether `refused` is required (the TF8 `bridge_booked` precedent,
  which breaks those five loudly) or defaults to `None`.

Fold:
- Name the set: `refused` is truthy only for the executor's gate kinds and `OFFLINE`, as a frozenset constant in
  `tools/base.py`. It is never `PERMISSION_DENIED` from a tool body.
- Route a gate refusal through the `book_machine_refusal` semantics, which `record_outcome` with `nac=None`
  already gives, and keep the carryover. Alternatively, state why gate refusals differ from #1185 and record it as an
  owner decision.
- Make `refused` keyword-only and required, as `bridge_booked` is, and list the five scripts in the item 3 commit.

### 7. [Wire] SHOULD-FIX, fail-closed, incomplete fold (Wire NIT 7): `ActionRecord` is not SHAPE-FROZEN, and the hand-off names only the consumer

Evidence: `console/schemas.py::ActionRecord` is a pydantic `BaseModel` with all-optional fields, not a CC3
SHAPE-FROZEN dataclass. The contract is `console/contract_surface.json` (which lists `ActionRecord`),
`console/openapi.json`, and the `actions.jsonl` header version. That header is written by
`simulation/report.py::save_action_log` (1.2), its lines are produced by `simulation/sinks.py`, and it is read by
`console/run_store.py`. Pulse is pinned through `ui_bundle.py`'s contract version. The hand-off in §7 names only the
schema field.

Fold: the S hand-off names the producer (`sinks.py` writes `error_kind` as the enum `.value`), the `actions.jsonl`
header bump (1.3) with `run_store`'s version gate, the `openapi.json` and `contract_surface.json` regeneration, and
the contract-version note to pulse. Drop "SHAPE-FROZEN" from §1A.2's table. If only the schema field ships, the field
is always null. That is visible, but it is a dead contract.

### 8. [Wire] NIT, fail-closed: `get_version_info` does not crash under offline, it swallows

Evidence: `maxim/__init__.py::get_version_info` runs `subprocess.run(["git", ...])` directly, not through `git_out`,
inside `except Exception: pass` (two sites). `OfflineRefusal(RuntimeError)` is silently absorbed, and `git_hash` is
omitted. §1A.7 lists it among the `git_out` callers that crash. Fold: move it to the swallow-census list, and give it
an `is_offline()` early return so the census shows no hit. Its callers (`cli.py` L235, `leader_proxy` `/version`,
`peer/cli.py`) read `version` only, or are refused anyway under solo.

### 9. [Executor] NIT: three build details for step 0a

- The lazy `__getattr__` must raise `AttributeError` for unknown names. `from maxim.utils import http`
  (`hivemind/substrate_client.py` L25, `hive_cli.py`) and `from maxim.runtime import substrate_proposal` /
  `tool_dispatch` / `heartbeat` (`agent_loop.py`, `loop_setup.py`, `loop_state.py`, `loop_substrate.py`, all
  mypy-public files) depend on Python's fallback to submodule import.
- Mirror the table under `if TYPE_CHECKING:` in each init, so mypy keeps real types instead of `Any`.
- The closure test should state its dispatch-time expectations. `Executor` imports `modes.definitions` per dispatch
  (L209/230/250), plus `agents.bus`, `decisions.causal_link` and `runtime.tool_dispatch` (about 45 modules, including
  `memory.hippocampus`) only behind a pain detector or a tool-pain bridge. The sealed executor must carry neither.

Step 0a's PR should also run a fresh-subprocess "import every module" sweep, because lazy inits can surface latent
import cycles that the eager order hid.

### 10. [Wire] NIT, fail-closed: `ToolErrorKind.OFFLINE` breaks a member-count pin

`tests/unit/test_stop_reasons.py::TestToolErrorKind::test_member_count` asserts 7, and `test_all_members_exist`
enumerates them. Update both in item 3's commit, together with the gate kinds. `proprioception/pain.py`'s map falls
back to `TOOL_FAILURE`, which is harmless because a refusal books no pain.

### 11. [Architecture] NIT, fail-closed: the canonical-parser line is a missing hand-off

CLAUDE.md §Environment Variables names `cluster_bias_annotation.annotation_disabled_via_env` as the canonical truthy
parser. After §1A.1 that function delegates to `maxim._offline.truthy`. The §7 CLAUDE.md hand-off lists only the
`closed_local` stub. Add that line, and the same line in any brief that cites the parser. Since 28 callers go through
the delegator, the old pointer keeps working.

### 12. [Architecture] NIT: advisor_mode.md markers

The "Superseded" markers are correct for the guard line and the context row. A marker on §"draft interfaces",
pointing at §1A.8, would close the loop that §1A.8's "and that file says so" promises, if one is not already there.

## Verdict

No BLOCKER. Findings 2, 3, 4 and 7 are SHOULD-FIX and fail-closed: a lint or an existing test catches the
contradiction, or the cost is a visible null. They go to the step-0a, item-2/3 or red-gate PRs. Finding 5 is
fail-open but in a covered class, the subprocess closure test. Findings 8–12 are NITs.

**Two SHOULD-FIXes are fail-open and not in a covered class: 1 (user profiles silently missing after V9's
relocation) and 6 (`refused` unbounded, and the departure from #1185 unstated).**

**One more delta round is owed (recommended, the strict option),** scoped exactly to the folds of findings 1 and 6:
- the profile-accessor rule and its grep, plus the red gate;
- the `refused` kind set, the `book_machine_refusal` alignment or the owner decision recording the difference, and
  required-vs-default for `refused`.

The alternative is to route finding 1 to step 0a's own mandated three-lens round, and finding 6 to item 3's. That is
weaker, because the plan text those rounds review would still say the opposite. If the two folds land as written
above, that round can be a single lens (Wire) with the stop rule.
