# Step 0a (#1315, V9): wire integrity lens

Reviewed: the staged diff on `feat/1315-step0a-lazy-inits`, now commit `7bfbee71` (27 files), against
`docs/plans/atonia_gate_approach_notes.md` §1A.1 and `docs/CODE_REVIEW.md`. Reviewed 2026-10-29. Main is the archived
`mainsrc/src` snapshot. While I reviewed, the working tree picked up unstaged folds to `profile_loader.py`,
`models/__init__.py`, `config.py`, `lane_backends.py::_lane_profile_is_cloud` and `benchmark_cross_session.py`. I
note them where they overlap a finding below. They do not touch W1.

Method: AST scans over `src/`, `scripts/` and `tests/` (the scripts are in the session scratchpad under `wire/`), then
reproductions of each finding in `python3 -I` children, one on the branch and one on main.

## Verdict

The namespace contract (1) is intact: no consumer in the repo reads a submodule attribute it did not import. The
executor's `suppress_nac` contract (4) is intact. Import-time side effects (3) carried nothing a consumer depended
on, except the profile merge. The profile tables (2) have no stray reader or writer.

**The one real finding (W1) is a lifecycle change on the failure path.** On main, a malformed `profiles.yml` killed
every process at import. Now it raises only when a profile is looked up, and about 15 `except Exception` handlers
outside the diff now see that `ConfigurationError` for the first time. One of them turns it into a silent model
downgrade. I reproduced it.

## Findings

### W1: SHOULD-FIX, fail-open. A malformed `profiles.yml` silently downgrades the large tier to smollm

`runtime/llm_server.py::profile_has_local_file` (outside the diff) wraps `normalize_llm_profile`,
`list_llm_profiles` and `load_llm_config` in `except Exception: return False`. `lane_models.detect_tiers` uses it as
`profile_available`, and the result is "no profile has a local file". The tier walk then falls back to the smallest
profile. It never reaches the `get_profile` call that would raise, so nothing is logged.

Reproduced. Sandbox `HOME` with a downloaded (stub) `Qwen2.5-14B-Instruct.Q4_K_M.gguf`, 16 GB VRAM:

| `profiles.yml` | main | branch |
|---|---|---|
| valid (`profiles: {}`) | large = `qwen2.5-14b-instruct` | large = `qwen2.5-14b-instruct` |
| malformed | `ConfigurationError` (at import) | large = **`smollm-1.7b-instruct`**, no error, no log |

The probe was
`detect_tiers(RuntimeCapabilities(has_gpu=True, gpu_type='cuda', vram_gb=16.0, ram_gb=32.0), profile_available=lane_backends._profile_has_local_file)`
(script: scratchpad `wire/tiers.py <src>`).

The same thing happens in `build_primary_router` (`lane_backends.py`, around L2155) and in the mid-run re-walk
`_ensure_lane_profiles_available` (around L1885). Later accessors in the same build may still raise. Whether they do
depends on which lanes are local and on the flags. Any lookup made after the tier walk has already chosen the model
can raise, but it cannot undo the choice.

This is the case the spec says it rules out. §1A.1 says "Failure is never cached … a swallowing `except Exception`
site cannot turn the first failure into a permanently silent 'no user profiles'". The guard does stay unset, but the
caller **caches the decision**: the tier assignment is computed once and kept for the whole run.

Other swallowing handlers that a malformed file now reaches for the first time (AST scan: an accessor called inside
the `try` body of an `except Exception` handler):

- `cli.py::_main_impl` (L1385: "LLM init failed — using template fallback"; L1725)
- `agents/exec_agent.py::_ensure_router` (returns `None`)
- `simulation/orchestrator.py::start_simulation_mode` (AUT router fallback)
- `simulation/research_orchestrator.py::_build_research_llm`
- `embodied_runtime/agentic_runtime.py::_start_agentic_runtime`
- `lane_backends.py`: `_build_local_backend`, `_build_remote_backend`, `_placement_entry_to_provider` (returns
  `(None, False)`), `_maybe_auto_spawn_server`, `_lane_profile_is_cloud` (debug only), `has_cloud_billing_surface`
  (returns `True`, the conservative answer)
- `models/download.py::delete_llm` (`pass`)
- `llama_backend._fetch_missing_model`
- `doctor/checks.py` VRAM checks (`projection = None`)
- `leader_proxy._handle_debug_vram`

Most of them log a WARNING and carry on degraded. On main all of them were unreachable in this state.

Scope of the proper fix, for an owner decision. Do not special-case `profile_has_local_file`:

- **(a) One seam.** The entry points that build LLM state call `_ensure_user_profiles()` up front and let it raise:
  `build_primary_router`, the CLI's sim/chat path, the leader/peer daemon start, and `maxim.create.router`. That
  restores "refuses at startup" for every command that needs a model. Commands that look up no profile keep the
  intended relaxation. **This is my recommendation.**
- **(b) Narrow every handler** in the list above to re-raise `ConfigurationError`. That is about 15 sites, and the
  next swallowing site will repeat the bug.

Either way, add a red gate to `test_profile_readers_v9.py`: with a malformed file, `detect_tiers` with the real
`profile_available` raises instead of returning tiers.

### W2: SHOULD-FIX (docs). External `pymaxim` users lose package-attribute access to unimported submodules

`import maxim.runtime; maxim.runtime.executor.Executor` worked on main because the eager init had imported the
submodule. On the branch it raises `AttributeError`. The same holds for `maxim.agents.bus`, `maxim.tools.registry`,
`maxim.utils.paths` and every other submodule that the eager inits used to load. In-repo there are **zero**
consumers. My scan covered every `maxim.<pkg>.<sub>…` attribute chain, aliases included, in `src/`, `scripts/` and
`tests/`, checked against the modules each file actually imports. It printed nothing, and it does flag a planted
case. The published API is still a public surface, though. The changelog fragment says "every other public import
keeps working", which is true of imports but not of this attribute access. Add one line naming the break and the
fix (`import maxim.runtime.executor`).

### W3: NIT (fail-closed). Doctor text for "User profiles" is stale

`doctor/checks.py::check_user_profiles` has two stale pieces of text:

- The docstring says a malformed file "blocks `maxim` startup (the loader raises `ConfigurationError` from module
  import)".
- The fix hint says "`maxim` will refuse to start until this resolves."

Neither is true now, and W1 is the reason. Separately, a positive change: on main, `maxim doctor --json` with a
malformed file died with `ConfigurationError` before printing anything, so this check could never report. On the
branch the doctor runs and reports `User profiles | fail`, plus warnings from the tier checks. I reproduced both.

### W4: NIT. `model_cli` help still says "profiles load at import"

`models/model_cli.py::_build_argparser` ("add" description). The user-visible meaning, usable on the next
invocation, still holds. Only the parenthetical is stale. The folds already fixed the matching docstrings in
`profile_loader.py` and `scripts/benchmark_cross_session.py`.

### W5: NIT, already folded. `_lane_profile_is_cloud` called `logger.debug` on a possibly-`None` logger

The code was moved verbatim. Before this change the malformed-file case could not reach that handler. The
working-tree fold adds the `None` check.

## Contract maps

### 1. Package namespaces `maxim.runtime` / `agents` / `tools` / `utils` (now PEP 562)

| Producer / consumer outside the diff | Status |
|---|---|
| Every `from maxim.<pkg> import Name` in `src/`, `scripts/`, `tests/`, `docs/` (AST scan, 4 packages + `maxim.models`) | Intact. Every name is in `_LAZY` or is a real submodule (submodule fallback). The only misses are the placeholder `X` in the approach notes. |
| Attribute chains `maxim.<pkg>.<sub>[.x]` with only the package imported, including aliases (`import maxim.runtime as rt`) and `from maxim import runtime` | Intact. Zero hits repo-wide (W2 covers external users). |
| Main's eager names compared with branch `_LAZY` / `__all__` | Identical sets for all four packages. Every `_LAZY` entry resolves (31/69/34/33). There are no `_LAZY`/submodule name collisions. |
| `mock.patch("maxim.runtime.Executor")`, `patch("maxim.agents.base.GoalAgent")`, `patch("maxim.models.LLM_MODELS")`, `patch.dict("maxim.models.LLM_MODELS")` | Intact (probed on Python 3.12: enter and exit both clean, and the name re-resolves afterwards). |
| `src/maxim/__init__.py::__getattr__` (`_API_VERBS` / `_API_SUBMODULES` / `_API_TYPES`) | Unchanged. It never exposed `runtime` and the other three packages. |
| String class-path resolvers (`import_module` + `getattr`), star imports, `hasattr`/`vars` introspection of the packages | None in the repo apart from the `maxim/__init__` map and `models/vision/registry.py`, neither of which targets these packages. |
| `scripts/lint_orphan_modules.py` | Clean on the branch (0 orphans). |
| Import cycles that the eager inits hid | Covered by `tests/integration/test_fresh_import_sweep.py` (slow). I did not run it. Spot imports of `maxim.runtime.executor`, `cli`, `api`, `agent_loop`, `simulation.orchestrator`, `lane_backends`, `doctor.checks`, `peer.cli` and `leader_proxy` in fresh `-I` children all pass. |

### 2. Profile tables and accessors

| Producer / consumer outside the diff | Status |
|---|---|
| Writer `profile_loader.py::apply_user_profiles` | Called only from `config._ensure_user_profiles` and from doctor (throwaway dicts). It makes no accessor calls, so there is no re-entrant merge. It validates everything before it mutates. Intact. |
| Runtime writers of `_BUILTIN_PROFILES` / `_PROFILE_ALIASES` / `LLM_MODELS` | None in `src/` or `scripts/` apart from the loader. `model_cli` (`maxim model add`/`remove`) writes the YAML file only and reads nothing back in-process, so no writer can run before the first ensure and get clobbered. |
| Readers in `src/` and `scripts/` | All go through `get_profile`, `iter_profiles`, `profile_aliases`, `normalize_llm_profile`, `list_llm_profiles`, `load_llm_config` (via `_normalize_profile`) or `llm_models()`. The only remaining raw reads are in `config.py::load_llm_config`, which runs after `_normalize_profile` has ensured, and inside the accessors themselves. The AST lint test passes. |
| Import-time accessor calls (module level, class bodies, defaults, decorators) | None in `src/`. With a malformed file, the nine major entry modules import cleanly. |
| Tests reading raw `_BUILTIN_PROFILES` / `LLM_MODELS` (`test_profile_metadata`, `test_lane_backends`, `test_doctor`, `test_phase1_fixes`, `test_download_consent_1343`) | They want the built-ins, so this is fine. `test_profile_metadata` now sees built-ins only, even outside the conftest isolation, which removes the 2026-10-27 rig failure class. |
| `patch.dict` wrapping the first ensure | No `patch.dict` on these tables exists in `tests/`. The spec's harmlessness argument (conftest isolates `XDG_CONFIG_HOME`) holds. |
| Failure path (every swallowing caller) | **W1.** |

### 3. Import-time side effects that the eager inits used to trigger

I took the closure of `import maxim.{runtime,agents,tools,utils}` on main (236 modules) and listed every top-level
call and every top-level assignment from a non-trivial call:

| Side effect | Who relies on it | Status |
|---|---|---|
| `models/language/config.py::_apply_user_profiles_at_import` | Every profile reader | Replaced by the ensuring accessors (contract 2). The failure path is W1. |
| `agents/modality.py` `_SUBSTRATE_MAP.update(...)` | Readers of that map, which import `modality` themselves | Intact. |
| `hardware/capabilities.py` constants, `content_safety.frame_rule()`, `filesystem_policy` locks, `substrate_proposal._NoSituationCue()` | Module-local | Intact. |
| `StoreFileOwnership.__init_subclass__` | Store classes, at their own definition | Intact. |
| `utils/http.py` `_offline.on_engage(close_all)` | Offline latch | Intact. `on_engage` runs immediately if the latch is already up. Lazy loading also makes `_set_imported_library_flags` reach fewer pre-imported libraries, but they read `HF_HUB_OFFLINE` from the env at their own import. |
| `simulation/sim_logger.py` `atexit` registrations (formerly loaded on every `Executor.execute`) | Only processes that use `sim_logger` state, and those import it themselves | Intact. |

`models/language/cancellation.py` has no import-time registration: there is no `register_stop_hook` in the tree.

### 4. `Executor` `suppress_nac` ↔ `ToolPainBridge`

| Consumer | Status |
|---|---|
| `_run_started` `record_tool_start` / `record_tool_complete` / `record_tool_embodiment_failure`, `_fail_ran` → `_book_failure` | Each one is gated on `self._tool_pain_bridge is not None` as well as `suppress_nac`. When the bridge is `None`, `suppress_nac` is never read for an effect, so forcing it to `False` is unobservable. |
| `_stamp_invocation` (pops RPE and pain) | Gated on the bridge, and does not read `suppress_nac`. |
| Post-construction bridge assignment (`tests/unit/test_build_executor.py`, `test_tool_failure_credit_1200.py`, `test_interoceptive_outcome.py`) | The check reads the attribute on each dispatch, so a late assignment is honoured. Intact. |
| `pain_interceptor.py`, `agentic_runtime.py`, `agent_loop.py`, `fear_agent.py` | They read `executor._tool_pain_bridge` and never `suppress_nac`. Intact. |
| `_SPECULATIVE` (prefetch) | Read only on the bridge branch, where it is meaningful. Intact. |
