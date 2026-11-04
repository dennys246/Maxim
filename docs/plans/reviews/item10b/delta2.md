# #1315 item 10b — delta round 2 (the staged fold of delta round 1, on top of HEAD bd5b43e0)

Reader: a different reader, delta only (`git diff --cached HEAD`, src: 10 files, +105/-61).

How the probes ran:
- In a copied tree: `scratchpad/review10b/d2/tree` (rsync, no `.git`). `maxim.__file__` was asserted to resolve to the copy.
- HOME, XDG_CONFIG_HOME and MAXIM_DATA_HOME pointed at scratch dirs, MAXIM_OFFLINE=1 was set, and the network guard
  was active.
- Every probe is a pytest test in `tests/unit/test_zz_delta2_probes.py` in the copy. No online path, sim or LLM was
  run.

Targeted tests on the copy:
- `test_api_expansion`, `test_offline_call_sites_10b`, the 5 benchmark files, `test_decision_log`,
  `test_offline_latch_unit` and `test_offline_startup_10a`: 366 passed, 1 xfailed (the known 10a SF1 gate).
- Every unit file that mentions warmup, `_check_one_robot`, `check_robot_reachable` or `_LlamaCppBackend`: 519
  passed, 3 xfailed (all pre-existing).
- `test_offline_call_sites_10b.py` run ahead of the lane, spawner, override and leader files: 209 passed.

Lints, run read-only in the worktree:
- `lint_function_length` is clean, the touch rule included.
- `lint_no_silent_swallows` is clean.
- `ruff check` is clean on the touched files, and `ruff format --check` is clean.

## Delta-1 SHOULD-FIX: status

| # | Status |
|---|---|
| N1 warmup thread | **Folded correctly.** The re-indent is verbatim. The only changes are the `SECURITY_REFUSALS` import, the try, and moving `_warmup_failed` / `_ready_event.set()` into `finally`. The inner `except Exception` in the no-`warmup` branch is unchanged. Probes: (a) a refusal from `backend.warmup()` sets ready, sets failed, and the thread raises nothing. (b) A refusal from `_get_backend()` is now caught too (same result). (c) A refusal from `complete()` in the no-warmup branch is caught by the existing inner handler (same result). (d) The success path is unchanged. (e) A NON-security exception such as `ValueError` from `warmup()` still escapes and kills the thread through `threading.excepthook`, exactly as before. The difference is that `finally` now sets ready and failed first, so `wait_ready` no longer stalls. That is strictly better, and no exception escapes differently. llama `.with_traceback(None)` verified: the traceback length stays at [2, 2, 2] across three calls. |
| N2 benchmark consumers | **Folded correctly.** `exit_code()` returns 2/0/1, and an empty results dict gives 0, as `all([])` did before. Both CLI exits use it. `api.benchmark` skips refused models (test pins it). Other consumers, from a grep of `src/` + `scripts/`: (a) `_compare_baseline` is harmless, because a refused model has `metrics={}` and so no shared keys. (b) `save_report` writes `refused`. (c) `summary_table` and the rankings were folded in round 1. (d) **`write_paper` still records a refused model as "Score: 0.00, passed: False"** in the experiment log. That is delta-1 NIT e, still open. (e) No `scripts/` file imports `BenchmarkRunner` or `ModelResult`, and `research` / `campaign` do not consume `BenchmarkReport`. |
| N3 robot rows | **Folded correctly.** Probe with two robots, offline: `[('robot:a','info','robot','refused (offline): …'), ('robot:b','info','robot','refused (offline): …')]`. No duplicate row is appended, because no probe dialled, so the ledger delta is empty. `--retry` only collects rows whose status is not `ok`/`info`, so these rows are never re-probed. The wrapper special-case is gone. |
| N4 / V28 pin | **Folded in shape, but the tier half is vacuous (F2), and one CPU-fallback copy is unmarked (F1).** On "same source, same key": yes. `_profile_is_pinned` reads `getattr(load_llm_config(), "_raw", {}).get("tiers")`, which is byte-for-byte the read `build_primary_router` feeds to `apply_tier_config_overrides`. It is keyed by tier name and compares the raw `model_profile` string, which the override copies through unchanged. Ordering is sound too: when `--llm` re-points a lane, the tier check misses and `resolve_setting` (env) pins it. The marked fallback returns False only when the env still holds that exact value. |

The other folds:
- `decision_log` online records keep the old shape (test pins both directions).
- The `test_api_expansion` mocks were updated.

These are correct.

## New findings

**F1 SHOULD-FIX (low; fail-closed): V28 marks one of two CPU-fallback copies.**
- Where: `embodied_runtime/agentic_runtime.py::_apply_llm_backend_fallback`.
- This method is a live duplicate of `cli_utils.configure_cpu_fallback_model`, called from `agentic_runtime.py`
  (its runtime start). It still does `os.environ.setdefault("MAXIM_LLM_PROFILE", "smollm-1.7b-instruct")` with no
  `mark_profile_maxims_own`.
- Scenario: a GPU-less agentic runtime, offline, with smollm missing and a smaller model on disk. The profile reads
  as pinned and refuses instead of re-walking. V28 names "the GPU-less CPU-fallback default", so the decision is
  half-applied.
- Verified by reading: grep of `MAXIM_LLM_PROFILE` writers in `src/`. The other writers are `--llm`, `api` /
  `create` `model=` and the sim's `aut_model`, all genuine choices.
- Fails closed (refuses), which is the class V28 already covers.
- Fix (root, not a second mark): have `_apply_llm_backend_fallback` call `cli_utils.configure_cpu_fallback_model`,
  or a shared helper, so that one function owns the default and its mark. Alternatively file it as a follow-up.

**F2 SHOULD-FIX (pre-existing root, surfaced by V28): `LLMConfig` has no `_raw`, so llm.json `tiers` never apply,
and V28's tier pin can never fire.**
- Where: `runtime/lane_backends.py::build_primary_router` (the `_raw` read, introduced in feb65551) and
  `lane_backends.py::_profile_is_pinned`.
- `LLMConfig` is `@dataclass(frozen=True, slots=True)`, and nothing in `src/` sets `_raw`. So
  `getattr(load_llm_config(), "_raw", {})` is always `{}`.
- Consequences:
  - `apply_tier_config_overrides` and `load_function_overrides` never receive llm.json `tiers` / `functions`.
    That is a long-standing dead config surface.
  - The fold's tier-pin branch is unreachable.
  - `test_an_llm_json_tier_profile_is_pinned` passes only because it monkeypatches `load_llm_config` to return a
    `SimpleNamespace(_raw=…)` that a real `LLMConfig` cannot be.
- Verified: probe p9 wrote a real llm.json `{"tiers":{"medium":{"model_profile":"phi-2"}}}` as the only
  candidate. The `build_primary_router` read gave lanes `{'large': None, 'medium': None, 'small': None}` (no
  override applied), and `_profile_is_pinned("phi-2"…)`, fed from those lanes, was False.
- Today this is neutral: no lane ever carries an operator tier profile, so nothing is mis-swapped.
- The hazard is latent. Whoever revives the tiers read (for example by reading the JSON directly) must change BOTH
  sites, or tier profiles silently become swappable.
- Do NOT band-aid in this branch. Activating the dead `tiers` / `functions` config changes routing behaviour beyond
  10b and needs the owner.
- Fix:
  1. File an issue: "llm.json tiers/functions never apply: `_raw` is not an `LLMConfig` attribute (since feb65551)".
     The proper fix is one helper, for example `_llm_json_section("tiers")`, used by both `build_primary_router` and
     `_profile_is_pinned`.
  2. In this branch, record in the V28 note that the tiers half is inert until that issue lands.
  3. Re-point the test docstring, or mark it as pinning the helper contract rather than shipped behaviour. That is a
     test/docstring-only touch, and I would not owe another round for it.

**NITs**
- a. `tests/unit/test_offline_call_sites_10b.py::test_the_cpu_fallback_default_is_not_a_pin` and
  `test_an_operator_profile_survives_the_cpu_fallback` leak `MAXIM_LLM_N_GPU_LAYERS=0` into later tests.
  `configure_cpu_fallback_model` setdefaults it on the real `os.environ`, and no conftest scrub covers it
  (`MAXIM_LLM_PROFILE` is scrubbed). Probe p10 printed `LEAK 0` in the next test. The lane, spawner and override files
  still passed after it (209 passed), so the leak is latent. Fix: add `monkeypatch.setenv("MAXIM_LLM_N_GPU_LAYERS",
  "x")` followed by `monkeypatch.delenv(...)` so it is restored.
- b. Delta-1 NIT a is still open: the comment above `_ensure_lane_profiles_available` in `build_primary_router`
  ("offline + non-tty … this is a no-op") is stale under V25.
- c. Delta-1 NIT e is still open: `write_paper` feeds refused models into the experiment log as scored 0.00 /
  failed. Offline, its `LLMRouter()` refusal is also absorbed by a pre-existing broad `except` and reported as "No LLM
  available".
- d. `_profile_is_pinned` treats a config.json `llm.profile` that equals the CPU default as unpinned once the fallback
  has exported it. The env layer shadows the config layer, so the operator's own identical pin swaps. This is a narrow
  edge on the fail-open side and costs a degraded record. It goes away if the fallback stops exporting through
  `MAXIM_LLM_PROFILE` (the F1 follow-up).

## Verdict

**STOP.** BLOCKER 0 · SHOULD-FIX 2 (F1, F2) · NIT 4.

All four delta-1 SHOULD-FIXes are folded correctly. Nothing regressed, and no exception escapes differently except
for the better.

F1 fails closed and sits in V28's own class, so it goes to a follow-up issue under the stop rule. F2 is a pre-existing
dead read that the fold built on consistently. It is neutral today, and its proper fix needs the owner. So it is a
follow-up issue plus a note, not a branch fold.

The owner should be told that V28's "llm.json tiers pin" half is currently vacuous, because llm.json `tiers` have
never applied.
