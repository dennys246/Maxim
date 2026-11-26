# Delta round 7: #1315 item 10b (PR #1735), commit 8a8e4f34 "fix(#1602)"

Lenses: Executor + Wire integrity. Reader: a different reader from the author. Tree: review_d7/tree (exported). Probes ran in review_d7/probes, del1, del2 and leakprobe. MAXIM_OFFLINE=1 with a scratch HOME/XDG/MAXIM_DATA_HOME.

## What checks out

- **Equivalence of the two callers.**
  - CLI (`cli_utils.py::configure_cpu_fallback_model`): the code is byte-equivalent. `in os.environ` was already the test there, and the N_GPU_LAYERS setdefault is unchanged.
  - Robot (`agentic_runtime.py::AgenticRuntimeMixin._apply_llm_backend_fallback`): the old `setdefault` and the new `"MAXIM_LLM_PROFILE" not in os.environ` agree on every input, including `MAXIM_LLM_PROFILE=""`. Neither writes an empty value, and every reader `.strip()`s it to "unset", so an auto-detect follows.
  - An operator profile is preserved. N_GPU_LAYERS is set to "0" only when absent.
  - Log change: the robot no longer logs "Using CPU-friendly LLM: smollm" when the operator's profile stands. That is an improvement (see N2 for the CLI).
- **Other writers of the default.** No other writer in src/ sets `MAXIM_LLM_PROFILE=smollm-1.7b-instruct` without marking it.
  - The remaining env writers are `normalize_args` (`--llm`, the operator's), `configure_cloud_solo_auto_detect` (a cloud profile, which `_ensure_lane_profiles_available` skips outright, so V28 does not apply), `_maybe_pin_pre_upgrade_profile`, `_restore_persisted_profile` and the `_llm_env_map` projection.
  - The other smollm literals are tier tables (`lane_models.detect_tiers` small lane, `_REVIEW_PROFILE`, the VRAM tables), the registry/aliases, and `download.DEFAULT_LLM`. None of them writes the env.
  - In scripts/, the writes (`run_maxim_cpu.sh`, the benchmark/exp44 harnesses) are operator-chosen exports into a child process. A child reads them as pinned, which fails closed, as the `_MAXIMS_OWN_PROFILES` comment states.
- **#1538 registry.** `test_no_new_writer_sets_the_session_profile_around_a_build` passes. It stays meaningful: `adopt_cpu_fallback_profile` writes through a literal subscript, which the scan catches, and a re-introduced `os.environ.setdefault("MAXIM_LLM_PROFILE", ...)` in agentic_runtime would be caught as a new writer.
- **Layering.** `embodied_runtime` has no `LAYER_RULES` entry in `utils/audit.py`, and the module already imports `maxim.runtime.*` (bootstrap, sim_adapter, lane_backends.build_run_router). The import is fine.
- **Deletion probes.**
  - `del1` reverts the robot caller to the old setdefault: `test_the_robot_runtime_cpu_fallback_default_is_not_a_pin` FAILS.
  - `del2` drops `mark_profile_maxims_own` inside the helper: both the CLI and the robot `*_default_is_not_a_pin` tests FAIL.
  - The unmodified tree: all 4 cpu_fallback tests pass. The guards bite.
- **Lints.** `ruff check` and `ruff format --check` are clean on the 5 files. `lint_function_length` is clean (its diff-scoped touch rule was skipped because the tree has no git base, but the touched functions are about 40 lines). `lint_no_silent_swallows` is clean, and the delta adds no handler.

## Findings

### S1: SHOULD-FIX. The fallback the fold claims never happens: Maxim's own default rides the env, and the re-walk and the backend both re-read the env

`lane_models.py::detect_tiers` (`m_profile = env_profile or _pick_medium_profile(...)`), `lane_models.py::build_profile`, `lane_backends.py::_ensure_lane_profiles_available` and `LaneBackendManager._build_local_backend`.

The fold makes `_profile_is_pinned("smollm-1.7b-instruct", ...)` False on the robot path. The changelog line it adds goes further: "a GPU-less offline robot run falls back instead of refusing". That does not hold.

The re-walk calls `detect_tiers(..., profile_available=<excludes smollm>, profile=None)`. `build_profile(None)` returns the env's `smollm-1.7b-instruct`, which beats `profile_available`, so the re-walk hands back the same missing model. After that, `_build_local_backend` serves `_build_profile(self.profile_override)`, which is the env profile again, whatever the lane was re-walked to. This is the "filed residual" in `build_run_router`'s docstring and in llm-routing.md, but in this case it swallows the whole V28 behaviour.

Probe (review_d7/probes/rewalk_probe.py: offline, no GPU, 16 GB, every model on disk except smollm):
```
adopted: True smollm-1.7b-instruct ; pinned? False
tiers: {'small': 'smollm-1.7b-instruct', 'medium': 'smollm-1.7b-instruct'}
after: {'small': 'smollm-1.7b-instruct', 'medium': None}
served by backend: smollm-1.7b-instruct
degraded: 'smollm-1.7b-instruct is not on disk; lane small runs smollm-1.7b-instruct', '... lane medium has no model'
```
Failure scenario: a GPU-less robot (or CLI agentic run), offline, without smollm on disk but with another model on disk.
- **Before the fold:** a named refusal (`OfflineRefusal`).
- **After the fold:** no refusal, but `build_run_router` returns no LLM. The run proceeds with no LLM, and the provenance `offline` block records a FALSE degraded entry, "lane small runs smollm-1.7b-instruct", for a model that is not on disk.

V27 makes provenance a claim surface, so the false record matters. The "runs {substitute}" record also fires whenever the re-walk returns the declined profile itself (the small lane is hard-coded smollm). That is item-10b code in this PR, outside the delta.

The new tests assert only the predicate (`_profile_is_pinned`), not the composition. This is the "shipped the pieces, not the composition" lesson. The CLI path has the same defect from the earlier 10b rounds, and the changelog line "or the CPU-fallback default ... falls back to a model already on disk" is untrue there too.

Root cause: Maxim's own default is expressed as the session's `MAXIM_LLM_PROFILE`, which every downstream resolver reads as authoritative. Proper scope, an owner decision under "no band-aid fixes":
- either the CPU fallback stops writing `MAXIM_LLM_PROFILE` and becomes a tier preference / auto pick (for example, `detect_tiers` prefers smollm on a GPU-less box through `profile_available`),
- or `build_profile` and `detect_tiers` treat a `_MAXIMS_OWN_PROFILES` env value as auto, so it yields to `profile_available` and the re-walk.

Either way:
- The degraded record must say "has no model" when the substitute is the declined profile.
- A composed test must drive `build_run_router` (or `_ensure_lane_profiles_available` plus `_build_local_backend`) and assert that a model on disk is served.
- Until then, the changelog line must not claim the fallback.

### S2: SHOULD-FIX. The single writer overwrites an operator's `llm.profile` from config.json, contrary to its own docstring and V25

`lane_backends.py::adopt_cpu_fallback_profile`.

The docstring says "unless the operator named a profile", but the guard checks only the env. `config.json` `llm.profile` reaches the env only later, at build time (`_apply_lane_config_to_env_locked`), and `resolve_setting` ranks env above config. The CPU fallback runs before any build, so it wins.

Probe (scratch config.json `{"llm":{"profile":"phi-2"}}`, env unset):
```
before: ('phi-2', 'config')
adopted: True
config override: llm.profile='smollm-1.7b-instruct' (env) shadows 'phi-2' (config.json)
after: ('smollm-1.7b-instruct', 'env')
phi-2 pinned for medium: False
```
Failure scenario: an operator runs `maxim config set llm.profile phi-2` on a GPU-less box, then runs the CLI agentic mode or the robot runtime. Their chosen model is silently swapped for smollm. Online, it is just swapped. Offline, a missing phi-2 is no longer refused by name. That violates V25 ("a profile you chose ... `llm.profile` ... is refused by name and never swapped", the changelog's own sentence).

This is pre-existing on both paths (the old code was identical). But the fold now names this function as THE writer of the operator-vs-Maxim contract, and the fix sits inside it and is small: adopt only when `"MAXIM_LLM_PROFILE" not in os.environ` and `resolve_setting("llm.profile")[1] == "default"`, with a test that config.json `llm.profile` survives. Under "filing is not fixing" (a small known fix is done now), fold it. If S1's proper fix removes the env write entirely, S2 goes away with it.

### S3: SHOULD-FIX. The new tests leak `MAXIM_LLM_N_GPU_LAYERS=0` into every later test

`tests/unit/test_offline_call_sites_10b.py::test_the_robot_runtime_cpu_fallback_default_is_not_a_pin` and `::test_an_operator_profile_survives_the_robot_runtime_cpu_fallback` (and the pre-existing CLI pair, same pattern).

`monkeypatch.delenv(name, raising=False)` records nothing when the variable is absent, which is the normal case (pytest `MonkeyPatch.delitem`). The code under test then sets it to "0", and nothing undoes that. tests/conftest.py scrubs `MAXIM_LLM_PROFILE` but not `MAXIM_LLM_N_GPU_LAYERS`.

Probe: the two new tests followed by a probe test print `N_GPU_LAYERS after: '0'`.

Failure scenario: a later test in the same worker that builds a local backend skips `cfg.n_gpu_layers` (`if "MAXIM_LLM_N_GPU_LAYERS" not in os.environ:` in `_build_local_backend`), and `load_llm_config` reads 0 GPU layers. The result is an order-dependent failure or coverage drift, the class behind the xdist revert.

Fix: `monkeypatch.setenv("MAXIM_LLM_N_GPU_LAYERS", "x")` and then `delenv`, so the restore is recorded. Do the same in the two CLI tests.

### N1: NIT. Stale comment
`lane_backends.py::_MAXIMS_OWN_PROFILES`: the comment still says "(the GPU-less CPU fallback, cli_utils)". Name `adopt_cpu_fallback_profile` instead.

### N2: NIT. The CLI still announces smollm when the operator's profile stands
`cli_utils.py::configure_cpu_fallback_model`: the CLI still logs the WARNING and prints "No GPU detected — using smollm-1.7b-instruct (CPU mode)" when `adopt_cpu_fallback_profile()` returns False (the operator's profile stands). The robot path now gates its log on the return value, so the two callers disagree. Gate the CLI message the same way.

### N3: NIT. The default string is duplicated
`CPU_FALLBACK_PROFILE` duplicates `models/download.py::DEFAULT_LLM`, and the log strings in both callers hard-code "smollm-1.7b-instruct". Interpolate the constant so a future change of the default cannot leave the messages wrong.

### N4: NIT. Import order
In `test_offline_call_sites_10b.py`, `import os` sits after `import threading`. Ruff passes (isort is not enforced), so this is cosmetic.

## Verdict

FOLD
