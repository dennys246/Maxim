# Delta round 8, Wire integrity: #1315 item 10b (PR #1735), staged fold on 8a8e4f34 (owner decision V30)

Reader: a different reader from the author. Tree: review_d8/tree (exported). Probes: review_d8/wire_probes/ (`robot_primary.py`, `robot_build.py`, `rewalk_pinned.py`), run with MAXIM_OFFLINE=1 and a scratch HOME/XDG/MAXIM_DATA_HOME. Narrow tests: `test_aut_router_profile_1538.py` + `test_offline_call_sites_10b.py`: 71 passed.

## What checks out

- **(d) Removed symbols.** `_MAXIMS_OWN_PROFILES`, `mark_profile_maxims_own`, `adopt_cpu_fallback_profile` and `CPU_FALLBACK_PROFILE` have no caller left in src/, scripts/, tests/, docs/agents/ or changelog.d/. The only hits are historical review reports (`docs/plans/reviews/item10b/delta2.md`, `delta7.md`) and the notes paragraph that says the marker is gone.
- **(e) `PROFILE_ENV_WRITERS`.** It is accurate. `test_no_new_writer_sets_the_session_profile_around_a_build` passes, and the only new direct writer is `cli_utils.py::configure_cpu_fallback_model` (a literal subscript, inside the scan's view). The robot runtime writes no `MAXIM_LLM_PROFILE`.
- **(b) The process-global preference.** Production callers of `detect_tiers` are `build_primary_router` and its re-walk (`_ensure_lane_profiles_available`) in the same process, plus `doctor/checks.py` (lines 89 and 2908/2909). Doctor runs in its own process and never sees the preference. Only `_apply_llm_backend_fallback` sets the preference. The CLI agentic path writes the env instead, and the env still beats the preference (`env_profile or _preferred_cpu_pick(...)`).
- **(c) Consumers of the degraded record.**
  - `simulation/report.py`'s `offline` block and `decision_log`'s `offline_degraded` both de-duplicate on `(site, reason)` and count nothing. The "exact count" in report.py is the refusal count, not the degraded count, so several records per re-walk break no consumer.
  - `test_a_pinned_missing_profile_refuses_and_an_auto_one_re_walks` pins `[d.site ...] == ["lanes.model_profile"]`, and that scenario still produces one record.
  - No test or doc outside the diff matches the old "runs X" string.
- **Config.json still wins on the lane path.** `_project_lane_env_for_build` projects `llm.profile` into the env before `detect_tiers`, so the env layer outranks the preference, as the changelog says.
- `_maybe_pin_pre_upgrade_profile` now sees no env on a GPU-less robot, but it only writes the marker without a CUDA GPU of 4 GB or more, so no pin results. `_restore_persisted_profile` now restores a persisted `--llm` choice on the robot, where smollm used to shadow it. That is the operator's model, so it is correct.

## Findings

### W1: SHOULD-FIX. On a GPU-less robot, the LLM the robot actually serves is no longer smollm but the llm.json / built-in default (mistral-7b), and the V30 preference never reaches it

`embodied_runtime/agentic_runtime.py::AgenticRuntimeMixin._apply_llm_backend_fallback` → `_start_agentic_runtime` → `runtime/lane_backends.py::build_run_router` (its `load_llm_config(profile_override=profile)` fallback) + `models/language/config.py::load_llm_config`.

How it happens:
1. On a box with no GPU, `detect_tiers` creates no `large` lane, and `_add_remote_large_placeholder` adds one only with a peer URL.
2. So `build_primary_router` returns `manager.get_backend("large") = None`, and `build_run_router` builds the robot's router (the `LLMWorker`'s, shared with ExecAgent through `share_run_llm`) from `load_llm_config(profile_override=None)`.
3. That function reads `MAXIM_LLM_PROFILE`, else llm.json's `profile`, else `LLMConfig.profile = "mistral-7b-instruct-v0.2"`.
4. Until this fold, the fallback's env write is what made that router smollm. The new preference feeds only `detect_tiers`' **medium** lane. The robot runtime never serves the medium lane (it uses only the primary router, and nothing in embodied_runtime/ calls `get_backend_for_function` or `get_backend("medium")`).

Probe (`wire_probes/robot_build.py`: real `_apply_llm_backend_fallback` plus real `build_run_router`, no GPU, 16 GB, offline; `old` = the 8a8e4f34 env write):

| models on disk | before (8a8e4f34) | after (this fold) |
|---|---|---|
| smollm | primary serves smollm | **primary None** (mistral-7b checked, `unusable_models={'mistral-7b-instruct-v0.2': 'not_downloaded'}`); medium lane = smollm, unused |
| smollm + mistral | smollm | **mistral-7b-instruct-v0.2** on CPU (`n_gpu_layers 0`) |
| phi-2 only | named `OfflineRefusal` (smollm) | primary None; degraded record names only the small lane |

Failure scenarios:
- **Online or offline, the default setup.** `download.DEFAULT_LLM` is smollm, so this is a GPU-less robot with smollm downloaded. The robot now runs with **no LLM**. The fix was meant to keep the default it already had, and it removes it. Off the main thread nothing is asked, so the run is silently template-only.
- **mistral also on disk.** The robot loads a 7B model on CPU. That is exactly what `_apply_llm_backend_fallback`'s docstring says the fallback exists to prevent ("so the runtime doesn't try to load a model it can't run"). On a small host (Pi-class RAM) it may not fit at all.
- **Offline, a model on disk but not mistral.** No refusal and no LLM, and the `offline` block does not say the run has no LLM (the primary's refusal goes only to `unusable_models`).

The changelog's robot sentence ("offline a missing default falls back to a model on disk") is false for the model the robot serves.

`test_a_gpu_less_offline_robot_run_serves_a_model_on_disk` drives `_ensure_lane_profiles_available` plus `build_profile(None)` on the medium lane. It never drives `build_run_router`, which round 7 asked for, so it passes while the robot serves nothing. This is the "shipped the pieces, not the composition" pattern again.

Root cause: on a GPU-less box the lane system has no lane that is the primary, and the primary's fallback (`load_llm_config`) has no hook other than the env.

Proper scope (owner decision, no band-aid): one of these.
- Make the lane system produce the primary on a GPU-less box. For example, `build_run_router`/`build_primary_router` serve the medium lane, or `detect_tiers` emits the CPU pick as the primary lane.
- Or have `build_run_router`'s fallback take the preferred/re-walked profile (`profile_override=` the medium lane's profile, which still respects the env and config layers) instead of a bare `load_llm_config()`.

Either way, the composed guard must call `build_run_router` with no GPU and assert that the served `router.cfg.profile` is smollm when smollm is on disk, and is the model on disk when smollm is missing offline. Prove it by deleting the mechanism.

### W2: SHOULD-FIX (pre-existing, now on the claim surface). The one re-walk overwrites a lane the operator pinned, and the model is on disk; the changelog says such a profile is "never swapped"

`runtime/lane_backends.py::_ensure_lane_profiles_available`: the `for new_name, new_cfg in new_tiers.items()` replace loop.

The re-walk triggered by one lane's missing model replaces `model_profile` on **every** lane it detects. That includes a lane whose profile came from llm.json `tiers.<lane>.model_profile` (which `_profile_is_pinned` calls pinned) and is on disk.

Probe (`wire_probes/rewalk_pinned.py`): small = smollm (missing), medium = phi-2 (on disk, pinned via a tier override), offline. Result:
```
{'small': 'smollm-1.7b-instruct', 'medium': 'phi-3-mini-4k-instruct'}
['smollm-1.7b-instruct is not on disk; lane small has no model on disk', 'the re-walk replaced phi-2; lane medium runs phi-3-mini-4k-instruct']
```

The swap predates this fold, and it happens online too. The fold's `_record_offline_rewalk` now states it honestly ("the re-walk replaced phi-2"). But the fragment's first sentence, "a profile you chose (... or a tier's `model_profile` in llm.json) is refused by name and never swapped", is contradicted by the code.

Small known fix: in the replace loop, skip a lane whose current profile is pinned (`lane_profile == profile or _profile_is_pinned(...)`), or one that is on disk and is not the lane being re-walked. Add a test where a pinned on-disk tier survives another lane's re-walk. Under "filing is not fixing", fold it. If the owner defers it, file it with a `Deferred because:` reason and soften the changelog sentence.

### N1: NIT. Two wordings for the same degraded state
`lane_backends.py::_ensure_lane_profiles_available` (the two `re_walked or capabilities is None` / re-walk-raised branches) still record "... lane X has no model", while `_record_offline_rewalk` records "... lane X has no model on disk", which is what the notes paragraph quotes. Any consumer or doc grep for one of them misses the other. Use one string.

### N2: NIT. The robot logs that it prefers smollm even when the operator's profile stands, and its docstring still says it switches the profile
`agentic_runtime.py::AgenticRuntimeMixin._apply_llm_backend_fallback` logs "Preferring a CPU-friendly LLM: smollm-1.7b-instruct" unconditionally. The 8a8e4f34 version logged only when the default was adopted, and the CLI side was fixed this round to stay quiet when the operator's profile stands, so the two callers disagree again in the other direction. Given W1, the line is also untrue about what the robot serves. The docstring's "switch the default profile to `smollm-1.7b-instruct`" is stale.

### N3: NIT. The preference is not in the decision record
`runtime/decision_log.py::_capture_env_snapshot` used to show `MAXIM_LLM_PROFILE=smollm-1.7b-instruct` for a GPU-less robot build. Now nothing in `lane_decisions.jsonl` says that the medium lane came from `_preferred_cpu_profile` rather than `_pick_medium_profile`. That is a small provenance gap, given that V27 makes this a claim surface.

### N4: NIT. No autouse reset for `lane_models._preferred_cpu_profile`
It is process state on a hot startup path that is never cleared. Today only the two new tests set it, and both monkeypatch it. A future test that runs the real `_apply_llm_backend_fallback` / `_start_agentic_runtime` without the patch would leak it into every later `detect_tiers` on GPU-less caps, so the medium lane would become smollm instead of the RAM pick. The env value it replaced had the conftest scrub `_isolate_maxim_llm_profile_env`. The same lesson applies (see CLAUDE.md "Opt-in env vars in hot startup paths need autouse scrubs"). Add a reset to that fixture.

## Verdict

FOLD
