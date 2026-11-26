# Delta round 8, Executor lens: #1315 item 10b (PR #1735), staged fold on 8a8e4f34 (owner decision V30)

Reader: a different reader from the author. Tree: review_d8/tree (exported). Probes are in review_d8/exec_probes (`robot_router_probe.py`, `rewalk_record_probe.py`, `old_loop/` = change 2 reverted, `dp1`..`dp4` = deletion copies, `base` = unmodified copy). Every probe ran with MAXIM_OFFLINE=1 (except the online cases, which drive `_offline` directly) and a scratch HOME/XDG/MAXIM_DATA_HOME.

## What checks out

- **config.json `llm.profile` wins on the robot's lane path.** `build_primary_router` calls `_project_lane_env_for_build` (the `_llm_env_map` projection of `llm.profile` into `MAXIM_LLM_PROFILE`, plus `_restore_persisted_profile`) before `detect_tiers`. So `env_profile` is set and beats `_preferred_cpu_pick`. The `load_llm_config` fallback also reads the projected env. Round 7's S2 is closed on the robot path.
- **The preference only applies after the availability check.** The first `detect_tiers` in `build_primary_router` passes `profile_available=_profile_has_local_file`, and the re-walk passes it with the excluded profile removed. Both run `_preferred_cpu_pick` through the check.
- **`_profile_is_pinned`.** Removing the `_MAXIMS_OWN_PROFILES` exemption is correct now that the robot no longer writes the env. The CLI's env write reads as a pin, as V30 intends.
- **#1538 registry.** `configure_cpu_fallback_model` is registered, and the scan still catches a reintroduced robot-side `setdefault` (dp4).
- **Test env hygiene.** `_clear_env` (setenv then delenv) records the restore, so round 7's S3 leak is closed.
- **Deletion probes.** Every guard bites on the unit it covers:
  - dp1 (change 2 reverted): the composed test FAILS.
  - dp2 (`_preferred_cpu_pick` dropped from `detect_tiers`): the preference test FAILS.
  - dp3 (the old one-line record restored): the composed test FAILS.
  - dp4 (the robot writes `MAXIM_LLM_PROFILE` again): 3 tests FAIL, including the #1538 writer scan.
  - base: all 7 targeted tests pass. The 4 related files pass 169/170; the one failure is my copy lacking `scripts/`, not the code.
- **Lints.** `ruff check` and `ruff format --check` are clean on the 6 files. `lint_no_silent_swallows` is clean, and the delta adds no handler.
- **Process-global `_preferred_cpu_profile`.** It is no wider than the env value it replaces: it touches only the GPU-less medium lane, and only in a process whose robot runtime ran the fallback. The new tests monkeypatch it, so they restore it.

## Findings

### S1: SHOULD-FIX. On a GPU-less box the robot's LLM is not the lane, so the preference never reaches it, and the robot now loads `mistral-7b-instruct-v0.2`

Where: `embodied_runtime/agentic_runtime.py::AgenticRuntimeMixin._apply_llm_backend_fallback`, `runtime/lane_backends.py::build_run_router`, `runtime/lane_backends.py::_primary_router_and_manager`, `models/language/config.py::load_llm_config`.

How the robot gets its LLM:
- `_start_agentic_runtime` takes its LLM from `build_run_router`.
- `build_run_router`'s router is `manager.get_backend("large")`. On a GPU-less box `detect_tiers` makes no large lane, so that router is `None`.
- `build_run_router` then takes its fallback, `load_llm_config(profile_override=None)`, which reads `MAXIM_LLM_PROFILE`, then llm.json's `profile`, then `LLMConfig.profile = "mistral-7b-instruct-v0.2"`.
- Before this delta (and before #1602), the env value made that fallback load smollm. V30 removed the env write, and the preference steers only the medium lane, which the robot's primary LLM never uses.

Probe (`exec_probes/robot_router_probe.py`: GPU-less, 16 GB, smollm on disk, LLM enabled, no llm.json):
```
env MAXIM_LLM_PROFILE: None
lanes: {'small': 'smollm-1.7b-instruct', 'medium': 'smollm-1.7b-instruct'}
primary router: None | fallback load_llm_config profile: mistral-7b-instruct-v0.2
```

Failure scenarios:
- **Online.** A GPU-less robot with smollm on disk now builds a 7B model for CPU inference. If that model is missing, the robot asks to download 4.5 GB or runs with no LLM. The "Preferring a CPU-friendly LLM: smollm" log line is false.
- **Offline.** The robot does not fall back to a model on disk. The fallback `consented_router` checks mistral, finds it missing and refuses, so the robot runs with no LLM. The `offline` record lists lane re-walks for lanes that serve nothing.

V30's headline claim ("a GPU-less offline robot run ... falls back to a model on disk") therefore does not hold end to end, and the online behaviour regresses.

The new "composed" test (`test_a_gpu_less_offline_robot_run_serves_a_model_on_disk`) misses this for two reasons:
- It stops at `_ensure_lane_profiles_available` plus `build_profile(None) == ""` and never drives `build_run_router`.
- It calls `detect_tiers(caps)` without `profile_available`, so it builds a lane state (medium = the missing smollm) that the real build never produces. The real first walk already gives medium a model on disk, and the re-walk is triggered by the small lane.

Root cause and scope (an owner decision under "no band-aid fixes"): the V30 preference lives in a component (the medium lane) that is not the robot's LLM on the one hardware class V30 targets. Two options:
- (a) `build_run_router`'s no-large-lane fallback takes the medium lane's (re-walked) profile when no profile is named.
- (b) The GPU-less primary router comes from the medium lane.

Either way, the guard must be a test that drives `build_run_router` (stubbing only the file check, the consent and the spawn), as the probe does, and asserts which profile is served online and offline. Until then, the changelog line must not claim the fallback.

### S2: SHOULD-FIX. Change 2 lets the re-walk silently swap an operator's llm.json tier pin it has not yet checked: no refusal offline (V25), no download prompt online

Where: `runtime/lane_backends.py::_ensure_lane_profiles_available`.

The one re-walk replaces `model_profile` on every lane `detect_tiers` returns, including a lane whose profile came from llm.json `tiers.<lane>.model_profile`. `detect_tiers` does not know that pin, so it overwrites it with an auto pick. With the snapshot loop, the pinned lane was then still judged by its own pinned profile. Now it is judged by the re-walked one, which is on disk, so the loop passes it.

Probe G/H (`rewalk_record_probe.py`, new tree against `old_loop/`): GPU box, llm.json `tiers.large.model_profile = mistral-7b` (pinned, missing), qwen-14b on disk, smollm missing.
```
NEW offline: large -> qwen2.5-14b-instruct; asked=[]; large_declined=set()
OLD offline: large -> None; asked=['mistral-7b-instruct-v0.2'] (ensure_available raises OfflineRefusal by name in reality)
NEW online (declines): asked=['smollm'] only; large runs qwen; large_declined_out empty
OLD online (declines): asked=['smollm','mistral']; large_declined_out={'mistral'}
```

Failure scenarios:
- **Offline.** The operator's pinned model is swapped for another, a V25 violation ("a profile you chose ... a tier's `model_profile` in llm.json is refused by name and never swapped").
- **Online.** The operator is never offered their pinned model, and `large_declined_out` / `unusable_models` lose it.

The trigger is common: the small lane is hard-coded to smollm, so any box without smollm on disk spends the re-walk first.

Probe D shows the same root cause for a pinned lane that IS on disk (llm.json large = phi-3 on disk, smollm missing). The re-walk replaces phi-3 with qwen, and the new record states it openly: "the re-walk replaced phi-3-mini-4k-instruct; lane large runs qwen2.5-14b-instruct". That part is pre-existing (the old loop also replaced it), but change 2 turned the missing-pin case from a refusal into a swap.

Proper fix: the re-walk replaces only lanes not yet judged, or whose current profile is missing, and never a pinned lane (`_profile_is_pinned` or the build's `profile`). Add a test with an llm.json tier pin plus a missing smollm, online and offline.

### S3: SHOULD-FIX. `_record_offline_rewalk` records "runs X" for a substitute that is not on disk

Where: `runtime/lane_backends.py::_record_offline_rewalk`.

The `ran` test treats only `now == was` (or no profile) as "has no model". It never checks whether `now` is on disk, and the re-walk can hand a lane a profile that skips the availability check:
- the small lane's hard-coded smollm,
- `_pick_infer_profile_with_ctx`'s universal fallback `_INFER_VRAM_TIERS[-1]`, returned regardless of `profile_available`,
- `env_profile`.

Probe I (offline, GPU, llm.json large = mistral pinned and missing, nothing on disk):
```
degraded: ['smollm-1.7b-instruct is not on disk; lane small has no model on disk',
           'mistral-7b-instruct-v0.2 is not on disk; lane large runs smollm-1.7b-instruct',   <- false
           'smollm-1.7b-instruct is not on disk; lane large has no model']
```
The report's `offline` block (a V27 claim surface) holds a false and self-contradicting pair. It is the same class round 7 flagged ("a false 'lane small runs smollm'").

Fix: `ran = f"runs {now}" if now and now != was and _profile_has_local_file(now) else "has no model on disk"`, plus a test case. S2's fix removes this trigger, but not the other unchecked sources listed above.

### S4: SHOULD-FIX, unless #1737 explicitly carries it. Round 7's S2 is still live on the CLI path

Where: `cli_utils.py::configure_cpu_fallback_model`.

The CLI agentic path runs this before any config projection (`cli.py`, `mode == "agentic"` → `_configure_cpu_fallback_model`). It still guards only on `"MAXIM_LLM_PROFILE" in os.environ`, so an operator's `maxim config set llm.profile phi-2` on a GPU-less box is overwritten by the smollm env write, which `resolve_setting` ranks above config.json.
- **Online.** The operator's model is swapped for smollm.
- **Offline.** smollm now reads as the pin and refuses by name, although the operator's phi-2 may be on disk.

The fold notes scope round 7's S2 fix to the robot ("(S2 there)"). The CLI remains, and the changelog's V25 sentence ("`llm.profile` ... never swapped") is untrue there.

The small fix from round 7 still applies: write only when `resolve_setting("llm.profile")[1] == "default"`, with a config.json test. If it is deferred instead, #1737 must name it with a `Deferred because:` reason.

### N1: NIT. The robot log line no longer tells whether the default applies
`agentic_runtime.py::AgenticRuntimeMixin._apply_llm_backend_fallback` logs "Preferring a CPU-friendly LLM: smollm" even when an env or config profile stands, which round 7 counted as an improvement. The CLI now gates its message on that, so the two callers disagree again. Log it only when no profile is named, or say "unless a profile is set".

### N2: NIT. No autouse reset for the new process global
`lane_models.py::_preferred_cpu_profile` has no reset fixture in tests/conftest.py, while `MAXIM_LLM_PROFILE`, which it replaces, has one. CI's Linux runners have no GPU, so any future test that reaches `_start_agentic_runtime` / `_apply_llm_backend_fallback` unpatched would make every later GPU-less `detect_tiers` test in that worker pick smollm, an order-dependent failure. Add an autouse `monkeypatch.setattr(lane_models, "_preferred_cpu_profile", None)`.

### N3: NIT. Lanes the re-walk adds are never checked or recorded
`_ensure_lane_profiles_available` iterates `list(out)`, so a lane the re-walk adds (`existing is None`) is never visited. `_record_offline_rewalk` also skips it (`not was`). Its profile can be missing yet go unchecked and unrecorded. This is pre-existing and rare, since `detect_tiers` with the same caps returns the same lane set. Visit added lanes, or record them as "added".

### N4: NIT. A lane left on a missing model is recorded but not cleared
The trigger small lane left on its missing smollm is recorded as "has no model on disk", yet `out["small"].model_profile` stays `smollm-1.7b-instruct`, so a later `get_backend("small")` still tries to load it. The record is honest, but the lane config disagrees with it. This is pre-existing, so clear it to `None` as the second-miss branch does.

## Verdict

FOLD
