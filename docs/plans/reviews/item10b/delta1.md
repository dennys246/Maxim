# #1315 item 10b — delta round 1 (fold b6015366..38a872c6)

Reader: a different reader, delta only. Probes ran in a copied tree
(`scratchpad/review10b/d1/tree`, no `.git`). `maxim.__file__` was checked to resolve to the copy, and HOME, XDG and
MAXIM_DATA_HOME pointed at scratch dirs. Every probe ran as a pytest test with the latch engaged and the repo's
network guard active. No online path was called.

The probe files are `tests/unit/test_zz_delta1_probes.py` and `tests/unit/test_zz_delta1_probes2.py` in the copy.

Targeted tests on the copy, all passing:
- `test_offline_call_sites_10b.py`, `test_offline_latch_unit.py`, the benchmark tests (5 files), `test_decision_log.py`
  and the robot doctor tests: 273 passed.
- `test_sandbox.py`, `test_docker_sandbox.py`, `test_offline_startup_10a.py`, the llama tests, the doctor tests and the
  1343 tests: 358 passed, 1 skipped, 1 xfailed.

Lints, run read-only in the worktree with PYTHONDONTWRITEBYTECODE: `lint_function_length` (touch rule included),
`lint_no_silent_swallows` and `lint_http_surface` are clean, and ruff is clean on the touched src files.

## Round-1 findings: status

| Finding | Status |
|---|---|
| Exec S1 / Wire F5: an optional pose weight refuses the whole YOLO engine | **Folded correctly.** Offline, a missing pose weight sets `_pose_load_error` and calls `mark_degraded`. `load_pose=True` still refuses. `_try_load_pose` short-circuits. Tested. |
| Exec S2 / Arch S3 / Wire F3: whisper relabels every failure as "not cached locally" | **Folded correctly.** It refuses only when `is_hub_local_miss(e)`, and re-raises everything else. The encoder and semantic reasons go through `load_failure_reason`. Verified on the real sentence-transformers 5.4: the chain is `OSError <- LocalEntryNotFoundError`, which classifies as a miss and reads "not cached locally". |
| Exec S3 / Arch S4: the doctor demoted unrelated list rows | **Folded, with a new defect.** The rows now stand and one info row is appended, and `_check_one_robot` is wrapped. The per-robot wrap loses each robot's identity (see N3). |
| Exec S4 / Wire F2: lane preflight aborts instead of re-walking | **Folded per V25, incomplete.** Pin detection misses one operator source and misreads one automatic source (N4). The comment above the call in `build_primary_router` ("offline + non-tty … this is a no-op") is still stale (NIT a). |
| Exec S5 / Wire F4: llama fetch swallows the refusal and re-books it on every call | **Folded, and it regressed a neighbour.** `complete()` is now typed and booked once. But `backend.warmup()` now raises inside the router's warmup thread, which has no handler there (N1). Wire F4's "move the refusal after the registry check" was not taken: a custom profile gets the typed refusal. I accept that. |
| Exec S6: the sentence-transformers floor | **Folded** (signature-gated kwarg; see decision 3). |
| Exec S7 / Arch S5: the benchmark aborts the run on one refusal | **Folded per V26 inside `BenchmarkRunner`, incomplete at its consumers.** The CLI exits 1 and `api.benchmark` scores the refused model 0.0 (N2). |
| Arch S1 / N4 / Wire N6: the duplicate CI grep | **Folded correctly.** The grep step and its string test are removed, and the notes credit `lint_http_surface` (which ran clean). |
| Arch S2 / Wire F6: `degraded` has no reader | **Folded correctly** (V27). `run_provenance` gets `_offline_stamp`, built after the preflight's degraded marks, at teardown. `LaneDecisionRecord` gets defaulted `offline` and `offline_degraded`, and `build_record` runs after `_ensure_lane_profiles_available`. Tested. |
| Wire F1: an offline sim runs with no sandbox | **Folded correctly.** `check_docker_available` returns False offline without probing. `auto` falls back to tmpdir and is marked degraded. `docker` refuses. `_setup_sim_sandbox` re-raises. The online swallow is deferred to #1596. |
| Arch N1: one helper instead of ten copies | Folded (`refuse_if_offline`). |
| Exec NIT 1 / Wire N1 (rtm pose), Exec NIT 2, Arch N5, Wire N3 (swallows) | Deferred, with a record (the notes §6 10c list names them). |
| Arch N3 / Wire N2 (transformers silently forces `trust_remote_code` off), Exec NIT 3 / Wire N4 (download CLI exits 1), Wire N5 (unbounded `mark_degraded`) | Not addressed and not recorded. These are NITs, so they do not block. The report dedupes degraded entries, so N5 matters only in memory. |

## Folder decisions taken without the owner

1. **Doctor (append a row, wrap `_check_one_robot`): sound in principle, with a defect in the wrap.**
   - The question was whether a list check's real `fail` row still makes `maxim doctor` exit 1 offline when the latch
     caused it. In today's list checks, no.
     - The probed `check_mesh_nodes` offline gives a node row with status **warn**: `other (…): OfflineRefusal …`.
       `_worst_status` is `warn`, so the exit code is 0.
     - The `fail` rows in `check_resolved_config` and `check_env_config` are config defects, and those are genuine.
     - Robot rows are marked per item.
   - So exiting 1 on a list `fail` row is acceptable, and fails toward honesty.
   - The defect is N3: each robot loses its row name.
   - Small: the mesh per-node probe (`_probe_mesh_node_to_check`) is not wrapped, so its rows stay `warn` with a
     self-describing OfflineRefusal message, and `--retry` re-probes them (NIT b).
2. **`_profile_is_pinned`: should change, and needs the owner.** Lane env overrides are remote-only, and placement
   carries cloud or remote, so neither matters here. Two sources are wrong (N4):
   - an `llm.json` `tiers.<lane>.model_profile` override (operator-chosen) reads as auto-detected, so it gets swapped;
   - `configure_cpu_fallback_model`'s `setdefault("MAXIM_LLM_PROFILE", "smollm-1.7b-instruct")` (automatic) reads as
     pinned, so it refuses.
3. **`local_files_only_kwargs(loader)`: sound.**
   - Verified: 5.4 accepts the kwarg. A `**kwargs` loader also passes, and a non-introspectable one passes.
   - Caveat, not verified because 2.2 is not installed: at the 2.2.x floor, sentence-transformers calls its own
     `snapshot_download`, which reaches the hub even for a cached model. The latch then refuses it. The result is a
     degraded hash fallback with the reason "failed to load (OfflineRefusal)", even when the model is on disk. That
     fails closed, so it is fine.
   - Raising the floor would be cleaner. That is a NIT.
4. **`is_hub_local_miss` by class name along the chain: sound.** Verified on the real sentence-transformers chain.
   - Hazard: because the walk follows `__context__`, a non-miss error that a library raises while handling a miss reads
     as a miss. That is a NIT, and following `__cause__` only would be tighter.
5. **The grep dropped in favour of `lint_http_surface`: sound** (ran clean: 11 raw dials in 9 files).
6. **The report lists the first 50 refusals plus the exact count, and dedupes degraded entries by (site, reason):
   sound.** `PROVENANCE_SCHEMA` was not bumped. The key is additive and no consumer checks exact keys, so that is fine.
7. **Benchmark (refused models excluded from both rankings, defaulted `refused` fields, "refused" in the JSON): sound
   inside the runner, but incomplete at its consumers** (N2).
   - `_load_baseline` does not read `refused` back. That is harmless, because a refused model has no metrics to
     compare.
   - A model with some runs refused and some errored gets `refused=None` and shows as "every run errored". That is a
     NIT.
8. **`_setup_sim_sandbox` re-raises `SECURITY_REFUSALS`: sound.** The tmpdir fallback for `auto` means only an
   explicit `--sandbox docker` reaches the re-raise.

## New findings

**N1 SHOULD-FIX: the fold regressed the router's warmup. Offline, the warmup thread dies on the remembered refusal,
so `wait_ready` stalls for its full timeout.**
- Where: `models/language/router.py::LLMRouter.warmup` (`_warmup_thread`), reached through
  `llama_backend.py::_LlamaCppBackend.warmup` → `_ensure` → `_fetch_missing_model`.
- Scenario:
  1. Offline, an in-process llama lane has a GGUF that is not on disk and that the preflight did not cover: a custom
     profile not in `LLM_MODELS`, a bare router, or a lane whose profile the preflight cleared.
  2. Before the fold, `warmup()` returned False and `_ready_event` was set.
  3. Now `warmup()` raises `OfflineRefusal`. The `backend.warmup()` branch of `_warmup_thread` has no try, so the
     thread dies through `threading.excepthook` and prints a traceback. `_ready_event.set()` never runs.
  4. Every `wait_ready` then blocks until it times out: `orchestrator.py` waits 120 s, twice, and ignores the result;
     `cli.py` waits 120 s or 180 s; `research_orchestrator` waits 120 s. Only after that does `complete()` raise the
     typed refusal.
- Verified:
  - Probe p3: `_LlamaCppBackend.warmup()` with a custom profile raised `OfflineRefusal … my-custom-profile is not on
    disk`.
  - Probe p4: `LLMRouter.warmup()` with a backend whose `warmup` raises gave `READY_SET False THREAD_EXC
    ['OfflineRefusal']`.
- Fix: in `_warmup_thread`, catch `SECURITY_REFUSALS` around `backend.warmup()` (set `_warmup_failed` and log
  "refused (offline)"), and put `self._ready_event.set()` in a `finally`. Alternatively, have
  `_LlamaCppBackend.warmup` return False on the remembered refusal, so that `complete()` still raises. Add a test
  with a raising backend.
- Also (NIT): `raise self._security_refusal` re-raises the same instance on every call, so its `__traceback__` grows
  by a few frames per `complete()`. Raise a fresh `OfflineRefusal(str(r))` chained with `from r`, without booking it.

**N2 SHOULD-FIX: V26 holds inside `BenchmarkRunner`, but its two consumers still treat a refused model as failed or
scored.**
- Where: `cli.py` at both benchmark sites (`all_passed = all(mr.passed …); sys.exit(0 if all_passed else 1)`), and
  `api.py::benchmark` (`scores[model] = {"overall": model_result.score, …}`).
- Scenario:
  - `maxim benchmark --models <cloud>` offline: every run is refused, the report says REFUSED (not scored), and the
    process exits **1**, the "failed" code. Before the fold it exited 2 through the refusal mapping. The refusal exit
    is 2 everywhere else (the robot run, server mode, `_run_stopped_on`).
  - `maxim.benchmark([...])` (the public API) returns `scores["claude"] = {"overall": 0.0}` for a refused model, so a
    caller ranks it last. That is "scored", against V26.
- Verified:
  - Probe p6: an all-refused `_aggregate_runs` gives `passed=False score=0.0`, so the CLI exits 1.
  - The `api.py` lines were checked by reading.
- Fix:
  - CLI: exclude refused models from `all_passed`, and exit 2 when any model was refused (or when all were; owner
    call, strict = any).
  - API: omit refused models from `scores`, or carry `refused` in `BenchmarkResult`.
  - Pin both with tests.

**N3 SHOULD-FIX (low): offline, each per-robot doctor row loses its identity, and a duplicate row is appended.**
- Where: `doctor/checks.py::_refused_offline_row`, as applied to `_check_one_robot`.
- Scenario:
  - `OfflineRefusal` is a `RuntimeError`, so `_check_one_robot`'s `except OSError` does not catch the refused
    `getaddrinfo`/`connect`. The per-item wrapper then builds `refused(e)` with
    `name = "_check_one_robot".removeprefix("check_").replace("_", " ")`, which gives `" check one robot"`. The row
    has no `retry_id`.
  - The outer `check_robot_reachable` wrapper then sees the ledger delta and appends a third row, "robot reachable".
  - With two robots, the operator sees two identical rows named " check one robot" and cannot tell which robot is
    which. That contradicts the fold's claim that "each robot marks itself".
- Verified: probe p7 printed `[(' check one robot','info',None,…), (' check one robot','info',None,…), ('robot
  reachable','info',None,…)]`.
- Fix:
  - Inside `_check_one_robot`, catch `_offline.OfflineRefusal` and return
    `CheckResult(name, "info", f"refused (offline): {e}", retry_id="robot")`, or let the wrapper take a row name.
  - Have the outer wrapper skip its appended row when every row already carries "refused (offline)".

**N4 SHOULD-FIX (owner call): V25's pin test misreads two sources.**
- Where: `runtime/lane_backends.py::_profile_is_pinned`.
- (a) An operator-chosen per-tier profile is treated as auto-detected and swapped. `llm.json` `tiers.<lane>.model_profile`,
  applied by `apply_tier_config_overrides`, is an operator's choice. `_profile_is_pinned` sees only `llm.profile`, so
  offline that profile is re-walked to another model, and only a degraded record says so. V25's wording, "a profile
  you chose", covers it, but its listed examples (`--llm`, `llm.profile`) do not.
- (b) An automatic choice is treated as pinned and refused. `cli.py` agentic mode on a GPU-less box calls
  `configure_cpu_fallback_model`, which does `os.environ.setdefault("MAXIM_LLM_PROFILE", "smollm-1.7b-instruct")`.
  `resolve_setting` then reports `source="env"`, so the profile counts as pinned. Offline with that model missing and a
  smaller one on disk, the run refuses instead of re-walking.
  - Probe p2 printed `PINNED True`.
  - This case fails closed, which is the strict side.
- Fix: decide whether tier overrides pin (strict: yes; carry a `pinned` flag on the lane config from the override
  stage). Make the CPU fallback not masquerade as an operator pin, either by not exporting it through
  `MAXIM_LLM_PROFILE` or by marking it. Or record in V25 that only `--llm` and `llm.profile` pin.

**NITs**
- a. `lane_backends.py::build_primary_router`: the comment above `_ensure_lane_profiles_available` ("If the user is
  offline + non-tty … this is a no-op") is stale under V25. Exec S4 asked for it to be updated.
- b. `_probe_mesh_node_to_check` (direct, and through the `--retry` reprobe) is not wrapped. Offline, its rows stay
  `warn` "other … OfflineRefusal", and the retry loop re-probes them.
- c. `decision_log`: `asdict` now writes `offline: false, offline_degraded: []` on every ONLINE record too, so an older
  `read_last_record` returns None for every new record, online included. This is the documented pre-existing pattern,
  but "empty online" holds only for the report stamp, not here.
- d. `is_hub_local_miss` follows `__context__` (decision 4).
- e. Benchmark: a model with some runs refused and some errored is shown as "every run errored", and `write_paper`
  still receives refused models.

## Verdict

**NOT STOP.** BLOCKER 0 · SHOULD-FIX 4 (N1–N4; N4 is an owner call) · NIT 5 (plus the unrecorded round-1 NITs above).
Fold N1 to N3 and decide N4. A follow-up delta round then covers only `router.py`'s warmup, the two benchmark
consumers, the robot row, and whatever N4 decides.
