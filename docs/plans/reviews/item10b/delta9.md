# #1315 item 10b, delta round 9 (commit 53b143c1): Executor + Wire integrity

I read only the exported tree. Probes ran from review_d9/probes/e2e with PYTHONPATH set to the tree (checked
`maxim.__file__`), a scratch HOME/XDG/MAXIM_DATA_HOME and MAXIM_OFFLINE=1. Nothing was downloaded and no sim ran.

## Claims

1. **The marker is gone. HOLDS.** No `_MAXIMS_OWN_PROFILES`, `mark_profile_maxims_own` or `maxims_own` remains in
   src/, scripts/ or tests/. The only mentions left are the history in docs/plans (approach notes §6 and the
   reviews/item10b files).
2. **The fallbacks. HOLDS.**
   - `cli_utils.configure_cpu_fallback_model` now sets `N_GPU_LAYERS=0` with setdefault first, then returns early when
     `MAXIM_LLM_PROFILE` is present. When the operator's profile is set, its warning and stderr print are now skipped.
     That is the intended change. Before the delta the print claimed smollm while phi-2 ran.
   - `AgenticRuntimeMixin._apply_llm_backend_fallback`: the old `setdefault` lines were the last statements in the
     method, and only the `log.info("Using CPU-friendly LLM…")` line followed them. The early return skips only that
     line, so nothing else is lost.
   - When `MAXIM_LLM_PROFILE` is present, both versions leave the env exactly as before. A blank value also counts as
     present, as it did before.
   - When both run in one process (the CLI agentic path), the second fallback now stays silent because the first
     already announced.
3. **The re-walk record. HOLDS.** At `lane_backends._ensure_lane_profiles_available`, `runs` requires a truthy
   substitute that differs from `lane_profile` and passes `_profile_has_local_file`. The two clearing branches use the
   same string, "has no model on disk". The positive branch ("runs smollm") is still pinned by
   `test_a_pinned_missing_profile_refuses_and_an_auto_one_re_walks`.
4. **The tests. HOLDS.**
   - `_clear_env` records a restore for each variable.
   - Each test has a deletion that it catches. Reverting the robot path to `setdefault` trips the caplog assertion.
     Restoring the unconditional CLI warning trips the capsys and caplog assertions. Restoring the marker flips both
     `_profile_is_pinned` asserts. Reverting the record expression gives two reasons in the set.
   - `test_offline_call_sites_10b.py`, `test_aut_router_profile_1538.py` and `test_talk_never_asks_1658.py`: 97
     passed.
5. **The changelog. PARTLY.** The wording is too broad. See finding S1.

## Fail-closed end to end (robot path): HOLDS, by probe

I ran `_apply_llm_backend_fallback` with no GPU and MAXIM_LLM_PROFILE unset. It set the env to smollm. Then, offline,
`build_run_router(capabilities=RuntimeCapabilities(has_gpu=False, ram_gb=16))` ran on an empty data home:

- It raised `maxim._offline.OfflineRefusal: models.download.ensure_available refused: smollm-1.7b-instruct is not on
  disk`.
- `warn_unless_refusal`, the robot runtime's catch-all, re-raised it.
- The degraded list was empty.

The refusal fires in the preflight. `detect_tiers` gives the small lane smollm, and the medium lane takes smollm from
the env. Both read as pinned (env value == lane profile), so `ensure_available` refuses before the `load_llm_config`
fallback is reached. The fallback leg would refuse the same way: `consented_router` → `router_model_problem` →
`ensure_available`. smollm cannot be in `declined` offline, because a pinned profile raises instead of returning
False.

Upstream callers re-raise as well: `media_loop` (`warn_unless_refusal` around `_start_agentic_runtime`) and
`_start_agentic_runtime`'s own LLM block. The robot therefore stops typed and by name, not silently with no LLM.

## Consumers of the changed string

- Only `tests/unit/test_offline_call_sites_10b.py` matches the exact text ("has no model on disk"). The other
  test at line 434 checks only the `site` and a "smollm" substring.
- No `src/`, `scripts/` or `docs/` code parses the reason. `report.json` and `lane_decisions.jsonl` carry it verbatim.
- `test_talk_never_asks_1658.py:249` has "has no model", but it is prose in a docstring.

## Silent swallows

None added. The `except Exception: pass` in `configure_cpu_fallback_model` predates this delta and its lines are
untouched.

## Findings

**S1 SHOULD-FIX: `changelog.d/1315-item10b-offline-call-sites.added.md` says the default applies "Without a GPU",
but only robot/agentic runs set it.**
- Both fallbacks run only on the robot path: `cli.py::_main_impl` in `mode == "agentic"`, and
  `AgenticRuntimeMixin._start_agentic_runtime`.
- A GPU-less `maxim --sim` never exports smollm. Its small lane is tier-detected, so offline a missing smollm
  re-walks and is recorded as degraded. It is not refused.
- Failure scenario: an operator reads the changelog, runs an offline sim on a CPU box without smollm, and expects a
  refusal by name. They get a degraded run instead. The changelog makes the claim in the wrong scope (V27: the
  record is a claim).
- Fix: "On a robot (agentic) run without a GPU, Maxim's default model … is set for the session like a profile you
  chose …".

**N1 NIT (predates the delta, behaviour unchanged): neither fallback consults `llm.profile` in config.json.**
- At `cli_utils.configure_cpu_fallback_model` and `AgenticRuntimeMixin._apply_llm_backend_fallback`, the check
  `"MAXIM_LLM_PROFILE" in os.environ` sees only the env layer. `resolve_setting` ranks env above config, so the
  exported smollm overrides an operator's `maxim config set llm.profile phi-2`.
- Online it serves smollm, which it did before this delta. Offline with smollm missing, it now refuses naming smollm
  even though the operator's phi-2 is on disk.
- That is fail-closed and by name, but the refusal names a model the operator never chose, and the comment "the
  operator's profile stands" reads more broadly than the code.
- This is #1737 territory (where the GPU-less default comes from). It would be worth one line in #1737.

**N2 NIT: the network guard reported "blocked 2 outbound call(s) (dns)"** across the three test files I ran.
I did not find out which test caused it. No test in the delta touches the network on its face, so this probably
predates the delta. It is worth one narrow look in the 10c pass.

## Verdict

FOLD: one SHOULD-FIX (S1, a one-phrase changelog edit, docs-only). After that fold, no further round is owed: it
touches no `src/`.
