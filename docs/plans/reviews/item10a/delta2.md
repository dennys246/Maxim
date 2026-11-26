# Delta round 2: #1315 item 10a, the fold of delta round 1

Scope: only what changed in response to delta round 1 (`git diff --cached fb3ae6f3` in `.worktrees/v-item10a`):
`exceptions.py`, `utils/logging.py`, `research_orchestrator.py`, `cli.py`, `doctor/checks.py`, `config_loader.py`,
`reachy_diagnostics.py`, the two docs, the two test files.

How I probed:
- I worked in a copied tree, `scratchpad/review10a/d2copy`, with `PYTHONPATH` set to the copy's `src`
  (`maxim.__file__` checked) and `HOME` set to `scratchpad/review10a/d2home`.
- Deletion probes ran in a further copy, `d2probe`.
- I did not edit the shared worktree, ran no state-changing git, and did not run mypy (the box is shared).
- Lint in the worktree (read-only): `ruff check` on the changed files is clean, and `ruff format --check src/ tests/`
  is clean.
- Targeted tests in the copy: 157 passed in `test_offline_latch`, `test_offline_startup_10a`,
  `test_leader_fail_closed` and `test_bounded_stop_side_1438`. The 4 failures are the known no-`.git` provenance
  scenarios (`ScenarioBroken: … this tree has no .git`), an artifact of the copy.
  `test_offline_module_is_stdlib_only` passes.

Counts: **0 BLOCKER, 0 SHOULD-FIX, 6 NIT.** Verdict: **STOP.**

---

## Delta-1 findings: fold status

| Finding | Status | Evidence |
|---|---|---|
| SF1: shipped callers turn OfflineRefusal into "run degraded" | **Folded, except the orchestrator AUT fallback**, which is deferred with its dependency recorded (decision 1). | `warn_unless_refusal("x %s", OfflineRefusal(...))` now re-raises (probe). The agentic runtime LLM worker and the media loop route through it. `_build_research_llm` re-raises; a deletion probe that reverts it to `except LeaderRefusal` fails `test_the_research_llm_build_does_not_degrade_an_offline_refusal` with the old "will use heuristics" WARNING. A deletion probe on `warn_unless_refusal` fails 2 tests. |
| N1: non-UTF-8 config gives a traceback | Folded | `except (OSError, UnicodeDecodeError)`. A deletion probe fails `test_a_config_that_is_not_utf8_refuses_typed`. |
| N2: `reachy_diagnostics.main` traceback | Folded (prints, returns 2) | A deletion probe fails `test_maxim_diagnostics_engages_first`. |
| N3: doctor rows for self-caught refusals | Folded (refusal-ledger delta) | Real `maxim doctor` offline in the copy: rc 0, no traceback. `server reachable`, `context window`, `vram pressure`, `inference coherence`, `llm server exposure`, `lan access` and `cloudflared` each read `refused (offline): <event>: <detail>`, and the false "auto-spawn will launch one" row is gone. A deletion probe fails `test_a_check_that_catches_the_refusal_still_reads_refused`. Residuals: new NIT-2 and NIT-3. |
| N4: guard gaps | Folded | New tests: `test_maxim_config_set_accepts_llm_offline`, `test_doctor_resolved_config_has_an_llm_offline_row`, `test_download_model_engages_first`, `test_maxim_diagnostics_engages_first`. |
| N5: bare `LLMRouter` | Folded (docs) | The user doc and §6 now say a bare `LLMRouter(...)` does not read the key, and name `MAXIM_OFFLINE=1`. |
| N6: `maxim serve` latches mid-process | Folded (docs) | The user doc says a running serve engages it on its next router build, and that going back online takes a restart. |

## Folder decisions

1. **The AUT fallback is deferred to after #1538. Acceptable, and the §6 claim is now accurate, with one caveat.**
   - **What §6 now says.** The "Open:" sentence states the actual behaviour: an offline sim whose AUT model is not
     in-process falls back to the shared router, and egress stays refused.
   - **The code matches it.** In the worktree, `orchestrator.py` still has the inline `except Exception` → "falling
     back to shared".
   - **The branch matches what was described.** On `fix/1538-aut-router-env-restore`, `_build_aut_router` exists and
     still has `except Exception` with no refusal pass-through, so 10a must add `except SECURITY_REFUSALS: raise`
     there after rebasing.
   - **Caveat 1: no PR for #1538 yet.** `gh pr list --search 1538` returns nothing. The ordering rests only on V22
     coupling 10a to 10b.
   - **Caveat 2: the earlier sentence reads wider than the code.** "…refused by name, never dropped to a fallback
     router" is scoped to `lane_backends`, but it sits two sentences before the Open item.
   - **Recommendation (NIT-6).** Name the AUT re-raise as a merge-checklist item in the 10b PR description. Also add
     a `xfail(strict=True)` red gate now that drives the AUT build with an `OfflineRefusal`. It flips when the raise
     lands, so it cannot be forgotten silently.
2. **The benchmark's per-scenario `except Exception` records the refusal as a failed scenario. Acceptable for 10a,
   but it is a typed abort feeding a score (NIT-5).**
   - It is not silent: a WARNING, `FAIL: <msg>`, and `RunResult.error` are all recorded.
   - However `_aggregate_runs` still produces a `Score:` for a model that never ran, and the refusal is a config
     fact identical for every scenario.
   - The same holds for `LeaderRefusal` today, so it is pre-existing.
   - Fix (10b or a follow-up): `except SECURITY_REFUSALS: raise` ahead of the per-scenario handler.
3. **The CLI top level reports any SECURITY_REFUSALS refusal as "Refusing to run" and exits 2. Sound**
   (`test_cli_main_reports_an_offline_refusal_and_exits_2`; `getattr(refusal, "fix_hint", None)` copes with
   `OfflineRefusal` having no hint). One path does not reach it; see NIT-1.
4. **`SECURITY_REFUSALS` in `exceptions.py` with an import-time lazy import: sound.**
   - `maxim._offline` imports only `os`, `socket`, `sys`, `threading` and `typing`, so there is no cycle.
   - Each of `maxim.exceptions`, `maxim._offline`, `maxim.utils.logging`, `research_orchestrator` and
     `doctor.checks` imports cleanly in a fresh `-I` interpreter.
   - `SECURITY_REFUSALS[1] is maxim._offline.OfflineRefusal` holds, including after `importlib.reload(exceptions)`.
   - `import maxim._offline` alone does not pull in `maxim.exceptions`, and `test_offline_module_is_stdlib_only`
     passes.
   - Importing `_offline` engages nothing, and there is no other import-time side effect.
5. **The doctor ledger check: works for the shipped CLI doctor, but it can misattribute in two cases (NIT-2,
   NIT-3).**
   - The CLI doctor runs its checks serially, and no `check_*` calls another wrapped `check_*`, so there is no
     double counting.
   - The ledger is process-global, unbounded, and only cleared by the test reset, so the `[booked:]` slice is sound.
   - The misattribution cases are listed below.

## Remaining `LeaderRefusal`-only handlers (the grep)

- `cli.py::_run_stopped_on`: `isinstance(error, (BackendError, LeaderRefusal, SpeechOutputUnavailable))` → NIT-1.
- `runtime/agent_pool.py::_generate_llm_response`: `except (ModelError, LeaderRefusal): raise`, then
  `except Exception` re-types to `ModelInferenceError` → NIT-4.
- `models/language/router.py::generate`: `except (LeaderRefusal, _offline.OfflineRefusal)` is equivalent; using
  `SECURITY_REFUSALS` there would be cosmetic.
- `cli.py::_early_leader_boot` (`except LeaderRefusal` → print, return 2; `except Exception` → `user_warn` and
  continue). An `OfflineRefusal` from the leader proxy boot would warn and continue. A leader under offline is
  contradictory, and everything it forwards to is refused anyway, so this is fail-closed. Census class, so 10b.
- Nothing consumes `SECURITY_REFUSALS`-caught exceptions as `LeaderRefusal`: `fix_hint` is only read through
  `getattr`.

---

## New findings

### NIT-1: a robot/agentic run that hits an OfflineRefusal exits 1 as a crash, not 2 as a refusal
- **Where:** `cli.py::_run_stopped_on`.
- **Scenario:** `media_loop` / `agentic_runtime` now re-raise the refusal through `warn_unless_refusal`, the fold
  made this path reachable, and the run loop hands it to `_run_stopped_on`.
- **Verified by probe:** `_run_stopped_on(OfflineRefusal(...))` logs `❌ Maxim stopped: offline …` and returns 1.
  So a supervisor cannot tell it from a crash, which is what #1468's convention exists for.
- **Fail-closed:** the run stops.
- **Fix:** add `*SECURITY_REFUSALS` to the tuple.

### NIT-2: the ledger check misattributes a refusal booked by another thread
- **Where:** `doctor/checks.py::_refused_offline_row`.
- **Scenario:** `maxim.diagnose()` in an embedder process (or, later, a console process) while another thread
  under the latch books a refusal, such as a talk loop's dispatch or a warmup. Whatever local check is running at
  that moment reads `refused (offline): <the other thread's event>`.
- **Verified:** a synthetic probe wrapped a 0.3 s `check_disk_space` while a background thread called
  `_offline.refuse(...)`. Result: `CheckResult(name='slow', status='info', message='refused (offline): bg-thread[http]: …')`.
- **Not reachable from `maxim doctor`,** which is single-threaded.
- **Fix:** book `threading.get_ident()` in `Refusal`, a NamedTuple whose new field could default, and filter on it.
  Or accept it and document "single-threaded attribution".

### NIT-3: on a check that returned normally, the ledger branch discards the check's own row
- **Where:** `doctor/checks.py::_refused_offline_row`.
- **What is lost:**
  - The check's name is replaced by the function-derived lowercase name (`server reachable` for
    `Auto-spawn server`, `lan access`, `cloudflared`). Offline rows are keyed differently from online ones.
  - Its status is replaced: a `fail` the check found before an incidental refusal becomes `info`.
  - Its message is replaced. Delta-1's N3 suggested keeping it as detail.
  - For the four list-returning checks (`check_env_config`, `check_resolved_config`, `check_robot_reachable`,
    `check_mesh_nodes`), one refused probe collapses every row into one, hiding unrelated rows. With several robots
    configured, one refused robot would hide the others.
- **Verified by reading,** plus the live doctor output naming above.
- **Not exercised:** no case I ran hit the fail-hidden or list-collapse paths.
- **Fix:** on the ledger branch, keep `result`'s name(s), and append `refused (offline): <event>` to its message.
  Keep `fail` as is, or downgrade only `warn`. For a list, mark only the rows instead of replacing the list.

### NIT-4: `agent_pool._generate_llm_response` re-types an OfflineRefusal to ModelInferenceError
- **Where:** `runtime/agent_pool.py::_generate_llm_response`.
- **Classification:** the same class as delta-0 Executor S2 (`generate()` re-typing). It is loud and typed
  (`TurnResult.error`), and it is reached only if a router built offline refuses at dispatch.
- **Fix:** `except (ModelError, *SECURITY_REFUSALS): raise`.

### NIT-5: benchmark scenarios score an offline refusal as a failed run
- Decision 2 above. Pre-existing for `LeaderRefusal`.

### NIT-6: the §6 "every handler" wording is broader than the code, and the AUT item has no mechanical reminder
- **Where:** `docs/plans/atonia_gate_approach_notes.md` §6 status block.
- **The wording:** "SECURITY_REFUSALS … is what every 're-raise before degrading' handler re-raises". NIT-1 and
  NIT-4 are not on it, and neither are benchmark or the AUT fallback (the last is listed as Open).
- **Fix:**
  - Say "the handlers listed", or fold NIT-1 and NIT-4.
  - Add the strict-xfail AUT gate, or a 10b PR checklist line, as in decision 1.

### Guard coverage note (not a finding)
A deletion probe reverting `_run_server_mode`'s `except SECURITY_REFUSALS as _refusal` to `LeaderRefusal` survives
the suite. The refusal then reaches `_main_guarded` and still exits 2 ("Refusing to run" instead of "Refusing to
serve"), so the behavioural cost is the message only. Every other new guard is killed by a deletion probe.

---

## Stop rule
- SF1 is folded, apart from the AUT fallback. That is deferred with its dependency written down, and V22 ties 10a's
  merge to 10b.
- All six new findings are NITs and fail-closed for egress: the latch refuses every socket and spawn regardless.
- NIT-1 and NIT-4 are loud exit-code and typing issues. NIT-2 and NIT-3 are doctor reporting. NIT-5 is pre-existing.
  NIT-6 is wording and a reminder.
- None needs another review round. They can be folded opportunistically, as one-line changes plus a test each, or
  carried to 10b.

**STOP.**
