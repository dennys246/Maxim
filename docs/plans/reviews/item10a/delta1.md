# Delta round 1: #1315 item 10a fold (`git diff --cached fb3ae6f3` in .worktrees/v-item10a)

Reviewer: delta lens (a different reader from round 1). Probes ran in a copied tree
(`scratchpad/review10a/dcopy`, plus `dprobecopy` for deletion probes), with `PYTHONPATH` set to the copy's `src`
(`maxim.__file__` checked) and `HOME`/`XDG_CONFIG_HOME` pointed at scratch dirs. I did not edit the shared worktree
and ran no state-changing git. In the worktree I ran only read-only lints: ruff check and format --check, the
no-silent-swallows, function-length and orphan-module lints. All were clean. I did not run mypy, because two
pytest/mypy processes from other sessions were already on the box.

Targeted tests in the copy:
- 61 passed: `test_offline_startup_10a.py` and `test_bounded_stop_side_1438.py`;
- 663 passed: config_loader, writer, cli, the 856 format version, doctor, diagnose-matches-doctor, offline_latch,
  lane_served_nctx, leader_fail_closed, persistence_compat and the router tests. The only 4 failures are the known
  no-`.git` provenance tests, an artifact of the copy.

Counts: **0 BLOCKER, 1 SHOULD-FIX, 6 NIT.** Verdict: **NOT STOP** (see the stop-rule section).

---

## Round-1 findings: fold status

### Executor lens
| Finding | Status | Evidence |
|---|---|---|
| S1: an unparseable config runs online | **Folded correctly** (V20). A NIT remains (N1 below). | Probes: invalid JSON, a trailing comma, a non-object root, chmod 000 on the file, and chmod 000 on the dir all make `cli.main` print `✗ …` and return 2. `_read_config_json` is now shared with `load_config`. |
| S2: `generate()` re-typed `OfflineRefusal` | **Folded correctly** | `except (LeaderRefusal, _offline.OfflineRefusal): raise`. Tested in process (`test_router_build_and_dispatch_refuse_in_process`), and a deletion probe fails it. |
| S3: `warmup()` bypassed the predicate | **Folded correctly** | `_refuse_offline_lanes("router[warmup]")` runs first, in the caller's thread. It covers (a) and (b), because it checks the provider list and not the backend cache. The optional `finally` on `_ready_event` was not done; with a synchronous refusal it is no longer needed. |
| S4: build refusal swallowed into a fallback router | **Incomplete.** The lane builders re-raise and `_validate_remote_urls` skips offline, so `build_primary_router` now raises by name. The "further swallows" S4 listed are untouched, and the refusal now reaches them. | See **SF1**. |
| S5: `llm.offline` not reachable from `maxim config` | **Folded correctly.** A guard gap remains (N4). | Probe: `maxim config get`, `set false`, `set maybe` (rc 2), `unset` all work on a latched process; the doctor resolved-config walk shows `✓ llm.offline: True [source=config.json]`. |
| S6: library entry points never read `llm.offline` | **Folded** at `build_primary_router` (`_router_build_preflight`), in `api.download_model` and in `reachy_diagnostics.main`. A bare `LLMRouter(...)` built outside `build_primary_router` still does not engage (N5). | Probe: `maxim.create.router(model="claude-sonnet")` with config offline leaves the latch up. |
| N1: empty file warns | Folded | `test_an_empty_file_is_absent_and_silent` |
| N2: `api._engage_offline` body probe | Folded | `test_api_engage_reads_the_config` |
| N3: in-process coverage twins | Folded | `test_router_build_and_dispatch_refuse_in_process` books all four ledger events |
| N4: `pytest.raises(Exception)` | Folded | It now names `ConfigurationError` |
| N5: blackwell / tunnel auto-spawn | Out of scope (10b, V22) | n/a |

### Architecture lens
| Finding | Status |
|---|---|
| B1: router-build refusal swallowed on the shipped path | **Folded at the lane-manager seam** and verified: a config-only peer lane (`lanes.large.remote_url`, no env) gives `build_primary_router` → `OfflineRefusal … router[build] refused: lanes that are not in-process: lane-large`, with ledger `['router[build]']`. **Incomplete one layer up**: three callers of `build_primary_router` still absorb it (SF1). |
| S1: not read at router build | Folded (`_router_build_preflight`; `test_build_primary_router_engages_from_config`). |
| S2: not settable through `maxim config` | Folded (`SETTABLE_FIELDS`). |
| S3: docs and changelog lack the "not yet usable" caveat | Resolved by **V22**: the PR merges only with 10b, so `main` never carries the half-done mode. |
| N1: the raw reader drifts from `load_config` | Folded (one shared `_read_config_json`). |
| N2: §1A.6 stale on version ownership | Folded ("As built" note). |
| N3: warmup outside `_complete_text` | Folded (as Executor S3). |

### Wire lens
| Finding | Status |
|---|---|
| B1: `test_g7` hard-coded `"1.4"` | **Folded correctly.** It now asserts first appearance in the fixture; passes. |
| S1: not read at router build | Folded at `build_primary_router` rather than at `LLMRouter.__init__` (N5). |
| S2: probe-and-drop, the builder swallow, the caller fallback | (1) probe-and-drop: folded (offline skip; nothing cached; tested). (2) builder swallow: folded (both builders; tested; the deletion probes `remote_reraise` and `validate_skip` are caught). (3) caller fallback: the orchestrator's primary build has no `try`, so the raise now passes its `None → LLMRouter(load_llm_config())` fallback. That part is fixed. `agentic_runtime` still swallows it (SF1). |
| S3: doctor crashes offline | Folded (V21). Verified end to end: `maxim doctor` and `maxim doctor --as peer …` with `llm.offline: true` run to the end with no traceback, and each uncaught refusal is its own `ℹ … refused (offline): …` row. A residual is in N3. |
| S4: `maxim config set llm.offline` refused | Folded. |
| N3, N4: entry points that do not engage | `download_model` and `maxim-diagnostics` are engaged. `selfy` and `models.download` are left to 10b (folder decision 6). |
| N6: auto-spawn lane loads in-process | Resolved by **V23**; §6 text updated. |

---

## Folder decisions (taken without the owner)

1. **`SETTABLE_FIELDS = frozenset(_FIELD_TO_ENV) | _CONFIG_ONLY_FIELDS`: sound.**
   - It keeps `_FIELD_TO_ENV`'s value type, so `_ABSORBED_ENV_VARS` and doctor's env→field table (`checks.py`, the
     `env_to_field` map) are unaffected.
   - `resolve_setting` skips the env leg when `env_name is None`.
   - Every consumer that needed the registry was switched: config_cli get/list/set/unset, the writer set/unset,
     `resolve_setting` and doctor. Grep across `src/` and `scripts/` found no remaining "settable" gate on
     `_FIELD_TO_ENV`.
   - Gap: the CLI half is unguarded (N4).
2. **Doctor `_wrap_checks_for_offline`: sound as a mechanism; should change in one cheap respect (N3), not owner
   territory.**
   - `run_all_checks` resolves the check names from module globals at call time, so it gets the wrapped versions.
   - The `__module__ == __name__` filter excludes imported names.
   - Every `check_*` is annotated `-> CheckResult` or `-> list[...]`, so the list/row detection holds.
   - `test_every_doctor_check_is_wrapped` guards it.
   - But "a check that catches it internally reports it its own way" misses V21's letter: those rows give wrong
     causes and wrong fixes (N3).
3. **`cli.main` prints the ConfigurationError and returns 2 before signal handling: sound.** Nothing is installed
   that would need tearing down. A consequence worth knowing: `maxim doctor` and `maxim config` cannot run on a
   broken config.json either (the old doctor `fail` row is unreachable). The message names the file and the parse
   error, and V20 asked for exactly this.
4. **A non-object root refuses: sound.** Verified: `load_config` and `engage_offline_if_configured` raise the
   identical message (`config.json: root must be a JSON object, got list`).
5. **`generate_json` refuses at `router[backend]`: sound.** It is still a typed `OfflineRefusal`, raised before
   any backend is built.
6. **`download_model` and `reachy_diagnostics.main` engaged; `python -m maxim.models.download` and
   `python -m maxim.embodied_runtime.selfy` left to 10b: sound under V22, provided 10b's scope names them.**
   - `selfy` is now partly covered, because its router build engages via `_router_build_preflight`. Anything selfy
     does before the router build (robot connect, media) runs unlatched.
   - `agentic_runtime` also swallows a refusal raised there (SF1).
   - Add both module mains to 10b's list explicitly.
7. **`_read_config_json` extracted: sound; `load_config` behaviour is unchanged.** It is a verbatim move of
   `config_file_present` → read (`OSError` → ConfigurationError) → strip-empty → `json.loads(stripped)`
   (JSONDecodeError → ConfigurationError, same message), and `_maybe_migrate_from_peer_yml` still runs first. The
   `test_config_loader` and persistence-compat tests pass. The behaviour change is deliberately in
   `engage_offline_if_configured` and so in `build_primary_router` (V20). It shares one inherited gap (N1).

---

## New findings in the delta

### SF1 (SHOULD-FIX): the refusal now leaves `build_primary_router` by name, but three shipped callers still turn it into "run degraded"
**Where:**
- `utils/logging.py::warn_unless_refusal` re-raises only `LeaderRefusal`. It is used by
  `embodied_runtime/agentic_runtime.py` (the LLM worker build, `except Exception`) and by `media_loop.py`.
- `simulation/research_orchestrator.py::_build_research_llm` (`except LeaderRefusal: raise` / `except Exception`
  → heuristics).
- `simulation/orchestrator.py` AUT build (`except Exception` → "falling back to shared").
- Smaller, same shape: `simulation/benchmark.py`'s paper writer (`LLMRouter()` → WARNING, returns None).
- `cli.py::_start_server_llm` returns a loud NOT READY reason, which is acceptable as is.

**Scenario:**
1. `llm.offline: true` with a peer, cloud or loopback lane.
2. Research mode logs `Research LLM init failed: offline (llm.offline): router[build] refused: … lane-large —
   Writer/Reviewer will use heuristics` and runs.
3. The embodied runtime logs "Failed to create LLM worker" and runs with no LLM.
4. A sim with `--aut-model <cloud profile>` logs a WARNING and runs the AUT on the **shared** model, a silent swap of
   the model under test.

The fold's own docs say otherwise. The §6 status block says the refusal is "refused by name, never dropped to a
fallback router", and the changelog says "refuses by name". Round-1 Executor S4 and Architecture B1 named exactly
these fallbacks.

**Classification:** fail-closed for egress (the latch still refuses every socket and spawn). It is **not** in a class
an existing check covers: §6's swallow census covers handlers *around a census site* (fetch, socket, subprocess
calls). These handlers wrap `build_primary_router`, not a census site, so 10b's census will not list them as
designed.

**How verified:**
- `scratchpad/review10a/dprobe3.py`: `_build_research_llm()` returns None with the WARNING above.
  `warn_unless_refusal("x %s", OfflineRefusal(...))` logs and returns instead of re-raising.
- The orchestrator AUT and benchmark paths were confirmed by reading.

**Fix (in the seam, as the codebase already does for LeaderRefusal):**
- Make `warn_unless_refusal` re-raise `OfflineRefusal` too. Better: one `SECURITY_REFUSALS = (LeaderRefusal,
  OfflineRefusal)` tuple that it and the `except LeaderRefusal: raise` sites use.
- Add `OfflineRefusal` to the pass-through in `_build_research_llm` and ahead of the orchestrator AUT
  `except Exception`.
- Add one test that drives `_build_research_llm` (or `warn_unless_refusal`) with a refusal.
- If the folder prefers to defer this to 10b, the §6 status line and the changelog must stop claiming "never dropped
  to a fallback", and 10b's scope must list these handlers by name.

### N1 (NIT): a config.json that is not valid UTF-8 refuses with a raw traceback, not the typed refusal
`runtime/config_loader.py::_read_config_json` catches `OSError` around `read_text`. A `UnicodeDecodeError` is a
`ValueError`, so `cli.main`'s `except ConfigurationError` misses it. The command still refuses (fail-closed), but as a
traceback. This is inherited verbatim from `load_config`.

Verified: bytes `\xff` in the file give `RAISED UnicodeDecodeError` from `cli.main(["--help"])`.

Fix: `except (OSError, UnicodeDecodeError)` → ConfigurationError, which also improves `load_config`.

### N2 (NIT): `reachy_diagnostics.main` lets V20's ConfigurationError escape as a traceback
`utils/reachy_diagnostics.py::main` calls `engage_offline_if_configured()` uncaught. It is fail-closed. Fix: mirror
`cli.main` (print, return 2).

### N3 (NIT; should change if anything is folded anyway): doctor rows for refusals a check catches itself give the wrong cause and fix
`doctor/checks.py::_refused_offline_row` turns only an *uncaught* refusal into a row. Checks that swallow it
internally report what looks like an outage. Observed offline:
- `Auto-spawn server: nothing responding at http://127.0.0.1:8100/v1 → Run maxim in another terminal — auto-spawn will launch one`.
  This is false offline, since auto-spawn is ruled out.
- `Inference coherence: server not reachable`.
- In peer mode: `✗ Remote leader probe: … unreachable: other (OfflineRefusal …) → check maxim peer logs` (doctor
  exits 1) and `Peer latency: all latency probes failed`.

These sit around `utils.http` census sites, so the class is 10b's swallow census: fail-closed and covered.

Cheap uniform fix in the same wrapper: record `len(_offline.ledger())` before the call. If it grew, mark the
returned row(s) `refused (offline)`, keeping the check's message as detail. That meets V21 for every check without
touching them one by one.

### N4 (NIT): guard gaps, where deletion probes survive
Each mutation below was applied in `dprobecopy` and run against `test_offline_startup_10a`, `test_config_cli`,
`test_doctor*`, `test_config_loader`, `test_diagnose_matches_doctor`, `test_api*` and `test_download_consent_1343`.
Every one still passed:
- `config_cli._cmd_set`'s gate reverted to `_FIELD_TO_ENV`. The round-trip test drives `config_writer` only, not the
  CLI, which is the path round 1 reproduced.
- `check_resolved_config` walking `_FIELD_TO_ENV` again (no `llm.offline` row).
- The engage removed from `api.download_model`. The AST verb test still lists seven verbs.
- The engage removed from `reachy_diagnostics.main`.

Fix: `cli.main(["config","set","llm.offline","true"])`, then `get`; an assertion that a doctor resolved-config row
named `llm.offline` exists; `download_model` added to the AST verb list; an AST or call assertion for
`reachy_diagnostics.main`.

### N5 (NIT): "the router build engages" holds only for `build_primary_router`
A bare `LLMRouter(load_llm_config())` from a library or a script (`scripts/exp44/rerun_ablated_offline.py`,
`behavioral_convergence_*`, or any user code) never reads `llm.offline`. The changelog and user doc say "the router
build".

Fix: narrow the wording to `build_primary_router` / `maxim.create.router`. Or engage lazily at the top of
`LLMRouter.__init__`; Wire round 1 notes the router already lazy-imports `maxim.runtime`.

### N6 (NIT): the user doc says a running `maxim serve` picks the key up "on restart"; it can now latch mid-process
`console/handle.py` rebuilds the talk router via `build_primary_router()` when the config generation changes. That
build now engages the latch, which is one-way, inside the running server. It is fail-closed: more restrictive, and
`config set llm.offline false` cannot undo it until a restart.

Fix: one sentence in `docs/user/configuration.md`, or skip the engage in `_router_build_preflight` when the process
is already serving. The doc sentence is enough.

---

## Stop rule
Per docs/lessons/development-flow-speed.md §1, a round stops only when every finding is fail-closed AND in a class
already covered.
- N1–N6 are all fail-closed. N3 is in the census class 10b covers. N1, N2, N4, N5 and N6 are small and can go to a
  follow-up or to 10b.
- **SF1 is fail-closed for egress, but no existing or planned check covers it.** The swallow census keys on handlers
  around census call sites, and these handlers wrap the router build. The fold's own status text also claims the
  opposite of the behaviour.

**NOT STOP.** Fold SF1, or narrow the claim and name the handlers in 10b's scope. Then run one more delta-scoped
round on that fold only (`utils/logging.py`, the research and orchestrator AUT handlers, and the test).
