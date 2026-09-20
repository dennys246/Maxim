# Evidence Agent B — Runtime correctness and Maintainability (pymaxim v1.3.0 @ ad541dd7)

Checkout: `/Users/dennyschaedig/Scripts/Maxim/.worktrees/rescore-v1.3.0` (`git rev-parse v1.3.0^{commit}` = ad541dd7211f…). Everything ran with `PYTHONPATH=$PWD/src` and `MAXIM_DATA_HOME` pointed at a scratch dir, so `~/.maxim` was never touched. No live LLM, Minecraft, or hardware.

---

## Axis 1: Runtime correctness. Proposed grade: **C+**

### Deciding findings

1. **A documented public API silently returns wrong data: `AgentInstance.export_memories()` always reports 0 episodic memories.** (VERIFIED)
   - `src/maxim/runtime/agent_factory.py::AgentInstance.export_memories` (around line 253) reads `self.hippocampus.memories`. `Hippocampus` has no such attribute (it has `_memories` and `__len__`). The `AttributeError` is caught by `except Exception: result["episodic_memories"] = 0`.
   - Reproduced with the exact example in `docs/user/python-api.md` (lines ~229-245): `store_observation(...)`, then `export_memories()` prints `0 memories, 0 links` while `len(agent.hippocampus) == 1`.
   - Both tests are vacuous. `tests/unit/test_agent_factory.py::TestAgentInstance::test_export_memories` asserts only `"episodic_memories" in export`. `tests/unit/test_composable_api.py::test_agent_export_memories` stores an observation and then asserts only `isinstance(export, dict)`.
   - The swallow lint does not catch it, because the handler body is an assignment rather than `pass`/`continue`.
   - The bug reaches `AgentPool.export_memories` / `export_all_memories` (`src/maxim/runtime/agent_pool.py:279-287`) and `load.agent`'s own docstring example.

2. **The `create.agent` docstring example crashes.** (VERIFIED)
   - `src/maxim/create.py:181`: `agent.hippocampus.capture(perception="dark cave ahead")` raises `AttributeError: 'str' object has no attribute 'salience'` at `hippocampus.py:608`.
   - `capture()` takes typed `Perception`, not a string, and does not validate its input. Nothing checks docstrings (no doctest).

3. **The persistence round trip is correct, and corrupt state fails loudly.** (VERIFIED)
   - With typed input, the sequence `create.agent` → `capture` → `shutdown` → `load.agent` writes `hippocampus.json`, `nac.json`, `atl.json`, `ec.json` and `scn.json`, each carrying `_format_version`. After reload, the Hippocampus reports `total_memories: 1`.
   - A corrupted `hippocampus.json` makes `load.agent` raise `MemoryCorruptionError` naming the file. A missing agent raises `FileNotFoundError`. This matches the D17 claim.
   - Sampled "FIXED" rows in the bugs ledger (`docs/bugs/README.md`) each have the fix, a non-test caller, and a test:
     - **D17:** `_note_corruption` at `agent_factory.py:115`, used at lines 822-951.
     - **D42:** `SCN(persistence_path=…)` at `runtime/bio_stack.py:347`, reached through `build_bio_stack`. Its callers are cli, orchestrator and minecraft_harness.
     - **D82:** `Executor.permits` at `runtime/executor.py:158`, with a live caller at `runtime/agent_loop.py:4859` and a test in `tests/unit/test_console_tool_allowlist.py`.
     - **D55:** `FocusLearner.save` uses `atomic_write_json` (`proprioception/focus_learner.py:641`), is constructed in `default_network/network.py:430`, and is tested by `test_persistence_compat.py::test_focus_learner_roundtrip_and_pre_v1_file`.

4. **A composition gap is admitted but still open.**
   - `maxim.console.make_pairing_announcer` and `utils.audio.make_device_speak_sink` have no non-test caller. `grep -rn` over `src/` and `scripts/` finds only definitions, `__all__` entries, and one docstring mention (`utils/audio.py:161`). So the spoken-code pairing loop is still pieces that nothing assembles.
   - CHANGELOG.md:44-45 states this plainly ("no shipped command constructs the announcer"), which earns credit for honesty. D87 (sample rate / TTL on hardware) is OPEN.

5. **Silent-failure surface is large, and the lint only covers one shape of it.**
   - `scripts/lint_no_silent_swallows.py` reports **430** bare `except Exception: pass/continue` sites in 116 files. The 16 measurement-path files are held at zero, and the lint is diff-scoped so the count cannot rise.
   - There are **1,788** `except Exception` sites in total across `src/maxim`: 75 in `orchestrator.py`, 59 in `memory_hub.py`, 56 in `agent_loop.py`.
   - Finding 1 is an example of a swallow the lint cannot see.
   - On the positive side, the NAc credit path (`decisions/nac.py`) routes its 11 broad handlers through `log_swallowed_exception()`, which logs at WARNING on first hit.

6. **The survival-world seam holds up at its boundaries.**
   - **Hivemind ingest** (`hivemind/ingest.py`) enforces:
     - an entry-count cap and per-entry and total decompressed-size caps;
     - a decompression read bounded at `MAX_ENTRY_UNCOMPRESSED_BYTES + 1`;
     - refusal of NaN/Infinity JSON constants, with range checks;
     - node, foreign-count and embedding-norm caps.
     These are covered by 60 tests in `tests/unit/test_hivemind_ingest.py`.
   - **Minecraft client, low-severity gap** (`simulation/minecraft.py::_absorb_state`, around line 263): the client does not filter non-finite values. Verified: `float("nan")` passes through to `latest_state()`. `world_set_axis` then clamps with `max(lo, min(hi, v))` (`embodiment/audio_localization.py:760`), which turns NaN into the **top** of the sensor's range. This is the same NaN min/max poisoning that ingest explicitly refuses ("row M").
     - Inferred: this is unreachable from the shipped JS bridge, because `JSON.stringify(NaN)` emits `null`, which is dropped. It is still an unguarded wire boundary.
     - A code comment also misdescribes the drop. The whole dict is replaced; "previous truth" holds only because the backend skips missing keys.

7. **Diagnostics run offline but disagree with the CLI.**
   - `python -m maxim --help` works (0.33 s).
   - `maxim.diagnose()` returns a `DiagnosticReport`: 76 checks (16 ok, 11 warn, 49 info), `all_passed=True`. It also prints bare `http_request_failed` lines to stdout.
   - `python -m maxim doctor --json` on the same machine exits 1 with `worst_status: fail`, from a "Remote leader probe" check (HTTP 502) that `diagnose()` never runs.
   - Result: the API facade reports green where the CLI reports red. (Environment-dependent: this machine has a configured remote leader.)

8. **The fast suite is green, and the tests are real.** (VERIFIED)
   - Result: **10,867 passed, 39 skipped, 42 deselected, 0 failed in 10:00**, exit 0.
   - MemoryHub integration gate: 25 passed.
   - One hygiene issue: after the summary line, the captured output ends with a stream of `Orchestrator planning next probe... (555s)` spinner frames. Some test leaves a spinner thread running past its own lifetime.

### Enforced vs documented (runtime)
- **Enforced:**
  - fast suite + MemoryHub gate in CI;
  - swallow lint (count ratchet, zero in 16 files);
  - D44 strict-xfail pattern;
  - two-process `PYTHONHASHSEED` stable-hash test (`tests/unit/test_stable_hash_two_process.py:48` sets differing seeds; verified present);
  - CI greps (urllib: zero matches, verified; raw Reachy SDK motion: passes locally, verified);
  - `MemoryCorruptionError` on corrupt load (verified).
- **Documented only:** the public-API usage examples. Nothing executes them, which is why findings 1 and 2 survive.

### What would raise it to B−
- Fix `export_memories` and turn its two tests into value assertions.
- Make `Hippocampus.capture` reject non-`Perception` input, or fix the docstring. Add a doctest or example-execution test for the `create`/`load`/`python-api.md` examples.
- Wire the pairing composition into a shipped command, or formally mark it out of scope.
- Make `diagnose()` run the same role-aware checks as `maxim doctor`.

---

## Axis 2: Maintainability. Proposed grade: **C**

### Deciding findings

1. **God functions of extreme size are pinned but not shrinking.**
   - `scripts/lint_function_length.py`:
     - `runtime/agent_loop.py::run_agentic_loop` = **3,484 lines**
     - `simulation/orchestrator.py::start_simulation_mode` = **3,324**
     - `cli.py::_main_impl` = **1,747**
   - All three are at their pins. `src/maxim/utils/function_length_baseline.json` history shows 3546 → 3484 over about three weeks, mostly by paying for new code with small extractions.
   - The ratchet bounds **only these three**. My AST scan (VERIFIED) of all 6,286 functions found:
     - 53 functions longer than 200 lines and 18 longer than 300;
     - unbounded ones include `embodied_runtime/agentic_runtime.py::_start_agentic_runtime` (921), `cli_parser.py::_build_parser` (636), `embodied_runtime/media_loop.py::live` (587), `runtime/bio_stack.py::build_bio_stack` (511) and `decisions/nac.py::recommend_action` (462). Any of these can grow without CI noticing.
   - The core runtime and sim control paths therefore sit inside two 3,000+-line functions. For example, D82's fix lives at `agent_loop.py:4859`, deep inside `run_agentic_loop`.

2. **Module sizes are large.** 520 modules and 226,516 LOC under `src/maxim`. Examples: `agent_loop.py` 5,348 lines, `nac.py` 3,988, `orchestrator.py` 3,686, `doctor/checks.py` 3,574, `lane_backends.py` 3,168.

3. **Type checking covers a thin slice.**
   - CI runs mypy 1.20.0 on 18 files (`__init__`, `api`, `session`, `create`, `load`, `hivemind/`) with `--follow-imports=silent` and no strictness config in `pyproject.toml`. That run passes (VERIFIED).
   - The same invocation over all of `src/maxim` reports **1,050 errors in 141 files** (VERIFIED).
   - Finding 2 of the runtime axis (a `str` passed where `Perception` is expected) is the kind of error this gap lets through.

4. **The architecture audit is real and enforced, but carries accepted debt.**
   - `python -m maxim --audit-architecture` reports 33 accepted-debt findings across 30 baseline entries: 14 `memory must_not_import agents`, 6 `agents→runtime`, 4 `tools→agents`, and others. No new, stale or unreviewed findings.
   - It is enforced in the fast suite by `tests/unit/test_architecture_audit.py`:
     - `test_no_findings_outside_the_baseline`
     - `test_no_stale_baseline_entries`
     - `test_every_baseline_entry_is_reviewed`
   - The lowest layer (memory) importing the agents layer 14 times is a genuine cycle-shaped debt.

5. **The lint and ratchet estate is unusually strong, and CI runs it.** `.github/workflows/test.yml` lint job, all run locally and clean:
   - orphan modules: 0, and any new orphan fails;
   - atomic-io rename ratchet: 12 sites, never rising;
   - swallow ratchet;
   - fix-touches-tests;
   - `[Unreleased]`-on-src-change;
   - harness provenance;
   - prereg-precedes-data;
   - body rest-neutral;
   - version sync;
   - CLAUDE.md invariant lint (7 docs, all guards present);
   - ruff check/format.
   Most carry positive-control tests. This is the main reason the grade is not lower.

6. **Spot-check of three CLAUDE.md "Regression guard" lines:**
   - (a) Stable-hash two-process test: real. It spawns interpreters with differing `PYTHONHASHSEED` (`tests/unit/test_stable_hash_two_process.py:41-48`).
   - (b) urllib CI grep: holds, with zero matches.
   - (c) Raw Reachy SDK motion CI grep: holds. It is a text regex with path allow-lists (test.yml:959-963), so it can be evaded by aliasing. I found no evasion in the tree; the other `.goto_target(`/`.look_at_image(` calls go through controller/Selfy wrappers.
   - Overall the guards do what they say. The weakest are grep-shaped.

### Enforced vs documented (maintainability)
- **Enforced:** all the lints in finding 5, the architecture audit via the fast suite, a mypy slice of 18 files, and the three-function length pin.
- **Not enforced:**
  - length bounds on every other function;
  - typing for the other ~500 modules;
  - any cap on module size or on total broad-`except` count (only the pass/continue subset is ratcheted).

### What would raise it to C+ or B−
- Decompose at least one of `run_agentic_loop` or `start_simulation_mode` below about 1,000 lines, with the pin lowered in the same commit.
- Extend the length ratchet to every function over some threshold (for example, grandfather all functions over 300 lines at their current span).
- Widen mypy to the runtime core (`runtime/`, `decisions/`, `memory/`) with a per-file error-count ratchet.

---

## Test suite result
- Command: `python -m pytest tests/ -x -q -m "not slow" --ignore=tests/integration/test_memory_hub.py`
- Result: 10867 passed, 39 skipped (mostly 'SemanticEngine not healthy (missing deps)'), 42 deselected, in 600 s, exit 0.
- `tests/integration/test_memory_hub.py`: 25 passed.
- Leaked spinner threads are still printing 'Orchestrator planning next probe' after the session ends (hygiene issue, not a failure).

## Verified vs inferred
- **Verified by running:**
  - the round trip, corrupt-load and missing-agent behaviour;
  - the `export_memories` zero-count bug with the python-api.md example;
  - the `create.agent` docstring crash;
  - `diagnose()` vs `doctor --json` disagreement;
  - `--help`;
  - every lint named above (outputs quoted);
  - mypy slice (clean) and repo-wide (1,050 errors);
  - architecture audit;
  - AST function-length scan;
  - NaN passing through `_absorb_state`;
  - the grep for zero callers of the pairing announcer;
  - for the four sampled bugs-ledger rows, the fix, a caller and a test (by grep and read).
- **Inferred, not executed:**
  - NaN being unreachable from the JS bridge (JSON.stringify semantics);
  - the NaN→top-of-range clamp (read from code and from Python `min`/`max` semantics, not run end to end through an embodiment);
  - the reason `diagnose()` skips the remote-leader probe (role detection; not traced);
  - that the Reachy grep has no aliasing evasion (a partial grep, not an exhaustive data-flow check).

## Independence
- I did not open `docs/limits/score_cards/`, `docs/plans/burndown_1_3.md`, the redacted roadmap section, or anything under `~/.claude/`.
- **One incidental contact.** The `_comment` field of `src/maxim/utils/function_length_baseline.json`, which I read as ratchet evidence, says "score card 2026-08-27 Maintainability 'Upgrade to C+'". That implies an earlier Maintainability grade at or below C.
  - I read it while gathering evidence, after I had already measured the function spans.
  - My C is based on the measured evidence above: pinned 3,000+-line god functions, 53 functions over 200 lines with no bound, and a 1,050-error typing surface. Offsetting those is the strong, CI-enforced lint estate.
- A CI comment at `.github/workflows/test.yml` (the fix-touches-tests step) mentions "the one the score card counts". It contains no grade.
- The CLAUDE.md system context mentions score cards but shows no letter grades.
