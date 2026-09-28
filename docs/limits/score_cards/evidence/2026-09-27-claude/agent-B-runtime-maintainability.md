# Evidence Agent B — Runtime correctness and Maintainability (pymaxim v1.3.1 @ 7e695a58)

**Checkout + environment:** `/Users/dennyschaedig/Scripts/Maxim/.worktrees/rescore-v1.3.1`, detached HEAD `7e695a58`. The tree is dirty only because of the firewall: score cards and three archive plans were deleted, and `docs/lessons/experiment-prereg-precedes-data.md`, `docs/plans/roadmap_1_3_x.md` and `src/maxim/utils/function_length_baseline.json` were redacted. Python 3.12.12 (repo `.venv`), `PYTHONPATH=<worktree>/src`, `MAXIM_DATA_HOME=<scratchpad>/rescore/B_home`. `maxim.__file__` resolves to the worktree. Tools: mypy 1.20.0, ruff 0.14.14.

---

## Runtime correctness

Proposed grade: **C+**

### Deciding findings

1. **VERIFIED: the required fast suite is green at the tag.** `python -m pytest tests/ -x -q -m "not slow" --ignore=tests/integration/test_memory_hub.py` gave **12105 passed, 39 skipped, 44 deselected in 638.76s, exit 0**. `tests/integration/test_memory_hub.py` gave **25 passed in 0.17s**. CI runs both (`.github/workflows/test.yml` steps "Run required fast suite" and "MemoryHub integration gate").

2. **VERIFIED: the documented public API runs.** Every snippet in the Bio-Subsystems, Modifying, Agents and Pools sections of `docs/user/python-api.md` (lines 166–290) executed without an exception.
   - `export_memories()` now counts correctly: "1 memories" after one store. This is the 1.3.1 fix.
   - The `create.agent` docstring example (`src/maxim/create.py:185-195`) runs.
   - The create → capture → `record_event` → shutdown → `load.agent` round trip restores the episodic memory, and recall finds it.
   - `load.agent("missing")` raises `FileNotFoundError` with a fix hint.
   - A corrupt `hippocampus.json` raises `MemoryCorruptionError` under `on_corrupt="raise"`. `"fresh"` gives an empty agent. A bogus `on_corrupt` raises `ValueError`. These match D17 and D41 in `docs/bugs/README.md`.
   - `tool_whitelist=` raises `ValueError` (D73).

3. **VERIFIED DEFECT: silent data loss through the documented create path.** `maxim.create.hippocampus(persistence_path=P)` never reads an existing `P`: `src/maxim/create.py:73-76` builds `Hippocampus(config)`, and only `Hippocampus.from_config` loads (`hippocampus.py:545-549`). The docstring says the path is for "saving/loading".
   - Measured: 3 memories on disk; reopened via `create`, recall returned 0; after one store and `save()`, 1 memory on disk. Two memories were destroyed with no warning.
   - `create.atl` does the same: 2 concepts on disk, 1 after reopen and save.
   - The same happens with a corrupt file: it loads empty with nothing logged at WARNING and is overwritten on save.
   - The docs' own example writes to `/tmp/memory.json`, so running that script twice loses the first run's memories.
   - This footgun is not in the `docs/bugs/README.md` ledger (checked by grep).

4. **VERIFIED: typed-error inconsistency.** `maxim.load.hippocampus(<corrupt>)` raises a raw `json.JSONDecodeError`, while `load.agent` raises `MemoryCorruptionError` for the same file.

5. **VERIFIED: `maxim.diagnose()` and `python -m maxim doctor --json` agree on the verdict.**
   - The CLI exits 1 with `worst_status: fail`. `diagnose()` gives `all_passed=False`, `failures=1` ("Remote leader probe"), which is the 1.3.1 fix.
   - Residual drift: the CLI reports 71 checks and `diagnose()` reports 70. The extra one is `legacy_env_migration` (warn), which `diagnose()` does not run.

6. **VERIFIED: the 1.3.1 security fixes behave as described.** Checked directly against `SandboxExecutor` (`src/maxim/utils/sandbox_executor.py`):
   - No approver → `APPROVAL_UNAVAILABLE`.
   - Approver returning `"yes"` rather than `True` → `BLOCKED`.
   - An approved `.py` script runs (#800).
   - A sibling-prefix directory `/…/B_sb2` → `PERMISSION_DENIED` (#801).
   - A symlink out of the sandbox → `PERMISSION_DENIED`.
   - A script opening `/etc/hosts` → `PermissionError`; `import os` → `ImportError`.
   - `http.fetch_url(public_only=True)` refused `127.0.0.1`, `localhost`, `[::ffff:10.0.0.1]` and `169.254.169.254` (#824). `tools/http_fetch.py:358,391` is the non-test caller.
   - Executor mode gate (#826): `executor.set_mode_source` has one non-test caller (`src/maxim/runtime/agent_loop.py:2151`), and `tests/unit/test_mode_tool_gate.py` covers it and passed in the suite. The gate is opt-in: `test_no_mode_source_means_no_mode_restriction` shows an executor with no mode source is unrestricted.
   - Follow-ups #921 (base-URL and download re-resolve) and #924 (passive agent can self-switch to active) are still OPEN.

7. **VERIFIED: the silent-failure surface is large.**
   - `scripts/lint_no_silent_swallows.py` reports **418 bare `except Exception: pass/continue` sites in 112 files**. The 16 measurement-path files are held at zero bare swallows, but **153 broad swallows on the measurement path do not report**.
   - `grep` finds **1807** `except Exception`/`except:` sites in `src/maxim`.
   - The lint only ratchets these counts down; it does not remove them. D11 is ACCEPTED.

8. **VERIFIED: open defects include live silent no-ops on core paths.** `gh issue list --state open` returns 34 issues. Two spot-checked:
   - **#840:** `src/maxim/bridges/fear_bridge.py:624-633` calls `self.nac.record_event(event_key, outcome, metadata=...)`. The signature at `decisions/nac.py:1043` has no `metadata` parameter, and the resulting `TypeError` is eaten by `except Exception: pass`. FearCircuitBridge has never reported to NAc.
   - **#908:** `runtime/bio_stack.py:458` builds `Cerebellum(config=CerebellumConfig())` with no `persistence_path`, so Cerebellum state is never saved.
   - Other open issues describe the same class (unverified by me): #841 (swallowed AttributeError), #873 (damage vanishes but reports success), #910 (unreachable annotation), #851 (leak), #816, #812, #818, #819.
   - The `docs/bugs/README.md` ledger additionally carries ~20 OPEN or PARTIAL rows (D1, D10, D23, D29, D30, D45, D49–D51, D63–D65, D83, D85–D87, among others). None of #840/#908/#910/#873 appears in that ledger (grep count 0).

9. **VERIFIED: D23 (OPEN) is visible in the suite's own output.** The captured pytest log contains raw spinner frames ("Orchestrator planning next probe... (582s)") written after the test summary.

### What holds

- A large (12k-test) suite that is green, plus the MemoryHub gate.
- The public verbs do what the docs say, and the round trip and corruption paths are correct on the `load.agent` route.
- Each security fix I exercised behaves as claimed.
- FIXED ledger rows spot-checked cite guard tests (D71, D72, D73, D54, D58), and those tests ran in the green suite.

### The deciding gap

Known, unfixed silent failures sit on shipped runtime paths: #840 and #908 were verified, and more issues are open in the same class. The swallow surface is ~1.8k broad excepts, with 153 unreported ones on the measurement path. On top of that, a public creation path silently destroys persisted data. The suite and the lints bound growth; they do not establish correctness of these paths.

### To reach B−

- `create.hippocampus`/`create.atl` with an existing `persistence_path` either load it or refuse/warn. Pin this with a test that fails on the current clobber.
- `load.hippocampus`/`load.nac`/`load.atl` raise `MemoryCorruptionError` on corrupt input, with tests.
- Close #840 and #908, each with a guard test that fails when the mechanism is deleted.
- The swallow lint's measurement-path unreported count reaches 0 (CI-enforced).
- `diagnose()` and `doctor --json` produce identical check sets, asserted by a test.

---

## Maintainability

Proposed grade: **C+**

### Deciding findings

1. **VERIFIED: god functions persist.** AST spans at the tag:
   - `run_agentic_loop` (`runtime/agent_loop.py`): **3389** lines
   - `start_simulation_mode` (`simulation/orchestrator.py`): **3322**
   - `_main_impl` (`cli.py`): **1696**
   - `_start_agentic_runtime`: 918

   Across 6544 functions (nested ones counted):

   | Span | Functions |
   |---|---|
   | >100 lines | 268 |
   | >200 lines | 53 |
   | >300 lines | 19 |
   | >500 lines | 8 |
   | >1000 lines | 3 |

2. **VERIFIED: the length ratchet covers exactly 3 functions and has two parallel mechanisms.**
   - `tests/unit/test_function_length_baseline.py` with `src/maxim/utils/function_length_baseline.json` is a strict-equality ceiling (3389/3322/1696) and passed in the suite.
   - `scripts/lint_function_length.py` (CI) has stale pins (3453/3324/1743) and prints "+64 — lower the pin". On its own it would let `run_agentic_loop` grow by 64 lines.
   - No general bound exists: the 918-line and 636-line functions, and the rest of the 53 over 200 lines, are unconstrained.
   - The baseline's HISTORY shows real extractions (3546 → 3389 since v1.1.0), but that is ~4% on the worst function.

3. **VERIFIED: mypy scope is tiny.**
   - The CI invocation (`test.yml` "mypy public API surface + hivemind") passes: "no issues found in 19 source files". That is 19 of 524 modules, **3.6%**, with `--follow-imports=silent`.
   - `mypy src/maxim --ignore-missing-imports`: **Found 1071 errors in 142 files (checked 524 source files)**.
   - 85 `# type: ignore` in src.

4. **VERIFIED: ruff enforces a minimal rule set.** `ruff check src/ tests/` → "All checks passed". `ruff format --check` → "1099 files already formatted". Both are enforced in CI (pinned 0.14.14). But `pyproject.toml:291-296` selects only `["E","F"]`, with E402 and E501 ignored: no complexity (C901), no blind-except (BLE), no bugbear.

5. **VERIFIED: the lint estate is real and CI-wired.** 13 `scripts/lint_*.py` all appear in `.github/workflows/test.yml` (lint job). Run locally:
   - `lint_orphan_modules`: 0 orphans, 0 grandfathered.
   - `lint_atomic_io_ratchet`: clean (residual hand-rolled renames listed).
   - `lint_version_sync`: OK at 1.3.1.
   - `lint_no_silent_swallows`: clean.
   - `lint_claude_md_invariants` FAILED locally on one broken link to `docs/plans/archive/roadmap_1_1_to_1_3.md`. That file was removed by the firewall, so this is an artifact of the blind copy, not a tag defect.
   - CI also runs grep invariants for `urllib.urlopen`, `InternetAccessPolicy` and PTB dormancy.

6. **VERIFIED: layering is enforced with accepted debt.** `python -m maxim --audit-architecture` reports "33 accepted-debt finding(s) across 30 baseline entries… No new, stale, or unreviewed findings". It is enforced in the fast suite by `tests/unit/test_architecture_audit.py::test_no_findings_outside_the_baseline`, `::test_no_stale_baseline_entries` and `::test_every_baseline_entry_is_reviewed`.

7. **VERIFIED: size.** 524 modules, 233,361 LOC in `src/maxim`, 48 top-level packages. 49 modules exceed 1000 lines; 13 exceed 2000, the largest being `agent_loop.py` (5543), `decisions/nac.py` (4046), `orchestrator.py` (3711) and `doctor/checks.py` (3620).

8. **VERIFIED: no coverage threshold.** `pytest-cov` is installed in CI and `[tool.coverage]` is configured, but `test.yml` and `pyproject.toml` contain no `fail_under`/`--cov-fail-under`.

### What holds

- An unusually thorough set of ratchets, all running in CI: orphan modules, architecture audit, silent-swallow count and de-instrumentation, atomic-io, version sync, harness provenance, prereg ordering.
- Formatting and base lint are clean.
- The worst functions shrink measurably, under a strict test.

### The deciding gap

The core control flow still lives in three functions totalling ~8,400 lines, plus 50 more functions over 200 lines with no bound. Static typing covers 3.6% of modules, with 1,071 errors outside that set. The ratchets prevent regression but leave the debt in place.

### To reach B−

- A repo-wide function-length ceiling in CI, such as a per-function baseline file where any function over 200 lines may not grow and no new function may exceed 200.
- The stale CI pins in `scripts/lint_function_length.py` removed or unified with the strict test.
- At least one god function under 1000 lines.
- mypy in CI extended to a named, growing module set (for example `memory/`, `decisions/`, `runtime/executor.py`), with a repo-wide error-count ratchet printed and enforced.
- ruff adding `C901` (with a threshold baseline) and `BLE001`, with a baseline.
- A coverage floor enforced in CI.

---

## Contacts

- `src/maxim/utils/function_length_baseline.json` `_comment` references "score card 2026-08-27 Maintainability '[redacted]'". The grade itself is redacted.
- `.github/workflows/test.yml` (God-function ratchet step comment) references "The 2026-08-27 card's standing complaint". It gives no grade.
- `docs/bugs/README.md` rows D40 and D41 cite "score card 2026-08-27 N1/N2". They give no grade.

None of these stated a grade, and none influenced mine.

## Commands run

`git rev-parse/status`; fast pytest suite (full); `pytest tests/integration/test_memory_hub.py`; `scripts/lint_function_length.py`, `lint_no_silent_swallows.py`, `lint_orphan_modules.py`, `lint_atomic_io_ratchet.py`, `lint_claude_md_invariants.py`, `lint_version_sync.py`; `mypy` (CI invocation, and `src/maxim` repo-wide); `ruff check` + `ruff format --check`; `python -m maxim --audit-architecture`; `python -m maxim doctor --json` vs `maxim.diagnose()`; ad hoc scripts exercising the `docs/user/python-api.md` snippets, the create/capture/shutdown/load round trip, corrupt-file loads and the `create.*` reopen clobber; `SandboxExecutor` approval/containment probes; `http.fetch_url(public_only=True)` private-address probes; AST span/size census; `gh issue list --state open --limit 60`; greps over `src/`, `tests/`, `.github/`, `docs/bugs/README.md`, `CHANGELOG.md` (1.3.1 section ranges only).
