# Publication Refinement Plan

> **Status:** Phase 0 DONE. Phase 1 DONE (1m deferred). ASH ALL PHASES DONE. Module Compartmentalization IN PROGRESS. POG-0 DONE.
> **Goal:** Fix the blockers, code quality issues, and documentation gaps identified in a comprehensive repo review before publishing pymaxim v0.2.0 to PyPI.
> **Estimated scope:** ~1,400 LOC of fixes + ~500 LOC of tests + ~400 LOC of docs changes across 5 phases (includes Mother Maxim on-ramp items 1m-1n)
> **Sequence:** Blockers (0) → Code Quality + Data Integrity + UX + API Contract (1) → Test Depth (2) → Docs & Packaging (3) → Publish (4)
> **Timeframe:** 3-4 focused sessions
>
> **UPDATE (2026-04-08):** A comprehensive repo review identified deeper API surface and module structure issues. Two companion plans now exist:
> - **[API Surface Hardening Plan](../archive/api_surface_hardening_plan.md)** — ALL PHASES COMPLETE. Wired stub verbs, fixed research protocol, error handling on user-facing paths, integration tests, README overhaul.
> - **[Module Compartmentalization Plan](../archive/module_compartmentalization_plan.md)** — COMPLETE (2026-04-09). 5 god-modules decomposed, 7 new files, 125 tests.
>
> **Revised sequence:** Phase 0 (DONE) → ASH (DONE) → Phase 1.5/2 (DONE) → POG-0 (DONE) → Packaging (DONE) → Module Compartmentalization (IN PROGRESS) → **Must-Fix Blockers** → Phase 4 (publish)
>
> ## Pre-Publish Must-Fix Checklist
>
> These four items **block publication**. ~1-2 sessions of work.
>
> | Item | Source | Status | What |
> |------|--------|--------|------|
> | **Fix `maxim.run()` TypeError** | Refinement 0a | **NOT DONE** | Flagship API call is broken. First thing in getting-started guide. |
> | **Wire 3 stub API verbs** | Refinement 0b / ASH Phase 1 | **VERIFY** | `campaign()`, `benchmark()`, `research()` — if ASH wired them, verify. If not, wire or raise `NotImplementedError` with CLI pointer. |
> | **API key fail-fast** | Refinement 1g / ASH 3f | **NOT DONE** | Missing `ANTHROPIC_API_KEY` should fail at startup, not minutes later. ~15 LOC. |
> | **Error type exports** | Refinement 0d | **PARTIAL** | 5 of 25 exception types exported. At minimum export the 7 category-level exceptions so `except maxim.ConfigurationError` works. |
>
> **Should-do but can defer to v0.2.1:**
> - Research protocol bugs (D-0a through D-0e) — `--research` is broken but power-user only. Wire `research()` as `NotImplementedError` for v0.2.0.
> - `@maxim.tool` schema inference — nice-to-have, 30 LOC.
> - Error handling audit (ASH Phase 3 remaining) — silent `except Exception: pass` blocks are tech debt, not user-facing blockers.

---

## Context

A deep review of the full codebase, documentation, plans, and publication readiness surfaced five categories of issues:

1. **Hard publication blockers** — `maxim.run()` TypeError, stub API verbs, silent error swallowing
2. **Code quality debt** — 653 bare `except Exception:` blocks, god classes, `Any` overuse
3. **Two threading bugs** — hippocampus queue race condition and flush sync
4. **Documentation gaps** — broken links, stale status claims, oversized CLAUDE.md
5. **Packaging/security gaps** — hardcoded paths, incomplete shell blocklist, license reference
6. **Data persistence gaps** — 6 files bypass atomic writes, no multi-process locking, no disk cleanup
7. **LLM robustness gaps** — OpenAI backend ignores 429 rate limits, no JSON parser tests, prompt injection surface
8. **UX edge cases** — missing API key produces late/vague error, no permission error handling, no file locking for concurrent instances
9. **API contract holes** — return types (`SimulationResult`, `RobotController`) not importable from top-level; 20+ exception types defined but not exported in `__all__`
10. **Config fragility** — malformed env vars (`MAXIM_PROXY_MAX_CONCURRENT=banana`) crash with unhandled ValueError; 40+ env vars undocumented; no config file schema validation
11. **Silent failure chains** — hippocampus swallows state snapshot errors, caller gets `AttributeError` on `None`; async capture losses invisible; daemon threads crash with `pass` or debug-level logging
12. **Composable API doc/behavior gaps** — `python-api.md` examples use wrong method signatures (recall, NAc observe); `load.hippocampus()` silently returned empty on missing file; `create.router()` mutates `os.environ` globally; `Session.observe()` returns error dicts while `get_session()` raises exceptions (inconsistent)

This plan addresses **what must be done before publish** (Phases 0-1), **what should be done** (Phases 2-3), and **what can ship in v0.2.1** (deferred items). It does NOT overlap with the GitHub Repo Management Plan or Mother Maxim Plan — those are post-publication work.

---

## Phase 0 — Hard Blockers (~4-6 hours)

These prevent publication. Do them first, in order.

### 0a. Fix `maxim.run()` TypeError

**Problem:** `publication_guide.md:63` flags this as open. `api.py` passes a wrong kwarg to `LLMWorker`. This is the flagship API call — if it's broken, the getting-started tutorial fails on the first example.

**Action:**
1. Read `src/maxim/api.py` — find the `run()` function
2. Trace the call into `LLMWorker.__init__()` — identify the mismatched kwarg
3. Fix the signature mismatch
4. Add a smoke test: `tests/unit/test_api_verbs.py::test_run_signature`
5. Verify `getting-started.md` example works

**Files:** `src/maxim/api.py`, `src/maxim/agents/llm_worker.py`
**Verify:** `python -c "import maxim; help(maxim.run)"` shows correct signature

### 0b. Wire campaign() and research() stubs — `[SKIP — ASH Phase 1]`

> **Superseded by [API Surface Hardening Plan](../archive/api_surface_hardening_plan.md) Phase 1.** That plan goes further: wires all 3 stub verbs (campaign, benchmark, research) to actual runtime, fixes the 5 research protocol bugs, and adds @maxim.tool schema inference. Warnings-only (the approach below) is insufficient — see ASH plan for rationale.

**Problem:** 2 of 13 API verbs return empty/None. Users who call them get silence.

**Options (pick one):**
- **Option A (preferred):** Wire them to actual runtime — `campaign()` delegates to `PartyDMRuntime`, `research()` delegates to `research_orchestrator`. ~100 LOC.
- **Option B:** If wiring is non-trivial, add explicit `raise NotImplementedError("campaign() will be available in v0.2.1 — use the CLI: maxim --sim path.yaml")` with a clear message. ~10 LOC.

**Action:** Try Option A first. Fall back to Option B if it requires >2 hours.

**Files:** `src/maxim/api.py`
**Verify:** `python -c "import maxim; help(maxim.campaign)"` shows real signature and docstring

### 0c. Error honesty — API surface audit

**Problem:** ~60 of 653 `except Exception:` blocks are in the API surface (api.py, router.py, lane_backends.py). Silent failures cause users to think the library is broken.

**Action:**
1. Grep for `except Exception` in these files: `api.py`, `router.py`, `lane_backends.py`, `__init__.py`, `cli.py`
2. For each, classify:
   - **Raise:** If the caller needs to know (API boundary, user-facing). Convert to `raise MaximError(...)` or re-raise with context.
   - **Log + continue:** If it's a best-effort subsystem (background capture, telemetry). Add `logger.warning(...)`.
   - **Keep silent:** Only if truly inconsequential (optional metric increment). Add a comment explaining why.
3. Target: zero silent `except Exception: pass` in the 5 API-surface files listed above.

**Files:** `src/maxim/api.py`, `src/maxim/models/language/router.py`, `src/maxim/runtime/lane_backends.py`, `src/maxim/cli.py`
**Verify:** `grep -n "except Exception" src/maxim/api.py` returns zero bare catches

### 0d. Fix missing return types in `__all__`

**Problem:** `connect()` returns `RobotController` but it's not in `_API_TYPES` in `__init__.py`. Users who write `from maxim import RobotController` get `ImportError`. (`SimulationResult` is no longer needed — `imagine()` now returns `Session`, which is already exported.)

Similarly, only 5 of 25 exception types are exported. Users who try `from maxim import ConnectionError` or `from maxim import ToolNotFoundError` fail.

**Fix:** Add missing types to `_API_TYPES` dict in `src/maxim/__init__.py:~41-54`:
```python
"RobotController": "maxim.hardware.controller",
```

For exceptions: export the full hierarchy (or at minimum the 7 category-level exceptions) so users can catch errors meaningfully. ~10 LOC.

**Files:** `src/maxim/__init__.py`
**Verify:** `python -c "from maxim import RobotController, ConnectionError, ToolNotFoundError"`

### 0e. Fix hippocampus threading bugs

**Problem 1 — Queue.Full race condition** (`hippocampus.py:~654-662`):
Between `get_nowait()` and the subsequent `put_nowait()`, another thread can fill the queue.

**Fix:** Replace the get-then-put pattern with a single blocking `put()` with a short timeout:
```python
try:
    self._capture_queue.put(request, timeout=0.1)
except queue.Full:
    logger.warning("Capture queue full, dropping oldest")
    try:
        self._capture_queue.get_nowait()
        self._capture_queue.task_done()
    except queue.Empty:
        pass
    # If this also fails, we accept the drop
    try:
        self._capture_queue.put_nowait(request)
    except queue.Full:
        pass
```

**Problem 2 — Flush sync uses unreliable `queue.empty()`** (`hippocampus.py:~671-685`):
`queue.empty()` can be stale immediately after returning.

**Fix:** Use `join()` with timeout instead of polling `empty()`:
```python
# Replace the while-not-empty polling loop with:
try:
    self._capture_queue.join()  # blocks until all items processed
except Exception:
    return False  # timeout or error
return True
```
Or keep the deadline pattern but check `unfinished_tasks` under the condition lock instead of `empty()`.

**Files:** `src/maxim/memory/hippocampus.py`
**Verify:** Add `tests/unit/test_hippocampus_threading.py` with 8-thread concurrent capture test

---

## Phase 1 — Code Quality Triage (~3-4 hours)

Not blockers, but significantly improves the published package's reliability. Do after Phase 0.

### 1a. Triage remaining except blocks (non-API) — `[SKIP — ASH Phase 3]`

> **User-facing path triage superseded by [API Surface Hardening Plan](../archive/api_surface_hardening_plan.md) Phase 3.** That plan audits api.py, bootstrap, router, hippocampus, and OpenAI backend specifically. The remaining ~560 non-user-facing blocks stay deferred to v0.2.1 as originally planned.

**Problem:** 593 remaining `except Exception:` blocks outside the API surface.

**Action:** This is too many to fix in one pass. Triage by subsystem priority:
1. **memory/** (hippocampus, atl, nac) — these manage state; silent failures corrupt data. Audit all.
2. **runtime/** (agent_loop, executor) — these run the main loop; silent failures cause stalls. Audit critical path.
3. **Everything else** — defer to v0.2.1.

**Target:** Convert the ~30 most dangerous catches (state-mutating code) from silent to logged. Leave non-critical ones for later.

**Files:** `src/maxim/memory/*.py`, `src/maxim/runtime/agent_loop.py`, `src/maxim/runtime/executor.py`

### 1b. Hardcoded sandbox paths

**Problem:** `simulation/sandbox.py` has hardcoded `/home/user` and `/home/maxim` paths in the sensitive file list. On macOS/Windows these never match, so sandbox security checks silently fail.

**Fix:** Make the sensitive file list OS-aware:
```python
_home = Path.home()
SENSITIVE_PATHS = [
    _home / ".ssh" / "id_rsa",
    _home / ".ssh",
    _home / ".env",
    _home / ".bash_history",
    ...
]
```
Keep the container-specific paths (`/home/maxim/...`) in a separate `_CONTAINER_SENSITIVE_PATHS` list used only when `runtime == "docker"`.

**Files:** `src/maxim/simulation/sandbox.py`, `src/maxim/simulation/container_runner.py`

### 1c. Shell command blocklist gaps

**Problem:** `sandbox_executor.py` uses a blacklist for shell commands. Missing: `bash -c`, `find -exec`, `eval`, `source`. Blacklists are inherently incomplete.

**Fix:** Add the missing entries to `BLOCKED_SHELL_COMMANDS`. Document in a comment that this is defense-in-depth alongside the builtins override, not a standalone security boundary. Consider adding a `_SHELL_PATTERN_BLOCKLIST` for patterns like `bash -c`, `sh -c`, etc.

**Files:** `src/maxim/utils/sandbox_executor.py`

### 1d. Replace critical `Any` annotations

**Problem:** 75+ `Any` type annotations, especially in `exec_agent.py` (`self._nac: Any`, `self._hippocampus: Any`).

**Action:** Define lightweight Protocol classes for the subsystem interfaces used by ExecAgent:
```python
class NacLike(Protocol):
    def predict(self, context: str) -> float: ...
    def record_outcome(self, ...) -> None: ...

class HippocampusLike(Protocol):
    def recall(self, query: str, k: int = 5) -> list: ...
    def capture(self, content: str, ...) -> None: ...
```

Replace `Any` with these protocols in the 3-4 files that are worst offenders. Don't boil the ocean — focus on the public-facing agent interfaces.

**Files:** `src/maxim/agents/exec_agent.py`, `src/maxim/agents/bus.py`
**Scope limit:** Only the subsystem handle annotations. Leave internal `Any` for v0.2.1.

### 1e. Atomic write bypasses (crash corruption risk)

**Problem:** 6 files write JSON directly with `open('w') + json.dump()`, bypassing the project's own `atomic_write_json()`. On crash during write, these files corrupt.

| File | Line | Data at risk |
|------|------|-------------|
| `decisions/nac.py` | ~815 | Causal model (thousands of links) |
| `utils/config.py` | ~203 | User configuration |
| `utils/web_cache.py` | ~311 | Web cache entries |
| `utils/last_run.py` | ~70 | Last run metadata |
| `utils/internet_access.py` | ~152 | Access control policy |
| `conscience/selfy.py` | ~830 | Motor history (has tmp+replace but missing fsync) |

**Fix:** Replace each `open('w') + json.dump()` with `atomic_write_json()`. For selfy.py, add `os.fsync(fp.fileno())` before the `os.replace()` call.

**Files:** Listed above
**Verify:** `grep -rn "json.dump" src/maxim/ | grep -v atomic | grep -v test` returns only non-persistence uses (logging, formatting)

### 1f. OpenAI backend rate limit handling — `[SKIP — ASH Phase 3e]`

> **Superseded by [API Surface Hardening Plan](../archive/api_surface_hardening_plan.md) Phase 3e.**

**Problem:** `openai_backend.py` uses linear backoff (0.5s, 1.0s) for ALL errors including 429 rate limits. The Anthropic backend correctly multiplies by 4x for 429s. OpenAI backend will hammer the API on rate limits.

**Fix:** Port the rate-limit-aware backoff from anthropic_backend.py:
```python
if _is_rate_limit_error(e):
    backoff = min(backoff * 4, 30.0)
```

**Files:** `src/maxim/models/language/openai_backend.py`

### 1g. API key validation at startup — `[SKIP — ASH Phase 3f]`

> **Superseded by [API Surface Hardening Plan](../archive/api_surface_hardening_plan.md) Phase 3f.**

**Problem:** When a user runs `maxim --language-model claude-sonnet` without `ANTHROPIC_API_KEY`, the system warns but continues. The error surfaces minutes later during inference with a confusing message.

**Fix:** In `cli.py`, after model selection and before entering the main loop, validate that the required API key exists:
```python
if profile.get("cloud") and not _has_api_key_for(profile):
    sys.exit(f"Error: {model} requires {key_name}. Set it with: export {key_name}=...")
```

**Files:** `src/maxim/cli.py`

### 1h. Permission error handling for ~/.maxim/

**Problem:** `paths.py:data_home()` calls `mkdir(parents=True, exist_ok=True)` with no exception handling. If the user lacks write permission, they get a raw `PermissionError` stack trace.

**Fix:** Wrap the mkdir call:
```python
try:
    base.mkdir(parents=True, exist_ok=True)
except PermissionError:
    sys.exit(
        f"Cannot create data directory: {base}\n"
        f"Fix: Check permissions, or set MAXIM_DATA_HOME to a writable path."
    )
```

**Files:** `src/maxim/utils/paths.py`

### 1i. Env var parsing crashes on invalid input

**Problem:** Several env vars are parsed with bare `int(os.environ.get(...))` — no try/except. A user typo like `MAXIM_PROXY_MAX_CONCURRENT=banana` causes an unhandled `ValueError` crash.

Known locations:
- `runtime/leader_proxy.py:~1105` — `MAXIM_PROXY_MAX_CONCURRENT`
- `runtime/lane_backends.py:~895` — `MAXIM_AUTO_SPAWN_PORT`
- `runtime/lane_backends.py:~946` — `MAXIM_AUTO_SPAWN_N_CTX`
- `runtime/lane_backends.py:~951` — `MAXIM_AUTO_SPAWN_TIMEOUT_S`
- `runtime/bootstrap.py:~368` — `MAXIM_COMMS_PORT`

**Fix:** Use the existing `_as_int()` / `_as_float()` helpers from `models/language/config.py`, or wrap each in try/except with a warning and sensible default. ~15 LOC total.

**Files:** Listed above

### 1j. Silent failure chains in hippocampus state capture — `[SKIP — ASH Phase 3d]`

> **Superseded by [API Surface Hardening Plan](../archive/api_surface_hardening_plan.md) Phase 3d.**

**Problem:** `hippocampus.py:~536-542` catches `Exception` on `state.snapshot()` and sets `state_snapshot = None`. The caller at line ~548 then does `state_snapshot.get("mode")` which raises `AttributeError: 'NoneType' object has no attribute 'get'`. The user sees an error pointing to the wrong place — the real failure (snapshot) was silently swallowed.

**Fix:** Either let the snapshot exception propagate, or guard the `.get()` call:
```python
mode = state_snapshot.get("mode") if state_snapshot else None
```
Apply the same pattern to all places that use a maybe-None result from a caught exception. Audit `hippocampus.py` for this pattern (~5 instances).

**Files:** `src/maxim/memory/hippocampus.py`

### 1k. Daemon threads with bare `except: pass` — `[SKIP — ASH Phase 3]`

> **Superseded by [API Surface Hardening Plan](../archive/api_surface_hardening_plan.md) Phase 3 (user-facing error audit).**

**Problem:** Some daemon threads catch all exceptions with `pass` — no log, no alert. When the thread dies, the system degrades silently.

Known locations:
- `conscience/media_loop.py:~608-615` — two bare `except Exception: pass` blocks in media capture
- `runtime/heartbeat.py:~109` — error logged at `debug` level (invisible at normal verbosity)

**Fix:** Replace `pass` with `logger.warning(...)` so thread failures are visible. Elevate heartbeat errors from `debug` to `warning`. ~10 LOC.

**Files:** `src/maxim/conscience/media_loop.py`, `src/maxim/runtime/heartbeat.py`

### 1l. `create.router()` mutates `os.environ` globally — ALREADY FIXED

**DONE** — `create.py:router()` already uses a scoped try/finally pattern that saves original env values, sets them for `build_primary_router()`, and restores originals in the finally block. No action needed.

### 1m. Wire store protocols into bio-system constructors (Mother Maxim on-ramp) — DEFERRED to v0.2.1

**Deferred because:** The store protocol interface (`save(list[dict])` / `load() → list[dict]`) is fundamentally different from the bio-system save/load pattern. Each bio-system's `save()` serializes complex state (version headers, associative graph, context indices, compressed memories) into a single JSON blob. Each `load()` mutates internal state and handles version migration. The File*Store implementations just do simple list-of-dicts I/O.

Wiring would require either:
1. Moving ~300 LOC of serialization logic from each bio-system into File*Store — high regression risk
2. Thin adapter wrappers that call existing save/load — complexity with no practical benefit until M-1

**What ships now:** Store protocols defined + exported (1n done). All bio-system persistence uses `atomic_write_json()` (1e done). The interface is locked for users.
**What ships in v0.2.1:** Redesigned store protocol that accounts for versioning, graph state, and the mutate-self contract. Designed alongside Mother Maxim M-1 database backend.

### 1n. Export store protocols and Concept from memory module

**Problem:** `maxim.memory.__init__.py` does not export store protocols (`EpisodicStore`, `CausalStore`, `SemanticStore`, `FileEpisodicStore`, `FileCausalStore`, `FileSemanticStore`) or `Concept`. Users and Mother Maxim must import from internal paths (`maxim.memory.store`, `maxim.memory.semantic_types`).

**Fix:** Add imports and `__all__` entries in `src/maxim/memory/__init__.py`:
```python
from maxim.memory.store import (
    EpisodicStore, CausalStore, SemanticStore,
    FileEpisodicStore, FileCausalStore, FileSemanticStore,
)
from maxim.memory.semantic_types import Concept
```

**Files:** `src/maxim/memory/__init__.py`
**Scope:** ~10 LOC
**Verify:** `python -c "from maxim.memory import EpisodicStore, Concept, FileCausalStore"`

---

## Phase 1.5 — Data Directory Migration (~1-2 hours)

Fully commit to `~/.maxim/` as the single data home. Eliminate the legacy `data/` directory in the repo root.

### 1.5a. Migrate `data/util/` → `~/.maxim/util/`

**Problem:** `data/util/llm.json` and `data/util/active_llm_model.txt` are the only actively-read files in `data/`. Both the leader and peer read from here.

**Fix:**
1. Update `models/language/config.py` to check `~/.maxim/util/llm.json` first, fall back to `data/util/llm.json` for backward compat
2. Update `runtime/lane_backends.py` (`_read_persisted_model`/`_write_persisted_model`) to use `~/.maxim/util/`
3. Copy existing `data/util/` files to `~/.maxim/util/` on first access (one-time migration)
4. Update `maxim peer llm` hot-swap to write to new location

**Files:** `src/maxim/models/language/config.py`, `src/maxim/runtime/lane_backends.py`

### 1.5b. Move `data/motion/` → `src/maxim/_data/motion/`

**Problem:** `data/motion/default_actions.json` is the only git-tracked file in `data/`. It's bundled seed data, not user data.

**Fix:** Move to `src/maxim/_data/motion/`, update any references. Already included by `package-data` glob `_data/**/*`.

### 1.5c. Migrate `data/models/` → `~/.maxim/models/`

**Problem:** Downloaded GGUFs (4-14GB each) live in `data/models/`. Can't just delete — re-downloading takes hours.

**Fix:**
1. Update `download_models.sh` to target `~/.maxim/models/`
2. On first run, if `~/.maxim/models/` is empty but `data/models/` has files, symlink or copy
3. Update all model path resolution in `config.py` to use `~/.maxim/models/`

### 1.5d. Clean up `data/` and gitignore

After migration is verified:
1. Delete: `data/sim_sandbox/` (~200 dirs), `data/agents/MagicMock_*` (~55 dirs), `data/sim_reports/`, `data/sim_orchestrator/`, `data/short_term_memory/`, `data/runtime/`
2. Replace granular `.gitignore` entries with single `data/` line
3. Add `data/` deprecation note: "Legacy directory. All runtime data lives in ~/.maxim/. Safe to delete after running maxim once (migrates automatically)."
4. Clean test artifacts from `~/.maxim/agents/` (test_*, iso_*, pool_*, cross_*, vs_* dirs created by composable API tests)

### 1.5e. Bugs found during testing (fix alongside migration)

Two bugs identified during DM campaign testing:

1. **Respond loop** — AUT enters infinite `respond` loop after dice roll results (61 actions on arena Turn 4). Needs a consecutive same-tool cap in the agent loop. Add `max_consecutive_same_tool: int = 5` to LoopController config; after N identical tool calls, force-advance to next turn.

2. **Ctrl+C during DM campaign crashes without saving** — `KeyboardInterrupt` in `dm_runtime.run()` propagates through `bridge.send_and_wait()` → `time.sleep()` and bypasses the report/save code in the orchestrator. Fix: wrap `dm.run()` in `try/except KeyboardInterrupt` at line ~1167 of orchestrator.py, set `finish_reason = "cancel"`, and continue to the save/report section.

---

## Phase 2 — Test Depth (~2-3 hours)

### 2a. Add threading tests for memory subsystems

**Problem:** Zero concurrency tests for hippocampus capture, NAc recording, or RWLock under contention.

**Action:** Add `tests/unit/test_hippocampus_threading.py`:
- 8 threads doing concurrent `capture()` calls — verify no data loss
- Concurrent `recall()` during `capture()` — verify no deadlock (timeout after 5s)
- Queue full scenario — verify graceful degradation
- Flush under load — verify all items processed

Add `tests/unit/test_nac_threading.py`:
- Concurrent `record_outcome()` — verify counter consistency
- Re-entrant lock behavior — verify no deadlock

**Scope:** ~150 LOC of tests total.

### 2b. Upgrade smoke tests to behavioral tests

**Problem:** Average 1.9 assertions/test. Many just check `is not None`.

**Action:** Pick the 20 weakest test files (those averaging <1.5 assertions/test) and add 1-2 behavioral assertions each. Example:
```python
# Before (smoke test):
def test_agent_factory():
    agent = factory.create("npc")
    assert agent is not None

# After (behavioral):
def test_agent_factory():
    agent = factory.create("npc")
    assert agent is not None
    assert agent.hippocampus is not None  # subsystems wired
    assert agent.name == "npc"
    assert agent.can_recall()  # functional check
```

**Scope:** ~200 LOC of test additions. Not a full rewrite — just adding teeth to existing tests.

### 2c. Add JSON parser tests

**Problem:** The 4-stage JSON repair pipeline (`json_parser.py`) has zero unit tests. This is the primary defense against LLM output malformation.

**Action:** Add `tests/unit/test_json_parser.py`:
- Stage 1: Valid JSON passes through unchanged
- Stage 2: Control characters in strings are escaped
- Stage 3: Trailing commas, single quotes, unquoted keys repaired
- Stage 4: Truncated JSON (unclosed braces/brackets) closed correctly
- Edge case: Deeply nested truncation
- Edge case: Array vs object confusion on truncation
- Compliance counter tracking (first-try vs repaired counts)

**Scope:** ~100 LOC of tests.

---

## Phase 3 — Docs & Packaging (~2 hours)

### 3a. Fix broken links and stale status — `[SKIP — ASH Phase 5a]`

> **Superseded by [API Surface Hardening Plan](../archive/api_surface_hardening_plan.md) Phase 5a.**

| Fix | File | Effort |
|-----|------|--------|
| Change `usage-guide.md` → `cli-reference.md` | `docs/user/api-quickstart.md:97` | 1 line |
| Document `list_models()` | `docs/user/python-api.md` | ~10 lines |
| Update `maxim.run()` status to FIXED (after 0a) | `docs/publication_guide.md:63` | 1 line |
| Update campaign()/research() status (after 0b) | `docs/publication_guide.md:69` | 1 line |
| Gate getting-started.md examples if run() deferred | `docs/user/getting-started.md:114-146` | 5 lines |
| Add AGPL warning for yolo extra | `docs/user/getting-started.md` | 2 lines |
| Clarify "no cloud dependency" = optional | `README.md:5` | Rephrase to "no cloud dependency required" |
| Fix `hippo.recall("wolf", limit=3)` → `recall(query="wolf", limit=3)` | `docs/user/python-api.md` | ~5 lines |
| Fix NAc examples: `record_event` doesn't create links, show `observe()` | `docs/user/python-api.md` | ~10 lines |

### 3b. Trim CLAUDE.md — `[SKIP — ASH Phase 5b]`

> **Superseded by [API Surface Hardening Plan](../archive/api_surface_hardening_plan.md) Phase 5b.**

**Problem:** 536 lines covering 22 sections. Too much for an agent orientation file.

**Action:** Extract these sections to dedicated docs (link from CLAUDE.md):
- Project structure table → already in ARCHITECTURE.md, remove duplicate
- Testing section → `docs/testing.md` or `CONTRIBUTING.md`
- Simulation reports → reference `docs/user/simulation.md`
- Research protocol details → reference `docs/user/simulation.md`
- Python API package management → reference `docs/publication_guide.md`
- Active initiatives → link to `docs/plans/future_plans.md`

**Target:** CLAUDE.md under 250 lines. Keep: overview, required checks, architectural invariants, doctor maintenance, sim running tips, env vars, quick-reference table.

### 3c. Packaging fixes

| Fix | File | Effort |
|-----|------|--------|
| Change `license = {text = "Apache-2.0"}` to `license = {file = "LICENSE"}` | `pyproject.toml:11` | 1 line |
| Add upper bound: `pyyaml>=6.0,<7.0` | `pyproject.toml:27` | 1 line |
| Verify `maxim-diagnostics` entry point imports cleanly | Manual test | 5 min |
| Verify `.env` gitignore covers `data/**/.env` | `.gitignore` | 1 line |

### 3d. htmls-guides/ housekeeping

**Problem:** 23 Jinja2 templates in the repo root look like documentation but aren't usable standalone.

**Action:** Add a one-line `htmls-guides/README.md`: "Jinja2 source templates for dennyschaedig.com. Rendered by the site's build system — not standalone HTML."

---

## Phase 4 — Publish (~1 hour)

After Phases 0-3 are complete:

1. Update `publication_guide.md` — mark all blocker items as FIXED with dates
2. Run the full pre-publication checklist from `publication_guide.md` (sims, tests, clean import)
3. Bump version if needed (0.2.0 unless API surface changed, then 0.2.1)
4. `python -m build && twine check dist/*`
5. `twine upload --repository testpypi dist/*` → test install in clean venv
6. `twine upload dist/*` → verify real install
7. Tag release: `git tag v0.2.0 && git push origin v0.2.0`

---

## Pecking Order Graph Prep (POG-0) — weave into publication

Three small, additive items that position the internal types for the [Pecking Order Graph](pecking_order_graph_plan.md) without changing any public API. These are optional for v0.2.0 but save rework post-publication.

| Item | What | LOC | Risk |
|------|------|-----|------|
| **POG-0a** | Add `NodeLoad` (gpu_util, queue_depth, ram_pressure, thermal) to `RuntimeCapabilities` | ~30 | None — additive fields, not in public API |
| **POG-0b** | Add optional `parent_id: str | None` to `AgentIdentity` | ~10 | None — optional field, backward-compatible serialization |
| **POG-0c** | Define `GateResult` and `EdgeCapacity` shared types in new `mesh/gate_types.py` | ~60 | None — new file, no behavior changes |
| **POG-0d** | Define `SpatialContext` frozen dataclass in new `memory/spatial.py` | ~40 | None — type only, no integration. Locks `Perception.location` serialization format |

**Why do these now:** `RuntimeCapabilities` and `AgentIdentity` are serialized in heartbeat messages. Adding fields post-publication means dealing with version skew between nodes running different versions. Adding them now (as optional/defaulted fields) means the wire format is stable from v0.2.0 onward.

**What does NOT need to happen before publication:** The graph itself (POG-1+), registration protocol (POG-2), cascades (POG-3), and Mother integration (POG-4) are all post-publication. The public Python API (`maxim.api`) is not affected by any POG work — it's all internal mesh/runtime infrastructure.

---

## Deferred to v0.2.1 (not blocking publication)

These are real issues but won't cause user-facing failures on day 1:

| Item | Why deferred |
|------|-------------|
| God class refactoring (agent_loop, cli, exec_agent, bus, orchestrator) | Now tracked in [Module Compartmentalization Plan](../archive/module_compartmentalization_plan.md). Executes after API Surface Hardening, before publish. |
| Remaining ~560 except blocks in non-API code | Internal resilience, not user-facing |
| Full `Any` → Protocol migration | Only affects type checker users, not runtime |
| `--list-models` CLI flag | Nice-to-have, not blocking |
| Switch shell blocklist to whitelist | Defense-in-depth already exists; hardening, not a hole |
| future_plans.md scope trimming | Internal planning doc, not user-facing |
| Windows path separator consistency in tools | Edge case; most users are Linux/macOS |
| `[all]` extras completeness (missing llm-torch, training, semantic) | Documented as intentional (AGPL/heavy deps) |
| Remaining 520 smoke tests → behavioral tests | Quality improvement, not a regression |
| Multi-process file locking (fcntl) for ~/.maxim/ | Uncommon scenario; atomic writes prevent corruption, just not data loss |
| Disk space cleanup/rotation for sim_reports, sessions | No user impact on day 1; long-term accumulation |
| Prompt injection sanitization in simulation_generator | Only affects sim scenarios, not production agent loop |
| Context window buffer (add safety margin to token estimates) | Conservative 3-char estimate already overestimates |
| Encryption at rest for hippocampus data | Privacy improvement, not a functional gap |
| Hippocampus version migration (auto-upgrade old schemas) | Edge case; clear error is acceptable for v0.2.0 |
| Per-call token ceiling in energy tracker | Session ceiling exists; per-call is defense-in-depth |
| Silent fallback when all LLM providers unavailable | Returns empty string; needs proper error, but rare scenario |
| Full exception hierarchy export (all 25 types) | 7 category-level exports is sufficient for v0.2.0; users rarely need leaf types |
| Config file schema validation (warn on unknown keys) | Prevents silent typos in llm.json, but power-user scenario |
| Document all 40+ undocumented MAXIM_* env vars | Real gap, but internal vars — users mostly use CLI flags |
| Config precedence documentation (env vs file vs CLI) | Important for power users, not day-1 blocker |
| Remove dead `AUTIntrospector` alias | Not exported, zero user impact, cleanup |
| Config hot-reload support | Feature request, not a bug |
| `Session.observe()` error dicts vs `get_session()` exceptions | Inconsistent error strategy, but both are informative; standardize in v0.2.1 |

---

## User Journey — How People Will Interact with pymaxim

This section describes the full experience from a user's perspective: discovery, install, first use, ongoing use, and where the current gaps are.

### Discovery

Users find `pymaxim` on PyPI (https://pypi.org/project/pymaxim/). The landing page is rendered from `README.md` — currently ~600 lines with feature overview, install instructions, CLI reference, and architecture glossary. The package name is `pymaxim` (because `maxim` was taken on PyPI), but the import name is `maxim`.

**Gap:** The README leads with robot/hardware language ("hardware-agnostic cognitive framework"), which may confuse the majority of users who will never connect a robot. The headless/simulation path is the primary use case and should be front-and-center.

### Installation

Users install with pip. The core package has only 5 dependencies (`numpy`, `scipy`, `pyyaml`, `json-repair`, `rich`):

```bash
# Minimal — cognitive architecture + simulation, no LLM
pip install pymaxim

# With a cloud LLM backend (most common for getting started)
pip install pymaxim[llm-anthropic]

# With local LLM inference
pip install pymaxim[llm-server]

# Everything (minus AGPL-licensed yolo)
pip install pymaxim[all]
```

After install, users get two console commands:
- `maxim` — the main CLI entry point
- `maxim-diagnostics` — Reachy-specific hardware diagnostics (should be gated behind `[reachy]` extra)

They can also use `python -m maxim` which works via `__main__.py`.

**Gaps:**
- `maxim-diagnostics` installs for everyone but only works with Reachy hardware. Confusing for non-robot users.
- The `[all]` extra is missing `training`, `tts`, `semantic`, and `database`. A user doing `pip install pymaxim[all]` won't get everything.
- `getting-started.md` still shows `git clone` + `pip install -e .` as the primary install path, not `pip install pymaxim`.
- Getting-started references `pip install -e ".[llm]"` but the extras group is actually called `llm-llama` in pyproject.toml.

### First Use — CLI Path

The most likely first interaction is the CLI:

```bash
# Check environment
maxim doctor

# Run a simulation (generative mode)
maxim --sim "test memory recall under interference"

# Run a DM campaign
maxim --sim scenarios/campaigns/heist_v1.yaml

# Run with a specific model
maxim --language-model claude-sonnet --sim "test safety"
```

`maxim doctor` is a strong first-touch experience — it detects the platform, checks GPU/RAM/disk, validates model availability, and prints actionable fix hints. This should be prominently recommended as step 1.

**Gaps:**
- If the user runs `maxim --language-model claude-sonnet` without `ANTHROPIC_API_KEY`, the error surfaces late and is vague. Should fail fast at startup with: `"Error: claude-sonnet requires ANTHROPIC_API_KEY. Set it with: export ANTHROPIC_API_KEY=sk-..."` (Phase 1g addresses this).
- `maxim --sim "goal"` requires a working LLM. A user who does `pip install pymaxim` and immediately runs `maxim --sim "test"` gets a confusing failure because no LLM is configured. Need a clear error: `"No LLM configured. Try: maxim --language-model claude-sonnet --sim 'test' (requires ANTHROPIC_API_KEY)"`.

### First Use — Python API Path

```python
import maxim  # Fast — no heavy imports

# Step 1: Check environment (works immediately, no LLM needed)
report = maxim.diagnose()
print(report.summary())

# Step 2: Discover available models
models = maxim.list_models()
for m in models["cloud"]:
    print(f"{m.name} (requires {m.api_key_env})")

# Step 3: Run a simulation
result = maxim.imagine(
    goal="test memory recall",
    persona="cooperative",
    model="claude-sonnet",
)

# Step 4: Inspect cognitive state after the sim
memories = maxim.observe("memory")
causal = maxim.observe("causal")
```

The verb-based API is clean and discoverable. `diagnose()` and `list_models()` work without any LLM setup, making them good entry points. `observe()` lets users inspect persisted state after a session without running a new one.

**Gaps (current):**
- `maxim.run()` has a TypeError — the flagship "start the agent" call is broken (Phase 0a).
- `maxim.campaign()` returns an empty `CampaignResult` — it loads the YAML but doesn't execute the campaign (Phase 0b).
- `maxim.benchmark()` returns an empty `BenchmarkResult` — complete stub (Phase 0b).
- `maxim.research()` returns an empty `ResearchResult` — complete stub (Phase 0b).
- None of the stubs warn or raise — they silently return empty objects. A user has no way to know they didn't work.
- The `@maxim.tool` decorator doesn't infer `input_schema` from function type annotations, so the LLM never knows what arguments the tool accepts.

### Ongoing Use — Power Users

Power users will:

1. **Write custom tools** — extend the agent's capabilities:
   ```python
   @maxim.tool
   def query_database(sql: str) -> str:
       """Execute a read-only SQL query."""
       return db.execute(sql)
   ```

2. **Create custom personas** — shape simulation behavior:
   ```python
   maxim.register_persona(
       name="security_auditor",
       context_prompt="You are testing for security vulnerabilities...",
       max_initiative=0.9,
   )
   ```

3. **Subscribe to events** — build monitoring/dashboards:
   ```python
   maxim.on("pain_signal", lambda ev: slack_alert(f"Pain: {ev}"))
   maxim.on("memory_capture", lambda ev: log_to_db(ev))
   ```

4. **Run benchmarks** — compare models on cognitive tasks:
   ```bash
   maxim --sim benchmark --models mistral-7b,qwen2.5-14b \
       --campaign scenarios/benchmarks/cognitive_suite.yaml
   ```

5. **Write YAML campaigns** — design structured experiments:
   ```yaml
   # my_experiment.yaml
   name: "Test fear response"
   turns:
     - percept: "A stranger approaches with a knife"
       salience: 0.9
   ```

6. **Inspect bio-subsystems programmatically** — post-hoc analysis:
   ```python
   state = maxim.observe()           # Full system summary
   maxim.observe("memory", keyword="danger", limit=5)
   maxim.observe("causal")           # NAc causal link graph
   ```

### Ongoing Use — Researcher Path

Researchers use the experiment pipeline:

```bash
# Run a structured experiment with pre/post analysis
maxim --sim "hippocampal recall under interference" \
    --research \
    --campaign scenarios/experiments/hippocampal_recall_short.yaml \
    --language-model claude-sonnet \
    --aut-model mistral-7b

# Results saved to data/sim_reports/{session_id}/
#   report.json    — metrics + LLM analysis
#   actions.jsonl  — full action trace
#   aut_hippocampus.json — memory state
#   aut_nac.json   — causal model
```

The dual-LLM setup (Claude orchestrates, local model experiences) lets researchers study how different models form memories and learn causally. The `--research` flag adds Writer + Reviewer agents that auto-generate experiment papers.

**Gap:** The Python API equivalent (`maxim.research()`) is a stub — CLI path works, API path doesn't.

### Ongoing Use — Robot Path

```python
# Connect to hardware
robot = maxim.connect("reachy_mini")

# Run with embodiment
maxim.run(model="mistral-7b", robot="reachy_mini")
```

Third-party robots register via the `maxim.robots` entry-point group — no core code changes needed. Currently only Reachy Mini and a simulated controller are available.

### Error Experience

When things go wrong, users see custom exceptions:

```python
from maxim import ConfigurationError, ModelError, ToolExecutionError

try:
    maxim.run(model="claude-sonnet")
except ConfigurationError as e:
    # "Model 'claude-sonnet' requires ANTHROPIC_API_KEY.
    #  Fix: export ANTHROPIC_API_KEY=<your-key>"
    print(e)
```

16 exception types organized by category (connection, model, tool, memory, planning, hardware, runtime, config). All include a `context` dict with debug info.

**Gap:** Many internal errors are still swallowed by bare `except Exception: pass` blocks (~60 in the API surface). Users see silent failures instead of actionable errors (Phase 0c).

### Data Persistence

All user data lives at `~/.maxim/`:
```
~/.maxim/
  memory/           # Hippocampus + NAc + ATL persisted state
  sessions/         # Per-session logs
  benchmarks/       # Benchmark results
  components/       # User-added SEM components
  config/           # User configuration
  data/sim_reports/ # Simulation reports
```

The `MAXIM_DATA_HOME` env var overrides the location. Persistence uses `atomic_write_json()` (fsync + tmp + replace) for crash safety — though 6 files still bypass this (Phase 1e).

### What v0.2.0 Delivers vs. What It Promises

| Feature | CLI | Python API | Status |
|---------|-----|------------|--------|
| Environment diagnostics | `maxim doctor` | `maxim.diagnose()` | **Works** |
| Model discovery | `maxim --list-models` | `maxim.list_models()` | **Works** |
| Agentic loop | `maxim --language-model X` | `maxim.run()` | **CLI works, API broken** |
| Simulation | `maxim --sim "goal"` | `maxim.imagine()` | **Both work** |
| DM campaigns | `maxim --sim path.yaml` | `maxim.campaign()` | **CLI works, API stub** |
| Benchmarks | `maxim --sim benchmark` | `maxim.benchmark()` | **CLI works, API stub** |
| Research protocol | `maxim --sim X --research` | `maxim.research()` | **CLI broken (no reports generated), API stub** |
| Robot connection | `maxim` (auto-detects) | `maxim.connect()` | **Both work** |
| Custom tools | N/A | `@maxim.tool` / `register_tool()` | **Works (no schema inference)** |
| Custom personas | N/A | `register_persona()` | **Works** |
| Event hooks | N/A | `maxim.on()` | **Registered but not wired to bus** |
| Bio-state inspection | N/A | `maxim.observe()` | **Works (from persisted state)** |

**Key takeaway:** The CLI is the more complete path. The Python API has the right shape and 8 of 13 verbs work, but the 3 most interesting verbs (campaign, benchmark, research) are stubs and the flagship verb (run) has a bug.

---

## Improvements from PyPI Review (2026-04-08)

Items identified during a comprehensive PyPI readiness review, integrated into the existing phase structure:

### Phase 0 additions (blockers)

- **0e. Add warnings to stub API verbs** — `[SKIP — ASH Phase 1]` — ASH plan goes further: wires them to actual runtime instead of just warning.

### Phase 3 additions (packaging)

- **3e. Fix `[all]` extras** — Add `training`, `tts`, `semantic`, `database` to the `[all]` group in `pyproject.toml`. Currently missing. 1-line fix.
- **3f. Gate `maxim-diagnostics` entry point** — Either remove from `[project.scripts]` or document it's Reachy-specific. Non-robot users shouldn't have a broken command in their PATH.
- **3g. Guard `get_version_info()` git calls** — Check for `.git` directory existence before shelling out to `git rev-parse`. Installed-from-PyPI users won't have `.git`, and the two subprocess calls are wasted. ~3 LOC.
- **3h. Update getting-started.md install path** — `[SKIP — ASH Phase 5c]` — Covered by README overhaul in ASH plan.
- **3i. Infer `input_schema` in `@maxim.tool` decorator** — `[SKIP — ASH Phase 1d]` — Covered by ASH plan Phase 1d.

### Phase 4 additions (publish)

- **4b. Add CI publish workflow** — `.github/workflows/publish.yml` triggered on GitHub release tags. Build wheel, validate, upload via trusted publishing. Prevents manual `twine upload` drift. ~50 LOC YAML.
- **4c. TestPyPI dry run** — Must complete before real publish. Install in a clean venv, verify `import maxim`, `maxim --help`, `maxim.diagnose()`, and `python -m maxim` all work.

---

## Post-Publication Enhancements (v0.2.1+)

Improvements that make the published package more useful without blocking the initial release.

### A. Fold `maxim-diagnostics` into `maxim doctor --hardware`

**Current state:** `maxim-diagnostics` (`utils/reachy_diagnostics.py`) is a standalone Reachy Mini hardware tester — pings the robot, probes ports 7447 (Zenoh motors), 8000 (dashboard), 8443 (WebRTC). It installs as a global console command for all users via `[project.scripts]`, even though it only works with Reachy hardware.

**Problem:** Two separate diagnostic tools is confusing. Non-robot users get a broken `maxim-diagnostics` command. Robot users don't know to run it because `maxim doctor` doesn't mention hardware.

**Proposal:** Fold Reachy port checks into `maxim doctor` as an optional hardware section:

```bash
# Auto-detect: if reachy SDK is installed, include hardware checks
maxim doctor

# Explicit hardware mode (checks all robot ports + services)
maxim doctor --hardware reachy
maxim doctor --hardware reachy --ip 192.168.1.42

# Future: other robot types
maxim doctor --hardware atlas
maxim doctor --hardware spot
```

**Implementation:**
1. Add a `_hardware_checks(info, robot_type, robot_ip)` section in `doctor/checks.py` that runs port connectivity tests (same logic as reachy_diagnostics.py, ~40 LOC).
2. Auto-detect: if `reachy_mini` is importable, add hardware checks to the default doctor run with auto-discovered IP.
3. `--hardware <type>` flag forces hardware checks even if the SDK isn't installed (useful for diagnosing "why can't I import reachy").
4. Keep `reachy_diagnostics.py` as-is for backward compat but deprecate the entry point. Remove from `[project.scripts]` in v0.3.0.
5. Each robot type (future: Atlas, Spot) registers its own checks via the `maxim.robots` entry-point group — the doctor discovers them the same way `connect()` discovers controllers.

**Why this is better:**
- One diagnostic tool, not two. `maxim doctor` becomes the single entry point for "is my setup working?"
- Hardware checks get the same platform-aware fix hints, `--json` output, and `--retry` loop that software checks already have.
- When Embodiment Hardware Adapter ships (future plan), each robot plugin can register its own doctor checks alongside its controller.

~80 LOC in doctor/checks.py + 20 LOC in doctor/cli.py.

### B. Composable Subsystem API — Power User Access

**Current state:** All bio-subsystems (Hippocampus, NAc, ATL, SCN, AngularGyrus, Embodiment, SEM) are fully instantiable standalone via direct imports. AgentFactory creates isolated agents, AgentPool orchestrates multiple agents. But **none of this is exposed through the public API** — users must know the internal module paths.

**Problem:** Power users who want to use individual components (e.g., just the Hippocampus for a research project, or just AgentFactory for multi-agent orchestration) have to read source code to discover what's available. The public API only offers monolithic verbs (`run`, `imagine`).

**Proposal:** Add a `maxim.create` namespace for composable access:

```python
import maxim

# ── Individual bio-subsystems ──────────────────────────────
hippo = maxim.create.hippocampus(persistence_path="/tmp/test")
hippo.capture(perception="saw a wolf", action="ran away", outcome="survived")
memories = hippo.recall("wolf", k=3)

nac = maxim.create.nac()
nac.observe("saw_wolf", "ran", "survived", valence=1.0, delta_seconds=5)
prediction = nac.predict("saw_wolf", "ran")

atl = maxim.create.atl(persistence_path="/tmp/test")
scn = maxim.create.scn()
angular = maxim.create.angular_gyrus()

# ── SEM entities ───────────────────────────────────────────
entity = maxim.create.entity("npcs/guard", name="captain_aldric")
body = maxim.create.embodiment(entity)
readings = body.read_all()

# ── Single agent (isolated subsystems, no loop) ───────────
agent = maxim.create.agent("scout", personality="cautious and observant")
agent.hippocampus.capture(...)
agent.nac.predict(...)
agent.shutdown()

# ── Multi-agent pool ──────────────────────────────────────
pool = maxim.create.agent_pool()
pool.add(maxim.create.agent("guard", personality="stern"))
pool.add(maxim.create.agent("merchant", personality="cunning"))
result = pool.run_turn("guard", percept="A stranger approaches.")
pool.shutdown()

# ── LLM router (for inference without the full loop) ──────
router = maxim.create.router(model="mistral-7b")
response = router.generate("What should I do?")
```

**Implementation:** Add `src/maxim/create.py` as a factory module with thin wrappers:

```python
def hippocampus(*, persistence_path=None, config=None):
    from maxim.memory.hippocampus import Hippocampus, HippocampusConfig
    cfg = config or HippocampusConfig(persistence_path=persistence_path)
    return Hippocampus(cfg)

def agent(name, *, personality=None, remembers=True, learns=True):
    from maxim.runtime.agent_factory import AgentFactory, AgentConfig
    factory = AgentFactory()
    return factory.create_agent(AgentConfig(agent_id=name, ...))

def entity(template_ref, *, name=None):
    from maxim.embodiment.component_registry import ComponentRegistry
    registry = ComponentRegistry()
    return registry.instantiate(template_ref, name=name)
# ... etc
```

Wire into `__init__.py` via lazy `__getattr__` so `maxim.create.hippocampus()` works.

**Why this matters:**
- Researchers can use Hippocampus or NAc in Jupyter notebooks without understanding the full architecture.
- Multi-agent experiments become first-class (create agents, run turns, inspect memory) without the CLI.
- SEM entity design becomes accessible (`maxim.create.entity("npcs/guard")`) instead of requiring 3 internal imports.
- The `router` factory gives users inference without the full agent loop — the most-requested missing piece.

~150 LOC for the factory module + ~20 LOC for lazy loading. Post-publication because it's additive API surface, not a fix.

### C. Benchmark Flexibility — Optional Models + LLM Availability Warning

**Current state:** `BenchmarkRunner` requires `models: list[str]` and `suite_path: str` — both mandatory. The CLI enforces this with hard errors:

```
Error: --models is required for --sim benchmark
Error: --campaign is required for --sim benchmark
```

If a model isn't available (no API key, no local server), the benchmark fails mid-run with whatever error the backend throws — no pre-flight check.

**Problems:**
1. **Models should be optional.** If the user has a model already loaded (via `--language-model` or env var), the benchmark should just use it. Requiring `--models` when there's only one model is friction.
2. **No LLM availability pre-flight.** The benchmark starts, runs for minutes, then fails when it hits a model that can't respond. Should check before running.
3. **No "use what's available" mode.** A user with both a local model and a cloud key should be able to say "benchmark whatever I have" without listing models manually.

**Proposal:**

```bash
# Current (still works)
maxim --sim benchmark --models mistral-7b,claude-sonnet --campaign suite.yaml

# New: use the currently loaded model (single-model benchmark)
maxim --sim benchmark --campaign suite.yaml
# → Uses $MAXIM_LLM_PROFILE or --language-model, warns if none set

# New: auto-discover available models
maxim --sim benchmark --models auto --campaign suite.yaml
# → Checks local server + API keys, benchmarks everything that responds

# New: Python API with defaults
result = maxim.benchmark(suite="cognitive")  # Uses loaded model
result = maxim.benchmark(models=["auto"], suite="cognitive")  # Auto-discover
```

**Implementation changes:**

1. **`BenchmarkRunner.__init__`**: Make `models` optional (`models: list[str] | None = None`). If `None`, read from `$MAXIM_LLM_PROFILE`. If that's also unset, raise `ConfigurationError("No models specified and no LLM configured. Pass --models or set --language-model.")`.

2. **Auto-discovery** (`models=["auto"]`): New method `_discover_available_models()`:
   ```python
   def _discover_available_models(self) -> list[str]:
       """Probe which models can actually respond right now."""
       from maxim.models.language.config import _BUILTIN_PROFILES
       available = []
       for name, profile in _BUILTIN_PROFILES.items():
           if profile.get("cloud"):
               key_env = profile.get("api_key_env", "")
               if key_env and os.environ.get(key_env):
                   available.append(name)
           else:
               # Local model — check if server is responding
               # (reuse doctor's check_llm_server_reachable logic)
               available.append(name)
       return available
   ```

3. **Pre-flight validation**: Before the run loop, verify each model can respond:
   ```python
   def _preflight_check(self) -> list[str]:
       """Verify all models are available before starting. Return failures."""
       failures = []
       for model in self.models:
           try:
               _validate_model(model)  # Reuse from api.py
           except Exception as e:
               failures.append(f"{model}: {e}")
       return failures
   ```
   Print warnings for failed models, continue with the ones that work. If ALL fail, abort with clear guidance.

4. **General "no LLM" warning**: This applies beyond benchmarks — any mode that needs inference should check early. Add a shared utility:
   ```python
   # In models/language/config.py or utils/
   def check_llm_available() -> tuple[bool, str]:
       """Return (available, reason) for the currently configured LLM."""
   ```
   Use this in `cli.py` before entering sim/benchmark/agent mode. Also useful for the future hibernate mode — on wake, check if LLM is available before attempting inference.

5. **`maxim.benchmark()` API stub fix**: Wire to actual `BenchmarkRunner` (part of Phase 0b), with `models` defaulting to `None` (use loaded model):
   ```python
   def benchmark(models=None, *, suite="cognitive", runs=1, ...):
       if models is None:
           models = [os.environ.get("MAXIM_LLM_PROFILE", "")]
           if not models[0]:
               raise ConfigurationError("No model specified. Pass models= or set MAXIM_LLM_PROFILE.")
       runner = BenchmarkRunner(models=models, suite_path=_resolve_suite(suite), runs=runs)
       return runner.run()
   ```

~60 LOC for auto-discovery + preflight, ~20 LOC for CLI changes, ~15 LOC for shared LLM check utility.

### D. Fix Research Pipeline + Persistent Sessions

The research/experiment system has **5 interconnected bugs** that prevent report generation. Beyond fixing those, the architecture should evolve toward persistent sessions where research is an operation on accumulated data, not a one-shot pipeline.

#### Critical Bugs (should be Phase 0, blocking)

**D-0a. Two separate ExperimentLog instances (root cause)**

The orchestrator (`orchestrator.py:~818`) creates one `ExperimentLog` and registers the `record_experiment` tool with it. The research protocol (`research_orchestrator.py:~79`) creates a *completely separate* `ExperimentLog` for the Writer/Reviewer. Researcher LLM's experiments go into log A; Writer reads from log B. Log B is always empty.

**Fix:** Pass the orchestrator's `ExperimentLog` instance through to `start_research_mode()` via the `SimulationResult`, or share a session-scoped log path that both sides read/write:
```python
# In SimulationResult, add:
experiment_log_path: str | None = None  # Path to experiments.jsonl

# In research_orchestrator, load from that path instead of creating a new log:
if sim_result.experiment_log_path:
    experiment_log = ExperimentLog.from_path(sim_result.experiment_log_path)
```

**D-0b. `campaign_analysis` only populated with `--campaign`**

`orchestrator.py:~1246` — the `campaign_analysis` dict is built inside an `if pre_campaign_turns:` block. Without `--campaign`, it's `{}`, so research_orchestrator's experiment recording branch (`if sim_result.campaign_analysis:`) is never entered.

**Fix:** Always run post-simulation analysis (introspector.full_analysis) regardless of whether a campaign was provided. The analysis is useful for generative runs too:
```python
# Move the analysis block OUTSIDE the campaign conditional:
# After simulation completes (whether campaign or generative):
try:
    analysis = aut_introspector.full_analysis(seed_keywords=seed_keywords or [])
    sim_result.campaign_analysis = analysis
except Exception as e:
    logger.warning("Post-sim analysis failed: %s", e)
```

**D-0c. WriterAgent early return on empty experiments**

`research_agents.py:149-151` — returns empty draft without saving to disk. `draft.output_path` stays `None`, no `paper.md`, bus message to Reviewer never sent.

**Fix:** Remove the early return. Let the Writer generate a draft even with no experiments (it can describe the simulation setup and note that no formal experiments were recorded). Always save to disk before returning:
```python
experiments = self.experiment_log.all_entries()
# Remove: if not experiments: return self.draft
# Instead, let the LLM handle empty experiments gracefully in its prompt
```

**D-0d. Hardcoded "memory_recall_verath" key**

`research_orchestrator.py:~158` — `analysis.get("memory_recall_verath", {})`. Only works for one specific scenario.

**Fix:** Use dynamic keyword extraction from the analysis dict:
```python
# Instead of hardcoded key:
recall_data = {k: v for k, v in analysis.items() if k.startswith("memory_recall_")}
```

**D-0e. Reviewer always rejects empty papers**

With no sections, consistency check fires "Results section is empty" and verdict is always "reject".

**Fix:** This resolves automatically once D-0c is fixed (Writer produces real content). Add a guard: if the paper has no sections because the Writer failed, skip review and report the Writer failure explicitly.

~80 LOC total for all 5 fixes.

#### Persistent Sessions (post-publication, the bigger redesign)

**The problem with the current architecture:** Every simulation is fire-and-forget. `--research` tries to do everything in one shot (run experiment → write paper → review), which is brittle and doesn't match how real research works. Users can't:
- Run multiple experiments and then generate a report spanning all of them
- Resume a simulation to add more data points
- Call `--research` on an existing session without re-running the sim
- Incrementally build up experiment logs across sessions

**Proposed: Session-based research**

A session is a persistent container for simulation state + experiment data + reports. Research becomes an operation on a session, not a monolithic pipeline.

```bash
# ── Run experiments (accumulate data in a session) ─────────
maxim --sim "test memory recall" --language-model mistral-7b
# → Session: 20260408_143022
# → data/sim_reports/20260408_143022/
#      actions.jsonl, aut_hippocampus.json, aut_nac.json

# Resume the same session, add more experiments
maxim --resume-sim 20260408 --sim "now test with interference"
# → Appends to existing session data

# Run a campaign into the same session
maxim --resume-sim 20260408 --sim scenarios/experiments/hippocampal_recall_short.yaml

# ── Generate research report from accumulated data ─────────
maxim --resume-sim 20260408 --research
# → Writer reads ALL experiments.jsonl entries across the session
# → Generates paper.md + review_r0.json
# → Can be re-run after adding more data

# ── Python API equivalent ──────────────────────────────────
session = maxim.imagine(goal="test memory recall", model="mistral-7b")
# session.session_id = "20260408_143022"

session = maxim.imagine(goal="add interference", resume=session.session_id)
# Resumes into same session, appends data

report = maxim.research(session=session.session_id)
# Generates report from accumulated session data
# report.paper_draft, report.review, report.experiment_count
```

**Key architectural changes:**

1. **`SessionStore` class** (~100 LOC) — manages the session directory lifecycle:
   ```python
   class SessionStore:
       def __init__(self, session_id: str, base_dir: str = "~/.maxim/sessions"):
           self.session_dir = Path(base_dir) / session_id
       
       def experiment_log(self) -> ExperimentLog:
           """Single ExperimentLog for the entire session (fixes Bug #1)."""
           return ExperimentLog(session_dir=self.session_dir)
       
       def append_sim_result(self, result: SimulationResult) -> None:
           """Append a simulation run's data to the session."""
       
       def load_bio_state(self) -> dict:
           """Load accumulated hippocampus/NAc/ATL state."""
       
       def list_runs(self) -> list[dict]:
           """List all simulation runs in this session."""
   ```

2. **`--resume-sim` uses SessionStore** — loads persisted bio-state (hippocampus, NAc) so the agent has memories from prior runs in the same session. This already partially exists (`--resume-sim SESSION_ID` is documented in CLAUDE.md) but isn't wired to experiment logs.

3. **`--research` becomes a session operation** — instead of running a sim + writing a paper in one shot, it reads from the session's experiment log and generates a report:
   ```python
   def start_research_mode(session_id=None, goal=None, ...):
       if session_id:
           # Report mode: analyze existing session data
           store = SessionStore(session_id)
           experiment_log = store.experiment_log()
           # Writer + Reviewer operate on accumulated data
       else:
           # Legacy mode: run sim + report in one shot (backward compat)
           ...
   ```

4. **`maxim.research()` API** — wired to either create a new session (run + report) or analyze an existing one:
   ```python
   # New session (one-shot, backward compat)
   result = maxim.research(goal="test memory recall", model="claude-sonnet")
   
   # Analyze existing session
   result = maxim.research(session="20260408_143022")
   
   # Analyze with additional campaign data injected
   result = maxim.research(
       session="20260408_143022",
       campaign="scenarios/experiments/hippocampal_recall_short.yaml",
   )
   ```

5. **Session directory layout:**
   ```
   data/sim_reports/20260408_143022/
     session.json           # Metadata: created, model, goal, run count
     experiments.jsonl       # ALL experiments across all runs (single log)
     runs/
       run_001/             # First simulation run
         actions.jsonl
         aut_hippocampus.json
         aut_nac.json
         report.json
       run_002/             # Resumed run (with prior memories loaded)
         actions.jsonl
         aut_hippocampus.json
         aut_nac.json
         report.json
     paper.md               # Generated by Writer (can be regenerated)
     review_r0.json          # Generated by Reviewer
     research_result.json    # Summary metadata
   ```

**Why this is better:**
- **Separates experimentation from reporting.** Users can run 5 sims, inspect the data, and then decide to generate a paper. No forced one-shot pipeline.
- **Fixes the ExperimentLog split.** One `SessionStore` owns one `ExperimentLog` — both the researcher tools and the Writer read from the same source.
- **Enables incremental research.** Run a baseline sim, analyze it, realize you need more data, resume the session, add interference, generate an updated report.
- **Matches how researchers actually work.** Hypothesis → experiment → observe → adjust → experiment more → write up. Not: hypothesis → magic → paper.
- **`--resume-sim` gets real value.** Currently it reloads memories but doesn't accumulate experiment data. With SessionStore, resuming means building on prior work.

**Implementation sequence:**
1. Fix the 5 bugs (D-0a through D-0e) — ~80 LOC, immediate, unblocks research for v0.2.0
2. Add `SessionStore` class — ~100 LOC, clean abstraction over session directories
3. Wire `--resume-sim` to SessionStore — ~50 LOC, loads bio-state + appends to shared experiment log
4. Refactor `start_research_mode()` to accept session_id — ~60 LOC, report-only mode
5. Wire `maxim.research()` API verb — ~40 LOC, both one-shot and session modes

Total: ~330 LOC for the full persistent session system. Bugs (step 1) ship in v0.2.0; steps 2-5 ship in v0.2.1 or v0.3.0.

---

## Success Criteria

**Ready to publish when:**
- [ ] `maxim.run()` works (or explicitly raises NotImplementedError with guidance)
- [ ] `campaign()` and `research()` either work or raise honest errors
- [ ] Zero silent `except Exception: pass` in api.py, router.py, lane_backends.py, cli.py
- [ ] Hippocampus queue race condition fixed + threading test passes
- [ ] NAc, config, web_cache, last_run, internet_access use atomic_write_json
- [ ] OpenAI backend has rate-limit-aware backoff for 429s
- [ ] Missing API key produces immediate, actionable error (not deferred failure)
- [ ] Permission errors on ~/.maxim/ produce helpful message, not stack trace
- [ ] `docs/user/getting-started.md` tutorial runs without errors
- [ ] `docs/user/api-quickstart.md` broken link fixed
- [ ] `publication_guide.md` blocker table all marked FIXED
- [ ] JSON parser has unit tests covering all 4 stages
- [ ] Test PyPI install succeeds in clean venv
- [ ] `from maxim import RobotController` works (return type from `connect()` importable)
- [ ] `from maxim import ConnectionError, ToolNotFoundError` works (exception types importable)
- [ ] `MAXIM_PROXY_MAX_CONCURRENT=banana maxim doctor` doesn't crash (env var validation)
- [ ] No daemon threads with bare `except: pass` (all log at warning level minimum)
- [ ] `python -c "import maxim; print(maxim.__version__)"` works from pip install

- [x] `maxim.create.hippocampus()` creates standalone subsystem + tests pass (DONE 2026-04-08)
- [x] `maxim.create.agent()` always fresh, `maxim.load.agent()` restores (DONE 2026-04-08)
- [x] `maxim.imagine()` returns `Session` with `.id`, `.observe()`, `.research()` (DONE 2026-04-08)
- [x] `maxim.load.session()` / `maxim.load.sessions()` canonical load path (DONE 2026-04-08)
- [x] `maxim.Entity` accessible from `__all__`; `Report` exportable (DONE 2026-04-08)
- [x] NAc + Angular Gyrus use atomic_write_json (DONE 2026-04-08)
- [x] Entity.to_dict()/from_dict()/save()/load() round-trip (DONE 2026-04-08)
- [x] `create.router()` doesn't mutate os.environ (DONE 2026-04-08)

---

## Composable Object API (IMPLEMENTED — 2026-04-08)

Shipped pre-publication to lock in the public API surface. All internal classes remain behind facades.

**New files:**
- `src/maxim/session.py` (~260 LOC) — Session wrapper with observe/research, from_disk/from_result
- `src/maxim/create.py` (~280 LOC) — Factory namespace: hippocampus, nac, atl, scn, angular_gyrus, agent, pool, entity, embodiment, templates, router
- `src/maxim/load.py` (~120 LOC) — Deserialization: hippocampus, nac, atl, session, entity

**Modified files:**
- `src/maxim/__init__.py` — Added `_API_SUBMODULES` for create/load, SEM types + Session in `_API_TYPES`, new verbs `get_session`/`list_sessions`
- `src/maxim/api.py` — `imagine()` returns `Session` (wraps SimulationResult), added `get_session()`/`list_sessions()`
- `src/maxim/simulation/orchestrator.py` — `SimulationResult` gains `session_id` + `session_dir` fields

**Tests:** `tests/unit/test_composable_api.py` — 53 tests covering lazy loading, subsystem creation, mutation, cross-object isolation, session persistence, save/load roundtrips.

**Docs:** `docs/user/python-api.md` fully rewritten with verb API + composable API + mutation examples. `docs/user/getting-started.md` updated with both API paths.

**Key design decisions:**
- `maxim.create.*` = always fresh. `maxim.load.*` = restore from disk. Two namespaces, clear semantics.
- `Session` wraps `SimulationResult` via delegation — backward compatible
- `Report` object with `.save("report.md")` / `.save("report.json")` — multi-format output
- Only `Entity` exported as public type (not `Sensor`/`Modulator` — those are Protocols, not constructable)
- `NACConfig.persistence_path` is a proper config field (not monkey-patched `_persistence_path`)
- `create.router()` scopes env vars with save/restore (no global mutation)
- `Entity.from_dict()` reconstructs SpecSensor/SpecModulator/FailureMode (full round-trip)

**Fixed during review (2026-04-08):**
- `dir(maxim)` now returns all public names (added `__dir__()` to `__init__.py`)
- `load.hippocampus()` raises `FileNotFoundError` on missing file (was silently returning empty)
- `create.py` and `load.py` gained `__all__` to reduce namespace pollution
- `session.py` bare `except: pass` blocks upgraded to `logger.warning()`
- Removed `get_session`/`list_sessions` verb duplicates — `maxim.load.session()`/`.sessions()` is canonical
- Removed `Sensor`/`Modulator` from `__all__` (Protocols, not constructable)
- Added `Report` type with `.save()` for md/json (future: pdf/docx via `pymaxim[docs]`)
- Added `maxim.load.agent()` for restoring persisted agents
- `session.research()` returns `Report` with warning if no data (not silent empty object)
- `pool.run_turn()` emits warning that responses are placeholder (memory ops still work)

**Remaining work (v0.2.1+):**
- Wire `session.research()` to actual research_orchestrator (blocked on research pipeline fixes D-0a..D-0e)
- Wire `campaign()` / `benchmark()` to actual runtimes (currently warn + return stub)
- Add `maxim.create.router()` tests (requires LLM setup)
- `@maxim.tool` decorator should infer `input_schema` from type annotations (Phase 3i)
- Report `.save("report.pdf")` / `.save("report.docx")` via `pymaxim[docs]` extra

---

## Relationship to Other Plans

- **publication_guide.md** — This plan subsumes the "remaining blockers" section (§4). After completing Phase 0, update that table.
- **future_plans.md** — Phase 12b items overlap. After this plan completes, mark 12b as DONE in future_plans.md.
- **github_repo_management_plan.md** — Independent. That plan is post-publication work (Phase 1+ not started).
- **mother_maxim_plan.md** — Independent. M-0 prep items were already done in buildout Phase 9.
- **pre_publication_hardening_plan.md** — This plan finishes the remaining items from that plan.
