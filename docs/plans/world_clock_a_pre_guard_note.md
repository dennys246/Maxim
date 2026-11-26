# Approach note: the wall-read guard for world clock A-pre (M72, extended)

> **PROPOSED 2026-11-24.** This is a one-page adversarial design pass, written before any code (CLAUDE.md "Faster
> without lowering the bar" (1)), for [world_clock_a_pre.md](world_clock_a_pre.md) D-WC4.
> - **Basis:** Overdrive's current position. Roadmap row OD-A2 calls for a static AST ban plus a long-horizon scripted
>   run (BD-14.2), which supersedes the draft's `sys.setprofile` tripwire (draft §3.3).
> - **Revised** after the plan's three-lens review on `1e1eed8d`. The review corrected four claims: the dynamic half's
>   reach, the resolver this note cited, the ratchet's semantics and the scope list.
> - **Built 2026-11-26 in PR 1b, amended by the owner at its start.** Each amendment is stricter than the
>   design below, which stays as written:
>   - **Absolute from birth.** Every in-scope read got an allowlist entry, so every file was at zero on day one and
>     the migration ratchet (§1 "Migration") would never have run. It is not built; a read without an entry fails.
>     The "swaps at an equal count" row below no longer has a ratchet to fool.
>   - **Each entry pins its function's exact read count** (`reads`). A read added to an allowlisted function fails,
>     closing the blind spot a `file::function` key alone leaves (a leak hidden beside an allowlisted read); a read
>     removed fails too, so the "stale entry" rule covers a partly stale one.
>   - **`proprioception/focus_learner.py` joins the machinery scope, whole.** M72's original row names its
>     `_apply_decay`; PR 0's `_clock_refs.MACHINERY_SCOPE` had omitted it.
>   - The allowlist is [`scripts/wall_clock_allowlist.json`](../../scripts/wall_clock_allowlist.json); the categories
>     are defined in experience_clock_decay.md §(g).

## What the guard is

It has two parts, and neither trusts the other.

### 1. The static half: M72, extended

- **One lint, not a second one:** `scripts/lint_wall_clock_reads.py`.
- **End state.** It is #1361's **absolute** lint (experience_clock_decay.md §Enforcement): no wall-clock reference in
  scope outside the allowlist.
- **Migration.** It starts as a per-file ratchet against `origin/main`, through `scripts/_lint_git.py::count_ratchet`
  and `base_ref`, so existing reads are grandfathered while A-pre and A-post convert them. A file pins **absolute**
  (zero outside the allowlist) as soon as its count reaches zero. When every in-scope file is pinned, the ratchet is
  deleted.
- **Precedence.**
  - An allowlisted `file::function` reference is never counted.
  - Every other reference in scope is counted.
  - The ratchet applies to the count, and a pinned file applies zero.
- **Scopes.**
  - **Machinery.** M72's original files, which are `decisions/`, `memory/`, `embodiment/body.py` and the two
    `_apply_decay`s. Added to them: `integration/memory_hub.py`, `agents/memory_agent.py` (whole file),
    `agents/context_pool.py`, `attention/`, `salience/`, `default_network/network.py`, `bridges/salience_bridge.py`,
    `bridges/planning_bridge.py`, `bridges/spatial_bridge.py`, `similarity/signature.py`, `embodiment/spec.py`,
    `proprioception/pain.py`, `proprioception/pain_bus.py`, `reactions/bus.py` and `models/language/cost_tracker.py`.
    Most of these start ratcheted with `pending:` entries, because D-WC6 moves their stamps later.
  - **Prompt.** `agents/prompt_builder.py`, `agents/context_pool.py`, `agents/llm_fallback.py`,
    `agents/exec_agent.py`, `agents/exec_prompts.py`, `agents/statistician_agent.py`, `retrieval/temporal_signal.py`,
    `tools/http_fetch.py`, `tools/introspection.py` and `simulation/introspection.py`.
- **Banned references** (references, not just calls):
  - `time.time`, `time.time_ns`, `time.monotonic(_ns)`, `time.perf_counter(_ns)`, `time.localtime`, `time.gmtime`,
    `time.strftime` and `time.ctime`;
  - `datetime.now`, `utcnow`, `today`, `date.today` and `fromtimestamp`.

  These are refused outside `time/agent_time.py` and the allowlist.
- **`uuid1` and `uuid4` are census-only.** The lint warns and does not fail, because D-WC1 offers no `IdSource`, so
  failing on them would be a new id rule the owner deferred.
- **The alias resolver is shared:** `scripts/_clock_refs.py`, built in PR 0 for the census, which PR 1's lint imports
  with the banned names and both scopes, so the two cannot drift. Neither `lint_dead_imports`' resolver
  (maxim-internal targets) nor `lint_atomic_io_ratchet._module_aliases` (`import X as Y` only) resolves local stdlib
  bindings. The census and the lint still judge independently: the lint never reads the census's verdicts. The
  resolver handles these forms, each with a planted test:
  - `import time as _t`;
  - `from time import time as now`;
  - `from datetime import datetime`;
  - `import datetime as dt`;
  - `from datetime import datetime as DT`.
- **Bindings it cannot resolve one way fail closed.** A name bound to two clock targets in one module (a
  function-local `from datetime import datetime` beside a module-level `import datetime`) or a star import from a
  clock module is reported, not guessed. The census prints it and the lint fails on it.

### 2. The dynamic half: G-long

- **The run.** A long-horizon scripted run through observe, consolidate, sleep and decay. `time.time` alone is patched
  to jump a huge, varying amount per read, and two jump schedules are compared. The persisted-state digest must be
  identical. Its scope is named in the test: A-pre's families first, and the full loop in A-post.
- **What it catches.** A wall read that goes through `time.time` at call time and reaches state, in any form.
- **What it cannot catch** (corrected from the first draft, which claimed "whatever form the value takes"):
  - `datetime.now()` and `datetime.fromtimestamp()`, which are C-level and never call `time.time`;
  - `field(default_factory=time.time)`, which binds the builtin at import (`tests/unit/_loop_harness.py`'s
    docstring states this limit);
  - `time.monotonic` reads, which are deliberately left unpatched, because huge jumps there would trip `Deadline` and
    `wait_session_ended` and abort the run instead of comparing.

  For these the static half is the only guard.

## Every input the guard consults, and how each could lie

| Input | How it lies | What catches it |
|---|---|---|
| **The scope lists** | A new module that renders or stamps time is in neither list | Machinery: G-long, where its reach allows (§2). Prompt: G-render, G-tz and G-scn, which build real prompts. The list itself: **M99**, which derives the prompt scope. *Stated blind spot until M99.* |
| **Import aliasing** | `import time as _time` (as `embodiment/spec.py` does), `from datetime import datetime`, and the other forms in §1 | The shared resolver (`scripts/_clock_refs.py`), with a planted test case for each form. A conflicting binding or a clock-module star import is reported and fails closed |
| **Non-call references** | `default_factory=time.time` binds at class definition, so a call-only scan misses it | `Attribute`/`Name` matching. Planted: a dataclass stamp. G-long cannot see this case (§2), so the static half must. |
| **C-level calendar reads** | `datetime.now()`, `fromtimestamp` | Static only (§2). Planted: one of each in a prompt-scope fixture. |
| **Dynamic access** | `getattr(time, "time")`, `importlib`, a callable passed in | *Static blind spot, stated.* G-long catches it if the read goes through `time.time` and reaches state. |
| **Transitive reads** | An in-scope function calls a `utils/` helper that reads the wall | *Static blind spot, stated.* G-long and G-pace catch it at the digest, within §2's reach. The census lists every cross-module helper a family calls. |
| **The allowlist** | A rubber-stamp entry, or an entry that outlives its site | Entries are keyed `file::function` with a fixed category (`process`, `scheduling`, `motor-io`, `real-world-io`, `cost`, `pending:<where>`) and a reason. A `pending` entry names the PR that removes it. An entry whose function no longer reads a banned name **fails** the lint. New entries show in the diff. |
| **The ratchet's reference** | `origin/main` is stale or missing, for example in a shallow clone | `_lint_git.base_ref`: a ref it cannot read exits 2 and is never a pass |
| **Swaps at an equal count** | `count_ratchet` counts hits in changed files, so replacing an allowlisted read with a leak in the same function, or one leak with another, passes | Allowlisted references are not counted, so a leak swapped *in* raises the count. A leak swapped for another leak is invisible to a count. That is why a file pins **absolute** once it reaches zero, and why the end state is absolute (§1). *Stated blind spot while ratcheted.* |
| **Legitimate durations** | A timeout's `monotonic()` looks the same as a leak | Category `process`. A duration meant to feed learning (OD-A4's tool duration) is never `process`: it is `pending:A-post`. It reaches no learning today (#1788), and G-tool-duration lands red in the PR that wires it. |
| **G-long's sentinel** | A frozen fake clock hides cadence bugs, and a wall read used only as a difference looks normal | The wall **advances** by huge, varying jumps, two schedules are compared, and the machinery clock advances too |
| **Threads** | The ConceptExtractor reads the wall off the stepped thread | G-long builds without ATL, so `MemoryHub._restore_on_open` cannot revive the extractor across a session cycle, and with `build_memory_hub(start_background_workers=False)`. *Stated blind spot:* host-driven threads are OD-A12's. |
| **TZ** | The prompt test runs where `time.tzset` is missing (Windows) and passes vacuously | G-tz skips **loudly** with a reason and runs in CI on Linux. A skip is never counted as a pass. |
| **The census** | It misclassifies a site, so a placement is wrong | It is AST-only and prints its classification for review. The lint and G-long rescan and never read its verdicts. M100's `--check` adds the pair check. |
| **The scan's own tree** | It flags its own fixtures, or skips a tracked file | It scans tracked `src/maxim/` only (`git ls-files`), and its tests use planted temp trees |

## Proof by deletion

Each planted case is red before it goes green, in `tests/unit/test_lint_wall_clock_reads.py`:
- one per alias form;
- a `default_factory` stamp;
- `datetime.now`;
- `fromtimestamp`;
- `strftime`;
- a stale allowlist entry;
- a count raised above `origin/main`;
- a pinned file going from zero to one.

`uuid4` is checked as a warning, not a failure.

G-long is proven by reverting PR 4's window read to `time.time()` under the patched schedules and watching the
digest differ.

## What it does not claim

- No failure on ids or randomness (D-WC1). Their sites are `pending:OD-A7`.
- No guard on the hot and fenced files beyond the ratchet. Their sites are `pending:A-post`, and G-long's full-loop
  half stays strict red until A-post.
- No guard on threads the host owns (OD-A12), or on the C-level and import-bound reads at run time (§2).
