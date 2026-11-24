# Approach note: the wall-read guard for world clock A-pre (M72, extended)

> **PROPOSED 2026-11-24.** A one-page adversarial design pass (CLAUDE.md "Faster without lowering the bar" (1)),
> written before any code, for [world_clock_a_pre.md](world_clock_a_pre.md) D-WC4. It is based on Overdrive's current
> position (roadmap OD-A2: a static AST ban plus a long-horizon scripted run, BD-14.2), which supersedes the 11-16
> draft's `sys.setprofile` tripwire (§3.3).

## What the guard is

It has two parts, and neither part trusts the other.

1. **The static half is M72, extended.** It is one lint, `scripts/lint_wall_clock_reads.py`, and adds no second lint.
   It has two scopes:
   - **Machinery:** the step-path set. That is M72's original files (`decisions/`, `memory/`, `embodiment/body.py`,
     the two `_apply_decay`s), plus A-pre's families from the census: `agents/context_pool.py`, `attention/`,
     `salience/`, `embodiment/spec.py`, `proprioception/pain_bus.py`, `reactions/bus.py` and
     `models/language/cost_tracker.py`.
   - **Prompt:** `agents/prompt_builder.py`, `agents/context_pool.py`, `agents/llm_fallback.py`,
     `agents/exec_agent.py`, `agents/exec_prompts.py`, `agents/statistician_agent.py`, `retrieval/temporal_signal.py`,
     `tools/http_fetch.py` and `tools/introspection.py`.

   The lint refuses any **reference** (not just a call) to `time.time`, `time.time_ns`, `time.monotonic(_ns)`,
   `time.perf_counter(_ns)`, `time.localtime`, `time.gmtime`, `time.strftime` or `time.ctime`. It also refuses
   `datetime.now`, `utcnow`, `today`, `date.today` and `fromtimestamp`, and `uuid.uuid1` and `uuid.uuid4`, outside
   `time/agent_time.py` and the allowlist. It is a **per-file ratchet**, the pattern `lint_atomic_io_ratchet.py` uses: each file is grandfathered at its
   `origin/main` count (through `scripts/_lint_git.py`'s shared ratchet), a branch may not raise a file's count, new
   files start at zero, and each A-pre PR lowers its family's counts. There is no baseline file to edit.
2. **The dynamic half is G-long.** It is a long-horizon scripted run through consolidation, sleep and decay. The wall
   clock is patched to jump a huge, varying amount on every read, and the run is repeated under two different jump
   schedules. The persisted-state digest (ids excluded, D-WC1) must be identical. A wall read that the lint missed
   shows up as a digest difference, whatever form the value takes.

## Every input the guard consults, and how each could lie

| Input | How it lies | What catches it |
|---|---|---|
| **The scope lists** | A new module that renders or stamps time is in neither list, so it is never scanned | Machinery: G-long, whose digest covers state wherever it is written. Prompt: G-render and G-tz build real prompts. The list itself: backlog **M99**, which derives the prompt scope from the prompt's producers. *Stated blind spot until M99.* |
| **Import aliasing** | `import time as _time` (as in `embodiment/spec.py`), `from time import time as now`, `from datetime import datetime`, `import datetime as dt` | The lint resolves every import binding in the module (the `lint_dead_imports` resolver) before matching. A planted test case covers each form. |
| **References that are not calls** | `field(default_factory=time.time)` binds the builtin at class definition, so a call-only scan misses it | The lint matches `Attribute` and `Name` references, not `Call` nodes. Planted case: a dataclass stamp. |
| **Dynamic access** | `getattr(time, "time")`, `importlib`, a callable passed in from outside scope | *Static blind spot, stated.* G-long catches it if the value reaches state. A read that never reaches state, such as one only logged, is not a correctness leak for this plan. |
| **Transitive reads** | an in-scope function calls a helper in an unscanned module (a `utils/` function) that reads the wall | *Static blind spot, stated.* G-long and G-pace catch it at the digest. The census lists every cross-module helper the families call. |
| **The allowlist** | An entry rubber-stamps a leak. An entry outlives the site it names. | Entries are keyed `file::function`, with a category from a fixed set (`process`, `scheduling`, `motor-io`, `real-world-io`, `cost`, `pending:<where>`) and a reason string. A `pending` entry must name the plan or PR that removes it. An entry whose function no longer reads a banned name **fails** the lint, so a stale entry cannot linger. New `motor-io` and `real-world-io` entries are review-visible in the diff. |
| **The ratchet's reference** | The `origin/main` ref is stale or missing (shallow clone), so the grandfathered counts are wrong | `scripts/_lint_git.py`'s shared ratchet, which `lint_atomic_io_ratchet.py` already relies on: a ref it cannot read is exit 2, never a pass. A raise needs a new allowlist entry, which is visible in the diff. |
| **Legitimate monotonic durations** | A timeout's `time.monotonic()` looks the same as a leak | It carries category `process`. A duration that feeds learning (OD-A4's tool duration) is never `process`: it is `pending:A-post` with G-tool-duration red. |
| **G-long's sentinel** | A wall read used only as a difference gives small numbers that look normal. A frozen fake clock hides cadence bugs. | The patched wall **advances** by a huge, varying jump per read, and two schedules are compared, so any wall-derived duration or stamp changes the digest. The machinery clock advances too, never frozen. |
| **Threads** | A background worker (the ConceptExtractor) reads the wall off the stepped thread | G-long builds with `build_memory_hub(start_background_workers=False)`, so worker paths are out of its reach. *Stated blind spot:* the host-driven thread is OD-A12's, and G-long does not claim it. |
| **TZ** | The prompt test runs where `time.tzset` is missing (Windows) and passes vacuously | G-tz skips **loudly** with a reason and runs in CI on Linux. A skip is never a pass. |
| **The census** | The census script misclassifies a site, so the plan's placement is wrong | The census is AST-only and prints its classification for review. The lint and G-long don't read the census's verdicts: they rescan. |
| **The scan's own tree** | The lint scans `tests/` or its own fixtures and flags itself, or skips a tracked file | The lint scans tracked `src/maxim/` files only (`git ls-files`), and its tests use planted temp trees (the `lint_self_scan` lesson). |

## Proof by deletion

Every planted case goes red before it goes green, for each form above, in `tests/unit/test_lint_wall_clock_reads.py`:
- a `default_factory` stamp;
- a from-import alias;
- `fromtimestamp`;
- `strftime`;
- `uuid4`;
- a stale allowlist entry;
- a count raised above `origin/main`.

G-long is proven by removing one machinery pass in PR 4 and watching the digest differ.

## What it does not claim

- No guard on ids or randomness: D-WC1 defers them, and the lint lists `uuid4` only to keep the census honest. Its
  sites are `pending:OD-A7`.
- No guard on the fenced files: they are `pending:A-post`, and G-long's full-loop half stays strict red until A-post.
- No guard on threads the host owns (OD-A12).
