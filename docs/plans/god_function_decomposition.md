# God-Function Decomposition — Measurement Integrity, Not Elegance

**Status:** DRAFT (2026-08-10) — sequenced AFTER measurement_path_fail_loud.md Stages 1-2
**Motivation:** External critique (2026-08-10), point 1, verified:

| Function | Lines | try blocks |
|---|---|---|
| `runtime/agent_loop.py::run_agentic_loop` | 3,298 | 47 |
| `simulation/orchestrator.py::start_simulation_mode` | 3,110 | 77 |
| `cli.py::_main_impl` | 1,737 | 37 |

The justification is measurement integrity: a behavioral delta emitted from a 535-branch
function with hundreds of locals cannot be audited for contamination. Every experiment's
provenance chain runs through `run_agentic_loop`. Elegance is not a goal; auditability is.

**Front-gate:** no new mechanism. The extraction target pattern ALREADY EXISTS —
`run_agentic_loop` has 56 numbered section banners (`# 0. CHECK STOP CONDITIONS`,
`# 1. PERCEPTION`, `# 8.5 NAc decay`, …) and `LoopController` (Phase 1+2 of a prior
refactor) already holds transient loop state. This plan finishes that started migration.

## Order of attack

1. **`run_agentic_loop`** — it IS the measurement path. Highest value.
2. **`start_simulation_mode`** — the experiment harness (77 try blocks, worst swallower).
3. **`cli.py::_main_impl`** — lowest risk-to-value; may stop after 1-2 if it stalls.

## Method (per function)

**Boundary = the existing section banners.** Each numbered section extracts to a
module-level function `_loop_<slug>(ctx) -> <narrow result>` taking a context object —
extend `LoopController` (agent_loop) / introduce an equivalent `SimContext`
(orchestrator) rather than threading 30-arg signatures. Known traps, all with existing
feedback memories:

- **No local aliases for mutable state** (`no-local-aliases-for-state`): during extraction,
  locals that alias controller state must collapse onto the controller field, not be passed
  as parameters and reassigned.
- **Module-extraction global binding** (CLAUDE.md lesson): any module-level mutable
  consulted by a section moves by module reference, never `from X import _var`.
- **Section-order is load-bearing** (e.g. pause-check before drift tick before idle gate,
  per the Track-1 invariant; NAc decay in 8.5). Extraction must not reorder; each PR diff
  is reviewed specifically for call-order preservation.

**Behavior-preservation gate per PR** (this is why fail_loud Stages 1-2 land first — the
swallow instrumentation makes silent divergence visible):

1. Full fast suite + memory-hub suite green.
2. One logged interactive sim (`MAXIM_LOG_FILE` JSONL): percept → tool-call → followup
   sequence compared against a pre-refactor capture of the same seed/fixture; event
   sequence must match (excluding timestamps/latency).
3. `MAXIM_PROVENANCE_VERBOSITY=2` trace eyeballed once per PR batch.
4. Zero new `swallowed_exception` firings vs the Stage-2 baseline.

**PR sizing:** 3-6 sections per PR, mechanical extraction only. Any bug discovered during
extraction is NOT fixed in the extraction PR (no-band-aid rule: surface it, file it,
fix separately) — mixing semantic fixes into mechanical moves destroys the reviewability
that is this plan's whole point. Standard two-lens pre-merge round per PR.

## What this is NOT

- Not a redesign of the loop's phases, ownership, or the ARCHITECTURE.md layer story.
- Not an async rewrite, not a plugin system, no new abstractions beyond the context objects.
- Not chasing a line-count number. Success = each section independently readable and
  testable, with the loop body reduced to the section-dispatch skeleton.

## Sequencing with 1.1

Lands on main while the 1.1 walk runs from its pinned worktree — no interference. But do
NOT re-baseline any in-flight graduation row across an extraction merge; re-runs after a
merge cite the new hash (Exp 42b provenance discipline).

**Regression guard:** the per-PR JSONL sequence-diff (step 2) is the working guard during
migration; terminal state adds a CI check asserting `run_agentic_loop` body length below a
threshold (~300 lines) so the ratchet can't silently reverse.
