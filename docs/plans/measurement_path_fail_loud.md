# Measurement-Path Fail-Loud Purge

**Status:** Stages 1 + 2 + 4 SHIPPED (1: #487; 4: the CI lock `scripts/lint_no_silent_swallows.py` 2026-08-13; 2: 2026-08-30, 1.1.2 Cluster A — **zero firings**, [artifact](../experiments/data/fail_loud_stage2/)). **Stage 3 remains open and is now green-lit** — roadmap 1.1.x item 6. (DRAFT 2026-08-10)
**Motivation:** External deep-dive critique (2026-08-10), point 2 — cross-confirmed by our own
history: the SCN drive path was dead for months behind exactly one bare-except-swallowed
`TypeError` (see `scn_event_producer_gap.md`). The research claim rests on detecting ~1 SD
behavioral deltas; a silently swallowed exception in the substrate path is contamination we
cannot see, cannot attribute, and will never find. When ablation arms move "in roughly
arbitrary directions," partially-broken-and-quiet is an unfalsified hypothesis until this lands.

**Front-gate:** rides entirely on existing infrastructure (logging, conftest, CI lint). No new
mechanism, no new bus, no new module.

## Scope — the measurement path ONLY

NOT a repo-wide purge of all 1,749 handlers. Exactly these files (inventory 2026-08-10;
`bare` = `except Exception:` with no binding, `swallow` = followed by bare `pass`/`continue`).
**The machine copy of this table is `scripts/lint_no_silent_swallows.py::MEASUREMENT_PATH`
(Stage 4's CI lock) — any edit to this table updates the lint in the same commit** (a
file added here but not there is silently unlinted; a removed/renamed file fails the
lint loudly):

| File | except | bare | swallow |
|---|---|---|---|
| decisions/nac.py | 11 | 11 | 8 |
| decisions/temporal_credit.py | 6 | 3 | 2 |
| runtime/tool_dispatch.py | 6 | 4 | 1 |
| runtime/bio_integration.py | 8 | 0 | 0 |
| runtime/agent_loop.py | 52 | 15 | 4 |
| similarity/encoder.py | 6 | 5 | 2 |
| similarity/ec.py | 2 | 1 | 0 |
| bridges/tool_pain_bridge.py | 8 | 6 | 2 |
| proprioception/pain_bus.py | 8 | 6 | 4 |
| embodiment/body.py | 8 | 5 | 4 |
| embodiment/tool_bridge.py | 8 | 8 | 4 |
| simulation/sim_logger.py | 5 | 4 | 3 |
| memory/hippocampus.py | 7 | 1 | 1 |
| memory/hippocampus_consolidation.py | 3 | 2 | 2 |
| integration/memory_hub.py | 59 | 2 | 1 |
| decisions/causal_link.py | 0 | 0 | 0 |

~190 handlers to classify, ~38 silent swallows to eliminate.

## Policy (the rule this plan installs)

Within the measurement path:

1. **Learning writes fail loud.** Any handler wrapping a substrate/learning WRITE
   (`record_outcome*`, `distribute_reward`, `encode*`, `pattern_complete_or_separate`,
   episodic capture, valence propagation, credit routing) must either narrow to the specific
   expected exception or propagate. A broken learning write that "keeps the sim running" is
   worse than a crash — it produces a plausible-looking wrong measurement.
2. **Telemetry emits may stay defensive, but never silent.** Tracers, sim_log, decision-log
   appends may catch-and-continue, but MUST log the exception (WARNING first occurrence,
   DEBUG after) with enough context to attribute. Bare `pass` is banned even here.
3. **Every kept handler gets a one-line comment stating what it defends against.** If the
   author can't name the expected exception, that's the signal it should propagate.

## Stages

**Stage 1 — instrument, zero behavior change.** Replace every silent swallow in the scope
table with `logger.warning("swallowed_exception", exc_info=True, ...)` (deduped per site via
the `warn_optional_fallback` pattern). No control-flow change. Full suite + ruff.

**Stage 2 — measure. ✅ SHIPPED 2026-08-30 (1.1.2 Cluster A) — result: ZERO firings.**
Artifact: [../experiments/data/fail_loud_stage2/](../experiments/data/fail_loud_stage2/)
(`baseline.json` + both raw captures, gzipped); tool:
[`scripts/fail_loud_stage2.py`](../../scripts/fail_loud_stage2.py). Captures measured at `6f3f3b7d` from a clean tree, artifact regenerated at `258389f8` also
clean (the first cut was regenerated from a DIRTY tree and said otherwise — see the artifact's
Provenance section; the review round caught it and the tool now REFUSES rather than stamps):
**0 firings across 75,654 records**, over the substrate mode
(`orient_substrate/2_full_path_probe.py`, no LLM, 72,239 records) and the generative mode
(`--sim "test basic recall" --interactive false --sim-max-turns 3 --llm qwen2.5-14b`, 3,415
records). Per this plan's own reading that is the informative-either-way outcome's good
branch: **the swallows are dead defensive weight, safe to narrow — Stage 3 is green-lit.**

Two things the artifact records rather than smooths over: (a) the instrumented-site count is
**50**, not the 48 of the Stage-1 note — recompute with `fail_loud_stage2.py inventory`, never
from prose; (b) per-site coverage is NOT proven — a zero over two modes is a comparison
baseline, not proof that all 50 sites are unreachable.

**The grep in the original instruction does not work as written.** "Grep the JSONL for
`swallowed_exception`" matches, but a *parser* written against the call site's
`extra={"event":…, "data":{…}}` shape reads nothing: the `MAXIM_LOG_FILE` handler formats via
`StructuredFormatter` → `LogRecord.to_compact()`, which keys the event `"e"` and flattens
`data` to the top level. That silently reports zero firings and prints a pass. Guarded by
`tests/unit/test_fail_loud_stage2.py::test_reads_compact_shape`. This is the same failure
class PR #487's review caught at the other end of the pipe, one layer further on.

**Stage 3 — fix.** Firing sites: root-cause fix. Silent sites: narrow the exception type or
convert to propagate per the policy above. One PR per 3-5 files, each with the standard
two-lens pre-merge round (these flips CAN change behavior — that is the point — so each PR
runs the fast suite + the memory-hub suite + one sim smoke).

**Stage 4 — lock. ✅ SHIPPED 2026-08-13 (ahead of Stages 2–3, which remain open).** The
scoped files were already at zero swallows after Stage 1, so the lock was safe to land
early: `scripts/lint_no_silent_swallows.py` in CI enforces (1) zero-total over the 16
scoped files and (2) diff-scoped no-count-increase repo-wide. Stages 2 (measure which
instrumented sites actually fire) and 3 (narrow/propagate per the policy) still to do.
Original spec: extend the CI lint: the diff-scoped no-new-swallows lint already tracked
in CLAUDE.md ships here, with the measurement-path files promoted to a **zero-total-swallows
allow-list** (not just no-new). Grep form (COMMENT-TOLERANT — the PR #487 review found the
original comment-blind pattern missed 10 `pass  # best-effort` swallows, including the
`record_event` wrap that motivated this plan):
`grep -EA1 "except Exception:\s*(#.*)?$" <files> | grep -E "^\s+(pass|continue)\s*(#.*)?$"`
must return zero matches in the scoped files.

**Stage 1 outcome note (2026-08-10):** shipped as PR #487 — 38 inventory sites + 10
comment-blind sites found by the review lens = **48 instrumented sites**. The review also
caught that the `swallowed_exception` event initially did not survive StructuredFormatter
into the MAXIM_LOG_FILE JSONL (event/data `extra` was required) — pinned by
`TestJsonlSerialization` so the Stage-2 grep provably sees what fires.

## Interaction with running experiments

Stage 3 flips must NOT land mid-walk on a branch the 1.1 graduation runs read from. The walk
runs from the pinned worktree (`../Maxim-heartbeat-1.1` @ f05c63aa), so main is safe to
change; but any behavioral delta observed AFTER a Stage 3 merge must note the fail-loud
change as a candidate cause (provenance discipline per Exp 42b).

## Non-goals

- Repo-wide except cleanup (grandfathered ~1,700 sites stay; the existing review-time grep
  covers new ones).
- Refactoring the functions these handlers live in (that is `god_function_decomposition.md`;
  this plan deliberately lands FIRST so the instrumented logging can verify extraction
  preserves behavior).

**Regression guard:** Stage 4 CI lint (zero-swallow allow-list over the 16 scoped files).
