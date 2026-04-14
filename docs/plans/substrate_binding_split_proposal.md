# Substrate Binding & Persistence — Plan Split Proposal

**Status:** PROPOSAL — authored 2026-04-14, awaiting approval before new plan files land.
**Parent plan:** [substrate_binding_persistence.md](substrate_binding_persistence.md) (currently monolithic across P3a → P8 + B3–B5)
**Motivation:** The P2 session retrospective — a 2,200 LOC plan required three internal stages and a forced metric pivot mid-sweep to ship. The P3 → P8 arc is ~4,100 LOC across 10 phases and 3 version bumps. A single plan file at that scope becomes unreadable, unreviewable, and un-stageable.
**Related:** [sem_execution_hook.md](sem_execution_hook.md) — NEW companion plan that closes a gap discovered during this audit.

## Why split

The P2 shipping pattern taught three lessons that make monolithic plans expensive:

1. **Every substrate phase needs its own three-stage template.** P2 required Stage 1 (metric extractor + synthetic mechanism test), Stage 2 (integration PoC on real bundled components), Stage 3 (real-embedding validation sweep with three forced metric pivots). Each stage was its own pre-merge review round, each had its own commit on the feature branch, each landed as a separate PR (or at minimum a separate commit). A 10-phase plan would need 30 stages. No plan file can usefully narrate 30 stages.

2. **Pre-merge review rounds require narrow surfaces.** The P2 Stage 2 review caught 3 critical + 5 important findings including the `_context_similarity` union-of-keys band-aid that would have shipped if the plan had been bigger. Executor + Architecture reviewers need to read the ENTIRE changed surface to give grounded findings. A 400-LOC phase is reviewable; a 4,100-LOC arc is not.

3. **Version-gate boundaries need to map to plan boundaries.** "0.3-target ships when substrate_binding_persistence is complete" is ambiguous — which phases of a 10-phase plan gate the version? "0.3-target ships when substrate_p3a + substrate_p3b + substrate_p3_5 + substrate_p4 are all `Status: COMPLETE`" is unambiguous.

The split is cheap (8 new files, each ~200–400 lines), reversible (the parent plan stays as an index), and matches the version-gate structure already in `docs/plans/README.md`.

## Inventory of existing scaffolding (audit result)

Before proposing the split I audited what's already built. Big finding: **P3a needs less new infrastructure than the parent plan suggests**, because the Hebbian within-ATL edge mechanism already has a home:

| Surface | Status | Notes |
|---|---|---|
| `DependencyGraph.add_edge(weight=...)` + `update_edge(weight=...)` on `agents/bus.py:1012` | ✅ exists | P3a Hebbian updates layer on `EdgeType.ASSOCIATES` — no new edge type needed. |
| `ATL.graph` — ATL's internal `DependencyGraph` | ✅ exists | Within-ATL Hebbian edges are one `add_edge` call away. |
| `CrossLayerGraph` on `memory/cross_layer.py` | ✅ exists | Inter-layer edges (hippocampus ↔ ATL) already typed: `DERIVED_FROM`, `INSTANCE_OF`, `INFORMS`. |
| `Hippocampus.capture()` + `capture_from_loop()` | ✅ exists | Single-event capture path. |
| `tests/substrate/persistence_harness.py` (S3) | ✅ exists | Subprocess round-trip harness for P3.5. |
| `PerceptTraceBuffer` + `substrate_node_id` + NAc reward bias | ✅ exists | Wired from P1/P2. |
| `Episode` dataclass (multi-event time window) | ❌ missing | Real P3a new work. |
| `EpisodeStore` (hippocampus-side episode → nodes index) | ❌ missing | Real P3a new work. |
| Episode boundary detector (tick gap + scene signal) | ❌ missing | ~80 LOC new. |
| Hebbian update rule on episode close | ❌ missing | Layers on `DependencyGraph.update_edge` — ~50 LOC new. |
| Partial-cue retrieval path | ❌ missing | Real P3a new work. |
| TF-IDF gate baseline | ❌ missing | ~100 LOC, standalone. |
| `BioSystemSnapshot` Protocol | ❌ missing | Real P3.5 new work. |
| NAc `_reward_bias` save/load fields | ⚠️ partial | P2 added the field persistence; P3.5 extends the protocol. |
| `PerceptTraceBuffer.save/load` | ❌ missing | Real P3.5 new work. |
| CLIP-based `VisionEncoder` | ❌ missing | Real P4 new work. Existing `src/maxim/models/vision/` is YOLO-style detection, wrong shape. Needs new optional dep. |

**Important gap discovered: there is no production path that executes SEM affordances.** `executor.py::execute(action)` handles TOOLS (bash, internet_search, etc.) via `ToolPainBridge.record_tool_start` (line 172–173). `motor.py` is a CATALOG layer for motor programs, not an execution layer. Agents decide "slash the sword" in LLM prose but nothing translates prose → `entity.modulators['combat'].affordances['slash']` → `evaluate_failures()` → pain cascade. P2 Stage 2's PoC faked this by calling `agent.record_action("slash")` + driving `durability` directly.

This is bigger than the RC3 follow-up note made it sound, and spans the whole binding-persistence arc. It's carved out into its own plan: [sem_execution_hook.md](sem_execution_hook.md).

## Proposed split — 8 focused plans replacing 1 monolith

The parent `substrate_binding_persistence.md` stays as the top-level index + exit-criteria tracker. Each phase-level plan becomes its own file with Stage 1/2/3 template, entry criteria, load-bearing invariants, and exit criteria. Each plan has a `Status:` line (`Draft` / `Active` / `Stage N in progress` / `COMPLETE`).

### Substrate track

| New file | Scope | Stages | Gates | Depends on |
|---|---|---|---|---|
| `substrate_p3a_episode_binding.md` | P3a only. ~400 LOC + 100 metric. Episode dataclass, EpisodeStore, boundary detector, Hebbian update, partial-cue retrieval, TF-IDF gate baseline. | S1 metric + mechanism tests; S2 fixture-based validation; S3 real-data sweep + pre-merge review | 0.3-target | substrate_recognition ✅ |
| `substrate_p3b_channel_integration.md` | P3b only. ~250 LOC + 100 metric. Per-channel boundary rules (SMS + narrative), channel-filtered retrieval, metadata-grep baseline. | S1 + S2 + S3 same shape | 0.3-target | p3a |
| `substrate_p3_5_persistence_snapshot.md` | P3.5 only. ~500 LOC. `BioSystemSnapshot` Protocol, NAc reward-bias save/load, PerceptTraceBuffer save/load, cross-layer round-trip harness, schema versioning. | S1 minimal shell (P3a entry gate); S2 full protocol; S3 migration + round-trip | 0.3-target | substrate_recognition ✅; minimal shell LANDS BEFORE p3a Stage 1 |
| `substrate_p4_cross_modal_binding.md` | **1.0-GATING.** P4 only. ~500 LOC + 100 metric. CLIP-based `VisionEncoder`, cross-modal retrieval path, mug-test fixture, OpenCLIP head-to-head baseline. | S1 mechanism (synthetic embeddings); S2 real CLIP + mug fixture; S3 20-seed sweep + OpenCLIP head-to-head | 0.3-target | p3a, p3b, p3_5 |
| `substrate_p5_stress_persistence.md` | P5 only. ~400 LOC. 10k+ node long-running sim, serialize every 100 episodes, reload verification. | S1 mechanism; S2 mid-scale; S3 full 10k+ sweep | 0.5 | p3_5, p4 |
| `substrate_p6_extinction.md` | P6 only. ~300 LOC + 100 metric. Decay without reinforcement, LRU head-to-head gate baseline. | S1 mechanism; S2 two-group sim; S3 LRU sweep | 0.5 | p3a |
| `substrate_p8_sleep_replay.md` | P8 only. ~350 LOC + 100 metric. Sleep-phase replay, Hebbian link updates, F1 improvement gate. **Activates** [memory_consolidation_practice.md](memory_consolidation_practice.md). | S1 mechanism; S2 within-session replay; S3 F1 delta sweep | 0.5 | p3a, p6 |

### Prompt track

| New file | Scope | Stages | Gates | Depends on |
|---|---|---|---|---|
| `prompt_b3_b5_track.md` | B3 Acting Coach + B5 Embodiment/narrative separation. ~450 LOC. | S1 each | 0.4 | B1 ✅ |
| `prompt_b4_replanning.md` (future split from above) | **1.0-GATING.** B4 replanning with failure diagnosis. ~400 LOC. Promoted to its own plan when opened because its failure mode gates 1.0 and it needs P3a episode retrieval of prior attempts. | S1 mechanism; S2 induced-failure scenario; S3 blind-A/B | 0.4 | B1 ✅, p3a |

### Cross-cutting (NEW)

| New file | Scope | Notes |
|---|---|---|
| `sem_execution_hook.md` | Forward path for SEM affordances — prose decision → affordance invocation → failure evaluation → pain cascade. ~300 LOC + tests. | **Can ship anytime.** Not on the substrate version-gate path. UNBLOCKS behavioral convergence experiments that need real SEM execution. Parallel-session safe (zero file overlap with P3/P4/P5). See plan doc for full scope. |

## Sequencing

```
substrate_recognition ✅ (SHIPPED)
      │
      ▼
substrate_p3_5 (S1 minimal shell — persistence scaffold for P3a tests)
      │
      ▼
substrate_p3a ──► substrate_p3b
      │               │
      └───────┬───────┘
              ▼
      substrate_p3_5 (S2+S3 full protocol)
              │
              ▼
      substrate_p4 (1.0-GATING)
              │
      ┌───────┼───────┐
      ▼       ▼       ▼
    p5      p6      p8 ──► activates memory_consolidation_practice
      │       │       │
      └───────┴───────┘
              │
              ▼
  behavioral_convergence_practice — experiments land as P-phases complete

Parallel (not gating substrate arc):
  prompt_b3_b5_track  ── Track B rewrites, depends on B1 ✅ + p3a for B4
  sem_execution_hook  ── Production SEM forward path, unblocks behavioral experiments
```

**The reversal vs the parent plan:** P3.5 opens FIRST (Stage 1 minimal shell) because P3a's pass criteria include "persistence round-trip" and the `BioSystemSnapshot` Protocol needs to exist for P3a's tests to even run. P3.5's full Stage 2+3 can happen after P3a and P3b are both green. This is the kind of ordering nuance a monolithic plan hides.

## What each plan file contains (template)

Every new plan file follows the same shape, baked in from day one. No plan should be opened without all of these sections:

```markdown
# <Plan name>

**Status:** Draft / Active / Stage N / COMPLETE
**Scope:** ~<LOC> across <N> stages
**Target version:** <0.3-target / 0.4 / 0.5>
**Gates:** <release gate or null>
**Depends on:** <list of completed plans>
**Blocks:** <list of downstream plans>
**Parent:** [substrate_binding_persistence.md](substrate_binding_persistence.md)
**Related:** <links to sibling plans, living docs>

## Goal (1 paragraph)

## Hypothesis (falsifiable claim)

## Dependencies (scaffolding audit — what exists, what's new)

## Stages

### Stage 1 — mechanism + metric
What's built: ...
Pass gate: ...
Tests: ...

### Stage 2 — fixture validation
What's built: ...
Pass gate: ...
Tests: ...

### Stage 3 — real-data sweep + pre-merge review
What's built: ...
Pass gate: ...
Baseline: <TF-IDF / metadata-grep / LRU / OpenCLIP>
Reviewers: Executor + Architecture lenses
Tests: ...

## Pass criteria (maps to version gate)

## Deferred follow-ups

## Load-bearing invariants (filled in AFTER shipping)
```

The Stage 1/2/3 template is the P2 pattern that shipped successfully. It's mandatory for every substrate phase going forward.

## What this PR does

This proposal PR lands:

1. **This proposal doc** (`substrate_binding_split_proposal.md`) — the narrative you're reading.
2. **`sem_execution_hook.md`** — the companion plan for the production SEM forward path gap discovered during the audit. This is a FULL plan, not a proposal, because the gap is concrete and the scope is scoped.
3. **Updates to [substrate_binding_persistence.md](substrate_binding_persistence.md)** — add a header pointer to this proposal, mark the monolithic plan as "splitting" until approval lands.
4. **Updates to [docs/plans/README.md](README.md)** — add the proposal + sem_execution_hook to the Active section with correct status, keep substrate_binding_persistence as the index until the split completes.

This PR does NOT yet create the 8 per-phase plan files. Those land in follow-up commits/PRs once you've approved the split narrative. The first one to land is `substrate_p3_5_persistence_snapshot.md` (minimal shell), followed by `substrate_p3a_episode_binding.md`. Both will be created at the kickoff of the new substrate session.

## Risks I want you to pressure-test

1. **Does 8 files feel like over-splitting?** The alternative is 2–3 mid-scale files (e.g., "substrate_p3_block" = P3a+P3b+P3.5, "substrate_p4_block" = P4 alone, "substrate_p5_p6_p8_block" = 0.5 phases). Mid-scale keeps related work together but loses the "one plan, one gate, one set of invariants" clarity. I lean on 8 because the P2 experience showed 3 forced pivots inside a single plan; each of the 8 needs room to pivot independently.

2. **`sem_execution_hook.md` as a parallel session.** This works cleanly ONLY if the parallel session respects the worktree + `PYTHONPATH` discipline from `feedback_installed_package_shadows_worktree.md`. Otherwise the two sessions will collide on imports. I'll spell this out in the kickoff prompt. It's worth the coordination cost because SEM execution is genuinely independent of substrate mechanism work.

3. **P3.5 opening before P3a is a reversal of the parent plan order.** The parent plan says P3a → P3b → P3.5. I'm proposing P3.5 Stage 1 (minimal shell) → P3a → P3b → P3.5 Stage 2+3. The reason is P3a's "persistence round-trip" pass criterion. If you'd rather keep the original ordering, P3a's round-trip test becomes a stub that gets filled in later — that works too but leaves a visible TODO.

4. **`prompt_b4_replanning.md` as its own plan** is a further split that I'm pre-committing to but not executing in this PR. If the user opens B3+B5 without B4, it's fine. If the user opens B4 first, it should live in its own plan file from the start because its 1.0-gating nature deserves full attention.

## Not in this PR

- The 8 per-phase plan files. Those land at kickoff of each phase's session.
- Any code changes. This is pure planning.
- Changes to archived plans. `archive/substrate_plan.md` stays as-is.

## Decision required

If you approve this split:
- I'll write `sem_execution_hook.md` as a full plan doc in this same PR
- I'll update the parent plan header + plans/README.md
- I'll commit the proposal + sem_execution_hook + updates + kickoff prompts + memory entry as one atomic planning pass

If you want to redirect:
- Smaller split (2–3 blocks instead of 8)?
- Keep P3a → P3b → P3.5 ordering?
- Defer sem_execution_hook?
