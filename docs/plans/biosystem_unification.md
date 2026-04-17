# Bio-System Unification — central tracking doc

**Status:** Living index doc. Captures the structural-enforcement pattern that came out of `executor_bootstrap_unification.md` and applies it to the rest of the bio-system construction surface.
**Created:** 2026-04-14
**Parent pattern:** [executor_bootstrap_unification.md](executor_bootstrap_unification.md) — the first instance of this work, which pushed the `ToolPainBridge` invariant into `build_executor`.
**Companion ceiling:** [agent_factory_canonicalization.md](agent_factory_canonicalization.md) — the eventual end state where `AgentFactory.create_agent` is the only door for constructing a Maxim agent. Bio-system unification is the *floor* it rests on.

## Why this document exists

The `executor_bootstrap_unification.md` PR closed a class of bugs that had bitten the Maxim codebase three times in three weeks (sem_execution_hook Stages 1, 2, and a planned 2c). All three were the same shape: "an agent entry point forgot to wire bio-system X, the system silently no-op'd on that path, the missing learning signal corrupted everything downstream until someone noticed in production." The fix was structural: push the invariant into the constructor signature so forgetting becomes a `TypeError`, not a silent no-op.

That fix only covered ONE bio-system (`ToolPainBridge`). The rest of the bio-system construction surface has the **same shape and the same risk** on every other entry point. This document catalogs the candidates, ranks them by silent-failure blast radius, sequences them in dependency order, and tracks status as each one ships.

## Lessons learned (the things to remember)

These came out of the `executor_bootstrap_unification.md` work and the pre-merge review round. Every bio-system unification plan in this catalog should encode them.

### L1 — Silent failure mode is the primary trigger, not the headcount

Don't wait for three repeated bugs before pushing an invariant down a layer. The rule is:

- **Loud-failure recurring bugs** (raise → log → user sees it) → helper-discipline + a CLAUDE.md comment is fine. The next caller hits the failure during testing and copies the right pattern.
- **Silent-failure recurring bugs** (no exception, no log, just wrong behavior or missing learning) → push the invariant into the type/constructor signature on instance ONE if the path is critical. By instance THREE the rule is no longer optional — the next entry point will reproduce it.

The `ToolPainBridge` instance hit threshold three before being structurally fixed. The bus subscribers below have the same silent failure mode and **should not wait for three**.

### L2 — Audit before designing

Every bio-system unification starts with a written audit of every call site (file:line, what's wired, what's missing, what an explicit decision would look like). The `executor_bootstrap_unification.md` audit found two pre-existing silent-no-op bugs the original Stage 2c plan had no idea existed (api.py headless mode + simulation/tools.py sub-AUT). Both surfaced because the audit forced every call site to explain itself.

**Rule:** for each unification plan, write the audit before any code. The audit IS the design — once you can fill in the table, the plan writes itself.

### L3 — Pre-merge review is non-negotiable

Two parallel reviewers (Executor lens + Architecture lens), independent, then fold cross-confirmed findings first. The `executor_bootstrap_unification.md` review caught:

- A docstring in an unrelated module (`pain_interceptor.py`) that was teaching the forbidden retrofit pattern → would have re-introduced the bug class on the next read.
- A signature design smell (bridge gating on `pain_bus|pain_detector` instead of `nac`) that was forcing the sim orchestrator into a no-op `PainDetector()` workaround.
- A latent `UnboundLocalError` in cli.py from a pre-existing bug the migration was about to make user-visible.

Tests caught zero of those. Reviews caught all three, and two of three were cross-confirmed (highest trust). Every plan in this catalog gets a review round before merge.

### L4 — Bridge construction goes on `nac is not None`, not subscription source

The C2 design lesson from `executor_bootstrap_unification.md` generalizes: **the primary value of a bio-system bridge is direct attribution; subscription is the secondary out-of-band path.** When designing a new builder, gate construction on the *learning subject* (NAc, Hippocampus, ATL), not on the *signal source* (PainBus, PainDetector, ReactionBus). A caller that has the learning subject and wants direct attribution should not need to pass a no-op signal source to trick the constructor.

### L5 — Declared fields beat attribute stashes

When a builder needs to surface an optional object to its caller (e.g., `Embodiment` from `build_executor`), declare the field on the constructed object. Don't `obj._foo = bar  # type: ignore[attr-defined]`. Untyped attribute stashes silently drop to `None` if a wrapper sits between the builder and the reader, and they're invisible to mypy.

### L6 — One unification per PR

The temptation is to write a mega-plan that fixes all six candidates in one PR. Resist. Each unification has its own audit surface, its own review questions, its own migration risk. Combining them turns a clean N-PR sequence into one ~2000-LOC monster nobody reviews well. The pattern is: one structural fix per PR, each unblocks the next.

### L7 — The doc + memory refinement is part of the work

Every unification plan ships with: (a) `CLAUDE.md` invariant update, (b) optionally a new `feedback_*.md` memory file, (c) `docs/plans/README.md` index update, (d) any cross-references in related plans. This is not optional polish — it's how the next session learns what shape the codebase enforces. The `executor_bootstrap_unification.md` PR allocated explicit scope for this and it added ~50 LOC of doc edits that materially changed how the next reader will think.

## General guidelines for any bio-system unification

When you open one of the shells below, follow this pattern:

1. **Audit pass** (written, ~300-500 words, as the first plan-doc section). Every call site of the system in `src/maxim/`. For each: file:line, what's currently wired, what's missing, what an explicit decision would look like.
2. **Surface pre-existing bugs.** The audit will surface 1-3 silent gaps that were never caught before. Document them in the plan as "pre-existing bugs surfaced." Decide for each whether to fix in this PR or defer to a follow-up plan.
3. **Design the constructor signature.** Required keyword args for the deps that gate construction; explicit `None` opt-out for sandboxes/tests; fail-fast preconditions at the top BEFORE any object construction.
4. **Apply L4** — gate construction on the learning subject, not the signal source.
5. **Apply L5** — declared fields, not attribute stashes.
6. **Migration in a second commit** — call sites updated one by one, each with an explicit decision.
7. **Doc + memory refinement in a third commit** — CLAUDE.md invariant, optional memory file, plan index update, cross-references.
8. **Pre-merge review round** — two parallel reviewers, fold cross-confirmed first.
9. **PR.**

Plan doc shape: Status / Goal / Audit / Design / Migration plan / Doc + memory refinement / Pass criteria / Out of scope / Pre-merge review questions / Estimated scope.

Test scope per plan: tests for the **new** constructor only. Bulk test migration is deferred to `agent_factory_canonicalization.md` Stage F6, which has hard-enforced test review. Tests that fail at suite-run time on a `TypeError` are part of the migration's audit signal, not a regression.

## Catalog of bio-systems needing unification

Ranked by silent-failure blast radius (highest first). Each row links to its shell plan (creating now as part of this commit) and lists dependencies.

| # | System | Shell plan | Status | Silent-failure risk | Depends on | Parallel-safe with |
|---|---|---|---|---|---|---|
| 0 | `ToolPainBridge` (the example) | [executor_bootstrap_unification.md](executor_bootstrap_unification.md) | **SHIPPED** (in review on `feat/exec-bootstrap-unify`) | Critical — 3 instances over 3 weeks | — | — |
| 1 | PainBus subscribers | [pain_bus_unification.md](pain_bus_unification.md) | DRAFT shell | **Critical** — same shape as #0, currently scattered across 5+ entry points | None (independent) | #2 |
| 2 | ReactionBus subscribers | [reaction_bus_unification.md](reaction_bus_unification.md) | DRAFT shell | High — typed isolation surface, same shape | None (independent) | #1 |
| 3 | MemoryHub bridges | [memory_hub_unification.md](memory_hub_unification.md) | DRAFT shell | Medium — silent partial coordination, fewer call sites | #1, #2 (consumes both buses) | — |
| 4 | DefaultNetwork | [default_network_unification.md](default_network_unification.md) | DRAFT shell | Medium — silent fear-gate skip | #1 (consumes PainBus) | #3 |
| 5 | Bio-stack umbrella | [bio_stack_unification.md](bio_stack_unification.md) | DRAFT shell | The umbrella that subsumes #1-#4 in a single builder | #1, #2, #3, #4 | — |
| 6 | LearnedToolIndex registration | (notes only — separate shape) | NOTE | Low — different shape (registry coupling, not bus subscription) | — | any |

### Item 6 — LearnedToolIndex (different shape, not getting a shell yet)

The bug shape is: "tool registered in `ToolRegistry`, forgotten in `LearnedToolIndex`." The fix is not a `build_*` constructor — it's coupling: `ToolRegistry.register()` should call `index.register_tool()` automatically when an index is bound. This is closer to the `LearnedToolIndex` constructor accepting a `tool_registry=` parameter and auto-subscribing to register events. Different surgery from the bus unifications. Worth its own plan when a second instance of the bug surfaces, OR opportunistically during one of the other plans if the same session is touching the registry. Logged here so it doesn't get lost.

## Logical chain + parallel-safety

```
                                executor_bootstrap_unification.md  (SHIPPED)
                                              │
                                              ▼
                  ┌────────────────────────────┴───────────────────────────┐
                  │                            │                           │
                  ▼                            ▼                           ▼
  pain_bus_unification.md       reaction_bus_unification.md       (independent — can ship in parallel)
                  │                            │
                  └─────────────┬──────────────┘
                                │
                                ▼
                  memory_hub_unification.md  (consumes both buses)
                                │
                                ▼
                  default_network_unification.md  (consumes PainBus + NAc)
                                │
                                ▼
                  bio_stack_unification.md  (umbrella over all of the above)
                                │
                                ▼
                  agent_factory_canonicalization.md  (the ceiling — calls bio_stack + build_executor)
```

### What can run in parallel

- **#1 + #2** (PainBus + ReactionBus) — fully independent. Touch different files. Two parallel sessions, no overlap. **Recommended:** ship these together as the next two PRs after the executor unification merges.
- **#3 + #4** (MemoryHub + DefaultNetwork) — both depend on #1 (PainBus). Once #1 ships, these can run in parallel sessions because they touch different modules (`integration/memory_hub.py` vs `default_network/network.py`). DefaultNetwork also needs NAc but doesn't touch the same files as MemoryHub.
- **#5** (bio_stack) — must come AFTER #1-#4 because it composes them. Single PR; not parallel-safe with anything in this list.
- **AgentFactory canonicalization** — must come AFTER #5. Single multi-PR plan; not parallel-safe with anything in this list.

### What CANNOT run in parallel

- Any of #1-#5 with `agent_factory_canonicalization.md` Stages F1-F5 — they touch overlapping files (cli.py, orchestrator.py, agentic_runtime.py, api.py).
- #5 with #1-#4 — bio_stack composes them, so it needs them done.
- Any of these with a substrate-track plan that touches the same bio-system file (verify with the substrate plan's "off-limits files" section before opening).

### Recommended ordering

1. **Wave 1 (parallel):** #1 PainBus + #2 ReactionBus. Two PRs, can ship in either order. ~2-3 days each.
2. **Wave 2 (parallel):** #3 MemoryHub + #4 DefaultNetwork. Two PRs after Wave 1 lands. ~1-2 days each.
3. **Wave 3 (single):** #5 bio_stack. One PR after Wave 2 lands. ~3-5 days. This is the umbrella that makes AgentFactory canonicalization a downhill rewrite.
4. **Wave 4 (multi-PR plan):** `agent_factory_canonicalization.md` Stages F1-F6. Multi-session.

Total estimate to reach the structural ceiling: ~2-3 weeks of focused work, spread across ~8-10 PRs, depending on how many parallel sessions run.

## Status tracking

| Wave | Plan | Branch | PR | Status | Date |
|---|---|---|---|---|---|
| 0 | executor_bootstrap_unification | feat/exec-bootstrap-unify | #114 (merged) | **SHIPPED** | 2026-04-14 |
| 1 | pain_bus_unification | feat/pain-bus-unification | (PR pending) | **Audit + builder + migration committed; pre-merge review next.** Closes Gap A (3 CLI sites silently skipping NAc bus subscription). Gap B (DefaultNetwork split ownership) deferred to memory_hub_unification.md per no-band-aid rule. Gap C (api.py headless) structural side resolved, user-facing API question stays at agent_factory_canonicalization.md F5. | 2026-04-14 |
| 1 | reaction_bus_unification | feat/reaction-bus-unification | (PR pending) | **Audit + builder + factory fix committed; pre-merge review next.** Surface differs from PainBus (N=1 construction site). Builder exists for Wave 3 downstream sequencing (`build_bio_stack` requires `build_reaction_bus` BEFORE `build_pain_bus`). Gap A (CerebellumModulator factory silently dropping `reaction_bus=`) fixed preemptively — factory has zero production callers today but the parameter now flows correctly. | 2026-04-16 |
| 2 | memory_hub_unification | — | — | Shell only — INHERITS Gap B from pain_bus_unification (DefaultNetwork split subscriber ownership). Address by giving DefaultNetwork a `hippocampus=` kwarg or routing it through `build_pain_bus(...)` once MemoryHub is structurally enforced. | — |
| 2 | default_network_unification | — | — | Shell only | — |
| 3 | bio_stack_unification | — | — | Shell only | — |
| 4 | agent_factory_canonicalization | — | — | Running doc — INHERITS Gap C from pain_bus_unification (api.py headless `pain_bus=None`). Stage F5 owns the user-facing default-on-vs-default-off bio-learning decision for headless `pymaxim` agents. The structural construction door already exists (`build_pain_bus`). | 2026-04-14 |

## Cross-references

- [feedback_structural_enforcement_over_helper_discipline.md](../../.claude/projects/-Users-dennyschaedig-Scripts-Maxim/memory/feedback_structural_enforcement_over_helper_discipline.md) — the silent-failure-mode rule articulated in memory.
- [feedback_review_before_ship.md](../../.claude/projects/-Users-dennyschaedig-Scripts-Maxim/memory/feedback_review_before_ship.md) — pre-merge review round template (two parallel lenses, fold cross-confirmed first).
- [feedback_audit_before_building.md](../../.claude/projects/-Users-dennyschaedig-Scripts-Maxim/memory/feedback_audit_before_building.md) — audit-first discipline (the `executor_bootstrap_unification` audit cut ~40% of original scope by surfacing pre-existing infrastructure).
- `CLAUDE.md` "Push silent-no-op invariants into types, not helpers" lesson — the load-bearing version of L1.
- `CLAUDE.md` "build_executor is the canonical bridge wiring site" invariant — the precedent every plan in this catalog mirrors.
