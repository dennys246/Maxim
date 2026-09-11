# 1.3 quality burndown — chipping at the 1.1/1.2 weak areas

**Drafted 2026-09-10**, after 1.2.1 shipped, from a three-lens sweep (score cards, plans +
deferred, self-identified debt + bugs ledger). This is the **incremental-improvement** track
that runs *alongside* the survival-world 1.3 build ([roadmap_1_3.md](roadmap_1_3.md)) — not part
of it. It is optional quality work: no behavioral-graduation row is `Stale` or `Broken`, so
nothing here gates a release.

## Three framing facts

1. **The score-card baseline is stale.** The newest card is the 1.1.0 re-score (2026-08-27,
   `docs/limits/score_cards/2026-08-27-claude.md`). **Three cards are owed: Codex 1.1.0, and
   both assessors' 1.2 cards.** Re-scoring is the highest-leverage move — everything below aims
   at current grades only once it lands.
2. **Most of the 1.1.0 card's *mechanical* complaints were already built in 1.2** — the
   don't-re-do list: `lint_version_sync.py`, `lint_prereg_precedes_data.py`,
   `lint_atomic_io_ratchet.py`, `test_api_surface.py` (`_API_VERBS`==README), the
   fix-touches-tests lint, the model-cache + slow-lane gates, the ARCHITECTURE EC correction,
   `pytest-timeout`, the N2 create→shutdown→load round-trip test, and the first god-function
   extraction.
3. **The cheapest wins are guards the repo already named as owed** — the house style is "push
   the invariant into a guard," and several guards are explicitly flagged unbuilt.

## Batch 0 — commission the owed score cards *(process, not code; do first)*

Re-score at the 1.2 cut, both assessors; produce the Codex 1.1.0 card blind to the Claude one.
Without it the burndown targets three-cuts-old grades. Owner/assessor action.

## Batch 1 — the "guard" PR *(cheap lints closing unenforced invariants)*

| item | closes | size | files |
|---|---|---|---|
| **AST function-length ratchet** — pin `run_agentic_loop` / `start_simulation_mode` / `_main_impl` at ≤ current lengths, ratchet-down only | Maintainability C→C+ — *the only un-built mechanization on any axis* (08-27 card) | **S** | new `scripts/lint_function_length.py` + CI lint job; `tests/unit/test_lint_function_length.py` |
| **roadmap-16.10 lint** — a post-tag `src/`-touching commit must add an `[Unreleased]` line | the half-enforced "main ahead of PyPI" policy (CLAUDE.md §Versioning "convention, not yet enforced") | **S/M** | `scripts/` (alongside `lint_version_sync.py`) + CI |
| **dead-code orphan-module lint** — `.py` basenames absent from any `import` | Docs/Maintainability (found ~8,500 LOC once; "no automated test enforces") | **S/M** | new `scripts/lint_orphan_modules.py` + CI |
| **D63** — wire `pr_merge_readiness`'s required-checks-present check into a real merge gate | the "green PR that ran nothing" class (bugs ledger D63) | **M** | `scripts/pr_merge_readiness.py` (advisory today) + gate wiring |

## Batch 2 — the "claim-truth" PR *(doc-only; lifts Research Integrity B−→B, Docs C+)*

- **L185/L186** EARNED rows get data citations or dated data-lost annotations
- **D10** — Exp 45 `az_bin` claim-linkage annotation on the graduation row
- **D29** — annotate/re-cut the 45-series `dry_run` records + fix the citation
- **Tier-3 dispositions** — the 16 Pending rows, zero dispositions since 2026-05-27
- **D46** — delete the dead percept-transport reference in `simulation/sources.py`
- **D50** — warn on `party_mode`/`choice_resolution` no-ops in `load_campaign`; drop the dead schema field
- **ARCHITECTURE.md** Key-Modules refresh (missing `comms/ doctor/ hivemind/ imagination/ motion/ reactions/ roy/ tunnel/ default_network/ console/`)

Sizes: all **S** except Tier-3 (**M**, judgement per row).

## Batch 3 — the "small correctness" PR *(real user-facing fixes, reachable now)*

- **N1** — `api.campaign()` threads-or-rejects `npc_model`/`interactive`/`prompt_handler` (pairs with the done N2) — `src/maxim/api.py::campaign`, `tests/unit/test_api_core.py` — **S/M**
- **D32** — load the foundational preamble from `CONSTITUTION.md` as package data + a drift guard (pip users currently get an empty preamble) — `src/maxim/agents/llm_context.py::_load_foundational_context` — **S/M**
- **D84** — SUPERVISED sandbox honestly *refuses* when no approval callback is wired instead of silently auto-approving — `src/maxim/tools/sandbox.py::ExecuteSandboxScriptTool.execute` — **S**
- **D49** — benchmark honesty: apply-or-delete `weight`, fix the running half-mean, drop-or-ship the missing tier2/tier3 suite files — `simulation/benchmark.py` — **M**

## Batch 4 — larger, now-unblocked engineering *(opportunistic; competes with the survival build)*

- **Fail-loud Stage 3** (roadmap item 6) — narrow the measurement-path swallows; **green-lit** (Stage 2 measured *zero* firings) — `docs/plans/measurement_path_fail_loud.md` — **S/M**
- **God-function decomposition** (items 7 + 12; do together) — more extractions against the AST baseline the Batch-1 ratchet pins — `docs/plans/god_function_decomposition.md` — **L**
- **Widen mypy** beyond the ~5 public-API files — `.github/workflows/test.yml` — **M**
- **D19 architecture-debt** ratchet toward zero — **M–L**
- **Behavioral-suite thickening** for Exp 52/53b/56 (still ~5 files) — `tests/behavioral/` — **M–L**

## Explicitly OUT of scope

- **Hardware/research-gated** (owed, not tractable-software): Exp 54 B/C, Exp 50, Exp 44b at-power, D30/D31/D87, place-code default-ON. *One pullable slice:* the **L8 record-stamping** fix (stamp model/endpoint/n_ctx/quantization on every run record) is Exp 44b's prerequisite and is cheap on its own.
- **Deferred plans whose trigger 1.2 just fired — NEW-MECHANISM, not debt**, and each its own decision: **HF hive Phase 1** ([deferred/hf_hive_repository.md](deferred/hf_hive_repository.md), the launch companion) and the **reactive-peer-mesh C10 event-slice** ([reactive_peer_mesh_roadmap.md](reactive_peer_mesh_roadmap.md) §5).
- **DORMANT-for-a-reason:** D51 (LSHIndex — "needs a design decision, not a patch"), D45 (SpatialMemoryBridge — mark Dormant), D9 producers (mechanism dormant).

## Recommended pull

**Batch 0 + Batch 1 + Batch 2** as the 1.3 quality burndown — almost all S/M, directly answering
the weakest 1.1/1.2 grades (Maintainability, Research Integrity, Docs), and it is the
push-into-a-guard work the project values most. Batch 3 is a strong follow-on; Batch 4 runs
opportunistically rather than gating 1.3.
