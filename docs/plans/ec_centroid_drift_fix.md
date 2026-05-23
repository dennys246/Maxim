# EC text-modality centroid drift fix

**Target version:** 0.9.1 or 1.0 (decision gated on Phase 2 regression results).
**Status:** Draft. Plan written 2026-05-23 after the paraphrase-collapse diagnostic ([24_roy_paraphrase_diagnostic.md](../experiments/24_roy_paraphrase_diagnostic.md), [PR #259](https://github.com/dennys246/Maxim/pull/259)) returned `CENTROID_DRIFT_COLLAPSE`.
**Owns:** `scripts/diagnose_roy_paraphrase_collapse.py` (extend), `data/roy_paraphrase_pairs.json` (reuse), `src/maxim/similarity/ec.py` (Phase 3 change), `docs/experiments/24_roy_paraphrase_diagnostic.md` (companion update).
**Companion plans:** [release_0_9_1.md](release_0_9_1.md) (0.9.1 Roy-3 ships independently; this can land alongside or in 1.0), [v1_refinement.md](v1_refinement.md) (V1 cross-session validation silently depends on this fix), [roy_5_encoder_alignment_disambiguator.md](roy_5_encoder_alignment_disambiguator.md) (Roy-5 / JEPA direction stays a separate gap, but downstream of this fix), [jepa_cross_modal_alignment.md](jepa_cross_modal_alignment.md) (1.2+ research; same downstream relationship).

## Why this plan exists

The 24 paraphrase diagnostic shipped one finding and dismissed two worries:

1. **Encoder is fine.** Pre-EC cosines separate paraphrase pairs (0.58–0.92) from semantically-distant distractors (0.07–0.30) cleanly with no overlap. The N=1 worry on H1a is not an embedding-quality problem.
2. **Isolated EC clustering is fine.** Fresh EC per pair → 10/10 pair collapse, 0/5 distractor collapse. The pattern-completion mechanism works correctly when uncontaminated.
3. **Sequential EC clustering breaks.** Walking 22 unique strings through one shared EC (the Roy production regime) produces runaway centroid drift: 19 of 20 unique pair-strings land in a single mega-node at threshold 0.40. The 60% distractor "collapse" is a side-effect of that mega-node.

Mechanism: [`EntorhinalCortex.pattern_complete_or_separate`](../../src/maxim/similarity/ec.py#L374-L387) updates the matched node's centroid as a running mean for non-frozen modalities. `frozen_centroid_modalities` only freezes `"interoception"`. Text drifts. Successive low-but-above-threshold matches pull the centroid toward a generic "second-person body sensation" prototype that pattern-completes everything sent to it.

**Implication:** the persona-convergence gap is **not purely cross-modal**. Any persona work that depends on text-modality EC nodes persisting as distinct concept clusters across a session is silently corrupted by centroid drift. 1.0 V1 cross-session validation silently depends on this same machinery — cross-session "recall" of a previously-encoded concept drifts toward "anything second-person-sensory" the more text the substrate has seen.

## Framing rule

**Run all configurations as ONE matrix experiment, not three sequential PRs.** The diagnostic doc named three follow-ups (decomposition, frozen-centroid, threshold sweep) as if they were independent fixes. They are NOT independent — they interact at the EC pattern-completion layer. Doing them as a 2×2×4 matrix in one script run gives apples-to-apples comparison and avoids the "ship A, find A is incomplete, ship B, find B regressed P2" pattern.

**Validate behaviorally before locking into 1.0.** A substrate-level diagnostic can pass clean-data tests but fail to change Roy persona convergence. The link from "EC clusters cleanly" to "persona inertness resolves" is empirical, not inferred. Phase 4 closes that gap.

## Sizing

| Phase | Item | LOC | Persistence | Risk |
|---|---|---|---|---|
| 1 | Extend `diagnose_roy_paraphrase_collapse.py` to sweep 2×2×4 matrix | ~150 | none | low — script-only |
| 2 | Re-run P1 paraphrase + P2 reward-modulation sweeps with winning config | ~50 (scripts touch) | none | **high — regression gate** |
| 3 | EC config change: add `"text"` to `frozen_centroid_modalities` and/or threshold tune | ~10 src + ~80 tests | possible `aut_ec.json` semantic change | medium |
| 4 | Roy-2c re-run with fixed EC; compare persona convergence to Roy-2c baseline | ~30 (analyzer touches) | none | medium — runner time |
| 5 | Update `v1_refinement.md` V1 cross-session validation plan; named pointer here | ~50 (doc only) | none | none |
| **Total** | | **~370 LOC + ~3 runner days** | one possible breaking change | |

The maximum scope assumes the winning config is "frozen text centroid + threshold 0.50" (the most likely Phase 1 outcome by mechanism). A more aggressive fix (member-count cap, sparse coding) would expand Phase 3 substantially; this plan does not pre-commit to those.

## Phase 1 — Matrix diagnostic (one PR)

**Goal:** name the winning EC configuration cell. Extend [`scripts/diagnose_roy_paraphrase_collapse.py`](../../scripts/diagnose_roy_paraphrase_collapse.py) to sweep a config matrix; report a heatmap-style table; choose the smallest-change config that satisfies the existing gate.

### Matrix axes

| Axis | Values | Rationale |
|---|---|---|
| `decomposition` | off, on (`MAXIM_CONCEPT_DECOMPOSITION=1` shape — wire a `ConceptDecomposer` into `LinguisticEncoder`) | Tests whether noun-chunked concepts ("food", "belly") suffer less drift than whole-sentence embeddings. Confirms the drift IS centroid-update-driven, not encoder-driven. |
| `frozen_text_centroid` | off (current), on (add `"text"` to `frozen_centroid_modalities`) | The bio-correct fix candidate. Mirrors existing `"interoception"` treatment. |
| `threshold` | 0.40 (current), 0.50, 0.60, 0.70 | Tightens rejection band. 0.40 is the P1-tuned default; 0.70 was the P2 frozen-pair-collapse winner on real-embedding sweeps. |

Total: 2 × 2 × 4 = 16 cells. Each cell emits sequential + isolated pair/distractor rates from the existing fixture.

### Winning-cell criteria (pre-registered)

The winning cell is the one that:

1. **Sequential pair collapse ≥ 70% AND sequential distractor collapse < 30%** (the original gate), AND
2. Among the cells that satisfy (1), the **smallest config delta from current defaults** — preferring frozen-centroid over threshold change, preferring smaller threshold change over larger.

If no cell satisfies (1), the diagnostic plan loops back: extend the matrix axes or commit to a more aggressive fix (member-count cap, sparse coding).

### Test surface

- The existing 10 pair / 5 distractor fixture stays unchanged. Reproducibility against the shipped diagnostic's numbers is part of the regression guard.
- Output: one JSON file per cell (`/tmp/roy_paraphrase_collapse_d{0,1}_f{0,1}_t{40,50,60,70}.json`) + a summary CSV/Markdown table the diagnostic doc consumes.

### Definition of Done

- Matrix sweep runs cleanly with `MAXIM_SUBSTRATE_PATH=1`.
- Summary table emitted to stdout AND persisted alongside the per-cell JSON outputs.
- Companion update to `docs/experiments/24_roy_paraphrase_diagnostic.md` (or a new `25_*.md`) reports the winning cell + rationale.
- PR labels the winning config as the **Phase 3 target**, not "merge me — I'm the fix." Phase 2 regression gate is the actual merge gate.

### What this phase does NOT do

- Does NOT change `ECConfig` defaults. Matrix runs use ad-hoc `ECConfig` instances passed into the diagnostic.
- Does NOT change persisted `aut_ec.json` format. Read-only on the substrate.
- Does NOT pre-commit to the winning fix shipping in 0.9.1 vs 1.0 — Phase 2+4 results decide.

## Phase 2 — Regression-guard against P1+P2 (one PR)

**Goal:** confirm the Phase 1 winning config doesn't regress the existing 91.7% P1 paraphrase-collapse pin or the +56pp P2 reward-modulation target gain.

### Why this is the load-bearing phase

P1 ([docs/experiments/p1_recognition_sweep.md](../experiments/p1_recognition_sweep.md)) and P2 ([docs/experiments/p2_reward_modulation_sweep.md](../experiments/p2_reward_modulation_sweep.md)) shipped on the current EC behavior (running-mean centroid, threshold 0.40). Both are 0.3-minimum gate references for 1.0. Any change that "fixes Roy drift" but drops P1 below ~85% or P2 below ~+40pp is **net-negative** — it'd close one gap by re-opening two harder ones.

### Implementation

- Re-run [`docs/experiments/protocols/p2_reward_modulation_reproduction.md`](../experiments/protocols/p2_reward_modulation_reproduction.md) with the Phase 1 winning config. Compare to the pinned `+56.0 ± 29.0 pp target gain / 0.0 ± 0.0 pp distractor drift / 94% monotone / 9-of-10 seeds`.
- Re-run P1's paraphrase-collapse sweep with the winning config. Compare to the 91.7% pin.
- Both must hold (within seed variance) for the Phase 1 winning config to graduate. If either regresses materially, **return to Phase 1 with the next-smallest-delta candidate**.

### Test surface

- The two sweep protocols are already documented; reproduction is a re-run, not new code.
- The regression delta tolerance is **pre-registered before re-run** to prevent post-hoc threshold movement: P1 ≥ 85% (vs 91.7% pin), P2 ≥ +40pp (vs +56pp pin), distractor drift ≤ +5pp (vs 0.0pp pin).

### Definition of Done

- P1 + P2 re-run results documented in a companion experiment doc (`docs/experiments/25_centroid_drift_fix_p1_p2_regression.md` or similar).
- Either: (a) winning config holds both gates → Phase 3 is unblocked, OR (b) at least one gate fails → return to Phase 1 with a smaller-delta candidate.

### What this phase does NOT do

- Does NOT run Roy iterations. That's Phase 4.
- Does NOT change ECConfig defaults or persisted formats. Read-only on the EC code; sweep configs are passed at construction time.

## Phase 3 — Ship the EC change to main (one small PR)

**Goal:** make the Phase 1+2-validated config the default in `ECConfig`. Smallest possible code surface.

### Implementation (depends on winning config)

The most likely winning config is `frozen_text_centroid=on, threshold=0.50, decomposition=off`. Under that:

- Update [`ECConfig.frozen_centroid_modalities`](../../src/maxim/similarity/ec.py#L199) default from `frozenset({"interoception"})` to `frozenset({"interoception", "text"})`.
- Update [`ECConfig.pattern_complete_threshold`](../../src/maxim/similarity/ec.py#L186) default from `0.40` to whatever Phase 1+2 named.
- Add a unit test pinning the diagnostic's sequential numbers post-fix (sequential pair collapse ≥ 70%, sequential distractor collapse < 30%, EC node count after 22-string walk ≥ N where N is the Phase 1 winning value).
- Add a regression test pinning P1 + P2 sweep numbers (or note that the existing P1+P2 tests carry the regression guard naturally).

### Persistence semantics

- Frozen-centroid affects writes only — existing `aut_ec.json` files load identically. The centroid stored on disk is whatever it was when persisted; reading does not re-frozenify or un-frozenify.
- Threshold change does not affect persisted shape; only future pattern-completion decisions on loaded EC state change.
- **No `_format_version` bump required** — the file shape is unchanged.

### What this phase does NOT do

- Does NOT add new ECConfig fields beyond the two existing ones. If the Phase 1 winner is "member-count cap" or "sparse coding", that's a substantially bigger Phase 3 that needs its own design pass — this plan does not pre-commit to it.
- Does NOT migrate existing `aut_ec.json` files. They keep their current centroids; only future updates use the new defaults.

### Definition of Done

- ECConfig defaults updated; tests pass.
- `python -m pytest tests/ -x -q -m "not slow" --ignore=tests/integration/test_memory_hub.py` clean.
- `mypy` clean on the affected files.
- Diagnostic re-run on main confirms the fix.

## Phase 4 — Behavioral validation in Roy harness (one PR + runner time)

**Goal:** test whether the substrate-level fix actually moves Roy persona convergence. This is the question Phase 1+2+3 cannot answer.

### Implementation

- Re-run Roy-2c ([scenarios/roy/roy_2c_iteration.yaml](../../scenarios/roy/roy_2c_iteration.yaml)) with the fixed EC. Same priming + holdout + arms structure as Roy-2pc.
- Compare `sense_food_source` counts across arms A / B / C against the Roy-2c baseline ([docs/experiments/20_roy_2c.md](../experiments/20_roy_2c.md)).
- Pre-registered pass criterion: **Arm A > B > C on `sense_food_source` counts AND the gap is materially larger than Roy-2c baseline's gap** (specific delta target named in the Phase 4 PR description, based on Roy-2c's pre-fix numbers).
- Cost: ~3 arms × Roy-2c runtime (cheapest behavioral fixture in the Roy suite).

### Decision routing

| Result | Verdict | Next step |
|---|---|---|
| Persona convergence sharpens materially | **Centroid drift was load-bearing for Roy inertness** | Fix ships in 0.9.1 (after Roy-3) or 1.0 V1 — operator decides. JEPA / cross-modal binding becomes 1.1+ work. |
| Persona convergence unchanged | **Centroid drift was real but not the dominant Roy failure mode** | Fix still ships (substrate hygiene + V1 prerequisite), but JEPA / cross-modal binding stays a 1.0-or-1.1 gate. Roy-5 plan unchanged. |
| Persona convergence regresses | **The fix surfaced an interaction we missed** | Halt Phase 3 merge to main; return to Phase 1 with the new finding scoping a different cell. |

### Test surface

- Roy-2c is reproducible via the existing runner; the only new code is whichever analyzer surfaces the persona-convergence delta vs the pre-fix baseline.
- If Roy-2c baseline numbers are not currently captured in a machine-readable form, this phase ships the baseline-capture alongside the post-fix comparison.

### Definition of Done

- Experiment doc (`docs/experiments/26_centroid_drift_fix_behavioral_validation.md` or similar) names the verdict + the routing decision.
- 0.9.1 vs 1.0 ship decision documented.

### What this phase does NOT do

- Does NOT re-run the full Roy suite (1a, 1b, 2, 2pc, 4, 5a). Roy-2c is the diagnostic crucible because it carries the engineered food/hunger overlap that maximally exposes the persona-convergence signal.
- Does NOT touch the cancelled `cross_modal_substrate_binding.md` plan's status — that stays cancelled regardless of Phase 4 outcome.

## Phase 5 — 1.0 V1 implications (planning only, no code)

**Goal:** thread the Phase 1-4 results into the 1.0 release plan.

### Implementation

- Update [`v1_refinement.md`](v1_refinement.md) V1 cross-session validation section to note that the EC centroid-drift fix is a prerequisite for credible V1 runs. Without it, "concept persists across sessions" silently degrades as more text accumulates.
- Add a named pointer from `v1_refinement.md` to this plan + the Phase 1+2+3+4 PRs.
- Update [`release_0_9_1.md`](release_0_9_1.md) only if Phase 4 routes the fix to 0.9.1 (operator decision).
- Add the diagnostic's lesson to `CLAUDE.md`'s "Lessons learned (bugs that bit us)" section: the running-mean centroid drift class of bug.

### Definition of Done

- `v1_refinement.md` updated.
- This plan + the Phase 1+2+3+4 PRs cross-linked from `docs/plans/README.md`.
- `CLAUDE.md` lessons updated.

## Rejected approaches

### Decomposition as the fix

The naive "just turn on `MAXIM_CONCEPT_DECOMPOSITION=1`" fix would reduce drift because shorter noun-chunked strings ("food", "belly") have fewer members per node and less centroid travel. Two reasons not to ship it as the fix:

1. **It doesn't address the root cause.** The centroid drift dynamic is still present; decomposition just shortens the inputs that trigger it. A longer session with more strings re-introduces drift.
2. **Over-fragmentation risk.** "food", "fullness", "hunger" become three separate concepts when they should arguably be one cluster. The persona-convergence work is already starved for shared concept structure; making concepts MORE atomized would be net-negative.

The matrix runs decomposition as confirming evidence that the drift IS centroid-update-driven, not encoder-driven. It does not ship as the fix.

### The three follow-ups as three sequential PRs

The 24 diagnostic doc named decomposition, frozen-centroid, and threshold sweep as three sequential follow-ups. That sequencing sets up the "ship A, find A is incomplete, ship B, find B regressed something, ship C" pattern that has cost real time on the Roy iteration arc (see [`feedback_three_iteration_metric_pivot.md`](../../.claude/projects/-Users-dennyschaedig-Scripts-Maxim/memory/feedback_three_iteration_metric_pivot.md)). Matrix-first is faster, gives apples-to-apples comparison, and exposes the interaction between axes (frozen-centroid + low threshold might pass; running-mean + high threshold might pass; the matrix tells you which).

### Member-count cap or sparse coding as the fix

A cap (force pattern separation after N members per node) or sparse coding (k-WTA on EC activations) would bound the worst-case drift more aggressively than frozen-centroid or threshold change. Both are substantially bigger changes:

- Member-count cap: needs a per-node member tracker, a separation policy at the cap boundary, persistence-format extension.
- Sparse coding: re-architects how `pattern_complete_or_separate` scores matches.

Either could become Phase 3's actual implementation IF Phase 1 shows that neither frozen-centroid nor threshold tune is sufficient. This plan does not pre-commit; it commits to Phase 1 naming the necessary fix scope.

## Open questions for reviewer

1. **Phase 4 routing decision (0.9.1 vs 1.0):** if Phase 4 passes, should the fix ship in 0.9.1 alongside Roy-3, or wait for 1.0? Argument for 0.9.1: it's a small structural fix with regression guards; shipping sooner means V1 runs benefit. Argument for 1.0: 0.9.1 is "annotation patterns" themed; the EC fix is structurally adjacent but thematically different.
2. **P1+P2 regression tolerance:** the pre-registered tolerances (P1 ≥ 85%, P2 ≥ +40pp) are conservative but not nailed-down. Reviewer can tighten before Phase 2 runs.
3. **Roy-2c baseline capture format:** if the existing Roy-2c numbers are documented as prose-only, Phase 4 ships baseline-capture as a side-deliverable. Confirm the format (CSV? JSON? in-doc table?) before Phase 4 runs.
