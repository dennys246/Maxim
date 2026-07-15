# 31 — Roy iteration convergence/divergence audit

**Status:** Audit complete 2026-05-27.
**Trigger:** [CLAUDE.md Principle 4 operationalization](../../CLAUDE.md#working-principles-for-new-mechanisms) (PR #271, 2026-05-26).
**Branch:** `docs/roy-divergence-audit`.

## Method

CLAUDE.md Principle 4 introduces the convergence/divergence frame for Roy iterations: convergence = the next iteration narrows the same kind of issue; divergence = the next iteration surfaces a new failure mode the predecessor did not name. **Two divergence iterations in a row on the same mechanism trigger a bird's-eye bisect** ("step back from the mechanism layer, ask what's actually changing").

This audit walks every Roy iteration in [persona_convergence_crucible.md § Iteration log](../plans/deferred/persona_convergence_crucible.md) plus the per-iteration outcome docs (`docs/experiments/16_*` through `30_*`), classifies each adjacent pair, and surfaces any divergence streak that did NOT trigger a bisect when it should have.

**Unit of analysis.** Iteration *pairs*, not individual iterations. A single iteration alone is neither convergent nor divergent; the relationship between Roy-N and Roy-(N+1) is. Pairs across mechanism-cycle boundaries are flagged as transitions and not classified — Principle 4 explicitly notes that a Roy iteration on Wire-A vs one on the cradle harness are different mechanism cycles even if chronologically adjacent.

**Mechanism cycles (this audit's grouping).** Three cycles span the 11 Roy iterations + 1 bird's-eye bisect:

- **Cycle A: Cluster wire behavioral expression.** Roy-0 → Roy-1a → Roy-1b → Roy-2 → Roy-2pc → Roy-2c. Same underlying question: "the cluster_reward_bias wire writes correctly during priming; does it translate to behavior at test time?"
- **Cycle B: Encoder structure / binding rescue.** Roy-4 → Roy-5a. Same underlying question: "Roy-2c surfaced encoder-subspace disjointness; can a Hebbian binding rule rescue it (Roy-4), and what's the dimensional shape of the gap (Roy-5a)?"
- **Cycle C: Wire-A annotation pattern.** Roy-3a → Roy-3b → (Roy-3c-bisect) → Roy-3a-retry. Same underlying question: "does substrate-annotates-LLM-context behaviorally express priming bias the cluster wire's consumer cannot?"

The boundary between cycle A and cycle B sits at Roy-2c's H1a verdict; the boundary between cycle B and cycle C sits at Roy-5a's H1a-confirm + 0.9.1 wire merges.

## Classification table

| # | From → To | Mechanism cycle | Verdict | Evidence | Bird's-eye triggered? |
|---|---|---|---|---|---|
| 1 | Roy-0 → Roy-1a | A: cluster wire | **Convergence** | Same structural finding reproduced (cluster_reward_bias_l2 within 0.4%); Roy-1a *narrowed* the gap to consumer coupling (llm-primary doesn't read the bias) and surfaced salience_KS=0.879 as load-bearing positive signal | n/a (first pair) |
| 2 | Roy-1a → Roy-1b | A: cluster wire | **Convergence** | Roy-1b swapped to substrate-primary at test (Roy-1a's recommended next step); narrowed: structural wire healthy (2.4678 cluster_reward_bias_l2), but held-out percepts don't fire priming EC clusters → symmetric inertness across AUT modes | n/a |
| 3 | Roy-1b → Roy-2 | A: cluster wire | **Convergence** | Multi-arc priming tested as a methodology widening (Roy-1b's recommended path (a)); ruled out as ineffective (cradle stages don't shift AUT proposer's cold-start cluster monoculture); surfaced positive A-vs-C tool-family divergence under llm-primary | n/a |
| 4 | Roy-2 → Roy-2pc | A: cluster wire | **Convergence** | Positive control with engineered semantic overlap (Roy-2's recommended next test); reproduced behavioral inertness; cluster wire structurally healthy (cluster_reward_bias_l2 byte-identical across 5 iterations now) but inert even under positive control → empirical floor for Wire 1 escalation | n/a |
| 5 | Roy-2pc → Roy-2c | A: cluster wire | **Convergence (strong)** | Classic disambiguator pair: Roy-2pc named H1 vs H2; Roy-2c set min_confidence=0.0; H1 confirmed (encoder alignment failure), H2 refuted; priming and test EC clusters structurally disjoint at cluster-key resolution | n/a |
| — | Roy-2c → Roy-4 | (boundary A→B) | transition pair, not classified | Roy-2c's H1a verdict closed cycle A; Roy-4 opened cycle B as the cross-modal binding plan's pre-implementation gate | n/a |
| 6 | Roy-4 → Roy-5a | B: encoder structure | **Convergence** | Roy-4 confirmed disjointness at per-tick EC resolution + FAIL across the entire reasonable Hebbian-binding parameter sweep (0 priming↔test edges); Roy-5a narrowed to H1a verdict (different-dim subspaces 384 vs 768; food strictly interoception-modality during priming, no text-modality food rep) | n/a |
| — | Roy-5a → Roy-3a | (boundary B→C) | transition pair, not classified | Roy-5a closed cycle B (binding plan cancelled, encoder replacement → 1.2+); Roy-3a opened cycle C (post-0.9.1-wires Wire-A annotation pattern) | n/a |
| 7 | Roy-3a → Roy-3b | C: Wire-A annotation | **Divergence (mixed)** | Roy-3a surfaced THREE findings: expected null annotation render (predicted) PLUS two unpredicted failure modes — priming-side cluster_reward_bias regressed (6 saturated → 2 partial entries) AND Roy-2's clean A-vs-C tool-family divergence weakened. Roy-3b reproduced both unpredicted findings AND added a new one: valence_KS divergence collapsed from p=0.023 (Roy-2) to p=0.998 — the wires *weakened* the previously-cleanest cross-arm signal | **YES — Roy-3b's verdict explicitly recommended the bisect** |
| ★ | Roy-3b → Roy-3c-bisect | (bird's-eye, not a Roy iteration in the narrowing sense) | Bird's-eye response | Bisected 5/13→5/22 PR window. Two outside causes found, NOT the wires themselves: (1) non-code encoder drift on Mac in 5/13→5/14 window (refuted: MAXIM_SUBSTRATE_PATH env var change; revised suspect: LLM narrator drift), (2) Wire-A's intentional `cluster_reward_bias_decay_tau=50` decay (bio-fidelity correction, not regression). Reframed three downstream decisions (affordance replay, render floor, re-validation) | YES (this row IS the bisect) |
| 8 | Roy-3c-bisect → Roy-3a-retry | C: Wire-A annotation (mechanism-seam — see below) | **Ambiguous under Principle 4 retroactive reading** — see "Classification ambiguity" section. Strict scope reading: **convergence + NULL** (calibration math validated, primary criterion failed for reasons outside the iteration's pre-registered scope). Broader cycle-question reading: **convergence on tau + divergence on downstream gaps** (registry gap, imagination substrate-blindness) | Convergence: Phase 1 tau-split (tau=300) merged via PR #267; Roy-3a-retry validated calibration math (annotation rendered `[strongly rewarding]` throughout, max\|bias\| 0.753-0.997 — exactly as Phase 1 predicted). The registry-gap + imagination-blindness findings are labeled "Architectural findings surfaced by post-result investigation" in [30_wire_a_tau_validation.md](30_wire_a_tau_validation.md) — they did not emerge from the iteration's planned measurements. Spawned [sense_tool_registry.md](../plans/deferred/sense_tool_registry.md) + [imagination_substrate_signals.md](../plans/deferred/imagination_substrate_signals.md) | n/a (single iteration post-bisect; classification ambiguity rules out a clean trigger reading) |

## Convergence streaks

- **Cycle A (cluster wire): 5 convergent pairs in a row.** Roy-0 → Roy-1a → Roy-1b → Roy-2 → Roy-2pc → Roy-2c. Each iteration's recommended next step was the predecessor's recommendation, and each narrowed the structural-vs-behavioral gap toward the H1a verdict (encoder alignment failure). The cluster wire output reproduced within 1% across all 5 iterations. Healthy convergence; cycle closed cleanly with a falsified-hypothesis verdict.
- **Cycle B (encoder structure): 1 convergent pair.** Roy-4 → Roy-5a. Roy-4's FAIL across the Hebbian binding parameter sweep prompted Roy-5a's cosine-localization disambiguator, which narrowed from "cluster disjointness" to the specific shape (different-dimensional subspaces, food strictly interoception during priming). Cycle closed with the H1a verdict (encoder replacement = 1.2+ research direction).

## Divergence streaks

- **Cycle C (Wire-A annotation): 1 divergent pair (Roy-3a → Roy-3b).** Both iterations surfaced unpredicted failure modes vs their predecessor. The bisect (Roy-3c-bisect) was triggered AFTER Roy-3b — i.e., after the second iteration where new failure modes appeared. This matches the Principle 4 "two divergence iterations in a row" trigger pattern.

  - Roy-3a (vs the cycle-A baseline, which was the comparison surface since cycle C started here): null annotation render (predicted), priming-side regression (NEW), weakened A-vs-C divergence (NEW).
  - Roy-3b (vs Roy-3a): same priming-side regression (reproduced — convergent on that finding), same null render (reproduced), BUT valence_KS regression p=0.023→p=0.998 (NEW failure mode).

  By the time Roy-3b's verdict was written, the cycle had accumulated four new failure modes across two iterations: priming-side cluster_reward_bias drop, weakened A-vs-C tool divergence, weakened valence_KS, null Wire-A render on engineered overlap fixture. The bisect was the correct response.

## Missed bird's-eye triggers

**None found.** The only divergence streak in the Roy history (Roy-3a → Roy-3b in cycle C) correctly triggered Roy-3c-bisect.

The two transition pairs (Roy-2c → Roy-4 and Roy-5a → Roy-3a) are not divergence-within-a-cycle and would not trigger bisects under Principle 4's frame.

The five convergent pairs in cycle A and the one convergent pair in cycle B are healthy narrowing — no bisect would be appropriate.

## Classification ambiguity: Roy-3a-retry's two readings

Principle 4 is new (introduced 2026-05-26). Applying it retroactively to Roy-3a-retry surfaces a genuine ambiguity in the "new failure modes" criterion that the two pre-merge review lenses landed on opposite sides of:

**Strict scope reading** (Roy-method lens): Roy-3a-retry was pre-registered narrowly as a tau-magnitude calibration test. PRIMARY: Arm A ≥1 `sense_food_source`. STRETCH: cross-arm divergence. The registry-gap and imagination-blindness findings are explicitly labeled **"Architectural findings surfaced by post-result investigation"** in [30_wire_a_tau_validation.md § "Architectural findings"](30_wire_a_tau_validation.md). They did not emerge from the iteration's planned measurements. Under strict Principle 4 reading: Roy-3a-retry is a **convergence on tau magnitude + a NULL on the primary criterion**, with post-hoc observations recorded but not in-cycle divergence signals.

**Broader cycle-question reading** (convergence/divergence lens): Cycle C's central question is "does substrate-annotates-LLM-context drive behavioral expression?" Any failure mode that blocks annotation→behavior is within cycle C's mechanism question, even if surfaced post-result. Under this reading: Roy-3a-retry is **convergence on tau + divergence on downstream gaps**, and cycle C has accumulated five failure modes across three iterations.

**The audit does not resolve which reading is correct.** Both have legitimate textual support: the iteration's narrow pre-registration vs the cycle's broader animating question. Principle 4's "new failure modes each iteration" criterion is silent on whether post-hoc observations count.

**Practical implication for the trigger.** Under strict reading, cycle C has 1 divergence pair (Roy-3a → Roy-3b, correctly bisected) and is currently in a post-bisect convergent-NULL state; the two-in-a-row counter resets after the bisect's tau-split fix lands. Under broader reading, cycle C has 1 divergence pair pre-bisect + 1 divergence-flavored post-bisect iteration, and one more new failure mode would arguably fire the trigger.

## Mechanism-cycle seam inside cycle C

Both lenses independently flagged a related concern: **Roy-3a/3b and Roy-3a-retry may be testing different mechanisms within the broader Wire-A framing**:

- **Roy-3a + Roy-3b** test *Wire-A as authored at PR #257* (tau=50). The annotation render is near-null on both fixtures.
- **Roy-3a-retry** tests *Wire-A with the Phase 1 tau-split applied via PR #267* (tau=300). The annotation render is strongly-rewarding throughout.

Under a stricter cycle definition ("Wire-A as authored" vs "Wire-A with tau-split"), Roy-3a-retry opens a NEW cycle and the Principle 4 two-in-a-row counter resets after the bisect. Under the audit's broader cycle definition ("the Wire-A annotation pattern in any form"), it's the next iteration of the same cycle. The audit's grouping is defensible but elides this seam — and the elision matters for the trigger count.

## Implications — what to watch, with caveats

The two lenses disagree on whether cycle C is currently in active divergence territory. But they agree on:

- **Cycle C has accumulated more failure modes than cycle A's full five-pair convergent streak.** Whether they count as in-cycle divergence under Principle 4 is the ambiguity above, but the *accumulation* is real:

  | Iteration | Failure modes surfaced (in-scope OR post-hoc) |
  |---|---|
  | Roy-3a | (1) priming-side regression, (2) weakened A-vs-C divergence |
  | Roy-3b | (3) valence_KS regression collapse (p=0.023 → 0.998) |
  | Roy-3a-retry | (4) sense-tool registry gap, (5) imagination substrate-blindness (both post-hoc per the experiment doc) |

- Roy-3c-bisect closed (1) (revised suspect: narrator drift, not the wires; partial closure as bio-fidelity reframe). (2), (3), (4), (5) remain unaddressed at the mechanism level.

- **The strategic question the audit cannot answer alone:** are (2)–(5) consequences of Wire-A's signal-shaping decisions interacting with the LLM proposer in ways the design did not anticipate, or are they downstream-system gaps that any substrate-annotation mechanism would surface? This is the question a future bird's-eye bisect would scope, IF the trigger fires.

**Strategic seam between cycle B and cycle C (flagged by convergence lens, not resolved):** Cycle B's verdict ("encoder fundamentally wrong-dimension, replacement = 1.2+ research direction") may put cycle C in a structurally hopeless position from inception — Wire-A annotates a substrate whose recall path Roy-2c/Roy-4 already proved disjoint. The audit treats Roy-5a → Roy-3a as a "transition, not classified," which is the correct Principle 4 reading (different mechanisms), but it hides the strategic question of whether cycle C should have been launched at all given cycle B's verdict. This is NOT a missed Principle-4 trigger — Principle 4 is per-cycle — but it IS a meta-question worth surfacing.

**Concrete escalation candidate for user decision (not a recommendation, surfacing per kickoff):**

- **The next Wire-A iteration is the watch point under both readings.** Under strict reading, it would need to surface *new in-scope failure modes* twice in a row (after Roy-3a-retry's NULL) to fire the trigger. Under broader reading, one more divergence-flavored iteration after Roy-3a-retry fires the trigger.
- **The bird's-eye question, if/when triggered**, would be scoped to: *what's the actual independent variable shaping arm A's behavior?* Candidate axes — Wire-A annotation signal strength (now saturated at strongly rewarding), scene-tool availability, imagination wiring, LLM proposer's reading of the annotation, the deeper cycle-B-verdict question of whether the substrate Wire-A annotates is dimensionally compatible with the LLM's expected signal shape.
- **NOT authorization to spawn a bisect now.** It's the surface the user should keep an eye on. The classification ambiguity is itself a signal that Principle 4's retroactive application has limits the user may want to refine.

## Lens cross-check (pre-merge review)

Two parallel review lenses ran on the audit. Their independent verdicts converged on the cycle A + cycle B classifications and on the Roy-3a → Roy-3b divergence + correct bisect trigger, but **diverged** on the Roy-3a-retry classification and on the "cycle C in active divergence territory" framing. The divergence between the two lenses is itself signal: Principle 4's retroactive application has classification ambiguity the audit now surfaces in the "Classification ambiguity" + "Mechanism-cycle seam" sections above.

**Convergence/divergence lens (independent spot-checks).** Three pairs spot-checked by re-reading the relevant experiment docs:

- *Pair 4 (Roy-2 → Roy-2pc):* Roy-2's verdict recommended the positive-control fixture; Roy-2pc reported the pre-registered A ≈ B ≈ C outcome. Same mechanism narrowed by ruling out engineered semantic overlap. **Convergence confirmed.**
- *Pair 6 (Roy-4 → Roy-5a):* Roy-4 recommended encoder-replacement direction; Roy-5a delivered the cosine-localization disambiguator. **Convergence confirmed.**
- *Pair 7 (Roy-3a → Roy-3b):* Both surfaced new failure modes vs predecessor; Roy-3b's verdict explicitly recommended the bisect. **Divergence confirmed.**

The convergence lens additionally raised the cycle B → cycle C strategic seam question (now surfaced in implications) and flagged that cycle C's failure-mode accumulation already exceeds cycle A's full convergent streak length.

**Roy-method lens (independent scope-fairness check).** Re-stated each iteration's pre-registered scope and tested whether the audit's classifications hold under a strict "in-scope only counts" reading:

- Cycle A iterations all stayed within their pre-registered single-variable scopes. **Classifications hold.**
- Cycle B iterations stayed in scope. **Classifications hold.**
- Roy-3a's pre-registered NULL branch explicitly named "prompt rendering / priming-side regressions" as the investigative direction. The unpredicted findings ARE within that named branch. **In-scope divergence; classification holds.**
- Roy-3b's valence_KS finding is borderline — Roy-3b's narrow pre-registration was the `sense_food_source` count, with valence_KS being one of the standard pairwise divergence panel metrics. **In-scope-but-borderline; classification holds with a caveat.**
- **Roy-3a-retry's registry-gap + imagination-blindness findings: OUT-OF-SCOPE under strict reading.** The experiment doc itself labels them "Architectural findings surfaced by post-result investigation." Roy-3a-retry's pre-registration was tau-magnitude calibration only. Under strict scope reading, Roy-3a-retry is a convergence on the bisect's tau hypothesis + a NULL on the primary criterion, not an in-cycle divergence iteration.

**The two lenses disagree on Roy-3a-retry.** The audit folds this disagreement into the "Classification ambiguity" section rather than picking one reading. Both readings have legitimate textual support; the trigger-count implication differs between them.

**Other fold-worthy findings folded:**
- **Mechanism-cycle seam inside cycle C.** Both lenses flagged that Roy-3a/3b (testing Wire-A as authored, tau=50) and Roy-3a-retry (testing Wire-A with Phase 1 tau-split, tau=300) may be testing different mechanisms within the broader Wire-A framing. The audit treats them as one cycle with a noted seam.
- **Strategic seam between cycle B and cycle C** (convergence lens). Cycle B's "encoder fundamentally wrong-dimension" verdict may have put cycle C in a structurally hopeless position from inception. Not a missed Principle-4 trigger (Principle 4 is per-cycle), but a meta-question worth surfacing.
- **Audit's own lens-cross-check was glossing the scope question** (Roy-method lens). The fold makes the scope handling explicit rather than rhetorical.

No classifications other than Roy-3a-retry's required revision; that one is now presented as ambiguous rather than forced to a single label.

## Summary

| Cycle | Iterations | Convergent pairs | Divergent pairs | Bird's-eye triggered? |
|---|---|---|---|---|
| A: cluster wire | Roy-0 → Roy-2c | 5 | 0 | n/a (cycle closed cleanly with H1a verdict) |
| B: encoder structure | Roy-4 → Roy-5a | 1 | 0 | n/a (cycle closed cleanly with H1a-confirm) |
| C: Wire-A annotation | Roy-3a → Roy-3a-retry | 0 (strict) or 0+1 mixed (broader) | 1 (strict) or 1+1 mixed (broader) | YES — Roy-3c-bisect (correct trigger after Roy-3b) |

**No missed bird's-eye triggers** under either reading.

**Whether cycle C is currently in active divergence territory depends on whether Roy-3a-retry's post-hoc findings count as in-cycle divergence signals** — the two pre-merge lenses disagree, and the audit declines to force one reading. The next Wire-A iteration is a watch point under both readings, with different trigger thresholds.

**Cycle C HAS accumulated more named failure modes (5) than cycle A's full convergent streak (5 pairs).** That accumulation is real regardless of how Principle 4 classifies the individual iteration pairs.

## Cross-references

- [CLAUDE.md § Working principles for new mechanisms](../../CLAUDE.md#working-principles-for-new-mechanisms) — Principle 4 definition.
- [persona_convergence_crucible.md § Iteration log](../plans/deferred/persona_convergence_crucible.md) — source-of-truth for each Roy iteration's pre-registration, result, verdict.
- [29_roy_3c_bisect.md](29_roy_3c_bisect.md) — the canonical bird's-eye response.
- [30_wire_a_tau_validation.md](30_wire_a_tau_validation.md) — Roy-3a-retry results that surfaced findings (4) and (5).
- [sense_tool_registry.md](../plans/deferred/sense_tool_registry.md) + [imagination_substrate_signals.md](../plans/deferred/imagination_substrate_signals.md) — downstream-gap follow-up plans currently sitting on the cycle C divergence ledger.

## Retroactive correction (2026-05-27, [32_wire_a_post_w1_w2.md](32_wire_a_post_w1_w2.md))

Row 8 of the iteration-pair table cites Roy-3a-retry as "annotation rendered `[strongly rewarding]` throughout" — that sub-claim is invalidated by exp 32's Bug A discovery. The actual rendering was empty for the full 0.9.1 window because of an upstream agent_id mismatch between priming and test-arm AUTs. **The classification of row 8 itself is not changed by this correction** — the row labels "calibration math validated, primary criterion failed for reasons outside the iteration's pre-registered scope" remain valid (the decay trajectory is agent_id-agnostic and confirmed the tau-split math regardless of whether the annotation reached the LLM). The "convergence + NULL" verdict stands; the supporting evidence shape changes. Future audits using this row should read it as "calibration math validated; LLM-side annotation reception unverified (instrument bug)." Cycle C's overall convergence/divergence reading is unaffected — the cycle's behavioral measurements were always behavior on the LLM's part, not on the annotation pipeline's part.
