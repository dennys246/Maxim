# Roy-5a — Cosine-localization disambiguator (1.1+ plan Stage 1)

**Date:** 2026-05-14 (run completed 2026-05-13 22:51 local)
**Plan:** [roy_5_encoder_alignment_disambiguator.md § Stage 1](../plans/archive/roy_5_encoder_alignment_disambiguator.md) · [persona_convergence_crucible.md § "Iteration log"](../plans/deferred/persona_convergence_crucible.md)
**Companion:** [21_roy_4.md](21_roy_4.md) (Roy-4 FAIL that cancelled cross-modal binding plan and motivated Roy-5) · [20_roy_2c.md](20_roy_2c.md) (H1 confirmation that lives inside Roy-5's three sub-hypotheses)
**Spec:** [scenarios/roy/roy_5a_iteration.yaml](../../scenarios/roy/roy_5a_iteration.yaml)
**Engineered fixture:** [scenarios/roy/roy_2pc_holdout.yaml](../../scenarios/roy/roy_2pc_holdout.yaml) (reused unchanged from Roy-2pc / Roy-2c / Roy-4)
**Reproduction:** [protocols/22_roy_5a_reproduction.md](protocols/22_roy_5a_reproduction.md)
**Analyzer:** [scripts/analyze_roy_5_cosine_localization.py](../../scripts/analyze_roy_5_cosine_localization.py)
**Persistence prerequisite:** [PR #248](https://github.com/dennys246/Maxim/pull/248) (wires `EC.save()` + `ATL.save()` into `simulation/report.py::save_aut_state`)

## Status

**VERDICT: H1a — encoder subspace incompatibility (confirmed across three runs, two distinct mechanisms).**

`max(M_tt food-bearing) = n/a` (no text-modality food-bearing priming centroids exist regardless of `MAXIM_SUBSTRATE_PATH` state). Pre-registered decoder maps `-inf` → `< 0.20` → **H1a**. Three runs confirm the verdict:

| Run | `MAXIM_SUBSTRATE_PATH` | Priming text/intero fires | Food-bearing text centroids | Mechanism |
|---|---|---|---|---|
| **Roy-5a** (initial) | unset | 0 / 151 | 0 (no text nodes at all) | H1a via "no text nodes exist" |
| **Roy-5a-confirm** | unset | 0 / 153 | 0 (same) | H1a — silence reproduces, not run-to-run variance |
| **Roy-5a-substrate-on** | **=1** | **162 / 154** | **0 (zero of 14 text centroids are food-bearing)** | **H1a via "food NAc bias keys are exclusively interoception node IDs"** |

Roy-4's 154 text fires on the identical spec turned out to be the result of `MAXIM_SUBSTRATE_PATH=1` being inherited from the shell environment, not a code regression. Roy-5a + Roy-5a-confirm ran without it; Roy-5a-substrate-on explicitly sets it and reproduces Roy-4's text-modality fire count (162 vs 154 — within run-to-run variance). **The fundamental finding holds across the env-var state:** food NAc-attributed cluster IDs are interoception-modality EC node IDs, never text-modality.

**Stronger structural finding — dimension mismatch (uncovered by the analyzer's dimension-mismatch warning, added in the pre-merge review fold):** `SensorEncoder` produces **384-dim** SHA-basis embeddings for interoception modality; `LinguisticEncoder` (`paraphrase-mpnet-base-v2`) produces **768-dim** embeddings for text modality. Cross-modality cosine (`M_dt`) is **mathematically undefined** — the vectors live in different dimensional spaces, and `_cosine` returns 0.0 silently on length mismatch. This means the plan's "encoder subspaces are far in cosine space" framing of H1a is **structurally weaker than the data actually shows**: the subspaces aren't far, they're **different-dimensional**, and any cosine-based cross-modal alignment is structurally impossible without a learned projection layer between them.

**Per [roy_5_encoder_alignment_disambiguator.md § Stage 2c](../plans/archive/roy_5_encoder_alignment_disambiguator.md), this verdict triggers Stage 3** — redesigned cradle priming arc with deliberate `(sensor, drive, narrator-utterance)` co-firing + Hebbian retest. **The Stage 3 design constraint sharpens with the dimension-mismatch finding:** the goal isn't "make text and interoception embeddings line up in cosine space" (they can't — different dimensions). It's "make the food concept LAND in text-modality EC nodes during priming, attributed to `sense_food_source` reward in NAc". The narrator utterance ("hungry") would need to fire as a text-modality EC node that the NAc reward bias keys to. **The Hebbian binding mechanism cancelled by Roy-4 was always structurally impossible at the cosine level** — `cross_modal_substrate_binding.md`'s resurrection conditions (Stage 4a) would need a learned dimension-reducing projection between encoders, not a naive Hebbian rule on raw cosine.

**Recommended next step for Stage 3:** explicitly design the redesigned cradle arc so the narrator utterance fires text-modality EC during the same tick the `sense_food_source` reward is attributed in NAc. The pass criterion is then **"at least one text-modality EC node ends up in NAc's `cluster_reward_bias` map keyed to `sense_food_source`"** — directly observable in the persisted `aut_nac.json`, no cosine math required. If that produces a text-modality food centroid, M_tt becomes non-empty in a Roy-5b re-run and the H1c / H1b discrimination becomes meaningful for the FIRST TIME across all Roy iterations.

## Pre-registered diagnostic logic

| `max(M_tt food-bearing)` | Verdict | Stage 2 next step |
|---|---|---|
| `≥ 0.40` | **H1c** — threshold/centroid tuning | 2a env-var sweep (`MAXIM_EC_PATTERN_THRESHOLD_TEXT` / `MAXIM_EC_FROZEN_TEXT`) |
| `0.20 ≤ max < 0.40` | **H1b** — encoder model fit | 2b encoder A/B (kill `LinguisticEncoder._get_encoder` singleton, sweep alternative sentence-transformer models) |
| **`< 0.20` (incl. n/a)** | **H1a — encoder subspace incompatibility** | **2c → Stage 3 cradle-arc redesign + Hebbian retest** |

The 0.40 boundary is the current `ECConfig.pattern_complete_threshold` (P1-tuned, paraphrase-mpnet@0.40); the 0.20 lower band is the conservative floor below which "the encoder doesn't even see these as related" is the most parsimonious explanation. Both bounds are pinned by [`tests/unit/test_roy_5_cosine_localization.py::TestVerdictDecoding`](../../tests/unit/test_roy_5_cosine_localization.py).

## What shipped

- [`scenarios/roy/roy_5a_iteration.yaml`](../../scenarios/roy/roy_5a_iteration.yaml) — re-run of the Roy-2c / Roy-4 priming + fixture + arms structure with `MAXIM_EC_TRACE_ACTIVATIONS=1` in the runner environment.
- [PR #248](https://github.com/dennys246/Maxim/pull/248) (merged before this run) — `EC.save()` and `ATL.save()` now wired into `simulation/report.py::save_aut_state`. Every session_dir under `~/.maxim/sim_reports/<sid>/` now contains `aut_ec.json` and `aut_atl.json` alongside the existing hippocampus + NAc dumps. The resume-from-session load path picks both up symmetrically.
- [`scripts/analyze_roy_5_cosine_localization.py`](../../scripts/analyze_roy_5_cosine_localization.py) — post-hoc analyzer that reads persisted EC centroids from priming + each arm session, computes `M_tt` / `M_dt` / `M_dd` pairwise cosine matrices, identifies food-bearing priming centroids via the same UTS-separator NAc compound-key parsing the Roy-4 analyzer uses, and decodes max cosine over arm A's matrices into the pre-registered verdict.
- 24 unit tests in [`tests/unit/test_roy_5_cosine_localization.py`](../../tests/unit/test_roy_5_cosine_localization.py) covering boundary semantics (0.19 / 0.20 / 0.39 / 0.40 / 0.41 / `-inf`), cross-modal `M_dt` non-emptiness regression guard, UTS-separator food-cluster extraction, graceful zero-centroid handling, and end-to-end synthetic-fixture decoding of all three sub-hypotheses.
- 4 unit tests in [`tests/unit/test_save_aut_state.py`](../../tests/unit/test_save_aut_state.py) (shipped via PR #248) covering the persistence-wiring extension.

## Result

Wall: **~1547s (~25.8 min)** — same shape as Roy-2c / Roy-4. Pre-flight clean (`outcome: ok`, `latency_ms: 211.2`). Priming completed all 5 stages × 10 turns; arm a/b/c each completed 10 turns.

### Per-arm

| Arm | Substrate | system_prompt | session_id | turns | duration_s | finish |
|---|---|---|---|---|---|---|
| a | from_priming | neutral | `20260513_223616` | 10 | (cancel) | cancel |
| b | blank | "You are a hungry infant" | `20260513_224138` | 10 | (cancel) | cancel |
| c | blank | neutral | `20260513_224641` | 10 | (cancel) | cancel |

Priming final session: `20260513_223436` (act3_consolidation, the session arm A resumed from).

### Headline cosine matrices — arm A

| Matrix | Modality pair | Rows × Cols | Max over food-bearing rows |
|---|---|---|---|
| **`M_tt`** | priming text × arm A text | **0 × 0** | **n/a (no text-modality nodes on either side)** |
| **`M_dt`** | priming interoception × arm A text | 2 × 0 | n/a (arm A has zero text-modality nodes) |
| **`M_dd`** | priming interoception × arm A interoception | 2 × 2 | **1.0000 (identical centroids)** |

**Food-bearing priming clusters** identified from `aut_nac.json` compound keys: 2 cluster IDs (`4660f22e-...` and `aebdb0c3-...`). **Both are in interoception modality**, neither in text modality. The food concept lives entirely in the interoception-encoded sensor channel; no text-modality representation of "food" exists in priming.

### Cross-arm sanity check — `M_dd` headline

Same cosine matrix computed against arms B and C confirms the "interoception identity survives at sensor level":

| Arm | `M_dd` max food cosine | `M_dd` rows × cols | Cluster IDs match priming? |
|---|---|---|---|
| **a** (substrate-primed) | **1.0000** | 2 × 2 | Yes (same UUIDs — substrate restored) |
| **b** (blank, persona) | 1.0000 | 2 × 2 | No (fresh UUIDs, but cosine ≈ 1.0 — same sensor pattern, frozen-prototype embeddings collide) |
| **c** (blank, neutral) | 1.0000 | 2 × 2 | No (fresh UUIDs, embedding match) |

This is the **"two identity schemes for the same concept"** pattern from [feedback_two_identity_schemes.md](../../.claude/projects/-Users-dennyschaedig-Scripts-Maxim/memory/feedback_two_identity_schemes.md): SensorEncoder produces deterministic SHA-basis hash embeddings, so identical sensor states across sessions produce identical interoception centroids — but each session allocates a **fresh UUID** for the "new node" creation path. Arms B/C share the embedding identity but not the cluster-ID identity that NAc keys off, which is why `cluster_reward_bias_l2` shows non-zero divergence even though M_dd shows 1.0 cosine.

### EC trace event capture (secondary, for cross-check)

Roy-5a ran with `MAXIM_EC_TRACE_ACTIVATIONS=1` so the JSONL trace events landed at `/tmp/roy_5a_ec_trace.jsonl`. The trace tells a different story from Roy-4's:

| Iteration | Total EC trace events | Per-modality fires | Per-modality NEW |
|---|---|---|---|
| Roy-4 (priming) | 306 | text=154, interoception=152 | text=**57**, interoception=12 |
| **Roy-5a (priming)** | **151** | **interoception=151, text=0** | **interoception=6, text=0** |

**Roy-4 had 57 NEW text-modality nodes fire during priming; Roy-5a had zero. Cause located after two follow-up runs**: `MAXIM_SUBSTRATE_PATH=1` was inherited from the shell environment when Roy-4 ran; Roy-5a + Roy-5a-confirm were kicked off in a shell without it set. The env var gates `MemoryHub._encoder` wiring at [`integration/memory_hub.py:247`](../../src/maxim/integration/memory_hub.py#L247); `BioEnrichmentPipeline` receives `encoder=getattr(memory_hub, "_encoder", None)` at [`runtime/bio_stack.py:378`](../../src/maxim/runtime/bio_stack.py#L378) and its text-modality EC fire at [`bio_enrichment.py:576`](../../src/maxim/integration/bio_enrichment.py#L576) short-circuits on `if self._encoder is not None`. With the env var off, every BioEnrichmentPipeline text query silently skips EC; with it on, queries fire `pattern_complete_or_separate(embedding, "text")` and produce trace events.

Roy-5a-substrate-on confirmed this end-to-end: explicit `MAXIM_SUBSTRATE_PATH=1` reproduces Roy-4's text-modality fire pattern (162 fires vs Roy-4's 154 — within run-to-run variance). **No code regression.** The 0.9.1 release plan explicitly ships substrate-path features behind this env var per [`release_0_9_1.md`](../plans/archive/release_0_9_1.md) Stage 2; Wire-A is what flips the default once it ships.

**Roy-4's hippocampus dumps had zero `cli_input` / `transcript` fields too — same as Roy-5a-*.** Roy-4's 154 text fires came through `BioEnrichmentPipeline`'s query path during sim turns, not from `transcript_chunk` direct routing. This explains why neither Roy-4 nor any Roy-5a variant shows text in the post-hoc hippocampus perception fields.

**The H1a verdict survives every variant:**

- **Substrate path OFF (Roy-5a / Roy-5a-confirm):** zero text-modality EC nodes exist at all → M_tt trivially empty for food rows.
- **Substrate path ON (Roy-5a-substrate-on):** 14 text-modality EC centroids exist in priming, but **zero of them are food-bearing**. NAc's `cluster_reward_bias` map keys 2 cluster IDs to `sense_food_source`; both are interoception-modality node IDs (`8986a04c-...`, `27c6f321-...`), neither in the 14-node text-modality set.
- **All three runs:** `M_dd ≈ 1.0` between priming and arm-A interoception food clusters (the surviving frozen-prototype identity scheme — same SHA-basis embedding regardless of run).

**Dimension mismatch — additional structural finding:**

The dimension-mismatch warning the pre-merge review fold added to `_compute_matrix` fired on real Roy-5a-substrate-on data with `row dims=[384] col dims=[768]`. Concretely:

| Modality | Encoder | Embedding dim | Source |
|---|---|---|---|
| `interoception` | `SensorEncoder._stable_basis` | **384** | SHA-derived basis vectors (deterministic per sensor name) |
| `text` | `LinguisticEncoder` (`paraphrase-mpnet-base-v2`) | **768** | sentence-transformers |

Cross-modality cosine (M_dt: priming interoception × arm A text) is **mathematically undefined** — `_cosine` returns 0.0 silently on length mismatch (`if len(u) != len(v) or not u: return 0.0` per [`scripts/analyze_roy_5_cosine_localization.py:104`](../../scripts/analyze_roy_5_cosine_localization.py#L104)). The 0.0000 M_dt value in the per-arm summary is **not** "the encoders see them as orthogonal" — it's "comparison undefined". The analyzer's warning surfaces this; the JSON bundle's M_dt 0.0 should be read as "undefined" not "computed".

**This finding is stronger than the plan's H1a framing.** The plan modeled H1a as "encoder subspaces are far in cosine space (max < 0.20)". The actual finding is "encoder subspaces are different-dimensional, cosine is undefined". Any cosine-based cross-modal alignment is structurally impossible without a learned projection layer between the encoders. The Hebbian binding mechanism cancelled by Roy-4 (`cross_modal_substrate_binding.md` Stages 2-6) was always structurally impossible at the cosine level — Stage 4a's resurrection conditions would need a dimension-reducing learned projection, not a Hebbian rule on raw cosine.

## What this means for Stage 2

**The pre-registered verdict triggers Stage 2c → Stage 3** (redesigned cradle priming arc with deliberate co-firing). Per [roy_5_encoder_alignment_disambiguator.md § Stage 3](../plans/archive/roy_5_encoder_alignment_disambiguator.md), Stage 3 ships:

- `_data/components/bodies/infant_humanoid_naming_v1.yaml` — new body with co-firing scaffold (additive, doesn't replace existing arc).
- `prompts/cradle_narrator.py` — narrator pattern that fires "hungry" / "thirsty" / "warm" utterances co-tick with the matching drive/sensor threshold.
- `scenarios/roy/roy_5b_iteration.yaml` — same shape as Roy-4 / Roy-5a but uses the redesigned arc.

**Sharpened Stage 3 pass criterion (refined by Roy-5a-substrate-on's findings):** the original plan said "produce co-firing + Hebbian binding succeeds". The data shows that's neither sufficient nor — at the cosine level — possible. The **directly observable + structurally meaningful** pass criterion is:

> **At least one text-modality EC node appears in NAc's `cluster_reward_bias` map keyed to `sense_food_source` after Stage 3 priming.**

This is observable directly in the persisted `aut_nac.json` + `aut_ec.json` — no cosine math required. If Stage 3's narrator-utterance scaffold produces a text-modality food centroid, M_tt becomes non-empty in a Roy-5b re-run and the H1c / H1b discrimination becomes meaningful for the **first time across all Roy iterations**. If Stage 3 still produces zero text-modality food centroids, then the Hebbian binding mechanism is dead even at the structural level and Stage 4b (encoder replacement to 1.2+) is triggered.

The plan's original two-part criterion (text fires + cross-modality binding) collapses into the single observable above: the text-modality food fire IS the prerequisite, and the cross-modality binding is structurally impossible at cosine level so doesn't gate further.

## Recommended next step

The disambiguation Stage 1 was meant to produce is now clean. Three actions for Stage 2 routing:

1. **Always set `MAXIM_SUBSTRATE_PATH=1` in any future Roy reproduction protocol** so the text-modality routing is active and the analyzer's matrices populate meaningfully. The Roy-5a reproduction protocol has been updated to reflect this.
2. **Stage 3 (cradle-arc redesign) is greenlit** per the H1a verdict, with the sharpened pass criterion above. The dimension-mismatch finding means Stage 4a's resurrection of `cross_modal_substrate_binding.md` should NOT proceed even if Stage 3 produces text-modality food centroids — the Hebbian binding rule on cross-modality cosine is structurally impossible regardless. Stage 3 PASS → ship the substrate-annotates-LLM-context path (Wire-A in 0.9.1) as the operator-visible answer; Stage 4a stays cancelled.
3. **Consider whether the dimension-mismatch finding requires a follow-up plan** to either (a) align SensorEncoder + LinguisticEncoder on a common embedding dim (additive change to SensorEncoder's hash basis projection), or (b) declare cross-modality cosine alignment out of scope for 1.0 and document the learned-projection requirement as 1.2+ research. This is a strategic 1.0 / 1.1+ scoping question, not a Stage 2 implementation step.

## Why this verdict is more confident than Roy-4's FAIL

Roy-4's FAIL was robust across a sweep of `(min_cofire, min_weight)` parameters. Roy-5a's H1a verdict has a similar robustness property: the verdict triggers on **the absence of text-modality food clusters**, which is **observable directly** in the persisted EC dump (`aut_ec.json`) without dependence on any analyzer hyperparameter. The pre-registered threshold boundaries (`0.20` / `0.40`) only matter if `max(M_tt)` is computable; here it isn't, and the analyzer's `-inf` handling — pinned by `test_negative_infinity_decodes_h1a` — flows directly to H1a.

The **`M_dd` cosine ≈ 1.0** result is the strongest positive signal in this iteration: priming food clusters DO land in arm A's interoception substrate, perfectly. Whatever future Stage 3 / 4a / 4b implementation ships, this confirms the food concept's interoception embedding survives arm A's substrate-restoration — the gap is strictly in projecting it to text modality (which is the substrate channel CLI fixture text routes through).

**Important caveat on `M_dd ≈ 1.0`:** this is a **baseline** (a frozen-prototype-determinism result), not a learning result. `SensorEncoder._stable_basis(name, dim, salt)` in [src/maxim/similarity/encoder.py](../../src/maxim/similarity/encoder.py) deterministically hashes the sensor name into a SHA-derived basis vector, so identical sensor states across sessions produce identical interoception centroids regardless of what the cradle arc taught. Arms B and C (blank substrate, no priming exposure) also show `M_dd ≈ 1.0` for the same reason — they encode the same sensor pattern at the same embedding. Future Stage 4a "scaffold rescues binding" interpretations should treat `M_dd ≈ 1.0` as a structural floor, not evidence the substrate acquired anything from priming; the **load-bearing learning result is `cluster_reward_bias_l2 ≈ 2.47 between arm A and arms B/C`** ([Roy iteration log § Roy-5a "Pairwise substrate divergence"](../plans/deferred/persona_convergence_crucible.md)), which is the NAc-side cluster-key learning that does NOT survive without substrate restoration.

## References

- [docs/plans/archive/roy_5_encoder_alignment_disambiguator.md](../plans/archive/roy_5_encoder_alignment_disambiguator.md) — the plan Roy-5a is Stage 1 of.
- [docs/experiments/21_roy_4.md](21_roy_4.md) — Roy-4 FAIL that cancelled the binding plan and motivated Roy-5.
- [docs/experiments/20_roy_2c.md](20_roy_2c.md) — H1 confirmation; Roy-5a disambiguates which sub-hypothesis (H1a / H1b / H1c) holds.
- [feedback_two_identity_schemes.md](../../.claude/projects/-Users-dennyschaedig-Scripts-Maxim/memory/feedback_two_identity_schemes.md) — the cross-Roy pattern Roy-5a re-confirms (interoception centroids survive at the embedding level, cluster_ids do not).
- [feedback_interim_contamination.md](../../.claude/projects/-Users-dennyschaedig-Scripts-Maxim/memory/feedback_interim_contamination.md) — the trap Roy-5a's verdict protects against (Stage 2c → Stage 3 explicitly does NOT route through a hand-curated lexicon, even if the bio-faithful path takes longer).
- [feedback_review_before_ship.md](../../.claude/projects/-Users-dennyschaedig-Scripts-Maxim/memory/feedback_review_before_ship.md) — two-lens pre-merge review pattern; applied to the Roy-5a analyzer PR.
- [`/tmp/roy_5a_analysis.json`](/tmp/roy_5a_analysis.json) — full analyzer output bundle (M_tt / M_dt / M_dd matrices + verdict + per-arm headlines), regenerated by re-running the analyzer per the reproduction protocol.
