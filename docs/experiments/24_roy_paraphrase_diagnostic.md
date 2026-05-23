# 24 — Roy paraphrase-collapse diagnostic

**Date:** 2026-05-23
**Branch:** [`feat/0-9-1-roy-paraphrase-diagnostic`](https://github.com/dennys246/Maxim/tree/feat/0-9-1-roy-paraphrase-diagnostic)
**Companions:** [22_roy_5a.md](22_roy_5a.md) (H1 → H1a sharpening), [21_roy_4.md](21_roy_4.md) (Roy-4 FAIL, cross-modal binding cancellation), [p1_recognition_sweep.md](p1_recognition_sweep.md) (P1: 91.7% paraphrase collapse on designed fixtures)
**Script:** [scripts/diagnose_roy_paraphrase_collapse.py](../../scripts/diagnose_roy_paraphrase_collapse.py)
**Fixture:** [data/roy_paraphrase_pairs.json](../../data/roy_paraphrase_pairs.json)
**Raw output:** `/tmp/roy_paraphrase_collapse.json` (run locally; not persisted)

## Status

**VERDICT: CENTROID_DRIFT_COLLAPSE.**

Isolated per-pair clustering is **clean** — 10/10 pair collapse, 0/5 distractor collapse. The encoder + EC handle Roy fixture text correctly when each pair is tested against a fresh EC. **Sequential** encoding through one shared EC across the full pair + distractor surface (the Roy production regime) produces runaway centroid drift: after 20 text strings enter `pattern_complete_or_separate`, 18 of them have collapsed into a single mega-node. Three of five distractor pairs (60%) "collapse" only because their halves were both already swept into that mega-node from earlier pair-walk steps.

The encoder is fine. The pair / distractor design is fine. The failure mode is the **running-mean centroid update** in `EntorhinalCortex.pattern_complete_or_separate` for the `text` modality — successive low-but-above-threshold matches pull the centroid toward a generic "second-person body sensation" prototype that then pattern-completes everything sent to it.

## Pre-registered diagnostic logic

The task brief (2026-05-23) named three outcomes and a follow-up; this run added a fourth that the data made unavoidable:

| Condition | Verdict | Next step |
|---|---|---|
| Pair collapse ≥ 70% AND distractor collapse < 30% | **CROSS_MODAL_ONLY** | H1a is the only gap. JEPA / cross-modal binding direction stands. |
| Pair collapse < 70% AND ≥1 pair has pre-EC cosine ≥ threshold but disjoint EC nodes | **EC_THRESHOLD_TUNING** (H1c-shape) | Sweep `MAXIM_EC_PATTERN_THRESHOLD_TEXT` |
| Pair collapse < 70% OR distractor collapse ≥ 30% (both modes) | **SUBSTRATE_BROKEN** | Encoder / substrate write path rework before further persona work |
| **Isolated mode passes the gate but sequential mode fails it** | **CENTROID_DRIFT_COLLAPSE** ← **this run** | Fix the running-mean centroid drift in `pattern_complete_or_separate` |

The `CENTROID_DRIFT_COLLAPSE` branch was added to the verdict logic during the run because neither of the three pre-registered outcomes captured the actual failure mode (isolated pairs cluster perfectly, sequential pairs trivially collapse alongside every other string in the walk). The decision boundary stays pre-registered — the named verdict is the new label on a previously-unanticipated data shape, not a moved goalpost.

## Setup

- 10 hand-curated paraphrase pairs sampled across [`scenarios/roy/roy_1_holdout.yaml`](../../scenarios/roy/roy_1_holdout.yaml) (somatosensory: thermal, texture, intero, cold-shock, pressure, vibration, social) and [`scenarios/roy/roy_2pc_holdout.yaml`](../../scenarios/roy/roy_2pc_holdout.yaml) (engineered food/hunger overlap).
- 5 distractor pairs spanning cross-modality and cross-class semantic distance.
- Encoder: `paraphrase-mpnet-base-v2` (production model, same as P1+P2 sweeps).
- EC: default `ECConfig` — `pattern_complete_threshold = 0.40`, text modality NOT in `frozen_centroid_modalities`.
- Decomposition: **off** (whole-string encoding). Mirrors default Roy regime (`MAXIM_SUBSTRATE_PATH=1` set, `MAXIM_CONCEPT_DECOMPOSITION` not set).
- Two encoding regimes per pair: **sequential** (all 20 unique pair-strings + 2 unique distractor-strings encoded in one EC in first-seen order) and **isolated** (fresh EC per pair, A then B, that pair only).

## Result

### Sequential mode (production-realistic)

| Metric | Value | Gate |
|---|---:|---|
| Pair collapse rate | **10/10 = 100%** | ≥ 70% ✓ |
| Distractor collapse rate | **3/5 = 60%** | < 30% ✗ |
| EC substrate nodes after walk of 22 unique strings | **3** | — |
| Pairs with pre-EC cosine ≥ threshold but disjoint nodes | 0 | — |

The headline "100% pair collapse" is a centroid-drift artifact: of the 20 unique pair strings, 19 of them join node `292c157a` (the first string's node). The collapse is not "pairs found their concept" — it's "everything found the same one mega-concept."

Walk trace (selected — pair strings encoded first, distractor strings last):

```
step  is_new  sim   node       text
  0   True    0.000 292c157a   you sense food nearby.
  1   False   0.754 292c157a   the smell of food fills the air.
  ...
  8   False   0.462 292c157a   heat blooms across your fingertips.
  ...
 10   False   0.425 292c157a   an abrupt chill grips your shoulders.
 ...
 19   False   0.455 292c157a   someone close by asks if you follow.
 20   True    0.000 cc4ce203   two people are arguing in the next room.
 21   True    0.000 8727971b   the room grows quiet.
```

All 20 pair strings (food, thermal, texture, vibration, pressure, social-question) land in one node. Only the two distractor-unique strings ("two people arguing" and "the room grows quiet") survive as separate nodes — and they survive only because they happened to be encoded after the mega-node had absorbed all the pair strings.

### Isolated mode (per-pair, fresh EC)

| Metric | Value | Gate |
|---|---:|---|
| Pair collapse rate | **10/10 = 100%** | ≥ 70% ✓ |
| Distractor collapse rate | **0/5 = 0%** | < 30% ✓ |

In isolation, every pair pattern-completes correctly and every distractor pattern-separates correctly. The encoder + EC handle Roy fixture text exactly as P1 (91.7% paraphrase collapse) suggested they would.

### Pre-EC cosine distribution

| Pair class | Range | All ≥ EC threshold (0.40)? |
|---|---|---|
| Food overlap (2pc, 4 pairs) | 0.583 – 0.754 | ✓ |
| Somatosensory matching/novel (5 pairs) | 0.717 – 0.916 | ✓ |
| Social/unrelated (1 pair) | 0.606 | ✓ |

| Distractor class | Range | All < EC threshold (0.40)? |
|---|---|---|
| Cross-modality (2 distractors) | 0.067 – 0.194 | ✓ |
| Cross-class (3 distractors) | 0.105 – 0.297 | ✓ |

The embedding model itself separates pairs from distractors cleanly — pair cosines (≥0.58) and distractor cosines (≤0.30) do not overlap. The bug is downstream of the embedding.

## Mechanism

`EntorhinalCortex.pattern_complete_or_separate` updates the matched node's centroid as a running mean on every successful pattern completion ([`src/maxim/similarity/ec.py:374-387`](../../src/maxim/similarity/ec.py#L374-L387)). The `frozen_centroid_modalities` config only freezes `"interoception"`; the `"text"` modality drifts.

Walk-through of the observed dynamic:

1. `"you sense food nearby."` creates node A with centroid = emb_0.
2. `"the smell of food fills the air."` matches A at cos = 0.754. Centroid updates to mean(emb_0, emb_1).
3. Subsequent food strings match the (slightly drifted) centroid at decreasing similarity (0.682, 0.703, 0.494, 0.574, 0.624) and pull the centroid further toward a generic food-related average.
4. `"heat blooms across your fingertips."` arrives. cos against the now-eight-string centroid is **0.462 — above 0.40**. It joins A.
5. Once thermal joins, the centroid contains mass in both food-detect AND thermal-sensation directions. The decision boundary widens further.
6. Cold-shock (0.425), pressure (0.455), vibration (0.529), social-question (0.426) all enter at similarities only marginally above threshold. Each updates the centroid toward an even more generic "second-person sensory description" prototype.

The same pairs encoded in isolation against a fresh EC trivially separate from any other concept, because no contaminating centroid exists.

## Interpretation

### What this confirms

- The **encoder** (paraphrase-mpnet-base-v2) embeds Roy fixture text correctly: pair cosines cluster well above distractor cosines, with no overlap. The N=1 worry on H1a is dismissed for text-modality embedding quality.
- The **EC pattern-completion mechanism** at threshold 0.40 produces correct pairwise clustering decisions when the centroid is uncontaminated.

### What this changes

- The persona-convergence gap is **not purely cross-modal**. There is also a structural intra-modal failure that surfaces only under sequential streaming input — which is the production Roy regime. Any persona work that depends on text-modality EC nodes persisting as **distinct concept clusters across a session** is silently corrupted by centroid drift.
- The JEPA / cross-modal binding direction (project_jepa_plan_drafted, [roy_5 Stage 4](../plans/roy_5_encoder_alignment_disambiguator.md)) **still stands as a separate gap**, but it is no longer the only blocker on the persona-convergence path. A learned cross-modal projection on top of a substrate that has already collapsed text concepts into one mega-node would inherit the collapse.
- 1.0's V1 cross-session validation silently depends on this same machinery. The current text-modality centroid behavior would cause cross-session "recall" of a previously-encoded concept to drift toward "anything second-person-sensory" the more text the substrate has seen.

### What this does NOT yet show

- **Decomposed mode** (`MAXIM_CONCEPT_DECOMPOSITION=1`) was not tested. Production Roy iterations do not set it, but if shorter noun-chunked concepts (`"food"`, `"belly"`, `"hunger"`) resist drift better than whole-sentence embeddings, the production-with-decomposition regime might behave differently. Worth a follow-up.
- **Per-modality threshold tuning.** A higher `pattern_complete_threshold` for text (e.g., 0.60) would reject the marginal matches that cause drift but at the cost of pair recall. Sweep before declaring the running-mean update is the only fix.
- **Frozen-centroid for text.** Adding `"text"` to `frozen_centroid_modalities` would freeze the first embedding as the prototype and eliminate drift. Mirrors the existing fix for the `"interoception"` modality. The trade-off is whether the first text embedding is a strong-enough prototype for the concept — for the food cluster, "you sense food nearby" is plausibly a fine prototype; for the somatosensory mega-cluster the first-encountered string sets the seed. Worth a sweep.
- **Member-count cap.** Forcing pattern separation after N members per node would bound the worst-case drift directly.

## Next-step routing

The brief's pre-registered routing was binary (cross-modal-only vs substrate-rework vs threshold-sweep). The actual data calls for a three-track follow-up sequenced before the cross-modal binding direction is re-prioritized:

1. **Decomposition sanity check.** Re-run this diagnostic with `MAXIM_CONCEPT_DECOMPOSITION=1` and `MAXIM_SUBSTRATE_PATH=1` to learn whether noun-chunked concepts ("food", "belly", "shoulders") resist drift. Cost: minutes; new code: none (script already supports it because `encode_decomposed` is the path the encoder takes when a decomposer is wired — modulo a small script change to construct the encoder with a `ConceptDecomposer`).
2. **Frozen-text-centroid sweep.** Add `"text"` to `ECConfig.frozen_centroid_modalities`, re-run the sequential walk. Pass if pair collapse stays ≥ 70% AND distractor collapse drops < 30%.
3. **Threshold sweep.** If frozen-centroid alone hurts pair collapse rate, sweep `MAXIM_EC_PATTERN_THRESHOLD_TEXT` in {0.45, 0.50, 0.55, 0.60} with the centroid still frozen. Tightens the rejection band.

Only after one of (2) or (3) restores clean sequential clustering does the persona-convergence work return to net-positive ROI on the cross-modal direction. Until then, anything that depends on text-modality EC node identity across a session is reading from a corrupted substrate.

## Reproduction

```bash
MAXIM_SUBSTRATE_PATH=1 python scripts/diagnose_roy_paraphrase_collapse.py \
    --input data/roy_paraphrase_pairs.json \
    --output /tmp/roy_paraphrase_collapse.json
```

Cost: zero $ (substrate-only, no LLM calls), ~10 s wall after the sentence-transformers model is warm.

## Provenance

- Curated pairs + distractors: hand-written for this experiment, register-matched to the two source Roy YAML fixtures.
- Decision thresholds (`pair_collapse_min=0.70`, `distractor_collapse_max=0.30`) pinned in [`data/roy_paraphrase_pairs.json`](../../data/roy_paraphrase_pairs.json) `_meta.decision_thresholds` so re-runs can't be tuned post-hoc.
- The `CENTROID_DRIFT_COLLAPSE` verdict label was added to the script's verdict logic during the same session as the run (the pre-registered three outcomes did not anticipate this data shape). The decision boundary itself — "isolated passes AND sequential fails on the same gate" — is mechanically defined; the label only renames the cell.
