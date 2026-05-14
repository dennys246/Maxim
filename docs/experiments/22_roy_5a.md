# Roy-5a — Cosine-localization disambiguator (1.1+ plan Stage 1)

**Date:** 2026-05-14 (run completed 2026-05-13 22:51 local)
**Plan:** [roy_5_encoder_alignment_disambiguator.md § Stage 1](../plans/roy_5_encoder_alignment_disambiguator.md) · [persona_convergence_crucible.md § "Iteration log"](../plans/persona_convergence_crucible.md)
**Companion:** [21_roy_4.md](21_roy_4.md) (Roy-4 FAIL that cancelled cross-modal binding plan and motivated Roy-5) · [20_roy_2c.md](20_roy_2c.md) (H1 confirmation that lives inside Roy-5's three sub-hypotheses)
**Spec:** [scenarios/roy/roy_5a_iteration.yaml](../../scenarios/roy/roy_5a_iteration.yaml)
**Engineered fixture:** [scenarios/roy/roy_2pc_holdout.yaml](../../scenarios/roy/roy_2pc_holdout.yaml) (reused unchanged from Roy-2pc / Roy-2c / Roy-4)
**Reproduction:** [protocols/22_roy_5a_reproduction.md](protocols/22_roy_5a_reproduction.md)
**Analyzer:** [scripts/analyze_roy_5_cosine_localization.py](../../scripts/analyze_roy_5_cosine_localization.py)
**Persistence prerequisite:** [PR #248](https://github.com/dennys246/Maxim/pull/248) (wires `EC.save()` + `ATL.save()` into `simulation/report.py::save_aut_state`)

## Status

**VERDICT: H1a — encoder subspace incompatibility.**

`max(M_tt food-bearing) = n/a` (no text-modality food-bearing priming centroids exist), which the pre-registered decoder maps to `< 0.20` → **H1a**. The empirical mechanism producing the verdict is **stronger than the plan modeled**: the issue is not "text food centroids exist but are far from arm-A text centroids in encoder space" (the plan's H1a framing), it is **"text-modality food centroids do not exist at all in either priming or arm A"**. The priming arc never lands a food concept in text modality; the agent's "sense_food_source" tool produces interoception-modality nodes only.

**Per [roy_5_encoder_alignment_disambiguator.md § Stage 2c](../plans/roy_5_encoder_alignment_disambiguator.md), this verdict triggers Stage 3** — redesigned cradle priming arc with deliberate `(sensor, drive, narrator-utterance)` co-firing + Hebbian retest. **But the secondary observation (text-modality silence on the food concept across both priming AND arm A) suggests the Stage 3 redesign needs to fix MORE than the plan anticipated.** Recommend the user inspect text-modality routing before Stage 3 begins. See "Recommended next step" below.

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

**Roy-4 had 57 NEW text-modality nodes fire during priming; Roy-5a had zero.** This is unexpected given that the priming spec is identical (same 5 cradle arcs, 10 turns each, same fixture). The likely explanations are operational rather than structural — different LLM context cache state, the first priming stage in Roy-5a had many AUT timeouts that may have suppressed narrator text routing, or some other run-to-run variance that doesn't reflect a code change. **No commit between Roy-4 (PR #246) and Roy-5a (PR #248 + this PR's branch) touches encoder routing, LinguisticEncoder, or the cradle narrator code path** — `git log c80190f..HEAD -- src/maxim/similarity/ src/maxim/agents/ src/maxim/runtime/agent_loop.py` returns empty.

Notably, **Roy-4's hippocampus dumps also showed zero `cli_input` / `transcript` fields**, so the 154 Roy-4 text-modality fires went through a code path that didn't land in the hippocampus perception fields the analyzer can inspect post-hoc. This is consistent with text fires from tool-output text, decomposed concept chunks, or other side-channel routings rather than direct cradle-narrator percepts.

**The H1a verdict survives either interpretation.** Both runs show:
- Food-bearing NAc cluster IDs all live in interoception modality.
- M_dd cosine ≈ 1.0 between priming and arm A food clusters (the surviving identity scheme).
- No text-modality food centroids exist in priming OR arm A.

## What this means for Stage 2

**The pre-registered verdict triggers Stage 2c → Stage 3** (redesigned cradle priming arc with deliberate co-firing). Per [roy_5_encoder_alignment_disambiguator.md § Stage 3](../plans/roy_5_encoder_alignment_disambiguator.md), Stage 3 ships:

- `_data/components/bodies/infant_humanoid_naming_v1.yaml` — new body with co-firing scaffold (additive, doesn't replace existing arc).
- `prompts/cradle_narrator.py` — narrator pattern that fires "hungry" / "thirsty" / "warm" utterances co-tick with the matching drive/sensor threshold.
- `scenarios/roy/roy_5b_iteration.yaml` — same shape as Roy-4 / Roy-5a but uses the redesigned arc.
- Re-run [`scripts/analyze_roy_4_coactivation.py`](../../scripts/analyze_roy_4_coactivation.py) on Roy-5b's trace. PASS → resurrect `cross_modal_substrate_binding.md` Stages 2-6; FAIL → promote encoder replacement to 1.2+ research.

**The Roy-5a secondary finding (text-modality silence on the food concept) refines the Stage 3 design requirement.** It's not enough for the redesigned arc to produce co-firing — it also needs to produce **non-zero text-modality fires on food-related percepts** so that the Hebbian binding rule has at least one cross-modality node pair to evaluate. If the narrator's "hungry" utterance routes through a code path that doesn't reach `LinguisticEncoder` (the same path that's silent in Roy-5a's run), Stage 3 will fail not because the binding mechanism is dead but because there are no text-modality nodes to bind.

## Recommended next step

**Before committing to Stage 3 implementation**, the user should:

1. **Re-run Roy-5a once or twice** to confirm the text-modality-silence finding is stable. If a re-run produces text-modality fires (matching Roy-4's 154), Roy-5a was anomalous and Stage 3 can proceed as planned. If text-modality silence reproduces, item (2) becomes load-bearing.
2. **Inspect why the cradle narrator's text isn't routing to `LinguisticEncoder` on food-related percepts.** Likely places to look: the orchestrator's narrator-percept routing in `simulation/orchestrator.py`, `LinguisticEncoder.encode`'s text extraction (`percept.transcript_chunk or percept.content`), and whether `embodiment/percepts.py::EmbodimentPerceptSource` populates either field on food-sensor ticks.

If text-modality routing turns out to be quietly broken (rather than variance), Stage 3's cradle redesign will need to fix the routing AS PART OF the redesigned arc rather than assuming it works.

## Why this verdict is more confident than Roy-4's FAIL

Roy-4's FAIL was robust across a sweep of `(min_cofire, min_weight)` parameters. Roy-5a's H1a verdict has a similar robustness property: the verdict triggers on **the absence of text-modality food clusters**, which is **observable directly** in the persisted EC dump (`aut_ec.json`) without dependence on any analyzer hyperparameter. The pre-registered threshold boundaries (`0.20` / `0.40`) only matter if `max(M_tt)` is computable; here it isn't, and the analyzer's `-inf` handling — pinned by `test_negative_infinity_decodes_h1a` — flows directly to H1a.

The **`M_dd` cosine ≈ 1.0** result is the strongest positive signal in this iteration: priming food clusters DO land in arm A's interoception substrate, perfectly. Whatever future Stage 3 / 4a / 4b implementation ships, this confirms the substrate has the right concept on the interoception side — the gap is strictly in projecting it to text modality (which is the substrate channel CLI fixture text routes through).

## References

- [docs/plans/roy_5_encoder_alignment_disambiguator.md](../plans/roy_5_encoder_alignment_disambiguator.md) — the plan Roy-5a is Stage 1 of.
- [docs/experiments/21_roy_4.md](21_roy_4.md) — Roy-4 FAIL that cancelled the binding plan and motivated Roy-5.
- [docs/experiments/20_roy_2c.md](20_roy_2c.md) — H1 confirmation; Roy-5a disambiguates which sub-hypothesis (H1a / H1b / H1c) holds.
- [feedback_two_identity_schemes.md](../../.claude/projects/-Users-dennyschaedig-Scripts-Maxim/memory/feedback_two_identity_schemes.md) — the cross-Roy pattern Roy-5a re-confirms (interoception centroids survive at the embedding level, cluster_ids do not).
- [feedback_interim_contamination.md](../../.claude/projects/-Users-dennyschaedig-Scripts-Maxim/memory/feedback_interim_contamination.md) — the trap Roy-5a's verdict protects against (Stage 2c → Stage 3 explicitly does NOT route through a hand-curated lexicon, even if the bio-faithful path takes longer).
- [feedback_review_before_ship.md](../../.claude/projects/-Users-dennyschaedig-Scripts-Maxim/memory/feedback_review_before_ship.md) — two-lens pre-merge review pattern; applied to the Roy-5a analyzer PR.
- [`/tmp/roy_5a_analysis.json`](/tmp/roy_5a_analysis.json) — full analyzer output bundle (M_tt / M_dt / M_dd matrices + verdict + per-arm headlines), regenerated by re-running the analyzer per the reproduction protocol.
