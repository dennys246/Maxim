# D8 read-side mutation pre-registration — measure, then accept or separate (1.2 gate 3)

**Frozen 2026-09-04, merged to main BEFORE any data.** Implementation:
[`scripts/exp_d8_read_mutation.py`](../../../scripts/exp_d8_read_mutation.py) — its module
docstring mirrors this document; on any divergence THIS file is the authority, and a change
to either after first data requires an amendment header here, per house prereg rules.

## Question

D8 (bugs ledger): `bio_enrichment`'s recall path calls
`EC.pattern_complete_or_separate(embedding, "text", geometry=…)` per enrichment query, and
`"text"` is not a frozen-centroid modality — so every recall that completes moves the
matched node's centroid by a running-mean step (~1/(n+1)) and increments its member count.
The in-code claim is "intentional reconsolidation"; the undocumented consequence is
"querying degrades text-cluster resolution over time." 1.2 gate 3: **that mutation must be
measured and accepted, or separated from recall** — before shared substrate, because a
receiver's recall traffic would silently reshape merged foreign nodes.

This protocol measures the degradation claim on the real encode/recall path and computes an
accept-or-separate verdict from a decision rule frozen here.

## Apparatus (all in-process, no LLM, no hardware)

- Real `EntorhinalCortex` (default `ECConfig` — text `pattern_complete_threshold` **0.44**,
  the production operating point; 0.85 is the SENSOR-substrate threshold and does not
  apply here) + real `LinguisticEncoder` embeddings (`paraphrase-mpnet-base-v2`). **The
  hash-fallback encoder is a refusal, not a fallback** (exit 4): fallback vectors are not
  the production text geometry, so drift measured on them answers nothing.
- **Substrate**: the corpus frozen in the script — 16 concepts × 3 encode variants,
  encoded via the production protocol (embed → `pattern_complete_or_separate` →
  `register_substrate_node` on separation), with the production geometry tag passed.
- **Recall workload**: 2 held-out query variants per concept × R=6 repetitions = 192
  recall calls, replaying `bio_enrichment`'s exact call shape (same modality, same
  geometry source). Held-out variants model the realistic case: recall queries about
  known concepts that are near, not identical to, the encoded members.
- **Probes**: the 48 encode texts themselves. A probe run completes each against a
  **save/load CLONE** of the store (probing must not itself mutate the store under test);
  churn compares completed node ids before vs after the workload on the same id space.

## Arms

- **BASELINE** — the shipped behavior: workload runs, centroid updates + count increments
  fire as in production.
- **FROZEN (instrument arm)** — identical workload with `"text"` added to
  `frozen_centroid_modalities`: centroid drift must be **exactly 0.0**; nonzero drift here
  is an instrument failure → refuse (exit 4), no verdict.
- **AMPLIFIED (instrument arm)** — BASELINE workload at R×5 (960 calls) on a fresh store:
  measured mean drift must be **strictly greater** than BASELINE's; otherwise the meter
  cannot see the effect it claims to measure → refuse (exit 4), no verdict.
  (Both instrument arms exist because a zero can mean "no effect" or "blind meter" — the
  D43/vacuous-guard shape.)

## Metrics (BASELINE arm)

- **M1 probe identity churn**: fraction of the 48 probes whose completed node id after the
  workload differs from before. The direct "recall changed what recall returns" signal.
- **M2 centroid drift**: per-node cosine similarity between pre- and post-workload
  centroids; report mean and min (worst node).
- **M3 count provenance (reported, structural — never folded into the verdict)**: member
  counts incremented by recall completions vs by encode-time observation. This is a
  code-reading fact this protocol makes visible, not a measured unknown: `ec_merge`
  weights centroids by member count, so recall traffic buys merge weight. Its
  disposition belongs to gate 4's merge/threat design (weight by observation-only
  counts, or stop counting on recall); a workload-ratio threshold here would be
  predetermined by our choice of R, so none is set.

## Decision rule (frozen — no post-hoc motion)

1. **`separate-required`** iff M1 churn > **2/48** (more than two probes change identity)
   **or** M2 min per-node cosine < **0.98**. Rationale: churn is recall visibly changing
   recall — near-zero tolerance, with a two-probe allowance for genuinely borderline
   probes; 0.98 bounds a single session-scale recall workload to ≲2% worst-case cosine
   motion per node — beyond that, query traffic is visibly resculpting the cluster space
   that later recalls (and any shared-substrate merge) will read.
2. **`accept`** otherwise — scoped: the ~1/(n+1) reconsolidation stays, documented as
   within resolution bounds for realistic recall traffic; the M3 count-provenance fact is
   handed to gate 4 regardless of verdict.
3. Instrument-arm failures → **no verdict**, exit 4 (fix the meter, re-run).

The verdict is computed by `scripts/exp_d8_read_mutation.py::decide_verdict(metrics)` —
the protocol's own function, no operator judgment in the loop.

## Validity gates (refusals, exit 4)

Real semantic encoder loaded (no fallback); every concept forms exactly one node at encode
time with ≥ 3 members (the corpus is designed to cluster; a corpus that shatters is an
apparatus failure, not a finding); ≥ 90% of workload queries complete **in every arm** (a
workload that mostly separates measures node creation, not reconsolidation — and on the
instrument arms it makes the drift checks vacuous); FROZEN arm drift == 0.0 (implemented
as bit-identical embeddings — a float cosine of a vector with itself rounds below 1.0);
AMPLIFIED mean drift > BASELINE mean drift.

## Data + provenance

`docs/experiments/data/d8_read_mutation_<date>.json` — gated: written only with
`--write-experiment-results`, behind `_provenance.in_process_code_provenance` (clean tree,
this repo's `maxim`, provenance block stamped into the record). Verdict lands in the bugs
ledger D8 row (accept → ACCEPTED with the numbers; separate-required → the fix plan is a
read-only recall scan for `bio_enrichment`, and D8 stays OPEN until it ships with a
caller).

## Pre-freeze apparatus disclosure (the L11 precedent: measured pre-freeze, stated here)

The corpus was tuned pre-freeze against the real encoder until it clustered (three
concepts replaced after collapse refusals at the production 0.44 threshold; final state:
no cross-concept pair ≥ 0.44, all within-concept pairs well above it). A full apparatus
smoke then ran end-to-end (2026-09-04, no gated write) and **previewed the BASELINE
numbers: churn 0/48, drift mean_cos ≈ 0.9757 / min_cos ≈ 0.9521, workload 192/192
completions** — which sits on the `separate-required` side of the rule above. The
decision thresholds were authored BEFORE any run and were **not moved after the
preview** — moving a frozen threshold after seeing the number it will judge is exactly
the post-hoc motion this protocol forbids, in either direction. The official gated run
(post-merge, clean tree) is the record; the smoke is disclosed so nobody mistakes this
pre-registration for a blind one. **The apparatus is deterministic** (no seed, no
sampling; a review-round rerun reproduced the smoke to six decimal places) — absent a
code change, the official run will reproduce these numbers, so the expected verdict is
`separate-required`. What the official run adds is the gated, provenance-stamped record
computed on a clean tree, not suspense.

## Known-limit acknowledgments

- The corpus is authored, not sampled from live sessions — realistic in shape (percept-like
  sentences, near-paraphrase variants) but not a traffic distribution. A pass here bounds
  the mechanism under controlled load; it does not certify every workload. Recorded,
  accepted: the gate asks for a measured accept-or-separate, not a workload census.
- R=6 models a session's repeated enrichment of hot concepts; heavier long-run traffic is
  covered only by extrapolation (drift per completion shrinks as 1/(n+1); the AMPLIFIED
  arm bounds the direction).
- Probe churn uses the encode texts themselves; a query distribution far from the encoded
  members could churn differently. Same acceptance as above.
