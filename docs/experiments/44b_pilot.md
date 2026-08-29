# Exp 44b Pilot — instrument shakedown + what it revealed (2026-08-10/11)

**Status:** PILOT COMPLETE — not a result. One seed per arm; the confirmatory campaign
(10 seeds/arm, [protocols/exp44b_preregistration.md](protocols/exp44b_preregistration.md))
is **not yet frozen** and its freeze is now gated on the findings below.
**Machine:** big-mac-mini, qwen2.5-32b-instruct local, isolated `MAXIM_DATA_HOME`s.
**Provenance:** learn stages at `9e3ebc4e`; verified capture + re-query at `7b51b949`.
**Reproduce:** [protocols/exp44b_reproduction.md](protocols/exp44b_reproduction.md).

## What the pilot was for

Shake down the confirmatory apparatus end-to-end (learn → capture → re-query → stats),
settle the two open gate questions (does the annotation reach prompts? does a transplanted
substrate surface across the `_b` name suffix?), and produce the numbers needed to sign the
pre-registration freeze. It did all three — and then found something that changes the claim.

## Results (1 seed/arm — descriptive only)

| | correct-content (A+B) | wrong-content (CTRL transplant) |
|---|---|---|
| runs | 2 | 1 |
| directional flips | **20 safe : 1 harm** | **9 annotated : 7 other** |
| commitment axis | **16 : 0** | **0 : 0** |
| ablated-side baseline | A 0.667 / B 0.778 safe-colored | 0.875 safe-colored (7:1) |
| phantom exclusions | 0 | 0 |

- **Primary sign test: +2 / −0 runs, p = 0.50.** That p is the *floor* at n=2 (two-sided
  exact binomial cannot go below 0.5 with two units); n ≥ 6 is needed for p < 0.05, and
  the pre-registered n = 20 reaches ≈2×10⁻⁶ if this direction holds. Correctly
  non-significant; direction maximal.
- **The effect size replicates the exploratory run almost exactly.** Raw flip rate looks
  3.4× higher than Exp 44 (0.51 vs 0.15), but that is annotation *coverage*: Exp 44 had
  ~29-30 annotated prompts inside 100 scored decisions (15/29 ≈ **0.52** per annotated
  decision); the pilot has `annotation_fraction = 1.0` (18/35 ≈ **0.51**). Same per-annotated
  effect, months apart, on a rebuilt harness.

## What the pilot revealed (the part that matters)

Reading the actual annotation text out of a capture row was worth more than the statistics.
The control arm's prompt carried:

```
=== Substrate associations from prior experience ===
  purple_flame_observe            [strongly rewarding from prior experience]
  green_flame_touch               [strongly rewarding from prior experience]
  focus_on_sound                  [strongly rewarding from prior experience]
  green_flame_warm_self           [strongly rewarding from prior experience]
  infant_humanoid_chilled_listen  [strongly rewarding from prior experience]
```

**F1 — The transplant control is name-mismatched, so it under-tests the channel.**
Those are `green_flame_*` / `purple_flame_*` signatures in a world that only offers
`green_flame_b_*` / `purple_flame_b_*` tools: the annotation names tools that do not exist
in the capture world. The weak 9:7 direction and the **zero** commitment flips are both
explained by "the named committed action was not callable" at least as well as by
"content was wrong". Cross-arm transplant is *inherently* name-mismatched because our
counterbalance is implemented by renaming entities. The control therefore cannot separate
**name-copying** from **learned-content following** — that separation is now the open
question, not a footnote.

**F2 — The annotation transmits (entity, affordance) pairs, not a preference.** BOTH
colors appear as "strongly rewarding". What discriminates is which affordance each color
is paired with — green with `warm_self`/`touch`, purple with `observe`. So the directional
and commitment axes are **not independent measurements**; the annotation encodes them
jointly. Exp 44's write-up presented them as two effects; that framing must be corrected.

**F3 — "Rewarding" conflates its own causes.** Drive relief (`warm_self`), bare tool
success (`observe`), and cross-modal audio credit (`focus_on_sound`,
`infant_humanoid_chilled_listen`) all render with the identical phrase. The LLM cannot
tell *why* something is rewarding and so cannot judge whether the reason transfers.

**F4 — The annotation discards context; the substrate does not.**
`NAc.get_agent_tool_biases` iterates `for (aid, _cid, tool_sig), bias in
self._cluster_reward_bias.items()` — the cluster id is **dropped**, and per-tool bias is
max-aggregated agent-wide. Meanwhile the substrate-primary policy (`recommend_action`)
consults `consulted_bias_by_modality` for the *active* clusters. Same knowledge: the
no-LLM path uses context, the LLM path cannot see it. Context-dependent value
("this affordance is good HERE and bad THERE") is structurally inexpressible in the
prompt channel today. (The agent-wide aggregation was a deliberate Roy-2c fix for
encoder-alignment disjointness — it is not a bug, but it is a ceiling.)

**F5 — Transfer is signature-keyed, not semantic.** The affordance-concept machinery
exists and works (`AffordanceDecompositionStrategy` → EC pattern completion, the
"flame"→"fire" transfer result), but `get_agent_tool_biases` keys on exact tool-signature
strings, so the annotation path bypasses it. That is why `green_flame` → `green_flame_b`
barely transfers.

**F6 — The signal is non-stationary within a run.** Cluster bias fell ≈0.997 → 0.059
across a capture despite the τ=1000 hold; bands are absolute (`≥0.5` = "strongly
rewarding"), so early decisions get a strong annotation and late ones a weak or absent
one. Early and late flips are not the same treatment. **MEASURED 2026-08-24 — see
[§S4](#s4-non-stationarity-measurement-2026-08-24-roadmap-11-item-11) below:** the
prediction holds in kind but not in shape — bands degrade strongly→mildly for the
non-target tools by the second half in every arm, the target's band holds, and no
tool drops out late.

**F7 — Hallucinated actions are being scored.** `approach_flame` appears in re-query
output and exists nowhere in the codebase (real affordances: `observe`/`warm_self`/
`touch`). Currently scored as neutral and not caught by the phantom guard. Needs an
explicit pre-registered rule.

## S4 non-stationarity measurement (2026-08-24, roadmap 1.1 item 11)

Offline analysis of the three pilot captures (1 seed/arm, 30–36 decisions each)
with [`scripts/exp44/analyze_nonstationarity.py`](../../scripts/exp44/analyze_nonstationarity.py)
at `main` `7aed652b`, re-query results supplied (`qwen2.5-32b-instruct`, e8, t0.7).
Captures + re-query JSONL are committed under
[`data/44b_pilot/`](data/44b_pilot/) (copied from big-mac-mini `~/exp44b/pilot/`;
per-arm model files excluded); the analyzer's JSON + stdout per arm under
[`data/44b_s4_nonstationarity/`](data/44b_s4_nonstationarity/). Descriptive only —
one seed per arm.

**Band tier per tracked tool, first half → second half of the run** (tier 2 =
"strongly rewarding", 1 = "mildly", 0 = neutral/absent; mean over decisions):

| arm | non-target tools (4) | target `warm_self` | decisions without any annotation |
|---|---|---|---|
| A green_safe | 1.78–1.83 → **1.00** (drift −0.78 … −0.83) | 1.94 → 2.00 (+0.06) | 6/36, scattered at [21–23, 32–34], not a trailing cliff |
| B purple_safe | 1.93–2.00 → **1.00–1.07** (−0.93 … −1.00) | 2.00 → 2.00 (0.00) | 0/30 |
| CTRL transplant | 1.79–1.86 → **1.00** (−0.79 … −0.86) | 1.79 → 1.00 (−0.79) | 5/35, scattered at [2–4, 23, 27] |

**Flip rate by run half** (re-query flip vs the ablated prompt): A **0.667 → 0.333**,
B 0.600 → 0.533, CTRL 0.412 → 0.611 (annotated-only: 0.667→0.500, 0.600→0.533,
0.500→0.688). **By strongest band in the prompt:** A strongly 0.600 (n=30) vs absent
0.000 (n=6); CTRL mildly 0.667 (n=18) vs strongly 0.500 (n=12) — not monotone in
the predicted direction. **Free determinism probe** (decisions whose full and
ablated prompts are byte-identical, so any flip is decoding noise): **0.000 on 11
pairs** (6 in A, 5 in CTRL) — temp-0 re-query agreed on every identical-prompt
pair. This is the first measurement of the method's core assumption (settle-item
2 below); one seed per arm, so it bounds noise at "not observed", not at zero.

**Reading.** F6 is confirmed in kind: the treatment is NOT stationary within a run
— in every arm the non-target tools' bands fall from strongly to mildly rewarding by
the second half while the target tool's band holds (in A and B; in the transplant
control it falls with the rest, which is itself consistent with F1's
signature-keyed transfer). It is NOT the shape F6 hypothesised: nothing decays to
neutral (the 0.997 → 0.059 bias fall lands in the "mildly" band, not below it), and
tools do not drop out of the annotation late — the unannotated decisions are
scattered. Flip rate does not track band tier monotonically and the half-to-half
change differs in sign across arms, so at n=1 seed/arm the *dose→flip* relation is
unmeasured, only the *dose drift* is. **Consequence for the freeze:** the
confirmatory analysis must either stratify decisions by run half (or by band tier
as a covariate) or the apparatus must hold the band tier across the run; pooling
early and late decisions pools two doses. Added as settle-item 5.

## Instrument bugs the pilot caught (before any claim rested on them)

1. **Wrong body** — `infant_humanoid` has no `cold` sensor, so the flames' relief was a
   silent no-op and no seed could ever pass the learn gate. Fixed: `infant_humanoid_chilled`
   (entropic cold). Symptom: `passed_gate=False`, every drive at set point, drift +0%.
2. **Gate false-negative** — `has_cluster_bias` is nested under `world_state`; the counter
   read the top level, so `annotation_fraction` was structurally 0.0 and the gate failed a
   healthy capture (30/34 prompts demonstrably annotated). Fixed + guarded (#489).
3. **Cross-arc discovery leakage** — registry-wide component-index enrichment surfaced the
   counterbalance twins in the other arm's prompts. Phantom-pick exclusion added (#488);
   pilot recorded 0 phantoms, so the guard is armed but not yet load-bearing.
4. **Triple double-launch** — two runners raced one workdir (two `campaign_start`s 82 s
   apart), interleaving one capture file. Single-runner lock added (#494).
5. **H3 invisible in its own report** — control numbers were written to `stats.json` but
   never printed (#495).

Reported as evidence the gating discipline works *and* as honest disclosure of harness
fragility; both belong in the paper's limitations section.

## What the pilot establishes

**Provenance disclosure (2026-08-29, surfaced by `scripts/lint_prereg_precedes_data.py`).** The
pilot's `campaign_start` record (`data/44b_pilot/manifest.jsonl`, 2026-08-10 11:53:49 local)
precedes the pre-registration's commit `1667ad19` (12:10:05 −0600) by 16 minutes — the prereg
was on a branch, not on `main`, when the pilot began, and the first `learn` sub-run started at
12:16. The pilot is an instrument shakedown, not the confirmatory run, and nothing in the 44b
gates rests on it; it is grandfathered in the lint by explicit entry with this reason. The
confirmatory run must start after the prereg (and any amendment) is on `main`.

- The apparatus runs end-to-end and every gate fires correctly.
- Both learn arms reach bias 0.9967; the counterbalance learns the opposite color.
- The transplant surfaces (validity gate passes) — the VOID branch was not needed.
- The per-annotated-decision effect replicates the exploratory result (~0.51).
- The primary test behaves exactly as pre-registered at n=2.

## What must be settled before the freeze

1. **Name-copying vs learned content** — F1 makes this the central open question. Design:
   [../plans/annotation_context_and_provenance.md](../plans/annotation_context_and_provenance.md).
2. **Determinism of temp-0 re-query** (#496) — the method's core assumption. **First
   measurement 2026-08-24 (§S4): 0.000 flips on 11 identical-prompt pairs** across
   two arms; one seed per arm, so the freeze should still budget a determinism
   arm at power rather than treat this as zero.
3. **Invalid-action rule** (F7) — recommendation: score as neutral (an unexecutable action
   *is* disengagement) and report the rate; frozen in the pre-registration either way.
4. **Corrected framing for the entangled axes** (F2) in both this doc's parent
   ([44_substrate_counterfactual.md](44_substrate_counterfactual.md)) and the paper outline.
5. **Non-stationary dose** (F6, measured in §S4) — stratify by run half / band tier,
   or hold the tier across the run, and say which in the frozen analysis parameters.
