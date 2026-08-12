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
one. Early and late flips are not the same treatment. Unmeasured; cheap to measure
offline from existing captures.

**F7 — Hallucinated actions are being scored.** `approach_flame` appears in re-query
output and exists nowhere in the codebase (real affordances: `observe`/`warm_self`/
`touch`). Currently scored as neutral and not caught by the phantom guard. Needs an
explicit pre-registered rule.

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

- The apparatus runs end-to-end and every gate fires correctly.
- Both learn arms reach bias 0.9967; the counterbalance learns the opposite color.
- The transplant surfaces (validity gate passes) — the VOID branch was not needed.
- The per-annotated-decision effect replicates the exploratory result (~0.51).
- The primary test behaves exactly as pre-registered at n=2.

## What must be settled before the freeze

1. **Name-copying vs learned content** — F1 makes this the central open question. Design:
   [../plans/annotation_context_and_provenance.md](../plans/annotation_context_and_provenance.md).
2. **Determinism of temp-0 re-query** (#496) — the method's core assumption, currently
   assumed rather than measured.
3. **Invalid-action rule** (F7) — recommendation: score as neutral (an unexecutable action
   *is* disengagement) and report the rate; frozen in the pre-registration either way.
4. **Corrected framing for the entangled axes** (F2) in both this doc's parent
   ([44_substrate_counterfactual.md](44_substrate_counterfactual.md)) and the paper outline.
