# Exp 37 — Cross-Model Results

**Status:** IN PROGRESS 2026-06-13 (4 open-source fires complete: Qwen14B, Qwen32B, Mistral24B → **Goldilocks-zone finding**; DeepSeek-R1-Distill-Qwen-32B → **reasoning amplifies substrate** [bucket R-A] + first clean Wire-A ablation; cloud comparisons deferred behind prompt-caching refactor).
**Companion doc:** [37_cross_session_graduation.md](37_cross_session_graduation.md) — pre-registration + per-fire verdict writeups.
**Plan:** [docs/plans/exp37_cross_model_characterization.md](../plans/exp37_cross_model_characterization.md) — methodology + sequencing.
**Purpose:** Cross-cutting interpretation across model fires. Each per-fire verdict stays in `37_cross_session_graduation.md`; this doc compiles the cross-model story that emerges from the union.

## Methodology — three orthogonal axes

The cross-model fires decompose the substrate-transfer question across three axes:

| Axis | Controls for | Compares |
|---|---|---|
| **Scale** (within Qwen family) | Family / training corpus | Qwen14B vs Qwen32B (vs Qwen72B if added post-1.0) |
| **Family** at moderate scale | Scale ballpark | Qwen32B vs Mistral24B (open) — and vs closed-source once unblocked |
| **Training paradigm — reasoning vs chat** | Family AND scale (cleanest isolation) | Qwen32B vs DeepSeek-R1-Distill-Qwen-32B (same base, same scale, only reasoning-training differs) |
| **Open vs closed** | None — multi-confound but anchors the story | Open-source fires vs Claude/GPT/DeepSeek (post-prompt-caching) |

Each axis informs different aspects of the interpretation:
- Scale → "does substrate signal scale with model capacity?"
- Family → "is the effect Qwen-specific or universal across open models?"
- **Reasoning paradigm → "does explicit-thinking training change how substrate context is processed?"**
- Open/closed → "does alignment regime / closed-source RLHF affect substrate-LLM interaction?"

## Model lineup

| Model | Where | Status | Axis contribution |
|---|---|---|---|
| Qwen2.5-14B-Instruct | leader (local) | **DONE** 2026-06-06 (60/60) | Scale baseline (small end) |
| Qwen2.5-32B-Instruct | leader (local) | **DONE** 2026-06-08 (60/60) | Scale comparison (large end, same family) |
| Mistral-Small-24B-Instruct-2501 | leader (local) | **DONE** 2026-06-11 (60/60) | Family comparison at intermediate scale |
| DeepSeek-R1-Distill-Qwen-32B | leader (local) | **DONE** 2026-06-13 (60/60) | Reasoning-axis isolation (same base as Qwen32B) |
| Llama 3.3 70B Instruct | leader (local) | CONDITIONAL on DeepSeek result | New family + biggest local scale; only if DeepSeek leaves open questions |
| Claude Sonnet 4.6 | peer (cloud) | DEFERRED behind prompt-caching refactor | Closed-source anchor #1 |
| GPT-4o | peer (cloud) | DEFERRED | Closed-source anchor #2 + OpenAI key-path validation |
| DeepSeek-V3 | peer (cloud) | DEFERRED | Radically different training corpus + cheap |

Cloud fires (Sonnet/GPT/DeepSeek) blocked behind [prompt_caching_for_cloud_backends.md](../plans/prompt_caching_for_cloud_backends.md) Phase 1+ work, which is deferred to post-1.0 per the 2026-06-09 decision (Qwen32B's positive result reduced 1.0 urgency for closed-source comparison; the refactor is multi-day work better done without 1.0 ship pressure).

## Headline results — primary metric (`positive_approach_engagement_fraction`)

The cradle scenario's primary metric on fire_pit, across all model fires:

| Model | Arm A mean ± SD | Arm B mean | Δ (SD units) | Primary verdict | Corroborating PASS | Robustness |
|---|---|---|---|---|---|---|
| Qwen2.5-14B-Instruct | 0.533 ± ~0.27 | 0.517 | −0.06 | **FAIL** | 0/4 | FAIL |
| Qwen2.5-32B-Instruct | 0.420 ± 0.27 | 0.800 | **+1.43** | **PASS** | **2/4 PASS** | **PASS** |
| Mistral-Small-24B-Instruct | **1.000 ± 0.000** | 0.600 | **−0.40** (wrong dir; zero-SD fallback) | **FAIL** | 0/4 | FAIL |
| DeepSeek-R1-Distill-Qwen-32B | 0.259 ± 0.145 | 0.566 | **+2.11** | **PASS** | 1/4 PASS | FAIL (diverges) |
| Llama 3.3 70B Instruct | TBD (conditional) | — | — | — | — | — |
| Claude Sonnet 4.6 | TBD (deferred) | — | — | — | — | — |
| GPT-4o | TBD (deferred) | — | — | — | — | — |
| DeepSeek-V3 | TBD (deferred) | — | — | — | — | — |

(SD-shift threshold for primary PASS: +1.0 SD per [exp37_sd_shift.md](../plans/exp37_sd_shift.md).)

## Secondary criterion — ablation attribution

The pre-reg's secondary criterion: ≥1 of 3 ablations should shrink Arm B's delta toward Arm A.

| Model | Wire-A off Δ shrink (SD) | Wire-1 off Δ shrink (SD) | NAc-bias off Δ shrink (SD) | Secondary verdict |
|---|---|---|---|---|
| Qwen2.5-14B-Instruct | −0.06 (overshoot) | −0.21 (overshoot) | −0.18 (overshoot) | **FAIL** (0/3) |
| Qwen2.5-32B-Instruct | +0.56 (suggestive) | −0.06 (no change) | +0.03 (no change) | **FAIL** (0/3, but Wire-A directionally consistent) |
| Mistral-Small-24B-Instruct | n/a (ablation degenerate) | n/a | n/a | **FAIL** (0/3 — not measurable; see note) |
| DeepSeek-R1-Distill-Qwen-32B | **+1.13 (PASS)** | +0.69 (FAIL) | overshoot (FAIL) | **PASS (1/3)** — first clean ablation across all fires |

**Mistral24B secondary criterion is structurally unmeasurable.** The ablation test asks "does turning off bio-mechanism X shrink B's delta toward A?" — but for Mistral24B the B-vs-A delta is already NEGATIVE (B=0.600 < A=1.000, Δ=−0.40). There is no positive delta to "shrink." The ablation arms (B-wire-a-off=0.600, B-wire-1-off=0.600, B-nac-bias-off=0.700) all sit at or near B, so the analyzer reports "insufficient data for ablation comparison." This is a direct consequence of the ceiling effect (Arm A already perfect — see headline finding below).

**Pattern across the completed fires:** the substrate effect is NOT cleanly attributable to any single bio-mechanism. At 32B (Qwen), Wire-A ablation moves the needle (+0.56 SD shrinkage) but Wire-1 and NAc-bias don't; at 24B (Mistral) the question is moot because there's no positive delta to attribute. Either multi-channel mediation (each channel contributes a little, no single ablation is decisive) or substrate-context-as-a-whole effects on LLM reasoning that no single annotation captures.

## Isolation arm (Arm C — "general caution" confound)

| Model | Arm C mean | Inside Arm A's band? | Verdict |
|---|---|---|---|
| Qwen2.5-14B-Instruct | 0.617 | YES (band [0.333, 0.96]) | PASS |
| Qwen2.5-32B-Instruct | 0.667 | NO (outside band [0.033, 0.660]) | **FAIL — confound** |
| Mistral-Small-24B-Instruct | 0.400 | NO (outside band [1.000, 1.000]) | **FAIL — but in the OPPOSITE direction** |
| DeepSeek-R1-Distill-Qwen-32B | 0.527 | NO (outside band [0.145, 0.479]) | **FAIL — confound (C ≈ B, both above A)** |

Arm C carries a peaceful-prior session (control for "any prior shifts behavior" vs "fire-failure prior shifts behavior specifically"). When C falls OUTSIDE A's empirical band AND looks more like B, that's the signal that the substrate effect generalizes across priors rather than being scenario-specific. **Qwen32B triggered this confound** in the upward direction (C ≈ B, both above A). **Mistral24B triggers it in the opposite direction:** C=0.400 is BELOW A=1.000, not above. Because Arm A is a degenerate point distribution (every value exactly 1.000, band [1.000, 1.000]), literally ANY non-perfect arm falls "outside the band." The confound flag here is a statistical artifact of A's zero variance, not evidence of cross-prior generalization. C being the WORST arm (lowest warm_self, has touches) is actually the predicted fire-specific-learning pattern — the peaceful-prior agent is the most dangerous near fire. **DeepSeek-R1 reproduces Qwen32B's upward confound** (C=0.527 ≈ B=0.566, both well above A=0.259): the peaceful-prior agent behaves much more like the fire-failure-prior agent than like the fresh agent. As with Qwen32B, this means the substrate effect is not cleanly fire-failure-specific — *any* resumed prior shifts behavior up — and the per-fire verdict is gated to investigation rather than a clean substrate-attribution PASS. The reasoning overlay does not resolve the confound.

## Descriptive corroborating — `fire_approach_action_count`

Counts of warm_self actions per session — not pre-reg gated, but directionally informative.

| Model | A mean | B mean | Δ | Direction |
|---|---|---|---|---|
| Qwen2.5-14B-Instruct | 1.60 | 1.40 | −0.20 | **WRONG** (B less than A) |
| Qwen2.5-32B-Instruct | 1.60 | 3.00 | **+1.40** | **PREDICTED** (B more than A) |
| Mistral-Small-24B-Instruct | 1.00 | 0.80 | −0.20 | **WRONG** (B less than A) — but A is at ceiling |
| DeepSeek-R1-Distill-Qwen-32B | 1.40 | 3.20 | **+1.80** | **PREDICTED** (B more than A) — largest of any fire |

## Sharp_rock scenario — degenerate at ALL FOUR fires

Qwen14B, Qwen32B, Mistral24B, AND DeepSeek-R1-Distill-Qwen-32B all produced near-zero engagement across every sharp_rock arm (R1: A=0.000, B=0.000 — primary FAIL via zero-SD fallback; all three ablations report "insufficient data"). The asymmetric-design concern from [exp37_metric_pivot.md](../plans/exp37_metric_pivot.md) (sharp_rock has no positive-approach analog like fire_pit's `warm_self`) is now realized at four model fires spanning two families, three scales, and a reasoning-trained variant. This is no longer a Qwen-specific quirk — **sharp_rock is structurally broken for cross-model use**, and the cradle scenario needs a positive-approach affordance for sharp_rock (or a different second scenario) before it can carry verdict weight. Tracked as cradle-redesign motivation for post-1.0 work. fire_pit alone carries the substantive evidence on all cross-model fires.

## HEADLINE FINDING — the Goldilocks zone of prior strength

**The three completed fires form a clean three-point story that the original pre-registered buckets (A/B/C/D) did not anticipate. Mistral24B did not land in any of them cleanly — it revealed a CEILING EFFECT that reframes the whole cross-model question.**

| Model | Arm A (fresh agent) mean | Arm B (resumed, has substrate) mean | Δ | Position |
|---|---|---|---|---|
| Qwen2.5-14B | 0.533 (variable) | 0.517 | −0.02 | **Below the zone** — priors too weak |
| Qwen2.5-32B | 0.420 (variable) | 0.800 | **+0.38 (+1.43 SD, PASS)** | **Inside the zone** — sweet spot |
| Mistral-Small-24B | **1.000 (SD=0, perfect)** | 0.600 | −0.40 | **Above the zone** — ceiling effect |
| DeepSeek-R1-Distill-Qwen-32B | 0.259 (variable) | 0.566 | **+0.31 (+2.11 SD, PASS)** | **Deep inside the zone** — reasoning overlay lowers Arm A, widens the headroom, amplifies the signal |

**The reframe:** substrate-transfer signal is detectable only in a *Goldilocks zone of prior strength* — the LLM's first-encounter priors must be good enough to act on substrate-derived context, but not so good the task is already solved on the first try.

- **Qwen14B sits BELOW the zone.** Its fresh-agent priors are mediocre (A=0.533), and they're not strong enough for substrate context to be leveraged — B stays at 0.517. No transfer because the model can't act on what the substrate is telling it.
- **Qwen32B sits INSIDE the zone.** Its fresh-agent priors leave headroom (A=0.420), and substrate context fills that headroom — B jumps to 0.800. This is the +1.43 SD PASS. The sweet spot.
- **Mistral24B sits ABOVE the zone.** Its fresh-agent priors already solve the task *perfectly* (A=1.000, SD=0.000 — every one of 5 trials picks warm_self, zero touches, reaches warming at action index 0.4). **There is no headroom for substrate to demonstrate improvement.** You cannot improve on 1.000. If anything, substrate context adds exploration noise: the resumed agent (B) dawdles to action index 2.6 before warming, and drops to 0.600.

### Why this is a training-method effect, not a scale effect

The striking part: **Mistral24B (24B params) hits the ceiling that Qwen32B (32B params) does NOT.** A smaller model has *stronger* cradle-task priors than a larger one. This is exactly the family/training-method axis the cross-model design was built to isolate — and it pays off cleanly. Mistral's instruction-tuning apparently produces near-optimal "infant near fire → warm yourself, don't touch" reasoning out of the box, where Qwen (even at 32B) leaves room for substrate to contribute.

So the cross-model picture is NOT "bigger models leverage substrate better" (the naive scale story). It is: **substrate transfer is detectable when there's a gap between the model's priors and optimal behavior — and that gap is governed by training method at least as much as parameter count.**

### The "tell" — time-to-first-warm-self

The single most diagnostic number in the Mistral fire:

| Arm | action index of first warm_self | reading |
|---|---|---|
| A (fresh, no substrate) | [0, 1, 0, 0, 1] → mean **0.4** | beelines straight to the safe warming action |
| B (resumed, has substrate) | [3, None, 3, 1, 4] → mean **2.6** | explores more before warming (one trial never warms) |

The fresh agent goes *immediately* to the optimal action. The substrate-having agent explores more first. For Mistral, substrate context is not "memory of past danger that makes you cautious" — it's added context that slightly perturbs an already-optimal first-move policy. This is the cleanest possible illustration of the ceiling effect: when priors already produce optimal behavior, substrate can only add variance.

### What the original buckets got wrong (honest pre-registration accounting)

The pre-registered buckets (A: PASS like Qwen32B / B: intermediate / C: null-family-specific / D: stronger-than-Qwen32B) all assumed Mistral's Arm A would be in the same variable-mediocre regime as Qwen's. None anticipated **A=1.000 with zero variance** — a degenerate point distribution. The result is technically "bucket C (null)" by the analyzer's verdict, but the *reason* is the ceiling effect, not "Qwen-family-specific signal" or "sharp phase transition" as bucket C predicted. The pre-registration was honest and useful — it forced us to write down what each outcome would mean — but the data revealed a mechanism (prior saturation) outside the pre-registered hypothesis space. That is exactly what exploratory cross-model characterization is for.

### 1.0 framing implication

This is a *stronger and more nuanced* 1.0 story than "substrate works at scale":

> "Substrate carries cross-session memory across all models tested (EARNED via Exp 10 + persistence across all three fires). Whether that carried memory produces a measurable behavioral shift depends on a Goldilocks condition: the base LLM's priors must leave headroom between first-encounter behavior and optimal behavior. Qwen2.5-32B sits in this zone and shows a clear +1.43 SD substrate-transfer effect; Qwen2.5-14B's priors are too weak to leverage substrate; Mistral-Small-24B's priors already solve the cradle task perfectly, leaving no headroom. The condition is governed by training method as much as parameter count. The specific bio-mechanism carrying the effect (Wire-A / Wire-1 / NAc-bias) remains unattributed — a substantive Exp 38 / post-1.0 research question."

The strong "substrate drives action selection via specific bio-mechanisms" claim STAYS pulled from 1.0 bio-framing. The Goldilocks finding actually *strengthens* the case for the post-1.0 substrate-primary direction (Exp 38): if you want to demonstrate substrate-driven behavior unambiguously, you need to remove the LLM-prior confound entirely, because under LLM-AUT the signal is only visible in a narrow prior-strength band.

## REASONING-AXIS RESULT — DeepSeek-R1-Distill-Qwen-32B → Bucket R-A (reasoning amplifies substrate)

**Fire complete 2026-06-13 (60/60). Verdict: PARTIAL — investigation gate** (same gate as Qwen32B, for the same Arm-C-confound reason), **but the strongest substrate signal of any fire AND the first clean ablation attribution.** Full analyzer output: [data/37_results_r1_distill_qwen_32b.md](data/37_results_r1_distill_qwen_32b.md).

The DeepSeek fire isolates the reasoning axis: DeepSeek-R1-Distill-Qwen-32B is the R1 reasoning-trace fine-tune of the SAME Qwen2.5-32B base we already fired — same architecture, tokenizer, scale, pretraining. The only methodologically-meaningful difference is the reasoning-training overlay, so the comparison vs Qwen32B is a clean A/B on the reasoning-paradigm axis. (Llama-distill variants would confound reasoning + new family + larger scale at once; avoided for the headline test.)

### Headline: this is Bucket R-A

| Quantity | Qwen32B (base) | DeepSeek-R1 (reasoning) | Reading |
|---|---|---|---|
| Arm A (fresh) mean ± SD | 0.420 ± 0.27 | **0.259 ± 0.145** | reasoning overlay LOWERS fresh-agent priors — deeper in the Goldilocks zone |
| Arm B (resumed) mean | 0.800 | 0.566 | absolute B is lower, but A is lower too |
| Δ (SD units) | +1.43 (PASS) | **+2.11 (PASS)** | **reasoning AMPLIFIES the substrate-transfer effect** |
| Best ablation shrinkage | Wire-A +0.56 (suggestive, FAIL) | **Wire-A +1.13 (PASS)** | **first clean ablation attribution across all four fires** |
| `fire_approach_action_count` Δ | +1.40 | **+1.80** | largest warm_self count lift of any fire |

**Two findings, both pointing the same way:**

1. **Reasoning amplifies substrate utilization.** The SD-shift jumps from +1.43 (Qwen32B) to +2.11 (R1) — the largest of any fire. The "think before responding" pattern lets the model engage more deliberately with substrate-derived prompt context: drives, recalled episodes, and valence annotations get weighted in an explicit reasoning chain rather than skimmed. This is bucket R-A as pre-registered.

2. **Reasoning makes the mechanism observable.** For the first time across all fires, a single bio-mechanism ablation cleanly bites: turning off **Wire-A's cluster-bias annotation** shrinks Arm B's delta by +1.13 SD (PASS threshold +1.0). At Qwen32B the same ablation was only +0.56 (suggestive, sub-threshold); Wire-1 (+0.69) and NAc-bias (overshoots past Arm A) still don't attribute. So Wire-A — the substrate-voice annotation that renders NAc cluster reward biases into the prompt — is the channel the reasoning model is reading. The deliberate-reasoning overlay appears to *surface* a mediation that was present but statistically buried in the non-reasoning model.

### The Goldilocks-aware check passes — this is a fair comparison, not a position artifact

The post-Mistral caveat demanded we report Arm A's position first, because the R1 distillation could have moved Arm A out of the zone (a ceiling effect masquerading as a reasoning effect). **It did not.** Arm A *dropped* to 0.259 — even further from the optimal-behavior ceiling than Qwen32B's 0.420, i.e. *more* headroom, deeper in the Goldilocks zone. So the +2.11 SD is a genuine larger-effect reading, not a consequence of A drifting toward saturation. The reasoning overlay made the fresh agent slightly *worse* at the cradle task on its first encounter (more exploration, less beeline-to-safe), which widened the gap substrate could fill — and then the deliberate reasoning chain filled more of it than the non-reasoning model did. Both directions of the Goldilocks logic are satisfied at once.

### Honest caveats (why PARTIAL, not EARNED)

- **Arm C confound (the reason for the investigation gate).** C=0.527 ≈ B=0.566, both well above A=0.259 — the peaceful-prior agent behaves like the fire-failure-prior agent, not like the fresh agent. As with Qwen32B, the effect is "any resumed prior shifts behavior up," not "fire-failure prior specifically." The reasoning overlay does not resolve this; clean fire-specificity attribution still requires the Exp 38 substrate-primary design.
- **Robustness diverges from primary.** The legacy per-action-failure-rate robustness metric FAILs even though the positive-approach primary PASSes. Substrate biases the model toward `warm_self` (more warming) without proportionally reducing `touch` (dangerous contact) — the two halves of "safe engagement" move semi-independently. Investigate before claiming a clean behavioral-safety improvement.
- **Only Wire-A attributes; Wire-1 and NAc-bias do not.** Single-channel attribution is progress over zero-channel, but it is not full mechanistic accounting. The NAc-bias-off ablation *overshoots* past Arm A (0.197 < 0.259) — an opposite-direction effect, not shrinkage.
- **sharp_rock degenerate (4th model).** Zero engagement on every sharp_rock arm; the scenario carries no weight.
- **Corroborating 1/4.** Only time-to-first-warm-self passes (A=0.8 → B=0.2, −1.34 SD); the other three corroborators fail. The primary + the Wire-A ablation + the descriptive count carry the story; the corroborating battery is weak.

### 1.0 framing implication

The reasoning row *strengthens* the Goldilocks story rather than complicating it. The pre-registered worry was bucket R-C (reasoning *drowns* substrate — "smarter models are worse substrate consumers," a bad look for the bio-harness positioning). The data went the other way: **reasoning-trained models are *better* substrate consumers, and the deliberate-reasoning trace makes the carrying mechanism (Wire-A) statistically legible for the first time.** This is a clean 1.1+ research direction — *substrate-aware reasoning models* as a complement to the post-1.0 substrate-primary (Exp 38) direction. The strong "substrate drives action selection via a specific bio-mechanism" claim still STAYS pulled from 1.0 bio-framing (the Arm-C confound gates it), but R1 is the first fire to put a single named mechanism on the board.

## Setup commands for DeepSeek-R1-Distill-Qwen-32B (copy-paste-ready, for reproduction)

To reproduce the DeepSeek fire, execute on the **leader**:

```bash
# 1. Download + add the profile to ~/.config/maxim/profiles.yml
#    HF repo (verify before downloading — quantizer source may have updated):
#    https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF
maxim model add deepseek-r1-distill-qwen-32b \
  --hf bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF:DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf \
  --chat-format chatml \
  --n-ctx 13312 \
  --alias deepseek-r1-32b \
  --alias r1-distill-32b

# 2. Verify the profile is recognized + downloaded
maxim model list | grep deepseek

# 3. Swap the leader to serve DeepSeek (will replace the running Mistral24B server)
tmux kill-session -t maxim-leader 2>/dev/null
pkill -f "maxim --llm"
pkill -f llama_cpp.server
sleep 5
maxim config set llm.profile deepseek-r1-distill-qwen-32b
tmux new -d -s maxim-leader "source .venv/bin/activate && maxim --llm deepseek-r1-distill-qwen-32b"
sleep 120
curl -s -m 10 -H "Authorization: Bearer OYpVQCIwcczWmsBhlm0h2SU9xWXfVP7mKv-f_EqV2q8" http://127.0.0.1:8100/v1/models

# 4. Fire the harness (use same arm/scenario structure as Qwen32B fire — direct A/B)
mkdir -p docs/experiments/data /tmp/exp37_deepseek
tmux new -d -s deepseek "source .venv/bin/activate && PYTHONPATH=src python scripts/benchmark_cross_session.py --scenario both --arms A,B,C,B-wire-a-off,B-wire-1-off,B-nac-bias-off --trials 5 --model deepseek-r1-distill-qwen-32b --sim-max-turns 12 --cost-cap 5 --out docs/experiments/data/37_results_deepseek_r1_distill_qwen_32b.jsonl --workdir /tmp/exp37_deepseek/workdir --cleanup-after-trial 2>&1 | tee /tmp/exp37_deepseek/harness.log"
tmux ls
```

**Sizing notes:** Q4_K_M is ~20GB (comfortable fit on 48GB Mac Mini). Wall-time estimate matches Qwen32B (~25-30 hours). R1 distillation does add inference-time reasoning chains, so per-token latency may be slightly higher than vanilla Qwen32B — could stretch to ~32-35 hours. Watch the per-record duration of the first few trials.

**Reasoning chain note (resolved by the fire):** R1-distilled models emit `<think>...</think>` reasoning chains before the actual response. Maxim's tool-call parser handled these cleanly — no `_llm_unavailable` fallbacks across all 60 records, engagement diversity ~5.8 (comparable to non-reasoning fires), and tool-call rate did NOT regress (the model did not "think out" of using tools). The reasoning chains do raise per-action latency to ~150s (vs ~30s for vanilla Qwen32B), which is why the AUT turn timeout had to be made configurable via `sim.aut_turn_timeout_s` (PR #369) before this fire could run — the hardcoded 30s ceiling was truncating R1's reasoning mid-chain. The `--subsim-timeout-s 10800` harness flag was also added to lift the per-sub-sim wall-clock cap for the longer fire.

## Path forward — sequencing of remaining work

- **2026-06-11 ✓ DONE:** Mistral24B fire complete. Revealed the **Goldilocks-zone / ceiling-effect** finding (A=1.000 SD=0, no headroom for substrate). Reframes the cross-model question from "how big must the model be?" to "is there headroom between priors and optimal behavior?"
- **2026-06-13 ✓ DONE:** DeepSeek-R1-Distill-Qwen-32B fire complete (60/60). **Bucket R-A** — reasoning AMPLIFIES substrate (+2.11 SD vs base +1.43) and surfaces the first clean Wire-A ablation (+1.13 SD). Goldilocks-aware check passes (A=0.259 deeper in zone than base's 0.420). Verdict PARTIAL — investigation gate (Arm C confound persists). Opens the 1.1+ "substrate-aware reasoning models" direction.
- **Conditional (if DeepSeek raised scale-ceiling questions):** R1 did NOT raise scale-ceiling questions — it sat deeper in the Goldilocks zone, not at the ceiling. Llama 3.3 70B Instruct stays a "future direction," not a required follow-on.
- **1.0 ship:** Open-source cross-scale + cross-family + reasoning-axis story is the substantive evidence base. [behavioral_graduation_candidates.md](../plans/behavioral_graduation_candidates.md) row 1b stays PARTIAL with scale-axis + paradigm-axis nuance; cloud comparison framed as 1.1 work.
- **Post-1.0 — prompt-caching refactor:** [prompt_caching_for_cloud_backends.md](../plans/prompt_caching_for_cloud_backends.md) Phase 1+ refactor unlocks cloud fires. Sonnet, GPT-4o, DeepSeek-V3 (cloud, full model) fires execute then.
- **Post-1.0 — Exp 38 substrate-primary:** the principled test of the strong substrate-drives-behavior claim. The mechanism-attribution open question from this exploratory work directly motivates Exp 38's design.

## Cross-references

- [docs/experiments/37_cross_session_graduation.md](37_cross_session_graduation.md) — pre-registration + per-fire verdicts (Qwen14B, Qwen32B; Mistral24B will be added when the fire completes).
- [docs/plans/exp37_cross_model_characterization.md](../plans/exp37_cross_model_characterization.md) — methodology, model lineup, sequencing.
- [docs/plans/behavioral_graduation_candidates.md](../plans/behavioral_graduation_candidates.md) row 1b — the 1.0 graduation gate this evidence informs.
- [docs/plans/prompt_caching_for_cloud_backends.md](../plans/prompt_caching_for_cloud_backends.md) — gating cloud fires; deferred post-1.0.
- [docs/experiments/data/37_results.jsonl](data/37_results.jsonl) — Qwen14B fire records.
- [docs/experiments/data/37_results_qwen32b.jsonl](data/37_results_qwen32b.jsonl) — Qwen32B fire records.
- `docs/experiments/data/37_results_mistral24b.jsonl` — Mistral24B fire records.
- [docs/experiments/data/37_results_r1_distill_qwen_32b.jsonl](data/37_results_r1_distill_qwen_32b.jsonl) — DeepSeek-R1-Distill-Qwen-32B reasoning-axis fire records (60/60).
- [docs/experiments/data/37_results_r1_distill_qwen_32b.md](data/37_results_r1_distill_qwen_32b.md) — DeepSeek-R1 analyzer output (full per-scenario tables).
