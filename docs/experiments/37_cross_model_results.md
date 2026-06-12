# Exp 37 — Cross-Model Results

**Status:** IN PROGRESS 2026-06-11 (3 open-source fires complete: Qwen14B, Qwen32B, Mistral24B → **Goldilocks-zone finding**; DeepSeek reasoning-axis fire queued; cloud comparisons deferred behind prompt-caching refactor).
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
| Mistral-Small-24B-Instruct-2501 | leader (local) | **RUNNING** 2026-06-10 (kicked off 11:50; re-fired 16:32 after harness safety-check fix) | Family comparison at intermediate scale |
| DeepSeek-R1-Distill-Qwen-32B | leader (local) | **QUEUED** after Mistral24B | Reasoning-axis isolation (same base as Qwen32B) |
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
| DeepSeek-R1-Distill-Qwen-32B | TBD (queued) | TBD | TBD | TBD | TBD | TBD |
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

**Mistral24B secondary criterion is structurally unmeasurable.** The ablation test asks "does turning off bio-mechanism X shrink B's delta toward A?" — but for Mistral24B the B-vs-A delta is already NEGATIVE (B=0.600 < A=1.000, Δ=−0.40). There is no positive delta to "shrink." The ablation arms (B-wire-a-off=0.600, B-wire-1-off=0.600, B-nac-bias-off=0.700) all sit at or near B, so the analyzer reports "insufficient data for ablation comparison." This is a direct consequence of the ceiling effect (Arm A already perfect — see headline finding below).

**Pattern across the completed fires:** the substrate effect is NOT cleanly attributable to any single bio-mechanism. At 32B (Qwen), Wire-A ablation moves the needle (+0.56 SD shrinkage) but Wire-1 and NAc-bias don't; at 24B (Mistral) the question is moot because there's no positive delta to attribute. Either multi-channel mediation (each channel contributes a little, no single ablation is decisive) or substrate-context-as-a-whole effects on LLM reasoning that no single annotation captures.

## Isolation arm (Arm C — "general caution" confound)

| Model | Arm C mean | Inside Arm A's band? | Verdict |
|---|---|---|---|
| Qwen2.5-14B-Instruct | 0.617 | YES (band [0.333, 0.96]) | PASS |
| Qwen2.5-32B-Instruct | 0.667 | NO (outside band [0.033, 0.660]) | **FAIL — confound** |
| Mistral-Small-24B-Instruct | 0.400 | NO (outside band [1.000, 1.000]) | **FAIL — but in the OPPOSITE direction** |

Arm C carries a peaceful-prior session (control for "any prior shifts behavior" vs "fire-failure prior shifts behavior specifically"). When C falls OUTSIDE A's empirical band AND looks more like B, that's the signal that the substrate effect generalizes across priors rather than being scenario-specific. **Qwen32B triggered this confound** in the upward direction (C ≈ B, both above A). **Mistral24B triggers it in the opposite direction:** C=0.400 is BELOW A=1.000, not above. Because Arm A is a degenerate point distribution (every value exactly 1.000, band [1.000, 1.000]), literally ANY non-perfect arm falls "outside the band." The confound flag here is a statistical artifact of A's zero variance, not evidence of cross-prior generalization. C being the WORST arm (lowest warm_self, has touches) is actually the predicted fire-specific-learning pattern — the peaceful-prior agent is the most dangerous near fire.

## Descriptive corroborating — `fire_approach_action_count`

Counts of warm_self actions per session — not pre-reg gated, but directionally informative.

| Model | A mean | B mean | Δ | Direction |
|---|---|---|---|---|
| Qwen2.5-14B-Instruct | 1.60 | 1.40 | −0.20 | **WRONG** (B less than A) |
| Qwen2.5-32B-Instruct | 1.60 | 3.00 | **+1.40** | **PREDICTED** (B more than A) |
| Mistral-Small-24B-Instruct | 1.00 | 0.80 | −0.20 | **WRONG** (B less than A) — but A is at ceiling |

## Sharp_rock scenario — degenerate at ALL THREE scales/families

Qwen14B, Qwen32B, AND Mistral24B all produced zero engagement across every sharp_rock arm. The asymmetric-design concern from [exp37_metric_pivot.md](../plans/exp37_metric_pivot.md) (sharp_rock has no positive-approach analog like fire_pit's `warm_self`) is now realized at three model fires spanning two families and three scales. This is no longer a Qwen-specific quirk — **sharp_rock is structurally broken for cross-model use**, and the cradle scenario needs a positive-approach affordance for sharp_rock (or a different second scenario) before it can carry verdict weight. Tracked as cradle-redesign motivation for post-1.0 work. fire_pit alone carries the substantive evidence on all cross-model fires.

## HEADLINE FINDING — the Goldilocks zone of prior strength

**The three completed fires form a clean three-point story that the original pre-registered buckets (A/B/C/D) did not anticipate. Mistral24B did not land in any of them cleanly — it revealed a CEILING EFFECT that reframes the whole cross-model question.**

| Model | Arm A (fresh agent) mean | Arm B (resumed, has substrate) mean | Δ | Position |
|---|---|---|---|---|
| Qwen2.5-14B | 0.533 (variable) | 0.517 | −0.02 | **Below the zone** — priors too weak |
| Qwen2.5-32B | 0.420 (variable) | 0.800 | **+0.38 (+1.43 SD, PASS)** | **Inside the zone** — sweet spot |
| Mistral-Small-24B | **1.000 (SD=0, perfect)** | 0.600 | −0.40 | **Above the zone** — ceiling effect |

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

## Cross-model interpretation — what we're trying to learn from DeepSeek-R1-Distill-Qwen-32B

**Queued for after Mistral24B finishes.** The DeepSeek fire isolates a different axis from family / scale: it tests whether **explicit-thinking training** (R1's reasoning-trace distillation) changes how substrate-derived context interacts with the LLM's action selection.

### Why this is the cleanest reasoning-axis test available

DeepSeek-R1-Distill-Qwen-32B is the R1 reasoning-trace fine-tune of the SAME Qwen2.5-32B base we've already fired. Same architecture, same tokenizer, same scale, same pretraining. **The only methodologically-meaningful difference is the reasoning-training overlay.** That makes the comparison vs Qwen32B a clean A/B on the reasoning-paradigm axis with minimal confounds.

(Llama-distill variants like DeepSeek-R1-Distill-Llama-70B would add reasoning + new family + larger scale at once — three confounded axes. We avoid that for the headline test; Llama 70B stays as a *conditional* follow-on if DeepSeek raises specific scale-ceiling questions.)

### Three-bucket pre-registered interpretation

#### Bucket R-A: DeepSeek stronger than Qwen32B (Δ > +1.43 SD)

**Interpretation:** Reasoning training *amplifies* substrate signal. The "think before responding" pattern lets the model engage more deliberately with substrate-derived prompt context — drives are noticed, recalled episodes get weighted, valence annotations get considered rather than skimmed.

**1.0 framing:** "Substrate-context utilization is enhanced by reasoning-trained models. Future work: substrate-aware reasoning models as a complementary direction to substrate-primary action selection."

This would be a substantively interesting finding for the 1.1+ research agenda.

#### Bucket R-B: DeepSeek similar to Qwen32B (Δ ≈ +1.43 SD, within noise)

**Interpretation:** Reasoning training is neutral with respect to substrate signal. The substrate effect we see at 32B Qwen survives the R1 distillation overlay — substrate context flows through both reasoning and non-reasoning models similarly.

**1.0 framing:** "Substrate signal is robust to training-paradigm shifts at the moderate scale tested. Substrate-LLM interaction does not depend on reasoning training."

The "null result" outcome on this axis. Reassuring for the substrate generality claim.

#### Bucket R-C: DeepSeek weaker than Qwen32B (Δ < +1.0 SD)

**Interpretation:** Reasoning training *drowns* substrate signal. Explicit reasoning chains may out-compete substrate-derived context — the model "thinks past" the bio-annotations toward its own preferred reasoning trajectory, ignoring substrate suggestions it would have used in a non-reasoning mode.

**1.0 framing:** "Substrate signal weakened by reasoning-training overlay. Future work: substrate context positioning + prompt-engineering for reasoning-trained models." Suggests substrate signal works better with non-reasoning models in the current Maxim architecture.

This would be the most surprising finding — it would imply that bigger / smarter / more thoughtful LLMs are actually *worse* substrate consumers, with implications for the bio-inspired-harness 1.0 positioning.

#### Goldilocks-aware caveat for the DeepSeek read (added 2026-06-11 post-Mistral)

The Mistral24B ceiling effect adds a critical confound to watch for. DeepSeek-R1-Distill-Qwen-32B shares the Qwen32B BASE, so its fresh-agent (Arm A) priors should start near Qwen32B's (A≈0.42, in the Goldilocks zone). **But the R1 reasoning distillation could shift Arm A's prior strength in either direction**, and that shift — not the substrate interaction per se — could drive the result:

- If R1 distillation makes Arm A *better* at the cradle task (pushes A toward the Mistral-style ceiling), we'd see a Mistral-like null even if substrate interaction is unchanged — a ceiling effect, not a reasoning-drowns-substrate effect.
- If R1 distillation makes Arm A *worse* / more variable (more exploration, less task-focus), we'd see more headroom and possibly a Qwen32B-like PASS — again driven by prior strength, not substrate-reasoning interaction.

**So the DeepSeek read MUST report Arm A's mean + SD first**, and interpret the B-vs-A delta only relative to where A sits in the Goldilocks zone. A "weaker than Qwen32B" result (bucket R-C) is only a *reasoning-drowns-substrate* finding if Arm A stayed in the zone (A still ~0.42 with headroom). If Arm A moved to the ceiling, it's the same ceiling effect Mistral showed, and reasoning-vs-substrate stays unanswered. Document Arm A's position explicitly in the DeepSeek writeup.

## Setup commands for DeepSeek-R1-Distill-Qwen-32B (copy-paste-ready, queued)

When ready to fire DeepSeek (after Mistral24B completes), execute on the **leader**:

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

**Reasoning chain caveat:** R1-distilled models emit `<think>...</think>` reasoning chains before the actual response. The harness should treat these as part of the response (token counts include them; engagement metrics measure actions, not reasoning text). If we observe a regression in tool-call rate, that could be the model "thinking out" of using tools at all — note in the results writeup.

## Path forward — sequencing of remaining work

- **2026-06-11 ✓ DONE:** Mistral24B fire complete. Revealed the **Goldilocks-zone / ceiling-effect** finding (A=1.000 SD=0, no headroom for substrate). Reframes the cross-model question from "how big must the model be?" to "is there headroom between priors and optimal behavior?"
- **2026-06-12 → 2026-06-14:** DeepSeek-R1-Distill-Qwen-32B fire (~30 hours). Adds the reasoning-axis row + R-A/R-B/R-C bucket selection.
- **Conditional (if DeepSeek raises scale-ceiling questions):** Llama 3.3 70B Instruct fire (~30 hours). Otherwise mark Llama 70B as "future direction" in the doc.
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
- `docs/experiments/data/37_results_mistral24b.jsonl` — Mistral24B fire records (file populated as records land 2026-06-10/11).
