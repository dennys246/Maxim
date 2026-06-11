# Exp 37 — Cross-Model Results

**Status:** IN PROGRESS 2026-06-10 (Mistral24B fire running; cloud comparisons deferred behind prompt-caching refactor).
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
| Mistral-Small-24B-Instruct | TBD | TBD | TBD | TBD | TBD | TBD |
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
| Mistral-Small-24B-Instruct | TBD | TBD | TBD | TBD |

**Pattern across both completed fires:** the substrate effect is NOT cleanly attributable to any single bio-mechanism. At 32B, Wire-A ablation moves the needle (+0.56 SD shrinkage) but Wire-1 and NAc-bias don't. Either multi-channel mediation (each channel contributes a little, no single ablation is decisive) or substrate-context-as-a-whole effects on LLM reasoning that no single annotation captures. Mistral24B's pattern will help disambiguate — if it ALSO shows Wire-A>others, multi-channel-with-Wire-A-dominant becomes the leading interpretation.

## Isolation arm (Arm C — "general caution" confound)

| Model | Arm C mean | Inside Arm A's band? | Verdict |
|---|---|---|---|
| Qwen2.5-14B-Instruct | 0.617 | YES (band [0.333, 0.96]) | PASS |
| Qwen2.5-32B-Instruct | 0.667 | NO (outside band [0.033, 0.660]) | **FAIL — confound** |
| Mistral-Small-24B-Instruct | TBD | TBD | TBD |

Arm C carries a peaceful-prior session (control for "any prior shifts behavior" vs "fire-failure prior shifts behavior specifically"). When C falls OUTSIDE A's empirical band AND looks more like B, that's the signal that the substrate effect generalizes across priors rather than being scenario-specific. **Qwen32B triggered this confound** — substrate effect at 32B is broader than fire-specific. Watch whether Mistral24B does too.

## Descriptive corroborating — `fire_approach_action_count`

Counts of warm_self actions per session — not pre-reg gated, but directionally informative.

| Model | A mean | B mean | Δ | Direction |
|---|---|---|---|---|
| Qwen2.5-14B-Instruct | 1.60 | 1.40 | −0.20 | **WRONG** (B less than A) |
| Qwen2.5-32B-Instruct | 1.60 | 3.00 | **+1.40** | **PREDICTED** (B more than A) |
| Mistral-Small-24B-Instruct | TBD | TBD | TBD | TBD |

## Sharp_rock scenario — degenerate at both scales

Both Qwen14B and Qwen32B produced zero engagement across all sharp_rock arms. The asymmetric-design concern from [exp37_metric_pivot.md](../plans/exp37_metric_pivot.md) (sharp_rock has no positive-approach analog like fire_pit's `warm_self`) realized at both completed scales. Sharp_rock contributes no information to the verdict at either model size; fire_pit alone carries the substantive evidence on cross-model fires too. **Watch whether Mistral24B sharp_rock engages differently** — if it does, that would suggest sharp_rock's degeneracy is Qwen-specific.

## Cross-model interpretation — what we're trying to learn from Mistral24B

The Mistral24B fire's results will collapse into one of four buckets, each carrying a distinct 1.0 framing implication:

### Bucket A: Mistral24B PASSES primary (Δ ≥ +1.0 SD) similar to Qwen32B

**Interpretation:** Substrate-transfer signal is detectable across model families at moderate (24B+) scale. The Qwen32B result is not Qwen-specific.

**1.0 framing:** "Substrate carries cross-session memory at all scales tested. The carried memory measurably shifts behavior at ≥24B scale across open-source families. Specific bio-mechanism attribution remains unclear — substantive Exp 38 / post-1.0 research question."

This is the strongest 1.0 narrative. Cross-family confirmation at the same scale band as Qwen32B.

### Bucket B: Mistral24B INTERMEDIATE (0 < Δ < +1.0 SD)

**Interpretation:** Smooth scaling with parameter count. There's a continuous spectrum between "null at 14B" and "PASS at 32B," with a detection threshold somewhere in the 24-30B range.

**1.0 framing:** "Substrate signal scales monotonically with model capacity. The behavioral threshold for SD-shift detectability sits at approximately ${cutoff}B parameters."

Cleanest theoretical interpretation. Suggests we'd see PASS at larger Mistral / Mixtral models.

### Bucket C: Mistral24B NULL (Δ < +1.0 SD or wrong direction)

**Interpretation:** Either substrate-transfer at scale is Qwen-family-specific OR there's a sharp phase transition between 24B and 32B (likely around the architectural changes that happen with bigger models).

**1.0 framing:** "Substrate signal detectable at 32B Qwen specifically; family-specific or sharp phase transition. Cross-family confirmation at scale remains open question."

Weakens the headline claim. The 32B finding becomes more tentative — possibly architecture-specific rather than capability-driven.

### Bucket D: Mistral24B STRONGER than Qwen32B (Δ > +1.43 SD)

**Interpretation:** Mistral's training (less aggressive instruction tuning? different alignment regime?) allows substrate signal through more readily than Qwen at the same scale.

**1.0 framing:** "Substrate signal strength is family-modulated, with Mistral showing stronger expression than Qwen at similar scale. Open question: what training characteristics correlate?"

Less likely but methodologically interesting if it happens.

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

- **2026-06-10/11:** Mistral24B fire completes. This results doc gets filled in with real numbers and the Mistral interpretation locked (bucket A/B/C/D).
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
