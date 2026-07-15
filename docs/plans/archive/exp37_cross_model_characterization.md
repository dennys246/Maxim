# Exp 37 — Cross-Model Characterization

**Status:** DRAFT 2026-06-07 (start after PR #347 lands + Phase 1 cloud-dispatch
validates)
**Author:** Denny + Claude
**Purpose:** Exploratory cross-model characterization of the
LLM-prior-dominance finding from the [2026-06-06 Qwen14B Exp 37 fire](../../experiments/37_cross_session_graduation.md).
Six models across scale (14B → 32B → ~200B-class), family (Qwen / Mistral),
and open/closed dimensions. Frames as **exploratory follow-up data informing the
1.0 results doc discussion section**, NOT a 5th pre-reg amendment.
**Out of scope:** Exp 37's metric / pre-reg structure (locked in PRs #332/#334/#336 +
the 2026-06-06 results PR #344); harness-on-leader structural fix (PR #339 +
PR #347's preflight refinement); cloud-dispatch debug (PR #346 + PR #347, which
this plan stacks on).

## Why this exists

The 2026-06-06 Qwen14B Exp 37 fire (60/60 records, verdict
`PARTIAL — investigation gate`) produced the empirical finding that
substrate signal is below the LLM-prior noise floor at N=5 paired trials on the
cradle scenario. The ablation pattern was the diagnostic signature: turning off
Wire-A, Wire-1, or NAc reward bias did NOT shrink Arm B's delta toward Arm A —
2 of 3 ablations OVERSHOT past A, the classic "LLM pretraining doing 95% of
decision-making + bio-mechanisms adding small perturbations" signature.

A single-model finding raises an obvious question: **how much of the
"LLM-prior-dominance" pattern is specific to Qwen14B?** Three orthogonal
factors could each be responsible:

1. **Scale.** Larger models have richer reasoning capacity. Maybe substrate
   signal can compete at 32B+ scales but gets drowned at 14B.
2. **Family / training corpus.** Qwen's pretraining is Chinese-heavy with a
   specific instruction-tuning recipe. Maybe a Mistral-trained model would
   behave differently.
3. **Open vs closed-source alignment.** Closed-source models (Claude, GPT, etc.)
   have more rigorous alignment training. Maybe their priors are different in
   ways that interact with substrate signal.

The 2026-06-07 design isolates each factor:

- **Scale axis:** Qwen14B (done) vs Qwen32B (proposed). Same family, same
  tokenizer, same pretraining recipe — only parameter count differs.
- **Family axis:** Qwen32B vs Mistral-Small-24B-Instruct. Similar scale,
  different family / corpus / tokenizer.
- **Open-vs-closed axis:** All open models vs Claude Sonnet (mainstream
  Western closed), GPT-4o (validates OpenAI key path + second closed anchor),
  DeepSeek-V3 (radically different training corpus, cheap to fire).

If ALL six show the same null result, we have a strong universality claim:
"LLM-prior-dominance is the substrate-effect-masking mechanism across model
scale, family, training corpus, and alignment regime, on the cradle scenario
at N=5." That's a meaningful contribution.

If results DIFFER, the matrix tells us which variable matters — and that
informs the Exp 38 substrate-primary research direction with concrete priors
about what model characteristics matter.

## Experimental design matrix

| Model | Scale | Family | Open/Closed | Where to fire | Wall time | Cost |
|---|---|---|---|---|---|---|
| Qwen2.5-14B-Instruct | 14B | Qwen | open | leader (done) | DONE | $0 |
| Qwen2.5-32B-Instruct | 32B | Qwen | open | leader | ~30 hr | $0 |
| Mistral-Small-24B-Instruct-2501 | 24B | Mistral | open | leader | ~25 hr | $0 |
| Claude Sonnet (claude-sonnet-4-6) | ~200B class | Anthropic | closed | peer (cloud API) | ~3 hr | ~$14 |
| GPT-4o | ~200B class | OpenAI | closed | peer (cloud API) | ~3 hr | ~$14 |
| DeepSeek-V3 (deepseek-chat) | 671B MoE / 37B active | DeepSeek | closed | peer (cloud API) | ~3 hr | ~$2 |

**Scale axis:** Qwen14B (done, null) vs Qwen32B (proposed). Controls family.
**Family axis:** Qwen32B vs Mistral24B. Controls scale roughly.
**Closed-source anchors:** Claude, GPT, DeepSeek — three radically different
training corpuses + alignment regimes, anchoring the "open vs closed" comparison.

**What we deliberately skip:**

- **Gemini-Pro:** mostly redundant with GPT-4o for our finding. Western
  mainstream alignment, similar character. Drop unless a specific reason emerges.
- **Reasoning-tuned models (o1, DeepSeek-R1):** different inference-time
  compute profile. Methodologically interesting but a separate research
  question that doesn't help characterize LLM-prior-dominance specifically.
- **Hosted-open models (Groq, Together, Fireworks):** these host open weights
  on different infra. Adds inference-platform variance without isolating any
  substantive variable.
- **Grok / xAI:** Maxim doesn't have a profile for it currently. Adding one
  is post-1.0 work.

## Bonus value: API key path validation

Beyond the experimental design, this plan **validates the closed-source key
dispatch paths end-to-end for the first time.** PR #337 wired the cloud-dispatch
prefixes for 7 providers; only `claude-*` has been tested (after the cascade of
fixes PR #337 → PR #338 → PR #346 → PR #347). The `gpt-*` and `deepseek-*`
paths are likely to surface the same kinds of bugs we found in the Anthropic
path. Better to find them now than after 1.0 ships.

The "4 bugs in 6 hours" pattern from 2026-06-07's Anthropic validation says
each cloud-dispatch path probably has lurking issues. We'd test each one anyway
before claiming `pip install pymaxim[llm-openai]` actually works.

## Parallelization sequence

The peer (laptop) handles cloud-API calls; the leader (Mac Mini) handles local
inference. Different machines, no resource conflict. **Run cloud + local in
parallel windows.**

### Day 1 — Phase 1 validation + first parallel kickoff

- Morning: **Phase 1 cloud-dispatch validation** (~30 min, peer). After PR #347
  merges, single Arm A Sonnet trial with patience. Confirms the layered fixes
  finally work end-to-end.
- Afternoon (parallel kickoff):
  - **Sonnet full replication** on peer — `--model claude-sonnet`, 65 runs, ~3 hr, ~$14
  - **Qwen32B full replication** on leader — `--model qwen2.5-32b-instruct`, 65 runs, ~30 hr, $0

### Day 2 — Sonnet finishes, fire GPT-4o + continue Qwen32B

- Morning: Sonnet's done (~3 hr in). Inspect records, confirm cross-model
  Anthropic pattern matches or differs from Qwen14B.
- Afternoon (peer parallel with leader):
  - **GPT-4o full replication** on peer — `--model gpt-4o`, 65 runs, ~3 hr, ~$14
  - Qwen32B still running on leader (~24 hr remaining)

### Day 3 — GPT-4o finishes, fire DeepSeek-V3 + continue Qwen32B

- Morning: GPT-4o done. Inspect records, validates the OpenAI key path
  end-to-end (and surfaces bug #5+ if present).
- Afternoon (peer parallel with leader):
  - **DeepSeek-V3 full replication** on peer — `--model deepseek-chat`, 65 runs, ~3 hr, ~$2
  - Qwen32B still running on leader

### Day 4 — Closed-source done, Qwen32B finishes, fire Mistral24B

- Morning: Qwen32B done after ~30 hours (started afternoon Day 1).
- Afternoon: Inspect Qwen32B vs Qwen14B (scale dimension). Kick off
  **Mistral-Small-24B-Instruct-2501** on leader — `--model mistral-small-24b-instruct`,
  65 runs, ~25 hr, $0.

### Day 5 — Mistral24B finishes, cross-model write-up

- Morning: Mistral24B done.
- Afternoon: Cross-model results doc — 6-model comparison table, scale axis
  analysis, family axis analysis, open/closed comparison, interpretation
  framework.

**Total wall time:** ~4-5 days (mostly background).
**Total cost:** ~$30 cloud + $0 local = ~$30.
**Total leader compute used:** ~55 hours (Qwen32B + Mistral24B).
**Total peer compute used:** ~10 hours (cloud calls).

## What success/failure looks like for the multi-model finding

### Outcome A: ALL six models show null behavioral delta (most likely)

The strongest possible cross-model evidence. **Headline:** "LLM-prior-dominance
is universal across model scale (14B → 32B → 200B+), family (Qwen / Mistral /
Anthropic / OpenAI / DeepSeek), training corpus (Western / Chinese-heavy /
mixed), and alignment regime (open instruction-tuned / closed-source RLHF). On
the cradle scenario at N=5 paired trials, substrate signal is below all tested
LLMs' prior noise floors."

This is what we'd hope to see for the 1.0 ship. Strengthens the
"bio-inspired LLM harness" framing with empirical breadth. Directly motivates
the substrate-primary direction (Exp 38 / Oasis) as the principled answer.

### Outcome B: Some open models show signal, all closed models null

Suggests alignment-trained models have stronger priors that overwhelm substrate
more aggressively. Closed-source models specifically resist substrate
influence; open models are more "honest" about admitting substrate-derived
context.

Implication for 1.0: substrate works better with less-aligned models. Reframe
the bio-inspired harness story to specifically endorse open-model use.

### Outcome C: Smaller models show signal, larger don't

The substrate-priors competition is scale-dependent: at 14B, substrate signal
shows through; at 32B+, the LLM's reasoning capacity drowns it. This would
make the Qwen14B fire's signal-not-detectable finding scale-specific.

(Unlikely given the 2026-06-06 ablation pattern, but possible.)

Implication: substrate is more useful for smaller models. Could be a
practical recommendation in the README.

### Outcome D: Mixed signals across models

Most informative diagnostically. The matrix tells us what variable matters.
Generates concrete hypotheses for Exp 38 substrate-primary design (which
characteristics of the substrate vs LLM interaction we'd want to test
without the LLM in the loop).

### Outcome E: Cloud-dispatch path breaks for GPT-4o / DeepSeek

The "find bugs in untested key paths" outcome. Each bug gets a small PR,
validated, the relevant model's replication is then fired after the fix.
**This is also a success outcome — better to find these now than after 1.0 ships.**

## Honest framing — these are NOT pre-reg amendments

The pre-reg originally specified Claude Sonnet as PRIMARY and Qwen14B as
tertiary. The 2026-06-06 Qwen14B fire is the locked PRIMARY EVIDENCE for the
1.0 graduation gate (with Sonnet replication pending).

These additional 4 model runs (Qwen32B, Mistral24B, GPT-4o, DeepSeek-V3) are
**exploratory follow-up data**, not pre-registered replications. The framing
in the cross-model results doc should be:

> "Following the 2026-06-06 Qwen14B fire's locked primary evidence, we ran
> additional exploratory characterization across model scale, family, training
> corpus, and open/closed alignment to test whether the LLM-prior-dominance
> finding generalizes. The pre-reg's verdict (PARTIAL — investigation gate)
> stands on the Qwen14B evidence; the additional models inform the discussion
> section but do not change the graduation status."

This preserves pre-reg credibility (no 5th amendment) while letting the data
strengthen / qualify the headline finding.

## Risks

1. **Each closed-source model may surface a cloud-dispatch bug.** Anthropic
   needed 4 bug fixes. OpenAI / Gemini / DeepSeek likely need at least one or
   two each. The plan budgets for this (each bug is a small PR + small wait).
2. **Mac Mini availability.** Leader compute is the bottleneck for the open
   models. If something else needs the leader during the 55-hour budget, this
   plan gets paused. Coordinate with anything else queued on the leader.
3. **Cost.** ~$30 total feels small but could balloon if a model fires more
   tokens per session (e.g., DeepSeek's reasoning model). Budget cap on each
   harness invocation (--cost-cap 20) is the hard limit; exceeds-cap fires
   abort cleanly.
4. **VRAM on the leader.** 48GB total per the 2026-06-07 correction.
   Qwen32B Q4_K_M ~19GB, Mistral24B Q4_K_M ~14GB, neither approaches the
   limit. No concern.
5. **Time-to-write-up.** Six models = more interpretation work. The
   cross-model results doc could take 1-2 days alone. Budget for this in the
   wall-time plan above.

## Out of scope (explicitly)

- **Substrate-primary measurement (Exp 38 / Oasis):** the principled
  follow-up. This plan characterizes the LLM-prior-dominance finding to
  motivate Exp 38, not replace it.
- **Reasoning-tuned models (o1, DeepSeek-R1, Claude-with-thinking):**
  different inference profile, different research question.
- **Cradle scenario enrichment:** the "electric heater" idea from 2026-06-04.
  Genuine scenario improvement but orthogonal to cross-model characterization.
  1.1+ cradle polish.
- **N=10 or N=20 re-fires:** statistical power scaling is a separate
  consideration. This plan stays at N=5 to keep wall-time and cost manageable.

## Sequencing

1. **PR #347 lands** (preflight cloud-profile exception).
2. **Phase 1 validates** — single Arm A Sonnet trial confirms cloud-dispatch
   end-to-end. ~30 min.
3. **Verify API keys present** — ANTHROPIC_API_KEY (✓ already), OPENAI_API_KEY
   (needs operator add), DEEPSEEK_API_KEY (needs operator add). Without these,
   the relevant runs auto-skip (or block until added).
4. **Verify open-model profiles available** — `maxim --list-models` confirms
   `qwen2.5-32b-instruct` and `mistral-small-24b-instruct` (or equivalent
   profile names) exist. Add missing profiles to `~/.config/maxim/profiles.yml`
   per the v0.9.3 leader-UX profile-management work.
5. **Execute the parallelization sequence** above. Each fire's records land at
   `docs/experiments/data/37_results_<model>.jsonl` to keep them separable.
6. **Run analyzer per model** — six separate analyzer runs, six `.md` outputs.
7. **Write cross-model results doc** — `docs/experiments/37_cross_session_cross_model.md`.
   Six-model comparison table, scale-axis section, family-axis section,
   open/closed-axis section, interpretation framework, implications for Exp 38.
8. **Append cross-model evidence to `behavioral_graduation_candidates.md` row 1b**
   as supplementary data. Row 1b's PARTIAL status doesn't change (locked by the
   2026-06-06 Qwen14B fire), but the additional evidence either reinforces or
   nuances the rationale.

## Cross-references

- [docs/experiments/37_cross_session_graduation.md](../../experiments/37_cross_session_graduation.md)
  — the 2026-06-06 Qwen14B fire results + LLM-prior-dominance interpretation.
- [docs/plans/behavioral_graduation_candidates.md](../behavioral_graduation_candidates.md)
  — row 1 split (1a EARNED memory persistence, 1b PARTIAL behavioral delta).
  This plan's cross-model data extends 1b's rationale without changing its status.
- [docs/plans/archive/exp37_metric_pivot.md](exp37_metric_pivot.md) — the metric pivot
  whose primary metric (positive_approach_engagement_fraction) is what gets
  measured across all 6 models.
- [docs/plans/archive/exp37_sd_shift.md](exp37_sd_shift.md) — the SD-shift test all
  models will be evaluated against.
- [docs/plans/archive/cloud_dispatch_debug.md](cloud_dispatch_debug.md) — Phase 1
  validation that has to pass before this plan starts.
- CLAUDE.md "Environment Variables" — canonical reference for the cloud
  provider API key env vars.
- PRs #337 / #338 / #346 / #347 — the layered cloud-dispatch fixes that
  unblock cross-model firing.
