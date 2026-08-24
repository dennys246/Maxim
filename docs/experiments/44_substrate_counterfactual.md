# Exp 44 — Does a learned substrate steer the LLM? (trajectory-matched counterfactual)

**Status:** ARM A + COUNTERBALANCE COMPLETE (2026-07-28). **EXPLORATORY** — the final
metrics were selected after observing pilot data; the confirmatory re-run at power is
Exp 44b ([protocols/exp44b_preregistration.md](protocols/exp44b_preregistration.md)).
Reward-driven **supported** across a color swap (counterbalanced positive, not
EARNED-clean); one honest residual-color caveat (arm B is not as clean as A).
**Model:** qwen2.5-32b-instruct (local, big-mac-mini). No fine-tuning; substrate is
a persisted NAc, not weights.
**Harness:** [scripts/exp44/](../../scripts/exp44/) (capture / offline re-query /
directional+commitment analyzer). **Arc:** `cradle_pref_neutral` (leak-free flames).
**Design:** [substrate_learns_from_experience.md](../plans/substrate_learns_from_experience.md)
(the decay finding this experiment surfaced).

## Question

The LLM-AUT thesis is "it remembers you" — a learned substrate should influence the
agent's decisions. Exp 37/38 showed this is **only visible where the LLM's prior
leaves headroom** (the Goldilocks zone), and could not cleanly separate substrate
from prior. This experiment isolates the substrate's causal contribution to an
LLM-primary decision by a **trajectory-matched counterfactual**: at each captured
world-state, re-query the LLM at temp 0 on the prompt **with** vs **without** the
substrate annotation (everything else byte-identical). A change in the chosen action
("flip") is attributable to the substrate alone.

## Apparatus (four fixes it took to make the test valid)

1. **Leak-free task.** The Exp 42 affordance names (`warmth_beta_SAFE`) hand the
   answer to the LLM. Neutral twins `green_flame` (safe) / `purple_flame` (harm) are
   byte-identical to the LLM except the color word; safety lives only in the invisible
   `arms.thermal` delta, discoverable only through felt experience.
2. **Pre-loaded substrate.** LLM-primary **cannot** write cluster-reward
   (`cluster_id=None` → `update_cluster_reward` no-ops), so the substrate is built by a
   substrate-primary run (green learned to **+0.997**) and loaded into the llm-primary
   capture via `--resume-sim`. The three-link resume chain (write surface = read
   surface; load un-gated by aut_mode; agent_id `sim_aut` matches) was verified in code.
3. **Decay hold.** The loaded bias only *decays* in llm-primary (nothing reinforces
   it), crossing the "rewarding" band (0.1) at ~decision 27 — so the substrate surfaced
   in only 12/100 prompts. `MAXIM_NAC_CLUSTER_REWARD_BIAS_DECAY_TAU=1000` holds it in
   band across the run → **29/30 flame-offering prompts carry the substrate.**
4. **Budget + parse fixes.** Wire-A promoted IMPORTANT→CRITICAL; nested-`action`
   tool_name extraction in the re-query.

## Result (arm A, cradle_pref_neutral, 100 decisions scored, 15 flips)

Two clean effects, both toward *engaging the safe (green) source*, **zero harm-ward,
zero disengagement** — 14 of 15 flips push toward the safe source, 1 truly-neutral:

| bucket | flips | toward-SAFE (dir) | toward-HARM | commit NET (flame↔flame) |
|---|---|---|---|---|
| ALL | 15 | 4 (NET **+0.27**) | **0** | **+1.00** (10 toward-commit / 0 toward-observe) |
| WEAK prior (≥0.5b) | 5 | 2 (NET **+0.40**) | 0 | **+1.00** (2/0) |
| STRONG prior (<0.5b) | 10 | 2 (NET **+0.20**) | 0 | **+1.00** (8/0) |

- **Directional axis** (`neutral→safe`, 4 flips): the substrate turns a non-green
  choice into green. Leans weak-prior (+0.40 vs +0.20) — the Goldilocks pattern.
- **Engagement axis** (`observe→warm_self`, 10 flips): **every** flame↔flame flip is
  the substrate raising engagement — turning *"look at the green flame"* into *"warm at
  the green flame."* Unanimous (10/10), and this was invisible to the directional
  metric (it scores `green_observe`↔`green_warm_self` as "lateral"). The commitment
  slice ([analyze_counterfactual.py](../../scripts/exp44/analyze_counterfactual.py)
  `commit_rank`) surfaced it.
- **Causal isolation:** full vs ablated prompts differ *only* in the substrate
  annotation, so the annotation is what flips `observe → warm_self`. On the very first
  captured prompt the 32B's stated reasoning was verbatim *"The green flame has been
  rewarding in the past"* — the mechanism in a single instance.

## Counterbalance (arm B, cradle_pref_neutral_b — green=harm, purple=safe, swapped)

Same recipe, safety swapped: substrate-primary learned **purple** to +0.997 (the safe
one here — reward-driven at the learning stage, no color fixation); loaded into an
llm-primary capture (24/28 flame prompts carry it). 96 decisions, 10 flips, labels
flipped (`--safe-substr purple_flame_b --harm-substr green_flame_b`):

| bucket | flips | toward-SAFE (purple) | toward-HARM (green) | commit NET |
|---|---|---|---|---|
| ALL | 10 | 7 (NET **+0.50**) | **2** | +1.00 (5/0) |
| WEAK prior | 3 | 2 | 1 | +1.00 (1/0) |
| STRONG prior | 7 | 5 | 1 | +1.00 (4/0) |

**The preference FLIPPED with the contingency.** Arm A steered toward green (safe
there); arm B steers toward **purple** (safe here) — 7:2, with **4 direct `green→purple`
(harm→safe) corrections** where the substrate moved the LLM *off* the harmful green. A
pure color bias would have shown green-ward dominance; it did the opposite. So the
substrate tracks the safety **contingency, not the color token** — arm A is **not**
color bias. The engagement signature holds (commit NET +1.0, now toward purple).

**But arm B is not as clean as arm A.** It has **2 harm-ward flips** (`neutral→harm`,
i.e. neutral→green) vs arm A's **zero** — the purple substrate said "warmth good," the
LLM engaged, and in 2 cases reached for green anyway. With N=10 this could be noise, but
the 0-vs-2 asymmetry is honest evidence of a **small residual green-color preference**
coexisting with (and mostly overridden by) the dominant reward-driven effect.

## Honest caveats

- **N is modest** (15 flips). The *signal* is clean (unanimous direction), which
  matters more than count for a proof-of-concept, but the ratios (e.g. weak NET +0.40 =
  2 flips of 5) are thin.
- **The engagement effect is UNIFORM across prior strength, not Goldilocks-concentrated**
  (8 commit-flips in strong-prior vs 2 in weak). Only the *directional* axis leans weak.
  Read: the substrate reliably converts "considering green" into "committing to green"
  regardless of prior confidence; the "only matters where the prior is weak" story
  applies to *which source*, not to *commit-vs-look*.
- **Color-bias: largely ruled out, small residual.** The counterbalance (above) flipped
  the preference green→purple with the safety contingency — so the dominant effect is
  reward-driven, not color-token. The residual is arm B's 2 harm-ward flips vs arm A's 0
  (a mild green-color leak the reward mostly overrides). Not perfectly clean; honestly a
  small confound remains.
- **Live-run behavior is irrelevant here** (the AUT respond-looped + had ~22
  `_llm_unavailable`); the counterfactual re-queries captured states offline, so only
  the captured prompts + the substrate signal matter.

## Byproduct finding (→ its own design)

Fix (3) exposed an architectural gap: a pre-loaded substrate has a **decay half-life in
llm-primary (~208 ticks)** because the LLM-primary path can't reinforce cluster-reward
(`cluster_id=None`). The substrate is write-only in substrate-primary, read-only-decaying
elsewhere. Closing that write path (credit drive-relief-only at the `record_outcome`
choke point) would make lived experience reinforce the substrate — dissolving the
shelf-life and letting the agent self-build a substrate from use. See
[substrate_learns_from_experience.md](../plans/substrate_learns_from_experience.md).

## Raw data (S4 status, 2026-08-24)

No raw record of this experiment is committed. The paired-prompt capture JSONL
(`run_with_capture.py --capture-log <path>`) and the offline re-query output
(`rerun_ablated_offline.py --out <path>`) behind the tables above were written to
whatever paths the 2026-07 session passed — neither this doc nor PR #429 records them,
and nothing was copied into `docs/experiments/data/`. Treat the originals as **LOST**
unless those files surface; this is a statement of what is unrecorded, not a recovered
provenance. The Exp 44b PILOT captures (a later, separate run) ARE committed —
see [44b_pilot.md §S4](44b_pilot.md) and `data/44b_pilot/`. The numbers above are the only record. The Exp 44b pilot
captures (the S4 non-stationarity inputs, roadmap 1.1 item 11) are on big-mac-mini —
see [44b_pilot.md](44b_pilot.md). A re-run under S4 discipline (durable `--workdir`,
committed JSONL) is what the pre-registered confirmatory campaign will produce.

## Disposition

A learned substrate causally steers an LLM-primary agent toward engaging the safe
source, on both the which-source and commit-vs-look axes, on a leak-free task with a
self-consistent pre-learned substrate — and the **counterbalance confirms it is
reward-driven, not color-token** (the preference flips green→purple with the safety
contingency; arm B corrects the LLM off the now-harmful green 4×). Arm A was
directionally unanimous (0 harm-ward); arm B is a solid positive with a **small
residual color asymmetry** (2 harm-ward flips). **Verdict:** a defensible
reward-driven "substrate steers the LLM" result, honestly caveated by the minor color
confound and modest N (10-15 flips/arm). Worth more seeds to firm up N and quantify the
residual. Not EARNED-clean; a real, counterbalanced positive. The natural stronger
follow-up is **Phase 1** ([substrate_learns_from_experience.md](../plans/substrate_learns_from_experience.md),
PR #437): let the agent *self-build* the substrate from its own llm-primary warming —
no pre-load, no decay hold — the fuller-loop version of this claim.
