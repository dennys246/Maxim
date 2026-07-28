# pymaxim.bio — website update suggestions (2026-07-28)

Spec for the **maxim-web** repo (Astro/Starlight → pymaxim.bio). This is *what to say
and how to frame it*, not the Astro code. Everything here is discovery-only + honest —
the site's credibility is the product's credibility, so under-claim before over-claim.

## The one new headline the site can now honestly make

> **A learned substrate changes what the agent decides — and it's the reward, not a coincidence.**

Backed by **Exp 44** ([experiment doc](../experiments/44_substrate_counterfactual.md)):
a trajectory-matched counterfactual on a *leak-free* task (two flames whose names don't
reveal which is safe) showed the substrate steers the LLM toward *engaging the safe
source* — and a **counterbalance** (swap which flame is safe) made the preference **flip
with the contingency**, ruling out a color/token bias. This is the confound-reduced,
"it's actually the memory" version of the thesis — much stronger for AI-engineer
skeptics than a bare "it remembers you."

**Honesty guardrails for the copy (do NOT drop these):**
- Say "influences / steers," not "controls." The effect is real but modest (10-15
  decision-flips per arm) and concentrates where the LLM's prior is weak (Goldilocks).
- Mention the residual: the counterbalance arm had a *small* color asymmetry (2
  harm-ward flips vs 0 in the other arm). A site that names its own caveat reads as
  science, not marketing.
- No fine-tuning: the substrate is a persisted NAc, not weights. That's a *selling
  point* — behavior change without touching the model.

## Suggested page / section changes

1. **Home hero** — refresh the one-liner around the above. Pair it with the two
   already-strong hardware/embodiment claims so the story is "learns → on real hardware
   → and it steers a frontier LLM."

2. **A "Proof" / "Evidence" page** (new or expand existing) — a short, honest gallery of
   the graduated/validated results, each one sentence + a link. Pull from the
   [graduation-candidates](../plans/behavioral_graduation_candidates.md) EARNED/POSITIVE rows:
   - **Real-hardware sensorimotor learning** (Reachy Mini orients to sound, direction +
     magnitude, no LLM in the action path) — Exp 45/45e.
   - **Operant orienting** (a caregiver teaches a driveless infant to orient) — Exp 48.
   - **Substrate-primary safe-vs-harm discrimination** — Exp 42 GRADUATE.
   - **Substrate steers the LLM (counterbalanced)** — Exp 44 (the new one).
   Each card: the claim, the honest caveat, the link. This page is the credibility spine.

3. **A "Vision → next" note** — the substrate now *learns from lived experience* in
   llm-primary (Phase 1, #437): the agent self-builds its substrate from use, so it stays
   fresh instead of only being pre-loaded. Frame as the direction, not a shipped headline
   claim yet (the behavioral self-build validation is future work).

4. **Local-first framing stays central** (already the position): Console is
   127.0.0.1-only, the tunnel carries the *resource* not the UI, Oasis contribution is a
   *local decision*. The website is discovery, not a service. Don't let the new results
   tempt a "sign up" / hosted framing — that contradicts the whole architecture.

## What NOT to put on the site yet

- Oasis as a live/available feature — it's the *next build*, not shipped. "Coming: peer
  substrate sharing (Oasis)" at most, clearly future.
- Any "the agent has memory of you across sessions that changes its behavior" claim
  without the Goldilocks + counterbalance caveats — Exp 37/38 showed the naive version
  over-claims; Exp 44 is the honest version and should be cited *with* its limits.
- Benchmark/leaderboard-style numbers — the results are mechanism demonstrations at
  modest N, not competitive benchmarks. Presenting them as benchmarks invites the wrong
  scrutiny.

## Cross-repo note

The engine facts these claims rest on live in this repo's `docs/experiments/` +
`docs/plans/behavioral_graduation_candidates.md`. When a claim on the site changes,
update it here first (the experiment docs are the source of truth), then the site — same
direction as the `maxim serve` OpenAPI contract flow.
