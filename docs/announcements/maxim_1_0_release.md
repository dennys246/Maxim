# Maxim 1.0 — announcement copy

Two registers below: a punchy short post (for social / the top of a blog) and a longer technical version (for the full writeup). Companion HTML guide (archived): `docs/archive/html-guides/maxim-1-0-release.html`.

---

## Short version (~250 words)

**Maxim 1.0: we built a bio-inspired cognitive architecture, then spent four months trying to prove it doesn't work.**

Every agent framework claims its memory makes the agent "learn and improve." Almost none of those claims survive a real falsification attempt — because the LLM is so capable that any sensible behavior gets credited to "the architecture."

So we built the experiment that separates them. It's one world with a single knob. An agent is cold to the point of pain, and a fire is its only warmth. In one version the fire is **safe** — warming helps, exactly as the model expects. In the other the hearth **burns** — warming relieves the cold but also hurts, against everything the model "knows." We carry Maxim's bio-substrate (memories, learned reward associations, concept clusters) into a resumed session and ask: does a resumed agent behave differently than a fresh one — and crucially, does it treat the *costly* hearth differently than the *safe* fire? The harm lives only in the world's physics, never in the words the agent sees.

Across three frontier models and a reasoning distill — Sonnet 4.6, GPT-4o, DeepSeek-V3, and DeepSeek-R1 — the answer was consistent: **the LLM prior dominates.** Carried experience, including direct memory of the burn, didn't override it.

That's not a failure to bury — it's the result a 1.0 needs to know before it claims its agents learn. And the *shape* of it is fascinating: on the safe-fire world a real signal *was* detectable, but only in a narrow "Goldilocks" band of model capability (Qwen-32B, +1.43 SD) — and when we went back and falsified the prior at that exact model, the signal **collapsed too** (prior-agreement, not scale, is the gating variable). Reasoning models *amplify* the substrate (the cleanest causal attribution we got — though they amplify the prior, not the correction). And with the LLM removed entirely, the bio-systems genuinely learn — they just fixate without an exploration policy.

Maxim 1.0 ships as a working bio-inspired LLM harness with earned, scoped claims — and a reproducible instrument that mapped exactly where biology matters and where it doesn't. We chose science over marketing. The release is stronger for it.

*Read the full breakdown → [maxim-1-0-release]*

---

## Long version (technical)

### The premise

Maxim is a bio-inspired cognitive architecture for LLM agents — a 5-agent pipeline wired to biological memory systems (hippocampus, ATL, entorhinal cortex, nucleus accumbens, SCN) and a reactive default network. 1.0 is the architecture, complete and production-wired. But shipping it honestly meant answering a question most projects avoid: **does the bio-substrate actually change what the agent does, or is the LLM doing all the work?**

### Two gates

We held 1.0 to two orthogonal gates. **Mechanism validation:** do the individual bio-systems work? **Benchmarking:** does the agent *perform* better because of them? A single test lets either failure hide behind the other. Both are now settled.

### The instrument

A pre-registered, paired fresh-vs-resume benchmark on a developmental "cradle" simulation. Arm A is fresh; Arm B resumes A's substrate; Arm C resumes an unrelated prior (the "any resumed state shifts behavior" control); three ablation arms switch off individual bio-mechanisms. Every metric and threshold was frozen *before* the first run.

The key design move is the **counter-prior**, built as a dilemma. The agent is cold to the point of pain; a fire is its only warmth. In the prior-aligned world the fire is safe — warming helps (but a null there is meaningless: the prior is right whether or not the substrate works). In the counter-prior world the hearth burns: warming relieves the cold *and* breaches the thermal comfort band into pain, against the model's instinct. Because the cold pulls every agent toward the hearth, the test isn't "does it avoid" (avoiding could mean freezing) — it's whether carried memory of the burn makes the agent treat the *costly* hearth differently than the *safe* fire. A matched safe-fire control runs alongside; the verdict is the interaction between them, which cancels the shared cold-need.

### The result: dominance, across four models

Sonnet 4.6, GPT-4o, DeepSeek-V3, and DeepSeek-R1-Distill-Qwen-32B — 60 paired runs each. All four: **dominance.** Carried substrate, including direct cross-session pain at that exact hearth, did not override the wrong `fire → warm` prior.

### Three nuances that make it a finding, not a flop

1. **The Goldilocks zone — and what kills it.** On the prior-aligned benchmark a signal *was* detectable — but only where the model's priors leave headroom. Qwen-14B (below, −0.06 SD), Qwen-32B (in the zone, **+1.43 SD**), Mistral-24B (above — a fresh agent already maxes the task); the band tracks training method as much as scale (a 24B hit a ceiling a 32B didn't). Then we ran the counter-prior at Qwen-32B itself — the one model where the signal lived — and it **collapsed to dominance** (interaction +0.04 SD; the resumed agent warmed the harmful hearth as readily as a fresh one). Same model, same substrate, prior flipped, opposite result: **prior-agreement is the gating variable, not scale.** The signal survives only when the task agrees with what the model already knows.

2. **Reasoning amplifies the substrate.** Distilling reasoning into the same 32B base made the substrate the largest-magnitude, cleanest-attributable effect in the dataset — the first clean single-mechanism ablation across all models. Reasoning models *use* the carried signal. The catch: they amplify the carried *prior*, not the corrective *experience* — confidently wrong against a counter-prior. Same mechanism, two faces. This opens the most actionable 1.1+ direction: substrate-aware reasoning.

3. **Remove the LLM and the substrate is real but immature.** In substrate-primary mode (no LLM in the action path), the bio-systems form clusters, accumulate causal links, and propose drive-conditioned actions — but fixate on the first confident habit instead of exploring. A real early-stage learner that needs a curiosity/exploration policy.

### What 1.0 claims

**Earned:** cross-session memory persistence; entorhinal pattern completion/separation; the pain→reward cascade; causal, ablation-attributable substrate learning in reasoning models; the bio-systems are real and reproducibly measurable.

**Scoped out, honestly:** that the substrate drives adaptive *behavioral improvement* under a strong LLM (it doesn't — the prior dominates); that substrate-primary acts adaptively unaided (it fixates); any "Maxim agents learn and get better" framing.

### The positioning

Maxim 1.0 is a bio-inspired LLM harness with rigorously-scoped, earned claims — and a research instrument that produced a real, quantified result about the boundary between bio-substrate and LLM priors. A foundation, not a verdict, with the next two threads precisely named.
