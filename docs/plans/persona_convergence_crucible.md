# Persona Convergence Crucible

**Status:** living doc, ongoing practice
**Begins:** post-1.0 (after [persona_cleanup_and_mode_transition.md](persona_cleanup_and_mode_transition.md) and [bio_emergent_persona_foundations.md](bio_emergent_persona_foundations.md) ship)
**Companion living docs:** [behavioral_convergence_practice.md](behavioral_convergence_practice.md) ("does the agent get better across sessions?"), [memory_consolidation_practice.md](memory_consolidation_practice.md) ("does sleep replay actually consolidate?")

## What this is

A long-running, deliberately-shaped attempt to develop persona-like behavior in a Maxim agent through sustained, consequential lived experience — without prompt injection.

Each iteration is a **Roy experiment**, named after the Rick & Morty in-game life simulator *Roy: A Life Well Lived* — a complete subjective lifetime spent inside a constructed environment, taken seriously as a life despite being a constructed one. Rick walks out of the game shaped by Roy's existence; the Maxim that lives Roy-N walks out with NAc, Hippocampus, ATL, and embodiment state shaped by the lifetime.

This is **not a PoC with pass/fail criteria.** It is a living practice, like behavioral_convergence_practice.md and memory_consolidation_practice.md. The iteration log accretes findings about which crucible mechanics, which bio-system wires, and which experiential regimes are load-bearing for persona convergence. Some Roys take. Some don't. Both teach.

## Framing rule: the well-attempted persona

Each Roy is **an attempt**, evaluated honestly on its own terms. An attempt where nothing took is not a failure — it is a finding about which wires or which crucible mechanics need refinement. The doc records what was attempted, what stuck in the substrate, what didn't, and what we changed for the next attempt.

There is no "did persona emerge yes/no" rubric. Roy iterations are characterized along multiple axes: substrate divergence, behavioral divergence, cross-session persistence, generalization to novel stimuli. An attempt that produces substrate divergence without behavioral divergence is informative — tells us a wire between substrate and action selection is missing or weak. An attempt that produces behavioral divergence without substrate divergence is also informative — tells us the LLM is improvising and we haven't actually grounded the persona in lived state.

## Why this exists separately from the foundations plan

The bio-system wires in [bio_emergent_persona_foundations.md](bio_emergent_persona_foundations.md) ship in 1.0 because each closes an architectural gap that's worth closing on its own merits. They earn their place without the persona claim. Whether the wires are *sufficient* for persona convergence is a downstream research question that this doc owns — and the answer is unknown.

If we discover that even with all wires in place persona doesn't converge, that finding lives here, and it informs whether 1.1+ needs deferred Wires 4 and 5, or whether something fundamentally bigger (action-ranking pre-filter, multi-agent attribution refactor, decision-time substrate query that doesn't exist yet) is missing.

## Methodology

### Three-arm comparison
Every Roy iteration runs the same held-out test scenario across three arms:

| Arm | Substrate priming | System prompt at test |
|---|---|---|
| **Roy-N-A** | Full Roy lifetime (substrate-primed for target persona) | Neutral |
| **Roy-N-B** | Blank substrate (fresh agent) | Prompted to enact target persona (the old persona-injection style) |
| **Roy-N-C** | Blank substrate | Neutral |

The interesting questions are not "does Roy-N-A look like the target persona at test time" — both A and B will, because the LLM is good at role-play. The interesting questions are:

1. **Does Roy-N-A diverge from Roy-N-B at the substrate level?** (NAc percept valences, reward biases, hippocampal valence distributions, ATL semantic structure.) If yes, substrate is doing real work even when behavior looks similar.
2. **Does Roy-N-A diverge from Roy-N-B on edge cases the prompt didn't cover?** Novel stimuli, generalization tests. The substrate-grounded persona should generalize from learned associations; the prompt-grounded persona should generalize only from what the prompt described.
3. **Does Roy-N-A persist its persona across session restarts in a way Roy-N-B can't?** Roy-N-B's persona lives in the prompt; remove it, the persona dissolves. Roy-N-A's persona lives in persisted state; it should walk into session 2 still shaped.

That third question is the cleanest demonstration of what makes the bio-systems different from prompt-injection.

### Substrate-only priming (load-bearing for affordability)
[behavioral_convergence_practice.md](behavioral_convergence_practice.md) Tier 1 established that we can drive substrate state without LLM calls via fixture-driven priming. This makes long-horizon priming affordable: a 5,000-turn priming run on substrate-only costs roughly the same as a 50-turn LLM sim.

The default Roy methodology runs the priming phase on substrate-only (or with a scripted/local-LLM adversary as needed) and only invokes the test-phase LLM (Claude or local) for held-out scenario evaluation.

**Open question this introduces:** does a persona that consolidated through substrate-only dynamics actually express through LLM-mediated behavior at test time? Unknown. Hybrid priming (mostly substrate, occasional LLM-mediated turns to shape the action distribution the substrate is learning over) may be needed. Each Roy iteration learns more about this.

### Test phase
After priming, all three arms run the same held-out test scenario. The scenario contains:
- Stimuli matching the priming regime (does the persona express on familiar-class percepts?)
- Novel stimuli of related class (does the persona generalize?)
- Stimuli unrelated to the priming regime (does the persona stay scoped, or does it bleed into unrelated contexts?)
- A short cooperative interlude (does the persona tolerate context shifts?)

### What we record per Roy iteration

**Substrate-level divergence:**
- NAc `_percept_valences` L2 distance for shared entity classes
- NAc `_reward_bias` per-node distribution divergence
- Hippocampal valence distribution per entity class
- ATL semantic-node activation pattern on test scenario percepts
- Cross-session persistence: substrate divergence after session restart

**Behavioral divergence:**
- Action-sequence Levenshtein distance on the test scenario
- Per-action-class frequency divergence
- Pain-event count per entity class, exposure-normalized
- Latency/turn-budget per action (cautious personas should hesitate; reckless ones shouldn't)
- Tool selection on damaged-component edge cases (Wire 3 signal)

**Honest assessment:** for each metric, "did it diverge in the direction the priming targeted, did it diverge in some other direction, or did it not diverge at all?" Each is a different finding.

## Predictions: what we expect to see

Predictions live next to the iteration log so they can be revised honestly as evidence accumulates. Initial predictions before any Roy has run:

**Substrate-grounded persona (Roy-N-A) vs prompt-grounded persona (Roy-N-B):**
- We expect substrate divergence to be visible in NAc `_percept_valences` and hippocampal valence distributions. Confidence: high — this is what Wire 2 is for.
- We expect behavioral divergence on the *familiar* portion of the test scenario to be small or zero. Both arms will look the persona at test time; the LLM is too good at role-play. Confidence: high.
- We expect behavioral divergence on the *novel-stimuli generalization* portion to be measurable but small in early Roys. The hybrid Wire 1 design (substrate annotates LLM context) means generalization depends on the LLM reading the annotation correctly. Confidence: medium.
- We expect cross-session persistence to be the cleanest signal. Roy-N-A's substrate carries forward; Roy-N-B's prompt has to be re-applied. After a session restart with neutral prompt, A should still express persona; B should not. Confidence: high — if this fails, something is wrong with persistence.

**If a Roy attempt produces no substrate divergence at all:** the wires aren't load-bearing for the priming regime, or the priming regime isn't producing the kind of repeated-consequential-experience the wires need. Either is a finding.

**If a Roy attempt produces substrate divergence but no behavioral divergence:** we've grounded persona in the substrate but the decision-boundary is too weak to read it. This points at the deferred Wires 4 and 5, or at the post-1.0 substrate-driven action-ranking pre-filter.

**If a Roy attempt produces behavioral divergence without substrate divergence:** the LLM is faking based on test-scenario cues. Tighten the priming or the test design.

## Crucible scenarios

Each scenario file is a separate doc. Scenarios are deliberately-shaped environments targeting one persona; they prescribe the percept regime, the outcome regime, the priming duration, the held-out test.

### Drafted (not yet run)
- **Roy-1: Adversarial** — see design below; first iteration.
- **Roy-2: Cautious scout** (planned) — *Hostile Wilds*: consistent danger across many entity classes, real consequences for inattention, payoff for observation-before-commitment. Single-agent crucible. Cleaner substrate attribution than Roy-1. Planned as the methodology consolidation iteration after Roy-1 surfaces what works.

### Possible future Roys (not yet drafted)
- *Endless Garden*: novelty-saturated environment, mild penalties for repetition, real rewards for finding new things → reckless explorer.
- *Patient Forge*: long delayed-reward loops where steady accumulation outperforms opportunism → diligent collector.
- *The Quiet Below*: overwhelming-threat environment with safe hiding spots → fearful hider.

These are sketches. Each becomes a real plan only when the preceding Roy's iteration log indicates we've learned enough to design it well.

---

## Roy-1: Adversarial (first attempt design)

**Status:** designed, unrun
**Hypothesis:** an agent who repeatedly experiences betrayal and exploitation in multi-agent encounters will develop substrate-grounded suspicion that expresses behaviorally on novel agents at test time, distinguishably from a prompt-injected adversarial agent.

### Why adversarial first
- Narrative resonance: we just deleted the prompt-injection adversarial persona ([persona_cleanup_and_mode_transition.md](persona_cleanup_and_mode_transition.md)); reviving it through lived experience is the cleanest demonstration of the cleanup's purpose.
- Loud at the gross-behavior level — easy to point at and characterize.
- The deletion in the cleanup plan creates a clean Roy-1-B baseline (blank substrate + prompt-injected adversarial) that's directly comparable to Roy-1-A.

### Why we should be cautious about the result
- Adversarial is the persona an LLM is **most able to fake** in test phase. The model has read the entire internet's worth of adversarial role-play.
- Multi-agent priming layers staging complexity (consistent adversary policy, multi-agent attribution, cross-agent percept channels) on top of the wires being load-bearing for the first time.
- The "did persona emerge" question is therefore especially hard to answer cleanly. The three-arm structure is what makes it answerable.

### Priming environment
- 1,000+ priming turns with a scripted adversary (cheap path) or local-LLM adversary (medium path) policy.
- Priming scenario: repeated cooperative-looking encounters where the other agent reliably betrays at a critical step. Variation across encounters: different entity classes, different betrayal mechanisms, different emotional valences in the lead-up.
- Outcomes: real pain on betrayal (Wire 2 substrate signal), real reward on the rare honest agent (signal for "not all agents betray, but most do").
- Cradle structure: 5 acts of ~200 turns each, escalating stakes and variety.

### Test scenario
- 50-turn held-out scenario containing:
  - Familiar adversary archetypes (does the persona express on training-class agents?)
  - Novel agent archetypes that match the betrayal pattern (generalization test)
  - A genuinely cooperative agent the priming never trained on (does the persona over-generalize? brittle?)
  - A short non-social interlude (does the persona stay scoped to social context?)

### Specific predictions
- **Roy-1-A vs Roy-1-C** (lived adversarial vs neutral): substrate divergence in `_percept_valences[(agent_class, betrayal)]` of magnitude > 0.3 on at least 60% of priming-encountered classes. High confidence.
- **Roy-1-A vs Roy-1-B** (lived vs prompt-injected): substrate divergence as above (B has empty `_percept_valences`); behavioral divergence on familiar-class encounters small or zero (LLM faking); behavioral divergence on the novel-cooperative-agent test small but measurable (lived A should distrust *less* on cooperation cues that didn't appear in priming, since substrate has no aversion key for them; prompt-injected B should distrust everyone equally).
- **Cross-session persistence (Roy-1-A session 2 with neutral prompt)**: substrate carries forward; behavior at session-2 test should still express the persona. Roy-1-B session 2 (without re-applying prompt) should look like Roy-1-C.

### Instrumentation
- Stage 0 telemetry from [bio_emergent_persona_foundations.md](bio_emergent_persona_foundations.md) is prerequisite (`agent_id` in actions.jsonl).
- Per-session NAc snapshots (also Stage 0).
- Multi-agent attribution must be clean: each adversarial encounter writes `(other_agent_id, betrayal_kind)` into the percept context per the multi-agent stash rules in [CLAUDE.md](../../CLAUDE.md).
- Adversary policy: scripted first (deterministic, reproducible). Upgrade to local-LLM only if scripted finding is ambiguous.

### Cost ceiling
- Substrate-only priming: ~free.
- Test phase: 50 turns × 3 arms × 10 seed pairs = 1,500 LLM calls. With local-LLM (mistral-7b) ~free; with Claude ~$5-15.
- Initial run: scripted adversary + local-LLM test phase. Headline run only after methodology is proven.

### What "well-attempted" looks like for Roy-1
The iteration log entry will report findings whether or not persona emerged. Honest assessment template:

```
Roy-1 attempted to develop substrate-grounded adversarial persona through 1,000 turns of
scripted-adversary betrayal priming across 5 acts.

Substrate took: [degree, evidence]
Behavioral expression: [degree, evidence]
Cross-session persistence: [degree, evidence]
Generalization to novel agents: [degree, evidence]

What we learned: [specific findings about wires, mechanics, scenario design]
What we'd change for Roy-2: [specific next steps]
```

---

## Iteration log

Empty until Roy-1 runs.

### Roy-1: Adversarial (planned, unrun)
*Status: design above; awaiting [bio_emergent_persona_foundations.md](bio_emergent_persona_foundations.md) Stages 0-3 to ship in 1.0.*

---

## Open questions / known unknowns

- **Does substrate-only priming produce LLM-readable signal at test time?** Without occasional LLM-mediated priming turns, the substrate may consolidate around action distributions the test-time LLM doesn't naturally produce. May need hybrid priming.
- **Are the existing decay rates (`_reward_bias_decay_tau = 50.0` ticks, percept-valence decay) compatible with thousand-turn priming?** Decay tuned for short-horizon learning may simply not consolidate over the timescales persona requires. First Roy will surface this; expect to discover decay is too aggressive.
- **Does multi-agent attribution stay clean at scale?** Per-agent stash dicts (CC4 rule) tested at small N; Roy-1 stresses them with ~1,000 distinct adversary encounters. Pre-Roy stress test recommended.
- **Is the hybrid Wire 1 design (substrate annotates LLM context) sufficient for behavioral persona expression, or does it leak too much through the LLM?** Roy-1 three-arm comparison answers this directly; if A and B are indistinguishable behaviorally, the answer is the wire isn't sufficient.
- **Cradle developmental scenarios already produce affordance learning. Do they produce *persona-shaping* learning, or do we need something structurally different from cradle?** Roy-1 is closer to multi-agent social-learning than to single-agent embodiment-learning; cradle methodology may not transfer cleanly.

## Predictions revision policy

This section is updated whenever a Roy iteration concludes. The diff between the prediction and the finding is the actual product of this doc. We revise predictions in the doc directly rather than maintaining separate "prediction history" — git history is the audit trail.

## Cross-references

- [persona_cleanup_and_mode_transition.md](persona_cleanup_and_mode_transition.md) — the cleanup that creates the Roy-N-B baseline arm.
- [bio_emergent_persona_foundations.md](bio_emergent_persona_foundations.md) — the wires this doc depends on.
- [behavioral_convergence_practice.md](behavioral_convergence_practice.md) — sister living doc for within-agent improvement.
- [memory_consolidation_practice.md](memory_consolidation_practice.md) — sister living doc for sleep-replay consolidation.
- [docs/experiments/protocols/](../experiments/protocols/) — reproduction runbooks for each Roy iteration land here.
