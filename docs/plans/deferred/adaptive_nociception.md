# Adaptive nociception: habituation and sensitisation of pain

> **DEFERRED 2026-09-21 (owner decision), out of scope for the memory-strength line.** Today's pain
> intensity is fixed: an **innate prior** (behaviour tier 2) whose gain nothing learns.
> **Revive when:** (a) a world produces **chronic pain that does no damage** (repeated harmless
> pain an animal would habituate to), or (b) a measurement shows **over-tagging from repeated pain**:
> memory-strength Phase 2+ tags, or NAc fear, saturating on one repeated harmless source. Whichever
> comes first. Before reviving, re-read the Exp 60–62 fear results; they depend on pain staying
> strong.

## Why this exists

During the memory-strength Phase 2 decisions, the owner questioned the rule "pain baseline is fixed,
never adaptive": *"I feel like we need some degree of habituation and desensitisation in somewhat of
an adaptive sense."* That is right about the biology. The memory plan still keeps pain fixed, for
two reasons:

1. **Pain adapts in both directions.** Repeated harmless stimuli habituate (the response fades).
   Repeated real harm sensitises (the response grows: wind-up, central sensitisation, hyperalgesia).
   A flat adaptive baseline would model only habituation. In a chronically dangerous pool it would
   stop tagging the one thing that matters, which would erase the drowning-fear signal Exp 60–62
   earned.
2. **Adaptation belongs upstream, in the pain producer, not in the memory baseline.** If pain
   intensity adapts at its source (the PainBus producers, the SEM drive-pain derivation), every
   consumer reads the adapted value for free: the memory tag, NAc credit and the reflexes. That gives
   one mechanism instead of a memory-side copy that drifts away from what the rest of the body feels.

## The shape it should take (sketch, for the revive design)

- **Per-source, per-agent adaptation state**, keyed by what hurts (the pain source or type, and the
  body part or drive), not one global gain.
- **Two opposing processes.**
  - **Habituation:** the gain falls with repeated exposure **that causes no damage**, where damage
    means integrity or health actually changed.
  - **Sensitisation:** the gain rises with repeated exposure **that does cause damage**, and recovers
    only slowly.
  - The discriminator is whether damage followed. It is measured, not assumed.
- **Recovery on the experience clock** (the memory plan's clock, not wall time): dishabituation
  after a gap, and slow decay of sensitisation.
- **Bounded gain.** A floor above zero, so dangerous pain can never be fully habituated away. That is
  the tier-2 rule: learning tunes the gain but never removes the prior.
- **Frozen during experiment campaigns** and recorded in the harness fingerprints. It changes the
  unconditioned stimulus that Exp 60–62 measure against.

## What it must not do

- Silence pain from a source that is still damaging the body.
- Adapt a copy of pain inside one consumer, such as memory, NAc or a reflex. Only the producer adapts.
- Change any default before an experiment earns it (two-tier rule: it enters as `[engineering]`).

## A current violation of the rule above (found 2026-09-24)

The narrative reflex path adapts the **stimulus itself**: `embodiment/reflex.py` habituation,
sensitization and pre-emption scale the damage `damage_component` inflicts. Its re-layering is
recorded in [reflex_layering.md](reflex_layering.md). That plan's step 3a hands felt-pain
adaptation to this plan rather than building a second one, so reviving either should read both.

## Links

- [memory_strength_and_forgetting.md](../memory_strength_and_forgetting.md), Phase 2 decision 4
  (pain `x = intensity`, fixed, pointing here).
- [behavior_tiers.md](behavior_tiers.md), migration M6.
- Exp 60 / 61 / 62 in [behavioral_graduation_candidates.md](../behavioral_graduation_candidates.md):
  the rows a pain-gain change would re-trigger.
