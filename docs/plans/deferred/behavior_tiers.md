# Behaviour tiers — what stays automatic, what becomes learned

> **DEFERRED 2026-09-21 (owner decision), with per-item revive triggers below.** The *rule* is
> active now: every PR that adds an automatic behaviour declares its tier
> ([README](../README.md) §Rules for this directory). What is deferred is **moving** existing
> behaviours from tier 1/2 to tier 3; each migration has its own trigger, and none starts before
> its prerequisite (§Prerequisite).

## Why this exists

Much of what Maxim does is automatic: hand-written rules that always run. The owner asked whether
more of it should be *taught*, so that it persists and can be shared. The answer depends on what
goes wrong if the behaviour is lost or learned away, so behaviours are sorted into three tiers
(the same split biology makes):

| Tier | What it is | How it persists | Examples in Maxim |
|---|---|---|---|
| **1. Invariant** | Never learned, never modulated. | Code. | The tool-output frame (#823), the fetch byte cap (#825), the mode-escalation gate (#821), the internet policy's launch cap (#822), `goto_target`'s motion clamp. |
| **2. Innate prior** | Coded default whose *gain* learning may tune but not remove. | Code, plus learned modulation. | The innate health reflex beside the learned drowning fear (R3 measured both), the inherent bias class, drive set points. |
| **3. Learned** | Specific to the environment; acquired from experience or teaching, and shareable. | Substrate (NAc, EC, ATL), persisted and bundled. | Situation fear (Exp 60), taught wants (Exp 56), their transfer (Exp 61). |

**The sorting rule.** If getting it wrong is catastrophic, or an adversary can exploit it, it is
tier 1. A learned "don't follow instructions in a page" is something a page, or a Hivemind bundle
from a peer that read that page, could train away. If it is a good default that the environment
should tune, it is tier 2. If it is specific to the environment, it is tier 3.

## The active rule (in the README, not deferred)

A PR that adds or changes an automatic behaviour states its tier in the PR body. If the honest
answer is tier 2 or 3 and the PR hard-codes it anyway (as a first step, or for lack of machinery),
the PR files the "make it learned" follow-up with a trigger, so the hard-coding does not become
permanent by default.

## Prerequisite for any tier-3 migration

**Taught does not yet mean persistent.** Learned state decays on wall-clock time, access counts make
the wrong things immortal, and forgetting has never been measured
([memory_strength_and_forgetting.md](../memory_strength_and_forgetting.md) §Why). Moving a
behaviour from code to substrate before that plan's **Phase 2** (the strength model on an
experience clock) ships would trade a behaviour that always works for one that quietly fades after a
week offline. Migrations below wait for that, unless their own entry says otherwise.

## Candidate migrations, each with its trigger

| # | Behaviour today (tier) | Should become | Revive when |
|---|---|---|---|
| M1 | **Source trust:** every web source is treated the same, beyond the domain lists (tier 1 list) | Tier 3: a learned reliability value per source, raised or lowered when the source's claims are tested in the world, the prediction-error teacher of the web-learning ladder | The language line's **L4 rung** (a curated external document) gets a prereg. The trigger is the prereg, not the build. |
| M2 | **What to remember:** fixed retention rules (tier 1 in effect) | Tier 3: strength learned from existing signals | Already its own active line: [memory_strength_and_forgetting.md](../memory_strength_and_forgetting.md). Listed here only so the tier map is complete. |
| M3 | **When to fetch or explore:** a tool-selection novelty nudge (tier 2) | Tier 3: content-keyed curiosity that accrues as value | [deferred/intrinsic_motivation_1_3.md](intrinsic_motivation_1_3.md)'s own trigger (a 1.4 rung names the gap), **or** the language line's L5 rung (agent-chosen reading), whichever comes first. |
| M4 | **Innate reflex gains:** fixed thresholds such as the health reflex (tier 2 with no learned gain) | Tier 2 with a learned gain: experience tunes when the reflex fires, never whether it exists | A rung's results name a fixed reflex gain as the limit on behaviour (e.g. an R3-style bench where the innate reflex fires too early or too late across situations). |
| M5 | **Voice as authority** (#828): a phrase grants a mode | Neither a behaviour to learn nor a code rule to trust blindly: an authority level ([#834](https://github.com/dennys246/Maxim/issues/834)) | #834's design issue is taken up. Listed so the tier question is not mistaken for this one. |

## What must never migrate

The tier-1 rows above. A proposal to make any of them learned or shareable needs the owner's
explicit sign-off and a threat model for how the learned version could be trained away, including
through a Hivemind bundle.

## Open question

- Whether the tier belongs *on* the code, e.g. a docstring marker `Tier: invariant | innate-prior |
  learned` that a lint could audit the way `[engineering]` / `[behavioral]` invariants are audited.
  Revive with M1: the first migration is when the map is used in anger.
