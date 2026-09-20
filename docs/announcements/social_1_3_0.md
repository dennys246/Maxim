# 1.3.0 "Oasis-2" — social announcement drafts

**Audience:** people outside the project — the release notes are the record, this is the invitation.
**Rule this file lives by:** a post may claim exactly what `release_1_3_0.md` claims, no more. Every
number below traces to an experiment's §Outcome. Three things are easy to overclaim and must survive
every edit: survival was a **ceiling** (nothing here is "the agent survives because it learned"), the
transfer was measured on **one situation in one world**, and **no language model is in the action
path** for these results — which is the interesting part, not a disclaimer.

Published 2026-09-19 · `pip install pymaxim==1.3.0` · https://pypi.org/project/pymaxim/1.3.0/

---

## Short post (X / Mastodon / Bluesky) — the one-idea version

> An agent felt air-hunger underwater in Minecraft. It learned to fear that situation and started
> leaving the water *before* the pain.
>
> Then we exported that fear to a second agent that had never drowned. On its first live submersion,
> it left the water too. 12/12, against 0/24 controls.
>
> pymaxim 1.3.0 "Oasis-2": the reward comes from the game, not from a teacher — and no LLM is in the
> action path. Details, limits and what we do *not* claim: <link>

**If a thread is wanted, the second and third posts:**

> 2/ The honest part: survival was a ceiling. Every agent in every arm lived. What the learned fear
> buys is the cost it removes — about 25 seconds of latency, 11 health points and 22 seconds of pain.
> Not life. We built a benchmark to measure exactly that, and it graduated nothing.

> 3/ Also not claimed: the fear is keyed to one pool, not to water everywhere. "Dark = danger" is
> blocked at the instrument — our sensors can't separate dark from safe. Eating when hungry turned
> out to be a prior, not learning. The release notes say all of this before they say anything else.

## Medium post (LinkedIn / a short blog note)

> **A fear that travelled between two agents.**
>
> pymaxim 1.3.0 is out. The previous release showed that a want a *teacher* put into one agent could
> be shared with another. This one takes the teacher out: the learning signal is the game's own pain.
>
> On a live Minecraft server, an agent held underwater until it felt air-hunger learns a fear keyed
> to that situation. Afterwards it leaves the water before the pain fires. A yoked twin with the same
> pain exposure but the fear mechanism detached never does — five seeds each, and the separation is
> complete.
>
> Then the interesting bit. The first agent exports its substrate through a signed bundle. A fresh
> agent ingests it, reboots, and meets water for the first time with its loop live. It leaves.
> Twelve out of twelve, against zero out of twenty-four isolated controls, and zero for two further
> controls that ship the pieces of the fear without the fear itself.
>
> None of this involves a language model choosing actions. The drives, the fear, the credit and the
> selection all run in the substrate.
>
> What we are careful not to say: the agents were never at risk of dying — regeneration was on and
> survival was a ceiling by design, so the benchmark measures what the drive *costs* an agent, not
> whether it lives. The fear is keyed to one pool; whether it reaches a second one is the next
> experiment, already designed and not yet run. And the contingency we originally planned, "dark =
> danger", is blocked at the instrument: the sensors cannot separate those situations, so the
> contingency was earned on water instead.
>
> Release notes, preregistrations and the data: <link>

## The paragraph to reuse anywhere

> pymaxim 1.3.0 "Oasis-2": an agent learns a fear of drowning from the game's own pain, acts on it
> before the pain returns, and that fear transfers to an agent that never felt it — 12/12 against
> 0/24, with no language model in the action path. Survival itself was a ceiling: what the drive buys
> is the cost it removes, not life.

## Answers for the obvious replies

- **"Is this just a hard-coded reflex?"** There is an innate reflex, and it is one of the benchmark's
  arms: it surfaces an untrained agent in about 28 seconds, after the damage. The learned fear
  surfaces in about 3, before any damage. The comparison is the point.
- **"Did the second agent just copy a policy?"** It received a substrate bundle — clusters and
  valences — not a policy or a script. A control that shipped the situation without the fear produced
  zero escapes, and one that shipped the fear without the situation produced zero as well.
- **"Why Minecraft?"** Because it is a world the system does not control, with pain, relief and
  states we can measure. It is the instrument, not a demo. Nothing here says the agent plays the game.
- **"Is this AGI / does it feel anything?"** No. Bio-inspired is an engineering stance, not a claim
  about experience, and the mechanisms are named after brain systems because that is where the design
  came from, not because they are simulations of them.

## Before posting

1. The linked notes must be live (`docs/announcements/release_1_3_0.md` on main, the GitHub release).
2. Re-read the "what is not claimed" section of the notes and check the post against it.
3. If any number moves in a later correction, correct the post the same way the repo corrects a
   release: add the correction, leave the original visible.
