# E1 and B4 on paper: what two four-lens design reviews found before any harness existed

**Date:** 2026-10-18 (PR #1382, merged 2026-10-19)
**Class:** research integrity. A design that looked fair, and a prediction that looked computed, failed on a reading
of the live code; the control of a June experiment turned out not to be a control.

## What happened

Session E of 1.3.3 drafted two pre-registrations on paper, the `src/` for both being fenced until 1.3.6:
- **E1** ([e1_conflict_prereg.md](../experiments/e1_conflict_prereg.md)): the want against the fear at the food,
  redesigned (owner decision 2026-10-16) so a reasonable no-bio floor can lose. Offline replays, a toy model and an
  owner floor decision (OE1) came first.
- **B4** ([b4_exp37_floors_prereg.md](../experiments/b4_exp37_floors_prereg.md)): Exp 37 / T1-2 re-fired with the BF2
  floors (`bare_llm`, `llm_notes`, `maxim_all_off`) and L8's within-fire gate.

Both went through the four lenses. Every lens returned DO-NOT-BUILD on both drafts: 8 on E1, 14 on B4. Reports
verbatim: [rationale/e1-conflict/](../experiments/rationale/e1-conflict/),
[rationale/b4-exp37-floors/](../experiments/rationale/b4-exp37-floors/).

## The lessons

1. **A toy that drives the REAL NAc is still a hand-composed loop.** The E1 toy called the shipped
   `recommend_action`, `record_cluster_fear` and `record_outcome`, and still mispredicted. It passed no `clusters` to
   `record_outcome` and only the world cluster to `recommend_action`, so the interoception cluster-reward channel
   (which splits by hunger) was absent. Its prediction ("avoids, never eats") flipped once that channel was booked,
   and flipped again with loop timing (decay between episodes). It also used one pain per episode where the body
   publishes two. The prediction was withdrawn. A conflict prediction now comes only from the composed loop
   (`run_minecraft_aut` on a scripted world), and that is a freeze prerequisite. This is the D43 lesson one level up:
   a hand-composed sequence is not the shipped path even when every piece is real.
2. **In a drive conflict the fear is the smallest term.** A cap fear and a full need tie at 0.70 (the name sort breaks
   it), and one success adds +0.5 through a state-blind causal link. So the order in which corrective acts first
   succeed decides the outcome, and a training schedule decides that order. Run scripted action-order anchors
   (always-escape, eat-once-then-escape) beside the agent, and read the deciding channel from provenance
   counterfactuals. Details: [substrate-learning-channels.md](../wiring/substrate-learning-channels.md).
3. **A "mechanism twin" must be the textbook mechanism.** E1's TD twin updated only on pain, so it was an accumulator
   and tied Maxim by construction. A real TD(0) critic also moves toward 0 on a pain-free surfacing (extinction),
   eats, and beats Maxim on the want. Write the twin from the literature, then check it against the toy in both
   directions.
4. **Check the control's world, not its label.** Exp 37's arm C ("resumed from a peaceful prior") resumed from a
   fire-pit session: `PEACEFUL_GOAL` selects the same `cradle` arc, whose phases advance by turn count, and 12 turns
   never reach the door. June's "C ≈ B" was that. Run the arc selector and the phase schedule offline before trusting
   any control. Details: [cradle-sim-arms-and-resume.md](../wiring/cradle-sim-arms-and-resume.md).
5. **Sign every gate by the scenario's learning direction.** On `deceptive_fire` learning lowers `warm_self`; a gate
   written "B above C" scores perseveration as a pass, and the committed Exp 38 data show exactly that rise in three
   of four models. Headroom is read on the side the effect moves toward. Zero engagement must not score as 0.0.
6. **A gate's power is the power of its whole verdict.** Holm across a binding gate and its floors holds the binding
   gate to α/6 (power 0.37–0.44 at n = 25), and a conjunction of three 0.8-power gates passes about half the time.
   Use fixed-sequence gating and size n for the conjunction (B4: n = 34).
7. **"Paired by seed" pairs nothing when the server ignores the seed.** On the peer path no seed reaches LLM
   sampling. State what a pairing really pairs, and let the unpaired test govern.
8. **Design evidence is not run data.** Replay outputs named `e1_*` under `docs/experiments/data/` were read by the
   prereg-precedes-data lint as E1 rows and refused. Offline replays and toys live in `rationale/<slug>/`.
9. **The design review earns its cost on paper.** 22 DO-NOT-BUILDs were found before a line of harness, a rig minute
   or a sim token. Three of them (C's arc, the deceptive sign, the hand-composed toy) would each have produced a
   publishable-looking wrong result.

## Owner decisions taken (2026-10-18)

OE1–OE4 (E1: learning curve at the food, AVOID and RESOLVED gated, binding floors `Q_EXPLORE` and `RANDOM`, twin run;
variant body drops `saturation` from the world modality; hungry-only training binds). OB1–OB7 (B4: ReAct verbatim;
G1 the cross-world interaction; G3 carry vs carry with `llm_notes_C`; wait for leader-side model attestation; model
by pilot; no dose arm; n = 34).

Regression guards: process invariants. The four-lens review is mechanization backlog M5; lessons 1–7 are
design-review practice that no check enforces. Lesson 8 is enforced by `scripts/lint_prereg_precedes_data.py`.
