# Exp 59 (DRAFT idea, not frozen) — layered cave: darkness × danger × treasure and how components shift centroid clustering

> **STATUS: IDEA CAPTURE (2026-09-14, owner). NOT a pre-registration, NOT frozen, NO
> harness.** Recorded so it isn't lost while Exp 58 (learned dark-fear) runs. When Exp 58
> lands and this is picked up, it gets the full four-lens design review
> (`docs/experiments/DESIGN_REVIEW.md`) before any harness — this doc is the brainstorm a
> prereg is drafted *from*, not the prereg.

## The idea (owner, 2026-09-14)

Exp 58 builds a cave with a **danger** component (a deep pit + a persistent hostile). The
follow-up: add a **treasure** — a valued/rewarding target placed *deeper and a little
hidden, but not too far off* from the danger — and then present the components
(**darkness**, **zombie-danger**, **treasure**) in **separate training-scheme sequences**,
watching how the *outcomes* and especially the **centroid clustering** change as each
component is layered in.

The question is not just "does the agent avoid the dark" (Exp 58) but: **as attractive and
aversive components are added to the same space, in different orders, how does the EC
centroid structure reorganize?** Does a reward deeper than the danger create an
approach–avoidance conflict that splits or merges clusters? Does the order of layering
(danger-first vs treasure-first) leave a different centroid geometry — a path-dependence in
how the agent carves the space into situations?

## Why it's a natural next rung (and what it reuses)

- The apparatus is the SAME depth cave (`setup_world.py classroom`), extended: the pit is
  built **adequately long and with depth headroom** precisely so a treasure chamber can
  sit *deeper* than the danger without a rebuild (see the pit-length comment there).
- The mechanisms exist: Wire-4 situation-fear (Exp 58) for the aversive component; the
  drive-relief / operant-credit path (Exp 52/56) or a game-native pickup for the
  attractive component; EC centroids + `cluster_reward_bias`/`cluster_fear` for the
  clustering readout.
- It rhymes with the R1 cluster-granularity line and the L11 dilution finding: layering
  components adds axes of variation, which is exactly what changes whether situations
  separate or merge. This experiment could *measure* that reorganization directly.

## Open design questions (for the future prereg + its four-lens review)

1. **What is the treasure, game-natively?** A minable ore / a pickup that unlocks a tool
   (the intrinsic-motivation line, `intrinsic_motivation_1_3.md`), or an operant-credited
   reward? Must be D1-clean (a real game affordance, not a bespoke "+reward here").
2. **The centroid-clustering DV.** How is "how the space is carved" measured — cluster
   count, centroid separation/drift over training, the reward/fear bias map across
   clusters? Instrument-first: define and validate the clustering metric before running
   (and beware L11 — small-signal axes may not separate; the danger axis needed a
   persistent hostile to be reliable).
3. **Layering as the independent variable.** Separate training-scheme sequences =
   danger-only, treasure-only, danger-then-treasure, treasure-then-danger, both-together.
   The claim is about how outcomes/centroids differ across these — a within/between-scheme
   design with the sequence as the IV. Confound care: fresh agent per scheme for a clean
   claim (the one-world/fresh-agent-per-claim rule).
4. **Approach–avoidance conflict.** Treasure deeper than danger means the path to reward
   runs through the feared zone — does the agent's behaviour (and its cluster values)
   resolve the conflict, and how does that resolution show up in the centroid geometry?
5. **Relation to Exp 58's substrate limits.** Exp 58 exposed that the 17-sensor world
   channel dilutes single-axis signals (L11) and clusters are jittery live; a
   layered-component experiment leans even harder on stable, separable clustering, so the
   clustering instrument likely needs the depth/hostile/position multi-axis contrasts (and
   maybe the A4-gain / channel-split work) resolved first.

## Not decided / parked

The exact treasure mechanic, the clustering metric, the sequence set, and whether this is
one experiment or a small series. All of that is the four-lens design review's job when
Exp 58 is done. This doc only fixes the *idea* and its reuse of the depth-cave apparatus.
