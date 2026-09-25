# Rung B SUPPORT — design review record (2026-09-25)

These two lens reports reviewed a pre-registration for a rig trace (`rungb_support_preregistration.md`,
with an analyzer `scripts/survival_world/rungb_support.py`) that was **withdrawn on this review and
never merged**. Both lenses returned DO-NOT-BUILD independently, for the same two reasons:

1. **Wrong place.** The protocol recorded a day at an open spawn, where sky light swings 15 → 4 and
   gives apparent SUPPORT (~0.27 of the day in the middle band). The fear was learned in a sealed
   shell, where light is 0 all day and only `time_of_day` moves: ~0.05 middle, no SUPPORT.
2. **Computable offline.** With an idle bot and mobs, weather and movement off, light is a closed form
   in `time_of_day`, so the "trace" is a known curve (corollary 3 of
   `docs/wiring/cosine-separation-is-directional.md`).

And the finding that outlived the protocol: Exp 62's "night pool 0.799" is `time_of_day` **0.99**, the
minute before the clock wraps. At the fear place midnight reads 0.903 (inside the 0.85 key). The miss
is the linear encoding of a circular clock — [#899](https://github.com/dennys246/Maxim/issues/899) —
not night. Reproduced by the main session with the Exp 62 replay's own embedding (its guard
reproduces the live gate cosine 0.7874).

Consequences: `outstanding.md` O3 closed; memory 2S-e (B) parked
(`docs/plans/deferred/generalization_by_pattern_completion.md`); the Exp 62 records, roadmap 1.4
Phase 5 and CLAUDE.md corrected. A 4-minute apparatus dry run (rig `/tmp`, never committed) preceded
the review and is not evidence.
