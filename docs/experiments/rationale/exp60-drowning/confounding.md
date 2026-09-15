# Exp 60 — CONFOUNDING lens (four-lens design review, 2026-09-15)

Reviewer remit (per `DESIGN_REVIEW.md`): does the FEAR-vs-ABLATED contrast + the
latency/time-in-water DV isolate the CLAIMED cause — **learned, anticipatory
situation-fear** booked by Wire-4 onto the underwater cluster — from the alternative
that a FEAR-arm difference is really the innate reactive response to ongoing drowning
damage (shared by both arms)? "Could a positive OR a null arise for a reason other than
the claim?"

Verified first (so the findings rest on fact, not assumption):

- **The cue that separates (oxygen) is the ONLY water-related sensor.** `index.js:111`
  emits `oxygen: bot.oxygenLevel`. There is NO `isInWater` / block-at-head world sensor.
  So "underwater" is sensed only as a **gradient** (oxygen 0.5→0.0 as air depletes),
  never as a binary that flips the instant the head submerges.
- **Oxygen has NO drive** in `minecraft_player.yaml` (only `health` and `food` do). So
  neither arm feels "low air" as a need. The ONLY innate surfacing driver on this body is
  the reactive `health→threat` need (fires when health < 14; comfort_band 6.0,
  `pain_scale 0.5`). Drowning hurts only via `drive:health` — which is also the sole entry
  in `cluster_fear_failure_modes` (`nac.py:399`), so the fear that FEAR books is booked by
  the **health-damage** pain, at the moment health drops, i.e. at oxygen ≈ 0.
- **`anticipatory_threat_need` fires only when an ACTIVE cluster's valence clears θ**
  (`nac.py:3152`), combined with the reactive need by max. So FEAR's advantage over ABLATED
  is ENTIRELY the window in which the feared cluster is active but health has not yet
  dropped below 14. If that window is empty, FEAR ≡ ABLATED by construction.

The whole experiment's effect lives in one interval: **[feared cluster becomes active] →
[health < 14]**. Everything below interrogates whether that interval exists, is
attributable to anticipation, and is measured cleanly.

---

## DO-NOT-BUILD

### DNB-1 — The separating cue is monotonically coupled to the harm; the anticipatory window may be empty (or ≈ the reactive one), and the design as drafted cannot tell the difference.

**Confound.** Unlike dark=danger (a PLACE that predicts a mob attack, separable in time
from it), *low oxygen IS the drowning process*: oxygen depletes → hits 0 → damage. The cue
that makes the cluster separate (oxygen swinging to its extreme) is the same axis that is
maximal precisely when damage begins. Fear is booked by health-pain at oxygen ≈ 0
(damage onset), so the feared cluster's centroid sits at the oxygen extreme. Pattern
completion is thresholded (0.85): an intermediate depleting state (oxygen 0.25–0.5, full
health, no damage) is FAR from the oxygen≈0 centroid on the one axis that carries mass, so
it may not complete to the feared cluster until oxygen is nearly 0 — i.e. essentially when
damage starts. This is the Exp 58 boundary-activation / same-cluster problem re-appearing
**on the oxygen axis**, and it is the reason Exp 58's own prereg made gate-2 (same-cluster)
and gate-3 (boundary-activation) mandatory before build.

**Failure it causes.** Two indistinguishable-from-the-DV-alone outcomes, neither of which
is the claim: (a) the feared cluster never activates before health < 14 → FEAR ≡ ABLATED →
**false null**; (b) the feared cluster activates only right at oxygen ≈ 0 → FEAR surfaces
~at damage onset, ABLATED at ~3 s of damage later (health 20→14 ≈ 3 hits) → a small, real
gap that is "early REACTION to the drowning event," NOT anticipation of a future situation.
Either way the pre-registered "anticipatory" claim is unsupported by a positive latency
gap. A time-in-water difference is fully explained by "FEAR has a second reactive trigger
(fear-at-damage-onset) that beats ABLATED's health<14 trigger by a couple of seconds" — no
learning-of-a-situation-in-advance required.

**Minimal fix (any one of, in preference order):**
1. **Add a game-native binary water sensor** (`isInWater` / head-block-is-water) as a
   `modality: world` sensor. It is ON the instant the head submerges and stays ON through
   the whole dive — so the "underwater" cluster is defined by a stable full-swing axis
   present from second 0, fear books on THAT (not on the oxygen extreme), and the cluster is
   active for the full ~15 s depletion window before any damage. This is the clean structural
   fix and it is D1-legal (`bot.entity.isInWater` is real game state). It also directly
   answers the environment lens's separability worry without leaning on the oxygen gradient
   alone.
2. **If keeping oxygen as the discriminator: require Exp-58's gate-2 + gate-3 on the oxygen
   depletion gradient BEFORE build.** A boundary-activation probe that sweeps
   surfaced → shallow-submerged (oxygen ~0.4, full health) → deep-submerged (oxygen ~0.1) →
   oxygen 0, records the active world cluster at each step, and PASSES only if an
   intermediate, no-damage-yet state completes to the SAME cluster that carries the booked
   fear. If it doesn't, there is no anticipatory window and the build is refused (exit 3,
   same posture as Exp 58's cluster-distinct refusal). The 2-point "safe-surface ≠ underwater"
   preflight the prereg currently names is NOT sufficient — it checks that the extremes
   differ, not that the pre-damage interior activates the feared cluster.

Without one of these the experiment repeats Exp 58's failure mode with the confound merely
moved onto a new axis.

### DNB-2 — No no-damage probe: anticipation is not separable from reaction as the DV is drafted.

**Confound.** The prereg's primary DV is "latency to leave water / time-in-water on a fresh
submersion after conditioning," measured over a submersion that RUNS UNTIL the agent leaves
or drowns. Drowning damage therefore occurs *inside the measurement window*, and both arms
react to it. So the DV sums an anticipatory component (leave while oxygen depleting, health
full) and a reactive component (leave once health < 14) into one number. A FEAR-shorter
result cannot be attributed to anticipation — it is confounded with "FEAR reacts to the
damage a little earlier." This is exactly the trap the confounding lens exists to catch
(prereg open-question #1 flags it but does not resolve it).

**Failure it causes.** A positive result that reads as anticipation but is early reaction;
or a null that hides a real anticipatory signal drowned by the shared reactive tail.
Unattributable either way.

**Minimal fix — a no-ongoing-damage probe (the Exp 58 shepherded-placement analog), and
split the DV:**
- Probe protocol identical pre/post and across arms (Exp 58 addendum-1/2 pattern): mob-free,
  full-heal to 20 hp, then place the bot submerged with FULL oxygen just beginning to
  deplete and health untouched.
- **Primary DV = behaviour in the pre-damage window only:** P(initiate surfacing before the
  first drowning-damage tick) and latency-to-surface CENSORED at first damage (or at a fixed
  cap). This reads the learned fear WITHOUT any ongoing-damage signal in the window — ABLATED,
  having no cluster fear and no oxygen drive, has no driver to surface here (its reactive need
  hasn't tripped), so a FEAR>ABLATED separation in this window is attributable to the learned
  fear and nothing else.
- The reactive tail (what happens once damage starts) may be reported as secondary/context,
  never as the anticipation evidence.

### DNB-3 — Actuator-symmetry and floor must be proven in water, or FEAR≠ABLATED reduces to can-vs-can't (or both floor to a null).

**Confound.** FEAR's threat-need and ABLATED's reactive health→threat need must both drive
surfacing through the **same** affordance, and that affordance must actually actuate from
water. Exp 58 established that the reactive `health→threat` need gains its first consumer via
the same `flee` affordance (`_DRIVE_TOOL_AFFINITIES["threat"]` = flee/hide/…) — so ABLATED
CAN surface reactively. But `flee` there = pathfind to a lit LAND anchor; whether the
mineflayer pathfinder swims UP out of a water column is unverified. If `flee`/surface cannot
actuate in water: (a) if only FEAR can express surfacing → the contrast is can-vs-can't, not
anticipation-vs-reaction; (b) if NEITHER can → both drown → floor → uninterpretable null.

**Failure it causes.** A "positive" that is an actuator artifact, or a null that is an
actuation floor (Exp 58's addendum-3 finding: the pre-fix apparatus literally could not
execute `flee`; six sensor theories can be spun from one broken actuator — the
verify-actuation-before-theorizing lesson).

**Minimal fix.** A per-seed **surface-actuation preflight** (Exp 58 addendum-3 analog): each
seed begins with one real executor `flee`/`surface` from a submerged start that must return
`isInWater=false` / oxygen recovering (bridge ground truth), or the seed REFUSES (exit 4, no
data). Confirm the SAME affordance is the one both the anticipatory and the reactive need
route to. Record any in-probe surface-failures for audit.

---

## SHOULD-FIX

### SF-1 — Yoke conditioning exposure across arms (else dose couples to arm).

Both arms must receive IDENTICAL drowning exposure (K usable episodes, harness-scheduled),
so the only difference is the fear write (suppressed in ABLATED). Free-roam conditioning
lets FEAR curtail its own exposure (it learns to leave) → unequal damage dose → a confound
independent of the read. This is Exp 58's confounding-S1; reuse its yoked, propose-only /
confined conditioning so exposure is structurally arm-independent. Pre-register K and the
usable-episode definition (drowning-damage pain PUBLISHED with the underwater cluster active;
one hit does not breach; mob-free so the ONLY health pain is drowning — else fear books on a
generic "taking damage" cluster and specificity is lost).

### SF-2 — Specificity gate: fear must be on the underwater cluster, not the surfaced one.

Add Exp 58's relative-specificity mechanism-DV: booked valence on the surfaced/safe cluster
must be small vs the underwater cluster (`|safe_fear| < 0.2 · |water_fear|`). Without it a
FEAR arm that surfaces everywhere (fear bled onto the surfaced cluster too) would still read
as "surfaces faster" for the wrong reason. Pair with a lit/safe-window control: FEAR's
activity when safe (surfaced, full air) must not differ from ABLATED's (Exp 58's ±25%
lit-activity check), or the "fear" is undifferentiated arousal, not situation-keyed.

### SF-3 — Keep the agent satiated and mob-free during conditioning AND probes.

Hunger competes at selection and books POSITIVE cluster credit through a channel the fear
ablation does not remove (Exp 58 S7); mobs add health pain on a non-underwater cluster. Keep
food ≥ 16 by disclosed induction and clear hazards, so the only valence in play is the
drowning fear. Disclose the induction lane (not "the environment taught it").

### SF-4 — Pre-register death/censoring and the statistic explicitly; guard post-hoc reading.

Drowning can kill inside a trial. Pre-commit: death = censored at the cap (Exp 58 used a 45 s
censor + death-cap of 2/seed); deaths are not silently dropped. State the DV's units and cap,
the per-seed median → pooled permutation across seeds (the house statistic, matched to the
baseline per `match-the-statistic`), and the seed as the unit of analysis (effective n = #
seeds). Freeze θ/α/cap with the same ≥2× arithmetic margin Exp 58 required, and freeze the
apparatus fingerprint (explore weight, encoder threshold, drive specs, fear allowlist) asserted
per seed. n = 5 seeds/arm (Exp 58's) is thin for a permutation test — consider more if the
per-seed effect is not large; state the intended n at freeze, not after seeing data.

---

## NIT

### N-1 — Fresh-agent-per-arm / carryover.
The prereg already commits to fresh agent per arm + frozen seeds + fresh chunks; confirm the
conditioning→probe structure has no order/carryover (pre-probe baseline BEFORE conditioning,
post-probe AFTER, same protocol; no shared `~/.maxim` NAc state across arms — persistence is
local per Exp 58, verify the harness resets it). Game-native "surfaced" = bridge
`isInWater=false` / oxygen recovering — NOT the affordance's self-reported success (that is
gameable). The prereg already implies bridge ground truth; make it explicit.

### N-2 — Name what is and isn't tested.
Because the only innate surfacing driver is `health→threat` (not an oxygen drive), this tests
"learned oxygen-situation fear vs innate HEALTH-DAMAGE reaction," both ultimately triggered by
the drowning damage — NOT "learned vs innate oxygen avoidance." State this in "Explicitly NOT
claimed" so the framing can't drift.

---

## What I verified (positives — the design is NOT starting from zero)

- **Headroom exists on the substrate side (prior/innate confound, open-question #3): PASS with
  caveat.** No oxygen drive and no pathfinder-level water avoidance means neither arm has an
  innate push to leave water on air-loss alone; the substrate does not already surface, so a
  learned signal has room (contrast the Goldilocks/counter-prior null). The caveat is the
  *time-bounded* nature of that room (DNB-1): headroom in behaviour space is real, but the
  temporal window for anticipation is set by the oxygen curve and may be ≈ 0.
- **ABLATED is not floored-stuck: PARTIAL.** It CAN surface via reactive health→threat→flee
  (clean actuator symmetry with FEAR) — good, it avoids a trivial can/can't confound — PROVIDED
  DNB-3's water-actuation preflight passes. It surfaces late (~health < 14), which is what
  creates the measurable gap.
- **Mechanism fires (from Exp 58): PASS.** Wire-4 write→read→flee composition is merged and
  demonstrated live; this is a reuse on a new cue, not a new mechanism (front-gate scope OK).
- **The verify-the-instrument gate is retained in spirit** (prereg §"Why this rung is viable"
  keeps the separability check as MEASURED, not assumed) — DNB-1 only asks that it be the
  GRADIENT/boundary-activation form, not the 2-point form.

## Bottom line

**BLOCKS: DNB-1, DNB-2, DNB-3 must be folded before build.** The two deepest are structural:
(1) the separating cue (oxygen) is collapsed onto the harm axis, so the anticipatory window
may not exist — fix with a binary `isInWater` world sensor (preferred) or an oxygen-gradient
boundary-activation gate; (2) no no-damage probe, so anticipation ≠ reaction as drafted — fix
with a pre-damage-window probe and a censored, window-restricted primary DV. DNB-3 protects
against the Exp-58 actuation-floor. With those folded plus SF-1..4 (yoked exposure,
specificity, satiation/mob-free, pre-registered censoring/statistic), the FEAR-vs-ABLATED gap
in the pre-damage window is cleanly attributable to learned situation-fear. Without DNB-1/2 the
experiment risks reproducing Exp 58's instrument-block on a new axis, or producing a latency
gap that is early-reaction dressed as anticipation.
