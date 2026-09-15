# Exp 60 (DRAFT, not frozen) — learned drowning-avoidance: Wire-4 situation-fear on a separable cue

> **STATUS: DRAFT for four-lens experiment-DESIGN review (2026-09-15). NOT frozen, NO harness.**
> The pivot from Exp 58 (dark=danger, blocked): the SAME Wire-4 situation-fear mechanism — which
> we proved *fires* (dry-run: fear accumulates, flee executes) — applied to a survival cue that
> ACTUALLY SEPARATES on today's substrate. Exp 58 died at the instrument (dark/safe wouldn't
> cluster-separate; verified unfixable by encoding remedy — `docs/plans/l11_slice2_channel_split.md`).
> Drowning is the fix for the apparatus half: `oxygen` (modality world, range [0,40], rest 20 =
> normalized 0.5) descends to 0 underwater — a full half-range swing to an extreme (gain weight
> 0→1.0), so "underwater/low-air" is a genuinely distinct world cluster. This prereg is the
> brainstorm the four-lens review reads; it is NOT the frozen prereg.

## The claim

A survival agent LEARNS to escape the drowning situation — it surfaces / leaves water sooner after
experiencing oxygen-deficit pain than a fear-ablated control, driven by Wire-4 cluster-fear booked
onto the underwater world cluster (anticipatory threat need → corrective surfacing), NOT merely by
the innate reactive response to ongoing damage.

- **Arms:** FEAR (Wire-4 active) vs ABLATED (Wire-4 zeroed), fresh agent per arm, frozen seeds.
- **Primary DV (the learned, anticipatory component):** latency to leave water / time-in-water on a
  fresh submersion AFTER conditioning — FEAR should surface *earlier* (before deep deficit) than
  ABLATED. The DV must isolate ANTICIPATION (acting on the learned cluster-fear) from innate
  damage-response (both arms feel drowning damage; only FEAR carries the learned situation-fear).
- **Statistic:** permutation test across seeds, matched to the baseline (per `match-the-statistic`).

## Why this rung is viable where Exp 58 was not

- **The situation separates (the whole point).** `oxygen` at 0 underwater vs 0.5 surfaced is a
  full-range directional swing — the exact property dark/safe lacked (both underground, light=0;
  `nearest_hostile_dist` moved only 0.09 on one side of neutral → cos 0.977, unfixable). An
  instrument-check (safe-surface cluster ≠ underwater cluster) is a preflight, expected to PASS —
  but it is MEASURED first, not assumed (the verify-the-instrument lesson, twice-learned).
- **The mechanism already exists and fired.** Wire-4 (`_cluster_fear`, `record_cluster_fear`,
  `note_active_clusters`, `anticipatory_threat_need`), the pain→cluster subscriber, and the `flee`
  affordance are built and merged (Exp 58 line). Front-gate scope: NO new mechanism — this reuses it
  on a working cue. The failure mode `drive:oxygen` (or the existing air/drowning drive) must be in
  the `cluster_fear_failure_modes` allowlist (currently `drive:health` only) — a config addition to
  verify, not a new Wire.
- **The corrective act is game-native (D1).** Surfacing = swim up / leave water — a real affordance.
  Confirm the bridge exposes an executable "surface / move to air" (the `flee`-up path or a new
  param-free affordance), reachable and measurable, before freezing.

## Open design questions (for the four-lens review)

1. **Confounding — learned vs innate.** Minecraft applies drowning damage innately; the AGENT's
   learned want is anticipatory surfacing driven by Wire-4 cluster-fear. Does the FEAR-vs-ABLATED
   contrast + the anticipation-latency DV cleanly isolate the LEARNED component, or does innate
   damage-avoidance confound it? Is there a "surfaced but recently-drowned" probe that reads the
   learned fear without ongoing damage (the Exp 58 probe analog)?
2. **Bio-faithful — is oxygen-deficit a declared failure mode that publishes pain?** Exp 58 found
   pain needs a DECLARED failure mode (`pain-needs-declared-failure-modes.md`); confirm the
   oxygen/air drive publishes pain (entry/deepening latch) so the subscriber can book fear. Is the
   drowning-danger cluster co-active and noted when the pain fires (encode hoisted above pain tick)?
3. **Wiring — the corrective action + credit path.** Does `flee`/surface execute live from the water
   (pathfinder can swim? bot won't sink-drown mid-flee?), and does the read path
   (`anticipatory_threat_need` → `recommend_action`) actually emit the surface action? Verify with
   the real consumer, not a hand-composed probe (D43).
4. **Environment — the apparatus.** A NEW classroom variant: a deep water column / pool the agent is
   submerged in, with a reachable air/shore escape, built game-natively (real water, not a bespoke
   "oxygen−1 here"). Is submersion + escape reliably stageable (the drowning must actually deplete
   oxygen; the escape must actually restore it)? Does the bot drown-die too fast to measure latency
   (death-cap + rescue like Exp 58's teleport)?
5. **Transferability (the 1.3 headline, optional stretch).** If drowning-fear forms a clean cluster,
   does it bundle/travel between agents (the Exp 56/57 shared-want path)? Parked unless the base
   claim lands — but noted because a separable cue is the prerequisite the dark=danger want lacked.

## Not decided / parked

The exact water apparatus geometry, the surface affordance (reuse `flee`-up vs a new `surface`
action), the failure-mode allowlist entry, the anticipation-latency DV's precise definition, and the
death-cap/rescue. All are the four-lens review's job. This doc fixes the IDEA and its reuse of the
merged Wire-4 mechanism on a cue that separates.
