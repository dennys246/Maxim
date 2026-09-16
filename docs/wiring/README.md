# Wiring the subsystems into an environment — a living field guide

**Purpose.** When you connect Maxim's bio-inspired subsystems (drives, credit, clusters, the
substrate selector, embodiment) to a *new environment* (Minecraft, Reachy, a sim), the same
non-obvious wiring facts get rediscovered the hard way. This directory is the running record of how
each subsystem actually behaves when wired to a world, so a discovery is made **once**.

**Scope.** Not architecture (that's [ARCHITECTURE.md](../../ARCHITECTURE.md)) and not decisions
(that's [DECISIONS.md](../../DECISIONS.md)). This is *operational*: "if you wire X to a world,
expect Y; measure it like Z; the trap is W." Each entry earns its place by having *bitten us* or
*shaped an experiment* — cite the run/PR/experiment that established it.

**How to use.** Read the relevant subsystem doc BEFORE wiring it into a new environment or
designing an experiment that leans on it. Refine as we go — every survival-loop / 1.3 discovery
that generalizes belongs here.

## Subsystem docs

- [harness-loop-must-be-proven-live.md](harness-loop-must-be-proven-live.md) — a substrate-primary
  HARNESS loop must be proven able to tick AND act before any window can measure anything: three
  instrument causes behind one `actions=0` symptom (idle gate had no substrate wake source; bridge
  cadence; the loop ran at PLANNING autonomy and never executed a body affordance), the measurement
  ladder that named them (window telemetry → thread profile → proposal timeline → read the gate →
  reproduce the LIVE condition offline), and the preflights every harness now carries. (Exp 60 trial
  harness, 2026-09-15/16.)

- [substrate-learning-channels.md](substrate-learning-channels.md) — the TWO learning channels a
  successful action writes (state-blind tool-success **causal link** vs state-conditioned
  **cluster reward bias**), why they confound behavioural credit-isolation, and how to design a
  probe that separates them. (R2 learned-bias, 2026-09-12.)

- [world-light-sensing.md](world-light-sensing.md) — block light ≠ brightness: why `light_level`
  read "dead 0 everywhere" in Exp 56 (raw block light is 0 under sunlight), the
  `max(block, sky − darkness(time))` fix, and the 1.18 block-light-0 spawn-rule trap.
  (First live 1.20.4 session, 2026-09-13.)

- [sensor-range-clamps.md](sensor-range-clamps.md) — world sensors clamp to their body-declared
  ranges (`y_altitude` caps at 128): compare harness expectations against the SENSED value via
  `_read_world_ranges`, never raw world truth; states beyond a cap are indistinguishable to the
  agent. (Survival Phase-0 first run, 2026-09-13.)

- [pain-needs-declared-failure-modes.md](pain-needs-declared-failure-modes.md) — CORRECTED
  entry: the full damage-fear topology. Drive pain DOES publish (band-crossing + latch
  semantics); action-blame is correctly B8-suppressed for bystander actions; Wire 2 percept
  aversion fires but is SITUATION-BLIND (keyed to the suffering body's class, not the
  co-active world cluster) — so "dark = danger" has NO write channel on current wiring: a
  Phase-1 design item. Plus four instrument traps, incl. `pain_bus.recent` being a lossy
  view. (Dark=danger probe, 7 iterations, 2026-09-13.)

- [cluster-dilution-blocks-situation-fear.md](cluster-dilution-blocks-situation-fear.md) — L11
  live: on a 17-sensor world channel, partial-axis contrasts between situations don't clear the
  0.85 separation threshold, so a situation-keyed contingency (Exp 58 dark/danger fear) can't form
  a distinct cluster to key on. Measure the live cosine geometry before building; remedies are
  substrate-level (channel-split / scaled threshold / gain). (Exp 58, 2026-09-14.) **Superseded as
  the operative mental model by ↓ cosine-separation-is-directional.**

- [cosine-separation-is-directional.md](cosine-separation-is-directional.md) — **the deep form of
  L11, read this first for any situation-keyed contingency.** Cosine sees DIRECTION not magnitude,
  so the A4 gain (a magnitude weight) can't make situations separate; only a sensor that rotates the
  summed vector (a neutral→extreme / full-range-across-neutral swing) separates them — a small
  one-sided move never does, at any gain. Corollaries: the gain-weight "mass" metric predicts
  contribution not separability; use a binary rest-neutral state-flag for a stable pre-event cluster;
  replay any remedy offline on real captured vectors first; adding a rest-neutral sensor is safe but
  re-tagging an existing one orphans persisted nodes; the pathfinder is dead in water. (Exp 58 →
  Slice-2 reject → Exp 60, 2026-09-15; verified `docs/experiments/data/*_cosine_check.py`.)

- [experiment-catalog-candidates.md](experiment-catalog-candidates.md) — a prioritized **backlog** of
  11 reusable wiring lessons harvested from all 92 past experiments (catalog workflow, 2026-09-13);
  each becomes a full page as we next touch that subsystem. Highest-priority: the substrate→LLM
  annotation channel being lossy (recurs across 9 cards), cross-agent want-transfer's coverage
  ceiling, and relief-sourced operant credit.

_(stubs to fill as we hit them):_
- `drives-and-corrective-affinity.md` — deriving a corrective NEED from a world-owned drive; the
  read-tool trap (R2 break 1).
- `measured-relief-credit.md` — booking real relief for interoceptive world drives; modality-gating;
  the bridge sync-timing trap (R2 break 2 + the eat-lag).
- `world-affordance-and-bridge.md` — making corrective acts executable game-natively (R2 break 3);
  the one-client bridge, connect-retry, RCON classroom setup.
- `frozen-apparatus-hygiene.md` — the config surface that silently changes substrate selection
  (explore-bonus, cluster-bias caps, EC thresholds, `MAXIM_OPERANT_ONLY_CREDIT`); fingerprint +
  assert-absolute.

## The meta-lesson

Most of these are the same shape: **a subsystem does something reasonable in isolation that
interacts surprisingly once a world drives it.** The fix is almost never "change the subsystem" —
it's "understand the interaction and design the wiring/experiment around it." When in doubt,
**run the smallest experiment that measures the interaction** rather than reasoning from the code.
