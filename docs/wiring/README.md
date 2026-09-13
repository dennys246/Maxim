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

- [substrate-learning-channels.md](substrate-learning-channels.md) — the TWO learning channels a
  successful action writes (state-blind tool-success **causal link** vs state-conditioned
  **cluster reward bias**), why they confound behavioural credit-isolation, and how to design a
  probe that separates them. (R2 learned-bias, 2026-09-12.)

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
