# L11 re-measure pre-registration — A4 on real world traces (retirement-grade)

**Frozen 2026-09-04, merged to main BEFORE any capture.** Implementation:
[`scripts/l11_real_trace_remeasure.py`](../../../scripts/l11_real_trace_remeasure.py) — its
module docstring mirrors this document; on any divergence THIS file is the authority, and a
change to either after first data requires an amendment header here, per house prereg rules.

## Question

Does the shipped A4 nonlinear gain (1.1.4; `SensorEncoderConfig.gain_modalities={"world"}`,
p=3.0, unchanged 0.85 threshold), on REAL correlated sensor traces from the live Minecraft
bridge at a per-channel count above the ~12 safe band (the 16-sensor
`bodies/minecraft_player.yaml`, ranges re-centered so rest = the gain's neutral — D1's lever,
measured necessary by the pre-freeze review round), outperform the ungained control on the
bake-off's frozen weakest-link metric — and does it clear the retirement bar?

## Arms

- **A4** — the shipped default config, replayed through the SHIPPED production path
  (`SensorEncoder.encode_sensors(modality="world")`, real `EntorhinalCortex`,
  register-on-separate protocol).
- **A0** — identical replay with `gain_modalities=frozenset()` (ungained control).

## Metrics (real-data analogs of the bake-off trio; PRIMARY = min of the three)

- **stability**: over quiet pairs — consecutive snapshots with no onset-kind event within
  ±3 s of either AND max normalized state delta ≤ 0.10 (un-evented world drift is real
  change; without the delta bound this analog punishes exactly the sensitivity A4 provides)
  — fraction resolving to the same cluster. The delta-gate counts as same-cluster (it is
  arm-independent: raw values, pre-gain).
- **separation**: over event onsets — the first snapshot STRICTLY after each
  damage/spawn/death event ts (equal timestamps are the pre-event push), deduplicated by
  snapshot index — fraction resolving to a different cluster than the preceding snapshot.
- **discrimination**: over pairs of different-kind onsets at different snapshot indices —
  fraction on different clusters.
- **economy** (cost, reported, never folded in): clusters/100 snapshots, same-cluster
  fraction, gate-eligible fraction, duplicate-snapshot fraction.

## Decision rule

1. **Mitigation confirmed** iff A4 PRIMARY ≥ A0 PRIMARY **and** A4 PRIMARY > 0 **and**
   A4 separation > 0, at N ≥ 13 encodable sensors. (The >0 clauses exist because two blind
   arms tie at 0.0 — the D43 shape; a tie at zero must never read as success.)
2. **RETIRED-eligible** iff additionally A4 PRIMARY ≥ **0.70** (the bake-off's
   next-best-arm band).
3. **Both arms at PRIMARY 0 → "refuted-blind"** — an apparatus/body-design refutation, its
   own outcome.
4. Anything else → "not-confirmed"; L11 stays MITIGATED with numbers recorded and the next
   step named. No post-hoc threshold motion.

## Validity (S3 — explicit refusals, exit 4)

≥ 600 deduped snapshots; ≥ 8 resolved onsets of ≥ 2 kinds; > 0 quiet pairs; > 0
discrimination pairs; ≥ 13 encodable sensors (state ∩ declared ranges) in ≥ 95% of
snapshots; declared world set == declared ranged set; capture refuses in its first seconds
if the bridge snapshot lacks any declared world sensor.

## Apparatus

Paper 1.16.5 (offline mode), one Mineflayer bridge, the capture bot passive; night,
daylight cycle off, ≥ 6 hostiles summoned across the session, difficulty normal; 10-minute
capture at 0.5 s cadence. Data: `docs/experiments/data/l11_world_trace_<date>.jsonl` +
`l11_remeasure_<date>.json` — both gated (clean-tree preflight enforced at write time).

## Known-limit acknowledgments (the limits-ledger design rule)

- Rest-at-extreme ranges were measured structurally blind pre-freeze (event cos 0.926 under
  them; 0.747 re-centered vs A0's 0.960) — the re-centered declarations ARE part of the
  design under test; a weak result attributes first to range declarations, then to A4.
- Stability here includes any un-evented drift that survives the 0.10 bound and so may
  under-read the seeing arm; recorded, accepted.
- `min_delta = 0.05` is in raw sensor units, so the gate fires mostly on truly-frozen
  frames under Minecraft's integer quantization; gate-eligibility is reported per arm-
  independently.
