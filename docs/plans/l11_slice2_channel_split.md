# L11 Slice 2 (Step C) — per-type channel-split of the `world` modality

**Status: DO-NOT-BUILD (2026-09-15) — four-lens design review + an offline computation on the
Slice-1 vectors killed the remedy before any substrate code was written.** All four lenses
returned DO-NOT-BUILD; a direct cosine computation on the real encoder bases (sanity-checked:
reconstructed full-channel cos = 0.9766 vs the probe's 0.9767) shows **every candidate remedy is
WORSE than the failing status quo and none approach the 0.85 threshold**:

| candidate | cos(safe,dark) |
|---|---|
| full 16-sensor channel (status quo) | 0.977 (fails) |
| per-type `world:threat` (5 sensors) | 0.991 (WORSE) |
| minimal threat (`nearest_hostile_dist` alone) | 0.992 (worse) |
| range-fix (re-center `light_level`+`time_of_day`) | 0.991 (worse) |

**Root cause (bio-faithful lens, verified): cosine separation is a DIRECTION problem, and the A4
gain is a MAGNITUDE weight cosine ignores.** The lone real mover `nearest_hostile_dist` shifts
0.179→0.087 — both *below* the neutral 0.5, so the embedding direction barely rotates (cos≈0.99).
Regrouping or de-massing the constant sensors cannot fix a near-collinear pair; you would need a
sensor that swings across a large directional arc (crosses neutral / full-range), or several
independent movers. The classroom also can't express dark=danger via the darkness cue at all —
the "safe" chamber at y=40 is *also* underground, so `light_level`=0 in both.

**Therefore the block is APPARATUS + REPRESENTATION, not the channel grouping**, and the four
lenses additionally found the split would silently break the mechanism it exists to serve (fear
WRITE hard-codes `.get("world")` → goes silent; bridge sensor allow-list `=="world"` → world
sensing dark; R2 relief-credit gate `=="world"` → credit stops) and DESTROY (not preserve) Exp
56/57 taught-wants (a modality re-tag orphans their EC nodes; the named migrate command only
handles geometry, not modality). Full findings: `docs/experiments/rationale/l11-slice2/`. The
design below is retained as the record of what was reviewed and rejected; §Decision at the end
names the real options.

---

**Original status line (superseded): DESIGN PLAN DRAFT (2026-09-15), for four-lens design review BEFORE any substrate code.**
This is Step C of the L11 approach (A = bank the Exp 58 block ✓ #713; B = geometry diagnostic ✓
#715/#716; C = build the remedy B nominated). Gated by the Slice-1 result
(`docs/experiments/data/l11_geometry_2026-09-15.json`, verdict `diluted_present`) and its plan
(`docs/plans/l11_world_channel_diagnostic.md`). **Nothing here authorizes a build** — it is the
artifact the design review reads; the build waits on the folded review + operator go-ahead.

## What the data says the fix must do

Slice-1 measured, on the live classroom, that safe and dark share a world cluster because
**constant full-weight sensors dominate the sum**: `light_level` (gain weight 1.0 in both, Δ0 —
the "safe" chamber at y=40 is also underground) and `time_of_day` (0.77 both, Δ0) carry maximal
mass but zero contrast, out-voting the one sensor doing real work — `nearest_hostile_dist` (gain
weight 0.27→0.56). Depth (`y_altitude`) is nearly static (Δnorm 0.037). So the remedy is not
"rescue a silenced signal" — the signal is passed — it is **stop the constant sensors from
drowning the discriminating ones**: put the discriminating sensors in their own small-N channel
where their contrast dominates.

## Scope decision (front-gate: does this need its own mechanism?) — NO, it rides existing

Channel membership is already **declaration-driven**: `_SUBSTRATE_CHANNELS` is a tuple of
`ModalityChannel(tag, read_states, read_ranges)`, and the `world` channel
(`ModalityChannel(WORLD_TAG, _read_world_states, _read_world_ranges)`) reads exactly the sensors
whose body-YAML `modality:` is `world` (`_read_declared_modality_states`, no hard-coded roster).
So the split needs **no new bus/bridge/bio-system**: re-tag sensors to sub-modalities in the body
YAML and add one `ModalityChannel` per sub-tag. And the NAc fear path already **iterates over all
modality tags** in `active_clusters` (`nac.py` `anticipatory_threat_need`, ~L2121) rather than
keying on `"world"`, so fear books onto whichever sub-channel is co-active when pain fires — i.e.
the threat sub-channel, automatically. This is the design's biggest asset and must be verified,
not assumed (wiring lens).

## Proposed split (from the data + sensor semantics)

- **`world:threat`** — the small-N discriminating channel: `nearest_hostile_dist`, `hostile_count`,
  `y_altitude`, `distance_from_spawn`, `nearest_player_dist` (~5 sensors). This is where the
  danger contrast lives and where Wire-4 fear should key.
- **`world:env`** — the ambient/constant diluters: `light_level`, `time_of_day`, `is_raining`.
- **`world:self`** — world-declared body/pose sensors: `health`, `food`, `saturation`, `oxygen`,
  `xp_level`, `on_ground`, `speed`, `look_pitch`. (Whether some of these should be re-homed to
  interoception is a real question but is **out of scope** here — keep membership within the
  world set, only sub-group it, unless the review argues otherwise.)

## Open design questions (the four-lens review must resolve these)

1. **Grouping principle vs tuning-to-apparatus (confounding / D1).** Per-type is principled; a
   *minimal* threat channel (just the danger sensors) would maximize contrast but risks tuning the
   instrument to make Exp 58 pass. The review decides: is threat/env/self the honest partition, or
   is any grouping that isolates `nearest_hostile_dist` inherently D1-adjacent? The composite
   `min(sep, stab, disc)` bar and a live re-encode are the guards, but the grouping itself must be
   defensible without reference to Exp 58.
2. **Gain per sub-channel (bio-faithful — the sharpest one).** `world ∈ gain_modalities` (A4 cubic
   p=3.0). The bake-off (`docs/limits/l11_sensor_dilution.md`) measured A4 as a *many-sensor* tool
   — stability 0.62 at N=6, WRONG for a small channel. The `world:threat` channel is ~5 sensors, so
   **the very split that concentrates the signal may need gain OFF or a lower exponent there.**
   `gain_modalities` is a flat set of tags today; splitting forces a per-sub-tag gain policy. This
   interacts with Q1: without gain, does `nearest_hostile_dist`'s raw contrast separate?
3. **Threshold per sub-channel.** The `1−k/N` scaled threshold is the plan's secondary arm; at
   small N the fixed 0.85 may already suffice (Slice-1 showed the problem was *mass*, not N). Decide
   whether to touch the threshold at all, or hold it and let the split do the work.
4. **Frozen-centroid preservation.** `world ∈ frozen_centroid_modalities` (`ec.py`). Each new
   sub-tag MUST be added to the frozen set, or the sub-channels start drifting (a new failure mode
   the split would introduce). Verify.
5. **Blast radius — who says `.get("world")` (wiring).** Splitting removes the `"world"` key from
   `_encode_current_clusters`' return. Known consumers to rewire or the split silently no-ops their
   world read: `scripts/survival_world/exp58_run.py` (`_encode_current_clusters(...).get("world")`),
   the geometry probe's own `analyze` (`modality="world"`), and any `INTEROCEPTION_MODALITY`-style
   hard key (`nac.py` ~L2044) / `fold_legacy_cluster_id` / hivemind scrub-merge that assumes the
   world tag. A fix ships with its callers (D43) — grep every `"world"` / `WORLD_TAG` consumer.
6. **Persisted substrate + Exp 56/57 non-regression (regression lens — the C gate).** Re-tagging
   changes the encoding **geometry tag**, which orphans persisted `world` nodes (the `ec.py`
   geometry-mismatch warning + the `maxim substrate invalidate --drop-geometry` migrate path).
   Exp 56/57 taught-wants are world-keyed, so this is the re-baseline the whole L11 line has been
   gating on. The split does not ship until Exp 56/57 are re-measured under it and shown intact (or
   the migration is executed and validated). "Preserves separations" is necessary, not sufficient.
7. **The live re-encode gate (environment / sole build authorization).** Rebuild the classroom,
   re-encode safe vs dark through the SPLIT channels, and confirm the `world:threat` sub-channel
   gives **distinct** safe/dark clusters at the composite bar, past the exact cluster-distinct
   preflight that refused Exp 58. Offline replay through production code nominates; only the live
   re-encode authorizes shipping.

## Build order (after the review folds)

1. Body-YAML re-tag → sub-modalities; add `ModalityChannel` entries to `_SUBSTRATE_CHANNELS`.
2. Per-sub-tag gain policy (Q2) + frozen-centroid set membership (Q4) + threshold decision (Q3).
3. Rewire the `.get("world")` consumers (Q5), incl. `exp58_run.py` and the geometry probe.
4. Two-lens code review (executor + architecture) → fold on the branch before merge.
5. Offline replay through production code (`l11_real_trace_remeasure.py::analyze` idiom) on the
   Slice-1 captured vectors, re-grouped per sub-channel — nominates.
6. Live re-encode past the Exp-58 preflight (Q7) — authorizes.
7. Exp 56/57 re-baseline under the split (Q6) — clears the regression gate. Only then does C ship.

## Deliverable of the review

Four lenses — **confounding** (Q1 grouping/D1, the metric), **bio-faithful** (Q2 gain, Q4 frozen,
reads `docs/agents/bio-memory.md` + the L11 ledger), **wiring** (Q5 consumers, Q6 persisted
substrate, D43), **environment** (Q7 live gate reachable + measurable) — each returns
DO-NOT-BUILD / SHOULD-FIX / NIT into `docs/experiments/rationale/l11-slice2/<lens>.md`; the main
session folds them into this plan and presents the cohesive design before any substrate code.

## Decision (2026-09-15) — do NOT build the split; the block is upstream

The channel-split is rejected: it is verified to make separation worse, it would silently break
fear-write / the bridge / R2 credit, and it would destroy Exp 56/57 taught-wants. The dark=danger
survival want is blocked at the REPRESENTATION+APPARATUS layer, not the encoding grouping. Real
options, for the owner to choose:

- **(A) Fix the apparatus to give a full-range, reliable cue.** Make "safe" genuinely differ from
  "dark" on a sensor that swings a large directional arc — e.g. a surface-lit safe area vs a
  cave-dark danger area so `light_level` moves 15→0. BUT Slice-1 + Exp 58 already found
  `light_level` unreliable underground (skylight contamination); this needs the light-sensing
  issue solved first, or a different full-range cue. Cheapest to try, but may hit the known light
  wall.
- **(B) The upstream substrate fix the ledger already names: set-point-aware neutral / habituation**
  (deferred by `_sensor_embed` decision D1). This makes a sensor's contribution relative to its
  *rest* value, so a small move off baseline rotates the embedding — directly addressing the
  direction problem. This is its own substrate project with its own design + review, larger than
  the split, but it is the mechanism-faithful fix and would help beyond this one experiment.
- **(C) Pivot the survival want to a contingency the sensors CAN separate** — one keyed on a sensor
  that swings full-range (crosses neutral) rather than dark=danger. Accepts dark=danger as
  infeasible on this body's sensor suite and banks it as a representation-limited null.

Recommended sequencing: bank this rejection as the L11-C finding (done), then decide A vs B vs C.
None is started; all are the owner's call.
