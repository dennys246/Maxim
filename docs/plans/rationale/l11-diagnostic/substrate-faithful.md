# L11 world-channel diagnostic — SUBSTRATE / BIO-FAITHFUL lens

**Reviewer lens:** does the diagnostic exercise the REAL encoder substrate the way production runs
it — same embedding equation, same gain, same threshold, same frozen-centroid policy, same ranges —
or does it measure a caricature / re-implementation that answers a different question?

**Verdict:** the diagnostic is the right idea and the read-only tap is achievable over real code —
BUT the plan as written would measure a **fiction** in two places: (a) its own component-4 tooling
(`encoding_bakeoff.py`) is a *re-implementation* of the encoder, contradicting the plan's own
"never a re-implementation" discipline; and (b) it cites *running-mean centroid drift* as the reason
to measure isolated-vs-sequential, but `world` is a **frozen-centroid** modality where that drift
does not occur — so the stated mechanism, and any replay built to it, is wrong for this channel.
One DO-NOT-BUILD (as-specified, fixable by re-pointing tooling), five SHOULD-FIX, four NIT.

Grounding read: `similarity/encoder.py` (`_normalize_value`, `_sensor_embed`, `SensorEncoder`,
`SensorEncoderConfig`), `similarity/ec.py` (`pattern_complete_or_separate`,
`pattern_complete_readonly`, `frozen_centroid_modalities`), `docs/limits/l11_sensor_dilution.md`,
`scripts/encoding_bakeoff.py`, `runtime/agent_loop.py::_read_world_states/_read_world_ranges`,
`bodies/minecraft_player.yaml` (17 `modality: world` sensors, confirmed).

---

## DO-NOT-BUILD

### DNB-1 — The remedy replay reuses a RE-IMPLEMENTATION of the encoder, and its "channel-split" is a stride, not per-type. Measuring the mirror defeats the diagnostic's entire purpose.

**Issue.** Component 4 says: *"Reuse the existing L11 tooling (`scripts/ec_scan_cost.py`, its
bake-off framework) — extend it from synthetic to these live vectors rather than re-deriving."* But
`scripts/encoding_bakeoff.py::_embed_state` is documented as *"Mirror
`similarity/encoder.py::_sensor_embed`"* — a hand-copied second implementation, **not** a call into
production `_sensor_embed`. And its grouping is `_partition = names[i::groups]` (a round-robin
**stride**), whereas the plan and `docs/wiring/cluster-dilution-blocks-situation-fear.md` both
specify **per-type** sub-channels. So a replay built on this tool measures (i) a copy of the encoder
that can silently drift from `_sensor_embed`, and (ii) a "channel-split" that mixes sensor types at
random and bears no relation to the per-type split the C-plan would ship.

**Consequence.** This is the exact failure the plan's component-1 discipline names — *"never a
re-implementation (or it measures a different thing than production)"* — reintroduced by its own
component-4 tooling choice. A remedy that separates the situations under the stride-mirror is not
evidence that the *shipped* per-type split will, and a `_sensor_embed` change (they happen — A4 gain,
range-aware normalization both landed here) would make the mirror stale without any test catching it.
It fully undermines the diagnostic's headline promise to *"measure the REAL contrast … through the
real encoder."*

**Minimal fix.** Two rules, both cheap:
1. The replay computes embeddings through **production** `similarity/encoder.py::_sensor_embed`
   (or better, a read-only `SensorEncoder` path — see SF-2), never a mirror. `encoding_bakeoff.py`
   may be the *harness scaffold* (arm loop, metric) but the embedding call must be the real function.
2. The channel-split arm partitions by **declared sensor type** (the C-plan's intended cut), with
   each sub-channel carrying the real frozen-centroid policy and the real per-sub-channel threshold;
   record the partition explicitly so the replayed remedy == the buildable remedy. A stride partition
   is not a stand-in for a type partition and must not be reported as one.

---

## SHOULD-FIX

### SF-1 — The live baseline is the A4-GAINED world channel, and the menu re-opens arms the bake-off already scored/rejected. Reframe or the record re-derives settled results and presents A4 as untried.

**Issue.** `world ∈ SensorEncoderConfig.gain_modalities` **by default**, so the live world channel is
**already gained at `gain_exponent = 3.0`**. The Exp 58 merge that motivated this plan happened
*with A4 active*. The remedy menu (channel-split / scaled-threshold / gain-exponent / combinations)
re-opens arms the 2026-09-01 bake-off already scored on synthetic data: A2 grouping-only = 0.00 at
N=100; A1 scaled-threshold degrades with N (0.70 at N=100); **A3 threshold+grouping measured WORSE
than A1 alone** (0.62 vs 0.76 @N=50); **A5 gain+threshold collapses to 0.00** ("never combine",
per `SensorEncoderConfig` docstring). The plan's line *"combinations, per the L11 bake-off's finding
that grouping + scaled-threshold interact"* reads the bake-off backwards — they interact
**negatively**.

**Consequence.** As framed, the diagnostic risks (a) re-deriving synthetic results already banked,
(b) presenting A4 as a candidate to "pick" when it is the shipped default that *already failed* on
these situations, and (c) spending the run on combinations the bake-off flagged harmful.

**Minimal fix.** State the live baseline explicitly as **A4-gained world @ θ=0.85** (the config the
merge occurred under). Frame the genuinely-new question narrowly: *these situations merge despite A4
— (1) WHY, per gain-weighted sensor (SF-4), and (2) does anything help on REAL correlated vectors
that the SYNTHETIC bake-off rejected?* The bake-off's scope note is the license for (2) — "real
drives correlate … says which candidate is worth building, not which is validated" — but the record
must cite the prior rejections beside each replayed arm so a re-derivation isn't mistaken for a new
finding, and must not list A5-style gain+threshold combinations as promising.

### SF-2 — Name `pattern_complete_readonly` (D8) as the tap seam, and add the read-only step ON `SensorEncoder`. The naive tap mutates EC.

**Issue.** Component 1 wants cosine + pattern-complete decision *"without mutating EC state (no new
centroid, no running-mean update)."* The seam that does exactly this **already exists**:
`EntorhinalCortex.pattern_complete_readonly` (D8, 1.2 gate 3) — structurally incapable of writing
(no centroid update, no member-count increment, no first-touch geometry stamp). The plan does not
name it, and a naive implementer will reach for `pattern_complete_or_separate`, which mutates **even
for frozen world**: it increments `_substrate_node_counts` (ec.py line ~721) and adopts a first-touch
geometry stamp (~711–715). That is the precise D8 defect ("a recall-shaped caller reaching for the
mutating method is a defect in review").

The embedding half has the same trap: the tap must not call raw `_sensor_embed`, because the
**gain-modality selection** and the **D2 zero-vector-at-rest** handling live in
`SensorEncoder.encode_sensors` (`applied_gain = gain_exponent if modality in gain_modalities`). Raw
`_sensor_embed(sensors)` with default args produces the **ungained** vector — a different space.

**Consequence.** Without naming the seam, the tap either mutates the live EC (perturbing the very
session it measures) or silently encodes the ungained world space (a fictional substrate).

**Minimal fix.** Tap = `pattern_complete_readonly` for the cosine/decision. For the embedding, add a
read-only method **on `SensorEncoder`** (e.g. `embed_readonly(sensors, modality, ranges)`) that
performs the exact `applied_gain` selection + `_sensor_embed` + geometry-tag construction of
`encode_sensors`, but skips the delta stash, provenance stamp, NAc eligibility, and
`register_substrate_node`. Reusing the class keeps the gain/range/tag logic single-sourced (no
re-implementation). Do NOT invent a parallel encoder class (see NIT-3).

### SF-3 — Component 3 cites the WRONG mechanism: `world` is frozen-centroid, so there is no running-mean drift. Measure-both is still right, but the sequential path must be frozen-faithful.

**Issue.** The plan justifies isolated-vs-sequential by *"running-mean centroid drift (bio-memory
brief invariant)."* But `ECConfig.frozen_centroid_modalities = {"interoception", "audio", "world"}`
— **`world` is frozen** (plan decision D6, deliberately frozen "before the channel's first caller
exists"). On completion the frozen path skips the running-mean update entirely (ec.py ~720–728);
the centroid never moves. The drift the invariant guards against **does not occur for this channel.**

The genuine sequential hazard for a *frozen* modality is different and still real: (a) the **first**
embedding to reach a node fixes the prototype (arrival order matters), and (b) a **separation
cascade** — once situation A allocates a cluster, situation B completes-or-separates against A's
*frozen* prototype, so whether they merge depends on which arrived first and how many samples of
each. Isolated (fresh EC per situation) cannot see this; sequential (one EC over the stream) can.

**Consequence.** If the implementer builds the sequential replay to the stated reason, they may run
it with running-mean semantics (unfrozen), measuring a world substrate that does not exist —
inverting the very "measure the real thing" goal. And the *interpretation* of any
isolated≠sequential gap would be mis-attributed to drift rather than arrival-order/cascade.

**Minimal fix.** State that `world` is frozen-centroid; keep the sequential replay frozen
(`frozen_centroid_modalities` must include the world tag / sub-channel tags — as `encoding_bakeoff`
already does via `frozen = frozenset(f"ch{j}"...)`); and describe the isolated-vs-sequential check as
detecting **first-touch prototype + separation-cascade** order effects, not running-mean drift. The
invariant (always measure both) still applies — for the right reason.

### SF-4 — The per-sensor contribution metric must be GAIN-WEIGHTED, or it mis-attributes which sensors carry the contrast.

**Issue.** Component 3's *"which sensors differ between safe and dark, by how much (normalized)"* is
computed on normalized `_normalize_value` deltas. But in the gained world space each sensor's
contribution to the summed vector is `w = (|v − 0.5|·2) ** 3.0` (`_sensor_embed`, `gain_exponent`
path). A sensor whose normalized value *moved* but sits near the neutral 0.5 contributes ~nothing;
a sensor at an extreme dominates. So a raw-normalized per-sensor diff does not tell you what the
**cosine** actually sees.

**Consequence.** This is very likely where the answer hides. The Exp 58 discriminators —
`y_altitude` (Δ12/128 ≈ 0.09 of range → normalized value near 0.5 either side) and
`nearest_hostile_dist` (partial swing) — sit **near neutral**, exactly where the p=3.0 gain
**silences** them. A raw-normalized report would flag them as "carrying contrast"; the gained
encoder throws their contribution away. Reporting normalized deltas alone would mis-diagnose the
mechanism and could send the C-plan after the wrong remedy.

**Minimal fix.** Report per-sensor **gain-weighted contribution** (`w·(1−v)·basis_low + w·v·basis_high`
magnitude, or at minimum the `w` weight beside the normalized delta) so the record shows both "the
sensor moved" and "the gained encoder could/couldn't see it." This distinction — present-but-silenced
vs genuinely-absent — is the plan's own stated question and it cannot be answered without the weight.

### SF-5 — Capture through `_read_world_states` + `_read_world_ranges`; the declared ranges are part of the design under test.

**Issue.** Component 2 captures "each raw 17-sensor `world` vector via the bridge." The faithful
seam is `agent_loop._read_world_states` (values) **and** `_read_world_ranges` (declared `(lo,hi)`
from `minecraft_player.yaml`). Ranges are not incidental: `_normalize_value` maps `(v−lo)/(hi−lo)`,
so they set where each sensor sits relative to the gain's neutral 0.5. The L11 re-measure prereg
found this is *load-bearing*: rest-at-extreme range declarations made the gained background maximally
loud (event cos 0.926 — blind) until ranges were re-centered so REST sits at neutral (0.747 vs
ungained 0.960). Wrong or invented ranges → wrong gain weights → a fictional geometry.

**Consequence.** A capture that hand-supplies ranges, or omits them (legacy range-blind fold), does
not measure the live channel; it measures whatever ranges the harness chose.

**Minimal fix.** Pull both from the live executor exactly as the loop does (`_read_world_states` /
`_read_world_ranges`), record the ranges in the decision record as part of the measured config, and
flag any sensor whose reading rests far from neutral 0.5 under its declared range (it will dominate
the gained sum regardless of contrast — the re-measure's exact finding).

---

## NIT

- **NIT-1 — `pattern_complete_readonly` still touches `_geometry_mismatch_seen` + logs.** Its geometry
  check calls `_note_geometry_mismatch`, which mutates a *diagnostic* set and can WARN. It is
  substrate-read-only (the code says so), and on a fresh diagnostic EC seeded with freshly-registered,
  same-geometry nodes it won't fire — but if the tap runs against the LIVE session EC it may emit
  warnings. Note this in the plan as "diagnostic-state only, not substrate mutation" so it isn't
  mistaken for a perturbation.

- **NIT-2 — Sensor count: reconcile "17" vs the live median 16.** `minecraft_player.yaml` declares
  **17** `modality: world` sensors (confirmed), but the L11 live re-measure recorded a *median of 16*
  (some read absent/None-gated per tick). Both the plan and the wiring doc say 17; state "17 declared,
  ~16 live median" so the record's per-sensor table isn't read as dropping one.

- **NIT-3 — Naming: keep the new surface ON the canonical classes.** The plan renames no bio-system
  (no `NucleusAccumbens`, no lane-tier drift) — clean. Just ensure the read-only tap is a method on
  `SensorEncoder` / `EntorhinalCortex`, not a new `TelemetryEncoder`/`WorldChannelProbe` that
  duplicates `_sensor_embed`. A parallel class is a re-implementation by another name (see DNB-1).

- **NIT-4 — Surface the gained embedding NORM per sample (D2 edge).** Under gain, a world reading with
  most sensors near neutral approaches the **zero vector** — `encode_sensors` returns `None` by
  design (D2 designed-rest), and cosine on a near-zero-norm vector is ill-conditioned. The telemetry
  should report each sample's gained-embedding norm and flag near-degenerate ones, rather than print a
  cosine computed on a vector the live path would have refused to encode.

---

## What the diagnostic will most likely find (bio-faithful prediction, for the record)

The situations merge because the discriminating world sensors (`y_altitude`,
`nearest_hostile_dist`) swing only **partially** and therefore sit **near the gain's neutral 0.5**,
where the p=3.0 A4 gain — the mitigation itself — **silences** them, while the many static
world sensors (which rest away from neutral under their declared ranges) dominate the gained sum.
If so, the honest remedy is **not** "sweep the gain exponent" (a lower exponent re-admits the N=17
dilution A4 exists to fix; a higher one silences the contrast further) and **not** the
stride-grouping the bake-off already rejected, but either a **per-type channel-split that isolates
the depth/threat sensors into their own small-N sub-channel** (real per-type, not stride) or a
**set-point-aware neutral** so a partial-but-meaningful swing is not treated as "at rest" — the
latter explicitly deferred by plan decision D1 in `_sensor_embed`'s docstring and therefore a
C-plan design question, not something to improvise in the replay. The diagnostic should be built to
be able to SEE this (SF-4 gain-weighted contribution is what distinguishes the two remedies), which
is the whole reason this lens rates SF-4 a fix and not a nit.
