# L11 Slice-2 (channel-split) — BIO-FAITHFUL design-review lens

Reviewer lens: does the per-type `world` channel-split respect how the real
`_sensor_embed` → EC substrate actually works, or does it break the mechanism's
semantics? Reads `docs/plans/archive/l11_slice2_channel_split.md`,
`docs/experiments/data/l11_geometry_2026-09-15.json`, `docs/agents/bio-memory.md`,
`docs/limits/l11_sensor_dilution.md`, and the code
(`similarity/encoder.py::_sensor_embed`, `::SensorEncoderConfig`,
`similarity/ec.py::pattern_complete_or_separate` / `::ECConfig`,
`runtime/agent_loop.py::_SUBSTRATE_CHANNELS` / `::_encode_current_clusters`,
`decisions/nac.py::anticipatory_threat_need` / `::note_active_clusters`,
`proprioception/pain_bus.py::create_pain_cluster_fear_subscriber`).

Verdict: **DO-NOT-BUILD** the substrate re-tag/config surface (plan build steps
1–4) until the cheap offline re-group of the already-captured Slice-1 vectors is
run first and clears 0.85. Two DO-NOT-BUILD findings and two SHOULD-FIX below;
hand-worked numbers against `_sensor_embed` are shown so the wiring/environment
lenses and the main session can check them.

---

## DO-NOT-BUILD 1 — A4 gain is a MAGNITUDE tool; the L11 limit is a DIRECTION problem. Worked against `_sensor_embed`, the proposed threat channel does NOT clear 0.85 — it is WORSE than the full channel.

**Issue.** The plan's remedy premise (plan §"What the data says") is: constant
full-weight sensors (`light_level` w=1.0, `time_of_day` w=0.77, both Δ0) drown
`nearest_hostile_dist`; isolate the discriminators into a small-N channel and the
contrast dominates. Worked through `similarity/encoder.py::_sensor_embed`, this is
only half true, and the missing half kills it.

Each sensor contributes `w·((1−v)·basis_low + v·basis_high)` where
`w = (|v−0.5|·2)**3` under A4. Cosine ignores magnitude, so **`w` cannot
separate two states — only the `(1−v):v` ratio (the direction) can.** Gain helps
in exactly one regime: one sensor far from neutral while the rest sit AT neutral,
so the mover's *direction* dominates. It does nothing when the mover's value sits
on the same side of neutral in both states — then `w` scales but the direction
barely rotates, and cosine sees ~no change.

The Slice-1 data is that adversarial case. `nearest_hostile_dist` is
`v_safe=0.1786 → v_dark=0.0871` — **both below neutral** (both lean `basis_low`).
Its own safe-vs-dark direction cosine is
`(0.8214·0.9129 + 0.1786·0.0871)/(|(0.8214,0.1786)|·|(0.9129,0.0871)|) ≈ 0.993`.
The weight nearly doubles (0.266 → 0.563), but that is pure magnitude — invisible
to cosine.

Now compute the whole proposed `world:threat` channel
(`nearest_hostile_dist, hostile_count, y_altitude, distance_from_spawn,
nearest_player_dist`) under A4, treating the SHA low/high bases as mutually
orthogonal (their expectation in 384-d):

- gained safe/dark contributions are dominated by `nearest_hostile_dist`;
  `y_altitude` (Δnorm 0.037, `"moved": false`) and `distance_from_spawn`
  (Δnorm 0.025) barely move and stay same-side; `hostile_count` (w≈0.001–0.003)
  and `nearest_player_dist` (v=0.5, **w=0**) contribute ~nothing.
- summing the per-sensor dot / norms gives **cos(safe, dark) ≈ 0.992**.

That is **worse** than Slice-1's full-channel `a4_gained = 0.977`. Why the split
makes it worse: the full channel's separation was carried in part by the
`moved_but_silenced` set — `saturation` (0.5→0.38), `health` (0.5→0.42),
`on_ground` (0.5→0.44) — which sit at neutral (w=0) in safe and acquire small
dark-only mass in *new* basis directions, rotating dark away from safe. The
plan's partition **banishes exactly those movers to `world:self`**, leaving
`nearest_hostile_dist` as the only real mover in `world:threat`, and its contrast
is single-sided. Concentrating mass on a single-sided discriminator does not
separate it. This is precisely the ledger's teaching
(`docs/limits/l11_sensor_dilution.md` §"The limit, precisely"): *"detection is
recoverable by moving the threshold, discrimination is not."* Discrimination is
about which direction the sum points, and A4 does not touch direction.

**Consequence.** Build steps 1–4 (YAML re-tag, per-tag gain/frozen/threshold,
consumer rewire, code review) all ship before the plan's own offline replay
(step 5). My hand-calc says step 5 fails, so that entire surface is built for a
split that its own nomination gate rejects — the opposite of the plan's own
"build cheapest first" lesson (MEMORY.md `[1-3-build-sequence-approved]`).

**Minimal fix.** Reorder: run the offline re-group **now**, before any substrate
code. The Slice-1 probe already captured the live vectors and
`scripts/survival_world/l11_geometry_probe.py::analyze` (and
`scripts/l11_real_trace_remeasure.py::analyze`) can re-encode them grouped per
proposed sub-channel through production `SensorEncoder` + frozen-centroid EC with
zero new code. Make "isolated `world:threat` gives distinct safe/dark cluster ids
at the composite bar" a **hard precondition** for steps 1–4, not a step-5
afterthought. If it fails (as predicted), the split is not the remedy and the
line returns to the diagnostic's other nominated arm / an upstream fix (see
SHOULD-FIX 4 and DO-NOT-BUILD 2).

---

## DO-NOT-BUILD 2 — The split routes fear AWAY from the to-be-feared cue, and the fear WRITE path is hardcoded to `"world"`.

**Issue (semantic).** The whole 1.3 thesis is a *transferable* want: **dark =
danger** — darkness (an ambient/env cue) PREDICTS harm, so the agent should learn
to fear the dark before a hostile touches it. Contextual fear conditioning is
hippocampus-dependent and is *about the conjunction* — "this dark place at this
time," bound into one pattern. The single `world` cluster IS that conjunction
(`_encode_current_clusters` returns one `world` node binding light + time +
threat + pose), and Wire-4 books fear onto it — bio-faithful contextual fear.

The plan frames "fear about the threat, not the ambient" (plan §Q-block, line 38)
as a WIN. It is the opposite. Splitting decomposes the conjunction into
independent cue clusters and, per the plan's partition, routes fear to
`world:threat` = hostile-proximity. But hostile-proximity is what the agent
**already senses reactively** (the innate `health→threat` need,
`nac.py::anticipatory_threat_need` combines the two by `max`). The *transferable,
anticipatory* content — darkness — lives in `world:env` (`light_level`,
`time_of_day`, `is_raining`), which `world:threat` excludes. So the split turns
contextual fear conditioning into cue fear conditioning and points it at the one
signal that is NOT the want. Even a perfectly separating `world:threat` would
teach "fear when a hostile is near," which is redundant, and would leave darkness
unconditioned.

Worse, `world:env` cannot rescue this either: in the Slice-1 data `light_level`
(0.0/0.0), `time_of_day` (0.0417/0.0417) and `is_raining` (0.5/0.5) are
**byte-identical safe vs dark** — the "safe" chamber is also underground/dark. The
cue that must carry the want has zero contrast in this apparatus. No channel
carving can make a constant cue informative (this reinforces SHOULD-FIX 4 and is
squarely the environment lens's to adjudicate, but it is fatal to the split's
bio-story).

**Issue (wiring, load-bearing for the semantic).** The plan's biggest-asset claim
(plan §Scope, lines 28–32) — "the NAc fear path already iterates over all
modality tags in `active_clusters` rather than keying on `world`" — is true only
for the READ (`nac.py::anticipatory_threat_need` iterates `clusters.values()`).
The **WRITE** path does not:
`proprioception/pain_bus.py::create_pain_cluster_fear_subscriber` hardcodes
`nac.active_clusters(agent_id).get("world")` (pain_bus.py:610) and books fear onto
that single key. After the split there is no `"world"` key
(`_encode_current_clusters` returns `world:threat` / `world:env` / `world:self`),
so `.get("world")` is `None`, the subscriber `return`s, and **fear is booked onto
NO cluster** — the mechanism goes silent. The plan's Q5 blast-radius list omits
this site entirely. This is the D43 / "a fix ships with a caller" shape: the read
generalizes, the write is single-keyed, and the composition breaks in the seam.

**Consequence.** Even granting separation (which DO-NOT-BUILD 1 denies), the
learned want is "fear proximate hostiles" (redundant with the reactive need), not
"fear the dark" (the transferable want the whole line exists to earn) — and, as
shipped, fear is written nowhere at all.

**Minimal fix.** Two parts. (a) Decide, at design time and in the prereg, WHICH
sub-cluster carries the want and rewire the pain subscriber accordingly — for
dark=danger that is the darkness cue, so fear must reach `world:env` (or the
conjunction must be preserved). Booking fear onto `world:threat` does not model
the transferable want. (b) Wherever the subscriber ends up, replace the hardcoded
`.get("world")` with an explicit, reviewed policy over the sub-tags (all of them,
or a named subset), added in the same commit as the split with a guard test — and
add `pain_bus.py::create_pain_cluster_fear_subscriber` to Q5's consumer list.

---

## SHOULD-FIX 3 — A new sub-tag silently falls out of FOUR+ modality-keyed config sets; the plan names only frozen-centroid.

**Issue.** `world`'s semantics are not defined by `_SUBSTRATE_CHANNELS`
membership alone — they are the intersection of several modality-keyed sets. A new
sub-tag (`world:threat`, …) must join every one of them in the same commit or its
semantics silently diverge from `world`:

- `ECConfig.frozen_centroid_modalities` (ec.py:428) — miss → running-mean centroid
  drift, the exact collapse `docs/agents/bio-memory.md` / the interoception lesson
  exist for.
- `SensorEncoderConfig.gain_modalities` (encoder.py:684) — miss → ungained
  (linear), which is **both** a different geometry tag (encoder.py:918-930 adds a
  `gain=` field only when gained, so an ungained sub-tag's nodes are mutually
  unreachable from a gained one) **and** back in the raw 1/N dilution regime. This
  is the SILENT one: no cross-pin test guards `gain_modalities` membership.
- `hivemind.merge.DEFAULT_FROZEN_CENTROID_MODALITIES` (merge.py:609) — pinned equal
  to `ECConfig().frozen_centroid_modalities` (`test_hivemind_merge.py:717`).
- `hivemind.merge.SENSOR_MODALITY_THRESHOLDS` (merge.py:635) — pinned to cover
  every frozen-centroid modality (`test_gate2_geometry_and_thresholds.py:70`).
- `embodiment.sensory_streams.DECLARABLE_MODALITY_TAGS` (sensory_streams.py:68) —
  `embodiment/spec.py:612` rejects an undeclared tag LOUDLY at parse (good — this
  one fails fast, the body won't load).

The frozen/threshold trio and DECLARABLE are test- or parse-guarded (fail-loud);
`gain_modalities` is not. The plan's Q4 names only `frozen_centroid_modalities`.

**Consequence.** Forgetting `gain_modalities` gives a sub-channel that is a
different encoding space AND diluted — an unmeasured drift configuration shipped
silently, the precise hazard `docs/agents/bio-memory.md`'s frozen-centroid note
guards against.

**Minimal fix.** Enumerate all five sets in the plan and add each sub-tag to each
in the same commit; add a membership guard test for `gain_modalities` mirroring
the two existing frozen-centroid pins so its omission also fails loud.

---

## SHOULD-FIX 4 — There is no defensible A4 setting for a ~5-sensor channel, and the ledger already measured grouping as the WRONG remedy. Confront that before building.

**Issue.** The plan's Q2 asks the gain question but leaves it open. Bio-faithfully
there is no good answer for `world:threat` at N≈5:

- **Gained (A4 p=3.0):** this is a many-sensor tool. The bake-off measured
  stability **0.62 at N=6** with cluster identity churning
  (`docs/limits/l11_sensor_dilution.md` §Bake-off N=6/8 rows; `encoder.py`
  `SensorEncoderConfig` docstring keeps interoception UNGAINED for exactly this
  reason). Applying A4 to a 5-sensor channel reproduces the regime interoception
  is kept out of.
- **Ungained:** back in the 1/N dilution regime — and 3 of the 5 proposed threat
  sensors (`nearest_player_dist` v=0.5 constant, `hostile_count` Δ0.024,
  `distance_from_spawn` Δ0.025) are constant/near-constant, so ungained they carry
  full constant mass and dilute. Effective N_moving on `world:threat` is ~1
  (`nearest_hostile_dist`).

Moreover the L11 ledger already ruled on this shape:
`docs/limits/l11_sensor_dilution.md` §"What raises the ceiling" lists
*"Per-type modality channels — near-useless alone (0.00 at N=100)"* and
*"[threshold + grouping] measured WORSE than the threshold alone… grouping shrinks
per-channel N."* The ledger keeps per-type channels only "for a **separate**
reason" (letting a sensor declare its own modality), **explicitly not as this
limit's mitigation**. The plan is re-proposing a measured-and-rejected arm under a
new framing ("concentrate the discriminator") without confronting that verdict.

The honest fix the ledger + `_sensor_embed`'s D1 note point at is upstream: a
**set-point-aware neutral** (habituation — a constant background sits at set point
→ weight 0 → silent, without carving channels) would silence `light_level` /
`time_of_day` on principle. `_sensor_embed`'s docstring (encoder.py:580-582)
DEFERS this (D1) and forbids improvising it — so it is not buildable today, but it
is the mechanism-faithful target, and channel-split is a workaround for its
absence. (And note: even set-point neutrality would not help here, because the
contrast location is itself dark — the apparatus, not the encoder, is the binding
constraint; environment lens owns that.)

**Consequence.** Whatever gain is chosen, the plan ships either a small-N
stability collapse or a dilution regime, against a ledger that already measured
grouping as worse than the shipped mitigation — a real risk of re-opening a
settled question with an unmeasured configuration.

**Minimal fix.** Before building, (a) state and justify the per-sub-tag gain with
a live number, (b) reckon explicitly with the ledger's "grouping near-useless /
measured worse" rows and say why THIS grouping (real correlated sensors, N≈1
mover) differs, and (c) record set-point-aware gain (D1) as the deferred
mechanism-faithful alternative the split is standing in for. The single decisive
datum for all of this is the offline re-group from DO-NOT-BUILD 1's fix.

---

## NIT 5 — Encode path is correctly single-sourced (confirm); tag hygiene notes.

- **Single-source: PASS, and it is the design's genuine asset.**
  `_encode_current_clusters` (agent_loop.py:1300) iterates `_SUBSTRATE_CHANNELS`
  and calls the SAME `sensor_encoder.encode_sensors(modality=ch.tag, …)` per
  channel; `_read_declared_modality_states(executor, modality)` (agent_loop.py:1040)
  is generic over the tag; the geometry tag carries `modality` as a field
  (encoder.py:919) so each sub-tag auto-gets its own space. Adding sub-tags is
  data + registry entries, NOT a fork of `_sensor_embed`. Keep it that way — do not
  add per-channel encode logic.
- **Colon-form tags are opaque strings everywhere I checked** (frozen/gain set
  membership, `encoding_geometry_tag`, EC's `_EC_TRACE_MODALITY_TAG_MAP` which
  falls through to `"sensor"`). Fine, but they MUST be added to
  `DECLARABLE_MODALITY_TAGS` or the body YAML won't parse (spec.py:612 — fail-loud,
  which is correct).
- **The proposed `world:threat` membership smuggles in 3 constant/near-constant
  sensors** (`nearest_player_dist`, `hostile_count`, `distance_from_spawn`). If the
  split proceeds at all, the "small-N channel where the contrast dominates" is
  really 1 mover + 4 ballast; either drop the ballast or expect ungained dilution.

---

## Summary for the fold

1. **DO-NOT-BUILD:** run the offline re-group of the ALREADY-captured Slice-1
   vectors before any substrate code; hand-calc predicts `world:threat`
   cos(safe,dark) ≈ 0.992 (worse than the full channel's 0.977) because A4 gain is
   magnitude-only and the discriminator is single-sided, and because the split
   banishes the real movers (saturation/health/on_ground) to `world:self`.
2. **DO-NOT-BUILD:** the split routes fear to `world:threat` (redundant with the
   reactive need) and away from the darkness cue (`world:env`, which is Δ0 here
   anyway) — turning contextual fear into cue fear and defeating the dark=danger
   thesis; and the WRITE path `pain_bus.py::create_pain_cluster_fear_subscriber`
   hardcodes `.get("world")`, so after the split fear is booked nowhere (Q5 misses
   this site).
3. **SHOULD-FIX:** a sub-tag must join 5 modality-keyed sets (frozen-centroid,
   gain, two hivemind sets, declarable); only `gain_modalities` is unguarded — the
   silent one; the plan names only frozen-centroid.
4. **SHOULD-FIX:** no defensible A4 setting exists at N≈5 (gained → 0.62 stability
   collapse; ungained → dilution); the ledger already measured grouping as WORSE
   than the shipped mitigation — confront that with a live number; record
   set-point-aware gain (D1) as the mechanism-faithful alternative the split
   substitutes for.
5. **NIT:** encode path is correctly single-sourced (keep it so); add colon tags
   to `DECLARABLE_MODALITY_TAGS`; the threat membership includes 3 constant
   sensors.
