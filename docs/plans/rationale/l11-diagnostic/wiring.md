# L11 world-channel diagnostic — WIRING lens review

**Verdict: SHOULD-FIX (one DO-NOT-BUILD-as-worded).** The diagnostic's *intent* is
wiring-sound — measure the real live cosine geometry through the real encoder before
building a substrate change. But as worded it points its remedy replay at a
**re-implementation** of the encoder (the bake-off's mirror), and never mentions the
faithful live-capture tool that already exists. Several seams need naming precisely or
the record will be TRUE of the harness, not of the running agent.

Reviewed: `docs/plans/archive/l11_world_channel_diagnostic.md` against the real path —
bridge `scripts/minecraft_bridge/index.js::snapshot` → `MinecraftClient.latest_state` →
`backends/minecraft.py::sync_world_sensors` (`world_set_axis`, clamps to declared range) →
`agent_loop._read_world_states`/`_read_world_ranges` (`_read_declared_modality_states`,
filters `reading_schema["modality"]=="world"`) → `SensorEncoder.encode_sensors(modality="world")`
(gain, `_sensor_embed`, `EC.pattern_complete_or_separate(threshold=0.85)`) →
`_encode_current_clusters` (the seam exp58 already keys on). Existing tooling:
`scripts/l11_real_trace_remeasure.py` (faithful) and `scripts/encoding_bakeoff.py` (a mirror).

---

## DO-NOT-BUILD (as worded)

### W1 — "Reuse the bake-off framework" measures a re-implementation, not production
Component 4 says *"Reuse the existing L11 tooling … its bake-off framework, `scripts/ec_scan_cost.py`
— extend it from synthetic to these live vectors."* But `scripts/encoding_bakeoff.py::_embed`
is by its own docstring a **"Mirror of `similarity/encoder.py::_sensor_embed`"**, and
`_partition(names, groups)` is `names[i::groups]` — a **round-robin stripe**, explicitly "the
per-type-modality stand-in", not the per-*type* grouping the plan's channel-split describes.
Building the remedy replay on that measures (a) a hand-copied encoder that can drift from the
shipped one, and (b) an arbitrary partition **no C-plan would ship**. This is precisely the
"shipped the pieces not the composition" / "verify with the real consumer" failure the plan's
own Discipline invokes.

- **Consequence:** the channel-split verdict would be true of a stripe partition + a mirror
  encoder; the C-plan would then ship a *different* (semantic, per-type) partition against the
  *real* encoder, and the gate would have validated the wrong thing.
- **Minimal fix:** build the replay on the faithful foundation **that already exists and the
  plan never cites** — `scripts/l11_real_trace_remeasure.py::analyze`, which replays captured
  live traces through the **shipped** `SensorEncoder.encode_sensors(modality="world")` against a
  real `EntorhinalCortex`, "register-on-separate protocol — **never a mirror**". Route each
  remedy through real production parameters wherever one exists:
  - **gain-exponent sweep** → `_sensor_embed(..., gain_exponent=X)` (the real function, the real
    parameter);
  - **scaled threshold `1−k/N`** → `pattern_complete_or_separate(threshold=…)` (the real param);
  - **channel-split** → **there is NO production seam today** (`_SUBSTRATE_CHANNELS` has one
    `WORLD_TAG` channel via `_read_world_states`). So its arm is unavoidably a *possibility
    measured offline*. That is a legitimate use of a diagnostic (measure a possibility to gate a
    build), but the decision record MUST pin the exact `sensor → sub-channel` mapping, the
    per-channel gain decision, and the per-channel threshold used, and the C-plan must ship *that*
    mapping verbatim or re-diagnose. A bare "channel-split separates: TRUE" that the C-plan cites
    while shipping a different partition re-arms D43.

---

## SHOULD-FIX

### W2 — Capture at the encoder's INPUT (declared-world subset + declared ranges), not the raw bridge dict
The bridge `snapshot()` emits **18** keys; only the **17** `modality: world` sensors declared in
`src/maxim/_data/components/bodies/minecraft_player.yaml` are the channel, and production feeds
**range-normalized, clamp-equivalent** values (`world_set_axis` clamps to the declared range; and
`_normalize_value` re-clamps through the same range, so a raw `y=150` and a clamped `128` both map
to `1.0` *iff the same ranges are used*). Capturing raw `latest_state()` and feeding all keys would
encode a **different vector than production**: drive keys (`health`, `food`) that are not `world`
would pollute the sum, and any sensor fed without its declared range would silently re-fold through
the legacy range-blind map.

- **Consequence:** the diagnostic would measure an easier/different problem than the live
  classroom — the *exact* false-confidence failure it exists to prevent
  (`cluster-dilution-blocks-situation-fear.md`: "the offline gate validated an easier problem").
- **Minimal fix:** capture/replay at `_read_world_states(executor)` + `_read_world_ranges(executor)`
  (or filter `latest_state` to `backend.world_owned_sensors` and normalize with the declared
  ranges) — which is exactly what the remeasure already does
  (`state = {k: v for k,v in rec["state"].items() if k in ranges}` +
  `_declared_world_ranges()` asserting *declared set == ranged set*). **Enumerate the set at
  runtime; never hardcode 17.**

### W3 — The tap must be read-only along the ENCODER's own state, not just EC's
The plan names "no new centroid, no running-mean update" — but `encode_sensors` also mutates the
**delta-gate stash** (`_last_sensors`/`_last_node_id`/`_last_ranges`) and the **NAc eligibility
trace**, and its `min_delta` gate can *return a cached node id without encoding*. If the tap is
implemented by reusing `encode_sensors`:
- it can **gate out** a subsequent real live encode (or be gated itself) via the shared per-`(agent_id,modality)` delta stash, and
- it writes NAc eligibility — perturbing the very live state a concurrent measurement reads.

- **Consequence:** "read-only telemetry" silently perturbs the thing being measured; tap and live
  encode disagree, and the disagreement is invisible.
- **Minimal fix:** the tap is a **new** method that composes only the pure pieces
  (`_sensor_embed` with the SAME `applied_gain = config.gain_exponent if modality in
  config.gain_modalities else None` resolution, `_normalize_value`, and `EC`'s cosine), reads
  `self.config` for gain/threshold/`gain_modalities` (so it can't drift), and touches **no stash,
  no NAc, no `register_substrate_node`**. Do NOT reimplement `_sensor_embed` (that's W1 again) —
  call it. Concurrency: `MinecraftSyncPump` writes `vital_metrics` on a background thread every
  0.25 s during exp58-style runs, so **snapshot the sensor dict once** and reuse it for both the
  tap and any paired real encode, or a torn read makes them disagree.

### W4 — "Isolated" needs a genuinely fresh EC; `make_fresh_encoder` reuses `aut.bio.ec`
`survival_world/common.make_fresh_encoder(aut)` builds a fresh `SensorEncoder` but wires
`ec=aut.bio.ec` — the **same** EC. Nodes registered while encoding situation A become
pattern-completion targets for situation B, so a "fresh encoder per situation" is not isolated.

- **Consequence:** the isolated/sequential disagreement the plan relies on as a false-read guard
  is contaminated — the "isolated" arm carries sequential leakage.
- **Minimal fix:** the isolated measure should be the **pure tap's direct embedding cosine (no EC
  at all)** — that is genuinely order-free. The sequential measure uses a real EC via the real
  `encode_sensors` path. State this split explicitly; do not use `make_fresh_encoder` for the
  isolated arm expecting EC isolation.

### W5 — Exp 56/57 non-regression is not capturable in B without those apparatuses
Exp 56 runs on `bodies/minecraft_bench` and Exp 57 on `minecraft_bench57` (see
`scripts/exp56/common.py` `BODY_REF` and the offset_x/offset_z note) — **different bodies with
different world sensor sets**, staged by the Exp 56 apparatus (its classroom, its contingency
slots). The Deliverable says the replay "MUST also run on captured Exp 56/57-style situation
vectors"; "-style" hand vectors would repeat the false-confidence trap (W2's shape). Two further
wiring facts:
- channel-split is a **geometry-tag change** (`encoding_geometry_tag` moves when the declared set
  is partitioned / gain per channel differs), which **invalidates persisted Exp 56/57 substrate**
  — "preserves separations" is necessary but not sufficient; the earned nodes need re-encoding.

- **Consequence:** the record could claim "clears Exp 56/57" from proxies that were never the
  earned vectors, or ignore that the winning remedy re-tags and orphans the earned substrate.
- **Minimal fix:** run the non-regression on **real** Exp 56/57 vectors under their standing
  apparatus (memory notes the ~51-min Exp 56 re-baseline is available before this step), **or**
  explicitly sequence the Exp 56/57 non-regression into **C's own re-baseline** and do NOT claim B
  clears it. Resolve the B-vs-C scope the plan flags in its own Q4 rather than leaving it open.

### W6 — Promote "live re-encode under the top candidate" from open question to required step
The plan's Q3 correctly worries that a remedy separating *captured* vectors offline could still
*merge* them live (running-mean / jitter / presentation order / the `min_delta` gate). This is the
live-vs-harness gap and it must be closed, not left conditional.

- **Consequence:** without a live confirmation the decision is TRUE of the offline harness, not of
  the running agent — the failure family the plan opens with.
- **Minimal fix:** for **gain / threshold** candidates this is cheap and must be required — set the
  config and re-encode live through `_encode_current_clusters` / `propose_via_substrate` on the
  running agent (the exact seam exp58 already uses). For **channel-split** there is **no live seam
  in B** (W1), so its verdict is inherently pre-build; the record must label channel-split's number
  "offline replay only, live confirmation deferred to C" and not present it beside the live-confirmed
  gain/threshold numbers as if equivalent.

---

## NIT

### W7 — Bind the numbers to the live capture in the provenance stamp
The Deliverable is a gated decision record the C-plan cites. Reuse exp58's discipline
(`in_process_code_provenance` + `evidence_out_paths_or_exit`, and the `FROZEN["fingerprint"]`
pattern) to stamp, **into the record**: the captured `world_owned_sensors` set, the declared
ranges, the encoder config fingerprint (`gain_exponent`, `gain_modalities`, `pattern_threshold`),
the classroom anchor file, and the code provenance. Without this, nothing connects "these cosines"
to "the live classroom", and the record could be produced from a stale or hand capture — the
provenance half of W2.

### W8 — The world-sensor count is already drifting across docs
`l11_real_trace_remeasure.py`'s docstring and the bridge `snapshot()` comment say **16** world
sensors; the body YAML (`grep -c "modality: world"`) and this plan say **17**. Whichever tool the
diagnostic extends must derive the set at runtime (`backend.world_owned_sensors` /
`_declared_world_ranges()`), never a frozen count, and the drift should be reconciled so the
retirement prereg and the diagnostic agree.

---

## What the plan already gets right (wiring)
- Keying the encode on `_encode_current_clusters` / `propose_via_substrate` is the real consumer —
  exp58 already reads clusters through it, so the diagnostic taps the same seam the agent runs.
- Insisting on "the same code the live path uses; never a re-implementation" is the correct
  instinct — W1/W3 are about *honoring* it, since the cited bake-off violates it.
- Measuring isolated AND sequential (centroid-drift invariant) is the right false-read guard —
  W4 only sharpens *how* to make "isolated" truly isolated.
- Flagging the Exp 56/57 scope (Q4) and the offline-vs-live risk (Q3) shows the author already
  sees W5/W6; the fix is to resolve them in the plan, not leave them as open questions gating a
  build.
