# Perception Pipeline Placement — per-stage workload distribution

**Status:** Shell plan, drafted 2026-06-22; corrected 2026-06-22 after a code-grounded review pass (see "Review corrections" below).
**Scope:** Abstraction (placement type + config surface + pinned/placeable model) is the deliverable; per-placement implementation is small and incremental, shipped one cut point at a time. The driving consumer (Reachy sound-localization cradle) is a separate, larger build with real new substrate + motor work — *not* "mostly free reuse" (see honest accounting below).
**Target versions:** 1.1 (placement abstraction + first cut point — all-local Reachy, zero wire); 1.2+ (leader-side segmentation cut point + raw-frame transport, on demand).
**Gates:** None as a release gate. Post-1.0 embodiment research infrastructure.

**Driving use case:** A Reachy Mini sound-localization cradle — camera + 4-source mic, where the agent learns to orient its head toward a sound. The perception pipeline for this has stages with sharply different resource demands (sub-millisecond DSP vs GPU segmentation vs stateful substrate encoding), so *where each stage runs* must be a configurable placement decision, not a frozen split point.

**Depends on:**
- [`runtime/worker_pool.py`](../../src/maxim/runtime/worker_pool.py) — `Origin`, `ProviderPlacement`, `validate_placement_coherence`: the **idioms** this plan borrows (not the runtime mechanism — see Front-gate).
- [`runtime/config_loader.py`](../../src/maxim/runtime/config_loader.py) — `LaneTierPlacement` declarative-twin pattern (`api_key_ref` → resolved-at-load).
- [`runtime/lane_backends.py`](../../src/maxim/runtime/lane_backends.py) — `derive_placement` (empty `()` = derive-from-legacy) is the back-compat move this plan reuses to keep `Percept.to_wire_dict` valid as a default.
- [`simulation/sources.py`](../../src/maxim/simulation/sources.py) — the 4-member `PerceptSource` Protocol (`name` / `next_percept()` / `is_exhausted()` / `capabilities`); pull-based, non-blocking.
- [`agents/perception_agent.py`](../../src/maxim/agents/perception_agent.py) + [`runtime/capture.py`](../../src/maxim/runtime/capture.py) + [`integration/bio_enrichment.py`](../../src/maxim/integration/bio_enrichment.py) — **existing perception infrastructure this plan must reuse, not reinvent** (`PerceptionAgent`, `CaptureManager` background-thread capture, `BioEnrichmentPipeline`). See Front-gate row 5.
- [`mesh_perception_transport.md`](mesh_perception_transport.md) — the transport that moves bytes between placed stages. This plan supersedes that plan's frozen single-cut-point assumption.
- [`similarity/encoder.py`](../../src/maxim/similarity/encoder.py) / [`similarity/ec.py`](../../src/maxim/similarity/ec.py) — `SensorEncoder` (384-dim, hard-tags `modality="interoception"`). Exteroceptive modalities are **new substrate work**, not a drop-in (see "Substrate side" below).
- [`grounded_language_acquisition.md`](grounded_language_acquisition.md) — substrate-primary AUT, the cognition layer that consumes placed percepts.

**Enables:**
- The Reachy cradle running **self-contained on-device** (all stages local, no wire) *and* later **distributing the GPU-heavy vision stage to a leader** without a rewrite — same codebase, config change.
- Any embodiment peer with heterogeneous-resource sensors (a camera-rich peer with no GPU, a mic-array peer, a multi-Maxim training rig).

---

## Review corrections (what the first draft got wrong)

A code-grounded review pass (substrate + systems lenses, 2026-06-22) found three blocking errors in the first draft. They are corrected throughout; recorded here so the diff is legible:

1. **The audio↔visual binding "cheap early win" hypothesis was wrong** — it rested on a misdiagnosis of why [`cross_modal_substrate_binding`](cross_modal_substrate_binding.md) was cancelled. Roy-4 ([`docs/experiments/21_roy_4.md`](../experiments/21_roy_4.md)) cancelled it because the priming and test EC nodes **never co-fire in time** (zero node-ID overlap; clusters firing in near-isolation) — a *temporal co-activation* failure. The 384-vs-768 dimensional mismatch is a **separate, later Roy-5a finding**. Dim-consistency removes only the *algebraic* obstacle; the load-bearing *co-activation* obstacle is untouched and, for audio↔visual, unmeasured. Binding is now a flagged open question, not a planned slice with a value claim (see "Future direction").
2. **"Rides ~80% on existing cradle machinery" was overstated (~50% is honest).** The cerebellum forward-model path is **dormant** (`cerebellum_modulator_factory` "Dormant since 2026-05-26", zero production callers) — using it requires un-dormant-ing a path that, per CLAUDE.md Principle 2, needs its own earning experiment. The first slice is rescoped to *not* depend on it.
3. **The control-loop latency boundary was misplaced.** The agent loop is idle-gated and LLM-blocking (up to ~120s awaiting a proposal), not a steady 2–30 Hz consumer. "Localize, then decide" closes fine through the percept→cognition path; "smooth continuous orienting" cannot and must close in a reflex/motor thread. The first slice targets discrete orient steps, not continuous tracking.

Plus should-fixes folded in: `SensorEncoder` is not a modality-agnostic drop-in; reuse `PerceptionAgent`/`CaptureManager`/`BioEnrichmentPipeline` rather than a greenfield executor; `PerceptContext` is frozen so azimuth rides in `metadata`/`SensoryTag`; background-DSP→sensor writes need the explicit lock discipline CLAUDE.md mandates.

---

## Correction this plan records (in the transport plan)

[`mesh_perception_transport.md`](mesh_perception_transport.md) shipped its 1.0 prep with a **frozen single cut point**: "peer always segments, ships event-shaped percepts; raw frames never cross the wire." That promoted a sensible default to a near-invariant — the mirror-image of the mistake [`lane_capability_placement_split.md`](lane_capability_placement_split.md) corrected for LLM lanes (conflating *what the work needs* with *where it runs*).

**The fix:** the perception pipeline is a sequence of stages with heterogeneous resource demand; *where each stage runs* is a placement decision per stage. The raw-frames ban demotes from invariant to **default-with-opt-out**. Nothing shipped needs undoing — only the transport *prep* landed; `to_wire_dict` stays valid as **one cut-point payload** (post-segmentation), and other cut points add their own `_format_version`'d payloads additively.

---

## Front-gate scope pressure (Principle 3)

**Question:** does per-stage placement need its own mechanism, or can it ride on existing infrastructure?

| Candidate | Verdict |
|---|---|
| 1. Reuse lane-placement **idioms** (frozen dataclass + CC3 `extra` hatch + `__post_init__` collision guard; coherence-validated at the producer boundary, permissive at runtime; declarative twin with ref→resolved-at-load; `derive_placement` empty-means-legacy) | **Sufficient — borrow.** Hard-won, proven patterns. Copying the shape (not the type) keeps perception consistent with the codebase and inherits the freeze discipline. |
| 2. Reuse lane-placement **runtime composition** (ordered tuple = failover alternatives, compiled onto `LLMRouter.provider_priority` via `_inject_placement_tail`) | **Insufficient — do NOT share.** Lane placement is a *failover list* (first healthy provider wins). A perception pipeline is a *sequence of stages run across nodes* — no failover, no router to compile onto. The compile step is meaningless here. |
| 3. Model placement with the LLM `Origin{LOCAL,CLOUD,PEER}` enum verbatim | **Partial.** The str-Enum idiom transfers; the *values* differ (perception placement targets a node identity/role, plus the "pinned by physics" constraint the LLM enum never had). Build a perception-specific target; borrow the enum shape. |
| 4. Let the transport's fixed cut point stand and just relax the raw-frames ban | **Insufficient.** Relaxing one rule without the placement abstraction re-creates the same frozen-assumption brittleness next time a different cut point is wanted. |
| 5. **Build a new "pipeline executor" for running stages** | **Mostly insufficient — reuse existing perception infra.** `PerceptionAgent` already "observe → classify → publish structured Percept to bus"; `CaptureManager` already runs a background capture thread with a single-worker publish ThreadPool; `BioEnrichmentPipeline` already runs ordered enrichment stages. The localized-source stage rides on these. The genuinely new part is small: the *placement resolution* (which node each stage targets) + the sensor front-end, not a new stage-runner. |

**Verdict:** **new placement type, borrowing config/type idioms; the stage-running rides on existing perception infra.** Per-stage placement is a *sibling* to lane placement that reuses the config/type discipline but owns its composition semantics (pipeline, not failover) and adds the pinned-by-physics constraint. This is the "typed-placement-per-purpose" extension of the repo's existing "typed-transport-per-purpose" invariant.

**Specific reason the placement type must exist separately:** failover-composition and pipeline-composition are different graphs. Lane placement answers "which one of these interchangeable providers serves this call?"; perception placement answers "on which node does each non-interchangeable stage run, given some stages are physically pinned?" Forcing them under one type makes the failover-vs-pipeline ambiguity load-bearing.

---

## The stage model

The perception pipeline is an ordered DAG of stages. Each stage carries a placement target; each stage is either **pinned** (placement fixed by physics) or **placeable** (placement is policy).

```
stage                    pinned?      why
─────                    ───────      ───
capture (cam + 4-mic)    PINNED       raw high-rate stream lives at the sensor
sub-ms DSP / ITD-TDOA    PINNED       needs raw sample-level timing (sub-ms);
                                      cannot survive an event-shaped percept tunnel
segmentation / STT       PLACEABLE    GPU-heavy, latency-tolerant — the movable middle
feature / sensor encode  PLACEABLE    cheap; usually co-located with the substrate owner
substrate (EC/ATL)       PINNED*      single-owner stateful — the substrate owner
cognition (NAc/loop)     PINNED*      co-located with the substrate
```

**The pinned/placeable distinction is the genuinely new concept** lane placement did not have. `*`The substrate/cognition pin is to "the substrate owner," which in the self-contained Reachy case **is the Reachy itself** — pinned does not mean "the leader," it means "wherever the single substrate owner is." A coherence check at the producer boundary rejects a config that tries to place a pinned stage elsewhere.

**What crosses the wire is determined by the cut point** — the boundary between a stage on node A and the next stage on node B. That boundary's payload is whatever the upstream stage emits:
- cut after segmentation → event-shaped percept (today's `to_wire_dict`).
- cut after capture (segmentation on a GPU leader) → compressed frame (the 1.2+ frame-transport path, under a size cap).
- cut after sensor-encode → encoded features (substrate placed remotely — uncommon). Note: "encoded features" here are pre-substrate feature vectors, **not** the EC embedding — this does not violate the transport plan's "embedding is NEVER on the wire" invariant, which is about the leader-owned EC/ATL substrate output.

---

## Verified reusability map

| Surface | Lane source | Perception action |
|---|---|---|
| Placement-target enum | `Origin(str, Enum)` | **Build** a perception-target type (node identity/role); borrow the str-Enum + `__str__` idiom. |
| Placement value type | `ProviderPlacement` | **Build** `PerceptionStagePlacement`; borrow the frozen + `extra` CC3 hatch + collision-guard shape. |
| Coherence rules | `validate_placement_coherence` | **Build** perception coherence (pinned-stage override rejected; placeable stage needs a reachable target); borrow validate-at-boundary / permissive-at-runtime. |
| Declarative twin | `LaneTierPlacement` (`api_key_ref` → resolved) | **Build** a `config.json::perception` twin; borrow unresolved-ref → resolved-at-load. |
| Back-compat default | `derive_placement` (empty = legacy) | **Reuse the pattern** so an unconfigured pipeline = today's default (peer segments, ships event percepts), byte-identical. |
| Stage running | `_inject_placement_tail` onto router | **Do NOT reuse — and do NOT build greenfield.** Ride on `PerceptionAgent` / `CaptureManager` / `BioEnrichmentPipeline`. |
| Wire payload | n/a | **Extend** `Percept.to_wire_dict` family with cut-point payload types, each `_format_version`'d. |

---

## Substrate side: exteroceptive modalities are new work

`SensorEncoder` (`similarity/encoder.py`) is real and fixed-384-dim, and `_sensor_embed` is mechanically modality-agnostic — but it is **not** a drop-in for exteroceptive sensors:

- It hard-tags every output `modality="interoception"` (`encoder.py:559`), and EC's `pattern_complete_or_separate` only compares **within** a modality (`ec.py:373`). Audio-azimuth / visual-region nodes tagged `"interoception"` would pool into the *drive* cluster space. The encoder's own docstring (`encoder.py:499-501`) says a `"sensor"` umbrella tag for exteroceptive surfaces is "a separate concern" — **it does not exist yet**.
- `"interoception"` is in `frozen_centroid_modalities` (`ec.py:211`) — frozen-prototype semantics, possibly wrong for continuous exteroceptive signals.
- `_normalize_value` (`encoder.py:405`) assumes `[0,1]`/`[-1,1]` ranges; raw azimuth needs new normalization.

**Scope:** adding exteroceptive modalities = a new modality tag (`"sensor"` or per-modality `"audio"`/`"vision-pattern"`) + EC routing + normalization + a frozen-vs-drifting decision (touches the EC centroid-drift lesson). This is explicit new substrate work, sequenced as its own commit, not assumed free. **Verify the dim before assuming anything cross-modal** (Q6).

Note: `PerceptContext.Modality` already includes `"audio"` (`percept_context.py:72`), so the *percept* side is partly there; the gap is the *substrate-encode* side.

---

## Driving consumer: the Reachy Mini sound-localization cradle

The earlier A-vs-B fork (self-contained agent vs. thin sensor peer) **dissolves into a continuum** under placement: ITD-DSP is pinned to the Reachy by physics; everything else is a placement knob. Start all-local (zero wire); move the GPU vision stage to a leader later without a rewrite.

### First slice — honest accounting (~50% reuse)

Audio-only, single source, no vision, no binding. **Discrete orient steps, not continuous tracking.**

**Reuses existing machinery (~half):**
- **Drive/pain cascade** — "centered" is a `HomeostaticDriveSpec` set-point; azimuth deviation → discomfort → `evaluate_failures` → `_publish_drive_pain` → NAc. (`sem.py`, `body.py`.)
- **NAc credit on discrete affordances** — crediting `turn_left`/`turn_right` *discrete* affordances by drive-pain reduction is the standard discrete-affordance credit path, not a novel continuous-control structure.
- **The observe→publish *skeleton* + percept bus** — borrow `PerceptionAgent`'s observe→classify→publish structure and `CaptureManager`'s background-thread + single-worker publish ThreadPool idiom. The localized-source reading rides in `Percept.metadata` (PerceptContext is frozen), modality `"audio"`. **Caveat (don't overclaim this as drop-in reuse):** `BioEnrichmentPipeline` is *text-only* (a thalamic relay for text — keyword extraction, no slot for a scalar azimuth), `PerceptionAgent`'s publish path is vision-detection/transcript-shaped, and `CaptureManager`'s audio path is *mono* (no multi-channel/per-mic raw access exists anywhere in the codebase today). So: borrow the threading/publish *idioms*; **build new** multi-channel capture + a new audio-percept construction path. This is the substance behind Q7; it nudges the audio-path reuse below 50%.

**Genuinely new (the other ~half — must be built, not assumed):**
- **A real-time audio front-end**: 4-mic capture + ITD-TDOA → a scalar `azimuth` + `centeredness` reading. Runs in a background DSP thread (pattern: `CaptureManager`), writing the latest value under a `threading.Lock` (the RMW discipline CLAUDE.md mandates); a `PerceptSource.next_percept()` samples the latest value — pull-per-tick, non-blocking.
- **Two discrete orient affordances** (`turn_left`/`turn_right` by a fixed yaw step) on the body — the infant body has *no* orienting affordance today; `reachy_mini.yaml` has `head_yaw` to model from.
- **An exteroceptive substrate modality** for the azimuth sensor (the substrate-side work above).
- **No world-model needed on real hardware**: after a head turn, the mic array **physically re-measures** azimuth — the loop closes through re-measurement, *not* through a `self_effect` static delta (which cannot express a world-coupled update). This is *only* a problem in pure sim; on the Reachy the world-coupling is free. (A sim harness would need a world-coupling rule; the hardware path does not.) **BUT re-measurement only moves the hidden assumption, it doesn't remove it:** it yields a signal only if the sound is still present on the *next* cognition cycle. Combined with the per-step latency below, the first slice requires a **sustained or repeating sound source** (or a buffered last-known-azimuth that persists across cycles); a single transient clap is gone before a discrete orient step completes and is unrecoverable through the LLM-gated path.
- **`tick_vital_drift` auto-recenters homeostatic sensors every tick** (called at the top of `evaluate_failures`, `body.py`), pulling any homeostatic-drive sensor toward `set_point`. For a *world-set* azimuth sensor this silently fabricates "comfort" between re-measurements. The sensor read MUST overwrite `vital_metrics[azimuth]` each tick before evaluation — a genuinely-new wiring detail, not free reuse of the drive cascade.

**Explicitly NOT in the first slice:**
- **Cerebellum forward model.** It is dormant (`cerebellum_modulator_factory` zero production callers) and predicting an *exteroceptive* sensor (set by the world, not the body's effector) is an untested causal structure. Un-dormant-ing it is a separate, optional extension that needs its own earning experiment — not a first-slice dependency.
- **Continuous/smooth orienting control.** Cannot close through the LLM-gated cognition path; if smooth tracking is ever wanted it closes in a reflex/motor thread (like the existing Reachy reactive path), a separate design.

**First-slice success criterion:** over a session, presented with a **sustained or repeating** sound source, the agent's *discrete* orient choices increasingly reduce azimuth-error (avoid drive-pain) — learned via NAc from embodied pain, no LLM in the action path. (The head turns toward sound, in steps.) **Honest cost:** each orient step is one full LLM-gated cognition cycle (seconds to minutes on a local model), so convergence is *many* cycles; this is a slow, deliberate orienting demo, not real-time tracking. Homeostatic drives emit pain only (no positive reward on recovery), so the learning signal is strictly pain-*avoidance*.

### Future direction (NOT a scoped slice): audio–visual registration

Binding "this sound source" to "that visual object." This is genuinely novel research and **not a cheap early win**:
- The blocker that cancelled [`cross_modal_substrate_binding`](cross_modal_substrate_binding.md) was **temporal co-activation never occurring**, not dimensionality. Dim-consistency (both exteroceptive sensors at 384-dim *if* they share a modality space — unverified) is *necessary but not sufficient*.
- Before any binding work, run a **Roy-4-style co-activation instrumentation pass** (`MAXIM_EC_TRACE_ACTIVATIONS=1`) on the actual audio↔visual trajectory to measure whether the two modalities' EC nodes ever co-fire. If they don't, binding is dead on arrival regardless of dims.
- `cross_modal_substrate_binding` carries a cancelled/"do-not-resurrect" disposition; reviving the mechanism for a new modality pair needs a **fresh plan**, not a dimension argument. JEPA ([`jepa_cross_modal_alignment`](jepa_cross_modal_alignment.md)) is the path if a learned projection is needed.

---

## v1 scope cuts (explicit non-goals)

- **No leader-side segmentation / no raw-frame transport in the first cut point.** First cut point is all-local Reachy (zero wire). Frame transport is the 1.2+ second cut point, built when a no-GPU-on-Reachy wall is actually hit.
- **No per-stage failover.** A placeable stage has one placement target. "Run segmentation on the leader, fall back to local" is a future extension — the point where the failover-vs-pipeline line would be re-examined.
- **No continuous/smooth orienting control through the cognition path** (see first slice).
- **No cerebellum forward model in the first slice** (dormant; separate earning experiment).
- **No audio-visual binding as a committed deliverable** (open research question, gated on a co-activation measurement).
- **No bidirectional perception, no dynamic runtime re-placement, no heterogeneous-hardware placements within one node.**

---

## Open design questions (answer in 1.1 pre-implementation review)

1. **Placement target type.** Node identity (`mesh.yml` name) vs. role vs. LLM-style `Origin`? Lean: node identity for placeable stages; a `SELF`/`SENSOR`/`SUBSTRATE_OWNER` symbolic target for pinned stages that resolves to a concrete node at construction.
2. **Where does the stage DAG live?** `config.json::perception` (declarative) vs. body-YAML vs. code. Lean: declarative `config.json` twin for *placement overrides*, DAG defined in code (stages are not operator-authored).
3. **Pinned-override rejection: warn or fail?** Lean: fail loud at the producer boundary (consistent with `validate_placement_coherence`).
4. **Cut-point payload registry — ownership.** The wire payload/envelope (how cut-point payloads are discriminated, e.g. a `cut_point` tag) is **owned by [`mesh_perception_transport.md`](mesh_perception_transport.md)**, not this plan — to avoid the "merge before multiplying" trap of two plans building the same registry. This plan owns only the cut-point *selection policy* (which boundary the placement implies); the transport plan owns what crosses it. Align the two when the transport ships.
5. **Exteroceptive modality semantics.** New `"sensor"` umbrella tag vs. per-modality tags; frozen-centroid vs. drifting; normalization for azimuth. (Touches the EC centroid-drift lesson.)
6. **Verify the dim-consistency premise** before *any* cross-modal reasoning — confirm audio + visual exteroceptive sensors actually encode at the same dim in the same modality space. Do not assume.
7. **Reuse boundary for the stage-runner.** Exactly which of `PerceptionAgent` / `CaptureManager` / `BioEnrichmentPipeline` carries the localized-source stage, and what (if anything) genuinely needs new code beyond placement resolution + the sensor front-end?

---

## Proposed sequencing

### Now
1. **Land this plan + the `mesh_perception_transport.md` correction + README updates** (docs only).

### 1.1 — placement abstraction + first cut point
2. **Pre-implementation review** (Executor + Architecture + the systems/substrate lenses that caught the first-draft errors) answering Q1–Q7.
3. **Commit 1:** `PerceptionStagePlacement` value type + perception-target enum + coherence validator (borrowing lane idioms). Tests pinning the borrowed-shape discipline.
4. **Commit 2:** stage DAG + pinned/placeable model + `derive_placement`-style "unconfigured = today's default." Tests pinning byte-identical default behavior.
5. **Commit 3:** exteroceptive substrate modality for a scalar sensor (new modality tag + EC routing + normalization + frozen/drift decision). Substrate-only; no robot needed; testable in isolation.
6. **Commit 4:** Reachy audio front-end — 4-mic capture + ITD-TDOA in a background thread (`CaptureManager` pattern, locked RMW) → `azimuth` sensor → `PerceptSource` → `Percept.metadata`. Reuses `PerceptionAgent`.
7. **Commit 5:** two discrete orient affordances on the body + drive-pain set-point wiring + NAc discrete-affordance credit. First-slice success criterion above.
8. **A future experiment doc** (`docs/experiments/NN_reachy_sound_localization.md`) pre-registers the orienting-learning measurement when the rig is ready.

### 1.2+ — on demand
9. Leader-side segmentation: frame-transport cut point + payload registry, built only when a real no-GPU-on-Reachy need arises.
10. (Research, gated) audio↔visual co-activation measurement → binding only if it passes → JEPA if a learned projection is needed.

---

## Re-check triggers

- **A second placement consumer arrives** (Minecraft adapter, second robot, training rig) — abstraction proven by parallel use.
- **The failover-vs-pipeline line is challenged** — if someone wants per-stage failover, re-examine whether that's a placement extension or a different mechanism.
- **The exteroceptive-modality dim is measured** (Q6) — updates any cross-modal reasoning.
- **`mesh_perception_transport.md` ships its transport** — the cut-point payload registry must align with its envelope.

---

## Related plans

- [`mesh_perception_transport.md`](mesh_perception_transport.md) — moves bytes between placed stages; this plan supersedes its frozen-cut-point assumption.
- [`grounded_language_acquisition.md`](grounded_language_acquisition.md) — substrate-primary cognition that consumes placed percepts.
- [`jepa_cross_modal_alignment.md`](jepa_cross_modal_alignment.md) / [`cross_modal_substrate_binding.md`](cross_modal_substrate_binding.md) — the binding path; cancelled for temporal-co-activation reasons, not dims. A new modality pair needs a fresh plan + a co-activation measurement.
- [`lane_capability_placement_split.md`](lane_capability_placement_split.md) — the capability/placement orthogonality this plan extends to perception (idioms borrowed, runtime mechanism not).
- [`maxim_hivemind.md`](maxim_hivemind.md) — a third typed transport/placement sibling; same playbook.
