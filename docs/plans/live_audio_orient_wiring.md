# Live audio-orient wiring — DoA into the standard embodied runtime

**Status:** Drafted 2026-07-28 from a four-lens parallel audit (runtime ingestion, audio front-end + sim wiring, hardware abstraction layer, body drives + DN behaviors). Not yet scheduled.

**Goal:** the standard Maxim runtime, embodied on a Reachy Mini, *sees* live sound direction (DoA), *reacts* to it (reflex + learned orient policy), and *reflects* on it (LLM prompt) — wired through a robot-agnostic seam so any `maxim.robots` plugin can provide an audio-localization source, with the infra N-axis-ready (elevation lands later as body-YAML data + one tuple entry, no new schema).

**Companion docs:** [perception_pipeline_placement.md](perception_pipeline_placement.md) (commits 4–5 umbrella; this plan supersedes its stale "background thread + RMW" text — the shipped front-end is pull-per-tick by design), [productive_orienting_affordance.md](productive_orienting_affordance.md) (elevation recipe + the one deferred 2-axis design decision), [docs/behaviors/audio_behaviors.md](../behaviors/audio_behaviors.md) (BL-1..BL-5 reflex-layer blockers), [deferred/transition_based_drive_pain.md](deferred/transition_based_drive_pain.md) (**its revive trigger names this plan**).

---

## Front-gate scope statement (does this need to be its own mechanism?)

**No new mechanism is introduced.** The audit's decisive finding: the hardware-agnostic half already exists and is production-quality — this plan is *wiring*, plus one small capability seam on the existing `RobotController` ABC. Specifically:

| Layer | Component | Status |
|---|---|---|
| Localization front-end | `DoAReader` contract, `AzimuthDoASource`, REST + onboard readers, transport chooser `build_reachy_audio_orienting_source` | **EXISTS** ([audio_localization.py](../../src/maxim/embodiment/audio_localization.py)) — the transport chooser has **zero callers** |
| Body write | `world_set_azimuth` (capability-gated, fail-soft) | **EXISTS** (`audio_localization.py:420`) — only sim-gated callers |
| Substrate routing | `ModalityChannel` split, `AUDIO_TAG`, split encode, `recommend_action(current_clusters=)`, `record_outcome(clusters=)` intero/extero credit routing | **EXISTS, production-wired** (PR #411) — value side empty on live because nothing writes `vital_metrics["azimuth"]` |
| Body declaration | `azimuth` world-set drive (`drift_rate: 0`), 2×2 turn affordances, `listen` | **EXISTS** ([reachy_mini.yaml](../../src/maxim/_data/components/bodies/reachy_mini.yaml)) |
| Prompt reflection | `body_state_summary` / `format_body_state_for_prompt` / acting-coach drive modulation | **EXISTS, fully per-sensor generic** — zero azimuth-specific code needed |
| Multiplexing | `CompositePerceptSource` + `build_audio_composite` | **EXISTS** (sim-attached only) |
| Robot plugin seam | per-robot optional DoA capability | **NEEDS BUILDING** (Stage 1 — the only new surface) |
| Live feed + consumers | poll thread, un-gating §1.16, learned-NAc load, DN reflex | **NEEDS BUILDING** (Stages 2–5) |

Rejected designs stay rejected (do NOT build): raw 4-mic TDOA (XVF3800 chip-level USB limit), background DSP thread inside the source (pull-per-tick is the shipped contract, `audio_localization.py:18-22`), a new `AxisSpec` type or axis config schema (`perception_pipeline_placement.md:127` guardrail — ride the body YAML), reviving the dead `SensorStream` ABC (DoA is a stateless pull, not a stream), continuous/smooth orienting through the cognition path.

---

## The five gaps (from the audit)

1. **No live feed.** `build_reachy_audio_orienting_source` has zero callers; the only `world_set_azimuth` writers are sim-gated (`agent_loop.py:1730` — `if sim.is_sim_mode and aut_mode != "substrate-primary"`) or sim scaffolds (`cradle_mother.py`). On the live robot, nothing writes `vital_metrics["azimuth"]`, ever.
2. **No robot-agnostic capability seam.** The Reachy transport chooser exists, but there is no way for a robot plugin to *advertise* a DoA source. The `RobotController` ABC is contract-frozen at 12 abstract methods (`docs/user/extension_api.md`), so the seam must be additive and optional.
3. **Relief credit is structurally dead on reachy (Gate A — highest-value single fix).** `reachy_mini.yaml` orient affordances write `self_effect: {head_yaw: ±0.3/±0.9}`, but `head_yaw` carries **no drive spec** — `tool_bridge.py:447-451` builds `pre_values` from `self_effect ∩ drive_specs ∩ vital_metrics` = `{}`, so `drive_potential_diff` is never emitted and orient relief credit never flows, no matter how well the sensor is fed. (`base_humanoid.yaml:101,106` uses the working shape: `self_effect: {azimuth: ±0.3}`.)
4. **Live runtime is LLM-primary (Gate C).** `agentic_runtime.py:870` passes no `aut_mode`; cluster reward requires `active_clusters`, which only `propose_via_substrate` populates. Nothing structurally blocks substrate-primary on the live path (plain kwarg, no sim checks inside), but the *standard* runtime needs a reflex-layer consumer (Stage 5) or the mode flag (measured runs).
5. **Drive-pain cadence hazard.** Drive pain is state-based (re-fires every `evaluate_failures` while out of band). It is latent for reachy **precisely because azimuth is world-set at initial 0.0** — this plan is the thing that de-latents it. `transition_based_drive_pain.md` explicitly says it "should land before Track 2 feeds live azimuth."

---

## Stage 0 — Prerequisites (before any feed lands)

**0a. Land [transition_based_drive_pain.md](deferred/transition_based_drive_pain.md) first.** Its revive trigger has fired twice now (the `tick_embodiment_drift` cadence change, and this plan). Once live DoA breaches the azimuth band, `drive:azimuth:discomfort` re-fires per loop iteration on both pain channels; the PainBus channel is unfiltered by B8 delta-attribution, and every bystander action executing while off-center gets blamed. The fix is ~25–40 LOC (latch per-`(entity_path, drive_name)` breach state on `Body`, fire on within→out transition only) but carries validation discipline: re-run Exp 37/38 triage, SEM pain cascade, Exp 42. **Recommendation: hard prerequisite, not parallel** — landing the feed first would validate the orient loop against a pain regime we already know is wrong and plan to change.

**0b. Recalibrate `azimuth.pain_scale` 1.0 → 0.2–0.3** in `reachy_mini.yaml` (the YAML's own `TODO(Track 2)`). At 1.0, a fully off-center sound (intensity 0.9) out-shouts `thermal_throttling` (0.4) and `camera_lost` (0.6) — "off to the side" must not hurt more than "motors overheating." `base_humanoid.yaml` already uses 0.3 and cites this exact mis-scale.

**0c. Fix Gate A: give the orient affordances an `azimuth` self_effect.** Change `self_effect: {head_yaw: ±X}` → `self_effect: {head_yaw: ±X, azimuth: SAME-sign gain·X}` (keep `head_yaw` for motor semantics; add the drive-bearing key). *(2026-07-31 pre-merge review, both lenses independently: an earlier draft of this line wrote `∓gain·X` — that glyph contradicted this stage's own mandatory sign check below and `base_humanoid.yaml`. Under `-1=left/+1=right`, turning left rotates a stationary sound's head-relative bearing toward `+`, so the azimuth delta shares head_yaw's sign. The shipped YAML is same-sign; do not "fix" it toward the old glyph.)* The azimuth deltas use the measured gain (0.57 az/rad post-headfix): normal ≈ ∓0.17, big ≈ ∓0.50 — matching the calibrated decision boundary (≈0.33) the Exp 45c policy was trained against. **Known honest limitation (Gate B):** the intra-execute `drive_potential_diff` credits the *modeled* shift; the *measured* shift arrives at the next DoA re-write, which corrects the value but not the already-booked credit. Acceptable for the first slice (sign is what's credited, and the modeled sign is correct by calibration); the act-now-credit-later pending-map design in `audio_behaviors.md` is the eventual fix and belongs to Stage 5, not here. Sign convention check is mandatory: `turn_left` self_effect on azimuth must move the value toward the set-point for a left-of-center sound under the `-1=left/+1=right` convention (`doa_to_azimuth`).

Regression guards: 0b/0c pin in `tests/unit/test_reachy_orient_body.py` (pain_scale ≤ 0.3; each orient affordance's self_effect carries an `azimuth` key whose sign opposes its direction; boundary math unchanged). 0a carries its own plan's guards.

---

## Stage 1 — The robot-agnostic capability seam

The one genuinely new surface. Design follows the codebase's own conventions (audited):

**1a. `RobotController.get_doa_reader(self) -> DoAReader | None: return None`** — a **non-abstract** method on the ABC ([hardware/controller.py](../../src/maxim/hardware/controller.py)), sited next to `get_audio_stream`, typed against `maxim.embodiment.audio_localization.DoAReader` under `TYPE_CHECKING` (the existing empty `TYPE_CHECKING` block is waiting for exactly this). Rationale, in order of strength:
- The ABC is contract-frozen ("12 abstract methods total") — a 13th `@abstractmethod` breaks every third-party `maxim.robots` plugin. Non-abstract default-`None` is the established convenience shape (`center_vision`, `reconnect`).
- `None` = absent capability is the house idiom (`get_video_stream/get_audio_stream → X | None`; `build_reachy_audio_orienting_source`'s own docstring: "absent capability ⇒ absent source, no dead config").
- Call sites probe with `getattr(controller, "get_doa_reader", None)` so plugins compiled against an older ABC keep working (`selfy.py:399` precedent).
- NOT a `StreamCapability`/`SensorStream`: DoA is a stateless pull, not a stream; `SensorStream` has been a zero-implementation placeholder since it was written.

**1b. `ReachyMiniController.get_doa_reader()`** — onboard (`self._mini` present) → `make_reachy_doa_reader(self._mini)`; networked → `make_reachy_rest_doa_reader(self._resolved_host)`. Prerequisite micro-fix: `connect()` resolves `probe_host` as a local and never stores it — stash it as `self._resolved_host` so the reader is self-sufficient (this also sidesteps the known `selfy.py:254-268` wart where the controller config is built inline, ignoring `robots.yaml`).

**1c. Advertise + enable.** Advertise via `RobotCapabilities.custom` (`"audio_localization"` — the zero-schema-change channel, same as `"antenna_left"`). Enable/disable is the wiring layer's decision via the **free-form `robots.yaml` config dict** (`config: {audio_localization: false}` opt-out; default ON when the capability + a body with an `azimuth` sensor are both present — capability-driven, mirroring the `config.body` / `resolve_body_ref` precedent and the CC10/NH-2 "ride the free-form dict, no typed field" rule). No new `MAXIM_*` env var (config-over-env standard).

**1d. Promote the script-only signal hygiene into the library (BL-3 partial).** `gated_azimuth` (median-of-k + speech-gate smoothing) lives only in `scripts/orient_backbone/live_common.py` — `src/` cannot import it. Promote it (plus `az_bin`/policy-meta helpers if Stage 5 lands in the same release) into `embodiment/audio_localization.py`. The measured DoA characterization justifies it: speech gate fires 23–100% (median 50%) per utterance, so raw single reads are too noisy for credit.

Regression guards: extend `tests/unit/test_multi_robot_extensibility.py` (default returns `None`; a fake controller overriding it is discoverable via `getattr`); `tests/unit/test_audio_localization.py` gains `TestGatedAzimuth` (promoted from script behavior); a `SimulatedController.get_doa_reader` returning a scripted reader pins the seam end-to-end without hardware.

---

## Stage 2 — The live feed (sensor lane; lowest risk, independently shippable)

A DoA poll thread in the live runtime that caches the latest gated reading and world-sets the body sensor. This alone makes `listen` return live direction, the azimuth drive breach, and body_state renderable — no agent-loop edits.

- **Thread:** follows the CaptureManager pattern (shared `stop_event`, registered in the `_stop_agentic_runtime` teardown sweep next to `("_audio_thread", "capture.audio")`). The thread exists to decouple the REST reader's synchronous GET from consumers — the `PerceptSource` non-blocking contract (`sources.py:68-71`, a pinned CLAUDE.md invariant) forbids pulling the REST reader inline. The thread polls `reader()`, applies `gated_azimuth`, and stores `(azimuth, timestamp)` under a lock; consumers read the cache. Poll cadence ~5–10 Hz (the chip converges in 0.23 s; faster is waste).
- **Write:** each fresh gated reading → `world_set_azimuth(executor.embodiment, az)`. Wiring site: `agentic_runtime`, adjacent to `_resolve_body_wiring` / the CaptureManager start gate — gated on (robot seam returns a reader) AND (body declares `azimuth`) AND (config opt-out absent). Absent any of the three: no thread, no log spam — a robot without a mic behaves exactly as today.
- **Multi-axis infra (the elevation-ready part, no wiring):** add `world_set_axis(embodiment, sensor_name, value)` as the generic helper; `world_set_azimuth` becomes a one-line delegate. This is one of the two parameterizations the code's own elevation note blesses (`world_set_azimuth` docstring, `audio_localization.py:422-432`). The poll thread iterates whatever axes the reader supplies (today: one).

**Prerequisite for the user:** `~/.maxim/robots.yaml` must declare `config: {body: bodies/reachy_mini}` (without it the runtime is bodiless and the gate correctly no-ops) — plus `host`/`connection_mode` for the networked topology.

Regression guards: unit test with a fake reader + real `Embodiment` over `reachy_mini.yaml` (thread writes clamp; stale reading not re-written; `drift_rate: 0` keeps the world-set value across `evaluate_failures`); teardown test (thread joins on stop_event); the existing `test_productive_listen.py` covers the read-back.

---

## Stage 3 — The percept lane (react/reflect for the LLM-primary runtime)

The §1.16 audio-orientation block in `agent_loop.py` (escalation tiers, reflex clamp, `_audio_escalate_this_tick`, English rendering via `format_audio_orientation`) is the designed consumer — but it's gated `if sim.is_sim_mode and ...`, and attaching a `percept_source` on the live path would flip `is_sim_mode` across 12 sites (consolidation downgrade, DN shutdown skip, sim logging...).

**Root-cause fix, not the band-aid:** the gate's real condition is "a modality-preserving percept is present," and sim-ness was its proxy. So:

- **3a.** Re-gate §1.16 on `sim.current_percept is not None` (keeping the `aut_mode != "substrate-primary"` exclusion — under substrate-primary the drive/EC path reads the sensor directly and §1.16 would double-write).
- **3b.** Teach `NullSimulationAdapter` to carry `current_percept`: the Stage-2 poll thread, on a *fresh* speech-gated reading, builds `make_audio_percept(az, source="reachy:audio-doa", agent_id=...)` and hands it to the adapter's side-channel slot (same modality-preserving side-channel design the sim uses; the percept never lands in persisted `state.data`). This deliberately does NOT attach a `percept_source` — no `is_sim_mode` flip, no `consolidation="full"` override needed, no DN-shutdown regression. (The `CompositePerceptSource` + `consolidation` override route remains documented as the fallback if a future adapter needs true source multiplexing.)
- **3c.** Reflection: with the sensor fed, `MAXIM_ENABLE_BODY_STATE_PROMPT=1` (the existing Reachy seam at `agentic_runtime.py:439-447`) puts `reachy_mini.azimuth: ... (DRIVE: outside comfort band, ...)` in the prompt and the acting coach names it per-sensor — audited: zero azimuth-specific code needed. §1.16's `format_audio_orientation` adds the English "You hear a sound well to your left" line (closing the raw-numerics Gap 1). The flag stays default-OFF per the pre-registered ablation discipline; flipping it for the Reachy profile is a run-level decision, not a default change.

Regression guards: `tests/unit/test_audio_orientation.py` gains a non-sim-adapter case (NullSimulationAdapter with a carried percept → §1.16 fires; without → block skipped); an `is_sim_mode`-stays-False assertion on the live wiring path; escalation/reflex tier tests already exist and re-cover.

---

## Stage 4 — Load the learned behavior

The bio-stack NAc on the live path is **in-memory only** (audited: `build_bio_stack` passes no `persistence_path`, unlike its hippocampus/ATL siblings) — nothing loads at boot, nothing saves at shutdown. Two changes:

- **4a. Give the runtime NAc persistence** matching its siblings: `persistence_dir/nac.json` in `build_bio_stack`, loaded via `NAc.load_safe` (canonical pattern: `agent_factory.py:558-563`), saved at session end. This is an omission fix, not orient-specific.
- **4b. Import the trained orient policy by MERGE, not replace.** `NAc.load_safe`/`load_state` is a replace — clobbering the runtime NAc's other learning to load orient biases is wrong. The purpose-built tool exists: `hivemind/merge.py::nac_merge` (pure function, Welford-correct, `left_source=`/`right_source=` validated). A one-shot import verb (rides the existing `maxim substrate import` surface, or a small `--import-nac` bootstrap step in the runtime) merges `~/.maxim/orient_live/nac_reachy_flip.json` into the runtime's `nac.json`. Agent-id alignment is already exact (`build_bio_stack(agent_id="reachy")` ↔ the trained policy's `agent_id: "reachy"` — audited, no divergence). The policy-meta sidecar (`bin_boundary`, `gain`, `action_deltas`) travels with the import per the make-the-definition-travel-with-the-data lesson.

Regression guards: round-trip test (merge orient fixture into a populated NAc → both the orient cluster biases and the pre-existing links survive; probe-style argmax per az-bin matches the source policy); `test_persistent_agent_campaign`-style boot-load test for 4a.

---

## Stage 5 — React at reflex speed: the DN audio-orient behavior (largest stage, separable)

Stages 2–4 give the *cognition-path* loop: sound → drive discomfort → substrate/LLM proposes a turn → relief credit. That is seconds-scale. "Focuses on sounds" at animal latency needs the Default-Network reflex layer, and the audit confirms `audio_behaviors.md`'s verdict: **not buildable as "just one Behavior"** — five structural blockers (BL-1..BL-5), all `[generic]` DN plumbing:

1. **BL-1:** new non-blocking `turn_body` motor-dispatch branch that ships an **explicit head matrix** (the head-frame counter-rotation invariant — `head=None` means the mics don't turn; this exact bug faked a sensor pathology for a full session). No 5–8 s blocking sleeps on the DN thread.
2. **BL-2:** un-gate `_process_tick`'s `if not detections: return` for audio-only ticks; push the latest gated azimuth + the reflex NAc into behaviors via the DN setter pattern (`TurnAround.set_head_yaw` precedent), fed from the Stage-2 cache.
3. **BL-3:** `az_bin` / policy-meta helpers promoted to `src/` (done in Stage 1d if co-scheduled).
4. **BL-4:** **isolate the reflex NAc** — reflex-rate `update_cluster_reward` into the shared NAc would leak into `get_agent_tool_biases` and Wire-A would tell the LLM "you've been rewarded for turn_body" from motor activity it never chose. Dedicated `agent_id` (e.g. `reachy:orient-reflex`) or a separate NAc instance seeded from the Stage-4 merged policy.
5. **BL-5:** register orient bearings into an azimuth IOR map so the robot doesn't chase every cough (the current salience/IOR gates pass target-less proposals through unchecked).

Plus the one-way inhibition rule (a voluntary LLM head move suppresses the reflex; never the reverse) and efference copy into body_state. Bearings use az-bin **string** keying via `recommend_action(current_cluster_id=...)` — the Exp 45 policy is usable as-is, no retrain. NH-5: an audio startle at priority 0.95 must habituate or it starves orient.

**Recommendation:** ship Stages 0–4 as one PR-sized arc (the standard runtime sees, feels, learns-from, and reflects on sound; reaction is cognition-speed), and Stage 5 as its own follow-up plan execution of `audio_behaviors.md` — it touches DN internals that deserve their own two-lens round. For measured substrate-only runs meanwhile, `aut_mode="substrate-primary"` on the live runtime works today (audited: plain kwarg, encoder + registry both populated, no sim checks inside).

---

## Elevation / other-audio-percept readiness (infra only — deliberately NOT wired)

What this plan leaves behind for a future non-coplanar-array robot (the Reachy's linear array physically cannot do elevation):

**Already N-axis-generic after Stages 1–3:** `Percept.metadata` (a dict — new keys are additive), `make_audio_percept(metadata=...)` extras channel, `world_set_axis` (Stage 2), `ModalityChannel` registry (a new audio-derived sensor = one body-YAML sensor + membership in the existing audio channel's read fns), `_EXTEROCEPTIVE_ROOT_SENSORS` (a deliberately named tuple — `"elevation"` is one entry), `encode_sensors` multi-key encode, per-sensor-generic body_state/coach rendering, and the body-YAML drive schema (`HomeostaticDriveSpec` needs **no new field** — an elevation drive block is pure data).

**Deliberately still 1-axis (deferred until a second-axis robot exists, per the N=1 rule):** the `DoAReading = (radians, is_speech)` 2-tuple and the ~5 remaining azimuth-named helpers (`doa_to_azimuth`, `reflex_oriented_azimuth`, `OrientingProfile.max_orient_azimuth`, `format_audio_orientation`, the §1.16 `metadata["azimuth"]` read). The 3-step extension recipe and the **one real open design decision** — two independent centeredness drives vs one combined `√(az²+el²)` angular distance — live in `productive_orienting_affordance.md:163-179` and are decided *when a 2-axis body forces it, not now*.

**Guardrail (restated from perception_pipeline_placement.md:127): if an `AxisSpec` type or axis config schema starts to appear, stop and use the body YAML.** "Other audio percepts" (loudness, speech-presence, novelty-of-source) follow the same path: declare the sensor on the body, add it to the audio channel's readers, done — no new types.

---

## Invariants this plan must respect (checklist)

- `PerceptSource.next_percept()` non-blocking — REST GET never inline; poll-thread + cache (Stage 2).
- Head-frame: any body turn that a head-mounted sensor must follow ships an explicit `head=` matrix (Stage 5 BL-1; `tests/unit/test_reachy_head_frame.py`).
- Pull-per-tick front-end; no retry loops in transports; typed errors.
- `maxim/utils/http.py` for all HTTP (REST reader already complies).
- No new `MAXIM_*` env vars for durable config (robots.yaml / config.json); any harness env toggle gets a conftest scrub in the same commit.
- Credit lands as state-conditioned positive relief (`potential_diff` sign), never pain-avoidance (Phase 0b); drive-relief → interoception cluster, operant/direction → audio cluster (`AUDIO_TAG` via the constant, never a bare literal).
- `is_sim_mode` is not flipped on the live path (Stage 3 re-gate instead).
- Declarative files stay operator-written; runtime state in `~/.maxim/` (NAc merge output lands in the runtime's own `nac.json`).
- New frozen dataclasses (if any) pick CC3 path (a) or (b) before merge. Current draft introduces none.

## Validation protocol

1. Offline: full fast suite + the new guards; `--dry-run`-style fake-reader wiring test.
2. Live smoke (`MAXIM_LOG_FILE` JSONL): standard runtime + robots.yaml body decl → speak off-center → verify (a) `vital_metrics["azimuth"]` tracks, (b) drive discomfort appears ≤ pain_scale 0.3 and only on band *entry* (Stage 0a), (c) body_state line + coach mention with the flag on, (d) an orient turn books positive relief credit into the intero cluster and operant credit into the audio cluster (check `consulted_bias_by_modality` telemetry).
3. Behavioral non-regression: Exp 42 triage arm re-run (Stage 0a obligation), Exp 44 cadence caveat check, orient full-path probe (`scripts/orient_substrate/2_full_path_probe.py`) unchanged.
4. Live learned-behavior check: after Stage 4 merge, frozen-policy probe over az-bins from the runtime NAc matches `orient_demo.py`'s probe on the same file.

## Decisions (settled with owner, 2026-07-28)

1. **Stage 0a ordering — HARD PREREQUISITE.** `transition_based_drive_pain.md` lands (with its Exp 37/38 + SEM cascade + Exp 42 re-validation) before any live azimuth feed. Rationale: every orient measurement is taken against the pain regime we intend to keep; no double validation.
2. **Stage 0c — DUAL-KEY self_effect.** `self_effect: {head_yaw: ±0.3, azimuth: ±0.17}` — SAME sign (big: ±0.9 / ±0.50). Keeps joint-space bookkeeping, adds the drive-bearing key at gain-calibrated deltas matching the Exp 45c decision boundary. *(Glyph corrected from `∓` at the 2026-07-31 pre-merge round — see the Stage 0c note; the sign check in that stage is the authority.)*
3. **Stage 4b — ONE-SHOT CLI VERB** in the `maxim substrate import` family; merges via `nac_merge` with the policy-meta sidecar. No boot-time auto-merge (re-merging the same file would double-count Welford observations — the idempotency-marker machinery a bootstrap flag would need is the footgun we're avoiding).
4. **Stage 5 — FOLLOW-UP PLAN.** Stages 0–4 ship as one arc (cognition-speed see/feel/learn/reflect); the DN reflex layer executes `audio_behaviors.md` as its own plan with its own two-lens round. Interim options for reflex-speed behavior: `orient_demo.py`, or `aut_mode="substrate-primary"` on the live runtime for measured runs.

## Post-review amendments (2026-07-31 pre-merge two-lens round)

1. **The motor gap: live orient RELIEF CREDIT is deferred to Stage 5, by guard, not by accident.** The Architecture lens surfaced what this plan's validation item 2(d) missed: the live runtime has NO affordance→motor bridge — `ModulatorAffordanceTool.execute` writes sensor deltas only, and physical motion lives in the disjoint robot tools. So a live `turn_left` would book modeled relief (+1 cluster reward) while the head never moves, the next DoA reading reverts the fabricated shift, and the same credit is available again — an unbounded phantom-credit mill polluting the (now persisted, #446) runtime NAc that Stage 4b seeds. **Shipped guard:** `Embodiment.live_world_set_sensors` — the DoA feed claims `azimuth`; `ModulatorAffordanceTool.execute` filters live-owned sensors out of the modeled self_effect entirely (no write, no credit, no B8 blame), so live turns behave exactly as pre-branch while sim/scripted-substrate semantics are untouched. Validation item 2(d) therefore moves to Stage 5, whose motor bridge (BL-1) + act-now-credit-later pending map make live credit REAL instead of modeled. The arc's live deliverables stand: see (listen/§1.16/body_state), feel (drive breach, severity-latched), import (merge-nac); react is cognition-speed via the robot's real motion tools, credited from Stage 5.
2. **§1.16's reflex tier is sim-only** (gated on `sim.is_sim_mode`): its world-set models a turn — on the live path that would fabricate a measurement (the head-frame lesson's failure class). Unreachable live today anyway (DoA percepts carry 0.5/0.3 vs the >0.9/>0.9 reflex thresholds), but the gate is data-driven (`orienting:` YAML), so the guard is structural, not incidental.
3. **B8 delta-attribution is direction-blind for signed set-point-0 drives** (noted, no code change — B8-owner territory): `|delta| > comfort_band` marks azimuth "intrinsically harmful" for every turn, so in SIM a mid-course relieving turn books a negative causal link alongside its +1 relief credit (pre-existing shape on base_humanoid; live path unaffected thanks to amendment 1). Revisit when B8 next opens.
4. **Stage 0c sign convention:** same-sign, not `∓` — corrected in Decisions #2 above; both review lenses independently confirmed the shipped YAML.
5. **`focus_on_sound` IS the cognition-speed react path (2026-08-03).** Amendment 1's "react is cognition-speed via the robot's real motion tools" acquired a purpose-built tool: `FocusOnSoundTool` — zero-numeric, closed-loop (reads the DoA feed's capture-frame-stamped reading at execution time; dispatches one-shot via the controller's `goto_target`, NOT `maxim.move()`'s 2°/call step-clamped path), designed off the mirror-turn post-mortem so no signed scalar crosses the LLM interface. Three orient vocabularies now coexist by deliberate layering: **deliberative voluntary** = `focus_on_sound` (llm-primary, real motion, tool-success credit only — relief credit stays deliberately zero until the Stage-5 pending map); **substrate-primary policy** = the SEM `turn_left`/`turn_right` affordances (virtual on live per the phantom-credit guard); **reflex** = Stage 5's DN motor bridge. Stage 5 MUST coordinate with this tool (both can command the head off the same DoA reading; `inhibit_during_tool_execution` was MEANT to cover half the race but is currently dead code — `DefaultNetworkController.inhibit_for_tool` has zero callers, so the flag silently no-ops (doc-truth correction 2026-08-13); Stage 5 must wire it before relying on it — the one-way voluntary-suppresses-reflex rule in `audio_behaviors.md` covers the rest). Known shared-surface follow-ups it inherits, not owns: the `move()`/`move_head` world-vs-body-relative yaw collapse when `body_yaw ≠ 0`, and freeing `MoveTool` from the per-call step clamp via the same controller dispatch.
