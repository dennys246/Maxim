# Design Decisions (Maxim)

This file tracks decisions that affect public behavior, repo structure, and long-term maintenance.

## 2026-10-10 — Tool-failure credit (#1200): TF1–TF3

### Decision

1. **TF1. Staged design B.** A failed tool teaches through the tool-pain bridge's direct, attributed
   path (a new `ToolPainBridge.record_tool_failure`), not through the PainBus; `tool_dispatch` skips its
   duplicate NEGATIVE `tool:X` booking only for a failure the bridge booked (a failure-only cut, declared by
   an executor stamp; success links unchanged; clarified at the plan review). A stale-success defect is fixed first, on
   its own issue. A felt-only FRUSTRATION PainBus signal comes later, once `Reaction` carries its kind.
2. **TF2. Only tools that ran** earn failure credit (the invocation reached `tool.run`, returned or raised); a hallucinated or
   inactive tool name is a cognitive error, not a tool failure.
3. **TF3. Suppressed while a human drives** (the existing interactive learning gate).

### Rationale

A five-angle dive (`docs/plans/tool_failure_credit.md`) showed tool failure was never wired on `main` (PR
#114 wired and un-wired it the same day), and that putting it on the PainBus as it stands makes learning
worse: the context-similarity NAc subscriber steals the bridge's pending event, the shared cooldown
misattributes failures across tools, and downstream consumers treat frustration as harm. The bridge's
direct path gives each failure one attributed booking and a real RPE, which the memory-strength design
asked for.

## 2026-10-09 — Thalamic relay GL3.B0 decisions (TR1, TR2)

### Decision

Asked at GL3.B0's start (`docs/plans/thalamic_relay.md` §6, §11), both on the strict recommendation:

1. **TR1. The dead preemption scaffolding goes Dormant, not deleted** (L7, #1179):
   `runtime/preemption.py::PreemptionCircuit`, `ExecutionTracker`, `MaximAgent.wire_preemption`, the
   `check_hold` branch and the `capture_before` guard will be marked `Dormant since 2026-10-07: never
   wired; superseded in vocabulary by AfferentTrackSpec` by #1179. Callers and the `maxim.runtime`
   re-exports stay.
2. **TR2. The substrate turn budget may delay only action, never nociception** (L2, #1177): a denied
   tick still transduces (the encode, `note_active_clusters` and `evaluate_failures`) and skips only the
   proposal. GL3.B0's red gate (e) encodes it and flips with #1177's fix.

### Rationale

TR1 is dormancy over deletion: the code is wired by re-export, and its vocabulary is close to what a
track needs. TR2: a slow signal (the narrator's turn) must not gate the fastest one; delaying a
drift-driven breach also stamps it late and misaligns NAc's temporal window (§3.4).

## 2026-10-09 — Grounding GL1 design-review decisions (G9–G20)

### Decision

The GL1 four-lens design review of `docs/plans/grounding.md` and `docs/plans/autonomic_layer.md`
(reports: `docs/plans/reviews/grounding_gl1/{confounding,bio-faithful,wiring,environment}.md`) returned
nine DO-NOT-BUILDs. The owner took G9–G16 on 2026-10-09, and the plans fold every DO-NOT-BUILD
(`grounding.md` §GL1 lists each with its resolving sentence). The GL1 code review (three lenses, no
blockers) followed the same day, and the owner took G17–G20 with it. G1–G8 stand, except where an
"amended 2026-10-09" clause in the 2026-10-07 entry below says otherwise.

1. **G9. GL2a ships the tool-path record only.** The out-of-band producer (the per-entity snapshot,
   `Embodiment.drain_outcomes()`, its queue and bound, and the lock over `evaluate_failures`' latch) lands
   after the decomposition fence, together with the narrated scope.
2. **G10. The queue bound,** once the out-of-band producer exists: measured from a scripted session's
   peak; overflow drops the oldest record, and every drop is counted and logged.
3. **G11. Valence is unweighted:** `valence = relief − harm − nociception`, an innate prior. Health counts
   once: a `drive:health` loss enters `nociception` as this event's injury and is excluded from `harm`.
   That is a definition of the inputs, not a weighting.
4. **G12. Urgency v1 is pressure only** (max pressure after the event); the slope waits for GL3's timing.
5. **G13. GL5 is an acquired-equivalence design** (Honey & Hall 1989): authored items whose sensed
   readings and consequences DISAGREE, experience of each item required, and the model gated against a
   no-learning sensed-similarity baseline. The jet triad becomes a control for generalisation from sensed
   features, not the test. T9 keeps its claim under this design.
6. **G14. GL2a's exempt set is widened inside `runtime/executor.py`** to `Executor._run_started` (the raw
   before-snapshot) and `Executor._stamp_invocation`. The tool-path record is scoped to the invoked
   affordance's own declared drives, read through `embodiment/sem.py::_resolve_sensor_slot` /
   `_read_sensor_value` (the #1125 resolver, merged in #1164), and reported net of the declared drift over
   the window (a pure helper factored from `Embodiment.tick_vital_drift`'s arithmetic; the gate has a
   `_StepClock` case that advances 20 s between the loop tick and the tool call). The existing trio
   (`drive_pressure_before`, `drive_relief`, `pain`) and `encoding_tag` / `storage_strength` stay
   byte-identical, pinned by a golden taken from the pre-change commit on the scripted cradle sequence and
   the Minecraft arm.
7. **G15. `PhysicalEventId` is session-unique at GL2a.** GL2a mints pids unique within a session (agent,
   session id, seq) and persists no sequencer state. The cross-session resume (the sequencer's own
   high-water mark, wired at both load seams, `runtime/bio_stack.py::build_bio_stack` and
   `simulation/orchestrator.py::_restore_aut_from_session`, and tested through the real load paths) lands
   after the fence and before GL4 S1, the first stage that persists pids where they are joined across
   sessions. *(Its GL2a half is superseded by G17: GL2a mints no pid at all.)*
8. **G16. A harness-scoped provenance kind, `apparatus`.** Harness writes to a drive (the water trial's
   rescue teleport and `/effect` heal, R3's respawn, Exp 56's teacher feed, Exp 52's mother feed) are
   recorded as `apparatus`: never a forward-model training target, never credit. The double-credit rule
   extends to `NAc.credit_operant_reward`. GL2c's batched re-run reports satiation counts per arm and per
   cause. Provenance kinds are now `experienced` / `narrated` (discounted, G6) / `imagined` / `apparatus`
   (excluded).
9. **G17. GL2a ships no event id (supersedes G15's GL2a half).** GL2a's `InteroceptiveOutcome` carries no
   pid. The `PhysicalEventId` type (agent_id, session_id, seq), the per-agent sequencer, its session-id
   source and the cross-session resume all land together at the post-fence resume stage, before GL4 S1
   (the first joiner). Records written before that stage carry no pid and are never forward-model
   training data. `CauseRef.cause_pid` lands at that stage too, and GL2a's exempt file set drops
   `embodiment/event_id.py`. The "one live sequencer per agent; a rebuild fails loudly" rule moves to the
   resume stage, which must define release (bio session end and executor shutdown), name the two
   harnesses that rebuild one agent in one process (`scripts/survival_world/r3_run.py`,
   `scripts/survival_world/exp61_run.py`) and add a conftest autouse reset. The session-id source
   (deterministic; who mints it; its scope; `--resume-sim` behaviour; a new required keyword on
   `build_executor` and its callers) is asked at the resume stage's start.
10. **G18. GL4 S0a is the audit roadmap 1.4 Phase 5 requires:** one audit, one verdict.
11. **G19. T1-5's successor (the GL5 claim) lives in the cradle.** The GL1 census found every harm-class
    collision in a cradle `touch` / `warm_self` variant. Minecraft stays a consequence-prediction arm.
12. **G20. The L0 gate is a measurement only.** PASS or FAIL is recorded as the word prior's quality and
    moves nothing; `grounded_word_binding` stays SUBSUMED (G5).

Filed with the review: [#1189](https://github.com/dennys246/Maxim/issues/1189). `ToolOutput`'s repr is
persisted and substring-searched, so the existing executor stamps already leak into memory retrieval.
GL2a's new field and GL4's `ActionContext` are declared `repr=False`, with a guard that `str(ToolOutput)`
is byte-identical, and GL2a's T1-16 discharge covers the substring path.

### Reason

- **The tool-path window was undefined** (wiring D1, confounding DNB-1, environment DNB-1). As drafted it
  either refilled the trio (a memory-strength change under a "record-only" label) or failed its own pin. A
  body-wide window took in narrator writes and wall-clock drift as the action's consequence. G14 makes the
  record action-scoped, netted of drift, and leaves the trio untouched.
- **Relief must depend on need** (bio-faithful DNB-1, alliesthesia). The plan now computes relief and harm
  from the change in `drive_pressure`, and persists the per-drive block so a later schema can be
  recomputed.
- **The resume could not be wired inside the exempt set, and its source loses traces** (wiring D2): G15.
- **The record would have entered recall through the `ToolOutput` repr** (wiring D3): `repr=False`,
  #1189.
- **Sensed properties and consequences moved together in every GL5 arm** (confounding DNB-2): G13. **S0b's
  stop gate could hardly fail** (confounding DNB-3): null consequences are excluded, and collisions and
  convergences are counted separately.
- **The apparatus caused the survival rows' satiation crossings** (environment DNB-2): G16, and G7's
  description is corrected.
- **No deterministic session id exists to mint a G15 pid with** (GL1 code review): G17. Every session id
  today is wall-clock `time.strftime` (`simulation/orchestrator.py`, `simulation/research_orchestrator.py`,
  `simulation/report.py`, `runtime/loop_setup.py`, `runtime/agent_loop.py`); the minting sites have none
  in scope (`build_executor` takes no session parameter); and R3 and Exp 61 rebuild the same `agent_id` in
  one process, which the one-live-sequencer rule would refuse. Choosing the source is a decision of its
  own, so the type waits for the stage that needs it.
- **G18–G20 close the three questions `grounding.md` left open for GL1's start** (the Phase 5 audit, T1-5's
  successor, the L0 gate's role).

### Tradeoffs

- GL2a no longer records out-of-band change. Minecraft's post-return oxygen refill and every narrator
  consequence wait for the post-fence producer, so GL2c and S0b's Minecraft floor depend on it.
- Pids do not join across sessions until the resume stage lands, so GL4 S1 waits for it. Since G17 GL2a
  records carry no pid at all, so none of them is ever forward-model training data.
- The acquired-equivalence arm needs authored items (Risk 9, circular authored physics, stays open). The
  Minecraft arm tests consequence prediction, not transfer, unless a transfer contrast is designed.

## 2026-10-07 — Entity format 1.1: component state lives on its modulator; a trigger with no reading does not fire (#1124)

### Decision

Owner decisions (2026-10-07, at the issue's start and after its review round):
- `<mod>.integrity` is derived on read (`Entity.component_integrities()`), never stored on an entity's `vital_metrics`. When failures are evaluated, a modulator's real sub-sensor beats a dotted `vital_metrics` key, and the collision is reported once.
- The Entity JSON format is **1.1** (`sem.ENTITY_FORMAT_VERSION`). Each modulator saves its sub-sensor `values`, its `integrity` function and its `damage_affinities`, and a reload rebuilds the modulator's back-reference to its entity. The writer saves no dotted `vital_metrics` key; it warns about any it drops.
- **Loading a 1.0 file:**
  - A damaged `<mod>.integrity` is migrated onto that modulator's weighted, non-drive sub-sensors, so integrity and derived health come back as saved.
  - Every other dotted key is dropped.
  - One WARNING per entity says which keys were migrated and which were dropped.
- **A ≤1.3.1 build reading a 1.1 file:** it ignores `values`, finds no stored integrity and fires integrity triggers on a healthy body. This is documented, not worked around: a compatibility copy would put derived keys back in the file.
- **A failure trigger whose field has no reading does not fire**, and a recovery condition with no reading does not clear. Each warns once. `_parse_entity` warns when a trigger names a field nothing on the entity produces. Behaviour tier: invariant (fail-closed: unknown is not a breach).

### Reason

A missing field read as 0.0, so a `<` trigger fired on any key nothing wrote. A reload emptied every modulator, so integrity froze at its saved value. A body saved before its first evaluation therefore reloaded concussed and crippled while unhurt, and a stored integrity or a pre-#874 orphan key shadowed the real sub-sensor.

### Tradeoffs

- A 1.0 file's integrity is migrated, not its true sub-sensor values, and its integrity function falls back to `weighted_mean`. Neither was ever saved. So a `min`/`max` component (the dragon's torso) comes back with its saved integrity but averages later damage. Where the saved integrity cannot be reproduced (a weighted drive sub-sensor), the WARNING says so.
- Affordance specs (`params`, `requires`, `self_effect`), latent affordances and failure-mode runtime state still do not round-trip (#1159).
- One shared sensor resolver is not part of this decision (#1156).

## 2026-10-07 — The grounding line: body first, a latent forward model, receptors and afferent tracks (#1120 audit)

### Decision

These are owner decisions G1–G4, taken after the #1120 five-angle audit, and G5–G8, taken in the review
round the same day. The audit found that the EC is two disconnected worlds:
- a BODY world: the `SensorEncoder` channels at 0.85 with frozen centroids. It is live by default and it
  is the only path that acts without the LLM. No EARNED ledger row depends on the other world (T1-16 is
  Hippocampus recall into the prompt; T1-4 rides PainBus / `ToolPainBridge`);
- a WORD world: EC `text` at 0.44 on a running mean, only with `MAXIM_SUBSTRATE_PATH=1`. Percepts encode
  there on any runtime with that flag; affordance NAMES encode there only from the `--sim` orchestrator.

No edge joins them, and no positive consequence producer exists, so concepts are similar by name and
never by what they do to the body. The 1.4 grounding line answers this in stages GL0–GL6
(`docs/plans/grounding.md`):

1. **Owner decision G1. Placement.**
   - The line fills roadmap 1.4 Phase 5's existing slots (the relief store's producer side; the graded
     predictor, as `latent_forward_model.md`). It does not open a new ladder or a parallel line.
   - A new release threshold **T8** (engineering only, like T7) gates 1.4.0.
   - **T9**, the first grounding claim, is conditional and never co-headlined with E3.
   - **Body world first** (Minecraft and the cradle, substrate-primary). The word world comes later, as
     the innate-prior tier.
   - GL0, GL1, GL3.B0 (census + red gates, tests only), GL4 S0a (audit) and the paper run now, beside
     1.3.2. GL4 S0b needs a fresh capture (a sim run). GL4 S1 waits only for GL2a (exempt set below);
     GL4 S3 and S4 are `src/` and wait for the fence; GL4 S2 is offline (`scripts/`).
   - `src/` waits for the `agent_loop`/orchestrator decomposition fence. The exception is record-only
     GL2a, plus GL4 S1. The exempt file set, exactly: GL2a edits `embodiment/body.py`,
     `embodiment/sem.py`, the new leaf `embodiment/event_id.py`, `runtime/executor.py`
     (`Executor._stamp_invocation` only), `tools/base.py::ToolOutput` and `runtime/bio_integration.py`;
     GL4 S1 edits `embodiment/tool_bridge.py` (the `ActionContext` assembly and the observe call),
     `embodiment/cerebellum.py` (payload `"1.2"`, the new payload key, `import_state` refusing newer
     versions), `runtime/executor.py` (`Executor._stamp_invocation` only), `runtime/bio_integration.py`
     (`EncodingSignals.extra["context"]` at the loop capture) and the new leaf
     `embodiment/action_context.py` (the `ActionContext` type). Anything else waits for the fence.
     *(Amended 2026-10-09 by G14: GL2a's `runtime/executor.py` scope is `Executor._run_started` plus
     `Executor._stamp_invocation`. Amended 2026-10-09 by G17: GL2a's set drops `embodiment/event_id.py`,
     which lands at the post-fence resume stage; GL4 S1 waits for that stage, so its exemption is moot.)*
2. **Owner decision G2. The 2026-09-18 decision 3 is reversed.**
   - "JEPA re-pointed, not revived" (recorded here 2026-09-19) no longer holds. The predictor enters as
     `docs/plans/latent_forward_model.md`, the name Phase 5 already reserved, so that pointer becomes true.
   - It inherits the four rules of `docs/plans/deferred/jepa_cross_modal_alignment.md`: no pretrained
     cross-modal weights; the contamination guard is a CI test; opt-in; existing encoders untouched.
   - It is **not called JEPA** until its target is a learned embedding of a rich percept. While its
     target is the fixed autonomic code, it is supervised regression in a JEPA shape.
   - The projection plan is subsumed, not revived.
3. **Owner decision G3. Names.**
   - A percept source is a `Receptor` in code; prose may say "engine". The pathway it emits onto is an
     `AfferentTrack`. The handoff record is an `AfferentEvent`, carrying a deterministic `PhysicalEventId`
     of (agent, seq): no uuid, no wall time. *(Amended 2026-10-09 by G15 and G17: the id is (agent,
     session id, seq), and the type lands at the post-fence resume stage, not GL2a.)*
   - **An engine is not a track.** One receptor emits onto one or more tracks, and one physical event
     fans out under one shared id. A burn, for example, goes to a fast nociceptive track and a slow
     affective track.
   - Tracks are **logical channels scheduled on the loop/tick clock**: deterministic, lockstep-testable,
     never OS threads. Real concurrency exists only at hardware edges, which post into an inbox the loop
     drains.
   - Stage IDs are GL0–GL6 (`GL` was unused). No existing bio class is renamed.
4. **Owner decision G4. The second-body gate is split.**
   - Minecraft satisfies the PERCEPTION abstraction. Five live sources (drives, the Minecraft world,
     Reachy DoA, narrator text, DN vision) already differ in receptor class, embedding space and clock.
   - So three plans re-key to a capability trigger, the relay's registry+provenance stage (GL3.B1):
     `deferred/cross_modal_perception_fabric.md`, `deferred/perception_pipeline_placement.md` and
     `deferred/modality_resolution_and_alignment.md`.
   - The robot hardware factory (`deferred/second_body_staging.md` Stage B, the orient line, the
     microduck) keeps its physical-robot trigger, because Minecraft is not a `maxim.robots` controller.
   - The re-keyed triggers, worded once: the fabric revives when GL3's registry+provenance stage ships
     AND a 1.4 rung needs cross-modal binding; placement revives when that stage ships AND a stage is
     placed across a wire; modality resolution revives when that stage ships.
5. **Owner decision G5. Dispositions.** Taken now:
   - `deferred/jepa_cross_modal_alignment.md` is **SUBSUMED** by `latent_forward_model.md` (its four
     rules carried over);
   - `grounded_language_acquisition.md` is **SUBSUMED** for grounding by `grounding.md` (its own scope
     note kept);
   - `deferred/grounded_word_binding.md`'s L0 gate is kept as GL1's innate-prior measurement, and its old
     "PASS → candidate 1.5 headline" consequence is **RETIRED**;
   - `deferred/nociception_layer.md` is **REVIVED** into `autonomic_layer.md`.
6. **Owner decision G6. Narrated provenance, discounted.**
   - Consequences written by the narrator's tools (`simulation/tools.py::SetEntitySensorTool`,
     `DamageComponentTool`, `OrchestratorActorTool`, and the reflex dispatch that goes through them) are
     stamped `narrated`, never `experienced`.
   - They are usable for forward-model training and for credit at a **declared discount**. Its value is
     an owner decision at GL4's start; the strict default for that decision is a small discount, and
     GL5 reports its results with AND without narrated data.
   - Reason (owner): excluding them would mute the world the LLM's language priors simulate.
   - Provenance kinds are `experienced` / `narrated` / `imagined`; `declared` and `reported` stay open at
     GL3.B1, the registry+provenance stage. No type defaults provenance to `experienced`. *(Amended
     2026-10-09 by G16: a fourth kind, `apparatus`, for harness writes; recorded, never trained on,
     never credited.)*
   - The contamination guard checks that a narrated record can never be relabelled `experienced` and
     that the discount is applied.
7. **Owner decision G7. Satiation fires the survival rows whatever the routing.**
   - On `minecraft_player` the homeostatic `oxygen` and `health` breach latches clear on every surfacing
     or regeneration (`embodiment/body.py::Embodiment.evaluate_failures`, `elif cleared:
     breach_latch.pop`) after a latched breach, which is Exp 60's own `escape_water` contingency; its
     `food` is entropic with `satisfaction_threshold: 16`, and the `d1` of `minecraft_bench` (T1-11) and
     of `minecraft_bench57` (T1-12, the same spec) is entropic with 0.3.
   - So GL2c fires T1-11, T1-12, T1-13, T1-14 and T1-15 under any routing (T1-12 added by the owner
     2026-10-08; its `Re-run on:` matches T1-11's). It ships off by default and lands only with a
     batched live re-run of Exp 60/61/62 (plus the T1-11 and T1-12 arguments or re-runs).
   - The staged-donor refusals are restated, not deleted:
     `scripts/survival_world/exp61_run.py::donor_sanity_staged` (non-empty `reward_bias`,
     `cluster_reward_bias`, `links`, `event_outcome_welford`) and
     `scripts/survival_world/r3_run.py::_R3._boundary` (`reward_bias`, `links`).
   - Routing (relief store, distributor, or both) stays an open GL2c decision.
   - *Amended 2026-10-09 (G16; GL1 environment review, DNB-2):* the decision stands, but the description
     above of where the crossing happens is wrong. On the earned rows the crossings land where the
     apparatus acts. Exp 60/61 training is propose-only, and every episode ends in
     `scripts/survival_world/water_trial.py::WaterTrial.rescue` (an RCON teleport to shore, a settle until
     oxygen ≥ `RECOVER_OXYGEN_MIN`, then `heal()`), so the `oxygen` latch clears during the rescue, not at
     `escape_water`. Exp 60's probes are capped before the breach and never latch. R3's respawns reset
     health and oxygen. Exp 56's teacher and Exp 52's mother write the drive directly. Those writes are
     `apparatus` (G16), and the rows still fire whatever the routing. `food` is not reachable natively in
     the earned campaigns.
8. **Owner decision G8. The registry enters with its consumer.**
   - GL3's `Receptor` registry lands together with provenance on the write path, in one stage (GL3.B1),
     whose consumers are GL5's experiment and the forward model's contamination guard. GL3's census and
     red gates (GL3.B0) stay first; they are tests only and run now, inside the fence.
   - GL3's stage IDs are always prefixed (`GL3.B0`–`GL3.B8`; a bare `B8` is the delta-attribution
     invariant), and `thalamic_relay.md` §6 is the canonical map. The first track slice (GL3.B3) is the
     thermal dual-track fan-out, with no preemption, built on GL2b's `NociceptorSpec`; it carries the
     seq-authority handover as its stage gate.
   - Nociceptive-fast preemption (GL3.B4) becomes a declared 1.4 rung arm, or waits for one; the owner
     names the rung later.
   - The timing defects L1, L2, L5 and the dead preemption scaffolding L7 are filed as defect issues
     (#1176 L1, #1177 L2, #1178 L5, #1179 L7), each fixable without the registry.

**The event-identity contract** (one statement; `grounding.md` carries it, `autonomic_layer.md` defines
the type):
- One type, `PhysicalEventId(agent_id: str, seq: int)`: frozen, SHAPE-FROZEN at 1.0 (CC3 path b);
  `__post_init__` rejects an empty `agent_id` and a negative `seq`; `__str__` is `"{agent_id}:{seq}"`. It
  lives in the leaf module `maxim/embodiment/event_id.py`, built at GL2a; GL3 imports it. *(Amended
  2026-10-09 by G15: the id is (agent, session id, seq), so the type carries a non-empty `session_id`
  between the two and `__str__` is `"{agent_id}:{session_id}:{seq}"`. Like the seq, the session id
  carries no uuid and no wall time, and `__post_init__` rejects an empty one. Amended again 2026-10-09 by
  G17: the type, its leaf module and the sequencer are built at the post-fence resume stage, not GL2a.)*
- One seq authority per agent: an `EventSequencer` held by the agent's primary `Embodiment`. Ephemeral,
  scene and foundry wrappers mint no records and no ids. `seq` persists per agent and resumes past the
  saved maximum (the `Hippocampus._resume_capture_seq` rule). At GL3.B3 the scheduler's drain point
  takes over, with the GL2a counter as its backing store; the handover is GL3.B3's stage gate.
  *(Amended 2026-10-09 by G15: GL2a's pids are unique within a session (agent, session id, seq), and
  GL2a persists no sequencer state. The persisted resume, with its own high-water mark and both load
  seams (`bio_stack.build_bio_stack`, `orchestrator._restore_aut_from_session`) tested through the real
  load paths, lands after the fence and before GL4 S1. Superseded in part by G17: GL2a builds no
  sequencer and mints no pid; the sequencer, its session-id source and the resume land together at
  that stage.)*
- A consequence names its cause by `CauseRef.cause_pid` (`PhysicalEventId | None`), the same type.
  *(Amended 2026-10-09 by G17: `cause_pid` lands with the type at the post-fence resume stage.)*
- `Embodiment.drain_outcomes()` has named production drainers (the loop capture; in `--sim`, the AUT
  loop's `capture_loop_action` on `sim.aut`), a drop-oldest bound with every drop counted and logged,
  and ephemeral wrappers never queue. *(Amended 2026-10-09 by G9/G10: it lands with the out-of-band
  producer, after the fence; GL2a builds none of it.)*
- The `--sim` orchestrator thread (the `start_simulation_mode` caller running the orchestrator agent's
  loop, where the narrator's tools, registered on `orch_registry`, call `evaluate_failures` while the AUT
  loop runs on `sim.aut`; not `sim.dm`, which only interactive DM campaigns use) is a declared edge: a
  lock from GL2a, an inbox drained once per pass from GL3.B3. The reflex dispatch's instances of the
  same classes run on the loop thread. *(Amended 2026-10-09 by G9: GL2a's sequencer takes its own
  private lock; the lock over `evaluate_failures`' latch and snapshot lands with the out-of-band
  producer. Amended again by G17: the sequencer, with its private lock, lands at the post-fence resume
  stage.)*
- Forward-model pairs join on the `pid`, never on the executor's `uuid4` invocation id; the tool path
  stamps the pid on `ToolOutput` beside it.

The plans:
- `docs/plans/grounding.md`, the umbrella;
- `docs/plans/autonomic_layer.md` (GL2): the signed body-consequence code and the regulatory defects.
  `deferred/nociception_layer.md` is REVIVED into it;
- `docs/plans/thalamic_relay.md` (GL3): receptors and afferent tracks;
- `docs/plans/latent_forward_model.md` (GL4).

The state page is `docs/wiring/body-and-word-worlds.md`. Every other choice the drafts raised stays open,
listed in `grounding.md` as "Owner decision at <stage> start". That includes T1-5's status and #1120's
retired-threshold red gate (both at GL0), #1161's order, and the joint review of GL2c with the relief
store.

### Reason

- **Nothing links a word to a consequence.**
  - Hebbian episode binding is Dormant (D6);
  - `Hippocampus.retrieve_cross_modal` has no `src/` caller;
  - `archive/cross_modal_substrate_binding.md` is cancelled.

  On shipped components, names and consequences disagree: a safe `touch` and a burning one share one
  node, as do a safe and a harmful `warm_self`. Name similarity cannot tell them apart, and the
  compound-name match is what #1120 actually measured under T1-5.
- **Body first is the only order whose results are readable.** No earned row depends on the word path,
  and the word path is fenced (the orchestrator) and LLM-coupled.
- **The predictor needs a target the body already has**, a signed consequence code. That is why the
  autonomic layer is built first. It is also why the projection plan (384 ↔ 768 alignment) is solving a
  different problem.
- **Phase 5's slots keep the rules.** Placing the line there keeps "no second store, no second predictor",
  the relief store's own review rule. It also lets T8 gate the release on engineering truth without
  gating it on a may-fail result.
- **The physical trigger conflated two arguments:** a robot-factory argument (designing
  `hardware/controller.py::RobotController` from one robot) and a perception argument that is already
  satisfied.

### Tradeoffs

- **GL2c's positive producer lapses a written discharge, under any routing (G7).** T1-13 (Exp 60) says
  "if a positive Reaction emitter is ever wired, this reasoning lapses", and the Minecraft breach latches
  clear at Exp 60's own contingency, so T1-11, T1-12, T1-13, T1-14 and T1-15 all fire whichever surface
  the producer writes. No routing avoids it. The producer is off by default and lands only with a
  batched live re-run of Exp 60/61/62 (+ T1-11 and T1-12) in its own rig slot after the 1.3.2 live Exp 60 re-run, never
  stacked on it.
- **Narrated data is used, not excluded (G6).** The cost is a contamination risk the guard must close
  (no relabelling, the discount applied), and every GL5 result is reported with and without it.
- **Two may-fail lines in one release make a null in either unreadable.** Hence T9 is conditional and
  never co-headlined, and no grounding flag is set in an E1–E3 arm unless declared (mechanization backlog
  M10, amended).
- **A relay contract is close to the percept-channel manifest that `archive/percept_testbed_audit.md`
  rejected "as conceived".** `thalamic_relay.md` must answer that in its front-gate. Per G8 the registry
  enters only with provenance and that stage's consumers, wrapping the existing receptors
  byte-identically; the first track slice (GL3.B3) is a flag-off fan-out with no preemption, and
  preemption (GL3.B4) enters only as a declared rung arm.
- **The rules followed by attention become backlog rows.**
  - Rows in `docs/plans/outstanding.md`: M39 (every percept enters the EC through the relay), M40
    (afferent tracks are never threads) and M41 (below).
  - The shared event identity is to be enforced by its type (structural guard: `PhysicalEventId` and
    frozen records with required keyword-only fields); no row.
  - The predictor's contamination guard is to be enforced by its CI test. Until `latent_forward_model.md`
    S3 lands it is backlog row M41; then it is structural.
- **Banners, not moves.** `grounded_language_acquisition.md` and `deferred/jepa_cross_modal_alignment.md`
  are cited by `src/` and the CLI, so they stay at their paths with dated banners.

## 2026-10-01 — config.json 1.2: the file holds exactly the operator's choices

### Decision

Owner decision (the root fix, over a narrow O19 prereg amendment):
- `config.json` format **1.2** changes what a key MEANS, not the schema: a key present is a value the operator set (`maxim config set`, a setup verb, a hand edit). The writer persists exactly those keys; every write path states what it assigns (`mutate_config(assigned=)`, `write_config(explicit=)`, both required).
- `maxim config set` pins even a value equal to the default (`llm.n_ctx 8192`). `maxim config unset` and `set <field> null` return a field to its default and drop it from the file.
- A file stamped below 1.2 is a full dump whose intent is unknowable, so it is read as before (a value equal to the default is not set) until the first write on a 1.2 build, which keeps only its non-default values plus the field being set. A file with NO version was written by hand (the writer has always stamped one) and is read by presence.
- C7a's cloud auto-detect reads the resolved config: a configured local or unknown `llm.profile`, `cloud.enabled false`, or an unreadable `config.json` stops it, and it never sets an env var whose field `config.json` sets (#1030).
- How each format version resolves is pinned in `tests/fixtures/config_resolution_by_version.json`, append-only under the #856 lint: a change of meaning is a new version.

### Reason

The writer dumped every field (`asdict`), defaults included, so no reader could tell a choice from a default and the loader guessed "value != default". An operator's `maxim config set llm.n_ctx 8192` therefore read as `default`. The O19 re-runs require `configured_n_ctx_source == "config"` and could never pass, and the documented off-switch `cloud.enabled false` (also the default) did nothing. Two design passes rejected the narrower fixes: reading key presence on full-dump files would have made every field read as set, and an O19 prereg amendment would have left the loader blind everywhere else.

### Tradeoffs

- An older build reading a 1.2 file still parses it, but reads a pinned default-equal value as `default`. The current 1.1 build refuses to write it (#974), and its `downgrade` writes a full 1.1 dump. PyPI 1.3.1 (format 1.0) rewrites it as a full 1.0 dump on `config set`. Values survive either way; only the "the operator set this" source is lost. A PyPI leader and a dev checkout that share `~/.config/maxim` will demote each other's pins.
- Behaviour that changes on a 1.2 file: a pinned `llm.enabled true` exports `MAXIM_LLM_ENABLED=1`; leader_proxy enforces admission at a pinned `n_ctx`; the env-vs-config convergence and divergence logs fire for pinned values (no longer for dumped defaults); doctor shows pinned values as `config.json`.

## 2026-09-28 — A newer config.json is refused, then transitioned explicitly (#974)

### Decision

Owner decision (the refuse-only option, plus a transition the owner asked for):
- `maxim config set` refuses a config.json from a newer format version.
- `maxim config downgrade` keeps the known settings and moves the unknown ones to a `config.preserved.json` sidecar that is never applied.
- `maxim config restore-preserved` restores them after an upgrade: re-validated, shown as a diff with security-relevant settings flagged, and confirmed at an interactive terminal.

### Reason

After a downgrade, an ordinary `config set` silently deleted every setting the newer build had added. Preserving them automatically would have let a stale security-relevant setting come back without the operator seeing it. Refusing alone would have left no path through a rollback short of hand-editing.

### Tradeoffs

- The owner's injection review, and a security review lens, set the requirements:
  - escaped output, including validation errors;
  - no `--yes`: the CLI never restores unattended;
  - EVERY setting flagged except an explicit list of tuning knobs, so a field added later is flagged by default;
  - preserved sections restored and shown field by field, never wholesale (a wholesale section once silently reset the fields it lacked);
  - each entry's source version and time shown;
  - the same validation as `config set`;
  - one bounded read.
- The terminal requirement makes a restore no EASIER than editing `config.json` directly. It is not a barrier against a same-user process, which could allocate a terminal, call the Python API, or edit the file itself.
- No new privilege boundary is crossed: the sidecar and `config.json` are both files owned by the same user, so a planted sidecar can do nothing a direct edit could not.
- A build without this change, which is every release before it, still drops a newer file's keys on write.


## 2026-09-28 — config.json format 1.1, and the schema is pinned to the version (#856)

### Decision

Owner decision: bump `CONFIG_FORMAT_VERSION` to `"1.1"` once, covering what shipped under `"1.0"` without one: the `console`, `tools`, `sim` and `memory` sections, and the fields added inside `llm` and `console`. Guard it so the next schema change cannot ship without a bump: every field path is pinned per version in `tests/fixtures/config_schema_by_version.json` (append-only; minors only add).

### Reason

The format-version contract was built for this case and never used. An older build tolerates the unknown keys of a FUTURE minor, but refuses them in a same-version file, and that stopped every `maxim` command after a downgrade. Documenting the incompatibility instead would have left the next section to repeat it.

### Tradeoffs

- A `"1.0"` file an earlier build wrote with those sections still stops an older build until it is rewritten. No change can reach files already on disk.
- The write side stays open: an older build that rewrites a newer file drops the keys it does not know (#974, decision owed there).
- The writer now stamps its own version rather than a loaded file's. A config loaded from an older file is this build's schema once parsed.


## 2026-09-28 — A memory store never saves over a file it did not read; `~` is the home directory (#939, #950)

### Decision

Owner decisions, four at the design step:

1. **The clobber is fixed at SAVE, not in the constructors.** A store may write a file it read, a file it created, or a file it was told to replace (`save(overwrite=True)`, `allow_overwrite()`). Any other existing file raises `StoreOverwriteRefused`. Enforced in `memory/store.py::StoreFileOwnership` for Hippocampus and ATL, so every construction path is covered, including future ones. The constructors keep the package contract that `create.*` is always empty.
2. **`create.agent(name)` refuses up front** when the agent's home already holds persisted state. With a guard on only some stores, a fresh agent there would have kept the old memories but replaced NAc/EC/SCN: a mixed agent. The guard for the remaining stores is #971.
3. **`~` is expanded** through one resolver (`utils/paths.py::store_file_path`): in every Hippocampus, ATL and NAc save/load, the `load.*` calls, and the agent-home paths (`persistence_dir`, `load.agent(base_dir=)`, `AgentFactory(base_data_dir=)`). SCN, EC, AngularGyrus and the cross-layer index get it with their guard (#971).
4. **`load()` of a missing file raises `FileNotFoundError`** in Hippocampus and ATL, as in NAc. `missing_ok=True` is the explicit load-if-present form (the hub's session-start ATL restore uses `load_safe`, which checks the file exists first).

### Reason

A user's memories were lost silently: the documented examples did it when run twice. The failure lived in the composition "construct empty, then save", which no single constructor sees, so the guard sits at the one place every path passes through, the write.

### Tradeoffs

- A corrupt Hippocampus/ATL file that a store starts fresh from is copied to `<name>.corrupt-<UTC timestamp>`, and then the store saves in its place (owner decision at review). Preserving it untouched instead would have left the agent half-persisted: its memories session-only while NAc/EC kept saving, the mixed state #939 prevents. The evidence is kept either way; if the copy fails, the original stays and saves over it are refused.
- The write-but-don't-read orchestrator still restores its ATL at session start (#972, pre-existing): its overwrite declaration only matters when that read fails.
- Running a script that calls `create.agent("scout")` twice now fails the second time, with a message naming `load.agent`. The same holds for `create.hippocampus(persistence_path=P)` followed by `save()`. (Superseded in part by #1071, 2026-10-04: `create.hippocampus` / `create.atl` now refuse an existing `P` at construction, not at `save()`.)
- NAc, EC, SCN, AngularGyrus and the cross-layer index are not guarded yet (#971). `create.agent`'s up-front refusal covers the public entry point meanwhile.


## 2026-09-28 — The internet policy is the operator's, frozen; on/off is composed at read time (#832)

### Decision

Owner decisions on #832 items 1, 3, 4 and 5 (item 2 waits for #922/#834). The freeze followed a parallel three-lens review of the type's shape: enforcement, planned work, and freeze mechanics.

1. **Search obeys the policy.** Results pass the policy's allow/block lists (`domain_refusal`, no DNS); the page limit is `max_pages_per_minute`, read per request.
2. **`build_tool_registry(internet_launch_enabled=...)` is required and builds the getter.** No caller can hand the tools a policy, and forgetting is a `TypeError`.
3. **The recorded internet state is the effective value at launch** in both runtimes. The live per-turn read waits for the `agent_loop` slices (#965).
4. **`InternetAccessPolicy` is frozen and operator-only:**
   - `enabled` moved out to a composed, never-persisted `EffectiveInternetPolicy(policy, enabled, source)`;
   - the three fields nothing read are retired;
   - the stale private domain copies are removed;
   - values are validated on construction.
5. **Forward compat is loader-owned** (not `extra`, not SHAPE-FROZEN): unknown keys fail closed, and retired keys warn. So any field added later turns internet off on an older build: a downgrade fails closed.
6. **A corrupt toggle file fails closed too.** Proposed during the review folds, beyond the owner's four answers, and approved by the owner on 2026-09-28 after #969 was up. It applies #822's stance (and item 5's) to the toggle: a corrupt or non-boolean `util/internet_access.json` used to fall back to ON.

### Reason

The operator's lists and limits reached `http_fetch` but not search. The shared cached instance was mutable, and its loader mutated it. Three fields claimed behaviour nothing enforced. `dataclasses.replace` kept a stale block list enforced. And an ignored unknown key meant a typo silently dropped a block list. Freezing without the review would have locked the dead fields and the `enabled` ownership split into the persisted shape.

### Tradeoffs

- A policy file with an unknown key now turns internet off until it is fixed. That is the #822 fail-closed stance, applied to typos, and it makes a downgrade across an added field turn internet off (the #856 class, chosen deliberately here).
- The recorded on/off can go stale mid-session until #965. No consumer decides on it today.
- The review's other findings are separate issues: the summary never reaches the model (#965), the pre-check's classifier and DNS-failure reason (#966), robots.txt failing open (#967), and IDNA 2003 vs 2008 for `ß`-style entries (#968).


## 2026-09-28 — The operational mode is a launch grant, separate from the run mode (#829)

### Decision

Owner decisions, three at the design step and one at review:

1. **Two axes, two flags.** `--mode` stays the run mode (exploration, agentic, sleep, live, train,
   reflection). `--operational-mode passive|active|singularity` is the operator's launch grant for
   capability.
   - It sets the mode dispatch enforces (`Executor.set_operational_override`), what the model is shown
     (`loop_state._effective_mode`: roster, context prompt, Default Network) and the registry's file
     containment. (Superseded by #963: `Executor.effective_operational_mode` + `loop_state.operational_mode`.)
   - It is honoured by the CLI agent loop (`--mode agentic`) and the robot runtime.
   - It needs an explicit `--mode`. It is refused (exit 2) with `--sim`/`--research`/`--benchmark`/
     `--foundry`, whose AUTs run active by design, and with the one-shot actions that run no agent.
   - Every runtime request may only LOWER the capability the process would run with afterwards: an
     operational name is that capability, and a run mode keeps the grant, else implies its own.
   - A request for the current state is a no-op, not a restart.
2. **Singularity is reachable only by the explicit launch flag**, announced on stderr and in the log. The
   tool, the heard phrase and the runtime re-exec all refuse it. The grant persists across run-mode
   re-execs (a sleep/wake cycle; without a grant, waking from a passive-class run mode into an
   active-class one is a raise and is refused). #922's time-boxed, in-session grant will supersede that persistence.
3. **`--mode agentic` is defined (active-class), and an unknown mode name fails closed**: it is enforced
   as passive, never as unrestricted. `raises_capability` judges an unknown current mode as passive too,
   so the predicate and dispatch agree. On shipped paths `agentic` reaches the predicate (the
   `ModeSwitchTool` reads the run mode) and the in-process re-exec fallback. No shipped writer puts it in
   the loop state, so no dispatch changes.
4. **A small, explicit exception to the 1.3.2 `agent_loop` fence** (owner, at review). Three read sites
   now resolve the mode through `_effective_mode`, so a raising grant is not a silent no-op at the
   prompt. (Superseded by #963: `Executor.effective_operational_mode` + `loop_state.operational_mode`, and every
   other mode read in the loop moved to them.)
5. **#963 (2026-10-09), Q1: one precedence.** In the agent loop and at dispatch, the grant-over-run-mode rule
   lives once, in `Executor.effective_operational_mode()`; `_mode_denial` and `loop_state.operational_mode` both
   read it, with one default (`observe`). Two copies remain outside it: `bootstrap.build_tool_registry`'s
   `get_mode` callback (#1193) and the CLI's `_current_operational_mode` / `_registry_operational_mode` /
   `_runtime_mode_switch_allowed` (#922).

### Reason

Operational names reached `--mode`, which rejected them (exit 2). There was no launch path to a chosen
capability, although #924's strict rule depends on one. Review found the grant reaching dispatch but not
the prompt, and silently ignored on sim paths. Both are the silent-no-op shape the rules forbid.

### Tradeoffs

- Without the flag, file containment and dispatch can still disagree: the robot runtime and the CLI sim
  fall-through, #960. That predates #829, and changing either side changes capability.
- It lands ahead of #834's typed `GrantAuthority`. The grant is a bare string for now, and #834 will
  retrofit it.
- The wake handler's `exploration` request is still not runtime-switchable (pre-existing). Without a
  grant, a passive-class run mode (sleep, train, reflection) cannot re-exec into an active-class one
  (live, agentic, active): the lower-only rule applies to run modes too. A robot meant to wake into an
  active run mode must be launched with `--operational-mode active`.


## 2026-09-28 — Downloads stay in the class of their first dial; operator paths let a proxy win (#921)

### Decision

Two owner decisions:

1. **`download_to_file` stays in the address class of its first dial.** Each download gets its own
   connection backend (`utils/http.py::_StartClassBackend`). The first connection's own resolution fixes
   the class. If every address is public, every later hop (each redirect is a new connection) must be
   public too. Anything else, meaning a LAN or loopback URL the operator configured such as a local Oasis
   or a model mirror, stays unrestricted.
2. **A configured proxy still carries operator-configured URLs** (downloads, backend base URLs, peer
   probes). The proxy can come from env vars or the macOS/Windows system proxy. The proxy dials, so the
   connect-time check cannot apply to what it carries, and one WARNING per process says so. Every checked
   client gets httpx's OWN per-URL proxy mounts (`_proxy_mounts`) over an address-checked default
   transport, so httpx's `NO_PROXY` semantics decide per hop, and every DIRECT dial is checked. The
   model-chosen fetch (#824) never uses a proxy.

The backend base-URL paths enforce `validate_base_url`'s rules at connect time, and the peer's probes are
included. One classifier, `utils/net.py::is_public_address`, serves both the validation and the connect.

### Reason

#824 made the model-chosen fetch safe and scoped these operator-configured paths out.
- A redirect from a public registry into the LAN or `169.254.169.254` is the SSRF a download can suffer.
- Refusing every private address would break LAN-hosted Oasis and mirror setups.
- Classifying the start with a separate lookup was the first draft. Two reviewers showed a hostile DNS
  owner could answer private to it and public to the connect, taking the whole download off the checked
  path. Classifying on the dial itself removes that lookup.
- Proxies: a custom transport silently drops proxies. Strict checks would have cut model downloads and
  every OpenAI-compatible cloud backend for anyone behind a corporate proxy. A first attempt re-derived
  httpx's proxy rules by hand and misread ports and schemes in `NO_PROXY`, ignored system proxies, and
  left a proxy-bypassing redirect hop unchecked. Reusing httpx's own rules fixed all three.

### Tradeoffs

- **What a proxy carries is not address-checked.** It is logged, not silent. `NO_PROXY` hosts and
  direct hops are checked.
- It relies on httpx's private `httpx._utils.get_environment_proxies`, pinned by a test that fails if
  httpx moves it.
- A URL whose own first dial is private is trusted as the operator's choice.
- CGNAT (100.64.0.0/10) moved from "public" to "non-public" in `validate_base_url` when the classifiers
  were unified. An `https://` backend on Tailscale now needs `allow_local_endpoints`.

## 2026-09-28 — Host coding tools: the working root, and git_commit is opt-in (#949)

### Decision

Two owner decisions, taken when #949's review found that `allowed_dirs[0]` is the scratch
`.maxim_workspace` in every mode's filesystem policy:

1. **The working root** of `run_tests`, `git_diff`, `git_commit` and `execute_file` is the project
   (the process working directory) when it lies inside the mode's `allowed_dirs`, and otherwise
   `allowed_dirs[0]` (a sim tmpdir or the console's override root). It is computed by one function,
   `tools/base.py::tool_workdir`. git is capped at that root with `GIT_CEILING_DIRECTORIES`. `bash`
   keeps its existing `allowed_dirs[0]` default.
2. **`git_commit` is opt-in** via `MAXIM_ALLOW_GIT_COMMIT`, off by default, the same shape as
   `MAXIM_ALLOW_BASH`, `MAXIM_ALLOW_RUN_TESTS` and `MAXIM_ALLOW_GIT_DIFF`.

### Reason

With `allowed_dirs[0]` as the root, `run_tests` collected nothing, a repo-relative `git_diff` path
returned success with an empty diff (a silent no-op), singularity crashed on a workspace that is never
created, and git searched upward from the workspace into the host repository, so `git_commit` could
commit whatever the user had staged. The project is inside `allowed_dirs` by the policy's own design, so
running there is contained. `git_commit` was the one host coding tool with no gate.

### Tradeoffs

- A user who relied on the agent committing must set the flag.
- A project started from a subdirectory of a repository cannot reach the repository root from these
  tools, because of the ceiling: contained, not convenient.
- The environment allowlist drops `SSH_AUTH_SOCK`, so SSH signing is unavailable to these tools.

## 2026-09-27 — The agent may never raise its own mode capability (#924, strict)

### Decision

The model's `mode_switch` tool refuses every switch that gains capability, and allows every switch that
does not. This covers passive → active, which gains acting on the host (#826's dispatch refusal), not only
switches into the code-executing singularity (#821). Lowering stays free. "Gains capability" is one
predicate, `maxim.modes.definitions.raises_capability`, derived from the enforcement itself. It
iterates the same `DISPATCH_CAPABILITIES` table the dispatch gate reads, so the two cannot drift. A
switch raises capability when:
- the target would run a tool the current mode refuses at dispatch;
- or the target gains code execution;
- or the target drops the confirmation requirement.

An unknown target counts as a raise. An unknown current mode (the agentic runtime's `"agentic"`) counts
as unrestricted, as dispatch treats it, but still may not enter a code-executing mode. (Superseded by #829: `agentic` is defined, and an unknown current mode is judged as passive.) The rule is latent
today: no shipped runtime registers `mode_switch`.

This holds until the in-session human approval surface (#922, 1.4) exists. A capability-raising request is
then that surface's decision.

### Reason

After #826, passive mode is enforced by capability at dispatch, so a passive agent that could switch itself
to active could step out of that refusal. Owner decision 2026-09-27: strict, the recommended default under
"weak evidence never gates". It reverses the 2026-09-26 decision to keep #821's narrower rule for 1.3.1,
which is recorded in #924's body, not in this file.

### Tradeoffs

- An agent that genuinely needs active mode now has to ask a human. Until #922 exists, that means a
  restart in active mode by the operator.
- The rule is enforced in the tool, not at dispatch, because the executor gates by tool name. A second
  mode-changing tool would have to call the predicate too. #834's typed authority is what makes that
  structural.
- Other channels use `executes_code` only:
  - the CLI `requested_mode` consumer (#829);
  - the heard-phrase path (#828). It changes `StateManager`'s mode, not the `state.data["mode"]` that
    dispatch reads, so it does not raise dispatch capability today;
  - the wake handler's sleep → exploration request (`agentic_runtime.py`), which the CLI's switchable-mode
    list rejects today.

  #834 unifies them.

## 2026-09-27 — The coding world revives coding_habits_oasis; Exp 55 leaves the Shared-perception deferral

### Decision

`docs/plans/deferred/coding_habits_oasis.md` is revived **by owner request** as a 1.4 parallel line,
`docs/plans/coding_world.md`. Its written trigger ("a rung names the gap") has not fired. The old file
stays as the record. The revival comes with five rules:

1. **Exp 55 re-points** from the Shared-perception deferral to this line. It was bundled with the
   physical trigger (a second body) but never needed one, because it is agent-to-agent transfer on one
   body type.
2. **Oracle harm labels** (for example "a failing test was deleted" or "a CWE pattern was introduced")
   ride the 2026-09-12 harness-injected lane under all four of its rules. The oracle lives under
   `scripts/`. It injects a pain valence into an interoceptive drive that is never on the world roster,
   and each prereg names it as the independent variable. Measured world state (test outcomes, exit codes,
   honeypot touches) stays D1.
3. **No second credit store.** The act-bound aversive write is a PainBus *producer* for the existing
   `NAc.credit_operant_reward` path.
   - Roadmap 1.4 Phase 5's relief-store review only *reserves* the opposite sign on that seam.
   - The producer is designed in its own review after the relief store's T5 decision.
   - It is off by default and never set in an E-rung arm.
4. **Default: no coding-world `src/` before the 1.4.0 cut**, so the line cannot gate 1.4.0 through T4.
5. **Stack Overflow is not grounded text.** It has no sensor side. No Stack Overflow content is committed
   and no LLM is trained on it. Its uses (held-out test phrasings, later advice) are owner decisions
   listed in the plan, with "no use before social referencing's trigger" as the default.

### Reason

The survival-world paired-data audit ruled "redesign the data source". The aversive-conscience hand-off
asks a question Maxim can answer honestly only where a harmful act is measurable. The deferred plan had
already designed that world. Two-lens review found that the act-bound write mostly exists already
(`credit_operant_reward`, Exp 56's teacher path). So the new piece is a producer, not a store.

### Tradeoffs

- This adds a parallel line with its own may-fail experiments. Mechanization backlog M12 allows at most
  one claim-bearing experiment per line at a time.
- Rule 4 delays any coding-world result to after 1.4.0.
- The by-attention rules are mechanization backlog rows M10–M14 in `docs/plans/outstanding.md`.

## 2026-09-27 — The agent preamble is read from the Constitution (bugs ledger D32)

### Decision

The foundational preamble every agent's prompt carries now lives in `CONSTITUTION.md` itself, in a marked
"Runtime Preamble" section, and the runtime reads that block verbatim from the copy shipped as package
data (`maxim/_data/CONSTITUTION.md`, drift-tested against the repo root). Owner decision, 1.3.1.

### Reason

The preamble was a hard-coded paraphrase in `agents/llm_context.py`, gated only on a repo-root
`CONSTITUTION.md` existing: the document and the prompt could drift silently, and every pip install (no
repo root) ran with an empty preamble. Reading the text from the document makes them one thing.

### Tradeoffs

Editing the principles agents are told now means editing the Constitution -- intended. The block is a
condensation, but its hard constraints are §1's bullets VERBATIM and a test requires every one of them
(`test_every_hard_constraint_reaches_the_prompt_verbatim`): the old prompt had dropped "Never operate
actuators at speeds that could cause injury", and 1.3.1 restores it (owner decision, 2026-09-27). The
rest of the block (core values, the agent behavior rules) stays a review duty, not a mechanical check.

## 2026-09-25 — Oasis release format v2: detached signature, entry index, ordering, license

Decision (design: [docs/plans/oasis_entry_index_v2.md](docs/plans/oasis_entry_index_v2.md); public_oasis
Phase 0 item 7, landing before the item-2 format freeze):

- **Signing scheme v2 is a DETACHED signature over raw member bytes.** A `signature.json` ZIP member
  signs every other member's uncompressed bytes under the domain tag `maxim-bundle-v2` — no canonical
  JSON in what is signed. The manifest (bundle schema **3**) carries `signer_identity`,
  `release_sequence`, `license` and `entry_index`, so all of them are inside the signature.
- **Only a signed release is schema 3.** An unsigned bundle stays schema 2: it needs nothing schema 3
  added, so 1.3.x peers and Oasis servers keep reading contributions, and they refuse only what they
  cannot verify. The schema number says what a reader must understand.
- **v1 is legacy.** A schema ≤ 2 bundle signed under v1 still verifies — the manifest is verified AS
  STORED, before the envelope migration (which rewrites a field v1 signs). A v1 signature on a schema-3
  manifest is refused as a downgrade; an unknown scheme is refused, never read as v1. The verifier
  takes `accept_v1` as a REQUIRED keyword, and the per-Oasis registry flag decides it for `hive pull`:
  a NEW registration writes `accept_v1: false` (a first-contact client has no v1 history, so it cannot
  be downgraded), an existing entry without the field keeps accepting v1 (item 7 PR B). v1 is removed
  at 2.0.
- **An entry is one situation cluster**, and its digest is sha256 of the RFC 8785 (JCS) serialization of
  its projection. The verifier refuses any situation state the signed index does not cover.
- **Releases are additive, so the sequence ORDERS them; it does not gate them.** A `(signing key,
  sequence)` pair binds to one signed payload (a second payload claiming it is equivocation); dedup
  keys on the signed-payload digest. There is no "refuse below the highest seen" rule. The payload
  covers `created_at`, so re-composing the same sequence is a NEW payload: the producer's counter
  advances on every compose, not every publish (item 7 PR C). The counter is keyed by the signing
  key's PUBLIC KEY (the Queen key and a development key never share one), and is committed with the
  release; the Oasis store verifies each release at publish (`oasis publish --queen-key`), ids it by
  its signed-payload digest and refuses equivocation with the same predicate receivers apply.
- **Downgrade: once a key's v2 release is admitted, a v1 bundle from that key is refused** in that
  receiver session (owner, kept on the PR B review). v1 has no sequence, so this also refuses an OLDER
  v1 lineage that arrives later; recovery is a journal hand-edit. A `created_at` narrowing was rejected:
  the key holder the rule guards against signs that timestamp. It needs one key that signed both
  formats — the Queen's own key never signed v1, and a new registry entry refuses v1 outright — so an
  explicit waiver is added only if it ever bites.
- **Every signed release carries a license** (SPDX); published bundles use `CDLA-Permissive-2.0`.
- **A release ships one agent's learning, and a receiver keeps only rows it can read** (owner, on the
  PR A review). NAc reads filter on the reader's agent id. The exporter keeps its own agent's rows
  (`--agent-id` when several), drops the rest with a count, and ships them under a fixed token
  (`_agent`), so local agent ids never ship and two agents' rows can never collapse onto one key. A
  receiver MUST re-key the situation rows to its own agent id (ingest refuses a token-keyed bundle
  without one) and drops the non-situation rows (percept valences, outcome stats, node-keyed bias)
  filed under another agent, which the receiving agent could never read. Making those transfer is a behaviour change,
  deferred on a trigger: [docs/plans/deferred/transfer_non_situation_nac_rows.md](docs/plans/deferred/transfer_non_situation_nac_rows.md).

Why: the index is what lets social_referencing select and admit Queen material per entry without a
server-cut slice. The detached raw-bytes signature ends the v1 canonicalization hazards and lets a
non-Python verifier check a release. The two-lens design review and its re-read are in
[docs/plans/reviews/oasis_entry_index_v2/](docs/plans/reviews/oasis_entry_index_v2/). The per-entry
journal and supersession (reserved in the format) are social_referencing S1's, with its own amendment
to the ingest contract §1.

## 2026-09-19 — Decision point 4 re-opened for PUBLICATION only: a project-hosted Oasis that publishes and never accepts

Decision:

- **The project runs a public Oasis** at `oasis.pymaxim.bio`, on the owner's own hardware behind a
  Cloudflare Tunnel, serving the Queen release tier **read-only**. This re-opens decision point 4
  ([docs/plans/archive/hivemind_p2p_scope.md](docs/plans/archive/hivemind_p2p_scope.md)
  §Decision points: *"no project-hosted Oasis in 1.2 … the project runs none"*), which was an
  operational-burden decision. Publication's burden is static signed blobs, no inbound trust and no
  curation labor.
- **The deferral stands for SUBMISSIONS.** `POST /v1/substrate/contribute` is not opened to the
  public — not into trusted state, and not into the quarantine tier — until the six conditions in
  [docs/plans/public_oasis.md](docs/plans/public_oasis.md) §Phase 2 all hold. `hive contribute`
  remains write-only, unchanged from 1.2 and from 1.4's stated posture.
- **The discovery-only website stance is PRESERVED, not reversed.** `pymaxim.bio` keeps linking to
  Oases rather than routing to them; the Oasis is a peer served from the rig on its own subdomain,
  which [docs/plans/maxim_hivemind.md](docs/plans/maxim_hivemind.md) already anticipates
  (*"Public Oasis — eventual reference instances … that anyone can connect to"*) under a topology
  that stays flat (*"No hierarchy … The Hivemind mesh has no root"*). Adopting the Queen role is a
  per-Oasis role, never a canonical root. The hosted-Console non-goal is untouched.
- **Entry condition:** the Phase 0 prerequisites — a CI lane that installs `cryptography` (1.3.1
  scope), the public format-freeze pass, a human privacy read of the exemplar's key material, and a
  licensing posture for published bundles.

Reason:

- The receiver's trust boundary is an operator-typed allowlist: `hivemind/ingest.py::ingest_bundle`
  refuses any `contributor_id` absent from `trusted_sources`, and `substrate ingest --trust` is
  `required=True`. The frozen threat model designates everything behind that door defense-in-depth
  rather than the boundary, and says the clamps *"bound magnitude, not INTENT."* Accepting strangers
  would delete the only trust decision and substitute nothing: the Queen promotion gauntlet was
  deferred 2026-09-06 with four prerequisites and none has landed.
- Publication, by contrast, adds **no new trust decision at all**. Consumers verify the Queen
  signature, never the host that served the bytes, so a mirror is untrusted by construction — the
  same property that made the Hugging Face plan's Phase 1 cheap.
- It makes the domain real, exercises the pull path against real strangers, and hardens the bundle
  wire boundary in public, without spending 1.4's ladder.

Tradeoffs:

- An uptime promise on hardware that also runs experiments; §Open questions Q2 owes an availability
  posture and a what-the-site-says-when-it-is-down answer.
- The "give back" half of the story stays aspirational and must be described that way — never
  rounded up.
- Publishing freezes the bundle shape in public: future format changes now break strangers, not two
  coordinated repos.
- Re-opening a deferred decision point for one half invites pressure to open the other. The six
  Phase 2 conditions exist so that pressure meets a list rather than a mood.

## 2026-09-19 — 1.4 re-pointed to the survival line; "Shared perception" deferred on a physical trigger

Decision:

- **1.4 continues the survival world** — generalization, multi-step credit and a trajectory
  instrument (`docs/plans/roadmap_1_4.md`, five-lens reviewed 2026-09-18) — instead of the
  "Shared perception" release (perception fabric + microduck + Exp 55 + breeding).
- **"Shared perception" is DEFERRED on a PHYSICAL trigger, not a date:** it revives the day a
  second robot body exists (a real backend registered through `maxim.robots`, or the operator
  records its arrival). Its plan stays intact (`docs/plans/deferred/second_body_staging.md`) and Stage A (the
  baseline measurement of the new body) runs unchanged on revival.
- **The deferred JEPA plan is re-pointed, not revived.** It is a PROJECTION layer (384-dim sensor ↔
  768-dim language); the survival line may need a PREDICTOR (a latent forward model), which is a
  different mechanism. A predictor enters only through its own plan, after the shipped Cerebellum
  forward model and `anticipatory_pre_activate` are audited.

Reason:

- The microduck is backordered for an unknown time; planning a release around hardware that does
  not exist turns a may-fail experiment into an indefinite wait.
- The survival line has momentum, an instrument (R3) and a working rig, and the three gaps it names
  (R1 exact-key substrate, R4 call-window credit, a reactive-only fear) are the next honest questions.

Tradeoffs:

- 1.4 has no new modality; the perception claims move out at least one release.
- A deferral with a physical trigger has no calendar check — the plans audit reads the README
  §Deferred entry, which names the trigger.

## 2026-09-19 — Variant bodies for new classrooms; what they shield and what they do not

Decision:

- A classroom that needs new affordances or sensors uses a **variant body**
  (`extends: bodies/minecraft_player`), never an edit to the shipped body, so the shipped body's
  EARNED rows (Exp 60, Exp 61) keep their apparatus.
- The variant shields ONLY body-change triggers. It does NOT shield a bridge protocol change, a
  `recommend_action` change, a credit-path change, a harness change against R3's §Outcome clause,
  a substrate/EC rule, or the minor-version heartbeat. Each of those fires by its letter and is
  re-run or discharged with a dated annotation on the ledger row.
- A variant renames every tool signature (`{body}_{affordance}`): cluster-keyed fear carries across
  bodies, biases do not, and bundle ingest across bodies refuses at gate 7. A rung on a variant
  re-runs its R3 baseline arms on the variant before it measures against them.

Reason: the 1.4 review (scope + wiring lenses) showed "a variant body" read as a general shield
while most 1.4 changes fire triggers the body never touches.

Tradeoffs: every new classroom pays a small baseline re-run; in exchange no EARNED row changes
apparatus silently.

## 2026-09-19 — A release is named at the transaction from its highest EARNED result

Decision:

- A release's working title is not its name. The name is fixed in the release transaction from
  the highest rung with a recorded EARNED; the CHANGELOG headline claims exactly that rung, names
  the highest rung attempted, and never describes a mechanism that did not enter. A release with
  no earned claim ships under an instrument name (as 1.1.4 "The world seam" did).
- A "recorded outcome" means a prereg frozen on main before the first data timestamp, a
  merge-committed data PR at one hash, a COMPLETE report (or amendments folded and floor-reviewed),
  and a §Outcome that leads with the frozen status verbatim.

Reason: 1.4's working title "Anticipation" names the mechanism its last rung (E3) tests, and E3 may
not run or may record a null before 1.4.0 ships. 1.2.1's headline ("end to end") is the recorded
example of a name outrunning what shipped.

Tradeoffs: a less evocative name when a hard rung misses; the name stays true.

## 2026-09-12 — Harness-injected interoceptive signals: a labelled second lane beside D1

Decision:

- **D1 (game-native pressure only — no synthetic sensor, no bespoke reward) continues to govern
  ENVIRONMENT-DRIVEN claims** — any claim of the form "the environment's / game's own drives moved
  behaviour" (e.g. the R2 survival-loop flip). These use ONLY game-exposed state and game-native
  reward; no injection, ever.
- **A separate, explicitly-labelled lane** permits harness-injected interoceptive signals (e.g. a
  homeostatic PAIN valence for an out-of-bounds state the environment under-models) as the
  **INDEPENDENT VARIABLE** of a **substrate-mechanism** claim: "given signal X, the substrate
  integrates it into its policy." Four rules make the lane safe:
  1. **World-state stays game-truthful** — inject a pain SIGNAL / body valence only; never fabricate
     damage or a state the environment cannot itself produce.
  2. **Harness-only** (never `src/`, never shipped), **deterministic**, and declared in the
     experiment's pre-registration.
  3. **The claim is scoped to the substrate's integration of the injected signal — NEVER "the
     environment taught it."** The injected signal is named as the manipulated variable in the
     prereg. (This is the load-bearing rule.)
  4. **Game-native-premise rungs never use injection** — the R2 flip (rung 1) and any
     environment-driven claim stay strictly game-native.

Reason:

- The bio-inspired substrate's value includes integrating interoceptive pain/valence the way a body
  does, but a test world (Minecraft) under-models many embodiable costs (over-fullness caps at 20
  with no penalty; the game never punishes it). Studying whether the substrate correctly integrates
  such a signal is legitimate — and is a DIFFERENT claim from "the environment afforded the lesson."
  D1 as written would forbid the study; a blanket exception would erode into outcome-engineering
  (the Exp 42b-retraction family: measuring a possibility, presenting it as proof it happened). The
  labelled-lane + claim-boundary keeps both honest without weakening D1.

Tradeoffs:

- The boundary between the two lanes is a CLAIM-discipline line, not a mechanical one — enforced by
  prereg review (rule #3 stated in every injected-signal prereg), not a lint. The risk is a future
  reader relaxing it into "we can inject when convenient"; rule #3 up front is the guard.
- Scope: injection is for mechanism rungs only. The first consumer is the planned R2 homeostasis
  rung (rung 2), NOT the R2 flip (rung 1, game-native). See `docs/experiments/r2_learned_bias_prereg.md`.

## 2026-08-19 — `maxim.run()` uses canonical ingress and owns its resources (D15/D16)

Decision:

- A non-`None` `goal` is seeded into `RuntimeState.pending_cli_input`, the same
  ingress consumed by interactive CLI input, so prefetch, memory capture, and LLM
  submission keep one path.
- `robot` requires `headless=False`; contradictory arguments raise
  `ConfigurationError` instead of silently ignoring hardware intent.
- Robot connection happens before tool-registry/executor construction. The
  selected controller is wired into the agent context and direct-motion path;
  controller-bound tools neither advertise nor accept `robot_id`, so a model
  cannot redirect a command to unrelated process-global hardware.
- `run()` wakes a controller that was asleep and attempts to restore that prior
  state on exit. It disconnects only registrations it atomically created; live
  pre-existing connections remain caller-owned. If safe sleep or disconnect
  cannot be confirmed, cleanup raises `HardwareError` and retains the
  registration for recovery.
- Controller-only runs do not advertise legacy capture, vision, command, or DoA
  tools whose required full-runtime state is absent.
- `run()`'s cleanup boundary begins before its LLM environment overrides and
  runtime-resource acquisition. Worker stop, bio shutdown, robot lifecycle, and
  restoration of those two overrides are independently guarded.
- Only one `run()` may be active per process because model routing uses process
  environment state. A second call fails with `ConfigurationError`.
- `goal` is initial input, not a lifecycle bound: goal completion does not stop
  the service loop. `goal=None` starts idle and the Python facade installs no
  terminal-input reader.

Reason:

- `goal` and `robot` were stable headline arguments with no effective runtime
  behavior, while setup exceptions could leak threads, connections, and process
  environment changes.

Tradeoffs:

- A caller must now say `headless=False` explicitly when requesting hardware.
  This preserves the meaning of both stable arguments instead of letting one
  silently override the other.
- CWD-relative loop-state persistence is still outside complete `home_dir`
  ownership, and equivalent early-cleanup work for `imagine()`/`campaign()` stays
  in the 1.1.x hardening line.

## 2026-08-19 — Simulation process status represents run integrity (D22)

Decision:

- Simulation libraries continue to return structured results and never terminate
  the embedding process.
- Process-level CLI entry points map generic `error` to exit 1 and incomplete or
  runtime-aborted outcomes to exit 4, matching the existing D12 hard-abort code.
- Benchmark, curriculum, Roy, research, and Console consumers apply the same
  centralized run-integrity classification before accepting metrics or artifacts.
- Semantic experiment verdicts (`failed`, `blocked`, `inconclusive`) remain valid
  data and therefore do not imply process failure.

Reason:

- D13/D14 made planning failures unwind cleanly with typed statuses, but the CLI
  still returned 0 and multi-run harnesses only recognized literal `error`. A
  partial campaign could therefore be counted as evidence precisely because its
  teardown worked.

Tradeoffs:

- Exit 4 covers both forced and clean aborts, so callers use `finish_reason` and
  the saved report when they need the precise cause.
- LLM-selected `failed` remains exit 0 because it describes the system under test,
  not a failure of the experimental apparatus.

## 2026-08-19 — 1.1 release closure and provider-neutral agent guidance

Decision:

- 1.1 remains under a mechanism freeze. Its remaining scope is correctness,
  stable-contract repair, verification, release truth, and completion of the
  already-started heartbeat—not Oasis, Hivemind, or another cognitive mechanism.
- `docs/plans/archive/roadmap_1_1_to_1_3.md` is the sole 1.1 scope authority. The July
  checklist is archived as a historical snapshot.
- **Single agent-guidance source, ratified 2026-08-19 in the INVERSE direction of
  this entry's first draft:** `CLAUDE.md` stays the canonical core (CI-linted,
  operator-reviewed, auto-loaded by the primary tooling); `AGENTS.md` was rewritten
  as a pointer-only provider-neutral ADAPTER with no copied checks, routing table, or
  hard rules. Its exact contents are enforced by the documentation lint. Subsystem
  knowledge remains in tracked `docs/agents/` briefs and incident history in
  `docs/lessons/`. The first draft (AGENTS.md canonical, CLAUDE.md demoted to an
  adapter) was reversed in review: it would have made the invariant lint pass
  vacuously, taxed every Claude session with indirection, and obsoleted a
  freshly operator-reviewed artifact — rationale recorded in the roadmap's
  single-source section. Any future canonical-filename migration uses content
  identity (generated copy + CI byte-check), never indirection.
- Oasis and Hivemind move to gated 1.2 work. Encoder provenance/compatibility,
  read-side EC safety, and a sharing threat model must close before implementation.

Reason:

- The 2026-08-19 review confirmed public API no-ops, planning liveness defects,
  non-hermetic required tests, a permanently red architecture audit, and release
  policy drift. Distributing state or adding mechanisms before closing those gaps
  would amplify failure modes the current runtime cannot diagnose reliably.
- Two large root instruction files had already diverged on Python support,
  dependencies, and API count. A single canonical source avoids provider-specific
  truth forks while a measured compatibility adapter avoids breaking Claude-based
  workflows by assumption.

Tradeoffs:

- 1.1 takes longer and carries less novelty, but becomes a defensible release rather
  than a bundle of already-merged features with unresolved contracts.
- The `AGENTS.md` adapter leaves two filenames in the repository, but its frozen,
  pointer-only shape prevents a second substantive instruction corpus from forming.
- The 33 current architecture findings may remain as reviewed debt in 1.1; CI must
  reject additions, and 1.1.x owns the burn-down.

## 2026-03-31 — Claw-Code Upgrade: Cognitive Pain, Coding Tools, Session Persistence

Adopted patterns from the claw-code Python port of Claude Code to improve Maxim's
coding assistant capabilities and architectural robustness:

- **Cognitive pain system** (#15): Tool errors routed through PainDetector → NAc → FearAgent.
  NAc learns which tools fail in which contexts via Rescorla-Wagner. ToolPainBridge,
  ToolHarmPredictor, MonitorRegistry created. [MonitorRegistry superseded — not in current codebase]
- **Coding tools** (#11a-d): EditFileTool (text-anchor edits), CodeSearchTool (regex search),
  RunTestsTool (structured test results), GitDiffTool, GitCommitTool.
- **Structured error vocabulary** (#9): StopReason (10 loop termination reasons),
  ToolErrorKind (7 error classifications) on ToolOutput.
- **Frozen value dataclasses** (#2): ToolOutput, LLMProposal, SkillResult [superseded — not in current codebase], LongHorizonConfig.
- **Session persistence** (#8): AgentSession [superseded — not in current codebase] with save/load, Percept serialization.
- **Context compaction** (#10): Sliding window with first-turn pinning for long-horizon plans.
- **Test-driven replan** (#12): CodingReplanContext with structured test/build failures.
- **Streaming events** (#5): StreamEvent + on_event callback for fine-grained loop events.
- **Permission extensions** (#3): SupervisionPolicy gains forbidden_prefixes, forbidden_categories.
- **Architecture audit** (#1): AST-based import validator, --audit-architecture CLI flag.
- **RuntimeCapabilities** (adaptive): Headless mode without robot, graceful degradation.
- **Bugfixes**: wave_scores crash, worker_pool prune, hippocampus atomic load, LLM timeout
  mutation, bare except:pass, recall_similar lock split, tool execution timeout.

## 2026-03-13: Conscience mixin decomposition and agents/ module extraction

> **Note:** The package was renamed `conscience/` → `embodied_runtime/` (commit ed59cc2, 2026-04-10, during v0.2.x development) to better describe its contents (robot mixin stack, not safety enforcement). All file references below use the new path.

Decision:
- `src/maxim/embodied_runtime/selfy.py` `Maxim` class decomposed into six mixins: `ConnectionMixin` (connection.py), `VisionStreamMixin` (vision_stream.py), `AgenticRuntimeMixin` (agentic_runtime.py), `MovementMixin` (movement.py), `InputHandlerMixin` (input_handlers.py), `MediaLoopMixin` (media_loop.py). Module-level worker functions live in `workers.py`.
- `src/maxim/agents/llm_worker.py` split into focused modules: `llm_types.py` (request/response dataclasses), `llm_context.py` (context building), `prompt_budgeter.py` (token budget management), `llm_fallback.py` (fallback behaviors), `prompt_builder.py` (prompt construction). All are re-exported from `llm_worker.py` for backward compatibility.

Reason:
- `selfy.py` had grown too large; decomposition improves readability and allows independent testing of each concern.
- `llm_worker.py` mixed data types, context logic, and worker thread code; extraction clarifies responsibilities.

Tradeoffs:
- More files to navigate, but each is focused and independently testable.
- Re-exports preserve all existing import paths.

## 2026-03-08: Contemplation loop for local chain-of-thought in ExecAgent

Decision:
- ExecAgent uses a multi-pass contemplation loop (draft → critique → refine) to improve plan quality when native extended thinking is unavailable.
- Contemplation triggers only for complex plans (2+ sub_goals or HIGH/CRITICAL priority) and is mutually exclusive with Anthropic extended thinking.
- Two modes: `standard` (separate critique and refine calls, 3 passes max) and `fast` (combined in one call, 2 passes max).
- Smart preemption: only urgent percepts (CLI, voice, comms, high-urgency escalations) interrupt contemplation. Normal vision percepts queue.
- Quality metrics: outcomes tracked via `GoalCompleted` bus subscription, fed to NAc for causal learning.
- Adaptive thresholds: NAc-learned outcomes auto-tune `confidence_threshold` and `min_sub_goals_to_trigger` within configurable bounds.
- Config stored in `LLMConfig.contemplation` as `tuple[tuple[str, Any], ...]` (frozen dataclass compatible), configured via `llm.json` `contemplation` key.

Reason:
- Local LLMs (1.7B–8B) produce lower-quality plans in a single pass. A structured self-critique loop improves output quality without changing the underlying model.
- Extended thinking is Anthropic-only. This provides equivalent capability for any provider.

Tradeoffs:
- Adds 2–3x latency for complex plans (~15–25s on local 1.7B vs ~10s without).
- Extra LLM calls consume additional energy (tracked via existing energy system).
- Contemplation can only improve plans, never destroy them — any failure returns the original draft unchanged.

## 2026-02-25: Optional cloud LLM backends with routing + cost tracking
Decision:
- Cloud LLM providers (Anthropic/OpenAI) integrate through `LLMRouter` (no parallel gateway).
- Cloud usage is explicit opt-in (`cloud_enabled: true`) and requires a redaction policy.
- Cost tracking persists to `data/util/cost_state.json` (migrated to `~/.maxim/util/cost_state.json` in later releases via `resolve_user_state`); audit logs append to `data/logs/cloud_audit.jsonl`.
- Budget enforcement uses routing policy thresholds with graceful fallback to local models.

Reason:
- Preserve the existing local LLM path while enabling cloud quality when explicitly requested.
- Prevent accidental data egress and unbounded spend.

Tradeoffs:
- Requires additional config for cloud usage (keys, redaction policy, pricing).
- Adds routing logic and state to LLMRouter.

## 2026-01-18: Agentic MaximAgent naming + GPU-gated runtime
Decision:
- The composite agentic implementation is now `MaximAgent` (alias `AgenticMaximAgent` preserved for compatibility).
- Removed `ReachyMiniAgent`; agentic control now relies on tools for Reachy SDK access.
- `--mode agentic` requires GPU availability before starting.
- Vision events are streamed to `data/vision/vision_events_<run_id>.jsonl` and surfaced as `latest_vision_event`.
- `execute_file` tool execution is opt-in via `MAXIM_ALLOW_EXECUTE_FILE=1`.

Reason:
- Align the primary agent name with the agentic implementation.
- Keep agents action-free and centralize SDK control in tools.
- Avoid running the agentic loop without accelerator support.
- Feed vision detections into the agentic perception loop without blocking control.
- Reduce the risk of transcript-triggered arbitrary file execution.

Tradeoffs:
- Existing imports referencing the old class names should update (aliases remain).
- Agentic runs now exit/skip on GPU-less machines.

## 2026-01-07: Add interactive terminal input for keyword actions
Decision:
- `maxim` starts a line-based terminal prompt (`maxim>`) when `--interactive true` (default).
- Prompted input is matched against `phrase_responses.json` (same as voice triggers).
- Single-key shortcut mode is disabled while interactive prompt input is enabled to avoid stdin conflicts.
- Interactive CLI input is recorded under `data/cli/cli_input_<run_id>.jsonl`.
- CLI and vision-overlay inputs bypass phrase cooldowns and `requires_agentic` gating.
- When the OpenCV display is active, the vision overlay includes a text input box with a Send button that routes input through the same phrase responses.

Reason:
- Provide a reliable non-audio control path for keyword actions without adding new dependencies.

Tradeoffs:
- Keypress-only shortcuts require `--interactive false` or Enter in the prompt.
- The overlay input requires a direct OpenCV display backend; process-based display modes cannot capture input.

## 2026-01-02: Queue-based capture + writer pipeline
Reason:
- Avoid blocking perception/motor control on disk I/O.
- Enable “record everything” semantics by applying backpressure (blocking queues) instead of dropping samples.

Tradeoffs:
- More moving parts (threads/process + shutdown signaling).
- When disk/CPU can’t keep up, capture blocks and effective FPS may decrease.

## 2026-01-02: Single-run artifacts (MP4 + WAV + JSONL transcript)
Reason:
- A single `videos/*.mp4` is more efficient and simpler than thousands of PNGs.
- A single `audio/*.wav` preserves a continuous audio stream; JSONL allows streaming transcript append.

Tradeoffs:
- Requires codecs/backends for MP4 writing (environment dependent).
- Large files require log/cleanup discipline.

## 2026-01-04: Store transcripts under `data/transcript/`
Decision:
- JSONL transcripts are written under `data/transcript/` (previously `data/text/`).

Reason:
- Avoid confusion with generic “text” outputs and make transcripts easier to locate.

## 2026-01-02: Whisper transcription runs in a separate process
Reason:
- Whisper inference is heavy and should not stall the control loop.
- Process isolation avoids GIL contention and keeps the rest of the system responsive.

Tradeoffs:
- Whisper dependency/model availability may be missing; transcription must degrade gracefully.
- Requires chunking audio and coordinating handoff via a queue.

## 2026-01-02: Optional audio pipeline via CLI flags
Decision:
- `--audio True/False` controls audio capture/transcription.
- `--audio_len <seconds>` controls chunk size for efficient streaming transcription.

Reason:
- Some runs are vision-only; audio should be skippable.
- Chunking balances latency (short chunks) vs throughput (long chunks).

## 2026-01-02: `--mode sleep` skips `wake_up()`
Decision:
- `--mode sleep` records/transcribes audio without running the camera/ML loop and does not call `ReachyMini.wake_up()`.

Reason:
- Support “leave motors asleep” debugging and audio-only dataset capture.

Tradeoffs:
- The run won’t auto-stop based on frame epochs; it runs until interrupted.

## 2026-01-02: Default mode is `exploration`
Decision:
- Default `--mode` is `exploration`.
- `Maxim(mode=...)` defaults to `exploration`.

Reason:
- Exploration mode actively discovers and learns about the environment.
- Maxim is immediately curious and engaged on startup.
- Aligns with the goal of building understanding through active observation.

Tradeoffs:
- Higher resource usage than passive modes like `sleep` or `reflection`.
- Users who want minimal activity can pass `--mode sleep` or `--mode reflection`.

## 2026-01-02: Per-run logs saved under `data/logs/`
Decision:
- Each CLI run writes logs to `data/logs/reachy_log_<run_id>.log`.

Reason:
- Makes runs debuggable after the fact without copying terminal output.
- Keeps artifacts grouped per session alongside video/audio/transcripts.

Tradeoffs:
- Produces additional files; users may need periodic cleanup.

## 2026-01-02: Inference code lives under `src/maxim/inference/`
Reason:
- Keep "runtime inference/control" separate from "robot orchestration" (`src/maxim/embodied_runtime/`, formerly `conscience/`) and "model definitions" (`src/maxim/models/`).

Tradeoffs:
- Requires stable re-export modules to preserve import paths during refactors.

## 2026-01-02: Vision via pluggable engine (RTMDet-m default, YOLOv8 optional)
Reason:
- Fast, general-purpose perception for “person/object of interest” detection.
- Pose keypoints enable eye/face target refinement when available.
- RTMDet-m + RTMPose-m (ONNX Runtime, Apache 2.0) is the default engine; YOLOv8 is available via `pip install “maxim[yolo]”` (AGPL-3.0).
- Vision engine registry in `src/maxim/models/vision/registry.py` maps names (“rtm”, “yolo”) to engine implementations.

Tradeoffs:
- Heavier runtime dependency; performance depends on hardware.
- Model weights and backends vary by environment.

## 2026-01-02: MotorCortex uses ConvNeXt-Tiny backbone
Decision:
- MotorCortex predicts head movement deltas: `[x, y, z, roll, pitch, yaw, duration]`.

Reason:
- Strong image feature extractor that trains well for regression with minimal custom code.

Tradeoffs:
- Requires TensorFlow/Keras for training/inference in this repo’s implementation.

## 2026-01-02: Add `maxim` CLI entrypoint
Decision:
- `pip install -e .` installs a `maxim` console script (entrypoint: `maxim.cli:main`).
- The importable package is `maxim` (code lives under `src/maxim/`); `src.*` imports are removed.
- `python scripts/main.py` remains supported as a compatibility entrypoint.

Reason:
- Reduce friction for new users (no need to remember the module/file path).
- Avoid confusion from a top-level package named `src`.

## 2026-01-02: JSON-configured key responses
Decision:
- Maxim loads `data/util/key_responses.json` on startup and listens for terminal key presses while running (override via `$MAXIM_KEY_RESPONSES`).

Reason:
- Allow quick, extensible runtime actions (e.g., recenter vision) without impacting the control loop.

## 2026-01-04: Training sample log under `data/training/`
Decision:
- When vision-driven movement is initiated, Maxim appends a JSONL record to `data/training/motor_training_set.jsonl` via a background writer.
- The `u` key writes a marked record (`user_marked=true`) for the most recent sample.

Reason:
- Keep an always-on stream of “trainable moments” for MotorCortex without blocking the control loop.
- Make it easy to curate a subset of samples for training by marking moments during a run.

Tradeoffs:
- Samples reference run artifacts (video/audio/transcript paths + timestamps); extracting frames is a post-processing step.

## 2026-01-04: Phrase-triggered actions from transcripts + event labels
Decision:
- Maxim can trigger actions from transcribed speech using `data/util/phrase_responses.json` (override via `$MAXIM_PHRASE_RESPONSES`).
- The default wake words are `Maxim` and `Reachy`, which call `wake_up()`, start the agentic runtime loop, and enable voice-triggered actions.
- Voice commands `Maxim shutdown`, `Maxim sleep`/`sleep maxim`, and `Maxim observe`/`observe maxim` request clean shutdown / mode switches (the CLI restarts Maxim into the requested mode).
- When a non-wake command phrase matches, wake-word triggers are suppressed for that transcript line to avoid double actions.
- Transcript text is normalized before matching (punctuation/possessives stripped; common alias `maximum` → `maxim`).
- When `maxim` is present in a transcript line, Maxim also attempts to infer the best matching non-wake command from the remaining words before falling back to the wake action (and does not re-fire the wake action once enabled).
- Runtime events (voice/key actions + user outcome labels) are appended to `data/training/action_events.jsonl` via the same background writer used for training samples.
- Keys `0`–`9` are reserved for simple outcome labels (`0` = no errors; `1`–`9` = generic error/odd behavior codes).

Reason:
- Tie transcripts, actions, and “trainable moments” together via time-aligned JSONL logs.
- Support lightweight human-in-the-loop labeling during runs without blocking the control loop.

## 2026-01-05: Optional local LLM routing for wake-word transcripts
Decision:
- When the agentic runtime is running, transcript lines that contain the wake word (`maxim` + common variants like `maximum`) may be routed through an optional local LLM to produce a single agentic action (`{"tool_name": ..., "params": ...}`).
- Hard keyword commands for mode switching (`sleep/observe/shutdown` with `maxim`) always override LLM routing.
- LLM configuration is stored in `data/util/llm.json` (override via `$MAXIM_LLM_CONFIG`) and is disabled by default.
- Initial reference backend uses `llama-cpp-python` (local GGUF) with built-in profiles for Mistral 7B and SmolLM 1.7B.
- LLM backends live under `src/maxim/models/language/` to keep them swappable.

Reason:
- Keep voice control deterministic for critical mode switches while enabling richer, optional transcript-driven behaviors when compute is available.

## 2026-01-06: Configure Whisper compute type via env var
Decision:
- `MAXIM_WHISPER_COMPUTE_TYPE` controls the `faster-whisper` compute type for transcription (default: `int8`).

Reason:
- Provide a safe fallback for Linux/WSL segfaults in CTranslate2/Whisper without code edits.

Tradeoffs:
- `float32` is slower and may increase CPU usage; `int8` is faster but less stable on some systems.

## 2026-01-06: Disable OpenCV display in headless/non-main thread runs
Decision:
- `MAXIM_DISABLE_IMSHOW=1` or `MAXIM_HEADLESS=1` skips `cv2.imshow` calls to avoid Qt/GTK thread crashes on WSL/headless setups.
- `MAXIM_IMSHOW_MODE=process` runs `cv2.imshow` in a dedicated process to keep GUI calls on that process's main thread.
- On Linux/WSL, default to the display process; set `MAXIM_IMSHOW_MODE=direct` to force main-thread imshow.

Reason:
- OpenCV GUI backends often crash when invoked from non-main threads or without a display server.

Tradeoffs:
- No on-screen visualization during runs; rely on saved videos/logs instead.
- Display process adds IPC overhead and may drop frames under load.

## 2026-01-05: CLI model selection flags
Decision:
- `--language-model <profile>` overrides the LLM profile for the run (prints available profiles on unknown).
- `--segmentation-model <name>` selects the vision engine (default: `rtm`; options: `rtm`, `yolo`; prints available models on unknown).

Reason:
- Make per-run experimentation easier without editing JSON/env vars.

## 2026-01-04: Agentic decision flow + single point of decision
Decision:
- Action selection happens in exactly one place: `src/maxim/planning/decision_engine.py`.
- Canonical flow (no skipping): Observe state → Agents propose intents → Planners propose candidate plans → Policies constrain plans → Decision engine selects one next action → Runtime executes.
- Planners generate plans but do not select final actions or mutate state.
- Policies are deterministic/auditable guardrails and do not plan or execute.
- “Hidden decisions” are forbidden: if a component chooses between alternatives, prioritizes options, or suppresses actions, that logic belongs in the decision engine.

Reason:
- Keep behavior predictable, testable, and debuggable as the codebase grows.
- Prevent side effects and control-flow decisions from leaking into the wrong layers.

Tradeoffs:
- Requires discipline and occasional refactors to keep boundaries intact.
- Some features may need more explicit state representation and dependency injection to remain testable.

## 2026-01-04: Standardize agentic plan/action schema
Decision:
- Canonical action schema: `{"tool_name": <str>, "params": <dict>}`.
- Canonical plan schema: `list[action]`.
- `DecisionEngine.decide()` returns a dict containing the selected `action` and its `plan` context.
- Agentic orchestration lives under `src/maxim/runtime/` and executes actions via `Executor` + `ToolRegistry`.

Reason:
- Keep planner outputs, policy checks, evaluators, and runtime execution interoperable.
- Reduce “stringly-typed” ambiguity and make plans serializable/debuggable.

## 2026-01-04: Persist agentic runtime state under `data/agents/`
Decision:
- Agentic runtime state snapshots are persisted to `data/agents/<STATE_NAME>/runtime/state_<run_id>.json`.
- `STATE_NAME` defaults to `Agent.agent_name`, but agents may override it via `state_name`.

Reason:
- Support resuming/debugging agent runs with a durable, per-agent state artifact outside the installed package.

## 2026-01-04: Add `--mode agentic` to the CLI
Decision:
- `maxim --mode agentic` runs the agentic runtime loop (`src/maxim/runtime/`) instead of the Reachy orchestration loop.

Reason:
- Provide a first-class entrypoint for agentic development/testing without requiring robot connectivity.

## 2026-01-04: Agentic runtime defaults
Decision:
- `--mode agentic` runs the composite `MaximAgent` (agentic architecture).
- Alternate agent selection is not exposed via the CLI at the moment.

Reason:
- Keep the agentic entrypoint focused on the primary architecture.

Tradeoffs:
- Switching agents requires code changes rather than a CLI flag.

## 2026-01-04: Keep agents in independent files
Decision:
- Each agent implementation should live in its own file under `src/maxim/agents/` (e.g., `maxim_agent.py`, `goal_agent.py`).
- `src/maxim/agents/base.py` should only contain shared interfaces/helpers (`Agent`, `AgentList`, utilities).
- Exception: agents that share nearly all logic via inheritance (or are tightly coupled variants) may be co-located.

Reason:
- Improves discoverability and reduces unrelated coupling as the agent set grows.

## 2026-01-03: Store motion presets under `data/`
Decision:
- Default motion actions load from `data/motion/default_actions.json`.

Reason:
- Keep editable JSON configs separate from code and easy to find.

## 2026-01-05: Preflight Matplotlib font cache before loading vision models
Decision:
- Before loading vision models, Maxim runs a Matplotlib font-cache preflight in a subprocess.
- The preflight uses `MPLCONFIGDIR` under the run's home directory when not already set and forces `MPLBACKEND=Agg`.
- `MAXIM_SKIP_MPL_PREFLIGHT=1` bypasses the preflight when needed.
- Maxim also preloads Matplotlib in-process early (before Reachy/GStreamer/Ultralytics init) to stabilize native font libs.
- `MAXIM_SKIP_MPL_PRELOAD=1` bypasses the early preload when needed.

Reason:
- On Linux/WSL, Matplotlib + FreeType can abort while scanning fonts; preflighting isolates failures and surfaces actionable errors.

Tradeoffs:
- Adds a small startup cost to vision initialization.
- Users may need to clean or repair system fonts if preflight fails.

## 2026-01-06: Allow disabling VAD filter for faster-whisper transcription
Decision:
- `MAXIM_VAD_FILTER=0` disables the faster-whisper VAD filter when running the transcription worker.

Reason:
- VAD uses onnxruntime (Silero ONNX) and can segfault on some Linux builds; the toggle lets users isolate or bypass that path.

Tradeoffs:
- Without VAD, transcription may be slower and include more silence.

## 2026-01-06: Default epochs to unlimited
Decision:
- CLI `--epochs` defaults to `0` (unlimited).
- `Maxim` treats epochs `<= 0` as unlimited; `Maxim.live(epochs=...)` overrides.
- Agentic `max_steps <= 0` runs without a step cap.

Reason:
- Prevent unexpected stops in long-running sessions unless the user explicitly sets a limit.

Tradeoffs:
- Users must pass `--epochs` to cap runtime by default.

## 2026-01-04: Store head poses under `data/motion/default_poses.json`
Decision:
- Default head poses (including the `centered` pose used by the `c` key) load from `data/motion/default_poses.json`.

Reason:
- Allow robot-specific calibration of “centered” without changing code.

## 2026-01-05: Clamp head movement step size
Decision:
- Head movement commands are clamped per call using `data/motion/movement_thresholds.json` to avoid large, sudden jumps.

Reason:
- Improve stability/safety and make movement behavior tunable without changing code.

## 2026-01-03: Store trained models under `data/models/`
Decision:
- Default model artifacts (MotorCortex checkpoints/history, vision engine weights) live under `data/models/`.

Reason:
- Keep model artifacts separate from run outputs under `data/`.

## 2026-01-03: Extract reusable helpers from nested defs
Decision:
- Avoid defining reusable helper functions inside other functions/methods.
- Put cross-cutting helpers under `src/maxim/utils/` (or at module scope) and import them where needed.

Reason:
- Improve reuse and reduce duplicated logic while keeping runtime loops readable.

## 2026-01-04: Keep Python code under the `maxim` namespace
Decision:
- Importable code lives under `src/maxim/` (packaged as `maxim*`).
- Avoid creating new top-level packages under `src/` (e.g., `src/agents/`) unless `pyproject.toml` explicitly includes them.

Reason:
- Ensures `pip install -e .` installs everything needed for imports and avoids collisions with overly-generic package names.

## 2026-01-19: Architecture migration - `live()` as hardware I/O layer

### Current State

The system has two parallel control paths:

1. **`live()` loop** (`src/maxim/embodied_runtime/selfy.py`, decomposed into mixins):
   - Hardware I/O: frame capture, audio capture, video/audio writing (`MediaLoopMixin`, `workers.py`)
   - Connection lifecycle (`ConnectionMixin`)
   - Vision capture and segmentation (`VisionStreamMixin`)
   - Agentic runtime bootstrap (`AgenticRuntimeMixin`)
   - Motor command helpers (`MovementMixin`)
   - CLI/keyboard/voice input routing (`InputHandlerMixin`)
   - Transcription pipeline: spawns Whisper process for speech-to-text
   - Observation functions: `passive_observation()` / `motor_cortex_control()`
   - Display: shows annotated frames via OpenCV

2. **Agentic runtime** (`src/maxim/runtime/`, `src/maxim/agents/`):
   - PerceptionAgent: processes frames/audio into Percepts
   - MemoryAgent: builds StructuredContext from percepts
   - AgenticGoalAgent: proposes goals based on context
   - ExecAgent: executes goals via tool calls
   - AutonomyController: gates tool execution by autonomy level
   - LLMWorker: non-blocking LLM inference

### Intended Migration Path

**Phase 1 (Current):** Keep both paths, document boundaries.
- `live()` remains the hardware interface layer
- `passive_observation()` / `motor_cortex_control()` are fallbacks when agentic runtime is inactive
- Agentic runtime consumes data from `live()` via shared state (`_last_frame`, transcripts)

**Phase 2:** Make agentic runtime the primary decision-maker.
- `live()` becomes a pure capture/recording layer (no observation logic)
- All perception → decision → action flows through the agentic system
- `passive_observation()` becomes a simple "display frame + detections" helper
- Remove `motor_cortex_control()` training logic (training moves to offline pipeline)

**Phase 3:** Merge capture threads into agentic runtime.
- Move frame/audio capture workers into `_start_agentic_runtime()`
- `live()` becomes a thin wrapper that starts the agentic runtime
- Single entry point for all modes (sleep/observe/agentic)

### Key Boundaries (Current)

| Component | Responsibility | Does NOT do |
|-----------|---------------|-------------|
| `live()` | Hardware I/O, recording, display | Decision-making, goal selection |
| `passive_observation()` | Legacy fallback: segment + display + simple tracking | Goal proposal, LLM reasoning |
| PerceptionAgent | Convert raw data → Percepts | Movement commands, tool calls |
| MemoryAgent | Build context, manage memories | Propose goals, execute actions |
| AgenticGoalAgent | Propose goals from context | Execute tools directly |
| ExecAgent | Execute approved actions via tools | Propose goals, bypass autonomy |
| AutonomyController | Gate tool execution | Make decisions, propose goals |

### Migration Checklist

**Phase 2 (Completed 2026-01-19):**
- [x] Display logic extracted from `passive_observation()` into `display_detections()` standalone helper
- [x] `passive_observation()` simplified to display-only (returns target info, no movement)
- [x] `motor_cortex_control()` removed from `live()` observation loop
- [x] `live()` now auto-starts agentic runtime when not in sleep mode
- [x] Target info stored in `_last_detection_target` for agentic system access

**Phase 3 (Completed 2026-01-19):**
- [x] Created `CaptureManager` class (`src/maxim/runtime/capture.py`) for unified frame/audio capture
- [x] PerceptionAgent directly receives frames via CaptureManager callbacks (bypasses JSONL polling)
- [x] MaximAgent accepts `capture_manager` parameter and passes to PerceptionAgent
- [x] `live()` observation loop uses CaptureManager's pre-segmented frames when available
- [x] `display_detections()` updated to handle both tuple and dict detection formats
- [x] Single entry point: `live()` is the unified entry (sleep/observe/agentic modes)
- [x] `sleep()` calls `live(vision=False, motor=False, wake_up=False)`

### Current Data Flow (Phase 3)

```
CaptureManager (agentic runtime)
    ↓ direct frame capture + vision engine segmentation (RTMEngine default)
    ↓ callback notification
PerceptionAgent._on_captured_frame()
    ↓
Percept published to AgentBus
    ↓
MemoryAgent → StructuredContext
    ↓
AgenticGoalAgent → Goals
    ↓
ExecAgent → Tool calls (via AutonomyController)

live() display loop (parallel)
    ↓ polls CaptureManager.get_latest_frame()
    ↓ display_detections() for visualization
```

### Key Changes in Phase 3

1. **CaptureManager** (`src/maxim/runtime/capture.py`):
   - Unified capture for frame and audio data
   - Direct vision engine segmentation in capture thread (RTMEngine by default)
   - Callback-based notification to PerceptionAgent
   - Bypasses JSONL intermediary for lower latency

2. **PerceptionAgent updates**:
   - Accepts optional `capture_manager` in constructor
   - Registers `_on_captured_frame()` callback for direct frame processing
   - `process_captured_frame()` public API for manual frame processing

3. **Detection format normalization**:
   - `_normalize_detection()` helper handles both tuple and dict formats
   - `display_detections()` works with CaptureManager's dict output

### Tradeoffs

- **Keeping `live()` for recording:** Still needed for video/audio file writing; CaptureManager focuses on agentic perception
- **Dual capture paths:** CaptureManager captures for agentic system; live()'s threads still write to disk
- **Fallback observation:** `passive_observation()` still available when CaptureManager unavailable
- **Direct callbacks:** Lower latency but tighter coupling between CaptureManager and PerceptionAgent

## 2026-01-19: Active visual tracking via TrackTargetTool

### Decision

Added `TrackTargetTool` to enable the agentic system to actively move the head to center detected objects of interest.

### Implementation

**New Tool:** `track_target` (`src/maxim/tools/reachy.py`)
- Reads detection targets from CaptureManager (Phase 3) or `_last_detection_target` (fallback)
- Computes if target is outside configurable deadzone from frame center
- Calls `look_at_image()` to move head and center the target
- Parameters:
  - `deadzone_px`: Minimum offset from center to trigger movement (default: 40)
  - `duration_s`: Movement duration in seconds (default: 0.3)
  - `prefer_people`: Prioritize people over other objects (default: true)

**ExecAgent Changes:**
- Added `track_target` to available tools in system prompt
- Updated default behavior: when detections present, propose `track_target` instead of just `focus_interests`
- Added guidelines encouraging tracking behavior for people/objects

**Flow:**
```
CaptureManager → vision detections (RTMEngine default)
    ↓
ExecAgent sees detected_objects/detected_people in StructuredContext
    ↓
Proposes track_target goal (MEDIUM priority)
    ↓
TrackTargetTool reads latest detections
    ↓
If target outside deadzone: look_at_image(u, v, duration)
    ↓
Head centers on target
```

### Reason

- Enables proactive visual engagement without explicit voice commands
- Makes Maxim appear more attentive and aware of surroundings
- Leverages existing detection pipeline for active tracking
- Respects deadzone to avoid jitter from small movements

### Tradeoffs

- **Continuous movement:** May be distracting; deadzone helps mitigate
- **Rate limiting:** 10 Hz cap prevents excessive motor commands
- **LLM-optional:** Default tracking works without LLM; LLM can propose higher-priority goals to override
- **People preference:** May miss interesting non-person objects when people are present

## 2026-01-19: Enhanced Verbosity System for Agentic Information Flow

### Decision

Updated the verbosity system to provide granular control over agentic logging, making it easier to debug and observe the perception-memory-goal-action pipeline.

### Implementation

**New Verbosity Levels** (`src/maxim/utils/structured_logging.py`):
- **Level 0 (QUIET):** Errors and critical warnings only
- **Level 1 (NORMAL):** Key events - goal proposals, tool executions, mode changes
- **Level 2 (VERBOSE):** + Perception events, memory updates, autonomy decisions
- **Level 3 (DEBUG):** + Loop iterations, rate limiting, internal state changes

**Event Categories:**
Events are categorized with minimum verbosity levels:
```python
EVENT_VERBOSITY = {
    # Level 0: Always shown
    "error": 0, "critical": 0, "hard_stop": 0,

    # Level 1: Key events
    "goal_proposed": 1, "tool_called": 1, "tool_result": 1,
    "mode_change": 1, "action_executed": 1, "action_rejected": 1,

    # Level 2: Detailed events
    "percept": 2, "detection": 2, "memory_store": 2,
    "autonomy_check": 2, "intent_proposed": 2,

    # Level 3: Debug events
    "loop_iteration": 3, "rate_limited": 3, "idle": 3,
}
```

**CLI Arguments:**
- `--agentic-verbosity {0,1,2,3}`: Set agentic logging verbosity (default: 1)
- `--no-agentic-console`: Disable agentic event output to console (enabled by default)

**Environment Variables:**
- `MAXIM_AGENTIC_VERBOSITY`: Default verbosity level (0-3)
- `MAXIM_AGENTIC_CONSOLE`: Enable console output ("1", "true", "yes")

**New API Functions:**
```python
# Configure globally
configure_agentic_verbosity(verbosity=2, console_output=True)

# Log directly to abstraction stream
log_agentic("track_target", "detection", {"target_u": 500, "is_person": True})

# Get buffer for inspection
buf = get_abstraction_buffer()
print(buf.get_recent_human(10))  # Human-readable output
print(buf.get_summary())  # Event/source counts
```

**LogRecord Formats:**
- `to_compact()`: Minimal JSON for LLM context (`{"t":1234.5,"s":"exec_agent","e":"goal_proposed"}`)
- `to_verbose()`: Full field names for debugging
- `to_human()`: Human-readable console format (`12:34:56.789 [exec_agent] goal_proposed | tool=track_target`)

**Agent Loop Integration:**
The `run_agentic_loop()` now logs throughout the pipeline:
- Loop iterations (level 3)
- Intent proposals from agent fallback (level 2)
- Autonomy checks (level 2)
- Tool calls and results (level 1)
- Errors and hard stops (level 0)

### Reason

- **Debugging:** Easier to trace why actions happen or don't happen
- **Observability:** Clear visibility into the perception→goal→action pipeline
- **Flexibility:** Different verbosity for development vs production
- **Non-intrusive:** Default level 1 shows key events without flooding logs

### Tradeoffs

- **Performance:** Level 3 logging adds overhead; use level 1-2 in production
- **Storage:** AbstractionBuffer is limited to 500 entries by default
- **Complexity:** Multiple output formats (compact/verbose/human) to maintain
