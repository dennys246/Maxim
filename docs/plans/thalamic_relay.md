# Thalamic relay: the Receptor registry and AfferentTracks (grounding line, GL3)

> **Status 2026-10-24: GL3.B0 BUILT (#1201: census, seven red gates, frozen latency record). This plan's
> four-lens design review (T8 (b)) is DONE 2026-10-24 and folded: owner decisions TR18–TR24 (DECISIONS.md
> G24) settle its twelve DO-NOT-BUILDs ([§The T8 (b) four-lens design review](#the-t8-b-four-lens-design-review-2026-10-24));
> the pre-merge and delta rounds added TR28–TR29 (G28, G29).
> GL3.B1 onward still waits on the decomposition fence.**
> **PROPOSED, 2026-10-07.** **Sub-plan of the 1.4 grounding line ([grounding.md](grounding.md)),
> stage GL3.** Not a rung and not a behavioural claim. Every mechanism here enters `[engineering]`.
> Every behaviour change **this plan's stages** make ships **behind a flag that defaults OFF**, with
> today's goldens pinned byte-identical while the flag is off. The one named exception is not a GL3
> change: GL3.B4's prerequisite, #1176's per-pass transduction (TR19), is a root-cause defect fix (L1),
> owned by #1176 and landed **unflagged** in #1176's own PR, with its own trigger walk and T1-13/14/15
> re-runs, outside GL3's three flags (CLAUDE.md "no band-aid": a toggle around a defect is the shape it
> forbids; §3.4, §6 GL3.B4). **`src/` work waits for the decomposition fence** (the fence lifts when 1.3.6's decomposition slices land — the point at which T0 can pass):
> `runtime/agent_loop.py`, `runtime/loop_*.py`, `runtime/substrate_proposal.py`'s call path and
> `simulation/orchestrator.py` belong to Session A until their slices land. Stage GL3.B0 below (tests and
> census only) is built (#1201). This plan's four-lens design review is part of release threshold **T8**
> (grounding truth and contracts, engineering only, gates 1.4.0); it ran 2026-10-24 and is folded. Building
> GL3 is not required for 1.4.0. The GL3.B1 promise above holds by TR20: GL3.B1 records provenance
> always, and every behaviour it changes (imagined eligibility, the narrated discount, the credit check)
> waits behind the one OFF credit flag.
> **Stage IDs are always written with the `GL3.` prefix** (`GL3.B0` … `GL3.B8`): a bare "B8" in the
> ledger is T1-6's "B8 delta-attribution change" trigger, which is unrelated to this plan.

**Owner decisions G1, G3, G4, G6 and G8, already taken (2026-10-07), recorded here, not re-opened**
(owner decision G2, the predictor, is GL4's; all are in [grounding.md](grounding.md)):
- **G1** The grounding line fills `roadmap_1_4.md` Phase 5's slots. The body world comes first (Minecraft
  plus cradle, substrate-primary). The word world comes later, as the innate-prior tier.
- **G3** Code names are `Receptor` (a receptor class or percept source; prose may say "engine"
  informally) and `AfferentTrack`. The handoff record is `AfferentEvent`, and it carries a deterministic
  shared `PhysicalEventId` = (agent, seq), with no uuid and no wall time (amended 2026-10-09 by G15:
  (agent, session id, seq); and by G17: the type is built at the post-fence resume stage, not GL2a).
  No existing bio class is renamed. `PhysicalEventId` is **not defined here**: the autonomic plan's
  post-fence resume stage builds it in the leaf module `maxim/embodiment/event_id.py` (no maxim
  imports), and this plan **uses** it (§5.2).
- **G4** Minecraft satisfies the **perception** abstraction. The perception plans' "second body" triggers
  are re-keyed to capability triggers that fire at GL3 (§9). The robot hardware factory keeps its
  physical-robot trigger.
- **A Receptor is not a track.** A Receptor emits onto one or more AfferentTracks. One physical event fans
  out under one `PhysicalEventId`. Tracks are **logical channels scheduled on the loop clock**. They are
  never OS threads, except at hardware edges, which post into an inbox that the loop drains.
- **G6** Consequences written by the narrator's tools (`simulation/tools.py::SetEntitySensorTool`,
  `DamageComponentTool`, `OrchestratorActorTool`, and the reflex dispatch that goes through them) are
  stamped `narrated`, never `experienced`. They are usable for forward-model training and for credit at a
  **declared discount** (value: owner decision at GL4 start; strict default for that later decision: a
  small discount). Provenance kinds were `experienced` / `narrated` / `imagined`; `declared` and `reported`
  stay open at GL3.B1 (§11). A narrated record can never be relabelled experienced. *Amended 2026-10-09
  by G16:* a fourth kind, `apparatus` (harness writes to a drive: rescue, heal, respawn, teacher and
  mother feeds), recorded and never trained on or credited. (Under TR20 the "never credited" half
  holds once GL3.B1's credit flag flips; until then every kind credits as today, `apparatus` included,
  and is only recorded; pre-merge Architecture S5. The cradle mother's credit is §5.7 item 1's case.)
- **G8** The Receptor registry lands **with** provenance (one stage, GL3.B1), whose consumers are GL5's
  experiment and the forward model's contamination guard. The nociceptive-fast preemption slice (GL3.B4)
  becomes a declared 1.4 rung arm, or waits for one (the owner names the rung later). The timing defects
  L1, L2, L5 and the dead preemption scaffolding L7 (§3.4, §3.5) are **filed as defect issues** (#1176 L1, #1177 L2, #1178 L5, #1179 L7);
  each is fixable without the registry. (#1179 is closed: the `Dormant since 2026-10-07` docstring is on
  `runtime/preemption.py`. #1176–#1178 are open, 2026-10-24, wiring N4.)

**Owns (proposed):** a new `src/maxim/perception/` package (the name `archive/thalamus_relay_design_pass.md`
design-pass decision 1, DP1, reserved): `ReceptorSpec`, `AfferentTrackSpec`, `AfferentEvent`,
`ReceptorRegistry`, `TrackScheduler`. It also owns a per-node provenance dict on
`similarity/ec.py::EntorhinalCortex` (GL3.B1). **Uses, does not own:** `PhysicalEventId` and the per-agent
`EventSequencer` (`maxim/embodiment/event_id.py`, built at the autonomic plan's post-fence resume stage,
G17; the sequencer is held by the agent's primary `Embodiment`). Seq authority moves to this plan's scheduler at the GL3.B3 stage gate (§5.5).
**Companion plans:** [autonomic_layer.md](autonomic_layer.md) (GL2: `InteroceptiveOutcome`,
`NociceptorSpec`; it supplies the first dual-track receptor); [latent_forward_model.md](latent_forward_model.md)
(GL4: it consumes `AfferentEvent`s joined on `pid`, never on the executor's `uuid4` invocation id,
which is not persisted-stable; the tool path stamps the pid on `ToolOutput` alongside the invocation id).
The state page is
[docs/wiring/body-and-word-worlds.md](../wiring/body-and-word-worlds.md).
**Prior art, cited as decided:** [archive/thalamus_hypothalamus_framing.md](archive/thalamus_hypothalamus_framing.md)
(the organizing frame), [archive/thalamus_relay_design_pass.md](archive/thalamus_relay_design_pass.md)
(SUBSUME), [archive/percept_testbed_audit.md](archive/percept_testbed_audit.md) ("do not build a
percept-channel manifest as conceived", answered in §4.1).
**Inputs, not plans of record:** [deferred/cross_modal_perception_fabric.md](deferred/cross_modal_perception_fabric.md),
[deferred/perception_pipeline_placement.md](deferred/perception_pipeline_placement.md),
[deferred/modality_resolution_and_alignment.md](deferred/modality_resolution_and_alignment.md),
[deferred/mesh_perception_transport.md](deferred/mesh_perception_transport.md),
[deferred/reflex_layering.md](deferred/reflex_layering.md), [deferred/nociception_layer.md](deferred/nociception_layer.md).

Evidence: everything below was read at `origin/main f1c833ab` and is cited as `file::symbol` (line
numbers drift). Facts are from code reads, not runtime probes, unless stated. **UNVERIFIED** marks an
inference that the drafts could not confirm.

### The T8 (b) four-lens design review (2026-10-24)

The four lenses read this plan, the built GL3.B0 tests and the code at `origin/main d9dffe6e`:
[confounding](reviews/grounding_gl3/confounding.md) (4 DO-NOT-BUILD / 10 SHOULD-FIX / 5 NIT),
[bio-faithful](reviews/grounding_gl3/bio-faithful.md) (2 / 8 / 6),
[wiring](reviews/grounding_gl3/wiring.md) (3 / 12 / 6) and
[environment](reviews/grounding_gl3/environment.md) (3 / 9 / 6). Three lenses re-ran the GL3.B0 latency
instrument at `d9dffe6e`: both arms still equal the frozen record. The GL4 wiring review's S8
([reviews/grounding_gl4/wiring.md](reviews/grounding_gl4/wiring.md)) is folded here on GL3's side. The
owner took the strict option on every DO-NOT-BUILD (2026-10-24, TR18–TR24, DECISIONS.md G24); each is folded
where it lives:

| DO-NOT-BUILD | Resolved by | Where |
|---|---|---|
| Bio D1, confounding D4, environment S1 (GL3.B3): track routing moves the `drive:oxygen` publish off the W-4 order and writes health fear twice; §5.4 and §5.8 disagree on `affective_slow`'s PainBus destination | TR18 / T8b-1 | §5.4 (`affective_slow` row, fan-out example), §5.5 H3, §5.8 PainBus row, §5.10, §6 GL3.B3 build and gates 2, 3, 7 |
| Bio D2, confounding D1, environment D1, wiring S10 (GL3.B4): nothing transduces a breach before H2, so the preemption check has no event; a latency gain would belong to the transduction or the reselect | TR19 / T8b-2 | §3.4 L1, §4.2, §5.6, §6 GL3.B4 entry and gates |
| Confounding D3, wiring D3 (GL3.B1): the imagined / narrated eligibility change is unflagged, has no narrated weight, and no golden runs its path | TR20 / T8b-3 | header, §5.7 rule and items 3–4, §5.9, §6 GL3.B0 note, §6 GL3.B1 gates 3–5 |
| Wiring D1 (GL3.B1): a registry on `LoopRun` cannot reach the water-trial trainers or red gate (g), and reopens the slice-3 pin | TR21 / T8b-4 | §5.3, §5.8 row 1, §6 GL3.B1 build, §8 |
| Wiring D2, confounding D2 (GL3.B1): provenance misses the writers that credit on the body path; `apparatus` has no producer; deletion probe 5 contradicts gate 1 | TR22 / T8b-5 | §5.7 items 1, 2, 4 (the credit-writer table), §5.10, §6 GL3.B1 build and gates, §8, §10 |
| Environment D2, wiring D2 (GL3.B3): loop-thread evaluations transduce the narrator's write first and mint it `experienced` | TR23 / T8b-6 | §3.2, §5.5, §5.7 item 1, §6 GL3.B3 build and gate 6 |
| Environment D3 (GL3.B3): the thermal burn has no real-loop harness and no hook on the tool path | TR24 / T8b-7 | §5.5 (hook H3t, the per-caller drain table), §6 GL3.B3 harness and gates |

GL4 wiring S8 (one contract, two producers): GL3.B2 now states the contract once, with the tool-path pid
per invocation; GL4 holds the producer (§6 GL3.B2, §8).

**SHOULD-FIX dispositions.** Folded, except where a reason is given:
- **Confounding** S3 (livelock on a deepening breach), S4 (once per pid; probe 3 re-worded; TR18), S5
  (drift keeps its kind; the mixed-node credit rule; a mixed-node gate and the mixed fraction), S7 (gate
  1's trajectory and the order-permutation probe), S8 (gate 6's positive half), S9 (GL3.B5's scripted
  gate), S10 (TR19): folded. S6 (red gate (b) flips with work GL3.B1 does not do): an owner decision
  at GL3.B1 start, TR25 (§11), because moving a red gate's flipping stage is a re-point the GL3.B0 banner
  reserves to the owner. S1 (positive controls for (a), (d), (g); (e) does not pin TR2's action half)
  and S2's GL3.B0 half ((d)'s world never picks a protective action): **follow-up issue (filed by the main
  session)**, because they change the built GL3.B0 tests; S2's GL3.B4 half (one scenario with a
  protective pick, a v2 record) is folded in §6 GL3.B4.
- **Bio-faithful** S1 (the `reflex` track relabelled; keyword reflexes are a narrated nociceptive
  receptor), S2 (`affective_slow` is a delayed phasic copy; coalesced pids; clamp), S3 (re-tiered as
  engineering priors over an oracle label), S4 (strict rule kept, its reach measured), S5 (the per-sensor
  Minecraft mapping), S6 (`threat_low_road` renamed to contextual fear), S7 (`receptor_class` per
  sensor / `composite`; `efference` and `linguistic` removed), S8 (refractory coalesces, never drops):
  folded. S4's per-sensor-moved option (b) stays the owner's pick at GL3.B1 start (§5.7 item 1).
- **Wiring** S1 (pull reads carry no pid at GL3.B1; §8's "GL2a (the pid)" fixed), S2 (a synchronous
  `encode_event` route), S3 (the `provenance=` callers; the per-percept kind table owed with TR11), S4
  (`hivemind/entry_index.py` and the accessor), S5 (`tick` returns `(channel_values, clusters)`), S7 (the
  touch list and how the narrator tools reach the inbox; posting the whole consequence is superseded by
  TR23's write-time stamp), S8 (latency-0 drained in place at every caller), S9 (`pass_index` = the
  non-idle pass count), S11 (Rule T extractions), S12 (three flags): folded. S6: (b)'s half is TR25; the
  (g2) / (g3) red gates are folded as owed before GL3.B1 and go to the same follow-up issue (they are new
  tests). S10: TR19.
- **Environment** S2 (one pass counter, a delivery log, the per-world table), S3 (a deepening damage
  stream, the hazard's time course), S4 (the record is a harness fact; the drift check), S5 (the narrator
  tool-output contract), S6 (TR22's apparatus window), S8 (the `caused_by` target and its strata), S9
  (held-action and DoA hardware facts): folded. S7 (the fabric trigger fires before its join key exists):
  **not folded**, an owner decision (§11 TR27), because §9's wording is "the one used everywhere" and changes in
  every banner at once (strict option (a): add "AND the join the binding needs exists"); its "five live
  sources" correction is folded in §9.
- **NITs** folded, except: confounding N2 and N3 (red gate (c) with two readings; (g)'s docstring), which
  edit built GL3.B0 tests and go to the follow-up issue; wiring N2 (`perception/`'s ARCHITECTURE.md
  layer entry), which is ARCHITECTURE.md's edit, made with TR4 at GL3.B1 (named in §11 TR4); wiring N6,
  moot under TR18 (`affective_slow` no longer writes cluster fear).
- **For other files** (the main session's): grounding.md's GL3 row restates the deletion-probe /
  byte-identity contradiction (TR22) and says the per-sensor Minecraft mapping lives in GL3.B0's census
  (bio S5: it lives in §5.4 here and is enforced by GL3.B3's membership test); wiring's proposed
  `docs/wiring/provenance-lives-at-the-write.md`; M10's row in `outstanding.md` (wiring S12: three flags,
  asserted by the T1 harnesses too).

### Pre-merge review round (2026-10-24)

The three-lens pre-merge round (Executor, Architecture, Wire integrity; `docs/CODE_REVIEW.md`) read the
T8 (b) fold as a staged diff against `d9dffe6e`. Findings on **this plan** (the rest concern
latent_forward_model.md, grounding.md, the roadmaps or `docs/wiring/`): Executor 2 BLOCKER / 7
SHOULD-FIX / 3 NIT; Architecture 0 / 4 / 2; Wire integrity 3 / 4 / 2. The three BLOCKER pairs were
resolved by the main session and are folded as stated; every other finding is folded where listed.

| Finding | Resolution | Where folded |
|---|---|---|
| Executor B1 = Wire B2: "(g) flips as written" and "reached unchanged" were false (no bind site in exp53 or (g); the typed error would FAIL a strict `raises=AssertionError` gate) | One binder, `perception/`'s `registry_for(embodiment)`, called from `runtime/`, deriving from the body YAML and caching on first use; typed error only with no body YAML (restated by the delta round: derived from the root entity tree's declared sensors, built once under an `RLock`, typed error when the tree declares no sensors); a known-answer test on (g)'s surface | §5.3, §5.8 row 1, §6 GL3.B1 build and gate 3, §8, TR21 |
| Executor B2 = Architecture S7: is #1176's per-pass transduction flagged? | Unflagged, in #1176's own PR, with its own walk and T1-13/14/15 re-runs; the record re-frozen as v2; gate 1 reads v2; (ii) gated, (i) reported; the negative control is the v1 record, cited (restated by the delta round: produced on the post-#1176 tree by a test-only monkeypatch and gated; ratified as unflagged by the owner, TR28 / G28) | header, §3.4 L1, §5.6, §6 GL3.B0 note, §6 GL3.B4 prerequisite / record / gates 1, 3, 5, 6, §7, TR19 |
| Wire B1: the last-writer slot on the `Embodiment` cannot be reached by writers that hold an `Entity`, and the actor tool holds `scene_emb` | The kind map beside `Entity.vital_metrics`; the writer kind a `ModulatorAffordanceTool` constructor field; constructors listed | §3.2, §5.7 item 1, §5.10, §6 GL3.B1 build, §8, TR23 |
| Wire B3: two pid contracts; the H3t emit preceded its `caused_by` target | One pid: minted at `Executor._run_started` before `tool.run`; the H3t emit reuses it; `_stamp_invocation` mints nothing | §5.2, §5.5 H3t, §5.10, §6 GL3.B2 contract and gate, §6 GL3.B3 build and gate 2; autonomic_layer.md §3.1.1, §3.1.2 |
| Executor S1: the slot's initial kind | `unknown`, written by the body builder, credited as `legacy`; (g′) restated. **Superseded by the delta round under G29 (TR29):** no `unknown` kind; a missing slot reads `legacy` | §5.2, §5.7 item 1, §6 GL3.B0 note |
| Executor S2: M39's allowlist seeded from `scripts/` | Seeded from `src/` without the moved sites; the counter on both entry points | §5.10, §6 GL3.B1 gate 5 |
| Executor S3: the refractory could defer a PainBus publish; "publishes and also delivers" read as two | `pain_bus` exempt (same-pass coalesce only); one publish; a count-test case | §5.4, §5.5, §5.10, §6 GL3.B3 build and gate 8 |
| Executor S4: #1450 does not carry N2, N3, (g2), (g3), (h), and its deadline is "before GL3.B4's gate design" | Filed as #1450, with its comment carrying N2, N3, (g2), (g3) and (h), owed before GL3.B1 (delta round, Executor N-1) | #1450 and its comment |
| Executor S5 = Wire SF-4: the `encode_sensors` caller list | Keyword-only, no default; every caller by directory (src 2 sites, scripts 16 files, tests 18 files) edited; the census category extended to `scripts/` | §5.7 item 3, §6 GL3.B1 blast radius |
| Executor S6: no llm-primary harness for GL3.B3 gates 2 and 6 | To be built in GL3.B3: a mocked-LLM llm-primary driver plus a scripted orchestrator thread; the gates are owed until it exists | §6 GL3.B3 |
| Executor S7 = Wire SF-7: the `WaterTrial` window and T1-9 | The window in the blast radius with T1-13/14/15/R3/B1 walked and two byte-identity checks; T1-9 walked; the mother's credit recorded, its fate at the flip the T1-9 walk's | §5.7 items 1 and 4, §6 GL3.B1 blast radius, §8, TR22 |
| Architecture S5: G16's "never credits" vs TR20's interim | Qualified where this plan restates G16 (holds once the credit flag flips); the tier row is grounding.md's | header G6 note (grounding.md: main session) |
| Architecture S8: `affective_slow` has no acting consumer | Record-only, stated; T4a: Dormant at 1.4.0 if nothing acts on it | §5.4 row |
| Architecture S10 = Executor N5: Rule T misses two pinned functions | `record_outcome` (399) and `ModulatorAffordanceTool.execute` (502) added with their extractions | §6 Rule T |
| Architecture N3: environment S7 had no §11 row | TR27 | §9, §11, this section |
| Architecture N4: Session V coordination | Named in GL3.B1's and GL3.B3's touches | §6 GL3.B3, §8 |
| Wire SF-5: the writer census; `spec.py`; derived `health` | Completed (Reachy motor backend, `backends/minecraft.py`, `cradle_mother`, `DoAFeed`, `loop_perception`, `exp53`, `exp56`); `spec.py` in §8; derived values inherit their source's kind | §5.7 item 1, §8 |
| Wire SF-6: EC lifecycle sites; counts on the `min_delta` short-circuit | `ingest_substrate_nodes`, `remove_substrate_node`, `_reset_store_state`; the short-circuit adds the kind, bumps no count | §5.7 item 2 |
| Wire N3: "provenance" collides with hivemind's vocabulary | Key kept (it is red gate (g)'s built read surface; a rename would re-point a strict gate); its namespace stated | §5.7 item 2 |
| Wire N7: two scripts missing from the `evaluate_failures` caller table | Added | §5.5 |
| Executor N4: "G3-n" labels collide with owner decision G3 | Renamed `T8b-1` … `T8b-7` | the table above, §11 |
| Executor N6: pre-stage goldens regenerated at review | Committed from the pre-stage tree first | §6 GL3.B1 gate 1 |

**Delta round (2026-10-24).** The same three lenses read the fold's unstaged delta only (stop rule):
Executor 0 BLOCKER / 6 SHOULD-FIX / 6 NIT, Architecture 0 / 5 / 5, Wire integrity 0 / 4 / 5, all folded
(the ones on other files in those files). Owner decisions G28 (#1176 unflagged, TR28), G29 (no `unknown`
kind; a missing slot reads `legacy`, TR29) and G30 (the matched-class floors bind T9 now, margins at S5
start; latent_forward_model.md) settle the three decision-shaped findings. On this plan: `registry_for`
derives from the root entity tree's declared sensors, built once under an `RLock`, typed error when the
tree declares none (§5.3, §6 GL3.B1, TR21); GL3.B4 gate 6's negative control is produced by a test-only
monkeypatch on the post-#1176 tree (§6 GL3.B4); the raw `vital_metrics[...] =` writes go through one
kind-taking setter with an AST lint (M78), `loop_perception`'s kind comes from the percept, and the
kind-map reader resolves through `_resolve_sensor_slot` and persists through `Entity.to_dict` /
`from_dict` (§5.7 item 1, §5.10); the pid and seq are drawn at `_run_started` under the sequencer's lock,
with `ConcurrentExecutor` outside the determinism claim (§5.2, §5.5); S5's resets are stamped
`apparatus` through the setter and window (§5.7 item 1); the constructor census, the imagined-entity
kind and the stale rows (Executor S4, the GL3.B4 "Waits for") are fixed. **No further round (owner, stop
rule; converged).**

## Why this plan exists

The relay the project actually has is not where the 2026-07 documents put it. For the body world the
de facto thalamus is `runtime/substrate_proposal.py::_SUBSTRATE_CHANNELS`, a tuple of `ModalityChannel`s.
The earned survival and orient rows ride on it, and no EARNED row depends on the word (EC text) path.
Around it there are six direct encoder call sites, a percept FIFO that moves one item per loop pass, and
pain ingress through several producers (§3.3) whose timing depends on whichever caller happens to run
`evaluate_failures`, on which thread. No percept carries **who produced it**, **what kind of truth it is**
(experienced, imagined or narrated), or **which physical event it belongs to**. As a result, an imagined
affordance can receive the same NAc credit as an experienced one. A burn cannot be told apart from
"the same burn, felt again later". A breach that begins after the substrate has selected an action cannot
stop that action. This plan adds the three missing identities at one registration point, and adds timing
as logical tracks, both at the smallest size that carries the identity.

---

## 1. What the 2026-07 thalamus framing decided, and what was built

| Design-pass decision DP1–DP4, plus framing (source) | Built? | Evidence |
|---|---|---|
| Frame: thalamus = exteroceptive relay, hypothalamus = drive integration; the two-organ split is correct anatomy, not a bug (`archive/thalamus_hypothalamus_framing.md`) | Framing only | n/a |
| **DP1** SUBSUME the fragmented relays (`ThalamicGate`, `BioEnrichmentPipeline`, the exec-agent call site); do not grow `ThalamicGate`. The coordinator does not exist until a second need forces it. The `perception/` name is reserved for it (`archive/thalamus_relay_design_pass.md`) | Direction adopted. **No coordinator and no `perception/` package exist** | `src/maxim/` has no `perception/` |
| **DP2** Un-flatten the sim percept through a typed side-channel, not a dict key | **BUILT** (PR #402) | `runtime/sim_adapter.py::SimulationAdapter.current_percept`, `NullSimulationAdapter.carry_percept`; guard `tests/unit/test_sim_adapter_unflatten.py` |
| First slice: `CompositePerceptSource`, an ordered first-non-None multiplexer that raises on a second injection implementer | **BUILT**, live only behind `MAXIM_SIM_AUDIO_ORIENT` | `simulation/composite_source.py::CompositePerceptSource`, `simulation/audio_orient_wiring.py`; guard `tests/unit/test_composite_source.py` |
| **DP3** `enabled` = one per-channel gate at the routing fork; `gain` is per-route and deferred | **NOT BUILT**: no channel gate exists | the only per-channel knobs are env (`MAXIM_SIM_AUDIO_SALIENCE` / `_NOVELTY`) |
| **DP4** Audio azimuth gets its own `"audio"` EC route, de-bundled from interoception | **BUILT** as the extero/intero seam (PR #411) | `embodiment/sensory_streams.py::ModalityChannel`, `::DECLARABLE_MODALITY_TAGS`; `substrate_proposal.py::_SUBSTRATE_CHANNELS` |
| Testbed audit M1 (flag consolidation), M2 (per-run active-config record), M3 (per-channel telemetry) | M1 and M3 not built as a channel surface. M2 **UNVERIFIED** (not traced) | the ablation flags are still per-env |
| Fabric (`deferred/cross_modal_perception_fabric.md`) | Zero code. DEFERRED 2026-09-19 on "a second robot body physically exists" | file header |
| Mesh percept transport | Wire prep only. `MeshMessageType.PERCEPT_PUSH` has **no producer** | `mesh/message.py`; `simulation/sources.py` docstring (D46) |

**What this means.** After PR #411 and 1.1.4 PR 2, the body-world relay has the right shape: one
`ModalityChannel(tag, read_values, read_ranges)` per channel, membership declared per sensor in body
YAML (`modality: world|audio`), one encode per channel and one EC cluster space per channel, with late
convergence in `NAc.recommend_action`. The "fragments" the July documents named (`ThalamicGate`,
`BioEnrichmentPipeline`) are both LLM-facing, and neither is on that path. **This plan therefore
subsumes from the body path outward.** `_SUBSTRATE_CHANNELS` becomes the core of the registry (GL3.B1). The
LLM-facing fragments are wrapped last, and only where a consumer needs them.

---

## 2. Receptor census: every live percept source

"EC route" means the substrate path. "LLM route" means the prompt or the percept bus.

| Receptor-to-be | Producer (file::symbol) | How it enters | EC modality / space | `agent_id` | Provenance | Clock |
|---|---|---|---|---|---|---|
| Body drives / interoception | `substrate_proposal.py::_read_drive_states` (plus the derived `cold` need) | `SensorEncoder` channel | `"interoception"`, 384-d hashed sum, ungained, frozen centroid, 0.85 | passed by the caller to `encode_sensors(agent_id=)`, not carried by the reading | none (implicitly experienced) | none on the reading; per tick |
| Minecraft world state | `simulation/minecraft.py::MinecraftClient.latest_state` → `embodiment/backends/minecraft.py::MinecraftWorldBackend.sync_world_sensors` → `world_set_axis(owner="minecraft_bridge")` → `vital_metrics` | `SensorEncoder` channel (`_read_world_states`, declared `modality: world`) | `"world"`, 384-d, **gained** (A4 p3.0), frozen | caller-passed | none | `state_age_s` exists on the client but is not carried |
| Minecraft health / food / oxygen | the same sensors **also** declare `drive:` (`_data/components/bodies/minecraft_player.yaml`) | **the same reading enters twice**: the world channel and the interoception channel | world + interoception | caller | none | none |
| Minecraft game events (chat / damage / death / block) | `simulation/minecraft.py::MinecraftPerceptSource.next_percept` → `make_text_percept("[minecraft:damage] …", channel="narrative")` | text percept; `SensoryTag(NARRATIVE)` by default | `"text"` (768-d mpnet, 0.44), **only with `MAXIM_SUBSTRATE_PATH=1`** | **None** (the factory is called without `agent_id`; `MemoryHub` falls back to the hub's agent) | none. **A damage event is tagged NARRATIVE, not nociceptive** | `Percept.timestamp` (wall) |
| Reachy DoA (live) | `embodiment/audio_localization.py::DoAFeed` | **two lanes**: the sensor lane via `world_set_azimuth` → audio channel, and the percept lane via `make_audio_percept` → `adapter.carry_percept` → the §1.16 prompt fold (`loop_perception.py::orient_to_audio`) | `"audio"` (ungained, frozen; optional place code `MAXIM_PLACE_CODE_EXTEROCEPTION`) | set on the percept | none | `(azimuth, timestamp)` cached; head/body yaw stamped at capture. **The two lanes share no event id** |
| Sim DoA | `AzimuthDoASource` inside a `CompositePerceptSource` | same as live, through `SimulationAdapter.current_percept` | `"audio"` | per the factory call | none | wall |
| Narrator / DM / CLI text | `simulation/conversational_source.py::ConversationalSource.inject_cli` → `make_text_percept(NARRATIVE/cli)` | LLM route; EC `"text"` only with `MAXIM_SUBSTRATE_PATH` | `"text"` | often None | none | wall |
| Sim pain injection | `ConversationalSource.inject_pain` → `make_intero_percept` | a text percept with an INTEROCEPTION tag, published as a `Reaction` straight onto `reaction_bus` | `agents/modality.py::_SUBSTRATE_MAP[INTEROCEPTION] = "text"`, so it **embeds as language** | caller | none | wall |
| Narrator consequence writes (`--sim`) | `simulation/tools.py::SetEntitySensorTool`, `DamageComponentTool`, `OrchestratorActorTool` (registered on `orch_registry` against `_aut_embodiment`), and the reflex dispatch, which builds separate instances of the same classes (`BioEnrichmentPipeline`'s `_reflex_damage_tool` / `_reflex_sensor_tool`) | writes the AUT's body sensors, then calls `_aut_embodiment.evaluate_failures()`: the narrator's tools **on the orchestrator thread (the `start_simulation_mode` caller running the orchestrator agent's loop)**, the reflex dispatch inside the AUT's `enrich` on the loop thread (`sim.aut`); `DamageComponentTool` also publishes a `PainSignal` directly | reaches EC later, through the body channels' pull read of the written sensors | the AUT's | **`narrated` (owner decision G6)**; today none | wall |
| Vision (DN / robot) | `agents/perception_agent.py` raw `Percept(modality="vision")` | DN `ThalamicGate`, then the LLM. On the EC route (`memory_agent` → `MemoryHub.on_percept_received`), `LinguisticEncoder.encode` embeds the percept's **text content** under modality `"vision"` (code-read), **only with `MAXIM_SUBSTRATE_PATH=1`** | a `"vision"` tag over a **text** embedding | via `_build_context()` | none | wall |
| Imagined entity affordances | `imagination/trigger.py::ImaginationTrigger._encode_entity_affordances` → `encode_decomposed(aff_name, "text", agent_id)` | EC `"text"` + ATL + **`NAc.update_eligibility`** | `"text"` | yes | **None on the node.** `register_substrate_node(source="local")`. `imagined` lives only on `ComponentRegistry` (`provenance="imagined"`), `ImaginationResult`, `Episode.imagined`, and on CausalLinks tagged **retroactively at session end** (orchestrator, "Tag + decay causal links from imagined entities") | `TemporalSignature.now()` (wall) |
| Own-body affordance names | `simulation/orchestrator.py` → `imagination/trigger.py::encode_entity_affordances` | the same encode path as the imagined case | `"text"` | yes | none; **indistinguishable from imagined** | wall |
| Mesh peer percepts | none (`PERCEPT_PUSH` has no producer) | n/a | n/a | n/a | n/a | n/a |

**GL3.B0 census (2026-10-09; corrected after its review).** The checked-in table is
`tests/unit/test_receptor_census.py::CENSUS`: every site with its liveness and the threads that run it;
the scan fails on any new or vanished site (it checks sites; the liveness and thread columns are
code-read, and a new caller one hop up is invisible to it). It adds producers the table above missed or
misdescribed:
- percepts: the messaging channels (`comms/`, on a webhook thread); `AgentPool.run_turn` (a pool worker
  in a concurrent round); the scenario fixture source (`simulation/scenario_source.py`, a raw `Percept`);
  the Dormant `EmbodimentPerceptSource`; and above all `agents/perception_agent.py::PerceptionAgent.process_observation`,
  **the per-pass percept of every `MaximAgent` loop** (CLI / transcript text, vision only when detections
  exist), published synchronously to the memory agent, on the loop thread (the "Vision" row understates it);
- `ConversationalSource.inject_cli` runs on six threads: the orchestrator, `sim.dm` (through
  `send_and_wait`), `sim.stdin` (human free text straight into the AUT's percept source), `sim.stall`
  (nudges, into the orchestrator's own source), a console request, and `main` (the cradle mother, and the
  generative, fixture, pre-campaign and non-interactive DM sends);
- `inject_pain`'s direct-Reaction branch needs `pain_bus=`, which no caller passes: the live path is the
  INTEROCEPTION percept, turned into a Reaction on the loop by `SimulationAdapter.next_observation` (so
  the row above, "published as a Reaction straight onto reaction_bus", is wrong); `inject_sensor` has no caller;
- body writers outside the narrator row: the **cradle reactive mother** (`simulation/cradle_mother.py::reactive_mother_tick`)
  writes the AUT's hunger and azimuth from `main` while `sim.aut` runs, and the DM cascade
  (`dm_runtime.py::CascadeResolver.resolve`) writes campaign entities on `sim.dm` (on `main` when the
  campaign runs non-interactively, after `sim.aut` starts), the AUT's body when a role resolves to it; the
  narrator's `DamageComponentTool` also writes the body through `apply_damage`;
- the imagined-affordance encode is also reached from the fixture runner's manifest on `main` while the
  AUT loop runs; `ComponentIndex._embed` is its own sentence model, not the EC.

**Gaps the census exposes** (all code-read):

1. **Identity.** No reading says which receptor produced it. The DoA feed is already a one-receptor,
   two-lane source built by hand, and its lanes share only a wall timestamp.
2. **`agent_id`.** `PerceptContext.agent_id` is documented as "until then, producers leave it None".
   Minecraft events and narrator percepts do leave it None. Sensor channels get `agent_id` from the
   caller, not from the reading. F0.5 is unfinished.
3. **Provenance.** No EC, ATL or NAc write carries a provenance kind. The only per-node provenance is
   `EntorhinalCortex._substrate_node_sources` ("local" or a hivemind contributor id, which records origin,
   not experienced vs imagined) and `_encoder_provenance` (encoder realized-state per recorder). An
   imagined affordance and the agent's own affordance land on the **same node with eligibility
   updated**, so credit can reach it. Confirmed at `similarity/encoder.py::LinguisticEncoder.encode_decomposed`
   (calls `self._nac.update_eligibility`) and `similarity/ec.py::EntorhinalCortex.register_substrate_node`
   (default `source="local"`).
4. **Misfiled pain.** A Minecraft `[minecraft:damage]` event is NARRATIVE text. A sim pain injection
   embeds as language. Neither enters as nociception. Body-sensor nociception reaches the body through the
   health sensor and `evaluate_failures`, but several producers publish pain without it (the ingress
   census in §3.3).
5. **Double entry.** Minecraft health, food and oxygen sit in both the world and the interoception
   channel. Under this plan that is legal (two receptors reading one sensor), but it must be named, and
   GL3.B1 must not de-duplicate it: doing so moves the geometry and fires Exp 60–62.
6. **Three clocks.** Percepts carry wall `timestamp`. Eligibility anchors use `TemporalSignature.now()`
   (wall) at encode. Hippocampus captures use the agent's experience clock (`experience_us`,
   `capture_seq`; `runtime/experience_time.py`). On turn-based worlds the eligibility anchor and the
   capture time are in different units. **UNVERIFIED** whether this mis-credits anything today.
7. **Dead gain field.** `agents/modality.py::SensoryTag.perceived_intensity` / `modulated_by` are
   documented as "set by SensoryGate". No `SensoryGate` class exists. The only assignment is
   `SensoryTag.from_dict` deserialization. No producer sets it.
8. **Vision as text.** A `"vision"` EC tag is applied over a sentence embedding of the percept text
   (code-read). Leave it alone until a vision receptor exists, and name it on the wiring page.
9. **Three modality vocabularies.** `SensoryModality` (7 values), `Percept.modality`
   (`Literal["text","vision","audio","intero"]`) and the open-string EC tags
   (`interoception/audio/world/text/vision`) are joined by the lossy `agents/modality.py::_SUBSTRATE_MAP`
   (SOUND / TOUCH / SMELL / INTEROCEPTION → `"text"`). The July testbed audit flagged this, and nothing
   changed.
10. **Embedding-space identity already exists, and must not be duplicated.**
    `similarity/encoder.py::encoding_geometry_tag` and `sensor_geometry_fields`, plus `geometry=` on
    `pattern_complete_or_separate`, already name the encoding space (encoder, modality, declared sensor
    set, normalization, dim, gain, ranges). A Receptor's embedding space is **derived** from these, never
    authored.

---

## 3. Timing findings: where a fast signal waits today

### 3.1 The loop pass and its clocks

`runtime/agent_loop.py::run_agentic_loop` runs on one thread. A "pass" is one iteration of its loop:

| § | What runs | Where |
|---|---|---|
| 0–0.6 | `pre_tick_gate`: stop/pause checks, the **live tick** (`_loop_live_tick` → `tick_embodiment_drift` [llm-primary only] + `ExperienceClockDriver.on_live_pass`), then the idle gate | `runtime/loop_gates.py::pre_tick_gate`, `::_loop_live_tick` |
| 1 | `sim.next_observation`: **one** percept per pass | `runtime/sim_adapter.py::SimulationAdapter.next_observation` |
| 1.1 / 1.15 / 1.16 | imagination, auto-sense, audio orientation (with the sim-only orienting reflex) | `loop_perception.py::perceive` (1.3.2 slice 5; `imagine`, `auto_sense`, `orient_to_audio` → `PerceptionOutcome`) |
| 1.2 | ThoughtGate, then `BioEnrichmentPipeline.enrich` (which runs the **percept reflexes**) | inline; `integration/bio_enrichment.py::_evaluate_reflexes` |
| 2 / 3 | poll the LLM worker (non-blocking); `agent.propose_intent` fallback | inline |
| 4 | **execute `ctrl.pending_proposal`** | inline → `runtime/tool_dispatch.py` |
| 6b | substrate tick (substrate-primary): `propose_via_substrate` sets `ctrl.pending_proposal` | `runtime/loop_substrate.py::substrate_tick` |
| 6 | LLM submit (llm-primary) | inline |
| 8.5 | NAc per-pass decay | `agent_loop.py::_loop_bio_tick_maintenance` |
| 9 | `time.sleep(target_period - elapsed)` | inline |

Four clocks are in use. **Wall time** drives the loop period, the substrate cadence
(`loop_state.py::_substrate_tick_due`, `loop_controller.py::LoopController.llm_submit_interval = 0.5`),
drive drift (`embodiment/body.py::evaluate_failures`, `drift_dt = now - _last_poll`), and the PainBus
(0.5 s per `(entity, failure_mode)`) and ReactionBus refractories. **Experience time**
(`memory/experience_clock.py::ExperienceClock`, integer µs, advanced once per live pass by
`runtime/experience_time.py::ExperienceClockDriver`) is read only by memory strength today. **Pass
count** is the clock of NAc decay (§8.5), which is why `_submitted_recently` waking the loop was left
alone (changing it changes Exp 56/57 measurements; `_substrate_tick_due` docstring). **Sequence numbers**:
`Hippocampus` gives each capture a per-agent `capture_seq`, resumed past the saved maximum on load
(`memory/hippocampus.py::Hippocampus._resume_capture_seq`, called from the load path in
`memory/hippocampus_persistence.py`). `time/temporal_event.py::TemporalEvent` only carries an `event_id`;
its producers mint it as `uuid4().hex` (`bridges/tool_pain_bridge.py::ToolPainBridge`,
`embodiment/body.py::_emit_drive_temporal_event`) with wall `TemporalSignature.now()`, which is not
deterministic.

**Determinism infrastructure.** `tests/unit/_loop_harness.py::_StepClock` patches the global `time`
module; only the loop thread advances it. `_ScriptedWorld` is an in-process Minecraft client that is a
pure function of the step clock. `test_agent_loop_selection_golden.py::test_golden_is_identical_across_processes_and_hash_seeds`
pins the selection trace across processes and `PYTHONHASHSEED`. **Every scheduler gate in this plan runs
through this harness.**

### 3.2 Where real concurrency exists

| Thread | Owner | Touches body or bus? |
|---|---|---|
| loop (`sim.aut` under `--sim`) | `run_agentic_loop` | yes; calls `evaluate_failures` from `propose_via_substrate` (substrate-primary) and `tick_embodiment_drift` (llm-primary), and, under `--sim`, through the reflex dispatch's `DamageComponentTool` / `SetEntitySensorTool` instances inside the AUT's `enrich`. **Not the only caller** (next row) |
| the orchestrator thread (`--sim` only: the `start_simulation_mode` caller running the orchestrator agent's loop) | `simulation/orchestrator.py` registers `OrchestratorActorTool` / `DamageComponentTool` / `SetEntitySensorTool` on `orch_registry` against `_aut_embodiment`, run inside `run_agentic_loop(orch_agent, …)` | **yes**: the narrator's tools write the AUT's sensors and call `_aut_embodiment.evaluate_failures()` on this thread while the AUT loop runs on `sim.aut`; `DamageComponentTool` publishes a `PainSignal` directly |
| `sim.dm` (interactive DM campaigns; a non-interactive campaign runs the same code on `main`) | `simulation/orchestrator.py` → `campaign_runner.run_dm_campaign` | never touches the narrator tools above, but it is **not** body-silent: its sends reach `PerceivedPainAssessor.assess_text` (pain into the AUT's buses), `inject_cli` writes the AUT's percept source, and the DM cascade can write the AUT's sensors (GL3.B0 census, §2 and §3.3) |
| `mc-sync-<agent>` (0.5 s default; 0.25 s in `exp58_run.py` / `b1_floor_run.py`, 0.05 s in `dark_danger_probe.py` / `exp58_offline_gates.py`; the bridge's own state interval is 500 ms by default, 100 ms in the Exp 60–62 / R3 harnesses; environment N2) | `simulation/minecraft_harness.py::MinecraftSyncPump._run` | writes `vital_metrics` via `world_set_axis`; publishes no pain and **never calls `evaluate_failures`** |
| `minecraft-bridge-reader` | `simulation/minecraft.py` | fills the snapshot and event queue; `call_action` **blocks** the loop thread until `action_result` |
| `doa-feed` | `embodied_runtime/agentic_runtime.py` → `DoAFeed.run` | `world_set` of azimuth plus `carry_percept` (one slot, latest wins) |
| DefaultNetwork `_run_loop` (30 Hz) | `default_network/network.py::DefaultNetwork` | behaviours, `PriorityArbiter`, ThalamicGate. Its thread also starts under `--sim` (the AUT's DN is built enabled, `loop_setup.py` starts it); with no robot it appears to publish nothing on these buses (**UNVERIFIED**, GL3.B0 review) |
| WorkerPool lanes | `runtime/worker_pool.py` | LLM jobs; `cancel_pending` drains queued jobs only |
| Hippocampus capture worker; `sim.stdin`, `sim.stall` | `memory/hippocampus.py`; `simulation/orchestrator.py` | capture; human edge, watchdog |

The pump writes `vital_metrics` while the loop reads channel values. Nothing guarantees that one encode
sees one snapshot (per-key writes are GIL-atomic; there is no snapshot lock). **UNVERIFIED** whether this
has ever mixed two snapshots in one encode. Tracks fix ordering, not tearing.

**The orchestrator thread is a declared edge.** Two threads run `evaluate_failures` on the AUT's body
under `--sim` (`sim.aut` and the orchestrator thread (the `start_simulation_mode` caller running the orchestrator agent's loop)), so the shared mutable state it touches is written from both:
`Entity.drive_breach_severity` (the breach latch dict, `embodiment/sem.py`) and GL2a's per-entity
previous-snapshot slot. No lock guards either today (code-read: `embodiment/body.py` and
`embodiment/sem.py` take none). The rule, in steps (amended 2026-10-09 by G9 and G17): **GL2a** mints
nothing and adds no lock over that state; **from the post-fence resume stage** the tool path mints on the
executor's thread, under the `EventSequencer`'s own private lock;
**with the out-of-band producer, after the fence,** records minted off the loop thread go through the
sequencer's lock, so ids never collide, and one `Embodiment`-owned lock guards the latch dict and the
snapshot slot; **from GL3.B3**, orchestrator-thread
transduction no longer calls `evaluate_failures` directly: it posts into the edge inbox, which the loop drains once per
pass (H0, §5.5), so the latch and the slot are touched from the loop thread only.

**The write stays cross-thread, so provenance is stamped at the write (TR23, environment D2, wiring D2).**
Moving the transduction behind the inbox does not move the write: the narrator tools write the AUT's
sensors on the orchestrator thread, and the loop's own evaluations (the llm-primary drift tick in
`_loop_live_tick`, which runs just before H0; an AUT tool in §4; a substrate tick) can transduce that
write first. Without a record of who wrote the sensor, they mint it `experienced`, which G6 forbids. The
rule: every body writer stores a per-sensor **last-writer provenance** at write time, in a kind map
beside the written `Entity`'s `vital_metrics` (not on the `Embodiment`: most writers hold only an
`Entity`, and the narrator's actor tool holds a scene `Embodiment`, §5.7 item 1, pre-merge Wire B1;
`world_set_axis` already takes `owner=`, a log tag today; the kind must be stored), and every
`evaluate_failures` caller reads it when it mints the event. The cradle reactive mother (`apparatus`,
from `main`) and the DM cascade (`sim.dm`) are covered by the same map.

### 3.3 The pain path

Body-sensor transduction happens in `embodiment/body.py::evaluate_failures`, which applies drift, latches
the breach on band entry, then `_publish_pain` → `proprioception/pain_bus.py::PainBus.publish`. Callers:
`propose_via_substrate` (substrate-primary), `loop_gates.py::tick_embodiment_drift` (llm-primary, once
per live pass), the reflex dispatch inside the AUT's `enrich` (loop thread), and the narrator's tools on
the orchestrator thread (§3.2). **It is not the only pain
ingress.** Producers that bypass it (code-read census, for GL3.B0's test):

| Producer | Path |
|---|---|
| `simulation/tools.py::DamageComponentTool` | constructs a `PainSignal` and publishes it directly, then also calls `evaluate_failures` |
| `runtime/sim_adapter.py::SimulationAdapter.next_observation` | a `proprioception` / `pain_signal` sim percept becomes a `Reaction` (source `sim_adapter:<pain_type>`) on the ReactionBus |
| `simulation/sandbox.py::PainTriggerLayer` | publishes a `Reaction` and a `PainSignal` onto the PainBus |
| `runtime/pain_interceptor.py::PainInterceptorExecutor` | publishes a `Reaction` onto the ReactionBus |
| `proprioception/perceived_pain.py::PerceivedPainAssessor` | constructs `PainSignal`s and publishes `Reaction`s |
| `bridges/pain_bridge.py::PainCircuitBridge` | the DefaultNetwork pain circuit (robot only) |
| `bridges/tool_pain_bridge.py::ToolPainBridge` | tool-coupled pain inside `execute` (T1-4) |

**GL3.B0 census corrections (2026-10-09; corrected after its review; `tests/unit/test_receptor_census.py::CENSUS`).**
- `ToolPainBridge` publishes no `PainSignal` or `Reaction` (it books attribution into NAc and emits SCN
  `TemporalEvent`s); `PainCircuitBridge` subscribes, and its `record_action_start` arms
  `PainDetector._check_movement_failure` (half of the robot's motion-pain wiring).
- **`PerceivedPainAssessor.assess_text` is a live, undeclared cross-thread pain ingress:** the
  orchestrator assigns it to `bridge.percept_anxiety_hook`, and `SimulationBridge.send_and_wait` calls it
  on every non-substrate-primary send, so it publishes into the AUT's PainBus / ReactionBus from whichever
  thread sends: the orchestrator, **`sim.dm`**, or `main`. `assess` runs inside
  `runtime/pain_interceptor.py::AnticipatoryPainExecutor` (before execute), not `PainInterceptorExecutor`.
- **Tool-failure pain is de-wired:** `PainDetector.record_tool_error` is reached only through
  `Executor._report_failure` when the executor has a `pain_detector`, and no `build_executor` caller passes
  one ([#1200](https://github.com/dennys246/Maxim/issues/1200); not fixed by GL3.B0). `record_tool_running` has no caller.
  #1200 is now "Stage 3: a PainBus FRUSTRATION signal (Stages 1–2 shipped)": its Stage 3 is the next new
  pain ingress, so it will trip `test_receptor_census.py`, and GL3.B3's ingress walk must expect it
  (wiring N3).
- `CerebellumModulator`'s prediction `Reaction`s are Dormant: `cerebellum_modulator_factory` has no caller.
- `PainTriggerLayer` publishes a `Reaction` when a ReactionBus exists, else a `PainSignal` (either, not both).
- `evaluate_failures` has three callers the list above omits: `ModulatorAffordanceTool.execute` (the
  AUT's tools on the loop; `OrchestratorActorTool`'s ephemeral tools on the orchestrator thread),
  `simulation/foundry.py::run_gauntlet` (no agent) and the Dormant `EmbodimentPerceptSource`.

`api.py`'s `on("pain_signal")` subscriber is a consumer, not a producer, but it subscribes to the bus
outside `build_pain_bus`'s ordered list. `PainBus.publish` is **synchronous on the publisher's thread**. Subscribers run in `build_pain_bus` order (memory → NAc outcome → Wire-2 percept
valence → Wire-4 cluster fear → extras), then a `Reaction` is forwarded to `ReactionBus`.
`proprioception/pain.py::PainKind` (NOCICEPTIVE / DRIVE / ANTICIPATORY / FRUSTRATION / EXHAUSTION) is the
closest existing thing to "which track", and today it decides memory encoding only.

**The fast paths that exist:** (1) tool-coupled pain, synchronous inside `execute` through `ToolPainBridge`
(row T1-4); (2) **same-call fear in substrate-primary**: inside `propose_via_substrate` the order is
encode channels → `nac.note_active_clusters` → `evaluate_failures` (pain → Wire-4 fear write) → re-read
drives → `anticipatory_threat_need` → `recommend_action`. **This order is load-bearing (Exp 58 W-4: the
encode must precede the pain tick, or the fear lands on the wrong cluster).** (3) The sim-only audio
orienting reflex (§1.16, excluded from substrate-primary). (4) The percept reflex registry
(`embodiment/reflex.py::ReflexRegistry`, `_data/reflexes/*.yaml`), wired only on the orchestrator's AUT.
(5) DefaultNetwork behaviours (robot only). (6) **CLI preemption**: in §1.5 a new non-follow-up CLI input
cancels `pending_action_followup` and any pending proposal whose `strategy_used` is `multi_step` or
`fallback`. **This is the only live preemption rule in the loop, and text triggers it, not the body.**

### 3.4 The delays, L1–L6 (code-read; magnitudes UNVERIFIED until GL3.B0 characterizes them)

**L1, L2 and L5 (and L7, §3.5) are filed as defect issues (owner decision G8): #1176 L1, #1177 L2, #1178 L5, #1179 L7.** Each is
fixable without the registry. Each issue names the root-cause seam its fix belongs in, and the issue's
review checks that the fix is not one of the special cases §4.2 warns about. Whether L1's fix is itself a
preemption check, and so the GL3.B4 arm, is the L1 issue's first question. The tracks (§5) later re-house
the timing these fixes repair; they do not wait for it. L7 is done (#1179 closed, the Dormant docstring
landed); L1, L2 and L5 are open.

**L1 and GL3.B4 (TR19).** GL3.B4's preemption check needs an event to read, and in substrate-primary
nothing transduces a breach between selection and §4 (§5.6). So **per-pass body transduction,
independent of selection, before H2, is a named prerequisite of GL3.B4**, settled with #1176. It keeps
the W-4 order: a per-pass transduction also encodes that pass, then `note_active_clusters`, then pain, in
one tick. It is the L1 seam's fix, made once, not a second `evaluate_failures` spot (§4.2).
**Flag status (pre-merge Executor B2, Architecture S7).** It is a root-cause defect fix, so it lands
**unflagged** in #1176's own PR, with its own trigger walk (per-pass encodes move NAc's per-pass inputs on
T1-13/14/15's path) and the T1-13/14/15 re-runs that walk owes. It is not behind GL3.B4's preemption key
or a key of its own: a toggle around a defect is the shape CLAUDE.md's "no band-aid" rule forbids, and it
sits outside GL3's three flags (header). After it lands, the frozen latency record is re-frozen as **v2**
on the post-#1176 tree; v1 stays as history (§6 GL3.B4).

- **Delay L1. Substrate-primary executes a proposal decided on the previous world state.** §6b sets
  `ctrl.pending_proposal` in pass N, and §4 executes it in pass N+1 after the sleep, with no
  re-evaluation. `_substrate_tick_due` requires `pending_proposal is None`, so no pain tick runs in
  between. A breach that starts after selection is not transduced until the next substrate tick (≥ 0.5 s
  later), and it cannot stop the stale action. During a held Minecraft primitive (`call_action` blocks),
  the pump keeps writing world state, but **no pain is evaluated until the hold returns**
  (`roadmap_1_4.md` §E budget: "0.58 s tick + its blocking hold").
- **Delay L2. In the orchestrator, the turn budget also gates nociception.** When
  `MAXIM_SUBSTRATE_ACTIONS_PER_TURN` denies a tick (`loop_substrate.py::substrate_tick`,
  `_substrate_gate_denied`), `propose_via_substrate` is skipped entirely, and with it
  `note_active_clusters` and `evaluate_failures`. For **drift-driven** breaches the band-entry latch then
  publishes late, stamped late, which misaligns NAc's temporal window (a narrator write is not delayed:
  its tool calls `evaluate_failures` immediately, on the orchestrator thread). A slow signal (the narrator
  turn) gates the fastest one.
- **Delay L3. LLM-primary cannot preempt deliberation.** A running LLM job cannot be cancelled. Pain does not
  invalidate a pending LLM proposal (`_drop_stale_proposal` drops only proposals older than
  `_STALE_PROPOSAL_AGE_S = 35 s`). `_run_deliberation_cycles` → `_wait_for_proposal` blocks the loop thread
  for up to 300 s per cycle, with no live tick, no intake and no decay during that time.
- **Delay L4. Sim percepts are one FIFO, one per pass.** `ConversationalSource._queue` is a `deque` with
  `popleft`, so a pain percept waits behind earlier text. `CompositePerceptSource.next_percept` takes the
  first non-None child. Priority was **deliberately deferred** by the design pass ("priority is exactly the
  N=1 policy-bake that earns the coordinator").
- **Delay L5. The "spinal" reflex sits behind the cortical gate.** `enrich` (and so `_evaluate_reflexes`) runs
  only if `_pfc_gate_passed`. `runtime/thought_gate.py::ThoughtGate.should_think` rejects on refractory
  (2 passes), on LLM **token** energy (< 0.15 of budget), on an empty working memory, or on an adaptive
  threshold that `goal_reward_bias` moves. A rejected percept has already been popped, so its reflex never
  fires. An innate reflex therefore depends on the token budget. That inverts the biology. How often it
  happens in a campaign is **UNVERIFIED**. (Bio-faithful S1: what the gate holds back is **narrated
  injury transduction**, not a motor reflex. Every shipped `_data/reflexes/*.yaml` response is
  `damage_component` or `set_entity_sensor`, a body write the body then transduces; see §5.4's `reflex`
  row.)
- **Delay L6. `ThalamicGate` is DN-only and vision-shaped.** `default_network/gate.py::ThalamicGate.evaluate`
  runs in `DefaultNetwork._process_tick` only when there are detections. There is no sim path.
  `DefaultNetwork.add_escalation_callback` has no caller.

### 3.5 L7: dead preemption scaffolding (verified by grep)

- `runtime/preemption.py::PreemptionCircuit` ("generalized interrupt system … Pain is the first registered
  source"), with `SourceConfig{threshold, cooldown, priority}`, `PreemptionSignal`, `ExecutionTracker` and
  `TOOL_REVERSALS`. **Nothing in `src/`, `tests/` or `scripts/` constructs it.** Nothing publishes a
  `PreemptionSignal`, and nothing calls `MaximAgent.wire_preemption`. It survives the orphan lint only
  because `runtime/__init__.py` re-exports `PreemptionCircuit` and `ExecutionTracker`.
- The `agent_loop.py` §3 branch `if hasattr(agent.goal, "check_hold")` is dead: no `check_hold` exists in
  `src/`.
- `runtime/tool_dispatch.py`: the `agent._execution_tracker.capture_before(...)` guard can never be true,
  because only `wire_preemption` sets it.

Its vocabulary (per-source threshold, cooldown, priority; graded withdrawal; a reversal per tool) is
close to what a track needs. Its mechanics (wall-clock cooldowns, AgentBus pub/sub, no producer, no
ordering) are not. **Done (TR1, #1179 closed):** it is marked `Dormant since
2026-10-07: never wired; superseded in vocabulary by AfferentTrackSpec` in its module docstring. The
recommendation that led there, kept as rationale: strict, dormancy over deletion, filed as
the L7 defect issue. Deletion is the non-strict alternative: it removes the `PreemptionCircuit` /
`ExecutionTracker` re-exports from `maxim.runtime` (a CHANGELOG line). The scheduler does **not** move into
`runtime/preemption.py`; it lives in `perception/` (§5.3).

---

## 4. Front-gate scope pressure

### 4.1 Why this is not the manifest the 2026-07 audit rejected

`archive/percept_testbed_audit.md` rejected "a percept-channel manifest that rides `PerceptSource`" on
four counts. Each count is answered by construction, not by assertion:

| Audit count | How this plan avoids it |
|---|---|
| (1) Two sources of truth: it collided with `perception_pipeline_placement.md`'s config surface and with the body YAML | **Body-sensor `ReceptorSpec`s are derived from body YAML at bind time** (the `modality:` / `drive:` keys that `_SUBSTRATE_CHANNELS` already reads, as instantiated on the root entity tree, which is what `registry_for` reads, §5.3), never authored in a parallel file. The embedding space is derived from `encoding_geometry_tag`. The new YAML keys (`tracks:`, `relay_gain:`) each get their reader in the same PR (the #1120 lesson: `coupled_to` / `modulated_by` are parsed and never read). |
| (2) `PerceptSource` is not a uniform seam, and substrate-primary reads only the body path, so a `PerceptSource` manifest is inert in the LLM-free mode | The registry's **core is the body path**, `_SUBSTRATE_CHANNELS`, not `PerceptSource`. `PerceptSource`-shaped producers are wrapped later (GL3.B5) and stay behind `CompositePerceptSource` as an edge multiplexer. |
| (3) Audio azimuth is double-represented by design (signed extero cluster + folded intero drive), so a flat `{enabled, gain, noise}` entry is ill-defined | The straddle stays two representations. It becomes **one Receptor emitting onto two tracks under one `PhysicalEventId`**. Gain stays per-route (design pass DP3): `relay_gain` on the relay route, motivational gain in the drive's `pain_scale`. |
| (4) Per-channel attribution could not be measured | The registry stamps every route it takes or gates, per receptor (`ec.record_encoder_provenance(f"receptor:{id}", …)`), which is the testbed audit's M3 telemetry. |

It also keeps SUBSUME: the coordinator is introduced now because the second need the design pass waited
for has arrived. **Provenance and event identity must be stamped in one place, or the next producer
forgets them.** Three silent misses have already happened: imagined writes are unflagged, `agent_id` is
None, and `perceived_intensity` is never set. Under CLAUDE.md's silent-no-op rule, that count calls for
structure, not another helper.

### 4.2 Per mechanism: why existing infrastructure cannot do this

**`ReceptorRegistry` + `ReceptorSpec`.** `ModalityChannel` already gives one channel, one encode and one
EC space, but it has no identity, no provenance and no `agent_id` of its own. Several `src/` sites call
encoders directly (the census's `linguistic_encode` and `sensor_encode` categories in
`tests/unit/test_receptor_census.py`, which count five EC sites: `_encode_current_clusters`,
`propose_via_substrate`, `MemoryHub.on_percept_received` and the two `imagination/trigger.py` affordance
paths, one of which is the orchestrator's self-entity encode; plus the internal `LinguisticEncoder.encode`
and the non-EC `ComponentIndex._embed`; wiring N1), and the survival scripts call the body-path
encoders directly too (§5.3), so there is no single place to stamp them. `CompositePerceptSource` is
deliberately priority-free and identity-free. **It rides on:** `ModalityChannel` (becomes the
`sensor_sum` adapter), `SensorEncoder.encode_sensors(modality=)`, `LinguisticEncoder`,
`encoding_geometry_tag` / `record_encoder_provenance`, `DECLARABLE_MODALITY_TAGS`, the body-YAML
declarations, and `PerceptSource` (CC8).

**`AfferentEvent` (on the autonomic plan's `PhysicalEventId`, built at its post-fence resume stage, G17).** No existing record joins two lanes of one physical
event. DoAFeed's lanes share only a wall timestamp, which is not a join key under turn-based or lockstep
clocks. `TemporalEvent` is the right credit envelope, but the uuid its producers mint and its wall-clock
signature are nondeterministic, and it is coupled to SCN registration. It is kept as the envelope, and it
carries `str(pid)` in `context` instead of changing its frozen shape. **It rides on:** the resume stage's
`PhysicalEventId` and per-agent `EventSequencer` (which follow the `Hippocampus.capture_seq` pattern,
including its resume-past-max-on-load rule) and `ExperienceClock`. This plan adds no second identity type.

**`TrackScheduler` + `AfferentTrackSpec`.** PainBus and ReactionBus are synchronous pub/sub on the
publisher's thread with wall-clock refractories. They have no latency, no per-signal destination subset,
no preemption right, and no way to show one physical event on two channels at different times. Adding
that to PainBus would turn a transport into a scheduler for every publisher at once (the T1-4 and T3-9
blast radius). `PreemptionCircuit` has the vocabulary but no producer, no consumer, wall-clock cooldowns
and no ordering. `ThalamicGate` runs on the DN thread, is vision-shaped, and does not run in sim. **It
rides on:** `ExperienceClock` (stamps), `loop_setup.py::LoopRun` / `build_loop_run` (one build site),
`pre_tick_gate`'s live-tick slot (the drain point), PainBus subscribers (destinations), `PainKind` (track
membership), `ReflexRegistry` maths, and `_loop_harness` (lockstep gates).

**EC per-node provenance (GL3.B1).** `Episode.imagined` is episode-level, CausalLink `imagined` is set
retroactively at session end, and EC `source` records hivemind origin. None reaches
`register_substrate_node` / `update_eligibility` at write time.

**What the defects need, and what they do not.** Delays L1, L2 and L5 are defects with root causes of
their own, and owner decision G8 files them as issues now, fixable without the registry (§3.4). What the
issues must avoid is the special-case shape: re-running `evaluate_failures` in a second spot before §4,
an "always run pain" bypass beside the turn gate, a reflex call duplicated outside the gate. Each issue
fixes its seam once (L2, for example, by separating transduction from selection inside the substrate tick,
so the turn budget gates action only). The per-pass transduction GL3.B4 needs (TR19, §3.4) is that same
shape, made once at the L1 seam (#1176): it is the seam fix, not a second spot. What the defect fixes cannot give, and the reason the registry and
tracks still exist, is the shared identity of one physical event seen on fast and slow channels, with its
provenance, for GL4 and the credit line.

**Count:** one new package with a thin coordinator and a scheduler, three types (`ReceptorSpec`,
`AfferentTrackSpec`, `AfferentEvent`; `PhysicalEventId` is the autonomic plan's, built at its post-fence
resume stage, G17), and one new EC parallel dict. No new
encoder, no new EC modality, no new bus, no class per modality. `ModalityChannel` stops being the registry
and becomes one encoder adapter.

---

## 5. Design

### 5.1 Principles

1. **Receptors declare; the registry routes.** A Receptor turns a world or body change into a reading and
   declares what it is. The registry is the single registration and routing point: it applies the gate
   and relay gain, stamps identity and provenance, and routes to EC, the LLM and tracks. **Receptors never
   call `encode_sensors`, `LinguisticEncoder.encode*` or `NAc.*` themselves.** Nuclei stay declarations.
2. **Transport stays; timing moves.** PainBus stays the subscriber registry and transport.
   `build_pain_bus`'s required-subscriber invariant is untouched. A track decides *when* a publish
   happens and *to which destinations*.
3. **Tracks are logical, on the loop's clock.** Real concurrency stays at the hardware edges that already
   exist (pump, bridge reader, DoA feed, stdin, WorkerPool, DN). Edges post into a lock-guarded **edge
   inbox**. The loop thread drains it at exactly one point per pass and assigns sequence numbers there.
   That single drain point is what makes ordering deterministic.
4. **The relay stays thin.** Multiplex, gate and gain, route, stamp. Nothing else. Stop rule: if registry
   code grows behaviour that belongs in a nucleus, push it back down (framing-doc rule).
5. **Thalamic by analogy only for the exteroceptive and nociceptive relay** (bio-faithful N1). The
   registry also routes imagination, language and body drives; much interoception reaches the
   hypothalamus and insula through NTS / parabrachial without a thalamic relay, a spinal reflex never
   reaches the thalamus, and imagery is not afferent at all. Elsewhere the registry is a registration and
   provenance point. A static `enabled` plus a fixed `relay_gain` is a thin relay, not thalamic gating
   (which is state-dependent: the reticular nucleus, burst vs tonic mode).

### 5.2 Types (one definition each; the drafts' `PerceptEngineSpec` → `ReceptorSpec`, `TrackSpec` → `AfferentTrackSpec`)

```python
# Imported, not defined here. The leaf module (no maxim imports), built at the autonomic plan's
# post-fence resume stage (G17), not GL2a:
#   from maxim.embodiment.event_id import PhysicalEventId
# PhysicalEventId(agent_id: str, session_id: str, seq: int): frozen, SHAPE-FROZEN at 1.0 (CC3 path b);
# __post_init__ rejects an empty agent_id, an empty session_id and a negative seq;
# __str__ is "{agent_id}:{session_id}:{seq}" (G15).
# ONE join rule: InteroceptiveOutcome.pid, CauseRef.cause_pid, AfferentEvent.pid,
# TemporalEvent.context["pid"] and GL4 training rows all join on the pid, never on the
# executor's uuid4 invocation id (not persisted-stable).

ProvenanceKind = Literal["experienced", "narrated", "imagined", "apparatus"]   # apparatus: G16 (harness writes;
                                                   # never trained on or credited); + "declared"/"reported":
                                                   # owner decision (§11, GL3.B1). No "unknown"
                                                   # kind (TR29, G29): a MISSING last-writer slot
                                                   # means "not yet written" and the reader maps it
                                                   # to "legacy" (credited as today); nothing
                                                   # defaults to "experienced" (§5.7 item 1)
ReceptorClass  = Literal["exteroceptive", "interoceptive", "proprioceptive",
                         "nociceptive", "composite"]   # bio-faithful S7: "composite" for a
                         # game-state channel that mixes classes (world.state carries vitals),
                         # unless the derived spec stamps receptor_class per sensor; "linguistic" is
                         # an encoding (EncoderKind covers it), and "efference" (corollary discharge,
                         # not afferent) is dropped until a producer exists
EncoderKind    = Literal["sensor_sum", "linguistic", "linguistic_affordance", "precomputed"]

@dataclass(frozen=True)
class ReceptorSpec:
    """One registered receptor. CC3 path (a): defaults on every field + extra, because its stamp
    (receptor_id, receptor_class, provenance_kinds, encoder, derived geometry) is persisted into
    EC encoder provenance (ec.json) and the bundle manifest's encoder_provenance."""
    receptor_id: str = ""                 # stable, dotted: "minecraft.world", "body.drives",
                                          # "reachy.doa", "sim.narrator", "imagination.affordance"
    receptor_class: ReceptorClass = "exteroceptive"
    modality_tag: str = ""                # EC namespace; validated against ONE registry
    encoder: EncoderKind = "sensor_sum"
    provenance_kinds: frozenset[str] = frozenset()   # what it MAY emit. The empty default is a
                                          # SENTINEL that __post_init__ rejects: no type defaults
                                          # provenance to "experienced"; every spec declares it
    tracks: tuple[str, ...] = ()          # validated against the AfferentTrack registry
    clock: Literal["realtime", "turn"] = "realtime"   # DERIVED from ExperienceClockDriver's kind
                                          # (percept_source.experience_turns), never authored
                                          # (environment N3: one source of truth, as TR5)
    relay_gain: float = 1.0               # per-ROUTE relay gain (design pass DP3); motivational
                                          # gain stays in the drive's pain_scale (hypothalamus side)
    enabled: bool = True                  # design pass DP3: one gate at the routing fork
    extra: dict = field(default_factory=dict, hash=False, compare=False)
    def __post_init__(self): ...          # non-empty id/tag; tag registered; provenance_kinds
                                          # non-empty and within ProvenanceKind; relay_gain finite
                                          # >= 0; extra keys must not collide with fields

@dataclass(frozen=True)
class AfferentTrackSpec:
    """One logical afferent pathway. CC3 path (a): it is recorded in the run fingerprint
    (M10 as amended 2026-10-07 / testbed-audit M2), so it persists."""
    name: str = ""
    bio_map: str = ""                     # e.g. "A-delta / spinothalamic (first pain)"
    latency_passes: int = 0               # 0 = same drain phase; k = deliverable k passes later
    priority: int = 0                     # higher drains first within a phase
    preempts: frozenset[str] = frozenset()     # {"pending_proposal","in_flight_plan","held_action"}
    preempt_threshold: float = 1.0        # intensity at/above which preemption fires
    gain: float = 1.0                     # innate prior; adaptive later (adaptive_nociception.md)
    destinations: frozenset[str] = frozenset() # {"reflex","autonomic","relay","credit","prompt","pain_bus"}
    refractory_passes: int = 0            # per (agent, receptor, locus, track), counted in PASSES
    coalesce: Literal["latest", "max", "sum"] = "latest"
    extra: dict = field(default_factory=dict, hash=False, compare=False)

@dataclass(frozen=True)
class AfferentEvent:
    """The receptor→registry handoff: ONE physical event. The scheduler delivers the SAME
    object on every track the receptor declares; a delivery is (AfferentTrackSpec, AfferentEvent),
    and track gain is applied by the destination from the spec. CC3 path (a)."""
    pid: PhysicalEventId | None = None    # required in practice: __post_init__ raises if None
    receptor_id: str = ""                 # required: non-empty, registered
    provenance: str = ""                  # required: the "" sentinel is rejected; must be in the
                                          # spec's provenance_kinds (never defaulted to "experienced")
    locus: str = ""                       # entity path / body part / sensor name
    intensity: float = 0.0                # raw, before track gain
    payload: Mapping[str, Any] = ...      # tagged by EncoderKind: sensor values+ranges | text; JSON-safe
    experience_us: int = 0                # ExperienceClock stamp at ingest (the Hippocampus unit)
    pass_index: int = 0                   # the ingest pass: the NON-IDLE pass count (the NAc-decay
                                          # clock), stamped once at H0 (wiring S9, environment S2)
    # salience / novelty (fabric §C convention) are not fields until a producer exists: they
    # would be dead attention fields, the gap-7 pattern (bio-faithful N2); they may ride in extra
    caused_by: PhysicalEventId | None = None   # e.g. "[minecraft:damage]" text → the health-drop event
    extra: dict = field(default_factory=dict, hash=False, compare=False)
```

**Reconciliation choices (put to the four-lens review):**
- **One `PhysicalEventId`, the autonomic plan's.** One planning draft hashed `(engine_id, agent_id,
  engine_seq)` and another used `(agent, seq)`. Owner decision G3 was `(agent, seq)`, amended by G15 to
  `(agent, session id, seq)`; the type lives once, in `maxim/embodiment/event_id.py`, built at the
  post-fence resume stage (G17). The receptor is a field of the event, not of its identity, so one burn
  seen by two receptors (thermal sensor plus the derived drive) can still share or link ids through
  `caused_by`.
- **One seq authority per agent at a time.** The per-agent `EventSequencer` (built at the post-fence
  resume stage, G17), held by the agent's primary `Embodiment`, assigns seq until GL3.B3; from GL3.B3 the scheduler's drain point assigns it and
  the sequencer becomes the scheduler's backing store (persisted high-water). Ephemeral, scene and foundry
  wrappers (`agent_id == ""`, `simulation/tools.py`'s `scene_emb`, the `simulation/foundry.py` wrappers)
  are not the AUT: they mint no records and no ids. (Amended 2026-10-09 by G15 and G17:) GL2a mints no
  pid; the type, the sequencer, its session-id source and the cross-session resume past the sequencer's
  own high-water mark (the `Hippocampus._resume_capture_seq` rule), wired at both load seams
  (`bio_stack.build_bio_stack`, `orchestrator._restore_aut_from_session`), land together after the fence
  and before GL4 S1, so they are in place before GL3.B3's handover.
- **`agent_id` lives once**, on the pid. The event carries no second copy that could disagree.
- **Pid granularity (wiring S1).** A pid names a physical event, never a read. Pull reads carry **no
  pid** at GL3.B1: EC provenance is per node, not per event, so `registry.tick` builds no `AfferentEvent`
  per read, and the required-pid rule applies only to events that are emitted (GL3.B3 onward). Once
  GL3.B3 emits, a pull read joins its transduction through `caused_by`. The pid granularity of the GL4
  pair is the tool-path invocation pid (§6 GL3.B2). **One pid per invocation (pre-merge Wire B3):** the
  invocation pid is minted at `Executor._run_started`, before `tool.run`, once the post-fence resume
  stage builds the type (G17), and every H3t `AfferentEvent` emitted inside that invocation's `tool.run`
  **reuses** it rather than drawing a new seq; `_stamp_invocation` stamps the same pid on the
  `ToolOutput` and mints nothing. **Where the pid and seq are drawn (delta round, Executor S-F, Wire
  N-5):** at the mint, `Executor._run_started`, on whatever thread runs `execute` (the loop thread on
  every loop path; a `ConcurrentExecutor` pool thread when that public, internally uncalled class is
  used), under the sequencer's lock. Before GL3.B3 that lock is the `EventSequencer`'s; from GL3.B3's
  handover it is the same counter as the scheduler's backing store, so there is still one authority.
  The total-order and cross-process determinism claim (§5.5, §5.10) is **scoped to the single-threaded
  loop paths**; `ConcurrentExecutor` is **outside the determinism claim** (its pids stay unique and
  monotonic under the lock, but their order follows pool scheduling). The carrier from `_run_started`
  to the H3t emit is named by the GL3.B3 PR: `Embodiment.outcome_window()` does **not** carry the pid
  today (`OutcomeWindow` holds only `drift`, `drift_dt_s` and `cleared`, and the window opens only on
  the outcome-tracked branch), so the PR either adds the pid to that thread-local window, with the
  window opened on every branch that reaches `tool.run`, or uses an execution-context value. So the burn's track copies, its
  `InteroceptiveOutcome` and the GL4 pair carry one pid, with no `caused_by` hop on the tool path. So at
  GL3.B1 an ad-hoc `Embodiment` (red gate (g)'s)
  and the scripts that rebuild one agent (`r3_run.py`, `exp61_run.py`) need no sequencer for their pull
  reads; G17's "a rebuild fails loudly" applies where events are emitted, from the sequencer the primary
  `Embodiment` holds.
- **No wall time on the event.** One draft had `wall_ts`. It is dropped: `experience_us` is the analysis
  clock, `Percept.timestamp` still exists for the LLM route, and a wall field would break byte-identical
  traces.
- **CC3 vs the silent-no-op rule.** CC3 path (a) wants defaults on every field. The silent-no-op rule
  wants the identity core to be impossible to omit. The recommendation is sentinel defaults that
  `__post_init__` rejects with `ValueError` (a loud failure at construction), which satisfies both. The
  alternative is keyword-only required fields (a `TypeError`) plus path (b). This is a review item, not an
  owner decision. Either way, **no provenance field has a usable default**: `ReceptorSpec.provenance_kinds`,
  `AfferentEvent.provenance`, GL2a's `InteroceptiveOutcome` and GL4's `ActionContext` (latent_forward_model.md) each reject a
  missing or sentinel provenance in `__post_init__`.

### 5.3 The registry

`ReceptorRegistry` (in `perception/`, built once per agent and **bound to the agent's primary
`Embodiment`**, TR21, through one named binder):

- **Where it lives (TR21, wiring D1).** **One binder (pre-merge Executor B1, Wire B2):**
  `perception/`'s `registry_for(embodiment)`, called from `runtime/` (`_encode_current_clusters` and
  `propose_via_substrate` call it with `executor.embodiment`, with no new keyword) and never imported by
  `embodiment/`. On first use it derives the registry deterministically from the `Embodiment`'s **root
  entity tree** (TR5): the sensors declared on `Embodiment.root` and its components (modulators), the
  same keys `_SUBSTRATE_CHANNELS`' readers already read. It is not derived from a YAML path: an
  `Embodiment` keeps `root`, `config`, `pain_bus`, `distributor` and `agent_id`, and `Entity` has no
  source-file field, so "built from a body file" is not observable (delta round, Executor S-A, Wire
  D-1, Architecture SF-2). It caches the result in the `Embodiment`'s `Receptor`-protocol slot, built
  **once per `Embodiment` under an `RLock`** (delta round, Architecture N5: the loop thread and an
  outcome-time encode on llm-primary could otherwise both build one, splitting gate 5's per-receptor
  route counter); every later call returns the cached one. Because the derivation is a pure function
  of the declared tree, it is a **derivation, not a fallback**: there is no second path that could
  produce a different registry, and a caller cannot reach an un-derived one. The typed error is kept
  for exactly one observable case, **a root entity tree that declares no sensors** (on the root or any
  component), so it can never silently encode through a bare `ModalityChannel`. Three things decide this. The slice-3 pin
  (`tests/unit/test_loop_substrate_characterization.py::test_the_proposer_gets_the_runs_handles_by_identity`,
  the proposer's exact keyword set) stays intact. The EARNED rows' shipped path is not the loop: the
  water-trial trainers (`scripts/survival_world/water_trial.py::WaterTrial`, imported by `exp60_run.py`,
  `exp61_run.py`, `exp62_run.py`, `exp62_precheck.py`, `r3_run.py`, `r3_pilot.py`, `b1_floor_run.py`,
  `scripted_water.py`), `exp58_run.py`, `exp58_offline_gates.py`,
  `orient_backbone/exp53_cross_context_readout.py`, `r2_learned_bias.py`, `dark_danger_probe.py`,
  `instrument_check.py` and, in `src/`, `substrate_proposal.py::_attach_live_situation` call the two
  functions directly. Every one of them reaches the registry **with no edit** because the binder runs
  inside the two functions on first use, whatever built the executor: `build_executor` (the water-trial
  trainers through `build_minecraft_aut`, the loop), `scripts/orient_backbone/exp53_cross_context_readout.py`'s
  bare `Embodiment(entity)` and hand-rolled `_Executor(registry, embodiment)`, and the test stubs
  (`test_modality_seam.py::_StubExecutor`, `_loop_harness`, `_cradle_loop_driver.py`). An earlier draft
  said "bound at bind time" and "reached unchanged" in the same breath; with no bind site in exp53 or in
  red gate (g), that was false, and the lazy binder is what makes it true. Red gate (g)'s read surface
  (`_encode_current_clusters(SensorEncoder(...), "aut-1", SimpleNamespace(embodiment=Embodiment(body)))`)
  reaches it the same way: `Embodiment(body)`'s root tree declares sensors (it is instantiated from the
  component registry, as exp53's is), so `registry_for` derives the registry and
  (g) can flip as written (it is `xfail(strict=True, raises=AssertionError)`, so a typed error would FAIL
  it, not flip it; GL3.B1 adds a known-answer test that (g)'s exact surface reaches a registry). Callers
  that pass a reader-only `SimpleNamespace(embodiment=SimpleNamespace(root=…))` are named in the GL3.B1
  PR: a stub whose root tree declares no sensors takes the typed error and is updated in that PR, and
  reader-only callers never reach the binder. `tests/unit/test_memory_2sc_signals.py`'s `MagicMock()`
  executor does not reach the binder at all: that test monkeypatches `_encode_current_clusters` away
  (delta round, Wire D-1).
  `embodiment/` must not import `perception/`: `Embodiment` declares the slot, and the import direction is
  `perception → embodiment` (wiring N2; the ARCHITECTURE.md layer entry lands with TR4). `LoopRun` carries
  only the GL3.B3 scheduler.
- `register(spec, receptor)`: the `Receptor` is a protocol with `spec` plus either a **pull** reader
  (`read(executor) -> payload | None`, which is today's `ModalityChannel` reader) or a **push** adapter for
  hardware edges, which posts into the scheduler's edge inbox. A duplicate `receptor_id` raises. An
  unregistered `modality_tag` raises. Registration order is preserved and **equals today's code order**,
  `_SUBSTRATE_CHANNELS` = (interoception, audio, world), because that order and the number of active
  channels set the summed cluster term in `recommend_action`.
- `tick(agent_id, executor)`: pulls every pull receptor, keeps the **empty-read → no-encode** rule,
  stamps provenance (no pid on a pull read, §5.2), and routes:
  - **EC route** per `encoder`: `sensor_sum` → `SensorEncoder.encode_sensors(modality=spec.modality_tag,
    ranges=…, provenance=…)` (the new keyword-only `provenance=`, TR22; otherwise unchanged);
    `linguistic*` → `LinguisticEncoder`. It returns **`(channel_values, clusters)`** (wiring S5):
    `clusters` is `{modality: cluster_id}`, the same dict `_encode_current_clusters` returns today, and
    `channel_values` is the raw reading `propose_via_substrate` hands to `_select_action` (a
    `SubstratePolicy` gets `observation=MappingProxyType(channel_values.get(WORLD_TAG))`, the input the
    BF2 floor policies in `scripts/survival_world/floor_policies.py` select on). Each caller keeps its
    current read/encode order: `propose_via_substrate` reads every channel and then encodes,
    `_encode_current_clusters` reads and encodes per channel; with the live pump writing between reads
    (§3.2), the two orders are not interchangeable.
  - **Synchronous event route (wiring S2):** `encode_event(receptor_id, payload, *, provenance)` for
    event-driven receptors at GL3.B1 (the imagination and own-body affordance encodes), run on the
    caller's thread: the loop thread, or `main` for the fixture runner's manifest and for
    `start_simulation_mode`'s self-entity encode before `sim.aut` starts. Routing them through the edge
    inbox would defer imagined encodes to the next H0, a behaviour change; the inbox stays GL3.B3's.
  - **LLM route:** the existing nuclei, unchanged: `BioEnrichmentPipeline` for text, `ThalamicGate` for
    DN vision, the §1.16 fold for audio.
  - **Track route:** one scheduler emit per event, fanned out to `spec.tracks` (GL3.B3 onward).
  - **Stamp:** `ec.record_encoder_provenance(f"receptor:{receptor_id}", {...spec stamp, derived geometry})`,
    which reuses the existing per-recorder stamp and its merge rules. **This stamp is a wire format, not a
    log line:** `EntorhinalCortex._encoder_provenance` is saved into `ec.json`'s `encoder_provenance` key,
    and `hivemind/cli.py` carries that key into `hivemind/bundle.py`'s `manifest["encoder_provenance"]`.
    GL3.B1's gate therefore covers the EC save bytes and the bundle-manifest bytes (§6), and its blast
    radius includes T1-10 and T1-14 (§7).
  - **Route counter (TR22, confounding D2):** the registry counts, per receptor, the encodes it routed in
    each `tick`. GL3.B1's deletion probe reads it (§6 GL3.B1 gate 5): a build that writes stamps but is
    not on the encode route reads zero.
- `enabled=False` drops the event at the fork for every route and records which routes were gated.

### 5.4 The seven AfferentTracks

Every number is an **innate prior**: hard-coded on purpose, reviewed at GL3.B3 start, and owed a follow-up
with a trigger (`adaptive_nociception.md` for gain, `nociception_layer.md` step 5 for membership). Latency
is in **loop passes** and is ordinal, not metric, and not distance-dependent (bio-faithful N6): passes
are irregular in wall time, so the Aδ-before-C ordering is preserved and no millisecond realism, and no
fixed first/second-pain interval, is claimed. Docstrings must say "FUNCTIONAL, not fibre dynamics", as
`PainBus` already does, and "FUNCTIONAL: one threshold for both fibre classes" (GL1 N4, bio-faithful N3).

**What a pass is (wiring S9, environment S2).** `pass_index` is one named counter: the **non-idle** pass
count, the clock NAc decay already runs on, stamped once at H0. The loop's `step_num` also counts idle
iterations, and the idle gate `continue`s before §8.5, so H3 and H4 do not run on idle iterations. A
delivery that comes due on an idle iteration waits for the next non-idle drain (until H5 exists). The
scheduler keeps a **delivery log** of `(pid, track, delivery_pass, experience_us_at_delivery)`, because the
shared `AfferentEvent` carries only its ingest pass; GL3.B3's gates read the log. Wall time per pass by
world:

| World | Wall time per pass |
|---|---|
| `_loop_harness` | 0.05 s idle, 0.25 s running (`_StepClock`) |
| Live Minecraft | a pass holds any blocking `call_action` (up to 8 s for `escape_water`, 15 s action timeout) |
| LLM-primary | up to 300 s per deliberation cycle (L3); the AUT turn window is 30 s (`sim.aut_turn_timeout_s`) |

In the turn-based cradle, `experience_us` advances only per narrator turn
(`ExperienceClockDriver.on_live_pass`, the `experience_turns` branch), so the fast and slow deliveries of
one burn usually share one `experience_us`. GL4 orders deliveries by `(seq, delivery_pass)`, never by
`experience_us`, on turn worlds. **Only `nociceptive_fast` and `affective_slow` are built, at
GL3.B3, as a fan-out with no preemption; `nociceptive_fast`'s preemption right is exercised only at
GL3.B4. The other five are proposals that enter only with a consumer.** Gain is 1.0 on every track at
entry, so no intensity changes until a stage earns a different value.

**Bio mapping notes.** Thermoreception and thermal pain run on the anterolateral (spinothalamic) system,
not the dorsal column, so a thermal receptor emits on `nociceptive_fast` / `affective_slow`, never on
`mechano_proprio`; its innocuous reading stays in the body channel through the registry's EC route.
Sound direction (DoA) is auditory (brainstem → MGN → auditory cortex), so it rides `extero_detail`, with
its orienting response on `reflex` (collicular), not `mechano_proprio`. `affective_slow` carrying DRIVE
breaches (hunger, air hunger) is justified **only** as Craig's lamina-I homeostatic pathway (lamina I →
spinothalamic / parabrachial → insula), which carries homeostatic as well as nociceptive afferents; it is
not a C-fibre pain claim about hunger. It is a route label only: air hunger (chemoreceptive dyspnea) is
one of the fastest, most aversive interoceptive alarms, so the **teaching signal is never delayed**:
every PainBus publish stays at the transduction pass (TR18, bio-faithful D1).

**The reflex emitters, honestly labelled (bio-faithful S1).** The keyword reflexes in `ReflexRegistry`
inflict damage: every shipped `_data/reflexes/*.yaml` response is `damage_component` or
`set_entity_sensor` (`attack_flinch` damages the torso; `thermal_contact` damages the arms), and the
module's own "Known layering debt" says its habituation scales the damage the world inflicts.
`deferred/reflex_layering.md` finds its stage 4 (withdraw / flinch) never executed. So the keyword
reflexes are a **narrated nociceptive receptor** (`receptor_class="nociceptive"`,
`provenance_kinds={"narrated"}`) whose output is a body write that the body then transduces onto
`nociceptive_fast`. The `reflex` track keeps only emitters whose response is a motor act; today that is
the sim-only §1.16 audio orienting reflex. Nociceptors sensitize; they do not habituate.

**Minecraft's world sensors, per sensor (bio-faithful S5, from grounding.md's track table).** Vitals
(`health`, `food`, `saturation`, `oxygen`) → `affective_slow` (with `nociceptive_fast` for
`drive:health`); distance senses (`light_level`, `nearest_hostile_dist`, `hostile_count`,
`nearest_player_dist`, `is_raining`) → `extero_detail`; allocentric position (`y_altitude`,
`distance_from_spawn`: hippocampal / path integration, not proprioception) → `extero_detail` as a
named spatial input; vestibular / proprioceptive (`speed`, `on_ground`, `look_pitch`) and contact
(`is_in_water`: cutaneous plus vestibular) → `mechano_proprio`; `time_of_day` → its own clock input. The mapping lives here, and GL3.B3's track-membership
test enforces it; the GL3.B0 census checks sites, not membership. The EC `world` modality stays
byte-identical (track membership only).

| Track | Bio mapping | Latency (passes) | Priority | Preemption rights | Gain | Destinations | Refractory / coalesce | Emitters today → subsumes |
|---|---|---|---|---|---|---|---|---|
| `reflex` | spinal reflex arc (withdrawal, flexor); superior colliculus orienting: **motor acts only** (bio-faithful S1) | 0 | 100 | pending proposal (held action later) | 1.0 | reflex, autonomic | track refractory 0 / latest; no adaptation of its own | the audio orienting reflex (§1.16 reflex tier). The keyword reflexes are **not** this track's emitters: they are a narrated nociceptive receptor whose body write transduces onto `nociceptive_fast` (note above); moving their evaluation **out of** `enrich` / ThoughtGate (L5) re-houses narrated injury transduction, not a motor reflex |
| `nociceptive_fast` | Aδ, spinothalamic "first pain": sharp, localized (incl. noxious heat) | 0 | 90 | pending proposal, in-flight plan (exercised only from GL3.B4) | 1.0 | pain_bus (the learning subscribers, unchanged, published **at the transduction pass**), credit (eligibility stamp), relay (situation) | 1 pass per (receptor, locus), **coalesce (max) and deliver at the window's end, never drop** (bio-faithful S8), **except the `pain_bus` destination**, which coalesces within its transduction pass only and is never deferred (pre-merge Executor S3, §5.5); at most one preemption per breach episode (§5.6) / max | `body.evaluate_failures` for `PainKind.NOCICEPTIVE` (incl. `drive:health`), tool damage, `inject_pain` → PainBus *timing* for NOCICEPTIVE; the dead `PreemptionCircuit`'s pain source |
| `affective_slow` | C fibre, spinoreticular / spinoparabrachial "second pain"; for DRIVE breaches, Craig's lamina-I homeostatic pathway (see the note above). **FUNCTIONAL: a delayed phasic copy of one latched event**; tonic persistence is carried by drive pressure and the latch severity, not by this track (bio-faithful S2) | 2 | 50 | none | 1.0 | **only sinks that are not PainBus subscribers** (TR18): autonomic (GL2's `InteroceptiveOutcome`), a pid-keyed consequence record. No cluster fear, no memory tag, no prompt line until a tonic source exists | 0 / sum, clamped at 1.0; a coalesced delivery carries the **set** of member pids | the same physical event as `nociceptive_fast`, plus `PainKind.DRIVE` breaches (whose PainBus publish stays at transduction, in place) → nothing today; this is GL2's input path. At GL3.B3 no consumer acts on the slow delivery differently from the fast one: it is capability with no caller until GL2's out-of-band producer or GL4 joins it. **After TR18 its consumers are record-only** (pre-merge Architecture S8): GL2a's `InteroceptiveOutcome` has no reader that acts on it, and the pid-keyed consequence record has none before GL4. So `affective_slow` is infrastructure under T4a: if no acting reader (GL4's join, or GL2's out-of-band producer feeding one) exists by 1.4.0, it is marked `Dormant since <date>` then. Credit reaches only through the PainBus publishes at the transduction pass (`nociceptive_fast`'s, and the in-place DRIVE publish), never through this track |
| `mechano_proprio` | dorsal column / medial lemniscus: touch, proprioception (not thermal, not auditory); the Cerebellum destination is FUNCTIONAL (spinocerebellar tracts feed it in biology, bio-faithful N5) | 0 | 60 | none | 1.0 | relay (body schema / interoception channel), credit (Cerebellum forward-model error) | 0 / latest | Minecraft `speed`, `on_ground`, `look_pitch`, `is_in_water` via the pump (the per-sensor note above), `world_set_axis`, SEM `self_effect` deltas on mechanical/postural sensors → the pump's untimed `vital_metrics` writes become ingested events |
| `threat_low_road` | **contextual fear (hippocampus → amygdala)**, not LeDoux's low road (bio-faithful S6): its emitter is a learned, post-encoding EC cluster, which cannot deliver earlier than the encode. The low-road name is kept only for an emitter fed by a raw pre-encode sensor feature (e.g. a thresholded `nearest_hostile_dist`); the low road is contested in primates (Pessoa & Adolphs 2010). The bio-map string is fixed before any spec persists | 0 | 80 | pending proposal (only at/above threshold) | 1.0 | autonomic, selection (`anticipatory_threat_need`) | 1 pass per cluster / max | a situation-cluster change whose learned fear clears θ → the Wire-4 READ moves from "next substrate tick" to "on cluster change"; `ExecAgent` urgent flag (**UNVERIFIED** live) |
| `extero_detail` | cortical route (V1→IT; auditory brainstem → MGN → A1→belt): detailed, slower | 1 | 30 | none | 1.0 | relay (EC / GL4, incl. the `audio` channel), prompt (gated) | 0 / latest | vision detections, DoA azimuth and audio percepts, Minecraft distance and spatial sensors (the per-sensor note above) → the ordered scan in `CompositePerceptSource` gains a priority; `ThalamicGate` becomes this track's prompt gate |
| `language` | language cortex route (heard or read words) | 1 | 20 | pending proposal **only** for `multi_step` / `fallback` strategies | 1.0 | prompt, relay (word world, under `MAXIM_SUBSTRATE_PATH`) | 0 / latest | CLI, narrator, transcript, and Minecraft game-log events (chat / damage / death: text, narrated or `reported` per TR11, with `caused_by` to the sensor event; bio-faithful N4) → the §1.5 CLI-preemption rule, **declared, not changed** |

**Fan-out example (one burn, after GL2b's `NociceptorSpec`).** `arms.thermal` crosses its noxious
threshold. The thermal receptor emits one `PhysicalEventId("aut", "<session>", 41)`, delivered on
`nociceptive_fast` (locus `arms`, delivered now, published onto PainBus at the transduction pass under the
nociceptor's failure mode, `drive:arms.thermal:noxious` as GL2b specifies it) and `affective_slow` (two
passes later, summed with any follow-on heat and clamped, feeding GL2's deviation/urgency and the
pid-keyed consequence record only). A `reflex` delivery (withdraw) exists only once
`reflex_layering.md` route 3 ships a withdrawal act (bio-faithful S1). The thermal value itself stays in the body channel
through the registry's EC route; it does not ride `mechano_proprio`. On PainBus, the memory and NAc
outcome subscribers see the burn **once**, at its transduction pass, subject to their learning
thresholds; the slow delivery writes no learning sink (TR18). **Wire-4 cluster fear does not
fire:** its allowlist (`decisions/nac.py::DEFAULT_CLUSTER_FEAR_FAILURE_MODES` = `drive:health`,
`drive:oxygen`) excludes the burn's failure mode, and no stage edits it (autonomic_layer.md §1.4: a hivemind wire
boundary). GL4 and credit join on `aut:41`, so "the sharp pain and its delayed copy are the same burn" is a
fact of the data, not an inference.

**A drowning breach under the flag (TR18).** `drive:oxygen` is `PainKind.DRIVE`
(`proprioception/pain.py::TISSUE_DAMAGE_DRIVES = {"drive:health"}`) **and** on the Wire-4 allowlist, and
Wire-4 books fear on `nac.active_clusters(agent_id)["world"]` at publish time. So its PainBus publish
keeps its in-place synchronous slot inside `propose_via_substrate`, in the W-4 order, flag on or off. The
frozen record's `drive:oxygen` publish at +0 passes is the cadence's phase, not a mechanism (confounding
N5), and Exp 60's water fear rides on it.

### 5.5 The deterministic `TrackScheduler`

Built once per agent in `runtime/loop_setup.py::build_loop_run` and carried on `LoopRun`. Once live, it
becomes a **required keyword** on `build_loop_run` (silent-no-op-into-types rule).

- `emit(receptor_id, *, locus, intensity, payload, provenance, caused_by=None) -> PhysicalEventId`: loop
  thread only. It assigns `seq` and enqueues one delivery per declared track with
  `due_pass = pass_index + spec.latency_passes`.
- **Seq authority handover (GL3.B3's stage gate).** Before GL3.B3, the per-agent `EventSequencer` built
  at the post-fence resume stage (G17; GL2a mints nothing) (held
  by the primary `Embodiment`, lock-guarded for out-of-band records minted off the loop thread) is the only
  seq authority. At GL3.B3, seq assignment moves to the drain point: `emit` draws from the same per-agent
  counter, which becomes the scheduler's backing store, so the persisted high-water and its
  resume-past-max-on-load rule (both from the post-fence resume stage, G15) carry over unchanged. One authority at a time: a guard test asserts that,
  with the scheduler live, the `Embodiment` never assigns a seq itself, and that seqs stay unique and
  monotonic across the handover and across a save/load.
- `edge_inbox.put(receptor_id, edge_local_seq, ...)`: **the only call legal from another thread.** It is a
  lock-guarded list. Under `--sim`, the orchestrator thread's narrator-tool transduction posts here (§3.2),
  as do the hardware edges. **Reachability (wiring S7):** the narrator tools (`orch_registry`) and the
  reflex instances (`aut_bio_enrichment_pipeline._reflex_damage_tool` / `_reflex_sensor_tool`) are built in
  `start_simulation_mode` against `_aut_embodiment` **before** `sim.aut` starts, while the scheduler is
  built later on `sim.aut` by `build_loop_run`. So they hold no reference to it. Recommended:
  `build_loop_run` attaches the inbox to the primary `Embodiment`'s slot (beside the registry
  `registry_for` caches there, TR21), and the
  tools post through their `_aut_embodiment`. The GL3.B3 PR names the route; either way it touches
  `orchestrator.py` and `integration/bio_enrichment.py`. The write itself stays on the orchestrator
  thread, stamped at write time (TR23, §3.2).
- `ingest_edges()`: loop thread, **once per pass**. It sorts the inbox by `(receptor_id, edge_local_seq)`,
  which removes any dependence on arrival interleaving, then calls `emit` for each entry.
- `drain(phase) -> list[tuple[AfferentTrackSpec, AfferentEvent]]`: every due delivery, ordered by
  `(due_pass, -priority, seq)`. That is a total order, so delivery is deterministic.
- `preempted_since(basis_seq, right) -> AfferentEvent | None`: has any event on a track holding that right,
  at or above its threshold, arrived after `basis_seq`?
- Refractory is counted in passes per `(agent, receptor, locus, track)`. A delivery inside the window is
  **coalesced (max) and delivered at the window's end, never dropped** (bio-faithful S8): the body latch
  already re-publishes on material deepening, and adaptation belongs at the producer
  (`adaptive_nociception.md`), never in the track. **The `pain_bus` destination is exempt from the
  deferral (pre-merge Executor S3, TR18):** two transductions of one locus in one pass (H3t in §4 and
  then H3 in the substrate tick, or a drift tick plus an AUT tool) coalesce **within that pass**: the
  second adds no publish unless it is deeper, and a deeper one publishes at that same pass (the latch's
  deepening rule); a PainBus publish is never moved to the window's end, because that would land it
  after the next `note_active_clusters`, the W-4 misfiling TR18 exists to prevent. A deferred delivery
  to any other destination keeps its transduction-pass situation key and is never a PainBus publish. **Wall time never touches a track
  decision.** The PainBus wall refractory (0.5 s per `(entity, failure_mode)`) still applies to non-track
  publishers until they migrate, and so does the ReactionBus refractory, which keys on
  `f"{kind}:{source}"` (`reactions/bus.py`); `reactions/compat.py::pain_signal_to_reaction` gives every
  body `PainSignal` the source `pain_detector:external_signal` (`embodiment/body.py` publishes every body
  pain as `PainType.EXTERNAL_SIGNAL`), so a drive Reaction and a nociceptor Reaction from one
  `evaluate_failures` call coalesce there within its refractory. The GL3.B3 PR must name every
  publisher's path (the §3.3 census), because the refractories coexist. (`preempted_since` is used only
  from GL3.B4.)

**Hook points in the pass:**

| Hook | Where | What |
|---|---|---|
| H0 `ingest_edges()` | in `pre_tick_gate`, right after `_loop_live_tick` (the slot the experience driver uses) | edge data (hardware edges and the orchestrator-thread declared edge) enters the run in a fixed order: **the edge inbox is drained exactly once per pass** |
| H1 percept → events | §1, `next_observation` | a sim pain percept becomes a `nociceptive_fast` / `affective_slow` emit and no longer waits in the FIFO; text becomes `language` |
| H2 fast drain + preemption check | **immediately before §4 executes `ctrl.pending_proposal`** | deliver due `reflex` / `nociceptive_fast` / `threat_low_road`; from GL3.B4 only: if `preempted_since(proposal.basis_seq, "pending_proposal")` → typed drop (§5.6) |
| H3 transduction | inside `propose_via_substrate`: `evaluate_failures` emits through the scheduler, latency-0 tracks are **drained in place**, and **every PainBus publish, NOCICEPTIVE and DRIVE alike, happens here, at the transduction pass** (TR18; `drive:oxygen` keeps its in-place synchronous publish) | **the Exp 58 W-4 order (encode → `note_active_clusters` → pain → Wire-4 fear write → re-read drives → `anticipatory_threat_need` → `recommend_action`) stays byte-identical**; `affective_slow` adds later deliveries only to non-PainBus sinks, so each learning sink sees a pid once, keyed on the situation at this pass |
| H3t tool-path transduction (TR24) | inside `embodiment/tool_bridge.py::ModulatorAffordanceTool.execute`'s `evaluate_failures` (the AUT's own affordance in §4) | latency-0 tracks are drained in place inside the executing call, so T1-4's synchronous tool-coupled pain is preserved and an experienced burn (`cradle_fire_pit.touch`) has a fast drain. The emitted event **reuses the invocation pid** minted at `Executor._run_started` (one pid, §5.2; pre-merge Wire B3) |
| H4 slow drain | §8.5, beside `_loop_bio_tick_maintenance` | deliver due `affective_slow`, `extero_detail` and the slow credit destinations |
| H5 wake source | the idle gate (`_loop_is_idle`) | a due event wakes the loop. **This fires T1-13/14/15's idle-gate trigger by its wording**, so it is not in GL3.B3/B4 (§6 GL3.B7) |

**Where each `evaluate_failures` caller drains (wiring S8, TR24).** The rule: **latency-0 deliveries are
drained in place at every caller**, so no "fast" delivery arrives later than today's synchronous publish
and Wire-4 keys stay on the situation of the transduction pass (deferring an `execute`-time pain to the
next H3 would land it after the next `note_active_clusters`, the Exp 58 W-4 misfiling).

| Caller | Thread | Latency-0 drain |
|---|---|---|
| `propose_via_substrate` (substrate-primary) | loop | in place (H3) |
| `ModulatorAffordanceTool.execute` (AUT tool in §4; `OrchestratorActorTool`'s ephemeral tools) | loop; orchestrator for the actor tools, which post to the inbox instead (§3.2) | in place (H3t) on the loop; at H0 for the posted ones |
| `loop_gates.py::tick_embodiment_drift` (llm-primary, inside `_loop_live_tick`) | loop | in place, inside the drift tick (llm-primary never calls `note_active_clusters`, so Wire-4 books nothing there either way; `docs/wiring/body-and-word-worlds.md`) |
| the reflex dispatch inside `enrich` | loop | in place |
| narrator tools (`SetEntitySensorTool`, `DamageComponentTool`) | orchestrator | posted to the inbox; drained in place at H0 |
| `simulation/foundry.py::run_gauntlet`, the Dormant `EmbodimentPerceptSource` | no agent / Dormant | none: no AUT, no scheduler |
| scripts that call `evaluate_failures()` directly (`scripts/survival_world/dark_danger_probe.py`, `scripts/check_oscillator_coldstart.py`; pre-merge Wire N7) | the script's own | none unless the script builds a scheduler; the GL3.B3 PR states each one's flag-off byte-identity (no scheduler means today's synchronous publish) |

**Determinism.** On the loop paths `seq` is assigned only on the loop thread: `emit` at the drain
point, and the invocation mint at `Executor._run_started`, which on every loop path runs on the loop
thread, under the sequencer's lock (§5.2). `ConcurrentExecutor`, whose `execute` runs on pool threads,
is declared outside this determinism claim (delta round, Executor S-F, Wire N-5). Edge entries are sorted at the one H0 point.
No uuid and no wall time enter any decision. `experience_us` is recorded for analysis and for GL4, and it
is never used to schedule. With the flag off, the existing goldens stay byte-identical. With the flag on,
a new arm and golden are pinned through `_loop_harness` (`_ScriptedWorld` is single-threaded, so H0 sees
no edges and the order is a pure function of the step clock). Real worlds stay non-deterministic at their
edges. Tracks make the post-edge order deterministic; they do not change the edges.

### 5.6 Preemption semantics (deterministic, never idle; GL3.B4 only)

Nothing in this subsection is built before GL3.B4, which is a declared 1.4 rung arm or waits for one
(owner decision G8).

- **What H2 reads, and who produces it (TR19; bio-faithful D2, confounding D1, environment D1).** In
  substrate-primary, `propose_via_substrate` is the only per-pass transducer, and it cannot run while a
  proposal is pending (`loop_state.py::_substrate_tick_due` requires `ctrl.pending_proposal is None`);
  `tick_embodiment_drift` is a no-op there; the sync pump writes but never transduces, and becomes an
  edge only at GL3.B8. In the frozen damage arm the stale `mine_block` is refused, so nothing transduces
  until the next substrate tick; in red gate (d)'s world the stale `eat` runs and transduces the breach
  itself, inside §4, after H2. So, built as first drafted, H2 never fires on the L1 case. **The event H2
  reads comes from per-pass body transduction, independent of selection, before H2**: a named GL3.B4
  prerequisite, settled with #1176 (§3.4) and landed **unflagged** there as the L1 defect fix, keeping
  the W-4 order. A nociceptor transduces when the
  stimulus arrives, not when the selector next asks for context.
- **On the rig the dominant term is the hold (environment D1).** A stale action that really runs is a
  blocking `MinecraftClient.call_action` (15 s timeout; `escape_water` holds up to 8 s), and the breach
  is transduced inside that call, after the hold. GL3.B4 does not cover held actions, so a rung arm that
  declares GL3.B4 must not expect a rig-visible latency change from it alone.
- **What fires (environment S3).** On `minecraft_player`, `health` has set point 20, `comfort_band` 6.0
  and `pain_scale` 0.5, so pain is near 0 at band entry and saturates only at 12 hp or below; with
  `preempt_threshold 1.0` preemption fires only on saturated pain. Typical game hazards step down a few hp
  at a time (game knowledge, **UNVERIFIED** here). The declaring rung's prereg names a game-native hazard
  and its damage time course; an RCON `/damage` reproduction is `apparatus` (G16) and never credited.

- **Basis stamp.** Every proposal carries `basis_seq`, the scheduler's high-water seq when its context was
  built: in `propose_via_substrate` for substrate proposals, and at submit time in §6 for LLM proposals
  (`LLMProposal` gains a defaulted field; it is runtime-ephemeral).
- **Pending proposal.** At H2: drop with reason `preempted_by_track:<track>:<pid>`, logged as a sim EXEC
  line plus a structured event. Then, **in the same pass**, substrate-primary re-runs selection (so the
  protective action happens this pass), and llm-primary re-arms submission with the preempting event in
  `auto_sense_context`. **Named for the gate design pass:** the same-pass reselect re-runs
  `propose_via_substrate`, which calls `nac.note_active_clusters` and runs the Wire-4 fear write again, so
  one pass would note active clusters twice and could write fear twice. The design pass must decide
  whether the reselect re-enters after the transduction step (selection only) or re-runs the whole
  function, and pin the answer with a count test.
- **A preemption drop is not a planning failure.** It does not count against the D13 budget, and it never
  falls through to idle. The D13 invariant (`docs/agents/runtime-tools.md`) requires every drop site to
  re-arm, so this drop gets its own typed handler next to `_handle_planning_failure`.
- **In-flight plan.** The HTTP call is never aborted (WorkerPool cannot). When the job's proposal is
  consumed in §2, the same `basis_seq` check routes it through the same handler.
- **Held action.** Only if a backend declares `interruptible`; none does. Not in GL3.B4. The facts
  (environment S9): the Minecraft bridge's `stop` action clears pathfinder goals only
  (`bot.pathfinder.setGoal(null)`) and does not cancel `dig`, `consume` or the `escape_water` surface loop;
  bridge actions run as concurrent promises, so a mid-hold `stop` could be delivered, but
  `MinecraftClient.call_action` blocks the loop thread until `action_result`, so the loop cannot send one.
  The Reachy controllers (`hardware/reachy/controller.py`, `hardware/simulation/controller.py`) have
  `goto_target` and no cancel (GL3.B8).
- **Livelock guard (invariant).** One **breach episode** preempts **at most once**, and refractory applies
  per locus. A pid is the wrong unit (confounding S3, environment S3): `evaluate_failures` re-publishes
  whenever a breach deepens (`severity > latched + eps`, `_BREACH_DEEPEN_FRACTION = 0.05`), each
  re-publish is a new pid, and on a deepening stream (drowning, whole-hp game damage) "once per pid"
  bounds nothing, while a truly standing breach publishes once and makes a per-pid probe vacuous. Strict
  default: a proposal produced by a preemption reselect cannot itself be preempted within the same
  episode, so the protective action executes.
- **Tool-coupled pain stays synchronous.** `ToolPainBridge`'s direct attribution inside `execute` is not
  deferred by any track (T1-4). Tracks add a timed delivery beside it; they never replace it.

### 5.7 Provenance on the EC / NAc write path (GL3.B1)

Rule (`[engineering]`, enforced in the type): **every EC write carries a provenance kind, recorded
always; with the credit flag ON, `experienced` creates NAc eligibility at full weight, `narrated` only at
the declared discount (owner decision G6), and `imagined` creates none (strict default, TR12).** A
narrated record can never be relabelled experienced.

**Record always, act only behind the flag (TR20; confounding D3, wiring D3).** GL3.B1 records provenance
on every write, flag on or off. **Acting** on it, the imagined eligibility refusal (item 3), the narrated
discount (item 3) and the credit check (item 4), happens only behind the one credit flag, default OFF.
With the flag off, eligibility and credit are today's for every kind. Two reasons. The narrated weight
does not exist at GL3.B1 (G6 sets it at GL4 start), and the body channel's pull read goes through
`SensorEncoder.encode_sensors`, which itself calls `update_eligibility`, so an unflagged change would move
eligibility in every `--sim` run with the narrator active. And an unflagged imagined refusal is not inert:
`decisions/temporal_credit.py::TemporalCreditDistributor.distribute` normalizes "so total credit =
reward" across the eligible nodes, and `runtime/bio_stack.py`'s `_distribute_reward_from_reaction` calls it
on every Reaction, so removing imagined nodes re-allocates the same reward onto the agent's other nodes
in every bio-stack run with an `ImaginationTrigger`; `_reward_bias` lowers that node's EC threshold, so
cluster identity and enrichment can move. None of the cited goldens runs that path (the selection golden
builds no imagination trigger and no narrator; the GL3.B1 gate-1 tuple carries no eligibility), so GL3.B1
adds an **eligibility golden** that runs the narrator and imagination paths (§6 GL3.B1 gate 4). Red gate
(a)'s read surface changes accordingly: it flips under the flag-on arm, which is the owner-approved
re-point GL3.B0's banner requires (TR20). **Named for the GL3.B1 gate design (wiring D3):** whether a
refused node's strength is dropped **before** `distribute`'s normalization (which re-allocates reward to
the remaining nodes) or **after** it (total credit falls below `reward`, the other nodes' credit
unchanged). They are different behaviours; the strict recommendation is after, so a refusal never raises
another node's credit. The owner picks with TR12 / TR13.

**Provenance kinds are oracle labels (bio-faithful S3).** The producer stamps provenance as ground
truth; the agent never infers it. Biology has reality monitoring, an *inferred* source judgement that
fails in known ways (Johnson & Raye 1981), and it does learn from imagination (imagined extinction,
mental practice, replay). So no GL5 or later claim may cite provenance kinds as learned source
monitoring. A learned `imagined`-vs-`experienced` discriminator scored against the oracle would be a
separate mechanism with its own trigger.

1. `AfferentEvent.provenance` is required, and the registry passes it to the EC route. Records minted by
   the narrator's tools (`SetEntitySensorTool`, `DamageComponentTool`, `OrchestratorActorTool`, and the
   reflex dispatch that goes through them) carry `narrated` (G6). A pull receptor's later read of a sensor
   the narrator wrote is a mixed state. **The mechanism (TR22, TR23):** a per-sensor last-writer
   provenance map that lives **beside `Entity.vital_metrics`** (and beside a `SpecModulator`'s own
   `vital_metrics`), written at write time by every body writer, each taking a keyword-only `kind=`, so
   forgetting it is a `TypeError`. **Why the entity, not the `Embodiment` (pre-merge Wire B1):** the
   writers hold only an `Entity` (`tool_bridge.py::_apply_sensor_deltas(body: Entity, …)`,
   `::_write_sensor(body: Entity, …)`, `embodiment/spec.py::SpecModulator.apply_damage`), and
   `OrchestratorActorTool`'s ephemeral `ModulatorAffordanceTool(scene_entity, …, embodiment=scene_emb)`
   writes the AUT through its target_effect branch while holding a scene `Embodiment`. A map on the
   `Embodiment` would be stamped on `scene_emb`, or not at all, and the AUT's next pull read would encode
   a narrated value as experienced (G6). The map sits where the values live, so the write and its kind
   cannot part. **The reader resolves the slot through the same resolver as the value (delta round,
   Wire D-3):** `embodiment/sem.py::_resolve_sensor_slot`, so a qualified sensor such as `arms.thermal`
   (the burn's own sensor, which lives on the root's modulator) reads that modulator's map, and an
   unqualified one the root's. A kind is never looked up by a second rule that could disagree with where
   the value was read. **Persistence (Wire D-3, Architecture SF-2):** the map is serialized beside
   `vital_metrics` by `Entity.to_dict` and restored by `Entity.from_dict` (so `Entity.save` / `Entity.load`
   and the public `maxim/load.py` carry it), with an `ENTITY_FORMAT_VERSION` bump and a
   `tests/integration/test_persistence_compat.py` case: a pre-bump file loads with no map, which under the
   missing-slot rule below reads `legacy` for every slot, never `experienced`.
   - **The writer kind is a constructor field of `ModulatorAffordanceTool`** (keyword-only), threaded into
     both the self_effect and target_effect `_apply_sensor_deltas` calls: the AUT's own tools pass
     `experienced`, and `OrchestratorActorTool`'s ephemeral tool passes `narrated`, so its write to the
     AUT's entity is stamped `narrated` on the AUT's entity whatever `Embodiment` the tool holds. The
     constructors that gain the keyword (completed in the delta round, Wire N-3):
     `tool_bridge.py::generate_tools_for_entity` and, through it, `simulation/foundry.py::run_gauntlet`
     (`apparatus`: a harness trial of a candidate body with no AUT), `simulation/generative_runner.py`
     (`imagined` for an `_imagine_entity_from_ref` entity, whose `metadata={"imagined": True}` says so;
     `narrated` for an arc or registry scene entity), `simulation/dm_runtime.py` (`narrated`, the DM's
     scene), `scripts/orient_backbone/exp53_cross_context_readout.py` (`experienced`; T1-10 evidence, so
     its row's walk fires) and `scripts/r2_drive_premise_probe.py` (`experienced`);
     `simulation/tools.py::OrchestratorActorTool`'s ephemeral tool; the 8 `scripts/orient_substrate/*`
     probes that construct `ModulatorAffordanceTool(` directly (2, 3, 4, 5, 6, 7, `9_hunger_relief_orient`,
     `9_tool_competition_repro`; each passes `experienced`); and the test files that construct either
     (20 at `d9dffe6e`: 12 construct `ModulatorAffordanceTool(` directly, 10 call
     `generate_tools_for_entity(`, some both), each edited in the GL3.B1 PR.
   - **Imagined-entity tools train the Cerebellum after GL4 S1 (delta round, Wire N-2).** Once S1 routes
     the training gain sites, an imagined entity's tools (`generative_runner`'s `_imagine_entity_from_ref`)
     train the Cerebellum like any other tool; their writes carry `imagined`, so GL4's training read
     sees the kind and the Q3-style question (may imagined outcomes train the forward model?) is
     latent_forward_model.md's to answer at S1, not decided here.
   - **The writers (complete census, pre-merge Wire SF-5):** `audio_localization.py::world_set_axis`
     (its `owner=` is a log tag today; the kind is stored) and its callers `embodiment/backends/minecraft.py`
     (the pump, `experienced`), `hardware/reachy/motor_backend.py` ×2 (`experienced`) and
     `scripts/orient_backbone/exp53_cross_context_readout.py`; `world_set_azimuth`, a delegate that takes
     its own `kind=` so it cannot bypass the guard, with callers `audio_localization.py::DoAFeed`,
     `runtime/loop_perception.py` ×2 (fenced; the kind comes from the percept's own provenance, TR11's
     per-percept kind table, not a fixed `experienced`: on `--sim` the sound is the narrator's, so the
     write is `narrated`, and the reflex `_oriented` write, which the code marks "SIM-ONLY … fabricated
     measurement", takes the same percept's kind; delta round, Wire D-2) and `simulation/cradle_mother.py` ×2
     (`apparatus`); `_apply_sensor_deltas`, with callers `ModulatorAffordanceTool.execute` (both
     branches, the constructor kind above), `simulation/cradle_mother.py` (`apparatus`) and
     `scripts/exp56/common.py`; `_write_sensor`, with callers `_apply_sensor_deltas` and
     `simulation/tools.py::SetEntitySensorTool` ×2 (`narrated`); `spec.py::SpecModulator.apply_damage`
     (it lives in `embodiment/spec.py`, not `body.py`), called by `simulation/tools.py::DamageComponentTool`
     (`narrated`); `simulation/dm_runtime.py::_write_sensor` (the DM cascade, `narrated`). A scene-only
     write (`simulation/generative_runner.py`'s `entity.vital_metrics["presence"] = 1.0` on a scene
     entity) goes through the same keyword, as `narrated` for a narrated scene entity and `imagined` for
     an `_imagine_entity_from_ref` entity (delta round, Wire N-2); no AUT receptor reads a scene entity.
   - **Raw `vital_metrics[...] =` writes and the one setter (delta round, Wire D-2).** A keyword-only
     `kind=` catches only writers that are called; a raw dict write calls none and silently keeps the
     slot's previous kind. The raw sites at `d9dffe6e`: in `src/`, `embodiment/sem.py::Entity.from_dict`
     (the restore of modulator values), `simulation/foundry.py` (×2) and the Dormant
     `embodiment/backends/cerebellum_modulator.py` (predictions written into `vital_metrics`, an
     `imagined` writer); in `scripts/`, the `scripts/orient_substrate/*` stimulus placements
     (`body.vital_metrics["azimuth"] = az` in 2, 3, 4, 5, 6, 7, 9_hunger, 9_tool; hunger writes in 3 and
     9_hunger), `scripts/survival_world/r2_learned_bias.py`, `scripts/r2_drive_premise_probe.py` and
     `scripts/check_oscillator_coldstart.py` (×2). **Rule:** GL3.B1 routes every body write through
     **one setter that takes `kind=`** (the named writers above become its callers), and an AST lint,
     `scripts/lint_vital_metrics_writes.py` (mechanization backlog M78), forbids a raw
     `vital_metrics[...] =` (and `.update(`) outside the setter in `src/`; `scripts/` sites are listed as
     a census the lint prints, each routed through the setter or stated harness-only (`apparatus`) in
     the GL3.B1 PR. A restore (`Entity.from_dict`) writes values **with their persisted kinds**, not as
     a new writer.
   - **Derived values inherit their source's kind (pre-merge Wire SF-5).** A value `evaluate_failures`
     derives (`Embodiment.evaluate_failures` writes `vital_metrics["health"]` from component integrity)
     is written with the kind of the source write it derives from; with several sources, their union, so
     under the mixed-state rule a narrated `DamageComponentTool` hit makes the derived `health` read
     `narrated`. It is never written with whatever kind the slot held before.
   - **A slot with no writer yet (pre-merge Executor S1; owner decision TR29, DECISIONS.md G29).** No
     provenance field has a usable default, so a slot with no writer yet is never `experienced`. There
     is **no `unknown` kind**: a **missing** writer-kind slot means "not yet written", and the reader
     (above) maps it to **`legacy`**, credited as today (TR13), so the rule changes no earned row's
     credit and the encoder, the EC node set and its per-kind counts see only kinds the vocabulary
     already has. This holds for every body, whatever built it (`ComponentRegistry().instantiate`,
     `spec._parse_entity`, `_imagine_entity_from_ref`, `Entity.from_dict` of a pre-bump file, a
     hand-built test entity): no builder has to write an initial kind, and nothing can default to
     `experienced`. The cost, stated: on a cradle body, a drive that only drifts from its YAML initial
     value (experienced hunger) reads `legacy` until one of the agent's own writers touches it (drift
     keeps the slot's kind, below, and a missing slot stays missing); GL3.B1 gate 4 reports that share
     beside the `narrated` share. Under the mixed-state rule a missing slot never outranks a written
     one: a read is `narrated` if any declared sensor's last writer was the narrator, otherwise it
     carries the union of the written kinds, plus `legacy` for the missing slots. Control (g′) is
     restated against this: a drift-only encode on a fresh body carries `legacy` and never `narrated`
     or `experienced`.
   The map lands at GL3.B1, its first reader; GL3.B3's `evaluate_failures` callers read it too (§3.2).
   **Drift keeps the slot's kind** (wiring D2,
   confounding S5 (i)): `tick_vital_drift` is not a writer of its own, so a drift step on a narrated value
   stays `narrated`; otherwise one tick of drift would relabel a narrated value experienced, which G6
   forbids. **The rule over the slots** (strict default, kept; bio-faithful S4 (a)): a channel read is
   `narrated` whenever any declared sensor's last writer was the narrator. Its reach is wide: the
   interoception channel sums every drive, so one narrator write to `arms.thermal` stamps every later
   interoception encode `narrated` until that sensor is written again, including experienced hunger
   drift. GL3.B1 therefore reports the share of interoception-channel encodes stamped `narrated` on a
   cradle arc trace (§6 GL3.B1 gate 4). The bio-faithful alternative (b), stamping the kinds of the
   sensors that **moved** since the prior read, is the owner's pick at GL3.B1 start, and choosing it is
   the owner-approved change to red gate (g)'s read surface.
   **`apparatus` gets a named producer (TR22; wiring D2, environment S6).** In Minecraft every world write
   has one owner, the pump (`world_set_axis(owner="minecraft_bridge")`), so the water trial's rescue
   (`WaterTrial.rescue`: an RCON teleport, a settle, `heal()`), the `/effect` heal and the respawn come
   back as bridge truth, and a last-writer rule cannot see them. The producer is the harness's
   rescue/heal **window** (`autonomic_layer.md`: "the apparatus scope is a window, not a thread-local"),
   opened and closed by the harness around `rescue` / `heal` / respawn, and consumed by the mixed-state
   rule. Strict: every pull read inside the window, and until the first snapshot taken after it closes,
   is stamped `apparatus`. The cradle mother's writes carry `apparatus` through the slot.
   **The window is an edit to `WaterTrial` (pre-merge Executor S7, Wire SF-7).** `water_trial.py` is the
   instrument of T1-13, T1-14, T1-15, R3 and B1 (and a cited guard of the Exp 60/61/62/R3 rows,
   latent_forward_model.md S0b), so GL3.B1's blast radius walks those rows on their "harness" wording
   (§6 GL3.B1), and the GL3.B1 PR proves `scripts/survival_world/scripted_water.py` and
   `tests/unit/test_water_trial_smoke.py::test_water_trial_ticks_acts_and_the_staging_close_persists_fear`
   byte-identical with the window present and the credit flag off (the window only stamps; it changes no
   value while nothing acts on the kind).
   **GL4 S5's harness resets (delta round, Executor N-3; latent_forward_model.md §7 S5).** A harness
   reset write (S5's forced-schedule driver restoring the trial's pre-state) goes through the setter with
   `kind="apparatus"`, so the slot holds `apparatus`, and the reset is inside the apparatus window by the
   same rule as a rescue: every pull read inside it, and until the first snapshot after it closes, is
   stamped `apparatus`; after that a post-reset situation encode reads `apparatus` for each reset
   sensor until one of the AUT's own writers touches it. S5 therefore depends on GL3.B1 (the map, the
   setter and this window).
   **The cradle mother's teaching credit (pre-merge Wire SF-7).** `simulation/cradle_mother.py::reactive_mother_tick`
   → `NAc.credit_operant_reward` is Exp 52's mechanism (T1-9). Its body writes (the azimuth stimulus
   through `world_set_azimuth`, the feeds) are `apparatus`, and the node it credits is the cluster of
   the infant's own recent action, whose kinds come from the reads that formed it; where a mother write
   reached that read (the audio channel after a stimulus), the node carries `apparatus` too, so a
   flipped check that refused `apparatus` could zero Exp 52's own teaching signal. Strict, and not a
   silent zero: the mother's credit is **recorded** with its `apparatus` source, flag on or off; with the
   flag off it credits as today; whether an `apparatus`-sourced reward, or a node carrying `apparatus`,
   is refused once the flag flips is decided by the T1-9 walk at the flip (TR13's re-run list), never by
   this plan's default.
2. EC gains a per-node provenance set, a parallel dict like `_substrate_node_sources`, recording which kinds
   have ever *reinforced* the node (a node first imagined and later experienced shows both). It must enter
   **every** EC persistence and exchange path, or it is silently dropped:
   - `similarity/ec.py::EntorhinalCortex.save` / `load` (`with_format_version` + `atomic_write_json`).
     Files written before GL3.B1 load with an explicit `legacy` marker and one warning: never relabelled
     `experienced` (no type and no loader defaults provenance to `experienced`). How the credit check
     treats `legacy` nodes is part of TR13's flip (strict default: credit as today, because every earned
     row's nodes predate the stamp and nothing reads the set while the flag is OFF).
   - `hivemind/bundle.py::_BUNDLE_EC_NODE_FIELDS`, which today keeps only `embedding`, `modality`,
     `count`, `member_count`, `geometry`, `domain`, `source`, `contributors`: a field not listed there is
     scrubbed from every exported node. Adding it may need a bundle format version bump (T1-14 wording).
   - `hivemind/merge.py::ec_merge_aligned`'s node fold: the union (OR) of both sides' sets, so a narrated
     or imagined kind can never be lost or overwritten by `experienced` in a merge.
   - `hivemind/ingest.py`'s admission pass over a foreign `ec.json` nodes slice: validate the field
     (known kinds only, bounded size), refuse otherwise.
   - `hivemind/entry_index.py` (wiring S4): the Queen-signed v2 per-entry digest projects
     `EC_NODE_FIELDS = ("modality", "embedding", "geometry", "count", "domain")`. A field outside the
     projection is unsigned (changeable in transit), and "equal digests mean equal CONTENT" would let a
     per-entry dedup treat a narrated node and an experienced node as one. In or out of the projection is
     an owner call at GL3.B1 start (TR26; strict: inside, with an `INDEX_VERSION` bump).
   - The readers: `similarity/ec.py::EntorhinalCortex.substrate_node_metadata` (merge.py's documented
     input shape, and red gate (g)'s read surface, which today omits the field), `analysis/substrate_diff.py`
     and the staged-donor readers in `scripts/survival_world/exp61_run.py`.
   - **The EC lifecycle sites (pre-merge Wire SF-6).** `EntorhinalCortex.ingest_substrate_nodes` (used by
     `scripts/exp57/common57.py`, T1-12's receiver, and `scripts/r1_cross_layout_probe.py`): a foreign
     node enters with its source's validated set, or `legacy` when the source carries none, never empty
     and never `experienced` by default. `EntorhinalCortex.remove_substrate_node` (the invalidate path)
     pops the provenance set and its counts with `_substrate_node_sources`, so a reused id inherits no
     stale kind. `EntorhinalCortex._reset_store_state`'s attribute list names the new dicts, so they do
     not survive a reset (the #939 class). Gate 2's round trip covers all three.
   - **One name, its namespace stated (pre-merge Wire N3).** The node key stays `provenance`, because it
     is red gate (g)'s built read surface (`meta.get("provenance")`) and renaming it would be a re-point
     of a strict red gate. Its meaning is fixed here: **the percept-source kind** (experienced / narrated
     / imagined / apparatus / legacy; no `unknown`, TR29). It is not hivemind's contributor-origin export provenance
     (`hivemind/bundle.py::provenance_for_export`, the node's `source` and `contributors`), not
     `memory/semantic_types.py`'s `provenance`, and not M1's run provenance stamps. The EC docstring and
     the bundle code comment state the distinction in the GL3.B1 PR.
   - **Counts, not only membership (confounding S5 (ii), strict option).** Narrator writes and drift
     reinforce the same frozen-centroid clusters, so most `--sim` nodes would become
     `{experienced, narrated}` and get full credit, and the G6 discount would almost never apply. The node
     therefore keeps a per-kind reinforcement count beside the set, and the flagged credit rule credits a
     mixed node's creditable share by those counts. GL3.B1 gates a mixed node, and the fraction of mixed
     nodes on a recorded `--sim` trace is reported before the credit flag can flip. **Counts stay
     consistent with `count` (pre-merge Wire SF-6):** the `min_delta` short-circuit in `encode_sensors`
     (item 3) adds the kind to the set but bumps no per-kind count, as it bumps no `count`, so the
     per-kind counts always sum to the node's `count`; gate 2 asserts the sum.
3. `LinguisticEncoder.encode*` **and `SensorEncoder.encode_sensors`** take `provenance=` **keyword-only
   with no default**, so forgetting it is a `TypeError` (TR22, wiring D2: the narrated writes red gate (g)
   is about reach EC through `SensorEncoder.encode_sensors`, which itself calls `update_eligibility`).
   `encode_sensors` records the kind even on its `min_delta` short-circuit, which returns the previous
   node id without touching EC. With the credit flag ON, `update_eligibility` is called at full weight
   for `experienced`, at the declared discount for `narrated`, and not at all for `imagined`; with it OFF,
   every kind keeps today's weight (TR20). `imagined` and `narrated` still reach EC and ATL
   (recognition, and the innate word prior that GL4 treats as the prior tier). The callers the
   keyword-only change reaches (wiring S3): `LinguisticEncoder.encode` (which calls `encode_decomposed`
   internally), `integration/memory_hub.py::MemoryHub.on_percept_received` (the EC route under
   `MAXIM_SUBSTRATE_PATH=1`, so `tests/integration/test_memory_hub.py` runs), `scripts/grounding_census.py`
   and `scripts/diagnose_roy_paraphrase_collapse.py`. **Every direct `encode_sensors(` caller, by
   directory (pre-merge Executor S5, Wire SF-4; grep `encode_sensors(` at `d9dffe6e`):** keyword-only,
   no default, strict, so every one is edited in the GL3.B1 PR, none is left to break silently, and none
   is recorded as accepted-broken.
   - `src/`: `runtime/substrate_proposal.py` (2 sites, moved behind the registry), the only `src/` call
     sites (the `similarity/encoder.py` class docstring's usage example is updated too).
   - `scripts/` (16 files; most are `experienced` probes, each gets an explicit kind): `exp56/common.py`
     (T1-11), `exp57/common57.py` (T1-12), `survival_world/l11_geometry_probe.py` (part of T1-13's
     regression guard), `l11_real_trace_remeasure.py`, `paired_data_audit.py`, `r1_cross_layout_probe.py`,
     `rsc_precheck.py`, and `orient_substrate/{1,2,3,4,5,6,7,9_hunger_relief_orient,9_tool_competition_repro}`
     (9 files, the T1-6 / T1-7 / T1-9 lineage). Scripts are not run by CI, so the GL3.B1 PR runs each
     touched evidence script's offline / dry-run guard where one exists (TR3), and states that a keyword
     with today's weight (flag off) changes no number.
   - `tests/` (18 files): edited in the same PR.
   The GL3.B0 census's `sensor_encode` category is extended to `scripts/` in the GL3.B1 PR, so a new
   direct caller is counted. MemoryHub's text percepts need a kind at
   GL3.B1, before the text receptors exist (GL3.B5), so TR11 comes with a **per-percept kind table**
   (narrator, DM, CLI human text, `[minecraft:damage]`, vision-as-text) at GL3.B1 start.
4. Credit readers check the node's provenance, **behind the OFF credit flag**: they refuse an EC node
   whose set holds no creditable kind (`experienced`, `narrated`, or `legacy` as TR13 rules), and cap a
   narrated share at the discount. This second, independent lock keeps a stale eligibility from an
   older save from carrying full credit. **The credit writers (TR22; wiring D2).** The body world does
   not learn through `credit_node` alone, so the check covers every writer that credits on the body path:

   | Credit writer | What it writes | Covered by |
   |---|---|---|
   | `NAc.credit_node` | the recognition `_reward_bias` | GL3.B1's flagged check |
   | `TemporalCreditDistributor.distribute` | the eligible-set fan-out into `_reward_bias` | GL3.B1's flagged check (EC node ids only) |
   | `runtime/tool_dispatch.py::record_outcome` | `NAc._cluster_reward_bias`, keyed on the proposal's `clusters` (EC node ids from the reading) | GL3.B1's flagged check, reading the clusters' provenance |
   | the PainBus subscribers (`create_pain_nac_subscriber` → `record_outcome_full`, `create_percept_valence_subscriber`, `create_pain_cluster_fear_subscriber` → `NAc.record_cluster_fear`, the pain-memory subscriber) | outcome, Wire-2 valence, Wire-4 cluster fear, the memory tag | GL3.B1's flagged check: the publish carries the kind (the GL3.B1 PR names the carrier; `PainSignal` has no provenance field today, and `DamageComponentTool` publishes one directly) |
   | `NAc.credit_operant_reward` (named by G16; callers include `simulation/cradle_mother.py::reactive_mother_tick`, Exp 52's teacher, T1-9) | operant reward | GL3.B1's flagged check; the mother's `apparatus`-sourced reward is recorded, and its fate at the flip is the T1-9 walk's (item 1) |

   The GL3.B1 interim, with the flag off: every writer credits as today, narrated included at full weight,
   until the G6 value exists. **Scope: EC node ids only.** Eligibility keys that are not EC
   nodes are out of scope and pass unchanged; the load-bearing case is `tool:<name>`, which
   `bridges/tool_pain_bridge.py::ToolPainBridge` emits as a `TemporalEvent` →
   `decisions/temporal_credit.py::TemporalCreditDistributor.record_event` → `NAc.update_eligibility`
   (refusing it would kill tool credit, T1-4). **The check ships behind a flag that defaults OFF.**
   Flipping it is a separate, owned decision with its own re-runs (§7), and it cannot flip before the G6
   discount value exists (decided at GL4 start).

The contamination guard (latent_forward_model.md's CI test, and this plan's `tests/unit/test_ec_provenance.py`)
checks both halves of G6: a narrated record can never be relabelled `experienced` (through the registry,
a save/load, a merge or an ingest), and the discount is applied wherever narrated data trains or credits.

This tags provenance **at write time**. The end-of-session retroactive CausalLink tagging in the
orchestrator stays as a belt (it is a no-op when the tag is already set) until a deletion probe proves it
redundant.

**Kinds and policy still open** (§11): whether the agent's own-body affordance names (YAML self-knowledge,
neither experienced nor imagined) get a `declared` kind; whether game-log text about a real event
(`[minecraft:damage]`) is `narrated` with `caused_by` or a fourth kind `reported`; and whether imagined
nodes are refused credit (invariant, the strict default) or credited at a discount (an innate prior, like
`FOREIGN_FEAR_DISCOUNT` and the 0.5 imagined-link decay). Narrated is decided (G6: discounted, value at
GL4 start). The namespace alternative (a separate `"text.imagined"` EC modality) is simpler, but it stops
an imagined name from completing onto an experienced node, which is exactly what GL5 needs to measure. The
provenance dict is recommended.

### 5.8 What it subsumes, and what it leaves alone

| Existing piece | Fate |
|---|---|
| `_SUBSTRATE_CHANNELS` + `ModalityChannel` | **The GL3.B1 core.** Three registered receptors in today's code order (`body.drives` for interoception, then `audio.doa` for audio, then `world.state` for world, matching §5.3) whose readers *are* today's functions. `ModalityChannel` survives as the `sensor_sum` adapter. `_encode_current_clusters` and `propose_via_substrate` call `registry.tick`, resolving the registry through `registry_for(executor.embodiment)` (TR21; derived from the body YAML and cached on first use), so `_attach_live_situation` and every script caller in §5.3 are reached with no new keyword and no edit. |
| `CompositePerceptSource` | Stays as an **edge** multiplexer. Priority lives in the scheduler, which is the coordinator the design pass said N ≥ 2 would earn. Its single-injection-implementer error becomes routing addressed by `receptor_id` (GL3.B5). |
| `DoAFeed`'s hand-built two lanes | One push receptor `reachy.doa` on `extero_detail` (auditory; both lanes, the `audio` EC channel and the prompt fold, carry one pid), with its orienting response on `reflex`; never `mechano_proprio` (§5.4 bio note). `world_set_azimuth` stays the body-write side effect (one writer per sensor). Rig-only stage (GL3.B8), T1-7 / T1-10 apply. |
| **Reflex timing** (`ReflexRegistry`) | Kept: `ReflexFiring.outcome` and its maths, which scale the damage the world inflicts (bio-faithful S1). Its keyword reflexes are a **narrated nociceptive receptor** (§5.4 note), not `reflex`-track motor programs. L5 itself (narrated injury transduction behind ThoughtGate) is a defect issue fixed before any track work, without the registry (§3.4). GL3.B6 is a **revive trigger of `deferred/reflex_layering.md`** (which already lists GL3's two-track receptor), not a re-housing of the damage-dealing maths on the `reflex` track. `enrich` reads the firing results for its latent affordances. §1.16's orienting reflex, a motor act, moves to the `reflex` track; the sim-only rule stays (head-frame lesson). |
| **PainBus** / `build_pain_bus` | **Kept as transport** and subscriber registry: the `pain_bus` destination of `nociceptive_fast` (NOCICEPTIVE), and the in-place publish of every DRIVE breach at its transduction pass (TR18). `affective_slow` has no `pain_bus` destination. Track-delivered publishes bypass its wall refractory, which the pass refractory replaces. |
| `ReactionBus` | Kept (typed surface). The `inject_pain` → `reaction_bus` shortcut becomes an H1 emit. |
| **CLI preemption rule** (§1.5) | Declared as the `language` track's preemption right, unchanged in behaviour (characterized, GL3.B7). |
| **`ThalamicGate`** | Unchanged DN nucleus. **Later** (GL3.B7) it becomes the prompt gate of `extero_detail` / `language`. The registry consults it only for `exteroceptive` / `vision` on the LLM route. |
| `BioEnrichmentPipeline` / `ExecAgent._run_pre_deliberation` | Unchanged relay-to-PFC nucleus. The call-site redirect named in design pass DP1 comes last (it is a call site, not a component). |
| Imagination and own-body affordance encodes | Two receptors: `imagination.affordance` (`provenance_kinds={"imagined"}`) and `body.affordance_names` (kind per §11) (GL3.B1). |
| `MinecraftPerceptSource`, narrator, CLI | Text receptors with `provenance_kinds={"narrated"}` (or `reported`, §11) and a non-empty agent (GL3.B5). |
| `TemporalEvent` | Kept as the credit envelope. It carries `str(pid)` in `context`, so its frozen shape does not change. The uuid its producers mint stays a log id only. |
| `ExperienceClock` | Ridden on, not replaced. |
| DefaultNetwork 30 Hz thread | **Left alone.** It is a hardware-edge reflex loop on the robot. Under G4, `deferred/hybrid_substrate_reflex_runtime.md` keeps its robot trigger. "DN consumes the `reflex` track through an outbox" is recorded there as a note, not as a trigger change. |
| `embodiment/percepts.py::EmbodimentPerceptSource` (Dormant since 2026-07-14) | **Stays Dormant.** The registry makes it unnecessary. It is not resurrected. |
| `SensoryTag.perceived_intensity` / `modulated_by` (no producer) | `relay_gain` is the real per-route gain. Owner decision: mark the field Dormant/dead, or wire it from `relay_gain`. Not a stage gate. |
| **`runtime/preemption.py::PreemptionCircuit`, `ExecutionTracker`, `wire_preemption`, the `check_hold` branch, the `capture_before` guard** | **Dead (never ran); defect L7, [#1179](https://github.com/dennys246/Maxim/issues/1179), closed: the Dormant docstring is in place.** Owner decision at GL3.B0 (TR1, Dormant). **Recommended then, and done (strict, dormancy over deletion):** mark them `Dormant since 2026-10-07` in the module docstring; callers and re-exports stay; `AfferentTrackSpec` borrows `SourceConfig`'s vocabulary without importing it. Non-strict alternative: delete them, which removes the `PreemptionCircuit` / `ExecutionTracker` re-exports from `maxim.runtime` (a CHANGELOG line). The scheduler lives in `perception/`, not in this module. |

### 5.9 Behaviour tiers

| Behaviour | Tier |
|---|---|
| Registration validation (unique id, registered tag, non-empty agent, provenance within declared kinds); one EC writer per receptor; YAML-derived specs | **invariant** (`[engineering]`) |
| The scheduler's total order; one `PhysicalEventId` across the fan-out; edge inbox drained once per pass; no wall time in track decisions | **invariant** |
| A preemption drop re-arms and never idles; at most one preemption per breach episode (§5.6) | **invariant** |
| The Exp 58 W-4 order inside `propose_via_substrate` | **invariant** (already pinned by the selection golden) |
| Provenance on every EC write; a narrated record is never relabelled experienced (GL3.B1) | **invariant** |
| `imagined` creates no eligibility (GL3.B1, behind the credit flag, TR20) | **engineering prior** (a contamination safeguard over an oracle source label; bio-faithful S3). TR12's strict *refuse* stays the recommendation; only the tier changes |
| `narrated` eligibility and credit at the declared discount (G6; value at GL4 start); any imagined discount, if TR12 ever chose one | **engineering prior** (experimenter-side provenance, G6; bio-faithful S3) |
| Each learning sink sees a pid once, at its transduction pass (TR18); a track adds no adaptation of its own (refractory coalesces, never drops) | **invariant** |
| The credit check on EC nodes (behind its flag) | **invariant** once flipped; non-EC keys (`tool:<name>`) out of scope |
| Every `AfferentTrackSpec` number (latency, priority, gain, thresholds, refractory, coalesce); PainKind → track membership; which tracks a receptor declares; `relay_gain` defaults | **innate prior**: hard-coded on purpose, with follow-ups owed (`adaptive_nociception.md`, `nociception_layer.md` step 5; the `behavior_tiers.md` M4 migration trigger) |
| What delivered events teach (cluster fear, bias, GL4 mappings) | **learned**, by existing mechanisms; nothing in this plan learns |

### 5.10 New `[engineering]` invariants and their guards

| Invariant | Regression guard |
|---|---|
| Every percept enters EC through the registry | `scripts/lint_relay_only_encode.py` (M39: an AST lint allowing `SensorEncoder.encode_*` / `LinguisticEncoder.encode*` / `EC.pattern_complete_or_separate` only from `perception/` plus an allowlist naming each current caller with its reason, seeded from the `src/` callers it scans; the body-path functions and the imagination / MemoryHub sites that move behind the registry are **not** on it, pre-merge Executor S2), **landed in the GL3.B1 PR** (TR22), plus the registry's per-receptor route counter (§5.3), asserted on both `_encode_current_clusters` and `propose_via_substrate`. M39 scans `src/maxim/` only: it does not see the earned-row harness encoders that call `encode_sensors` directly (`scripts/exp56/common.py`, `scripts/exp57/common57.py`), so the rule is not over-read for them (wiring N5); the TR21 callers reach the registry through `executor`. Until GL3.B1 ships: **process invariant, mechanization backlog M39**. |
| Each learning sink (pain memory, NAc outcome, Wire-2 valence, Wire-4 cluster fear, eligibility) sees a pid at most once, keyed on the situation noted at its transduction pass (TR18) | `tests/unit/test_afferent_tracks.py` (GL3.B3): a count of `record_cluster_fear` calls and pain-memory tags per pid on a scripted health hit and a scripted oxygen breach, one each, with the oxygen breach's fear key equal to the cluster noted at the breach pass; plus a same-pass double transduction of one locus (H3t then H3), whose PainBus publish stays at that pass (pre-merge Executor S3) |
| Provenance is stamped at the write: every body writer stores a per-sensor last-writer kind in the map beside the written `Entity.vital_metrics`, never on an `Embodiment` (TR22, TR23; pre-merge Wire B1); a derived value inherits its source write's kind; a missing slot (no writer yet) reads `legacy`, never `experienced` (TR29); no raw `vital_metrics[...] =` outside the one kind-taking setter (delta round, Wire D-2) | Structural: the keyword-only kind on the setter, each writer and `ModulatorAffordanceTool`'s constructor (`TypeError`), and `provenance=` keyword-only on `SensorEncoder.encode_sensors`; `scripts/lint_vital_metrics_writes.py` for the raw writes (until built: process invariant, mechanization backlog M78); `tests/unit/test_ec_provenance.py` drives `OrchestratorActorTool`'s target_effect write and asserts `narrated` on the AUT's entity map |
| Tracks are logical, never OS threads | M40: a grep/AST lint, no `threading.Thread` / `asyncio.create_task` in `perception/` except allowlisted hardware-edge adapters. Process invariant until built. |
| One physical event, one pid, across every track; a consequence carries its cause. On the tool path the invocation pid is minted at `Executor._run_started`, before `tool.run`, and every H3t emit inside that `tool.run` reuses it, so the track copies, the `InteroceptiveOutcome` and the GL4 pair share one pid (pre-merge Wire B3; autonomic_layer.md §3.1.2) | Structural guard: `AfferentEvent.__post_init__` rejects a missing pid / receptor / provenance, and `PhysicalEventId.__post_init__` (from the post-fence resume stage, G17) rejects an empty agent, an empty session id or a negative seq; `tests/unit/test_afferent_tracks.py` pins the fan-out, and its tool-path arm asserts the burn's two deliveries and the `ToolOutput`'s record carry the same pid. |
| One seq authority per agent at a time; seqs unique and monotonic across the GL3.B3 handover and a save/load | `tests/unit/test_afferent_tracks.py` (handover arm) + the resume stage's test through both real load paths (G15, G17) |
| No provenance default; a narrated record is never relabelled experienced, through the registry, save/load, merge or ingest | Structural guard: the sentinel rejections in `ReceptorSpec` / `AfferentEvent` / `InteroceptiveOutcome` / `ActionContext` `__post_init__`; `tests/unit/test_ec_provenance.py` |
| Total order, no wall time, cross-process determinism | `tests/unit/test_afferent_tracks.py` + the flag-on arm of `test_agent_loop_selection_golden.py` across two processes and two `PYTHONHASHSEED`s |
| Preemption re-arms, at most once per breach episode (§5.6) | `tests/unit/test_tracks_preemption_loop.py` (livelock probe on a deepening breach) + a `TestLoopWiringPins`-style pin on the handler |
| `provenance=` cannot be omitted on a text encode | Structural: keyword-only, no default (`TypeError`) on `LinguisticEncoder.encode*` |

---

## 6. Stages

**Renumbered 2026-10-07 (owner decision G8).** Old → new: B0 → GL3.B0; B1 + B3 → GL3.B1 (the registry
lands with provenance); B5 → GL3.B2; B2b → GL3.B3 (the fan-out, now the first track slice); B2a → GL3.B4
(preemption, a declared rung arm or waits); B4 → GL3.B5; B6 → GL3.B6; B7 → GL3.B7; B8 → GL3.B8.

Every `src/` stage gets the three-lens pre-merge review (Executor, Architecture, Wire integrity;
`docs/CODE_REVIEW.md`). This plan got its **four-lens design review** before GL3.B1 (part of T8; done
2026-10-24 and folded, §The T8 (b) four-lens design review), and it answers the testbed audit and SUBSUME
in writing (§4.1).

**Rule T (touch it, shrink it; wiring S11).** The stages edit functions pinned over 200 lines in
`src/maxim/utils/function_length_baseline.json`: `propose_via_substrate` (GL3.B1, H3),
`SensorEncoder.encode_sensors` (GL3.B1, TR22), `Embodiment.evaluate_failures` (GL3.B3),
`run_agentic_loop` (GL3.B3 H4, GL3.B4 H2), `start_simulation_mode` (the self-entity encode at GL3.B1, the
inbox route at GL3.B3), `compose_bundle` and `ingest_bundle` (GL3.B1), and (pre-merge Architecture S10,
Executor N5) `runtime/tool_dispatch.py::record_outcome` (pinned at 399; GL3.B1's flagged credit check on
`_cluster_reward_bias`) and `embodiment/tool_bridge.py::ModulatorAffordanceTool.execute` (pinned at 502;
GL3.B1's writer kind threaded into both `_apply_sensor_deltas` branches, GL3.B3's H3t). Each stage names
the extraction that carries its edit: for GL3.B1, the per-channel read+encode loop leaves
`propose_via_substrate` as `registry.tick`; the credit check enters `record_outcome` as an extracted
helper that carries the cluster-reward booking with it; `execute`'s self_effect / target_effect blocks
move into a helper that takes the constructor kind. Any expected `touch_exemptions` record is named in the stage's PR description up front.

**Three flags, not one (wiring S12).** GL3.B1's credit flag (TR20), GL3.B3's fan-out flag and GL3.B4's
preemption flag are separate `maxim config` keys, so studying the fan-out never arms preemption, which G8
reserves to a declaring arm. `~/.maxim/` is shared across worktrees, so an operator's leftover flag could
change an earned-row re-run: the effective values are stamped into every `report.json` (M1) and asserted
OFF by the T1 re-run harnesses (`exp60_run.py`, `exp61_run.py`, `exp62_run.py`, `r3_run.py`,
`b1_floor_run.py`), not only by the E-rung harness (M10's row text changes to match, in `outstanding.md`). GL3.B4 also gets a **gate design pass** on
a one-page approach note, because the preemption check is a gate. **Fence:** GL3.B1 and GL3.B3–B7 change
code Session A owns (`agent_loop.py`, `loop_*.py`, `substrate_proposal.py`'s call path,
`orchestrator.py`). They start only after the relevant decomposition slices merge, and they must not
reopen a slice's characterization tests. The defect issues for L1, L2, L5 and L7 (§3.4, §3.5) are not
stages of this plan: they are filed now and fixed in their own seams, under the same fence.

### GL3.B0: census and red gates (tests only; runnable now, inside the fence)

> **BUILT 2026-10-09** (tests only; branch `docs/gl3-b0-afferent-census`). The census is
> `tests/unit/test_receptor_census.py` (its corrections are folded into §2 and §3.3); the red gates are
> `tests/unit/test_afferent_red_gates_units.py` (a, b, c, g) and `tests/unit/test_afferent_red_gates_loop.py`
> (d, e, f; (f) in a refractory and an energy-exhausted variant), each `xfail(strict=True,
> raises=AssertionError)` so a broken fixture fails loudly instead of counting as red. TR1 and TR2 are
> decided (§11, DECISIONS.md 2026-10-09).
>
> **The read surfaces are B0's choice, and the flipping stage must flip them as written** or record an
> owner-approved change (never a silent re-point; a red gate that does not flip is data): (a) reads NAc
> eligibility above zero on the imagined nodes and assumes TR12's strict "refuse" (**owner-approved
> change, TR20, 2026-10-24:** (a) flips under GL3.B1's flag-on arm, because acting on provenance waits
> behind the credit flag); (b) drives the real
> composition (`build_minecraft_aut(..., client=)` → `aut.percept_source.next_percept()`) and reads the
> percept's `agent_id`, so passing the AUT's `agent_id` through `build_minecraft_aut` flips it; (d) matches
> the stale execution by proposal identity AND by (the next pass, the same tool), so a fix that copies the
> action dict cannot flip it falsely; (c) reads a `pid=` keyword on
> `world_set_azimuth` and the percept's `.pid` / `context.pid` / `metadata["pid"]` (GL3.B8 decides where
> the pid is readable, explicitly); (g) reads a `"provenance"` key on
> `EntorhinalCortex.substrate_node_metadata` and assumes the strict mixed-state rule (§5.7 item 1:
> `narrated` whenever any declared sensor's last writer was the narrator).
>
> **The frozen characterization** (`tests/unit/test_afferent_latency_characterization.py`, fixture
> `tests/fixtures/afferent_latency_characterization_v1.json`, measured at 7d18347e with `src/` equal to
> `main`). It is a frozen RECORD, not a golden: it is never regenerated once `src/` moves (the L1/L2 fixes
> and GL3.B4 will move the numbers; since the pre-merge round, #1176's unflagged fix is followed by a
> re-frozen v2 record that GL3.B4's drift check reads, and this v1 stays as pre-#1176 history; GL3.B4's
> negative control is produced on the post-#1176 tree by a test-only monkeypatch, §6 GL3.B4 gate 6).
> Its test checks equality only
> while `src/` is the measured tree, and the shape otherwise; the instrument's blob hashes are recorded
> as provenance and never asserted, so the shared `_loop_harness.py` stays free to change. "Both arms" are read as the two `_loop_harness` worlds, substrate-primary, in loop passes:
> **damage** (shore world, health 20 → 8 landing while a proposal is pending: the L1 case GL3.B4
> compares against) — pain publish +2 passes, first protective action (`flee`) +3 passes, and the
> stale pre-damage proposal is dispatched; **drowning** (the fear_water world without seeded fear, so
> the breach itself writes the Wire-4 fear) — `drive:oxygen` publish +0 passes, first successful
> protective action (`escape_water`) +25 passes. llm-primary is not characterized: its protective action
> cannot be measured deterministically without an LLM. The loop driver duplicates `_loop_harness.run_arm`'s
> setup in the gate file because `run_arm` takes no world, gate or hook arguments and the harness is
> Session A's; it can fold into the harness if those arguments are added there.
>
> **What the record is (environment S4, confounding N5).** It is a step-clock harness fact:
> `_ScriptedWorld` re-syncs the body on every loop sleep, so the body sees a breach with zero staleness,
> and the record is identical on a CI runner and a laptop. It measures nothing about the rig (pump at
> 0.5 s default, bridge state interval 500 ms default, blocking holds). The drowning arm's "+0 passes"
> is the substrate cadence's phase, not a mechanism: never cite it as "drive pain is immediate". The
> damage arm's +2 publish depends on the stale `mine_block` being refused. The 2026-10-24 review re-ran
> `measure_all()` at `d9dffe6e` (137 `src/` commits after the measured tree): both arms equal the record.
>
> **Owed to a follow-up issue (filed by the main session; confounding S1, S2, N2, N3; wiring S6).** "Each
> fails today" shows a gate can fail, not that only the claimed fix flips it. Each built gate gets a green
> control that a non-claim fix would break: (a′) an experienced own-sensor encode writes eligibility
> above zero on the same read surface; (g′) a drift-only encode on a fresh body carries `legacy`
> (its slots are missing, "not yet written", TR29) and never `narrated` or `experienced` (restated by
> the pre-merge round, Executor S1, and again by the delta round under G29: the draft's "carries
> `experienced`" assumed a slot default the no-default rule forbids, and the interim `unknown` kind is
> dropped, §5.7 item 1; #1450's body still says `experienced`, and the main session corrects it there); (d′) with no hurt, tick k's proposal executes at pass k+1; (e′) on a body whose free run acts, the
> denied run installs no proposal and makes no executor call (TR2's action half). Gate (d)'s world never
> picks a protective action (its post-damage ticks pick `eat` again), so (d) pins re-evaluation only.
> (c) uses two readings and asserts two distinct pids, each shared across its two lanes; (g)'s docstring
> says its "`experienced` not in provenance" holds only on a fresh EC. New strict red gates (g2)
> `DamageComponentTool` and (g3) `OrchestratorActorTool` targeting the AUT, and (h) `WaterTrial.rescue`
> on `ScriptedWaterBridge` expecting `apparatus` on the surfacing read's EC node (environment S6), all
> flipping with GL3.B1, land before GL3.B1 builds. Where an assertion inside a strict red gate changes,
> it is recorded as an owner-approved change.

- **Build:** `tests/unit/test_receptor_census.py` enumerates every `make_*_percept` caller, every raw
  `Percept(` site, every `encode_sensors` / `LinguisticEncoder.encode*` caller and `_SUBSTRATE_CHANNELS`,
  against a checked-in table matching §2, **plus** every pain producer in §3.3's ingress census and every
  `evaluate_failures` caller with the thread it runs on (§3.2: the narrator tools on the orchestrator
  thread, the reflex dispatch on the loop thread, never `sim.dm`). A new unregistered producer fails it. This
  is the check behind the census, so no backlog row is needed. Strict red gates (`xfail(strict=True)`):
  (a) an imagined affordance encode produces no eligibility on its node;
  (b) a Minecraft event percept carries a non-empty `agent_id`;
  (c) DoAFeed's two lanes share a pid;
  (d) `test_tracks_d1_stale_substrate_proposal`: through `_loop_harness`, health drops between passes and
  the pre-damage proposal is **not** executed;
  (e) `test_tracks_d2_turn_gate_skips_pain`: with a denying `substrate_action_gate`, a drift-driven breach
  is still published on the breach pass;
  (f) `test_tracks_d5_reflex_behind_thought_gate`: with a refractory or energy-exhausted gate, a matching
  reflex still fires;
  (g) the EC node a narrator-tool consequence (`SetEntitySensorTool` / `DamageComponentTool`) reaches
  carries per-node provenance `narrated`, never `experienced` (the EC-node provenance write, which flips
  with GL3.B1; the record half, `InteroceptiveOutcome` stamped `narrated`, is GL2a's own guard, not
  this red gate).
  Which change flips each: (e) and (f) flip with the L2 and L5 defect issues; (d) with the L1 issue's fix
  or with GL3.B4, whichever the L1 issue decides; (a), (b) and (g) with GL3.B1; (c) with GL3.B8.
  Plus a **characterization** (not a gate): breach → publish and breach → protective-action latency, in
  passes, on both arms. **It is frozen**: committed, with the commit it measured, before GL3.B4's gate
  design pass begins. GL3.B4 never compares against a re-measurement: its drift check reads the frozen
  v2 record (re-frozen once after #1176 lands), its gated comparison is flag on vs off on one tree, and
  this v1 is history (§6 GL3.B4).
- **Falsifiable gate:** the census equals §2's and §3.3's tables. Each red gate fails today for its named
  reason, and each has (or, for (a), (d) and (g), is owed, per the follow-up note above) a control that a
  non-claim fix would break (confounding N1, S1). Coordinate the harness use with Session A (shared `_loop_harness`).
- **Blast radius:** none (tests only).
- **Owner decisions at start:** TR1 (L7: Dormant, recommended, or delete); TR2 (L2's semantics: may the
  turn budget delay *nociception*, or only *action*? Recommendation: only action). Filing L1, L2, L5 and L7
  as issues is already decided (G8).

### GL3.B1: the Receptor registry with provenance on the write path (lands with its consumer)

- **When:** only together with a consumer of provenance (owner decision G8): GL5's experiment, or the
  forward model's contamination guard (latent_forward_model.md). The registry does not land alone.
- **Build:** `ReceptorSpec`, `Receptor`, `ReceptorRegistry.register/tick/encode_event`, the
  `AfferentEvent` type (pull reads mint no pid, §5.2; emitted events draw pids from the resume stage's
  `EventSequencer`, G17; the seq handover is GL3.B3's). The registry is bound to the agent's primary
  `Embodiment` through `perception/`'s `registry_for(embodiment)`, derived from the root entity tree's
  declared sensors (`Embodiment.root` and its components, not a YAML path) and built once per
  `Embodiment` under an `RLock` on first use (TR21, §5.3; delta round). The three `ModalityChannel`s
  registered as receptors in today's code order (interoception, audio, world), with specs derived from
  the declared sensors (the body YAML's keys as instantiated on that tree);
  `_encode_current_clusters` and `propose_via_substrate` call `registry.tick` (returning
  `(channel_values, clusters)`). Provenance per §5.7: the
  imagination and own-body affordance sites become receptors on the synchronous `encode_event` route;
  `provenance=` is required on `LinguisticEncoder.encode*` and `SensorEncoder.encode_sensors`; the
  per-sensor last-writer kind map beside `Entity.vital_metrics`, its writers and the writer-kind field
  on `ModulatorAffordanceTool` (TR22, §5.7 item 1); the binder `registry_for(embodiment)` (TR21, §5.3);
  the apparatus window's
  consumer (TR22); the narrator tools' records carry `narrated`; the EC per-node provenance set and its
  per-kind counts go through `EC.save/load`, `_BUNDLE_EC_NODE_FIELDS`, `ec_merge_aligned`, ingest and
  `entry_index.py` (TR26); eligibility weighting by kind and the credit check (over the §5.7 item 4
  table) ship behind the one credit flag, default OFF, and cover EC node ids only (TR20). The M39 lint
  lands in this PR (TR22). No tracks, no new YAML keys.
- **Falsifiable gate (all must hold):**
  1. **Selection path, byte-identical:** a golden fixture over the `minecraft_player`, `infant_operant` and
     `reachy_mini` bodies across a scripted sensor trajectory; the sequence of
     `(modality, cluster_id, geometry_tag, margin)` is byte-identical before and after. The selection
     golden (both arms, cross-process, cross-hash-seed), `test_encoder_golden_v1.py`,
     `test_place_code_wiring.py::test_value_and_range_walks_stay_in_lockstep` and
     `scripts/selection_dynamics_rebaseline.py` output are unchanged. **The trajectory (confounding S7):**
     on each body, at least one step on which each channel goes from reading to empty and back
     (`minecraft_player` has no azimuth sensor, so its audio channel is always empty); at least one step
     with all three channels active (`reachy_mini` or a fixture body); `reachy_mini` run with
     `MAXIM_PLACE_CODE_EXTEROCEPTION` off and on, because the audio encoding differs (environment N4). The
     golden tuple also carries the floor policy's `observation`, and the golden gains a B1-floor-policy arm
     (wiring S5). **The "before" side of gates 1, 2 and 4 is committed first** (pre-merge Executor N6):
     each pre-stage golden is generated on the pre-stage tree and committed in its own commit before the
     build commit, frozen-record style, so "before" is never a regeneration at review.
  2. **Persistence and wire bytes:** a saved `ec.json` and an exported bundle manifest differ from the
     pre-stage output **only** by the added `receptor:*` keys under `encoder_provenance` and the per-node
     provenance field, pinned by a golden diff. A pre-GL3.B1 `ec.json` loads with `legacy` and one warning
     (`tests/integration/test_persistence_compat.py`). A bundle round-trip (export → ingest → merge) keeps
     every node's provenance set, and a narrated kind survives a merge against an experienced node. An
     entry-index round-trip keeps it too (wiring S4).
  3. Red gates (g), (g2), (g3) and (h) flip as written, each reaching the registry through
     `registry_for` with no edit to the gate (a known-answer test pins that (g)'s exact surface,
     `SimpleNamespace(embodiment=Embodiment(body))`, whose root tree declares sensors, resolves a
     registry rather than the typed error, and a second case that a root tree declaring no sensors takes
     the typed error; pre-merge Executor B1, Wire B2; delta round, Executor S-A, Wire D-1); (a) flips under the flag-on arm (TR20's recorded re-point); (b) per
     TR25.
  4. **Flag off (TR20):** an **eligibility golden**, per-node eligibility after a scripted `--sim`
     trajectory with one narrator write and one imagination encode, composed with an `ImaginationTrigger`
     and a Reaction so `distribute` runs, is byte-identical to the pre-stage tree. **Flag on:**
     `tests/integration/test_memory_hub.py` passes; an imagined-only node receives zero credit; a
     narrated-only node receives exactly the declared discount; a mixed node receives its creditable share
     by the per-kind counts (confounding S5 (iii)); `tool:<name>` credit is unchanged; every writer in §5.7
     item 4's table honours the check. **Reported, not gated:** the share of interoception-channel encodes
     stamped `narrated` on a cradle arc trace (bio-faithful S4) and the fraction of mixed nodes on a
     recorded `--sim` trace (confounding S5 (iv)), both before the credit flag may flip.
  5. **Deletion probes** (TR22: gate 1 is byte-identical with or without the registry, so a probe that
     "bypasses the registry and fails gate 1" cannot go red; confounding D2): add a direct
     `encode_sensors` call outside `perception/` and M39 fails; bypass the registry and its per-receptor
     route counter, asserted equal to the number of encodes the gate-1 trajectory performs, reads zero;
     permute the registration order and gate 1's golden fails (confounding S7); remove the flag guard and
     gate 4's flag-off eligibility golden fails (wiring D3); remove the credit check and an imagined-only
     node gets credit; widen the check to non-EC keys and the `tool:<name>` credit test fails; drop the
     field from `_BUNDLE_EC_NODE_FIELDS` and gate 2's round-trip fails. **The route counter is asserted
     on both entry points** (pre-merge Executor S2): the gate-1 trajectory drives `_encode_current_clusters`
     **and** `propose_via_substrate`, and the counter equals each one's encode count, so a build that
     leaves either encoding directly reads short. M39's lint passes with its allowlist, **seeded from the
     `src/` callers it scans** (M39 scans `src/maxim/` only, §10): the two body-path functions and the
     imagination / MemoryHub sites that move behind the registry are **not** on the GL3.B1 allowlist, so
     a build that leaves one of them calling an encoder directly fails M39. The `scripts/` body-path
     callers (wiring D1's census category, "body-path encode / propose callers") are a census row, not an
     allowlist seed, because M39 does not scan `scripts/`.
- **Guard tests:** `tests/unit/test_receptor_registry.py` (validation, order, empty-read rule),
  `tests/unit/test_receptor_registry_golden.py`, `tests/unit/test_ec_provenance.py`, an addition to
  `tests/integration/test_persistence_compat.py`, and a two-process test that the persisted provenance and
  pid strings are identical under differing `PYTHONHASHSEED`.
- **Blast radius (strict reading of the trigger wording):** **T1-6** (Exp 42) through its
  "SensorEncoder / EC-interoception change" trigger: the drives channel becomes a registered receptor, so
  it fires functionally through the encoder call site, not through any body wording (no body changes
  here). **T1-10** (Exp 53) twice: "`_encode_current_clusters` / … change", and "NAc/EC persistence format
  change", because the receptor stamp and the provenance field change `ec.json` **even with the flag
  off**. **T1-11** (Exp 56) and **T1-12** (Exp 57, PARTIAL): "`SensorEncoder` / EC world-modality change";
  T1-12 also through "`ec_merge_aligned` … change" (the provenance fold). **T1-15** (Exp 62): "the
  `minecraft_player` world-sensor roster" and "`_sensor_embed` / `gain_exponent` / `gain_modalities`",
  whose call site moves. **T1-14** (Exp 61): "`hivemind/bundle.py` scrub" (`_BUNDLE_EC_NODE_FIELDS` and
  the manifest's `encoder_provenance`), "`ingest.py` bounds", "bundle format version bump" if one is taken,
  "`NAc.credit_node` write-path change" when the check is on, and every Exp 60 trigger by inheritance.
  **T1-13** (Exp 60) by its world-modality wording, and through "`TemporalCreditDistributor` credit-path
  change" when the check is on. **The `WaterTrial` apparatus window** (pre-merge Executor S7, Wire SF-7)
  is an edit to the instrument of **T1-13, T1-14, T1-15, R3 and B1**: each is walked on its "harness"
  wording, with `scripted_water.py` and `test_water_trial_smoke.py` proven byte-identical (§5.7 item 1).
  **T1-9** (Exp 52, and T1-8, superseded) by its "`NAc.credit_operant_reward` change" wording: the
  flagged check covers that writer and the cradle mother's credit is recorded with its `apparatus`
  source; with the flag off nothing changes, and the flip's T1-9 walk decides the mother's credit
  (§5.7 item 1). The `encode_sensors` keyword reaches the evidence scripts of T1-6, T1-7, T1-9, T1-11,
  T1-12 and T1-13 (§5.7 item 3); each touched script's offline guard is re-run (TR3). Offline guards re-run at GL3.B1; the rig re-runs for credit belong to the
  flag flip, not the build.
- **Owner decisions at start:** which consumer it lands with; TR3 (does a byte-identity-proven registry
  fire these rows? Strict default: yes; re-run each row's offline / dry-run guard, and record a committed
  machine-readable exception for the rig-only re-runs if no rig time is spent. The alternative, that the
  goldens discharge the triggers structurally without firing them, is **open, not settled**; the #888
  owner-decision-C precedent is its argument); TR4 home; TR5 derived specs; TR11 kinds (with the
  per-percept kind table, wiring S3); TR12 imagined refuse vs discount, with the before/after-normalization
  choice (§5.7); TR13 credit-check flag default and `legacy` handling; TR14 dict vs namespace; TR25 red
  gate (b)'s home; TR26 the entry-index projection; the mixed-state rule's alternative (b) (§5.7 item 1).
  TR20–TR22 are decided (2026-10-24).

### GL3.B2: handoff to the latent forward model (GL4)

- **The contract, stated once (GL4 wiring S8, folded on GL3's side).** The forward model's pair is built
  **per tool invocation**, and its pid is the **invocation pid** (the `InteroceptiveOutcome.pid`), drawn
  from the same per-agent `EventSequencer` (G3) as every `AfferentEvent` pid. **One pid (pre-merge Wire
  B3):** it is minted at `Executor._run_started`, before `tool.run` (from the post-fence resume stage,
  G17), not after `tool.run` returns; the H3t `AfferentEvent`s emitted inside that `tool.run` reuse it,
  and `_stamp_invocation` stamps it on the `ToolOutput` alongside the executor's invocation id and mints
  nothing. So on the tool path the pair, the record and the burn's track copies carry the same pid, with
  no `caused_by` hop (the earlier "mint after `tool.run`" wording could not be joined: the H3t emit
  happened before its target pid existed). `AfferentEvent`s that are not this invocation's own
  transduction (a drift tick, a narrator post, a later substrate tick) carry their own pids in the same
  sequencer space and join through `caused_by`; a pull read carries none (§5.2). **GL4 holds the
  producer** (`latent_forward_model.md` S1). This plan's obligation is only that the join key exists, is
  deterministic, and that the H3t emit reuses the invocation pid once GL3.B3 emits.
- **Falsifiable gate:** on recorded traces from the `_loop_harness` arms and `_cradle_loop_driver.py`
  (both offline, no rig or LLM spend; no pid-bearing trace exists yet; environment N5), ≥ 1 joined
  `(context, action, consequence)` row exists per scripted consequence, every row's pid resolves to
  exactly one physical event (one `InteroceptiveOutcome` with its own track copies, on the tool path, or
  one `AfferentEvent` otherwise), and no row joins on an invocation id. Row counts
  are reported, with no threshold. Cradle-mother traces (`benchmark_cradle_mother.py`) are `apparatus`
  consequences under G16 and are counted apart.
- **Blast radius:** none (no consumer; GL4's T4a rule applies).

### GL3.B3: AfferentTracks, the first slice: the thermal dual-track fan-out (no preemption; flag default OFF)

- **Needs:** GL3.B1, GL2a's `InteroceptiveOutcome` and GL2b's `NociceptorSpec`.
- **Build:** `AfferentTrackSpec` and `TrackScheduler` with **two** tracks, `nociceptive_fast` and
  `affective_slow`. H0 (the edge inbox, drained once per pass; under `--sim` the orchestrator thread's
  narrator-tool transduction posts there, §3.2), H3 (transduction emits through the scheduler and is
  drained in place, so the W-4 order is unchanged), H3t (the tool path, TR24) and H4, with latency-0
  drained in place at every `evaluate_failures` caller (§5.5's table). **Every PainBus publish, of every
  kind, stays at the transduction pass (TR18):** NOCICEPTIVE publishes **once**, in place, and that
  in-place publish **is** `nociceptive_fast`'s `pain_bus` delivery (§5.4; its other destinations, credit
  and relay, are non-PainBus deliveries of the same event; pre-merge Executor S3); DRIVE (`drive:oxygen` included) keeps its in-place synchronous publish in the W-4
  order; `affective_slow` then adds a delivery, for both kinds, only to sinks that are not PainBus
  subscribers (autonomic, the pid-keyed consequence record). `str(pid)` is carried into
  `TemporalEvent.context`, and `InteroceptiveOutcome.pid` is the same pid: on the tool path the H3t emit
  reuses the invocation pid minted at `Executor._run_started` (§5.2, §6 GL3.B2; pre-merge Wire B3). The **seq authority handover**
  (§5.5). Every `evaluate_failures` caller reads the per-sensor last-writer kind map when it mints the event
  (TR23). The narrator tools post through their `_aut_embodiment` (§5.5, reachability). No preemption
  right is exercised and no `basis_seq` exists yet. Flag: `maxim config set runtime.afferent_tracks
  true`, the fan-out's own key, separate from GL3.B4's preemption flag (wiring S12; config over env; if an
  env var is unavoidable, add an autouse conftest scrub in the same commit). **Touches** (wiring S7):
  `loop_setup.py::build_loop_run`, `loop_substrate.py`, `loop_gates.py`, `agent_loop.py` (H4 beside
  `_loop_bio_tick_maintenance` in `run_agentic_loop`), `embodiment/body.py` (`evaluate_failures` emitting
  through the scheduler), `embodiment/tool_bridge.py` (H3t), `simulation/tools.py`,
  `simulation/orchestrator.py` and `integration/bio_enrichment.py` (the inbox route), and
  `runtime/executor.py` (carrying the invocation pid from `_run_started` to the H3t emit, §5.2).
  `simulation/` files are Session V's in `roadmap_1_3_x.md`: the PR names that coordination (no fence
  conflict, both land after 1.3.6; pre-merge Architecture N4), as GL3.B1 does for `simulation/tools.py`,
  `cradle_mother.py`, `dm_runtime.py` and TR25's `simulation/minecraft_harness.py::build_minecraft_aut`.
- **The harness (TR24, environment D3).** The burn's nociceptor is declared only on a GL2b **cradle
  variant body** (`minecraft_player.yaml` has no thermal sensor; the bridge's `entityHurt` carries no
  cause), and `_loop_harness` is Minecraft only. The thermal gates run on
  `tests/unit/_cradle_loop_driver.py` on the GL2b variant body, substrate-primary, with an **experienced**
  `cradle_fire_pit.touch` from a starting state inside GL2b's calibrated band (GL2b §3.2's "noxious from
  ≥ X"), transduced in §4 inside `ModulatorAffordanceTool.execute` and drained at H3t. The driver
  replaces the substrate proposer with a script, so H3 is exercised by the `_loop_harness` arms, H3t by
  the cradle arm. The thermal gate is substrate-primary only: on llm-primary every cradle burn is
  narrated by construction (the orchestrator passes `embodiment=None` to the scene tools there).
  `MAXIM_DETERMINISTIC_SCENE_EMBODIMENT` double-counts one contact (the scene tool's write plus the
  narrator's harm: two physical events, two pids, and `caused_by` cannot link them because the narrator's
  write holds no handle to the tool-path pid). The double-count is fixed before any GL3 arm uses that
  variable; until then it is excluded from every GL3 arm.
- **Falsifiable gate (all must hold):**
  1. Flag off: both existing goldens are byte-identical, cross-process and cross-hash-seed.
  2. One scripted burn on the cradle arm yields exactly two track deliveries with one pid; the delivery
     log shows the slow one `latency_passes` non-idle passes later (a restatement of `emit`'s definition
     that pins the scheduler, not a consumer; confounding N4); the autonomic record and the credit
     record join on the pid (the part that carries information), and that pid is the invocation pid the
     `ToolOutput` carries (minted at `_run_started`, reused by the H3t emit; pre-merge Wire B3). The
     same holds on an llm-primary arm for a drift-tick transduction, drained in place inside the drift
     tick (wiring S8), where the event has its own pid.
  3. **Deletion probe on the slow track:** removing it removes exactly the deliveries it owns, the
     autonomic and consequence-record deliveries, and nothing else: the PainBus learning sinks' writes,
     the Wire-4 fear write and the fast path's goldens are unchanged (TR18; confounding S4).
  4. **Handover:** with the scheduler live, the `Embodiment` never assigns a seq itself; seqs stay unique
     and monotonic across the handover and across a save/load.
  5. Flag on: a new golden is identical across two processes and two `PYTHONHASHSEED` values. It
     includes the cradle arm, not only the Minecraft ones (TR24).
  6. **Narrator writes (TR23).** Run **on llm-primary with the drift tick live**: a scripted `--sim`
     narrator write (a narrator tool run on the orchestrator thread, not on `sim.dm`) is published
     **exactly once** and the record carries `narrated`, whichever evaluation transduces it first (the
     drift tick, an AUT tool in §4, or H0); a second case lands the write during an AUT `tool.run`, with
     the same result. The breach latch and the snapshot slot are touched from the loop thread only (a
     thread-identity assertion). The positive half closes the gap where the inbox post is dropped and
     the thread assertion still passes (confounding S8).
  7. **Flag on, the drowning arm (TR18, Exp 60 / T1-13).** On `_loop_harness`'s `fear_water` arm:
     `drive:oxygen`'s publish latency in passes equals the frozen record's (+0); its Wire-4 fear write
     lands on the same `(agent, cluster)`, in the same pass, with the flag on and off, and that key is the
     cluster noted at the breach pass; breach → `escape_water` latency is no worse with the flag on, in
     passes and in step-clock seconds. And the once-per-pid count test of §5.10 passes on a scripted
     health hit and a scripted oxygen breach.
  8. **Refractory (bio-faithful S8):** a deepening publish one pass after onset is delivered, at the
     deepened intensity; and a same-pass double transduction of one locus publishes on PainBus at that
     pass, never at the window's end (pre-merge Executor S3).
- **The llm-primary harness for gates 2 and 6 (pre-merge Executor S6): to be built in GL3.B3.**
  `_loop_harness` and `_cradle_loop_driver.py` are substrate-primary only (their docstrings), and GL3.B0
  recorded that llm-primary cannot be measured deterministically with a live LLM. GL3.B3 builds a
  **mocked-LLM llm-primary loop driver** (a scripted backend through the router, no live LLM, with the
  drift tick live) plus a **scripted orchestrator thread** that runs one narrator tool against
  `_aut_embodiment` at a scripted pass, and gates 2 and 6 run on it; if the fixture runner can host an
  orchestrator thread, the GL3.B3 PR may use it instead and says so. Until it exists, gates 2 (its
  llm-primary sentence) and 6 are owed, and GL3.B3 cannot pass without them.
- **Narrator tool output under flag-on (environment S5).** The narrator tools return their synchronous
  `evaluate_failures` results to the orchestrator LLM (`DamageComponentTool`'s `pain_triggered` /
  `failure_modes`; `OrchestratorActorTool`'s `side_effects["embodiment_failures"]`, "for the orchestrator
  to narrate"). With the transduction deferred to the inbox, the tool output's channel-1 failures are
  computed state-based (no latch, no drift), as they are today; latched channel-2 pain arrives at the
  drain.
- **Guard tests:** `tests/unit/test_afferent_tracks.py` (order, identity, pass refractory, inbox sort,
  handover, the once-per-pid count, track membership per §5.4's per-sensor mapping).
- **Blast radius:** with the flag **off**, nothing fires, and the golden proves it. With the flag **on**,
  any validation run is a new condition, and no earned row may be cited under it without a re-run; the
  orchestrator-thread move also changes when narrator-written pain publishes (at the next drain, not inside the tool
  call), so the narrator's narration and the AUT's next action see the burn later, and drift runs
  between the write and its transduction (a chilled body's `cold` drifts at 0.08/s, so a 30-s turn can
  erase a narrated warming before it is transduced; environment S5); and track publishes bypass
  PainBus's 0.5 s wall refractory for a one-pass refractory, up to twice the publishes per second into
  the learning subscribers on a deepening stream (environment N6). The drowning signal is not moved
  (gate 7). No idle gate moves (H5 is not built), so T1-13's idle-gate wording does not fire. **T3-9** (Exp 09:
  "PainBus / ReactionBus / NAc reward pipeline change") arguably fires once a track-routed publish exists:
  walk it before building. **T1-4**: tool-coupled pain stays synchronous inside `execute`, pinned by the
  existing T1-4 tests (`tests/substrate/test_sem_execution_production.py`,
  `tests/unit/test_tool_pain_pending_leak.py`).
- **Owner decisions at start:** TR6 (the two built tracks' numbers; recommendation as tabled in §5.4);
  TR7 (latency in passes, ordinal, recommended, vs experience µs, which is deterministic under `_StepClock`
  but couples scheduling to world kind); TR10 (does it fire T3-9? strict default: yes, walk and re-run its
  offline guard). TR18, TR23 and TR24 are decided (2026-10-24).

### GL3.B4: `nociceptive_fast` preempts a pending substrate proposal (a declared rung arm, or it waits)

- **Entry:** only as an arm a 1.4 rung declares (owner decision G8; the owner names the rung later), and
  only after GL3.B0's characterization is frozen. Outside its declaring arm the flag stays off, asserted by
  M10 as amended 2026-10-07. A gate design pass on a one-page approach note comes first; it must settle
  the same-pass reselect's double `note_active_clusters` / Wire-4 write (§5.6).
- **Named prerequisite (TR19):** per-pass body transduction, independent of selection, before H2,
  settled with #1176 (L1) and keeping the W-4 order (§3.4, §5.6). It is H2's event source. **Flag status
  (pre-merge Executor B2, Architecture S7):** it is a root-cause defect fix, so it lands **unflagged** in
  #1176's own PR, with its own trigger walk (per-pass encodes move NAc's per-pass inputs, so
  T1-13/14/15's wording is walked there) and the T1-13/14/15 re-runs that walk owes. It is not behind
  this stage's preemption key: a toggle around a defect is the shape CLAUDE.md's "no band-aid" rule
  forbids. GL3.B4 is therefore built and measured on the **post-#1176 tree**, and its flag is the
  preemption key only.
- **The frozen record, v2 (pre-merge Executor B2).** After #1176 lands, the latency record is re-frozen
  as `afferent_latency_characterization_v2.json`, measured on the post-#1176 tree with the same
  instrument: both existing arms (damage, drowning) plus the one-scenario arm below. v1 stays as history
  (the pre-#1176 numbers). Gate 1's drift check reads v2.
- **The scenario (confounding S2, GL3.B4 half).** One scenario serves gates 2 and 3: a stale action that
  is a body affordance that really runs, **and** a post-damage pick that differs (protective). It is
  frozen in v2 with the other arms. Red gate (d) pins re-evaluation only, because its world never picks
  a protective action.
- **Build:** `basis_seq` on substrate proposals, H2 (the preemption check before §4, with an in-pass
  reselect), `nociceptive_fast`'s `pending_proposal` preemption right, and the typed preemption handler.
  Flag: GL3.B4's own preemption key, separate from GL3.B3's fan-out key (wiring S12).
- **Falsifiable gate (all must hold, or the stage fails):**
  1. Flag off: both existing goldens are byte-identical, cross-process and cross-hash-seed. **Drift
     check (environment S4, confounding D1):** the flag-off measurement on the build tree, with the same
     instrument blobs, equals the frozen **v2** record (post-#1176; v1 is pre-#1176 history and is not
     compared, because #1176 moves the damage arm's +2 publish by design); if not, stop and record the
     drift, and never compare a flag-on number against a drifted baseline.
  2. Flag on: red gate (d) flips (if the L1 issue has not already flipped it). The pre-damage proposal is
     never executed, the pain event's `seq > basis_seq`, the event comes from the named prerequisite's
     transduction, and reselection runs in the same pass and picks the protective action.
  3. **Latency, split in two, paired on the same post-#1176 tree (TR19; bio-faithful D2, confounding D1,
     S10, wiring S10; pre-merge Executor B2).** Preemption flag on against off, both on the GL3.B4 tree
     (which already carries #1176's unflagged transduction), on **both** frozen arms, in loop passes
     **and** step-clock seconds (idle passes are 0.05 s, running passes 0.25 s, so a change in idle
     passes cannot fake a pass-count win): **(ii) breach → protective action given a publish before §4 is
     the gated claim**, owed to preemption: strictly lower on the arm the claim names, no worse on the
     other. **(i) breach → publish latency is REPORTED, not gated**: it is #1176's (the per-pass
     transduction is in both arms, so flag on vs off on this tree is ~0 by construction), reported beside
     v1's pre-#1176 number so the prerequisite's own gain is visible and never credited to preemption.
     Both are pre-registered in the arm's prereg. The frozen v2 record is gate 1's drift check, not the
     comparison baseline.
  4. Flag on: a new golden is identical across two processes and two `PYTHONHASHSEED` values.
  5. **Deletion probes:** remove the H2 check and (2) and (3)(ii) go red, while the reported (3)(i) does
     not move; remove the in-pass reselect and (3)(ii) goes red.
  6. **Controls (TR19).** A **sham drop-and-reselect**: drop and reselect at the same pass with no
     qualifying event (or one below `preempt_threshold`); if the sham lowers (3)(ii) as much as the real
     arm, the gain is the reselect or the cadence, not preemption. A **negative control**: on the damage
     arm, H2 with no pre-§4 transduction does not preempt and does not flip (d). It is **produced, and
     gated, on the post-#1176 tree** (delta round, Executor S-B; the earlier "cite the v1 record" could
     not fail, since v1 was measured with no H2 at all): a **test-only `monkeypatch`** replaces #1176's
     per-pass transduction call with a no-op on the GL3.B4 tree, with the preemption flag on, and the
     test asserts no preemption and no (d) flip. This is not a runtime switch, so it is not the
     forbidden toggle. It needs no `src/` seam: the loop already calls the substrate path through a
     module reference (`runtime/loop_substrate.py`: `_sp.propose_via_substrate(...)`, "a test that
     replaces it patches ONE place", the pattern `tests/unit/_cradle_loop_driver.py` uses), so #1176's
     PR keeps its per-pass transduction a module-level function called through a module reference, and
     the GL3.B4 test patches that one attribute. If #1176 lands it in a shape that cannot be patched at
     a module attribute, the control becomes **reported, not gated**, and the GL3.B4 PR says what the
     gate then loses (proof that the preemption needs the pre-§4 event). The v1 record stays cited as
     pre-#1176 history only.
  7. **Livelock probe (confounding S3, environment S3):** a **deepening** breach (the drowning arm, and
     a damage stream of 1 hp per pass from 14 to 4 hp) preempts at most once per breach episode, and the
     protective action executes within k passes of the first above-threshold event (k pre-registered). A
     standing breach publishes once after entry, so it is not the probe.
  8. A count test pins `note_active_clusters` calls and Wire-4 writes per pass at the number the design
     pass chose.
- **The rig (environment S4).** A rig arm measures breach → protective action in wall time and bridge
  ticks, carries `MinecraftClient.state_age_s()` into `AfferentEvent.extra` at ingest, and inherits the
  Exp 60 preflights (bridge cadence ≤ 0.15 s, `_measure_bridge_cadence`; pump ≤ 0.25 s). GL3.B4's whole
  possible gain on the damage arm is ≤ 3 passes (0.55 step-clock s), the size of one default bridge
  interval.

  These engineering gates are tests. Any behavioural claim belongs to the declaring rung arm's prereg,
  under the weak-evidence rule.
- **Guard tests:** `tests/unit/test_tracks_preemption_loop.py` (the `_loop_harness` arms above).
- **Blast radius:** with the flag off, nothing fires. With the flag on, it is the declared arm's
  condition; T3-9 as at GL3.B3; T1-4 pinned unchanged. If the owner ever turns the flag on by default,
  T1-13/14/15 fire and need the batched rig slot with GL2c's Exp 60 re-run.
- **Owner decisions at start:** which rung declares it; TR6 (`nociceptive_fast`'s preemption right and
  threshold); TR8 (the protective action after a preemption: same-pass reselect, recommended, vs the
  `reflex` track's motor program first). TR19 is decided (2026-10-24).

### GL3.B5: text receptors and the multiplex (after the orchestrator slices)

- **Build:** `minecraft.events`, `sim.narrator` and `cli` register. `CompositePerceptSource` routing is
  addressed by `receptor_id`. Minecraft `[damage]` events carry `caused_by` = the health-drop sensor
  event, which is the first cross-lane join. This is the word world's entry point, so under G1 it follows
  the body-world stages.
- **Falsifiable gate:** a scripted-source run (`ConversationalSource` plus a mocked LLM, or the fixture
  runner) compares the percept sequence byte for byte before and after (confounding S9: a live-LLM
  comparison cannot isolate the change). The logged interactive session (`MAXIM_LOG_FILE` +
  `--interactive false`, 3 turns, `ACTION_FOLLOWUP` present) stays as the CLAUDE.md smoke, not the gate.
  Every text percept's agent is non-empty. **The `caused_by` join (environment S8):** the bridge's damage
  event is `event("damage", "took damage (health …)")` with no cause and no snapshot sequence, and a
  health-drop sensor event exists only when the latch publishes (below 14 hp, or on deepening), so
  in-band hits (20 → 17) have nothing to join to. The cause target is restricted to band-crossing damage
  (or, alternatively, a `world.state` read in which `health` decreased, which needs delta detection); the
  bridge's snapshot counter is added to each event line (a game-native field); the join rate is measured
  on a recorded Minecraft trace and reported stratified by in-band vs breach damage; no threshold is
  claimed.
- **Blast radius:** no T1 row (no EARNED row depends on the word path, #1120). T3-9's trigger wording does
  not name percept factories, so it fires only if `inject_pain`'s `reaction_bus` path moves, and that move
  is GL3.B6's.
- **Owner decision at start:** TR15 (whether the Dormant `perceived_intensity` field is marked dead or
  wired from `relay_gain`).

### Later track slices (not scheduled in 1.4; each enters only when a rung names it)

- **GL3.B6 `reflex` track** (re-houses L5's timing, which the L5 defect issue has already fixed, and fixes
  the L4 pain half): H1 for `inject_pain`, and the §1.16 orienting reflex (a motor act) onto `reflex`.
  The keyword reflexes register as a narrated nociceptive receptor whose body write transduces onto
  `nociceptive_fast` (§5.4 note); GL3.B6 is a revive trigger of `deferred/reflex_layering.md`, not a
  re-housing of the damage-dealing maths on `reflex` (bio-faithful S1). **Gate:** red gate (f) stays green, and on an Exp 09-style offline fixture the firing
  counts with the gate passing are unchanged. **Blast radius: fires T3-9** (it changes H1 firing counts
  directly). Decide before building.
- **GL3.B7 LLM-primary and the sensory tracks:** in-flight-plan preemption (part of L3; the 300 s blocking
  deliberation stays a decomposition item), `extero_detail` / `language` with ThalamicGate as their prompt
  gate, the CLI rule characterized as unchanged, and **H5 (idle-loop wake)**. **Blast radius: H5 fires
  T1-13/14/15** ("`run_agentic_loop` idle-gate or autonomy handling change"). NAc decays per pass, so a
  woken loop changes eligibility and bias magnitudes. Schedule it with their rig re-runs.
- **GL3.B8 hardware edges:** the Minecraft pump, DoA and stdin go through `edge_inbox`, plus held-action
  preemption where a backend can interrupt. Rig only. Red gate (c) flips here. **Blast radius: the DoA
  feed change fires T1-7** ("DoA front-end / Reachy transport change") **and T1-10** ("`DoAFeed` …
  change"). Verify actuation with `yaw_verify.py` first (embodiment brief §1). **Hardware safety
  (environment S9):** no held-motion preemption on Reachy without a controller-level abort behind the
  clamp and the lock (`ReachyMiniController.goto_target`, the single clamped and locked dispatch point),
  with a deletion-probed test in `tests/unit/test_reachy_workspace_safety.py`, and a stated rule for what
  the retained-axes stash holds after an interrupted goto (a commanded pose the head never reached). The
  `reachy.doa` reading is head-relative at capture, and its stamp trails the samples by up to the ~2 s
  gate cycle (`DoAFeed.latest` carries `window_start` for that reason): its payload carries
  `(window_start, capture_head_yaw, capture_body_yaw)` as capture-frame data, never as a scheduling input,
  so the no-wall-time rule never strips the capture tuple. The sensor lane writes nothing unless
  `robots.yaml` declares a body with an `azimuth` sensor (`world_set_azimuth` returns False otherwise), so
  red gate (c) runs on a declared body.

---

## 7. Blast radius summary (rows by ID)

| Stage | Rows that fire by wording | Mitigation |
|---|---|---|
| GL3.B0 | none | tests only |
| GL3.B1 (registry + provenance) | strict reading: T1-6 (functionally, "SensorEncoder / EC-interoception"), T1-10 (encode call site **and** EC persistence format, even flag off), T1-11, T1-12 (world-modality wording + `ec_merge_aligned`), T1-15, T1-14 (bundle scrub / manifest `encoder_provenance` / ingest bounds; `credit_node` when on; Exp 60 by inheritance), T1-13 (world-modality wording; credit path when on) | selection golden byte-identical; persistence/manifest golden diff limited to the added keys; owner ruling TR3 (strict default: fire, re-run offline guards, committed exception for rig re-runs); credit check OFF, rig re-runs tied to the flip |
| GL3.B2 | none | no consumer |
| GL3.B3, flag off | none | goldens prove it |
| GL3.B3, flag on | new condition (incl. narrator-pain timing and narrator tool output under `--sim`, the one-pass refractory); T3-9 arguably; T1-4 pinned unchanged; the drowning signal (T1-13) not moved, by TR18 | default OFF; walk T3-9 before building; gate 7 on the drowning arm |
| GL3.B4 | flag off: none; flag on: the declaring arm's condition, T3-9; default-on would fire T1-13/14/15. Its named prerequisite (per-pass transduction, TR19) is #1176's unflagged defect fix: it walks T1-13/14/15's wording and owes their re-runs in #1176's own PR, not GL3.B4's | declared arm only (M10); batched rig slot with GL2c if ever default-on |
| GL3.B5 | none of T1; T3-9 only if the reaction path moves | log check |
| GL3.B6 reflex track | T3-9 | Exp 09 offline fixture |
| GL3.B7 idle-loop wake (H5) | T1-13, T1-14, T1-15 | batched rig slot after the 1.3.2 live Exp 60 re-run |
| GL3.B8 DoA feed | T1-7, T1-10 | rig; yaw verification |
| L1 / L2 / L5 / L7 defect issues | walked in each issue (L2 changes when drift-driven pain publishes under the orchestrator's turn gate, so T3-9's wording applies; L1's fix may touch the substrate selection path) | each issue walks its rows before building |

The 1.4.0 minor-version heartbeat walks every Tier-1 row regardless (T4).

**Other risks.** *God-object drift* (the stop rule in §5.1). *A selection-dynamics change hidden inside a
refactor*: channel order and the active-channel count set `recommend_action`'s summed cluster term, so
GL3.B1 preserves both, plus the empty-read rule. *Cross-thread `vital_metrics` tearing*: tracks fix
ordering, not snapshots (**UNVERIFIED** impact) until GL3.B8. *Two threads in `evaluate_failures` under
`--sim`* (§3.2): unlocked today; the resume stage's sequencer lock (G17) guards ids only, and GL3.B3 moves the orchestrator
thread behind the inbox. *Bio over-claim*: pass latency is ordinal only. *Divergence*: if two consecutive stages
each surface a new failure mode, stop and audit the layer beneath (the body sensor resolvers, #1124 /
#1156 / #1159, are the likely layer).

---

## 8. Dependency on the decomposition fence

`runtime/agent_loop.py`, `runtime/loop_*.py` and `simulation/orchestrator.py` belong to **Session A**'s
1.3.2 decomposition (`agent_loop`'s phase-1 slices are built: slice 4 merged as #1187, slice 5, the perception
sections → `loop_perception.py`, built 2026-10-10; `start_simulation_mode` remains). Under G1, src
work waits for the fence, with one exception, GL2a, which lives outside it
(`maxim/embodiment/event_id.py`, which this plan imports, is built at the autonomic plan's post-fence
resume stage, G17, not GL2a). For this plan:

| Stage | Touches | Waits for |
|---|---|---|
| GL3.B0 | tests + `_loop_harness` only | nothing; coordinate harness changes with Session A |
| GL3.B1 | `substrate_proposal.py::_encode_current_clusters` / `propose_via_substrate` / `_attach_live_situation` (the body path, registry resolved from `executor`, TR21); `similarity/encoder.py` (`provenance=` on both encoders); `integration/memory_hub.py`; `imagination/trigger.py`, `orchestrator.py` self-entity encode; `perception/`'s `registry_for` (TR21); the body writers and their callers (`embodiment/body.py`, `embodiment/spec.py` (`SpecModulator.apply_damage`), `embodiment/tool_bridge.py`, `embodiment/audio_localization.py`, `embodiment/backends/minecraft.py`, `hardware/reachy/motor_backend.py`, `runtime/loop_perception.py` (fenced), `simulation/tools.py`, `simulation/cradle_mother.py`, `simulation/dm_runtime.py`, `simulation/generative_runner.py`, `simulation/foundry.py`; the last-writer kind map beside `Entity.vital_metrics`, TR22, §5.7 item 1; `simulation/` is Session V's, coordinated); the `encode_sensors` and `ModulatorAffordanceTool(` callers in `scripts/` and `tests/` (§5.7 items 1 and 3); `scripts/survival_world/water_trial.py` (the apparatus window, T1-13/14/15/R3/B1 walked); EC / NAc / `runtime/tool_dispatch.py::record_outcome` / the PainBus subscribers / `hivemind/bundle.py`, `merge.py`, `ingest.py`, `entry_index.py`; `scripts/lint_relay_only_encode.py` (M39) | `agent_loop`'s phase-2 slices (the LLM-primary sections the body path crosses, per `roadmap_1_3_x.md`) **and** the `start_simulation_mode` slices; the post-fence resume stage (G17: the pid, the sequencer); a provenance consumer (G8) |
| GL3.B2 | none here | GL4 S1 (the producer; GL4 wiring S8) |
| GL3.B3 | `loop_setup.py::build_loop_run`, `loop_substrate.py`, `loop_gates.py::pre_tick_gate`, `agent_loop.py` (H4), `embodiment/body.py`, `embodiment/tool_bridge.py` (H3t), `simulation/tools.py` (the narrator tools post to the inbox), `simulation/orchestrator.py` and `integration/bio_enrichment.py` (the inbox route) | the orchestrator slices; GL2b. (`agent_loop` slice 4's PLANNING arm, #1187, and slice 5's imagination / auto-sense / audio / `state.update` arm, `tests/unit/test_loop_perception_characterization.py`, 2026-10-10, are in place.) |
| GL3.B4 | `agent_loop.py` §4, `substrate_proposal.py` | as GL3.B3, plus a declaring rung arm, the re-frozen **v2** latency record (`afferent_latency_characterization_v2.json`, measured on the post-#1176 tree; v1 is history; delta round, Executor N-6) and the per-pass transduction prerequisite landed unflagged in #1176 (TR19, TR28) |
| GL3.B5 | `orchestrator.py`, `conversational_source.py`, `minecraft.py` | the orchestrator slices |

The behaviour-preservation gates the slices already pin (`test_agent_loop_selection_golden.py`,
`test_decision_provenance.py`, `test_encoder_golden_v1.py`) are exactly the gates GL3.B1 and GL3.B3's
flag-off arm reuse. For the survival rows, the check must **execute the producer**: the scripted
water-trial smoke (`scripts/survival_world/scripted_water.py`, which paces a socket bridge with the loop
on wall time, #951, so it is timing-sensitive on a loaded CI runner) and
`tests/unit/test_water_trial_smoke.py::test_water_trial_ticks_acts_and_the_staging_close_persists_fear`.
(`tests/unit/test_exp61_run.py`, cited here before, runs no loop and is not such a check; environment
N1.) A
reproduction of the Exp 60/61/R3 verdicts through `compute_verdict` is not such a check: it re-reads the
committed JSONL and never runs the changed code. Nothing in this plan edits these gates.

---

## 9. Second-body re-keying (G4)

The "deferred until a second body exists" gate mixed two arguments. **Robot-factory honesty**
(`deferred/second_body_staging.md` Stage B: do not design `hardware/controller.py::RobotController` and the
`maxim.robots` entry point from one robot) is about motion and hardware. Minecraft does not satisfy it,
because it is a SEM modulator backend (`MinecraftWorldBackend` via `minecraft_modulator_factory`), not a
`maxim.robots` controller. **That trigger stays.** **Perception-abstraction honesty** is already satisfied:
at least five live sources with different receptor classes, spaces and clocks (drives,
Minecraft world state, Reachy DoA, narrator and Minecraft text, DN vision). Corrected 2026-10-24
(environment S7): they use **two** encoders (`sensor_sum`, linguistic); "vision" is a sentence embedding
of percept text (§2 gap 8); DN vision is live only on a robot with detections; and Minecraft text
reaches EC only with `MAXIM_SUBSTRATE_PATH=1`. The re-keyed fabric trigger below can also fire before
the join its binding needs exists (DoA's two lanes share a pid only at GL3.B8; text↔sensor joins at
GL3.B5): whether to amend its wording is an open owner decision (environment S7, §11 TR27; strict: add
"AND the join the binding needs exists"), not changed here because the wording is shared by every banner. The fabric's own Stage 0b/0c
never needed a second body either; they run on the Reachy.

Re-keyed triggers. This wording is the one used everywhere (banners, README, roadmap, grounding.md):

| Plan | New trigger (capability, fires at GL3) |
|---|---|
| `deferred/cross_modal_perception_fabric.md` | Revives when **GL3's registry+provenance stage ships AND a 1.4 rung needs cross-modal binding** (GL3.B1 carries the event identity and provenance the binding's join key needs). Until then it is an input to this plan (binding convention, two-level attention, artifact contract), not a plan of record. |
| `deferred/perception_pipeline_placement.md` + `runtime/perception_placement.py` (Dormant type layer) | Revives when **GL3's registry+provenance stage ships AND a stage is placed across a wire** (GL3.B1). The `src` Dormant docstring's trigger changes in the same PR (a CHANGELOG line). |
| `deferred/modality_resolution_and_alignment.md` | Revives when **GL3's registry+provenance stage ships** (GL3.B1). Its discriminability facts are inputs. |
| `deferred/second_body_staging.md`, the microduck, the orient line (incl. `hybrid_substrate_reflex_runtime.md`) | **Unchanged:** physical-robot trigger. One added line: "the grounding line's Receptor is a percept receptor, not this plan's robot 'engine seam'." |

The naming follows: `Receptor`, never a bare `Engine`, so it cannot collide with the robot "engine seam",
`VisionEngine` or `TTSEngine`.

---

## 10. Mechanization rows this plan relies on

| Row | Rule | Check |
|---|---|---|
| M39 | Every percept enters EC through the registry | AST lint with an allowlist naming each current caller and its reason (the M38 pattern), `src/maxim/` only. Lands **in the GL3.B1 PR** (TR22); its deletion probe is GL3.B1 gate 5. |
| M40 | Tracks are logical, never OS threads | grep/AST lint over `perception/`, hardware-edge adapters allowlisted. Lands with GL3.B3. |
| M10 (amended 2026-10-07) | No grounding flag is active in an E1–E3 arm unless the arm declares it | M10's list names GL3 track preemption; under wiring S12 `runtime.afferent_tracks` is GL3.B3's fan-out key and GL3.B4's preemption flag is its own key; the GL3.B1 PR adds the credit-check flag to it. Under wiring S12 the fan-out, preemption and credit flags are three keys, stamped into `report.json` and asserted OFF by the T1 re-run harnesses too; the row text in `outstanding.md` changes to match. |

The pid-and-cause rule and the no-provenance-default rule need no backlog row: they are **structural
guards** (`AfferentEvent.__post_init__`, `PhysicalEventId.__post_init__` from the resume stage, the sentinel rejections),
cited as such in §5.10.

---

## 11. Open owner decisions (each asked at the named stage's start; strict option is the default recommendation where one exists)

Decided, not re-opened: G6 (narrated is discounted, never experienced; the discount value is GL4's
start decision), G8 (registry with provenance; GL3.B4 a declared arm or waits; L1/L2/L5/L7 filed as
#1176–#1179), and the identity contract (`PhysicalEventId` (agent, session id, seq), built with its
sequencer, session-id source and cross-session resume at the post-fence resume stage, G15 and G17; one
seq authority per agent; the orchestrator thread a declared edge). TR1, TR2 (2026-10-09) and TR18–TR24
(2026-10-24, the T8 (b) review, DECISIONS.md G24) are decided too; the rows below record them.

| # | Decision | At | Recommendation |
|---|---|---|---|
| TR1 | Dead `PreemptionCircuit` / `ExecutionTracker` / `wire_preemption` / `check_hold` / `capture_before` (L7): `Dormant since` or delete | GL3.B0 (via the L7 issue) | **Decided 2026-10-09 (owner): Dormant** (`Dormant since 2026-10-07`; callers and re-exports stay), carried out by #1179. |
| TR2 | L2: may the turn budget delay nociception, or only action? | GL3.B0 (via the L2 issue) | **Decided 2026-10-09 (owner): only action.** Red gate (e) encodes it; #1177 fixes it. |
| TR3 | Does a byte-identity-proven GL3.B1 fire T1-6/10/11/12/15 (+13/14)? | GL3.B1 | Strict: yes, re-run offline guards, committed exception for rig re-runs. The structural discharge is the open alternative. |
| TR4 | Home and names: `perception/`, `ReceptorRegistry`, `TrackScheduler` | GL3.B1 | Yes (the design pass reserved `perception/`). The package gets an ARCHITECTURE.md §Layer Ownership entry and an `architecture_baseline.json` decision, import direction `perception → embodiment` (wiring N2). |
| TR5 | Body-sensor specs derived from YAML (recommended) or authored; each new YAML key ships with its reader | GL3.B1 | Derived. |
| TR6 | Track numbers: latency, priority, preemption rights, gain, refractory | GL3.B3, GL3.B4 | As tabled in §5.4; only `nociceptive_fast` / `affective_slow` built; the preemption right only at GL3.B4. |
| TR7 | Latency in passes (ordinal) or experience µs | GL3.B3 | Passes. |
| TR8 | Protective action after preemption: same-pass reselect, or reflex motor program first | GL3.B4 | Same-pass reselect, with the double-`note_active_clusters` question settled by the gate design pass. |
| TR9 | (Settled by the identity contract as amended by G15 and G17: no pid at GL2a; the type, the sequencer and its session-id source are built, persisted per agent and resumed past the saved maximum at the post-fence resume stage, before GL4 S1.) | n/a | n/a |
| TR10 | Do GL3.B3, GL3.B4 and GL3.B6 fire T3-9? | GL3.B3, GL3.B4, GL3.B6 | Strict: yes; walk and re-run Exp 09's offline guard. |
| TR11 | Provenance kinds beyond experienced / narrated / imagined: add `declared`? `reported`, or `narrated` + `caused_by`? | GL3.B1 | `declared` yes (the planning drafts' recommendation); `narrated` + `caused_by` (this plan's lean; no draft recommendation). |
| TR12 | Refuse vs discount credit for **imagined** nodes (narrated is decided: G6); with it, whether a refused node's strength is dropped before or after `distribute`'s normalization (wiring D3) | GL3.B1 | Strict: refuse (tiered an engineering prior over an oracle label, bio-faithful S3; acting on it behind the credit flag, TR20); drop after normalization, so a refusal never raises another node's credit. |
| TR13 | Credit-check flag default; how `legacy` (pre-GL3.B1) nodes are treated when it flips | GL3.B1 | OFF until a separate, owned flip with its re-runs, after the G6 discount value exists; `legacy` credited as today (refusing it would silently change every earned agent's credit at the flip). |
| TR14 | Provenance dict vs `"text.imagined"` EC namespace | GL3.B1 | Dict. |
| TR15 | `SensoryTag.perceived_intensity`: mark dead, or wire from `relay_gain` | GL3.B5 | Owner's pick; not a gate. |
| TR16 | Minecraft health/food/oxygen double entry: keep both, and record the drive side as derived (`caused_by` the world event) once GL2 owns the drive receptor | GL3.B1 / GL2 | Keep both. Never de-duplicate inside GL3.B1. |
| TR17 | Which 1.4 rung declares GL3.B4, and which consumer GL3.B1 lands with | GL3.B4, GL3.B1 | The owner names them (G8). |
| TR18 | T8b-1 (bio-faithful D1, confounding D4, environment S1; GL3.B3): where PainBus publishes under the fan-out | GL3.B3 | **Decided 2026-10-24 (owner, T8 (b) review; DECISIONS.md G24):** every PainBus publish stays at the transduction pass (`drive:oxygen` keeps its in-place synchronous publish in the W-4 order); `affective_slow` adds deliveries only to sinks that are not PainBus subscribers (no cluster fear, no memory tag); invariant "each learning sink sees a pid once, keyed on the situation at its transduction pass", with a count test; the §5.4 / §5.8 contradiction resolved in writing; flag-on drowning-arm gates (Exp 60 / T1-13). Folded in §5.4, §5.5, §5.8, §5.10, §6 GL3.B3. |
| TR19 | T8b-2 (bio-faithful D2, confounding D1, environment D1, wiring S10; GL3.B4): H2's event source and the latency gate | GL3.B4 | **Decided 2026-10-24 (owner, T8 (b) review; DECISIONS.md G24):** per-pass body transduction, independent of selection, before H2, is a named prerequisite of GL3.B4, settled with #1176 (L1), keeping the W-4 order (*the pre-merge review round's addition, not part of G24; its unflagged half ratified by the owner as TR28 / G28:* landed unflagged in #1176's own PR as the defect fix, the record re-frozen as v2 after it, the gated claim action latency given publish, publish latency reported; the negative control, first written as the pre-#1176 v1 record, is produced on the post-#1176 tree by a test-only monkeypatch per the delta round); gate on flag-on vs flag-off on the same tree; a sham drop-and-reselect control and the negative control (no pre-§4 transduction → no preemption); gate 3 split into publish latency and action-latency-given-publish, gated in passes and seconds, on both frozen arms (strictly lower on the named arm, no worse on the other). Folded in §3.4, §4.2, §5.6, §6 GL3.B4. |
| TR20 | T8b-3 (confounding D3, wiring D3; GL3.B1): is the encode-time eligibility change flagged? | GL3.B1 | **Decided 2026-10-24 (owner, T8 (b) review; DECISIONS.md G24):** record provenance always, but act on it (the imagined eligibility refusal, the narrated discount) only behind the same OFF credit flag; an eligibility golden runs the narrator and imagination paths; red gate (a)'s read-surface change is recorded as the owner-approved re-point GL3.B0's banner requires. Folded in the header, §5.7, §5.9, §6 GL3.B0 note and GL3.B1 gates. |
| TR21 | T8b-4 (wiring D1; GL3.B1): where the registry lives | GL3.B1 | **Decided 2026-10-24 (owner, T8 (b) review; DECISIONS.md G24):** bound to the agent's primary `Embodiment` (not `LoopRun`), so the water-trial trainers and red gate (g)'s surface reach it and the slice-3 pin stays intact. Folded in §5.3, §5.8, §6 GL3.B1, §8. *Pre-merge round (Executor B1, Wire B2):* the bind is one named binder, `perception/`'s `registry_for(embodiment)`, called from `runtime/`, deriving and caching the registry on first use; the typed error only for an `Embodiment` with no body YAML. *Delta round (Executor S-A, Wire D-1, Architecture SF-2, N5):* the derivation reads the root entity tree's declared sensors (`Embodiment.root` and its components), not a YAML path, which no `Embodiment` or `Entity` records; the typed error fires when that tree declares no sensors; built once per `Embodiment` under an `RLock`. |
| TR22 | T8b-5 (wiring D2, confounding D2; GL3.B1): provenance on the crediting writers, `apparatus`'s producer, the deletion probe | GL3.B1 | **Decided 2026-10-24 (owner, T8 (b) review; DECISIONS.md G24):** provenance reaches the writers that actually credit on the body path (`SensorEncoder.encode_sensors`, `record_outcome`'s cluster reward, the PainBus subscribers); `apparatus` gets a named producer (the harness's rescue/heal window, consumed by the mixed-state rule); the deletion-probe / byte-identity contradiction is fixed by landing the M39 lint in the GL3.B1 PR plus a per-receptor route counter. Folded in §5.3, §5.7, §5.10, §6 GL3.B1, §8, §10. grounding.md's GL3 row needs the same fix (the main session's edit). *Pre-merge round (Executor S2, S5, S7; Wire SF-4, SF-6, SF-7):* M39's allowlist seeded from `src/` without the moved sites, the counter on both entry points; every `encode_sensors` caller edited; the EC lifecycle sites; the `WaterTrial` window and T1-9 walked; the mother's credit recorded, decided by the T1-9 walk. |
| TR23 | T8b-6 (environment D2, wiring D2; GL3.B3): how a narrated write keeps its stamp when the loop transduces it first | GL3.B3 | **Decided 2026-10-24 (owner, T8 (b) review; DECISIONS.md G24):** per-sensor last-writer provenance stamped AT WRITE TIME (`world_set_axis` already takes `owner=`; it must be stored), read by every `evaluate_failures` caller when it mints the event; gate 6 runs on llm-primary with the drift tick live. Folded in §3.2, §5.5, §5.7 item 1, §6 GL3.B3. *Pre-merge round (Wire B1, SF-5; Executor S1):* the map lives beside `Entity.vital_metrics`, not on the `Embodiment`; the writer kind is a `ModulatorAffordanceTool` constructor field; a derived value inherits its source's kind; the initial kind is `unknown`, credited as `legacy` (superseded by TR29: no `unknown` kind; a missing slot reads `legacy`). *Delta round (Wire D-2, D-3):* one kind-taking setter, raw writes linted (M78); the reader resolves through `_resolve_sensor_slot`; the map persists through `Entity.to_dict` / `from_dict` with a format-version bump. |
| TR24 | T8b-7 (environment D3; GL3.B3): the burn's harness and hook | GL3.B3 | **Decided 2026-10-24 (owner, T8 (b) review; DECISIONS.md G24):** the harness is `_cradle_loop_driver.py` on the GL2b variant body, substrate-primary, an experienced touch; a tool-path drain hook (H3t) on `ModulatorAffordanceTool.execute`'s path; the `MAXIM_DETERMINISTIC_SCENE_EMBODIMENT` double-count fixed. Folded in §5.5, §6 GL3.B3. |
| TR25 | Red gate (b) (the Minecraft event percept's `agent_id`) flips at GL3.B1, which does not touch `minecraft_harness.py` or `minecraft.py` (confounding S6, wiring S6) | GL3.B1 | Strict: add the F0.5 `agent_id` pass-through through `build_minecraft_aut` to GL3.B1's build, so (b) flips as written with no re-point. Alternative: an owner-approved move of (b) to GL3.B5 (or its own F0.5 issue). |
| TR26 | Per-node provenance inside or outside `hivemind/entry_index.py`'s signed `EC_NODE_FIELDS` projection (wiring S4) | GL3.B1 | Strict: inside, with an `INDEX_VERSION` bump (outside, the field is unsigned and a per-entry dedup merges a narrated and an experienced node). |
| TR27 | The re-keyed fabric trigger (§9) can fire before the join its binding needs exists (environment S7; recorded as a row by the pre-merge Architecture N3) | GL3.B1 (the stage whose ship the trigger names) | Strict: add "AND the join the binding needs exists" to the trigger, changed in every banner that carries it at once. |
| TR28 | Is #1176's per-pass body transduction (GL3.B4's prerequisite, TR19) flagged? (delta round, Architecture SF-1, Executor B2) | #1176 | **Decided 2026-10-24 (owner; DECISIONS.md G28):** it lands **unflagged** in #1176's own PR as a root-cause defect fix (CLAUDE.md "no band-aid"), with its own trigger walk (per-pass encodes move NAc's per-pass inputs on T1-13/14/15's path) and the T1-13/14/15 re-runs that walk owes; after it lands the latency record is re-frozen as v2. Folded in the header, §3.4, §6 GL3.B4, §8, TR19. |
| TR29 | The initial writer kind of a body slot with no writer yet (delta round, Architecture SF-4, Executor S-C, Wire D-3) | GL3.B1 | **Decided 2026-10-24 (owner; DECISIONS.md G29):** no `unknown` kind. A **missing** writer-kind slot means "not yet written" and the reader maps it to `legacy` (credited as today); nothing can default to `experienced`. Experienced hunger drift reads `legacy` until the agent's own writer touches it, and that share is reported (GL3.B1 gate 4). Control (g′): a drift-only encode on a fresh body carries `legacy`, never `narrated` or `experienced`. G16's vocabulary is unchanged. Folded in §5.2, §5.7 items 1 and 2, §5.10, §6 GL3.B0 note, TR23. |
