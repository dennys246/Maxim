# GL3 design review — WIRING lens (thalamic_relay.md)

Reviewer lens: real consumers and the real credit path (D43); right encoding and seams; no
hand-composed shortcut that passes while the shipped loop fails ("a fix ships with a caller"). Read at
worktree `.worktrees/e-t8b`, `origin/main d9dffe6e`. Read in full: `docs/plans/thalamic_relay.md`
(1,217 lines). Read for context: `docs/experiments/DESIGN_REVIEW.md`, `docs/plans/grounding.md`
§Owner decisions (G1–G20 taken as settled, not re-litigated), `docs/plans/reviews/grounding_gl1/wiring.md`
(format), `docs/wiring/README.md` + `body-and-word-worlds.md`, ARCHITECTURE.md §Layer Ownership,
`docs/plans/outstanding.md` rows M10/M39/M40, and `src/maxim/utils/function_length_baseline.json`.
Every code claim below was read at `file::symbol`. Runtime probes (read-only): the four GL3.B0 test files
(`9 passed, 8 xfailed`), and a fresh run of the GL3.B0 latency instrument against today's tree (both arms
EQUAL to the frozen record). Anything else is marked UNVERIFIED. Issue states were read with
`gh issue view` (#1176/#1177/#1178 OPEN, #1179 CLOSED, #1200 OPEN, #1201 MERGED).

**Verdict: 3 DO-NOT-BUILD / 12 SHOULD-FIX / 6 NIT.** All three DO-NOT-BUILDs are on GL3.B1, the first
`src/` stage. Each one can be folded with a plan edit before code. D1 and D3 each carry an owner
decision with options, strict option first. None re-opens G1–G20. GL3.B0 as built matches its claims
(see §Verified). Recommended reading order: D1 (where the registry lives decides most of the rest), D2,
D3, then S5, S6, S7, S10.

---

## DO-NOT-BUILD

### D1 — GL3.B1: the registry's home (`LoopRun`) cannot reach the shipped callers of the body path, and the B0 red gate's read surface already rules it out

**Stage:** GL3.B1 (§5.3 "built once per agent, carried on `LoopRun` beside the scheduler"; §5.8 row 1
"`_encode_current_clusters` and `propose_via_substrate` call `registry.tick`"; §8 GL3.B1 row).

**The issue.** The plan puts the registry on `LoopRun` and has the two body-path encoders call it. No
stage says how those two functions get hold of it, and every route the code allows breaks something the
plan says it keeps.

1. **The loop route reopens a pinned characterization test.** `runtime/loop_substrate.py::substrate_tick`
   takes explicit keywords, "never the loop's `LoopRun`" (its module docstring, rule (d)). The proposer's
   keyword set is pinned exactly:
   `tests/unit/test_loop_substrate_characterization.py::test_the_proposer_gets_the_runs_handles_by_identity`
   asserts `set(kw) == {"nac", "agent_id", "executor", "situation_cue", "sensor_encoder"}`. GL3.B1 has no
   flag, so a `registry=` keyword is passed on every run, the pin fails, and §6 says a stage "must not
   reopen a slice's characterization tests". The `policy_kw` "forwarded only when set" trick does not
   help, because the registry is always set.
2. **The EARNED rows' shipped path is not the loop.** The survival trainer calls the proposer and the
   outcome encoder directly, with no `LoopRun`:
   `scripts/survival_world/water_trial.py::WaterTrial` calls `propose_via_substrate(...)` in its training
   loop and `_encode_current_clusters(...)` for the world cluster. `exp60_run.py`, `exp61_run.py`,
   `exp62_run.py`, `exp62_precheck.py`, `r3_run.py`, `r3_pilot.py`, `b1_floor_run.py` and `scripted_water.py`
   all import it. More direct callers: `exp58_run.py` and `exp58_offline_gates.py` (both functions),
   `orient_backbone/exp53_cross_context_readout.py` (both), `r2_learned_bias.py`, `dark_danger_probe.py` and
   `instrument_check.py`. In `src/`, `substrate_proposal.py::_attach_live_situation` (llm-primary
   outcome time) is a third caller. §8 itself says the survival-row check "must **execute the producer**"
   (`scripted_water.py`, `test_exp61_run.py`). Under a `LoopRun`-only registry those producers either
   bypass it (so the EARNED rows run the path M39 exists to forbid, with no provenance stamp) or need a
   new argument at every harness call site (a harness change on T1-13/14/15's own instrument).
3. **Red gate (g), as written, already decides the question.**
   `tests/unit/test_afferent_red_gates_units.py::test_gl3b0_g_narrator_tool_consequence_reaches_ec_as_narrated`
   calls `_encode_current_clusters(SensorEncoder(...), "aut-1", SimpleNamespace(embodiment=emb))`. It
   passes no registry and no `LoopRun`. The B0 banner says the flipping stage "must flip them as written".
   So the provenance must be reachable from the executor's `embodiment`. That contradicts §5.3.
4. **The census does not look.** `tests/unit/test_receptor_census.py`'s docstring scopes out `scripts/`
   ("none injects into a running loop"). That is true, but it misses the point: the scripts *are* the
   loop for the survival rows. M39 (`outstanding.md`) is also `src/maxim/`-only.

**Why it matters.** This is the shipped-the-pieces failure in advance. A registry that only the
`run_agentic_loop` path reaches passes GL3.B1's selection golden (which runs through `_loop_harness`,
i.e. the loop), while Exp 58/60/61/62/R3/B1 keep encoding around it.

**Proposed fold — owner decision (strict first):**
- **(A, strict) Bind the registry to the executor's primary `Embodiment` at bind time.** Derive it from
  body YAML, which is what TR5 already recommends. `_encode_current_clusters` and `propose_via_substrate`
  then resolve it from `executor` with no new keyword: the pinned keyword set stays, every script caller
  is reached unchanged, and gate (g) flips as written. An embodied executor with no registry fails
  loudly; this is the silent-no-op rule, so it is a typed error, not a fallback. `LoopRun` carries only
  the GL3.B3 scheduler. Layering: `embodiment/` must not import `perception/`, so declare a `Receptor`
  protocol slot on `Embodiment` (see N2).
- **(B) A required keyword on both functions.** Every harness above gets a `TypeError` until it is edited.
  The slice-3 pin is reopened under a committed owner exception, and each harness edit is walked against
  its T1 row.
- Either way, §5.3, §5.8 row 1 and §8's GL3.B1 row must name the choice and list
  `_attach_live_situation` and the script callers. The census should gain a `scripts/` category,
  "body-path encode / propose callers", as the M39 allowlist seed.

### D2 — GL3.B1: provenance does not reach the writers that actually credit on the body path, and `apparatus` has no producer

**Stage:** GL3.B1 (§5.7 items 1–4; §5.2 `provenance_kinds`; gate 4; the contamination guard).

**The issue.** §5.7's rule is "every EC write carries a provenance kind", with credit gated by
provenance. The plumbing it specifies reaches neither the encoder the body path uses nor the credit maps
the body world learns through.

1. **The sensor encoder gets no provenance.** Item 3 makes `provenance=` keyword-only on
   `LinguisticEncoder.encode*` only, and §5.3 says the `sensor_sum` route through
   `SensorEncoder.encode_sensors` is "unchanged". But the narrated writes red gate (g) is about reach EC
   through `SensorEncoder.encode_sensors`, and that function itself calls `self._nac.update_eligibility(...)`
   (`similarity/encoder.py::SensorEncoder.encode_sensors`) with no provenance in scope. Its `min_delta`
   gate also returns the previous node id without touching EC, so a stamp placed after the scan is skipped
   on gated calls.
2. **No per-sensor last writer exists to stamp from.** §5.7 item 1's strict mixed-state rule ("`narrated`
   whenever any declared sensor's last writer was the narrator") needs a writer record. `vital_metrics`
   is a plain dict (`embodiment/audio_localization.py::world_set_axis` ends `vm[sensor_name] = v`).
   `owner=` is "a SELF-DECLARATION for logs and review, not an ACL" (its docstring) and is not stored.
   `SetEntitySensorTool.execute(source="narrator")` stores nothing either. The narrator's preferred tool,
   `OrchestratorActorTool.execute`, writes the AUT body through an ephemeral
   `embodiment/tool_bridge.py::ModulatorAffordanceTool` `target_effect`, i.e. through `_apply_sensor_deltas`.
   That is the same helper the AUT's own (experienced) actions use. GL3.B1's touch list (§8) names none of
   `embodiment/body.py`, `embodiment/tool_bridge.py` or `audio_localization.py`.
3. **The credit check guards the wrong maps.** Item 4 names `NAc.credit_node` and the
   `TemporalCreditDistributor` fan-out, which together write the recognition `_reward_bias`. The body
   world's earned learning goes elsewhere:
   - `runtime/tool_dispatch.py::record_outcome` credits `NAc._cluster_reward_bias` keyed on the proposal's
     `clusters`, which are EC node ids from the narrated or experienced reading;
   - `proprioception/pain_bus.py::build_pain_bus` subscribes `create_pain_nac_subscriber`
     (`record_outcome_full`), `create_percept_valence_subscriber` and `create_pain_cluster_fear_subscriber`
     (`NAc.record_cluster_fear`, the Exp 58/60/61/62 mechanism). `DamageComponentTool.execute` publishes a
     `PainSignal` straight onto that bus, and `PainSignal` has no provenance field;
   - `NAc.credit_operant_reward`, which G16 explicitly names.

   So under G6 a narrated burn still writes cluster fear and cluster reward at full weight with the check
   ON. The contamination guard's claim, "the discount is applied wherever narrated data trains or
   credits", has no enforcement point on these paths.
4. **`apparatus` (G16) has no producer.** The water trial's rescue is an RCON teleport, a settle and
   `heal()` (`WaterTrial.rescue`, called right before `submerge` in the training loop). It comes back
   through the Minecraft pump as `world_set_axis(owner="minecraft_bridge")`, which is the same receptor
   and the same writer as an experienced reading. The cradle mother writes hunger from `main`
   (`simulation/cradle_mother.py::reactive_mother_tick`, a census row). No `ReceptorSpec` declares
   `apparatus`, and nothing at the relay can tell those writes apart. GL3.B1 would therefore stamp G16's
   excluded writes `experienced`, which is wrong by construction. Its consumers, GL5 and the forward
   model's contamination guard (the reason G8 lets GL3.B1 land at all), would inherit the error.
5. **The narrated discount does not exist yet.** Item 3 says narrated eligibility is applied "at the
   declared discount", but G6 sets that value at GL4 start, after GL3.B1. The plan gives no weight for the
   interval.

**Proposed fold (plan text):**
- Move provenance to the **write seam**. One per-sensor writer-kind slot lives on `Embodiment`, written by
  every `BODY_WRITERS` site in the census (`world_set_axis`, `_apply_sensor_deltas`, `_write_sensor`,
  `apply_damage`, the cradle mother, the DM cascade). Each site takes a keyword-only kind, so forgetting it
  is a `TypeError`. Apparatus writes are marked by the harness, e.g. an apparatus scope on the
  embodiment around `rescue` / `heal` / respawn. Name the drift rule: a drift step on a narrated value
  keeps `narrated`. Otherwise one tick of drift relabels a narrated value experienced, which G6 forbids.
- `SensorEncoder.encode_sensors` takes `provenance=` keyword-only and records it even on the `min_delta`
  short-circuit.
- Replace item 4's list with the full credit-writer table: `credit_node`, `distribute`,
  `record_outcome` cluster credit, the four PainBus subscribers, and `credit_operant_reward`. For each,
  state whether GL3.B1's flagged check covers it, or which stage does, and with which red gate.
- State the GL3.B1 interim: narrated eligibility stays at full weight (today's behaviour) until the G6
  value exists.
- Add `embodiment/body.py`, `tool_bridge.py`, `audio_localization.py` and `simulation/tools.py` to §8's
  GL3.B1 row.
- If the owner prefers to keep GL3.B1 smaller: **owner decision**, (A, strict) the above at GL3.B1, or
  (B) GL3.B1 claims only "provenance on text and imagined nodes". Under (B), G6/G16 enforcement on the body
  path moves to a named later stage with its own strict red gates landed now, and GL5 may not consume
  body-path provenance until then.

### D3 — GL3.B1: "flag default OFF, goldens byte-identical" does not hold structurally; the imagined-eligibility change is unflagged and moves credit on the shipped path

**Stage:** GL3.B1 (header: "Every behaviour change ships behind a flag that defaults OFF, with today's
goldens pinned byte-identical while the flag is off"; §5.7 item 3; gate 1; red gate (a)).

**The issue.** GL3.B1's only flag is the credit check. Item 3 removes eligibility for `imagined`
unconditionally, and red gate (a) as built (`test_gl3b0_a_imagined_affordance_encode_creates_no_eligibility`,
no flag) can only flip if it is unconditional. That change is not inert:
`decisions/temporal_credit.py::TemporalCreditDistributor.distribute` normalizes "so total credit =
reward" across every eligible node, and `runtime/bio_stack.py`'s `_distribute_reward_from_reaction`
calls it on every Reaction (`reaction_bus.subscribe_all`). Taking imagined nodes out of the eligible set
**re-allocates the same reward onto the remaining nodes**, the agent's sensor and text nodes, in every
bio-stack run with an `ImaginationTrigger` (every `--sim` AUT). `_reward_bias` lowers that node's EC
threshold (`NAc` docstring: "threshold_override = base − reward_bias(agent_id, node)") and gates
`integration/bio_enrichment.py` / `tools/discovery.py`, so cluster identity and enrichment can move.
Gate 1's golden runs through `_loop_harness`, which builds no imagination trigger, so it cannot see this.
Gate 4 checks credit only flag-on. The ec.json byte change with the flag off is declared (§7). This one
is not.

**Proposed fold — owner decision (strict first):**
- **(A, strict)** Put the write-path eligibility change behind the same OFF flag as the credit check (one
  flag, named in M10). Record an owner-approved change to red gate (a)'s read surface so it flips under
  the flag-on arm, as the B0 banner requires for any change. Add a flag-off golden over a composition
  that includes an `ImaginationTrigger` and a Reaction (deletion probe: remove the flag guard and that
  golden fails).
- **(B)** Keep it unconditional, strike "every behaviour change ... behind a flag" for GL3.B1, declare it
  a credit-path change for `--sim`, and walk T3-9 / the T1 rows' "`TemporalCreditDistributor`
  credit-path change" wording (T1-13) at build, not at a later flip.
- Also specify whether a refused node's strength is dropped **before** normalization (which
  re-allocates) or **after** (total credit falls below `reward`). The plan never says, and the two are
  different behaviours.

---

## SHOULD-FIX

### S1 — GL3.B1: pid granularity on pull reads is undefined; §8 still cites "GL2a (the pid)"

`AfferentEvent.__post_init__` "raises if [pid is] None", and GL3.B1's `tick` builds an event for every
pull read: three channels per substrate tick, plus every `_encode_current_clusters` call at outcome time
and in the scripts. So pids are minted per *read*, not per physical event. The burn in §5.4 then has a
pull-read pid on the EC route (GL3.B1) and a transduction pid on the track route (GL3.B3), and no rule
links them. GL3.B2's gate, "every row's pid resolves to exactly one `AfferentEvent`", becomes trivially
true and means little. §8's GL3.B1 row says it waits for "GL2a (the pid)". G17 moved the pid to the
post-fence resume stage, and §5.2 says so too, so the table contradicts the plan's own text. The G17 rule
"a rebuild fails loudly" also meets gate (g) (an ad-hoc `Embodiment`) and the scripts that rebuild one
agent (`r3_run.py`, `exp61_run.py`). **Fold:** define pid granularity. Recommended: pull reads carry
no pid at GL3.B1, because EC provenance is per node, not per event, so `pid` is required only for
`emit`ted events. Alternatively, a pull read joins its transduction through `caused_by`. Fix §8 to "the
post-fence resume stage (G17)". Say which `Embodiment` holds the sequencer for gate (g) and the scripts.

### S2 — GL3.B1: push receptors have no route until GL3.B3

§5.3 gives a receptor a pull reader or "a push adapter ... which posts into the scheduler's edge inbox".
The scheduler is GL3.B3's. GL3.B1 nonetheless turns the imagination and own-body affordance sites into
receptors. These are event-driven (`imagination/trigger.py::ImaginationTrigger._encode_entity_affordances`,
reached on the loop thread and on `main` from the fixture runner per the census;
`encode_entity_affordances` from `start_simulation_mode` before `sim.aut` starts). Routing them through a
future inbox would defer imagined encodes to the next pass's H0, which is a behaviour change. **Fold:**
add a synchronous, caller-thread route to the GL3.B1 API (`registry.encode_event(receptor_id, payload,
*, provenance)`), name its thread contract for `main`, and keep the inbox for GL3.B3.

### S3 — GL3.B1: the `provenance=` keyword-only change has callers the touch list misses, and text percepts need a kind before TR11 is decided

`LinguisticEncoder.encode(percept)` calls `encode_decomposed` internally (`similarity/encoder.py`).
`integration/memory_hub.py::MemoryHub.on_percept_received` calls the encoder on the EC route under
`MAXIM_SUBSTRATE_PATH=1`, so it is the CLAUDE.md `test_memory_hub.py` trigger. The scripts
`grounding_census.py` and `diagnose_roy_paraphrase_collapse.py` call `encode_decomposed`. All of them
`TypeError` at GL3.B1, and §8's GL3.B1 row names none of them. MemoryHub's percepts have to pass *some*
kind at GL3.B1: narrator, DM, CLI human text, `[minecraft:damage]` (TR11 `reported`?), vision-as-text.
But the text receptors that would own that decision land at GL3.B5. **Fold:** add
`integration/memory_hub.py` and the internal and script callers to GL3.B1. Either move TR11 and a
per-percept kind table into GL3.B1's start decisions, or keep the text route's kind at a declared
interim value that the credit check treats like `legacy`.

### S4 — GL3.B1: an exchange path the per-node provenance set must enter is missing

§5.7 item 2 lists `EC.save/load`, `_BUNDLE_EC_NODE_FIELDS`, `ec_merge_aligned` and ingest. It misses
**`hivemind/entry_index.py`**: the Queen-signed v2 per-entry digest projects
`EC_NODE_FIELDS = ("modality", "embedding", "geometry", "count", "domain")`. A provenance field outside
the projection is unsigned, so it can be changed in transit. Its docstring also says "Equal digests mean
equal CONTENT". A per-entry dedup would therefore treat a narrated node and an experienced node as one.
Also: `similarity/ec.py::EntorhinalCortex.substrate_node_metadata` (merge.py's documented input shape,
which today omits it, so gate (g)'s read surface needs it added); `analysis/substrate_diff.py`; and the
staged-donor readers in `scripts/survival_world/exp61_run.py`. **Fold:** add `entry_index.py` (in or
out of the projection is an owner call: strict means inside, with an `INDEX_VERSION` bump) and the
accessor to item 2, and add an entry-index round-trip to gate 2.

### S5 — GL3.B1: `registry.tick`'s return shape drops a live consumer, the B1 floor policies

§5.3 says `tick` returns "the same dict `_encode_current_clusters` returns", `{modality: cluster_id}`.
But `propose_via_substrate` first reads `channel_values` for every channel and hands them to
`_select_action`. For a `SubstratePolicy`, that is
`observation=MappingProxyType(channel_values.get(WORLD_TAG))` (`substrate_proposal.py::_select_action`),
the raw world reading the BF2 floor policies select on (`scripts/survival_world/floor_policies.py`, the
`select` methods; `b1_floor_run.py`). Those policies are what T1-13/14/15's PARTIAL status rests on. The
two functions also interleave differently: one reads all channels and then encodes, the other reads and
encodes per channel. That matters with the live pump writing between reads (§3.2, tearing). The
single-threaded `_ScriptedWorld` golden cannot tell them apart. **Fold:** `tick` returns
`(channel_values, clusters)`, and each caller keeps its current read/encode order. Add the floor policy's
`observation` to gate 1's golden tuple and a B1-floor-policy arm to the golden.

### S6 — GL3.B0/GL3.B1: red gate (b) flips at a stage that doesn't touch its code; gate (g) covers one of G6's three narrator tools

(b) reads `percept.context.agent_id` through `simulation/minecraft_harness.py::build_minecraft_aut`, and
§6 says it flips with GL3.B1. But GL3.B1's build and touch list contain neither `minecraft_harness.py`
nor `minecraft.py`. GL3.B5 is the stage that touches `minecraft.py` and registers `minecraft.events`.
(g) exercises only `SetEntitySensorTool`. A fix that tags the two `simulation/tools.py` classes flips it
while `OrchestratorActorTool`, which writes through `tool_bridge._apply_sensor_deltas`, and
`DamageComponentTool`, which writes through `apply_damage` plus a direct `PainSignal`, stay
`experienced`. That is a hand-composed pass while the shipped path fails. **Fold:** either add the F0.5
`agent_id` pass-through to GL3.B1's build list, or record an owner-approved move of (b) to GL3.B5. Add two
new strict red gates now, (g2) `DamageComponentTool` and (g3) `OrchestratorActorTool` targeting the AUT,
both flipping with GL3.B1. New gates are not re-points.

### S7 — GL3.B3: the narrator tools cannot reach the AUT's inbox, the write stays cross-thread, and the touch list is short

The narrator tools are built in `simulation/orchestrator.py::start_simulation_mode` against
`_aut_embodiment` (`orch_registry`, and the reflex instances assigned at
`aut_bio_enrichment_pipeline._reflex_damage_tool` / `_reflex_sensor_tool`) **before** `sim.aut` starts.
The AUT's `LoopRun`, and with it the scheduler, is built later on `sim.aut` by
`loop_setup.py::build_loop_run`. So the tools hold no reference to the scheduler unless it lives on the
`Embodiment` (D1 (A)) or is injected afterwards. Either way that is an `orchestrator.py` change, plus
`integration/bio_enrichment.py` for the reflex instances, and neither is in §8's GL3.B3 row. Moving only
`evaluate_failures` behind the inbox still leaves the *write* on the orchestrator thread
(`SetEntitySensorTool` writes `vital_metrics`; `DamageComponentTool` calls `apply_damage` and publishes a
`PainSignal` directly). The AUT's own substrate tick can then transduce the narrator's write first, as an
experienced pain on the loop thread, and the inbox entry later finds the latch already set. Gate 6 passes
under the lockstep harness while a real `--sim` race mis-stamps the pain. **Fold:** post the whole
*consequence* (the write plus its transduction, carrying `narrated`) into the inbox and apply it at H0.
Route `DamageComponentTool`'s direct `PainSignal` through it as well. Add `orchestrator.py`,
`bio_enrichment.py`, `agent_loop.py` (H4 sits beside `_loop_bio_tick_maintenance` in
`run_agentic_loop`) and `embodiment/body.py` (`evaluate_failures` emitting through the scheduler) to
GL3.B3's row.

### S8 — GL3.B3: per-caller drain points for `evaluate_failures` are undefined; "fast" can become slower than today

H3 covers only `propose_via_substrate`'s own `evaluate_failures`. The census lists more live callers:
`ModulatorAffordanceTool.execute` (inside §4 execute), `tick_embodiment_drift` (llm-primary), the reflex
dispatch inside `enrich`, and `foundry.run_gauntlet`. GL3.B3 builds H0/H3/H4 but **not H2**. So in
llm-primary a `nociceptive_fast` event has no drain until H4 (§8.5), after the pass's action. Today it
publishes synchronously, so the "fast" track would arrive later than it does now. If a pain caused by an
affordance inside `execute` is deferred to the next H3, it is drained **after** the next tick's
`note_active_clusters`. Wire 4 then books it on the next situation's clusters, which is the misfiling
class Exp 58 W-4 fixed. (`docs/wiring/body-and-word-worlds.md` already records that llm-primary never
calls `note_active_clusters`, so `affective_slow`'s "credit (cluster fear)" destination is inert there.)
**Fold:** add a table to §5.5 giving each `evaluate_failures` caller and where its latency-0 deliveries
drain. Recommended: latency-0 is drained in place at every caller, which keeps T1-4 and W-4 keying. Gate 2
should also run on an llm-primary arm.

### S9 — GL3.B3: `pass_index` has no source, and drains do not run on idle iterations

`latency_passes` and `refractory_passes` are "loop passes", but nothing says which counter. The loop is
`for step_num in step_iter` with `continue` on `GateOutcome.IDLE` (`runtime/agent_loop.py::run_agentic_loop`),
so `step_num` counts idle iterations. Substrate-primary is idle on most iterations: it wakes only on
`_substrate_tick_due` (0.5 s wall) or on pending work. H0 runs before the idle gate
(`loop_gates.py::pre_tick_gate`, right after `_loop_live_tick`), but H3 and H4 do not run on idle
iterations. `affective_slow`'s "2 passes" would therefore mean two iterations, a few milliseconds, and
it would actually be delivered at the next non-idle H4, whose timing depends on the wall clock. §3.1
already names the clock that fits: NAc decay runs per non-idle pass. **Fold:** define
`pass_index` = the non-idle pass count (the NAc-decay clock), stamped once at H0, and state that a due
delivery waits for the next non-idle drain until H5.

### S10 — GL3.B4: the frozen-record comparison is not attributable once #1176 lands first

#1176 (L1) is OPEN, and its first question is whether its fix is itself the preemption check (§3.4). If L1
is fixed in its own seam before GL3.B4, the damage arm's breach → protective latency falls against the
frozen record (`afferent_latency_characterization_v1.json`, damage: attempt `3` passes, `stale_executed`
non-empty). GL3.B4's gate 3 ("strictly lower than the frozen") then passes on the L1 fix, and gate 5's
deletion probe ("remove the H2 check and (3) goes red") does **not** go red, because L1's fix still
holds. The plan's "never a re-measurement" rule protects against a moved baseline, but it makes the
gate's attribution wrong. **Owner decision (strict first):** (A) GL3.B4's gate 3 compares against the
frozen record **and** against a re-measurement at GL3.B4's base commit with the same instrument blobs. The
deletion probe is read against the base-commit number. (B) Keep frozen-only, and require GL3.B4 to land
before any L1 fix, which contradicts G8's "fixable without the registry".

### S11 — all `src/` stages: Rule T ("touch it, shrink it") is not planned for

The stages edit functions pinned over 200 lines in `utils/function_length_baseline.json`:
`propose_via_substrate` (249; GL3.B1, H3), `SensorEncoder.encode_sensors` (262; D2's fold),
`Embodiment.evaluate_failures` (224; GL3.B3), `run_agentic_loop` (2,411; GL3.B3 H4, GL3.B4 H2),
`start_simulation_mode` (2,889; the self-entity encode at GL3.B1, the tool injection at GL3.B3),
`compose_bundle` (262) and `ingest_bundle` (437; GL3.B1). The plan predates the rule (2026-10-16).
**Fold:** for each stage, name the extraction that carries the edit. For GL3.B1 it is natural: the
per-channel read+encode loop leaves `propose_via_substrate` as `registry.tick`. Name any expected
`touch_exemptions` record up front.

### S12 — GL3.B1/B3/B4: the flags' reach and assertion stop at E1–E3 arms

`runtime.afferent_tracks` is a `maxim config` key, and `~/.maxim/` is shared across worktrees (CLAUDE.md).
M10 asserts grounding flags OFF only in **E1–E3 arms**. The T1 re-run harnesses (`exp60_run.py`,
`exp61_run.py`, `exp62_run.py`, `r3_run.py`, `b1_floor_run.py`) are not E-arms, so an operator's leftover
flag would silently change an earned-row re-run. The plan also gives GL3.B3's fan-out and GL3.B4's
preemption **one** flag name (§6 GL3.B3; M10's text names "GL3 track preemption (GL3.B4)" under
`runtime.afferent_tracks`). Turning on B3's fan-out to study it would then arm B4's preemption, which G8
reserves to a declaring arm. **Fold:** use separate flags for the fan-out, the preemption and the
credit check (GL3.B1). The effective values are stamped into every `report.json` (M1) and asserted OFF by
the T1 harnesses, not only by the E-rung harness. Amend M10's row text accordingly.

---

## NIT

- **N1 (§4.2).** "Six sites call encoders directly" double-counts. `orchestrator.py`'s self-entity
  encode *is* `imagination/trigger.py::encode_entity_affordances`. The census (`linguistic_encode` +
  `sensor_encode`) has five EC sites in `src/`, plus the internal `LinguisticEncoder.encode` and the
  non-EC `ComponentIndex._embed`. Cite the census instead of a count.
- **N2 (§4.2 / TR4).** The new `perception/` package needs an ARCHITECTURE.md §Layer Ownership entry and
  an `architecture_baseline.json` decision. ARCHITECTURE.md already gives `environment/` "observation of
  the world". Under D1 (A) the import direction must be `perception → embodiment`, never the reverse.
- **N3 (§3.3).** #1200 is now titled "Stage 3: a PainBus FRUSTRATION signal (Stages 1–2 shipped)". The
  census row "`record_tool_error` … no live caller, #1200" still matches the code, but Stage 3 will be
  the next new pain ingress, and it will trip `test_receptor_census.py`. Name it in §3.3 so GL3.B3's
  ingress walk expects it.
- **N4 (header, §3.5, §5.8).** #1179 is CLOSED and `runtime/preemption.py` carries the
  `Dormant since 2026-10-07` docstring. §3.5 still reads as a recommendation and §3.4 as "filed".
  Update both to "done (#1179)".
- **N5 (§5.10 / §10).** Say outright that M39's lint is `src/maxim/`-only, and list the earned-row
  harness encoders it does not see (`scripts/exp56/common.py`, `scripts/exp57/common57.py` call
  `encode_sensors` directly, plus D1's list), so "every percept enters EC through the registry" is not
  over-read.
- **N6 (§5.4).** In llm-primary, `affective_slow`'s "credit (cluster fear)" destination books nothing,
  because only `substrate_proposal.py` calls `note_active_clusters` (the wiring page's audit row). Say so
  in the track table.

---

## Proposed `docs/wiring/` entry

**`provenance-lives-at-the-write.md`.** For the body channels, a reading's provenance is only knowable
where the sensor was **written**, never where it is **read**. `vital_metrics` is a plain dict.
`world_set_axis(owner=)` is a log tag, not a record. Experienced (pump, DoA, the AUT's own affordances),
narrated (`SetEntitySensorTool`, `DamageComponentTool`, `OrchestratorActorTool` → `_apply_sensor_deltas`)
and apparatus (the water trial's RCON rescue and heal arriving through the pump, the cradle mother on
`main`) all land in the same slots. The body world credits through `_cluster_reward_bias`
(`record_outcome`) and the PainBus subscribers (`record_cluster_fear` and the others), not through
`credit_node`. So a provenance check on `credit_node` alone does not touch the earned path. Second
trap: **the survival rows' "loop" is `scripts/survival_world/water_trial.py`**, which calls
`propose_via_substrate` / `_encode_current_clusters` directly. A mechanism carried only on `LoopRun`
never reaches Exp 58/60/61/62/R3/B1. (GL3 wiring review, 2026-10-24.)

---

## Verified

- **GL3.B0 as claimed (#1201 MERGED).** `tests/unit/test_receptor_census.py` (`CENSUS`, an AST scan over
  eight categories, with `scripts/` explicitly out of scope and the limits stated in its docstring);
  `tests/unit/test_afferent_red_gates_units.py` (a, b, c, g) and `test_afferent_red_gates_loop.py`
  (d, e, f×2 [refractory, energy]), each `xfail(strict=True, raises=AssertionError)`, with passing
  controls `test_d2_control_a_free_loop_publishes_the_drift_breach` and
  `test_d5_control_an_open_gate_fires_the_reflex`. Run: `9 passed, 8 xfailed`. The frozen record
  `tests/fixtures/afferent_latency_characterization_v1.json` (measured at `7d18347e`, `src_tree 74a0fa82`)
  has equality gated by `_src_is_the_measured_tree`. A fresh `measure_all` on today's tree is **EQUAL** on
  both arms (damage: publish +2, flee +3, stale `mine_block` executed; drowning: publish +0,
  `escape_water` +25), so the record is still reproducible today.
- **L1:** `loop_state.py::_substrate_tick_due` requires `ctrl.pending_proposal is None`. **L2:**
  `loop_substrate.py::substrate_tick` skips `propose_via_substrate` (and with it the note, the
  transduction and selection) when `_substrate_gate_denied`. **L3:** `_STALE_PROPOSAL_AGE_S = 35.0`,
  `_wait_for_proposal(timeout=300.0)` (agent_loop.py). **L4:** `ConversationalSource._queue: deque`,
  `popleft`. **L5:** `enrich` runs only under `_pfc_gate_passed`, which is set by `thought_gate.should_think`
  (agent_loop.py), and `_evaluate_reflexes` is inside `enrich` (bio_enrichment.py). **L6:**
  `add_escalation_callback` has no caller. **L7:** `check_hold` (agent_loop.py) and `wire_preemption`
  (maxim_agent.py) have no producer; the Dormant docstring is in place.
- The **W-4 order** in `propose_via_substrate` is as stated (read → encode per channel →
  `note_active_clusters` → situation cue → `evaluate_failures` → re-read drives →
  `anticipatory_threat_need` → `_select_action`). `_SUBSTRATE_CHANNELS` = (interoception, audio, world).
- H0's slot: `pre_tick_gate` calls `_loop_live_tick` (drift + `experience_driver.on_live_pass`) after the
  pause check and before the idle gate.
- `LinguisticEncoder.encode_decomposed` and `SensorEncoder.encode_sensors` both call
  `register_substrate_node` (default `source="local"`) and `NAc.update_eligibility`. Those are the only
  `register_substrate_node` callers in `src/`.
- `DEFAULT_CLUSTER_FEAR_FAILURE_MODES = {"drive:health", "drive:oxygen"}`. `_BUNDLE_EC_NODE_FIELDS` is as
  quoted. `record_encoder_provenance`'s merge rules and its `ec.json` / `hivemind/cli.py` →
  `compose_bundle` manifest path are as stated. `entry_index.EC_NODE_FIELDS` is as quoted in S4.
- The narrator tools on `orch_registry` against `_aut_embodiment` (orchestrator.py) and the reflex tool
  attributes assigned to `aut_bio_enrichment_pipeline` are as stated. `sim.aut` is
  `threading.Thread(name="sim.aut")`, started after setup.
- **UNVERIFIED:** whether any live `--sim` campaign has imagined nodes in the eligible set at a Reaction
  (D3's magnitude). Whether the JS bridge can cancel a hold (inherited from the plan).
