# GL4–GL5 design review — ENVIRONMENT lens (latent_forward_model.md)

Plan reviewed: `docs/plans/latent_forward_model.md` (all 1,214 lines), at worktree `.worktrees/e-t8b` =
`origin/main` d9dffe6e. Charter: `docs/experiments/DESIGN_REVIEW.md`. Also read: `docs/plans/grounding.md`
(G1–G20 are settled and not re-opened here), `docs/plans/reviews/grounding_gl1/environment.md` (its
DNB-1/DNB-2 and SF-1–SF-9 are already folded into the plan; this review does not repeat them),
`docs/wiring/{README, sensor-range-clamps, world-light-sensing, body-and-word-worlds,
harness-loop-must-be-proven-live, cradle-sim-arms-and-resume}.md`, and the committed rig and cradle records
under `docs/experiments/data/` (`42*_results.jsonl`, `exp60_trials.jsonl`, `r3_bench.jsonl`,
`census_name_consequence/`). Lens question: can the named worlds (the cradle `--sim`, Paper 1.20.4 through
the mineflayer bridge, the scripted water bridge, Reachy) produce, game-natively and at a realistic rate, the
paired (state, action, consequence) data S0b gates on and the items S5 needs? Every code claim below was read
at `file::symbol`; anything not read is marked UNVERIFIED. No sim was run.

**Verdict: 2 DO-NOT-BUILD / 8 SHOULD-FIX / 6 NIT.**

Reading order: D1, then D2. Both block S0b's prereg as written, and D2 also blocks S2's gate and S5's design.
Then S1 (the runtime that D1's fold needs), S2 and S3 (the Minecraft half), S6 (S5's Stage 2), and the rest.

- **D1 (S0b, cradle):** the shipped cradle body saturates. `arms.thermal` recovers at 0.0008/s, a harm
  contact adds +0.6 and a safe warm +0.05, and substrate selection (Exp 42: 99.6% of contacts on one item)
  piles warms onto the arm. Within about a dozen contacts nearly every later record sits at the range edge,
  which the plan's own rule excludes from sign counts. Explore-first gives each other key about one trial, in
  a fixed (score, then name) order, so "key order counterbalanced" cannot be done in one campaign. The
  floors cannot be met as designed.
- **D2 (S0b/S2/S5):** the `object` block has no game-native source. 24 of the 34 consequence-bearing cradle
  keys sit on items that declare **no sensors**. The rest are constants that nothing writes, and the AUT
  receives them only through a deliberate `sense` act. The block as specified reads ground truth the agent
  never perceives (a D1 synthetic sensor), and in Minecraft the "target" is the avatar itself. So S2's
  sensed-similarity baseline is undefined on most keys, and S5's "reads alike" has no perceptual channel.

---

## DO-NOT-BUILD

### D1 — S0b cradle: the body saturates and selection concentrates, so one campaign cannot reach the band-clean floors

- **Stage:** S0b (cradle PASS/FAIL, which "stops the line"), and through it S2.
- **Issue: arithmetic from the shipped YAML and drift code.**
  - `bodies/infant_humanoid.yaml`: `arms.thermal` is homeostatic, range [−1, 1], set point 0, `drift_rate`
    0.0008/s, comfort band 0.5. `embodiment/sem.py::drift_step` moves it linearly toward the set point.
  - A harm contact (`fire_pit.touch`, `hearth.*`, every `*_harm` / `purple_flame` / `green_flame_b` …
    `warm_self` and `touch`) adds +0.6. After one harm contact from rest, the arm takes (0.6 − 0.5)/0.0008 =
    **125 s** to drop back into the band and **750 s** to reach rest.
  - The safe twins add +0.05 per contact. From 0.6, **eight** safe warms clamp the arm at **1.0**. Drift
    removes about 0.0004 per 0.5 s loop tick (`target_hz=2.0` in `simulation/orchestrator.py`), which is far
    below +0.05 per contact.
  - `bodies/infant_humanoid_chilled.yaml`'s own note says so: "There is no drift rate that gives a clean
    transient breach … a slow rate lets the breach linger and mis-attribute to later safe actions."
- **Issue: what selection does with that body (committed data plus code).**
  - `docs/experiments/data/42d53_results.jsonl` (substrate-primary, chilled body, two items, ~40 narrator
    turns) has `n_actions` ≈ 410–425 per session, `n_contact` 99–284, `n_exploit` = `n_contact` − 3, and
    `safe_pref` 0.996. After the forced trials, every contact goes to the one safe source. Within ~10
    contacts that source is warming an arm pinned at 1.0, so `drive_delta` is 0 and the pre-state is at the
    edge. The plan's S0b rule excludes those records from sign counts. Most of a 500-invocation capture would
    therefore be reported and never counted.
  - `decisions/nac.py::NAc.recommend_action`'s explore-first gate gives each untried scored tool **one**
    trial (`max(untried, key=lambda t: (scores[t], t))`). The order is fixed by score, with ties broken by
    name, so it cannot be counterbalanced inside one session. Each key's first trial also happens at a
    pre-state its alphabetical/score predecessors set. For example, `green_flame_touch` is tried right next
    to `green_flame_b_touch`, on an arm the first already raised. That puts a name-ordered confound inside the
    very collision pairs S0b counts.
  - The drive gate (`drive_gate_threshold` 0.5) then limits exploitation to tools whose names carry a `cold`
    affinity keyword (`warm`, `fire`, `blanket`, `huddle`; `_DRIVE_TOOL_AFFINITIES`). Chilled `cold`
    regenerates at 0.08/s, so it is above 0.5 almost all the time. `cool_air_*`, `food_source_eat`,
    `sharp_rock_touch`, every `*_touch` on a glow item, and `turn_left`/`turn_right` get about one sample per
    session.
  - The body choice removes classes:
    - On `infant_humanoid_chilled`, hunger drifts at 0.0008/s and stays below 0.5 by design, so
      `food_source.eat` (hunger −0.4) clamps at 0 and is a null at the edge.
    - On `infant_humanoid` there is no `cold` sensor, so all 24 glow-item keys lose their relief and reduce
      to "+0.05, inside band" (null) or "+0.6" (harm).
    - No single shipped body makes all 34 keys consequence-bearing.
- **Why it matters:** the S0b floors (≥ 500 invocations with a computable consequence, ≥ 30 keys, and
  dissociation counts "per (key, pre-state band)" excluding edge pre-states) were set as if each key were
  sampled repeatedly across bands from non-saturated states. The world gives about one clean record per
  non-preferred key per fresh body, and a long tail of edge records on one key. A FAIL here "stops the line",
  and it would be a world artefact, not evidence about grounding.
- **Proposed fold (plan edit to S0b, before its prereg is drafted):**
  1. Make "one campaign" **N short fresh-body sessions**. A fresh `--sim` builds a fresh body (separate
     subprocess, as `scripts/benchmark_exp42_preference.py` does). The scene's entity order and the item
     names' alphabetical order are permuted across sessions to counterbalance first-trial order. That is the
     only counterbalance the explore-first tie-break allows.
  2. Require the prereg to **compute, from the YAML deltas and `drift_step`, the expected number of non-edge
     records per key per band per session**, and to derive N from it, before any capture. This is the same
     independent arithmetic as the known-answer identity check. Freeze the floors in those per-key terms.
  3. Owner decision at S0b start, strict first:
     - (a) shipped bodies only, N sessions sized by step 2;
     - (b) a declared S0b-only body variant with a faster `arms.thermal` recovery. GL2a's `outcome_window`
       now nets in-bracket drift (G14), so a fast recovery no longer erases the record's breach; that was the
       reason the chilled YAML gave for keeping it slow. It is a new YAML, authored before the capture, never
       an edit to a shipped body, and it touches no earned row;
     - (c) apparatus resets of the body between blocks, stamped `apparatus` (G16) and used only as
       pre-states, never as outcomes.

     Recommend (a), with (b) only if step 2 shows (a) needs more than about 50 sessions.
  4. Set the exploration weight through `maxim config set sim.substrate_explore_bonus_weight` (default 0.0;
     explore-first is off without it) and freeze `MAXIM_SUBSTRATE_ACTIONS_PER_TURN`. Unset, actions per turn
     = narrator wall-clock ÷ 0.5 s, which is machine-dependent (`simulation/bridge.py`, the S6 budget
     comment in `orchestrator.py`), so volume and composition would vary by box.
  5. Pick the body, then list which of the 34 keys are non-null on it, as the plan's Q9 asks. State it as a
     table in the prereg: chilled = cold relief + thermal harm, eat null; infant = no cold class.

### D2 — S0b/S2/S5: the `object` block has no game-native source in either world

- **Stage:** S0b (the block is recorded), S2 (the sensed-similarity baseline and the full-vs-identity
  margin), S5 (acquired equivalence, "sensed readings that disagree with consequences").
- **Issue:**
  - **Most cradle items have no sensors.** A walk of `_data/components/items/*` finds that all twelve
    glow-modulator items (`green_flame`, `green_flame_b`, `green_hearth`, `green_hearth_b`, `purple_flame`,
    `purple_flame_b`, `purple_hearth`, `purple_hearth_b`, `warmth_alpha_harm`, `warmth_alpha_safe`,
    `warmth_beta_harm`, `warmth_beta_safe`) declare **`sensors: []`**. That is 24 of the 34
    consequence-bearing keys. Their `object` block, `tool_bridge.py::ModulatorAffordanceTool.execute`'s
    `entity_state` loop over `self._entity.sensors`, is empty. The census's same-name harm flips
    (`warm_self` on `*_safe` vs `*_harm`, and the `_b` twins) sit exactly there.
  - **Where items do have sensors, the sensors are constants.** `fire_pit`/`hearth` carry
    `heat_output`/`fuel`, `blanket` `texture`/`thermal`, `cool_air` `temperature`/`intensity`, and so on.
    The modulators are `abstract: true` ("capability-only — no sensors (C4)"), and a grep finds no `src/`
    writer of `heat_output`. Each block is therefore a fixed fingerprint per entity.
    `similarity/encoder.py::_sensor_embed` hashes per **sensor name** (SHA bases), so cross-entity similarity
    is mostly "do they share sensor names". Two items "read alike" when an author gives them the same sensor
    names.
  - **The AUT never perceives these readings by default.** `simulation/orchestrator.py` deregisters the
    per-entity sensor tools and registers `tools/discovery.py::UniversalSenseTool`. A target's readings reach
    the agent only if it invokes `sense(entity_name=…)`, and then as text. Substrate selection encodes drive
    state (`runtime/substrate_proposal.py::_read_drive_states`), not target readings. The `object` block
    built at the tool (S1) and read at decision time (S3/S4 shadow, S5 selection) is ground truth, delivered
    without a receptor. That is the D1 "no synthetic sensor" rule, applied to perception.
  - **Minecraft has no target object.** Every affordance sits on the `avatar` modulator of the player root
    (`bodies/minecraft_player.yaml`: `move_to`, `turn`, `mine_block`, `place_block`, `eat`,
    `attack_nearest`, `flee`, `escape_water`). The "target entity" is the body, so `object` duplicates
    `situation`, and `identity` is "<affordance> minecraft_player" for every key.
- **Why it matters:**
  - S2's G13 gate ("beats the no-learning sensed-similarity baseline") is undefined on empty blocks. Every
    empty block is the same (zero) vector, so the baseline either ties everything or divides by zero.
  - On held-out entities with no object block, the full model has nothing beyond identity, so "beats
    identity-only by ≥ 0.2" is unreachable by construction. A FAIL there would be a property of the YAML,
    not of ridge.
  - S5's acquired-equivalence test needs readings the agent can sense before first contact. As specified,
    the readings are injected.
- **Proposed fold (plan edit to §4.1 `ActionContext`, S0b, S2 and §6):**
  1. Define the `object` block as **what a receptor delivers to the AUT**. Before GL3.B1 that is the AUT's
     own `sense` result for that entity, captured in its percept stream, and only if it was sensed before the
     decision. After GL3.B1 it is an exteroceptive `Receptor`. An entity the agent has not sensed has an
     **absent** block (recorded as absent, never a zero vector).
  2. S0b reports, per key, whether a block exists and how often it was sensed. S2's dissociation gates are
     read only over keys with a non-empty block, and the empty-block keys are reported as their own
     "identity-only by construction" row.
  3. For the S0b scene, either accept that only about 10 keys carry readings, and **restate the "≥ 30 keys"
     floor over keys with a block**, or author sensors on the glow items. Authoring follows S5's blind
     rule, uses new YAMLs, and leaves the shipped items untouched, since Exp 42/44 cite them. Strict
     recommendation: author new items. Do not edit shipped ones.
  4. In §6, state that "reads alike" means **shared sensor names with near values under `_sensor_embed`**,
     and that each S5 item's readings must reach the AUT through the receptor path in step 1 before Stage 1
     contact.
  5. In Minecraft, drop the `object` block (mark it absent) instead of copying the body into it.

---

## SHOULD-FIX

### S1 — S0b runtime: the fixture orchestrator cannot activate cradle items; the only runtime has an LLM narrator

- **Stage:** S0b (the plan's "candidate cradle runtime … UNVERIFIED"), Q7, Q9.
- **Issue (resolves the plan's UNVERIFIED):**
  - `simulation/fixture_orchestrator.py::FixtureDrivenOrchestrator.run` drives only the fixture's text and
    sensor percepts through `bridge.send_and_wait`.
  - `simulation/campaign_runner.py::run_fixture_campaign` threads no `embodiment`/`entity_map` into it.
    Scene items are activated only by `simulation/generative_runner.py::_load_world_entities` /
    `_activate_phase_entities`, inside `run_generative_campaign`, whose `Narrator(llm=llm, …)` is an LLM.
  - The `scenarios/substrate/*.yaml` fixtures are text scenarios with no cradle entities. A fixture-path AUT
    holds only its own body's affordances.
  - So the one production runtime that puts the 34 keys on an AUT is `maxim --sim --arc <yaml> --aut-mode
    substrate-primary`, the Exp 42 recipe.
  - On that path the AUT's `agent_id` is `"sim_aut"` (`AgentConfig(agent_id="sim_aut")`), so pids can be
    minted. The orchestrator also registers `SetEntitySensorTool`/`DamageComponentTool`/
    `OrchestratorActorTool` for the narrator.
  - Autonomous dispatch reaches `tool_dispatch.execute_and_learn` → `capture_loop_action`
    (`runtime/agent_loop.py`, the `if can_execute:` branch). That substrate proposals take this branch is
    consistent with the proposal-slot reading in `harness-loop-must-be-proven-live.md`, but was not traced
    end to end (UNVERIFIED).
- **Why it matters:**
  - Q7's rationale ("scripted-substrate + Minecraft only", "no LLM") is wrong about this runtime. Selection
    has no LLM, but the narrator does, so the run costs narrator calls and puts narrated writes into the
    world.
  - The narrator box is subject to the no-co-location rule (CLAUDE.md; `no-harness-on-leader-machine.md`).
  - The AUT turn is a 30 s wall window (`generative_runner.py::_aut_turn_timeout_s`).
- **Proposed fold:**
  - Name this runtime in S0b and Q9. Use an arc YAML whose phase `instruction` strings are empty, as
    `arcs.py::_make_pref_phases` does, so the narrator is never told to write the AUT's sensors.
  - State the narrator model, its box, and the sim box (never the leader).
  - Keep the narrated-write count as a reported row, plus a sentinel: S0b refuses a session whose narrated
    body-write count is not zero, or reports it as narrated-contaminated.
  - Delete the fixture-orchestrator candidate.
  - Note that the `--sim` path's Cerebellum is saved at `agent_factory` shutdown, not at `on_session_end`
    (the plan's S3 seam).

### S2 — Minecraft half: on every shipped water harness, the post-escape refill happens after an apparatus teleport

- **Stage:** S0b Minecraft half, S5 Minecraft arm.
- **Issue:**
  - `scripts/survival_world/water_trial.py`'s lethal-event loop does `if not s["in_water"]: end, t_surface =
    "surface", s["t"]; self.rcon.teleport(self.username, self.shore)  # the exit, BEFORE the loop stops`.
    `r3_bench.jsonl` shows `t_end` − `t_surface` ≈ 0.002 s.
  - `WaterTrial.rescue` teleports, then `settle_until(oxygen ≥ RECOVER_OXYGEN_MIN)`, then `heal()`.
  - So on R3 and Exp 60/61 the oxygen refill that `escape_water` should "cause" happens on the shore, where
    the apparatus put the bot.
  - Without the teleport, the live bridge holds jump for only 600 ms after the eyes clear
    (`scripts/minecraft_bridge/index.js` `escape_water`, `SURFACE_HOLD_MS`). The measured anchor has
    `t_sinkback_min_s` 2.119 (`r3_bench.jsonl` `anchor_measured`), so the bot is back under water within
    about 2 s.
- **Why it matters:** the `cause_pid` join over a "frozen consequence window" (S0b) would pair the
  escape's context with a refill that G16 must stamp `apparatus`, or that is cut off by sinkback. On the
  shipped harnesses, the experienced Δoxygen of `escape_water` is near zero, or mixes the escape with the
  apparatus.
- **Proposed fold:**
  - The S0b prereg either (a) builds a **new apparatus** that leaves the bot surfaced without a teleport for
    the whole window: a pool with a reachable ledge/air pocket, checked by `check_surface_cell_air`-style
    preflight and a measured sinkback time; or (b) ends the window at the first harness write and accepts
    the short in-water refill.
  - Freeze the window shorter than the measured sinkback. Read air refill in sensed units off a live trace
    (UNVERIFIED rate: about +4 air/tick ≈ 5.3 units/s on the bridge's 0–20 scale).
  - Recommend (a). Never edit `water_trial.py`'s exit: it is a cited guard of the Exp 60/61/62/R3 rows.

### S3 — Minecraft half: oxygen bands are filled by different learner states, and the "no-op pass" control has no record path

- **Stage:** S0b Minecraft half.
- **Issue:**
  - In `r3_bench.jsonl` (61 events in 2.03 h, so about 30 events per hour), surface times cluster in three
    modes: about 3.0 s (oxygen above 14, before pain), 8.3–9.1 s (pain band) and 27.5–28.5 s (oxygen 0,
    after damage). The modes track the arm: fear-learned agents escape early, innate-only agents late.
  - So "≥ 3 oxygen bands" is reachable, but each band is populated by a different agent history, which is
    confounded with what the context encodes.
  - Exp 60's probes are capped at 4.3 s, below the pain edge. Its record has 65 `escape_water` calls over
    228 placements, all in one band.
  - The control "a no-op pass in the same band" is not an invocation. It has no `pid`, so a non-act cannot
    carry a `cause_pid` join.
  - `turn` was marked UNVERIFIED. The bridge's `turn` is `bot.look(yaw, pitch, true)`, which runs whether or
    not the bot is submerged and has no oxygen effect, so it is a valid same-band executing control.
- **Proposed fold:**
  - Drive band coverage by placement **depth/delay within one agent type**, not by mixing arms, and record
    the band of each invocation's pre-state.
  - Use `turn` as the executing control. Drop the no-op pass, or define it as the drowning baseline trace of
    the same agent at the same band (a reported comparator with no pid).
  - Budget: 100 escapes at R3's rate is about 3.5 h of rig time plus setup, in a Track C slot never stacked
    on an E-rung (as recommended). Exp 60's probe regime cannot reach the floor.

### S4 — `nociception` is structurally zero in the cradle, and caused vs felt cannot yet be told apart

- **Stage:** S0b, S2 ("gates read per-dimension signs … `nociception`, `harm`, `relief`"), §4.1 "caused,
  not felt".
- **Issue:**
  - `proprioception/pain.py::TISSUE_DAMAGE_DRIVES` = `{"drive:health"}`, and the infant's `health` is
    `derived` (`bodies/base_humanoid.yaml`). The only nociceptive cradle pain is `damage_component`, which is
    narrator/reflex (narrated).
  - So in ans-v1 every experienced cradle record has nociception 0, and the cradle's per-dimension
    nociception gates are vacuous.
  - GL2a, merged at this commit, records nociception as one value: `sem.py::interoceptive_outcome` sets
    `"nociception_basis": "caused_or_felt"` ("Caused and felt pain are one value until ToolPainBridge splits
    them (after the fence)").
  - S0b's "caused versus felt-only nociception" count and S2's "caused only" target cannot be measured
    before the fence.
- **Proposed fold:**
  - The prereg declares the cradle gates on `harm` and `relief` only, with nociception reported, and names
    the target schema: ans-v1 now, or ans-v2 after GL2b(ii). Never both (the plan's own rule).
  - It states that the caused/felt split is unavailable before the fence. On the cradle tool path every
    entry is caused (`caused=tuple((name, True) …)`), so the split matters only in Minecraft.

### S5 — The identity block's geometry on the capture box, and names that leak the answer

- **Stage:** S0b dissociation counts, S2.
- **Issue:**
  - The census (`docs/experiments/grounding_census.md`) encoded affordance **names** with mpnet
    `paraphrase-mpnet-base-v2` at revision 6cc9279c on CPU.
  - The plan's identity block is a different string, "<affordance> <entity name>": 0.469 for touch
    blanket ↔ touch fire pit, against the census's 1.000 for touch ↔ touch.
  - `cradle-sim-arms-and-resume.md` records that every committed local cradle fire ran on the hash fallback
    (`sentence_transformers_version: "not-installed"`).
  - The rig's M1 provenance block (`exp60_trials.jsonl` `provenance`) does not stamp the encoder.
  - Three Exp 42 items put the answer in the name: `warmth_alpha_harm`/`warmth_beta_safe`.
- **Why it matters:** the census-derived floors would be compared against counts made under another string,
  and possibly another geometry. A capture on a fallback box would count 384-d bag-of-words "collisions".
- **Proposed fold:**
  - Stamp the encoder id, revision and dimension in S0b's provenance, and refuse a fallback run.
  - Recompute the census prediction over the identity-block strings before setting the floors.
  - Exclude or flag the `warmth_*_{harm,safe}` items. The purple/green families are the neutral-name ones
    (`arcs.py`, the Exp 44 comment).

### S6 — S5 Stage 2: the cradle world has no mechanism to "retrain A to O3"

- **Stage:** S5 (acquired equivalence, §6 step 2).
- **Issue:**
  - An item's consequence is its YAML `self_effect`. Scene activation is additive only:
    `generative_runner.py::_activate_phase_entities` skips already-activated refs and never deactivates.
  - A same-named A′ with O3 physics would silently replace A's tool in the registry
    (`tools/registry.py::ToolRegistry.register` is `self._tools[tool.name] = tool`).
  - How `EntityMap` resolves two entities with one name is UNVERIFIED.
- **Why it matters:** an in-session swap is an experimenter write to the world's physics, and the
  mechanism that would do it is a silent overwrite.
- **Proposed fold:**
  - Run Stage 2 as a **resumed session** (`--resume-sim`) under a second arc whose A-named item carries O3.
    `_restore_aut_from_session` carries hippocampus, NAc, EC and ATL only, with NAc undecayed (wiring doc).
    The predictor reaches Stage 2 only through S3's `on_session_end` refit, which on `--sim` needs S3's
    cradle seam.
  - Declare the physics swap an apparatus event (G16).
  - Add a preflight that refuses a scene whose entity names collide.

### S7 — S5 Minecraft: a game-native transfer contrast is not available on today's bridge

- **Stage:** S5 owner decision "whether a game-native transfer contrast is added to the Minecraft arm".
- **Issue:**
  - The affordance list (above) has no object-directed contact act, and the sensors have no per-object
    property (`bodies/minecraft_player.yaml`).
  - The bridge's `damage` event has no source (GL1 SF-7).
  - Food cannot drain natively in campaign time, and `ScriptedWaterBridge` holds food at 20 (GL1 SF-5).
  - Any contrast with differing or colliding names and game consequences (magma vs campfire, berry bush vs
    cactus) needs new bridge affordances and sensors, i.e. a protocol change. `scripted_water.py`'s own note
    says such a change fires the re-run trigger on four EARNED/PARTIAL rows.
- **Proposed fold:** state this in S5's owner-decision list as the price of the option. Strict default:
  no contrast, and the arm stays consequence prediction only, as G19 already places T9 in the cradle.

### S8 — §3.8 compute and persistence: dual-form ridge stores its training set, and n grows every session

- **Stage:** S2 ("under a second on a laptop", "Pi profile"), S3 (session-end refit, JSON persistence).
- **Issue:**
  - Dual-form kernel ridge predicts from all n training contexts.
  - At about 1.5–2k dims (768 identity + 384 object + 384 per situation receptor + params), 10³ pairs is
    about 2M floats, roughly 35–40 MB of JSON in `consequence_predictor.json`. S1 adds the same vectors to
    every loop trace in `aut_hippocampus.json`, about 30 KB per trace.
  - S3 refits on "the persisted training set" plus each session's traces, so n grows without bound. The
    solve is O(n³), and the n² float64 kernel is 800 MB at n = 10⁴.
  - The owner's hardware (RTX 5080 leader, 24 GB Mac peer, big-mac-mini rig) includes no Pi, so "the Pi
    profile" is a proxy.
- **Proposed fold:**
  - S2 measures solve time, peak memory and file size on the named machines at n = 10³ and 10⁴.
  - S3 freezes a cap on n (or a Nyström/primal fallback, declared as its own stage) and a file-size budget.
  - "Pi" becomes "measured on <box>; Pi UNVERIFIED".

---

## NIT

- **N1 — GL2a has merged; update the S0b text.** `embodiment/sem.py::interoceptive_outcome`,
  `InteroceptiveOutcome`, `drift_step` and `embodiment/body.py::Embodiment.outcome_window` exist at
  d9dffe6e, so the "frozen copy in the script if GL2a has not merged" branch is moot. Note too that the
  record covers **only the invoked affordance's declared drives** (G14), so the "per-body block … in the
  body's declared drive order" (§4.1) is a per-affordance subset. The target projection must fill the
  undeclared drives as absent, never as 0.
- **N2 — `turn` acts underwater** (S3 above): resolve the plan's UNVERIFIED by citing `index.js` `case
  "turn"`.
- **N3 — Freeze oxygen band edges in sensed units.** `oxygen` is declared on [0, 40], while the game value is
  0–20 (`minecraft_player.yaml`). Per `sensor-range-clamps.md`, compare against sensed values read through
  `substrate_proposal._read_world_ranges`.
- **N4 — Reachy yields no experienced pairs; say so.** `bodies/reachy_mini_infant.yaml` sets the azimuth
  drive to `null`, and its hunger and thirst are relieved only by mother feeds (apparatus, G16). On the live
  robot, azimuth is live-owned, so `tool_bridge` withholds the modeled self_effect. Add one line to S0b:
  Reachy is out of scope (G4), not "not run".
- **N5 — Define "computable consequence".** In Exp 42 between a third and two thirds of actions are
  non-contact (`sense`, `listen`, `look`), whose record covers no drive. The prereg says whether a
  null-drive record counts toward "≥ 500".
- **N6 — Capture loss bound.** `HippocampusConfig.capture_queue_size` is 100 with drop-oldest
  (`Hippocampus.capture_from_loop_async`), and `max_nodes` is 10,000. Both are ample at 2 Hz, but S0b should
  report drops per session, as it already plans.

---

## Verified

At d9dffe6e, by reading code or committed data:

- `similarity/encoder.py::_sensor_embed`: per-sensor-name SHA bases, value-rotating.
- `embodiment/sem.py::drift_step`: linear homeostatic/entropic drift, clamped.
- `sem.py::interoceptive_outcome`: `"nociception_basis": "caused_or_felt"`; on the tool path every covered
  drive is `caused=True`.
- `body.py::tick_vital_drift` runs on `drift_step`; `Embodiment.outcome_window` exists.
- `tool_bridge.py::ModulatorAffordanceTool.execute`: `entity_state` is read from the owner entity's sensors
  after `modulator.execute`; self_effect is applied via `_apply_sensor_deltas`.
- Item walk of `_data/components/items/*`: twelve glow items declare no sensors; the shipped
  self_effects are as listed in D2; there is no `src/` writer of `heat_output`.
- `infant_humanoid.yaml`: `arms.thermal` drift 0.0008/s, band 0.5; hunger 0.006/s; core 0.001/s.
  `infant_humanoid_chilled.yaml`: `cold` 0.08/s entropic; hunger/thirst 0.0008/s; the "no clean transient
  breach" note. `base_humanoid.yaml`: `turn_left`/`turn_right` ±0.3 azimuth, drift 0, band 0.1.
  `reachy_mini_infant.yaml`: azimuth `drive: null`.
- `minecraft_player.yaml`: sensors and ranges, eight avatar affordances.
- `decisions/nac.py::NAc.recommend_action`: explore-first one-trial gate with the (score, name) tie-break,
  active only when `substrate_explore_bonus_weight` > 0 (config default 0.0); the drive gate at 0.5;
  `_DRIVE_TOOL_AFFINITIES`.
- `simulation/orchestrator.py`: AUT `agent_id="sim_aut"`; `target_hz=2.0`; narrator tool registration; the
  fixture branch passes no embodiment.
- `fixture_orchestrator.py::FixtureDrivenOrchestrator.run` and `campaign_runner.py::run_fixture_campaign`
  never activate scene entities. `generative_runner.py::_activate_phase_entities` is additive.
  `tools/registry.py::ToolRegistry.register` overwrites by name.
- `scripts/minecraft_bridge/index.js`: `turn` = `bot.look`; `escape_water` with 600 ms hold and 8 s cap;
  `flee` refuses when submerged.
- `scripts/survival_world/water_trial.py`: the lethal-event teleport at surface, `rescue`, `heal`.
- `proprioception/pain.py::TISSUE_DAMAGE_DRIVES` = `{"drive:health"}`.
- `memory/hippocampus.py`: queue size 100, `max_nodes` 10,000. `pyproject.toml`: numpy/scipy core, `pi`
  extra.
- Committed data:
  - `42_results`/`42b_results`/`42d53_results.jsonl`: n_actions ≈ 405–442, exploit = contact − 3,
    safe_pref 0.996.
  - `exp60_trials.jsonl`: 20 rows, 228 placements, 65 `escape_water` calls, probe cap 4.335 s.
  - `r3_bench.jsonl`: 61 events in 2.03 h, the trimodal surface times, `t_sinkback_min_s` 2.119,
    `t_pain_edge_min_s` 5.153, damage onset at about 16.1 s.
  - The census header (mpnet revision, CPU, 0.44).
- Local probe: sentence-transformers and torch import on this Mac.

**UNVERIFIED:**
- That substrate proposals on `--sim` take the `execute_and_learn` branch end to end (consistent with the
  wiring docs; not traced).
- The live air-refill rate after `escape_water` (derived from 1.20.4 mechanics).
- How `EntityMap` resolves duplicate entity names.
- Wall time per Exp 42 session (the rows carry none).
- The size estimates in S8 (arithmetic, not measured).
