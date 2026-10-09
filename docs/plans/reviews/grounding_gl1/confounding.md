# GL1 four-lens design review: CONFOUNDING lens

Reviewed: `docs/plans/grounding.md` (umbrella) and `docs/plans/autonomic_layer.md` (GL2), at the gl1
worktree (origin/main with #1167 + GL0 #1186). I read `latent_forward_model.md` where the umbrella's
stage map restates its gates (S0b, S2, S5 = GL5). I did not re-open G1–G8 or the 2026-10-09 GL2a owner
decisions (tool-path record only, measured drop-oldest bound, unweighted valence
`relief − harm − nociception`, urgency = pressure only). Where a finding touches one of them, it is
about which number a gate reads, not about the decision.

The question for every stage: could a positive, or a null, come from something other than the claimed
cause?

**Verdict: 3 DO-NOT-BUILD / 8 SHOULD-FIX / 4 NIT.**

- DO-NOT-BUILD **GL2a**: refilling the existing trio "from the same computation" makes the
  `relief == max(drive_relief)` pin either vacuous or false, and it silently changes memory-strength
  inputs under a "record-only" label. No stated gate would catch it.
- DO-NOT-BUILD **GL5 / GL4 S2**: in every arm, what the agent senses about an object and what the object
  does to the body move together. The jet-triad reversal is already predicted by untrained object-block
  similarity. No arm separates "similar by consequence" from "similar by sensed properties".
- DO-NOT-BUILD **GL4 S0b** (restated in the umbrella's GL4 row): the "≥ 5 dissociation pairs" stop gate
  counts "< 0.44 cosine, same sign" pairs, null consequences included. As written it can hardly fail.

---

## DO-NOT-BUILD

### DNB-1 (GL2a): the trio refill makes the relief pin vacuous or false, and changes what memory strength reads

**Issue.** autonomic_layer.md §3.1.4 says: "The existing trio (`drive_pressure_before`, `drive_relief`,
`pain`) is filled from the same computation and stays." §3.1 adds: "a guard pins
`relief == max(drive_relief)` so the two cannot diverge". GL2a must also read qualified drives
(§3.1.4, the #1125 dependency): it "never ships a record that omits `arms.thermal`" and "must not derive
`drive_delta` from `side_effects["drive_progress_by_drive"]` while #1161 keeps that side effect
blind". These three statements cannot all hold.

- **Today's `drive_relief` is blind to `arms.thermal` and covers declared drives only.** In
  `embodiment/tool_bridge.py::ModulatorAffordanceTool.execute`,
  `pre_values = {name: _metrics[name] for name in _self_effect if name in _drive_specs and name in _metrics}`
  reads root `vital_metrics` only ("Root reads only: BLIND to modulator drives (`arms.thermal`) on
  purpose until #1161"). `_drive_progress_by_drive` then iterates only the affordance's declared
  `self_effect`. `runtime/executor.py::Executor._drive_relief` normalises exactly that side effect.
  Example: `cradle_fire_pit.yaml` `touch` declares `arms.thermal: 0.6` and `core_temperature: 0.15`.
  Today's `drive_relief` sees only `core_temperature`.
- **Case 1: the trio is refilled from the new computation** (every ranged drive, qualified drives
  resolved). Then `relief == max(drive_relief)` compares a value with itself and cannot fail. The trio
  also changes value and key set. That matters because the trio is not a passive record:
  `memory/encoding.py::encoding_tag` treats "the PRESENCE of a relief key" as relevance, and filters
  pressure to drives that carry a relief key. `Hippocampus._stamp_encoding_strength` turns the tag into
  `storage_strength`, which `memory/strategies.py::StrengthStrategy` reads. (`StrengthStrategy` is
  selected by strategy name `"strength"` in `Hippocampus`; whether it is the production default is
  UNVERIFIED.) A new relief key on `arms.thermal`, or on any drive the affordance did not declare,
  changes encoding tags and retention on cradle bodies.
- **Case 2: the trio is not refilled.** Then the pin is false whenever `arms.thermal` (or any
  undeclared drive) moves toward comfort. The record sees that relief and `drive_relief` does not.
- **Neither case is caught by the stated gates.** The selection golden
  (`tests/unit/test_agent_loop_selection_golden.py`) runs only the scripted Minecraft-shaped `shore` /
  `fear_water` arms through `tests/unit/_loop_harness.py`. It has no cradle body and does not record
  encoding fields. The encoder golden pins `_sensor_embed`. The T1-16 discharge argues only that
  `_rank_by_relevance` reads no `extra` key, so it misses the trio → `encoding_tag` → storage-strength
  route entirely.

**Why it matters.**
1. "Record-only, no reader acts" (grounding.md GL2a row) would be false.
2. The memory-strength line's measurements would move under a label saying nothing moved.
3. The one consistency guard offered would either be vacuous or would red on the burn, which is the
   record's reason to exist.

**Fix (plan text, before GL2a is built).**
- State that the trio stays computed by today's code, byte for byte, until #1161 decides otherwise.
  Changing it is a memory-strength change with its own T1-16 walk.
- Restate the pin over the trio's own drive set:
  `max(positive drive_delta restricted to the drives present in drive_relief) == max(drive_relief)`.
  Record the qualified-drive difference as a named, #1161-scoped divergence. That gives the pin a
  comparison it can fail.
- Add a GL2a gate: a golden taken from the pre-change commit of the trio, `encoding_tag` and
  `storage_strength` for every capture in a scripted cradle sequence (the one GL2a already names:
  `cool_air` ×2, `warm_self` ×2, `touch`) and in the Minecraft `fear_water` arm. It must be
  byte-identical after the change.
- Add the deletion probe the other way: revert the record and the golden still passes. That proves the
  record never fed the trio.

### DNB-2 (GL5 claim; GL4 S2's jet-triad gate): sensed properties and consequences are confounded in every arm

**Issue.** The thesis is "concepts become similar by what they do to the body". The design puts the
signal that must override the name in the **`object` block**, "the TARGET entity's own sensed readings"
(latent_forward_model.md §4.1). It says outright that the jet-triad reversal is "carried by the
`object` block: the source entity's own sensed heat versus wet/cold" (§6).

In every arm, what the object reads on its sensors and what it does to the body are authored together:
- `cradle_fire_pit.yaml` declares `heat_output: initial 0.9` beside `touch` → `arms.thermal: 0.6`.
- The GL5 "warm rug that burns" / "fire stone that is cold" items must, under the blind-author rule,
  carry sensors consistent with their effects. The fire stone reads cold and is cold.

So the cosine between object blocks orders the triad `fire_breath > water_jet` **before any consequence
is learned**. Ordinary nearest-neighbour generalisation over sensed readings, with no forward model,
produces the reversal. The S2 gate compares the full model only with **identity-only** (refuse if
identity-only ≥ 0.8). It names an object-only ablation but gates nothing on it. The GL5 arms compare
with an identity-only arm and a yoked no-predictor arm, and never with a "sensed similarity, no
consequence model" baseline.

The game-native Minecraft arm, the one meant to close the circularity of authored physics (umbrella risk
9), measures a graded Δoxygen/Δhealth regression for `escape_water` vs `move_to`. It does **not** test
transfer between affordances whose names differ or collide. So the arms that bear the claim are all
authored, and the one arm that is not authored does not bear the claim.

**Why it matters.** A PASS on all three arms is exactly what "perceptual similarity beats names" also
predicts. That is a real and interesting result, but it is not "similar by consequence", and T9 would
over-claim. The circularity mitigations (a blind author, effects derived from each object's own
declaration) make the confound stronger: they guarantee that each object's sensors and effects are
mutually consistent.

**Fix (fold into grounding.md §GL5 and latent_forward_model.md S2/S5 before GL5's prereg is drafted).**
1. **Add an untrained sensed-similarity baseline to S2 and to every GL5 arm.** Spread the trained
   consequence to held-out items by object-block cosine alone, with no fitted map. Gate the full model
   **against that baseline**, not only against identity-only.
2. **Author a perception–consequence dissociation set**, under the blind-author rule:
   - items whose sensed readings match but whose consequences differ (a hidden property, e.g. an
     insulated hot-reading object that does not burn), and
   - items whose readings differ but whose consequences match.

   Only those items can show learning by consequence rather than by perception. Pre-register the
   gate on them.
3. **Or re-scope T9's claim honestly**: "sensed properties override the name prior". Keep "similar by
   consequence" out of the claim until (1) and (2) pass.
4. **Either give the Minecraft arm a transfer item** (game-native, names that differ or collide), **or
   state plainly that it is a forward-model readout and not evidence for T9's transfer claim.**

### DNB-3 (GL4 S0b, restated in grounding.md's GL4 row): the line's stop gate can hardly fail

**Issue.** S0b's PASS requires "≥ 5 name–consequence dissociation pairs: keys whose identity-block cosine
is ≥ 0.44 but whose consequence sign differs, **or < 0.44 with the same sign**". The consequence classes
include "nothing" (S0b: "e.g. burn, relief, nothing"). The audit (both plans, §5.5 / §9) says 363 of 405
shipped affordances declare no body effect, and S0b's roster needs ≥ 30 keys, so many will be
`observe`/`look`-type keys with a null consequence. Any two of them whose names fall below 0.44 count as
a "same sign" dissociation. As written, the floor of 5 is met by null/null pairs, which say nothing about
grounding. The FAIL branch ("< 2 dissociation pairs … stops the line") is close to unreachable.

**Why it matters.** This is the gate that "stops the line". A gate that cannot fail is the vacuous-guard
class (prove-a-guard-by-deletion). The S2 gate is then evaluated on dissociation keys that may be mostly
trivial.

**Fix.**
- Exclude null-consequence pairs from both clauses.
- Count the two kinds separately, each with its own frozen floor:
  - **collisions**: ≥ 0.44 with opposite non-null sign, the "blocks it" half of T9;
  - **convergences**: < 0.44 with the same non-null sign, the "transfers" half.
- Derive "sign" per (key, pre-state band), per SF-3.
- Pin the definitions in the S0b prereg. Have GL1's census predict both counts from YAML, so the
  prereg's floors are set against a known expectation.

---

## SHOULD-FIX

### SF-1 (GL2a): the tool-path window can carry narrated or drift changes stamped `experienced`, and delayed consequences fall outside it

**Issue.**
- **The window is undefined.** `Executor.execute` snapshots pressure before `tool.run`
  (`_drive_pressure_snapshot`) and stamps after it. The plan does not say whether `drive_delta` and
  `deviation_after` cover (a) only the affordance's declared or B8-attributed drives, or (b) every
  ranged drive between the executor's two snapshots.
- **Under (b), changes from other threads land in an `experienced` record.**
  - In `--sim`, the narrator's tools write the AUT body from the orchestrator thread (§1.2a). The GL2a
    `RLock` guards only `evaluate_failures`' latch, snapshot and mint section, not the tool's
    before/after window. A narrator write landing inside the window is recorded as experienced. That is
    the relabel G6 forbids, made silently.
  - On Minecraft, `MinecraftSyncPump._run` writes vitals continuously from the game, so passive drift
    during a slow actuation (`escape_water`) is attributed to the action. Its duration is UNVERIFIED.
- **Delayed consequences are lost under the strict default.** Out-of-band records are not minted. If
  oxygen recovers after `escape_water` returns, that relief is in **no** record. The tool-path record
  then understates the very consequence S0b's Minecraft floor and the GL5 Minecraft arm measure.

**Fix.**
- Pin the window and drive set in §3.1.4. Recommended: a whole-body before/after delta, plus an
  `attributed` mask, meaning the declared or B8-harmed drives.
- Add a concurrency stamp: the body's write counter, or the sequencer's last seq, read under the lock
  at both ends of the window. Any foreign write in between sets a `mixed` flag in `extra`, and S0b/S2
  exclude or separate those records.
- Name a consequence horizon for actions whose effect lands after the tool returns. For example, join
  the next out-of-band record by `cause_pid` once the out-of-band producer ships. Until then, S0b's
  Minecraft half must state that the tool-path record truncates surfacing relief.

### SF-2 (GL2a → GL2b(ii)): the target's definition moves under the predictor

**Issue.**
- **GL2b(ii) changes what `nociception` means for the same event.** Before GL2b(ii) the infant burn is
  DRIVE pain, so `nociception = 0` and `drive_pain = 0.04`. After it, the burn is NOCICEPTIVE at
  ≥ 0.4 (calibration). The core vector, and `valence`, change for the same contact.
- **The schema id does not bump.** `schema_id="ans-v1"` is fixed in §3.1.1, and nothing requires a bump
  at GL2b(ii). S0b's capture can predate or straddle GL2b(ii), so a predictor trained or evaluated
  across the change learns a moved target.
- **`drive:health` is counted twice in valence.** It is carried as nociception (§3.1), and as a
  homeostatic drive its loss is also a negative `drive_delta`, which is harm. Under the unweighted
  formula, valence = relief − harm − nociception, an injury enters twice while a cold or air deficit
  enters once. I am not re-opening the weights. The overlap is a property of the inputs, and it makes
  Minecraft health events look more aversive than equal-sized oxygen events.

**Fix.**
- GL2b(ii) bumps the schema id (`ans-v2`). S0b and S2 train and evaluate on one schema id, and refuse a
  mixed set.
- State the `drive:health` overlap in the record's docstring.
- Have S2's and GL5's gates read per-dimension signs (`nociception`, `harm`, `relief`), not `valence`.
  If valence is reported, report it beside them as the innate-prior summary.

### SF-3 (GL1 census, GL4 S0b, GL5): a key's consequence sign depends on state, repetition and saturation

**Issue.** "Opposite declared effect" and "consequence sign differs" assume each key has one sign. The
shipped data says otherwise:
- **Mixed effects whose valence depends on the body's state.** `fire_pit.touch` is harm on
  `arms.thermal` (+0.6) **and** relief on `core_temperature` (+0.15) for a cold infant.
- **Repetition.** The nociceptor reads state (§3.2), so repeated `warm_self` stacks past threshold.
- **Saturation.** `arms.thermal` range `[-1,1]`: a touch on an arm already at 1.0 gives `drive_delta`
  0, so harm 0. Before GL2b, with no nociception either, a saturated burn records as "nothing".
- **Floors.** Minecraft health and oxygen cap at their set point (20), so their deviation is one-sided
  (§9 notes this).

A "dissociation" or "collision" count can therefore be produced by the order and state of the scripted
sequence, not by the keys.

**Fix.**
- Define sign per (key, pre-state band), with the bands frozen in the S0b prereg.
- Report saturated pre-states (value at the range edge) separately and exclude them from sign counts.
- Have the GL1 census report sign per band from the declarations, and counterbalance key order in S0b's
  scripted capture.
- Add to GL2b(ii)'s gate a "`warm_self` after `touch`, arm still hot" row next to the from-rest rows
  (also NIT-3).

### SF-4 (GL2c): a flag-on PASS on Exp 60/61/62 would not show the claimed cause still holds

**Issue.** GL2c's flag-on gate is "Exp 60/61/62's frozen gates still PASS on the rig". The satiation
crossing fires at Exp 60's own `escape_water` contingency (G7). With a positive value on that act, the
agent can surface on time because surfacing is **rewarded** (`_reward_bias`, or the relief store's
term) rather than because of the situation fear T1-13 claims. A PASS would be recorded MAINTAINED for a
different cause. That is the substrate-learning-channels trap (a second channel saturating the
behavioural signal), with the sign reversed.

**Fix.** The re-run reports the per-component score at every scored decision:
- `NAc.recommend_action` already emits `components` (`causal`, `reward_bias`, `learned_bias`,
  `drive`, `explore`).
- T1-13's annotation already records the `reward_bias` term, 0.0 at all 108 decisions today.

MAINTAINED requires the frozen gate to PASS **and** the decision to be unchanged with the satiation
term zeroed, a counterfactual recomputed offline from the recorded components. A PASS that holds only
with the term is recorded as the new cause, not as MAINTAINED. Put this in GL2c's gate row and in the
joint four-lens review's brief.

### SF-5 (GL4 S0b Minecraft floor; GL5 Minecraft arm): situation drift alone meets the Δoxygen floor

**Issue.** S0b's Minecraft PASS asks for "≥ 100 executed `escape_water` / `move_to` invocations over ≥ 3
oxygen bands with a non-zero Δoxygen spread by band". Underwater, oxygen falls whatever the action is,
so a spread by band arises from the situation alone. The GL5 Minecraft readout ("graded prediction of
Δoxygen/Δhealth by situation") can likewise pass from situation → drift with no action conditioning.

**Fix.** Make the floor and the gate action-contrastive within a band: Δoxygen(`escape_water`) −
Δoxygen(`move_to`), plus an idle/no-op control in the same band, with a non-zero difference required.
Gate the predictor on that difference, not on Δoxygen by band.

### SF-6 (GL5; also GL2b(i)): the word prior has a second path into selection that the plan does not model

**Issue.** The umbrella says "the word embedding is the innate prior" and "the two inputs meet only in
the forward model". The GL5 baseline is an "identity-only (word prior) arm". But
`decisions/nac.py::NAc.recommend_action` scores the tool **name** directly:
- **Component 3** adds `drive_value` when a drive name is a substring of the tool name, and
  `drive_value × 0.7` when a `_DRIVE_TOOL_AFFINITIES` keyword is a substring.
- **These rows fire on cradle bodies.** `runtime/substrate_proposal.py::_DRIVE_CORRECTIVE_NEEDS` maps
  `temp`/`thermal` → `"cold"`, and the `"cold"` row's keywords are `("warm", "fire", "blanket",
  "huddle")`.
- **Tool names carry the entity name.** They are `<entity>_<affordance>`
  (`tool_bridge.py::_resolve_tool_name`, `f"{ent.name}_{aff_name}"`), so the plan's own example
  items match: a cold infant's `warm_rug_touch` and `fire_stone_touch` both get the bonus.
  `water_jet` contains `"water"`, which is the `thirst` keyword.
- **GL2b(i) adds more name keywords** (a `"heat"` affinity row).

This path is the same in every arm, so it cannot by itself make a spurious difference between arms. It
can produce a **null**: a name bonus of up to 0.7 on both misleading items swamps a predicted aversion
at first contact. It also biases first-contact choice toward the names.

**Fix.**
- Name this path in the umbrella's thesis and in S5's blast radius.
- In GL5's prereg, do one of the following:
  - (a) choose fixture names that contain no drive name and no affinity keyword (checked by a test over
    `_DRIVE_TOOL_AFFINITIES` and `_DRIVE_CORRECTIVE_NEEDS`);
  - (b) hold every drive at or below 0.5 at the probe; or
  - (c) count Component 3 as part of the word-prior baseline, and gate on the recorded `components`, so
    the predictor's term is read net of `drive`.
- Freeze GL2b(i)'s keyword list before GL5's fixtures are named.

### SF-7 (GL4 S2, GL4 S0b, GL5): which narrated number gates must be fixed now, because narrated targets carry the word prior

**Issue.** G6 is decided: narrated consequences train at a declared discount, and results are reported
with and without them. I am not re-opening that. But narrated consequences are written by an LLM
narrator reading entity and affordance **names** (`cradle_fire_pit.yaml`: "Proximity effects are
handled by orchestrator sensor writes"). Their targets carry the language prior the claim says
experience overrides. A "with narrated" result is therefore confounded by construction, in the
direction of the claim's opposite (or of a spurious agreement, where the narrator's physics happens to
be sensible).
- S5 already sets the strict default: the gate reads the "without narrated" number.
- S2 says only "the gate is read on the pre-registered one of the two".
- S0b's dissociation counts do not say which provenance they use.

**Fix.** State in the umbrella (§GL5 row) and in S2/S0b that every gate, the dissociation floors
included, reads experienced-only data, and that with-narrated numbers are reported, never gated. Give
the reason: the narrated target is itself generated from names.

### SF-8 (GL5): "untried" leaves out the other per-tool and per-name channels

**Issue.** S5 limits the prior term to "untried tools only (zero `reward_bias` and zero cluster
history)". Other channels can carry value to a held-out item without the forward model:
- **The causal link.** A `tool:<name>` causal link forms on any success (`nac.observe`) and on
  `ToolPainBridge` negatives. `recommend_action` reads it as `causal_pos` / `causal_neg`.
- **EC text-node sharing.** With `MAXIM_SUBSTRATE_PATH=1`, the #1181 widening lets credit spread across
  names that share a text node.
- **Wire-4 situation fear.** It transfers by situation cluster, but is limited to `drive:health` and
  `drive:oxygen`. That matters for any cradle body with a health drive and for the whole Minecraft arm.

Any of these can produce a "transfer" positive, or mask one.

**Fix.**
- Define a held-out item as: no causal link on its tool signature, no `reward_bias`, no cluster
  history, and no shared EC text node with any trained item at the production 0.44 (or report the
  share as the measured word-prior channel).
- Keep every one of these channels identical across the predictor arm and the yoked no-predictor arm.
- Read the gate on decisions where the recorded `components` show `causal == 0`, `learned_bias == 0`
  and no Wire-4 fear term for the probed tools.

---

## NIT

- **NIT-1 (GL2a gate).** The "hand-computed table" for the scripted cradle sequence must be computed
  independently, from the YAML deltas and the specs by hand or in a separate fixture. It must not come
  from calling `sem.drive_comfort_progress` / `drive_span`, or the known-answer check shares code with
  what it checks.
- **NIT-2 (GL1 census).** "Pairs sharing a 0.44 node" depends on encoding order, because the running
  mean moves centroids; #1120 showed order-driven drift at 0.40. Report pairwise cosine ≥ 0.44
  (order-free) as the census statistic, and node co-membership under the production encoding order as
  a second column.
- **NIT-3 (GL2b(ii) gate).** "A single `warm_self` / `*_safe` produces 0 nociception" is tested from
  rest only. Add the post-`touch` state (arm still hot) so the gate covers the order/repetition case the
  plan already names (§3.2, "repeated `warm_self` that stacks").
- **NIT-4 (GL4 S2 guard).** The shuffled-target control should permute consequences **across keys**,
  keeping each key's invocations together, not across invocations. A within-key shuffle leaves key
  identity predictive and can sit above chance for a reason unrelated to the model.

---

## What I verified (in code, at this worktree)

- `embodiment/tool_bridge.py::ModulatorAffordanceTool.execute`: `pre_values` reads root
  `vital_metrics` only, with the "BLIND to modulator drives … until #1161" comment.
  `_drive_progress_by_drive` iterates the declared `self_effect` only.
- `runtime/executor.py`:
  - `_drive_relief` normalises `drive_progress_by_drive` and records movement away from comfort as 0.0.
  - `_drive_pressure_snapshot` resolves qualified drives (`_read_sensor_value`, the #1125 rule) and runs
    before `tool.run`.
  - `_stamp_invocation` writes the trio.
- `memory/encoding.py::encoding_tag` reads `drive_relief` key presence and `drive_pressure` filtered to
  relief keys. `memory/hippocampus.py::_stamp_encoding_strength` sets `storage_strength`, which
  `memory/strategies.py::StrengthStrategy` reads. That strategy is selected by name `"strength"`;
  whether it is the production default is UNVERIFIED.
- `tests/unit/test_agent_loop_selection_golden.py` + `_loop_harness.py`: scripted Minecraft-shaped arms
  only, no cradle/infant body. Captures are recorded as tool + situation, not encoding fields.
- `_data/components/items/cradle_fire_pit.yaml`: `heat_output` initial 0.9; `touch` self_effect is
  `arms.thermal: 0.6`, `core_temperature: 0.15`; `warm_self` is `+0.2 / +0.2`.
- `decisions/nac.py`:
  - `recommend_action` Component 3 does the name-substring drive match and the
    `_DRIVE_TOOL_AFFINITIES` keyword match (`cold`/`thermal` → `warm`, `fire`, `blanket`, `huddle`;
    `thirst` → `water`).
  - The causal links on `tool:<name>` are read as `causal_pos` / `causal_neg`.
  - The `components` dict is recorded per tool.
- `runtime/substrate_proposal.py::_DRIVE_CORRECTIVE_NEEDS`: `temp`/`thermal` → `cold`.
- `embodiment/tool_bridge.py::_resolve_tool_name` / the tool generator: tool names are
  `<entity>_<affordance>`.
- `docs/wiring/substrate-learning-channels.md` (a second, state-blind channel saturates the
  behavioural signal) and `docs/wiring/cosine-separation-is-directional.md`: read and applied in SF-4,
  SF-8 and DNB-2.

**UNVERIFIED:**
- How long `escape_water` actuates, and whether oxygen recovery lands after the tool returns (SF-1).
- The exact share of null-consequence keys in S0b's planned roster (DNB-3). I took the 363/405 figure
  from the plans, not recounted.
- Whether `StrengthStrategy` is the production default (DNB-1). DNB-1 stands either way: the trio is a
  persisted measurement the memory-strength line reads.
