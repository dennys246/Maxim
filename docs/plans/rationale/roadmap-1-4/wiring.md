# Roadmap 1.4 "Anticipation" — WIRING lens (design review, 2026-09-18)

**Charter** (`docs/experiments/DESIGN_REVIEW.md`): real consumers + real credit path (D43), right
encoding/seams, no hand-composed shortcut that passes while the loop fails. Read: the roadmap in full;
every `docs/wiring/*.md`; CLAUDE.md "A fix ships with a CALLER";
`docs/lessons/shipped-the-pieces-not-the-composition.md`; the code named below. This lens judges
Phase 0 and the wiring the later phases lean on; it does not judge the thesis.

**Verdict in one line:** Phase 0 is the right first step and its five items are the right five, but
three of them are specified in a way that lets the D43 shape through — item 1 names the wrong seam,
item 2 sets a byte-equivalence that is unsatisfiable as written, item 4 specifies a primitive the
substrate cannot select — and two later phases (E1's descent, E2's "relief keyed to a place") describe
mechanisms that do not exist on current wiring. All are fixable in the roadmap text before any build.

---

## Ranked findings

### DO-NOT-BUILD

#### DNB-1 — Phase 0 item 1 names the READ seam for a WRITE-side quantity; the credit write has no event today, and the ledger it asks for would show credit at every step

**What the roadmap says.** "the NAc recommend event carries an aggregate `drive` score with causal
and learned components but no per-need / per-step breakdown … The read must come from the runtime's
own credit path."

**What the code does (verified).**

- The *recommend* event is the SELECTION side: `src/maxim/decisions/nac.py::_emit_recommend_action_event`
  emits `NAc_RECOMMEND` once per `recommend_action` call with `score_components` for the winner
  (`causal`, `reward_bias`, `learned_bias`, `drive`, `explore`). It fires BEFORE the action executes.
  By construction it cannot carry "the credit each earlier step received" — no credit exists yet.
- The *credit* write is a different seam, after execution:
  `src/maxim/runtime/tool_dispatch.py::record_outcome` → `nac.update_cluster_reward(agent_id,
  cluster_id=credit_cluster, tool_signature=sig, reward=±1, source=...)` (line ≈538). **No structured
  event is emitted for this write.** The only nearby emissions are `sim_nac(...)` (the CAUSAL link:
  tool, valence, rpe, confidence — no cluster, no reward) and `tool_bridge.py`'s
  `motor_credit.measured` (tool, affordance, transitions, potential_diff, measured_total — no cluster
  id, no reward that landed). The NAc itself records provenance only as
  `_note_cluster_reward_source` (first-source-wins, "mixed" on conflict) — a stored label, not a
  per-write row.
- `comp["drive"]` is the SUM over every drive that matched the tool
  (`nac.py::recommend_action`, the `for drive_name, drive_value in drive_items` loop adds
  `drive_value` or `drive_value*0.7` per matching drive into one `comp["drive"]`). So "which need
  won" (E1's DV, §Phase 2) is unreadable from the event — a second, separate `src/` change on the
  read side.

**Why the ledger as specified would mislead (verified).** In substrate-primary mode
`agent_loop.py` sets `_drive_relief_only = aut_mode != "substrate-primary"` → **False**, so
`tool_dispatch.record_outcome` falls through to the tool-success floor: every successful executor call
books `cluster_reward = +1.0, source="tool_success"` on the interoception cluster (line ≈515). A
per-step credit column will therefore read **+1 at every step** of any successful primitive sequence.
R4's question "did credit reach step three?" answered by magnitude is vacuously YES. The only
discriminator is `source` (`drive_relief` vs `tool_success`) — and today `source` is not in any row.
This is the `substrate-learning-channels.md` trap restated for a ledger: the state-blind channel gets
there first and looks like the answer.

**Where the relief is anchored (verified).** `tool_bridge.py` captures `pre_values` before
`_apply_sensor_deltas` and measures `_intero_before`/after around the SAME backend call
(`_measured_intero_sensors = _self_pre & _live_pre & _drive_pre` filtered by declared
`modality: world`, lines ≈511–535, ≈596–705). So credit is **executor-CALL-window-anchored**, not
tick-anchored: relief that lands after `call_action` returns (oxygen refilling over the next second
after the head clears; the food packet that `index.js` polls for up to 1.5 s precisely because of this)
credits the NEXT call's tool. For primitives of ≈0.6 s this makes "step ten vs step eleven"
mis-anchoring as likely as "step ten vs step three". The roadmap's R4 statement ("tick-anchored to the
action co-active at relief") should say call-window.

**Also verified: a relief-credit requires a declared `self_effect`.** `_measured_intero_sensors` is
the intersection with `_self_pre` (the affordance's declared `self_effect` keys). A primitive with no
`self_effect: {oxygen: …}` in the body YAML can never receive `drive_relief` credit however
much oxygen it restores — it books the +1 floor only. Item 4's primitives must declare it, and that
declaration is a modeled stub the live backend filters (`drive_credit_withheld` logic) — fine, but it
must be in the variant body or the ledger's `drive_relief` column is structurally empty.

**Required `src/` change (name it in the roadmap):**

1. `src/maxim/decisions/nac.py::update_cluster_reward` emits ONE structured event per write —
   `sim_log("NAc_CREDIT", …)` with `{agent_id, cluster_id, tool_signature, reward, source,
   bias_before, bias_after, tick}` — emitted INSIDE the NAc (the single writer), so every caller is
   covered without a per-call-site edit: `tool_dispatch.record_outcome`, `NAc.credit_operant_reward`
   (→ `cradle_mother.reactive_mother_tick`), and any future relief store. Same fail-soft shape as
   `_emit_recommend_action_event` (ImportError swallowed, everything else propagates). Consider the
   same for `credit_node` (see SF-3: a multi-step credit path partly exists on that surface).
2. `nac.py::recommend_action` adds `comp["drive_by_need"] = {need: contribution}` beside the
   aggregate, carried on `NAc_RECOMMEND.score_components`, so `exp61_run.decision_decisive`'s
   successor can name the winning need (E1's DV).
3. A `drive_decisive` counterfactual mirroring `explore_decisive` (re-run the argmax with
   `comp["drive"]` removed; different outcome → the drive decided). Reason: the current
   `decision_decisive` clause requires `causal == 0 and learned == 0`, which is **impossible after the
   first successful primitive** (the causal link forms on every success, R2's finding). Every E1/E2/E3
   descent would read "not decisive" under the R3 clause. The roadmap should say the R3 clause does
   not transfer.

**The strict red gate (T1's first item) — specify it so a reconstruction cannot pass it:**

`tests/unit/test_trajectory_credit_read.py::test_relief_credit_event_from_the_real_loop`, marked
`xfail(strict=True)` until the emitter ships:

- Build the AUT through `run_minecraft_aut` (`src/maxim/simulation/minecraft_harness.py`), the
  substrate-primary loop with the AUTONOMOUS controller (`_loop_kwargs`), against
  `scripts/survival_world/scripted_water.py::ScriptedWaterBridge` — the SAME path R3's offline campaign
  test uses (`tests/unit/test_r3_run.py::test_offline_campaign_apparatus_and_one_event_per_in_process_arm`).
- Submerge; let the loop select and execute `escape_water`; wait for the shore snapshot.
- Capture with a `RecommendCapture`-shaped sink filtered on `"NAc_CREDIT"` and assert ONE event with
  `tool_signature == "tool:minecraft_player_escape_water"`, `source == "drive_relief"`, `reward > 0`,
  and `cluster_id` equal to the interoception cluster active at the call.
- **Anti-vacuity arms (the D44 rule 5):** (a) the test body never calls `update_cluster_reward`
  itself — assert the event count equals the number of executor calls whose measured relief was
  non-zero (read `motor_credit.measured` beside it); (b) a negative control on
  `ScriptedWaterBridge(escape_delay_s=<longer than the call>)` where the refill lands AFTER
  `action_result` returns, asserting the credit lands on the NEXT call's signature — which pins the
  call-window anchoring as a measured fact in the ledger rather than a sentence in the prereg.

**Recommended roadmap text (replace item 1's last two sentences):**

> **Requires two `src/` changes on two different seams:** (a) the credit WRITE —
> `NAc.update_cluster_reward` emits a structured `NAc_CREDIT` event (agent, cluster, tool signature,
> reward, **source**, before/after bias) from inside the NAc so every caller is covered; today the write
> has no event and only the causal link (`sim_nac`) and the measured potential diff
> (`motor_credit.measured`) are observable, neither carrying the cluster or the reward that landed;
> (b) the READ — `recommend_action` carries a per-need decomposition of `drive` and a `drive_decisive`
> counterfactual (the R3 `causal == 0` clause cannot hold past the first successful primitive). The
> ledger column is `source`, not magnitude: in substrate-primary the tool-success floor books +1 on
> every successful step, so "credit reached step three" means "**drive-relief** credit reached step
> three". Credit is anchored to the executor CALL window (relief after return credits the next call),
> and a relief credit requires the affordance to declare a `self_effect` on the drive sensor.

---

#### DNB-2 — Phase 0 item 4: `move(direction, duration)` cannot be selected by the substrate; and no current route selects ANY primitive toward food, so E1's descent has no consumer

**Verified consumer chain, primitive → credit:** body YAML affordance → `embodiment/tool_bridge.py::generate_tools_for_entity`
(`ModulatorAffordanceTool`, named `_resolve_tool_name(f"{ent.name}_{aff_name}")`) → registry →
`agent_loop.py::propose_via_substrate` (`registry.list()` minus `INTROSPECTION_TOOL_NAMES`) →
`nac.recommend_action(available_tools, current_drives, current_clusters)` → proposal with
`"params": recommendation.get("params", {})` → `run_agentic_loop` → `executor.execute` →
`embodiment/backends/minecraft.py::MinecraftWorldBackend.execute(affordance, params)` →
`MinecraftClient.call_action(name, params)` → `index.js runAction` → `bot.setControlState(...)`
→ `action_result` (+ snapshot absorbed, `sync_world_sensors`) → `tool_dispatch.record_outcome` →
`update_cluster_reward` / `nac.observe` (causal link).

**Three wiring facts the item ignores (all verified):**

1. **`recommend_action` returns `"params": {}`** (nac.py, the final return) and `propose_via_substrate`
   forwards it. The shipped body says it outright: `escape_water` is "Param-free (substrate-primary
   emits no params)" (`minecraft_player.yaml`). A `move(direction, duration)` affordance is
   un-invokable from the substrate path — it would reach the bridge with empty params.
2. **`build_tool_signature` keys `tool:<name>` only** (`tool_dispatch.py::build_tool_signature`; params
   are folded in only for the LLM-era `use` tool). One parameterized `move` is ONE learning key for
   forward/back/left/right/up/down — the substrate could never learn "sink" apart from "swim up".
3. **Nothing scores a primitive on a fresh agent.** `nac.py::_DRIVE_TOOL_AFFINITIES["hunger"] =
   ("eat", "pick_up", "food", "consume", "feed")`; `["threat"] = ("flee", "hide", "retreat",
   "escape", "withdraw", "defend", "shelter")`. `move_forward` / `sink` / `swim_up` match no keyword;
   cluster bias is zero at first contact; the explore bonus is OFF by default
   (`substrate_explore_bonus_weight` 0.0 — and turning it on is the frozen-apparatus config surface
   `docs/wiring/README.md` warns about, and makes exploration the descent driver: a confound). So a
   hungry fresh agent at the top of a lit column scores `eat` (fails: `no food in inventory` → a
   NEGATIVE causal link) and nothing else > 0 → IDLE. **E1's "fresh agent that never descends (a floor
   by design)" is a floor by WIRING** — it never descends because no primitive is selectable, not
   because fear or hunger weighed anything. The carried-fear arm cannot descend either. E1 as sketched
   has no consumer for the descent.

**Where a hand-composed shortcut passes while the loop fails.** The apparatus row for item 4 ("each
primitive's displacement per duration … recorded as an apparatus row") is correctly measured through
`client.call_action` (the "actuation through the BACKEND, never the executor" preflight of
`harness-loop-must-be-proven-live.md`). That measurement proves the bridge moves the bot. It proves
nothing about whether the LOOP can ever pick a primitive — and a harness that then scripts the descent
(teleport to depth, or `call_action("sink")` from the harness thread) to "reach the food" and reads the
resulting eat/relief would produce a real-looking E1 row with zero substrate decisions in it. The
roadmap's own D1 language ("game-native") does not cover this: the shortcut is game-native and still
not the loop. **State the rule:** every primitive that moves the agent in a DV window must appear as
an executed `NAc_RECOMMEND(passed_gate=True)` → `calls[]` pair in the trajectory row, and the row
refuses if any displacement in the window has no such pair (the `settle_guard` idea applied to
position).

**Required text for item 4:**

> Primitives are PARAM-FREE affordances on the variant body — `move_forward`, `move_back`,
> `strafe_left`, `strafe_right`, `swim_up` (jump), `sink` (sneak) — each with a frozen bridge-side
> duration (a per-affordance constant in `index.js`, recorded in the apparatus row), because
> `recommend_action` emits no params and `build_tool_signature` keys on the name alone. Python
> protocol: unchanged (name and params are opaque to `MinecraftClient`). Bridge: new `case`s over
> `bot.setControlState`, each returning after its hold so the executor call spans the motion
> (the credit window). Each declares the `self_effect` its motion plausibly touches
> (`swim_up: {oxygen: +}` stub) or it can never receive drive-relief credit (DNB-1).

**And a new Phase 0 item the roadmap must add — the selection route for a primitive.** Options,
each with its cost; the roadmap should pick one or make the choice E1's prereg's first decision:
(i) a taught bias (Exp 52/56 shape — `credit_operant_reward` from a teacher credits `sink` in the
hungry cluster): the want is TAUGHT, not learned from the game; the "learning" claim narrows to
"the taught want trades against the carried fear"; (ii) a new drive-affinity keyword
(`_DRIVE_TOOL_AFFINITIES["hunger"] += ("descend", "dive")` and naming the primitive `dive`): a
hand-coded prior, fires the Exp 60 row's "`recommend_action` drive-activation floor change" trigger
by its spirit if not its letter; (iii) exploration ON as a frozen apparatus constant: the descent is
exploration, the arm × depth interaction is then a statement about exploration under fear; (iv) an
innate `oxygen`-mirror: none exists for hunger-toward-food-below. None is free. What is NOT an option is
leaving it unstated — the R2 lesson ("the credit is a MESSENGER") already says the prior + causal
link carry selection; a primitive with no prior and no link is not in the argmax.

---

#### DNB-3 — Phase 3 (E2) describes "the relief half of Wire-4's job" — there is no relief half, and relief credit never keys to a world/place cluster on current wiring

**Verified.**
- `nac.py::record_cluster_fear` clamps to `[-max_cluster_fear, 0]`; the field comment at nac.py:399
  reads "fear only in v1 — counter-conditioning …". `_cluster_fear` is keyed
  `(agent_id, cluster_id, failure_mode)`. There is no positive, cluster-only valence store.
- The reward side's only cluster-keyed store is `_cluster_reward_bias`, keyed
  `(agent_id, cluster_id, tool_signature)` — a TOOL is always in the key.
- Which cluster receives drive relief: `tool_dispatch.py::record_outcome` sets
  `credit_cluster = intero_cluster` (`active_clusters.get("interoception")`, line ≈326/446); only
  `drive_relief_channel == "exteroceptive"` reroutes to `operant_cluster` (AUDIO_TAG, else the first
  non-interoception tag). For `oxygen`/`food` — declared `modality: world` on the body, interoception
  membership by `drive:` — the relief credit lands on the **interoception cluster** (the drive-state
  cluster: "drowning", "hungry"), NOT the world cluster (the place). The world cluster is written by
  reward only through `credit_operant_reward` (`cradle_mother.py:257` — an external teacher).

**Consequence.** "Does surfacing into an air pocket earn an oxygen-relief credit keyed to the pocket's
world cluster" has one honest answer on current wiring: **no, by construction** — it earns
`(interoception:drowning-cluster, tool:<whatever was executing>) += 1`. The roadmap's own falsifier
("relief credits the primitive that happened to execute at surfacing, not the place") is not a
falsifier; it is the certain outcome. E2 as written would be run, would record that outcome, and the
roadmap would call it a null on a mechanism that was never in the loop — the D43 shape one level up
(measuring the absence of a composition nothing ships).

**Fix (roadmap text, not code):** move "a cluster-keyed relief store — the positive mirror of
`record_cluster_fear`, or a world-cluster route for measured relief" into §Phase 5's list, name E2 as
the rung that decides whether it is needed, and restate E2's measurable version on current wiring:
"does `(world:pocket-cluster, tool:swim_up)` bias form and read decisive on the return dive" — which
is the Exp 56/57 (operant, world-keyed) shape, reachable only through `credit_operant_reward` today,
i.e. taught. If the roadmap wants game-native place-keyed relief, it is a Phase 5 mechanism entering
BEFORE Phase 3, with its own four-lens review, and §Bodies/§Release T5 should say so.

---

### SHOULD-FIX

#### SF-1 — Phase 0 item 2: "byte-equivalent reports" is unsatisfiable for re-run campaigns; the three copies also DISAGREE, so extraction is a design decision in disguise

**What the three harnesses actually share (verified by reading `r3_run.py`, `exp61_run.py`,
`exp60_run.py`):**

| concern | exp60_run | exp61_run | r3_run | byte-identical? |
|---|---|---|---|---|
| provenance preamble (`evidence_out_paths_or_exit` + `in_process_code_provenance`, exit 3) | `_run` head | `_run` head | `_setup` | yes (≈15 lines ×3) |
| `campaign_id` mint + `--resume requires --campaign-id` | — (run_id) | yes | yes | 2 of 3 |
| `base_row` | own shape | `pair_seed`, `gate_record_code_hash` | `seed`, `apparatus_record`, `anchor_measured` | no |
| `write` | own | `json.dumps(row)` | `json.dumps(row, default=str)` | **no** (a byte difference already) |
| `_existing_clean` (resume key) | `select_run` | `(kind, arm, pair_seed)`, refusal is None | `(arm, seed)`, kind == event, not refusal | **no** |
| supersede rule | duplicate (arm, seed) → INCOMPLETE | later CLEAN supersedes earlier REFUSED; two CLEAN → refused | refused rows skipped on resume | **three different rules** |
| one-hash rule | none | `compute_verdict` → refused list | `report` → INCOMPLETE (+ Amendment 1 ancestry escape) | **no** |
| `trial(...)` → `WaterTrial(...)` | n/a (closures) | yes | yes | yes |
| `wilson_interval`, `_median`, `rss_mb` | — | owns | imports via alias | yes |
| drift (`campaign_drift`, `gauntlet_drift`) | — | quartile medians on ts | gauntlet bands | no |
| report | `compute_verdict` (permutation) | `compute_verdict` (Fisher) | `report` (bootstrap seed 0, MW) | no — and should not be |

So the extractable part is real but smaller than the roadmap implies: the provenance/campaign preamble,
`trial()`, `write` (after choosing `default=str`), the stats helpers. The supersede / hash / resume-key
rules are where the 2026-09-18 amendment defects lived, and they are the part that DIFFERS. Choosing
one rule per concern changes at least two harnesses' behaviour — that is a design decision, and it
should be recorded as one (a table like the above in the extraction PR), not hidden in a "pure
refactor".

**Why byte-equivalence fails as written.** Rows carry `ts`, `campaign_id` (uuid unless passed),
`rss_mb`, `provenance.executed_git_hash`/`python`/`pythonpath`, and every timing DV; the offline
campaign is REAL-TIME (`tests/unit/test_r3_run.py` header: "the frozen stale-sample rule (0.15 s) is a
REAL-TIME property of that process … a 'deterministic' clock would disable the check with extra
steps"; commit d1aa70a4 moved it to the slow lane for exactly this). Two offline runs of the same
campaign do not produce byte-equal rows, so their reports cannot be byte-equal either. The report
functions themselves ARE pure (verified: `r3_run.report` has no clock, `bootstrap_median_ci(seed=0)`;
`exp61_run.compute_verdict` pure; `exp60_run.compute_verdict` pure) — so byte-equivalence is achievable
over FIXED rows.

**Recommended text (replace the diff sentence):**

> …then prove the extraction two ways: (a) the extracted `report`/`compute_verdict` over the COMMITTED
> rows (`docs/experiments/data/r3_cal.jsonl` + `r3_bench.jsonl` + `r3_gauntlet.json` →
> `r3_report_amended.json`; `exp61_pairs.jsonl` → `exp61_verdict.json`; `exp60_trials.jsonl` →
> `exp60_verdict.json`) is byte-equal to the committed report (key order included — `json.dumps`
> insertion order is part of the bytes); (b) the two slow-lane offline campaign tests
> (`test_r3_run.py::test_offline_campaign_*`, `test_exp61_run.py`'s scripted campaign) pass UNCHANGED
> against the core — schema and refusal-reason equivalence, never bytes, because the offline campaign
> is real-time. The three copies disagree on the supersede, resume-key and one-hash rules; the
> extraction PR records which rule each harness keeps (a table), and a harness whose rule changes
> re-runs its offline campaign under the new rule and says so.

#### SF-2 — Phase 0 item 3 (scripted grid world): what it must model for offline runs to prove COMPOSITION rather than a recipe

The current `ScriptedWaterBridge` proves composition because the R3 offline test drives the REAL loop
(`run_minecraft_aut` → `run_agentic_loop`, AUTONOMOUS, `SubstrateTelemetry`) against it and the
bridge answers only what the game answers (NDJSON `state`/`action_result`, RCON reply shapes). A
grid world keeps that property only if it models the things the LOOP's wiring reacts to, not the things
a human thinks of as "the world":

1. **The variant body's roster EXACTLY** — `REQUIRED_BRIDGE_SENSORS` / `missing_bridge_sensors` refuse
   at preflight on a missing key; the body carries every declared sensor at its initial value whether
   or not the bridge writes it (`harness-loop-must-be-proven-live.md`), so a partial scripted roster
   passes a body-side check and hides a dead sensor.
2. **`is_in_water` from EYE height per cell** (the bridge's rule, `index.js` eyeBlock), **oxygen
   drain/refill at the game's rate**, drowning damage 2 hp/s after onset, the respawn seam with the
   scoreboard LEADING the snapshot (already in `ScriptedWaterBridge` — keep it; a grid must not lose it).
3. **Positions, per-cell `light_level`, per-cell air, `y_altitude`** with the variant body's declared
   ranges and clamps (`sensor-range-clamps.md`), and the constant/capped sensors at the values the NEW
   site reads (`distance_from_spawn`, `offset_x/z`) — corollary 6 of
   `cosine-separation-is-directional.md`.
4. **Per-primitive displacement per duration in water vs air — FED FROM item 4's apparatus row**, so
   the scripted constants are live-measured, not typed. A hand-typed 1 block/0.6 s is the stub the
   `run-rig-scripts-offline-first` lesson warns about.
5. **Action latency**: a primitive's `action_result` returns AFTER its hold, so the executor call spans
   the motion; and the **relief lag** (oxygen refills over the next ≈1 s after the head clears; the
   food packet after `consume`) so the call-window mis-anchoring (DNB-1) is REPRODUCED offline and the
   ledger test can pin it.
6. **`eat` semantics**: `no food in inventory` → `ok:false` (a negative causal link — the wiring fact
   that makes the fresh arm's `eat` attempts matter); food/saturation on eat; `data get entity`
   replies for saturation (`ScriptedWaterControl` already answers them).
7. **`flee` fail-fast in water and `move_to` NoPath in water** (the pathfinder-dead contract).
8. **State cadence 100 ms and stale refusals** as real-time properties (do NOT add a fake clock —
   the R3 test header says why).
9. **RCON verbs for the classroom**: `fill`/`setblock` for the light gradient and pocket cells, and
   `execute if block <pocket> minecraft:air` per cell (the surface-air check generalized).

And the offline campaign test for every new harness must run the loop through `run_minecraft_aut`,
never `propose_via_substrate` hand-ticked (the `exp56.common.BenchSession` shape is a hand-ticked
encode/act/record — a recipe by design, fine for Exp 56's question, wrong for a loop claim).

#### SF-3 — Q5, learning-curve rows: no existing row carries within-agent episodes WITH a DV; the pieces to reuse are Exp 57's τ and R3's `lethal_event`, not Exp 60's training loop

**Verified.** `water_trial.py::WaterTrial.train` runs K yoked, propose-only episodes by calling
`propose_via_substrate` directly (no executor, no loop, no execution), and returns
`{usable_episodes, attempts, oxygen_pain_signals, health_pain_signals, episode_clusters, deaths}` — a
conditioning count, no per-episode DV. It is a SETUP loop and cannot become a dependent variable by
relabeling: nothing in it is a behavioural row.

The row shape that DOES carry within-agent repeated episodes with a DV and an interval is Exp 57's
trials-to-criterion τ (right-censored at K_max+1, JT ordered-trend, two censoring-artifact guards —
`docs/plans/behavioral_graduation_candidates.md` row "Dose–response scaling", `scripts/exp57/`), on
the hand-ticked BenchSession. The per-episode UNIT that runs the real loop is R3's
`WaterTrial.lethal_event` (one loop-live submersion; samples, calls, ticks, decision events, pain
seconds, joined on `t0_wall`).

**Recommended text for item 5:** "Within-agent repeated episodes = a list of `lethal_event`-shaped rows
under one `agent_id` with R3's `_boundary` between them (stop motion, reopen the hub session, heal —
the Exp 61 hub-session trap); episodes-to-criterion reuses Exp 57's τ definition, censoring sentinel
and JT trend with its interval. Exp 60's `train` stays what it is: propose-only conditioning, never a
DV row." Also note: the first `decision_decisive`/first-contact read across episodes must handle the
causal link forming after episode 1 (DNB-1 item 3).

#### SF-4 — Statements that describe a mechanism as present or absent contrary to the code (the D43 shape, both directions)

1. **§Thesis R4, "Nothing today credits step three."** Two things partly do (verified):
   (a) the tool-success floor credits every successful step +1 on the intero cluster in
   substrate-primary (DNB-1); (b) `src/maxim/decisions/temporal_credit.py::TemporalCreditDistributor`
   — eligibility traces (`nac.update_eligibility` on `record_event`) → `distribute` →
   `nac.credit_node` — is wired in `runtime/bio_stack.py:472/539` and credits EARLIER events on the
   `_reward_bias` surface (capped `max_reward_bias` 0.20, keyed `(agent, event_signature)`), which
   `recommend_action` reads as `comp["reward_bias"]` (`self.reward_bias(agent_id, event_sig)`). It is
   cluster-blind and capped, so it cannot carry a state-conditioned multi-step policy, but it is a
   multi-step credit path on the selection surface. Whether any Minecraft substrate-primary run ever
   reaches `distribute()` (reaction-driven rewards) is UNVERIFIED by me. Phase 5's "eligibility-trace
   shape the bio-memory brief implies" is describing this; the audit should start by MEASURING it
   (the `NAc_CREDIT` event family should cover `credit_node` too). Recommended text: "no
   state-conditioned credit reaches step three on the selection surface; the cluster-blind eligibility
   path (`TemporalCreditDistributor` → `credit_node`, cap 0.20) exists and is audited first."
2. **§Thesis "Anticipation — no mechanism predicts that a state leads to pain before the pain."**
   `nac.py::anticipatory_threat_need` is literally an anticipatory read: fear keyed to the pre-pain
   cluster fires on re-entry BEFORE the pain (R3 arm C surfaces at ≈3.3 s, before the oxygen-12
   publish at ≈5.8 s — the roadmap cites it). What is missing is prediction across a GRADIENT (a state
   that is not the pain-keyed cluster but leads to it), and `TemporalCreditDistributor.anticipatory_pre_activate`
   exists and is dormant ("the production agent loop never registered the per-tick …",
   temporal_credit.py:128). Say "no mechanism predicts pain from a state that is not itself
   fear-keyed", or Phase 5's forward model will re-implement Wire-4's read.
3. **§Phase 3 "the relief half of Wire-4's job"** — no relief half exists (DNB-3).
4. **§Bodies "Adding primitive affordances (or any sensor) … changes the world nodes' geometry."**
   Affordances are not encoded; only sensors are. Adding an affordance changes the argmax option set
   (true, and enough to fire the triggers); adding a REST-NEUTRAL sensor leaves every encoding
   byte-identical (`cosine-separation-is-directional.md` corollary 4); only a non-neutral or re-tagged
   sensor changes geometry. The sentence over-claims and under-claims at once.
5. **§Phase 0 item 5 "Exp 60's training loop already runs repeated episodes as setup; the dive makes
   them the dependent variable"** — SF-3.

#### SF-5 — §Bodies: the variant body renames every tool signature; say what "carried" carries

**Verified.** Tool names are `f"{ent.name}_{aff_name}"` (`tool_bridge.py`, `generate_tools_for_entity`),
and a child body declares its own `component.name` (`minecraft_bench57.yaml`: `name: minecraft_bench57`
under `extends`). A variant named anything but `minecraft_player` yields `tool:<variant>_escape_water`
etc. Consequences the roadmap must state:

- **Fear carries; biases do not.** `_cluster_fear` is `(agent, cluster, failure_mode)` — tool-free — so
  a C-protocol agent's drowning fear is readable on the variant IF the world cluster id matches (the
  variant's world roster must encode byte-identically for the same values: rest-neutral additions
  only, or the carried fear keys a cluster the variant never activates). `_cluster_reward_bias` keys
  carry `tool:minecraft_player_*` and match nothing on the variant. E1's arms must say whether the C
  agent is trained ON the variant (its own C protocol) or carried from the shipped body (fear only).
- **Bundle-carried (Exp 61 arm D shape) refuses across bodies:** gate 7
  (`hivemind/ingest.py`, `BundleBodyMismatch`, `derive_capability_map`) — a shipped-body donor cannot
  be ingested into a variant-body receiver unless the capability map matches
  `<modulator>/<affordance>`; adding primitives changes the map.
- The variant's tool prefix should be chosen once and frozen (it is in every ledger key).

#### SF-6 — §Phase 0 exit / T1: "verified with the real consumer on one Exp 60-style relief" — say which consumer

The real consumer of the credit is `recommend_action` on the NEXT tick, not the event sink. T1 should
require both: the `NAc_CREDIT` event (the write happened) AND a subsequent `NAc_RECOMMEND` whose
`consulted_bias_by_modality[interoception]` reflects it (the read happened) — the Exp 56/D44 rule
"read the outcome at the real consumer … never the emitted action" applied to the instrument itself.

---

### NIT

- **N-1** `state_age_s` is already on every sample (`WaterTrial.sample_full`) and the loop refuses stale
  snapshots (`minecraft_harness.py:33/309`); the trajectory row is the JOIN `lethal_event` already
  performs on `t0_wall` (samples × ticks × calls × decision events). Item 1's new column is the credit;
  say "extend `lethal_event`'s row", not "build a row".
- **N-2** `exp61_run._Campaign.write` uses `json.dumps(row)`; `r3_run._R3.write` uses `default=str`. The
  extraction changes one of them — pick `default=str` and note it.
- **N-3** `RecommendCapture` is a PROCESS-WIDE sink (`register_sim_sink`); `lethal_event` already
  filters by `agent_id`. Any `NAc_CREDIT` sink must filter the same way or a leaked loop from a refused
  row bleeds credit events into the next row (the comment in `lethal_event` describes exactly this).
- **N-4** "the same RCON verbs" (item 3): `ScriptedWaterControl.command` answers seven verb shapes;
  a classroom with pockets needs `setblock`/`fill` per cell and `execute if block` per pocket — list
  them so the scripted control cannot silently answer `""` (its default) to a verb the harness gates on.
- **N-5** §Phase 0 cadence paragraph: `loop_tick_probe.py` exists (verified); the tie-break tax is
  `flee` sorting before `escape_water` by NAME in the argmax — adding six primitives adds six names to
  that sort; if two score equal the name decides. Freeze primitive names with that in mind
  (a `sink` that sorts before `swim_up` wins ties underwater).
- **N-6** Cited files all exist (verified): `deferred/credit_on_progress_not_execution.md`,
  `deferred/transition_based_drive_pain.md`, `three_factor_credit_assignment.md`,
  `deferred/retrosplenial_spatial_frames.md`, `deferred/jepa_cross_modal_alignment.md`,
  `docs/experiments/data/exp62_cross_pool_replay.py`, `roadmap_1_3_path.md`.

---

## Answers to the six charter questions

1. **Per-step credit read** — NOT specified from the real credit path as written: it names the
   `NAc_RECOMMEND` (selection) event, whereas the credit is written in
   `tool_dispatch.record_outcome → NAc.update_cluster_reward` with no event today. Required:
   `nac.py::update_cluster_reward` emits `NAc_CREDIT` (and `credit_node` the same); `recommend_action`
   adds `drive_by_need` + `drive_decisive`. Red gate: the strict test in DNB-1, on `run_minecraft_aut` +
   `ScriptedWaterBridge`, with the two anti-vacuity arms (no test-side write; the after-return relief
   lands on the next call).
2. **Campaign core** — honest in intent, wrong in its exit: the shared part is the provenance/campaign
   preamble, `trial()`, `write`, stats helpers; the supersede/resume/one-hash rules DIFFER across the
   three and are the layer the amendment defects lived in. Byte-equivalence is real only for the pure
   report functions over the COMMITTED rows; the offline campaigns are real-time and can only be
   schema-equivalent. Nondeterminism sources: `ts`, uuid `campaign_id`, `rss_mb`, `provenance`, every
   timing DV, stale-sample refusals by host timing.
3. **Primitives** — no Python protocol change (names/params opaque); bridge `case`s; a VARIANT body
   (fires no shipped-body trigger but renames every tool signature, SF-5); an executor tool per
   primitive is generated automatically — BUT they must be param-free and each declare a
   `self_effect`, and no current scoring route selects one toward food (DNB-2). Shortcut that passes
   while the loop fails: harness-driven `call_action`/teleport descent with a real relief at the end.
4. **Scripted grid world** — SF-2's nine items; composition is proven only by driving
   `run_minecraft_aut` against it, with primitive displacements fed from the live apparatus row.
5. **Learning-curve rows** — no existing row carries within-agent episodes with a DV; reuse Exp 57's τ
   + R3's `lethal_event` as the episode unit with `_boundary` between; Exp 60's `train` is not it.
6. **D43-shape statements** — five, listed in SF-4 (R4 "nothing credits step three"; "no mechanism
   predicts"; "relief half of Wire-4"; "affordances change geometry"; "training loop becomes the DV").

## Verified vs inferred

**Verified by reading code/tests/data:** everything cited with `file::symbol` above — `nac.py`
(`_emit_recommend_action_event`, `recommend_action` scoring loop and final return, `update_cluster_reward`,
`_note_cluster_reward_source`, `record_cluster_fear` clamp, `anticipatory_threat_need`, `credit_node`
surface, `_DRIVE_TOOL_AFFINITIES`); `tool_dispatch.py` (`build_tool_signature`, `record_outcome`'s
credit branches and `credit_cluster`/`operant_cluster` routing); `tool_bridge.py` (`_drive_potential_diff`,
the `_self_pre & _live_pre & _drive_pre` measured set, `motor_credit.measured` fields, tool naming,
`derive_capability_map`); `agent_loop.py` (`propose_via_substrate`, `_drive_relief_only`);
`temporal_credit.py` + `bio_stack.py:472/539`; `minecraft.py` protocol + `state_age_s`; `index.js` verbs
and snapshot; `scripted_water.py`; `water_trial.py` (`lethal_event`, `sample_full`, `train`,
`_telemetry_ticks`); `r3_run.py` / `exp61_run.py` / `exp60_run.py` campaign layers and reports;
`tests/unit/test_r3_run.py` offline campaign test; `component_registry.py` `extends`;
`minecraft_bench57.yaml` child `name`; the graduation ledger's Exp 60/61 triggers; the cited files'
existence; `docs/experiments/data/` committed rows and reports.

**Inferred / not verified:** whether any Minecraft substrate-primary run ever reaches
`TemporalCreditDistributor.distribute()` (SF-4.1); the exact per-cell semantics Mineflayer exposes for
air pockets and light under water (SF-2, environment lens's domain); Exp 57's harness path
(`scripts/exp57/`) was cited from the ledger row, not re-read; that `escape_water`'s YAML entry declares
a `self_effect` on `oxygen` (I read its comment block, not the effect keys).
