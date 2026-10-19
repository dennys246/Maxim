# E1 v3 design review: wiring lens (real consumers, the real credit path, seams)

Reviewed: e1_conflict_prereg.md DRAFT v1 (2026-10-18)

Charter: [../../DESIGN_REVIEW.md](../../DESIGN_REVIEW.md), the wiring lens. The question: does every quantity the
prereg depends on flow through the shipped loop, or does a replay or the toy hand-compose something the live loop
does differently?

Scratch measurements (offline, shipped code, nothing committed). They are under
`/private/tmp/claude-501/-Users-dennyschaedig-Scripts-Maxim/15239c88-ef6e-47b7-8925-0c2ae097dbd4/scratchpad/`:
`intero_clusters.py` (the real body, the real `_read_drive_states` / `_read_drive_ranges`, a real `SensorEncoder`
over a fresh `EntorhinalCortex`), and `toy_livelike.py` / `toy_livelike_decay*.py`. These import
`learner_sim.py` and add the live loop's bookings to its MAXIM arm, one toggle at a time.

## What I verified

**The argmax replay matches the live selection step on these points:**

- The drives are the real `_read_drive_states` output: raw `food`/`health`/`oxygen` plus the derived `hunger`. The raw values are skipped by the `> 1.0` guard, and the gate's `max(drives.values())` sees them exactly as the live loop does.
- The Wire-4 merge copies `propose_via_substrate`'s three lines: max, never sum.
- `min_confidence` defaults to 0.3. That equals `_resolve_min_confidence` only while `MAXIM_NAC_MIN_CONFIDENCE` is unset.
- B1's stamped 26-tool roster is the post-introspection-filter set.
- θ is `>=` and the activation floor is a strict `> 0.5`. So 0.49 and 0.5 are inert, as the replay says.
- The tie-break is `max(scores, key=(score, name))`, so `flee` beats `escape_water`. This holds under the `minecraft_e1_` prefix too.
- `eat` has no false-match in any roster name.
- The same-tool cap is irrelevant to a single-shot argmax.

**The replay does not match the live selection step on one point.** It calls `tool_dispatch.record_outcome` with no
`clusters`. So it books causal links but no cluster reward bias, and the cluster term of `recommend_action` is
structurally 0 in every cell (DNB-1).

**`record_outcome` as the replays and toy call it, against what `execute_and_learn` passes live:**

- The live loop also passes:
  - `clusters` and `cluster_id`, from the proposal;
  - `drive_potential_diff`, eat's measured food relief;
  - `drive_credit_withheld`, `drive_relief_channel` and `outcome_valence`;
  - `tool_params={}`, `active_goal="survive in the world"`, `elapsed_s`;
  - `bridge_booked`;
  - the proposal's reasoning string.
- Outside `record_outcome`, every executed call also books:
  - the ToolPainBridge's own result link (`record_tool_complete` / `record_tool_failure`, ctx `{"params": {}}`), which compounds across invocations;
  - the PlanHistoryBridge's plan link (`bio_integration.record_plan_outcome` → `planning_bridge.record_plan_outcome`);
  - a TemporalEvent to the distributor, which writes eligibility on `tool:X`.
- At loop level:
  - every iteration runs `_NAC_PER_TICK_DECAYS`, which includes `decay_cluster_reward_biases` at τ = 300 iterations;
  - every loop window is a full session, so `on_session_end` runs `nac.decay_all(0.95)` once per episode.

**Pain wiring.**

- `build_pain_bus` auto-wires four subscribers with an NAc: memory, `pain_nac`, percept valence and Wire 4. The ToolPainBridge adds a fifth. No flag detaches Wire 4. `water_trial`'s ABLATED detach by `__qualname__` is the only precedent.
- Drive pain re-publishes on every 0.3-bubble severity step: `body.py::evaluate_failures` breach latch, with `_BREACH_DEEPEN_FRACTION` 0.05 × comfort band 6.
- `tool_bridge` runs `evaluate_failures` inside every affordance call. Pain therefore publishes mid-call, and the B8 filter keeps it off the bystander tool.

**The policy seam and the floors.**

- `SubstratePolicy` gets `available_tools`, `clusters` and the WORLD values. It gets no NAc and no drives.
- `b1_floor_run.make_policy` returns `None` for FEAR, so FEAR is never armed.
- `WaterTrial.train` (Exp 60/B1 FEAR training) is propose-only. It calls `propose_via_substrate` directly and never executes. `FloorWaterTrial.train_executed` asserts that an `ArmedPolicy` exists.

**The body.** `deep_merge` replaces keys and cannot delete them. `spec.py` sets no modality when `modality: null`, so the
world reader excludes that sensor. `MinecraftWorldBackend.sync_world_sensors` writes only sensors declared
`modality: world`.

**Interoceptive clusters on this body, measured.** The interoception channel splits food into three clusters:

| Encoder | I0 (or J1) | I1 (or J2) | I2 (or J0) |
|---|---|---|---|
| After the training sequence | food 10–18 | food 0–9 | food ≥ 19 (≥ 16 at oxygen 13) |
| A fresh encoder, food descending | food 10–15 | food 0–9 | food 16–20 |

Oxygen 12–20 never splits a cluster. The boundary is order-dependent.

## Findings

### DO-NOT-BUILD

**DNB-1. The predicted outcome rests on a toy and a replay that omit the cluster-reward channel the live credit path
writes. With that channel present, the toy's MAXIM resolves the conflict.**

*Claim attacked:*

- §Predicted outcome: "H-WANT fails in the direction MAXIM < floors … `AVOID_ONLY` … The want's suppression is the shipped wiring's … names the Phase 5 mechanism".
- §What the replays found: "One success books a causal link worth +0.5 … locks the order".

*Evidence:*

- **Live code path.**
  - `tool_dispatch.record_outcome` gets `clusters=proposal.clusters` from `execute_and_learn`.
  - It books `update_cluster_reward` on the interoception cluster: +1 for tool success, ±1 for drive relief.
  - The rate is `reward_bias_alpha` 0.15 per event, capped at 1.0.
  - `recommend_action` sums the cluster term across the active clusters.
- **What the toy and the replay do instead.**
  - `learner_sim.Maxim.outcome` and `e1_argmax_replay._book` pass no `clusters`.
  - The toy's `recommend_action` sees `{"world": "w_water"}` only.
  - The replay sees a constant `"i_any"` that never received credit.
- **The interoception cluster carries the hunger split** (table above). The E1 bands map as follows:

  | Band | Use | Cluster |
  |---|---|---|
  | [11, 13] | train | I0 |
  | [3, 5] | train | I1 |
  | [13, 15] | probe | I0 |
  | [7, 9] | probe | I1 |
  | [1, 3] | probe | I1 |

- **Measured, `toy_livelike.py`.** The toy's own MAXIM, with `clusters={world, interoception}` on both the read and the write, and nothing else changed:
  - probe (bread/pain): food 14 → 0/0, food 8 → 1/0, food 2 → 2/0;
  - 12 bread eaten in training;
  - biases at the end: `I1:eat` +1.00, `I0:escape_water` +1.00;
  - first-select reasoning at food 2: `causal_pos=0.85; causal_neg=0.50; cluster_bias[interoception]=+1.00; drive:hunger(1.00) →eat`.
  - The agent eats twice and then escapes, which is the conflict resolved. The toy's `AVOID_ONLY` shape (0/0 everywhere) holds only with the channel off.
- **The outcome is bistable on harness timing, because the cluster bias decays per loop iteration and not per episode** (`toy_livelike_decay3.py`, cluster decay ×(1 − 1/300) per iteration, session-end `decay_all(0.95)` per window):
  - with ≤ 16 loop iterations outside the episode per episode, MAXIM eats (probe 0/0, 1/0, 2/0);
  - with ≥ 24, `I1:eat` decays to about 0.1 and the toy's `AVOID_ONLY` returns.
  - B1's window warms the loop on the shore for `loop_warm_s` 1 s at `loop_hz` 4 (about 4 iterations), so the shipped schedule sits on the eating side. That is still a toy number.

*Fix (prereg text):*

- Withdraw the predicted outcome and the "suppression is the shipped wiring's" sentence until the prediction is re-derived on the composed loop. The composed loop means:
  - `run_minecraft_aut` against a scripted world with food and eat (SF-5);
  - or at minimum, the toy extended with `clusters`, eat's `drive_potential_diff`, per-iteration decay at the frozen `loop_hz`, and session-end decay.
- State both branches in advance, with the mechanism each implies:
  - **eats:** the state-conditioned cluster credit (`I1:eat`) out-votes the fear's affinity term;
  - **does not eat:** the decay out-runs the credit, or the causal order wins.
- Freeze `loop_hz`, `loop_warm_s` and the window schedule as apparatus. Record loop iterations per window, which is MAXIM's decay exposure. A schedule change is a protocol change.
- Add a `learned_decisive` counterfactual beside `drive_decisive` and the proposed `causal_decisive`: re-run the argmax with `comp["learned_bias"]` removed.
- Report `consulted_bias_by_modality` at each probe's first select. It is already on `NAc_RECOMMEND`.
- Phase 0 item 1b (the `NAc_CREDIT` write event) is now load-bearing for E1. Without it, a WANT result cannot be attributed between the tool-success floor and drive relief, because both write `I1:eat`. Move 1b into E1's instrument (T1), or say why the read side alone suffices.

**DNB-2. MAXIM has no arming, so a hungry MAXIM holding bread acts on the shore. It eats the episode's bread before
the teleport, and books eat credit on the very interoceptive cluster the probes read.**

*Claim attacked:*

- §Protocol: "teleport submerged … arm the policy, run the loop", "Bread: two in hand at every arming".
- §Arms: "Only the POLICY … differs".

*Evidence:*

- `water_trial.WaterTrial.loop_window` starts the loop and sleeps `loop_warm_s` on the shore before `enter()` teleports.
- Only floors are wrapped in `ArmedPolicy`: `b1_floor_run.make_policy` returns `None` for FEAR, and `FloorWaterTrial.train_executed` asserts `self.armed`.
- B1 §The hook relies on "FEAR returns nothing on the shore". That was true for a fed, fearless agent. In E1 a shore agent at food 4 scores `eat` at 0.7 × 1.0 = 0.70. At food 12 the hunger need, 0.4, is below the 0.5 activation floor, so `eat` scores nothing. Any band at food ≤ 10 (need > 0.5) clears `min_confidence` 0.3 on the shore, with no link and no fear. That covers the [3, 5] train band and the [7, 9] and [1, 3] probe bands.
- The interoception cluster is situation-blind (I1 on the shore and in the water alike). Shore eats therefore write `I1:eat` exactly where the water probes read.
- MAXIM's executed training through loop windows has never run on this harness. Exp 60 and B1 trained FEAR propose-only (`WaterTrial.train`). Exp 60's "positive escape link after the preflight refuses" and "assert `get_positive_outcomes(tool) == []`" checks would refuse every E1 MAXIM seed.

*Fix (prereg text):* name the arming seam for the `policy=None` path before the freeze. The options, in order:

- **(a) Recommended.** A typed arming gate on the substrate selection step that idles the `recommend_action` path while disarmed, with B1 rule 17's water-only form.
  - It is a `src/` change behind the fence. It must be byte-identical when unset, pinned by `test_substrate_policy_hook.py`'s pattern, and add a red gate in which a hungry agent with bread executes nothing before the first in-water sample.
  - In `propose_via_substrate` it fires T1-6's "substrate exploration-policy change" by its letter (B1's precedent).
  - In the loop's idle gate it fires Exp 60's "run_agentic_loop idle gate" trigger (T1-13 → T1-14/T1-15).
- **(b) Rejected.** An `ArmedPolicy` around a delegate that calls `nac.recommend_action`. The policy sees no drives and bypasses `min_confidence`, so the delegate would re-compose `propose_via_substrate`'s middle (the D43 family).
- **(c) Insufficient alone.** Give bread by RCON only after the teleport. It stops the bread loss, but not the shore `eat` failure, which books a negative link and `I1:eat` −0.15 every warm-up.

Also:

- Drop the two Exp 60 no-link checks for MAXIM explicitly.
- Refuse a MAXIM arm-seed that has any executed call before its first in-water sample (B1 rule 17's FEAR check).
- Count bread by RCON at the moment of arming, after the teleport, rather than at "prepare".

### SHOULD-FIX

**SF-1. The live body publishes two oxygen pains per pain episode; the toy books one.** *Claim attacked:* the toy's
pain model, "MAXIM never does after its second pain", and the MAXIM sanity gate.

*Evidence:*

- Oxygen is an integer in bubbles.
- The intensity is `drive_pain_for_value` = min(1, (|ox − 20| − 6) × 0.5): 0.5 at 13 and 1.0 at 12.
- The breach latch re-publishes on each step: `body.py`, a 0.3-bubble ε.
- Wire 4 keeps intensity ≥ 0.3 (`create_pain_cluster_fear_subscriber`).
- The committed Exp 60 FEAR row records `oxygen_pain_signals: 19` over 10 episodes (`exp60_trials.jsonl`, seed 11).
- The fear after the first pain episode is therefore −0.75 (the replay's D level), not the toy's −0.5. With the fear at −0.5 the toy MAXIM is inert in episode 2, because 0.5 is not above the activation floor.
- With clusters on, MAXIM takes the 0.5 pain on every food-4 episode while eating, then escapes before 12: the trace shows `PAIN0.5 escape+`.

*Fix:*

- Model the latch sequence in the toy.
- Define AVOID on the saturating publish (intensity ≥ `usable_pain_intensity_min` 1.0, by the harness's pain record), and report sub-saturating publishes per probe.
- Phrase the MAXIM sanity gate in publishes: "≥ 1 saturating `drive:oxygen` publish in training".

**SF-2. A pain inside a blocking call fails that call in the toy, but not live.**

*Evidence:*

- `learner_sim.episode` books the in-flight action `success=False` when the pain lands inside it.
- Live, the tool returns its own result: a successful `eat` books POSITIVE.
- The pain publishes from `tool_bridge`'s `evaluate_failures`, and B8 keeps it off the bystander tool.
- With clusters on, the toy's rule produces 12 training pains against the live rule's 2, and pain on every probe (`toy_livelike.py`).

*Fix:*

- The toy follows the live rule.
- The harness records every call in flight at each pain and at each terminal, with start and return times. B1 rule 14's `calls_all` already does this.
- For the Q floors, state what `end_episode("pain")` credits to an `eat` still in flight.

**SF-3. The TD twin has no seam where the prereg puts it, and the twin as written is not TD(0).** *Claim attacked:*
§Instrument "the TD twin's store behind the same seam" (`floor_policies` / `SubstratePolicy`), and §Arms "TD(0) state
value".

*Evidence:*

- `SubstratePolicy` replaces `recommend_action` and sees no NAc and no drives, so it cannot host a twin of MAXIM.
- A seam does exist without `src/`, using two public pieces:
  - the pain bus: detach Wire 4 (`water_trial`'s ABLATED detach) and subscribe a replacement;
  - the replacement writes the SAME store through `NAc.record_cluster_fear` with intensity′ = max(0, I − |V|), where V = `nac.cluster_fear(agent, world)`. At `cluster_fear_alpha` 0.5 this is exactly V ← V + 0.5(−I − V).
- The read stays `anticipatory_threat_need`, plus the shipped max-merge in `propose_via_substrate`. No trigger fires. A `record_cluster_fear` change in `src/` would fire T1-13/14/15.
- The rule updates only on pain. A TD(0) value also updates toward the observed return on pain-free visits, which is extinction. That cannot be written through `record_cluster_fear`: the valence is clamped ≤ 0 and the intensity is ≥ 0, so the write is monotone.

*Fix:*

- Specify the twin as the replacement subscriber above, pinned by a unit test: publishes 0.5, 1.0 → V −0.25, −0.625; the detach finds exactly one Wire-4 subscriber.
- Call it a "pain-only exponential accumulator (TD target, no extinction)".
- If a true TD(0) is wanted, say it needs `src/` and fires T1-13/14/15.

**SF-4. The floors' code changes are not additive as drafted, and B1's committed rows depend on that code.**

*Evidence:*

- `floor_policies.PRIOR_SUFFIXES = ("_flee", "_escape_water")` and records a violation unless there are exactly two. Q_PRIOR needs three tools (`eat`).
- `raw_key` is (in-water, oxygen bin), with no hunger band.
- There is no world-only `key_mode` for Q_WORLD.
- The reward has no relief term.
- `b1_floor_run` imports these, and its verdict over the committed rows is Phase 0 item 2's acceptance check (i).

*Fix:* every E1 change is a new parameter whose default is B1's behaviour, and `tests/unit/test_b1_floor_policies.py`
passes unchanged. Define the relief exactly:

- **Rule:** +1 when the observation's `food` rose by ≥ 1 across an interval whose pending pick was `eat`. Not Δfood/5: food caps at 20, so a bread eaten in the [13, 15] probe band adds 5–7, not a fixed amount.
- **Lag:** the bridge's eat polls up to 1.5 s for the food packet (`index.js` `case "eat"`). The credit must survive a delta that lands at the next select.
- **WANT source:** take WANT for every arm from one instrument, the RCON bread count before and after each episode. Do not infer it.

**SF-5. The instrument list is missing what the DVs and the composition proof need.** Add:

- **The offline scripted world** (roadmap item 3), through `run_minecraft_aut`. `ScriptedWaterBridge` serves `food` 20 constantly and answers every non-water action `did <name>`. It needs:
  - a settable food level;
  - an `eat` that consumes bread, fails on no bread ("no food in inventory") and fails at food 20;
  - a hunger-setter RCON verb.

  `docs/wiring/harness-loop-must-be-proven-live.md` requires this before any window, and DNB-1's re-derivation runs on it.
- **Phase 0 item (5),** learning-curve rows. E1 v3 is now a learning curve.
- **The decisive counterfactuals:** `causal_decisive` and `learned_decisive` (DNB-1).
- **Per probe select:**
  - `consulted_bias_by_modality`;
  - the world AND interoception cluster ids;
  - `cluster_fear` (the twin's V);
  - the positive and negative links per tool.
- **Per window:**
  - every `drive:*` publish with t and intensity, including `drive:food`. Food < 6 publishes hunger pain in the [3, 5] train band and the [1, 3] probe band. It reaches `pain_nac`, the distributor and memory, but not Wire 4 or the floors' `PAIN_MODES`;
  - executed calls with start and return times;
  - same-tool-cap drops for MAXIM, not only for Q;
  - loop iterations.

**SF-6. The "held-out hunger levels" are not held out for any cluster-keyed learner.**

*Evidence:*

- The measured map puts probe [13, 15] in I0, the [11, 13] train band's cluster, and probes [7, 9] and [1, 3] in I1, the [3, 5] train band's cluster.
- Probe [7, 9] straddles the order-dependent 9|10 boundary.
- The toy keyed Q_EXPLORE on four hunger bands. Its food-8 probe was therefore an untrained key, with all Q₀ ties broken at random. That inflates Q_EXPLORE's 0.58 pain rate there.
- Open question 3, answered: live, B1's cluster tuple gives two water keys across the five bands, not four.

*Fix:*

- The per-row NODE gate reads the interoception cluster as well as the world cluster, loop-off. It labels any probe whose interoception cluster is not one a training episode visited, and refuses on it if that is the intent.
- Word the claim as "food levels not trained at, which the interoceptive encoder maps onto trained clusters".
- Re-key the toy's Q_EXPLORE on the measured split.

**SF-7. Two selection-relevant settings are unpinned.**

- `drive_gate_enabled` resolves from `config.json::sim.drive_gate_enabled` in `build_bio_stack`. `~/.maxim` is shared across worktrees. `water_trial` stamps the explore weight but not the gate, and the replay shows the gate moves a corner.
- The prereg says "loop at 2 Hz", but B1's loop runs at `loop_hz` 4 and proposes at 2 Hz. Decay counts loop iterations.

*Fix:*

- Freeze, stamp and refuse on a mismatch for `drive_gate_enabled`, `substrate_explore_bonus_weight` and `loop_hz`.
- Assert `MAXIM_NAC_MIN_CONFIDENCE` is unset, as B1 does for the tool whitelist.

**SF-8. The variant body works through `extends` only in one form, and three harness consumers move with it.**

*Evidence:*

- `deep_merge` cannot delete. The working form is `saturation: {modality: null}`: `spec.py` sets no modality when `None`, and `_read_world_states` drops it.
- Then `sync_world_sensors` stops writing saturation, because `world_owned_sensors` comes from `modality: world`. `WaterTrial.rescue`'s settle guard (`saturation >= SATURATION_REST`) then reads the initial 10 forever, and passes vacuously. Its food ≥ 16 satiation contradicts E1's hunger protocol in any case.
- `water_trial.APPARATUS_ENCODING` is the shipped body's encoding identity. Its comment requires a variant's identity in the experiment's own `FROZEN` block, passed as `WaterTrial(encoding=…)`.
- The rename happens only if the child sets `entity.name`.
- `exp60_run` and `b1_floor_run` hard-code `entity_ref="bodies/minecraft_player"`.
- Open question 4: `flee` cannot be dropped through `extends`. `flee: null` crashes `spec.py`'s affordance loop (`aff_spec.get` on `None`). Dropping it needs a non-`extends` body copy or a `src/` change to merge semantics, which changes the merge for the ten bodies that extend.

*Fix:*

- Write the variant body YAML in the prereg: `entity.name`, `saturation.modality: null`.
- Give E1 its own rescue (heal, then the hunger setter, no satiation) in a `WaterTrial` subclass. `FloorWaterTrial` is the precedent.
- Give E1 its own `encoding` record.
- Keep `flee` (recommendation for open question 4).

**SF-9. Triggers this prereg fires on other ledger rows by their letter.**

| Change | What it fires | How to discharge |
|---|---|---|
| Item 1a, and any `*_decisive` counterfactual in `recommend_action` | T1-7's "NAc `cluster_reward_bias`/`recommend_action` change" and Exp 45's identical wording (the roadmap names only Exp 56's) | The byte-identical-selection guard |
| An arming seam (DNB-2) | T1-6 or T1-13 → T1-14/T1-15, as listed there | Its own guard |
| A `src/` twin | T1-13/14/15 | Avoid it (SF-3) |
| Edits to `water_trial.py` / `scripted_water.py` | Touch T1-14/T1-15's regression guards | Additive or subclassed only, with the existing tests unchanged |
| Edits to `floor_policies.py` | B1's verdict reproduction | It must hold (SF-4) |
| The variant body, the hunger setter (RCON only) and bread | Nothing (no bridge protocol change) | — |

Put this table in the prereg.

### NIT

- **Causal-link sinks.** The toy's single causal sink stands in for three live ones.
  - Dispatch's `tool_result` link is keyed by the reasoning string, which changes with the scores, so it fragments.
  - The ToolPainBridge result link (ctx `{"params": {}}`) compounds at 0.5 + 0.1√n, as the toy's does.
  - The plan link adds a third.

  Measured: adding the bridge link changes no toy cell. Say so in the toy's docstring.
- **Session-end decay.** `on_session_end` runs `decay_all(0.95)` and a full sleep consolidation once per loop window, so once per episode, because `consolidation="full"`. Measured: no toy cell changes. Disclose it.
- **Temporal credit.** It can raise `reward_bias` on `tool:X`: ToolPainBridge temporal events, then the reaction distributor, then `credit_node`, positive-only, ≤ 0.2. Unmodelled. `score_components.reward_bias` already carries it, so report it.
- **The replay's variant roster.** It appends item 4's verbs and guesses the prefix. The live variant roster must be stamped (B1's `tool_sets`). RANDOM's pain rate depends on roster size: 26 tools versus 34.
- **The same-tool cap.** It has no substrate exemption (`agent_loop._MAX_CONSECUTIVE_SAME_TOOL` 5). A greedy MAXIM's repeated failing `eat` loses every sixth tick. Record those drops (SF-5).
