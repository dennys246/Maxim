# Roadmap 1.4 "Anticipation" — bio-faithful lens (design review of a ROADMAP)

**Reviewed:** `docs/plans/roadmap_1_4.md` DRAFT v1 (2026-09-18). **Lens charter:** does each rung test the
mechanism's REAL job, not a caricature; does the manipulation respect how the substrate/body/drives actually
work; does the metric measure the mechanism's actual purpose. **Read:** the roadmap in full; `docs/agents/bio-memory.md`,
`docs/agents/runtime-tools.md`, `docs/agents/simulation-experiments.md` (survival-world rows); `docs/wiring/pain-needs-declared-failure-modes.md`,
`docs/wiring/cosine-separation-is-directional.md`, `docs/wiring/substrate-learning-channels.md`, `docs/wiring/world-light-sensing.md`;
R3 §Outcome + §Dependent measures; Exp 62 v2 §"What this means for the pressure idea"; Exp 60 §Outcome; the three
ledger rows (Exp 60/61, Exp 56). **Code read (not paraphrased from docs):** `decisions/nac.py` (`recommend_action`,
`update_cluster_reward`, `cluster_reward_bias`, `set_pending_operant_action`, `credit_operant_reward`, `note_active_clusters`,
`record_cluster_fear`, `cluster_fear`, `anticipatory_threat_need`, `update_eligibility`, `distribute_reward`, `_DRIVE_TOOL_AFFINITIES`,
`NACConfig` defaults), `decisions/temporal_credit.py` (`record_event`, `anticipatory_pre_activate`, `distribute`),
`runtime/agent_loop.py` (`_DRIVE_CORRECTIVE_NEEDS`, `_read_drive_states`, `_SUBSTRATE_CHANNELS`, `propose_via_substrate`,
`_loop_bio_tick_maintenance`), `runtime/tool_dispatch.py` (`record_outcome` credit routing, `build_tool_signature`),
`embodiment/tool_bridge.py` (the measured-relief block), `embodiment/sem.py` (`drive_pain_for_value`, `corrective_need_intensity`,
`drive_comfort_progress`), `embodiment/body.py` (`evaluate_failures` latch, `_publish_drive_pain`, `_emit_drive_temporal_event`),
`proprioception/pain_bus.py::create_pain_cluster_fear_subscriber`, `bridges/tool_pain_bridge.py` (TemporalEvent producer),
`runtime/bio_stack.py` (the `distribute` caller), `embodiment/component_registry.py::_resolve` (`extends`),
`_data/components/bodies/minecraft_player.yaml`, `scripts/minecraft_bridge/index.js` (the oxygen read).

**Verified vs inferred** is stated per finding; the summary list is at the end.

---

## Findings, ranked

### DO-NOT-BUILD

**DNB-1 — Phase 3 (E2) "relief keyed to a place" tests a write channel that does not exist; its stated falsifier is the shipped behaviour.**

*Roadmap sentence attacked:* "Does surfacing into an air pocket earn an oxygen-relief credit keyed to the pocket's world cluster …
This is the relief half of Wire-4's job stated for a *place*" and "**Falsifier:** relief credits the primitive that happened to
execute at surfacing, not the place."

*Mechanism facts (all VERIFIED by reading):*
1. **Wire-4 has no relief half.** `decisions/nac.py::NAc.record_cluster_fear` is the only in-session writer of `_cluster_fear`;
   the update is `current − alpha·intensity`, clamped `max(−max_cluster_fear, min(updated, 0.0))`. There are exactly two writes to
   `_cluster_fear[...]` in the file (the record and `load_state`), both clamped ≤ 0. No `record_cluster_relief` / extinction
   write exists; the only decrease in |fear| is `apply_wall_clock_decay` (the SLOW 7-day class). Wire-4 is a fear-only,
   one-way store.
2. **No survival-path relief credit ever lands on the WORLD cluster.** `runtime/tool_dispatch.py::record_outcome`: `credit_cluster =
   intero_cluster` by default; the extero (`operant_cluster`) route fires only when `drive_relief_channel == "exteroceptive"`, which
   `embodiment/tool_bridge.py` sets only from *backend-reported* `measured_drive_transitions` (the DoA/azimuth motor backend) — never for
   a world-owned drive. The tool-success floor is intero-only by the "seam routing" comment. The single shipped world-keyed positive
   write is `NAc.credit_operant_reward` under `MAXIM_OPERANT_ONLY_CREDIT` with a TEACHER (cradle_mother, Exp 52/56 — the ledger row
   for Exp 56 says exactly "operant-only credit on the WORLD cluster … teacher-delivered").
3. **Surfacing produces no measured oxygen relief at all.** `tool_bridge.py`'s locally-measured set is
   `self_effect ∩ live_world_set_sensors ∩ drive_specs ∩ modality:world`. `escape_water` in `minecraft_player.yaml` declares NO
   `self_effect`, so `_intero_before` is empty and `_measured_intero_sensors` is ∅. What surfacing books today is (a) a state-blind
   POSITIVE causal link on `tool:escape_water` (`nac.observe`, the Exp 60 caveat: 47–51 links per seed) and (b) the +1 tool-success
   floor on the INTEROCEPTION cluster that was active at PROPOSAL time (`clusters` are stashed on the proposal in
   `propose_via_substrate`, not re-encoded at surfacing).
4. Therefore the E2 falsifier ("credits the primitive, not the place") is not a falsifier — it is the code, readable in
   `record_outcome`. A live E2 on the shipped credit path can only return the null the code predicts; the "trajectory ledger" would
   confirm what the read already says.

*Bio-faithful reading.* The real biological job E2 is reaching for is conditioned place preference / safety-signal learning
(relief as a positive reinforcer bound to a context) — a genuine job, and a different store from Pavlovian fear. The codebase has
the fear store (Wire-4) and an action-keyed relief credit (interoception cluster), but no context-keyed relief store. That is a
Phase-5 mechanism by the roadmap's own rule ("each entering only when a rung names it"), and it is already nameable from the code
without a live run.

*Substitute text (Phase 3):* "E2 is gated on a named mechanism, not on a run. Reading `tool_dispatch.record_outcome` and
`tool_bridge` establishes today: (i) `escape_water` declares no oxygen `self_effect`, so surfacing measures no relief; (ii) relief
and tool-success credit route to the interoception cluster only; (iii) `_cluster_fear` is a fear-only store with no in-session
relief/extinction write. A place-keyed relief credit is therefore a Phase-5 mechanism (candidate shapes: a surfacing affordance
that declares `self_effect: {oxygen: …}` so measured relief exists + a world-cluster credit route for interoceptive relief, or a
positive sibling of `_cluster_fear` with its own allowlist) that enters through its own plan and four-lens review BEFORE E2 is a
rung. E2's live question is then 'does the new write make the agent return to the pocket', with the shipped path as the
recorded-null control arm."

---

**DNB-2 — Phase 2 (E1) "a want against a fear" has no want-side consumer on the primitive motor layer; as sketched it measures
fear against exploration noise, and its decisive DV cannot be read.**

*Roadmap sentence attacked:* "A hungry agent, food at depth *d* … Does the resolution of hunger-relief against the carried
drowning-fear change with *d*?" and "decision provenance on the executed descent (drive-decisive — which need won)".

*Mechanism facts (VERIFIED):*
1. The hunger need is real and game-derived: `agent_loop.py::_DRIVE_CORRECTIVE_NEEDS` maps `food → "hunger"` and
   `sem.py::corrective_need_intensity` grades it from `satisfaction_threshold` 16 down to `deprivation_threshold` 6 on the bridge-written
   `food` value. So the want in E1 is a drive, not an LLM prior (see Q4 below). But a need only moves behaviour through
   `nac.py::_DRIVE_TOOL_AFFINITIES["hunger"] = ("eat", "pick_up", "food", "consume", "feed")` or a name substring match. A primitive
   roster (`move_forward`/`sink`/`swim_up`/…) matches none of these; `eat` requires held food; `move_to(x, z)` needs params and
   `recommend_action` emits none. **On a fresh substrate nothing hunger-driven can take the agent DOWN.**
2. The fear side has a direct innate consumer: `threat` (max-combined from `anticipatory_threat_need`) → the "escape" keyword →
   `escape_water`. So the "conflict" is one reflex arc against no arc. The only thing that descends a fresh agent is the
   explore-first hard gate (`substrate_explore_bonus_weight > 0` forces one trial of EVERY untried tool) — i.e. novelty, which also
   defeats the proposed anti-vacuity floor ("a fed agent that never descends"): with primitives on the roster and exploration on,
   a fed agent descends by the gate at least once.
3. Once a descent has happened, the only learned signal that could make hunger select `move_forward` is `cluster_reward_bias` on the
   interoception cluster — the channel R2 measured as MESSENGER, not cause (`docs/wiring/substrate-learning-channels.md`,
   "do NOT design an experiment whose claim requires the cluster credit to be the behavioural cause").
4. The DV "which need won" is unreadable today: R3 §Dependent measures records that the `NAc_RECOMMEND` event "carries no per-need
   breakdown … a `src/` change" — Phase 0 item 1 promises the fix, but E1 cannot freeze until it lands (T1 before T3 is in the
   roadmap; the dependency should be explicit in E1's text).

*Substitute text (Phase 2):* "E1 requires a named APPETITIVE consumer on the variant body before its prereg: a game-native,
param-free approach affordance the hunger affinity matches (the `pick_up`/`food` family — e.g. `swim_to_food`, a fixed action
pattern like `escape_water`), so that hunger has an arc of the same kind as fear. If no such affordance is added, E1 is a
fear-vs-exploration test and must say so. The 'which need won' DV depends on Phase 0 item 1 and is a freeze precondition. The
anti-vacuity floors are re-stated with exploration OFF (or the explore-first gate's one forced trial subtracted)."
Note the tension with owner decision (4): a `swim_to_food` fixed action pattern is a compound verb. The bio-faithful position is
that the two sides of a conflict must live on the SAME motor layer (both compound, or both primitive with a learned approach),
otherwise the result names the layer asymmetry, not the drive resolution.

---

### SHOULD-FIX

**SF-1 — "The drives are reactive … No mechanism *predicts* that a state leads to pain before the pain" is false as stated; the
true gap is that the prediction is UNGRADED and UNTIMED.**

*Roadmap sentence attacked:* Thesis bullet "Anticipation — the drives are reactive. Pain arrives, fear is keyed to the state
co-active with it (Wire-4). No mechanism predicts that a state leads to pain before the pain."

*Mechanism facts (VERIFIED):* Wire-4 IS a state→pain predictor: `create_pain_cluster_fear_subscriber` keys fear to the co-active
world cluster; `anticipatory_threat_need` reads it on the NEXT entry into that cluster; Exp 60 EARNED "leaves the water a median
3.4 s BEFORE its own air-hunger pain would have fired", and R3 arm C surfaces at 3.18 s against a 5.1–5.4 s pain edge. That is a
Pavlovian CS→US prediction firing before the US — biologically exactly "a state predicts pain". What it is NOT: graded or timed.
`anticipatory_threat_need` returns `min(1, −deepest_fear)` gated at θ = 0.5 — a step on cluster IDENTITY; `is_in_water` is on from
dive second 0, so the need is 1.0 at the surface and 1.0 at the bottom. `cluster_fear_alpha` 0.5 caps fear in two publishes.
Nothing reads oxygen level, time-under, or distance-to-band (`corrective_need_intensity` fires only PAST the band, and `oxygen` has
no corrective need by design — the YAML comment). The timed predictor the codebase does have, `TemporalCreditDistributor.anticipatory_pre_activate`
(oscillator-predicted event → primed eligibility), is **Dormant since 2026-05-26**, and its drive-side producer
`Body._emit_drive_temporal_event` is **Dormant (D9)**.

*Substitute text:* "Anticipation — the drives are reactive and Wire-4's anticipation is ungraded. Wire-4 already predicts pain
from a state (a Pavlovian CS on the world cluster; Exp 60/R3-C fire ≈ 2–3 s before the pain edge) but as a step on cluster identity:
the need is 1.0 on entry and 1.0 at the bottom. No mechanism predicts WHEN pain arrives or how much budget remains, and nothing lets
a want hold the agent inside a feared situation for a bounded time. The dormant SCN anticipatory pre-activation
(`temporal_credit.py::anticipatory_pre_activate`, producer D9) is the in-tree timed predictor any forward-model plan must
front-gate against, alongside Wire-4." (Also add it to the §JEPA front-gate list, which today names only Wire-4 and the
temporal-credit path.)

---

**SF-2 — R4 "credit is tick-anchored to the action co-active at relief; a payoff ten actions away credits the tenth; nothing
today credits step three" misdescribes three different credit paths and hides the one that already reaches step three.**

*Roadmap sentence attacked:* the R4 thesis bullet and Phase 5 "Sequence credit (R4) … a credit rule that reaches earlier steps".

*Mechanism facts (VERIFIED):*
1. **Measured relief credit** (`tool_bridge` → `tool_dispatch`): SAME-CALL, zero-lag by construction — the acting affordance's own
   before/after on its DECLARED drives, signed to ±1, keyed `(agent, interoception cluster at proposal, tool:<name>)`. Not
   "tick-anchored"; it cannot reach any other action even one tick back.
2. **Operant credit**: `set_pending_operant_action` is an explicit one-step memory ("Overwrites any prior pending action … the
   relief that arrives next credits exactly the action that produced the rewarded state"); `credit_operant_reward` credits that
   last action. "Credits the tenth" is accurate for THIS path only — and it runs only under `MAXIM_OPERANT_ONLY_CREDIT` with a
   teacher (cradle_mother / Exp 56), not on the survival loop.
3. **Eligibility traces DO reach step three today.** `bridges/tool_pain_bridge.py::record_tool_start` emits a `tool:<name>`
   `TemporalEvent` → `TemporalCreditDistributor.record_event` → `NAc.update_eligibility`; `agent_loop.py::_loop_bio_tick_maintenance`
   decays them at 0.9/tick; `distribute` credits ALL traced signatures proportionally, with the SCN phase fallback at 0.3×. At the
   measured 0.58 s idle tick, step 3 of a ten-step path still holds 0.9⁷ ≈ 0.48 of its activation at step 10. What is wrong with
   this path is not reach but SURFACE and FEED: it writes `_reward_bias` (`credit_node`), capped at `max_reward_bias` 0.20, read by
   `recommend_action` as component 2 (positive-only) — it cannot out-vote a +1 cluster bias or a ≥ 0.5 causal link; and its only
   caller is the reaction-path reward subscriber in `runtime/bio_stack.py` (pain-valence driven). Whether a surfacing/eating relief
   on this world ever produces a POSITIVE reward on that path is **INFERRED not verified** (the subscriber is a pain/reaction
   subscriber; the entropic "satisfaction Reaction" the `EntropicDriveSpec` docstring names would be the food-side producer, and
   homeostatic oxygen has none).

*Substitute text:* "R4 — the selection-surface credits are zero-step (measured relief: the acting call only) or one-step
(operant: the last action). A multi-step, decaying credit over ACTIONS already exists (tool eligibility traces via
`ToolPainBridge` → `TemporalCreditDistributor.distribute`, 0.9/tick, phase fallback) but lands on the 0.20-capped recognition
bias and is fed only by the reaction-path reward. Phase 5's first audit is therefore 'route the existing trace credit to the
selection surface / feed it from measured relief' (rides on existing infrastructure — the front-gate rule), before any new
credit rule." Add this to the Phase 5 candidate list ABOVE the three deferred plans.

---

**SF-3 — The pain-free dive budget is ≈ 5.1–5.4 s, not ≈ 6 s; the ascent and first-tick costs leave ≈ 3 s of bottom time,
i.e. one primitive in ~4–5, not "one primitive in ten".**

*Roadmap sentence attacked:* "against a pain-free dive budget of ≈ 6 s (oxygen 20 → the oxygen-12 publish at ≈ 5.8–6.1 s,
R3 arm B) they are one primitive in ten."

*Mechanism facts:* `sem.py::drive_pain_for_value` (homeostatic): `excess = |v − 20| − 6`; pain > 0 when oxygen < 14, i.e. at the
first reading of 13 (VERIFIED). The bridge writes `bot.oxygenLevel` (VERIFIED, `index.js`), a 0–20 bubble scale; Minecraft
depletes one bubble per 15 game ticks = 0.75 s (INFERRED from the game, corroborated: 20 → 13 = 7 × 0.75 = 5.25 s, matching R3's
apparatus row "pain edge 5.09–5.44 s" and Exp 60's `pain_edge_min_s` 5.085). The "oxygen-12 publish" is when arm B's FEAR BOOKING
was observed — at 13 the pain is 0.5, at 12 it saturates at 1.0 (`(8 − 6) × 0.5`), and the loop samples at 0.58 s, so the first
publish the harness sees is 12 at ≈ 6 s. That is the fear-acquisition timestamp, not the pain edge; the pain the DV integrates
(`drive_pain_for_value` over the sample series, R3's own rule "NEVER from the pain publishes") starts at 13. Subtract R3's measured
components (first proposal tick ≈ 1.0 s after the teleport; ascent from depth 5 = 1.43 s; a `flee` tie-break tax of 0.77 s if it
fires) and the usable bottom time at depth 5 is ≈ 2.0–2.9 s ≈ 3–5 loop ticks. "Pain-seconds" as a DV is essentially
duration-below-14 (pain saturates by 12), so "oxygen minimum per dive" carries the graded information.

*Substitute text:* "against a pain-free budget of ≈ 5.1–5.4 s from submersion to the first oxygen-13 reading (R3 apparatus
record; the oxygen-12 publish at ≈ 6 s is the fear-booking timestamp, not the edge), of which ≈ 1 s is the first proposal tick and
1.43 s the ascent from depth 5, they are one primitive in four or five." Carry the corrected number into E1's depth choice and E3's
feasibility pilot.

---

**SF-4 — E3's within-agent learning curve cannot come from fear modulation: `_cluster_fear` is a one-way ratchet with no
in-session extinction, and the two needs tie at the argmax.**

*Roadmap sentence attacked:* Phase 4 "do dive lengths converge on the pain-free budget *from below* … while the food is still
reached?" and "episodes-to-criterion".

*Mechanism facts (VERIFIED):* fear only deepens in session (DNB-1 fact 1) and reaches the −1.0 cap in two publishes
(`cluster_fear_alpha` 0.5). The need it emits is a constant 1.0 on the water cluster thereafter. In `recommend_action` a
drive-affinity match scores `need × 0.7`; hunger at its 1.0 cap on an affinity-matched approach tool also scores 0.7; ties resolve by
name sort — deterministic, not graded by depth or oxygen. So "convergence from below" across episodes has no fear-side gradient
to ride; it could only come from the cluster-bias channel (the messenger, R2) or from a mechanism not yet built. Biologically:
extinction requires unreinforced CS exposure (the `NACConfig` comment at line ~342 says so), and no unreinforced-exposure write
exists.

*Substitute text:* "E3 is explicitly a MAY-FAIL rung whose null is predicted by the shipped wiring: the fear need is a saturating
step with no in-session extinction, so a learning curve on dive length requires either a graded/timed need (SF-1's gap) or a
selection-surface credit that reaches earlier steps (SF-2). The prereg must name which of the two its arms can distinguish, and a
null is the expected result on the shipped path."

---

**SF-5 — Owner decision (4), "primitive movements", is compatible with the substrate only as param-free PER-DIRECTION tools,
and the fear consumer must sit on the same layer as the want consumer.**

*Roadmap sentence attacked:* "Primitive motor verbs on the real bridge — `move(direction, duration)` over Mineflayer control
states".

*Mechanism facts (VERIFIED):* `tool_dispatch.py::build_tool_signature` returns `tool:<name>` for every tool except `use`, so
`move(direction=…)` collapses to ONE learning key `tool:move` — the substrate could not learn "forward is good here, back is bad
here". `recommend_action` emits no params, so a parametrised `move` is not selectable by the substrate at all. The 1.2 cradle
orient line learned precisely because `turn_left` / `turn_right` were separate affordances. Durations cannot be learned either — a
fixed duration per tool is the only shape the option set supports. Separately, if `escape_water` (compound: swim up UNTIL air) stays on
the variant roster beside a `swim_up` primitive, the fear need always selects the compound (only `escape_water` matches
`_DRIVE_TOOL_AFFINITIES["threat"]` by name unless the primitive is named with a threat keyword), so the fear side bypasses the
primitive layer while the want side (if it ever gets a consumer, DNB-2) lives on it.

*Bio-faithful reading.* A roster of fixed action patterns per direction is a legitimate motor layer (that is how the orient policy
formed), and richer repertoires are how motor development unfolds; the substrate learns state→FAP. What is not faithful is
mixing layers across the two sides of a conflict. *Substitute text:* "Primitive motor verbs — one param-free affordance per
direction with a fixed pilot-measured duration (`move_forward`, `move_back`, `strafe_left/right`, `swim_up`, `sink`, plus the two
turns); a parametrised `move(direction, duration)` is not a substrate-selectable or substrate-learnable shape
(`build_tool_signature` keys on the name). The fear and the want consumers must be declared on the same layer for E1/E3 — either
both fixed action patterns, or `escape_water` withdrawn from the variant roster and `swim_up` given the threat-affinity name."

---

**SF-6 — E1's "carried fear" arm is only coherent if the C-protocol training runs on the VARIANT body or the node gate is
re-run.**

*Roadmap sentence attacked:* §Bodies "The classroom therefore uses a variant body (`extends: bodies/minecraft_player`) … the
shipped body's EARNED rows stay EARNED" and Phase 2 "carried fear (a C-protocol agent)".

*Mechanism facts (VERIFIED):* `component_registry.py::_resolve` deep-merges the child's entity spec over the parent's, so a variant
that adds only AFFORDANCES leaves the sensor roster — and therefore every world-channel vector and EC node id — byte-identical: a fear
learned on the shipped body resolves on the variant. That is biologically coherent (same animal, richer repertoire, same
nervous system). It stops being coherent the moment the variant adds a SENSOR (the §Pressure step): the world vector changes and
completion to the shipped-trained node is no longer guaranteed by construction (Exp 62's replay says the pressure move is
< 0.001 in cosine, so it will almost certainly still complete — but that is a replay claim, and Exp 62 rung A's loop-OFF NODE gate
exists precisely to check it live).

*Substitute text:* "A carried-fear arm on the variant body either trains on the variant itself, or passes Exp 62's loop-OFF node
gate (the variant's submerged reading completes into the shipped-trained node) before any row; a variant that adds a sensor
must pass it in every case."

---

### NIT

**N-1 — "a light gradient to find it by":** `light_level` is a scalar at the agent's position with `rest: null` (a known
constant-mass contributor); the body senses no direction of light. Gradient-following therefore needs distinct world clusters
along the path (unmeasured — replay the descent column on real vectors first, corollary 3) or a "light increased after my last
move" credit, which is the same-call relief credit with `light_level` promoted to a drive — a mechanism. Say which.

**N-2 — "fresh (innate only)" arm:** `oxygen` has NO innate corrective need by design (`_DRIVE_CORRECTIVE_NEEDS` has no oxygen
entry; the YAML comment forbids adding one), so the fresh arm's only surfacing is `health → threat` at health < 14 (≈ 25 s, R3
arm A). Its "dives attempted" count is exploration, not a want. Label it as the R3-A floor, not an innate-want arm.

**N-3 — "relief keyed to a place … precondition for any dive optimization: without it the agent has nothing to be strategic
*with*":** the bio-faithful precondition is the other way round — a graded air-hunger signal (SF-1) is what a diving animal is
strategic WITH; a remembered air pocket is what it is strategic ABOUT. Order Phases 3 and 4 by which gap E1 actually names.

**N-4 — "Hivemind promotion … unchanged":** fine, but note the variant body's fear keys re-key through `rekey_nac_state` by EC
`id_map` — a fear learned on a sensor-extended variant will `fear_dropped` at ingest on a shipped-body receiver. Only matters if
1.4 ever exports from the classroom.

---

## The seven questions, answered from the code

**(1) Is "anticipation" the right biological job, or does the machinery already do part of it? Where does "reactive" end?**
Part of it exists and is EARNED. Reactive = the innate corrective needs (`corrective_need_intensity`: deficit past the band, no
lookahead; oxygen has none). Predictive-but-ungraded = Wire-4: a Pavlovian CS on the world cluster (`is_in_water` from second 0),
θ-gated step, max-combined with the reactive threat need — fires ≈ 2–3 s before the pain edge (Exp 60, R3-C). The NAc causal
links (`predict`, Rescorla-Wagner) predict tool→outcome, not state→pain. `drive_pain_for_value`'s bands define WHEN pain fires but no
consumer reads distance-to-band. The NAc cluster bias is state-conditioned (interoception cluster) but is a messenger (R2). The
codebase's timed predictor (SCN `anticipatory_pre_activate` + drive TemporalEvents) is dormant on both ends. **Reactive ends at the
cluster boundary: the prediction is binary on situation identity, not graded on oxygen or time.** The roadmap's gap is real but
mis-stated (SF-1); the right job is a graded/timed expectation (interval timing of the conditioned response), and a want that can
hold against a fear for a bounded time.

**(2) Is "relief keyed to a place" the relief credit's real job?** No — caricature of a channel that does not exist. Today the relief
credit keys on the ACTION (`tool:<name>`, same-call measured before/after on the affordance's DECLARED drives) in the INTEROCEPTION
cluster active at proposal; `escape_water` declares no oxygen `self_effect` so surfacing measures nothing and books the tool-success
floor + a state-blind causal link. The only world-keyed positive write is teacher-delivered operant credit. Wire-4 has no relief
half. (DNB-1.)

**(3) What does tick-anchored credit credit today; is "credits the tenth only" accurate?** Three paths: measured relief =
zero-step (the acting call); operant = one-step (last action, teacher path only) — "the tenth" is accurate here only; eligibility
traces = multi-step with 0.9/tick decay over `tool:<name>` signatures (`ToolPainBridge` producer, live) plus SCN phase fallback —
these DO reach step three, but write the 0.20-capped recognition bias and are fed by the reaction-path reward only. (SF-2.)

**(4) Is hunger in E1 a genuine drive or an LLM prior in a drive's clothes?** A genuine drive: `food` is bridge-written game
truth, the need is `corrective_need_intensity` graded 16→6, and R2's "prior-driven" means the INNATE affinity prior
(`_DRIVE_TOOL_AFFINITIES`), not an LLM — there is no LLM on the substrate-primary path. The problem is not the want's provenance but
its consumer: on a primitive roster it has none, so it cannot express itself as a descent. (DNB-2.)

**(5) Does the primitive layer change what the substrate learns in a bio-faithful way?** It changes the option set to a roster of
fixed action patterns and the learning key to one `tool:<name>` per direction — faithful as motor development, and the shape the
orient line already learned on. It is merely convenient (and confounding) if (a) it is one parametrised `move` (collapses to a
single key, unselectable), or (b) the fear keeps its compound consumer while the want gets primitives. (SF-5.)

**(6) Is the variant-body rule biologically coherent?** Yes for affordance-only variants (same sensors → same EC nodes → the
carried fear resolves; same animal, richer repertoire). Conditionally for sensor-adding variants — coherence then rests on the
live node gate, not on `extends`. (SF-6.)

**(7) Is "strategic breathing" measurable in the drives' own currency, and is the budget arithmetic right?** Measurable: yes —
`drive_pain_for_value` integrated over the sample series (R3's rule) plus oxygen minimum; pain saturates by oxygen 12 so
pain-seconds ≈ duration below 14. Arithmetic: no — the edge is 13 bubbles at ≈ 5.1–5.4 s (band 6 below set-point 20, 0.75 s per
bubble), not the ≈ 6 s oxygen-12 fear-booking publish; net bottom time ≈ 2–3 s at depth 5. (SF-3.) And the criterion "pain-free"
removes the mechanism's own teaching signal: pain reduction IS the credited relief and fear has no extinction, so a converging
curve cannot be fear-driven (SF-4).

---

## Verified by reading vs inferred

**Verified (read the symbol):** every `nac.py`, `agent_loop.py`, `tool_dispatch.py`, `tool_bridge.py`, `sem.py`, `body.py`,
`pain_bus.py`, `tool_pain_bridge.py`, `temporal_credit.py`, `component_registry.py`, YAML and `index.js` fact cited above by
`file::symbol`; the R3 pain-edge and component timings (§Outcome, §Dependent measures, the apparatus table); Exp 60's §Outcome
latencies; the ledger rows' mechanism columns; the two `_cluster_fear[...]` write sites.
**Inferred:** the 0.75 s-per-bubble depletion (from the game, corroborated by three independent measured edges in R3/Exp 60);
that the reaction-path reward in `bio_stack.py` never produces a positive reward on surfacing (read the caller, not the
subscriber's valence source); that a light gradient over a 5-block column does not rotate the world vector enough for distinct
clusters (corollary-based, not replayed).
