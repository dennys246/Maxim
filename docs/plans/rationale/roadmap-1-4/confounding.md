# Confounding lens — Roadmap 1.4 "Anticipation" (DRAFT v1, 2026-09-18)

**Reviewer:** confounding lens (DESIGN_REVIEW.md charter: does the metric isolate the claimed
cause; right controls; a statistic matched to the baseline; could a positive OR a null arise for a
reason other than the claim). **Object reviewed:** `docs/plans/roadmap_1_4.md` — a roadmap, not a
prereg, so the findings attack the SKETCHED DVs/arms/statistics and the release thresholds, and
each fix is text the roadmap can adopt before any rung's prereg is written.

**Verdict: DO-NOT-BUILD as sketched for E1 (three independent confounds, two of them measured
offline in this review) and for E2 (a structural null predictable from the credit-routing code,
the R1 shape); SHOULD-FIX on E3's arm separation, on the variant-body baseline, on the
learning-curve statistics and on two release thresholds. Phase 0 and Exp 62 are sound; the
thesis paragraph misstates the anticipation gap. Nothing here argues against the ladder ORDER
or against building the instrument first — it argues that the conflict rung's mechanism route
is missing from the sketch, and that without it every 1.4 headline candidate has a cheaper
explanation than the one it would claim.**

---

## What I verified by reading vs what I inferred

**Verified (file:symbol or file:line):**

- `src/maxim/decisions/nac.py::recommend_action` (2020–2444): score = causal link (state-blind,
  `tool:X`) + per-tool `reward_bias` + per-cluster `cluster_reward_bias` + drive relevance
  (`need × 1.0` on a name match, `need × 0.7` on an affinity keyword, **and nothing at all when the
  need ≤ 0.5** — the activation floor at nac.py:2167) + explore bonus (weight 0.0 in every survival
  fingerprint). Ties resolve by name sort. **The proposal carries `"params": {}` (nac.py:2444)** and
  `agent_loop.propose_via_substrate` passes that through unchanged (agent_loop.py:1578).
- `nac.py:579 _DRIVE_TOOL_AFFINITIES`: `hunger → (eat, pick_up, food, consume, feed)`;
  `threat → (flee, hide, retreat, escape, withdraw, defend, shelter)`. `agent_loop.py:891
  _DRIVE_CORRECTIVE_NEEDS`: `food → hunger`, `health → threat`. **No drive has an affinity for any
  locomotion verb**, and `oxygen` has NO innate corrective need (body YAML, `escape_water` comment:
  "Surfacing is DELIBERATELY a LEARNED response only").
- `nac.py::anticipatory_threat_need` (3186) + `agent_loop.py:1554–1562`: the learned fear of the
  ACTIVE cluster becomes `drives["threat"]` by MAX with the innate need, thresholded at θ, capped
  at 1.0. It is a per-cluster CONSTANT — not graded by oxygen, depth or time-in-state.
- `nac.py::record_cluster_fear` (3147): fear keys on `(agent, cluster_id, failure_mode)`, allowlist
  filtered; nac.py:3801 "fear has NO tick-anchored decay".
- `src/maxim/embodiment/tool_bridge.py:632–700`: a locally-measured world-owned drive transition
  (food, health, **oxygen**) sets `drive_relief_channel = "interoceptive"`; only a BACKEND-measured
  (azimuth) transition is `"exteroceptive"`.
- `src/maxim/runtime/tool_dispatch.py:~446–545`: `credit_cluster = intero_cluster`; the extero
  route is taken only for `drive_relief_channel == "exteroceptive"`; comment: "Seam routing:
  drive-relief AND generic tool-success write the INTEROCEPTION cluster only — never an
  exteroceptive cluster." **There is no write path that puts drive relief on a WORLD cluster.**
- `nac.py::credit_node` (2475) writes `_reward_bias[(agent, node_id)]`; `recommend_action` reads
  `reward_bias(agent, "tool:X")` (nac.py:2125) — different key spaces, so the node-level
  eligibility channel (`SensorEncoder` → `update_eligibility` on every encode, encoder.py:1046;
  `TemporalCreditDistributor`, bio_stack.py:~470) modulates RECOGNITION and cannot reach action
  selection. The roadmap's "nothing today credits step three" is TRUE for selection.
- `src/maxim/simulation/minecraft_harness.py:20`: "Substrate-primary, no LLM in the action path."
  The survival scripts (`water_trial.py`, `r3_run.py`) call `propose_via_substrate` directly. **The
  LLM lane is NOT a confound on this line** — but it also means no one supplies params.
- `src/maxim/_data/components/bodies/minecraft_player.yaml`: `saturation` range `[0,20]` rest 10
  (= the bridge clamp, fed); `food` `[0,40]` rest 20; `oxygen` `[0,40]` rest 20, comfort band 6 (pain
  below 14); `y_altitude` `[0,128]` rest 64; `is_in_water` `[-1,1]` rest 0.
- `docs/experiments/data/exp62_cross_pool_replay.py` reproduces the live gate-(ii) cosine (0.7874)
  before printing; I ran variants through it (below). R3 §Outcome baselines as quoted in the brief
  (A 27.995 / B 8.575 / C 3.180 / D 3.131 / E 28.084 s; C's 3.18 s = 0.99 first proposal + 0.77
  `flee` tie-break + 1.43 ascent; B's first proposal at 6.36 s after the oxygen-12 publish).
- Ledger re-run triggers for Exp 60/61 (`behavioral_graduation_candidates.md` rows 196–197) name
  `recommend_action` drive-activation floor, `record_cluster_fear`/θ/allowlist, encoder/EC world
  modality, `minecraft_player` sensor-range, `escape_water`/bridge water handling, `credit_node`
  write path, bundle/ingest — **not** "affordance roster" or "recommend EVENT payload" by letter.
- The three deferred credit plans and `retrosplenial_spatial_frames.md` the roadmap cites exist
  under `docs/plans/deferred/` (three_factor is at `docs/plans/three_factor_credit_assignment.md`).

**Measured in this review (offline replay on the committed Exp 60 gate-(ii) vectors, shipped
roster, shipped gain law; threshold 0.85):**

| pair | cos | reading |
|---|---|---|
| submerged FED vs submerged HUNGRY (food 5/40, saturation 0) | **0.826** | different cluster — a fed-trained fear MISSES a hungry prober |
| submerged HUNGRY vs SHORE HUNGRY | **0.858** | SAME cluster — a hungry agent has no distinct water cluster to key a fear on |
| submerged fed vs submerged food 12, saturation 0 | 0.849 | still a miss; saturation carries it |
| submerged fed vs submerged food 12, saturation unchanged | 0.9995 | `food` alone is harmless; `saturation` at 0 (w = 1.0 constant) is the whole effect |
| submerged oxygen 20 vs oxygen 12 (pain publish) | 0.9999 | one cluster through the pre-pain window |
| submerged oxygen 20 vs oxygen 2 | 0.924 | one cluster to the damage edge |
| submerged y 35 vs y 30 / 25 / 20 | 0.9996 / 0.998 / 0.992 | depth never separates (consistent with Exp 62's 0.92–1.00) |

(Minecraft drains saturation to 0 BEFORE food falls, so every game-natively hungry agent carries
saturation 0 — the hungry rows above are the state the protocol will actually produce, not a corner.)

**Inferred (not verified live):** primitive call durations, the pain-publish-inside-a-blocking-call
timing, the cost of a 108-event E1 campaign, and that a variant body's tie-break order changes with
new affordance names (it follows from "ties resolved by name sort" but I did not run it).

---

## Findings, ranked

### DO-NOT-BUILD (as sketched)

**DNB-1 — E1's "want" has no motor route to the food, so "dives attempted" is zero in EVERY arm by
construction, and the depth × arm interaction has no variance to carry a claim.**
*Claim attacked:* Phase 2 — "Does the resolution of hunger-relief against the carried drowning-fear
change with d?", DVs "dives attempted, dives that reach the food".
*Alternative explanation:* the hunger need reaches only `eat`-family names (nac.py:580); no drive
has an affinity for a locomotion primitive; the substrate emits `params: {}` so `move(direction,
duration)` cannot even be proposed; curiosity is unbuilt (explore weight 0.0 in the R3 fingerprint);
no LLM is in the action path. A hungry agent on the shore will propose `eat` (which throws with no
food in hand → a NEGATIVE causal link on `eat`) or nothing. The roadmap's own anti-vacuity row
("a fresh agent that never descends — a floor by design") is not anti-vacuity; it is the predicted
outcome of the carried-fear arm too. A null reads the architecture; a positive means the harness
supplied the descent (a teleport, a scripted path, or a primitive the prereg forgot to name), and
then "dives attempted" measures the harness.
*The deeper form:* even if the agent is PLACED at the food (the R3 event shape, which is the honest
version), the "resolution" is the argmax of `hunger_need × 0.7` on `eat` against `fear × 0.7` on
`escape_water`/`flee`, with a hard 0.5 floor on both needs, a cap of 1.0 on the fear, and name-sort
ties. Against a C-protocol fear at the cap (−1.0) hunger cannot win at ANY deficit (h ≤ 1.0); against
a D-style −0.75 it wins iff h > 0.75. **Depth does not enter the decision at all**: the fear is a
cluster constant, `oxygen` is not a need, and hunger is depth-independent. What varies with depth is
ascent physics (1.43 s from depth 5) and whether a pain publish lands inside a blocking call. So the
"depth × arm interaction" can only come from (i) cluster identity changing with depth (DNB-3) or
(ii) call-timing artifacts (SF-7) — never from a graded resolution, because none exists.
*Fix (text for the roadmap):* "E1's prereg names the mechanism by which each arm can DESCEND and by
which depth enters the DECISION. Today neither exists: no drive reaches a locomotion verb and Wire-4
fear is a per-cluster constant. Until a rung names and reviews a descent driver, E1 is the
TELEPORTED conflict: the agent is placed at the food at a fixed depth, hungry, and the DV is the
first executed action (`eat` vs `escape_water`) with its provenance, TITRATED over the hunger need
(the axis that actually enters the argmax) against the two fear magnitudes the line already owns
(self-learned −1.0, ingested −0.75). The pre-registered result is the CROSSING POINT with intervals,
computed FIRST offline as a pure function of `recommend_action` (the R2 mold — a live run adds noise,
not signal, to an argmax over constants) and then confirmed live on n rows at the predicted crossing
± one step. 'Graded by depth' is dropped unless a mechanism that grades fear by depth or by oxygen
enters through Phase 5." Note the R2 lesson applies with full force: the hunger side of this
conflict is the innate prior (the cluster credit is a messenger), so E1 measures prior-vs-Wire-4,
and the prereg says so.

**DNB-2 — The hungry arm breaks the fear's cluster key, and cannot form one in situ: the E1 arm
structure compares different clusters, not different drives.**
*Claim attacked:* Phase 2 arms — "carried fear vs fresh vs fear present but not hungry".
*Alternative explanation (measured above):* the C protocol trains FED (Exp 60's heal + satiate,
saturation at the bridge clamp 10 = rest = silent). A hungry prober reads saturation 0 = a w = 1.0
constant, and its submerged vector falls OUT of the trained cluster (0.826 < 0.85): the carried fear
is an exact-key MISS on the hungry arm, so "hunger wins over fear" is a cache miss, not a
resolution. Worse, the hungry agent's shore and water MERGE (0.858 > 0.85): the "fresh, hungry"
arm cannot acquire a water-specific fear in situ either (the Exp 60 gate-(ii) failure mode,
corollary 6 of the cosine doc, now on the interoceptive side). The "fear present but not hungry"
control is the only arm where the fear is actually present — so the arm contrast is confounded with
cluster identity 1:1.
*Fix:* "E1's prereg carries a MANDATORY hunger-state replay (the same status as the pressure replay
step): the protocol's induced `food`/`saturation` values on the VARIANT roster, reporting (a)
cos(trained-state submerged, probe-state submerged) ≥ 0.85 and (b) cos(probe-state shore,
probe-state submerged) < 0.85, before build. Fear is TRAINED and PROBED in the same interoceptive
state, and every fear arm carries Exp 61's step-4 NODE gate (loop-OFF read at the probe resolves to
the trained node id) as a per-row refusal. If the hungry state's saturation constant merges shore and
water, the variant body re-declares `saturation` so the HUNGRY state is the silent one (range
`[-10, 10]`, rest 0 — additive-safe on a variant, destructive on the shipped body, per corollary 4)
and the fed control is read at its OWN trained node. Fed-vs-hungry is reported as a cross-cluster
contrast, never as the same fear under two hungers."

**DNB-3 — The pressure/depth replay step has its success criterion inverted relative to the DV: a
sensor that separates depths is a sensor that makes the fear MISS across depths.**
*Claim attacked:* §Pressure — "if `y_altitude` separates the depths, pressure is not built; if it
does not and pressure does, it enters"; Phase 2 — depths as an axis of one fear.
*Alternative explanation:* the fear is exact-key on cluster id (R1). If two depths are two clusters,
a fear trained at one is absent at the other, and the "depth × arm interaction" is a keying
artifact indistinguishable from a graded conflict. With the shipped roster depth does NOT separate
(y 35 → 20 stays 0.992; Exp 62's 0.92–1.00), so the fear applies uniformly across depths — which is
what an interpretable conflict rung needs. Pressure, declared to cross neutral between depths, is
precisely the tool that would create the confound; and a same-side pressure move cannot open the arc
anyway (corollary 7), so the step as worded either does nothing or builds the confound.
*Fix:* "The replay step's decision rule is: the depth set must remain ONE cluster (pairwise cos ≥
0.85 on the variant roster, at the protocol's interoceptive state) for the conflict to be
interpretable; the cluster id at each depth is recorded per row and a row whose reading resolves to
a node the fear was not trained on is a keying REFUSAL, not a data point. A sensor that SPLITS depths
is a Phase-5 keying question (within-modality generalization), not an E1 sensor. Pressure therefore
does not enter through E1 as a depth separator; the door it keeps is a within-pool shore/water
sharpener (the 0.787 → 0.679 measured), which is not a contrast E1 measures."

**DNB-4 — E2's positive ("relief keyed to the pocket's WORLD cluster") is unreachable by the
shipped credit path; as written E2 is a structural null in the R1 shape, and its only live positive
has a cheaper explanation.**
*Claim attacked:* Phase 3 — "Does surfacing into an air pocket earn an oxygen-relief credit keyed to
the pocket's world cluster?"
*Alternative explanation:* oxygen is a locally-measured world-owned drive → `drive_relief_channel =
"interoceptive"` (tool_bridge.py:698) → `credit_cluster = intero_cluster` (tool_dispatch.py); the
extero route exists only for backend-measured azimuth transitions, and the comment states the rule:
relief and tool-success never write an exteroceptive cluster. The pocket's world cluster is not a
credit target. What CAN form is `(agent, interoception-cluster-at-relief, tool:<the primitive that
ran>)` — the falsifier the roadmap names — and it forms on EVERY surfacing (the R2 causal link also
forms on every successful primitive, state-blind). So a "return to the pocket" positive is
(a) the state-blind causal link on the most-executed primitive, or (b) an interoception-keyed
bias on that primitive, and a null is the routing code. Neither outcome tests "relief keyed to a
place". Two further alternative routes to a "which pocket" preference: the UNREACHABLE pocket's
route earns Wire-4 FEAR (pain on the way), so avoidance of the far pocket reads as preference for the
near one; and repeated episodes accumulate fear without decay (nac.py:3801), so the preference
drifts with exposure count.
*Fix:* "E2 does not run on the shipped credit routing. Before any E2 prereg: an offline structural
check in the R1 mold (`r1_cross_layout_probe.py`'s shape) that drives one oxygen relief through
`record_outcome` at a pocket and reads whether ANY world-cluster bias can form — predicted NO from
tool_dispatch. If NO, E2's premise is a Phase-5 mechanism (a world-cluster relief route, front-gated
against the existing extero route) with its own four-lens review, and the rung follows it. If E2
runs, its arms include a pocket-SCRAMBLED control (pockets swapped between episodes, so a place
preference must re-learn while a primitive preference persists) and a fear-detached (E-style) arm,
and the per-step ledger reports the causal, interoception-bias and world-bias components of the
chosen primitive per episode."

### SHOULD-FIX

**SF-1 — E3's "anticipation" arm cannot separate a forward model from the Wire-4 CS the line
already ships; the pain-free DV is the status quo for every fear-carrying arm.**
*Claim attacked:* Phase 4 — "does the agent turn toward the pocket BEFORE oxygen pain, i.e. on a
state that predicts pain rather than on pain?"; DVs "fraction of dives ending pain-free; oxygen at
surfacing".
*Alternative explanation:* the underwater cluster is ONE cluster from oxygen 20 to oxygen 2 (0.9999
at the pain publish, 0.924 at the damage edge); Wire-4 keys the fear to it; the fear fires on the
FIRST in-water tick (R3 C: first proposal at 0.99 s, `escape_water` dispatched at 1.76 s, head clear
at 3.18 s — at oxygen ≈ 18). Every fear-carrying arm therefore ALREADY surfaces before the first pain,
at zero food. "Pain-free dives" is a positive with a cheaper explanation than a predictor; and since
the cluster id is constant through the dive, a predictor keyed on the substrate's latent has nothing
graded to read — the only graded pre-pain variable is the raw `oxygen` sensor, and surfacing on a
threshold of it is a reflex, not a prediction. Conversely a null ("never surfaces before pain while
reaching food") is what you get when the fear has been outcompeted or never keyed. The two-arm
separation (one-step vs long path; path length varied) does not touch this: the CS fires at the same
oxygen (≈ 20) at every path length, so "the prediction has to fire at different oxygen levels" is
satisfied by no mechanism and refuted by none.
*Fix:* "Anticipation is pre-registered as a DECISION on a graded pre-pain variable that Wire-4 does
not carry: the DV is the oxygen level at the turn-toward-air, and the claim is that it TRACKS the
remaining path length (a slope with an interval), which a per-cluster constant cannot produce. A
fear-present arm surfacing at oxygen ≈ 20 regardless of path is recorded as Wire-4 (the CS), not as
anticipation. The arms include a fear-DETACHED arm (R3's E) so a pain-free dive is attributable to
something other than the CS, and a fear-present arm with the pocket REMOVED (so 'turn toward the
pocket' cannot be the CS's `escape_water` in disguise). The thesis paragraph is reworded: the gap is
not 'no mechanism predicts pain before pain' (Wire-4 is exactly that, and 'anticipatory' is the Exp
60 headline word) but 'no mechanism predicts HOW FAR pain is' — a graded/temporal prediction."

**SF-2 — E3's one-step vs long-path arms confound credit reach with the oxygen budget and with
sampling; the one-step arm's "learning" is the causal link.**
*Claim attacked:* Phase 4 — "Sequence credit (R4): the same task with a one-step path (credit
trivially reaches) vs the long path"; Risks — "the two-arm separation above is a prereg requirement".
*Alternative explanations for a long-path null:* (i) the longer path costs more oxygen, so pain
arrives regardless of credit (budget, not credit); (ii) with no curiosity, no LLM and no locomotion
prior, the sequence is never EXECUTED once, so there is nothing to credit (sampling, not credit);
(iii) the state-blind causal link on the primitive executed most often dominates the argmax
(nac.py: causal_pos is component 1) — a "sequence" that repeats one primitive is fixation, not
learning. *For the one-step positive:* `eat` after one primitive is the R2 result — the causal link
plus the innate prior select it without any cluster credit (`substrate-learning-channels.md`), so the
one-step arm is a tautology baseline, not a demonstration that credit reaches step one.
*Fix:* "E3's prereg separates the three by arm: (a) a long-path arm with NO oxygen cost (a dry
corridor of the same primitive count to the food) isolates credit reach from the budget; (b) a
harness-executed (propose-only, Exp 60-training style) demonstration of the sequence before the free
run isolates 'never sampled' from 'not credited' — the per-step ledger (T1) is read on the
demonstration, the behavioural DV on the free run; (c) the one-step arm is declared as the causal
link's baseline, with its provenance read (causal > 0, learned 0) as the anti-vacuity row, never as
evidence of credit. A rung that cannot fill (a) and (b) does not freeze."

**SF-3 — The R3 baselines are the SHIPPED body's; every 1.4 rung runs a VARIANT body whose option
set changes the argmax and the tie-break, so "measured against A/B/C/D/E" is measured against a
different instrument.**
*Claim attacked:* §Bodies + the brief's "the baseline every rung is measured against".
*Alternative explanation:* C's 3.18 s contains a 0.77 s `flee` tie-break — `flee` and
`escape_water` tie on the threat affinity and the winner is the name sort; adding `move_*`, `sink`,
`swim_up`, `turn` (or any name containing "escape/retreat/withdraw/hide/shelter" or
"food/pick_up/consume/feed") changes the tie set and the drive-relevant set. A latency that moves by
a whole tick on the variant body is an option-set effect, not a mechanism effect; and Exp 62's
keying result (shipped roster) does not transfer to a variant that adds ANY sensor (the geometry
changes; corollary 4 applies to the variant, not to the shipped rows).
*Fix:* "Before E1: re-run the R3 gauntlet on the variant body for arms A and C only (one cell, ≈ 1 h)
and freeze THOSE as the variant baseline; primitives are named to avoid every `_DRIVE_TOOL_AFFINITIES`
keyword; the tie-break order at the boundary is an apparatus row; any sensor the variant adds is
re-replayed offline (the Exp 62 script) before the variant is used for a claim."

**SF-4 — The within-agent learning curve (E2/E3) has three drivers that are not learning: fear
accumulation without decay, causal-link frequency, and tick phase.**
*Claim attacked:* Phase 0 item 5 + E2/E3 "episodes-to-criterion", "return rate across episodes".
*Alternative explanations:* fear has no decay and nothing discounts it (Exp 62 §Not claimed), so
repeated painful episodes drive the fear to cap and the curve converges to ZERO dives — and "fraction
of dives pain-free" over zero dives is 0/0, which a naive report renders as 1.0; the state-blind
causal link grows on every primitive success, so the most-used primitive wins regardless of
place/outcome (a frequency curve); the interoception cluster changes with oxygen and hunger across an
episode, so cluster biases spread over several ids and never accumulate on one; and per-episode
latencies quantize on the tick phase (the R3 band lesson).
*Fix:* "Episodes-to-criterion requires a criterion of REACH and pain-free jointly; a zero-dive
episode is a failure row, never excluded; a yoked control (same number of primitive executions with
relief decoupled from the pocket) is an arm; the ledger reports causal vs cluster components per
primitive per episode; any cadence band is frozen on idle ticks over a like-for-like window
(`instrument-band-statistic-matches-window.md`)."

**SF-5 — E1's statistic is undefined on the cells the sketch predicts.**
*Claim attacked:* "the depth × arm interaction on the reach rate and on pain-seconds".
*Gap:* two of three arms are predicted at reach rate 0/12 (the floor by design) and pain-seconds is
structurally zero for a row that never dives; an interaction over zero-variance cells is undefined,
and n = 12 per cell × 3 arms × 3 depths = 108 events is beyond one rig session. The R3 lesson —
match the statistic to the shortest window/smallest n any arm produces — applies.
*Fix:* "Pre-register CONDITIONAL DVs (given a dive) and the handling of zero-dive rows; the primary
is the arm contrast at ONE pre-registered cell (the titration crossing point of DNB-1) with an exact
test; depth (if kept) is descriptive; n is sized from the pilot, not assumed."

**SF-6 — T1's "byte-equivalent" and T4's "every trigger fired" are not measurable as written.**
*T1:* `r3_report.json` carries campaign ids, hashes, timestamps and paths; two honest runs of the
same rows through two harness copies differ in bytes. *Fix:* "equal after a declared normalization
(the listed volatile fields), compared as canonical JSON — the list is part of the Phase 0 PR."
*T4:* the ledger triggers are named by symbol; the per-need breakdown on the recommend EVENT touches
`_emit_recommend_action_event`, which no trigger names by letter, and a bridge that gains `move`
verbs is a "bridge water-handling change" only if `escape_water`'s path is touched — so T4 can be
satisfied vacuously by reading the letters narrowly. *Fix:* "T4 requires a dated table mapping every
1.4 `src/` and bridge commit to the trigger letters it touches — including 'none, because …' — reviewed
by the two-lens round; a row is discharged by that table, never by absence of a match."

**SF-7 — The cadence budget is a confound on every "before/after pain" DV, not only a constraint.**
*Gap:* a blocking `move(duration)` call is invisible to the loop; a pain publish that lands inside it
is acted on a full tick later; the 5.8–6.1 s budget minus descent minus ascent (1.43 s from depth 5,
more from deeper) leaves the ROUND-TRIP pain-free budget at ≈ 3–4 s, not 6. Any "surfaced after
pain" can be a call-duration artifact. *Fix:* "primitives are ≤ one loop tick; the ledger stamps call
start/return (already) and reports pain-seconds accrued INSIDE calls separately; the prereg states the
round-trip budget, not the one-way one."

**SF-8 — The headline rule has two holes: the release NAME and an undefined "EARNED".**
*Gap:* "claims exactly the highest rung with a recorded EARNED … never describes a mechanism that did
not enter" is the right rule, but the release is already named "Anticipation" — the E3 mechanism —
and T6 is open; a 1.4.0 that ships T1–T4 under that name overclaims by its title. And "EARNED" is
defined per prereg, none of which exists yet; nothing stops an E1 prereg from defining EARNED as a
depth interaction that (DNB-1/2/3) has no mechanism route. *Fix:* "The release name is provisional and
set at the release transaction from the highest EARNED rung ('Instrument', 'Conflict', …);
'Anticipation' is used only if T6 records EARNED. Each rung's EARNED criterion names the mechanism
route by which the DV can vary (which score component moves, by what) — a DV with no route is not a
claim."

### NIT

- **N-1** Phase 0 item 1: the R3 drive-decisive refusal rule (`drive > 0, causal 0, learned 0`)
  cannot be reused in any rung where learning is the DV — those components are non-zero by design.
  The per-component read is the replacement; say so in Phase 0.
- **N-2** Phase 0 item 5: "three consecutive pain-free reaches" — the criterion must say REACH; a
  non-diving agent is pain-free forever.
- **N-3** Phase 0 item 4: `move(direction, duration)` is unproposable by the substrate (`params:
  {}`); the primitive set must be param-free verbs (`swim_up`, `sink`, `move_fwd`, …), which is also
  the option-set change SF-3 prices. State it in Phase 0, not in E1's prereg.
- **N-4** Exp 62's outcome map ("pools never share a cluster → keying gap", …) is on the SHIPPED
  body; add: "the variant body inherits none of these readings until re-replayed".
- **N-5** §Pressure's "one door left open" should say which contrast the door is for (within-pool
  shore/water sharpening), since DNB-3 closes the depth-separation door.
- **N-6** The thesis' "R4 — a payoff ten actions away credits the tenth" is verified for the
  (cluster, tool) channel; note in the same sentence that the node-level eligibility channel exists
  and is inert for selection (`credit_node` → node keys; `recommend_action` reads `tool:` keys), so
  a reader does not mistake a ledger row of node credit for step credit.
- **N-7** "Idle tick 0.58 s … one primitive in ten" — the 6 s is one-way to the publish; see SF-7.

---

## Direct answers to the brief's questions

- **Are T1–T6 measurable as written?** T2, T3, T5, T6: yes (a report status on a frozen prereg).
  T1: not until "byte-equivalent" gets a normalization list. T4: not until "fired by its letter" gets
  a commit-to-trigger table; as written it can be satisfied by an absence.
- **Does the headline rule prevent overclaiming?** For the CHANGELOG sentence, yes. For the release
  NAME, no (SF-8). And it delegates "EARNED" to preregs that do not yet exist, so it is only as good
  as DNB-1/2/3 being folded into E1's.
- **Is E3's two-arm separation sufficient to separate sequence credit from anticipation?** No. The
  one-step/long-path arm confounds credit with budget and sampling (SF-2), and the path-length arm
  cannot separate a predictor from the shipped Wire-4 CS, which already fires pre-pain at a constant
  oxygen (SF-1). Four arms minimum: dry long path, demonstrated sequence, fear-detached, pocket-removed.
- **The single cheapest next step** the roadmap can take before any Phase 0 code: the E1 argmax
  replay (a pure `recommend_action` over hunger need × fear magnitude × the variant tool roster) and
  the hunger-state cosine replay — both are ~30 lines on committed inputs and together they decide
  whether the conflict rung has a mechanism route at all.
