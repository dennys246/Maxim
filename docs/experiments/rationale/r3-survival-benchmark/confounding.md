# R3 survival benchmark — CONFOUNDING lens (four-lens design review, 2026-09-17)

Reviewed: `docs/experiments/r3_survival_benchmark_prereg.md` (DRAFT v1, uncommitted on
`r3/design-draft`). Charter: `docs/experiments/DESIGN_REVIEW.md` — does the metric isolate the
claimed cause; could a positive OR a null arise for a reason other than the claim; controls; a
statistic matched to the baseline; no alternative explanation (prior, causal link, repetition, drift,
floor). Fed by `docs/wiring/substrate-learning-channels.md` (the state-blind causal link),
`docs/wiring/cosine-separation-is-directional.md` (corollaries 3 and 6: replay offline; interoceptive
state the protocol sets moves the cluster), `docs/wiring/harness-loop-must-be-proven-live.md`, and
the Exp 61 confounding report (`rationale/exp61-shared-fear/confounding.md`, whose V1–V4 channel
audit I reuse rather than re-derive).

Verdict line is at the bottom. Three DO-NOT-BUILD (as drafted), eight SHOULD-FIX, five NIT. The
short version: **the draft's load-bearing premise — "a naive agent's survival on this gauntlet is a
constant ≈ 0 because nothing but the learned fear answers the water" — is false on the apparatus's
own measured numbers, for two independent reasons, and the calibration option it recommends (c)
calibrates on a hazard that cannot kill on this world and that the very drive under test starves of
its corrective act.** The design as written would predictably STOP at R3-cal by its own stop rule,
or, if forced through, would report a saturated ceiling in every arm and call the schedule's
arithmetic "what the drive buys".

## Verified first (evidence, not the draft's assertions)

**V1 — Exp 60's "0/60 naive placements" is a US-FREE-window null, not a lethal-window null.** The
Exp 60 probe cap is the measured air-hunger pain edge (min) − 0.75 s = 4.335 s
(`exp60_run.FROZEN["probe_cap_margin_s"]`; `docs/experiments/exp60_drowning_avoidance_prereg.md`
§Design (iii): "A test trial must be US-free"). Every "0 calls" pre-probe placement in
`exp60_trials.jsonl` ends at 4.3 s and is followed by a rescue. Exp 60 §Not claimed says it in so
many words: "this tests learned situation-fear in a US-free window, where the innate `health→threat`
reaction cannot fire." R3 removes the rescue and opens a ≈ 26 s window. The draft carries the 0/60
number across that boundary ("a naive agent dies at its first event ≈ 26 s in, and its survival
fraction is ≈ 0 in every cell") without any measurement in the new regime.

**V2 — Route 1: a live Wire-4 agent one-shot-conditions to −1.0 INSIDE a single submersion, ~7 s in,
and self-rescues ~9–10 s in.** From `exp60_water_apparatus.json` (3 cycles, the bot held at the pool
floor to damage): oxygen 13 at 5.19 / 5.09 / 5.44 s, 12 at 5.98 / 6.29 / 6.13 s, 11 at 6.75 / 6.68 /
6.92 s; `t_damage_onset` 16.24 / 16.07 / 16.65 s; the game's 2 hp/s from 20 hp ⇒ death ≈ 26 s.
The oxygen drive (`minecraft_player.yaml`): homeostatic, set_point 20, comfort_band 6, pain_scale 0.5
⇒ pain = min(1, (|Δ| − 6) × 0.5): 0.5 at oxygen 13, 1.0 at ≤ 12. The PainBus latch
(`embodiment/body.py::evaluate_failures`) re-publishes whenever `severity > latched + eps` with
`eps = max(_BREACH_MIN_EPS, 0.05 × 6) = 0.3` — every 1-bubble drop is a new publish. Wire-4
(`NAc.record_cluster_fear`, α 0.5) therefore books −0.25 at ~5.2 s, −0.75 at ~6.1 s, −1.0 (clamped)
at ~6.8 s. `anticipatory_threat_need` returns 0.75 (> θ 0.5, strictly above `recommend_action`'s
`drive_value <= 0.5` skip) at ~6.1 s; the loop proposes at 2 Hz (`flee` first by name tie-break,
fails fast in water, then `escape_water`); measured escape 1.45–1.83 s from the call
(`w4_escape.t_surface`). Surfaced by ≈ 9–10 s, 6 s before damage, 16 s before death. Exp 60's own
training loop confirms the per-episode count: it BREAKS at the first intensity-1.0 publish
(`water_trial.train`, `usable_oxygen_max` 12) and rescues, which is why each seed logged exactly
2 publishes per episode (20 over K = 10); an unrescued submersion keeps publishing. The draft's
arm 1 is "fresh persistence" on the full AUT, whose PainBus auto-wires the fear subscriber
(`build_pain_bus`; only Exp 60's ABLATED arm detaches it via `_detach_fear_subscriber`). **So the
draft's "naive" arm is Exp 60's FEAR arm training itself live, and survives event 1.**

**V3 — Route 2: the innate `health→threat` corrective need fires inside the lethal window in EVERY
arm, including ABLATED.** `agent_loop._DRIVE_CORRECTIVE_NEEDS` maps `("health", "threat")`;
`corrective_need_intensity` (homeostatic) returns `min(1, |Δ|)` once `Δ < −comfort_band` (health
< 14); `_DRIVE_TOOL_AFFINITIES["threat"]` contains `"escape"` and `"flee"` (score 0.7 × need ≥
`min_confidence` 0.3); the loop folds it with the learned need by max
(`agent_loop.py` ~L1552–1562). Drowning damage at 2 hp/s from 16.1 s: health 12 at ≈ 19.1 s ⇒
threat need 1.0 ⇒ `flee` (fails) ⇒ `escape_water` ⇒ surfaced ≈ 21.5–22 s with ≈ 8–10 hp, death
at ≈ 26 s. Margin ≈ 4 s against a first-placement latency Exp 60 measured at 2.9–3.3 s (incl. the
tie-break). Exp 60 kept this reflex OUT of its window by design (V1); R3's window puts it back in.
**The ablated arm therefore does not "die at the first event" either; it survives on a ~4 s timing
margin, and any mid-range fraction it shows is a measurement of loop/bridge latency.**

**V4 — After the first successful escape the state-blind positive link takes over selection in
EVERY arm, on the shore as well as in the water.** Exp 60 run 2 (`301ebe…`, FEAR seed 11): post-probe
`shore_roam.actions` = 14 successful `minecraft_player_escape_water` calls in the 10 s shore window
(zero drive on the shore); placements 1–5 show `escape_water` calls at t = −0.75 s, i.e. DURING the
1 s shore warm-up before the teleport; `positive_escape_links` 47–51 per probe. Mechanism:
`recommend_action` component 1 adds the best positive-link confidence with no cluster or drive
condition (`substrate-learning-channels.md`); the bridge returns "already at surface" as a SUCCESS
when the head is dry (`index.js::escape_water`), so every shore call books another positive link
(a snowball, not a decay). `drive_gate_enabled` defaults False (`NACConfig`), so nothing narrows the
pick to a drive-relevant tool. Consequences: (i) events 2..k are identical across arms (arm 4
acquires the same link the moment its innate-reflex escape succeeds); (ii) in every link-carrying
arm `eat` (max score 0.7 × hunger need) loses to the link (≈ 0.9) for the rest of the episode.

**V5 — Hunger cannot kill on this world, and is inert at the draft's horizon.** `setup_world.py`
writes `difficulty=normal` and `prepare` re-asserts `difficulty normal`; on Normal, starvation damage
stops at 1 hp (it is lethal only on Hard). Difficulty is not a gamerule, so it is in neither
`water_trial.GAMERULES` nor the frozen fingerprint — nothing pins it. The drain is slow (`break3_smoke.py`
docstring: "Minecraft hunger drains SLOWLY (a saturation buffer burns off first)"; the smoke has to
apply `effect give … hunger` to reach a deficit inside 120 s); every rescue/settle satiates to food
≥ 16, saturation ≥ 10 (`WaterTrial.rescue`). At H = 300 s (the draft's D4 budget example) a fresh
satiated bot does not reach the food-drive's deficit at all. And even where it does, V4 says the arms
that carry the link never eat. So option (c)'s "calibrating axis" is a pressure that (a) is not
lethal, (b) does not bite inside the budgeted horizon, and (c) is answered by an act the drive under
test structurally suppresses.

**V6 — Hunger drain MOVES THE CLUSTER the fear is keyed to.** `food` and `saturation` are `world`
modality sensors declared to "descend loud" from their midpoint rest (`minecraft_player.yaml`;
Exp 60 fixed `saturation`'s range precisely because a drained saturation sat at an extreme and
raised cos(shore, submerged) from 0.787 to 0.8502 — `cosine-separation-is-directional.md`
corollary 6). R1 established that the fear/bias read is an exact-key cache (`minecraft_benchmark.md`
§R1, CACHE-CONFIRMED). A gauntlet that lets food fall to the deprivation band and saturation to 0
changes the submerged reading's direction, so the trained/transferred fear key may simply not be
the cluster the hungry agent completes into. Not replayed by the draft; must be, offline, before any
hunger axis is adopted (corollary 3).

**V7 — Arm 3's protocol rests on a campaign that has not run.** The draft's status line says
"transfers (Exp 61, EARNED 2026-09-17)". `docs/experiments/data/` holds `exp61_dryrun2_2026-09-17.jsonl`
only (a one-pair diagnostic the verdict never reads); there is no `exp61_pairs.jsonl`, no
`exp61_verdict.json`, no §Outcome in the Exp 61 prereg, no graduation-ledger row; CHANGELOG line 27:
"the behavioural claim waits for the Exp 61 run". Exp 61 is FROZEN (#747), not EARNED.

**V8 — Arm 2 vs arm 3 is structurally zero on any selection-based DV.** Both fears clear θ
(−1.0 → need 1.0; −0.75 → need 0.75); `recommend_action` is a deterministic argmax over
0.7 × need with the same name tie-break; the only competitor at event 1 would be `eat` at
0.7 × hunger need, which needs hunger > 0.75 (food < 8.5 against satisfaction 16) to beat arm 3's
0.525 — impossible for a freshly satiated agent — and can never beat arm 2's 0.7 (hunger caps at
1.0, tie goes to `escape` by name). After event 1 the link dominates both (V4). Exp 61's own §Not
claimed already records: "the DV cannot distinguish 0.75 from 1.0 here."

**V9 — What the draft's refusal rule can and cannot attribute.** Death is read from the `deaths`
scoreboard objective (`WaterTrial.deaths`) at 4 Hz sampling over a 100 ms bridge cadence; the
bridge also emits a `death` event (`index.js` L156). "Starvation" as a death class is unreachable
on Normal (V5), so the rule's disjunction reduces to "oxygen == 0 at death"; a death that hunger
set up (1 hp from starvation, killed by the first drowning tick at ~16 s instead of ~26 s) reads as
"drowning". The rule cannot see a mixed cause.

## Findings

### DO-NOT-BUILD (as drafted)

**F1 — The floor premise is false; R3-cal cannot land and R3-bench would read a ceiling in every arm.**
Failure scenario: the campaign runs; arms 1–4 all survive event 1 (V2 for arm 1, V3 for arm 4, the
prior fear for 2/3), acquire the link, escape every later event at t ≈ 0, and reach H alive; the
"frozen baseline" is 12/12 in four arms and the reported contrasts are 0 ± the Wilson width. The
draft's own §R3-cal stop rule ("no accepted cell after the whole sweep → R3 stops") fires
FIRST if it is honoured — the naive sweep cannot produce a fraction in [0.20, 0.80] because the
naive agent SURVIVES, not because it dies; the draft got the sign of the saturation wrong. Evidence:
V1–V3. What changes: the design must MEASURE the naive lethal-window outcome before any calibration
argument (one seed, live Wire-4, no rescue, ~30 s — the cheapest possible pilot), and must declare
per arm whether the fear subscriber is attached (arm 1 as drafted is a live learner, not a floor;
the only no-drive floor available is the ablated arm, and its floor is the innate reflex on a ~4 s
margin).

**F2 — Option (c) calibrates on a hazard that cannot kill on this world, that the horizon never
reaches, and that the drive under test suppresses the answer to.** Failure scenario: R3-cal sweeps
`(H, food)` on fear-carrying agents and finds them at 1.0 in every cell (starvation stops at 1 hp on
Normal — V5), or, if `difficulty hard` is quietly added, finds food → 0 at a schedule-determined time
identical across arms because none of them eats once the link exists (V4) — either way the fraction
is a step function of the food supply, not a distribution, and no cell lands in the middle except by
knife-edge luck; meanwhile the drained interoceptive sensors have rotated the submerged reading off
the feared cluster (V6), so a hungry fear-arm agent that DOES drown reads as "the drive did not
buy survival under hunger" when in fact its key missed. Tuning-to-pass is the lesser problem here;
the greater one is that (c) is not calibratable at all. Rejecting (a) and (b) is right, but the draft's
reasons are weaker than the real ones: (a) fails not because it "tunes the world" but because a
naive agent makes NO actions (nothing scores → `recommend_action` returns None; the apparatus shows
`sink_hold: True`), so there is no random walk to titrate and P(surface by chance) is 0 at any
depth; (b) fails not at "both ends" but at the ceiling in all arms (F1).

**F3 — The DV is a function of the schedule after event 1, so "the learned fear buys X seconds" is
the experimenter's choice of H and T, not a property of the agent.** Failure scenario: with T = 30 s,
H = 300 s, every arm that survives event 1 survives 10 events on the link; "X" = H − 26 s for any
arm whose event-1 route works, and the arm 2 − arm 4 "drive vs exposure" contrast reads 0 while the
two arms differ in ~8 s of air-hunger pain and ~7 hp of damage at event 1 — the draft's SECONDARY,
never-gated DV (drive integrity) is where every bit of arm information lives, and the primary DV
throws it away. This is the Exp 60 caveat ("placements 2–6 are read through fear PLUS a positive
link") promoted to the entire design: repetition is the confound, and it operates in every arm
including the ablated one (V4). A per-event DV does not rescue it (events ≥ 2 carry no arm
information); a link-reset between events would be a hand-composed intervention on the substrate
inside the measurement (D43's shape). The only clean unit is ONE lethal-window event per fresh
agent — which is Exp 60/61's shape with the rescue removed, not a survival benchmark.

### SHOULD-FIX

**F4 — Exp 61 is not EARNED; arm 3 must be conditional on its verdict.** Failure scenario: R3 runs
arm 3 on the Exp 61 protocol, Exp 61 later reads NULL or INCOMPLETE (e.g. the F2 timing failure
its prereg pre-registers), and R3's frozen baseline carries a "shared" arm whose ingredient was
never shown to work. Evidence: V7. Fix the status line; gate arm 3 on `exp61_verdict.json == EARNED`
on main, or drop it from v1 and add it as a declared extension.

**F5 — Declare the fear-subscriber state of every arm and add the true floor.** As drafted arm 1
(live Wire-4) and arm 4 (detached) differ in TWO things — the exposure history AND the live
learning channel — so arm 4 − arm 1 is not "exposure without the drive"; it is "exposure minus a
live learner". Fix: arm 1 = naive + live Wire-4 (rename it: "one-shot in-situ learner"; that number
is genuinely interesting for 1.3's thesis — can an agent learn fast enough to survive its FIRST
drowning?), arm 1′ = naive + detached (innate reflex only, the actual no-drive floor), arm 4 as is.
Then arm 1′ vs arm 4 isolates exposure, arm 1 vs arm 1′ isolates live Wire-4, arm 2 vs arm 1
isolates prior training. Cost: one more cheap arm (no training).

**F6 — Drop or re-label the arm 2 − arm 3 contrast; it is structurally zero.** Failure scenario:
n = 12 vs 12 produces 12/12 vs 11/12 by a timing failure and the §Outcome quotes "the discount's
price" with an interval — a noise-only number read as a mechanism cost. Evidence: V8. If kept,
label it "structurally 0 on this DV (Exp 61 §Not claimed); reported as a tripwire only"; the one
regime in which 0.75 could matter — a hungry receiver at event 1 — is excluded by the satiated
start and by the link thereafter.

**F7 — The statistic is not matched to the data the design would produce.** (i) Log-rank on two
arms that are both fully censored at H, or one fully censored vs one dying at a constant, is
degenerate (no variance to test); (ii) "median survival time (bootstrap 95 %)" is UNDEFINED
whenever more than half the arm is censored at H, which F1 predicts for every arm — the draft should
say what it reports then (survival fraction only, or restricted mean survival time, which is
defined under censoring and is the standard companion to a fraction-at-H); (iii) Fisher on the
fraction at H is matched to Exp 60/61 — keep it, but at n = 12 vs 12 it cannot resolve the contrast
V3 predicts (ceiling vs ~10–12/12: 12/12 vs 10/12 gives p ≈ 0.24 one-sided), and the 12/12 Wilson
lower bound is 0.76 — an interval a quarter of the range wide freezes no baseline. Either the
contrasts reported must be ones n = 12 can see, or n must follow from a pilot's observed variance
as D4 half-promises ("set by the calibration's observed variance") — which needs the pilot F1
asks for. (iv) Refused episodes must be excluded, never censored; say so.

**F8 — The death-attribution refusal rule cannot do what it is asked to.** Evidence: V9. Fix: read
the bridge `death` event plus the last two snapshots before it; classify by the LAST health-loss
cause the samples show (oxygen == 0 in the last ≥ 2 samples → drowning; food == 0 and oxygen > 0 →
starvation, reachable only on Hard); record the health trajectory of the whole episode so a
hunger-set-up drowning is visible as "drowned from ≤ 2 hp"; and PIN difficulty in the fingerprint
(RCON `difficulty` query) since it is not a gamerule and decides whether starvation is a death class.

**F9 — The anti-vacuity row contradicts the mechanism.** "One naive agent's ZERO calls through its
first event" will FAIL for a live-Wire-4 naive agent (V2) — the campaign would read INCOMPLETE by
its own rule; and if the row is instead read in a 4.3 s US-free window it does not exercise the
gauntlet's regime. Replace with: one live-Wire-4 naive agent's first lethal-window event recorded
tick by tick (fear trajectory, first proposal, first call, surface time) and one detached agent's
(innate-reflex latency) — the two ends of what the gauntlet can actually show.

**F10 — Hunger's role must be re-derived or removed.** At H = 300 s it is inert (V5); to bite it
needs either a long H (the budget line 48 × H explodes) or an induced effect (`effect give hunger`,
which the break-3 smoke uses for WIRING tests only and which the D1 spirit forbids as a pressure);
to kill it needs Hard; and once it bites it changes the cluster key (V6) and is starved of `eat` by
the link (V4). If v1 keeps "food on hand" it should be as a CONSTANT (satiated at episode start,
exactly the Exp 60/61 rescue settle) with a recorded food/saturation trajectory, not as an axis.

**F11 — Replay the gauntlet's representation offline before any live cell.** Whatever axis survives
(depth, F12; hunger, F10), replay on the real captured vectors the cosine between the trained water
node and the submerged reading at the gauntlet's extreme (deepest placement; lowest food/saturation
the schedule permits) — corollary 3. A fear that does not read at the gauntlet's extreme is a cache
miss, not a survival result.

### NIT

**F12 — A fourth calibration axis the draft missed, which is game-native and ORDERS the arms:
placement depth.** Time-to-surface scales with depth (`escape_water` swims up at the game's rate;
1.45–1.83 s from 5 blocks); the arms' escape routes START at different times — prior fear ≈ 1 s
(arms 2/3), live Wire-4 ≈ 6–7 s (arm 1), innate reflex ≈ 19–20 s (arms 1′/4) — against fixed
deadlines (damage 16 s, death 26 s). Depth therefore titrates the routes out in a fixed order:
first the innate reflex, then the one-shot learner, last the pre-trained fear. "The learned drive
buys N blocks of depth" is a benchmark number the schedule does not set, and the Goldilocks middle
is a depth at which the floor arm fails and the fear arms pass — calibratable on the FLOOR arm
(the sweep gate: the ablated/detached survival falls from 1 to 0 across the depth cells), which is
the right arm to calibrate on. Caveats that make this a candidate, not a recommendation: the
`escape_water` handler holds jump for a hard 8 s cap and returns "still submerged (capped)" beyond
it (a deeper column needs repeated calls, at a 5/6 duty cycle under the same-tool cap); training
must happen at the SAME depth as the gauntlet cell (the exact-key cache; a per-cell training cost);
`y_altitude` moves within the water cluster on the same side of its midpoint (small arc — probably
the same cluster, but replay it, F11); and the classroom column must stay a walled shaft with
reachable air (`setup_world.water_classroom_geometry`). Still ONE event per fresh agent (F3).

**F13 — "Reported with intervals, never graduated" is a real distinction only if the number cannot
be read as a result; as drafted the contrasts are named after causes** ("what self-learning buys",
"drive vs exposure", "the discount's price"). Name them after the arms, and state next to each which
mechanism (prior fear / live Wire-4 / innate reflex / link) the design predicts carries it — a
reader then sees which are structurally 0 (F6) and which are timing (V3).

**F14 — Make drive integrity (or event-1 cost: latency to surface, min oxygen, damage taken, pain
publishes) the PRIMARY DV, and survival the secondary.** After F1–F3 it is the only DV with arm
information, and it is graded, not binary — it is what "how much is the drive worth" actually means
on an apparatus where every route eventually surfaces.

**F15 — Arm 4's training is clean (propose-only, rescued, no execution ⇒ no links).** Verified in
Exp 61's lens (V1) and live in dry run 2 (`reward_bias == {}`, `links == {}` on both donors). Arm 4
does carry the Wire-2 percept valence, which `propose_via_substrate` never reads — no leak. The
ablated-vs-naive contrast is confounded only by F5, not by the training.

**F16 — Interleaving arms by episode seed and one code hash are right; add the Exp 61 drift refusal
(first- vs last-quartile median of the apparatus `t_surface`)**, since V3 makes the floor arm's
outcome a timing quantity and a slow afternoon would move it.

## Answers to the brief's questions, in one line each

1. *Smuggled claim?* Yes, via the contrast names and via "buys X seconds" — X is H − 26 s, a design
   parameter (F3, F13). The "never graduated" hedge does not stop the number being quoted.
2. *Is (c) sound?* No — non-lethal on Normal, inert at H = 300 s, starved of `eat` by the link,
   and it rotates the cluster the fear is keyed to (F2, V4–V6). (a) fails for a stronger reason than
   the draft gives (no stochastic policy exists to titrate); (b) saturates at the ceiling in all arms,
   not at both ends. Fourth option: depth (F12), one event per fresh agent.
3. *Within-episode learning?* Yes, and it collapses all arms after event 1 — including the ablated
   arm, which acquires the link through the innate reflex (V3–V4). Only a first-event DV survives; a
   link-reset is a hand-composed intervention. Hunger's `eat` books no arm-differing links because
   the link-carrying arms never eat (V4).
4. *Arm 4 vs arm 1 clean?* Arm 4's training is clean (F15); the contrast is not, because arm 1 as
   drafted carries a live learning channel (F5).
5. *0.75 vs 1.0?* Structurally invisible on any selection DV (V8); the draft should not report it as
   "the discount's price" (F6).
6. *Statistic?* Fisher is matched; log-rank and the bootstrap median are degenerate/undefined under
   the censoring F1 predicts; n = 12 cannot resolve the contrast the mechanism predicts (F7).
7. *Hunger?* Adds no death cause on Normal; the refusal rule's "starvation" class is unreachable and
   a hunger-set-up drowning is mislabelled (F8, F10, V9).
8. *Saturation per arm under (c):* arms 2/3 ceiling (prior fear, ~1 s); arm 1 ceiling (live Wire-4,
   ~9–10 s surface vs 16 s damage); arm 4 high-with-timing-risk (innate reflex, ~4 s margin) — the
   only interval that can move is arm 4's, and it moves with loop/bridge latency, not with learning.
9. *Else:* Exp 61 is not EARNED (F4); difficulty is unpinned (F8); the anti-vacuity row contradicts
   the mechanism (F9).

## What I verified

- `docs/experiments/DESIGN_REVIEW.md`; the R3 draft; `minecraft_benchmark.md` §R1/§R2/§R3 + D1/D2;
  `roadmap_1_3.md` §Phase 2–3 + discipline; Exp 60 prereg §Design (iii), §Outcome (incl. the
  positive-link caveat and the "innate reaction cannot fire in a US-free window" not-claimed line);
  Exp 61 prereg §Arms, §Lifecycle, §DVs/gates, §Not claimed, build step 4 (dry runs), and the
  ABSENCE of a §Outcome / verdict file (`ls docs/experiments/data/`, `git log`, CHANGELOG L27).
- Code: `water_trial.py` (`rescue` settle values, `submerge`, `loop_window` rescue-first, `placement`,
  `probe` incl. the shore-roam window, `train` break-at-first-saturating-publish, `deaths`,
  `GAMERULES`); `exp60_run.FROZEN`; `exp61_run.FROZEN` + `ingest.FOREIGN_FEAR_DISCOUNT`;
  `nac.py::recommend_action` (components 1–3, the `<= 0.5` drive floor, the `> 1.0` raw guard, the
  `(score, name)` tie-break, the disabled drive gate), `record_cluster_fear` (α 0.5, clamp),
  `anticipatory_threat_need` (θ 0.5, `>=`), `_cluster_fear` wall-decay-on-load only,
  `_DRIVE_TOOL_AFFINITIES["threat"]`; `agent_loop._DRIVE_CORRECTIVE_NEEDS` + the max-fold of learned
  and innate threat; `sem.corrective_need_intensity`; `body.py` latch (`eps`, deepen re-publish);
  `minecraft_player.yaml` (oxygen/health/food/saturation specs, `world` modality, pain_scale 0.5);
  `index.js::escape_water` ("already at surface" success, 8 s cap) and `eat`; `setup_world.py`
  (`difficulty=normal`, `keepInventory`, `doImmediateRespawn`); `minecraft_harness` AUTONOMOUS.
- Data: `exp60_water_apparatus.json` (oxygen series, pain edge, damage onset, escape timing, all
  three cycles); `exp60_trials.jsonl` run 2 (pre 0 calls; post shore roam 14 `escape_water` calls;
  warm-up calls at t < 0; per-placement call lists).

**Not verified (other lenses / owner):** whether `bot.consume()` succeeds while submerged (matters
only if a hungry agent is ever placed, F10); the exact starvation floor per difficulty (Minecraft
rule: Easy 10 hp / Normal 1 hp / Hard lethal — from game knowledge, not measured here; the F8 pin
makes it a recorded fact either way); the swim-up rate vs depth and the walled-column geometry for
F12 (environment lens); whether `y_altitude` at a deeper placement keeps the water cluster id
(replay, F11); anything bio-faithful about what "survival worth" should mean for a one-hazard body.

## Verdict

**DO-NOT-BUILD (as drafted).** The design's floor is not a floor (V2–V3), its recommended calibration
axis cannot kill and cannot be reached (V5) and rotates the key the drives read (V6), and its
primary DV is owned by the state-blind link from event 2 onward in every arm (V4). None of this is
fixable by n or by the statistic; it is the gauntlet's shape. The salvage that keeps R3's purpose —
a frozen, un-gameable measuring stick — is: one lethal-window event per fresh agent; arms declared
by learning channel (prior fear / live Wire-4 / innate-only / ablated / shared-if-Exp-61-EARNED);
event-1 cost as the primary DV (latency to surface, min oxygen, damage, pain publishes) with survival
as the binary secondary; depth as the pre-declared difficulty axis calibrated on the floor arm
(F12), replayed offline first (F11); hunger held constant at the Exp 60 settle. Cheapest next step
before ANY of that: a two-seed pilot (one live-Wire-4 naive, one detached) with no rescue, ~30 s
each, to replace the draft's extrapolated 0/60 with a measured lethal-window outcome.
