# Roadmap 1.4 (working title "Anticipation") — complex behaviour, sequence credit, and an instrument that can measure them

**Status:** DRAFT v2 (2026-09-18) — v1 went to a five-lens parallel design review the same day
(confounding, bio-faithful, wiring, environment, scope + engineering; reports preserved verbatim in
[rationale/roadmap-1-4/](rationale/roadmap-1-4/)). Every lens returned DO-NOT-BUILD on at least one
rung *as sketched* and none attacked the thesis or the ladder's order. v2 folds all of them
(§Review record at the end lists what changed and what was dismissed). **Owner decisions taken
2026-09-18:** (1) the "Shared perception" 1.4 (perception fabric + microduck + Exp 55 + breeding,
sequenced in [deferred/second_body_staging.md](deferred/second_body_staging.md)) is **DEFERRED on a physical trigger** — it
revives the day a second body exists (a real backend registered through `maxim.robots`, or the
operator records the body's arrival), Stage A unchanged, the plan intact; (2) 1.4 continues the
survival line on the rig that already runs it: **generalization + anticipation**; (3) the deferred
JEPA plan is **re-pointed, not revived** (§JEPA); (4) the motor layer for complex behaviour is
**primitive movements** on a **variant body**; (5) the **instrument is refined first**, scoped by the
first rung that consumes each piece — never as a framework.

**Shape, on purpose:** open where 1.3's plan was not. Fixed here: the thesis, the ladder's order, the
instrument each rung needs, the release thresholds. Open: which rungs beyond the first two ship in
1.4.0, the classroom geometry (nine pilot rows decide it), and — above all — *which* mechanism the
results name. Each mechanism enters through its own plan and four-lens review when a rung names the
gap; none is built ahead of that reading. **The release NAME is fixed at the release transaction
from the highest rung with a recorded EARNED** ("Anticipation" is a working title that E3 may not
earn; §Release thresholds).

Every rung inherits 1.3's disciplines: D1 (game-native pressure and relief only), verify the
instrument first, prereg frozen on main before the first data timestamp, refusal rows never fudged
rows, the four-lens design review before a harness, the two-lens code review before merge, offline on
the scripted bridge before the rig, and the divergence trigger.

---

## The thesis

1.2 shared a *taught* want. 1.3 shared a *survival* want and measured on a frozen gauntlet what a
carried drive buys at the one moment it matters (R3 §Outcome: ≈ 25 s of latency, ≈ 11 hp, ≈ 22 s of
oxygen pain — not life). Every EARNED result is **one agent, one situation, one executed action**.
That was the right unit while the question was whether a drive moves behaviour at all. It is the
wrong unit for the next one:

> **Can the system learn a behaviour whose payoff is several steps away, in a situation it has not
> seen in exactly that form, and trade one drive against another on the way?**

Three gaps, each stated against what the code does today (the review's reading; file::symbol in the
lens reports):

- **R1 — the substrate is exact-key.** The fear C carries in R3 is keyed to one pool's sensor
  geometry; a second context is a different key. Exp 62 (prereg v2 on main) measures what the shipped
  body already carries across pools and where the context wall is.
- **R4 — credit is anchored to the executor CALL window, and relief never writes a world cluster.**
  Measured relief is credited around the same backend call that produced it (`tool_bridge` pre/post),
  so relief landing after a call returns credits the NEXT call. Relief and tool-success credit route
  to the INTEROCEPTION cluster (`tool_dispatch.record_outcome`); the only world-keyed positive write
  is the teacher path (`credit_operant_reward`). In substrate-primary mode the tool-success floor
  books +1 on EVERY successful call, so "did credit reach step three?" is vacuously YES by magnitude —
  only the credit's *source* discriminates. A multi-step path DOES exist — eligibility traces
  (`ToolPainBridge.record_tool_start` → `TemporalCreditDistributor.distribute` → `NAc.credit_node`,
  0.9/tick decay, reaching step three at ≈ 0.48) — but it lands on the cluster-blind, 0.20-capped
  recognition surface that `recommend_action` does not read. **The first R4 audit is routing, not a
  new rule.**
- **Anticipation — the prediction is a step, not a slope.** Wire-4 IS a CS→US predictor (Exp 60's
  headline word is "anticipatory"; the fear fires on the first in-water tick, at oxygen ≈ 18). What
  does not exist is a *graded* prediction of HOW FAR pain is: the underwater cluster is one cluster
  from oxygen 20 to 2, the fear is a θ-gated step on cluster identity, and nothing reads oxygen level
  or time-to-pain. Two shipped pieces are on point and dormant: `anticipatory_pre_activate` (the
  timed predictor over drive TemporalEvents) and `embodiment/cerebellum.py` (a per-affordance sensory
  forward model — write live, read dormant). **The first anticipation audit is those two, not a new
  model.**

The classroom that puts all three under one roof is the **treasure dive**: food in a roofed alcove at
depth for a hungry agent, in a sealed dark column, air pockets on the way, oxygen depleting the whole
time. Relief is game-native (hunger relief on eating, oxygen refill in air), cost is game-native
(oxygen pain, drowning damage), it is built by RCON, and its dependent measures are the drives' own
currency. The review changed what it looks like (§The classroom) and what each rung can claim.

## What is fixed and what is open

| Fixed in v2 | Open (decided by results, each with its own plan + review) |
|---|---|
| Thesis; ladder ORDER (Exp 62 ∥ E1's instrument → E1 → [relief store] → E2 → E3) | Which rungs beyond E1 ship in 1.4.0 vs 1.4.x |
| Primitives as the motor layer; param-free, fixed-duration; a VARIANT body | The exact primitive set and durations (pilot rows 1–2) |
| The instrument, split by consumer (§Phase 0); E1's part ships before E1 | The campaign core's API (extracted; acceptance test below) |
| Release thresholds T1–T4; T5/T6 conditional | Which mechanism E2/E3 name: routing, a relief store, a graded predictor, keying, none |
| JEPA re-pointed; the survival-world paired-data audit is a DIFFERENT audit from its Stage 0 | Whether any predictor is JEPA-shaped |
| "Shared perception" deferred on a physical trigger; Phase 1b deferred on the hostile-window trigger; intrinsic motivation stays a parallel line, not in this ladder | — |

---

## The classroom, as the world affords it (environment lens; every number is measured or marked pilot-measure)

- **Sealed, dark.** Light is the context wall itself (a lit pool reads cos 0.588 to the cave pool —
  Exp 62 replay), every carried fear on file was booked at light 0, and the underground light read is
  patchy across restarts (Exp 58 addendum). A light gradient can KEY a situation; it cannot STEER a
  primitive (`perceivedLight` reads at the bot's feet). So: a stone shell, one block-light source only
  if a cell-by-cell profile repeats identically across a bridge restart and a rebuild (pilot row 3),
  the fear trained IN the column.
- **Food in a roofed alcove.** No chest verb is needed — pickup on hitbox overlap is game-native — but
  item entities float up in water, so the bread sits under a roof at the entry cell; `Age:-32768s`
  NBT; per-episode `clear` and a shore `eat`-throws preflight (the inventory holds 64 bread from
  `prepare`, kept across deaths). Hunger onset is the apparatus's `effect give hunger` → `effect
  clear` (natural drain measured zero over 60 s); `foodLevel` ≤ 15 for bread's +5 to register.
- **Start by teleport-in.** No innate need proposes a movement primitive (hunger → eat/pick_up/food/
  consume/feed; threat → flee/escape), exploration is off, no LLM sits in the action path. A shore
  start never descends in any arm. E1 starts submerged at the food (R3's `submerge`), and "dives
  attempted" becomes "descents executed after entry".
- **The budget.** First pain lands at **5.15–5.28 s** (apparatus `t_pain_edge`; the oxygen-12 publish
  at ≈ 6 s is the SECOND publish), the US-free cap is 4.40 s. Each primitive costs a 0.58 s tick +
  its blocking hold + the 0.77 s `flee` tie-break tax on every tick the fear proposes. Round trip with
  the food directly below (descent + eat ≈ 1.6 s + ≤ 1.5 s pickup poll + ascent 0.31–0.43 s/block +
  ticks): **depth 3 marginal (4.5–6.6 s), depth 4–5 exceeded.** That is 2–3 one-second primitives, not
  ten. Consequences: E1's shallowest depth is 3; E3 needs a pocket every 2–3 primitives, so **E2's
  pocket is a precondition of E3's design**, not a later rung.
- **Air pockets are active states.** A 1-tall air cell under a solid ceiling is buildable (water never
  flows up; verify with `execute if block`), sensed correctly (`is_in_water` reads the EYE block),
  refill measured 3.6–3.9 s to ≥ 19 — but the bot sinks when `jump` is released (sink-back ≥ 2.1 s),
  so "in the pocket" is `swim_up` every tick. Roofed 3×3 layers, never 1×1 chimneys; never bubble
  columns (they read as water).
- **Sneak does not sink.** In the bot's installed physics sneak only scales horizontal input in water;
  the only vertical control is `jump` (+0.04/tick); descent is passive gravity (0.02/tick, a pre-1.13
  constant — vanilla 1.20.4 differs). Every displacement is measured on the rig (pilot row 1), none
  derived from vanilla.
- **Rig constraints the campaign core carries:** one client per bridge and the flee anchor fixed at
  bridge start (bridge restart per classroom); oxygen quantized 0.75 s/bubble; RCON player commands
  need the bot online + forceload; the same-tool cap (5 consecutive) throttles any primitive run
  longer than five — a `src/` exemption or bounded paths, the prereg names which.

**Nine pilot rows, each an apparatus row on the built column, REQUIRED before E1's prereg freezes:**
(1) primitive displacements — `sink`, `swim_up`, `swim_forward`, `turn_left/right` — at 1.0 s holds,
water and air, n ≥ 3, start/end positions in the bridge `detail`; (2) round trip at depths 3/4/5 with
bread in the alcove: oxygen minimum and whether the pain edge fired; (3) the light profile ×3
(restart, rebuild); (4) item behaviour — floats in the open column, stays in the alcove, auto-pickup
from the adjacent cell, `clear`; (5) eating submerged (`bot.consume()` with head in water) and its
wall time; (6) pocket build + hold + refill + sink-back; (7) hunger onset holds ≥ 60 s with the loop
live; (8) the same-tool cap live (sixth `swim_forward` dropped, or the exemption shipped); (9)
regeneration–hunger coupling (food/saturation before and after one drowning; regen after damage burns
≈ 6 exhaustion/hp, so the fear arm's cost makes it hungrier — the arms couple).

## Phase 0 — the instrument, split by the rung that consumes it

The R3 process layer carries unchanged. Instrument increments enter WITH their consuming rung, never
before it (a per-step ledger with no experiment is the harness analogue of a function with no caller
— the D43 family, one level up). Each is a two-lens code-reviewed PR run offline before the rig.

**Shipped already, consumed by Rung B's entry condition (2026-09-20):** the EC match MARGIN
(`PatternResult.best_similarity`, `SensorEncoder.last_encode_margin`, issue #786). The EC computed
how close every pattern decision came and discarded it — a separation reported `similarity=0.0`, so
a near miss and a far one were indistinguishable in every record. Read-only: no consumer decides on
it, nothing re-keys. It is the instrument the SHAPE/SUPPORT question above reads, and every Exp 62
row now carries `node_gate.read_margin` beside the node id.

**E1's instrument (T1):**

- **(1a) Per-need provenance on the selection side.** `NAc.recommend_action` already iterates the
  drives by name and records each contribution as text in `parts`; add `comp["drive_by_need"]`
  beside the aggregate `comp["drive"]`, carried on `NAc_RECOMMEND.score_components`, guarded by the
  existing byte-identical-selection test. **R3's decisiveness clause does not transfer:** it requires
  `causal == 0 and learned == 0`, which is impossible after the first successful primitive (the causal
  link forms on every success). Add a `drive_decisive` counterfactual mirroring `explore_decisive`
  (re-run the argmax with `comp["drive"]` removed; a different winner ⇒ the drive decided).
- **(2) A campaign core, extracted not designed.** Exp 60/61/R3 each re-implemented arms × seeds,
  resume, supersede, refusals, provenance, drift and the report; the three copies DISAGREE on the
  supersede / resume-key / one-hash rules (the wiring report tabulates them), so the extraction is a
  design decision and must say which rule wins and why. **Front-gate:** two shared modules already
  exist and both harnesses import them — `scripts/exp56/common.py` (scripted bridge server,
  `RecommendCapture`, sessions, RCON, bundles) and `scripts/survival_world/common.py` — so the
  question is "why not grow `survival_world/common.py`", and the answer is recorded in the PR.
  **Acceptance (all five, in the PR):** (i) `exp61_run verdict` and `r3_run report --amended` over
  the COMMITTED rows produce JSON `cmp`-identical to `exp61_verdict.json` and the frozen amended R3
  report (an offline campaign RE-RUN can never be byte-equivalent to rig rows — the rows carry
  timestamps, RSS, real-time DVs; only the pure stages over committed data can); (ii) `r3_run.py`
  imports nothing from `exp61_run.py`, `exp61_run.py` nothing from `exp60_run.py` except `FROZEN` /
  record paths; (iii) no public name that did not exist in one of the three harnesses (a NEW name is a
  design and needs its own reason); (iv) the Exp 60/61 unit tests and the R3 offline campaign test
  (`slow` lane) pass unchanged; (v) net `scripts/` LOC negative. **The first NEW consumer is
  `exp62_run.py`** — a fourth copy is the D43 test of whether the seam is real.
- **(4) Primitive motor verbs on the real bridge.** `recommend_action` returns `"params": {}` for
  every substrate proposal, and `build_tool_signature` keys `tool:<name>` only — a parameterized
  `move(direction, duration)` is neither selectable nor learnable (and `turn` with `{}` computes a
  NaN yaw today). So: **param-free, fixed-duration control-state verbs** (`swim_forward/back/left/
  right`, `swim_up`, `sink`, `turn_left`, `turn_right`), each releasing its control state in
  `finally`, reporting DISPLACEMENT not elapsed time (`ok` means displaced; a call spanning a `deaths`
  increment is respawn-terminated), with `stop` extended to `bot.clearControlStates()`. Names chosen
  to match NO `_DRIVE_TOOL_AFFINITIES` keyword (escape/retreat/withdraw/hide/shelter/food/pick_up/
  consume/feed) so the tie set is not silently changed. A relief credit requires a declared
  `self_effect` on the drive sensor (the measured set is `self_effect ∩ live ∩ drives`), so `swim_up`
  declares `oxygen` in the VARIANT body or the ledger's `drive_relief` column is structurally empty.
  This is a **bridge protocol change** (additive verbs; the Python protocol header and the fake
  bridge kept in lockstep) and it fires the Exp 56/60/61 "bridge protocol change" trigger by its
  letter — discharged with a dated annotation once the existing verbs are shown byte-identical.
  Live-measured before any claim (pilot row 1) and recorded as a gated apparatus record.
- **Always-succeed tools flood the causal channel.** Primitives succeed by construction, so the
  state-blind `causal_pos` link on the most-executed primitive will dominate the argmax (the
  fixation shape). `deferred/credit_on_progress_not_execution.md` documents exactly this and is
  relevant HERE, to item 4, not to R4; audit it before E1's prereg.

**Later rungs' instruments (enter with T5/T6, each with its own strict red gate written when its
rung's prereg freezes):**

- **(1b) The credit ledger, read at the WRITE seam.** The recommend event fires before execution and
  cannot carry credit. The write is `tool_dispatch.record_outcome` → `NAc.update_cluster_reward`,
  which emits no structured event today. Required `src/`: `update_cluster_reward` emits ONE event per
  write — `NAc_CREDIT {agent_id, cluster_id, tool_signature, reward, source, bias_before,
  bias_after, tick}` — from INSIDE the NAc so every caller is covered (tool dispatch, the teacher path,
  any future relief store); the same for `credit_node` (the trace surface). `source` is the only
  discriminator between the tool-success floor and drive relief. **Red gate (strict `xfail`):** build
  the AUT through `run_minecraft_aut` against `ScriptedWaterBridge` (the R3 offline path), submerge,
  let the loop select `escape_water`, assert ONE `NAc_CREDIT` with `source == "drive_relief"` on the
  interoception cluster active at the call; anti-vacuity arms: the test never calls the write itself
  (event count == executor calls with non-zero measured relief), and a negative control where the
  refill lands AFTER the call returns credits the next call's signature. The read is also checked at
  `recommend_action` (`consulted_bias_by_modality`), not only at the event.
- **(3) Scripted world extensions.** E1 needs `scripted_water.py` + a second anchor (the food) and
  depth — not a grid. E2/E3 add pockets, positions and displacement constants FED from item 4's
  measured apparatus row (never typed), eye-height `in_water`, the relief lag, the eat-without-food
  failure, the pocket RCON verbs, real-time cadence. Composition is proven only through
  `run_minecraft_aut`.
- **(5) Learning-curve rows.** No existing row carries within-agent episodes with a DV (Exp 60's
  training loop is propose-only conditioning). Reuse Exp 57's τ and R3's `lethal_event` as the
  episode unit with `_boundary` between; episodes-to-criterion with intervals.
- **Cadence as a budget** — measured with `loop_tick_probe.py`, optimized only if a rung's prereg
  shows the budget is the binding constraint (§The classroom's table says it may be, for E3).

## Phase 1 — Exp 62, cross-context transfer by the body (runs FIRST, concurrently)

**EARNED 2026-09-20 — PHASE 1 IS CLOSED.** Campaign `exp62-rungA-1` at one hash, 27 rows, ZERO
refusals, 46.9 min: NODE 12/12, CROSS 12/12, SAME 12/12, ABLATED 0/3 with zero executor calls, REPLAY
agrees; Fisher p 0.0022; cross-arm Wilson **[0.758, 1.000]** — the interval is the statistic, since
both fear arms sit at the ceiling by design. Prereg v3 (the freeze) + §Outcome; data #790, ledger row
#791. **Which branch of the keying tree below fired: TRANSFER** — so the keying mechanism does NOT
enter, and the wall is further out. That is why Rung B's entry condition was made independent of this
rung (Phase 5). Bound, and it is load-bearing: the apparatus has ONE discriminating world sensor, so
what transferred is invariance to the two LOW-GAIN place absolutes, not to a changed situation; a
night pool still reads 0.799, so the fear misses at night and the frozen day hides it. **Corrected
2026-09-25:** the 0.799 is `time_of_day` 0.99 — the minute before the clock WRAPS — not night; at the
fear place midnight reads 0.903 (inside the key) and the fear misses only at time ≈ 0.94–0.99, the
linear encoding of a circular clock ([#899](https://github.com/dennys246/Maxim/issues/899)). The lit-pond wall (0.588) stands.

Prereg v3 on main; owner decisions D1–D4 taken; its instrument is `WaterTrial` per pool plus the
committed replay. **It needs nothing from Phase 0** and does not wait for it (schedule below). It
names the keying gap: pools never share a cluster → within-modality keying (R1's home); share but no
transfer → the write channel (Wire-4's key); transfer → the wall is further out. **Slot:** it closes
Exp 60/61's "fears water anywhere" §Not-claimed line, a 1.3-line result — 1.3.0 if it lands before
the release transaction, else 1.3.1. T2 makes it a 1.4.0 gate as a sequencing choice, stated.

## Phase 2 — E1, the conflict rung: a want against a fear, at the food

**v1 said "graded by depth, a short path".** All five lenses refused it: no drive has a locomotion
affinity, so no arm descends; the fear's key MISSES the hungry state (cos(submerged fed, submerged
hungry) = 0.826 < 0.85, measured on committed vectors — `saturation` at 0 is a full-weight constant on
the interoceptive side, and Minecraft drains saturation before food, so every game-native hungry
agent is in that state); and depth does not enter the decision at all (the fear is a per-cluster
constant; oxygen has no innate need).

**v2 — the teleported conflict, titrated over hunger need (the R2 mould).** The agent starts
submerged AT the food, hungry, fear present. The only decision is `eat` vs `escape_water`, and its
argmax is a pure function of protocol constants: `hunger × 0.7` vs `fear × 0.7` with the 0.5 floor —
a C-strength fear (−1.0) cannot be beaten at any deficit; a D-strength fear (−0.75) is beaten iff
need > 0.75. **Before any build:** (i) the argmax replay (pure `recommend_action` over hunger need ×
fear strength × the variant roster, ≈ 30 lines on committed inputs) predicts the crossing; (ii) the
hunger-state cosine replay on real vectors decides how the fear is trained — **train and probe in the
SAME hunger state**, with a per-row NODE gate (Exp 62's loop-OFF read); a variant body may re-declare
`saturation`'s range so "hungry" is silent to the encoder, recorded as a geometry change with its own
replay. The live rung then tests the replay's prediction, and the DV is the executed choice with its
`drive_by_need` provenance (item 1a) and the counterfactual, across hunger need and fear strength.
"Depth" leaves this rung; the rung's positive is the crossing point where the replay put it, and its
null is a crossing elsewhere or no crossing — both informative about the constants, neither about
learning. **Depths must remain ONE cluster** (see §Pressure).

## Phase 3 — E2, the air-pocket rung: relief keyed to a place — MECHANISM-GATED

v1's E2 described a mechanism that does not exist (three lenses, from the same code): the cluster-fear
write is fear-only (clamped ≤ 0, no relief or extinction, only wall-clock decay); relief and
tool-success credit route to the INTEROCEPTION cluster, never the world cluster; the only world-keyed
positive write is the teacher's. v1's "falsifier" — relief credits the primitive that happened to
execute, not the place — is the shipped behaviour. **E2 therefore waits on a Phase 5 mechanism, a
cluster-keyed relief store, entered through its own plan and four-lens review (the front-gate
question against the existing eligibility trace and the Exp 56 taught-bias shape answered in
writing).** An offline structural check on the committed vectors comes first: if the routing as
shipped cannot place relief on a world cluster, E2 does not freeze until the store exists. When it
runs: two pockets at different bearings, one reachable within the budget; DVs are pocket visits per
dive, which pocket, oxygen at entry, the cluster the relief credited (item 1b), and the return rate
across episodes (item 5). Its pocket is also E3's design precondition.

## Phase 4 — E3, the optimization rung: strategic breathing (the may-fail headline)

**Not "pain-free dives".** Every fear-carrying arm ALREADY surfaces before the first pain, at oxygen
≈ 18, with zero food (R3 C: first proposal 0.99 s, head clear 3.18 s); a per-cluster constant fires
at the same oxygen at every path length. **Anticipation is pre-registered as a decision on a graded
pre-pain variable Wire-4 does not carry:** the DV is the oxygen level at the turn-toward-air, and the
claim is that it TRACKS the remaining path length (a slope with an interval). A fear-present arm that
surfaces at oxygen ≈ 20 regardless of path is recorded as Wire-4, not anticipation.

**Arms (four minimum; a rung that cannot fill them does not freeze):** (a) fear-DETACHED (R3's E), so a
pain-free dive is attributable to something other than the CS; (b) fear-present with the pocket
REMOVED, so "turn toward the pocket" cannot be `escape_water` in disguise; (c) a **dry corridor** of
the same primitive count with no oxygen cost, isolating credit reach from the budget; (d) a
harness-executed (propose-only, Exp 60-training style) DEMONSTRATION of the sequence before the free
run, isolating "never sampled" from "not credited" — the credit ledger (1b) is read on the
demonstration, the behavioural DV on the free run. The one-step arm is declared the causal link's
baseline (provenance causal > 0, learned 0 — the anti-vacuity row), never evidence of credit.
**The null is predicted on shipped wiring:** fear is a one-way ratchet with no in-session extinction
and the two needs tie at 0.7 with a name-sort tie-break, so "dive lengths converge from below" has no
fear-side gradient today; the prereg says so, and a recorded failure ships as a failure. E3 may miss
1.4.0 (T6).

## Phase 5 — mechanisms, each entering only when a rung names it

None is built in advance; each gets a plan, a written front-gate answer against the shipped
infrastructure the review located, and the full four-lens review.

- **The B8 disposition rides with this audit (owner decision 2026-09-19).**
  `deferred/transition_based_drive_pain.md` shipped its latch in July; its last phase proposed
  retiring B8's delta-attribution filter and its own review then found B8 load-bearing, so the
  premise is gone. The audit below has to understand attribution anyway: settle whether that phase
  has content left, and if the code cannot settle it, run a pre-registered arm either way rather
  than arguing it.
- **Credit routing (R4, first).** The eligibility trace already reaches step three; it lands on the
  recognition `_reward_bias` (0.20 cap, cluster-blind, fed by the reaction-path reward) which the
  selection surface does not read. Audit: route existing trace credit to the selection surface before
  any new rule. `three_factor_credit_assignment.md` is the R4 map (it names the trace and the
  Cerebellum); its learnable part goes with the fabric deferral.
  **The look-back design review ran ahead of R4's build (owner, 2026-09-24; design only — a stated
  exception to this section's rule) and DECIDED: no new look-back store**
  ([lookback_primitive.md](lookback_primitive.md)). R4's credit stays on `NAc._eligibility`; retroactive
  tagging looks back over the Hippocampus record by enqueue-time experience µs; word binding decides at
  revival; `PerceptTraceBuffer` is Dormant (CI enforces it). Two live defects it found on this very path
  are R4's first work: [#888](https://github.com/dennys246/Maxim/issues/888) (temporal anchors never
  expire in-session and dilute every reward's credit) and [#889](https://github.com/dennys246/Maxim/issues/889)
  (the reward-bias ablation switch does not ablate the live path). *Wording correction from that
  review:* `_reward_bias` **is** read by selection for `tool:*` keys (a ≤0.20 nudge, cluster-blind); it
  is the cluster-keyed credit that selection never sees. The path is live on the EARNED survival loop
  but **unfingerprinted**.
- **A cluster-keyed relief store (needed by E2).** New: a positive, world-keyed write from measured
  relief, beside the fear-only store. Front-gate against `credit_operant_reward` (teacher) and the
  trace. Enters BEFORE Phase 3.
- **A graded predictor (anticipation).** Audit first: `anticipatory_pre_activate` + drive
  TemporalEvents (dormant on both ends) and `embodiment/cerebellum.py` (write live, read dormant).
  Only if neither can carry "how far pain is" does a new plan (`latent_forward_model.md`) open — and
  that is where the §JEPA predictive idea and the §Pressure candidate input would live.
- **Keying / generalization (R1's home) — and Exp 62 §Rung B, whose entry condition is stated here
  because the branch below cannot reach it.** The tree Phase 1 writes ("never share → keying; share
  but no transfer → the write channel; transfer → the wall is further out") arms this mechanism only
  on the FAILURE branches. The dry run points at **transfer**, so on the plain reading a successful
  rung A closes the door and the measured context wall — a lit pond at 0.588, a NIGHT POOL at 0.799,
  both under the 0.85 threshold — is left with no owner. That matters: it means Exp 60/61's EARNED
  drowning fear **misses at night** for representational reasons, and the frozen-day protocol is
  what hides it. **Corrected 2026-09-25:** the 0.799 row is time 0.99, just before the `time_of_day`
  wrap; midnight at the fear place reads 0.903 (inside the key), so the "night miss" is the wrap
  (#899, a keying defect), not night. The lit-pond wall (0.588) stands — and at 0.588 it is beyond any
  graded read, so it too is a keying question. An entry condition that can only fire when an
  experiment FAILS is a trapdoor, not a gate. So Rung B gets its own trigger, independent of rung A's outcome, in two ordered parts:

  - **SHAPE (free, offline, done 2026-09-20).** `docs/experiments/data/world_channel_landscape.py`
    sweeps the REAL declared roster across its REAL declared ranges through the shipped
    `_sensor_embed`. Measured: displacing k sensors from rest, cosine crosses 0.85 **gradually** at
    every k (per-step drops 0.03–0.07, against a ±0.05 band), with 10.3 % of the grid sitting inside
    that band. **The function has a middle.** Two traps it caught in its own first draft are recorded
    in the file: uniform sampling over the declared box makes every state maximally unlike every
    other (median cos 0.24) and answers the wrong question; and anchoring the perturbation at the
    range midpoint rather than each sensor's own rest produced the all-silent ZERO vector at f=0,
    which read as a finding until the f=0 ≡ 1.0 identity caught it.
  - **SUPPORT (rig, only if shape is continuous — it is).** A trace with `doDaylightCycle` **on**,
    so `time_of_day` actually varies. Shape says the landscape has a middle; support says whether the
    world ever visits it, and a graded read is worth building only where both hold. The one committed
    open-world trace cannot answer this: `light_level` 0.0 in 1193/1193 and `time_of_day` pinned in
    1193/1193. **Answered offline, 2026-09-25 — no rig run needed or built.** With an idle bot and mobs,
    weather and movement off, light is a closed form in `time_of_day`, so a day is computable. At the
    FEAR place (sealed shell: light 0 all day, only time moves) a day is 0.95 `exact`, 0.05 `middle`
    [0.75, 0.85), 0.00 beyond — **no SUPPORT**, and the whole middle is time 0.94–0.99, the wrap. The
    pre-registered rig protocol was withdrawn on a two-lens design review (confounding + environment,
    both DO-NOT-BUILD): it measured an open spawn, where sky light moves and gives apparent SUPPORT
    (0.27) for a state the fear was never learned in —
    [rationale/rungb-support/](../experiments/rationale/rungb-support/). Memory 2S-e (B), a graded
    read's consumer, is parked: [deferred/generalization_by_pattern_completion.md](deferred/generalization_by_pattern_completion.md).

  **Recorded limit, so nobody re-measures this in the wrong place:** the water classroom cannot
  answer a generalization question at any n. Its `live_contributors` is `["is_in_water"]` — ONE
  binary discriminator — so its situation space is two points and its shape is a cliff BY
  CONSTRUCTION. Measuring there and reporting a body property is the apparatus-for-body confound the
  2026-09-20 four-lens round was about.

  If Exp 62 says pools never share a cluster: a substrate
  rule for feature-invariant keys or cluster merging on the world channel, constrained by
  `docs/wiring/cosine-separation-is-directional.md`. Fires Exp 53b/56/60 triggers and the
  retrosplenial §5 registry — the variant body shields none of that.
- **Spatial frames.** `deferred/retrosplenial_spatial_frames.md`'s T3 trigger IS E2's question
  verbatim and fires at E2's DESIGN, not its null; audit it when E2's plan opens.
- **Intrinsic motivation (1.3 Phase 6)** stays a parallel line, NOT in this ladder; its guardrail
  restated: it must not silently power E1–E3 (a declared ablation arm or its own line, never an
  undeclared default). Disposition recorded in archive/roadmap_1_3.md Phase 6.

## JEPA — re-pointed, not revived

[deferred/jepa_cross_modal_alignment.md](deferred/jepa_cross_modal_alignment.md) is a **projection
layer** aligning the 384-dim sensor encoder with the 768-dim language encoder; its revive trigger is
"a problem that is structurally cross-modal AND unsolvable by threshold tuning". Nothing in this
roadmap fires it: pressure, light, oxygen and position are sensor-encoded in one family and cosine
between them is defined. What the survival line may need is the *predictive* idea — a latent forward
model — which shares the acronym, not the mechanism, and enters (if at all) through Phase 5's graded
predictor, after the Cerebellum and the timed predictor are audited. **The survival-world paired-data
audit** (a sensor vector + a language percept per tick) is a DIFFERENT audit from that file's Stage 0
(defined over Roy-5b cradle sessions with an NAc-reward pairing rule): it gets its own ≈ 50-line
script and pairing rule, is a measurement that commits to nothing, and does not satisfy or replace
Stage 0. The file's banner is edited to say so (§Record edits).

## Pressure — the instinct, its measured limits, and the corrected criterion

Exp 62 v1 → v2 measured the `pressure` SENSOR at 0.037 gain weight (moves every cross-pool cosine by
< 0.001; as a shared component it can only raise cross-pool similarity; at full weight it sharpens
pool 1's own shore/water separation, a within-pool gain) and deferred the DRIVE version with a
sharpened bar (Minecraft deals no pressure cost — D1's pain half; `drive:pressure` is outside the
Wire-4 allowlist; air hunger, not baroreception, drives surfacing — `oxygen` already senses it).

**v1 of this roadmap inverted the criterion.** It said E1 would build pressure if it separated the
depths. For the conflict to be interpretable the depths must stay ONE cluster: a sensor that splits
them makes the fear MISS across depths and turns any depth effect into a cache-miss artifact. On the
shipped roster the depths already stay one cluster (`y_altitude` is 0.02–0.07 of its range over
3–9 blocks, gain weight 0.09; Exp 62 measured 0.92–1.00), which is what E1 needs. **So: no pressure
sensor in E1; a depth-splitter is a Phase 5 keying question.** The one door left: the graded predictor
needs a graded pre-pain input, and pressure/depth (remaining ascent) is a candidate beside the raw
`oxygen` level — recorded as an input CANDIDATE for that plan, entering only through it, with the
replay as its evidence.

## Bodies: a variant, never the shipped one — and what that does and does not shield

The classroom uses `extends: bodies/minecraft_player` (deep-merge; tested; five bodies use it). This
shields the `minecraft_player` sensor-range and body-change triggers, so the shipped body's EARNED
rows stay EARNED. It does NOT shield: the bridge protocol change (item 4 — fires Exp 56/60/61); the
recommend-event field (1a — fires Exp 56's `recommend_action` letter; discharged by the byte-identical
guard); any credit-path write (1b, Phase 5 — fires Exp 56/61 and R3's §Outcome clause); the campaign
core (a HARNESS change — fires R3's clause; discharged by the acceptance test); the minor-version
heartbeat (every Tier-1 row); any substrate rule (Exp 53b/56/60 + retrosplenial §5). The variant also
**renames every tool signature** (`{body}_{affordance}`): the fear carries (cluster-keyed) but biases
do not, and bundle ingest across bodies refuses at gate 7 — so arm D-style receivers are shipped-body
only unless the prereg says otherwise. **Before E1: re-run the R3 gauntlet on the variant for arms A
and C only (one cell, ≈ 1 h) and freeze THOSE as the variant baseline** — the shipped baseline's
3.18 s contains a 0.77 s tie-break that a changed option set moves by a whole tick without any
mechanism changing.

## Release thresholds — what 1.4.0 must contain (set); T5/T6 conditional

Definitions, so a threshold cannot be met by an absent check: a **recorded outcome** = a prereg
frozen on main before the first data timestamp, a merge-committed data PR at one hash, a report
COMPLETE (or amendments folded and floor-reviewed, as R3's were), and a §Outcome that leads with the
frozen status verbatim; **EARNED** per rung = every frozen gate of that rung's prereg PASS.

- **T1 — E1's instrument shipped and proven.** Items 1a, 2 (all five acceptance checks), 4 (the
  gated apparatus record of measured primitives) merged with their offline runs; the R3 A+C variant
  baseline frozen; the nine pilot rows committed.
- **T2 — Exp 62 has a recorded outcome** (its slot is 1.3.x; gating 1.4.0 on it is a sequencing
  choice).
- **T3 — E1 has a recorded outcome** with the argmax and hunger-state replays committed beside it.
- **T4 — ledger WALKED, not merely clean.** The release PR carries a trigger-walk table: every
  Tier-1 EARNED row × every 1.4 `src/`, bridge and harness change (the recommend-event field; the
  primitive verbs; the campaign core against R3's clause; any credit path or substrate rule that
  entered), each cell FIRED / NOT FIRED, each FIRED cell re-run or discharged with a dated annotation
  on the row; the minor-version heartbeat walk executed and dated; no `Stale` or `Broken` token. **The
  Exp 56 re-baseline on 1.20.4 is a 1.3.0 debt** (never run; Exp 61 reused the fabric without it):
  done before 1.3.0, or the row carries the dated scope annotation "1.16.5-pinned; the fabric was
  reused on 1.20.4 by Exp 61 — recorded, not re-earned" and 1.4.0 inherits nothing unstated.
- **T4a — consumerless mechanisms walked.** Every mechanism that shipped wired but with no consumer
  and a stated consumer trigger either has had its trigger fire (and a consumer entered through its
  own review) or is marked `Dormant since <date>` in its docstring, with the dormancy noted in the
  release PR. Today that is one entry: memory 2S-d's situation recall
  ([memory_2s_d_situation_cue.md](memory_2s_d_situation_cue.md) §When the recall gets a consumer).
- **T5 (conditional) — E2.** Ships in 1.4.0 only if its Phase 5 relief store entered through its own
  review AND its outcome is recorded by the time T1–T4 hold; else 1.4.x, and the CHANGELOG headline
  says so. Its instrument (1b, 3, 5) enters with it, not with T1.
- **T6 (conditional) — E3.** Same rule; a recorded failure ships as a failure; an unrun E3 is neither
  claimed nor a failure.

**Name and headline rule.** The release name is fixed at the transaction from the highest rung with a
recorded EARNED ("Anticipation" only if E3 earns it; "Sequence credit" if E2/E3's credit arm does;
otherwise an instrument name, as 1.1.4 shipped "The world seam" with no claim). The CHANGELOG
headline claims exactly that rung, names the highest rung attempted, and never describes a mechanism
that did not enter.

## Before 1.4 — the 1.3.x hardening line

[roadmap_1_3_x.md](roadmap_1_3_x.md) (drafted 2026-09-19 from the v1.3.0 blind re-score): **1.3.1**
fixes the defects that card found and ships the enforcement each one needs — a gating lane that
installs the console + crypto extras (today NO lane runs the signed-bundle or console tests, the
path both the 1.2 and 1.3 headlines travel), the nightly model-cache lane green, the
prereg-before-data lint extended to cover 1.3's own experiments, `export_memories()` which always
reports 0, and the function-length + mypy ratchets. **1.3.2** is the `agent_loop` decomposition with
behaviour-preservation gates and the Exp 60/61 re-run triggers discharged. Both are infrastructure
only, no behavioural claim.

They come FIRST because Phase 0 below builds its instrument on `agent_loop.py`: decomposing
afterwards means building that instrument twice, and refactoring while a may-fail experiment runs
confounds a null with the refactor (the divergence rule). Exp 62 depends on neither and runs in
parallel.

## Schedule that keeps the rig busy (scope lens SF-8)

```
dev box                                        rig (big-mac-mini)
─────────────────────────────────────────────  ──────────────────────────────────────────
Exp 62 two-pool plumbing PR (two-lens)     →   Exp 62 live pre-check (4 rows)
Exp 62 harness PR on WaterTrial            →   Exp 62 dry run → FREEZE → campaign (≈ 2 h)
Phase 0 1a (per-need field + counterfactual)   [idle only during review rounds]
Phase 0 2 (campaign core; exp62_run first) →   pilot rows 1–9 on the built column (≈ 2 h)
Phase 0 4 (primitive verbs)                →   primitives live-measured (gated record)
credit_on_progress audit; argmax + hunger  →   R3 A+C on the variant (one cell, ≈ 1 h)
  replays (committed)                          Exp 56 re-baseline (≈ 51 min) if T4 chooses DONE
E1 prereg → four-lens → freeze             →   E1 pilot → campaign
```

## Parallel lines — outside the ladder, with their own entry conditions

None is a rung; none may silently power E1–E3. A line touching a survival rung enters as a
declared arm or not at all (1.3's D1 posture, applied to research lines).

- **Intrinsic motivation** (`archive/roadmap_1_3.md` Phase 6): `success × novelty` vs learning-progress, with
  the mining classroom as its testbed. Unscheduled.
- **Grounded language + the cross-modal projection** ([grounded_language_acquisition.md](grounded_language_acquisition.md),
  revived 2026-09-19 as a parallel line; [deferred/jepa_cross_modal_alignment.md](deferred/jepa_cross_modal_alignment.md)
  revives with it). **Entry condition: the paired-data audit** — ~50 lines over committed
  survival-world runs, counting (sensor vector, text percept) pairs per situation, how many distinct
  texts there are, and how many are TEMPLATED game strings rather than language. The survival world
  supplies on every tick what the cradle never did: a sensor vector and a text percept of the same
  moment. The audit commits to nothing and is the honest test of whether the pairing is language or
  labels. It passes → both plans revive together behind a prereg and a four-lens review. It fails →
  both go to `archive/` with the measurement recorded.
  **Why parallel and not a rung:** 1.4's ladder already carries a may-fail headline (E3) and an
  instrument rebuild; a second research line inside it would make a null in either unreadable.
  *(2026-09-21: audit ran; disposition REDESIGN THE SOURCE; exploratory re-audit #810.)*
  *(2026-09-24: its concrete path, [deferred/grounded_word_binding.md](deferred/grounded_word_binding.md), is DEFERRED behind a frozen offline gate — do blind-authored phrasings of five situations cluster by situation? Pass → a candidate 1.5 headline; fail → archive. Language stays off this ladder.)*
- **Memory strength and forgetting** ([memory_strength_and_forgetting.md](memory_strength_and_forgetting.md),
  opened 2026-09-21): a hippocampal forgetting model — storage strength from existing signals
  (salience, novelty, RPE, pain, relevance-gated drive pressure, relief, failure; noisy-OR over
  baseline deviations), retrievability on an experience
  clock, honest activation, sleep with floors, a byte budget in `maxim config`. **Entry condition:
  its Phase 0** (input-integrity defects #813–#817). Opt-in,
  defaults pinned — the survival harnesses already run `sleep()`, so every sleep change is behind
  the strategy selection and in their fingerprints; fear and cluster-bias decay out of round one. **Ties to the theme:** its
  retroactive tagging is the same look-back as R4's delayed credit and the language line's binding —
  **Decided 2026-09-24 by R4's look-back review ([lookback_primitive.md](lookback_primitive.md)): no new
  store** — tagging looks back over the Hippocampus record (enqueue-time experience µs at every capture
  door); `PerceptTraceBuffer` is Dormant.
- **Social referencing** ([social_referencing.md](social_referencing.md), PROPOSED 2026-09-24; split
  from the language line, needs no language): consult a locally held, verified Oasis mirror only when
  the agent is both ignorant and being hurt, holding the answer as advice apart from its own
  experience. **Depends on** [public_oasis.md](public_oasis.md) Phase 0 (scheduled for it); **src after
  1.3.2**; **rig after E3's campaign**; opt-in and never on in an E1–E3 arm. Its Exp C is a may-fail
  bet of its own, which is why it waits for E3 rather than sharing the release's headline.

## What is NOT in 1.4

- **Shared perception** — deferred on a physical trigger (a second body); Stage A unchanged on
  revival; plan intact.
- **The survival reflex tier (1.3 Phase 1b)** — deferred on its trigger, now recorded in
  archive/roadmap_1_3.md: the measured onset-to-death window of a hostile against this loop's ≈ 1 s reaction;
  longer than a second ⇒ never built.
- **Intrinsic motivation (1.3 Phase 6)** — a parallel line, not this ladder (§Phase 5).
- **Hostile classrooms, crafting, farming, shelter** — behind R4's routing audit and E3.
- **Hivemind promotion** — unchanged from 1.2's WRITE-ONLY posture.
- **A benchmark framework.** Phase 0 extracts what three harnesses share and stops.

## Risks, and what each turns into

- **The credit ledger reads +1 everywhere.** The tool-success floor books on every success in
  substrate-primary mode; only `source` discriminates. Mitigation: 1b's event carries `source`, the
  red gate asserts on it, and "credit reached step k" is never read by magnitude.
- **Fixation on the most-executed primitive** (always-succeed tools flood `causal_pos`). Mitigation:
  the `credit_on_progress` audit before E1; the demonstration arm in E3; provenance on every descent.
- **The budget makes E3 impossible rather than hard.** Mitigation: pockets every 2–3 primitives are
  E3's design precondition; the pilot's round-trip table decides depth; the dry-corridor arm
  separates budget from credit.
- **A shortcut that passes while the loop fails:** harness-driven descent (teleport or `call_action`)
  with a real relief at the end. Rule: every displacement inside a DV window pairs with an executed
  `NAc_RECOMMEND(passed_gate)` → `calls[]` entry, or the row is refused.
- **Divergence.** Two consecutive rungs each surfacing a NEW failure mode ⇒ stop and audit the layer
  beneath (instrument, then bridge).

## Record edits made with this plan (so the plans audit reads one story)

[README.md](README.md) (1.4 row → this plan; a "deferred on trigger" row for Shared perception;
§Deferred entry with the trigger), [archive/roadmap_1_1_to_1_3.md](archive/roadmap_1_1_to_1_3.md) (both 1.4 rows;
a dated line in the rescope note), [archive/roadmap_1_3.md](archive/roadmap_1_3.md) (Phase 1b deferral + trigger;
Phases 4–5 → here; Phase 6 disposition; §Not in 1.3 → deferred on a physical trigger),
[deferred/second_body_staging.md](deferred/second_body_staging.md) (banner: DEFERRED on a physical trigger, not the 1.4
plan), [deferred/jepa_cross_modal_alignment.md](deferred/jepa_cross_modal_alignment.md) (re-pointed;
the survival-world audit is a different audit), CLAUDE.md §Active initiatives. DECISIONS.md records
owed with the 1.3.0 transaction: (a) the 1.4 re-point + physical-trigger deferral; (b) the
variant-body ledger rule; (c) the release-name-from-highest-EARNED rule.

## Review record (v1 → v2, 2026-09-18)

Five lenses, all reports in [rationale/roadmap-1-4/](rationale/roadmap-1-4/). Cross-confirmed and
folded: E2 has no write channel (bio-faithful, wiring, confounding — three readings of
`record_cluster_fear` / `record_outcome`); no drive selects a primitive toward food and parameterized
verbs are unselectable (all five); the credit read belongs at the write seam with `source` (wiring,
scope); the pain-free budget is 2–3 primitives, first pain 5.15–5.28 s (bio-faithful, environment,
confounding); light is the context wall, not a steering cue, and sneak does not sink (environment);
the hungry state breaks the fear's key, measured (confounding); v1's pressure criterion was inverted
(confounding); Phase 0 built ahead of named gaps and T1 gated a release on instruments with optional
consumers (scope); the Cerebellum and the timed predictor were omitted from the anticipation
front-gate (bio-faithful, scope); Exp 62 should not wait for Phase 0 (scope). **Dismissed, with
reason:** "regen off is a cleaner E1 cost line" (environment NIT) — declined: D5 froze regeneration on
for the whole survival line and the R3 baseline is regen-on; a regen-off cell is a separate,
declared apparatus if a rung ever needs a lethal floor. "Move deferred/second_body_staging.md into deferred/"
(scope SF-7 option) — declined for the link cascade; the banner carries the deferral and README
§Deferred lists it. Nothing else was dismissed.
